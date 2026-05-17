/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// String-codec decode kernels and orchestrator for the GPU-native scan path.
//
// Top-down structure of this file:
//   1. Per-codec on-disk header structs + a compact FSST decoder layout.
//   2. Per-codec device descriptors (one CTA per descriptor at launch).
//   3. Anonymous-namespace helpers (bit unpacking, FSST-length precompute,
//      SM-targeting chunk expansion).
//   4. DICTIONARY pass-1 + pass-2 kernels.
//   5. FSST pass-1 (split A+B per-segment / phase-C chunked) + pass-2 chunked.
//   6. DICT_FSST pass-1, pass-2, and inline-NULL mark kernel.
//   7. Validity-overlay helper (mirrors gpu_native_decode.cu).
//   8. Public `gpu_decode_strings_column` orchestrator.
//
// Header parsing runs on device, matching the dispatcher contract in
// gpu_native_decode.cuh. The one host-side step is the FSST symbol table
// — `duckdb_fsst_import` is opaque, so per-segment decoders are deserialized
// host-side and concatenated into the descriptor arena. The shared host
// scan over `max_string_length` (used to upper-bound the chars allocation)
// also lets us skip the post-prefix-sum sync in the common case.
//
// Phase-2 perf opportunities (deferred; bench-gate before merge):
//  - DICTIONARY pass-2: cp.async stage of offsets+lengths so warp-shared dict
//    entries hit shmem instead of repeating gmem loads through L1.
//  - FSST length scan: BlockScan today; WarpScan + cross-warp aggregate is
//    ~equivalent perf with one fewer __syncthreads. Measure before switching.
//  - DICT_FSST inline-NULL walk: byte-by-byte today; uint4 + popcount stride
//    is ~4x fewer load issues for hot validity bitmaps.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_strings.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

//===----------------------------------------------------------------------===//
// On-disk header structs.
//
// Layouts mirror duckdb's compression headers. Sizes / field meanings are
// authoritative from the references in:
//   duckdb/src/storage/compression/dictionary.cpp     (DICTIONARY)
//   duckdb/src/storage/compression/fsst.cpp           (FSST)
//   duckdb/src/storage/compression/dict_fsst/{compression,decompression}.cpp
//                                                     (DICT_FSST)
//===----------------------------------------------------------------------===//

/// DICTIONARY compression header (20 bytes, packed contiguous).
struct dict_header_t {
  uint32_t dict_size;             ///< total uncompressed dictionary size in bytes
  uint32_t dict_end;              ///< offset within segment to dict-end sentinel
  uint32_t index_buffer_offset;   ///< offset to forward-cumulative dict-byte index
  uint32_t index_buffer_count;    ///< number of dict entries
  uint32_t bitpacking_width;      ///< width of the selection-buffer bitpacking
};

/// FSST compression header (16 bytes).
struct fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t bitpacking_width;
  uint32_t fsst_symbol_table_offset;
};

/// DICT_FSST compression header (16 bytes — confirmed by sizeof()).
struct dict_fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_count;            ///< includes reserved idx 0 (= NULL, length 0)
  uint8_t  mode;                  ///< 0=DICTIONARY, 1=DICT_FSST, 2=FSST_ONLY
  uint8_t  string_lengths_width;
  uint8_t  dictionary_indices_width;
  uint8_t  _pad;
  uint32_t symbol_table_size;
};

enum : uint8_t {
  DICT_FSST_MODE_DICTIONARY = 0,
  DICT_FSST_MODE_DICT_FSST  = 1,
  DICT_FSST_MODE_FSST_ONLY  = 2,
};

/// FSST escape-byte sentinel (reserved code, not a symbol index).
constexpr uint8_t FSST_ESC = 255;

/// Compact FSST decoder for GPU upload — just len + symbol, no version
/// header. Layout matches the leading region of duckdb_fsst_decoder_t so a
/// host memcpy from `duckdb_fsst_import` writes the right bytes.
struct fsst_decoder_compact {
  uint8_t len[255];
  unsigned long long symbol[255];
};

/// FSST decoder layout used by `duckdb_fsst_import`. Layout-compatible with
/// duckdb_fsst_decoder_t — NOT inspected outside the import → memcpy step.
struct fsst_decoder_full {
  unsigned long long version;
  unsigned char zeroTerminated;
  unsigned char len[255];
  unsigned long long symbol[255];
};

// Imported from libduckdb — declared here so we don't have to pull the FSST
// header path into the CUDA compile.
extern "C" unsigned int duckdb_fsst_import(void* decoder, unsigned char* buf);

//===----------------------------------------------------------------------===//
// Per-codec device descriptors. Each descriptor describes one chunk of work
// for one CTA at launch time — chunked DICTIONARY / FSST descriptors are
// produced by the SM-targeting expansion below.
//===----------------------------------------------------------------------===//

/// DICTIONARY / per-chunk descriptor. Re-used for FSST length pass A+B (one
/// CTA per *segment*, not per chunk — the prefix-sum state is per-segment)
/// and for FSST gather phase-C chunking (one CTA per chunk).
struct alignas(8) string_chunk_desc {
  uint8_t const* d_bytes;     ///< segment-base device pointer
  uint32_t bytes_size;        ///< staged segment size (bound for kernel reads)
  uint32_t row_count;         ///< rows in this chunk
  uint32_t global_row_start;  ///< column-relative output position
  uint32_t seg_row_start;     ///< rows skipped at the segment head when chunked
};

/// FSST chunked-gather descriptor. Adds the FSST-local row offset (so the
/// chunk can index the per-segment compressed-offsets array) and the
/// per-segment decoder index (segments share decoders across chunks).
struct alignas(8) fsst_chunk_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
  uint32_t global_row_start;
  uint32_t fsst_row_start;     ///< offset into d_comp_offsets
  uint32_t seg_decoder_idx;    ///< index into shared decoder array
  uint32_t is_first_chunk;     ///< 1 if first chunk of its segment, else 0
};

/// DICT_FSST per-segment descriptor (one CTA per segment — chunking the
/// per-row gather inside one segment buys little once dict_count is high).
struct alignas(8) dict_fsst_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
  uint32_t global_row_start;
  uint32_t seg_row_start;
  uint32_t dict_data_offset;       ///< within segment, to raw/FSST-compressed dict bytes
  uint32_t dict_indices_offset;    ///< within segment, to bitpacked dictionary_indices
  uint32_t seg_dict_offset_base;   ///< index into shared d_byte_offsets / d_decoded_offsets
  uint32_t seg_decoder_idx;        ///< index into shared decoder array (unused for mode 0)
  uint32_t dict_count;             ///< entries in this segment's dict (incl. reserved idx 0)
  uint32_t predecode_seg_offset;   ///< offset into the global predecode buffer for mode-1
  uint8_t  dict_indices_width;
  uint8_t  mode;
  uint8_t  _pad[6];
};

//===----------------------------------------------------------------------===//
// Anonymous-namespace helpers.
//===----------------------------------------------------------------------===//

constexpr uint32_t BLOCK_DIM     = 256;
constexpr uint32_t MIN_CHUNK_ROW = 64;
/// Cap host-upper-bound chars allocation; above this we sync once and use
/// the exact total. Keeps a pathological `max_string_length` from triggering
/// gigabyte-class over-allocation.
constexpr size_t HOST_UPPER_BOUND_LIMIT = size_t{512} * 1024u * 1024u;

/// Round n up to the nearest multiple of 8 (DuckDB's `AlignValue<idx_t>`).
constexpr uint32_t align_up8(uint32_t n) { return (n + 7u) & ~7u; }

/// Read one width-bit value at logical index `idx` from a 32-bit-word bit
/// stream. NOTE: this duplicates the helper in gpu_decode_bitpacking.cu's
/// anonymous namespace — both are private. A future cleanup PR will hoist
/// to a shared `bitpack_unpack.cuh` once both have landed.
template <typename T>
__device__ __forceinline__ T unpack_value(uint32_t const* packed,
                                          uint32_t idx,
                                          uint32_t width)
{
  if (width == 0) return T(0);
  uint64_t bit_pos  = static_cast<uint64_t>(idx) * width;
  uint32_t word_idx = static_cast<uint32_t>(bit_pos / 32);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 31);
  uint64_t combined = static_cast<uint64_t>(packed[word_idx]);
  if (bit_off + width > 32) {
    combined |= static_cast<uint64_t>(packed[word_idx + 1]) << 32;
  }
  uint64_t result = combined >> bit_off;
  if constexpr (sizeof(T) > 4) {
    if (bit_off > 0 && bit_off + width > 64) {
      result |= static_cast<uint64_t>(packed[word_idx + 2]) << (64 - bit_off);
    }
  }
  uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return static_cast<T>(result & mask);
}

/// Host-side bitpack unpack — DuckDB's BitpackingPrimitives packs values in
/// 32-element groups that are bit-dense (4*width bytes per group), so group
/// boundaries fall on 32-bit boundaries with no rollover. Reading 8 bytes
/// straddles a group boundary but the bit math still works out for the
/// values we care about (dict-entry lengths up to 32 bits).
template <typename T>
inline T host_unpack_bitpacked(uint8_t const* packed, uint32_t idx, uint32_t width)
{
  if (width == 0) return T(0);
  uint64_t bit_pos  = static_cast<uint64_t>(idx) * width;
  uint32_t byte_off = static_cast<uint32_t>(bit_pos >> 3);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 7u);
  uint64_t combined;
  std::memcpy(&combined, packed + byte_off, sizeof(uint64_t));
  uint64_t mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return static_cast<T>((combined >> bit_off) & mask);
}

/// Walk an FSST-compressed byte run to count its decoded length, without
/// producing output. Used to precompute per-dict-entry decoded offsets so
/// the GPU gather knows where to put each row's bytes.
inline size_t fsst_decompressed_length_host(fsst_decoder_compact const& dec,
                                            uint8_t const* src,
                                            size_t src_len)
{
  size_t pos = 0, dec_len = 0;
  while (pos < src_len) {
    uint8_t code = src[pos++];
    if (code < FSST_ESC) {
      dec_len += dec.len[code];
    } else {
      ++pos;
      ++dec_len;
    }
  }
  return dec_len;
}

/// Cached SM count → target CTA count for chunk expansion. We aim for two
/// waves of 8-way occupancy at 256-thread blocks; on Turing/Ampere/Hopper
/// that's 132 SMs * 4 * 2 ≈ 1056 chunks at the high end.
uint32_t get_target_ctas()
{
  static uint32_t cached = 0;
  if (cached != 0) return cached;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 256u;
  int occupancy_blocks = prop.maxThreadsPerMultiProcessor / 256;
  cached = static_cast<uint32_t>(prop.multiProcessorCount * occupancy_blocks * 2);
  return cached;
}

/// Split DICTIONARY / UNCOMPRESSED descriptors into smaller chunks so the
/// kernel grid fills available SMs. Below the 64-row floor the per-CTA
/// overhead dominates — keep the original descriptor.
/// Chunk-expand a vector of dict_fsst_desc so each chunk descriptor covers a
/// row sub-range of its parent segment. The kernel sees N CTAs instead of 1
/// per segment, filling the GPU. Only the per-row work (compute_lengths,
/// gather) uses chunked descriptors; per-dict work (predecode, mark_nulls)
/// keeps one-CTA-per-segment so each segment's dict gets decoded once.
inline std::vector<dict_fsst_desc> expand_chunks_dict_fsst(
  std::vector<dict_fsst_desc> const& descs, uint32_t target_ctas)
{
  uint32_t total_rows = 0;
  for (auto const& d : descs) total_rows += d.row_count;
  if (descs.size() >= target_ctas || total_rows == 0) return descs;

  uint32_t chunk_size = total_rows / target_ctas;
  chunk_size = std::max(chunk_size, MIN_CHUNK_ROW);
  chunk_size = (chunk_size / 32u) * 32u;
  if (chunk_size == 0) chunk_size = 32u;

  std::vector<dict_fsst_desc> out;
  out.reserve(target_ctas + descs.size());
  for (auto const& seg : descs) {
    uint32_t remaining = seg.row_count;
    uint32_t off = 0;
    while (remaining > 0) {
      uint32_t n = std::min(remaining, chunk_size);
      dict_fsst_desc chunk    = seg;
      chunk.row_count         = n;
      chunk.global_row_start  = seg.global_row_start + off;
      chunk.seg_row_start     = seg.seg_row_start + off;
      out.push_back(chunk);
      off += n;
      remaining -= n;
    }
  }
  return out;
}

std::vector<string_chunk_desc> expand_chunks(std::vector<string_chunk_desc> const& descs,
                                             uint32_t target_ctas)
{
  uint32_t total_rows = 0;
  for (auto const& d : descs) total_rows += d.row_count;
  if (descs.size() >= target_ctas || total_rows == 0) return descs;

  uint32_t chunk_size = total_rows / target_ctas;
  chunk_size = std::max(chunk_size, MIN_CHUNK_ROW);
  // Round down to warp size for store coalescing within a chunk.
  chunk_size = (chunk_size / 32u) * 32u;
  if (chunk_size == 0) chunk_size = 32u;

  std::vector<string_chunk_desc> out;
  out.reserve(target_ctas + descs.size());
  for (auto const& seg : descs) {
    uint32_t remaining = seg.row_count;
    uint32_t off       = 0;
    while (remaining > 0) {
      uint32_t n = std::min(remaining, chunk_size);
      out.push_back({seg.d_bytes,
                     seg.bytes_size,
                     n,
                     seg.global_row_start + off,
                     seg.seg_row_start + off});
      off += n;
      remaining -= n;
    }
  }
  return out;
}

//===----------------------------------------------------------------------===//
// DICTIONARY codec.
//
// On-disk layout per segment:
//   [header 20B]
//   [bitpacked selection_buffer at offset 20]
//   [forward-cumulative index_buffer at hdr.index_buffer_offset]
//   [dict bytes ending at hdr.dict_end]
//
// Each row's selection index points to one entry of `index_buffer`. Index 0
// is reserved for NULL (decoded length 0). Length(i) = idx_buf[i] -
// idx_buf[i-1]; bytes for entry i live in `[dict_end - idx_buf[i],
// dict_end - idx_buf[i-1])`.
//
// One CTA per chunk, grid-stride within the chunk.
//===----------------------------------------------------------------------===//

// UNCOMPRESSED VARCHAR (ported from pr651/a2-decode-strings-floats).
// DuckDB layout per segment: [dict_size(4)] [dict_end(4)] [offsets(4*N)] [chars]
// offsets[i] is signed int32, backward-cumulative from `dict_end`. The sign bit
// is unused here (DuckDB's inline-vs-pointer distinction is per-row and lives
// in the string_t slot, which UNCOMPRESSED *blocks* don't use — the segment
// layout above is the canonical form for storage scans).
__global__ void kernel_compute_lengths_uncomp(string_chunk_desc const* __restrict__ descs,
                                              uint32_t* __restrict__ d_lengths,
                                              uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  // Defensive bounds check: at minimum [dict_size(4)][dict_end(4)][offsets(4*end_row)].
  __shared__ uint8_t sm_ok;
  if (threadIdx.x == 0) {
    uint32_t end_row = desc.seg_row_start + desc.row_count;
    sm_ok = (limit >= 8u + size_t{end_row} * 4u) ? 1u : 0u;
  }
  __syncthreads();
  if (!sm_ok) {
    for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0u;
    }
    return;
  }

  const int32_t* duck_offsets = reinterpret_cast<const int32_t*>(base + 8);
  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i    = desc.seg_row_start + i;
    int32_t cur       = duck_offsets[seg_i];
    int32_t prev      = (seg_i > 0) ? duck_offsets[seg_i - 1] : 0;
    uint32_t abs_cur  = static_cast<uint32_t>(cur >= 0 ? cur : -cur);
    uint32_t abs_prev = static_cast<uint32_t>(prev >= 0 ? prev : -prev);
    d_lengths[desc.global_row_start + i] = abs_cur - abs_prev;
  }
}

__global__ void kernel_gather_uncomp(string_chunk_desc const* __restrict__ descs,
                                     int32_t const* __restrict__ d_offsets,
                                     uint8_t* __restrict__ d_chars,
                                     uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t sm_ok;
  __shared__ uint32_t sm_dict_end;
  if (threadIdx.x == 0) {
    uint32_t end_row = desc.seg_row_start + desc.row_count;
    sm_ok = (limit >= 8u + size_t{end_row} * 4u) ? 1u : 0u;
    if (sm_ok) {
      uint32_t de;
      memcpy(&de, base + 4, sizeof(de));
      sm_dict_end = de;
    }
  }
  __syncthreads();
  if (!sm_ok) return;  // pass 1 zero-filled, nothing to gather.

  const int32_t* duck_offsets = reinterpret_cast<const int32_t*>(base + 8);
  const uint8_t* dict_end     = base + sm_dict_end;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i    = desc.seg_row_start + i;
    int32_t cur       = duck_offsets[seg_i];
    int32_t prev      = (seg_i > 0) ? duck_offsets[seg_i - 1] : 0;
    uint32_t abs_cur  = static_cast<uint32_t>(cur >= 0 ? cur : -cur);
    uint32_t abs_prev = static_cast<uint32_t>(prev >= 0 ? prev : -prev);
    uint32_t str_len  = abs_cur - abs_prev;

    int32_t out_pos    = d_offsets[desc.global_row_start + i];
    const uint8_t* src = dict_end - abs_cur;
    memcpy(d_chars + out_pos, src, str_len);
  }
}

__global__ void kernel_compute_lengths_dict(string_chunk_desc const* __restrict__ descs,
                                            uint32_t* __restrict__ d_lengths,
                                            uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t  sm_ok;
  __shared__ uint32_t sm_width;
  __shared__ uint32_t sm_idx_buf_off;
  __shared__ uint32_t sm_sel_buf_off;

  if (threadIdx.x == 0) {
    sm_ok = 0;
    if (limit >= sizeof(dict_header_t)) {
      dict_header_t hdr;
      memcpy(&hdr, base, sizeof(hdr));
      bool bounds_ok = hdr.index_buffer_offset + hdr.index_buffer_count * sizeof(uint32_t) <= limit
                    && hdr.dict_end <= limit
                    && hdr.bitpacking_width <= 32u;
      if (bounds_ok) {
        sm_ok           = 1;
        sm_width        = hdr.bitpacking_width;
        sm_idx_buf_off  = hdr.index_buffer_offset;
        sm_sel_buf_off  = sizeof(dict_header_t);
      }
    }
  }
  __syncthreads();

  // Malformed metadata → zero-fill the chunk using the trusted descriptor
  // row count (parsed fields may have been the source of demotion).
  if (!sm_ok) {
    for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0u;
    }
    return;
  }

  uint32_t const* d_sel = reinterpret_cast<uint32_t const*>(base + sm_sel_buf_off);
  uint32_t const* d_idx = reinterpret_cast<uint32_t const*>(base + sm_idx_buf_off);
  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t sel   = unpack_value<uint32_t>(d_sel, seg_i, sm_width);
    uint32_t len   = (sel == 0) ? 0u : (d_idx[sel] - d_idx[sel - 1]);
    d_lengths[desc.global_row_start + i] = len;
  }
}

__global__ void kernel_gather_dict(string_chunk_desc const* __restrict__ descs,
                                   int32_t const* __restrict__ d_offsets,
                                   uint8_t* __restrict__ d_chars,
                                   uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t  sm_ok;
  __shared__ uint32_t sm_width;
  __shared__ uint32_t sm_idx_buf_off;
  __shared__ uint32_t sm_sel_buf_off;
  __shared__ uint32_t sm_dict_end;

  if (threadIdx.x == 0) {
    sm_ok = 0;
    if (limit >= sizeof(dict_header_t)) {
      dict_header_t hdr;
      memcpy(&hdr, base, sizeof(hdr));
      bool bounds_ok = hdr.index_buffer_offset + hdr.index_buffer_count * sizeof(uint32_t) <= limit
                    && hdr.dict_end <= limit
                    && hdr.bitpacking_width <= 32u;
      if (bounds_ok) {
        sm_ok           = 1;
        sm_width        = hdr.bitpacking_width;
        sm_idx_buf_off  = hdr.index_buffer_offset;
        sm_sel_buf_off  = sizeof(dict_header_t);
        sm_dict_end     = hdr.dict_end;
      }
    }
  }
  __syncthreads();
  // Pass 1 already zero-filled lengths on malformed metadata — d_offsets[i+1]
  // == d_offsets[i] for those rows, so there's nothing to write here.
  if (!sm_ok) return;

  uint32_t const* d_sel    = reinterpret_cast<uint32_t const*>(base + sm_sel_buf_off);
  uint32_t const* d_idx    = reinterpret_cast<uint32_t const*>(base + sm_idx_buf_off);
  uint8_t const* dict_end  = base + sm_dict_end;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t sel   = unpack_value<uint32_t>(d_sel, seg_i, sm_width);
    if (sel == 0) continue;
    uint32_t end_off  = d_idx[sel];
    uint32_t str_len  = end_off - d_idx[sel - 1];
    int32_t  out_pos  = d_offsets[desc.global_row_start + i];
    uint8_t const* src = dict_end - end_off;
    memcpy(d_chars + out_pos, src, str_len);
  }
}

//===----------------------------------------------------------------------===//
// FSST codec.
//
// On-disk layout per segment:
//   [header 16B]
//   [bitpacked compressed_lengths at offset 16]
//   [FSST symbol table at hdr.fsst_symbol_table_offset]
//   [compressed chars ending at hdr.dict_end]
//
// Pass 1 is split into A+B (per-segment) and C (chunked):
//   A — unpack compressed_lengths into d_comp_offsets[fsst_base..]
//   B — in-CTA InclusiveSum producing per-row cumulative compressed bytes
//   C — for each row, walk its compressed bytes summing decoded lengths
//
// Why split: A+B has per-segment prefix-sum state that one CTA owns; C is
// the expensive serial byte-scan and benefits from chunking onto more CTAs.
// The same compressed-offset array is reused for the gather pass-2 chunked
// kernel — no recomputation.
//===----------------------------------------------------------------------===//

__global__ void kernel_compute_lengths_fsst(
  string_chunk_desc const* __restrict__ descs,
  uint32_t* __restrict__ /*d_lengths*/,    // written by phase-C, not here
  uint32_t* __restrict__ d_comp_offsets,
  uint32_t const* __restrict__ d_fsst_row_starts,
  fsst_decoder_compact const* __restrict__ /*d_decoders*/,
  uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc      = descs[seg_idx];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t  sm_ok;
  __shared__ uint32_t sm_bp_width;

  if (threadIdx.x == 0) {
    sm_ok = 0;
    if (limit >= sizeof(fsst_header_t)) {
      fsst_header_t hdr;
      memcpy(&hdr, base, sizeof(hdr));
      // hdr.dict_end MUST point into the segment buffer; bitpacking width
      // for compressed lengths is ≤ 32.
      bool bounds_ok = hdr.dict_end <= limit
                    && hdr.fsst_symbol_table_offset < hdr.dict_end
                    && hdr.bitpacking_width <= 32u;
      if (bounds_ok) {
        sm_ok       = 1;
        sm_bp_width = hdr.bitpacking_width;
      }
    }
  }
  __syncthreads();

  uint32_t fsst_base = d_fsst_row_starts[seg_idx];
  uint32_t row_count = desc.row_count;
  uint32_t* my_comp  = d_comp_offsets + fsst_base;

  if (!sm_ok) {
    // Zero the per-segment compressed-offset slice so phase-C reads zeros
    // and emits zero decoded lengths — chars-buffer rows all collapse to
    // empty strings, no OOB reads.
    for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) my_comp[i] = 0u;
    return;
  }

  // Phase A: unpack compressed lengths.
  uint32_t const* packed =
    reinterpret_cast<uint32_t const*>(base + sizeof(fsst_header_t));
  for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
    my_comp[i] = unpack_value<uint32_t>(packed, i, sm_bp_width);
  }
  __syncthreads();

  // Phase B: in-CTA InclusiveSum (multi-pass — each thread scans its slice
  // of `row_count`, then a single BlockScan of per-thread totals composes).
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_DIM>;
  __shared__ typename BlockScanT::TempStorage scan_temp;
  uint32_t chunk_size = (row_count + blockDim.x - 1u) / blockDim.x;
  uint32_t start      = threadIdx.x * chunk_size;
  uint32_t end        = min(start + chunk_size, row_count);
  uint32_t local_sum  = 0;
  for (uint32_t i = start; i < end; ++i) {
    local_sum += my_comp[i];
    my_comp[i] = local_sum;
  }
  uint32_t scanned;
  BlockScanT(scan_temp).ExclusiveSum(local_sum, scanned);
  if (scanned > 0) {
    for (uint32_t i = start; i < end; ++i) my_comp[i] += scanned;
  }
}

__global__ void kernel_compute_lengths_fsst_phase_c(
  fsst_chunk_desc const* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  uint32_t const* __restrict__ d_comp_offsets,
  fsst_decoder_compact const* __restrict__ d_decoders,
  uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t  sm_ok;
  __shared__ uint32_t sm_dict_end;
  __shared__ uint8_t  sm_len[255];

  if (threadIdx.x == 0) {
    sm_ok = 0;
    if (limit >= sizeof(fsst_header_t)) {
      fsst_header_t hdr;
      memcpy(&hdr, base, sizeof(hdr));
      if (hdr.dict_end <= limit) {
        sm_ok       = 1;
        sm_dict_end = hdr.dict_end;
      }
    }
  }
  __syncthreads();

  if (!sm_ok) {
    for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0u;
    }
    return;
  }

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) sm_len[i] = dec.len[i];
  __syncthreads();

  uint8_t  const* dict_end_ptr = base + sm_dict_end;
  uint32_t const* my_comp      = d_comp_offsets + desc.fsst_row_start;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t cum  = my_comp[i];
    uint32_t prev = (i > 0) ? my_comp[i - 1]
                            : (desc.is_first_chunk ? 0u : *(my_comp - 1));
    uint32_t comp_len = cum - prev;
    if (comp_len == 0) {
      d_lengths[desc.global_row_start + i] = 0u;
      continue;
    }
    uint8_t const* comp_ptr = dict_end_ptr - cum;
    uint32_t decomp_len = 0;
    uint32_t pos        = 0;
    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        decomp_len += sm_len[code];
      } else {
        ++pos;
        ++decomp_len;
      }
    }
    d_lengths[desc.global_row_start + i] = decomp_len;
  }
}

__global__ void kernel_gather_fsst_chunked(fsst_chunk_desc const* __restrict__ descs,
                                           int32_t const* __restrict__ d_offsets,
                                           uint8_t* __restrict__ d_chars,
                                           uint32_t const* __restrict__ d_comp_offsets,
                                           fsst_decoder_compact const* __restrict__ d_decoders,
                                           uint32_t num_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= num_chunks) return;
  auto const desc      = descs[cid];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const limit = desc.bytes_size;

  __shared__ uint8_t  sm_ok;
  __shared__ uint32_t sm_dict_end;
  __shared__ uint8_t  sm_len[255];
  __shared__ unsigned long long sm_sym[255];

  if (threadIdx.x == 0) {
    sm_ok = 0;
    if (limit >= sizeof(fsst_header_t)) {
      fsst_header_t hdr;
      memcpy(&hdr, base, sizeof(hdr));
      if (hdr.dict_end <= limit) {
        sm_ok       = 1;
        sm_dict_end = hdr.dict_end;
      }
    }
  }
  __syncthreads();
  if (!sm_ok) return;  // pass-1 emitted zero lengths → nothing to gather

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) {
    sm_len[i] = dec.len[i];
    sm_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  uint8_t  const* dict_end_ptr = base + sm_dict_end;
  uint32_t const* my_comp      = d_comp_offsets + desc.fsst_row_start;

  // Warp-cooperative decode: 1 warp per row, 32 lanes load 32 compressed
  // bytes in one coalesced gmem transaction, then run a warp-scan to compute
  // per-lane output offsets. Each lane emits its decoded symbol bytes (or
  // literal byte) at its computed offset. Beats the per-thread byte-by-byte
  // loop by ~10-30× for typical row shapes because (a) gmem loads coalesce
  // 32×, (b) per-iteration work parallelizes 32×, (c) the
  // sequential-output-position dependency is broken by the warp-scan.
  //
  // BLOCK_DIM=256 → 8 warps per CTA → 8 rows decoded in parallel per CTA.
  uint32_t const lane    = threadIdx.x & 0x1Fu;
  uint32_t const warp_id = threadIdx.x >> 5;
  uint32_t const warps_per_cta = blockDim.x >> 5;

  for (uint32_t i = warp_id; i < desc.row_count; i += warps_per_cta) {
    uint32_t cum  = my_comp[i];
    uint32_t prev = (i > 0) ? my_comp[i - 1]
                            : (desc.is_first_chunk ? 0u : *(my_comp - 1));
    uint32_t comp_len = cum - prev;
    if (comp_len == 0) continue;

    uint8_t const* comp_ptr = dict_end_ptr - cum;
    uint32_t out_base = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);

    uint32_t output_running          = 0;
    uint32_t prev_chunk_last_is_esc  = 0;  // carries escape state across 32-byte chunks

    for (uint32_t off = 0; off < comp_len; off += 32u) {
      uint32_t bytes_in_chunk = comp_len - off;
      if (bytes_in_chunk > 32u) bytes_in_chunk = 32u;

      // Coalesced 32-byte load: lane t picks up byte t of this chunk.
      uint8_t my_byte = (lane < bytes_in_chunk) ? comp_ptr[off + lane] : 0u;
      uint32_t active = (lane < bytes_in_chunk) ? 1u : 0u;
      uint32_t is_esc = (active && my_byte == FSST_ESC) ? 1u : 0u;

      // For lane t > 0, "prev byte was escape" comes from lane t-1's is_esc.
      // For lane 0, it comes from the previous chunk's last-byte carry.
      uint32_t neighbor_esc = __shfl_up_sync(0xFFFFFFFFu, is_esc, 1);
      uint32_t prev_was_esc = (lane == 0) ? prev_chunk_last_is_esc : neighbor_esc;

      // A position contributes:
      //   1 byte (the literal byte itself) iff the previous byte was an escape sentinel
      //   sm_len[code] bytes iff this byte is neither escape nor a literal
      //   0 bytes iff this byte IS the escape sentinel
      uint32_t my_len = 0u;
      unsigned long long my_sym = 0ull;
      uint32_t my_emit_kind = 0u;  // 0 = nothing, 1 = code, 2 = literal byte
      if (active) {
        if (prev_was_esc) {
          my_len = 1u;
          my_sym = static_cast<unsigned long long>(my_byte);
          my_emit_kind = 2u;
        } else if (!is_esc) {
          uint8_t code = my_byte;
          my_len = sm_len[code];
          my_sym = sm_sym[code];
          my_emit_kind = 1u;
        }
      }

      // Inclusive warp-scan over my_len → per-lane prefix.
      uint32_t scan = my_len;
      #pragma unroll
      for (uint32_t step = 1u; step < 32u; step <<= 1) {
        uint32_t add = __shfl_up_sync(0xFFFFFFFFu, scan, step);
        if (lane >= step) scan += add;
      }
      uint32_t exclusive = scan - my_len;
      uint32_t my_out_pos = out_base + output_running + exclusive;

      // Sized store of this lane's contribution.
      if (my_emit_kind == 1u) {
        switch (my_len) {
          case 1: d_chars[my_out_pos] = static_cast<uint8_t>(my_sym); break;
          case 2: memcpy(d_chars + my_out_pos, &my_sym, 2); break;
          case 3: memcpy(d_chars + my_out_pos, &my_sym, 3); break;
          case 4: memcpy(d_chars + my_out_pos, &my_sym, 4); break;
          default: memcpy(d_chars + my_out_pos, &my_sym, my_len); break;
        }
      } else if (my_emit_kind == 2u) {
        d_chars[my_out_pos] = static_cast<uint8_t>(my_sym);
      }

      // Advance running output by total decoded bytes in this chunk.
      uint32_t chunk_total = __shfl_sync(0xFFFFFFFFu, scan, 31);
      output_running += chunk_total;

      // Carry: did this chunk's last active byte sit at an escape sentinel?
      // If so, the literal lives at the start of the next chunk.
      uint32_t last_active_lane = bytes_in_chunk - 1u;
      uint32_t last_is_esc = __shfl_sync(0xFFFFFFFFu, is_esc, last_active_lane);
      prev_chunk_last_is_esc = last_is_esc;
    }
  }
}

//===----------------------------------------------------------------------===//
// DICT_FSST codec.
//
// On-disk layout per segment (each region 8-byte aligned):
//   [header 16B]
//   [dict bytes               raw or FSST-compressed]
//   [FSST symbol table        only if mode != DICTIONARY]
//   [bitpacked string_lengths dict_count entries]
//   [bitpacked dict_indices   absent for FSST_ONLY; row i → entry i+1 there]
//
// dict_idx == 0 is reserved for NULL (length 0). DuckDB writes
// COMPRESSION_EMPTY for the validity segment in that case, so the inline
// NULL has to be propagated to the column null mask via mark_nulls below.
//
// Per-segment host work (orchestrator) precomputes byte offsets and decoded
// offsets for the segment's dict entries; the kernel just looks them up.
// One CTA per segment; the per-row gather is grid-stride within the CTA.
//===----------------------------------------------------------------------===//

__global__ void kernel_compute_lengths_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                                 uint32_t* __restrict__ d_lengths,
                                                 uint32_t const* __restrict__ d_decoded_offsets,
                                                 uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc        = descs[seg_idx];
  uint32_t const* dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
  uint32_t const* d_idx   = reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);
  bool const fsst_only    = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx   = fsst_only ? (seg_i + 1u)
                               : unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    uint32_t len = (idx == 0u) ? 0u : (dec_off[idx + 1] - dec_off[idx]);
    d_lengths[desc.global_row_start + i] = len;
  }
}

/// One-shot dict-entry predecode for DICT_FSST mode 1. Each thread takes one
/// dict entry (or strides over `dict_count`), FSST-decompresses its bytes
/// once into the global `predecode_buf`. The gather kernel then turns into a
/// pure memcpy from this buffer for every row that selects the same entry.
///
/// Without this, `kernel_gather_dict_fsst` re-decompresses the same entry
/// once per row that selects it. For typical TPC-H l_comment shapes
/// (~256 dict entries, ~1M rows), that's thousands of redundant FSST walks
/// per entry — predecode collapses it to one walk.
__global__ void kernel_predecode_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                           uint32_t const* __restrict__ d_byte_offsets,
                                           uint32_t const* __restrict__ d_decoded_offsets,
                                           fsst_decoder_compact const* __restrict__ d_decoders,
                                           uint8_t* __restrict__ predecode_buf,
                                           uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc = descs[seg_idx];
  if (desc.mode != DICT_FSST_MODE_DICT_FSST) return;
  if (desc.dict_count <= 1u) return;  // only entry 0 = NULL, nothing to decode

  __shared__ uint8_t            sm_len[255];
  __shared__ unsigned long long sm_sym[255];

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) {
    sm_len[i] = dec.len[i];
    sm_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  uint8_t  const* dict_data  = desc.d_bytes + desc.dict_data_offset;
  uint32_t const* byte_off   = d_byte_offsets    + desc.seg_dict_offset_base;
  uint32_t const* dec_off    = d_decoded_offsets + desc.seg_dict_offset_base;
  uint8_t* out_base = predecode_buf + desc.predecode_seg_offset;

  // Skip k=0 (reserved NULL slot, length 0). Threads cooperate over the
  // remaining entries — one thread per dict entry.
  for (uint32_t k = threadIdx.x + 1u; k < desc.dict_count; k += blockDim.x) {
    uint32_t comp_start = byte_off[k];
    uint32_t comp_end   = byte_off[k + 1];
    uint32_t comp_len   = comp_end - comp_start;
    uint32_t out_pos    = dec_off[k];

    uint8_t const* comp_ptr = dict_data + comp_start;
    uint8_t*       out_ptr  = out_base + out_pos;
    uint32_t pos = 0;
    uint32_t op  = 0;
    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        unsigned long long sym = sm_sym[code];
        uint8_t            sym_len = sm_len[code];
        switch (sym_len) {
          case 1: out_ptr[op] = static_cast<uint8_t>(sym); break;
          case 2: memcpy(out_ptr + op, &sym, 2); break;
          case 3: memcpy(out_ptr + op, &sym, 3); break;
          case 4: memcpy(out_ptr + op, &sym, 4); break;
          default: memcpy(out_ptr + op, &sym, sym_len); break;
        }
        op += sym_len;
      } else {
        out_ptr[op++] = comp_ptr[pos++];
      }
    }
  }
}

__global__ void kernel_gather_dict_fsst(dict_fsst_desc const* __restrict__ descs,
                                        int32_t const* __restrict__ d_offsets,
                                        uint8_t* __restrict__ d_chars,
                                        uint32_t const* __restrict__ d_byte_offsets,
                                        uint32_t const* __restrict__ d_decoded_offsets,
                                        uint8_t const* __restrict__ predecode_buf,
                                        fsst_decoder_compact const* __restrict__ d_decoders,
                                        uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc      = descs[seg_idx];
  uint8_t const* base  = desc.d_bytes;
  uint32_t const* dict_byte_off = d_byte_offsets + desc.seg_dict_offset_base;
  uint32_t const* d_idx = reinterpret_cast<uint32_t const*>(base + desc.dict_indices_offset);
  bool const fsst_only  = (desc.mode == DICT_FSST_MODE_FSST_ONLY);
  bool const has_fsst   = (desc.mode != DICT_FSST_MODE_DICTIONARY);

  __shared__ uint8_t            sm_len[255];
  __shared__ unsigned long long sm_sym[255];

  if (has_fsst) {
    fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
    for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) {
      sm_len[i] = dec.len[i];
      sm_sym[i] = dec.symbol[i];
    }
    __syncthreads();
  }

  uint8_t const*  dict_data = base + desc.dict_data_offset;
  uint32_t const* dict_dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
  uint8_t  const* predecode_seg = predecode_buf + desc.predecode_seg_offset;
  bool const mode_dict_fsst = (desc.mode == DICT_FSST_MODE_DICT_FSST);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx   = fsst_only ? (seg_i + 1u)
                               : unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx == 0u) continue;  // NULL row — pass-1 already emitted length 0

    int32_t  base_pos   = d_offsets[desc.global_row_start + i];
    uint32_t op         = static_cast<uint32_t>(base_pos);

    if (!has_fsst) {
      // DICTIONARY mode — raw dict bytes, straight memcpy.
      uint32_t byte_start = dict_byte_off[idx];
      uint32_t comp_len   = dict_byte_off[idx + 1] - byte_start;
      memcpy(d_chars + op, dict_data + byte_start, comp_len);
      continue;
    }

    if (mode_dict_fsst) {
      // Mode 1: dict was predecoded once into `predecode_seg`. Per-row work
      // is now a vectorized memcpy from the precomputed decoded bytes.
      // Explicit 8-byte chunked copy forces NVCC to emit ld.global.b64 +
      // st.global.b64 instead of byte-by-byte; ~3-4× higher write throughput
      // on the hot path for ≥16-byte rows (typical TPC-H l_comment shape).
      uint32_t dec_start = dict_dec_off[idx];
      uint32_t dec_len   = dict_dec_off[idx + 1] - dec_start;
      uint8_t const* src = predecode_seg + dec_start;
      uint8_t*       dst = d_chars + op;
      uint32_t copied = 0;
      while (copied + 8u <= dec_len) {
        unsigned long long v;
        memcpy(&v, src + copied, 8);
        memcpy(dst + copied, &v, 8);
        copied += 8;
      }
      if (copied + 4u <= dec_len) {
        uint32_t v;
        memcpy(&v, src + copied, 4);
        memcpy(dst + copied, &v, 4);
        copied += 4;
      }
      while (copied < dec_len) {
        dst[copied] = src[copied];
        copied++;
      }
      continue;
    }

    // Mode 2 (FSST_ONLY) — inline FSST decompress per dict entry. No
    // predecode here because dict_count == row_count + 1; predecode would
    // do the same total work in a different kernel.
    uint32_t byte_start = dict_byte_off[idx];
    uint32_t comp_len   = dict_byte_off[idx + 1] - byte_start;
    uint8_t const* comp_ptr = dict_data + byte_start;
    uint32_t pos            = 0;
    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        unsigned long long sym = sm_sym[code];
        uint8_t            sym_len = sm_len[code];
        switch (sym_len) {
          case 1: d_chars[op] = static_cast<uint8_t>(sym); break;
          case 2: memcpy(d_chars + op, &sym, 2); break;
          case 3: memcpy(d_chars + op, &sym, 3); break;
          case 4: memcpy(d_chars + op, &sym, 4); break;
          default: memcpy(d_chars + op, &sym, sym_len); break;
        }
        op += sym_len;
      } else {
        d_chars[op++] = comp_ptr[pos++];
      }
    }
  }
}

/// Write the sentinel offset (offsets[total_rows] = sum of all lengths) into
/// the prefix-sum output. Avoids a host round-trip — without this we'd have
/// to read total_chars back to host before the gather pass.
__global__ void kernel_write_offsets_sentinel(uint32_t* __restrict__ d_offsets_u32,
                                              uint32_t const* __restrict__ d_lengths,
                                              uint32_t total_rows)
{
  if (threadIdx.x == 0 && blockIdx.x == 0 && total_rows > 0) {
    d_offsets_u32[total_rows] = d_offsets_u32[total_rows - 1] + d_lengths[total_rows - 1];
  }
}

/// DICT_FSST inline-NULL pass: clear null_mask bits for rows where the
/// dictionary index is 0. Validity-segment overlay missed these because
/// DuckDB ships an "Empty Validity" segment (skipped by decode_validity).
///
/// `d_mask` points at the column's null_mask buffer in cudf bitmask layout.
/// Bit ordering is LSB-first within byte: clearing bit k of byte (row/8)
/// marks row `row` as NULL when AND-combined with the all-valid base.
__global__ void kernel_dict_fsst_mark_nulls(dict_fsst_desc const* __restrict__ descs,
                                            uint8_t* __restrict__ d_mask,
                                            uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc       = descs[seg_idx];
  uint32_t const* d_idx = reinterpret_cast<uint32_t const*>(desc.d_bytes + desc.dict_indices_offset);
  bool const fsst_only  = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  // FSST_ONLY can't encode NULL via idx==0 (row i → entry i+1 always non-zero).
  if (fsst_only) return;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx   = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx != 0u) continue;
    uint32_t row    = desc.global_row_start + i;
    uint32_t byte_i = row >> 3;
    uint8_t  bit    = static_cast<uint8_t>(1u << (row & 7u));
    atomicAnd(reinterpret_cast<unsigned int*>(&reinterpret_cast<uint32_t*>(d_mask)[byte_i >> 2]),
              ~(static_cast<unsigned int>(bit) << ((byte_i & 3u) * 8u)));
  }
}

//===----------------------------------------------------------------------===//
// Validity overlay. Mirrors `dispatch_validity_run` in gpu_native_decode.cu;
// we don't share the function because the strings orchestrator wants to
// also compose with the DICT_FSST mark-nulls pass under the same null mask.
//===----------------------------------------------------------------------===//

void overlay_validity_run(gpu_codec_run const& run, uint8_t* d_mask, rmm::cuda_stream_view stream)
{
  if (run.codec != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
    throw std::runtime_error("gpu_decode_strings_column: viability invariant violated — "
                             "validity codec " +
                             std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.row_offset % 8 != 0) {
      throw std::runtime_error("gpu_decode_strings_column: validity row_offset (" +
                               std::to_string(seg.row_offset) + ") not byte-aligned");
    }
    size_t bytes = (size_t{seg.row_count} + 7) / 8;
    if (size_t{seg.bytes_size} < bytes) {
      throw std::runtime_error("gpu_decode_strings_column: validity segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(bytes));
    }
    RMM_CUDA_TRY(cudaMemcpyAsync(
      d_mask + seg.row_offset / 8, seg.d_bytes, bytes, cudaMemcpyDeviceToDevice, stream.value()));
  }
}

//===----------------------------------------------------------------------===//
// Per-codec host-side preparation. Each block walks the input runs, parses
// just enough metadata host-side to (a) build per-codec descriptors, (b)
// upper-bound the chars allocation, and (c) deserialize FSST decoders and
// compute per-dict-entry decoded offsets for DICT_FSST.
//===----------------------------------------------------------------------===//

struct prepared_uncomp {
  std::vector<string_chunk_desc> descs;
};

struct prepared_dict {
  std::vector<string_chunk_desc> descs;
};

struct prepared_fsst {
  std::vector<string_chunk_desc>   length_descs;       ///< pass-1 A+B (per segment)
  std::vector<fsst_chunk_desc>     gather_chunks;      ///< pass-1 phase-C + pass-2 chunks
  std::vector<fsst_decoder_compact> decoders;
  std::vector<uint32_t>            row_starts;         ///< prefix sum of FSST row counts
  uint32_t total_fsst_rows;
};

struct prepared_dict_fsst {
  std::vector<dict_fsst_desc>       descs;
  std::vector<fsst_decoder_compact> decoders;
  std::vector<uint32_t>             byte_offsets;     ///< per-segment, dict_count+1 entries each
  std::vector<uint32_t>             decoded_offsets;  ///< per-segment, dict_count+1 entries each
  bool                              any_inline_nulls;
  uint32_t                          total_predecode_bytes;  ///< sum of mode-1 dict-decoded bytes
};

prepared_uncomp prepare_uncomp(gpu_string_codec_run const& run)
{
  prepared_uncomp out;
  out.descs.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    out.descs.push_back({seg.d_bytes,
                         seg.bytes_size,
                         seg.row_count,
                         seg.row_offset,
                         seg.seg_row_start});
  }
  return out;
}

prepared_dict prepare_dict(gpu_string_codec_run const& run)
{
  prepared_dict out;
  out.descs.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    out.descs.push_back({seg.d_bytes,
                         seg.bytes_size,
                         seg.row_count,
                         seg.row_offset,
                         seg.seg_row_start});
  }
  return out;
}

prepared_fsst prepare_fsst(gpu_string_codec_run const& run, rmm::cuda_stream_view stream)
{
  (void)stream;
  prepared_fsst out;
  out.total_fsst_rows = 0;
  out.length_descs.reserve(run.segments.size());
  out.decoders.reserve(run.segments.size());
  out.row_starts.reserve(run.segments.size());

  for (size_t si = 0; si < run.segments.size(); ++si) {
    auto const& seg = run.segments[si];
    if (seg.row_count == 0) continue;
    out.row_starts.push_back(out.total_fsst_rows);
    out.total_fsst_rows += seg.row_count;
    out.length_descs.push_back({seg.d_bytes,
                                seg.bytes_size,
                                seg.row_count,
                                seg.row_offset,
                                seg.seg_row_start});

    // Per-segment header + symbol-table read. Synchronous cudaMemcpy is the
    // right tradeoff for typical segment counts: cudaMemcpy is ~5 µs/call
    // (vs ~1-3 ms for cudaMallocHost which would be needed for true async
    // pipelining via pinned host scratch). For columns with many segments,
    // a stacked follow-up could replace this with a pre-allocated persistent
    // pinned-host pool. Today's bench shape (single-segment 1M row) is
    // dominated by kernel time, not host-prep.
    fsst_header_t hdr{};
    if (seg.bytes_size < sizeof(hdr)) {
      out.decoders.emplace_back();
      continue;
    }
    RMM_CUDA_TRY(cudaMemcpy(&hdr, seg.d_bytes, sizeof(hdr), cudaMemcpyDeviceToHost));
    if (hdr.fsst_symbol_table_offset >= seg.bytes_size ||
        hdr.fsst_symbol_table_offset >= hdr.dict_end ||
        hdr.dict_end > seg.bytes_size) {
      out.decoders.emplace_back();
      continue;
    }
    size_t symtab_max = std::min<size_t>(seg.bytes_size - hdr.fsst_symbol_table_offset, 8192u);
    std::vector<uint8_t> symtab(symtab_max);
    RMM_CUDA_TRY(cudaMemcpy(symtab.data(),
                            seg.d_bytes + hdr.fsst_symbol_table_offset,
                            symtab_max,
                            cudaMemcpyDeviceToHost));
    fsst_decoder_full full{};
    duckdb_fsst_import(&full, symtab.data());
    fsst_decoder_compact compact;
    std::memcpy(compact.len,    full.len,    sizeof(compact.len));
    std::memcpy(compact.symbol, full.symbol, sizeof(compact.symbol));
    out.decoders.push_back(compact);
  }

  // Build the FSST-chunked descriptors used by phase-C and gather. Total
  // FSST rows < target_ctas → split each segment to fill SMs; otherwise one
  // chunk per segment (the original shape).
  uint32_t target_ctas      = get_target_ctas();
  uint32_t chunk_size_fsst  = 0;
  if (out.length_descs.size() < target_ctas && out.total_fsst_rows > 0) {
    chunk_size_fsst = std::max(out.total_fsst_rows / target_ctas, MIN_CHUNK_ROW);
    chunk_size_fsst = (chunk_size_fsst / 32u) * 32u;
    if (chunk_size_fsst == 0) chunk_size_fsst = 32u;
  }
  for (size_t si = 0; si < out.length_descs.size(); ++si) {
    auto const& seg        = out.length_descs[si];
    uint32_t fsst_base_row = out.row_starts[si];
    if (chunk_size_fsst == 0) {
      out.gather_chunks.push_back({seg.d_bytes,
                                   seg.bytes_size,
                                   seg.row_count,
                                   seg.global_row_start,
                                   fsst_base_row,
                                   static_cast<uint32_t>(si),
                                   1u});
    } else {
      uint32_t remaining = seg.row_count;
      uint32_t off       = 0;
      bool first         = true;
      while (remaining > 0) {
        uint32_t n = std::min(remaining, chunk_size_fsst);
        out.gather_chunks.push_back({seg.d_bytes,
                                     seg.bytes_size,
                                     n,
                                     seg.global_row_start + off,
                                     fsst_base_row + off,
                                     static_cast<uint32_t>(si),
                                     first ? 1u : 0u});
        off += n;
        remaining -= n;
        first = false;
      }
    }
  }
  return out;
}

prepared_dict_fsst prepare_dict_fsst(gpu_string_codec_run const& run, rmm::cuda_stream_view stream)
{
  (void)stream;
  prepared_dict_fsst out;
  out.any_inline_nulls = false;
  out.total_predecode_bytes = 0;
  out.descs.reserve(run.segments.size());
  out.decoders.reserve(run.segments.size());

  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.bytes_size < sizeof(dict_fsst_header_t)) {
      // Malformed — emit a stub descriptor that pass-1 will zero-fill.
      out.descs.push_back({seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset,
                           seg.seg_row_start, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, {0,0,0,0,0,0}});
      continue;
    }

    // Read the header + bitpacked-lengths region host-side. Each segment ≤
    // 256 KiB so a synchronous whole-segment cudaMemcpy is the simplest
    // correct path. Per-call overhead is ~5-10 µs which is dwarfed by the
    // kernel decode for any row count that's worth the trip.
    std::vector<uint8_t> host_bytes_vec(seg.bytes_size);
    uint8_t* host_bytes = host_bytes_vec.data();
    RMM_CUDA_TRY(cudaMemcpy(host_bytes, seg.d_bytes, seg.bytes_size,
                            cudaMemcpyDeviceToHost));
    dict_fsst_header_t hdr;
    std::memcpy(&hdr, host_bytes, sizeof(hdr));

    if (hdr.mode > DICT_FSST_MODE_FSST_ONLY) {
      out.descs.push_back({seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset,
                           seg.seg_row_start, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, {0,0,0,0,0,0}});
      continue;
    }

    // Region offsets — each region is 8-byte aligned per DuckDB AlignValue.
    uint32_t off_dict   = align_up8(static_cast<uint32_t>(sizeof(hdr)));
    uint32_t off_symtab = align_up8(off_dict + hdr.dict_size);
    uint32_t off_slens  = (hdr.mode == DICT_FSST_MODE_DICTIONARY)
                            ? off_dict + align_up8(hdr.dict_size)
                            : align_up8(off_symtab + hdr.symbol_table_size);
    // dict_count includes reserved idx 0 (length 0).
    uint32_t slens_bits = hdr.dict_count * hdr.string_lengths_width;
    uint32_t off_didx   = align_up8(off_slens + (slens_bits + 7u) / 8u);

    if (off_didx > seg.bytes_size && hdr.mode != DICT_FSST_MODE_FSST_ONLY) {
      out.descs.push_back({seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset,
                           seg.seg_row_start, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, {0,0,0,0,0,0}});
      continue;
    }

    // Unpack per-entry string lengths (compressed lengths in DICT_FSST modes,
    // raw decoded lengths in DICTIONARY mode) into a host scratch vector.
    std::vector<uint32_t> entry_lens(hdr.dict_count, 0);
    uint8_t const* slens_ptr = host_bytes + off_slens;
    for (uint32_t k = 0; k < hdr.dict_count; ++k) {
      entry_lens[k] = host_unpack_bitpacked<uint32_t>(slens_ptr, k, hdr.string_lengths_width);
    }

    // byte_offsets[k] = cumulative compressed-byte position of dict entry k
    // within the dict region. Always uses entry_lens regardless of mode.
    uint32_t base_off = static_cast<uint32_t>(out.byte_offsets.size());
    out.byte_offsets.resize(base_off + hdr.dict_count + 1u);
    out.byte_offsets[base_off] = 0u;
    for (uint32_t k = 0; k < hdr.dict_count; ++k) {
      out.byte_offsets[base_off + k + 1] = out.byte_offsets[base_off + k] + entry_lens[k];
    }

    // decoded_offsets[k] = cumulative decoded-byte position. For DICTIONARY
    // mode the dict bytes are raw, so decoded_offsets == byte_offsets. For
    // FSST modes we walk each entry's compressed bytes counting decoded
    // length via the imported decoder.
    fsst_decoder_compact compact{};
    if (hdr.mode != DICT_FSST_MODE_DICTIONARY) {
      fsst_decoder_full full{};
      duckdb_fsst_import(&full, host_bytes + off_symtab);
      std::memcpy(compact.len,    full.len,    sizeof(compact.len));
      std::memcpy(compact.symbol, full.symbol, sizeof(compact.symbol));
    }
    out.decoders.push_back(compact);

    out.decoded_offsets.resize(base_off + hdr.dict_count + 1u);
    out.decoded_offsets[base_off] = 0u;
    if (hdr.mode == DICT_FSST_MODE_DICTIONARY) {
      // Raw dict — decoded == compressed.
      for (uint32_t k = 0; k <= hdr.dict_count; ++k) {
        out.decoded_offsets[base_off + k] = out.byte_offsets[base_off + k];
      }
    } else {
      uint8_t const* dict_bytes = host_bytes + off_dict;
      for (uint32_t k = 0; k < hdr.dict_count; ++k) {
        size_t comp_start = out.byte_offsets[base_off + k];
        size_t comp_len   = entry_lens[k];
        size_t dec_len    = (k == 0) ? 0
                                     : fsst_decompressed_length_host(
                                         compact, dict_bytes + comp_start, comp_len);
        out.decoded_offsets[base_off + k + 1] =
          out.decoded_offsets[base_off + k] + static_cast<uint32_t>(dec_len);
      }
    }

    // DICT_FSST modes ship "Empty Validity" when nulls are encoded inline.
    out.any_inline_nulls = out.any_inline_nulls
                           || (hdr.mode != DICT_FSST_MODE_FSST_ONLY && hdr.dict_count > 1
                               && entry_lens[0] == 0u);

    // For mode 1 (DICT_FSST), record the predecode-buffer offset for this
    // segment and reserve space for its decoded dict bytes. Modes 0 and 2 do
    // not use the predecode path (mode 0 = raw memcpy from dict; mode 2 =
    // per-row inline FSST decompress because each row is a unique entry).
    uint32_t predecode_off = 0u;
    if (hdr.mode == DICT_FSST_MODE_DICT_FSST) {
      predecode_off = out.total_predecode_bytes;
      out.total_predecode_bytes += out.decoded_offsets[base_off + hdr.dict_count];
    }

    out.descs.push_back({seg.d_bytes,
                         seg.bytes_size,
                         seg.row_count,
                         seg.row_offset,
                         seg.seg_row_start,
                         off_dict,
                         (hdr.mode == DICT_FSST_MODE_FSST_ONLY) ? 0u : off_didx,
                         base_off,
                         static_cast<uint32_t>(out.decoders.size() - 1u),
                         hdr.dict_count,
                         predecode_off,
                         hdr.dictionary_indices_width,
                         hdr.mode,
                         {0, 0, 0, 0, 0, 0}});
  }
  return out;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public orchestrator.
//
// Walks col.data once to prepare per-codec descriptor arrays, runs the two-
// pass shape, builds the cudf strings column. The chars buffer is sized
// from a host upper bound so the pass-1 → pass-2 boundary doesn't sync.
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> gpu_decode_strings_column(
  gpu_string_column_decode_input const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const total_rows = col.total_rows;
  if (total_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }
  if (total_rows > static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error("gpu_decode_strings_column: total_rows (" +
                             std::to_string(total_rows) + ") > cudf::size_type max");
  }

  // Per-codec preparation + per-codec viability gate. Anything other than
  // the four supported codecs throws — viability is upstream's job, this is
  // a defensive backstop.
  prepared_uncomp     prep_uncomp;
  prepared_dict       prep_dict;
  prepared_fsst       prep_fsst;
  prepared_dict_fsst  prep_dict_fsst;
  prep_dict_fsst.any_inline_nulls = false;
  prep_dict_fsst.total_predecode_bytes = 0;
  size_t cum_chars_upper = 0;
  for (auto const& run : col.data) {
    switch (run.codec) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        auto p = prepare_uncomp(run);
        for (auto& d : p.descs) prep_uncomp.descs.push_back(d);
        break;
      }
      case duckdb::CompressionType::COMPRESSION_DICTIONARY: {
        auto p = prepare_dict(run);
        for (auto& d : p.descs) prep_dict.descs.push_back(d);
        break;
      }
      case duckdb::CompressionType::COMPRESSION_FSST: {
        auto p = prepare_fsst(run, stream);
        // Caller-side concat: rebase row_starts into the merged FSST set.
        uint32_t base = prep_fsst.total_fsst_rows;
        for (auto& d : p.length_descs) prep_fsst.length_descs.push_back(d);
        for (auto& s : p.row_starts)   prep_fsst.row_starts.push_back(base + s);
        for (auto& d : p.decoders)     prep_fsst.decoders.push_back(d);
        for (auto& c : p.gather_chunks) {
          c.fsst_row_start += base;
          c.seg_decoder_idx += static_cast<uint32_t>(prep_fsst.decoders.size() - p.decoders.size());
          prep_fsst.gather_chunks.push_back(c);
        }
        prep_fsst.total_fsst_rows += p.total_fsst_rows;
        break;
      }
      case duckdb::CompressionType::COMPRESSION_DICT_FSST: {
        auto p = prepare_dict_fsst(run, stream);
        uint32_t bo_base = static_cast<uint32_t>(prep_dict_fsst.byte_offsets.size());
        uint32_t dec_base = static_cast<uint32_t>(prep_dict_fsst.decoders.size());
        uint32_t predecode_base = prep_dict_fsst.total_predecode_bytes;
        for (auto v : p.byte_offsets)    prep_dict_fsst.byte_offsets.push_back(v);
        for (auto v : p.decoded_offsets) prep_dict_fsst.decoded_offsets.push_back(v);
        for (auto& d : p.decoders)       prep_dict_fsst.decoders.push_back(d);
        for (auto& d : p.descs) {
          d.seg_dict_offset_base += bo_base;
          d.seg_decoder_idx      += dec_base;
          if (d.mode == DICT_FSST_MODE_DICT_FSST) {
            d.predecode_seg_offset += predecode_base;
          }
          prep_dict_fsst.descs.push_back(d);
        }
        prep_dict_fsst.any_inline_nulls = prep_dict_fsst.any_inline_nulls || p.any_inline_nulls;
        prep_dict_fsst.total_predecode_bytes += p.total_predecode_bytes;
        break;
      }
      default:
        throw std::runtime_error("gpu_decode_strings_column: viability invariant violated — "
                                 "data codec " +
                                 std::to_string(static_cast<int>(run.codec)) + " not implemented");
    }
    // Upper-bound chars contribution per run, using the walker's per-segment
    // `max_string_length` stat (carried by PR 6 walker → PR 9 converter →
    // here). A single segment with `max_string_length == 0` (unknown) forces
    // the post-prefix-sum sync path below — better to take one D2H than to
    // over-allocate by orders of magnitude.
    for (auto const& seg : run.segments) {
      if (seg.max_string_length == 0u) { cum_chars_upper = SIZE_MAX; break; }
      cum_chars_upper += size_t{seg.row_count} * seg.max_string_length;
    }
  }

  // Allocate work buffers.
  rmm::device_uvector<uint32_t> d_lengths(total_rows, stream, mr);
  rmm::device_uvector<int32_t>  d_offsets(size_t{total_rows} + 1u, stream, mr);
  // Lengths buffer must be zero-initialised — codecs that "skip" a row
  // (NULL via idx==0) leave their slot untouched, and the prefix sum needs
  // a known starting value.
  RMM_CUDA_TRY(cudaMemsetAsync(d_lengths.data(), 0,
                               size_t{total_rows} * sizeof(uint32_t),
                               stream.value()));

  // FSST shared comp_offsets buffer (exists only when there are FSST rows).
  rmm::device_buffer comp_offsets_buf(
    prep_fsst.total_fsst_rows > 0 ? prep_fsst.total_fsst_rows * sizeof(uint32_t) : 0,
    stream, mr);
  uint32_t* d_comp_offsets = static_cast<uint32_t*>(comp_offsets_buf.data());

  auto upload = [&](void const* src, size_t bytes) {
    rmm::device_buffer buf(bytes, stream, mr);
    if (bytes > 0) {
      RMM_CUDA_TRY(cudaMemcpyAsync(buf.data(), src, bytes, cudaMemcpyHostToDevice, stream.value()));
    }
    return buf;
  };

  // Expand DICTIONARY + DICT_FSST descriptors to chunks for SM-fill. The
  // DICT_FSST chunked array is used by the per-row compute_lengths + gather
  // kernels; the un-chunked `prep_dict_fsst.descs` stays for predecode and
  // mark_nulls (per-segment work that should NOT be duplicated per chunk).
  uint32_t target_ctas  = get_target_ctas();
  auto uncomp_chunks    = expand_chunks(prep_uncomp.descs, target_ctas);
  auto dict_chunks      = expand_chunks(prep_dict.descs, target_ctas);
  auto dict_fsst_chunks = expand_chunks_dict_fsst(prep_dict_fsst.descs, target_ctas);

  rmm::device_buffer d_uncomp_chunks_buf =
    upload(uncomp_chunks.data(), uncomp_chunks.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_chunks_buf =
    upload(dict_chunks.data(), dict_chunks.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_dict_fsst_chunks_buf =
    upload(dict_fsst_chunks.data(), dict_fsst_chunks.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_fsst_lengths_buf =
    upload(prep_fsst.length_descs.data(),
           prep_fsst.length_descs.size() * sizeof(string_chunk_desc));
  rmm::device_buffer d_fsst_chunks_buf =
    upload(prep_fsst.gather_chunks.data(),
           prep_fsst.gather_chunks.size() * sizeof(fsst_chunk_desc));
  rmm::device_buffer d_fsst_starts_buf =
    upload(prep_fsst.row_starts.data(),
           prep_fsst.row_starts.size() * sizeof(uint32_t));
  rmm::device_buffer d_fsst_decoders_buf =
    upload(prep_fsst.decoders.data(),
           prep_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_dict_fsst_descs_buf =
    upload(prep_dict_fsst.descs.data(),
           prep_dict_fsst.descs.size() * sizeof(dict_fsst_desc));
  rmm::device_buffer d_dict_fsst_decoders_buf =
    upload(prep_dict_fsst.decoders.data(),
           prep_dict_fsst.decoders.size() * sizeof(fsst_decoder_compact));
  rmm::device_buffer d_byte_offsets_buf =
    upload(prep_dict_fsst.byte_offsets.data(),
           prep_dict_fsst.byte_offsets.size() * sizeof(uint32_t));
  rmm::device_buffer d_decoded_offsets_buf =
    upload(prep_dict_fsst.decoded_offsets.data(),
           prep_dict_fsst.decoded_offsets.size() * sizeof(uint32_t));

  auto* d_uncomp_chunks_p = static_cast<string_chunk_desc*>(d_uncomp_chunks_buf.data());
  auto* d_dict_chunks_p   = static_cast<string_chunk_desc*>(d_dict_chunks_buf.data());
  auto* d_fsst_lengths_p  = static_cast<string_chunk_desc*>(d_fsst_lengths_buf.data());
  auto* d_fsst_chunks_p   = static_cast<fsst_chunk_desc*>(d_fsst_chunks_buf.data());
  auto* d_fsst_starts_p   = static_cast<uint32_t*>(d_fsst_starts_buf.data());
  auto* d_fsst_decs_p     = static_cast<fsst_decoder_compact*>(d_fsst_decoders_buf.data());
  auto* d_dict_fsst_p     = static_cast<dict_fsst_desc*>(d_dict_fsst_descs_buf.data());
  auto* d_dict_fsst_chunks_p = static_cast<dict_fsst_desc*>(d_dict_fsst_chunks_buf.data());
  auto* d_dict_fsst_decs_p =
    static_cast<fsst_decoder_compact*>(d_dict_fsst_decoders_buf.data());
  auto* d_byte_off_p     = static_cast<uint32_t*>(d_byte_offsets_buf.data());
  auto* d_decoded_off_p  = static_cast<uint32_t*>(d_decoded_offsets_buf.data());

  // Pass 1: compute lengths.
  if (!uncomp_chunks.empty()) {
    kernel_compute_lengths_uncomp<<<static_cast<uint32_t>(uncomp_chunks.size()),
                                    BLOCK_DIM, 0, stream.value()>>>(
      d_uncomp_chunks_p, d_lengths.data(), static_cast<uint32_t>(uncomp_chunks.size()));
  }
  if (!dict_chunks.empty()) {
    kernel_compute_lengths_dict<<<static_cast<uint32_t>(dict_chunks.size()),
                                  BLOCK_DIM, 0, stream.value()>>>(
      d_dict_chunks_p, d_lengths.data(), static_cast<uint32_t>(dict_chunks.size()));
  }
  if (!prep_fsst.length_descs.empty()) {
    kernel_compute_lengths_fsst<<<static_cast<uint32_t>(prep_fsst.length_descs.size()),
                                  BLOCK_DIM, 0, stream.value()>>>(
      d_fsst_lengths_p, d_lengths.data(), d_comp_offsets, d_fsst_starts_p, d_fsst_decs_p,
      static_cast<uint32_t>(prep_fsst.length_descs.size()));
    kernel_compute_lengths_fsst_phase_c<<<static_cast<uint32_t>(prep_fsst.gather_chunks.size()),
                                          BLOCK_DIM, 0, stream.value()>>>(
      d_fsst_chunks_p, d_lengths.data(), d_comp_offsets, d_fsst_decs_p,
      static_cast<uint32_t>(prep_fsst.gather_chunks.size()));
  }
  // Allocate predecode buffer for DICT_FSST mode-1 segments. Sized at the
  // sum of decoded dict bytes across all such segments — typically tiny
  // (~256 entries × ~30 B / segment ≈ 8 KB per column at TPC-H scale).
  rmm::device_buffer d_predecode_buf(
    prep_dict_fsst.total_predecode_bytes > 0 ? prep_dict_fsst.total_predecode_bytes : 1u,
    stream, mr);
  auto* d_predecode_p = static_cast<uint8_t*>(d_predecode_buf.data());

  if (!prep_dict_fsst.descs.empty()) {
    // compute_lengths is per-row work — chunked descriptors fan out to many
    // CTAs so a single-segment column still fills the GPU.
    kernel_compute_lengths_dict_fsst<<<static_cast<uint32_t>(dict_fsst_chunks.size()),
                                       BLOCK_DIM, 0, stream.value()>>>(
      d_dict_fsst_chunks_p, d_lengths.data(), d_decoded_off_p,
      static_cast<uint32_t>(dict_fsst_chunks.size()));
    // Predecode is per-segment work (one decode per dict entry, regardless
    // of how many rows reference it) — keep one-CTA-per-segment so we don't
    // duplicate the predecode across chunks of the same segment.
    if (prep_dict_fsst.total_predecode_bytes > 0) {
      kernel_predecode_dict_fsst<<<static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                                   BLOCK_DIM, 0, stream.value()>>>(
        d_dict_fsst_p, d_byte_off_p, d_decoded_off_p, d_dict_fsst_decs_p,
        d_predecode_p,
        static_cast<uint32_t>(prep_dict_fsst.descs.size()));
    }
  }

  // Single CUB ExclusiveSum across the whole d_lengths array → d_offsets.
  size_t cub_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                static_cast<int>(total_rows), stream.value());
  rmm::device_buffer cub_temp_buf(cub_bytes, stream, mr);
  cub::DeviceScan::ExclusiveSum(cub_temp_buf.data(), cub_bytes, d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                static_cast<int>(total_rows), stream.value());

  // Sentinel: d_offsets[total_rows] = sum_of_all_lengths. Tiny kernel
  // computing it from d_offsets[total_rows-1] + d_lengths[total_rows-1]
  // avoids a host round-trip on the common path.
  kernel_write_offsets_sentinel<<<1, 1, 0, stream.value()>>>(
    reinterpret_cast<uint32_t*>(d_offsets.data()), d_lengths.data(), total_rows);

  // Chars allocation. Common case: host upper bound below cap → no sync.
  // Fallback: pathological max_string_length → sync once for exact total.
  size_t alloc_chars = 0;
  if (cum_chars_upper <= HOST_UPPER_BOUND_LIMIT) {
    alloc_chars = cum_chars_upper;
  } else {
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    int32_t total_chars_signed = 0;
    RMM_CUDA_TRY(cudaMemcpy(&total_chars_signed,
                            d_offsets.data() + total_rows,
                            sizeof(int32_t),
                            cudaMemcpyDeviceToHost));
    if (total_chars_signed < 0) {
      throw std::runtime_error("gpu_decode_strings_column: total_chars overflowed int32 (= " +
                               std::to_string(total_chars_signed) + ")");
    }
    alloc_chars = static_cast<size_t>(total_chars_signed);
  }

  rmm::device_buffer d_chars(alloc_chars > 0 ? alloc_chars : 1u, stream, mr);
  auto* d_chars_p = static_cast<uint8_t*>(d_chars.data());

  // Pass 2: gather.
  if (!uncomp_chunks.empty()) {
    kernel_gather_uncomp<<<static_cast<uint32_t>(uncomp_chunks.size()),
                           BLOCK_DIM, 0, stream.value()>>>(
      d_uncomp_chunks_p, d_offsets.data(), d_chars_p,
      static_cast<uint32_t>(uncomp_chunks.size()));
  }
  if (!dict_chunks.empty()) {
    kernel_gather_dict<<<static_cast<uint32_t>(dict_chunks.size()),
                         BLOCK_DIM, 0, stream.value()>>>(
      d_dict_chunks_p, d_offsets.data(), d_chars_p, static_cast<uint32_t>(dict_chunks.size()));
  }
  if (!prep_fsst.gather_chunks.empty()) {
    kernel_gather_fsst_chunked<<<static_cast<uint32_t>(prep_fsst.gather_chunks.size()),
                                 BLOCK_DIM, 0, stream.value()>>>(
      d_fsst_chunks_p, d_offsets.data(), d_chars_p, d_comp_offsets, d_fsst_decs_p,
      static_cast<uint32_t>(prep_fsst.gather_chunks.size()));
  }
  if (!prep_dict_fsst.descs.empty()) {
    // gather is per-row work — use chunked descriptors for SM-fill.
    kernel_gather_dict_fsst<<<static_cast<uint32_t>(dict_fsst_chunks.size()),
                              BLOCK_DIM, 0, stream.value()>>>(
      d_dict_fsst_chunks_p, d_offsets.data(), d_chars_p, d_byte_off_p, d_decoded_off_p,
      d_predecode_p, d_dict_fsst_decs_p,
      static_cast<uint32_t>(dict_fsst_chunks.size()));
  }

  // Validity. Mirrors decode_column_validity in gpu_native_decode.cu —
  // start from all-valid mask, overlay UNCOMPRESSED runs, then for
  // DICT_FSST run mark_nulls so inline NULLs (idx==0) clear the mask.
  rmm::device_buffer null_mask{};
  cudf::size_type    null_count = 0;
  bool need_mask = col.has_nulls || prep_dict_fsst.any_inline_nulls;
  if (need_mask) {
    null_mask = cudf::create_null_mask(static_cast<cudf::size_type>(total_rows),
                                       cudf::mask_state::ALL_VALID, stream, mr);
    for (auto const& run : col.validity) {
      overlay_validity_run(run, static_cast<uint8_t*>(null_mask.data()), stream);
    }
    if (prep_dict_fsst.any_inline_nulls && !prep_dict_fsst.descs.empty()) {
      kernel_dict_fsst_mark_nulls<<<static_cast<uint32_t>(prep_dict_fsst.descs.size()),
                                    BLOCK_DIM, 0, stream.value()>>>(
        d_dict_fsst_p, static_cast<uint8_t*>(null_mask.data()),
        static_cast<uint32_t>(prep_dict_fsst.descs.size()));
    }
    null_count = cudf::null_count(static_cast<cudf::bitmask_type const*>(null_mask.data()),
                                  0, static_cast<cudf::size_type>(total_rows), stream);
  }

  // Build the cudf strings column.
  auto offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32},
    static_cast<cudf::size_type>(total_rows + 1u),
    d_offsets.release(),
    rmm::device_buffer{0, stream, mr},
    0);

  RMM_CUDA_TRY(cudaPeekAtLastError());
  return cudf::make_strings_column(static_cast<cudf::size_type>(total_rows),
                                   std::move(offsets_col),
                                   std::move(d_chars),
                                   null_count,
                                   std::move(null_mask));
}

}  // namespace sirius::cuda::scan
