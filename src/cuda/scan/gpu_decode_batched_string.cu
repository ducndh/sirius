/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 *
 * Batched two-pass string decode for DuckDB native segments.
 *
 * Instead of N per-segment kernel launches + N syncs, this implementation:
 *   Pass 1: ONE batched kernel per compression type → computes string lengths
 *           ONE global CUB ExclusiveSum → produces contiguous offsets (no gaps)
 *           ONE sync → reads total_chars for exact allocation
 *   Pass 2: ONE batched kernel per compression type → gathers strings
 *
 * Total: ~6 kernel launches + 1 sync regardless of segment count.
 * Previous: ~6N launches + N syncs where N can be 4000+ on ClickBench.
 */

#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/gpu_decode_batched_string.cuh"
#include "cuda/scan/gpu_decode_validity.cuh"
#include "cuda/scan/pinned_bounce.cuh"
#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <duckdb/common/types.hpp>

#include <cstring>
#include <vector>

namespace sirius::cuda::scan {

using sirius::op::scan::column_scan_result;

//===----------------------------------------------------------------------===//
// GPU-side segment descriptor — uploaded once for all segments
//===----------------------------------------------------------------------===//

struct alignas(8) batched_seg_desc {
  const uint8_t* d_block;  // Device pointer to 256KB block
  uint32_t block_offset;   // Offset within block to segment start
  uint32_t row_count;
  uint32_t global_row_start;  // Prefix sum of row counts (output indexing)
  uint32_t seg_row_start;     // Offset within segment (input indexing for chunked)
};

/// Extended descriptor for chunked FSST gather — includes the FSST-local
/// row start offset so each chunk CTA can index into d_comp_offsets.
struct alignas(8) batched_fsst_chunk_desc {
  const uint8_t* d_block;
  uint32_t block_offset;
  uint32_t row_count;          // Rows in this chunk (≤ chunk_size)
  uint32_t global_row_start;   // Global row offset (for d_offsets indexing)
  uint32_t fsst_row_start;     // FSST-local row offset (for d_comp_offsets indexing)
  uint32_t seg_decoder_idx;    // Index into d_decoders array
  uint32_t is_first_chunk;     // 1 if this is the first chunk of its segment
};

//===----------------------------------------------------------------------===//
// Adaptive chunking — expand segment descriptors to fill GPU SMs
//===----------------------------------------------------------------------===//

/// Expand segment descriptors into smaller chunk descriptors so that the
/// kernel grid is large enough to utilize all GPU SMs.  Each chunk descriptor
/// has the same d_block and block_offset as its parent segment, but covers
/// a subset of rows (chunk_row_count ≤ rows per chunk).
///
/// The target_ctas parameter is computed from cudaDeviceProp::multiProcessorCount
/// to ensure the GPU is fully utilized regardless of hardware.
static std::vector<batched_seg_desc> expand_to_chunks(
  const std::vector<batched_seg_desc>& seg_descs,
  uint32_t target_ctas)
{
  // Count total rows across all segments
  uint32_t total_rows = 0;
  for (auto const& d : seg_descs) total_rows += d.row_count;

  // If we already have enough CTAs, don't split
  if (seg_descs.size() >= target_ctas || total_rows == 0) {
    return seg_descs;
  }

  // Compute chunk size: target enough CTAs to fill the GPU
  uint32_t chunk_size = total_rows / target_ctas;
  // Clamp: at least 64 rows (below this, CTA overhead dominates)
  chunk_size = std::max(chunk_size, 64u);
  // Round down to warp size for coalescing
  chunk_size = (chunk_size / 32) * 32;
  if (chunk_size == 0) chunk_size = 32;

  std::vector<batched_seg_desc> chunks;
  chunks.reserve(target_ctas + seg_descs.size());

  for (auto const& seg : seg_descs) {
    uint32_t remaining = seg.row_count;
    uint32_t offset    = 0;
    while (remaining > 0) {
      uint32_t n = std::min(remaining, chunk_size);
      chunks.push_back({seg.d_block,
                        seg.block_offset,
                        n,
                        seg.global_row_start + offset,
                        offset});  // seg_row_start: offset within segment
      offset += n;
      remaining -= n;
    }
  }

  return chunks;
}

/// Query GPU SM count (cached after first call).
static uint32_t get_target_ctas()
{
  static uint32_t cached = 0;
  if (cached == 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // Target: fill all SMs with 2 waves of 8-way occupancy (256-thread blocks)
    int occupancy_blocks = prop.maxThreadsPerMultiProcessor / 256;
    cached = static_cast<uint32_t>(prop.multiProcessorCount * occupancy_blocks * 2);
  }
  return cached;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// DUCKDB_BLOCK_SIZE is defined in cuda/scan/gpu_decode.cuh

/// FSST escape code.
#define FSST_ESC 255

/// Dictionary compression header.
struct dict_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t index_buffer_offset;
  uint32_t index_buffer_count;
  uint32_t bitpacking_width;
};

/// FSST compression header.
struct fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t bitpacking_width;
  uint32_t fsst_symbol_table_offset;
};

// duckdb_fsst_import — declared here to avoid FSST include path in CUDA.
extern "C" unsigned int duckdb_fsst_import(void* decoder, unsigned char* buf);

/// FSST decoder struct (layout-compatible with duckdb_fsst_decoder_t).
struct fsst_decoder_gpu {
  unsigned long long version;
  unsigned char zeroTerminated;
  unsigned char len[255];
  unsigned long long symbol[255];
};

/// Compact FSST decoder for GPU upload — just len + symbol, no version header.
struct fsst_decoder_compact {
  uint8_t len[255];
  unsigned long long symbol[255];
};

//===----------------------------------------------------------------------===//
// DICT_FSST: header + descriptor + host helpers
//
// Segment on-disk layout (each region 8-byte aligned):
//   [header 16B]
//   [dict bytes        dict_size bytes]               (raw UTF-8 or FSST-compressed)
//   [symbol table      symbol_table_size bytes]       (only if mode != DICTIONARY)
//   [string_lengths    bitpacked at string_lengths_width]
//   [dictionary_indices bitpacked at dictionary_indices_width]   (absent for FSST_ONLY)
//
// dict_count includes a reserved index 0 that represents NULL (length 0).
// For FSST_ONLY, dictionary_indices is implicit — row i maps to dict entry i+1.
//===----------------------------------------------------------------------===//

/// DICT_FSST compression header — mirrors duckdb::dict_fsst::dict_fsst_compression_header_t (16B).
struct dict_fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_count;
  uint8_t  mode;
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

/// Per-segment descriptor for the DICT_FSST decode path.
struct alignas(8) batched_dict_fsst_desc {
  const uint8_t* d_block;             // device pointer to the 256KB block
  uint32_t block_offset;              // offset of segment base within block
  uint32_t row_count;
  uint32_t global_row_start;          // prefix sum of row counts across this column
  uint32_t seg_row_start;             // row offset within segment (0 for non-chunked)
  uint32_t dict_data_block_off;       // within block, to raw/compressed dict bytes
  uint32_t dict_indices_block_off;    // within block, to bitpacked dictionary_indices
  uint32_t seg_dict_offset_base;      // starting index into d_byte_offsets/d_decoded_offsets
                                      // (both arrays carry dict_count+1 entries per segment)
  uint32_t seg_decoder_idx;           // index into d_decoders; unused when mode == DICTIONARY
  uint8_t dict_indices_width;
  uint8_t mode;
  uint8_t _pad[2];
};

/// Round `n` up to the nearest multiple of 8 — matches DuckDB's AlignValue<idx_t>.
static inline uint32_t align_up8(uint32_t n) { return (n + 7u) & ~7u; }

/// Unpack one bitpacked value on host.  DuckDB's BitpackingPrimitives packs
/// values in 32-element groups that are bit-dense and exactly (32*width/8) =
/// (4*width) bytes each, so group boundaries fall on 32-bit boundaries with
/// no bit rollover — straightforward horizontal bit extraction matches the
/// on-disk layout.  Matches the GPU-side `unpack_value` in gpu_decode.cuh.
template <typename T>
static inline T host_unpack_bitpacked(const uint8_t* packed, uint32_t idx, uint32_t width)
{
  if (width == 0) return T(0);
  uint64_t bit_pos = static_cast<uint64_t>(idx) * width;
  uint32_t byte_off = static_cast<uint32_t>(bit_pos >> 3);
  uint32_t bit_off  = static_cast<uint32_t>(bit_pos & 7u);
  uint64_t combined;
  std::memcpy(&combined, packed + byte_off, sizeof(uint64_t));
  uint64_t mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
  return static_cast<T>((combined >> bit_off) & mask);
}

/// Return the decompressed length of an FSST-compressed byte run without
/// producing output — used to precompute per-dict-entry decoded lengths.
static inline size_t fsst_decompressed_length_host(
  const fsst_decoder_compact& dec, const uint8_t* src, size_t src_len)
{
  size_t pos = 0;
  size_t dec_len = 0;
  while (pos < src_len) {
    uint8_t code = src[pos++];
    if (code < FSST_ESC) {
      dec_len += dec.len[code];
    } else {
      ++pos;       // skip escaped literal byte
      ++dec_len;
    }
  }
  return dec_len;
}

//===----------------------------------------------------------------------===//
// Pass 1 kernels: compute string lengths
//===----------------------------------------------------------------------===//

/// Dictionary: one CTA per segment. Each thread unpacks indices and looks up
/// string length from the index buffer.  Grid-stride for large segments.
__global__ void kernel_compute_lengths_dict(const batched_seg_desc* __restrict__ descs,
                                            uint32_t* __restrict__ d_lengths,
                                            uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  // Parse 20-byte header in thread 0, broadcast via shared memory
  __shared__ uint32_t sh_width;
  __shared__ uint32_t sh_idx_buf_off;
  __shared__ uint32_t sh_sel_buf_off;

  if (threadIdx.x == 0) {
    dict_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_width       = hdr.bitpacking_width;
    sh_idx_buf_off = desc.block_offset + hdr.index_buffer_offset;
    sh_sel_buf_off = desc.block_offset + 20;  // DICTIONARY_HEADER_SIZE
  }
  __syncthreads();

  const uint32_t* d_sel_buf = reinterpret_cast<const uint32_t*>(desc.d_block + sh_sel_buf_off);
  const uint32_t* d_idx_buf = reinterpret_cast<const uint32_t*>(desc.d_block + sh_idx_buf_off);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i                       = desc.seg_row_start + i;
    uint32_t sel                         = unpack_value<uint32_t>(d_sel_buf, seg_i, sh_width);
    uint32_t len                         = (sel == 0) ? 0 : (d_idx_buf[sel] - d_idx_buf[sel - 1]);
    d_lengths[desc.global_row_start + i] = len;
  }
}

/// FSST: one CTA per segment.  Unpacks compressed lengths, computes
/// per-segment InclusiveSum in shared memory + multi-pass, then scans
/// compressed bytes to determine decompressed length per string.
///
/// Requires d_comp_offsets as temporary storage (same size as FSST rows).
__global__ void kernel_compute_lengths_fsst(
  const batched_seg_desc* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  uint32_t* __restrict__ d_comp_offsets,
  const uint32_t* __restrict__ d_fsst_row_starts,
  const fsst_decoder_compact* __restrict__ d_decoders,  // pre-deserialized on host
  uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  // Parse header
  __shared__ uint32_t sh_bp_width;
  __shared__ uint32_t sh_dict_end;
  __shared__ uint8_t sh_len[255];

  if (threadIdx.x == 0) {
    fsst_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_bp_width = hdr.bitpacking_width;
    sh_dict_end = hdr.dict_end;
  }
  __syncthreads();

  // Load pre-deserialized FSST len table into shared memory
  const fsst_decoder_compact& dec = d_decoders[seg_idx];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    sh_len[i] = dec.len[i];
  }
  __syncthreads();

  typedef cub::BlockScan<uint32_t, 256> BlockScanT;
  __shared__ typename BlockScanT::TempStorage scan_temp;

  uint32_t fsst_base = d_fsst_row_starts[seg_idx];
  uint32_t row_count = desc.row_count;

  // Phase A: Unpack compressed lengths
  const uint32_t* packed = reinterpret_cast<const uint32_t*>(base + sizeof(fsst_header_t));

  uint32_t* my_comp = d_comp_offsets + fsst_base;
  for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
    my_comp[i] = unpack_value<uint32_t>(packed, i, sh_bp_width);
  }
  __syncthreads();

  // Phase B: In-CTA InclusiveSum of compressed lengths (multi-pass).
  {
    uint32_t chunk_size = (row_count + blockDim.x - 1) / blockDim.x;
    uint32_t start      = threadIdx.x * chunk_size;
    uint32_t end        = min(start + chunk_size, row_count);

    uint32_t local_sum = 0;
    for (uint32_t i = start; i < end; i++) {
      local_sum += my_comp[i];
      my_comp[i] = local_sum;
    }

    uint32_t thread_total = local_sum;
    uint32_t scanned;
    BlockScanT(scan_temp).ExclusiveSum(thread_total, scanned);

    if (scanned > 0) {
      for (uint32_t i = start; i < end; i++) {
        my_comp[i] += scanned;
      }
    }
  }
  __syncthreads();

  // Phase C is split into a separate chunked kernel below
  // (kernel_compute_lengths_fsst_phase_c) for SM-filling grids.
}

/// FSST Phase C: compute decompressed lengths from precomputed compressed
/// offsets.  Separated from Phase A+B to enable chunking — this phase does
/// the expensive serial byte scan and benefits from more CTAs.
__global__ void kernel_compute_lengths_fsst_phase_c(
  const batched_fsst_chunk_desc* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  const uint32_t* __restrict__ d_comp_offsets,
  const fsst_decoder_compact* __restrict__ d_decoders,
  uint32_t num_chunks)
{
  uint32_t chunk_idx = blockIdx.x;
  if (chunk_idx >= num_chunks) return;

  const auto& desc    = descs[chunk_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  __shared__ uint32_t sh_dict_end;
  __shared__ uint8_t sh_len[255];

  if (threadIdx.x == 0) {
    fsst_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_dict_end = hdr.dict_end;
  }
  __syncthreads();

  const fsst_decoder_compact& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    sh_len[i] = dec.len[i];
  }
  __syncthreads();

  const uint8_t* d_dict_end_ptr = base + sh_dict_end;
  const uint32_t* my_comp       = d_comp_offsets + desc.fsst_row_start;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t cum  = my_comp[i];
    uint32_t prev = (i > 0) ? my_comp[i - 1]
                             : (desc.is_first_chunk ? 0 : *(my_comp - 1));
    uint32_t comp_len = cum - prev;

    if (comp_len == 0) {
      d_lengths[desc.global_row_start + i] = 0;
      continue;
    }

    const uint8_t* comp_ptr = d_dict_end_ptr - cum;
    uint32_t decomp_len     = 0;
    uint32_t pos            = 0;
    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        decomp_len += sh_len[code];
      } else {
        pos++;
        decomp_len++;
      }
    }
    d_lengths[desc.global_row_start + i] = decomp_len;
  }
}

/// Uncompressed strings: one CTA per segment.  Reads DuckDB backward-cumulative
/// offsets and computes per-string lengths.
__global__ void kernel_compute_lengths_uncompressed(const batched_seg_desc* __restrict__ descs,
                                                    uint32_t* __restrict__ d_lengths,
                                                    uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  // DuckDB uncompressed: [dict_size(4)] [dict_end(4)] [offsets(4*N)] [chars]
  // offsets are int32, backward-cumulative from dict_end
  const int32_t* duck_offsets = reinterpret_cast<const int32_t*>(base + 8);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i                       = desc.seg_row_start + i;
    int32_t cur                          = duck_offsets[seg_i];
    int32_t prev                         = (seg_i > 0) ? duck_offsets[seg_i - 1] : 0;
    uint32_t abs_cur                     = static_cast<uint32_t>(cur >= 0 ? cur : -cur);
    uint32_t abs_prev                    = static_cast<uint32_t>(prev >= 0 ? prev : -prev);
    d_lengths[desc.global_row_start + i] = abs_cur - abs_prev;
  }
}

/// Write sentinel: d_offsets[total_rows] = d_offsets[total_rows-1] + d_lengths[total_rows-1]
__global__ void kernel_write_sentinel(const uint32_t* __restrict__ d_offsets_u32,
                                      const uint32_t* __restrict__ d_lengths,
                                      uint32_t* __restrict__ d_sentinel,
                                      uint32_t total_rows)
{
  if (threadIdx.x == 0 && blockIdx.x == 0 && total_rows > 0) {
    *d_sentinel = d_offsets_u32[total_rows - 1] + d_lengths[total_rows - 1];
  }
}

//===----------------------------------------------------------------------===//
// Pass 2 kernels: gather strings into chars buffer
//===----------------------------------------------------------------------===//

/// Dictionary gather: one CTA per segment.  Re-unpacks indices and copies
/// strings from the dictionary at positions given by global d_offsets.
__global__ void kernel_gather_dict(const batched_seg_desc* __restrict__ descs,
                                   const int32_t* __restrict__ d_offsets,
                                   uint8_t* __restrict__ d_chars,
                                   uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  __shared__ uint32_t sh_width;
  __shared__ uint32_t sh_idx_buf_off;
  __shared__ uint32_t sh_sel_buf_off;
  __shared__ uint32_t sh_dict_end_off;

  if (threadIdx.x == 0) {
    dict_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_width        = hdr.bitpacking_width;
    sh_idx_buf_off  = desc.block_offset + hdr.index_buffer_offset;
    sh_sel_buf_off  = desc.block_offset + 20;
    sh_dict_end_off = desc.block_offset + hdr.dict_end;
  }
  __syncthreads();

  const uint32_t* d_sel_buf = reinterpret_cast<const uint32_t*>(desc.d_block + sh_sel_buf_off);
  const uint32_t* d_idx_buf = reinterpret_cast<const uint32_t*>(desc.d_block + sh_idx_buf_off);
  const uint8_t* d_dict_end = desc.d_block + sh_dict_end_off;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t sel   = unpack_value<uint32_t>(d_sel_buf, seg_i, sh_width);
    if (sel == 0) continue;

    uint32_t dict_offset = d_idx_buf[sel];
    uint32_t str_len     = d_idx_buf[sel] - d_idx_buf[sel - 1];
    int32_t out_pos      = d_offsets[desc.global_row_start + i];

    const uint8_t* src = d_dict_end - dict_offset;
    memcpy(d_chars + out_pos, src, str_len);
  }
}

/// FSST gather: one CTA per segment.  Re-computes compressed offsets via
/// multi-pass InclusiveSum, then decompresses each string to d_chars.
__global__ void kernel_gather_fsst(const batched_seg_desc* __restrict__ descs,
                                   const int32_t* __restrict__ d_offsets,
                                   uint8_t* __restrict__ d_chars,
                                   uint32_t* __restrict__ d_comp_offsets,
                                   const uint32_t* __restrict__ d_fsst_row_starts,
                                   const fsst_decoder_compact* __restrict__ d_decoders,
                                   uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  __shared__ uint32_t sh_bp_width;
  __shared__ uint32_t sh_dict_end;
  __shared__ uint8_t sh_len[255];
  __shared__ unsigned long long sh_sym[255];

  if (threadIdx.x == 0) {
    fsst_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_bp_width = hdr.bitpacking_width;
    sh_dict_end = hdr.dict_end;
  }
  __syncthreads();

  // Load pre-deserialized FSST decoder tables into shared memory
  const fsst_decoder_compact& dec = d_decoders[seg_idx];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    sh_len[i] = dec.len[i];
    sh_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  uint32_t fsst_base = d_fsst_row_starts[seg_idx];
  uint32_t row_count = desc.row_count;

  // Re-unpack compressed lengths and re-compute InclusiveSum
  const uint32_t* packed = reinterpret_cast<const uint32_t*>(base + sizeof(fsst_header_t));

  // BlockScan temp storage — must be at function scope
  typedef cub::BlockScan<uint32_t, 256> GatherBlockScanT;
  __shared__ typename GatherBlockScanT::TempStorage gather_scan_temp;

  uint32_t* my_comp = d_comp_offsets + fsst_base;
  for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
    my_comp[i] = unpack_value<uint32_t>(packed, i, sh_bp_width);
  }
  __syncthreads();

  // Multi-pass InclusiveSum (same as pass 1)
  {
    uint32_t chunk_size = (row_count + blockDim.x - 1) / blockDim.x;
    uint32_t start      = threadIdx.x * chunk_size;
    uint32_t end        = min(start + chunk_size, row_count);

    uint32_t local_sum = 0;
    for (uint32_t i = start; i < end; i++) {
      local_sum += my_comp[i];
      my_comp[i] = local_sum;
    }

    uint32_t thread_total = local_sum;
    uint32_t scanned;
    GatherBlockScanT(gather_scan_temp).ExclusiveSum(thread_total, scanned);

    if (scanned > 0) {
      for (uint32_t i = start; i < end; i++) {
        my_comp[i] += scanned;
      }
    }
  }
  __syncthreads();

  // Decompress strings
  const uint8_t* d_dict_end_ptr = base + sh_dict_end;

  for (uint32_t i = threadIdx.x; i < row_count; i += blockDim.x) {
    uint32_t cum      = my_comp[i];
    uint32_t prev     = (i > 0) ? my_comp[i - 1] : 0;
    uint32_t comp_len = cum - prev;
    if (comp_len == 0) continue;

    const uint8_t* comp_ptr = d_dict_end_ptr - cum;
    uint32_t out_pos        = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);
    uint32_t pos            = 0;

    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        unsigned long long sym = sh_sym[code];
        uint8_t sym_len        = sh_len[code];
        // Adaptive write: use sized stores for common lengths (1-2 bytes
        // dominate in practice per FSST paper). Avoids both the overhead
        // of the byte loop AND the bandwidth waste of an 8-byte store.
        switch (sym_len) {
          case 1:
            d_chars[out_pos] = static_cast<uint8_t>(sym);
            break;
          case 2:
            memcpy(d_chars + out_pos, &sym, 2);
            break;
          case 3:
            memcpy(d_chars + out_pos, &sym, 3);
            break;
          case 4:
            memcpy(d_chars + out_pos, &sym, 4);
            break;
          default:
            // 5-8 byte symbols (rare): use memcpy
            memcpy(d_chars + out_pos, &sym, sym_len);
            break;
        }
        out_pos += sym_len;
      } else {
        d_chars[out_pos++] = comp_ptr[pos++];
      }
    }
  }
}

/// Chunked FSST gather: uses precomputed d_comp_offsets from pass 1 (no
/// re-computation of the prefix sum), enabling SM-filling chunk grids.
/// Each CTA handles a subset of rows within a segment.
__global__ void kernel_gather_fsst_chunked(
  const batched_fsst_chunk_desc* __restrict__ descs,
  const int32_t* __restrict__ d_offsets,
  uint8_t* __restrict__ d_chars,
  const uint32_t* __restrict__ d_comp_offsets,
  const fsst_decoder_compact* __restrict__ d_decoders,
  uint32_t num_chunks)
{
  uint32_t chunk_idx = blockIdx.x;
  if (chunk_idx >= num_chunks) return;

  const auto& desc    = descs[chunk_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  __shared__ uint32_t sh_dict_end;
  __shared__ uint8_t sh_len[255];
  __shared__ unsigned long long sh_sym[255];

  if (threadIdx.x == 0) {
    fsst_header_t hdr;
    memcpy(&hdr, base, sizeof(hdr));
    sh_dict_end = hdr.dict_end;
  }
  __syncthreads();

  // Load FSST decoder from the segment's decoder (shared across chunks of same segment)
  const fsst_decoder_compact& dec = d_decoders[desc.seg_decoder_idx];
  for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
    sh_len[i] = dec.len[i];
    sh_sym[i] = dec.symbol[i];
  }
  __syncthreads();

  const uint8_t* d_dict_end_ptr = base + sh_dict_end;
  const uint32_t* my_comp       = d_comp_offsets + desc.fsst_row_start;

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t cum  = my_comp[i];
    // For the first row of a non-first chunk, read the previous element
    // from d_comp_offsets to get the correct cumulative compressed length.
    uint32_t prev = (i > 0) ? my_comp[i - 1]
                             : (desc.is_first_chunk ? 0 : *(my_comp - 1));
    uint32_t comp_len = cum - prev;
    if (comp_len == 0) continue;

    const uint8_t* comp_ptr = d_dict_end_ptr - cum;
    uint32_t out_pos        = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);
    uint32_t pos            = 0;

    while (pos < comp_len) {
      uint8_t code = comp_ptr[pos++];
      if (code < FSST_ESC) {
        unsigned long long sym = sh_sym[code];
        uint8_t sym_len        = sh_len[code];
        switch (sym_len) {
          case 1: d_chars[out_pos] = static_cast<uint8_t>(sym); break;
          case 2: memcpy(d_chars + out_pos, &sym, 2); break;
          case 3: memcpy(d_chars + out_pos, &sym, 3); break;
          case 4: memcpy(d_chars + out_pos, &sym, 4); break;
          default: memcpy(d_chars + out_pos, &sym, sym_len); break;
        }
        out_pos += sym_len;
      } else {
        d_chars[out_pos++] = comp_ptr[pos++];
      }
    }
  }
}

/// Uncompressed string gather: one CTA per segment.
__global__ void kernel_gather_uncompressed(const batched_seg_desc* __restrict__ descs,
                                           const int32_t* __restrict__ d_offsets,
                                           uint8_t* __restrict__ d_chars,
                                           uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc    = descs[seg_idx];
  const uint8_t* base = desc.d_block + desc.block_offset;

  __shared__ uint32_t sh_dict_end;
  if (threadIdx.x == 0) {
    uint32_t de;
    memcpy(&de, base + 4, sizeof(de));
    sh_dict_end = de;
  }
  __syncthreads();

  const int32_t* duck_offsets = reinterpret_cast<const int32_t*>(base + 8);
  const uint8_t* dict_end     = base + sh_dict_end;

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

//===----------------------------------------------------------------------===//
// DICT_FSST pass 1: compute decoded string length per row.
//
// Host precomputes cumulative decoded-byte offsets per dict entry
// (d_decoded_offsets, size = dict_count+1 per segment).  Length of entry i
// is decoded_offsets[i+1] - decoded_offsets[i].  Index 0 is NULL → length 0.
// FSST_ONLY has no bitpacked indices; row i maps to entry i+1.
//===----------------------------------------------------------------------===//
__global__ void kernel_compute_lengths_dict_fsst(
  const batched_dict_fsst_desc* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  const uint32_t* __restrict__ d_decoded_offsets,
  uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc        = descs[seg_idx];
  const uint32_t* dec_off = d_decoded_offsets + desc.seg_dict_offset_base;
  const uint32_t* d_idx   = reinterpret_cast<const uint32_t*>(
    desc.d_block + desc.dict_indices_block_off);
  const bool fsst_only = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx;
    if (fsst_only) {
      idx = seg_i + 1;
    } else {
      idx = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    }
    uint32_t len = (idx == 0) ? 0u : (dec_off[idx + 1] - dec_off[idx]);
    d_lengths[desc.global_row_start + i] = len;
  }
}

//===----------------------------------------------------------------------===//
// DICT_FSST pass 2: gather strings.
//
// For DICTIONARY mode (raw dict bytes), straight memcpy from dict region.
// For DICT_FSST / FSST_ONLY, inline FSST decompress from the dict region
// using a per-segment symbol table loaded into shared memory.
//===----------------------------------------------------------------------===//
__global__ void kernel_gather_dict_fsst(
  const batched_dict_fsst_desc* __restrict__ descs,
  const int32_t* __restrict__ d_offsets,
  uint8_t* __restrict__ d_chars,
  const uint32_t* __restrict__ d_byte_offsets,
  const fsst_decoder_compact* __restrict__ d_decoders,
  uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc          = descs[seg_idx];
  const uint8_t* dict_base  = desc.d_block + desc.dict_data_block_off;
  const uint32_t* byte_off  = d_byte_offsets + desc.seg_dict_offset_base;
  const uint32_t* d_idx     = reinterpret_cast<const uint32_t*>(
    desc.d_block + desc.dict_indices_block_off);
  const bool need_fsst = (desc.mode != DICT_FSST_MODE_DICTIONARY);
  const bool fsst_only = (desc.mode == DICT_FSST_MODE_FSST_ONLY);

  __shared__ uint8_t sh_len[255];
  __shared__ unsigned long long sh_sym[255];
  if (need_fsst) {
    const fsst_decoder_compact& dec = d_decoders[desc.seg_decoder_idx];
    for (uint32_t i = threadIdx.x; i < 255; i += blockDim.x) {
      sh_len[i] = dec.len[i];
      sh_sym[i] = dec.symbol[i];
    }
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx;
    if (fsst_only) {
      idx = seg_i + 1;
    } else {
      idx = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    }
    if (idx == 0) continue;

    uint32_t stored_start = byte_off[idx];
    uint32_t stored_len   = byte_off[idx + 1] - stored_start;
    const uint8_t* src    = dict_base + stored_start;
    uint32_t out_pos      = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);

    if (!need_fsst) {
      for (uint32_t k = 0; k < stored_len; ++k) {
        d_chars[out_pos + k] = src[k];
      }
    } else {
      uint32_t pos = 0;
      while (pos < stored_len) {
        uint8_t code = src[pos++];
        if (code < FSST_ESC) {
          unsigned long long sym = sh_sym[code];
          uint8_t sym_len        = sh_len[code];
          switch (sym_len) {
            case 1: d_chars[out_pos] = static_cast<uint8_t>(sym); break;
            case 2: memcpy(d_chars + out_pos, &sym, 2); break;
            case 3: memcpy(d_chars + out_pos, &sym, 3); break;
            case 4: memcpy(d_chars + out_pos, &sym, 4); break;
            default: memcpy(d_chars + out_pos, &sym, sym_len); break;
          }
          out_pos += sym_len;
        } else {
          d_chars[out_pos++] = src[pos++];
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// DICT_FSST null marking.
//
// DICT_FSST (and DICTIONARY) encode NULL inline in the data codec: dict
// index 0 is reserved for NULL.  On disk this means the validity segment
// is usually "Empty Validity" (COMPRESSION_EMPTY), so the bitmap overlay
// path skips the column and everything looks valid.  For correctness the
// null mask must reflect rows whose dict_idx == 0 — clear those bits.
// FSST_ONLY maps row i to idx = i+1, so it can never produce idx == 0
// and this kernel becomes a no-op there.
//===----------------------------------------------------------------------===//
__global__ void kernel_dict_fsst_mark_nulls(
  const batched_dict_fsst_desc* __restrict__ descs,
  unsigned long long* __restrict__ null_mask,
  uint32_t num_segments)
{
  uint32_t seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;

  const auto& desc     = descs[seg_idx];
  const bool fsst_only = (desc.mode == DICT_FSST_MODE_FSST_ONLY);
  if (fsst_only) return;

  const uint32_t* d_idx = reinterpret_cast<const uint32_t*>(
    desc.d_block + desc.dict_indices_block_off);

  for (uint32_t i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    uint32_t seg_i = desc.seg_row_start + i;
    uint32_t idx   = unpack_value<uint32_t>(d_idx, seg_i, desc.dict_indices_width);
    if (idx == 0) {
      uint32_t abs_row = desc.global_row_start + i;
      uint32_t word    = abs_row >> 6;
      uint32_t bit     = abs_row & 63u;
      atomicAnd(null_mask + word, ~(1ULL << bit));
    }
  }
}

//===----------------------------------------------------------------------===//
// Validity + null count kernels are shared with gpu_native_decode.cu
// via gpu_decode_validity.cuh — do not duplicate them here.
//===----------------------------------------------------------------------===//
// Host-side orchestrator
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_string_column_batched(
  column_scan_result& col_scan,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  const std::unordered_map<int64_t, size_t>* device_blocks,
  uint8_t* device_staging,
  uint32_t* d_valid_count_out)
{
  size_t total_rows = col_scan.data.total_rows;
  // Check for pre-existing CUDA errors
  auto pre_err = cudaPeekAtLastError();
  if (pre_err != cudaSuccess) {
    SIRIUS_LOG_ERROR("[batched_string] pre-existing CUDA error on entry: {}",
                     cudaGetErrorString(pre_err));
    cudaGetLastError();  // clear sticky error
  }
  if (total_rows == 0) { return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING)); }

  //===--------------------------------------------------------------------===//
  // Phase 1: Collect unique blocks and bulk H2D (or reuse pipelined staging)
  //===--------------------------------------------------------------------===//

  struct block_entry {
    const uint8_t* host_base;
    size_t device_offset;
    size_t copy_size;
  };
  std::unordered_map<intptr_t, block_entry> block_map;
  bool reuse_pipelined = (device_blocks != nullptr && device_staging != nullptr);

  rmm::device_buffer staging_buf;
  uint8_t* d_staging_ptr = nullptr;

  if (reuse_pipelined) {
    // Blocks are already on device from the pipelined path — reuse directly
    d_staging_ptr = device_staging;
    for (auto const& seg : col_scan.data.segments) {
      if (!seg.persistent || !seg.data_ptr || seg.row_count == 0) continue;
      if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) continue;
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      auto key                  = reinterpret_cast<intptr_t>(block_base);
      if (block_map.find(key) == block_map.end()) {
        auto it = device_blocks->find(seg.block_id);
        if (it != device_blocks->end()) {
          block_map[key] = {block_base, it->second};
        } else {
          // Block not in staging — fall back to H2D (shouldn't happen in pipelined path)
          reuse_pipelined = false;
          break;
        }
      }
    }
  }

  if (!reuse_pipelined) {
    // Collect unique blocks and compute required copy size per block.
    // DuckDB blocks may be smaller than 256KB for small tables, so we track
    // the max (block_offset + segment_size) across all segments in each block.
    block_map.clear();
    for (auto const& seg : col_scan.data.segments) {
      if (!seg.persistent || !seg.data_ptr || seg.row_count == 0) continue;
      if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) continue;
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      auto key                  = reinterpret_cast<intptr_t>(block_base);
      size_t needed =
        seg.block_offset + (seg.segment_size > 0 ? seg.segment_size : DUCKDB_BLOCK_SIZE);
      needed  = std::min(needed, DUCKDB_BLOCK_SIZE);
      auto it = block_map.find(key);
      if (it == block_map.end()) {
        size_t offset  = block_map.size() * DUCKDB_BLOCK_SIZE;  // compute BEFORE insert
        block_map[key] = {block_base, offset, needed};
      } else {
        it->second.copy_size = std::max(it->second.copy_size, needed);
      }
    }

    size_t staging_bytes = block_map.size() * DUCKDB_BLOCK_SIZE;
    if (staging_bytes > 0) {
      staging_buf   = rmm::device_buffer(staging_bytes, stream, mr);
      d_staging_ptr = static_cast<uint8_t*>(staging_buf.data());
      for (auto& [key, entry] : block_map) {
        bounce_h2d_async(d_staging_ptr + entry.device_offset,
                         entry.host_base,
                         entry.copy_size,
                         stream.value());
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Phase 2: Build per-type segment descriptor arrays
  //===--------------------------------------------------------------------===//

  std::vector<batched_seg_desc> dict_descs, fsst_descs, uncomp_descs;
  std::vector<uint32_t> fsst_row_starts;
  std::vector<fsst_decoder_compact> fsst_decoders;  // host-deserialized decoders
  uint32_t total_fsst_rows = 0;
  size_t cum_rows          = 0;
  size_t cum_chars_upper   = 0;  // CPU upper-bound on total string bytes

  // DICT_FSST host-precomputed per-dict-entry offset tables.  Both offset
  // arrays hold dict_count+1 cumulative entries per segment, packed
  // back-to-back across all DICT_FSST segments; desc.seg_dict_offset_base
  // points at entry 0 of that segment's tables.
  std::vector<batched_dict_fsst_desc> dict_fsst_descs;
  std::vector<uint32_t> dict_fsst_byte_offsets;     // stored bytes (raw or FSST-compressed)
  std::vector<uint32_t> dict_fsst_decoded_offsets;  // decoded (user-visible) bytes
  std::vector<fsst_decoder_compact> dict_fsst_decoders;

  for (auto const& seg : col_scan.data.segments) {
    if (seg.row_count == 0) {
      cum_rows += seg.row_count;
      continue;
    }

    if (!seg.persistent || !seg.data_ptr ||
        seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
      cum_rows += seg.row_count;
      continue;
    }

    // Accumulate char upper bound from segment metadata
    cum_chars_upper += static_cast<size_t>(seg.row_count) * seg.max_string_length;

    // Resolve device block pointer
    const uint8_t* block_base = seg.data_ptr - seg.block_offset;
    auto key                  = reinterpret_cast<intptr_t>(block_base);
    const uint8_t* d_block    = d_staging_ptr + block_map[key].device_offset;

    batched_seg_desc desc;
    desc.d_block          = d_block;
    desc.block_offset     = static_cast<uint32_t>(seg.block_offset);
    desc.row_count        = static_cast<uint32_t>(seg.row_count);
    desc.global_row_start = static_cast<uint32_t>(cum_rows);
    desc.seg_row_start    = 0;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_DICTIONARY: dict_descs.push_back(desc); break;
      case duckdb::CompressionType::COMPRESSION_FSST: {
        fsst_row_starts.push_back(total_fsst_rows);
        total_fsst_rows += desc.row_count;
        fsst_descs.push_back(desc);

        // Deserialize FSST symbol table on HOST (opaque format, can't parse on GPU)
        const uint8_t* seg_host_base = seg.data_ptr - seg.block_offset + seg.block_offset;
        fsst_header_t fsst_hdr;
        std::memcpy(&fsst_hdr, seg_host_base, sizeof(fsst_hdr));
        fsst_decoder_gpu full_dec;
        std::memset(&full_dec, 0, sizeof(full_dec));
        duckdb_fsst_import(
          &full_dec, const_cast<unsigned char*>(seg_host_base + fsst_hdr.fsst_symbol_table_offset));
        fsst_decoder_compact compact;
        std::memcpy(compact.len, full_dec.len, 255);
        std::memcpy(compact.symbol, full_dec.symbol, 255 * sizeof(unsigned long long));
        fsst_decoders.push_back(compact);
        break;
      }
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: uncomp_descs.push_back(desc); break;
      case duckdb::CompressionType::COMPRESSION_DICT_FSST: {
        // Host pre-parses this segment: header → region offsets → unpack
        // string_lengths → (optionally) import FSST decoder + decode each
        // dict entry to compute its decoded length.  The final byte/decoded
        // offset arrays are pushed back on the shared vectors and the
        // descriptor records the base index.
        const uint8_t* seg_host_base = seg.data_ptr;
        dict_fsst_header_t hdr;
        std::memcpy(&hdr, seg_host_base, sizeof(hdr));

        uint32_t dict_count         = hdr.dict_count;
        uint32_t dict_size          = hdr.dict_size;
        uint32_t sym_table_size     = hdr.symbol_table_size;
        uint8_t  mode               = hdr.mode;
        uint8_t  slen_width         = hdr.string_lengths_width;
        uint8_t  didx_width         = hdr.dictionary_indices_width;

        uint32_t dict_dest          = align_up8(sizeof(dict_fsst_header_t));
        uint32_t sym_table_dest     = align_up8(dict_dest + dict_size);
        uint32_t string_lengths_dest = align_up8(sym_table_dest + sym_table_size);
        // BitpackingPrimitives::GetRequiredSize: round dict_count up to group
        // size (32), then bytes = (rounded * width) / 8.
        uint32_t slen_rounded       = (dict_count + 31u) & ~31u;
        uint32_t slen_space         = (slen_rounded * slen_width) / 8u;
        uint32_t dict_indices_dest  = align_up8(string_lengths_dest + slen_space);

        // Unpack per-entry stored lengths, build forward cumulative byte offsets.
        std::vector<uint32_t> entry_stored_len(dict_count);
        const uint8_t* slen_packed = seg_host_base + string_lengths_dest;
        for (uint32_t i = 0; i < dict_count; ++i) {
          entry_stored_len[i] = host_unpack_bitpacked<uint32_t>(slen_packed, i, slen_width);
        }

        uint32_t base_off = static_cast<uint32_t>(dict_fsst_byte_offsets.size());
        dict_fsst_byte_offsets.resize(base_off + dict_count + 1);
        dict_fsst_decoded_offsets.resize(base_off + dict_count + 1);
        dict_fsst_byte_offsets[base_off]    = 0;
        dict_fsst_decoded_offsets[base_off] = 0;

        uint32_t seg_decoder_idx = static_cast<uint32_t>(dict_fsst_decoders.size());
        fsst_decoder_compact compact{};
        bool fsst_mode = (mode != DICT_FSST_MODE_DICTIONARY);
        if (fsst_mode) {
          fsst_decoder_gpu full_dec;
          std::memset(&full_dec, 0, sizeof(full_dec));
          duckdb_fsst_import(&full_dec,
                             const_cast<unsigned char*>(seg_host_base + sym_table_dest));
          std::memcpy(compact.len, full_dec.len, 255);
          std::memcpy(compact.symbol, full_dec.symbol, 255 * sizeof(unsigned long long));
        }
        // Always push a decoder slot so seg_decoder_idx is valid — the
        // kernel only reads it for non-DICTIONARY modes.
        dict_fsst_decoders.push_back(compact);

        const uint8_t* dict_bytes = seg_host_base + dict_dest;
        uint32_t stored_cum  = 0;
        uint32_t decoded_cum = 0;
        for (uint32_t i = 0; i < dict_count; ++i) {
          stored_cum += entry_stored_len[i];
          uint32_t decoded_len;
          if (!fsst_mode) {
            decoded_len = entry_stored_len[i];
          } else if (i == 0) {
            // Reserved NULL entry: length always 0 regardless of stored bytes.
            decoded_len = 0;
          } else {
            uint32_t start = (i == 0) ? 0 : dict_fsst_byte_offsets[base_off + i];
            decoded_len = static_cast<uint32_t>(
              fsst_decompressed_length_host(compact, dict_bytes + start, entry_stored_len[i]));
          }
          decoded_cum += decoded_len;
          dict_fsst_byte_offsets[base_off + i + 1]    = stored_cum;
          dict_fsst_decoded_offsets[base_off + i + 1] = decoded_cum;
        }

        batched_dict_fsst_desc df{};
        df.d_block                 = d_block;
        df.block_offset            = static_cast<uint32_t>(seg.block_offset);
        df.row_count               = static_cast<uint32_t>(seg.row_count);
        df.global_row_start        = static_cast<uint32_t>(cum_rows);
        df.seg_row_start           = 0;
        df.dict_data_block_off     = static_cast<uint32_t>(seg.block_offset) + dict_dest;
        df.dict_indices_block_off  = static_cast<uint32_t>(seg.block_offset) + dict_indices_dest;
        df.seg_dict_offset_base    = base_off;
        df.seg_decoder_idx         = seg_decoder_idx;
        df.dict_indices_width      = didx_width;
        df.mode                    = mode;
        dict_fsst_descs.push_back(df);
        break;
      }
      default: break;
    }
    cum_rows += seg.row_count;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Allocate work buffers
  //===--------------------------------------------------------------------===//

  SIRIUS_LOG_DEBUG(
    "[batched_string] phase 3: {} total rows, {} dict, {} fsst, {} uncomp segs, "
    "{} blocks, staging_ptr={}, reuse={}",
    total_rows,
    dict_descs.size(),
    fsst_descs.size(),
    uncomp_descs.size(),
    block_map.size(),
    (void*)d_staging_ptr,
    reuse_pipelined);
  // Verify all descriptors have valid device block pointers
  for (auto const& d : dict_descs) {
    if (!d.d_block) SIRIUS_LOG_ERROR("[batched] dict desc has null d_block!");
  }
  for (auto const& d : fsst_descs) {
    if (!d.d_block) SIRIUS_LOG_ERROR("[batched] fsst desc has null d_block!");
  }
  for (auto const& d : uncomp_descs) {
    if (!d.d_block)
      SIRIUS_LOG_ERROR("[batched] uncomp desc has null d_block! row_count={} global_row_start={}",
                       d.row_count,
                       d.global_row_start);
  }

  // d_lengths: one uint32 per row — pass 1 output, CUB input
  rmm::device_uvector<uint32_t> d_lengths(total_rows, stream, mr);

  // d_offsets: one int32 per row + sentinel — CUB output, pass 2 input
  rmm::device_uvector<int32_t> d_offsets(total_rows + 1, stream, mr);

  // Zero-init d_lengths (covers CONSTANT segments that have length 0)
  cudaMemsetAsync(d_lengths.data(), 0, total_rows * sizeof(uint32_t), stream.value());

  // FSST temp: compressed offsets (via RMM)
  rmm::device_buffer comp_offsets_buf(
    total_fsst_rows > 0 ? total_fsst_rows * sizeof(uint32_t) : 0, stream, mr);
  uint32_t* d_comp_offsets = static_cast<uint32_t*>(comp_offsets_buf.data());

  // Upload segment descriptors (via RMM).
  //
  // Previously five separate RMM allocations + cudaMemcpyAsync calls — two
  // of which (dict_descs, uncomp_descs) were dead (only FSST path reads
  // d_*_descs directly, the dict/uncomp paths operate on *_chunks uploaded
  // below). Coalesce the three live FSST descriptors into one arena so each
  // call site sees one allocation + one H2D instead of three, while the
  // chunks path upstream still does its own uploads.
  auto make_device_copy = [&](const void* src, size_t bytes) -> rmm::device_buffer {
    rmm::device_buffer buf(bytes, stream, mr);
    if (bytes > 0) {
      bounce_h2d_async(buf.data(), src, bytes, stream.value());
    }
    return buf;
  };

  auto align_up = [](size_t x, size_t a) { return (x + a - 1) & ~(a - 1); };
  constexpr size_t kAlign   = alignof(batched_seg_desc);
  const size_t bytes_fsst   = fsst_descs.size() * sizeof(batched_seg_desc);
  const size_t bytes_starts = fsst_row_starts.size() * sizeof(uint32_t);
  const size_t bytes_decs   = fsst_decoders.size() * sizeof(fsst_decoder_compact);
  const size_t off_fsst     = 0;
  const size_t off_starts   = align_up(off_fsst + bytes_fsst, kAlign);
  const size_t off_decs     = align_up(off_starts + bytes_starts, kAlign);
  const size_t arena_bytes  = off_decs + bytes_decs;

  std::vector<uint8_t> h_arena(arena_bytes);
  if (bytes_fsst)   std::memcpy(h_arena.data() + off_fsst,   fsst_descs.data(),      bytes_fsst);
  if (bytes_starts) std::memcpy(h_arena.data() + off_starts, fsst_row_starts.data(), bytes_starts);
  if (bytes_decs)   std::memcpy(h_arena.data() + off_decs,   fsst_decoders.data(),   bytes_decs);

  rmm::device_buffer fsst_arena_buf(arena_bytes, stream, mr);
  if (arena_bytes > 0) {
    bounce_h2d_async(fsst_arena_buf.data(), h_arena.data(), arena_bytes, stream.value());
  }
  auto* d_fsst_descs      = reinterpret_cast<batched_seg_desc*>(
                              static_cast<uint8_t*>(fsst_arena_buf.data()) + off_fsst);
  auto* d_fsst_row_starts = reinterpret_cast<uint32_t*>(
                              static_cast<uint8_t*>(fsst_arena_buf.data()) + off_starts);
  auto* d_fsst_decoders   = reinterpret_cast<fsst_decoder_compact*>(
                              static_cast<uint8_t*>(fsst_arena_buf.data()) + off_decs);

  // DICT_FSST arena: descriptors + byte_offsets + decoded_offsets + decoders.
  const size_t bytes_df_descs   = dict_fsst_descs.size() * sizeof(batched_dict_fsst_desc);
  const size_t bytes_df_byte    = dict_fsst_byte_offsets.size() * sizeof(uint32_t);
  const size_t bytes_df_decoded = dict_fsst_decoded_offsets.size() * sizeof(uint32_t);
  const size_t bytes_df_decs    = dict_fsst_decoders.size() * sizeof(fsst_decoder_compact);
  const size_t off_df_descs     = 0;
  const size_t off_df_byte      = align_up(off_df_descs + bytes_df_descs, kAlign);
  const size_t off_df_decoded   = align_up(off_df_byte + bytes_df_byte, kAlign);
  const size_t off_df_decs      = align_up(off_df_decoded + bytes_df_decoded, kAlign);
  const size_t df_arena_bytes   = off_df_decs + bytes_df_decs;

  std::vector<uint8_t> h_df_arena(df_arena_bytes);
  if (bytes_df_descs)
    std::memcpy(h_df_arena.data() + off_df_descs, dict_fsst_descs.data(), bytes_df_descs);
  if (bytes_df_byte)
    std::memcpy(h_df_arena.data() + off_df_byte, dict_fsst_byte_offsets.data(), bytes_df_byte);
  if (bytes_df_decoded)
    std::memcpy(h_df_arena.data() + off_df_decoded, dict_fsst_decoded_offsets.data(),
                bytes_df_decoded);
  if (bytes_df_decs)
    std::memcpy(h_df_arena.data() + off_df_decs, dict_fsst_decoders.data(), bytes_df_decs);

  rmm::device_buffer df_arena_buf(df_arena_bytes, stream, mr);
  if (df_arena_bytes > 0) {
    bounce_h2d_async(df_arena_buf.data(), h_df_arena.data(), df_arena_bytes, stream.value());
  }
  auto* d_df_descs    = reinterpret_cast<batched_dict_fsst_desc*>(
                          static_cast<uint8_t*>(df_arena_buf.data()) + off_df_descs);
  auto* d_df_byte     = reinterpret_cast<uint32_t*>(
                          static_cast<uint8_t*>(df_arena_buf.data()) + off_df_byte);
  auto* d_df_decoded  = reinterpret_cast<uint32_t*>(
                          static_cast<uint8_t*>(df_arena_buf.data()) + off_df_decoded);
  auto* d_df_decoders = reinterpret_cast<fsst_decoder_compact*>(
                          static_cast<uint8_t*>(df_arena_buf.data()) + off_df_decs);

  //===--------------------------------------------------------------------===//
  // Phase 4: Pass 1 — compute string lengths (batched)
  //===--------------------------------------------------------------------===//

  constexpr uint32_t THREADS = 256;
  uint32_t target_ctas       = get_target_ctas();

  // Expand dict descriptors to chunks for better SM utilization.
  // With 17 segments on 132 SMs, only 13% of SMs are active.
  // Chunking to ~2K CTAs fills all SMs with 2 waves.
  auto dict_chunks     = expand_to_chunks(dict_descs, target_ctas);
  auto dict_chunks_buf = make_device_copy(
    dict_chunks.data(), dict_chunks.size() * sizeof(batched_seg_desc));
  auto* d_dict_chunks  = static_cast<batched_seg_desc*>(dict_chunks_buf.data());

  auto uncomp_chunks     = expand_to_chunks(uncomp_descs, target_ctas);
  auto uncomp_chunks_buf = make_device_copy(
    uncomp_chunks.data(), uncomp_chunks.size() * sizeof(batched_seg_desc));
  auto* d_uncomp_chunks  = static_cast<batched_seg_desc*>(uncomp_chunks_buf.data());

  if (!dict_chunks.empty()) {
    kernel_compute_lengths_dict<<<static_cast<uint32_t>(dict_chunks.size()),
                                  THREADS,
                                  0,
                                  stream.value()>>>(
      d_dict_chunks, d_lengths.data(), static_cast<uint32_t>(dict_chunks.size()));
  }

  // Build chunked FSST descriptors once — reused for both lengths Phase C and gather.
  std::vector<batched_fsst_chunk_desc> fsst_chunks;
  rmm::device_buffer fsst_chunks_buf_dev(0, stream, mr);
  batched_fsst_chunk_desc* d_fsst_chunks = nullptr;

  if (!fsst_descs.empty()) {
    // Phase A+B: unpack compressed lengths + prefix sum (per-segment, unchunkable)
    kernel_compute_lengths_fsst<<<static_cast<unsigned>(fsst_descs.size()),
                                  THREADS,
                                  0,
                                  stream.value()>>>(d_fsst_descs,
                                                    d_lengths.data(),
                                                    d_comp_offsets,
                                                    d_fsst_row_starts,
                                                    d_fsst_decoders,
                                                    static_cast<uint32_t>(fsst_descs.size()));

    // Build FSST chunk descriptors for Phase C and later gather
    uint32_t chunk_size_fsst = 0;
    {
      uint32_t total_fsst = 0;
      for (auto const& d : fsst_descs) total_fsst += d.row_count;
      if (fsst_descs.size() < target_ctas && total_fsst > 0) {
        chunk_size_fsst = std::max(total_fsst / target_ctas, 64u);
        chunk_size_fsst = (chunk_size_fsst / 32) * 32;
        if (chunk_size_fsst == 0) chunk_size_fsst = 32;
      }
    }

    for (size_t si = 0; si < fsst_descs.size(); ++si) {
      auto const& seg = fsst_descs[si];
      uint32_t fsst_base_row = fsst_row_starts[si];
      if (chunk_size_fsst == 0) {
        fsst_chunks.push_back({seg.d_block, seg.block_offset, seg.row_count,
                               seg.global_row_start, fsst_base_row,
                               static_cast<uint32_t>(si), 1});
      } else {
        uint32_t remaining = seg.row_count;
        uint32_t offset = 0;
        bool first = true;
        while (remaining > 0) {
          uint32_t n = std::min(remaining, chunk_size_fsst);
          fsst_chunks.push_back({seg.d_block, seg.block_offset, n,
                                 seg.global_row_start + offset,
                                 fsst_base_row + offset,
                                 static_cast<uint32_t>(si),
                                 first ? 1u : 0u});
          offset += n;
          remaining -= n;
          first = false;
        }
      }
    }

    fsst_chunks_buf_dev = make_device_copy(
      fsst_chunks.data(), fsst_chunks.size() * sizeof(batched_fsst_chunk_desc));
    d_fsst_chunks = static_cast<batched_fsst_chunk_desc*>(fsst_chunks_buf_dev.data());

    // Phase C: compute decompressed lengths (chunked, SM-filling grid)
    kernel_compute_lengths_fsst_phase_c<<<static_cast<uint32_t>(fsst_chunks.size()),
                                          THREADS, 0, stream.value()>>>(
      d_fsst_chunks, d_lengths.data(), d_comp_offsets, d_fsst_decoders,
      static_cast<uint32_t>(fsst_chunks.size()));
  }

  if (!uncomp_chunks.empty()) {
    kernel_compute_lengths_uncompressed<<<static_cast<unsigned>(uncomp_chunks.size()),
                                          THREADS,
                                          0,
                                          stream.value()>>>(
      d_uncomp_chunks, d_lengths.data(), static_cast<uint32_t>(uncomp_chunks.size()));
  }

  if (!dict_fsst_descs.empty()) {
    kernel_compute_lengths_dict_fsst<<<static_cast<uint32_t>(dict_fsst_descs.size()),
                                       THREADS,
                                       0,
                                       stream.value()>>>(
      d_df_descs, d_lengths.data(), d_df_decoded,
      static_cast<uint32_t>(dict_fsst_descs.size()));
  }

  //===--------------------------------------------------------------------===//
  // Phase 5: Global CUB ExclusiveSum + sentinel + ONE sync
  //===--------------------------------------------------------------------===//

  size_t cub_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                static_cast<int>(total_rows),
                                stream.value());

  rmm::device_buffer cub_temp_buf(cub_bytes, stream, mr);
  void* d_cub_temp = cub_temp_buf.data();

  cub::DeviceScan::ExclusiveSum(d_cub_temp,
                                cub_bytes,
                                d_lengths.data(),
                                reinterpret_cast<uint32_t*>(d_offsets.data()),
                                static_cast<int>(total_rows),
                                stream.value());

  // Write sentinel: offsets[total_rows] = offsets[total_rows-1] + lengths[total_rows-1]
  kernel_write_sentinel<<<1, 1, 0, stream.value()>>>(
    reinterpret_cast<uint32_t*>(d_offsets.data()),
    d_lengths.data(),
    reinterpret_cast<uint32_t*>(d_offsets.data()) + total_rows,
    static_cast<uint32_t>(total_rows));

  // Allocate char buffer using CPU upper bound — avoids inter-pass sync.
  // The upper bound comes from sum(seg.row_count × seg.max_string_length).
  // Pass 2 kernels write at offsets computed by CUB (exact), so only the
  // leading portion of d_chars is used.  Excess is wasted but harmless.
  //
  // Safety: fall back to exact allocation (with sync) if upper bound
  // exceeds 512MB to prevent OOM from pathological max_string_length.
  constexpr size_t UPPER_BOUND_LIMIT = 512ULL * 1024 * 1024;
  bool use_upper_bound               = cum_chars_upper <= UPPER_BOUND_LIMIT;

  size_t alloc_chars = 0;
  if (use_upper_bound) {
    alloc_chars = cum_chars_upper;
  } else {
    // Two-pass fallback: sync to get exact total_chars
    stream.synchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("batched string pass 1 CUDA error: ") +
                               cudaGetErrorString(err));
    }
    int32_t total_chars = 0;
    cudaMemcpy(
      &total_chars, d_offsets.data() + total_rows, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (total_chars < 0) {
      throw std::runtime_error("batched string decode: negative total_chars=" +
                               std::to_string(total_chars));
    }
    alloc_chars = static_cast<size_t>(total_chars);
  }

  rmm::device_buffer d_chars(alloc_chars > 0 ? alloc_chars : 1, stream, mr);
  auto* d_chars_ptr = static_cast<uint8_t*>(d_chars.data());

  //===--------------------------------------------------------------------===//
  // Phase 7: Pass 2 — gather strings (batched)
  //===--------------------------------------------------------------------===//

  if (!dict_chunks.empty()) {
    kernel_gather_dict<<<dict_chunks.size(), THREADS, 0, stream.value()>>>(
      d_dict_chunks, d_offsets.data(), d_chars_ptr, static_cast<uint32_t>(dict_chunks.size()));
  }

  if (!fsst_chunks.empty()) {
    // Reuse chunked FSST descriptors built during Phase 4.
    kernel_gather_fsst_chunked<<<static_cast<uint32_t>(fsst_chunks.size()),
                                 THREADS, 0, stream.value()>>>(
      d_fsst_chunks,
      d_offsets.data(),
      d_chars_ptr,
      d_comp_offsets,
      d_fsst_decoders,
      static_cast<uint32_t>(fsst_chunks.size()));
  }

  if (!uncomp_chunks.empty()) {
    kernel_gather_uncompressed<<<uncomp_chunks.size(), THREADS, 0, stream.value()>>>(
      d_uncomp_chunks, d_offsets.data(), d_chars_ptr, static_cast<uint32_t>(uncomp_chunks.size()));
  }

  if (!dict_fsst_descs.empty()) {
    kernel_gather_dict_fsst<<<static_cast<uint32_t>(dict_fsst_descs.size()),
                              THREADS, 0, stream.value()>>>(
      d_df_descs, d_offsets.data(), d_chars_ptr, d_df_byte, d_df_decoders,
      static_cast<uint32_t>(dict_fsst_descs.size()));
  }

  //===--------------------------------------------------------------------===//
  // Phase 8: Cleanup temp buffers
  //===--------------------------------------------------------------------===//

  // All temp buffers (staging, descriptors, CUB scratch, comp_offsets)
  // are rmm::device_buffer — cleaned up by RAII at scope exit.

  //===--------------------------------------------------------------------===//
  // Phase 9: Build cudf string column
  //===--------------------------------------------------------------------===//

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(total_rows + 1),
                                                    d_offsets.release(),
                                                    rmm::device_buffer{0, stream, mr},
                                                    0);

  // Validity
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;

  if (col_scan.has_nulls) {
    // Match cuDF's Arrow-spec 64-byte padded null_mask size. Using the bit-tight
    // (rows+63)/64*8 size causes cudf::column deep-copy to memcpy past the end
    // of the buffer and fail with cudaErrorInvalidValue for non-aligned row
    // counts (see gpu_native_decode.cu::decode_validity_mask).
    size_t mask_bytes =
      cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(total_rows));
    null_mask    = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask = static_cast<uint64_t*>(null_mask.data());

    uint32_t num_words = static_cast<uint32_t>((total_rows + 63) / 64);
    kernel_fill_valid<<<(num_words + 255) / 256, 256, 0, stream.value()>>>(d_mask, num_words);

    size_t val_row_offset = 0;
    for (auto& vseg : col_scan.validity.segments) {
      if (vseg.row_count == 0) {
        val_row_offset += vseg.row_count;
        continue;
      }
      if (vseg.persistent && vseg.data_ptr &&
          vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        size_t seg_mask_bytes = (vseg.row_count + 7) / 8;
        if (val_row_offset % 64 == 0) {
          bounce_h2d_async(d_mask + val_row_offset / 64,
                           vseg.data_ptr,
                           seg_mask_bytes,
                           stream.value());
        } else {
          bounce_h2d_async(reinterpret_cast<uint8_t*>(d_mask) + val_row_offset / 8,
                           vseg.data_ptr,
                           seg_mask_bytes,
                           stream.value());
        }
      }
      val_row_offset += vseg.row_count;
    }

    // DICT_FSST encodes NULL inline via dict_idx == 0.  The validity segment
    // is usually "Empty Validity" for those columns, so without this step
    // the mask stays all-valid and NULL rows appear as empty strings.
    if (!dict_fsst_descs.empty()) {
      kernel_dict_fsst_mark_nulls<<<static_cast<uint32_t>(dict_fsst_descs.size()),
                                    THREADS, 0, stream.value()>>>(
        d_df_descs,
        reinterpret_cast<unsigned long long*>(d_mask),
        static_cast<uint32_t>(dict_fsst_descs.size()));
    }

    rmm::device_uvector<uint32_t> d_vc(1, stream, mr);
    cudaMemsetAsync(d_vc.data(), 0, sizeof(uint32_t), stream.value());
    if (d_valid_count_out) {
      // Deferred: write valid count to caller's slot, NO sync.
      cudaMemsetAsync(d_valid_count_out, 0, sizeof(uint32_t), stream.value());
      kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_valid_count_out);
    } else {
      // Legacy path: sync per column.
      kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_vc.data());
      stream.synchronize();
      uint32_t vc;
      cudaMemcpy(&vc, d_vc.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
      null_count = static_cast<cudf::size_type>(total_rows - vc);
    }
  }

  SIRIUS_LOG_INFO(
    "[gpu_native_decode] batched string col: {} rows, {} chars, "
    "{} dict segs, {} fsst segs, {} uncomp segs, {} dict_fsst segs",
    total_rows,
    alloc_chars,
    dict_descs.size(),
    fsst_descs.size(),
    uncomp_descs.size(),
    dict_fsst_descs.size());

  return cudf::make_strings_column(static_cast<cudf::size_type>(total_rows),
                                   std::move(offsets_col),
                                   std::move(d_chars),
                                   null_count,
                                   std::move(null_mask));
}

}  // namespace sirius::cuda::scan
