/*
 * Copyright 2026, Sirius Contributors.
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

// sirius_context.hpp must be included before anything that opens the
// sirius::cuda namespace (e.g. cuda/scan/gpu_native_decode.cuh). Otherwise
// resource_ref_utils.hpp's unqualified `cuda::stream_ref` reference inside
// namespace sirius::memory resolves to the (empty) sirius::cuda namespace
// instead of the global ::cuda namespace from CCCL.
#include "sirius_context.hpp"

#include "op/scan/duckdb_native_scan_task.hpp"

#include "cuda/scan/gpu_native_decode.cuh"
#include "cudf/cudf_utils.hpp"
#include "helper/type_conversions.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_block_layout.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/detail/error.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/storage_manager.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

using ::sirius::cuda::scan::gpu_codec_run;
using ::sirius::cuda::scan::gpu_column_decode_input;
using ::sirius::cuda::scan::gpu_segment_desc;

constexpr char const* kTag = "[sirius_gpu_duckdb_native_scan]";

void throw_unsupported(std::string what)
{
  throw std::runtime_error(std::string(kTag) + " unsupported in V1 spike: " + std::move(what));
}

bool is_constant_or_empty_validity(duckdb::CompressionType c)
{
  return c == duckdb::CompressionType::COMPRESSION_CONSTANT ||
         c == duckdb::CompressionType::COMPRESSION_EMPTY;
}

bool is_supported_v1_data_codec(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_RLE:
    case duckdb::CompressionType::COMPRESSION_BITPACKING: return true;
    default: return false;
  }
}

bool column_has_real_nulls(duckdb_column_metadata const& col)
{
  for (auto const& v : col.validity_segments) {
    if (v.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) { return true; }
  }
  return false;
}

cudf::data_type sirius_to_cudf_type(sirius::logical_type const& t)
{
  // Round-trip through DuckDB::LogicalType to reuse GetCudfType, which already
  // handles DECIMAL precision/scale and timestamp units correctly.
  duckdb::vector<sirius::logical_type> wrap{t};
  auto duckdb_vec = sirius::to_duckdb_vec(wrap);
  return duckdb::GetCudfType(duckdb_vec[0]);
}

//===----------------------------------------------------------------------===//
// Host-side block pinning
//
// V1 uses BlockManager::RegisterBlock + BufferManager::Pin to get block-level
// host bytes. Each pinned BufferHandle keeps the underlying buffer resident
// for its lifetime; we hold a vector of them until the H2D memcpy is queued
// on the stream. The kernel reads only what its codec header dictates, so
// the bytes_size upper bound (block-end minus block_offset) is safe.
//===----------------------------------------------------------------------===//

struct pinned_segment_bytes {
  duckdb::BufferHandle handle;
  uint8_t const* host_ptr = nullptr;
  std::size_t bytes       = 0;
};

pinned_segment_bytes pin_block(duckdb::BlockManager& block_manager,
                               duckdb::BufferManager& buffer_manager,
                               duckdb::block_id_t block_id,
                               std::size_t block_offset)
{
  if (block_id < 0) {
    throw std::runtime_error(std::string(kTag) +
                             " pin_block called with block_id<0 (CONSTANT segment?)");
  }
  auto handle  = block_manager.RegisterBlock(block_id);
  auto pinned  = buffer_manager.Pin(handle);
  auto* base   = pinned.Ptr();
  auto bytes   = block_manager.GetBlockSize();
  if (block_offset > bytes) {
    throw std::runtime_error(std::string(kTag) + " block_offset (" + std::to_string(block_offset) +
                             ") exceeds block size (" + std::to_string(bytes) + ")");
  }
  pinned_segment_bytes out;
  out.handle   = std::move(pinned);
  out.host_ptr = base + block_offset;
  out.bytes    = bytes - block_offset;
  return out;
}

// Pin main block + each additional block; concatenate into one owned buffer.
// The descriptor's block_offset applies only to the main block; additional
// blocks are taken whole-block. The resulting buffer has main-block bytes
// (from block_offset to end) followed by each additional block's full bytes.
//
// Whether the on-disk codec actually arranges its dictionary/heap to be
// readable as one contiguous slab is codec-dependent. For FSST/DICT_FSST
// inline-symbol-table segments the main block alone is enough; the concat
// is here for codecs that visit_block_ids.
pinned_segment_bytes pin_block_with_additional(
  duckdb::BlockManager& block_manager,
  duckdb::BufferManager& buffer_manager,
  duckdb::block_id_t main_block_id,
  std::size_t block_offset,
  std::vector<duckdb::block_id_t> const& additional_block_ids)
{
  auto main_pinned       = block_manager.RegisterBlock(main_block_id);
  auto main_handle       = buffer_manager.Pin(main_pinned);
  auto block_size        = block_manager.GetBlockSize();
  auto main_payload_size = block_size > block_offset ? block_size - block_offset : std::size_t{0};

  std::vector<duckdb::BufferHandle> handles;
  handles.push_back(std::move(main_handle));
  std::vector<uint8_t> concat;
  concat.resize(main_payload_size);
  std::memcpy(concat.data(), handles.front().Ptr() + block_offset, main_payload_size);

  for (auto add_id : additional_block_ids) {
    auto add_handle = block_manager.RegisterBlock(add_id);
    auto h          = buffer_manager.Pin(add_handle);
    auto offset     = concat.size();
    concat.resize(offset + block_size);
    std::memcpy(concat.data() + offset, h.Ptr(), block_size);
    handles.push_back(std::move(h));
  }

  pinned_segment_bytes out;
  out.owned_bytes = std::move(concat);
  out.handles     = std::move(handles);
  out.host_ptr    = out.owned_bytes.data();
  out.bytes       = out.owned_bytes.size();
  return out;
}

//===----------------------------------------------------------------------===//
// sirius_io variant: read .db block payloads via sirius_ioctx::host_read.
//
// Equivalent to pin_block / pin_block_with_additional but bypasses DuckDB's
// BufferManager. Used when the scan_manager is configured with
// use_sirius_datasource=true so the duckdb-native scan path goes through the
// same I/O substrate as parquet. Output shape matches BufferManager.Pin: a
// payload-only host pointer with `block_size - block_offset` bytes for the
// main block, plus full-block payloads concatenated for additional blocks.
//===----------------------------------------------------------------------===//

pinned_segment_bytes read_block_via_io(::sirius::io::sirius_ioctx& ctx,
                                       ::sirius::io::sirius_io_object& obj,
                                       duckdb::BlockManager const& bm,
                                       duckdb::block_id_t block_id,
                                       std::size_t block_offset)
{
  const std::size_t block_size  = bm.GetBlockSize();
  const std::size_t payload_off = duckdb_block_payload_offset(bm, block_id);
  pinned_segment_bytes out;
  out.owned_bytes.resize(block_size);
  const std::size_t got = ctx.host_read(obj, payload_off, block_size, out.owned_bytes.data());
  if (got != block_size) {
    throw std::runtime_error(std::string(kTag) + " short host_read for block_id " +
                             std::to_string(block_id) + ": got " + std::to_string(got) +
                             " expected " + std::to_string(block_size));
  }
  if (block_offset > block_size) {
    throw std::runtime_error(std::string(kTag) + " block_offset (" + std::to_string(block_offset) +
                             ") exceeds block size (" + std::to_string(block_size) + ")");
  }
  out.host_ptr = out.owned_bytes.data() + block_offset;
  out.bytes    = block_size - block_offset;
  return out;
}

pinned_segment_bytes read_blocks_with_additional_via_io(
  ::sirius::io::sirius_ioctx& ctx,
  ::sirius::io::sirius_io_object& obj,
  duckdb::BlockManager const& bm,
  duckdb::block_id_t main_block_id,
  std::size_t block_offset,
  std::vector<duckdb::block_id_t> const& additional_block_ids)
{
  const std::size_t block_size = bm.GetBlockSize();
  const std::size_t main_payload_size =
    block_size > block_offset ? block_size - block_offset : std::size_t{0};

  std::vector<uint8_t> concat;
  concat.resize(main_payload_size + additional_block_ids.size() * block_size);

  // Main block: read full payload then memcpy starting at block_offset.
  {
    std::vector<uint8_t> main_buf(block_size);
    const std::size_t got =
      ctx.host_read(obj, duckdb_block_payload_offset(bm, main_block_id), block_size, main_buf.data());
    if (got != block_size) {
      throw std::runtime_error(std::string(kTag) + " short host_read for main block_id " +
                               std::to_string(main_block_id));
    }
    if (main_payload_size > 0) {
      std::memcpy(concat.data(), main_buf.data() + block_offset, main_payload_size);
    }
  }

  // Additional blocks: read each full-payload directly into the concat buffer.
  std::size_t dst_off = main_payload_size;
  for (auto add_id : additional_block_ids) {
    const std::size_t got = ctx.host_read(
      obj, duckdb_block_payload_offset(bm, add_id), block_size, concat.data() + dst_off);
    if (got != block_size) {
      throw std::runtime_error(std::string(kTag) + " short host_read for additional block_id " +
                               std::to_string(add_id));
    }
    dst_off += block_size;
  }

  pinned_segment_bytes out;
  out.owned_bytes = std::move(concat);
  out.host_ptr    = out.owned_bytes.data();
  out.bytes       = out.owned_bytes.size();
  return out;
}

//===----------------------------------------------------------------------===//
// CONSTANT extraction.
//
// CONSTANT segments have block_id == -1; the constant value lives in
// per-(rg, col) statistics. We pull stats from PartitionRowGroup at scan
// time and copy the value into an owned buffer the kernel can read.
//===----------------------------------------------------------------------===//

template <typename T>
void store_typed_min(duckdb::BaseStatistics const& stats, std::vector<uint8_t>& out)
{
  auto v = duckdb::NumericStats::GetMin<T>(stats);
  out.resize(sizeof(T));
  std::memcpy(out.data(), &v, sizeof(T));
}

pinned_segment_bytes extract_constant_bytes(duckdb::BaseStatistics const& stats,
                                            sirius::logical_type const& sirius_type)
{
  auto duckdb_type = sirius_to_duckdb_type(sirius_type);
  pinned_segment_bytes out;
  switch (duckdb_type.InternalType()) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8: store_typed_min<int8_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT16: store_typed_min<int16_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT32: store_typed_min<int32_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT64: store_typed_min<int64_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT8: store_typed_min<uint8_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT16: store_typed_min<uint16_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT32: store_typed_min<uint32_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT64: store_typed_min<uint64_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::FLOAT: store_typed_min<float>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::DOUBLE: store_typed_min<double>(stats, out.owned_bytes); break;
    default:
      throw_unsupported("CONSTANT extraction for physical type " +
                        std::to_string(static_cast<int>(duckdb_type.InternalType())));
  }
  out.host_ptr = out.owned_bytes.data();
  out.bytes    = out.owned_bytes.size();
  return out;
}

//===----------------------------------------------------------------------===//
// ROARING validity host-decode.
//
// Walker output gives us block_id + block_offset + segment_count but no
// ColumnSegment*. We reconstruct one via ColumnSegment::CreatePersistentSegment
// (public factory) and drive RoaringScanState to fill a host bitmap. The
// bitmap then ships to the device as plain UNCOMPRESSED validity.
//===----------------------------------------------------------------------===//

pinned_segment_bytes decode_roaring_validity(duckdb::DatabaseInstance& db,
                                             duckdb::BlockManager& block_manager,
                                             duckdb_segment_descriptor const& desc)
{
  auto validity_type = duckdb::LogicalType(duckdb::LogicalTypeId::VALIDITY);
  auto seg           = duckdb::ColumnSegment::CreatePersistentSegment(
    db,
    block_manager,
    desc.block_id,
    desc.block_offset,
    validity_type,
    desc.segment_count,
    duckdb::CompressionType::COMPRESSION_ROARING,
    duckdb::BaseStatistics::CreateEmpty(validity_type),
    /*segment_state=*/nullptr);

  auto const row_count = static_cast<duckdb::idx_t>(desc.segment_count);
  pinned_segment_bytes out;
  std::size_t const words = (row_count + 63) / 64;
  out.owned_bytes.assign(words * sizeof(uint64_t), 0xff);

  duckdb::roaring::RoaringScanState rs(*seg);
  constexpr duckdb::idx_t CHUNK =
    static_cast<duckdb::idx_t>(duckdb::roaring::ROARING_CONTAINER_SIZE);
  duckdb::Vector tmp(duckdb::LogicalType::BOOLEAN, CHUNK);

  for (duckdb::idx_t scanned = 0; scanned < row_count; scanned += CHUNK) {
    auto const to_scan = std::min<duckdb::idx_t>(CHUNK, row_count - scanned);
    auto& vm           = duckdb::FlatVector::Validity(tmp);
    vm.SetAllValid(CHUNK);
    rs.ScanPartial(scanned, tmp, /*offset=*/0, to_scan);
    if (!vm.AllValid()) {
      std::size_t const byte_offset   = scanned / 8;
      std::size_t const bytes_to_copy = (to_scan + 7) / 8;
      std::memcpy(out.owned_bytes.data() + byte_offset,
                  reinterpret_cast<uint8_t const*>(vm.GetData()),
                  bytes_to_copy);
    }
  }

  out.host_ptr = out.owned_bytes.data();
  out.bytes    = out.owned_bytes.size();
  return out;
}

<<<<<<< HEAD
=======
//===----------------------------------------------------------------------===//
// Per-split staging
//===----------------------------------------------------------------------===//

struct staged_segment {
  std::size_t device_offset    = 0;
  std::size_t bytes            = 0;
  uint32_t row_offset          = 0;
  uint32_t row_count           = 0;
  uint32_t max_string_length   = 0;
  duckdb::CompressionType compression{duckdb::CompressionType::COMPRESSION_AUTO};
};

struct staged_column {
  std::vector<staged_segment> data;
  std::vector<staged_segment> validity;
  bool has_nulls         = false;
  std::size_t total_rows = 0;
  bool is_varchar        = false;
};

struct staging_state {
  std::vector<pinned_segment_bytes> pinned;
  std::vector<uint8_t const*> src_ptrs;
  std::vector<std::size_t> src_sizes;
  std::vector<std::size_t> dst_offsets;
  std::size_t running_offset = 0;
};

void record_staged(staging_state& s, pinned_segment_bytes p, staged_segment& out)
{
  // Align each segment's device destination to 16 bytes. Several decode
  // kernels cast `seg.d_bytes + internal_offset` to T*/uint32_t* (e.g. RLE
  // values, dictionary indices), and back-to-back packing made later segments
  // start at arbitrary offsets whenever an earlier segment's bytes_size was
  // not a multiple of the consumer type's alignof. 16 bytes covers every
  // integer width we read up to uint128. Padding bytes are uninitialized but
  // unreferenced — kernels only read inside [device_offset, device_offset +
  // bytes_size). See gpu_decode_rle.cu:467 (uint64) where misaligned d_values
  // raised cudaErrorMisalignedAddress on q40.
  constexpr std::size_t SEGMENT_ALIGN = 16;
  s.running_offset = (s.running_offset + SEGMENT_ALIGN - 1) & ~(SEGMENT_ALIGN - 1);
  out.bytes         = p.bytes;
  out.device_offset = s.running_offset;
  s.src_ptrs.push_back(p.host_ptr);
  s.src_sizes.push_back(p.bytes);
  s.dst_offsets.push_back(s.running_offset);
  s.running_offset += p.bytes;
  s.pinned.push_back(std::move(p));
}

duckdb::BaseStatistics const& constant_stats_for(
  std::vector<duckdb::PartitionStatistics> const& partition_stats,
  duckdb::idx_t rg_idx,
  duckdb::idx_t storage_idx,
  std::vector<std::unique_ptr<duckdb::BaseStatistics>>& owned_stats_cache)
{
  if (rg_idx >= partition_stats.size() || !partition_stats[rg_idx].partition_row_group) {
    throw std::runtime_error(std::string(kTag) +
                             " no PartitionRowGroup for CONSTANT lookup on rg " +
                             std::to_string(rg_idx));
  }
  auto stats = partition_stats[rg_idx].partition_row_group->GetColumnStatistics(
    duckdb::StorageIndex(storage_idx));
  if (!stats) {
    throw std::runtime_error(std::string(kTag) +
                             " PartitionRowGroup returned null stats for CONSTANT lookup");
  }
  owned_stats_cache.push_back(std::move(stats));
  return *owned_stats_cache.back();
}

// Read a single block payload via the io substrate when both ctx + obj are
// non-null, otherwise fall back to DuckDB's BufferManager. Output shape
// matches pin_block: payload pointer + remaining bytes from block_offset.
pinned_segment_bytes read_block_payload(::sirius::io::sirius_ioctx* io_ctx,
                                        ::sirius::io::sirius_io_object* io_obj,
                                        duckdb::BlockManager& bm,
                                        duckdb::BufferManager& buffer_manager,
                                        duckdb::block_id_t block_id,
                                        std::size_t block_offset,
                                        std::vector<duckdb::block_id_t> const& additional_block_ids)
{
  const bool via_io = (io_ctx != nullptr && io_obj != nullptr);
  if (via_io) {
    if (additional_block_ids.empty()) {
      return read_block_via_io(*io_ctx, *io_obj, bm, block_id, block_offset);
>>>>>>> 9e9bda03 (fix(native-scan): pad staging dst_offset to 16B per segment)
    }
    staged_cols.push_back(stage_one_column(pinned,
                                           src_ptrs,
                                           src_sizes,
                                           dst_offsets,
                                           running_offset,
                                           block_manager,
                                           buffer_manager,
                                           split.row_groups,
                                           ci));
  }

  rmm::device_buffer device_buf(running_offset, stream, mr_ref);
  if (running_offset > 0) {
    copy_staged_to_device(device_buf, src_ptrs, src_sizes, dst_offsets, stream);
  }

  // Build fixed-width column inputs for gpu_decode_table.
  std::vector<gpu_column_decode_input> fw_inputs;
  std::vector<std::size_t> fw_to_final_idx;
  fw_inputs.reserve(num_cols);
  fw_to_final_idx.reserve(num_cols);

  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (is_rowid_col[ci]) continue;

    auto const& staged = staged_cols[ci];
    gpu_column_decode_input input;
    input.out_type   = sirius_to_cudf_type(scan_info.projected_types[ci]);
    input.total_rows = static_cast<uint32_t>(staged.total_rows);
    input.has_nulls  = staged.has_nulls;
    fill_runs(staged.data, device_buf, input.data);
    fill_runs(staged.validity, device_buf, input.validity);
    fw_inputs.push_back(std::move(input));
    fw_to_final_idx.push_back(ci);
  }

  std::unique_ptr<cudf::table> decoded_table;
  if (!fw_inputs.empty()) {
    decoded_table = ::sirius::cuda::scan::gpu_decode_table(fw_inputs, stream, mr_ref);
  }

  std::vector<std::unique_ptr<cudf::column>> final_cols(num_cols);
  std::vector<std::unique_ptr<cudf::column>> fw_cols;
  if (decoded_table) { fw_cols = decoded_table->release(); }

  for (std::size_t fi = 0; fi < fw_cols.size(); ++fi) {
    final_cols[fw_to_final_idx[fi]] = std::move(fw_cols[fi]);
  }
  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (!is_rowid_col[ci]) continue;
    final_cols[ci] = build_rowid_column(split.row_groups,
                                        static_cast<cudf::size_type>(total_rows),
                                        stream,
                                        mr_ref);
  }

  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace sirius::op::scan
