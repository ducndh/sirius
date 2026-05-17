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
#include "log/logging.hpp"
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

//===----------------------------------------------------------------------===//
// Per-split staging
//
// All segments across all columns get concatenated into one device buffer.
// Each gpu_segment_desc points at the right slice. One H2D memcpy per
// segment via cudaMemcpyAsync — coalescing into a single bulk copy is a
// V2 optimization.
//===----------------------------------------------------------------------===//

struct staged_segment {
  std::size_t device_offset = 0;
  std::size_t bytes         = 0;
  uint32_t row_offset       = 0;
  uint32_t row_count        = 0;
  duckdb::CompressionType compression{duckdb::CompressionType::COMPRESSION_AUTO};
};

struct staged_column {
  std::vector<staged_segment> data;
  std::vector<staged_segment> validity;
  bool has_nulls = false;
  std::size_t total_rows = 0;
};

void stage_segment(std::vector<pinned_segment_bytes>& pinned,
                   std::vector<uint8_t const*>& src_ptrs,
                   std::vector<std::size_t>& src_sizes,
                   std::vector<std::size_t>& dst_offsets,
                   std::size_t& running_offset,
                   duckdb::BlockManager& block_manager,
                   duckdb::BufferManager& buffer_manager,
                   duckdb_segment_descriptor const& desc,
                   staged_segment& out_seg)
{
  auto p = pin_block(block_manager, buffer_manager, desc.block_id, desc.block_offset);
  // additional_blocks (FSST/DICT heaps) are V2; the V1 codec set
  // (UNCOMPRESSED/RLE/BITPACKING) never produces them on fixed-width columns.
  if (!desc.additional_blocks.empty()) {
    throw_unsupported("additional_blocks (variable-width payload) not staged in V1");
  }

  out_seg.bytes         = p.bytes;
  out_seg.device_offset = running_offset;
  src_ptrs.push_back(p.host_ptr);
  src_sizes.push_back(p.bytes);
  dst_offsets.push_back(running_offset);
  running_offset += p.bytes;
  pinned.push_back(std::move(p));
}

staged_column stage_one_column(std::vector<pinned_segment_bytes>& pinned,
                               std::vector<uint8_t const*>& src_ptrs,
                               std::vector<std::size_t>& src_sizes,
                               std::vector<std::size_t>& dst_offsets,
                               std::size_t& running_offset,
                               duckdb::BlockManager& block_manager,
                               duckdb::BufferManager& buffer_manager,
                               std::vector<duckdb_row_group_metadata> const& row_groups,
                               std::size_t projected_col_idx)
{
  staged_column out;

  uint32_t row_cursor = 0;
  for (auto const& rg : row_groups) {
    auto const& col_md = rg.columns.at(projected_col_idx);
    for (auto const& seg : col_md.data_segments) {
      if (!is_supported_v1_data_codec(seg.compression)) {
        throw_unsupported("data codec " + std::to_string(static_cast<int>(seg.compression)) +
                          " (column " + std::to_string(col_md.column_id) + ")");
      }
      staged_segment s;
      s.row_offset  = row_cursor + static_cast<uint32_t>(seg.segment_start);
      s.row_count   = static_cast<uint32_t>(seg.segment_count);
      s.compression = seg.compression;
      stage_segment(pinned,
                    src_ptrs,
                    src_sizes,
                    dst_offsets,
                    running_offset,
                    block_manager,
                    buffer_manager,
                    seg,
                    s);
      out.data.push_back(s);
    }

    if (column_has_real_nulls(col_md)) { out.has_nulls = true; }
    for (auto const& vseg : col_md.validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression)) {
        // Decoder's all-valid pre-fill is correct; do not stage.
        continue;
      }
      if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        throw_unsupported("ROARING validity (column " + std::to_string(col_md.column_id) + ")");
      }
      if (vseg.compression != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        throw_unsupported("validity codec " + std::to_string(static_cast<int>(vseg.compression)) +
                          " (column " + std::to_string(col_md.column_id) + ")");
      }
      staged_segment vs;
      vs.row_offset  = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count   = static_cast<uint32_t>(vseg.segment_count);
      vs.compression = vseg.compression;
      stage_segment(pinned,
                    src_ptrs,
                    src_sizes,
                    dst_offsets,
                    running_offset,
                    block_manager,
                    buffer_manager,
                    vseg,
                    vs);
      out.validity.push_back(vs);
    }
    row_cursor += static_cast<uint32_t>(rg.row_count);
  }

  out.total_rows = row_cursor;
  return out;
}

//===----------------------------------------------------------------------===//
// Bulk H2D copy onto the stream.
//===----------------------------------------------------------------------===//

void copy_staged_to_device(rmm::device_buffer& device_buf,
                           std::vector<uint8_t const*> const& src_ptrs,
                           std::vector<std::size_t> const& src_sizes,
                           std::vector<std::size_t> const& dst_offsets,
                           rmm::cuda_stream_view stream)
{
  auto* device_base = static_cast<uint8_t*>(device_buf.data());
  for (std::size_t i = 0; i < src_ptrs.size(); ++i) {
    RMM_CUDA_TRY(cudaMemcpyAsync(device_base + dst_offsets[i],
                                 src_ptrs[i],
                                 src_sizes[i],
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
  }
  // Buffer-handle release after this returns is safe: pageable-source H2D
  // semantics block on the source buffer being valid for the duration of
  // the call.  cudaMemcpyAsync from pageable host into device is in fact
  // synchronous w.r.t. the host (CUDA API guarantee) so by the time we
  // unwind, every byte has left host memory.
}

//===----------------------------------------------------------------------===//
// Build per-codec runs from a staged_column.
//===----------------------------------------------------------------------===//

void fill_runs(std::vector<staged_segment> const& staged,
               rmm::device_buffer const& device_buf,
               std::vector<gpu_codec_run>& out_runs)
{
  out_runs.clear();
  duckdb::CompressionType current = duckdb::CompressionType::COMPRESSION_AUTO;
  auto* device_base = static_cast<uint8_t const*>(device_buf.data());
  for (auto const& s : staged) {
    if (out_runs.empty() || s.compression != current) {
      out_runs.push_back({s.compression, {}});
      current = s.compression;
    }
    gpu_segment_desc seg{};
    seg.d_bytes    = device_base + s.device_offset;
    seg.bytes_size = static_cast<uint32_t>(std::min<std::size_t>(s.bytes, UINT32_MAX));
    seg.row_offset = s.row_offset;
    seg.row_count  = s.row_count;
    out_runs.back().segments.push_back(seg);
  }
}

//===----------------------------------------------------------------------===//
// Rowid synthesis.
//
// For each row group in the split, write a sequence starting at
// rg.row_group_start. Output is a non-nullable INT64 column whose rows are
// the absolute row IDs of the scanned rows.
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> build_rowid_column(
  std::vector<duckdb_row_group_metadata> const& row_groups,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> per_rg;
  per_rg.reserve(row_groups.size());
  for (auto const& rg : row_groups) {
    if (rg.row_count == 0) continue;
    auto init = cudf::numeric_scalar<std::int64_t>(static_cast<std::int64_t>(rg.row_group_start),
                                                   true,
                                                   stream,
                                                   mr);
    per_rg.push_back(cudf::sequence(static_cast<cudf::size_type>(rg.row_count), init, stream, mr));
  }
  if (per_rg.empty()) {
    auto empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           total_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
    return empty;
  }
  if (per_rg.size() == 1) { return std::move(per_rg[0]); }
  std::vector<cudf::column_view> views;
  views.reserve(per_rg.size());
  for (auto const& c : per_rg) { views.push_back(c->view()); }
  return cudf::concatenate(views, stream, mr);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Public entry: pick_gpu_memory_space_for_duckdb_native_scan
//===----------------------------------------------------------------------===//

cucascade::memory::memory_space* pick_gpu_memory_space_for_duckdb_native_scan(
  duckdb_native_scan_info const& scan_info)
{
  if (scan_info.context == nullptr) {
    throw std::runtime_error(std::string(kTag) + " scan_info.context is null");
  }
  auto& ctx       = *scan_info.context;
  auto sirius_st  = ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_st) {
    throw std::runtime_error(std::string(kTag) + " no sirius_state on the ClientContext");
  }
  auto& mem_mgr   = sirius_st->get_memory_manager();
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error(std::string(kTag) + " no GPU-tier memory space registered");
  }
  return const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
}

//===----------------------------------------------------------------------===//
// Public entry: decode_duckdb_native_split
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> decode_duckdb_native_split(
  scan_manager::duckdb_native_split_provider::split_payload const& split,
  cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  if (split.row_groups.empty()) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }
  auto const& scan_info = *split.scan_info;
  auto& storage         = *scan_info.storage;
  auto& context         = *scan_info.context;

  auto& sm             = storage.GetAttached().GetStorageManager();
  auto& block_manager  = sm.GetBlockManager();
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  auto mr_ref = mem_space.get_default_allocator();

  std::size_t const num_cols = scan_info.projected_cols.size();

  std::size_t total_rows = 0;
  for (auto const& rg : split.row_groups) { total_rows += rg.row_count; }

  if (total_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error(std::string(kTag) + " split rows (" + std::to_string(total_rows) +
                             ") exceed cudf::size_type max");
  }

  std::vector<pinned_segment_bytes> pinned;
  std::vector<uint8_t const*> src_ptrs;
  std::vector<std::size_t> src_sizes;
  std::vector<std::size_t> dst_offsets;
  std::size_t running_offset = 0;

  std::vector<staged_column> staged_cols;
  staged_cols.reserve(num_cols);
  std::vector<bool> is_rowid_col(num_cols, false);

  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    auto const& pcol = scan_info.projected_cols[ci];
    if (pcol.is_rowid) {
      is_rowid_col[ci] = true;
      staged_cols.emplace_back();  // placeholder; rowids handled separately
      staged_cols.back().total_rows = total_rows;
      continue;
    }
    if (scan_info.projected_types[ci].is_varchar()) {
      throw_unsupported("VARCHAR column (projected_idx=" + std::to_string(ci) + ")");
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
