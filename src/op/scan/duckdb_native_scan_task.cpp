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

#include "cuda/scan/gpu_decode_strings.cuh"
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

#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/compression/roaring/roaring.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/column_segment.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

using ::sirius::cuda::scan::gpu_codec_run;
using ::sirius::cuda::scan::gpu_column_decode_input;
using ::sirius::cuda::scan::gpu_segment_desc;
using ::sirius::cuda::scan::gpu_string_codec_run;
using ::sirius::cuda::scan::gpu_string_column_decode_input;
using ::sirius::cuda::scan::gpu_string_segment_desc;

constexpr char const* kTag = "[sirius_gpu_duckdb_native_scan]";

void throw_unsupported(std::string what)
{
  throw std::runtime_error(std::string(kTag) + " unsupported: " + std::move(what));
}

bool is_constant_or_empty_validity(duckdb::CompressionType c)
{
  return c == duckdb::CompressionType::COMPRESSION_CONSTANT ||
         c == duckdb::CompressionType::COMPRESSION_EMPTY;
}

bool is_supported_fixed_width_codec(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_RLE:
    case duckdb::CompressionType::COMPRESSION_BITPACKING: return true;
    default: return false;
  }
}

bool is_supported_varchar_codec(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_DICTIONARY:
    case duckdb::CompressionType::COMPRESSION_FSST:
    case duckdb::CompressionType::COMPRESSION_DICT_FSST: return true;
    default: return false;
  }
}

bool column_has_real_nulls(duckdb_column_metadata const& col)
{
  for (auto const& v : col.validity_segments) {
    auto c = v.compression;
    if (c == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED ||
        c == duckdb::CompressionType::COMPRESSION_ROARING) {
      return true;
    }
  }
  return false;
}

cudf::data_type sirius_to_cudf_type(sirius::logical_type const& t)
{
  duckdb::vector<sirius::logical_type> wrap{t};
  auto duckdb_vec = sirius::to_duckdb_vec(wrap);
  return duckdb::GetCudfType(duckdb_vec[0]);
}

duckdb::LogicalType sirius_to_duckdb_type(sirius::logical_type const& t)
{
  duckdb::vector<sirius::logical_type> wrap{t};
  auto duckdb_vec = sirius::to_duckdb_vec(wrap);
  return duckdb_vec[0];
}

//===----------------------------------------------------------------------===//
// Host bytes for a segment.
//
// Three sources: a pinned BufferHandle (normal block), owned host bytes
// (CONSTANT extracted from stats, ROARING host-decoded, or main+additional
// blocks concatenated), or both (BufferHandle pinning the main block plus
// owned bytes from the additional-block concat that follows). We keep the
// handle alive until H2D is queued.
//===----------------------------------------------------------------------===//

struct pinned_segment_bytes {
  std::vector<duckdb::BufferHandle> handles;  // empty when source is owned_bytes only
  std::vector<uint8_t> owned_bytes;           // used for CONSTANT, ROARING, concat
  uint8_t const* host_ptr = nullptr;
  std::size_t bytes       = 0;
};

// Pin a single block and return its host pointer + remaining bytes from
// `block_offset` to end-of-block.
pinned_segment_bytes pin_block(duckdb::BlockManager& block_manager,
                               duckdb::BufferManager& buffer_manager,
                               duckdb::block_id_t block_id,
                               std::size_t block_offset)
{
  auto handle = block_manager.RegisterBlock(block_id);
  auto pinned = buffer_manager.Pin(handle);
  auto* base  = pinned.Ptr();
  auto bytes  = block_manager.GetBlockSize();
  if (block_offset > bytes) {
    throw std::runtime_error(std::string(kTag) + " block_offset (" + std::to_string(block_offset) +
                             ") exceeds block size (" + std::to_string(bytes) + ")");
  }
  pinned_segment_bytes out;
  out.handles.push_back(std::move(pinned));
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
    }
  }

  rmm::device_buffer device_buf(staging.running_offset, stream, mr_ref);
  if (staging.running_offset > 0) { copy_staged_to_device(device_buf, staging, stream); }

  // Group fixed-width columns for a single gpu_decode_table call; varchar
  // columns each go through gpu_decode_strings_column separately.
  std::vector<gpu_column_decode_input> fw_inputs;
  std::vector<std::size_t> fw_to_final_idx;
  std::vector<gpu_string_column_decode_input> vc_inputs;
  std::vector<std::size_t> vc_to_final_idx;
  fw_inputs.reserve(num_cols);
  fw_to_final_idx.reserve(num_cols);

  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (is_rowid_col[ci]) continue;
    auto const& staged = staged_cols[ci];
    if (staged.is_varchar) {
      gpu_string_column_decode_input input;
      input.total_rows = static_cast<uint32_t>(staged.total_rows);
      input.has_nulls  = staged.has_nulls;
      fill_string_runs(staged.data, device_buf, input.data);
      fill_fixed_width_runs(staged.validity, device_buf, input.validity);
      vc_inputs.push_back(std::move(input));
      vc_to_final_idx.push_back(ci);
    } else {
      gpu_column_decode_input input;
      input.out_type   = sirius_to_cudf_type(scan_info.projected_types[ci]);
      input.total_rows = static_cast<uint32_t>(staged.total_rows);
      input.has_nulls  = staged.has_nulls;
      fill_fixed_width_runs(staged.data, device_buf, input.data);
      fill_fixed_width_runs(staged.validity, device_buf, input.validity);
      fw_inputs.push_back(std::move(input));
      fw_to_final_idx.push_back(ci);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> fw_cols;
  if (!fw_inputs.empty()) {
    auto fw_table = ::sirius::cuda::scan::gpu_decode_table(fw_inputs, stream, mr_ref);
    fw_cols       = fw_table->release();
  }

  std::vector<std::unique_ptr<cudf::column>> vc_cols;
  vc_cols.reserve(vc_inputs.size());
  for (auto const& vc : vc_inputs) {
    vc_cols.push_back(::sirius::cuda::scan::gpu_decode_strings_column(vc, stream, mr_ref));
  }

  std::vector<std::unique_ptr<cudf::column>> final_cols(num_cols);
  for (std::size_t fi = 0; fi < fw_cols.size(); ++fi) {
    final_cols[fw_to_final_idx[fi]] = std::move(fw_cols[fi]);
  }
  for (std::size_t vi = 0; vi < vc_cols.size(); ++vi) {
    final_cols[vc_to_final_idx[vi]] = std::move(vc_cols[vi]);
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
