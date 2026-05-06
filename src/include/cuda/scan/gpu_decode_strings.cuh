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

#pragma once

//===----------------------------------------------------------------------===//
// String-codec decode entry for the GPU-native scan path.
//
// Sibling to gpu_native_decode.cuh's `gpu_decode_table` — that entry handles
// fixed-width data because its output shape is a flat `rows * type_size`
// buffer per column. Strings need an offsets+chars pair sized by per-row
// decoded lengths, so they get their own orchestrator that runs the standard
// two-pass shape:
//
//   pass 1   per-codec kernel writes per-row decoded lengths into d_lengths
//   prefix   one cub::DeviceScan::ExclusiveSum over d_lengths produces offsets
//   pass 2   per-codec kernel writes char bytes at d_offsets[i]
//
// Codecs supported: DICTIONARY, FSST, DICT_FSST. The legacy two-phase
// pipeline (`decode_string_column_batched` in gpu_decode_batched_string.cu)
// stays in place for the iceberg / parquet paths; this header is for the
// GPU-native DuckDB scan converter only.
//
// Header parsing happens on device (matches gpu_native_decode.cuh's "kernels
// parse their own headers" contract). One exception: FSST and DICT_FSST
// symbol tables are deserialized via DuckDB's opaque `duckdb_fsst_import` on
// host, then concatenated and uploaded as part of the descriptor arena —
// the format is opaque to user code and reimplementing it on device would
// pin us to a specific FSST library version.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_native_decode.cuh"

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::cuda::scan {

/// Per-segment descriptor for the string-codec decode path.
///
/// Mirrors `gpu_segment_desc` from gpu_native_decode.cuh — same row_offset /
/// row_count meaning — and adds `seg_row_start` so the orchestrator can chunk
/// large segments across CTAs without breaking per-segment prefix-sum state.
/// Codec-specific metadata (FSST symbol-table index, DICT_FSST mode etc.)
/// is parsed on device from the segment bytes; nothing leaks into this struct.
struct gpu_string_segment_desc {
  uint8_t const* d_bytes;       ///< device pointer to the segment's first byte
  uint32_t bytes_size;          ///< size of the buffer behind d_bytes
  uint32_t row_offset;          ///< column-relative starting row index
  uint32_t row_count;           ///< rows produced by this segment (or chunk slice)
  uint32_t seg_row_start;       ///< rows skipped at the segment head when chunked
  uint32_t max_string_length;   ///< DuckDB segment-stat upper bound on decoded
                                ///  chars per row; 0 = unknown (forces a sync)
};

/// A run of string segments that share the same codec — the unit one
/// per-codec kernel batches over.
struct gpu_string_codec_run {
  duckdb::CompressionType codec;
  std::vector<gpu_string_segment_desc> segments;
};

/// Inputs for one varchar column. Multiple codec runs allowed; the single
/// orchestrator below shares d_lengths, the prefix sum, and d_chars across
/// all runs so a column with mixed DICTIONARY + FSST + DICT_FSST segments
/// pays one prefix sum, not three.
///
/// `validity` follows the same UNCOMPRESSED-only contract as the fixed-width
/// dispatcher (other validity codecs are decoded host-side upstream and
/// arrive here as plain validity bytes).
struct gpu_string_column_decode_input {
  std::vector<gpu_string_codec_run> data;
  std::vector<gpu_codec_run> validity;
  uint32_t total_rows;
  bool has_nulls;
};

/// Decode one varchar column. Pulls every codec_run in `col.data` through
/// the two-pass shape and returns a cudf strings column. Validity, if
/// present, is built from `col.validity`; for DICT_FSST a mark_nulls pass
/// also clears bits whose dictionary index is 0 (DuckDB encodes inline
/// NULLs that way and ships an "Empty Validity" segment for them).
///
/// All device work is enqueued on `stream`. The orchestrator does ONE host
/// sync — the read-back of total_chars — and only when the host-side upper
/// bound from `max_string_length` exceeds a safety cap; otherwise the chars
/// buffer is sized from the upper bound and the call is fully async.
///
/// Allocations go through `mr`. Throws on malformed segment metadata
/// (segment row range out of bounds, validity non-byte-aligned, codec not
/// in the supported set above).
std::unique_ptr<cudf::column> gpu_decode_strings_column(
  gpu_string_column_decode_input const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
