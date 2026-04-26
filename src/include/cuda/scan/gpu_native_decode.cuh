/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <op/scan/direct_block_scan.hpp>

#include <cudf/table/table.hpp>
#include <duckdb/common/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::cuda::scan {

/// @brief Stage every unique block referenced by any column of `col_scans` into
/// one contiguous device buffer. Returns (block_id → byte offset in buffer, owning buffer).
/// Reused by the preloaded-scan-region SQL function to populate an ENCODED region.
std::pair<std::unordered_map<int64_t, std::size_t>, rmm::device_buffer> stage_blocks_bulk_h2d(
  const std::vector<sirius::op::scan::column_scan_result>& col_scans,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);


/// @brief Decode all columns from direct block scan results into a cudf::table on GPU.
///
/// Pre-stages every unique block referenced by any column to device memory in one
/// coalesced bulk H2D (runs of contiguous block_ids share a single cudaMemcpyAsync),
/// then decodes all segments from device staging.  Supported compression types are
/// decoded via GPU kernels (bitpacking, dictionary, FSST, RLE, uncompressed,
/// constant); unsupported segments cause the scan to fall back to duckdb_scan_task.
///
/// A single stream.synchronize() runs at the end, after all per-column null-count
/// kernels have been enqueued — replacing N per-column syncs with 1.
///
/// @param col_scans   Per-column scan results (data + validity segments)
/// @param col_types   DuckDB logical types for each column (same order as col_scans)
/// @param stream      CUDA stream for async operations
/// @param mr          Device memory resource for GPU allocations
/// @param preloaded_block_offsets  Optional: when non-null, skip the bulk H2D and
///                    treat preloaded_base as a device buffer already containing
///                    every block referenced by col_scans. Lookup is by block_id
///                    using this offset map. Used by the preloaded-scan-region
///                    experiment.
/// @param preloaded_base  Optional device pointer corresponding to offset 0 in
///                    preloaded_block_offsets. Required iff preloaded_block_offsets
///                    is non-null.
/// @return Owning cudf::table with all columns decoded in GPU memory
std::unique_ptr<cudf::table> gpu_decode_table(
  std::vector<sirius::op::scan::column_scan_result>& col_scans,
  const std::vector<duckdb::LogicalType>& col_types,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  const std::unordered_map<int64_t, std::size_t>* preloaded_block_offsets = nullptr,
  const std::uint8_t* preloaded_base                                      = nullptr);

}  // namespace sirius::cuda::scan
