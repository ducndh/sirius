/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/table/row_group.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

// Per-(table, column) preloaded GPU bytes.
// ENCODED mode populates encoded_pool + block_offsets only.
// DECODED mode populates decoded_columns only.
// One preloaded_table is created for each table named in a gpu_preload_region call.
struct preloaded_table {
  // ENCODED: contiguous device buffer with all unique blocks for the listed columns.
  // block_offsets maps DuckDB block_id -> byte offset within encoded_pool.
  // The base pointer (encoded_pool.data()) is added to that offset to get the
  // device pointer for any block.
  rmm::device_buffer encoded_pool;
  std::unordered_map<int64_t, std::size_t> block_offsets;

  // DECODED: keyed by RowGroup* (stable for the table's lifetime in this DB session)
  // then by storage column ordinal (StorageIndex::GetPrimaryIndex()).
  // Each entry owns a fully decoded cudf::column for that one (rg, col).
  std::unordered_map<duckdb::RowGroup*,
                     std::unordered_map<duckdb::idx_t, std::unique_ptr<cudf::column>>>
    decoded_columns;
};

// Per-query staging area, owned by SiriusContext. Lifetime bracketed explicitly
// by gpu_preload_region(...) and gpu_release_region(). Not a cache: no eviction,
// no admission policy, no LRU.
struct scan_region {
  enum class mode { ENCODED, DECODED };
  mode m = mode::ENCODED;

  // Keyed by the DataTable pointer, which is stable while the table exists in
  // this connection's catalog. The scan task fetches its DataTable* from
  // gpu_native_scan_global_state::storage() and looks it up here.
  std::unordered_map<duckdb::DataTable*, preloaded_table> tables;
};

}  // namespace sirius::op::scan
