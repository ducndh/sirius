/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "helper/logical_type.hpp"
#include "op/scan/duckdb_native_metadata.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>

#include <cstddef>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// duckdb_native_scan_info
//
// Bind-data parked on the gpu duckdb-native scan operator at plan time and
// consumed by `sirius_scan_manager::prepare_for_query` to construct a
// `duckdb_native_split_provider`.
//
// Mirrors `parquet_scan_info` (PR #731) for the DuckDB-native storage path.
// The fields here drive `walk_duckdb_native_metadata` exactly: storage table,
// projected columns + types, row groups + their absolute starts, and a
// reference to the live client context for BufferManager access during the
// segment-tree walk.
//
// WIP scope (PR H1): this struct exists; the consumer-side wiring that
// converts task_creator's `seq_scan` dispatch into a scan_manager registration
// is still pending. See `duckdb_native_split_provider.hpp` for the producer
// half.
//===----------------------------------------------------------------------===//
struct duckdb_native_scan_info {
  /// Live storage table — must outlive the provider's start() future.
  duckdb::DataTable* storage = nullptr;

  /// Projected columns, parallel to `projected_types`. May contain rowid
  /// sentinels (handled by the walker; sentinel columns produce empty
  /// `column_scan_result`s and are decoded as synthesized rowids by the
  /// scan-task consumer).
  std::vector<projected_column> projected_cols;

  /// Logical types of `projected_cols`, parallel.
  std::vector<sirius::logical_type> projected_types;

  /// Row groups to scan; non-empty. Walker consumes via segment tree.
  std::vector<duckdb::RowGroup*> row_groups;

  /// Absolute row index of `row_groups[i]`'s first row within the table.
  /// Parallel to `row_groups`. Drives rowid synthesis.
  std::vector<duckdb::idx_t> row_group_starts;

  /// Client context for the running query; used by the walker for
  /// BufferManager pinning. The provider does not own this — it lives on the
  /// query's lifecycle.
  duckdb::ClientContext* context = nullptr;

  /// Target uncompressed bytes per row-group partition emitted as a split.
  /// Provider chunks `row_groups` into batches respecting this budget.
  std::size_t approximate_batch_size = 0;
};

}  // namespace sirius::op::scan
