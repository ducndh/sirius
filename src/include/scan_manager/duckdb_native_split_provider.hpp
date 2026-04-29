/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "op/scan/duckdb_native_metadata.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"
#include "op/sirius_physical_operator.hpp"
#include "scan_manager/split_provider.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::exec {
class thread_pool;
}  // namespace sirius::exec

namespace sirius::scan_manager {

class split_connector;

//===----------------------------------------------------------------------===//
// duckdb_native_split_provider
//
// Producer that walks DuckDB-native segment trees off the GPU pipeline's
// critical path and pushes per-row-group-batch splits into a split_connector.
//
// Architectural shape (mirrors parquet_split_provider in PR #731):
//   - Construction: takes a populated duckdb_native_scan_info (DataTable,
//     projected cols/types, row groups, row_group_starts, context, batch size).
//   - start(pool, connector): dispatches one or more metadata-walk tasks onto
//     `pool`. Each task processes a chunk of row groups via
//     `walk_duckdb_native_metadata`, wraps the result as a split, and pushes
//     it into `connector`. Closes the connector exactly once when all
//     dispatched work has completed (success or failure path).
//
// Consumer side (gpu duckdb_native scan operator): receives splits via
// `split_connector::get_next_split()`. Each split carries a
// `partitioned_duckdb_native_metadata` payload — the same struct PR E's
// `duckdb_native_scan_global_state` consumes today, just delivered via the
// split queue instead of computed inline.
//
//===---------------------------------------------------------------------===//
// WIP STATUS (PR H1, 2026-04-29)
//===---------------------------------------------------------------------===//
//
// What's in this skeleton:
//   ✅ Bind-data shape (`duckdb_native_scan_info`)
//   ✅ Provider class with start() that wraps `walk_duckdb_native_metadata`
//   ✅ Sequential single-task body (one walk → one split). Tested in
//      test_duckdb_native_split_provider.cpp via a synthetic in-memory table.
//
// What's still TODO (intentionally deferred for early review):
//   ❌ Row-group batching to respect `approximate_batch_size`. Today the
//      provider emits ONE split covering all row groups. Real impl walks
//      per-batch like parquet's `next_task_input`.
//   ❌ Parallel dispatch onto `pool`. Today the provider runs the walk
//      synchronously inside start() and returns a ready future.
//   ❌ Removal of the existing task_creator dispatch path. Today
//      `task_creator.cpp` still constructs `duckdb_native_scan_task` directly
//      from `seq_scan` operator data; integration with scan_manager
//      (`prepare_for_query` + a new `sirius_gpu_duckdb_native_scan_operator`
//      consuming connector splits) is a follow-up.
//   ❌ Scan_manager dispatch: `prepare_for_query` filters by
//      GPU_PARQUET_SCAN today. Adding GPU_DUCKDB_NATIVE_SCAN handling +
//      `create_provider_for(sirius_gpu_duckdb_native_scan_operator*)`
//      overload (§4b stopgap) is part of the integration follow-up.
//
//===----------------------------------------------------------------------===//
class duckdb_native_split_provider : public split_provider {
 public:
  /// One emitted split: the metadata snapshot for a row-group batch. Wrapped
  /// in `op::operator_data` so it flows through the existing connector
  /// machinery. The consumer (gpu duckdb-native scan operator) downcasts.
  struct split_payload : public op::operator_data {
    explicit split_payload(op::scan::partitioned_duckdb_native_metadata m) : metadata(std::move(m))
    {
    }
    op::scan::partitioned_duckdb_native_metadata metadata;
  };

  explicit duckdb_native_split_provider(op::scan::duckdb_native_scan_info info);

  ~duckdb_native_split_provider() override;

  duckdb_native_split_provider(const duckdb_native_split_provider&)            = delete;
  duckdb_native_split_provider& operator=(const duckdb_native_split_provider&) = delete;
  duckdb_native_split_provider(duckdb_native_split_provider&&)                 = delete;
  duckdb_native_split_provider& operator=(duckdb_native_split_provider&&)      = delete;

  std::future<void> start(exec::thread_pool& pool, split_connector& connector) override;

 private:
  /// Walk one row-group batch and push its split into `connector`. Throws on
  /// internal errors; caller catches and ensures `connector.close()` runs.
  void run_batch(std::vector<duckdb::RowGroup*> const& batch_row_groups,
                 std::vector<duckdb::idx_t> const& batch_row_group_starts,
                 split_connector& connector);

  op::scan::duckdb_native_scan_info _info;
};

}  // namespace sirius::scan_manager
