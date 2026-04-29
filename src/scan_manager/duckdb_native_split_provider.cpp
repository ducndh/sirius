/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "scan_manager/duckdb_native_split_provider.hpp"

#include "exec/thread_pool.hpp"
#include "log/logging.hpp"
#include "scan_manager/split_connector.hpp"

#include <exception>
#include <future>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

duckdb_native_split_provider::duckdb_native_split_provider(op::scan::duckdb_native_scan_info info)
  : _info(std::move(info))
{
  if (_info.storage == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_split_provider] scan_info.storage must be non-null");
  }
  if (_info.context == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_split_provider] scan_info.context must be non-null");
  }
  if (_info.row_groups.empty()) {
    throw std::invalid_argument(
      "[duckdb_native_split_provider] scan_info.row_groups must be non-empty");
  }
  if (_info.row_groups.size() != _info.row_group_starts.size()) {
    throw std::invalid_argument(
      "[duckdb_native_split_provider] row_groups and row_group_starts must be parallel");
  }
  if (_info.projected_cols.size() != _info.projected_types.size()) {
    throw std::invalid_argument(
      "[duckdb_native_split_provider] projected_cols and projected_types must be parallel");
  }
}

duckdb_native_split_provider::~duckdb_native_split_provider() = default;

std::future<void> duckdb_native_split_provider::start(exec::thread_pool& /*pool*/,
                                                      split_connector& connector)
{
  // WIP: synchronous single-batch walk. Future iteration dispatches per-batch
  // walks onto `pool` and chains a future that completes when the last batch
  // finishes. For now, return a ready future so the scan_manager driver loop
  // can advance immediately.
  std::promise<void> promise;
  auto future = promise.get_future();

  try {
    run_batch(_info.row_groups, _info.row_group_starts, connector);
    connector.close();
    promise.set_value();
  } catch (...) {
    SIRIUS_LOG_ERROR("[duckdb_native_split_provider] start failed; closing connector");
    connector.close();
    promise.set_exception(std::current_exception());
  }

  return future;
}

void duckdb_native_split_provider::run_batch(
  std::vector<duckdb::RowGroup*> const& batch_row_groups,
  std::vector<duckdb::idx_t> const& batch_row_group_starts,
  split_connector& connector)
{
  auto metadata = op::scan::walk_duckdb_native_metadata(*_info.storage,
                                                        _info.projected_cols,
                                                        _info.projected_types,
                                                        batch_row_groups,
                                                        batch_row_group_starts,
                                                        *_info.context);

  // Even when viable=false we still emit the split so the consumer can record
  // the failure reason and fall back to duckdb_scan_task. The consumer is
  // responsible for the fallback decision; the provider's contract is just
  // "deliver a metadata snapshot."
  connector.push_split(std::make_unique<split_payload>(std::move(metadata)));
}

}  // namespace sirius::scan_manager
