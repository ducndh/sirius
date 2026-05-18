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

#include "scan_manager/sirius_scan_manager.hpp"

#include "exec/thread_pool.hpp"
#include "io/prefetching_cache.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/scan_info.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/sirius_gpu_duckdb_native_scan_operator.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/cached_split_provider.hpp"
#include "scan_manager/duckdb_native_split_provider.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(
  scan_manager_config config, cucascade::memory::fixed_size_host_memory_resource* host_mr)
  : _config(std::move(config)),
    _thread_pool(_config.thread_pool.num_threads,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads))
{
  if (_config.use_sirius_datasource) {
    if (host_mr == nullptr) {
      throw std::runtime_error(
        "[sirius_scan_manager] use_sirius_datasource is true but no host "
        "fixed_size_host_memory_resource was provided");
    }
    _io_ctx = std::make_shared<sirius::io::uring_ioctx>(
      _config.uring_n_reactors, _config.uring_ring_entries, *host_mr);
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource enabled (uring_ioctx n_reactors={} ring_entries={})",
      _config.uring_n_reactors,
      _config.uring_ring_entries);

    if (_config.enable_prefetch_cache) {
      // Slab size = CHUNKS_PER_SLAB blocks at the resource's block size.
      // Round the byte budget up so the user gets at least what they asked for.
      auto const slab_bytes = host_mr->get_block_size() *
                              static_cast<std::size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB);
      auto const max_slabs =
        static_cast<uint32_t>((_config.prefetch_buffer_pool_bytes + slab_bytes - 1) / slab_bytes);
      _buffer_pool = std::make_unique<sirius::io::buffer_pool>(*host_mr, max_slabs);
      _io_ctx->initialize_cache(*_buffer_pool, _config.prefetch_inflight_budget_chunks);
      SIRIUS_LOG_DEBUG(
        "[sirius_scan_manager] prefetch cache enabled (slabs={} budget_bytes={} "
        "inflight_chunks={})",
        max_slabs,
        max_slabs * slab_bytes,
        _config.prefetch_inflight_budget_chunks);
    }
  } else {
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — falling back to "
      "cudf::io::datasource::create");
  }
}

sirius_scan_manager::~sirius_scan_manager()
{
  if (_io_ctx && _io_ctx->cache() != nullptr) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }
  stop();
}

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  // Advance the cache age so the evictor can score this query's inserts
  // against entries left over from prior queries.
  if (_io_ctx && _io_ctx->cache()) { _io_ctx->cache()->refresh_cache(); }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={}",
                   query.get_pipelines().size());

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }

    if (source->type == ::sirius::op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN) {
      auto* op = &source->Cast<op::scan::sirius_gpu_parquet_scan_operator>();
      if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

      auto provider = create_provider_for(op);
      if (!provider) { continue; }
      op->set_split_connector(std::make_unique<split_connector>());
      _providers_by_op.emplace(op, std::move(provider));
      _scan_op_order.push_back(op);

      SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered parquet op_id={}",
                       op->get_operator_id());
    } else if (source->type ==
               ::sirius::op::SiriusPhysicalOperatorType::GPU_DUCKDB_NATIVE_SCAN) {
      // Interim dispatch — replace with polymorphic make_provider when Kevin's
      // PR-F lands. See scan_manager_hpp for the parallel map declarations.
      auto* op = &source->Cast<op::scan::sirius_gpu_duckdb_native_scan_operator>();
      if (_duckdb_native_providers_by_op.find(op) != _duckdb_native_providers_by_op.end()) {
        continue;
      }
      auto provider = create_provider_for(op);
      if (!provider) { continue; }
      op->set_split_connector(std::make_unique<split_connector>());
      _duckdb_native_providers_by_op.emplace(op, std::move(provider));
      _duckdb_native_scan_op_order.push_back(op);

      SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered duckdb-native op_id={}",
                       op->get_operator_id());
    }
  }

  if (_scan_op_order.empty() && _duckdb_native_scan_op_order.empty()) { return; }

  start_metadata_processing();
}

std::unique_ptr<split_provider> sirius_scan_manager::create_provider_for(
  op::scan::sirius_gpu_duckdb_native_scan_operator* op)
{
  auto info = op->take_scan_info();
  if (!info) { return nullptr; }

  // Pinned-entry lookup. Match by (db_path, table_name): pinned entry's file_paths
  // must be exactly [db_path] AND its name must equal table_name. Only GPU tier is
  // wired here; HOST tier falls through to the regular decode path.
  if (!info->db_path.empty() && !info->table_name.empty()) {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (entry.tier != cucascade::memory::Tier::GPU) { continue; }
      if (entry.file_paths.size() != 1) { continue; }
      if (entry.file_paths[0] != info->db_path) { continue; }
      if (pinned_name != info->table_name) { continue; }
      if (entry.memory_space == nullptr) { continue; }

      // Resolve shared columns in projected_cols order. Names were captured by the
      // converter into scan_info->projected_names so we don't need to re-query the
      // catalog here. Cache miss on rowid columns or any name not in the pin.
      std::vector<std::shared_ptr<cudf::column>> resolved;
      resolved.reserve(info->projected_cols.size());
      bool all_resolved = info->projected_names.size() == info->projected_cols.size();
      for (std::size_t i = 0; all_resolved && i < info->projected_cols.size(); ++i) {
        if (info->projected_cols[i].is_rowid) {
          all_resolved = false;
          break;
        }
        auto const& col_name = info->projected_names[i];
        auto col_it          = entry.data_batches_by_column.find(col_name);
        if (col_it == entry.data_batches_by_column.end() || col_it->second.empty()) {
          all_resolved = false;
          break;
        }
        // Single-chunk only (which is what .duckdb pin emits today). Multi-chunk would
        // need concatenation here.
        if (col_it->second.size() != 1) {
          all_resolved = false;
          break;
        }
        resolved.push_back(col_it->second.front());
      }
      if (!all_resolved) { continue; }

      SIRIUS_LOG_INFO(
        "[sirius_scan_manager::create_provider_for] duckdb-native cache hit: pinned='{}' "
        "db_path='{}'",
        pinned_name,
        info->db_path);
      auto info_shared =
        std::shared_ptr<op::scan::duckdb_native_scan_info const>(std::move(info));
      return std::make_unique<duckdb_native_cached_split_provider>(
        std::move(info_shared), std::move(resolved), *entry.memory_space);
    }
  }

  return std::make_unique<duckdb_native_split_provider>(std::move(*info), _io_ctx);
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
