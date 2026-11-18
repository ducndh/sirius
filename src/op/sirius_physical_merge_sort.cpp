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

#include "op/sirius_physical_merge_sort.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <algorithm>

#include <cudf/copying.hpp>

namespace sirius {
namespace op {

sirius_physical_merge_sort::sirius_physical_merge_sort(sirius_physical_order* order_by)
  : sirius_physical_merge_sort(
      order_by->types,                // copied by value
      copy_orders(order_by->orders),  // deep copy (contains unique_ptr<Expression>)
      order_by->projections,          // copied by value
      order_by->estimated_cardinality,
      order_by->is_index_sort)
{
  child_op = order_by;
}

sirius_physical_merge_sort::sirius_physical_merge_sort(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality,
  bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_SORT, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  // Drain all batches from one partition per call (one merge task per partition).
  // Follows the same pattern as sirius_physical_grouped_aggregate_merge.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  while (_current_partition_index < repo->num_partitions()) {
    std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
    while (true) {
      auto batch = repo->pop_next_data_batch(_current_partition_index);
      if (!batch) { break; }
      all_batches.push_back(std::move(batch));
    }
    _current_partition_index++;
    SIRIUS_LOG_DEBUG("merge_sort: drained {} batches for partition {}",
                     all_batches.size(),
                     _current_partition_index - 1);
    if (all_batches.empty()) { continue; }
    return std::make_unique<pipelineable_operator_data>(all_batches, _current_partition_index - 1);
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(const operator_data& input_data,
                                                                   ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_merge_sort::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Find memory space
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
  }

  if (input_batches.empty() || !space) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // Helper lambda to apply final projection to a batch (removes sort-key-only columns)
  auto apply_final_projection =
    [this, stream, space](
      const cucascade::read_only_data_batch& ro) -> std::shared_ptr<cucascade::data_batch> {
    auto table_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> projected_cols;
    for (auto idx : _final_projections) {
      projected_cols.push_back(table_view.column(static_cast<cudf::size_type>(idx)));
    }
    auto projected_table = std::make_unique<cudf::table>(
      cudf::table_view(projected_cols), stream, space->get_default_allocator());
    return sirius::make_data_batch(std::move(projected_table), *space, stream, batch_telemetry());
  };

  // Single batch: no merge needed
  if (input_batches.size() == 1) {
    std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
    if (_final_projections.empty()) {
      // Forward the owned input batch (idle at park); no read lock carried.
      outputs.push_back(input.get_data_batches()[0]);
    } else {
      outputs.push_back(apply_final_projection(input_batches[0]));
    }
    if (_max_output_batch_bytes > 0) {
      return std::make_unique<pipelineable_operator_data>(outputs);
    }
    return std::make_unique<pipelineable_operator_data>(outputs, input.get_position());
  }

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Merge sort only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  auto merged_batch = gpu_merge_impl::merge_order_by(
    input_batches, order_key_idx, column_order, null_precedence, stream, *space, batch_telemetry());

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (merged_batch) {
    auto const merged_bytes = merged_batch->to_read_only().get_data()->get_size_in_bytes();
    if (_final_projections.empty() && _max_output_batch_bytes > 0 &&
        merged_bytes > _max_output_batch_bytes) {
      // Cut by rows into even pieces of about the asked size; each is a copy, so the merged
      // batch is released with this task.
      auto const ro     = merged_batch->to_read_only();
      auto const view   = sirius::get_cudf_table_view(ro);
      auto const rows   = static_cast<std::size_t>(view.num_rows());
      auto const pieces = std::min<std::size_t>(
        std::max<std::size_t>(1, rows),
        (merged_bytes + _max_output_batch_bytes - 1) / _max_output_batch_bytes);
      std::vector<cudf::size_type> cuts;
      for (std::size_t i = 1; i < pieces; ++i) {
        cuts.push_back(static_cast<cudf::size_type>(i * rows / pieces));
      }
      for (auto const& piece : cudf::split(view, cuts, stream)) {
        if (piece.num_rows() == 0) { continue; }
        auto table = std::make_unique<cudf::table>(piece, stream, space->get_default_allocator());
        outputs.push_back(sirius::make_data_batch(std::move(table), *space, stream, batch_telemetry()));
      }
    } else if (_final_projections.empty()) {
      outputs.push_back(std::move(merged_batch));
    } else {
      auto ro = merged_batch->to_read_only();
      outputs.push_back(apply_final_projection(ro));
    }
  }
  // A position stands for exactly one batch; sized batches go out unpositioned, all of them.
  if (_max_output_batch_bytes > 0) { return std::make_unique<pipelineable_operator_data>(outputs); }
  return std::make_unique<pipelineable_operator_data>(outputs, input.get_position());
}

}  // namespace op
}  // namespace sirius
