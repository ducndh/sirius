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

#include "op/sirius_physical_order.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/order/gpu_order_impl.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>

namespace sirius {
namespace op {

sirius_physical_order::sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                                             duckdb::vector<std::size_t> projections_p,
                                             std::size_t estimated_cardinality,
                                             bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_order::execute(const operator_data& input_data,
                                                              ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_order::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Order by only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  std::vector<int> proj_idx(projections.begin(), projections.end());

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }

    auto sorted_batch = gpu_order_impl::local_order_by(batch,
                                                       order_key_idx,
                                                       column_order,
                                                       null_precedence,
                                                       proj_idx,
                                                       stream,
                                                       *space,
                                                       batch_telemetry());
    if (sorted_batch) {
      // A few of the batch's key rows, at even steps through it, for the boundaries of the range
      // sort above (see sorted_key_sample()). Only where the keys are among the output columns.
      std::vector<cudf::size_type> key_at;
      for (auto const key : order_key_idx) {
        auto const it = std::find(proj_idx.begin(), proj_idx.end(), key);
        if (it == proj_idx.end()) { break; }
        key_at.push_back(static_cast<cudf::size_type>(std::distance(proj_idx.begin(), it)));
      }
      auto const sorted_ro = sorted_batch->to_read_only();
      auto const rows      = sirius::get_cudf_table_view(sorted_ro).num_rows();
      if (key_at.size() == order_key_idx.size() && rows > 0) {
        constexpr cudf::size_type kKeptPerBatch = 512;
        auto const step  = std::max<cudf::size_type>(1, rows / kKeptPerBatch);
        auto const count = (rows + step - 1) / step;
        rmm::cuda_stream_view const sv{stream.get()};
        auto const mr = space->get_default_allocator();
        cudf::numeric_scalar<cudf::size_type> const first{0, true, sv, mr};
        cudf::numeric_scalar<cudf::size_type> const by{step, true, sv, mr};
        auto const picks = cudf::sequence(count, first, by, sv, mr);
        auto kept        = cudf::gather(sirius::get_cudf_table_view(sorted_ro).select(key_at),
                                        picks->view(),
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        sv,
                                        mr);
        std::lock_guard<std::mutex> const lock(_key_sample_mutex);
        _key_samples.push_back(std::move(kept));
      }
      output_batches.push_back(std::move(sorted_batch));
    }
  }

  return std::make_unique<pipelineable_operator_data>(output_batches);
}

std::unique_ptr<cudf::table> sirius_physical_order::sorted_key_sample(
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  std::lock_guard<std::mutex> const lock(_key_sample_mutex);
  if (_key_samples.empty()) { return nullptr; }
  std::vector<cudf::table_view> views;
  views.reserve(_key_samples.size());
  for (auto const& sample : _key_samples) {
    views.push_back(sample->view());
  }
  rmm::cuda_stream_view const sv{stream.get()};
  auto const all = cudf::concatenate(views, sv, mr);
  return cudf::sort(all->view(), column_order, null_precedence, sv, mr);
}

}  // namespace op
}  // namespace sirius
