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

#include "duckdb/planner/bound_query_node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>
#include <mutex>
#include <vector>

namespace sirius {
namespace op {

// Helper to deep copy BoundOrderByNode vector (contains unique_ptr<Expression>)
inline duckdb::vector<duckdb::BoundOrderByNode> copy_orders(
  const duckdb::vector<duckdb::BoundOrderByNode>& src)
{
  duckdb::vector<duckdb::BoundOrderByNode> result;
  result.reserve(src.size());
  for (const auto& order : src) {
    result.push_back(order.Copy());
  }
  return result;
}

class sirius_physical_order : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    // Local sorting does not establish a sequence across independently completed batches.
    return sirius::OrderPreservationType::NO_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ORDER_BY;

  sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                        duckdb::vector<duckdb::BoundOrderByNode> orders,
                        duckdb::vector<std::size_t> projections_p,
                        std::size_t estimated_cardinality,
                        bool is_index_sort_p = false);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// The key rows execute() kept, a few of every batch it sorted, sorted together (key columns
  /// only, in @c orders' order). Null when none were kept. A range sort cuts its partition
  /// boundaries from these: every batch has passed through here on the device, whereas the
  /// batches a later operator can afford to bring back are a handful, and where the input's order
  /// follows the key at all a handful of batches does not look like the input.
  [[nodiscard]] std::unique_ptr<cudf::table> sorted_key_sample(
    const std::vector<cudf::order>& column_order,
    const std::vector<cudf::null_order>& null_precedence,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

 private:
  mutable std::mutex _key_sample_mutex;
  std::vector<std::unique_ptr<cudf::table>> _key_samples;
};

}  // namespace op
}  // namespace sirius
