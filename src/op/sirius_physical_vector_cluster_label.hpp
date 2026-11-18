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

#include "op/sirius_physical_operator.hpp"

#include <cudf/column/column.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>

namespace sirius {
namespace op {

/**
 * @brief Adds each row's cluster to the batches of a vector join's scanned build side.
 *
 * A clustered join reads its corpus as runs of equal cluster ids, which otherwise needs a table
 * written in cluster order beforehand, with the id as a column. This labels the rows as the scan
 * delivers them instead -- the nearest centroid of the join's clustering, as an INT32 column at
 * @c label_index -- so an ORDER BY on that column above it is all that stands between an
 * unordered table and the layout the join reads. The other columns pass through as views of the
 * input batch, unless the batch is to leave sorted by label (sort_by_label()).
 */
class sirius_physical_vector_cluster_label : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_CLUSTER_LABEL;

  /// @p centroids is borrowed from the session's index cache, which keeps a clustering for the
  /// life of the query. @p metric must be the one the clustering was fitted under.
  sirius_physical_vector_cluster_label(duckdb::vector<sirius::logical_type> types,
                                       std::size_t vector_column,
                                       std::size_t label_index,
                                       const cudf::column* centroids,
                                       std::int64_t dim,
                                       cuvs::distance::DistanceType metric,
                                       std::size_t estimated_cardinality);

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

  /// Emit each batch sorted by label, as a copy, instead of as views of the input in its own
  /// order: what a partition on cluster-id ranges directly above this operator reads.
  void sort_by_label(bool on) { _sort_by_label = on; }

  /// Emit the vector column in half precision: each value's IEEE half bits in an INT16 child, which
  /// the join's chunk source widens as it stages a chunk. The tiers then hold and move half the
  /// bytes. Only for values that all survive the rounding (the join would otherwise score rounded
  /// rows): a batch with one that does not fails the query.
  void store_half(bool on) { _store_half = on; }

  /// The labeler holds a half-precision copy of a batch's vectors and the candidates of its rows;
  /// a sorted batch is a second copy of it while the input is still held.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override
  {
    return _sort_by_label ? memory::saturating_mul(stats.bytes, 2) : stats.bytes;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

 private:
  std::size_t _vector_column;
  std::size_t _label_index;
  const cudf::column* _centroids;
  std::int64_t _dim;
  cuvs::distance::DistanceType _metric;
  bool _sort_by_label{false};
  bool _store_half{false};
};

}  // namespace op
}  // namespace sirius
