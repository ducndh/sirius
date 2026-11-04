/*
 * Copyright 2026, Sirius Contributors.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// The probe side's routing edges, on the device: one edge per (row, cluster it was routed to), from
// a row-major [n_rows x n_probes] matrix of cluster ids, nearest cluster first. A join's cost here
// is n_rows x n_probes whatever the data, so only the per-cluster boundaries and each row's nearest
// cluster cross to the host.
namespace sirius::vss {

/// The routing checked and keyed for sort_routed_edges.
struct keyed_edges {
  std::int64_t n_rows{0};
  std::int64_t n_probes{0};
  std::int32_t n_clusters{0};
  std::optional<rmm::device_uvector<std::uint64_t>> keys;
  std::vector<std::int32_t> nearest_cluster;  ///< [n_rows]
  /// The first row routed outside the clusters or reaching fewer than k corpus rows, else -1, and
  /// that row's clusters.
  std::int64_t refused_row{-1};
  std::vector<std::int32_t> refused_clusters;
};

/// The edges by cluster; within a cluster, the rows for which it is the nearest first, then rows in
/// ascending order.
struct sorted_edges {
  rmm::device_uvector<std::uint64_t> rows;  ///< Probe row per edge.
  std::vector<std::int64_t> edge_begin;     ///< [n_clusters + 1]
  std::vector<std::int64_t> nearest_edges;  ///< [n_clusters]
};

/**
 * @brief Check the routing and build the sort's keys.
 *
 * @param clusters    [n_rows x n_probes] cluster ids; a row's are distinct, nearest first.
 * @param cluster_rows Corpus rows per cluster, which also gives the number of clusters.
 * @param k            Fewest corpus rows a row's clusters must hold; at least 1 is required.
 */
[[nodiscard]] keyed_edges key_routed_edges(rmm::device_uvector<std::int32_t> clusters,
                                           std::int64_t n_rows,
                                           std::int64_t n_probes,
                                           std::vector<std::int64_t> const& cluster_rows,
                                           std::int64_t k,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/// A stable sort on the keys' upper half, so rows keep their order within a key. Holds 16 B an
/// edge while it sorts, 8 B an edge on return.
[[nodiscard]] sorted_edges sort_routed_edges(keyed_edges edges,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
