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

#include "vss/brute_force_search.hpp"

#include <cstdlib>
#include <string_view>

#include <cudf/column/column_factories.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/neighbors/brute_force.hpp>

namespace sirius::vss {

knn_result brute_force_knn(raft::device_resources const& res,
                           dataset_matrix_view dataset,
                           dataset_matrix_view queries,
                           int64_t k,
                           cuvs::distance::DistanceType metric,
                           rmm::device_async_resource_ref mr)
{
  namespace bf = cuvs::neighbors::brute_force;

#ifdef SIRIUS_ENABLE_FAISS_KERNEL
  // Read once: this is on the per-corpus-chunk path, so a getenv per call would show up.
  static bool const use_faiss = [] {
    const char* k = std::getenv("SIRIUS_VSS_KERNEL");
    return k != nullptr && std::string_view{k} == "faiss";
  }();
  if (use_faiss) { return brute_force_knn_faiss(res, dataset, queries, k, metric, mr); }
#endif

  auto const n_rows    = dataset.extent(0);
  auto const n_queries = queries.extent(0);

  CUDF_EXPECTS(dataset.extent(1) == queries.extent(1),
               "VSS dataset and query dimensionality must match");
  CUDF_EXPECTS(k >= 1 && k <= n_rows, "VSS k must satisfy 1 <= k <= n_rows");

  // Everything runs on res's stream so the search, the output allocations, and
  // the caller's downstream work all order on that single stream. res is
  // caller-owned and reused across chunks, so its handle setup is paid once.
  auto const stream = raft::resource::get_cuda_stream(res);

  // Build the brute-force index. With the non-owning dataset view this stores a
  // reference to Sirius-owned memory and precomputes norms.
  bf::index_params index_params;
  index_params.metric = metric;
  auto index          = bf::build(res, index_params, dataset);

  // Allocate flattened [n_queries * k] outputs through the caller's resource
  // (mr) so they are reserved against the owning memory space.
  // cuVS dispatches k <= 64 (row-major, any L2 variant) away from its tiled cuBLAS path and
  // into fusedL2Knn, a hand-written SIMT kernel whose policy is hard-wired to a 2x8 register
  // tile with scalar loads -- the vectorized alternative is dead code behind a
  // std::conditional<true, ...>. On d=128 that path is slower at every probe count measured
  // from M=128 to M=131072 (1.22x to 1.72x), and our own operator is 26% faster asking for
  // k=65 than k=64. So ask for one neighbour past the boundary and trim, which costs a gather
  // over n_queries*k elements and buys the better kernel.
  //
  // The trade is memory: the intermediate is n_queries*65 rather than n_queries*k. Set
  // SIRIUS_VSS_CUVS_OVERSEARCH=0 to disable, which is also how the two are A/B'd.
  static bool const oversearch_enabled = [] {
    const char* v = std::getenv("SIRIUS_VSS_CUVS_OVERSEARCH");
    return v == nullptr || std::string_view{v} != "0";
  }();
  constexpr int64_t kFusedDispatchMaxK = 64;
  int64_t const k_search =
    (oversearch_enabled && k <= kFusedDispatchMaxK && n_rows > kFusedDispatchMaxK)
      ? kFusedDispatchMaxK + 1
      : k;

  auto const out_size = static_cast<cudf::size_type>(n_queries * k_search);
  auto neighbors_col  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
    neighbors_col->mutable_view().data<int64_t>(), n_queries, k_search);
  auto distances_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    distances_col->mutable_view().data<float>(), n_queries, k_search);

  bf::search_params search_params;
  bf::search(res, search_params, index, queries, neighbors_view, distances_view);

  // Keep the first k of each row: results are nearest-first, so element j of row i lives at
  // i*k_search + j and the trim is a gather rather than a contiguous slice.
  if (k_search != k) {
    auto const kept = static_cast<cudf::size_type>(n_queries * k);
    cudf::numeric_scalar<int32_t> const zero(0, true, stream);
    cudf::numeric_scalar<int32_t> const one(1, true, stream);
    auto const positions = cudf::sequence(kept, zero, one, stream, mr);

    cudf::numeric_scalar<int32_t> const k_out(static_cast<int32_t>(k), true, stream);
    cudf::numeric_scalar<int32_t> const k_in(static_cast<int32_t>(k_search), true, stream);
    auto const row = cudf::binary_operation(
      positions->view(), k_out, cudf::binary_operator::DIV, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    auto const col = cudf::binary_operation(
      positions->view(), k_out, cudf::binary_operator::MOD, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    auto const base = cudf::binary_operation(
      row->view(), k_in, cudf::binary_operator::MUL, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    auto const gather_map = cudf::binary_operation(
      base->view(), col->view(), cudf::binary_operator::ADD, cudf::data_type{cudf::type_id::INT32}, stream, mr);

    auto trimmed = cudf::gather(cudf::table_view{{neighbors_col->view(), distances_col->view()}},
                                gather_map->view(),
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                mr);
    auto cols      = trimmed->release();
    neighbors_col  = std::move(cols[0]);
    distances_col  = std::move(cols[1]);
  }

  // The search runs async on res's stream.
  return knn_result{std::move(neighbors_col), std::move(distances_col), n_queries, k};

}

}  // namespace sirius::vss
