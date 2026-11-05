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

#include "cagra_index.hpp"

#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/extrema.h>

#include <cuda_fp16.h>
#include <cuvs/neighbors/cagra.hpp>

#include <cmath>
#include <limits>
#include <utility>
#include <variant>

namespace sirius::vss {

namespace {

constexpr float float16_limit = 65520.f;  // rounds to infinity from here

// One warp per row, each lane converting and summing every 32nd value.
template <typename T>
__global__ void convert_rows_kernel(float const* __restrict__ in,
                                    std::int64_t n,
                                    int d,
                                    T* __restrict__ out,
                                    std::int64_t stride,
                                    float* __restrict__ norm_sq)
{
  int const lane = threadIdx.x % 32;
  for (std::int64_t r = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / 32;
       r < n;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32) {
    float const* row = in + r * d;
    float s          = 0.f;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
      std::uint16_t* to = out == nullptr ? nullptr : out + r * stride;
      for (int c = lane; c < d; c += 32) {
        float const v = row[c];
        if (to != nullptr) { to[c] = __half_as_ushort(__float2half_rn(v)); }
        s = fabsf(v) < float16_limit ? fmaf(v, v, s) : INFINITY;
      }
    } else {
      T* to = out == nullptr ? nullptr : out + r * stride;
      for (int c = lane; c < d; c += 32) {
        float const v = row[c];
        if (to != nullptr) { to[c] = v; }
        s = fmaf(v, v, s);
      }
    }
    for (int step = 16; step > 0; step /= 2) {
      s += __shfl_xor_sync(0xffffffffu, s, step);
    }
    if (lane == 0) { norm_sq[r] = isfinite(s) ? s : INFINITY; }
  }
}

__global__ void partition_distances_kernel(float const* __restrict__ rows,
                                           std::int64_t n,
                                           int d,
                                           float const* __restrict__ centers,
                                           float const* __restrict__ offsets,
                                           int P,
                                           float* __restrict__ out)
{
  int const lane = threadIdx.x % 32;
  for (std::int64_t r = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / 32;
       r < n;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32) {
    float const* row = rows + r * d;
    for (int p = 0; p < P; ++p) {
      float const* center = centers + static_cast<std::int64_t>(p) * d;
      float s             = 0.f;
      for (int c = lane; c < d; c += 32) {
        float const t = row[c] - center[c];
        s             = fmaf(t, t, s);
      }
      for (int step = 16; step > 0; step /= 2) {
        s += __shfl_xor_sync(0xffffffffu, s, step);
      }
      if (lane == 0) { out[r * P + p] = s + offsets[p]; }
    }
  }
}

__global__ void widen_ids_kernel(std::uint32_t const* __restrict__ neighbors,
                                 std::int64_t n,
                                 std::int64_t n_rows,
                                 std::int64_t* __restrict__ ids)
{
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const id = static_cast<std::int64_t>(neighbors[i]);
    ids[i]        = id < n_rows ? id : -1;
  }
}

__global__ void sort_rows_kernel(float* __restrict__ distances,
                                 std::int64_t* __restrict__ ids,
                                 std::int64_t m,
                                 std::int64_t k)
{
  for (std::int64_t r = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    float* dist      = distances + r * k;
    std::int64_t* id = ids + r * k;
    for (std::int64_t i = 0; i < k; ++i) {
      if (id[i] < 0) { dist[i] = INFINITY; }
    }
    for (std::int64_t i = 1; i < k; ++i) {
      float const d_i      = dist[i];
      std::int64_t const v = id[i];
      std::int64_t j       = i;
      while (j > 0 && (dist[j - 1] > d_i || (dist[j - 1] == d_i && id[j - 1] > v))) {
        dist[j] = dist[j - 1];
        id[j]   = id[j - 1];
        --j;
      }
      dist[j] = d_i;
      id[j]   = v;
    }
    // A walk can reach a row twice; it counts once, and the row then has fewer than k pairs.
    std::int64_t kept = 0;
    for (std::int64_t i = 0; i < k; ++i) {
      if (id[i] >= 0 && (kept == 0 || id[kept - 1] != id[i])) {
        dist[kept] = dist[i];
        id[kept]   = id[i];
        ++kept;
      }
    }
    for (std::int64_t i = kept; i < k; ++i) {
      dist[i] = INFINITY;
      id[i]   = -1;
    }
  }
}

struct is_miss {
  __device__ bool operator()(std::int64_t id) const { return id < 0; }
};

template <typename T>
using padded_index = cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t>;

template <typename T>
padded_index<T> build_index(cuvs::distance::DistanceType metric,
                            cagra_build_settings const& s,
                            T const* rows,
                            std::int64_t n_rows,
                            std::int64_t dim,
                            std::size_t stride_bytes,
                            raft::device_resources const& res)
{
  namespace cagra = cuvs::neighbors::cagra;
  cagra::index_params params;
  params.metric                    = metric;
  params.graph_degree              = static_cast<std::size_t>(s.graph_degree);
  params.intermediate_graph_degree = static_cast<std::size_t>(s.intermediate_graph_degree);
  switch (s.builder) {
    case cagra_graph_builder::nn_descent: {
      cagra::graph_build_params::nn_descent_params nn(params.intermediate_graph_degree, metric);
      nn.max_iterations         = static_cast<std::size_t>(s.nn_descent_niter);
      params.graph_build_params = nn;
      break;
    }
    case cagra_graph_builder::ivf_pq: {
      auto pq = cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<std::int64_t>(n_rows, dim), metric);
      if (s.ivf_pq_n_lists > 0) {
        pq.build_params.n_lists = static_cast<std::uint32_t>(s.ivf_pq_n_lists);
      }
      if (s.ivf_pq_n_probes > 0) {
        pq.search_params.n_probes = static_cast<std::uint32_t>(s.ivf_pq_n_probes);
      }
      params.graph_build_params = pq;
      break;
    }
    case cagra_graph_builder::iterative_search:
      params.graph_build_params = cagra::graph_build_params::iterative_search_params{};
      break;
  }
  auto const stride = static_cast<std::int64_t>(stride_bytes / sizeof(T));
  auto view =
    raft::make_device_strided_matrix_view<T const, std::int64_t>(rows, n_rows, dim, stride);
  auto dataset = cuvs::neighbors::make_device_padded_dataset_view(res, view);
  return cagra::build(res, params, dataset);
}

class cagra_index_impl final : public cagra_index {
 public:
  template <typename T>
  cagra_index_impl(padded_index<T>&& index,
                   std::unique_ptr<rmm::device_buffer> rows,
                   std::unique_ptr<rmm::device_buffer> graph,
                   std::int64_t n_rows,
                   std::int64_t dim,
                   cagra_build_settings const& settings,
                   bool rows_unit)
    : _rows(std::move(rows)),
      _graph(std::move(graph)),
      _index(std::move(index)),
      _n_rows(n_rows),
      _dim(dim),
      _settings(settings),
      _rows_unit(rows_unit)
  {
  }

  [[nodiscard]] cagra_build_settings const& settings() const noexcept override { return _settings; }
  [[nodiscard]] std::int64_t n_rows() const noexcept override { return _n_rows; }
  [[nodiscard]] std::int64_t dim() const noexcept override { return _dim; }
  [[nodiscard]] bool rows_unit() const noexcept override { return _rows_unit; }

  void copy_graph_to_host(std::uint32_t* out, rmm::cuda_stream_view stream) const override
  {
    std::visit(
      [&](auto const& index) {
        auto const graph = index.graph();
        CUDF_CUDA_TRY(
          cudaMemcpyAsync(out,
                          graph.data_handle(),
                          static_cast<std::size_t>(graph.size()) * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()));
      },
      _index);
    stream.sync();
  }

  void search(void const* queries,
              std::int64_t m,
              std::int64_t k,
              cagra_search_settings const& s,
              std::uint32_t* neighbors,
              float* distances,
              rmm::cuda_stream_view stream) const override
  {
    namespace cagra = cuvs::neighbors::cagra;
    raft::device_resources res{stream};
    cagra::search_params params;
    params.itopk_size   = static_cast<std::size_t>(std::max(s.itopk, k));
    params.search_width = static_cast<std::size_t>(s.search_width);
    auto const out_ids =
      raft::make_device_matrix_view<std::uint32_t, std::int64_t, raft::row_major>(neighbors, m, k);
    auto const out_distances =
      raft::make_device_matrix_view<float, std::int64_t, raft::row_major>(distances, m, k);
    std::visit(
      [&](auto const& index) {
        using T      = typename std::decay_t<decltype(index)>::value_type;
        auto const q = raft::make_device_matrix_view<T const, std::int64_t, raft::row_major>(
          static_cast<T const*>(queries), m, _dim);
        cagra::search(res, params, index, q, out_ids, out_distances);
      },
      _index);
    raft::resource::sync_stream(res);
  }

 private:
  // Declared first: the index views these, and is destroyed before them.
  std::unique_ptr<rmm::device_buffer> _rows;
  std::unique_ptr<rmm::device_buffer> _graph;
  std::variant<padded_index<half>, padded_index<float>> _index;
  std::int64_t _n_rows;
  std::int64_t _dim;
  cagra_build_settings _settings;
  bool _rows_unit;
};

}  // namespace

void cagra_convert_rows(float const* in,
                        std::int64_t n,
                        std::int64_t dim,
                        cagra_row_type type,
                        void* out,
                        std::size_t out_stride_bytes,
                        row_norm_range& range,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  if (n == 0) { return; }
  rmm::device_uvector<float> norms(static_cast<std::size_t>(n), stream, mr);
  int constexpr block_threads = 256;
  int const warps_per_block   = block_threads / 32;
  auto const grid             = static_cast<int>(
    std::clamp<std::int64_t>((n + warps_per_block - 1) / warps_per_block, 1, 65535));
  if (type == cagra_row_type::float16) {
    convert_rows_kernel<std::uint16_t><<<grid, block_threads, 0, stream.get()>>>(
      in,
      n,
      static_cast<int>(dim),
      static_cast<std::uint16_t*>(out),
      static_cast<std::int64_t>(out_stride_bytes / sizeof(std::uint16_t)),
      norms.data());
  } else {
    convert_rows_kernel<float><<<grid, block_threads, 0, stream.get()>>>(
      in,
      n,
      static_cast<int>(dim),
      static_cast<float*>(out),
      static_cast<std::int64_t>(out_stride_bytes / sizeof(float)),
      norms.data());
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const extremes =
    thrust::minmax_element(rmm::exec_policy(stream), norms.begin(), norms.end());
  float const lo = norms.element(static_cast<std::size_t>(extremes.first - norms.begin()), stream);
  float const hi = norms.element(static_cast<std::size_t>(extremes.second - norms.begin()), stream);
  range.min_sq   = std::min(range.min_sq, lo);
  range.max_sq   = std::max(range.max_sq, hi);
}

std::unique_ptr<cagra_index> build_cagra_index(std::unique_ptr<rmm::device_buffer> rows,
                                               std::int64_t n_rows,
                                               std::int64_t dim,
                                               std::size_t stride_bytes,
                                               cagra_build_settings const& settings,
                                               bool rows_unit,
                                               rmm::cuda_stream_view stream)
{
  auto const metric = cuvs::distance::DistanceType::L2Expanded;
  raft::device_resources res{stream};
  if (settings.row_type == cagra_row_type::float16) {
    auto index = build_index<half>(
      metric, settings, static_cast<half const*>(rows->data()), n_rows, dim, stride_bytes, res);
    raft::resource::sync_stream(res);
    return std::make_unique<cagra_index_impl>(
      std::move(index), std::move(rows), nullptr, n_rows, dim, settings, rows_unit);
  }
  auto index = build_index<float>(
    metric, settings, static_cast<float const*>(rows->data()), n_rows, dim, stride_bytes, res);
  raft::resource::sync_stream(res);
  return std::make_unique<cagra_index_impl>(
    std::move(index), std::move(rows), nullptr, n_rows, dim, settings, rows_unit);
}

namespace {

template <typename T>
std::unique_ptr<cagra_index> index_over(std::unique_ptr<rmm::device_buffer> rows,
                                        std::unique_ptr<rmm::device_buffer> graph,
                                        std::int64_t n_rows,
                                        std::int64_t dim,
                                        std::size_t stride_bytes,
                                        cagra_build_settings const& settings,
                                        bool rows_unit,
                                        rmm::cuda_stream_view stream)
{
  raft::device_resources res{stream};
  auto view = raft::make_device_strided_matrix_view<T const, std::int64_t>(
    static_cast<T const*>(rows->data()),
    n_rows,
    dim,
    static_cast<std::int64_t>(stride_bytes / sizeof(T)));
  auto const graph_view = raft::make_device_matrix_view<std::uint32_t const, std::int64_t>(
    static_cast<std::uint32_t const*>(graph->data()), n_rows, settings.graph_degree);
  padded_index<T> index(res,
                        cuvs::distance::DistanceType::L2Expanded,
                        cuvs::neighbors::make_device_padded_dataset_view(res, view),
                        graph_view);
  raft::resource::sync_stream(res);
  return std::make_unique<cagra_index_impl>(
    std::move(index), std::move(rows), std::move(graph), n_rows, dim, settings, rows_unit);
}

}  // namespace

std::unique_ptr<cagra_index> make_cagra_index_over(std::unique_ptr<rmm::device_buffer> rows,
                                                   std::unique_ptr<rmm::device_buffer> graph,
                                                   std::int64_t n_rows,
                                                   std::int64_t dim,
                                                   std::size_t stride_bytes,
                                                   cagra_build_settings const& settings,
                                                   bool rows_unit,
                                                   rmm::cuda_stream_view stream)
{
  if (settings.row_type == cagra_row_type::float16) {
    return index_over<half>(
      std::move(rows), std::move(graph), n_rows, dim, stride_bytes, settings, rows_unit, stream);
  }
  return index_over<float>(
    std::move(rows), std::move(graph), n_rows, dim, stride_bytes, settings, rows_unit, stream);
}

void cagra_partition_distances(float const* rows,
                               std::int64_t n,
                               std::int64_t dim,
                               float const* centers,
                               float const* offsets,
                               std::int64_t P,
                               float* out,
                               rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  int constexpr block_threads = 256;
  int const warps_per_block   = block_threads / 32;
  auto const grid             = static_cast<int>(
    std::clamp<std::int64_t>((n + warps_per_block - 1) / warps_per_block, 1, 65535));
  partition_distances_kernel<<<grid, block_threads, 0, stream.get()>>>(
    rows, n, static_cast<int>(dim), centers, offsets, static_cast<int>(P), out);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void cagra_widen_ids(std::uint32_t const* neighbors,
                     std::int64_t n,
                     std::int64_t n_rows,
                     std::int64_t* ids,
                     rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  int constexpr block_threads = 256;
  auto const grid =
    static_cast<int>(std::clamp<std::int64_t>((n + block_threads - 1) / block_threads, 1, 65535));
  widen_ids_kernel<<<grid, block_threads, 0, stream.get()>>>(neighbors, n, n_rows, ids);
  CUDF_CUDA_TRY(cudaGetLastError());
}

std::int64_t cagra_count_misses(std::int64_t const* ids,
                                std::int64_t n,
                                rmm::cuda_stream_view stream)
{
  return static_cast<std::int64_t>(
    thrust::count_if(rmm::exec_policy(stream), ids, ids + n, is_miss{}));
}

void cagra_sort_rows(
  float* distances, std::int64_t* ids, std::int64_t m, std::int64_t k, rmm::cuda_stream_view stream)
{
  if (m == 0 || k == 0) { return; }
  int constexpr block_threads = 256;
  auto const grid =
    static_cast<int>(std::clamp<std::int64_t>((m + block_threads - 1) / block_threads, 1, 65535));
  sort_rows_kernel<<<grid, block_threads, 0, stream.get()>>>(distances, ids, m, k);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace sirius::vss
