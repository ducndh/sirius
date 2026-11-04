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

#include "vss/routed_edges.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;
/// An edge's sort key is ((cluster << 1) | not nearest) << kKeyShift | row.
constexpr int kKeyShift = 32;

int grid_for(int64_t n)
{
  return static_cast<int>(std::max<int64_t>((n + kBlock - 1) / kBlock, 1));
}

__global__ void key_edges_kernel(int32_t const* clusters,
                                 int64_t const* cluster_rows,
                                 int64_t n_rows,
                                 int64_t n_probes,
                                 int32_t n_clusters,
                                 int64_t k,
                                 uint64_t* keys,
                                 int32_t* nearest,
                                 int32_t* first_refused)
{
  auto const r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (r >= n_rows) { return; }
  auto const first   = r * n_probes;
  int64_t candidates = 0;
  bool in_range      = true;
  for (int64_t t = 0; t < n_probes; ++t) {
    auto const c = clusters[first + t];
    in_range     = in_range && c >= 0 && c < n_clusters;
    if (in_range) { candidates += cluster_rows[c]; }
    auto const key  = (static_cast<uint64_t>(static_cast<uint32_t>(c)) << 1) | (t > 0 ? 1u : 0u);
    keys[first + t] = (key << kKeyShift) | static_cast<uint64_t>(r);
  }
  nearest[r] = clusters[first];
  if (!in_range || candidates < k) { atomicMin(first_refused, static_cast<int32_t>(r)); }
}

// bounds[v] = the first edge whose key is >= v.
__global__ void key_bounds_kernel(uint64_t const* sorted_keys,
                                  int64_t n_edges,
                                  int64_t* bounds,
                                  int64_t n_bounds)
{
  auto const v = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (v >= n_bounds) { return; }
  int64_t lo = 0;
  int64_t hi = n_edges;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(sorted_keys[mid] >> kKeyShift) < v) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  bounds[v] = lo;
}

// The row is the key's low half; the buffer becomes the rows, as int64.
__global__ void keys_to_rows_kernel(uint64_t* keys, int64_t n)
{
  auto const i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) { keys[i] &= (uint64_t{1} << kKeyShift) - 1; }
}

}  // namespace

keyed_edges key_routed_edges(rmm::device_uvector<int32_t> clusters,
                             int64_t n_rows,
                             int64_t n_probes,
                             std::vector<int64_t> const& cluster_rows,
                             int64_t k,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const n_edges = n_rows * n_probes;
  CUDF_EXPECTS(static_cast<int64_t>(clusters.size()) == n_edges,
               "key_routed_edges: the routing is not n_rows x n_probes");
  CUDF_EXPECTS(n_rows <= std::numeric_limits<int32_t>::max(),
               "key_routed_edges: too many rows for 32-bit row ids");

  keyed_edges out;
  out.n_rows     = n_rows;
  out.n_probes   = n_probes;
  out.n_clusters = static_cast<int32_t>(cluster_rows.size());
  out.keys.emplace(static_cast<std::size_t>(n_edges), stream, mr);
  out.nearest_cluster.resize(static_cast<std::size_t>(n_rows));

  rmm::device_uvector<int64_t> device_cluster_rows(cluster_rows.size(), stream, mr);
  rmm::device_uvector<int32_t> nearest(static_cast<std::size_t>(n_rows), stream, mr);
  rmm::device_uvector<int32_t> first_refused(1, stream, mr);
  auto const none = std::numeric_limits<int32_t>::max();
  first_refused.set_element_async(0, none, stream);
  // From a pageable vector, so the copy has completed when the call returns.
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_cluster_rows.data(),
                                cluster_rows.data(),
                                cluster_rows.size() * sizeof(int64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  key_edges_kernel<<<grid_for(n_rows), kBlock, 0, stream.value()>>>(clusters.data(),
                                                                    device_cluster_rows.data(),
                                                                    n_rows,
                                                                    n_probes,
                                                                    out.n_clusters,
                                                                    std::max<int64_t>(k, 1),
                                                                    out.keys->data(),
                                                                    nearest.data(),
                                                                    first_refused.data());
  CUDF_CHECK_CUDA(stream.value());
  int32_t refused = none;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &refused, first_refused.data(), sizeof(refused), cudaMemcpyDeviceToHost, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(out.nearest_cluster.data(),
                                nearest.data(),
                                out.nearest_cluster.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  if (refused != none) {
    out.refused_row = refused;
    out.refused_clusters.resize(static_cast<std::size_t>(n_probes));
    CUDF_CUDA_TRY(cudaMemcpyAsync(out.refused_clusters.data(),
                                  clusters.data() + refused * n_probes,
                                  out.refused_clusters.size() * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
  }
  return out;
}

sorted_edges sort_routed_edges(keyed_edges edges,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const n_edges    = edges.n_rows * edges.n_probes;
  auto const n_clusters = static_cast<int64_t>(edges.n_clusters);
  auto const n_keys     = 2 * n_clusters;
  int key_bits          = 1;
  while ((int64_t{1} << key_bits) < n_keys) {
    ++key_bits;
  }

  auto first = std::move(*edges.keys);
  rmm::device_uvector<uint64_t> second(static_cast<std::size_t>(n_edges), stream, mr);
  cub::DoubleBuffer<uint64_t> keys(first.data(), second.data());
  auto const items       = static_cast<int32_t>(n_edges);
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
    nullptr, temp_bytes, keys, items, kKeyShift, kKeyShift + key_bits, stream.value()));
  rmm::device_buffer temp(temp_bytes, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
    temp.data(), temp_bytes, keys, items, kKeyShift, kKeyShift + key_bits, stream.value()));

  auto const n_bounds = n_keys + 1;
  rmm::device_uvector<int64_t> bounds(static_cast<std::size_t>(n_bounds), stream, mr);
  key_bounds_kernel<<<grid_for(n_bounds), kBlock, 0, stream.value()>>>(
    keys.Current(), n_edges, bounds.data(), n_bounds);
  keys_to_rows_kernel<<<grid_for(n_edges), kBlock, 0, stream.value()>>>(keys.Current(), n_edges);
  CUDF_CHECK_CUDA(stream.value());
  std::vector<int64_t> host_bounds(static_cast<std::size_t>(n_bounds));
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_bounds.data(),
                                bounds.data(),
                                host_bounds.size() * sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  std::vector<int64_t> edge_begin(static_cast<std::size_t>(n_clusters) + 1);
  std::vector<int64_t> nearest_edges(static_cast<std::size_t>(n_clusters));
  for (std::size_t c = 0; c < static_cast<std::size_t>(n_clusters); ++c) {
    edge_begin[c]    = host_bounds[2 * c];
    nearest_edges[c] = host_bounds[2 * c + 1] - host_bounds[2 * c];
  }
  edge_begin.back() = n_edges;
  return sorted_edges{keys.Current() == first.data() ? std::move(first) : std::move(second),
                      std::move(edge_begin),
                      std::move(nearest_edges)};
}

}  // namespace sirius::vss
