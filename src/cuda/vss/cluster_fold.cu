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

#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"

#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstdint>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;

int grid_for(int64_t n)
{
  return static_cast<int>(std::max<int64_t>((n + kBlock - 1) / kBlock, 1));
}

__global__ void gather_rows_kernel(
  float const* src, int64_t dim, int64_t const* rows, int64_t m, float* out)
{
  auto const total = m * dim;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i = p / dim;
    out[p]       = src[rows[i] * dim + (p - i * dim)];
  }
}

// One thread per part row. The merged row is written back into the accumulator row it was
// read from, which is safe back to front: first count how many of the k survivors come from
// the accumulator, then fill positions k-1 .. 0 with the larger remaining head. Every write
// lands at or after the accumulator entry still to be read, so nothing is overwritten unread
// and no scratch row is needed.
__global__ void fold_topk_rows_kernel(float* acc_d,
                                      int64_t* acc_n,
                                      int64_t k,
                                      float const* part_d,
                                      int64_t const* part_n,
                                      int64_t part_width,
                                      int64_t k_eff,
                                      int64_t const* rows,
                                      int64_t m,
                                      int64_t id_base,
                                      int64_t const* id_map)
{
  auto const i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= m) { return; }
  float* ad      = acc_d + rows[i] * k;
  int64_t* an    = acc_n + rows[i] * k;
  auto const* pd = part_d + i * part_width;
  auto const* pn = part_n + i * part_width;

  // Forward pass, reads only: ties go to the accumulator.
  int64_t ia = 0;
  int64_t ib = 0;
  while (ia + ib < k) {
    if (ib < k_eff && (ia >= k || pd[ib] < ad[ia])) {
      ++ib;
    } else {
      ++ia;
    }
  }
  if (ib == 0) { return; }

  // Backward pass: of the two heads, the later one in the forward order is written last. On a
  // tie that is the part's, mirroring the forward rule.
  for (int64_t p = k - 1; p >= 0; --p) {
    if (ia == 0 || (ib > 0 && pd[ib - 1] >= ad[ia - 1])) {
      --ib;
      ad[p] = pd[ib];
      an[p] = id_map != nullptr ? id_map[pn[ib]] : pn[ib] + id_base;
    } else {
      --ia;
      ad[p] = ad[ia];
      an[p] = an[ia];
    }
  }
}

__global__ void remap_radius_edges_kernel(int64_t const* query_rows,
                                          int64_t const* rows,
                                          int32_t* left,
                                          int64_t* neighbors,
                                          int64_t n_edges,
                                          int64_t id_base,
                                          int64_t const* id_map)
{
  for (int64_t e = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; e < n_edges;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    left[e]      = static_cast<int32_t>(rows[query_rows[e]]);
    neighbors[e] = id_map != nullptr ? id_map[neighbors[e]] : neighbors[e] + id_base;
  }
}

__global__ void scatter_row_ids_kernel(int64_t const* dest,
                                       int64_t n,
                                       int64_t row_base,
                                       int64_t* row_ids)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_ids[dest[i]] = row_base + i;
  }
}

__global__ void fill_list_offsets_kernel(int32_t* out, int64_t n, int64_t dim)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i <= n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = static_cast<int32_t>(i * dim);
  }
}

}  // namespace

void scatter_row_ids(
  int64_t const* dest, int64_t n, int64_t row_base, int64_t* row_ids, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  auto const grid = std::min(grid_for(n), 65535);
  scatter_row_ids_kernel<<<grid, kBlock, 0, stream.value()>>>(dest, n, row_base, row_ids);
  CUDF_CHECK_CUDA(stream.value());
}

void fill_list_offsets(int32_t* out, int64_t n, int64_t dim, rmm::cuda_stream_view stream)
{
  auto const grid = std::min(grid_for(n + 1), 65535);
  fill_list_offsets_kernel<<<grid, kBlock, 0, stream.value()>>>(out, n, dim);
  CUDF_CHECK_CUDA(stream.value());
}

void gather_rows(float const* src,
                 int64_t dim,
                 int64_t const* rows,
                 int64_t m,
                 float* out,
                 rmm::cuda_stream_view stream)
{
  if (m == 0) { return; }
  auto const grid = std::min(grid_for(m * dim), 65535);
  gather_rows_kernel<<<grid, kBlock, 0, stream.value()>>>(src, dim, rows, m, out);
  CUDF_CHECK_CUDA(stream.value());
}

void fold_topk_rows(float* acc_distances,
                    int64_t* acc_neighbors,
                    int64_t k,
                    float const* part_distances,
                    int64_t const* part_neighbors,
                    int64_t part_width,
                    int64_t k_eff,
                    int64_t const* rows,
                    int64_t m,
                    int64_t id_base,
                    rmm::cuda_stream_view stream,
                    int64_t const* id_map)
{
  if (m == 0 || k_eff == 0) { return; }
  CUDF_EXPECTS(k_eff <= part_width && k_eff <= k, "fold_topk_rows: k_eff exceeds its row");
  fold_topk_rows_kernel<<<grid_for(m), kBlock, 0, stream.value()>>>(acc_distances,
                                                                    acc_neighbors,
                                                                    k,
                                                                    part_distances,
                                                                    part_neighbors,
                                                                    part_width,
                                                                    k_eff,
                                                                    rows,
                                                                    m,
                                                                    id_base,
                                                                    id_map);
  CUDF_CHECK_CUDA(stream.value());
}

void remap_radius_edges(int64_t const* query_rows,
                        int64_t const* rows,
                        int32_t* left,
                        int64_t* neighbors,
                        int64_t n_edges,
                        int64_t id_base,
                        rmm::cuda_stream_view stream,
                        int64_t const* id_map)
{
  if (n_edges == 0) { return; }
  auto const grid = std::min(grid_for(n_edges), 65535);
  remap_radius_edges_kernel<<<grid, kBlock, 0, stream.value()>>>(
    query_rows, rows, left, neighbors, n_edges, id_base, id_map);
  CUDF_CHECK_CUDA(stream.value());
}

}  // namespace sirius::vss
