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

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>
#include <cuvs/selection/select_k.hpp>

#include <algorithm>
#include <cstdint>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;

// [-2x, |x|^2, 0...] per corpus row, one warp per row.
__global__ void augment_dataset_kernel(float const* x, int64_t n, int64_t d, int64_t dp, float* out)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < n; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v    = x[r * d + j];
      out[r * dp + j] = -2.f * v;
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    for (int64_t j = d + 1 + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) { out[r * dp + d] = acc; }
  }
}

// [q, 1, 0...] per query row, and |q|^2 beside it.
__global__ void augment_queries_kernel(
  float const* q, int64_t m, int64_t d, int64_t dp, float* out, float* norms)
{
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < m; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v    = q[r * d + j];
      out[r * dp + j] = v;
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    for (int64_t j = d + 1 + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) {
      out[r * dp + d] = 1.f;
      norms[r]        = acc;
    }
  }
}

// score = |x|^2 - 2 q.x becomes the distance: + |q|^2, clamped at the zero the rounding can
// undershoot, and square-rooted for the unsquared metric.
__global__ void finish_l2_kernel(
  float* vals, int64_t rows, int64_t k, float const* norms, bool take_sqrt)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const v = fmaxf(vals[p] + norms[p / k], 0.f);
    vals[p]      = take_sqrt ? sqrtf(v) : v;
  }
}

int grid_for_warps(int64_t rows)
{
  constexpr int64_t per_block = kBlock / kWarp;
  return static_cast<int>(std::clamp<int64_t>((rows + per_block - 1) / per_block, 1, 65535));
}

}  // namespace

knn_result gemm_l2_topk(raft::device_resources const& res,
                        dataset_matrix_view dataset,
                        dataset_matrix_view queries,
                        int64_t k,
                        bool take_sqrt,
                        rmm::device_async_resource_ref mr)
{
  auto const n = dataset.extent(0);
  auto const m = queries.extent(0);
  auto const d = dataset.extent(1);
  CUDF_EXPECTS(queries.extent(1) == d, "VSS dataset and query dimensionality must match");
  CUDF_EXPECTS(k >= 1 && k <= n, "VSS k must satisfy 1 <= k <= n_rows");
  CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "gemm_l2_topk: corpus chunk too large");
  auto const stream = raft::resource::get_cuda_stream(res);
  // One extra column carries |x|^2 through the GEMM; the pad keeps rows 16-byte aligned.
  auto const dp = (d + 1 + 3) / 4 * 4;

  rmm::device_uvector<float> xa(static_cast<std::size_t>(n * dp), stream, mr);
  rmm::device_uvector<float> qa(static_cast<std::size_t>(m * dp), stream, mr);
  rmm::device_uvector<float> qn(static_cast<std::size_t>(m), stream, mr);
  augment_dataset_kernel<<<grid_for_warps(n), kBlock, 0, stream.value()>>>(
    dataset.data_handle(), n, d, dp, xa.data());
  augment_queries_kernel<<<grid_for_warps(m), kBlock, 0, stream.value()>>>(
    queries.data_handle(), m, d, dp, qa.data(), qn.data());
  CUDF_CHECK_CUDA(stream.value());

  auto const out_size = static_cast<cudf::size_type>(m * k);
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  // The score tile is the only large buffer: sized so it stays near 1 GiB whatever the corpus
  // chunk, which bounds peak memory the way cuVS's own tiling does.
  constexpr std::size_t kTileBytes = std::size_t{1} << 30;
  auto const tile_rows             = std::clamp<int64_t>(
    static_cast<int64_t>(kTileBytes / (static_cast<std::size_t>(n) * sizeof(float))), 1, m);
  rmm::device_uvector<float> scores(static_cast<std::size_t>(tile_rows * n), stream, mr);

  auto handle = raft::resource::get_cublas_handle(res);
  CUDF_EXPECTS(cublasSetStream(handle, stream.value()) == CUBLAS_STATUS_SUCCESS,
               "gemm_l2_topk: cublasSetStream failed");
  float const alpha = 1.f;
  float const beta  = 0.f;
  for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
    auto const t = std::min(tile_rows, m - q0);
    // Row-major scores[t x n] = qa_tile[t x dp] * xa[n x dp]^T, which column-major cuBLAS sees
    // as scores^T[n x t] = xa^T(op T of a dp x n matrix) * qa_tile^T. FP32 compute, no TF32:
    // the join's answer is exact to FP32 rounding, like the brute-force search it replaces.
    auto const status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     static_cast<int>(n),
                                     static_cast<int>(t),
                                     static_cast<int>(dp),
                                     &alpha,
                                     xa.data(),
                                     CUDA_R_32F,
                                     static_cast<int>(dp),
                                     qa.data() + q0 * dp,
                                     CUDA_R_32F,
                                     static_cast<int>(dp),
                                     &beta,
                                     scores.data(),
                                     CUDA_R_32F,
                                     static_cast<int>(n),
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS, "gemm_l2_topk: cublasGemmEx failed");

    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(scores.data(), t, n),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  }
  auto const total = m * k;
  auto const grid  = static_cast<int>(std::clamp<int64_t>((total + kBlock - 1) / kBlock, 1, 65535));
  finish_l2_kernel<<<grid, kBlock, 0, stream.value()>>>(out_d, m, k, qn.data(), take_sqrt);
  CUDF_CHECK_CUDA(stream.value());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

}  // namespace sirius::vss
