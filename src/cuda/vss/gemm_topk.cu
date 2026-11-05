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

#include "cuda/vss/brute_force_search.hpp"
#include "cuda/vss/brute_force_threshold.hpp"
#include "vss/bound_gemm.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/exact_rows.hpp"
#include "vss/size_limits.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cublas_v2.h>
#include <cuvs/selection/select_k.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;

// How a metric becomes "smaller GEMM score is closer". L2 ranks by |x|^2 - 2 q.x, which is one
// GEMM over [q, 1] and [-2x, |x|^2]; cosine ranks by -q^.x^, a GEMM over unit vectors. Either
// way the selection or the threshold reads the GEMM output directly, and only survivors are
// turned into the metric's own distance.
// A cosine THRESHOLD instead keeps the raw dot product and applies 1 - q.x / (|q||x|) per pair:
// on the boundary that is what decides membership, and it is the formula DuckDB and cuVS round
// the same way, where normalizing first moves a few boundary pairs.
// Inner product ranks by -2 q.x, a GEMM over [q] and [-2x]: the score is the join's distance
// itself (twice the negative inner product, exact in FP32 since the factor is a power of two).
enum class score_kind : int { l2_squared, l2, cosine, cosine_raw, inner_product };

std::optional<score_kind> score_kind_of(cuvs::distance::DistanceType metric)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded: return score_kind::l2_squared;
    case cuvs::distance::DistanceType::L2SqrtExpanded: return score_kind::l2;
    case cuvs::distance::DistanceType::CosineExpanded: return score_kind::cosine;
    case cuvs::distance::DistanceType::InnerProduct: return score_kind::inner_product;
    default: return std::nullopt;
  }
}

// Corpus rows, one warp per row: [-2x, |x|^2, 0...] for L2, [x/|x|, 0...] for cosine. A zero
// row stays zero under cosine, i.e. similarity 0 to everything.
__global__ void prepare_dataset_kernel(
  float const* x, int64_t n, int64_t d, int64_t dp, int kind, float* out, float* root_norms)
{
  bool const cosine = kind == static_cast<int>(score_kind::cosine) ||
                      kind == static_cast<int>(score_kind::cosine_raw);
  bool const ip    = kind == static_cast<int>(score_kind::inner_product);
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < n; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v = x[r * d + j];
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    auto const scale = kind == static_cast<int>(score_kind::cosine_raw)
                         ? 1.f
                         : (cosine ? (acc > 0.f ? rsqrtf(acc) : 0.f) : -2.f);
    for (int64_t j = lane; j < d; j += kWarp) {
      out[r * dp + j] = scale * x[r * d + j];
    }
    auto const extra = cosine || ip ? d : d + 1;
    for (int64_t j = extra + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) {
      if (!cosine && !ip) { out[r * dp + d] = acc; }
      root_norms[r] = sqrtf(acc);
    }
  }
}

// Query rows: [q, 1, 0...] and |q|^2 for L2, [-q/|q|, 0...] for cosine (negated so that the
// smallest score is the most similar, as for L2).
__global__ void prepare_queries_kernel(
  float const* q, int64_t m, int64_t d, int64_t dp, int kind, float* out, float* norms)
{
  bool const cosine = kind == static_cast<int>(score_kind::cosine) ||
                      kind == static_cast<int>(score_kind::cosine_raw);
  bool const ip    = kind == static_cast<int>(score_kind::inner_product);
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < m; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v = q[r * d + j];
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    auto const scale = kind == static_cast<int>(score_kind::cosine_raw)
                         ? 1.f
                         : (cosine ? (acc > 0.f ? -rsqrtf(acc) : 0.f) : 1.f);
    for (int64_t j = lane; j < d; j += kWarp) {
      out[r * dp + j] = scale * q[r * d + j];
    }
    auto const extra = cosine || ip ? d : d + 1;
    for (int64_t j = extra + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) {
      if (!cosine && !ip) { out[r * dp + d] = 1.f; }
      norms[r] = acc;
    }
  }
}

// The metric's distance from a GEMM score: + |q|^2 (clamped at the zero rounding can undershoot)
// and square-rooted for L2; 1 + score = 1 - cos for cosine.
__device__ __forceinline__ float score_to_distance(float s, float qn, int kind)
{
  if (kind == static_cast<int>(score_kind::cosine)) { return 1.f + s; }
  // -2 q.x is the distance itself; it is negative for every pair with a positive inner product.
  if (kind == static_cast<int>(score_kind::inner_product)) { return s; }
  auto const v = fmaxf(s + qn, 0.f);
  return kind == static_cast<int>(score_kind::l2) ? sqrtf(v) : v;
}

__global__ void finish_topk_kernel(
  float* vals, int64_t rows, int64_t k, float const* norms, int kind)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    vals[p] = score_to_distance(vals[p], norms[p / k], kind);
  }
}

// One pass over a [t x n] score tile: every entry at or under its row's bound is appended as an
// edge. blockIdx.y is the row, so no per-element division; each thread takes kPerThread
// consecutive columns and a block scan places the block's matches with ONE global atomic, which
// is what keeps a dense threshold (hundreds of millions of pairs) from serializing on the counter.
constexpr int kPerThread = 8;

__global__ void threshold_emit_kernel(float const* scores,
                                      int64_t n,
                                      float const* bounds,
                                      float const* norms,
                                      int kind,
                                      float const* col_root_norms,
                                      float eps,
                                      int64_t row0,
                                      int64_t* out_rows,
                                      int64_t* out_cols,
                                      float* out_dist,
                                      unsigned long long* count,
                                      unsigned long long capacity)
{
  using block_scan = cub::BlockScan<int, kBlock>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ unsigned long long block_base;

  auto const r     = static_cast<int64_t>(blockIdx.y);
  auto const bound = bounds[r];
  auto const* row  = scores + r * n;
  auto const c0 =
    (static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x) * static_cast<int64_t>(kPerThread);

  bool const raw_cosine = kind == static_cast<int>(score_kind::cosine_raw);
  auto const q_root     = raw_cosine ? sqrtf(norms[r]) : 0.f;
  // For the raw-dot cosine, v holds the distance itself and the bound is eps; a zero-norm row
  // on either side has no direction and joins nothing.
  float v[kPerThread];
  int local = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    auto const c = c0 + i;
    if (c >= n) {
      v[i] = INFINITY;
    } else if (raw_cosine) {
      auto const denom = q_root * col_root_norms[c];
      v[i]             = denom > 0.f ? 1.f - row[c] / denom : INFINITY;
    } else {
      v[i] = row[c];
    }
    local += v[i] <= (raw_cosine ? eps : bound);
  }
  int offset = 0;
  int total  = 0;
  block_scan(scan_storage).ExclusiveSum(local, offset, total);
  if (threadIdx.x == 0) {
    block_base = total == 0 ? 0 : atomicAdd(count, static_cast<unsigned long long>(total));
  }
  __syncthreads();
  if (local == 0) { return; }
  auto pos      = block_base + static_cast<unsigned long long>(offset);
  auto const qn = norms[r];
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    if (v[i] <= (raw_cosine ? eps : bound)) {
      if (pos < capacity) {
        out_rows[pos] = row0 + r;
        out_cols[pos] = c0 + i;
        out_dist[pos] = raw_cosine ? v[i] : score_to_distance(v[i], qn, kind);
      }
      ++pos;
    }
  }
}

// Per query row, the score bound equivalent to distance <= eps.
__global__ void threshold_bounds_kernel(
  float const* norms, int64_t m, float eps, int kind, float* bounds)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (kind == static_cast<int>(score_kind::cosine)) {
      bounds[r] = eps - 1.f;  // 1 + s <= eps
    } else if (kind == static_cast<int>(score_kind::inner_product)) {
      bounds[r] = eps;  // the score is the distance, -2 q.x <= eps
    } else {
      auto const eps2 = kind == static_cast<int>(score_kind::l2) ? eps * eps : eps;
      bounds[r]       = eps2 - norms[r];  // s + |q|^2 <= eps^2
    }
  }
}

int grid_for_warps(int64_t rows)
{
  constexpr int64_t per_block = kBlock / kWarp;
  return static_cast<int>(std::clamp<int64_t>((rows + per_block - 1) / per_block, 1, 65535));
}

int grid_for(int64_t n)
{
  return static_cast<int>(std::clamp<int64_t>((n + kBlock - 1) / kBlock, 1, 65535));
}

/// The prepared operands of a GEMM-ranked search and the tiled GEMM over them.
struct gemm_search {
  int64_t n, m, d, dp;
  score_kind kind;
  rmm::device_uvector<float> xa, qa, qn;
  rmm::device_uvector<float> x_root_norms;  ///< |x| per corpus row
  rmm::device_uvector<float> scores;
  int64_t tile_rows;

  gemm_search(raft::device_resources const& res,
              dataset_matrix_view dataset,
              dataset_matrix_view queries,
              score_kind kind_,
              rmm::device_async_resource_ref mr)
    : n(dataset.extent(0)),
      m(queries.extent(0)),
      d(dataset.extent(1)),
      // L2 carries |x|^2 through one extra column; the pad keeps rows 16-byte aligned.
      dp((d +
          (kind_ == score_kind::cosine || kind_ == score_kind::cosine_raw ||
               kind_ == score_kind::inner_product
             ? 0
             : 1) +
          3) /
         4 * 4),
      kind(kind_),
      xa(static_cast<std::size_t>(n * dp), raft::resource::get_cuda_stream(res), mr),
      qa(static_cast<std::size_t>(m * dp), raft::resource::get_cuda_stream(res), mr),
      qn(static_cast<std::size_t>(m), raft::resource::get_cuda_stream(res), mr),
      x_root_norms(static_cast<std::size_t>(n), raft::resource::get_cuda_stream(res), mr),
      // The score tile is the only large buffer, bounded whatever the corpus chunk, which
      // bounds peak memory the way cuVS's own tiling does.
      scores(0, raft::resource::get_cuda_stream(res), mr),
      tile_rows(
        std::clamp<int64_t>(static_cast<int64_t>(gemm_search_tile_bytes() /
                                                 (static_cast<std::size_t>(n) * sizeof(float))),
                            1,
                            // The threshold emission puts a tile's rows on the grid's y dimension.
                            std::min<int64_t>(std::max<int64_t>(m, 1), 65535)))
  {
    CUDF_EXPECTS(queries.extent(1) == d, "VSS dataset and query dimensionality must match");
    CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "GEMM search: corpus chunk too large");
    auto const stream = raft::resource::get_cuda_stream(res);
    prepare_dataset_kernel<<<grid_for_warps(n), kBlock, 0, stream.get()>>>(
      dataset.data_handle(), n, d, dp, static_cast<int>(kind), xa.data(), x_root_norms.data());
    prepare_queries_kernel<<<grid_for_warps(m), kBlock, 0, stream.get()>>>(
      queries.data_handle(), m, d, dp, static_cast<int>(kind), qa.data(), qn.data());
    CUDF_CHECK_CUDA(stream.get());
    scores.resize(static_cast<std::size_t>(tile_rows * n), stream);
  }

  /// Run the GEMM tile by tile; @p consume(q0, t) reads scores [t x n] for queries [q0, q0 + t).
  void for_each_tile(raft::device_resources const& res,
                     std::function<void(int64_t, int64_t)> const& consume)
  {
    for_each_tile_while(res, [&](int64_t q0, int64_t t) {
      consume(q0, t);
      return true;
    });
  }

  /// The same, stopping when @p consume returns false.
  void for_each_tile_while(raft::device_resources const& res,
                           std::function<bool(int64_t, int64_t)> const& consume)
  {
    auto const stream = raft::resource::get_cuda_stream(res);
    auto handle       = raft::resource::get_cublas_handle(res);
    CUDF_EXPECTS(cublasSetStream(handle, stream.get()) == CUBLAS_STATUS_SUCCESS,
                 "GEMM search: cublasSetStream failed");
    float const alpha = 1.f;
    float const beta  = 0.f;
    for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
      auto const t = std::min(tile_rows, m - q0);
      // Row-major scores[t x n] = qa_tile[t x dp] * xa[n x dp]^T, which column-major cuBLAS sees
      // as scores^T[n x t] = xa^T(op T of a dp x n matrix) * qa_tile^T. FP32 compute, no TF32:
      // the answer is exact to FP32 rounding, like the brute-force search it replaces.
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
      CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS, "GEMM search: cublasGemmEx failed");
      if (!consume(q0, t)) { return; }
    }
  }
};

template <typename T>
std::unique_ptr<cudf::column> uvector_to_column(rmm::device_uvector<T>&& v, cudf::type_id id)
{
  auto const size = column_size(static_cast<std::int64_t>(v.size()), "vector join threshold");
  return std::make_unique<cudf::column>(
    cudf::data_type{id}, size, v.release(), rmm::device_buffer{}, 0);
}

// int32 dot products -> float scores |x|^2 - 2 q.x, in place (same width). Exact: both terms are
// integers under 2^24 for byte-valued data of dimension up to 512.
// Rows are ldc wide (n rounded up to 4, which the int8 GEMM requires of its output); the padding
// becomes +inf so the selection never picks it. blockIdx.y is the row.
__global__ void int8_scores_kernel(int32_t* tile, int64_t n, int64_t ldc, int32_t const* x_sq)
{
  auto* row   = tile + static_cast<int64_t>(blockIdx.y) * ldc;
  auto* row_f = reinterpret_cast<float*>(row);
  for (int64_t c = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; c < ldc;
       c += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_f[c] = c < n ? static_cast<float>(x_sq[c] - 2 * row[c]) : INFINITY;
  }
}

__global__ void finish_int8_topk_kernel(
  float* vals, int64_t rows, int64_t k, int32_t const* q_sq, bool take_sqrt)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const v = fmaxf(vals[p] + static_cast<float>(q_sq[p / k]), 0.f);
    vals[p]      = take_sqrt ? sqrtf(v) : v;
  }
}

// The FP16 counterpart of int8_scores_kernel: the tile already holds FP32 dot products.
__global__ void f16_scores_kernel(float* tile, int64_t n, int64_t ldc, float const* x_sq)
{
  auto* row = tile + static_cast<int64_t>(blockIdx.y) * ldc;
  for (int64_t c = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; c < ldc;
       c += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row[c] = c < n ? x_sq[c] - 2.f * row[c] : INFINITY;
  }
}

__global__ void finish_f16_topk_kernel(
  float* vals, int64_t rows, int64_t k, float const* q_sq, bool take_sqrt)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const v = fmaxf(vals[p] + q_sq[p / k], 0.f);
    vals[p]      = take_sqrt ? sqrtf(v) : v;
  }
}

}  // namespace

std::size_t gemm_search_tile_bytes()
{
  static std::size_t const bytes = [] {
    auto const* v = std::getenv("SIRIUS_VSS_GEMM_TILE_MB");
    return (v != nullptr ? std::strtoull(v, nullptr, 10) : 512ull) << 20;
  }();
  return bytes;
}

bool gemm_search_supports(cuvs::distance::DistanceType metric)
{
  return score_kind_of(metric).has_value();
}

knn_result gemm_topk(raft::device_resources const& res,
                     dataset_matrix_view dataset,
                     dataset_matrix_view queries,
                     int64_t k,
                     cuvs::distance::DistanceType metric,
                     rmm::device_async_resource_ref mr)
{
  auto const kind = score_kind_of(metric);
  CUDF_EXPECTS(kind.has_value(), "gemm_topk: unsupported metric");
  CUDF_EXPECTS(k >= 1 && k <= dataset.extent(0), "VSS k must satisfy 1 <= k <= n_rows");
  auto const stream = raft::resource::get_cuda_stream(res);
  gemm_search search(res, dataset, queries, *kind, mr);
  auto const m = search.m;
  auto const n = search.n;

  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  search.for_each_tile(res, [&](int64_t q0, int64_t t) {
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        search.scores.data(), t, n),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  });
  finish_topk_kernel<<<grid_for(m * k), kBlock, 0, stream.get()>>>(
    out_d, m, k, search.qn.data(), static_cast<int>(*kind));
  CUDF_CHECK_CUDA(stream.get());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

knn_result gemm_l2_topk(raft::device_resources const& res,
                        dataset_matrix_view dataset,
                        dataset_matrix_view queries,
                        int64_t k,
                        bool take_sqrt,
                        rmm::device_async_resource_ref mr)
{
  return gemm_topk(res,
                   dataset,
                   queries,
                   k,
                   take_sqrt ? cuvs::distance::DistanceType::L2SqrtExpanded
                             : cuvs::distance::DistanceType::L2Expanded,
                   mr);
}

threshold_join_result gemm_threshold(raft::device_resources const& res,
                                     dataset_matrix_view dataset,
                                     dataset_matrix_view queries,
                                     float eps,
                                     cuvs::distance::DistanceType metric,
                                     rmm::device_async_resource_ref mr,
                                     bool count_only)
{
  auto kind = score_kind_of(metric);
  CUDF_EXPECTS(kind.has_value(), "gemm_threshold: unsupported metric");
  if (*kind == score_kind::cosine) { kind = score_kind::cosine_raw; }
  auto const stream = raft::resource::get_cuda_stream(res);
  gemm_search search(res, dataset, queries, *kind, mr);
  auto const m = search.m;
  auto const n = search.n;

  rmm::device_uvector<float> bounds(static_cast<std::size_t>(m), stream, mr);
  threshold_bounds_kernel<<<grid_for(m), kBlock, 0, stream.get()>>>(
    search.qn.data(), m, eps, static_cast<int>(*kind), bounds.data());

  // Edges accumulate in one buffer that doubles when a tile overflows it. The overflowing tile's
  // scores are still in place, so only its emission is re-run, never its GEMM. Counting only,
  // the buffer has no room and the kernel writes nothing: the counter is the answer.
  std::size_t capacity = count_only ? 0 : std::size_t{1} << 20;
  std::size_t size     = 0;
  rmm::device_uvector<int64_t> rows(capacity, stream, mr);
  rmm::device_uvector<int64_t> cols(capacity, stream, mr);
  rmm::device_uvector<float> dist(capacity, stream, mr);
  rmm::device_uvector<unsigned long long> counter(1, stream, mr);

  auto const per_block = static_cast<int64_t>(kBlock) * kPerThread;
  search.for_each_tile(res, [&](int64_t q0, int64_t t) {
    for (;;) {
      CUDF_CUDA_TRY(cudaMemsetAsync(counter.data(), 0, sizeof(unsigned long long), stream.get()));
      dim3 const grid(static_cast<unsigned>((n + per_block - 1) / per_block),
                      static_cast<unsigned>(t));
      threshold_emit_kernel<<<grid, kBlock, 0, stream.get()>>>(
        search.scores.data(),
        n,
        bounds.data() + q0,
        search.qn.data() + q0,
        static_cast<int>(*kind),
        search.x_root_norms.data(),
        eps,
        q0,
        count_only ? nullptr : rows.data() + size,
        count_only ? nullptr : cols.data() + size,
        count_only ? nullptr : dist.data() + size,
        counter.data(),
        count_only ? 0 : capacity - size);
      CUDF_CHECK_CUDA(stream.get());
      unsigned long long emitted = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        &emitted, counter.data(), sizeof(emitted), cudaMemcpyDeviceToHost, stream.get()));
      stream.sync();
      if (count_only || size + emitted <= capacity) {
        size += emitted;
        return;
      }
      auto const grown = static_cast<std::size_t>(grown_pair_capacity(
        capacity, size + emitted, 2 * sizeof(int64_t) + sizeof(float), "gemm_threshold"));
      rows.resize(grown, stream);
      cols.resize(grown, stream);
      dist.resize(grown, stream);
      capacity = grown;
    }
  });
  if (count_only) {
    return threshold_join_result{nullptr, nullptr, nullptr, static_cast<int64_t>(size)};
  }
  rows.resize(size, stream);
  cols.resize(size, stream);
  dist.resize(size, stream);
  rows.shrink_to_fit(stream);
  cols.shrink_to_fit(stream);
  dist.shrink_to_fit(stream);
  return threshold_join_result{uvector_to_column(std::move(rows), cudf::type_id::INT64),
                               uvector_to_column(std::move(cols), cudf::type_id::INT64),
                               uvector_to_column(std::move(dist), cudf::type_id::FLOAT32),
                               static_cast<int64_t>(size)};
}

namespace {

// An exact search answers from the differences,
//   R(q, x) = sum_j (q_j - x_j)^2 in FP32,
// which the GEMM's form cancels for rows far from the origin and close together, so here the GEMM
// only chooses the pairs R is computed for: a pair may pass and not be in the answer, but no pair
// in the answer may fail. Over the rows as stored, in real arithmetic, let
// D = sum_j (q_j - x_j)^2, A = |q|, B = |x|, u = 2^-24, with d u <= 0.01. Every FP32 operation
// rounds within u, and a sum's terms meet at most d - 1 additions in any order. The tensor cores'
// order is unspecified, so as for the FP16 filter the GEMM's error is taken at twice that.
//  1. The differences. A term of R meets a subtraction, a product and at most d additions, so
//     |R - D| <= 1.01 (d + 2) u D. A Euclidean distance decides by sqrtf(R) <= eps, with sqrtf
//     within 4 u, which gives the same with r = eps^2 (1 + 9 u).
//  2. The score. s is the GEMM of [q, 1] and [-2x, X], where X = FP32 |x|^2 = B^2 (1 + t) and
//     |t| <= 1.01 (d + 1) u, and -2 x_j is exact. So |s - (X - 2 q.x)| <= 2 (d + 1) u (2 A B + X).
//     With Q = FP32 |q|^2 and D = A^2 + B^2 - 2 q.x,
//       |s + Q - D| <= 2 (d + 1) u (2 A B + X) + 1.01 (d + 1) u (A^2 + B^2)
//                   <= 3.03 (d + 1) u (A + B)^2.
//     Let E = 4 (d + 4) u (sqrtf(Q) + x_max)^2 + d 2^-122, x_max the chunk's largest |x|. It holds
//     that bound, at least (0.9 d + 11) u of the norm term over it for this expression's own
//     rounding and sqrtf's in the norms. The last term is underflow: a pair's 4 d + 8 operations
//     each flush at most 2^-126.
//  3. A radius. R <= r gives D <= r (1 + 1.01 (d + 2) u), so a pair within r passes when
//       s <= r (1 + 2 (d + 16) u) + E - Q.
//     A running top-k is the same with r its k-th: a pair enters only with R at most that.
//  4. A top k. Let s_k be a row's k-th smallest score. Those k pairs have D <= s_k + Q + E, so
//     R <= (s_k + Q + E) (1 + 1.01 (d + 2) u), and the row's k-th smallest R is at most that. By
//     (1) a pair at or under it has D <= (s_k + Q + E) (1 + 2.1 (d + 2) u), so it passes when
//       s <= s_k + 2 E + 3 (d + 16) u max(s_k + Q + E, 0).
//     Ties at the k-th are inside. Every pair under that limit must be seen: a row whose last
//     selected score is still under it may have more, and goes to the differences.
//  5. Overflow. With (A + B)^2 finite no partial sum of s overflows. A row whose norm is not finite
//     can still have a finite R (components near 3e19 square past FP32's range, their differences
//     need not), and its s is then NaN and fails every comparison. So a chunk with a non-finite
//     norm, or a NaN or infinite component, which makes one, is decided by differences alone.
// Far from the origin E can exceed every distance in the data: the limits stay correct and pass
// nearly everything, which the checks below turn back to the differences.

/// The largest share of a tile's pairs a radius screen may pass. A re-score reads two rows per pair
/// and a tile of differences reads them in bulk, so past this share the screen and its re-score
/// cost more than the differences alone. It decides speed and memory, never the answer.
constexpr int kScreenPassShift       = 10;
constexpr int64_t kMinScreenCapacity = int64_t{1} << 12;

/// Pairs a top-k screen selects per row past k. A row with more under its limit goes to the
/// differences, so it decides speed only: past k + this, exact ties are the common reason.
constexpr int64_t kScreenSelectSlack = 32;

/// Below these an exact search keeps the differences. The GEMM rewrites both sides into their
/// prepared layout per call, and the screen adds about 150 us of setup and kernels: on
/// every-cluster FP32 lists at 96-d, 1,000 probe rows lost to the differences up to 3e6 pairs a
/// slice and won at 1e7, 10,000 probe rows won from 1e6.
constexpr int64_t kScreenMinProbeRows = 512;
constexpr int64_t kScreenMinPairs     = 10'000'000;

bool exact_screen_enabled()
{
  static bool const enabled = [] {
    auto const off = [](char const* name) {
      auto const* v = std::getenv(name);
      return v != nullptr && std::string_view{v} == "0";
    };
    auto const* kernel = std::getenv("SIRIUS_VSS_KERNEL");
    return !off("SIRIUS_VSS_EXACT_SCREEN") && !off("SIRIUS_VSS_GEMM_TOPK") &&
           !(kernel != nullptr && std::string_view{kernel} == "faiss");
  }();
  return enabled;
}

struct finite_or_infinite {
  __device__ float operator()(float v) const { return v < INFINITY ? v : INFINITY; }
};

struct within_radius {
  float const* squared;
  float eps;
  bool sqrt_radius;
  __device__ bool operator()(int64_t i) const
  {
    return (sqrt_radius ? sqrtf(squared[i]) : squared[i]) <= eps;
  }
};

struct place_kept_pair {
  int64_t const* kept;
  int32_t const* from_rows;
  int64_t const* from_cols;
  float const* from_dist;
  int64_t* to_rows;
  int64_t* to_cols;
  float* to_dist;
  bool sqrt_radius;
  __device__ void operator()(int64_t i) const
  {
    auto const p = kept[i];
    to_rows[i]   = from_rows[p];
    to_cols[i]   = from_cols[p];
    to_dist[i]   = sqrt_radius ? sqrtf(from_dist[p]) : from_dist[p];
  }
};

struct place_shifted_pair {
  int64_t const* from_rows;
  int64_t const* from_cols;
  float const* from_dist;
  int64_t row_shift;
  int64_t* to_rows;
  int64_t* to_cols;
  float* to_dist;
  __device__ void operator()(int64_t i) const
  {
    to_rows[i] = from_rows[i] + row_shift;
    to_cols[i] = from_cols[i];
    to_dist[i] = from_dist[i];
  }
};

struct mapped_row {
  int64_t const* from;
  int64_t const* map;
  int64_t* to;
  __device__ void operator()(int64_t i) const { to[i] = map[from[i]]; }
};

struct left_to_differences {
  std::uint8_t const* flagged;
  int64_t first_unscreened;
  __device__ bool operator()(int64_t r) const { return r >= first_unscreened || flagged[r] != 0; }
};

// Per probe row, E and the limit (3) under one radius, or under its own running k-th of @p running
// (values under a square root when @p running_is_root), which is +inf until the row has k.
__global__ void screen_limits_kernel(float const* q_sq,
                                     int64_t m,
                                     float norm_factor,
                                     float underflow,
                                     float x_max,
                                     float radius,
                                     float const* running,
                                     int64_t const* running_rows,
                                     int64_t running_k,
                                     bool running_is_root,
                                     float widen,
                                     float* limits,
                                     float* norm_terms)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const reach = sqrtf(q_sq[r]) + x_max;
    float const e     = norm_factor * reach * reach + underflow;
    float limit       = radius;
    if (running != nullptr) {
      auto const row  = running_rows != nullptr ? running_rows[r] : r;
      float const kth = running[row * running_k + running_k - 1];
      limit           = (running_is_root ? kth * kth : kth) * widen;
    }
    norm_terms[r] = e;
    limits[r]     = limit + e - q_sq[r];
  }
}

// One pass over a [t x n] score tile: every pair at or under its row's limit is appended. Placed
// as threshold_emit_kernel places them, with one global atomic a block.
__global__ void screen_emit_kernel(float const* scores,
                                   int64_t n,
                                   float const* limits,
                                   int64_t row0,
                                   int32_t* out_rows,
                                   int64_t* out_cols,
                                   unsigned long long* count,
                                   unsigned long long capacity)
{
  using block_scan = cub::BlockScan<int, kBlock>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ unsigned long long block_base;

  auto const r     = static_cast<int64_t>(blockIdx.y);
  auto const limit = limits[r];
  auto const* row  = scores + r * n;
  auto const c0 =
    (static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x) * static_cast<int64_t>(kPerThread);
  bool pass[kPerThread];
  int local = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    auto const c = c0 + i;
    pass[i]      = c < n && row[c] <= limit;
    local += pass[i];
  }
  int offset = 0;
  int total  = 0;
  block_scan(scan_storage).ExclusiveSum(local, offset, total);
  if (threadIdx.x == 0) {
    block_base = total == 0 ? 0 : atomicAdd(count, static_cast<unsigned long long>(total));
  }
  __syncthreads();
  if (local == 0) { return; }
  auto pos = block_base + static_cast<unsigned long long>(offset);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    if (pass[i]) {
      if (pos < capacity) {
        out_rows[pos] = static_cast<int32_t>(row0 + r);
        out_cols[pos] = c0 + i;
      }
      ++pos;
    }
  }
}

// A selected tile, K scores a row, nearest first: each row's limit is the smaller of (3) and (4),
// and what passes it is appended. A row whose K-th still passes appends nothing and is flagged.
__global__ void screen_select_kernel(float const* selected,
                                     int64_t const* cols,
                                     int64_t t,
                                     int64_t k,
                                     int64_t K,
                                     int64_t n,
                                     float const* limits,
                                     float const* q_sq,
                                     float const* norm_terms,
                                     float widen,
                                     int64_t row0,
                                     int32_t* out_rows,
                                     int64_t* out_cols,
                                     std::uint8_t* flagged,
                                     unsigned long long* count,
                                     unsigned long long* flagged_count,
                                     unsigned long long capacity)
{
  auto const row = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (row >= t) { return; }
  auto const* s    = selected + row * K;
  auto const s_k   = s[k - 1];
  auto const e     = norm_terms[row];
  auto const top   = s_k + 2.f * e + widen * fmaxf(s_k + q_sq[row] + e, 0.f);
  auto const limit = fminf(limits[row], top);
  if (K < n && s[K - 1] <= limit) {
    flagged[row0 + row] = 1;
    atomicAdd(flagged_count, 1ull);
    return;
  }
  for (int64_t i = 0; i < K && s[i] <= limit; ++i) {
    auto const pos = atomicAdd(count, 1ull);
    if (pos < capacity) {
      out_rows[pos] = static_cast<int32_t>(row0 + row);
      out_cols[pos] = cols[row * K + i];
    }
  }
}

// Pairs by accumulator row, then distance: the bits of a distance >= 0 order as its value.
__global__ void screen_merge_keys_kernel(int32_t const* rows,
                                         int64_t const* row_map,
                                         float const* distances,
                                         int64_t n,
                                         uint64_t* keys,
                                         int64_t* order)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const row = row_map != nullptr ? row_map[rows[i]] : static_cast<int64_t>(rows[i]);
    keys[i]        = (static_cast<uint64_t>(row) << 32) | __float_as_uint(distances[i]);
    order[i]       = i;
  }
}

// The thread at a row's first pair merges the row's pairs into its accumulator row, in place, as
// fold_topk_rows does: ties go to the accumulator.
__global__ void screen_merge_kernel(uint64_t const* keys,
                                    int64_t const* order,
                                    float const* distances,
                                    int64_t const* ids,
                                    int64_t n,
                                    int64_t const* id_map,
                                    int64_t id_base,
                                    float* acc_d,
                                    int64_t* acc_n,
                                    int64_t k)
{
  for (int64_t first = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; first < n;
       first += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const row = keys[first] >> 32;
    if (first > 0 && (keys[first - 1] >> 32) == row) { continue; }
    int64_t end = first + 1;
    while (end < n && (keys[end] >> 32) == row && end - first < k) {
      ++end;
    }
    auto const count = end - first;
    auto* ad         = acc_d + static_cast<int64_t>(row) * k;
    auto* an         = acc_n + static_cast<int64_t>(row) * k;
    auto const dist  = [&](int64_t i) { return distances[order[first + i]]; };
    int64_t ia       = 0;
    int64_t ib       = 0;
    while (ia + ib < k) {
      if (ib < count && (ia >= k || dist(ib) < ad[ia])) {
        ++ib;
      } else {
        ++ia;
      }
    }
    if (ib == 0) { continue; }
    for (int64_t p = k - 1; p >= 0; --p) {
      if (ia == 0 || (ib > 0 && dist(ib - 1) >= ad[ia - 1])) {
        --ib;
        auto const id = ids[order[first + ib]];
        ad[p]         = dist(ib);
        an[p]         = id_map != nullptr ? id_map[id] : id + id_base;
      } else {
        --ia;
        ad[p] = ad[ia];
        an[p] = an[ia];
      }
    }
  }
}

/// What the screen reads off a prepared search: the chunk's largest norm, and whether its limits
/// can be computed at all (5).
struct screen_reach {
  float x_max;
  bool usable;
};

screen_reach reach_of(gemm_search const& search, raft::device_resources const& res)
{
  auto const policy  = raft::resource::get_thrust_policy(res);
  auto const largest = [&](float const* v, int64_t count) {
    return thrust::transform_reduce(
      policy, v, v + count, finite_or_infinite{}, 0.f, thrust::maximum<float>{});
  };
  auto const x_max = largest(search.x_root_norms.data(), search.n);
  auto const sum   = static_cast<double>(x_max) +
                   std::sqrt(static_cast<double>(largest(search.qn.data(), search.m)));
  return {x_max, 2.0 * sum * sum < static_cast<double>(std::numeric_limits<float>::max())};
}

double unit_roundoff() { return std::ldexp(1.0, -24); }

/// The chunk's rows as exact_distances reads them: one block holding them all.
struct chunk_rows_source {
  rmm::device_uvector<float const*> block;
  exact_rows view;

  chunk_rows_source(dataset_matrix_view dataset,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
    : block(1, stream, mr)
  {
    block.set_element(0, dataset.data_handle(), stream);
    view.copy_blocks    = block.data();
    view.rows_per_block = std::numeric_limits<int64_t>::max();
  }
};

}  // namespace

bool exact_screen_applies(cuvs::distance::DistanceType metric,
                          int64_t probe_rows,
                          int64_t corpus_rows)
{
  return exact_screen_enabled() &&
         (metric == cuvs::distance::DistanceType::L2Unexpanded ||
          metric == cuvs::distance::DistanceType::L2SqrtUnexpanded) &&
         probe_rows >= kScreenMinProbeRows && probe_rows * corpus_rows >= kScreenMinPairs;
}

threshold_join_result screened_threshold(raft::device_resources const& res,
                                         dataset_matrix_view dataset,
                                         dataset_matrix_view queries,
                                         float eps,
                                         cuvs::distance::DistanceType metric,
                                         rmm::device_async_resource_ref mr,
                                         bool count_only,
                                         bool* screen_worthwhile)
{
  bool const sqrt_radius = metric == cuvs::distance::DistanceType::L2SqrtUnexpanded;
  CUDF_EXPECTS(sqrt_radius || metric == cuvs::distance::DistanceType::L2Unexpanded,
               "screened_threshold: unsupported metric");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const policy = raft::resource::get_thrust_policy(res);
  auto const n      = dataset.extent(0);
  auto const m      = queries.extent(0);
  auto const d      = dataset.extent(1);
  CUDF_EXPECTS(queries.extent(1) == d, "VSS dataset and query dimensionality must match");
  auto const u = unit_roundoff();
  if (n == 0 || m == 0 || static_cast<double>(d) * u > 0.01) {
    return brute_force_threshold(res, dataset, queries, eps, metric, mr);
  }

  std::size_t capacity = count_only ? 0 : std::size_t{1} << 20;
  std::size_t size     = 0;
  rmm::device_uvector<int64_t> rows(capacity, stream, mr);
  rmm::device_uvector<int64_t> cols(capacity, stream, mr);
  rmm::device_uvector<float> dist(capacity, stream, mr);
  auto const grow_to = [&](std::size_t need) {
    if (need <= capacity) { return; }
    auto const grown = static_cast<std::size_t>(grown_pair_capacity(
      capacity, need, 2 * sizeof(int64_t) + sizeof(float), "screened_threshold"));
    rows.resize(grown, stream);
    cols.resize(grown, stream);
    dist.resize(grown, stream);
    capacity = grown;
  };

  // The first probe row the screen did not answer. Its operands are gone when this returns, which
  // leaves the differences the room they would have had without it.
  auto const answered = [&]() -> int64_t {
    gemm_search search(res, dataset, queries, score_kind::l2_squared, mr);
    auto const reach = reach_of(search, res);
    if (!reach.usable) { return 0; }

    double const r      = sqrt_radius ? static_cast<double>(eps) * eps : static_cast<double>(eps);
    double const widen  = 2.0 * static_cast<double>(d + 16) * u;
    double const radius = r + std::fabs(r) * widen;
    rmm::device_uvector<float> limits(static_cast<std::size_t>(m), stream, mr);
    rmm::device_uvector<float> norm_terms(static_cast<std::size_t>(m), stream, mr);
    screen_limits_kernel<<<grid_for(m), kBlock, 0, stream.get()>>>(
      search.qn.data(),
      m,
      static_cast<float>(4.0 * static_cast<double>(d + 4) * u),
      static_cast<float>(static_cast<double>(d) * std::ldexp(1.0, -122)),
      reach.x_max,
      std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity()),
      nullptr,
      nullptr,
      0,
      false,
      0.f,
      limits.data(),
      norm_terms.data());

    auto const tile_pairs = search.tile_rows * n;
    auto const tile_room  = static_cast<std::size_t>(
      std::min(tile_pairs, std::max(kMinScreenCapacity, tile_pairs >> kScreenPassShift)));
    rmm::device_uvector<int32_t> tile_rows(tile_room, stream, mr);
    rmm::device_uvector<int64_t> tile_cols(tile_room, stream, mr);
    rmm::device_uvector<float> tile_dist(tile_room, stream, mr);
    rmm::device_uvector<int64_t> kept(tile_room, stream, mr);
    rmm::device_uvector<unsigned long long> counter(1, stream, mr);
    chunk_rows_source const source(dataset, stream, mr);

    int64_t first_unanswered = m;
    auto const per_block     = static_cast<int64_t>(kBlock) * kPerThread;
    search.for_each_tile_while(res, [&](int64_t q0, int64_t t) {
      CUDF_CUDA_TRY(cudaMemsetAsync(counter.data(), 0, sizeof(unsigned long long), stream.get()));
      dim3 const grid(static_cast<unsigned>((n + per_block - 1) / per_block),
                      static_cast<unsigned>(t));
      screen_emit_kernel<<<grid, kBlock, 0, stream.get()>>>(search.scores.data(),
                                                              n,
                                                              limits.data() + q0,
                                                              q0,
                                                              tile_rows.data(),
                                                              tile_cols.data(),
                                                              counter.data(),
                                                              tile_room);
      CUDF_CHECK_CUDA(stream.get());
      unsigned long long passed = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        &passed, counter.data(), sizeof(passed), cudaMemcpyDeviceToHost, stream.get()));
      stream.sync();
      if (passed > tile_room) {
        first_unanswered = q0;
        return false;
      }
      if (passed == 0) { return true; }
      auto const pairs = static_cast<int64_t>(passed);
      exact_distances(queries.data_handle(),
                      tile_rows.data(),
                      0,
                      tile_cols.data(),
                      tile_dist.data(),
                      pairs,
                      source.view,
                      d,
                      stream,
                      mr);
      auto const first  = thrust::make_counting_iterator<int64_t>(0);
      auto* kept_end    = thrust::copy_if(policy,
                                       first,
                                       first + pairs,
                                       kept.data(),
                                       within_radius{tile_dist.data(), eps, sqrt_radius});
      auto const n_kept = static_cast<std::size_t>(kept_end - kept.data());
      if (!count_only && n_kept > 0) {
        grow_to(size + n_kept);
        thrust::for_each(policy,
                         first,
                         first + static_cast<int64_t>(n_kept),
                         place_kept_pair{kept.data(),
                                         tile_rows.data(),
                                         tile_cols.data(),
                                         tile_dist.data(),
                                         rows.data() + size,
                                         cols.data() + size,
                                         dist.data() + size,
                                         sqrt_radius});
      }
      size += n_kept;
      return true;
    });
    return first_unanswered;
  }();

  if (answered < m) {
    if (screen_worthwhile != nullptr) { *screen_worthwhile = false; }
    auto const rest_queries = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      queries.data_handle() + answered * d, m - answered, d);
    auto rest = brute_force_threshold(res, dataset, rest_queries, eps, metric, mr);
    if (answered == 0) { return rest; }
    auto const rest_edges = static_cast<std::size_t>(rest.n_edges);
    if (!count_only && rest_edges > 0) {
      grow_to(size + rest_edges);
      auto const first = thrust::make_counting_iterator<int64_t>(0);
      thrust::for_each(policy,
                       first,
                       first + rest.n_edges,
                       place_shifted_pair{rest.query_rows->view().data<int64_t>(),
                                          rest.neighbors->view().data<int64_t>(),
                                          rest.distances->view().data<float>(),
                                          answered,
                                          rows.data() + size,
                                          cols.data() + size,
                                          dist.data() + size});
    }
    size += rest_edges;
  }
  if (count_only) {
    return threshold_join_result{nullptr, nullptr, nullptr, static_cast<int64_t>(size)};
  }
  rows.resize(size, stream);
  cols.resize(size, stream);
  dist.resize(size, stream);
  rows.shrink_to_fit(stream);
  cols.shrink_to_fit(stream);
  dist.shrink_to_fit(stream);
  return threshold_join_result{uvector_to_column(std::move(rows), cudf::type_id::INT64),
                               uvector_to_column(std::move(cols), cudf::type_id::INT64),
                               uvector_to_column(std::move(dist), cudf::type_id::FLOAT32),
                               static_cast<int64_t>(size)};
}

void screened_topk_fold(raft::device_resources const& res,
                        dataset_matrix_view dataset,
                        dataset_matrix_view queries,
                        int64_t k,
                        cuvs::distance::DistanceType metric,
                        screened_fold_target const& target,
                        rmm::device_async_resource_ref mr,
                        bool* screen_worthwhile)
{
  bool const take_sqrt = metric == cuvs::distance::DistanceType::L2SqrtUnexpanded;
  CUDF_EXPECTS(take_sqrt || metric == cuvs::distance::DistanceType::L2Unexpanded,
               "screened_topk_fold: unsupported metric");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const policy = raft::resource::get_thrust_policy(res);
  auto const n      = dataset.extent(0);
  auto const m      = queries.extent(0);
  auto const d      = dataset.extent(1);
  CUDF_EXPECTS(queries.extent(1) == d, "VSS dataset and query dimensionality must match");
  CUDF_EXPECTS(k >= 1 && k <= n && k <= target.k, "VSS k must satisfy 1 <= k <= n_rows");
  if (m == 0) { return; }
  auto const u = unit_roundoff();

  rmm::device_uvector<std::uint8_t> flagged(static_cast<std::size_t>(m), stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(flagged.data(), 0, flagged.size(), stream.get()));

  // The first probe row the screen did not reach. Its operands are gone when this returns, which
  // leaves the differences the room they would have had without it.
  auto const first_unscreened = [&]() -> int64_t {
    if (static_cast<double>(d) * u > 0.01) { return 0; }
    gemm_search search(res, dataset, queries, score_kind::l2_squared, mr);
    auto const reach = reach_of(search, res);
    if (!reach.usable) { return 0; }

    rmm::device_uvector<float> limits(static_cast<std::size_t>(m), stream, mr);
    rmm::device_uvector<float> norm_terms(static_cast<std::size_t>(m), stream, mr);
    screen_limits_kernel<<<grid_for(m), kBlock, 0, stream.get()>>>(
      search.qn.data(),
      m,
      static_cast<float>(4.0 * static_cast<double>(d + 4) * u),
      static_cast<float>(static_cast<double>(d) * std::ldexp(1.0, -122)),
      reach.x_max,
      0.f,
      target.distances,
      target.rows,
      target.k,
      take_sqrt,
      static_cast<float>(1.0 + 2.0 * static_cast<double>(d + 16) * u),
      limits.data(),
      norm_terms.data());
    // A row short of k (+inf) has no running bound: the tiles select until every row has one.
    bool const bounded =
      thrust::transform_reduce(
        policy, limits.begin(), limits.end(), finite_or_infinite{}, 0.f, thrust::maximum<float>{}) <
      std::numeric_limits<float>::infinity();

    auto const K          = std::min(n, k + kScreenSelectSlack);
    auto const select_top = static_cast<float>(3.0 * static_cast<double>(d + 16) * u);
    rmm::device_uvector<float> selected(static_cast<std::size_t>(search.tile_rows * K), stream, mr);
    rmm::device_uvector<int64_t> selected_cols(
      static_cast<std::size_t>(search.tile_rows * K), stream, mr);
    rmm::device_uvector<unsigned long long> counters(2, stream, mr);

    auto const room = std::min(m, 4 * search.tile_rows) * K;
    bound_candidates candidates(room, stream, mr);
    chunk_rows_source const source(dataset, stream, mr);

    int row_bits = 1;
    while ((int64_t{1} << row_bits) < target.n_rows) {
      ++row_bits;
    }
    int64_t size     = 0;
    auto const merge = [&] {
      if (size == 0) { return; }
      exact_distances(queries.data_handle(),
                      candidates.rows.data(),
                      0,
                      candidates.ids.data(),
                      candidates.distances.data(),
                      size,
                      source.view,
                      d,
                      stream,
                      mr);
      if (take_sqrt) { sqrt_in_place(candidates.distances.data(), size, stream); }
      rmm::device_uvector<uint64_t> keys(static_cast<std::size_t>(size), stream, mr);
      rmm::device_uvector<uint64_t> keys_sorted(static_cast<std::size_t>(size), stream, mr);
      rmm::device_uvector<int64_t> order(static_cast<std::size_t>(size), stream, mr);
      rmm::device_uvector<int64_t> order_sorted(static_cast<std::size_t>(size), stream, mr);
      screen_merge_keys_kernel<<<grid_for(size), kBlock, 0, stream.get()>>>(
        candidates.rows.data(),
        target.rows,
        candidates.distances.data(),
        size,
        keys.data(),
        order.data());
      std::size_t sort_bytes = 0;
      CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                    sort_bytes,
                                                    keys.data(),
                                                    keys_sorted.data(),
                                                    order.data(),
                                                    order_sorted.data(),
                                                    static_cast<int>(size),
                                                    0,
                                                    32 + row_bits,
                                                    stream.get()));
      rmm::device_buffer temp(sort_bytes, stream, mr);
      CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(temp.data(),
                                                    sort_bytes,
                                                    keys.data(),
                                                    keys_sorted.data(),
                                                    order.data(),
                                                    order_sorted.data(),
                                                    static_cast<int>(size),
                                                    0,
                                                    32 + row_bits,
                                                    stream.get()));
      screen_merge_kernel<<<grid_for(size), kBlock, 0, stream.get()>>>(
        keys_sorted.data(),
        order_sorted.data(),
        candidates.distances.data(),
        candidates.ids.data(),
        size,
        target.id_map,
        target.id_base,
        target.distances,
        target.neighbors,
        target.k);
      CUDF_CHECK_CUDA(stream.get());
      // Its keys go with this scope, and a concurrent task may be handed their memory.
      stream.sync();
      size = 0;
    };

    int64_t stopped_at   = m;
    auto const per_block = static_cast<int64_t>(kBlock) * kPerThread;
    search.for_each_tile_while(res, [&](int64_t q0, int64_t t) {
      auto const tile_most = t * K;
      if (size + tile_most > room) { merge(); }
      CUDF_CUDA_TRY(
        cudaMemsetAsync(counters.data(), 0, 2 * sizeof(unsigned long long), stream.get()));
      unsigned long long seen[2] = {0, 0};
      if (bounded) {
        dim3 const grid(static_cast<unsigned>((n + per_block - 1) / per_block),
                        static_cast<unsigned>(t));
        screen_emit_kernel<<<grid, kBlock, 0, stream.get()>>>(
          search.scores.data(),
          n,
          limits.data() + q0,
          q0,
          candidates.rows.data() + size,
          candidates.ids.data() + size,
          counters.data(),
          static_cast<unsigned long long>(room - size));
        CUDF_CHECK_CUDA(stream.get());
        CUDF_CUDA_TRY(cudaMemcpyAsync(seen,
                                      counters.data(),
                                      sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
        if (static_cast<int64_t>(seen[0]) <= tile_most) {
          size += static_cast<int64_t>(seen[0]);
          return true;
        }
        CUDF_CUDA_TRY(
          cudaMemsetAsync(counters.data(), 0, sizeof(unsigned long long), stream.get()));
      }
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
          search.scores.data(), t, n),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t, raft::row_major>(selected.data(), t, K),
        raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
          selected_cols.data(), t, K),
        /*select_min=*/true,
        /*sorted=*/true);
      screen_select_kernel<<<grid_for(t), kBlock, 0, stream.get()>>>(
        selected.data(),
        selected_cols.data(),
        t,
        k,
        K,
        n,
        limits.data() + q0,
        search.qn.data() + q0,
        norm_terms.data() + q0,
        select_top,
        q0,
        candidates.rows.data() + size,
        candidates.ids.data() + size,
        flagged.data(),
        counters.data(),
        counters.data() + 1,
        static_cast<unsigned long long>(room - size));
      CUDF_CHECK_CUDA(stream.get());
      CUDF_CUDA_TRY(cudaMemcpyAsync(seen,
                                    counters.data(),
                                    2 * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
      size += static_cast<int64_t>(seen[0]);
      if (2 * static_cast<int64_t>(seen[1]) > t) {
        stopped_at = q0 + t;
        return false;
      }
      return true;
    });
    merge();
    return stopped_at;
  }();

  if (first_unscreened < m && screen_worthwhile != nullptr) { *screen_worthwhile = false; }
  rmm::device_uvector<int64_t> rest(static_cast<std::size_t>(m), stream, mr);
  auto const first = thrust::make_counting_iterator<int64_t>(0);
  auto* rest_end   = thrust::copy_if(
    policy, first, first + m, rest.data(), left_to_differences{flagged.data(), first_unscreened});
  auto const rest_rows = static_cast<int64_t>(rest_end - rest.data());
  if (rest_rows == 0) { return; }
  rmm::device_uvector<float> rest_queries(static_cast<std::size_t>(rest_rows * d), stream, mr);
  gather_rows(queries.data_handle(), d, rest.data(), rest_rows, rest_queries.data(), stream);
  auto const knn =
    brute_force_knn(res,
                    dataset,
                    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                      rest_queries.data(), rest_rows, d),
                    k,
                    metric,
                    mr);
  auto const* fold_rows = rest.data();
  rmm::device_uvector<int64_t> mapped(0, stream, mr);
  if (target.rows != nullptr) {
    mapped.resize(static_cast<std::size_t>(rest_rows), stream);
    thrust::for_each(
      policy, first, first + rest_rows, mapped_row{rest.data(), target.rows, mapped.data()});
    fold_rows = mapped.data();
  }
  fold_topk_rows(target.distances,
                 target.neighbors,
                 target.k,
                 knn.distances->view().data<float>(),
                 knn.neighbors->view().data<int64_t>(),
                 knn.k,
                 k,
                 fold_rows,
                 rest_rows,
                 target.id_base,
                 stream,
                 target.id_map);
  stream.sync();
}

knn_result exact_topk(raft::device_resources const& res,
                      dataset_matrix_view dataset,
                      dataset_matrix_view queries,
                      int64_t k,
                      cuvs::distance::DistanceType metric,
                      rmm::device_async_resource_ref mr)
{
  if (!exact_screen_applies(metric, queries.extent(0), dataset.extent(0))) {
    return brute_force_knn(res, dataset, queries, k, metric, mr);
  }
  auto const stream   = raft::resource::get_cuda_stream(res);
  auto const m        = queries.extent(0);
  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();
  fill_misses(out_d, out_n, m * k, stream);
  screened_topk_fold(res, dataset, queries, k, metric, {out_d, out_n, m, k}, mr);
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

threshold_join_result exact_threshold(raft::device_resources const& res,
                                      dataset_matrix_view dataset,
                                      dataset_matrix_view queries,
                                      float eps,
                                      cuvs::distance::DistanceType metric,
                                      rmm::device_async_resource_ref mr,
                                      bool count_only,
                                      bool* screen_worthwhile)
{
  return exact_screen_applies(metric, queries.extent(0), dataset.extent(0))
           ? screened_threshold(
               res, dataset, queries, eps, metric, mr, count_only, screen_worthwhile)
           : brute_force_threshold(res, dataset, queries, eps, metric, mr);
}

// scores[j * ldc + i] = x_i . q_j for i < rows, j < cols (int8 in, exact int32 out), on the tensor
// cores. cuBLAS finds no int8 kernel for some shapes -- e.g. 110,904 rows x 17..40 columns at
// d = 96, while 110,904 x 16 and 200,000 x 40 run -- so a NOT_SUPPORTED product is re-issued in
// column slices of at most 16, and a slice cuBLAS still refuses in two row halves (multiples of 4).
static cublasStatus_t int8_gemm_tn(cublasHandle_t handle,
                                   std::int8_t const* x,
                                   std::int8_t const* q,
                                   int32_t* scores,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t d,
                                   int ldc)
{
  int32_t const alpha = 1;
  int32_t const beta  = 0;
  auto const status   = cublasGemmEx(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   static_cast<int>(rows),
                                   static_cast<int>(cols),
                                   static_cast<int>(d),
                                   &alpha,
                                   x,
                                   CUDA_R_8I,
                                   static_cast<int>(d),
                                   q,
                                   CUDA_R_8I,
                                   static_cast<int>(d),
                                   &beta,
                                   scores,
                                   CUDA_R_32I,
                                   ldc,
                                   CUBLAS_COMPUTE_32I,
                                   CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_NOT_SUPPORTED) { return status; }
  if (cols > 16) {
    for (int64_t c0 = 0; c0 < cols; c0 += 16) {
      auto const st = int8_gemm_tn(
        handle, x, q + c0 * d, scores + c0 * ldc, rows, std::min<int64_t>(16, cols - c0), d, ldc);
      if (st != CUBLAS_STATUS_SUCCESS) { return st; }
    }
    return CUBLAS_STATUS_SUCCESS;
  }
  if (rows >= 8) {
    auto const half = rows / 8 * 4;
    auto const st   = int8_gemm_tn(handle, x, q, scores, half, cols, d, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) { return st; }
    return int8_gemm_tn(handle, x + half * d, q, scores + half, rows - half, cols, d, ldc);
  }
  return status;
}

knn_result gemm_int8_topk(raft::device_resources const& res,
                          std::int8_t const* x,
                          std::int32_t const* x_sq,
                          int64_t n,
                          std::int8_t const* q,
                          std::int32_t const* q_sq,
                          int64_t m,
                          int64_t d,
                          int64_t k,
                          bool take_sqrt,
                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 1 && k <= n, "VSS k must satisfy 1 <= k <= n_rows");
  CUDF_EXPECTS(d % 4 == 0, "gemm_int8_topk: dimension must be a multiple of 4");
  CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "gemm_int8_topk: corpus slice too large");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const ldc    = (n + 3) / 4 * 4;
  auto const tile_rows =
    std::clamp<int64_t>(static_cast<int64_t>(gemm_search_tile_bytes() /
                                             (static_cast<std::size_t>(ldc) * sizeof(float))),
                        1,
                        std::min<int64_t>(std::max<int64_t>(m, 1), 65535));
  rmm::device_uvector<int32_t> scores(static_cast<std::size_t>(tile_rows * ldc), stream, mr);

  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  auto handle = raft::resource::get_cublas_handle(res);
  CUDF_EXPECTS(cublasSetStream(handle, stream.get()) == CUBLAS_STATUS_SUCCESS,
               "gemm_int8_topk: cublasSetStream failed");
  for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
    auto const t = std::min(tile_rows, m - q0);
    // Same TN layout as the FP32 search: scores^T[n x t] = x^T * q_tile^T, on the int8 tensor
    // cores with int32 accumulation -- every dot product exact.
    // The int8 GEMM wants its row count a multiple of 4 (NOT_SUPPORTED otherwise), so it runs over
    // ldc corpus rows: the up-to-3 past the slice are the next slice's, or the slack the lists keep
    // at their end, and their scores are masked to +inf below.
    auto const status =
      int8_gemm_tn(handle, x, q + q0 * d, scores.data(), ldc, t, d, static_cast<int>(ldc));
    CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS,
                 "gemm_int8_topk: cublasGemmEx failed with status " + std::to_string(status));
    dim3 const grid(
      static_cast<unsigned>(std::clamp<int64_t>((ldc + kBlock - 1) / kBlock, 1, 65535)),
      static_cast<unsigned>(t));
    int8_scores_kernel<<<grid, kBlock, 0, stream.get()>>>(scores.data(), n, ldc, x_sq);
    CUDF_CHECK_CUDA(stream.get());
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        reinterpret_cast<float const*>(scores.data()), t, ldc),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  }
  finish_int8_topk_kernel<<<grid_for(m * k), kBlock, 0, stream.get()>>>(
    out_d, m, k, q_sq, take_sqrt);
  CUDF_CHECK_CUDA(stream.get());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

knn_result gemm_f16_topk(raft::device_resources const& res,
                         std::uint16_t const* x,
                         float const* x_sq,
                         int64_t n,
                         std::uint16_t const* q,
                         float const* q_sq,
                         int64_t m,
                         int64_t d,
                         int64_t k,
                         bool take_sqrt,
                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 1 && k <= n, "VSS k must satisfy 1 <= k <= n_rows");
  CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "gemm_f16_topk: corpus slice too large");
  auto const stream = raft::resource::get_cuda_stream(res);
  // Rows padded to 4 floats so each score row starts 16-byte aligned for the selection.
  auto const ldc = (n + 3) / 4 * 4;
  auto const tile_rows =
    std::clamp<int64_t>(static_cast<int64_t>(gemm_search_tile_bytes() /
                                             (static_cast<std::size_t>(ldc) * sizeof(float))),
                        1,
                        std::min<int64_t>(std::max<int64_t>(m, 1), 65535));
  rmm::device_uvector<float> scores(static_cast<std::size_t>(tile_rows * ldc), stream, mr);

  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  auto handle = raft::resource::get_cublas_handle(res);
  CUDF_EXPECTS(cublasSetStream(handle, stream.get()) == CUBLAS_STATUS_SUCCESS,
               "gemm_f16_topk: cublasSetStream failed");
  float const alpha = 1.f;
  float const beta  = 0.f;
  for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
    auto const t = std::min(tile_rows, m - q0);
    // Same TN layout as the FP32 and INT8 searches: scores^T[n x t] = x^T * q_tile^T.
    auto const status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     static_cast<int>(n),
                                     static_cast<int>(t),
                                     static_cast<int>(d),
                                     &alpha,
                                     x,
                                     CUDA_R_16F,
                                     static_cast<int>(d),
                                     q + q0 * d,
                                     CUDA_R_16F,
                                     static_cast<int>(d),
                                     &beta,
                                     scores.data(),
                                     CUDA_R_32F,
                                     static_cast<int>(ldc),
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS,
                 "gemm_f16_topk: cublasGemmEx failed with status " + std::to_string(status));
    dim3 const grid(
      static_cast<unsigned>(std::clamp<int64_t>((ldc + kBlock - 1) / kBlock, 1, 65535)),
      static_cast<unsigned>(t));
    f16_scores_kernel<<<grid, kBlock, 0, stream.get()>>>(scores.data(), n, ldc, x_sq);
    CUDF_CHECK_CUDA(stream.get());
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(scores.data(), t, ldc),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  }
  finish_f16_topk_kernel<<<grid_for(m * k), kBlock, 0, stream.get()>>>(
    out_d, m, k, q_sq, take_sqrt);
  CUDF_CHECK_CUDA(stream.get());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

}  // namespace sirius::vss
