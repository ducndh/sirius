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

#include "vss/bound_gemm.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <mma.h>

#include <algorithm>
#include <cstdint>

namespace sirius::vss {

namespace {

// Block tile 128 probe rows x 128 corpus rows, 8 warps in a 2 x 4 grid of 64 x 32 warp tiles.
// The reduction dimension is staged through shared memory 64 bytes at a time; the 16-byte row
// pad keeps the fragment loads off a single bank.
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kChunk = 64;
constexpr int kPitch = kChunk + 16;
constexpr int kWarps = 8;

// Two blocks per SM: under separable compilation ptxas otherwise spends 177 registers on this
// kernel and runs one block per SM, 1.5x slower.
__global__ void __launch_bounds__(kWarps * 32, 2)
  bound_filter_int8_kernel(int8_t const* __restrict__ x,
                           int32_t const* __restrict__ x_sq,
                           int64_t n,
                           int64_t id_base,
                           int64_t const* __restrict__ id_map,
                           int8_t const* __restrict__ probe,
                           int32_t const* __restrict__ probe_sq,
                           int64_t const* __restrict__ rows,
                           int64_t m,
                           int d,
                           float const* __restrict__ bound,
                           int32_t* out_rows,
                           int64_t* out_ids,
                           float* out_d,
                           unsigned long long* count,
                           unsigned long long capacity)
{
#if __CUDA_ARCH__ >= 720
  using namespace nvcuda;
  __shared__ __align__(32) int8_t tile_a[kTileM * kPitch];
  __shared__ __align__(32) int8_t tile_b[kTileN * kPitch];
  __shared__ __align__(32) int32_t scores[kWarps][16 * 16];
  __shared__ int64_t tile_rows[kTileM];

  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const wm = warp / 4, wn = warp % 4;
  int64_t const q0 = static_cast<int64_t>(blockIdx.y) * kTileM;
  int64_t const x0 = static_cast<int64_t>(blockIdx.x) * kTileN;
  for (int r = threadIdx.x; r < kTileM; r += blockDim.x) {
    tile_rows[r] = q0 + r < m ? rows[q0 + r] : -1;
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(acc[i][j], 0);
    }
  }
  for (int k0 = 0; k0 < d; k0 += kChunk) {
    int const kc = min(kChunk, d - k0);
    for (int e = threadIdx.x; e < kTileM * (kChunk / 16); e += blockDim.x) {
      int const r = e / (kChunk / 16), c = (e % (kChunk / 16)) * 16;
      int4 a = make_int4(0, 0, 0, 0), b = make_int4(0, 0, 0, 0);
      if (tile_rows[r] >= 0 && c < kc) {
        a = *reinterpret_cast<int4 const*>(probe + tile_rows[r] * d + k0 + c);
      }
      if (x0 + r < n && c < kc) { b = *reinterpret_cast<int4 const*>(x + (x0 + r) * d + k0 + c); }
      *reinterpret_cast<int4*>(tile_a + r * kPitch + c) = a;
      *reinterpret_cast<int4*>(tile_b + r * kPitch + c) = b;
    }
    __syncthreads();
    for (int kk = 0; kk < kc; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        wmma::load_matrix_sync(fa[i], tile_a + (wm * 64 + i * 16) * kPitch + kk, kPitch);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::load_matrix_sync(fb[j], tile_b + (wn * 32 + j * 16) * kPitch + kk, kPitch);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Epilogue, one 16 x 16 fragment at a time through the warp's own scratch tile: a lane scores
  // 8 pairs, and each warp-wide batch of survivors takes one atomic.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(scores[warp], acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int const rb     = wm * 64 + i * 16;
      int64_t const xb = x0 + wn * 32 + j * 16;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        int const e = t * 32 + lane, r = rb + e / 16;
        int64_t const xj  = xb + e % 16;
        int64_t const row = tile_rows[r];
        bool keep         = false;
        float dist        = 0.f;
        if (row >= 0 && xj < n) {
          dist = static_cast<float>(probe_sq[row] + x_sq[xj] - 2 * scores[warp][e]);
          keep = dist <= bound[row];
        }
        unsigned const mask = __ballot_sync(0xffffffffu, keep);
        if (mask != 0) {
          int const leader        = __ffs(mask) - 1;
          unsigned long long base = 0;
          if (lane == leader) {
            base = atomicAdd(count, static_cast<unsigned long long>(__popc(mask)));
          }
          base = __shfl_sync(0xffffffffu, base, leader);
          if (keep) {
            auto const p = base + __popc(mask & ((1u << lane) - 1));
            if (p < capacity) {
              out_rows[p] = static_cast<int32_t>(row);
              out_ids[p]  = id_map != nullptr ? id_map[xj] : id_base + xj;
              out_d[p]    = dist;
            }
          }
        }
      }
      __syncwarp();
    }
  }
#endif
}

__global__ void merge_keys_kernel(float const* acc_d,
                                  int64_t acc_n,
                                  int64_t k,
                                  int32_t const* cand_rows,
                                  float const* cand_d,
                                  int64_t n_cand,
                                  uint64_t* keys,
                                  int64_t* order,
                                  int32_t* per_row)
{
  auto const total = acc_n + n_cand;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint64_t row;
    float d;
    if (i < acc_n) {
      row = static_cast<uint64_t>(i / k);
      d   = acc_d[i];
    } else {
      row = static_cast<uint64_t>(cand_rows[i - acc_n]);
      d   = cand_d[i - acc_n];
      atomicAdd(per_row + row, 1);
    }
    // Distances are >= 0, so their bit patterns order as unsigned integers (+inf included).
    keys[i]  = (row << 32) | __float_as_uint(fmaxf(d, 0.f));
    order[i] = i;
  }
}

__global__ void merge_take_kernel(int64_t const* sorted_order,
                                  int32_t const* cand_before,
                                  int64_t n_rows,
                                  int64_t k,
                                  float const* old_d,
                                  int64_t const* old_n,
                                  float const* cand_d,
                                  int64_t const* cand_ids,
                                  float* acc_d,
                                  int64_t* acc_n,
                                  float* bound)
{
  auto const total = n_rows * k;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r   = i / k;
    auto const src = sorted_order[i + cand_before[r]];
    float d;
    int64_t id;
    if (src < total) {
      d  = old_d[src];
      id = old_n[src];
    } else {
      d  = cand_d[src - total];
      id = cand_ids[src - total];
    }
    acc_d[i] = d;
    acc_n[i] = id;
    if (i - r * k == k - 1) { bound[r] = d; }
  }
}

__global__ void kth_bound_kernel(float const* acc_d, int64_t n_rows, int64_t k, float* bound)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < n_rows;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    bound[r] = acc_d[r * k + k - 1];
  }
}

__global__ void sqrt_kernel(float* d, int64_t n)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    d[i] = sqrtf(d[i]);
  }
}

int grid_for(int64_t n) { return static_cast<int>(std::clamp<int64_t>((n + 255) / 256, 1, 65535)); }

}  // namespace

bound_candidates::bound_candidates(int64_t capacity,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
  : rows(static_cast<std::size_t>(capacity), stream, mr),
    ids(static_cast<std::size_t>(capacity), stream, mr),
    distances(static_cast<std::size_t>(capacity), stream, mr),
    count(1, stream, mr)
{
  CUDF_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(unsigned long long), stream.value()));
}

bool bound_filter_int8_supports(int64_t dim)
{
  // 16-byte loads of whole row segments, and exact int32 distances that also convert to float
  // exactly: 4 * 128^2 * dim < 2^24.
  return dim > 0 && dim % 16 == 0 && dim <= 256;
}

void bound_filter_int8(int8_t const* x,
                       int32_t const* x_sq,
                       int64_t n,
                       int64_t id_base,
                       int64_t const* id_map,
                       int8_t const* probe,
                       int32_t const* probe_sq,
                       int64_t const* rows,
                       int64_t m,
                       int64_t dim,
                       float const* bound,
                       bound_candidates& out,
                       rmm::cuda_stream_view stream)
{
  if (n == 0 || m == 0) { return; }
  CUDF_EXPECTS(bound_filter_int8_supports(dim), "bound_filter_int8: unsupported vector width");
  CUDF_EXPECTS((m + kTileM - 1) / kTileM <= 65535, "bound_filter_int8: too many probe rows");
  dim3 const grid(static_cast<unsigned>((n + kTileN - 1) / kTileN),
                  static_cast<unsigned>((m + kTileM - 1) / kTileM));
  bound_filter_int8_kernel<<<grid, kWarps * 32, 0, stream.value()>>>(
    x,
    x_sq,
    n,
    id_base,
    id_map,
    probe,
    probe_sq,
    rows,
    m,
    static_cast<int>(dim),
    bound,
    out.rows.data(),
    out.ids.data(),
    out.distances.data(),
    out.count.data(),
    static_cast<unsigned long long>(out.capacity()));
  CUDF_CUDA_TRY(cudaGetLastError());
}

void merge_bound_candidates(float* acc_distances,
                            int64_t* acc_neighbors,
                            int64_t n_rows,
                            int64_t k,
                            bound_candidates const& candidates,
                            int64_t n_candidates,
                            float* bound,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (n_candidates == 0) { return; }
  auto const acc_n = n_rows * k;
  auto const total = acc_n + n_candidates;
  rmm::device_uvector<uint64_t> keys(total, stream, mr), keys_sorted(total, stream, mr);
  rmm::device_uvector<int64_t> order(total, stream, mr), order_sorted(total, stream, mr);
  rmm::device_uvector<int32_t> per_row(n_rows + 1, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(per_row.data(), 0, per_row.size() * sizeof(int32_t), stream.value()));
  merge_keys_kernel<<<grid_for(total), 256, 0, stream.value()>>>(acc_distances,
                                                                 acc_n,
                                                                 k,
                                                                 candidates.rows.data(),
                                                                 candidates.distances.data(),
                                                                 n_candidates,
                                                                 keys.data(),
                                                                 order.data(),
                                                                 per_row.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Radix sort is stable and the accumulator entries come first, so they win ties.
  int row_bits = 1;
  while ((int64_t{1} << row_bits) < n_rows) {
    ++row_bits;
  }
  std::size_t sort_bytes = 0, scan_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                order.data(),
                                                order_sorted.data(),
                                                total,
                                                0,
                                                32 + row_bits,
                                                stream.value()));
  rmm::device_uvector<int32_t> cand_before(n_rows + 1, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, per_row.data(), cand_before.data(), n_rows + 1, stream.value()));
  rmm::device_buffer temp(std::max(sort_bytes, scan_bytes), stream, mr);
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(temp.data(),
                                                sort_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                order.data(),
                                                order_sorted.data(),
                                                total,
                                                0,
                                                32 + row_bits,
                                                stream.value()));
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    temp.data(), scan_bytes, per_row.data(), cand_before.data(), n_rows + 1, stream.value()));

  rmm::device_uvector<float> old_d(acc_n, stream, mr);
  rmm::device_uvector<int64_t> old_n(acc_n, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    old_d.data(), acc_distances, acc_n * sizeof(float), cudaMemcpyDeviceToDevice, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(old_n.data(),
                                acc_neighbors,
                                acc_n * sizeof(int64_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()));
  merge_take_kernel<<<grid_for(acc_n), 256, 0, stream.value()>>>(order_sorted.data(),
                                                                 cand_before.data(),
                                                                 n_rows,
                                                                 k,
                                                                 old_d.data(),
                                                                 old_n.data(),
                                                                 candidates.distances.data(),
                                                                 candidates.ids.data(),
                                                                 acc_distances,
                                                                 acc_neighbors,
                                                                 bound);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void kth_distance_bound(
  float const* acc_distances, int64_t n_rows, int64_t k, float* bound, rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }
  kth_bound_kernel<<<grid_for(n_rows), 256, 0, stream.value()>>>(acc_distances, n_rows, k, bound);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void sqrt_in_place(float* d, int64_t n, rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }
  sqrt_kernel<<<grid_for(n), 256, 0, stream.value()>>>(d, n);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace sirius::vss
