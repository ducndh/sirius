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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "vss/bound_gemm_cutlass.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

namespace sirius::vss {
namespace {

// CUTLASS's multistage mainloop with the filter applied to its accumulators in registers: no
// distance reaches memory. Shared memory also holds the tile's per-row values, staged before the
// mainloop so that their loads hide under it.
template <class TB, class WS, int Stages>
struct tile_filter {
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using EpOp    = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
  // kNone: an access past the slice loads nothing and leaves shared memory as it was, so what a
  // padding row or column accumulates is arbitrary (see the NaNs below), and a width that ends
  // inside a 32-component step would multiply stale operands (bound_filter_f16_uses_cutlass).
  using Default =
    typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,
                                                cutlass::layout::RowMajor,
                                                8,
                                                cutlass::half_t,
                                                cutlass::layout::ColumnMajor,
                                                8,
                                                float,
                                                cutlass::layout::RowMajor,
                                                float,
                                                cutlass::arch::OpClassTensorOp,
                                                cutlass::arch::Sm80,
                                                TB,
                                                WS,
                                                cutlass::gemm::GemmShape<16, 8, 16>,
                                                EpOp,
                                                Swizzle,
                                                Stages,
                                                false,
                                                cutlass::arch::OpMultiplyAdd,
                                                cutlass::gemm::SharedMemoryClearOption::kNone,
                                                true /* gather the probe rows */>::GemmKernel;
  using Mma       = typename Default::Mma;
  using WarpCount = typename Mma::WarpCount;
  using half_ptr  = typename Mma::IteratorA::Element*;

  static constexpr int kThreads = 32 * WarpCount::kCount;
  static constexpr int kMI = WS::kM / 16, kNI = WS::kN / 8;
  static_assert(kNI * 4 <= 32, "the survivor mask of a lane is 32 bits");
  static_assert(Mma::FragmentC::kElements == kMI * kNI * 4, "m16n8 accumulators, 4 per lane");

  struct Shared {
    typename Mma::SharedStorage main_loop;
    int32_t rows[TB::kM];
    float qsq[TB::kM];
    float limit[TB::kM];
    float xsq[TB::kN];
  };

  struct Params {
    typename Mma::IteratorA::Params params_a;
    typename Mma::IteratorB::Params params_b;
    cutlass_filter_args args;
    bool has_slack;
    bound_slice single;
    int swizzle_log_tile;
    cutlass::gemm::GemmCoord grid_tiled;
    bound_slice const* slices;
    int64_t const* prefix;
    int n_slices;  // 0: a single-slice launch
    int walk;
  };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static __device__ void filter(Params const& p, Shared& smem)
  {
    bound_slice sl;
    int64_t m_tile, n_tile, n_stop;
    if (p.n_slices > 0) {
      int64_t const b = blockIdx.x;
      int lo = 0, hi = p.n_slices;
      while (hi - lo > 1) {
        int const mid = (lo + hi) >> 1;
        if (p.prefix[mid] <= b) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      sl                  = p.slices[lo];
      int64_t const local = b - p.prefix[lo];
      int64_t const tiles = (sl.n + TB::kN - 1) / TB::kN;
      int64_t const walks = (tiles + p.walk - 1) / p.walk;
      // Corpus tiles fastest: a slice's rows stay in L2 while its probe tiles go by.
      m_tile = local / walks;
      n_tile = (local % walks) * p.walk;
      n_stop = min(tiles, n_tile + p.walk);
    } else {
      Swizzle sw;
      cutlass::gemm::GemmCoord const t = sw.get_tile_offset(p.swizzle_log_tile);
      if (p.grid_tiled.m() <= t.m() || p.grid_tiled.n() <= t.n()) { return; }
      sl     = p.single;
      m_tile = t.m();
      n_tile = t.n();
      n_stop = n_tile + 1;
    }
    int const d    = p.args.dim;
    int const m0   = static_cast<int>(m_tile) * TB::kM;
    int const tid  = threadIdx.x;
    int const warp = __shfl_sync(0xffffffffu, tid / 32, 0), lane = tid % 32;
    auto const nan = __int_as_float(0x7fc00000);
    for (int t = tid; t < TB::kM; t += kThreads) {
      int const i = m0 + t;
      if (i < sl.m) {
        auto const row = sl.rows[i];
        auto const qsq = __ldg(p.args.probe_sq + row);
        auto limit     = __ldg(p.args.bound + row);
        if (p.has_slack) {
          auto const qn = sqrtf(qsq);
          auto const x  = p.args.slack.x_max;
          limit += p.args.slack.a * qn * x + p.args.slack.b * (qn + x) * (qn + x) +
                   p.args.slack.c * (qn + x);
        }
        smem.rows[t]  = static_cast<int32_t>(row);
        smem.qsq[t]   = qsq;
        smem.limit[t] = limit;
      } else {
        // NaN fails every comparison whatever the padding row's stale accumulators hold, +inf
        // included: -inf as the limit would let that through.
        smem.rows[t]  = -1;
        smem.qsq[t]   = 0.f;
        smem.limit[t] = nan;
      }
    }
    // The iterator loads indices[row] at every access: from the tile's staged rows that is a
    // shared-memory read, not a global load each cp.async would wait on.
    int32_t const* const gather = smem.rows - m0;
    auto const* xsq_rows        = static_cast<float const*>(sl.x_sq);
    bool first_tile             = true;
    for (; n_tile < n_stop; ++n_tile, first_tile = false) {
      int const n0 = static_cast<int>(n_tile) * TB::kN;
      // Every warp is done with the previous tile's norms and mainloop buffers.
      if (!first_tile) { __syncthreads(); }
      for (int t = tid; t < TB::kN; t += kThreads) {
        int const j = n0 + t;
        smem.xsq[t] = j < sl.n ? __ldg(xsq_rows + j) : nan;
      }
      __syncthreads();
      typename Mma::IteratorA it_a(p.params_a,
                                   static_cast<half_ptr>(const_cast<void*>(p.args.probe)),
                                   {static_cast<int>(sl.m), d},
                                   tid,
                                   {m0, 0},
                                   gather);
      typename Mma::IteratorB it_b(p.params_b,
                                   static_cast<half_ptr>(const_cast<void*>(sl.x)),
                                   {d, static_cast<int>(sl.n)},
                                   tid,
                                   {0, n0});
      Mma mma(smem.main_loop, tid, warp, lane);
      typename Mma::FragmentC acc;
      acc.clear();
      mma((d + TB::kK - 1) / TB::kK, acc, it_a, it_b, acc);
      __syncthreads();

      // The sm_80 tensor-op fragment holds instruction tile (m, n) of the warp tile at elements
      // 4 (m + n kMI) + e, whose element e sits at row 16 m + lane / 4 + 8 (e / 2), column
      // 8 n + 2 (lane % 4) + (e % 2); warp w sits at (w % kM, w / kM) of WarpCount
      // (gemm/warp/mma_tensor_op.h, threadblock/mma_multistage.h).
      int const wmn = warp % (WarpCount::kM * WarpCount::kN);
      int const wr = (wmn % WarpCount::kM) * WS::kM, wc = (wmn / WarpCount::kM) * WS::kN;
      int const g = lane >> 2, q = lane & 3;
      float xs[kNI][2];
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kNI; ++n) {
        xs[n][0] = smem.xsq[wc + n * 8 + 2 * q];
        xs[n][1] = smem.xsq[wc + n * 8 + 2 * q + 1];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kMI; ++m) {
        int const r0        = wr + m * 16 + g;
        float const qs[2]   = {smem.qsq[r0], smem.qsq[r0 + 8]};
        float const lims[2] = {smem.limit[r0], smem.limit[r0 + 8]};
        unsigned keep       = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kNI; ++n) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < 4; ++e) {
            float const dist = qs[e >> 1] + xs[n][e & 1] - 2.f * acc[(m + n * kMI) * 4 + e];
            keep |= dist <= lims[e >> 1] ? (1u << (n * 4 + e)) : 0u;
          }
        }
        if (!__any_sync(0xffffffffu, keep != 0)) { continue; }
        // One atomic for the warp: an inclusive scan of the lanes' survivor counts places each.
        int const own = __popc(keep);
        int before    = own;
        CUTLASS_PRAGMA_UNROLL
        for (int o = 1; o < 32; o <<= 1) {
          int const v = __shfl_up_sync(0xffffffffu, before, o);
          if (lane >= o) { before += v; }
        }
        unsigned long long base = 0;
        if (lane == 31) { base = atomicAdd(p.args.count, static_cast<unsigned long long>(before)); }
        auto pos =
          __shfl_sync(0xffffffffu, base, 31) + static_cast<unsigned long long>(before - own);
        if (keep == 0) { continue; }
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kNI; ++n) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < 4; ++e) {
            if (((keep >> (n * 4 + e)) & 1u) == 0) { continue; }
            if (pos < p.args.capacity) {
              int64_t const j      = n0 + wc + n * 8 + 2 * q + (e & 1);
              p.args.out_rows[pos] = smem.rows[r0 + 8 * (e >> 1)];
              p.args.out_ids[pos]  = sl.id_map != nullptr ? sl.id_map[j] : sl.id_base + j;
              p.args.out_d[pos]    = qs[e >> 1] + xs[n][e & 1] - 2.f * acc[(m + n * kMI) * 4 + e];
            }
            ++pos;
          }
        }
      }
    }
  }
#endif
};

template <class K>
__global__ void __launch_bounds__(K::kThreads, 1) tile_filter_kernel(typename K::Params p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(16) unsigned char tile_filter_smem[];
  K::filter(p, *reinterpret_cast<typename K::Shared*>(tile_filter_smem));
#endif
}

using filter_128x256 = tile_filter<cutlass::gemm::GemmShape<kCutlassTileM, kCutlassTileN, 32>,
                                   cutlass::gemm::GemmShape<64, 64, 32>,
                                   3>;
static_assert(filter_128x256::kThreads == kCutlassThreads);

template <class K>
int shared_bytes()
{
  static int const bytes = [] {
    auto const size = static_cast<int>(sizeof(typename K::Shared));
    if (size >= 48 * 1024) {
      CUDF_CUDA_TRY(cudaFuncSetAttribute(
        tile_filter_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
    }
    return size;
  }();
  return bytes;
}

template <class K>
typename K::Params base_params(cutlass_filter_args const& a)
{
  typename K::Params p{};
  p.params_a  = typename K::Mma::IteratorA::Params(cutlass::layout::RowMajor(a.dim));
  p.params_b  = typename K::Mma::IteratorB::Params(cutlass::layout::ColumnMajor(a.dim));
  p.args      = a;
  p.has_slack = a.slack.a != 0.f || a.slack.b != 0.f || a.slack.c != 0.f;
  return p;
}

}  // namespace

void bound_filter_f16_cutlass(bound_slice const& s,
                              cutlass_filter_args const& a,
                              rmm::cuda_stream_view stream)
{
  using K         = filter_128x256;
  auto const smem = shared_bytes<K>();
  // A grid axis past the first holds at most 65535 entries.
  constexpr int64_t kChunk = int64_t{1} << 18;
  for (int64_t c0 = 0; c0 < s.n; c0 += kChunk) {
    auto p           = base_params<K>(a);
    p.single         = s;
    p.single.x       = static_cast<std::uint16_t const*>(s.x) + c0 * a.dim;
    p.single.x_sq    = static_cast<float const*>(s.x_sq) + c0;
    p.single.n       = std::min(kChunk, s.n - c0);
    p.single.id_base = s.id_base + c0;
    p.single.id_map  = s.id_map != nullptr ? s.id_map + c0 : nullptr;
    typename K::Swizzle sw;
    cutlass::gemm::GemmCoord const problem(
      static_cast<int>(s.m), static_cast<int>(p.single.n), a.dim);
    p.grid_tiled       = sw.get_tiled_shape(problem, {kCutlassTileM, kCutlassTileN, 32}, 1);
    p.swizzle_log_tile = sw.get_log_tile(p.grid_tiled);
    tile_filter_kernel<K><<<sw.get_grid_shape(p.grid_tiled), K::kThreads, smem, stream.get()>>>(p);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

void bound_filter_f16_cutlass_group(bound_slice const* slices,
                                    int64_t const* prefix,
                                    int n_slices,
                                    unsigned blocks,
                                    int walk,
                                    cutlass_filter_args const& a,
                                    rmm::cuda_stream_view stream)
{
  using K         = filter_128x256;
  auto const smem = shared_bytes<K>();
  auto p          = base_params<K>(a);
  p.slices        = slices;
  p.prefix        = prefix;
  p.n_slices      = n_slices;
  p.walk          = walk;
  tile_filter_kernel<K><<<blocks, K::kThreads, smem, stream.get()>>>(p);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace sirius::vss
