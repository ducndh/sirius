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

#include "vss/bound_gemm.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

// bound_filter_f16's tile search as a CUTLASS GEMM, with the fused tile kernel's contract. Callers
// go through bound_filter_f16 and bound_filter_group, which decide when it runs.
namespace sirius::vss {

// 8 warps of 64 x 64 tiles at about 230 registers: one block per SM, so a slice launched alone
// fills the GPU only from about as many tiles as it has SMs.
inline constexpr int kCutlassTileM   = 128;
inline constexpr int kCutlassTileN   = 256;
inline constexpr int kCutlassThreads = 256;

struct cutlass_filter_args {
  void const* probe;      ///< the whole probe side, IEEE half, read through each slice's rows
  float const* probe_sq;  ///< indexed like probe
  float const* bound;     ///< indexed like probe
  float16_slack slack;
  int dim;  ///< a multiple of 32
  int32_t* out_rows;
  int64_t* out_ids;
  float* out_d;
  unsigned long long* count;
  unsigned long long capacity;
};

void bound_filter_f16_cutlass(bound_slice const& slice,
                              cutlass_filter_args const& args,
                              rmm::cuda_stream_view stream);

/// Many slices in one launch of @p blocks blocks. Block b is the (b - prefix[s])-th of slice s with
/// prefix[s] <= b < prefix[s + 1]; a slice takes ceil(m / kCutlassTileM) x
/// ceil(ceil(n / kCutlassTileN) / walk) blocks, each walking @p walk of its corpus tiles for one
/// probe tile. @p slices and @p prefix are device memory, valid until the launch completes.
void bound_filter_f16_cutlass_group(bound_slice const* slices,
                                    int64_t const* prefix,
                                    int n_slices,
                                    unsigned blocks,
                                    int walk,
                                    cutlass_filter_args const& args,
                                    rmm::cuda_stream_view stream);

}  // namespace sirius::vss
