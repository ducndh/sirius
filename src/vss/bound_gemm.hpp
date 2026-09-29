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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

// A GEMM whose epilogue keeps only the pairs under a per-row bound. A ranked search that writes
// every score and then selects from them moves m x n x 12 bytes for m x k answers; with a bound
// that already sits near each row's k-th distance, almost every score is rejected inside the SM
// and the search becomes bound by the tensor cores instead of by memory.
namespace sirius::vss {

/// (probe row, corpus id, squared distance) triples appended by the bound-filtered search.
struct bound_candidates {
  bound_candidates(int64_t capacity,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr);

  rmm::device_uvector<int32_t> rows;
  rmm::device_uvector<int64_t> ids;
  rmm::device_uvector<float> distances;
  /// Pairs that passed, including any that did not fit: count > capacity() means the buffer
  /// overflowed and the triples past capacity() were dropped.
  rmm::device_uvector<unsigned long long> count;

  int64_t capacity() const { return static_cast<int64_t>(rows.size()); }
};

/// True when bound_filter_int8 can search vectors of this width.
bool bound_filter_int8_supports(int64_t dim);

/**
 * @brief Append every pair (rows[i], j) whose squared L2 distance is <= bound[rows[i]].
 *
 * @p x is the slice, @p n rows of shifted int8 (value - 128) with int32 squared norms @p x_sq;
 * @p probe and @p probe_sq are the whole probe side in the same encoding, read through @p rows
 * (m probe rows). Dots are exact int32, so the distances are exact. A passing pair is appended
 * as (rows[i], id, distance) with id = id_map[j] when @p id_map is given, else id_base + j.
 */
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
                       rmm::cuda_stream_view stream);

/// True when bound_filter_f16 can search vectors of this width.
bool bound_filter_f16_supports(int64_t dim);

/// The per-row slack a bounded FP16 search adds to its bound; see bound_filter_f16.
struct float16_slack {
  float a{0}, b{0}, c{0}, x_max{0};
};

/**
 * @brief bound_filter_int8 for IEEE half vectors, with FP32 norms @p x_sq / @p probe_sq of the
 * half-rounded rows.
 *
 * A pair passes when its distance, computed from the half rows, is <= bound[row] plus
 * slack.a |q| x_max + slack.b (|q| + x_max)^2 + slack.c (|q| + x_max). Callers pass a bound that
 * already covers the rounding (float16_bound_limit) and a zero slack; what passes are the
 * candidates exact_distances re-scores. Ids are id_base + j (layout rows).
 */
void bound_filter_f16(std::uint16_t const* x,
                      float const* x_sq,
                      int64_t n,
                      int64_t id_base,
                      std::uint16_t const* probe,
                      float const* probe_sq,
                      int64_t const* rows,
                      int64_t m,
                      int64_t dim,
                      float const* bound,
                      float16_slack const& slack,
                      bound_candidates& out,
                      rmm::cuda_stream_view stream);

/**
 * @brief distances[i] = |probe[r] - x[ids[i]]|^2 in FP32, r = rows[i] (or i / k when @p rows is
 * null), with x read from pinned host blocks of @p rows_per_block rows (device pointers in
 * @p blocks). Pairs with ids[i] < 0 are left alone.
 * With @p certain_below, a pair whose current distance is <= certain_below[its row] -- a filter
 * distance already proven to be within the caller's limit -- is not re-scored but set to
 * @p certain_value.
 */
void exact_distances(float const* probe,
                     int32_t const* rows,
                     int64_t k,
                     int64_t const* ids,
                     float* distances,
                     int64_t n_pairs,
                     float const* const* blocks,
                     int64_t rows_per_block,
                     int64_t dim,
                     rmm::cuda_stream_view stream,
                     float const* certain_below = nullptr,
                     float certain_value        = 0.f);

/// bound[r] = the largest of row r's k distances (+inf when any is a miss).
void row_max_bound(float const* acc_distances,
                   int64_t n_rows,
                   int64_t k,
                   float* bound,
                   rmm::cuda_stream_view stream);

/// Every one of the n entries becomes a miss: id -1 at +inf.
void fill_misses(float* distances, int64_t* ids, int64_t n, rmm::cuda_stream_view stream);

/// ids[i] = id_map[ids[i]] for every ids[i] >= 0.
void map_ids(int64_t* ids, int64_t n, int64_t const* id_map, rmm::cuda_stream_view stream);

/**
 * @brief Merge the first @p n_candidates of @p candidates into a nearest-first [n_rows x k]
 * accumulator and set bound[r] to row r's new k-th distance.
 *
 * On equal distances an accumulator entry ranks before a candidate, the order fold_topk_rows
 * gives. A candidate must not repeat a pair already in the accumulator.
 */
void merge_bound_candidates(float* acc_distances,
                            int64_t* acc_neighbors,
                            int64_t n_rows,
                            int64_t k,
                            bound_candidates const& candidates,
                            int64_t n_candidates,
                            float* bound,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/// bound[r] = acc_distances[r * k + k - 1].
void kth_distance_bound(float const* acc_distances,
                        int64_t n_rows,
                        int64_t k,
                        float* bound,
                        rmm::cuda_stream_view stream);

/// d[i] = sqrt(d[i]) for i in [0, n).
void sqrt_in_place(float* d, int64_t n, rmm::cuda_stream_view stream);

/// bound[i] = value for i in [0, n): the fixed radius a threshold join searches with.
void fill_bound(float* bound, int64_t n, float value, rmm::cuda_stream_view stream);

/// The pairs a radius join keeps from a candidate buffer, in buffer order.
struct radius_pairs {
  rmm::device_uvector<int32_t> rows;
  rmm::device_uvector<int64_t> ids;
  rmm::device_uvector<float> distances;
};

/**
 * @brief The first @p n_candidates of @p candidates whose distance is <= @p max_distance, with
 * ids mapped through @p id_map when it is given and distances square-rooted when @p take_sqrt.
 * Blocks until the count is known, so the result is sized exactly.
 */
radius_pairs take_within(bound_candidates const& candidates,
                         int64_t n_candidates,
                         float max_distance,
                         int64_t const* id_map,
                         bool take_sqrt,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
