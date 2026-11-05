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

#pragma once

#include "vss/cagra_settings.hpp"
#include "vss/cuvs_index_cache.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace sirius::vss {

/// The extremes of a set of rows' squared norms. A value past float16's range, a NaN or an infinity
/// makes the row's norm +inf.
struct row_norm_range {
  float min_sq{std::numeric_limits<float>::infinity()};
  float max_sq{0.f};
};

/// out = @p in, @p n rows @p dim wide, as @p type, in rows @p out_stride_bytes apart, and its
/// squared norms' extremes combined into @p range. @p out null only computes the norms.
void cagra_convert_rows(float const* in,
                        std::int64_t n,
                        std::int64_t dim,
                        cagra_row_type type,
                        void* out,
                        std::size_t out_stride_bytes,
                        row_norm_range& range,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

/// A CAGRA index: its graph, and the rows it walks, which it owns. Rows are in the pin's row order.
/// Everything it holds was allocated on the stream it was built on; the cache entry keeps that
/// stream for as long as the index lives.
class cagra_index : public any_cuvs_index {
 public:
  [[nodiscard]] virtual cagra_build_settings const& settings() const noexcept = 0;
  [[nodiscard]] virtual std::int64_t n_rows() const noexcept                  = 0;
  [[nodiscard]] virtual std::int64_t dim() const noexcept                     = 0;
  /// Whether every row was within unit_norm_tolerance of length 1 when the index was built.
  [[nodiscard]] virtual bool rows_unit() const noexcept = 0;

  /// The graph, [n_rows, graph_degree], copied to @p out. Synchronous.
  virtual void copy_graph_to_host(std::uint32_t* out, rmm::cuda_stream_view stream) const = 0;

  /// [m, k] neighbours of the @p m queries at @p queries, in the index's row type, @p dim tight.
  /// Ids are rows of the index, or at least n_rows() where a row found fewer than k. Synchronous.
  virtual void search(void const* queries,
                      std::int64_t m,
                      std::int64_t k,
                      cagra_search_settings const& params,
                      std::uint32_t* neighbors,
                      float* distances,
                      rmm::cuda_stream_view stream) const = 0;
};

/// How far a row's length may be from 1 and still count as a unit row. |q - x|^2 / 2 is then the
/// cosine distance to a relative 2e-5 at worst; rows normalized in FP32 sit near 1e-7.
inline constexpr float unit_norm_tolerance = 1e-5f;

[[nodiscard]] inline bool norms_are_unit(row_norm_range const& r)
{
  float const lo = (1.f - unit_norm_tolerance) * (1.f - unit_norm_tolerance);
  float const hi = (1.f + unit_norm_tolerance) * (1.f + unit_norm_tolerance);
  return r.min_sq >= lo && r.max_sq <= hi;
}

/// Build over @p rows, [n_rows, dim] of settings.row_type @p stride_bytes apart, which the index
/// takes over. The rows and the graph are allocated on @p stream through @p mr, the current device
/// resource must be @p mr's (it is Sirius's on every device), and the call is synchronous.
[[nodiscard]] std::unique_ptr<cagra_index> build_cagra_index(
  std::unique_ptr<rmm::device_buffer> rows,
  std::int64_t n_rows,
  std::int64_t dim,
  std::size_t stride_bytes,
  cagra_build_settings const& settings,
  bool rows_unit,
  rmm::cuda_stream_view stream);

/// An index over rows and a graph already built and already on the device, which it takes over.
[[nodiscard]] std::unique_ptr<cagra_index> make_cagra_index_over(
  std::unique_ptr<rmm::device_buffer> rows,
  std::unique_ptr<rmm::device_buffer> graph,
  std::int64_t n_rows,
  std::int64_t dim,
  std::size_t stride_bytes,
  cagra_build_settings const& settings,
  bool rows_unit,
  rmm::cuda_stream_view stream);

/// out[r * P + p] = |rows[r] - centers[p]|^2 + offsets[p], for @p P partitions.
void cagra_partition_distances(float const* rows,
                               std::int64_t n,
                               std::int64_t dim,
                               float const* centers,
                               float const* offsets,
                               std::int64_t P,
                               float* out,
                               rmm::cuda_stream_view stream);

/// ids[i] = neighbors[i], or -1 where it is at least @p n_rows.
void cagra_widen_ids(std::uint32_t const* neighbors,
                     std::int64_t n,
                     std::int64_t n_rows,
                     std::int64_t* ids,
                     rmm::cuda_stream_view stream);

/// How many of @p n ids are misses (below zero).
[[nodiscard]] std::int64_t cagra_count_misses(std::int64_t const* ids,
                                              std::int64_t n,
                                              rmm::cuda_stream_view stream);

/// Sorts each of @p m rows of @p k pairs by (distance, id), misses last, and turns a row's repeated
/// ids into misses. The rows arrive nearly sorted, in the order of the index's rows, so this is an
/// insertion sort, a row at a time.
void cagra_sort_rows(float* distances,
                     std::int64_t* ids,
                     std::int64_t m,
                     std::int64_t k,
                     rmm::cuda_stream_view stream);

}  // namespace sirius::vss
