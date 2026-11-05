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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sirius::vss {

enum class cagra_row_type : std::uint8_t { float16, float32 };
enum class cagra_graph_builder : std::uint8_t { nn_descent, ivf_pq, iterative_search };

/// What decides the graph and the rows an index holds. Two indexes with equal settings over the
/// same pin answer alike. Defaults, and why:
/// - degree 32 / intermediate 64: degree 16 needs far more NN-descent rounds, and on some sets
///   does not saturate by 20; 32 saturates at 8 on four of five widths and sets measured.
/// - NN-descent, 8 rounds: an index built inside a statement is paid for once, and past 8 rounds
///   it gains tenths of a recall point at any itopk. An index searched many times may ask for 20.
/// - FP16 rows: half the device memory, and the walk is limited by memory bandwidth, so 1.5-1.9x
///   faster from 256-d up and about level below; the answer's distances come from the FP32 rows.
struct cagra_build_settings {
  std::int64_t graph_degree{32};
  std::int64_t intermediate_graph_degree{64};
  cagra_graph_builder builder{cagra_graph_builder::nn_descent};
  std::int64_t nn_descent_niter{8};
  std::int64_t ivf_pq_n_lists{0};   ///< 0: cuVS's default for the table's size
  std::int64_t ivf_pq_n_probes{0};  ///< 0: cuVS's default
  cagra_row_type row_type{cagra_row_type::float16};

  [[nodiscard]] bool operator==(cagra_build_settings const&) const = default;
};

/// itopk 64 and width 1 are cuVS's own defaults: recall at a given itopk moves too much between
/// datasets for any other single value to mean a recall.
struct cagra_search_settings {
  std::int64_t itopk{64};
  std::int64_t search_width{1};
};

/// A join's CAGRA settings. With no build setting given in SQL, any index already on the column
/// answers; with one given, only an index built that way does.
struct cagra_join_settings {
  cagra_build_settings build;
  cagra_search_settings search;
  bool build_settings_given{false};
};

[[nodiscard]] inline char const* cagra_builder_name(cagra_graph_builder builder)
{
  switch (builder) {
    case cagra_graph_builder::ivf_pq: return "ivf_pq";
    case cagra_graph_builder::iterative_search: return "iterative_search";
    default: return "nn_descent";
  }
}

[[nodiscard]] inline char const* cagra_row_type_name(cagra_row_type type)
{ return type == cagra_row_type::float16 ? "float16" : "float32"; }

[[nodiscard]] inline std::size_t cagra_value_bytes(cagra_row_type type)
{ return type == cagra_row_type::float16 ? 2 : 4; }

/// cuVS requires a padded dataset's rows on 16-byte boundaries.
[[nodiscard]] inline std::size_t cagra_row_stride_bytes(std::int64_t dim, cagra_row_type type)
{
  auto const bytes = static_cast<std::size_t>(dim) * cagra_value_bytes(type);
  return (bytes + 15) / 16 * 16;
}

/// Device bytes a finished index holds: its rows and its graph.
[[nodiscard]] inline std::size_t cagra_resident_bytes(std::int64_t n_rows,
                                                      std::int64_t dim,
                                                      cagra_build_settings const& s)
{
  return static_cast<std::size_t>(n_rows) *
         (cagra_row_stride_bytes(dim, s.row_type) +
          static_cast<std::size_t>(s.graph_degree) * sizeof(std::uint32_t));
}

/// Device bytes a build needs beside the finished index. NN-descent holds a second copy of the rows
/// and per row 32 neighbour ids, 32 distances, a lock and two list sizes (measured over 1M rows at
/// 1,024-d: 4.05 GiB at the peak, 2.03 GiB resident). The other builders search the rows for their
/// graph, with scratch of the same order.
[[nodiscard]] inline std::size_t cagra_build_scratch_bytes(std::int64_t n_rows,
                                                           std::int64_t dim,
                                                           cagra_build_settings const& s)
{
  auto const n    = static_cast<std::size_t>(n_rows);
  auto const rows = n * cagra_row_stride_bytes(dim, s.row_type);
  if (s.builder == cagra_graph_builder::nn_descent) {
    return rows + n * (32 * 2 * sizeof(std::uint32_t) + 3 * sizeof(std::uint32_t)) +
           (std::size_t{256} << 20);
  }
  return rows + (std::size_t{1} << 30);
}

/// Bytes of device scratch one probe row takes inside cagra::search: the visited-node hash table
/// and, past itopk 512, the multi-CTA search's per-row result buffers. From cuVS's search plan
/// (detail/cagra/search_plan.cuh); at itopk 512 and below the walk's own state lives in shared
/// memory. itopk 1,024, width 8, degree 32 takes 16 KiB a row.
[[nodiscard]] inline std::size_t cagra_search_scratch_per_row(std::int64_t n_rows,
                                                              std::int64_t graph_degree,
                                                              std::int64_t k,
                                                              cagra_search_settings const& s)
{
  auto const hash_bits = [](std::int64_t visited, int bits) {
    while (static_cast<double>(visited) > 0.5 * static_cast<double>(std::int64_t{1} << bits)) {
      ++bits;
    }
    return bits;
  };
  auto const itopk   = (std::max(s.itopk, k) + 31) / 32 * 32;
  std::int64_t steps = 0;
  for (std::int64_t reach = 1; reach < n_rows;
       reach *= std::max<std::int64_t>(2, graph_degree / 2)) {
    ++steps;
  }
  if (itopk <= 512) {
    if (hash_bits(itopk + s.search_width * graph_degree, 8) <= 13) { return 0; }
    auto const iterations = itopk / s.search_width + steps;
    auto const visited    = itopk + s.search_width * graph_degree * iterations;
    return (std::size_t{1} << hash_bits(visited, 11)) * sizeof(std::uint32_t);
  }
  auto const per_query = std::max(s.search_width, (itopk + 31) / 32);
  auto const traversed = per_query * std::max<std::int64_t>(32, 32 + steps);
  auto const results   = static_cast<std::size_t>(per_query) * 32;
  return (std::size_t{1} << hash_bits(traversed, 11)) * sizeof(std::uint32_t) +
         results * 4 * sizeof(std::uint32_t);
}

/// Device bytes one probe row of a batch holds while it is searched and re-scored: the scratch
/// above, its rows as the index reads them, the [k] ids and distances out of the search, and the
/// sort keys of the re-score.
[[nodiscard]] inline std::size_t cagra_batch_bytes_per_row(std::int64_t n_rows,
                                                           std::int64_t dim,
                                                           std::int64_t k,
                                                           cagra_build_settings const& build,
                                                           cagra_search_settings const& search)
{
  return cagra_search_scratch_per_row(n_rows, build.graph_degree, k, search) +
         static_cast<std::size_t>(dim) * cagra_value_bytes(build.row_type) +
         static_cast<std::size_t>(k) * (2 * sizeof(std::uint32_t) + 4 * sizeof(std::int64_t));
}

}  // namespace sirius::vss
