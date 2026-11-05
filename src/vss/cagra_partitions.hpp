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

#include "cuda/vss/cagra_index.hpp"
#include "vss/cagra_join_index.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace cucascade::memory {
class memory_space;
class reservation;
}  // namespace cucascade::memory

namespace sirius::vss {

/// A CAGRA index in a few big partitions, for a table whose one graph would not fit the device: a
/// graph per partition, kept with its rows in host memory and brought to the device for one pass
/// over the probes that go there. A row goes to the partition nearest it after a per-partition
/// offset that evens the sizes out.
class cagra_partitioned_index final : public cagra_index {
 public:
  struct partition {
    std::int64_t n_rows{0};
    std::vector<std::byte> rows;       ///< settings().row_type, stride_bytes apart
    std::vector<std::uint32_t> graph;  ///< [n_rows, graph_degree]
    std::vector<std::int64_t> ids;     ///< pin row of each row
  };

  cagra_partitioned_index(std::int64_t n_rows,
                          std::int64_t dim,
                          cagra_build_settings const& settings,
                          bool rows_unit)
    : _n_rows(n_rows), _dim(dim), _settings(settings), _rows_unit(rows_unit)
  {
  }

  [[nodiscard]] cagra_build_settings const& settings() const noexcept override { return _settings; }
  [[nodiscard]] std::int64_t n_rows() const noexcept override { return _n_rows; }
  [[nodiscard]] std::int64_t dim() const noexcept override { return _dim; }
  [[nodiscard]] bool rows_unit() const noexcept override { return _rows_unit; }
  void set_rows_unit(bool unit) noexcept { _rows_unit = unit; }
  void search(void const*,
              std::int64_t,
              std::int64_t,
              cagra_search_settings const&,
              std::uint32_t*,
              float*,
              rmm::cuda_stream_view) const override;
  void copy_graph_to_host(std::uint32_t*, rmm::cuda_stream_view) const override;

  std::vector<partition> partitions;
  std::vector<float> centers;  ///< [P, dim]
  std::vector<float> offsets;  ///< [P], added to a row's squared distance to the centre
  /// Holds what the partitions take in the host pool's count for as long as the index lives.
  std::unique_ptr<cucascade::memory::reservation> host_charge;

 private:
  std::int64_t _n_rows;
  std::int64_t _dim;
  cagra_build_settings _settings;
  bool _rows_unit;
};

/// One offset per partition from a sample's squared distances [m, P]: raised while a partition
/// holds over its share, until every partition is within 3 % of it or the steps run out.
[[nodiscard]] std::vector<float> tune_partition_offsets(std::vector<float> const& distances,
                                                        std::int64_t m,
                                                        std::int64_t partitions);

/// Everything a build of @p settings.partitions partitions over @p pin needs on the device at the
/// peak of its largest partition, and in the host pool overall. The partition is taken 10 % over
/// its even share, the offsets' worst case plus what the sample leaves.
[[nodiscard]] std::size_t cagra_partition_peak_bytes(std::int64_t n_rows,
                                                     std::int64_t dim,
                                                     cagra_build_settings const& settings);
[[nodiscard]] std::size_t cagra_partition_host_bytes(std::int64_t n_rows,
                                                     std::int64_t dim,
                                                     cagra_build_settings const& settings);

/// Builds @p target's partitioned index from @p pin and replaces whatever index the column had.
/// Rows reach the device a chunk at a time to be labelled and converted; each partition then
/// crosses the link once to have its graph built and goes back. Call it on a thread that holds no
/// reservation of its own.
[[nodiscard]] std::shared_ptr<const pinned_index_entry> build_partitioned_cagra_index_for_pin(
  std::string const& fn,
  cuvs_index_cache& cache,
  cagra_index_target const& target,
  std::shared_ptr<const scan_manager::pinned_entry> const& pin,
  cagra_build_settings const& settings,
  cucascade::memory::memory_space& space,
  cucascade::memory::memory_space const& host_space,
  telemetry::batch_telemetry_info const& telemetry_info,
  cagra_build_report& report);

/// What every probe row found in its nearest partitions, on the host: [rows, n_probes, k] ids (pin
/// rows, or -1) and distances in the index's row type. The partitions were visited once each.
struct cagra_partition_answers {
  std::int64_t n_probes{0};
  std::int64_t k{0};
  std::vector<std::int64_t> chunk_first;  ///< first answer row of each probe chunk, then the total
  std::vector<std::int64_t> ids;
  std::vector<float> distances;
  double route_seconds{0};
  double move_seconds{0};
  double search_seconds{0};
  std::int64_t batches{0};
};

/// Hands fn the FP32 rows of probe chunk @p i, valid during the call.
using cagra_probe_reader = std::function<void(
  std::size_t i, std::function<void(float const* rows, std::int64_t n)> const& fn)>;

/// Route every probe row, then bring each partition to the device once and search its probes.
/// Charged to reservations of its own, so call it on a thread that holds none. @p cosine checks the
/// probes are unit rows.
[[nodiscard]] cagra_partition_answers cagra_search_partitions(
  cagra_partitioned_index const& index,
  cagra_probe_reader const& read_probe,
  std::size_t n_chunks,
  std::int64_t n_probe_rows,
  std::int64_t k,
  cagra_search_settings const& search,
  std::int64_t n_probes,
  bool cosine,
  cuvs_index_cache& cache,
  cucascade::memory::memory_space& space);

/// [m, k] ids and squared distances for answer rows [first, first + m): each row's candidates from
/// its partitions, nearest first, without repeats. Misses are id -1 at +inf.
void cagra_merge_partition_answers(cagra_partition_answers const& answers,
                                   std::int64_t first,
                                   std::int64_t m,
                                   std::vector<std::int64_t>& ids);

}  // namespace sirius::vss
