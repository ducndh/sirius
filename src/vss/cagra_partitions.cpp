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

#include "vss/cagra_partitions.hpp"

#include "cuda/vss/cudf_raft_interop.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/fit_defaults.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_clustering.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sirius::vss {

namespace {

double seconds_since(std::chrono::steady_clock::time_point start)
{ return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }

double gib(std::size_t bytes) { return static_cast<double>(bytes) / 1073741824.0; }

std::string cache_key(cagra_index_target const& t)
{ return build_ann_index_cache_key(t.catalog, t.schema, t.table, t.column, "cagra"); }

/// Rows the sample that fits the centres and offsets takes: enough for a boundary to be placed to
/// about a percent, a small part of any table this path is for.
constexpr std::int64_t partition_sample_rows = 65536;

/// A device reservation of @p bytes, attached to @p stream for its life, or an error with the
/// numbers. On the calling thread, which must hold no other.
class scoped_stream_reservation {
 public:
  scoped_stream_reservation(std::string const& what,
                            cuvs_index_cache& cache,
                            cucascade::memory::memory_space& space,
                            std::size_t bytes,
                            rmm::cuda_stream_view stream)
    : _stream(stream)
  {
    auto reservation = cache.reserve_index_memory(bytes, space.get_device_id());
    if (!reservation) {
      throw duckdb::InvalidInputException(
        std::format("{}: could not reserve {:.2f} GiB of device memory; {:.2f} GiB is free of "
                    "{:.2f} GiB",
                    what,
                    gib(bytes),
                    gib(space.get_available_memory()),
                    gib(space.get_max_memory())));
    }
    _allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    if (_allocator == nullptr ||
        !_allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
      throw duckdb::InvalidInputException(what + ": failed to bind to its reservation");
    }
  }
  scoped_stream_reservation(scoped_stream_reservation const&)            = delete;
  scoped_stream_reservation& operator=(scoped_stream_reservation const&) = delete;
  ~scoped_stream_reservation() { _allocator->reset_stream_reservation(_stream); }

 private:
  rmm::cuda_stream_view _stream;
  cucascade::memory::reservation_aware_resource_adaptor* _allocator{nullptr};
};

void download(void* to, void const* from, std::size_t bytes, rmm::cuda_stream_view stream)
{
  CUDF_CUDA_TRY(cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
}

void upload(void* to, void const* from, std::size_t bytes, rmm::cuda_stream_view stream)
{
  CUDF_CUDA_TRY(cudaMemcpyAsync(to, from, bytes, cudaMemcpyHostToDevice, stream.get()));
  stream.sync();
}

}  // namespace

void cagra_partitioned_index::search(void const*,
                                     std::int64_t,
                                     std::int64_t,
                                     cagra_search_settings const&,
                                     std::uint32_t*,
                                     float*,
                                     rmm::cuda_stream_view) const
{ throw std::logic_error("[vss] a partitioned CAGRA index is searched through its partitions"); }

void cagra_partitioned_index::copy_graph_to_host(std::uint32_t*, rmm::cuda_stream_view) const
{ throw std::logic_error("[vss] a partitioned CAGRA index has a graph per partition"); }

std::vector<float> tune_partition_offsets(std::vector<float> const& distances,
                                          std::int64_t m,
                                          std::int64_t partitions)
{
  auto const P = static_cast<std::size_t>(partitions);
  std::vector<float> offsets(P, 0.f);
  auto const target = static_cast<double>(m) / static_cast<double>(partitions);
  double mean = 0, var = 0;
  for (auto const v : distances) {
    mean += v;
  }
  mean /= static_cast<double>(distances.size());
  for (auto const v : distances) {
    var += (v - mean) * (v - mean);
  }
  double step = 0.5 * std::sqrt(var / static_cast<double>(distances.size())) + 1e-3;
  std::vector<std::int64_t> counts(P);
  for (int round = 0; round < 400; ++round) {
    std::fill(counts.begin(), counts.end(), 0);
    for (std::int64_t r = 0; r < m; ++r) {
      std::size_t best = 0;
      float best_d     = std::numeric_limits<float>::infinity();
      for (std::size_t p = 0; p < P; ++p) {
        auto const d = distances[static_cast<std::size_t>(r) * P + p] + offsets[p];
        if (d < best_d) {
          best_d = d;
          best   = p;
        }
      }
      ++counts[best];
    }
    double worst = 0;
    for (auto const c : counts) {
      worst = std::max(worst, std::abs(static_cast<double>(c) - target));
    }
    if (worst <= 0.03 * target) { break; }
    double mean_offset = 0;
    for (std::size_t p = 0; p < P; ++p) {
      offsets[p] += static_cast<float>(step * (static_cast<double>(counts[p]) - target) / target);
      mean_offset += offsets[p];
    }
    mean_offset /= static_cast<double>(P);
    for (auto& o : offsets) {
      o -= static_cast<float>(mean_offset);
    }
    step *= 0.99;
  }
  return offsets;
}

std::size_t cagra_partition_peak_bytes(std::int64_t n_rows,
                                       std::int64_t dim,
                                       cagra_build_settings const& settings)
{
  auto const largest = n_rows / settings.partitions + n_rows / settings.partitions / 10 + 1;
  return cagra_resident_bytes(largest, dim, settings) +
         cagra_build_scratch_bytes(largest, dim, settings);
}

std::size_t cagra_partition_host_bytes(std::int64_t n_rows,
                                       std::int64_t dim,
                                       cagra_build_settings const& settings)
{
  // Rows, graph and ids of every partition.
  return cagra_resident_bytes(n_rows, dim, settings) +
         static_cast<std::size_t>(n_rows) * sizeof(std::int64_t);
}

std::shared_ptr<const pinned_index_entry> build_partitioned_cagra_index_for_pin(
  std::string const& fn,
  cuvs_index_cache& cache,
  cagra_index_target const& target,
  std::shared_ptr<const scan_manager::pinned_entry> const& pin,
  cagra_build_settings const& settings,
  cucascade::memory::memory_space& space,
  cucascade::memory::memory_space const& host_space,
  telemetry::batch_telemetry_info const& telemetry_info,
  cagra_build_report& report)
{
  auto const n_rows = static_cast<std::int64_t>(pin->num_rows);
  auto const dim    = target.dim;
  auto const P      = settings.partitions;
  if (n_rows / P <= settings.intermediate_graph_degree) {
    throw duckdb::InvalidInputException(
      std::format("{}: {} partitions of '{}.{}' need more rows each than the intermediate graph "
                  "degree: the table has {} rows and intermediate_graph_degree is {}",
                  fn,
                  P,
                  target.table,
                  target.column,
                  n_rows,
                  settings.intermediate_graph_degree));
  }
  cache.erase(cache_key(target));
  auto host_charge = const_cast<cucascade::memory::memory_space&>(host_space)
                       .make_reservation_or_null(cagra_partition_host_bytes(n_rows, dim, settings));
  if (!host_charge) {
    throw duckdb::InvalidInputException(std::format(
      "{}: {} partitions of '{}.{}' need {:.2f} GiB in the host pool, and {:.2f} GiB is free",
      fn,
      P,
      target.table,
      target.column,
      gib(cagra_partition_host_bytes(n_rows, dim, settings)),
      gib(host_space.get_available_memory())));
  }

  rmm::cuda_stream stream_owner;
  rmm::cuda_stream_view const stream = stream_owner;
  auto const mr                      = space.get_default_allocator();
  auto const stride                  = cagra_row_stride_bytes(dim, settings.row_type);
  auto index = std::make_unique<cagra_partitioned_index>(n_rows, dim, settings, true);
  index->partitions.resize(static_cast<std::size_t>(P));

  auto const convert_start = std::chrono::steady_clock::now();
  auto const map           = map_pinned_rows(*pin, target.column, dim, space, stream, mr);
  if (!map) {
    throw duckdb::InvalidInputException(
      std::format("{}: the pin of '{}.{}' cannot be read a row at a time (compressed, or not plain "
                  "rows); pin it uncompressed",
                  fn,
                  target.table,
                  target.column));
  }
  std::int64_t const sample = std::min(partition_sample_rows, n_rows);
  auto const sample_rows    = gather_pinned_rows(
    map->view(), 0, std::max<std::int64_t>(n_rows / sample, 1), sample, dim, stream, mr);
  clustering_spec spec;
  spec.n_clusters = P;
  spec.train_rows = sample;
  spec.metric     = cuvs::distance::DistanceType::L2SqrtExpanded;
  spec.n_iters    = default_fit_iterations;
  auto const centers_col =
    train_centroids(std::vector<cudf::column_view>{sample_rows->view()}, dim, spec, stream, mr);
  auto const centers_view = list_column_as_dataset_view(centers_col->view(), dim);
  index->centers.resize(static_cast<std::size_t>(P * dim));
  download(index->centers.data(),
           centers_view.data_handle(),
           index->centers.size() * sizeof(float),
           stream);
  {
    auto const sample_view = list_column_as_dataset_view(sample_rows->view(), dim);
    rmm::device_uvector<float> sample_d(static_cast<std::size_t>(sample * P), stream, mr);
    rmm::device_uvector<float> zero(static_cast<std::size_t>(P), stream, mr);
    std::vector<float> const zeros(static_cast<std::size_t>(P), 0.f);
    upload(zero.data(), zeros.data(), zeros.size() * sizeof(float), stream);
    cagra_partition_distances(sample_view.data_handle(),
                              sample,
                              dim,
                              centers_view.data_handle(),
                              zero.data(),
                              P,
                              sample_d.data(),
                              stream);
    std::vector<float> host_d(sample_d.size());
    download(host_d.data(), sample_d.data(), host_d.size() * sizeof(float), stream);
    index->offsets = tune_partition_offsets(host_d, sample, P);
  }
  rmm::device_uvector<float> device_offsets(static_cast<std::size_t>(P), stream, mr);
  upload(
    device_offsets.data(), index->offsets.data(), index->offsets.size() * sizeof(float), stream);

  // Rows to their partitions, on the host: labelled and converted on the device a chunk at a time.
  auto const expected = static_cast<std::size_t>(n_rows / P + n_rows / P / 10 + 1);
  for (auto& part : index->partitions) {
    part.rows.reserve(expected * stride);
    part.ids.reserve(expected);
  }
  row_norm_range norms;
  {
    pinned_chunk_walk walk(*pin, target.column, space, stream, telemetry_info);
    std::int64_t at = 0;
    for (std::size_t i = 0; i < walk.size(); ++i) {
      auto const chunk = walk.take(i);
      auto const view  = list_column_as_dataset_view(chunk.view, dim);
      auto const m     = view.extent(0);
      rmm::device_buffer narrowed(static_cast<std::size_t>(m) * stride, stream, mr);
      if (stride != static_cast<std::size_t>(dim) * cagra_value_bytes(settings.row_type)) {
        CUDF_CUDA_TRY(cudaMemsetAsync(narrowed.data(), 0, narrowed.size(), stream.get()));
      }
      cagra_convert_rows(
        view.data_handle(), m, dim, settings.row_type, narrowed.data(), stride, norms, stream, mr);
      rmm::device_uvector<float> distances(static_cast<std::size_t>(m * P), stream, mr);
      cagra_partition_distances(view.data_handle(),
                                m,
                                dim,
                                centers_view.data_handle(),
                                device_offsets.data(),
                                P,
                                distances.data(),
                                stream);
      std::vector<std::byte> host_rows(narrowed.size());
      std::vector<float> host_d(distances.size());
      download(host_rows.data(), narrowed.data(), host_rows.size(), stream);
      download(host_d.data(), distances.data(), host_d.size() * sizeof(float), stream);
      for (std::int64_t r = 0; r < m; ++r) {
        auto const* d    = host_d.data() + r * P;
        auto const p     = static_cast<std::size_t>(std::min_element(d, d + P) - d);
        auto& part       = index->partitions[p];
        auto const* from = host_rows.data() + static_cast<std::size_t>(r) * stride;
        part.rows.insert(part.rows.end(), from, from + stride);
        part.ids.push_back(at + r);
        ++part.n_rows;
      }
      walk.prefetch_after(i);
      stream.sync();
      at += m;
    }
    if (at != n_rows) {
      throw std::runtime_error(
        std::format("{}: the pin's chunks hold {} rows of {}", fn, at, n_rows));
    }
  }
  if (!(norms.max_sq < std::numeric_limits<float>::infinity())) {
    throw duckdb::InvalidInputException(
      std::format("{}: '{}.{}' holds a value past float16's range, a NaN or an infinity",
                  fn,
                  target.table,
                  target.column));
  }
  index->set_rows_unit(norms_are_unit(norms));
  index->host_charge     = std::move(host_charge);
  report.convert_seconds = seconds_since(convert_start);

  // A graph per partition, each partition on the device only while its graph is built.
  auto const build_start = std::chrono::steady_clock::now();
  std::size_t peak       = 0;
  for (auto& part : index->partitions) {
    if (part.n_rows <= settings.intermediate_graph_degree) {
      throw duckdb::InvalidInputException(
        std::format("{}: a partition of '{}.{}' holds {} rows, not more than the intermediate "
                    "graph degree; use fewer partitions",
                    fn,
                    target.table,
                    target.column,
                    part.n_rows));
    }
    auto const need = cagra_resident_bytes(part.n_rows, dim, settings) +
                      cagra_build_scratch_bytes(part.n_rows, dim, settings);
    peak            = std::max(peak, need);
    rmm::cuda_stream part_stream;
    rmm::cuda_stream_view const view = part_stream;
    scoped_stream_reservation const charge(
      std::format("{}: a partition of '{}.{}'", fn, target.table, target.column),
      cache,
      space,
      need,
      view);
    auto rows        = std::make_unique<rmm::device_buffer>(part.rows.size(), view, mr);
    auto const moved = std::chrono::steady_clock::now();
    upload(rows->data(), part.rows.data(), part.rows.size(), view);
    report.move_seconds += seconds_since(moved);
    std::unique_ptr<cagra_index> built;
    try {
      built = build_cagra_index(
        std::move(rows), part.n_rows, dim, stride, settings, index->rows_unit(), view);
    } catch (rmm::out_of_memory const& oom) {
      throw duckdb::InvalidInputException(
        std::format("{}: ran out of device memory building a partition of '{}.{}', {} rows x "
                    "{}-d, with {:.2f} GiB reserved for it: {}",
                    fn,
                    target.table,
                    target.column,
                    part.n_rows,
                    dim,
                    gib(need),
                    oom.what()));
    }
    part.graph.resize(static_cast<std::size_t>(part.n_rows * settings.graph_degree));
    auto const back = std::chrono::steady_clock::now();
    built->copy_graph_to_host(part.graph.data(), view);
    report.move_seconds += seconds_since(back);
    built.reset();
    view.sync();
  }
  report.build_seconds  = seconds_since(build_start);
  report.peak_bytes     = peak;
  report.resident_bytes = 0;

  index_metadata meta;
  meta.kind           = index_kind::cagra;
  meta.catalog_name   = target.catalog;
  meta.schema_name    = target.schema;
  meta.table_name     = target.table;
  meta.table_identity = target.identity;
  meta.pin_snapshot   = pin->snapshot_identity;
  meta.column_name    = target.column;
  meta.dim            = dim;
  meta.num_rows       = n_rows;
  meta.metric         = cuvs::distance::DistanceType::L2SqrtExpanded;
  meta.resident_bytes = 0;
  auto key            = cache_key(target);
  cache.insert(key, std::move(meta), std::move(index), std::move(stream_owner));
  return cache.find(key);
}

cagra_partition_answers cagra_search_partitions(cagra_partitioned_index const& index,
                                                cagra_probe_reader const& read_probe,
                                                std::size_t n_chunks,
                                                std::int64_t n_probe_rows,
                                                std::int64_t k,
                                                cagra_search_settings const& search,
                                                std::int64_t n_probes,
                                                bool cosine,
                                                cuvs_index_cache& cache,
                                                cucascade::memory::memory_space& space)
{
  auto const settings    = index.settings();
  auto const P           = settings.partitions;
  auto const dim         = index.dim();
  auto const value_bytes = cagra_value_bytes(settings.row_type);
  auto const row_bytes   = static_cast<std::size_t>(dim) * value_bytes;
  auto const stride      = cagra_row_stride_bytes(dim, settings.row_type);
  auto const mr          = space.get_default_allocator();

  cagra_partition_answers answers;
  answers.n_probes = n_probes;
  answers.k        = k;
  auto const pairs = static_cast<std::size_t>(n_probe_rows * n_probes * k);
  answers.ids.assign(pairs, -1);
  answers.distances.assign(pairs, std::numeric_limits<float>::infinity());
  answers.chunk_first.reserve(n_chunks + 1);

  // Route: each probe row's nearest partitions, and its rows in the index's type, on the host.
  auto const route_start = std::chrono::steady_clock::now();
  std::vector<std::byte> probe_rows(static_cast<std::size_t>(n_probe_rows) * row_bytes);
  std::vector<std::vector<std::int64_t>> who(static_cast<std::size_t>(P));  // probe row x rank
  {
    rmm::cuda_stream route_stream;
    rmm::cuda_stream_view const stream = route_stream;
    rmm::device_uvector<float> centers(index.centers.size(), stream, mr);
    rmm::device_uvector<float> offsets(index.offsets.size(), stream, mr);
    upload(centers.data(), index.centers.data(), index.centers.size() * sizeof(float), stream);
    upload(offsets.data(), index.offsets.data(), index.offsets.size() * sizeof(float), stream);
    std::int64_t at = 0;
    std::vector<std::int64_t> order(static_cast<std::size_t>(P));
    for (std::size_t c = 0; c < n_chunks; ++c) {
      answers.chunk_first.push_back(at);
      read_probe(c, [&](float const* rows, std::int64_t m) {
        if (m == 0) { return; }
        rmm::device_buffer narrowed(static_cast<std::size_t>(m) * row_bytes, stream, mr);
        rmm::device_uvector<float> distances(static_cast<std::size_t>(m * P), stream, mr);
        row_norm_range norms;
        cagra_convert_rows(
          rows, m, dim, settings.row_type, narrowed.data(), row_bytes, norms, stream, mr);
        if (!(norms.max_sq < std::numeric_limits<float>::infinity())) {
          throw duckdb::InvalidInputException(
            "sirius_knn_join: a probe row holds a value past float16's range, a NaN or an "
            "infinity; row_type => 'float32' keeps values float16 cannot hold");
        }
        if (cosine && !norms_are_unit(norms)) {
          throw duckdb::InvalidInputException(
            "sirius_knn_join: a CAGRA join under metric => 'cosine' needs unit rows, and a probe "
            "row is not one; normalize them, or join under metric => 'l2'");
        }
        cagra_partition_distances(
          rows, m, dim, centers.data(), offsets.data(), P, distances.data(), stream);
        std::vector<float> host_d(distances.size());
        download(probe_rows.data() + static_cast<std::size_t>(at) * row_bytes,
                 narrowed.data(),
                 narrowed.size(),
                 stream);
        download(host_d.data(), distances.data(), host_d.size() * sizeof(float), stream);
        for (std::int64_t r = 0; r < m; ++r) {
          auto const* d = host_d.data() + r * P;
          std::iota(order.begin(), order.end(), std::int64_t{0});
          std::partial_sort(
            order.begin(), order.begin() + n_probes, order.end(), [d](auto a, auto b) {
              return d[a] < d[b] || (d[a] == d[b] && a < b);
            });
          for (std::int64_t rank = 0; rank < n_probes; ++rank) {
            who[static_cast<std::size_t>(order[static_cast<std::size_t>(rank)])].push_back(
              (at + r) * n_probes + rank);
          }
        }
        stream.sync();
        at += m;
      });
    }
    answers.chunk_first.push_back(at);
    if (at != n_probe_rows) {
      throw std::runtime_error(
        std::format("sirius_knn_join: the probe chunks hold {} rows of {}", at, n_probe_rows));
    }
  }
  answers.route_seconds = seconds_since(route_start);

  auto const per_row = cagra_batch_bytes_per_row(index.n_rows(), dim, k, settings, search);

  // Each partition crosses the link once, and its probes are searched while it is there.
  std::vector<std::byte> query_stage;
  for (std::int64_t p = 0; p < P; ++p) {
    auto const& part  = index.partitions[static_cast<std::size_t>(p)];
    auto const& slots = who[static_cast<std::size_t>(p)];
    if (slots.empty()) { continue; }
    auto const resident  = cagra_resident_bytes(part.n_rows, dim, settings);
    auto const available = space.get_available_memory();
    auto const room      = available > resident ? available - resident : 0;
    auto batch_rows      = static_cast<std::int64_t>(room / 2 / std::max<std::size_t>(per_row, 1));
    batch_rows =
      std::clamp<std::int64_t>(batch_rows, 1024, static_cast<std::int64_t>(slots.size()));

    rmm::cuda_stream part_stream;
    rmm::cuda_stream_view const stream = part_stream;
    scoped_stream_reservation const charge(
      "sirius_knn_join: a partition of the index",
      cache,
      space,
      resident + static_cast<std::size_t>(batch_rows) * per_row,
      stream);
    auto mark = std::chrono::steady_clock::now();
    auto rows = std::make_unique<rmm::device_buffer>(part.rows.size(), stream, mr);
    auto graph =
      std::make_unique<rmm::device_buffer>(part.graph.size() * sizeof(std::uint32_t), stream, mr);
    upload(rows->data(), part.rows.data(), part.rows.size(), stream);
    upload(graph->data(), part.graph.data(), graph->size(), stream);
    answers.move_seconds += seconds_since(mark);
    auto const on_device = make_cagra_index_over(std::move(rows),
                                                 std::move(graph),
                                                 part.n_rows,
                                                 dim,
                                                 stride,
                                                 settings,
                                                 index.rows_unit(),
                                                 stream);

    std::size_t at = 0;
    while (at < slots.size()) {
      auto const m = std::min<std::size_t>(static_cast<std::size_t>(batch_rows), slots.size() - at);
      try {
        mark = std::chrono::steady_clock::now();
        query_stage.resize(m * row_bytes);
        for (std::size_t q = 0; q < m; ++q) {
          auto const row = slots[at + q] / n_probes;
          std::memcpy(query_stage.data() + q * row_bytes,
                      probe_rows.data() + static_cast<std::size_t>(row) * row_bytes,
                      row_bytes);
        }
        rmm::device_buffer queries(query_stage.size(), stream, mr);
        upload(queries.data(), query_stage.data(), query_stage.size(), stream);
        answers.move_seconds += seconds_since(mark);

        rmm::device_uvector<std::uint32_t> found(m * static_cast<std::size_t>(k), stream, mr);
        rmm::device_uvector<float> found_d(m * static_cast<std::size_t>(k), stream, mr);
        mark = std::chrono::steady_clock::now();
        on_device->search(queries.data(),
                          static_cast<std::int64_t>(m),
                          k,
                          search,
                          found.data(),
                          found_d.data(),
                          stream);
        answers.search_seconds += seconds_since(mark);
        ++answers.batches;

        mark = std::chrono::steady_clock::now();
        std::vector<std::uint32_t> host_ids(found.size());
        std::vector<float> host_d(found_d.size());
        download(host_ids.data(), found.data(), found.size() * sizeof(std::uint32_t), stream);
        download(host_d.data(), found_d.data(), found_d.size() * sizeof(float), stream);
        for (std::size_t q = 0; q < m; ++q) {
          auto const to = static_cast<std::size_t>(slots[at + q]) * static_cast<std::size_t>(k);
          for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
            auto const id = host_ids[q * static_cast<std::size_t>(k) + j];
            if (id < static_cast<std::uint64_t>(part.n_rows)) {
              answers.ids[to + j]       = part.ids[id];
              answers.distances[to + j] = host_d[q * static_cast<std::size_t>(k) + j];
            }
          }
        }
        answers.move_seconds += seconds_since(mark);
      } catch (rmm::out_of_memory const&) {
        if (m <= 1024) { throw; }
        batch_rows = std::max<std::int64_t>(static_cast<std::int64_t>(m) / 2, 1024);
        stream.sync();
        continue;
      }
      at += m;
    }
  }
  return answers;
}

void cagra_merge_partition_answers(cagra_partition_answers const& answers,
                                   std::int64_t first,
                                   std::int64_t m,
                                   std::vector<std::int64_t>& ids)
{
  auto const k = answers.k;
  auto const p = answers.n_probes;
  ids.assign(static_cast<std::size_t>(m * k), -1);
  std::vector<std::pair<float, std::int64_t>> row;
  row.reserve(static_cast<std::size_t>(k * p));
  for (std::int64_t r = 0; r < m; ++r) {
    row.clear();
    auto const base = static_cast<std::size_t>((first + r) * p * k);
    for (std::size_t j = 0; j < static_cast<std::size_t>(p * k); ++j) {
      if (answers.ids[base + j] >= 0) {
        row.emplace_back(answers.distances[base + j], answers.ids[base + j]);
      }
    }
    std::sort(row.begin(), row.end());
    row.erase(
      std::unique(
        row.begin(), row.end(), [](auto const& a, auto const& b) { return a.second == b.second; }),
      row.end());
    auto const kept = std::min<std::size_t>(row.size(), static_cast<std::size_t>(k));
    for (std::size_t j = 0; j < kept; ++j) {
      ids[static_cast<std::size_t>(r * k) + j] = row[j].second;
    }
  }
}

}  // namespace sirius::vss
