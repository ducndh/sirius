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

#include "vss/kmeans_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "telemetry/data_batch_probe.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/distance_metric.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

/// Handles every kmeans function needs before it can touch a pinned column.
struct kmeans_context {
  cucascade::memory::memory_space* space;
  const cucascade::memory::memory_space* host_space;
  const scan_manager::pinned_entry* pin;
  int target_gpu;
};

kmeans_context resolve_context(duckdb::SiriusContext& ctx,
                               const std::string& fn,
                               const std::string& catalog,
                               const std::string& schema,
                               const std::string& table)
{
  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no GPU memory space available");
  }
  auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no HOST memory space available");
  }

  const auto* pin =
    ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(catalog, schema, table);
  if (pin == nullptr) {
    throw duckdb::InvalidInputException(fn + ": table '" + table +
                                        "' must be pinned (GPU or HOST tier)");
  }

  auto* space = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  return kmeans_context{space, host_spaces.front(), pin, space->get_device_id()};
}

/// The clustering stored under @p name, or nullptr when the name is free or holds something
/// that is not a clustering.
const pinned_index_entry* find_clustering_entry(duckdb::SiriusContext& ctx, const std::string& name)
{
  const auto* entry = ctx.get_cuvs_index_cache().find(name);
  if (entry == nullptr || entry->meta.kind != index_kind::kmeans_centroids) { return nullptr; }
  return entry;
}

/// The lists live beside their clustering under a derived name, so re-fitting the clustering
/// can drop them without the cache knowing the two are related.
std::string lists_key(const std::string& clustering) { return clustering + "::lists"; }

}  // namespace

const cudf::column* find_clustering_centroids(duckdb::SiriusContext& ctx, const std::string& name)
{
  const auto* entry = find_clustering_entry(ctx, name);
  if (entry == nullptr) { return nullptr; }
  auto* held = entry->index_as<std::unique_ptr<cudf::column>>();
  return held != nullptr ? held->get() : nullptr;
}

kmeans_fit_result run_kmeans_fit(duckdb::SiriusContext& ctx, const kmeans_fit_request& req)
{
  static const std::string fn = "sirius_kmeans_fit";
  auto const c                = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{c.target_gpu}};
  // The centroids outlive this call by design -- they stay in the index cache until the session
  // ends. rmm::device_buffer records the stream it was allocated on and deallocates on that same
  // stream, so an owned stream here would be destroyed long before the cache frees the column,
  // and the free would run on a dead stream. The default stream outlives both.
  auto stream = cudf::get_default_stream();

  auto const n_rows = static_cast<std::int64_t>(c.pin->num_rows);
  if (n_rows == 0) {
    throw duckdb::InvalidInputException(fn + ": table '" + req.table + "' is empty");
  }

  auto const metric     = ann_distance_type_from_metric(req.metric);
  auto const n_clusters = resolve_n_clusters(req.spec.n_clusters, n_rows);
  auto const train_rows = resolve_train_rows(req.spec.train_rows, n_rows, n_clusters);

  // Over-reserved to cover the sample, the concatenated copy train_centroids makes of it, and
  // k-means' own scratch; shrunk to the centroids alone once the fit is done.
  auto const vec_bytes        = static_cast<std::size_t>(req.dim) * sizeof(float);
  std::size_t const footprint = static_cast<std::size_t>(train_rows) * vec_bytes * 3 +
                                static_cast<std::size_t>(n_clusters) * vec_bytes * 2 +
                                (std::size_t{1} << 20);

  auto& index_cache = ctx.get_cuvs_index_cache();
  // Free the old clustering's reservation before asking for the new one, so re-fitting under
  // the same name does not have to fit both at once.
  index_cache.erase(req.name);
  index_cache.erase(lists_key(req.name));
  auto reservation = index_cache.reserve_index_memory(footprint, c.target_gpu);
  if (!reservation) {
    throw duckdb::InvalidInputException(fn + ": not enough free GPU memory to train " +
                                        std::to_string(n_clusters) + " centroids: need ~" +
                                        std::to_string(footprint >> 20) + " MiB");
  }
  auto const mr = reservation->get_memory_resource();

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);

  std::unique_ptr<cudf::column> centroids;
  std::int64_t sampled = 0;
  {
    // Each chunk is staged, reduced to its share of the sample, and released before the next
    // is staged, so a corpus larger than device memory can still be clustered.
    std::vector<std::unique_ptr<cudf::column>> samples;
    for (std::size_t i = 0; i < n_chunks; ++i) {
      auto staged =
        stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }

      auto take = std::min<std::int64_t>(rows, ((train_rows * rows) + n_rows - 1) / n_rows);
      take      = std::min<std::int64_t>(take, train_rows - sampled);
      if (take <= 0) { break; }
      samples.push_back(sample_vector_rows(staged.view, take, req.spec.seed, stream, mr));
      // The sample borrows the staged chunk, which this iteration is about to release.
      stream.synchronize();
      sampled += take;
    }
    if (samples.empty()) {
      throw duckdb::InvalidInputException(fn + ": column '" + req.column + "' has no rows");
    }

    std::vector<cudf::column_view> views;
    views.reserve(samples.size());
    for (auto const& s : samples) {
      views.push_back(s->view());
    }

    // Both counts are already resolved against the full table; passing them through stops
    // train_centroids from re-deriving them from the sample it is handed.
    clustering_spec spec = req.spec;
    spec.n_clusters      = n_clusters;
    spec.train_rows      = sampled;
    spec.metric          = metric;
    centroids            = train_centroids(views, req.dim, spec, stream, mr);
  }
  reservation->shrink_to_fit();

  index_metadata meta;
  meta.kind           = index_kind::kmeans_centroids;
  meta.table_name     = req.table;
  meta.column_name    = req.column;
  meta.dim            = req.dim;
  meta.num_rows       = n_rows;
  meta.n_lists        = n_clusters;
  meta.metric         = metric;
  meta.reserved_bytes = reservation->size();
  index_cache.insert(
    req.name, std::move(meta), make_cuvs_index(std::move(centroids)), std::move(reservation));

  return kmeans_fit_result{n_clusters, req.dim, sampled, n_rows};
}

std::unique_ptr<cucascade::host_data_representation> run_kmeans_assign(
  duckdb::SiriusContext& ctx, const kmeans_assign_request& req)
{
  static const std::string fn = "sirius_kmeans_assign";
  auto const c                = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{c.target_gpu}};
  rmm::cuda_stream stream_owner;
  auto stream   = stream_owner.view();
  auto const mr = c.space->get_default_allocator();

  const auto* entry = find_clustering_entry(ctx, req.clustering);
  if (entry == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + req.clustering +
                                        "'; run sirius_kmeans_fit first");
  }
  if (entry->meta.dim != req.dim) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' is over FLOAT[" + std::to_string(entry->meta.dim) +
                                        "] vectors but column '" + req.column + "' is FLOAT[" +
                                        std::to_string(req.dim) + "]");
  }
  const auto* centroids = find_clustering_centroids(ctx, req.clustering);
  if (centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' holds no centroids");
  }

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);

  raft::device_resources res{stream};
  std::vector<std::unique_ptr<cudf::column>> row_ids;
  std::vector<std::unique_ptr<cudf::column>> cluster_ids;
  std::vector<std::unique_ptr<cudf::column>> distances;
  std::int64_t base = 0;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto staged =
      stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
    auto const rows = static_cast<std::int64_t>(staged.view.size());
    if (rows == 0) { continue; }

    auto assignment = assign_to_centroids(
      res, staged.view, centroids->view(), req.dim, req.spec, base, entry->meta.metric, stream, mr);
    // The assignment reads the staged chunk, which is released at the end of this iteration.
    stream.synchronize();
    row_ids.push_back(std::move(assignment.row_ids));
    cluster_ids.push_back(std::move(assignment.cluster_ids));
    distances.push_back(std::move(assignment.distances));
    base += rows;
  }
  if (row_ids.empty()) {
    throw duckdb::InvalidInputException(fn + ": column '" + req.column + "' has no rows");
  }

  auto merge = [&](std::vector<std::unique_ptr<cudf::column>> const& parts) {
    if (parts.size() == 1) {
      return std::make_unique<cudf::column>(parts.front()->view(), stream, mr);
    }
    std::vector<cudf::column_view> views;
    views.reserve(parts.size());
    for (auto const& p : parts) {
      views.push_back(p->view());
    }
    return cudf::concatenate(views, stream, mr);
  };

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(merge(row_ids));
  columns.push_back(merge(cluster_ids));
  columns.push_back(merge(distances));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  return vss_table_to_host(*c.space, *c.host_space, stream, std::move(table));
}

const cluster_lists* find_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering)
{
  const auto* entry = ctx.get_cuvs_index_cache().find(lists_key(clustering));
  if (entry == nullptr || entry->meta.kind != index_kind::cluster_lists) { return nullptr; }
  return entry->index_as<cluster_lists>();
}

void erase_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering)
{
  ctx.get_cuvs_index_cache().erase(lists_key(clustering));
}

cluster_lists_result run_kmeans_build_lists(duckdb::SiriusContext& ctx,
                                            const kmeans_assign_request& req,
                                            list_storage storage)
{
  static const std::string fn = "sirius_kmeans_build_lists";
  auto const c                = resolve_context(ctx, fn, req.catalog, req.schema, req.table);
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{c.target_gpu}};
  // The lists outlive this call in the index cache, and a device_buffer frees on the stream it
  // was allocated on, so the persistent buffers use the default stream (as the fit's centroids
  // do); per-chunk scratch uses the owned one.
  auto const persistent = cudf::get_default_stream();
  rmm::cuda_stream stream_owner;
  auto stream       = stream_owner.view();
  auto const mr     = c.space->get_default_allocator();
  auto& index_cache = ctx.get_cuvs_index_cache();

  const auto* entry     = find_clustering_entry(ctx, req.clustering);
  const auto* centroids = find_clustering_centroids(ctx, req.clustering);
  if (entry == nullptr || centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + req.clustering +
                                        "'; run sirius_kmeans_fit first");
  }
  if (entry->meta.dim != req.dim) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' is over FLOAT[" + std::to_string(entry->meta.dim) +
                                        "] vectors but column '" + req.column + "' is FLOAT[" +
                                        std::to_string(req.dim) + "]");
  }
  // Built afresh every time, so the old copy is freed before the new one is allocated.
  index_cache.erase(lists_key(req.clustering));

  auto const n_rows     = static_cast<std::int64_t>(c.pin->num_rows);
  auto const dim        = req.dim;
  auto const n_clusters = entry->meta.n_lists;
  if (n_rows == 0) {
    throw duckdb::InvalidInputException(fn + ": table '" + req.table + "' is empty");
  }

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);
  raft::device_resources res{stream};
  // Same switch as the join's phase timer; synchronizes, so only for attribution.
  auto const dbg = std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr;
  auto phase_t0  = std::chrono::steady_clock::now();
  auto phase     = [&](const char* name) {
    if (!dbg) { return; }
    stream.synchronize();
    auto const now = std::chrono::steady_clock::now();
    std::fprintf(stderr,
                 "[vecjoin-lists] %-18s %8.3f s\n",
                 name,
                 std::chrono::duration<double>(now - phase_t0).count());
    phase_t0 = now;
  };

  // Pass 1: every row's nearest centroid, to the host. 4 bytes a row, so even 100M rows is
  // 400 MB, and the host is where the counting sort below is cheapest to run deterministically.
  std::vector<std::int32_t> labels(static_cast<std::size_t>(n_rows));
  // Whether every component is a byte, counted while the chunks are resident anyway: that is
  // what decides if the lists may be stored as UINT8 without changing a single value.
  bool const check_uint8 = storage != list_storage::float32;
  rmm::device_uvector<unsigned long long> non_uint8(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(non_uint8.data(), 0, sizeof(unsigned long long), stream.value()));
  {
    std::int64_t base = 0;
    for (std::size_t i = 0; i < n_chunks; ++i) {
      auto staged =
        stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }
      if (base + rows > n_rows) {
        throw std::runtime_error(fn + ": pinned column holds more rows than the pin reports");
      }
      rmm::device_uvector<std::int32_t> chunk_labels(static_cast<std::size_t>(rows), stream, mr);
      nearest_centroid_labels(
        res, staged.view, centroids->view(), dim, entry->meta.metric, chunk_labels.data(), mr);
      if (check_uint8) {
        count_non_uint8(list_column_as_dataset_view(staged.view, dim).data_handle(),
                        rows * dim,
                        non_uint8.data(),
                        stream);
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(labels.data() + base,
                                    chunk_labels.data(),
                                    static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      stream.synchronize();
      base += rows;
    }
    if (base != n_rows) {
      throw std::runtime_error(fn + ": pinned column holds " + std::to_string(base) +
                               " rows but the pin reports " + std::to_string(n_rows));
    }
  }

  phase("pass 1 labels");

  unsigned long long non_uint8_host = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&non_uint8_host,
                                non_uint8.data(),
                                sizeof(non_uint8_host),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  if (storage == list_storage::uint8 && non_uint8_host != 0) {
    throw duckdb::InvalidInputException(fn + ": storage => 'uint8' needs every value of '" +
                                        req.column + "' to be an integer in [0, 255]; " +
                                        std::to_string(non_uint8_host) + " are not");
  }
  auto const encoding =
    check_uint8 && non_uint8_host == 0 ? list_encoding::uint8 : list_encoding::float32;

  // A stable counting sort, split over threads by row range: per-thread histograms, then each
  // thread's start inside each list is the list's start plus the rows earlier threads put
  // there. Rows keep their pin order within a list, so the result does not depend on the
  // thread count.
  auto const n_threads =
    static_cast<std::int64_t>(std::clamp<unsigned>(std::thread::hardware_concurrency(), 1u, 16u));
  auto const span = (n_rows + n_threads - 1) / n_threads;
  std::vector<std::vector<std::int64_t>> hist(
    static_cast<std::size_t>(n_threads),
    std::vector<std::int64_t>(static_cast<std::size_t>(n_clusters), 0));
  auto parallel = [&](auto&& body) {
    std::vector<std::thread> threads;
    for (std::int64_t t = 0; t < n_threads; ++t) {
      threads.emplace_back(
        [&, t] { body(t, std::min(n_rows, t * span), std::min(n_rows, (t + 1) * span)); });
    }
    for (auto& th : threads) {
      th.join();
    }
  };
  std::atomic<bool> bad_label{false};
  parallel([&](std::int64_t t, std::int64_t lo, std::int64_t hi) {
    auto& h = hist[static_cast<std::size_t>(t)];
    for (auto r = lo; r < hi; ++r) {
      auto const l = labels[static_cast<std::size_t>(r)];
      if (l < 0 || l >= n_clusters) {
        bad_label = true;
        return;
      }
      ++h[static_cast<std::size_t>(l)];
    }
  });
  if (bad_label) {
    throw std::runtime_error(fn + ": assignment produced a label outside the clustering");
  }

  cluster_lists lists;
  lists.pin        = c.pin;
  lists.table      = req.table;
  lists.column     = req.column;
  lists.n_rows     = n_rows;
  lists.dim        = dim;
  lists.n_clusters = n_clusters;
  lists.offsets.assign(static_cast<std::size_t>(n_clusters) + 1, 0);
  std::vector<std::vector<std::int64_t>> start = hist;
  {
    std::int64_t run = 0;
    for (std::int64_t k = 0; k < n_clusters; ++k) {
      lists.offsets[static_cast<std::size_t>(k)] = run;
      for (std::int64_t t = 0; t < n_threads; ++t) {
        start[static_cast<std::size_t>(t)][static_cast<std::size_t>(k)] = run;
        run += hist[static_cast<std::size_t>(t)][static_cast<std::size_t>(k)];
      }
    }
    lists.offsets.back() = run;
  }
  std::vector<std::int64_t> dest(static_cast<std::size_t>(n_rows));
  parallel([&](std::int64_t t, std::int64_t lo, std::int64_t hi) {
    auto& s = start[static_cast<std::size_t>(t)];
    for (auto r = lo; r < hi; ++r) {
      dest[static_cast<std::size_t>(r)] =
        s[static_cast<std::size_t>(labels[static_cast<std::size_t>(r)])]++;
    }
  });
  phase("counting sort");

  // Storage. The row map is device-resident on both tiers because the fold reads it per slice;
  // the vectors go to the device when the pin is there and they fit, else to pinned host blocks.
  auto const elem_bytes = encoding == list_encoding::uint8 ? std::size_t{1} : sizeof(float);
  auto const vec_bytes =
    static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(dim) * elem_bytes;
  auto const row_bytes = static_cast<std::size_t>(n_rows) * sizeof(std::int64_t);
  auto* host_mr =
    c.host_space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  auto const block_bytes = host_mr != nullptr ? host_mr->get_block_size() : std::size_t{0};
  auto const rows_per_block =
    static_cast<std::int64_t>(block_bytes / (static_cast<std::size_t>(dim) * elem_bytes));
  // A staged chunk is ~64 MiB of FP32 whatever the encoding, and on the HOST tier a whole number
  // of blocks. An FP32 chunk on the device is a view and needs no bound but cudf's element
  // count; a UINT8 one is widened into a buffer when staged, so it is bounded like a host chunk.
  constexpr std::size_t kStagedChunkBytes = std::size_t{64} << 20;
  auto const staged_rows                  = std::max<std::int64_t>(
    1,
    static_cast<std::int64_t>(kStagedChunkBytes / (static_cast<std::size_t>(dim) * sizeof(float))));
  auto const max_list_elements =
    static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max());

  std::unique_ptr<cucascade::memory::reservation> reservation;
  // FP32 lists live where the pin does; UINT8 ones are a quarter of the corpus and go to the
  // device whenever they fit, which is what takes the corpus stream off the query path.
  bool on_device = c.pin->tier == cucascade::memory::Tier::GPU || encoding == list_encoding::uint8;
  if (on_device) {
    reservation = index_cache.reserve_index_memory(vec_bytes + row_bytes + (std::size_t{1} << 24),
                                                   c.target_gpu);
    on_device   = reservation != nullptr;
  }
  if (!on_device) {
    if (rows_per_block == 0) {
      throw duckdb::InvalidInputException(fn + ": a " + std::to_string(dim) +
                                          "-component row does not fit one host block");
    }
    reservation =
      index_cache.reserve_index_memory(row_bytes + (std::size_t{1} << 24), c.target_gpu);
    if (!reservation) {
      throw duckdb::InvalidInputException(fn + ": not enough free GPU memory for the lists' " +
                                          std::to_string(row_bytes >> 20) + " MiB row map");
    }
  }
  // The reservation above is only the admission check. A reservation charges the allocations
  // made on a stream it is attached to, and these buffers outlive every stream here, so drawing
  // them through it would count each byte twice -- reserved and allocated -- which at 100M rows is
  // enough to exhaust the pool. They are allocated from the space directly and counted once.
  auto const reserved_bytes = reservation->size();
  reservation.reset();
  auto const persistent_mr = c.space->get_default_allocator();

  lists.encoding = encoding;
  std::vector<std::byte*> block_ptrs;
  if (on_device) {
    lists.tier           = cucascade::memory::Tier::GPU;
    lists.chunk_rows     = encoding == list_encoding::uint8
                             ? std::min<std::int64_t>(n_rows, staged_rows)
                             : std::min<std::int64_t>(n_rows, max_list_elements / dim);
    lists.rows_per_block = n_rows;
    lists.device_vectors =
      std::make_unique<rmm::device_buffer>(vec_bytes, persistent, persistent_mr);
    block_ptrs.push_back(static_cast<std::byte*>(lists.device_vectors->data()));
  } else {
    lists.tier           = cucascade::memory::Tier::HOST;
    lists.rows_per_block = rows_per_block;
    lists.chunk_rows     = std::min<std::int64_t>(
      std::max<std::int64_t>(1, staged_rows / rows_per_block) * rows_per_block,
      max_list_elements / dim);
    auto const n_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    lists.host_vectors =
      host_mr->allocate_multiple_blocks(static_cast<std::size_t>(n_blocks) * block_bytes);
    block_ptrs.reserve(static_cast<std::size_t>(n_blocks));
    for (std::int64_t b = 0; b < n_blocks; ++b) {
      block_ptrs.push_back(lists.host_vectors->at(static_cast<std::size_t>(b)).data());
    }
  }
  lists.row_ids      = std::make_unique<rmm::device_buffer>(row_bytes, persistent, persistent_mr);
  lists.list_offsets = std::make_unique<rmm::device_buffer>(
    static_cast<std::size_t>(lists.chunk_rows + 1) * sizeof(std::int32_t),
    persistent,
    persistent_mr);
  fill_list_offsets(
    static_cast<std::int32_t*>(lists.list_offsets->data()), lists.chunk_rows, dim, persistent);
  persistent.synchronize();

  phase("allocate");

  // Pass 2: stage each chunk again and move its rows into the lists. A list's rows keep pin
  // order, so the rows one chunk contributes to a list are one contiguous run of that list: the
  // chunk is grouped by cluster on the device and each group is a single copy (two where it
  // straddles a host block). Writing rows through mapped host memory from a kernel instead ran
  // at ~5 GB/s, one scattered PCIe write per row.
  {
    struct run {
      std::int64_t first;  // position in the grouped chunk
      std::int64_t dest;   // first layout row
      std::int64_t rows;
    };
    std::vector<std::int64_t> order;
    std::vector<std::int64_t> count(static_cast<std::size_t>(n_clusters));
    std::vector<run> runs;
    auto const row_bytes_v = lists.row_bytes();
    auto const copy_kind   = on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    std::int64_t base      = 0;
    for (std::size_t i = 0; i < n_chunks; ++i) {
      auto staged =
        stage_pinned_column_chunk(*c.pin, req.column, i, *c.space, stream, telemetry_info);
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (rows == 0) { continue; }
      auto const vectors       = list_column_as_dataset_view(staged.view, dim);
      auto const* chunk_labels = labels.data() + base;

      std::fill(count.begin(), count.end(), 0);
      for (std::int64_t r = 0; r < rows; ++r) {
        ++count[static_cast<std::size_t>(chunk_labels[r])];
      }
      runs.clear();
      std::int64_t pos = 0;
      for (std::int64_t k = 0; k < n_clusters; ++k) {
        auto const n = count[static_cast<std::size_t>(k)];
        if (n == 0) { continue; }
        runs.push_back(run{pos, -1, n});
        count[static_cast<std::size_t>(k)] = pos;  // now the next slot of cluster k
        pos += n;
      }
      order.resize(static_cast<std::size_t>(rows));
      for (std::int64_t r = 0; r < rows; ++r) {
        auto& slot                              = count[static_cast<std::size_t>(chunk_labels[r])];
        order[static_cast<std::size_t>(slot++)] = r;
      }
      for (auto& g : runs) {
        g.dest = dest[static_cast<std::size_t>(base + order[static_cast<std::size_t>(g.first)])];
      }

      rmm::device_uvector<std::int64_t> order_d(static_cast<std::size_t>(rows), stream, mr);
      rmm::device_uvector<std::int64_t> chunk_dest(static_cast<std::size_t>(rows), stream, mr);
      rmm::device_uvector<float> grouped(static_cast<std::size_t>(rows * dim), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(order_d.data(),
                                    order.data(),
                                    static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(chunk_dest.data(),
                                    dest.data() + base,
                                    static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      gather_rows(vectors.data_handle(), dim, order_d.data(), rows, grouped.data(), stream);
      // The copies below read the grouped rows in the list encoding.
      std::optional<rmm::device_uvector<std::uint8_t>> grouped_u8;
      auto const* grouped_bytes = reinterpret_cast<std::byte const*>(grouped.data());
      if (encoding == list_encoding::uint8) {
        grouped_u8.emplace(static_cast<std::size_t>(rows * dim), stream, mr);
        narrow_to_uint8(grouped.data(), rows * dim, grouped_u8->data(), stream);
        grouped_bytes = reinterpret_cast<std::byte const*>(grouped_u8->data());
      }
      scatter_row_ids(
        chunk_dest.data(), rows, base, static_cast<std::int64_t*>(lists.row_ids->data()), stream);
      for (auto const& g : runs) {
        for (std::int64_t t = 0; t < g.rows;) {
          auto const r     = g.dest + t;
          auto const in_bl = r % lists.rows_per_block;
          auto const n     = std::min(g.rows - t, lists.rows_per_block - in_bl);
          CUDF_CUDA_TRY(
            cudaMemcpyAsync(block_ptrs[static_cast<std::size_t>(r / lists.rows_per_block)] +
                              static_cast<std::size_t>(in_bl) * row_bytes_v,
                            grouped_bytes + static_cast<std::size_t>(g.first + t) * row_bytes_v,
                            static_cast<std::size_t>(n) * row_bytes_v,
                            copy_kind,
                            stream.value()));
          t += n;
        }
      }
      // The staged chunk and the pageable sources of the copies above are released or reused
      // once this iteration ends.
      stream.synchronize();
      base += rows;
    }
  }
  labels.clear();
  labels.shrink_to_fit();

  phase("pass 2 scatter");

  cluster_lists_result result;
  result.n_rows     = n_rows;
  result.n_clusters = n_clusters;
  result.tier       = on_device ? "gpu" : "host";
  result.encoding   = encoding == list_encoding::uint8 ? "uint8" : "float32";
  result.min_list   = std::numeric_limits<std::int64_t>::max();
  for (std::int64_t k = 0; k < n_clusters; ++k) {
    auto const size =
      lists.offsets[static_cast<std::size_t>(k) + 1] - lists.offsets[static_cast<std::size_t>(k)];
    result.min_list = std::min(result.min_list, size);
    result.max_list = std::max(result.max_list, size);
    if (size == 0) { ++result.empty_lists; }
  }

  index_metadata meta;
  meta.kind           = index_kind::cluster_lists;
  meta.table_name     = req.table;
  meta.column_name    = req.column;
  meta.dim            = dim;
  meta.num_rows       = n_rows;
  meta.n_lists        = n_clusters;
  meta.metric         = entry->meta.metric;
  meta.reserved_bytes = reserved_bytes;
  index_cache.insert(lists_key(req.clustering),
                     std::move(meta),
                     make_cuvs_index(std::move(lists)),
                     std::move(reservation));
  return result;
}

std::unique_ptr<cucascade::host_data_representation> run_kmeans_centroids(
  duckdb::SiriusContext& ctx, const std::string& clustering)
{
  static const std::string fn = "sirius_kmeans_centroids";

  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto host_spaces     = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (gpu_spaces.empty() || host_spaces.empty()) {
    throw duckdb::InvalidInputException(fn + ": no GPU or HOST memory space available");
  }
  auto* space = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space->get_device_id()}};
  rmm::cuda_stream stream_owner;
  auto stream   = stream_owner.view();
  auto const mr = space->get_default_allocator();

  const auto* entry = find_clustering_entry(ctx, clustering);
  if (entry == nullptr) {
    throw duckdb::InvalidInputException(fn + ": no clustering named '" + clustering + "'");
  }
  const auto* centroids = find_clustering_centroids(ctx, clustering);
  if (centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + clustering +
                                        "' holds no centroids");
  }

  auto const dim    = entry->meta.dim;
  auto const values = cudf::lists_column_view(centroids->view()).child();
  auto const total  = values.size();

  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const one(1, true, stream);
  cudf::numeric_scalar<std::int32_t> const width(static_cast<std::int32_t>(dim), true, stream);

  // Position in the flattened [n_clusters, dim] buffer splits into its two coordinates by
  // division and remainder, which is what puts the centroids in long form.
  auto const positions = cudf::sequence(total, zero, one, stream, mr);
  auto cluster_ids     = cudf::binary_operation(positions->view(),
                                            width,
                                            cudf::binary_operator::DIV,
                                            cudf::data_type{cudf::type_id::INT32},
                                            stream,
                                            mr);
  auto dim_index       = cudf::binary_operation(positions->view(),
                                          width,
                                          cudf::binary_operator::MOD,
                                          cudf::data_type{cudf::type_id::INT32},
                                          stream,
                                          mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(cluster_ids));
  columns.push_back(std::move(dim_index));
  columns.push_back(std::make_unique<cudf::column>(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  return vss_table_to_host(*space, *host_spaces.front(), stream, std::move(table));
}

}  // namespace sirius::vss
