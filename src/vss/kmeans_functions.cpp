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
#include "vss/cuvs_index_cache.hpp"
#include "vss/distance_metric.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <string>
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

  const auto* pin = ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(catalog, schema, table);
  if (pin == nullptr) {
    throw duckdb::InvalidInputException(fn + ": table '" + table +
                                        "' must be pinned (GPU or HOST tier)");
  }

  auto* space = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  return kmeans_context{space, host_spaces.front(), pin, space->get_device_id()};
}

/// The clustering stored under @p name, or nullptr when the name is free or holds something
/// that is not a clustering.
const pinned_index_entry* find_clustering_entry(duckdb::SiriusContext& ctx,
                                                const std::string& name)
{
  const auto* entry = ctx.get_cuvs_index_cache().find(name);
  if (entry == nullptr || entry->meta.kind != index_kind::kmeans_centroids) { return nullptr; }
  return entry;
}

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
  if (n_rows == 0) { throw duckdb::InvalidInputException(fn + ": table '" + req.table + "' is empty"); }

  auto const metric     = ann_distance_type_from_metric(req.metric);
  auto const n_clusters = resolve_n_clusters(req.spec.n_clusters, n_rows);
  auto const train_rows = resolve_train_rows(req.spec.train_rows, n_rows, n_clusters);

  // Over-reserved to cover the sample, the concatenated copy train_centroids makes of it, and
  // k-means' own scratch; shrunk to the centroids alone once the fit is done.
  auto const vec_bytes = static_cast<std::size_t>(req.dim) * sizeof(float);
  std::size_t const footprint = static_cast<std::size_t>(train_rows) * vec_bytes * 3 +
                                static_cast<std::size_t>(n_clusters) * vec_bytes * 2 +
                                (std::size_t{1} << 20);

  auto& index_cache = ctx.get_cuvs_index_cache();
  // Free the old clustering's reservation before asking for the new one, so re-fitting under
  // the same name does not have to fit both at once.
  index_cache.erase(req.name);
  auto reservation = index_cache.reserve_index_memory(footprint, c.target_gpu);
  if (!reservation) {
    throw duckdb::InvalidInputException(
      fn + ": not enough free GPU memory to train " + std::to_string(n_clusters) +
      " centroids: need ~" + std::to_string(footprint >> 20) + " MiB");
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
      auto staged     = stage_pinned_column_chunk(
        *c.pin, req.column, i, *c.space, stream, telemetry_info);
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
    throw duckdb::InvalidInputException(
      fn + ": clustering '" + req.clustering + "' is over FLOAT[" +
      std::to_string(entry->meta.dim) + "] vectors but column '" + req.column + "' is FLOAT[" +
      std::to_string(req.dim) + "]");
  }
  const auto* centroids = find_clustering_centroids(ctx, req.clustering);
  if (centroids == nullptr) {
    throw duckdb::InvalidInputException(fn + ": clustering '" + req.clustering +
                                        "' holds no centroids");
  }

  telemetry::batch_telemetry_info const telemetry_info{};
  auto const n_chunks = pinned_column_chunk_count(*c.pin, req.column);

  std::vector<std::unique_ptr<cudf::column>> row_ids;
  std::vector<std::unique_ptr<cudf::column>> cluster_ids;
  std::vector<std::unique_ptr<cudf::column>> distances;
  std::int64_t base = 0;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto staged     = stage_pinned_column_chunk(
      *c.pin, req.column, i, *c.space, stream, telemetry_info);
    auto const rows = static_cast<std::int64_t>(staged.view.size());
    if (rows == 0) { continue; }

    auto assignment = assign_to_centroids(staged.view,
                                          centroids->view(),
                                          req.dim,
                                          req.spec,
                                          base,
                                          entry->meta.metric,
                                          stream,
                                          mr);
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
    if (parts.size() == 1) { return std::make_unique<cudf::column>(parts.front()->view(), stream, mr); }
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

}  // namespace sirius::vss
