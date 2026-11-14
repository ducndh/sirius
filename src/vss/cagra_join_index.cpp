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

#include "vss/cagra_join_index.hpp"

#include "cuda/vss/cagra_index.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <stdexcept>

namespace sirius::vss {

namespace {

std::string cache_key(cagra_index_target const& t)
{ return build_ann_index_cache_key(t.catalog, t.schema, t.table, t.column, "cagra"); }

double gib(std::size_t bytes) { return static_cast<double>(bytes) / 1073741824.0; }

/// A HOST-tier pin stages chunks to convert them. A chunk is at most one row group of rows.
std::size_t largest_staged_chunk(scan_manager::pinned_entry const& pin,
                                 cagra_index_target const& target)
{
  if (pin.tier != cucascade::memory::Tier::HOST || pin.host_chunks.empty()) { return 0; }
  auto const chunks = pin.host_chunks.size();
  auto const rows   = (pin.num_rows + chunks - 1) / chunks;
  // Chunks need not be equal; twice the mean holds what a row group's cut leaves over.
  return 2 * static_cast<std::size_t>(rows) * static_cast<std::size_t>(target.dim) * sizeof(float);
}

}  // namespace

std::shared_ptr<const pinned_index_entry> find_usable_cagra_index(
  cuvs_index_cache const& cache,
  cagra_index_target const& target,
  scan_manager::pinned_entry const& pin,
  std::optional<cagra_build_settings> const& wanted)
{
  auto entry = cache.find(cache_key(target));
  if (entry == nullptr || entry->meta.kind != index_kind::cagra ||
      !entry->meta.table_identity.matches(target.identity) ||
      !entry->meta.matches_pin(pin.snapshot_identity)) {
    return nullptr;
  }
  if (wanted && !(cagra_index_of(*entry).settings() == *wanted)) { return nullptr; }
  return entry;
}

cagra_index const& cagra_index_of(pinned_index_entry const& entry)
{
  auto const* index = dynamic_cast<cagra_index const*>(entry.index.get());
  if (index == nullptr) { throw std::runtime_error("[vss] the entry holds no CAGRA index"); }
  return *index;
}

std::size_t cagra_build_peak_bytes(std::int64_t n_rows,
                                   std::int64_t dim,
                                   cagra_build_settings const& settings,
                                   std::size_t staged_chunk_bytes)
{
  // A partitioned build holds one partition at a time, its largest taken 10 % over an even share.
  if (settings.partitions > 1) {
    n_rows = n_rows / settings.partitions + n_rows / settings.partitions / 10 + 1;
  }
  // Host chunks are staged while the rows are converted, before the graph build's scratch exists.
  auto const rows =
    static_cast<std::size_t>(n_rows) * cagra_row_stride_bytes(dim, settings.row_type);
  auto const converting = rows + 2 * staged_chunk_bytes;
  auto const building =
    cagra_resident_bytes(n_rows, dim, settings) + cagra_build_scratch_bytes(n_rows, dim, settings);
  return std::max(converting, building);
}

void check_cagra_build_fits(std::string const& fn,
                            cagra_index_target const& target,
                            scan_manager::pinned_entry const& pin,
                            cagra_build_settings const& settings,
                            cucascade::memory::memory_space const& space)
{
  auto const n_rows = static_cast<std::int64_t>(pin.num_rows);
  auto const peak =
    cagra_build_peak_bytes(n_rows, target.dim, settings, largest_staged_chunk(pin, target));
  auto const available = space.get_available_memory();
  if (peak <= available) { return; }
  auto const largest = settings.partitions > 1
                         ? n_rows / settings.partitions + n_rows / settings.partitions / 10 + 1
                         : n_rows;
  auto const rows =
    static_cast<std::size_t>(largest) * cagra_row_stride_bytes(target.dim, settings.row_type);
  auto const resident = cagra_resident_bytes(largest, target.dim, settings);
  throw duckdb::InvalidInputException(std::format(
    "{}: a CAGRA index over '{}.{}' does not fit: {} rows x {}-d{} as {} are {:.2f} GiB, the graph "
    "at degree {} is {:.2f} GiB and the build's scratch {:.2f} GiB, {:.2f} GiB at its peak, and "
    "the device has {:.2f} GiB free of {:.2f} GiB{}",
    fn,
    target.table,
    target.column,
    largest,
    target.dim,
    settings.partitions > 1 ? std::format(" in the largest of {} partitions", settings.partitions)
                            : std::string{},
    cagra_row_type_name(settings.row_type),
    gib(rows),
    settings.graph_degree,
    gib(resident - rows),
    gib(peak > resident ? peak - resident : 0),
    gib(peak),
    gib(available),
    gib(space.get_max_memory()),
    pin.tier == cucascade::memory::Tier::GPU
      ? "; pinning the table to the host tier frees what its FP32 rows hold on the device, and "
        "partitions => 2, 4 or 8 holds one partition at a time"
      : "; partitions => 2, 4 or 8 holds one partition at a time"));
}

std::shared_ptr<const pinned_index_entry> build_cagra_index_for_pin(
  std::string const& fn,
  cuvs_index_cache& cache,
  cagra_index_target const& target,
  std::shared_ptr<const scan_manager::pinned_entry> const& pin,
  cagra_build_settings const& settings,
  cucascade::memory::memory_space& space,
  telemetry::batch_telemetry_info const& telemetry_info,
  cagra_build_report& report)
{
  auto const n_rows = static_cast<std::int64_t>(pin->num_rows);
  if (n_rows <= settings.intermediate_graph_degree) {
    throw duckdb::InvalidInputException(
      std::format("{}: a CAGRA index over '{}.{}' needs more rows than its intermediate graph "
                  "degree: the table has {} rows and intermediate_graph_degree is {}",
                  fn,
                  target.table,
                  target.column,
                  n_rows,
                  settings.intermediate_graph_degree));
  }
  // A column has one index, and this one replaces whatever it had: what the old one holds is room
  // this build may need, and it cannot answer for these settings or this pin anyway.
  cache.erase(cache_key(target));
  check_cagra_build_fits(fn, target, *pin, settings, space);

  auto const peak =
    cagra_build_peak_bytes(n_rows, target.dim, settings, largest_staged_chunk(*pin, target));
  auto reservation = cache.reserve_index_memory(peak, space.get_device_id());
  if (!reservation) {
    throw duckdb::InvalidInputException(
      std::format("{}: could not reserve {:.2f} GiB of device memory to build a CAGRA index over "
                  "'{}.{}'; {:.2f} GiB is free of {:.2f} GiB",
                  fn,
                  gib(peak),
                  target.table,
                  target.column,
                  gib(space.get_available_memory()),
                  gib(space.get_max_memory())));
  }

  rmm::cuda_stream build_stream;
  rmm::cuda_stream_view const stream = build_stream;
  auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (allocator == nullptr ||
      !allocator->attach_reservation_to_tracker(build_stream, std::move(reservation))) {
    throw duckdb::InvalidInputException(fn + ": failed to bind the index build to its reservation");
  }
  absl::Cleanup reset_reservation = [allocator, &build_stream] {
    allocator->reset_stream_reservation(build_stream);
  };
  auto const mr     = space.get_default_allocator();
  auto const stride = cagra_row_stride_bytes(target.dim, settings.row_type);

  std::unique_ptr<cagra_index> index;
  try {
    auto const convert_start = std::chrono::steady_clock::now();
    auto rows =
      std::make_unique<rmm::device_buffer>(static_cast<std::size_t>(n_rows) * stride, stream, mr);
    // Rows padded to their stride: the graph build reads the padding as dimensions.
    if (stride != static_cast<std::size_t>(target.dim) * cagra_value_bytes(settings.row_type)) {
      CUDF_CUDA_TRY(cudaMemsetAsync(rows->data(), 0, rows->size(), stream.get()));
    }
    row_norm_range norms;
    {
      pinned_chunk_walk walk(*pin, target.column, space, stream, telemetry_info);
      std::int64_t at = 0;
      for (std::size_t i = 0; i < walk.size(); ++i) {
        auto const chunk = walk.take(i);
        auto const view  = list_column_as_dataset_view(chunk.view, target.dim);
        cagra_convert_rows(view.data_handle(),
                           view.extent(0),
                           target.dim,
                           settings.row_type,
                           static_cast<std::byte*>(rows->data()) + at * stride,
                           stride,
                           norms,
                           stream,
                           mr);
        walk.prefetch_after(i);
        stream.sync();
        at += view.extent(0);
      }
      if (at != n_rows) {
        throw std::runtime_error(
          std::format("{}: the pin's chunks hold {} rows of {}", fn, at, n_rows));
      }
    }
    if (!(norms.max_sq < std::numeric_limits<float>::infinity())) {
      throw duckdb::InvalidInputException(
        settings.row_type == cagra_row_type::float16
          ? std::format("{}: '{}.{}' holds a value past float16's range, a NaN or an infinity; "
                        "row_type => 'float32' keeps values float16 cannot hold",
                        fn,
                        target.table,
                        target.column)
          : std::format("{}: '{}.{}' holds a NaN or an infinity, which CAGRA cannot rank",
                        fn,
                        target.table,
                        target.column));
    }
    auto const build_start = std::chrono::steady_clock::now();
    report.convert_seconds = std::chrono::duration<double>(build_start - convert_start).count();
    index                  = build_cagra_index(
      std::move(rows), n_rows, target.dim, stride, settings, norms_are_unit(norms), stream);
    report.build_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
  } catch (rmm::out_of_memory const& oom) {
    // Rebuilding from the first row under the engine's retry cannot help: everything here is the
    // index's own, and nothing else in the statement can be moved to make room for it.
    throw duckdb::InvalidInputException(
      std::format("{}: ran out of device memory building a CAGRA index over '{}.{}', {} rows x "
                  "{}-d, with {:.2f} GiB reserved for it: {}",
                  fn,
                  target.table,
                  target.column,
                  n_rows,
                  target.dim,
                  gib(peak),
                  oom.what()));
  }

  index_metadata meta;
  meta.kind             = index_kind::cagra;
  meta.catalog_name     = target.catalog;
  meta.schema_name      = target.schema;
  meta.table_name       = target.table;
  meta.table_identity   = target.identity;
  meta.pin_snapshot     = pin->snapshot_identity;
  meta.column_name      = target.column;
  meta.dim              = target.dim;
  meta.num_rows         = n_rows;
  meta.metric           = cuvs::distance::DistanceType::L2SqrtExpanded;
  meta.resident_bytes   = allocator->get_allocated_bytes(build_stream);
  report.resident_bytes = meta.resident_bytes;
  report.peak_bytes     = allocator->get_peak_allocated_bytes(build_stream);
  std::move(reset_reservation).Invoke();

  auto key = cache_key(target);
  cache.insert(key, std::move(meta), std::move(index), std::move(build_stream));
  return cache.find(key);
}

}  // namespace sirius::vss
