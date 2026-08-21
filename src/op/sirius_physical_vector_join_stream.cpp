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

#include "vss/sirius_physical_vector_join_stream.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"
#include "vss/join_result_shaping.hpp"
#include "vss/knn_merge.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_clustering.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

/// GPU-tier pin: every chunk is already device-resident, so staging hands back a view.
class gpu_pinned_chunk_source : public vector_chunk_source {
 public:
  gpu_pinned_chunk_source(const scan_manager::pinned_entry& pin,
                          const std::string& column,
                          cucascade::memory::memory_space& space)
    : _views(vss::pinned_column_chunk_views(pin, column, space))
  {
  }

  [[nodiscard]] std::size_t num_chunks() const override { return _views.size(); }
  [[nodiscard]] bool is_streaming() const override { return false; }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override
  {
    return static_cast<std::size_t>(_views.at(i).size());
  }
  [[nodiscard]] std::size_t chunk_bytes(std::size_t /*i*/) const override { return 0; }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& /*space*/,
                            rmm::cuda_stream_view /*stream*/) override
  {
    return staged_vector_chunk{_views.at(i), nullptr, nullptr};
  }

 private:
  std::vector<cudf::column_view> _views;
};

/// HOST-tier pin: each chunk is copied device-side on demand through the same converter
/// the scan path uses, and freed when the caller drops the returned owner. Peak device
/// memory is therefore set by the chunks in flight, not by the corpus size.
class host_pinned_chunk_source : public vector_chunk_source {
 public:
  host_pinned_chunk_source(const scan_manager::pinned_entry& pin,
                           const std::string& column,
                           std::int64_t dim,
                           const telemetry::batch_telemetry_info& telemetry_info)
    : _pin(pin), _telemetry_info(telemetry_info), _dim(dim)
  {
    auto const& names = _pin.cache_info.column_names();
    auto const it     = std::find(names.begin(), names.end(), column);
    if (it == names.end()) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] host-tier pin is missing "
        "column '" +
        column + "'");
    }
    _column_index = static_cast<std::size_t>(std::distance(names.begin(), it));
  }

  [[nodiscard]] std::size_t num_chunks() const override { return _pin.host_chunks.size(); }
  [[nodiscard]] bool is_streaming() const override { return true; }

  // Row and byte counts without staging, so the memory estimate can size a task before any
  // copy happens. A host chunk reports bytes; rows follow from the fixed vector width.
  [[nodiscard]] std::size_t chunk_bytes(std::size_t i) const override
  {
    auto const& chunk = _pin.host_chunks.at(i);
    return chunk ? chunk->column_size(_column_index) : 0;
  }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override
  {
    auto const width = static_cast<std::size_t>(_dim) * sizeof(float);
    return width == 0 ? 0 : chunk_bytes(i) / width;
  }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) override
  {
    auto const& chunk = _pin.host_chunks.at(i);
    if (!chunk) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] host chunk " +
                               std::to_string(i) + " is null");
    }
    // Slice to the vector column alone: the rest of the pinned table is dead weight on
    // the wire and this copy is the operator's bandwidth budget.
    std::array<std::size_t, 1> const cols{_column_index};
    auto data_rep    = chunk->slice(cols);
    auto const bytes = data_rep->get_size_in_bytes();

    // Draw the staged copy from the task's budget. A null reservation means the chunk does
    // not fit what this task was granted, which is a sizing problem to surface, not to
    // silently exceed.
    std::shared_ptr<cucascade::memory::reservation> reservation{
      space.make_reservation_or_null(bytes)};
    if (!reservation) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] corpus chunk " +
                               std::to_string(i) + " needs " + std::to_string(bytes) +
                               " bytes device-side, which exceeds this task's budget");
    }

    auto const batch_id = sirius::get_next_batch_id();
    auto batch          = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));

    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), *reservation, stream);
    }
    auto const table = sirius::get_cudf_table_view(*batch);
    return staged_vector_chunk{table.column(0), std::move(batch), std::move(reservation)};
  }

 private:
  const scan_manager::pinned_entry& _pin;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
  std::int64_t _dim{0};
};

/// Build phase: the corpus is whatever the child scan deposited in the build port, walked in
/// the shared snapshot's order. A batch still device-resident is borrowed in place under a
/// read lock; one the downgrade executor has spilled is copied back device-side for the fold
/// step and released after it, exactly as a HOST-tier pin chunk is. So the same source serves
/// a corpus that fits and one that does not, and which case applies is decided per chunk at
/// staging time rather than per query at planning time.
class materialized_chunk_source : public vector_chunk_source {
 public:
  materialized_chunk_source(vss::materialized_side_buffer& buffer,
                            std::size_t column_index,
                            std::int64_t dim,
                            const telemetry::batch_telemetry_info& telemetry_info)
    : _repo(buffer.repo()),
      _batch_ids(buffer.batch_ids()),
      _telemetry_info(telemetry_info),
      _column_index(column_index),
      _dim(dim)
  {
    if (_repo == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] build side has no snapshot; the build pipeline "
        "has not finished");
    }
    // Sizes up front: the task estimate and the neighbour-id base both need per-chunk rows
    // before anything is staged.
    _rows.reserve(_batch_ids.size());
    _bytes.reserve(_batch_ids.size());
    for (auto const id : _batch_ids) {
      auto batch = fetch(id);
      auto ro    = batch->to_read_only();
      if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
        auto const table = sirius::get_cudf_table_view(ro);
        auto const col   = table.column(static_cast<cudf::size_type>(_column_index));
        _rows.push_back(static_cast<std::size_t>(col.size()));
        _bytes.push_back(row_bytes() * _rows.back());
      } else {
        auto const bytes = host_repr(ro).column_size(_column_index);
        _bytes.push_back(bytes);
        _rows.push_back(row_bytes() == 0 ? 0 : bytes / row_bytes());
      }
    }
  }

  [[nodiscard]] std::size_t num_chunks() const override { return _batch_ids.size(); }
  /// Reported as streaming: any chunk may have been spilled by the time it is staged, so a
  /// task must be sized as though it will have to copy one back.
  [[nodiscard]] bool is_streaming() const override { return true; }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override { return _rows.at(i); }
  [[nodiscard]] std::size_t chunk_bytes(std::size_t i) const override { return _bytes.at(i); }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) override
  {
    auto batch = fetch(_batch_ids.at(i));
    auto ro    = batch->to_read_only();
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      auto const table = sirius::get_cudf_table_view(ro);
      auto const view  = table.column(static_cast<cudf::size_type>(_column_index));
      // Borrowed, not owned: the reader holds the batch alive as well as locked, and `owner`
      // stays null so the fold does not try to rebind a stream on memory it did not allocate.
      return staged_vector_chunk{view, nullptr, nullptr, std::move(ro)};
    }

    // Spilled: copy just the vector column back, the rest of the batch is dead weight on the
    // wire. Mirrors the HOST-tier pin path, including drawing from the task's own budget so a
    // chunk that does not fit surfaces as a sizing error instead of silently overcommitting.
    // The slice references the batch's host allocation, so the borrow is held across the copy.
    std::array<std::size_t, 1> const cols{_column_index};
    auto data_rep    = host_repr(ro).slice(cols);
    auto const bytes = data_rep->get_size_in_bytes();

    std::shared_ptr<cucascade::memory::reservation> reservation{
      space.make_reservation_or_null(bytes)};
    if (!reservation) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] corpus chunk " +
                               std::to_string(i) + " needs " + std::to_string(bytes) +
                               " bytes device-side, which exceeds this task's budget");
    }

    auto const batch_id = sirius::get_next_batch_id();
    auto staged         = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
    {
      auto mut = staged->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), *reservation, stream);
    }
    auto const table = sirius::get_cudf_table_view(*staged);
    return staged_vector_chunk{
      table.column(0), std::move(staged), std::move(reservation), std::move(ro)};
  }

 private:
  [[nodiscard]] std::size_t row_bytes() const
  {
    return static_cast<std::size_t>(_dim) * sizeof(float);
  }

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> fetch(std::uint64_t id) const
  {
    auto batch = _repo->get_data_batch_by_id(id, /*partition_idx=*/0);
    if (!batch) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] build-side batch " + std::to_string(id) +
        " is no longer in the repository; the corpus must outlive every probe chunk");
    }
    return batch;
  }

  static const cucascade::host_data_representation& host_repr(
    const cucascade::read_only_data_batch& ro)
  {
    const auto* data = ro.get_data();
    if (data == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] build-side batch has no data representation");
    }
    return data->cast<cucascade::host_data_representation>();
  }

  cucascade::shared_data_repository* _repo;
  std::vector<std::uint64_t> _batch_ids;
  std::vector<std::size_t> _rows;
  std::vector<std::size_t> _bytes;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
  std::int64_t _dim{0};
};

}  // namespace

std::unique_ptr<vector_chunk_source> make_materialized_chunk_source(
  sirius::vss::materialized_side_buffer& buffer,
  std::size_t column_index,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  return std::make_unique<materialized_chunk_source>(buffer, column_index, dim, telemetry_info);
}

std::unique_ptr<vector_chunk_source> make_gpu_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  cucascade::memory::memory_space& space)
{
  return std::make_unique<gpu_pinned_chunk_source>(pin, column, space);
}

std::unique_ptr<vector_chunk_source> make_host_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  return std::make_unique<host_pinned_chunk_source>(pin, column, dim, telemetry_info);
}

sirius_physical_vector_join_stream::sirius_physical_vector_join_stream(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager,
  std::shared_ptr<sirius::vss::materialized_side_buffer> build_side,
  std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side,
  const cudf::column* centroids,
  duckdb::SiriusContext* sirius_ctx)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_STREAM, std::move(types), estimated_cardinality),
    _build_side(std::move(build_side)),
    _probe_side(std::move(probe_side)),
    _request(std::move(request)),
    _scan_manager(scan_manager),
    _centroids(centroids),
    _sirius_ctx(sirius_ctx)
{
}

void sirius_physical_vector_join_stream::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (children.empty()) {
    sirius_physical_operator::build_pipelines(current, meta_pipeline);
    return;
  }

  // Mirrors sirius_physical_nested_loop_join::build_pipelines, with one child: the corpus.
  // The child is the wrap chain's CONCAT, whose own child is the PARTITION that materializes
  // the scan into this operator's build port.
  pipeline::sirius_meta_pipeline* host_meta;
  pipeline::sirius_pipeline* host_current;
  if (is_sink()) {
    auto& sink_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
    host_meta       = &sink_meta;
    host_current    = sink_meta.get_base_pipeline().get();
  } else {
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    host_meta    = &meta_pipeline;
    host_current = &current;
  }

  // One child meta pipeline per fed side; each child is the wrap chain's CONCAT, whose own
  // child is the PARTITION that materializes that side's scan into this operator's port.
  for (auto& child_slot : children) {
    auto& child = *child_slot;
    D_ASSERT(child.is_sink());
    D_ASSERT(!child.children.empty());
    auto& child_meta = host_meta->create_child_meta_pipeline(*host_current, child);
    child_meta.build(*child.children[0]);
  }
}

bool sirius_physical_vector_join_stream::build_side_ready_locked()
{
  // A side is not complete -- and so its row order is not yet fixed -- until the pipeline
  // feeding it has finished. Snapshotting before that would silently join against a prefix.
  auto ready = [&](const char* port_id) {
    auto* port = get_port(port_id);
    if (port == nullptr || port->repo == nullptr) { return false; }
    return !port->src_pipeline || port->src_pipeline->is_pipeline_finished();
  };
  if (_build_side && !ready("build")) { return false; }
  if (_probe_side && !ready("default")) { return false; }
  return true;
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::ensure_initialized_locked()
{
  if (_initialized) { return; }
  if (!build_side_ready_locked()) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  const auto* left_pin = _probe_side ? nullptr
                                     : _scan_manager->find_pinned_entry_for_duckdb_table(
                                         left.catalog, left.schema, left.table);
  // The corpus comes from the build port on the build path, so only the probe side has to be
  // pinned there.
  const auto* right_pin = _build_side ? nullptr
                                      : _scan_manager->find_pinned_entry_for_duckdb_table(
                                          right.catalog, right.schema, right.table);
  if ((!_probe_side && left_pin == nullptr) || (!_build_side && right_pin == nullptr)) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] left or right table is no longer pinned");
  }

  // Both sides go behind the same seam. The probe side used to be required GPU-resident on
  // the argument that it is the small one; that is false for any join where both sides are
  // large (rec-sys candidate generation is the motivating case). One task already handles one
  // probe chunk, so streaming the probe is the same change staging the corpus was: stage the
  // chunk this task owns, search the whole corpus against it, release.
  if (_probe_side) {
    auto* port = get_port("default");
    _probe_side->ensure_snapshot(*port->repo);
    _probe = make_materialized_chunk_source(
      *_probe_side, /*column_index=*/0, _request.dim, batch_telemetry());
  } else if (left_pin->tier == cucascade::memory::Tier::HOST) {
    _probe = make_host_pinned_chunk_source(*left_pin, left.column, _request.dim, batch_telemetry());
  } else {
    _probe =
      make_gpu_pinned_chunk_source(*left_pin, left.column, vss::pinned_entry_gpu_space(*left_pin));
  }

  if (_build_side) {
    // Column 0 of every build batch is the vector column: the plan generator projects the
    // corpus scan that way precisely so the fold needs no name lookup here.
    auto* port = get_port("build");
    _build_side->ensure_snapshot(*port->repo);
    _corpus = make_materialized_chunk_source(
      *_build_side, /*column_index=*/0, _request.dim, batch_telemetry());
    for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
      _max_chunk_bytes = std::max(_max_chunk_bytes, _corpus->chunk_bytes(i));
    }
  } else if (right_pin->tier == cucascade::memory::Tier::HOST) {
    _corpus =
      make_host_pinned_chunk_source(*right_pin, right.column, _request.dim, batch_telemetry());
    // Sized from the widest chunk, since any one of them may be the one in flight when
    // the reservation is granted.
    auto const& names = right_pin->cache_info.column_names();
    auto const it     = std::find(names.begin(), names.end(), right.column);
    if (it != names.end()) {
      auto const col = static_cast<std::size_t>(std::distance(names.begin(), it));
      for (auto const& chunk : right_pin->host_chunks) {
        if (chunk) { _max_chunk_bytes = std::max(_max_chunk_bytes, chunk->column_size(col)); }
      }
    }
  } else {
    _corpus = make_gpu_pinned_chunk_source(
      *right_pin, right.column, vss::pinned_entry_gpu_space(*right_pin));
  }

  // Row counts per chunk are not known until a chunk is staged, so the neighbor-id base is
  // accumulated while streaming rather than pre-summed. The total comes from the pin, or on
  // the build path from the materialized batches, which report rows without being staged.
  if (_build_side) {
    _right_total_rows = 0;
    for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
      _right_total_rows += static_cast<std::int64_t>(_corpus->chunk_rows(i));
    }
  } else {
    _right_total_rows = static_cast<std::int64_t>(right_pin->num_rows);
  }

  _num_left = _probe->num_chunks();
  if (_probe->is_streaming()) {
    for (std::size_t i = 0; i < _num_left; ++i) {
      _max_probe_chunk_bytes = std::max(_max_probe_chunk_bytes, _probe->chunk_bytes(i));
    }
  }
  _initialized = true;
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_vector_join_stream::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  // Before the corpus is complete the base rule names the build pipeline as the producer to
  // wait on; the per-probe-chunk schedule below only starts once it says READY.
  if (!build_side_ready_locked()) { return sirius_physical_operator::get_next_task_hint(); }
  ensure_initialized_locked();
  if (_num_left == 0 || _next_left >= _num_left || _hint_returned) { return std::nullopt; }
  _hint_returned = true;
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_vector_join_stream::all_ports_empty()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  // Work is still to come, it just cannot be scheduled yet.
  if (!build_side_ready_locked()) { return false; }
  ensure_initialized_locked();
  return _next_left >= _num_left;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  if (!build_side_ready_locked()) { return nullptr; }
  ensure_initialized_locked();
  if (_next_left >= _num_left) { return nullptr; }

  auto const left_idx = _next_left++;
  return std::make_unique<vector_join_stream_input>(left_idx, per_left_batch_estimate(left_idx));
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::ensure_cluster_index(
  ::cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  raft::device_resources const& res,
  cuvs::distance::DistanceType metric)
{
  std::lock_guard<std::mutex> const lock(_op_mutex);
  if (_cluster_index_built) { return; }

  // The labels come from wherever the vectors come from, and chunk for chunk: a label's meaning
  // is its position in the corpus row order, so reading them from a second source -- a pin
  // behind a build-phase scan, say -- would be reading a different row order.
  std::unique_ptr<vector_chunk_source> build_labels;
  const scan_manager::pinned_entry* pin = nullptr;
  if (_build_side) {
    // The plan generator projects the corpus scan as [vector, emitted columns..., cluster id],
    // so the cluster column's position is arithmetic rather than a name lookup -- the same
    // arrangement that puts the vector column at 0. A `dim` of 1 gives the source a four-byte
    // row width, which is what one INT32 label per row occupies.
    build_labels =
      make_materialized_chunk_source(*_build_side,
                                     /*column_index=*/1 + _request.right.output_columns.size(),
                                     /*dim=*/1,
                                     batch_telemetry());
  } else {
    pin = _scan_manager->find_pinned_entry_for_duckdb_table(
      _request.right.catalog, _request.right.schema, _request.right.table);
    if (pin == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] clustering needs the corpus table '" +
        _request.right.table + "' pinned");
    }
  }

  auto const n_chunks = build_labels
                          ? build_labels->num_chunks()
                          : vss::pinned_column_chunk_count(*pin, _request.build_cluster_column);
  if (n_chunks != _corpus->num_chunks()) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] cluster column '" + _request.build_cluster_column +
      "' has " + std::to_string(n_chunks) + " chunks but the vector column has " +
      std::to_string(_corpus->num_chunks()) + "; both must come from the same corpus");
  }

  _n_clusters = static_cast<std::int64_t>(
    vss::list_column_as_dataset_view(_centroids->view(), _request.dim).extent(0));

  _cluster_rows.assign(static_cast<std::size_t>(_n_clusters), 0);
  _chunk_cluster_runs.clear();
  _chunk_cluster_runs.resize(n_chunks);
  _chunk_row_base.assign(n_chunks, 0);

  // The cluster column is read to the host one chunk at a time: it is one INT32 per corpus
  // row, so even a 100M-row corpus is 400 MB, and having it host-side turns slice lookup into
  // arithmetic instead of a device round-trip on every probe run. Read per chunk rather than
  // flattened because a slice is only searchable once its own chunk is staged, so what the
  // fold needs is chunk-local rows -- a corpus row index would have to be undone again.
  std::vector<std::int32_t> labels;
  std::int64_t row_base = 0;
  for (std::size_t j = 0; j < n_chunks; ++j) {
    _chunk_row_base[j] = row_base;

    // Both staged chunks are declared here so whichever one holds the labels outlives the copy.
    staged_vector_chunk staged_build;
    vss::staged_pinned_chunk staged_pin;
    cudf::column_view labels_view;
    if (build_labels) {
      staged_build = build_labels->stage(j, space, stream);
      labels_view  = staged_build.view;
    } else {
      staged_pin = vss::stage_pinned_column_chunk(
        *pin, _request.build_cluster_column, j, space, stream, batch_telemetry());
      labels_view = staged_pin.view;
    }
    if (labels_view.type().id() != cudf::type_id::INT32) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column '" +
                               _request.build_cluster_column +
                               "' must be INTEGER; sirius_kmeans_assign emits cluster_id that way");
    }
    auto const rows = static_cast<std::size_t>(labels_view.size());
    if (rows == 0) { continue; }

    labels.resize(rows);
    CUDF_CUDA_TRY(cudaMemcpyAsync(labels.data(),
                                  labels_view.data<std::int32_t>(),
                                  rows * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    row_base += static_cast<std::int64_t>(rows);

    // Within a chunk the labels must be non-decreasing, which makes each cluster one run and
    // bounds the runs per chunk by the cluster count. Chunks themselves may arrive in any
    // order -- a slice carries its own chunk, so nothing reads across a chunk boundary. That
    // is weaker than requiring the whole corpus to be sorted, and it is what a build phase can
    // actually promise: its batches are ordered spans of an ORDER BY, but the order the
    // batches are handed back in is a race.
    std::size_t start = 0;
    for (std::size_t i = 1; i <= rows; ++i) {
      if (i < rows && labels[i] == labels[start]) { continue; }
      auto const c = labels[start];
      if (c < 0 || c >= _n_clusters) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column '" +
                                 _request.build_cluster_column + "' holds id " + std::to_string(c) +
                                 ", outside the clustering's " + std::to_string(_n_clusters) +
                                 " clusters");
      }
      auto& runs = _chunk_cluster_runs[j];
      if (!runs.empty() && c <= runs.back().cluster) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream] corpus chunk " + std::to_string(j) +
          " is not stored in cluster order: cluster id " + std::to_string(c) + " follows " +
          std::to_string(runs.back().cluster) + ". Materialize the corpus with ORDER BY " +
          _request.build_cluster_column);
      }
      runs.push_back(
        chunk_cluster_run{c, static_cast<std::int64_t>(start), static_cast<std::int64_t>(i)});
      _cluster_rows[static_cast<std::size_t>(c)] += static_cast<std::int64_t>(i - start);
      start = i;
    }
  }
  if (row_base == 0) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column '" +
                             _request.build_cluster_column + "' is empty");
  }
  // Only on the pinned path: there _right_total_rows is the pin's own count and exact, while on
  // the build path it is summed from per-batch sizes that a spilled batch reports in bytes, so a
  // mismatch there would mean the estimate is loose rather than that the labels are wrong.
  if (pin != nullptr && row_base != _right_total_rows) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column has " +
                             std::to_string(row_base) + " rows but the corpus has " +
                             std::to_string(_right_total_rows) +
                             "; both must come from the same pin");
  }

  // Each cluster's nearest clusters, from the centroids alone. This is what a search index
  // cannot precompute: its query side has no cluster structure, while a join's does.
  auto const n_probes = std::clamp<std::int64_t>(_request.n_probes, 1, _n_clusters);
  auto const centers  = vss::list_column_as_dataset_view(_centroids->view(), _request.dim);
  auto knn            = vss::brute_force_knn(res, centers, centers, n_probes, metric, mr);
  stream.synchronize();
  _cluster_neighbors.resize(static_cast<std::size_t>(_n_clusters * n_probes));
  {
    std::vector<std::int64_t> ids(_cluster_neighbors.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(ids.data(),
                                  knn.neighbors->view().data<std::int64_t>(),
                                  ids.size() * sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    for (std::size_t i = 0; i < ids.size(); ++i) {
      _cluster_neighbors[i] = static_cast<std::int32_t>(ids[i]);
    }
  }
  _cluster_index_built = true;

  if (std::getenv("SIRIUS_VECTOR_JOIN_PRUNE_DEBUG") != nullptr) {
    std::size_t empty = 0;
    std::size_t runs  = 0;
    for (auto const rows : _cluster_rows) {
      if (rows == 0) { ++empty; }
    }
    for (auto const& per_chunk : _chunk_cluster_runs) {
      runs += per_chunk.size();
    }
    std::fprintf(stderr,
                 "[vecjoin] cluster index: %ld clusters over %ld rows in %zu chunks, %zu runs, "
                 "%zu empty, n_probes=%ld\n",
                 static_cast<long>(_n_clusters),
                 static_cast<long>(row_base),
                 n_chunks,
                 runs,
                 empty,
                 static_cast<long>(n_probes));
  }
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_stream::execute"};

  auto const* join_in = dynamic_cast<const vector_join_stream_input*>(&input_data);
  if (join_in == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream::execute] expected vector_join_stream_input; got " +
      std::string(typeid(input_data).name()));
  }
  auto* mem_space = join_in->get_gpu_memory_space();
  if (mem_space == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream::execute] no memory space set; prepare_for_processing "
      "was not called");
  }

  auto const left_idx = join_in->left_idx();
  auto const dim      = _request.dim;
  auto const mr       = mem_space->get_default_allocator();

  // Held for the whole task: every corpus chunk is searched against this probe chunk. Staged
  // on the compute stream, so its eventual free is already ordered behind the searches that
  // read it and needs no rebind, unlike the corpus chunks below.
  auto staged_probe  = _probe->stage(left_idx, *mem_space, stream);
  auto const queries = vss::list_column_as_dataset_view(staged_probe.view, dim);
  auto const n_left  = static_cast<std::int64_t>(queries.extent(0));

  // Every mode is served by searching each left row to some depth and then deciding which
  // of those candidates survive. The depth differs: global top-k needs k_global per left
  // row (a single left row may own the entire global answer), and threshold needs a cap on
  // how many in-range neighbours one left row may have, which is what k supplies there.
  auto const k_join = std::min<std::int64_t>(_request.k, _right_total_rows);

  raft::device_resources res{stream};
  auto const exact_unexpanded = _request.search_mode == vss::vector_join_search_mode::exact;
  auto const metric =
    vss::join_selection_distance_type_from_metric(_request.metric, exact_unexpanded);

  // Running [n_left x k_join] accumulator. Seeded by the first right batch, then
  // every later batch is folded in and released, so peak device memory does not
  // grow with the number of right batches.
  std::unique_ptr<cudf::column> acc_neighbors;
  std::unique_ptr<cudf::column> acc_distances;

  // The clustered path replaces the whole-corpus fold below. It is a separate branch rather
  // than a predicate inside it because the two iterate different things: the exhaustive fold
  // walks corpus chunks, while this walks (probe run x neighbouring cluster) pairs.
  if (_centroids != nullptr) {
    auto const dbg = std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr;
    auto phase_t0  = std::chrono::steady_clock::now();
    auto phase     = [&](const char* name) {
      if (!dbg) { return; }
      stream.synchronize();
      auto const now = std::chrono::steady_clock::now();
      std::fprintf(stderr,
                   "[vecjoin-phase] %-22s %8.3f s\n",
                   name,
                   std::chrono::duration<double>(now - phase_t0).count());
      phase_t0 = now;
    };

    ensure_cluster_index(*mem_space, stream, mr, res, metric);
    phase("cluster index");

    auto const n_probes = std::clamp<std::int64_t>(_request.n_probes, 1, _n_clusters);

    // Sorting the probe batch by cluster is what makes a run contiguous, and a contiguous run
    // is what lets a search read a slice instead of a gathered copy. The permutation is kept so
    // the answer can be put back in the caller's row order at the end.
    vss::assignment_spec nearest;
    nearest.n_probes = 1;
    auto assignment  = vss::assign_to_centroids(
      res, staged_probe.view, _centroids->view(), dim, nearest, 0, metric, stream, mr);

    phase("assign probe->cluster");
    auto const order =
      cudf::sorted_order(cudf::table_view{{assignment.cluster_ids->view()}}, {}, {}, stream, mr);
    phase("sort order");
    auto const sorted_probe  = cudf::gather(cudf::table_view{{staged_probe.view}},
                                           order->view(),
                                           cudf::out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           mr);
    auto const sorted_labels = cudf::gather(cudf::table_view{{assignment.cluster_ids->view()}},
                                            order->view(),
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            stream,
                                            mr);
    phase("gather probe vectors");
    auto const probe_sorted_view =
      vss::list_column_as_dataset_view(sorted_probe->get_column(0).view(), dim);

    std::vector<std::int32_t> host_labels(static_cast<std::size_t>(n_left));
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_labels.data(),
                                  sorted_labels->get_column(0).view().data<std::int32_t>(),
                                  host_labels.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    phase("labels to host");

    // One run per probe cluster present in this batch, in sorted order.
    struct probe_run {
      std::int32_t cluster;
      std::int64_t begin;
      std::int64_t end;
    };
    std::vector<probe_run> runs;
    for (std::int64_t b = 0; b < n_left;) {
      auto const c   = host_labels[static_cast<std::size_t>(b)];
      std::int64_t e = b + 1;
      while (e < n_left && host_labels[static_cast<std::size_t>(e)] == c) {
        ++e;
      }
      runs.push_back(probe_run{c, b, e});
      b = e;
    }

    // Inverted: for each corpus cluster, the runs that want it. Built per probe batch because
    // it depends on which clusters this batch's rows landed in; the (cluster -> nearest
    // clusters) table it reads is computed once for the whole join. Having it this way round is
    // what lets the search walk chunks -- a chunk is staged once and every run that wants a
    // slice of it is served before it is released.
    std::vector<std::vector<std::size_t>> runs_wanting(static_cast<std::size_t>(_n_clusters));
    for (std::size_t r = 0; r < runs.size(); ++r) {
      auto const c = runs[r].cluster;
      for (std::int64_t t = 0; t < n_probes; ++t) {
        auto const nc = _cluster_neighbors[static_cast<std::size_t>(c * n_probes + t)];
        if (nc < 0 || nc >= _n_clusters) { continue; }
        if (_cluster_rows[static_cast<std::size_t>(nc)] == 0) { continue; }
        auto& wanting = runs_wanting[static_cast<std::size_t>(nc)];
        // A cluster reached twice would fold the same corpus rows in twice and put a duplicate
        // neighbour in the answer, so a repeated id is dropped rather than trusted not to occur.
        if (std::find(wanting.begin(), wanting.end(), r) == wanting.end()) { wanting.push_back(r); }
      }
    }

    // A run's candidates are the rows of the clusters it wants, which the cluster index already
    // totals -- so a run that cannot supply k is refused before a single search is issued
    // rather than after the merge has silently padded the answer out with misses.
    std::vector<std::int64_t> run_candidates(runs.size(), 0);
    for (std::int64_t nc = 0; nc < _n_clusters; ++nc) {
      for (auto const r : runs_wanting[static_cast<std::size_t>(nc)]) {
        run_candidates[r] += _cluster_rows[static_cast<std::size_t>(nc)];
      }
    }
    for (std::size_t r = 0; r < runs.size(); ++r) {
      if (run_candidates[r] == 0) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe cluster " +
                                 std::to_string(runs[r].cluster) +
                                 " reached no non-empty corpus cluster");
      }
      if (run_candidates[r] < k_join) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe cluster " +
                                 std::to_string(runs[r].cluster) + " reaches only " +
                                 std::to_string(run_candidates[r]) +
                                 " corpus rows, fewer than k=" + std::to_string(k_join) +
                                 "; raise n_probes or cluster with fewer, larger clusters");
      }
    }

    // The prune, made explicit: a chunk holding no cluster any run wants is never staged, so
    // an out-of-core clustered corpus pays no transfer for the part it skips.
    std::vector<std::size_t> needed_chunks;
    for (std::size_t j = 0; j < _chunk_cluster_runs.size(); ++j) {
      for (auto const& slice : _chunk_cluster_runs[j]) {
        if (!runs_wanting[static_cast<std::size_t>(slice.cluster)].empty()) {
          needed_chunks.push_back(j);
          break;
        }
      }
    }

    // knn_merge_parts needs a uniform k across the parts it merges, and a cluster sliced by a
    // chunk boundary can be shorter than k. Padding the short part with a miss -- id -1 at
    // infinite distance -- is what keeps that slice mergeable; the padding can only be selected
    // when a run had fewer than k real candidates, which was refused above. Shifting the ids to
    // corpus row space happens BEFORE this, or the -1 would be shifted into a real row.
    auto pad_part = [&](std::unique_ptr<cudf::column> neighbors,
                        std::unique_ptr<cudf::column> distances,
                        std::int64_t rows,
                        std::int64_t k_eff) {
      if (k_eff >= k_join) { return std::pair{std::move(neighbors), std::move(distances)}; }
      cudf::numeric_scalar<std::int64_t> const miss_id(-1, true, stream);
      cudf::numeric_scalar<float> const miss_distance(
        std::numeric_limits<float>::infinity(), true, stream);
      auto const total = static_cast<cudf::size_type>(rows * k_join);
      std::vector<std::unique_ptr<cudf::column>> target_cols;
      target_cols.push_back(cudf::make_column_from_scalar(miss_id, total, stream, mr));
      target_cols.push_back(cudf::make_column_from_scalar(miss_distance, total, stream, mr));
      cudf::table const target{std::move(target_cols)};

      // Row-major, so element p of the [rows x k_eff] part belongs at
      // (p / k_eff) * k_join + (p % k_eff) in the [rows x k_join] one.
      cudf::numeric_scalar<std::int32_t> const zero32(0, true, stream);
      cudf::numeric_scalar<std::int32_t> const one32(1, true, stream);
      cudf::numeric_scalar<std::int32_t> const keff32(
        static_cast<std::int32_t>(k_eff), true, stream);
      cudf::numeric_scalar<std::int32_t> const kjoin32(
        static_cast<std::int32_t>(k_join), true, stream);
      auto const positions =
        cudf::sequence(static_cast<cudf::size_type>(rows * k_eff), zero32, one32, stream, mr);
      auto const src_row   = cudf::binary_operation(positions->view(),
                                                  keff32,
                                                  cudf::binary_operator::DIV,
                                                  cudf::data_type{cudf::type_id::INT32},
                                                  stream,
                                                  mr);
      auto const in_row    = cudf::binary_operation(positions->view(),
                                                 keff32,
                                                 cudf::binary_operator::MOD,
                                                 cudf::data_type{cudf::type_id::INT32},
                                                 stream,
                                                 mr);
      auto const dest_base = cudf::binary_operation(src_row->view(),
                                                    kjoin32,
                                                    cudf::binary_operator::MUL,
                                                    cudf::data_type{cudf::type_id::INT32},
                                                    stream,
                                                    mr);
      auto const dest      = cudf::binary_operation(dest_base->view(),
                                               in_row->view(),
                                               cudf::binary_operator::ADD,
                                               cudf::data_type{cudf::type_id::INT32},
                                               stream,
                                               mr);
      auto padded          = cudf::scatter(cudf::table_view{{neighbors->view(), distances->view()}},
                                  dest->view(),
                                  target.view(),
                                  stream,
                                  mr);
      auto cols            = padded->release();
      return std::pair{std::move(cols[0]), std::move(cols[1])};
    };

    // One accumulator per run rather than one for the batch: a run's answer is merged from the
    // slices it wants, and those arrive spread across chunks. Their total size is still
    // [n_left x k_join] -- the runs partition the batch -- so this costs no more device memory
    // than the single accumulator the exhaustive fold keeps.
    std::vector<std::unique_ptr<cudf::column>> acc_n(runs.size());
    std::vector<std::unique_ptr<cudf::column>> acc_d(runs.size());

    // Staging runs on its own stream so the next needed chunk's H2D overlaps this one's
    // compute, exactly as in the exhaustive fold below.
    std::optional<rmm::cuda_stream> staging_stream;
    if (_corpus->is_streaming()) { staging_stream.emplace(); }
    auto const stage_on = staging_stream ? staging_stream->view() : stream;
    std::vector<cucascade::read_only_data_batch> borrowed;
    auto release_staged = [&](staged_vector_chunk& chunk) {
      if (chunk.owner) {
        auto mut = chunk.owner->to_mutable();
        mut.rebind_stream(stream);
      }
      if (chunk.reader) { borrowed.push_back(std::move(*chunk.reader)); }
      chunk = staged_vector_chunk{};
    };

    std::int64_t scanned_pairs = 0;
    auto prefetched            = needed_chunks.empty()
                                   ? staged_vector_chunk{}
                                   : _corpus->stage(needed_chunks[0], *mem_space, stage_on);

    for (std::size_t ci = 0; ci < needed_chunks.size(); ++ci) {
      auto const j          = needed_chunks[ci];
      auto staged           = std::move(prefetched);
      auto const chunk_view = vss::list_column_as_dataset_view(staged.view, dim);
      auto const chunk_base = _chunk_row_base[j];

      for (auto const& slice : _chunk_cluster_runs[j]) {
        auto const& wanting = runs_wanting[static_cast<std::size_t>(slice.cluster)];
        if (wanting.empty()) { continue; }
        // The slice was cut from the cluster column's chunk j; this is the first point at
        // which the vector column's chunk j is resident and its row count exactly known. The
        // two are the same pin's row groups, so a mismatch is a broken invariant rather than
        // a user error -- but it would read past the end of the chunk, so it is checked.
        if (slice.end > static_cast<std::int64_t>(chunk_view.extent(0))) {
          throw std::runtime_error(
            "[sirius_physical_vector_join_stream] cluster column chunk " + std::to_string(j) +
            " describes row " + std::to_string(slice.end) + " but the vector column's chunk " +
            "holds " + std::to_string(chunk_view.extent(0)) + "; both must come from the same pin");
        }
        auto const slice_rows = slice.end - slice.begin;
        auto const slice_view =
          raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
            chunk_view.data_handle() + slice.begin * dim, slice_rows, dim);
        auto const k_eff = std::min<std::int64_t>(k_join, slice_rows);

        for (auto const r : wanting) {
          auto const run_rows = runs[r].end - runs[r].begin;
          auto const queries_run =
            raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
              probe_sorted_view.data_handle() + runs[r].begin * dim, run_rows, dim);

          auto knn = vss::brute_force_knn(res, slice_view, queries_run, k_eff, metric, mr);
          scanned_pairs += slice_rows * run_rows;

          // Neighbour ids come back local to the slice; the slice's own start in corpus row
          // space is the base that makes them corpus row ids, exactly as the chunk offset does
          // in the exhaustive fold.
          cudf::numeric_scalar<std::int64_t> const base(chunk_base + slice.begin, true, stream);
          auto shifted = cudf::binary_operation(knn.neighbors->view(),
                                                base,
                                                cudf::binary_operator::ADD,
                                                cudf::data_type{cudf::type_id::INT64},
                                                stream,
                                                mr);
          auto part    = pad_part(std::move(shifted), std::move(knn.distances), run_rows, k_eff);
          if (!acc_n[r]) {
            acc_n[r] = std::move(part.first);
            acc_d[r] = std::move(part.second);
            continue;
          }
          auto const stacked_d = cudf::concatenate(
            std::vector<cudf::column_view>{acc_d[r]->view(), part.second->view()}, stream, mr);
          auto const stacked_n = cudf::concatenate(
            std::vector<cudf::column_view>{acc_n[r]->view(), part.first->view()}, stream, mr);
          auto merged = vss::knn_merge_parts_topk(
            res, stacked_d->view(), stacked_n->view(), run_rows, 2, k_join, stream, mr);
          acc_n[r] = std::move(merged.neighbors);
          acc_d[r] = std::move(merged.distances);
        }
      }

      // The searches above are issued, not finished. Staging the next needed chunk now runs its
      // H2D while the GPU works on this one.
      prefetched = (ci + 1 < needed_chunks.size())
                     ? _corpus->stage(needed_chunks[ci + 1], *mem_space, stage_on)
                     : staged_vector_chunk{};
      release_staged(staged);
    }

    // What the pruning did, reported as a value rather than only a print: an approximate join
    // that skipped nothing returns a correct answer, so nothing in the result set distinguishes
    // it from one that pruned hard. This is what a test can assert on.
    auto const exhaustive_pairs = n_left * _right_total_rows;
    if (_sirius_ctx != nullptr) {
      _sirius_ctx->record_vector_join_prune(static_cast<std::uint64_t>(scanned_pairs),
                                            static_cast<std::uint64_t>(exhaustive_pairs),
                                            needed_chunks.size(),
                                            _chunk_cluster_runs.size());
    }

    if (std::getenv("SIRIUS_VECTOR_JOIN_PRUNE_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[vecjoin] batch: %zu probe runs, %zu of %zu corpus chunks staged, "
                   "%lld of %lld probe-row x corpus-row pairs scored (%.2f%%)\n",
                   runs.size(),
                   needed_chunks.size(),
                   _chunk_cluster_runs.size(),
                   static_cast<long long>(scanned_pairs),
                   static_cast<long long>(exhaustive_pairs),
                   exhaustive_pairs == 0 ? 0.0
                                         : 100.0 * static_cast<double>(scanned_pairs) /
                                             static_cast<double>(exhaustive_pairs));
    }

    phase("search+merge runs");
    // Runs are emitted in sorted order; concatenating them rebuilds the sorted answer, and
    // scattering by the sort permutation puts it back in the caller's row order.
    std::vector<cudf::column_view> nviews;
    std::vector<cudf::column_view> dviews;
    nviews.reserve(acc_n.size());
    dviews.reserve(acc_d.size());
    for (std::size_t r = 0; r < acc_n.size(); ++r) {
      if (!acc_n[r]) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe cluster " +
                                 std::to_string(runs[r].cluster) +
                                 " was answered by no corpus slice");
      }
      nviews.push_back(acc_n[r]->view());
      dviews.push_back(acc_d[r]->view());
    }
    auto sorted_n = cudf::concatenate(nviews, stream, mr);
    auto sorted_d = cudf::concatenate(dviews, stream, mr);
    phase("concat runs");

    // The accumulator is [n_left x k_join] row-major, so a row moves as a block of k_join
    // entries: expand the row permutation to element positions before scattering.
    cudf::numeric_scalar<std::int32_t> const kscalar(
      static_cast<std::int32_t>(k_join), true, stream);
    cudf::numeric_scalar<std::int32_t> const zero32(0, true, stream);
    cudf::numeric_scalar<std::int32_t> const one32(1, true, stream);
    auto const positions =
      cudf::sequence(static_cast<cudf::size_type>(n_left * k_join), zero32, one32, stream, mr);
    auto const src_row   = cudf::binary_operation(positions->view(),
                                                kscalar,
                                                cudf::binary_operator::DIV,
                                                cudf::data_type{cudf::type_id::INT32},
                                                stream,
                                                mr);
    auto const in_row    = cudf::binary_operation(positions->view(),
                                               kscalar,
                                               cudf::binary_operator::MOD,
                                               cudf::data_type{cudf::type_id::INT32},
                                               stream,
                                               mr);
    auto const dest_row  = cudf::gather(cudf::table_view{{order->view()}},
                                       src_row->view(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);
    auto const dest_base = cudf::binary_operation(dest_row->get_column(0).view(),
                                                  kscalar,
                                                  cudf::binary_operator::MUL,
                                                  cudf::data_type{cudf::type_id::INT32},
                                                  stream,
                                                  mr);
    auto const dest      = cudf::binary_operation(dest_base->view(),
                                             in_row->view(),
                                             cudf::binary_operator::ADD,
                                             cudf::data_type{cudf::type_id::INT32},
                                             stream,
                                             mr);
    auto scattered       = cudf::scatter(cudf::table_view{{sorted_n->view(), sorted_d->view()}},
                                   dest->view(),
                                   cudf::table_view{{sorted_n->view(), sorted_d->view()}},
                                   stream,
                                   mr);
    auto cols            = scattered->release();
    acc_neighbors        = std::move(cols[0]);
    acc_distances        = std::move(cols[1]);
    phase("scatter to row order");
  } else {
    auto const n_chunks = _corpus->num_chunks();
    std::int64_t offset = 0;  // running base of the current chunk in right-table row space

    // Staging runs on its own stream so chunk j+1's H2D overlaps chunk j's compute. The
    // converter host-synchronizes at the end of its copy, so what is actually overlapped is
    // "host blocked on the copy" against "GPU busy with the previous chunk" -- the compute
    // below is issued asynchronously and does not block the host. Only worth a stream when
    // the corpus actually streams; a GPU-tier pin stages nothing.
    std::optional<rmm::cuda_stream> staging_stream;
    if (_corpus->is_streaming()) { staging_stream.emplace(); }
    auto const stage_on = staging_stream ? staging_stream->view() : stream;

    auto prefetched =
      n_chunks > 0 ? _corpus->stage(0, *mem_space, stage_on) : staged_vector_chunk{};

    auto advance = [&](std::size_t next) {
      prefetched =
        next < n_chunks ? _corpus->stage(next, *mem_space, stage_on) : staged_vector_chunk{};
    };

    // Borrowed build-side batches whose read locks have to outlive the searches reading them.
    // A borrow costs no device memory -- the batch was already resident, which is why it was
    // borrowed rather than copied -- so holding it to the end of the task only means the
    // downgrade executor cannot spill that batch while this task is still reading it.
    std::vector<cucascade::read_only_data_batch> borrowed;

    // The staged copy is read by kernels that are still pending on the compute stream, but it
    // was allocated on the staging stream, so dropping it here would hand the buffer back to
    // RMM's free list for that other stream while a kernel is still reading it. Rebinding
    // moves the deallocation onto the compute stream, where it is ordered behind that kernel.
    // Only ever applied to a copy this task made: `owner` is null for a borrow, and upgrading a
    // borrowed batch to mutable while its own read lock is held would deadlock against itself.
    auto release_staged = [&](staged_vector_chunk& chunk) {
      if (chunk.owner) {
        auto mut = chunk.owner->to_mutable();
        mut.rebind_stream(stream);
      }
      if (chunk.reader) { borrowed.push_back(std::move(*chunk.reader)); }
      chunk = staged_vector_chunk{};
    };

    for (std::size_t j = 0; j < n_chunks; ++j) {
      // Held for this iteration only; released at the bottom once its compute is ordered,
      // which is what keeps device memory bounded by the chunks in flight, not the corpus.
      auto staged           = std::move(prefetched);
      auto const dataset    = vss::list_column_as_dataset_view(staged.view, dim);
      auto const batch_rows = static_cast<std::int64_t>(dataset.extent(0));
      if (batch_rows == 0) {
        advance(j + 1);
        continue;
      }

      // knn_merge_parts requires a uniform k across the parts it merges. Every batch
      // therefore has to supply k_join candidates; a batch shorter than k_join cannot,
      // and padding it row-major is not expressible without a dedicated kernel.
      if (n_chunks > 1 && batch_rows < k_join) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream] right batch " + std::to_string(j) + " has " +
          std::to_string(batch_rows) + " rows, fewer than k=" + std::to_string(k_join) +
          "; repartition the right table so every batch holds at least k rows");
      }
      auto const k_eff = std::min<std::int64_t>(k_join, batch_rows);

      auto knn = vss::brute_force_knn(res, dataset, queries, k_eff, metric, mr);

      // The search above is issued, not finished. Staging the next chunk now runs its H2D
      // while the GPU works on this one; the host blocks inside the converter, the device
      // does not.
      advance(j + 1);
      release_staged(staged);

      std::unique_ptr<cudf::column> neighbors = std::move(knn.neighbors);
      auto const chunk_base                   = offset;
      offset += batch_rows;
      if (chunk_base != 0) {
        cudf::numeric_scalar<std::int64_t> const off_scalar(chunk_base, true, stream);
        neighbors = cudf::binary_operation(neighbors->view(),
                                           off_scalar,
                                           cudf::binary_operator::ADD,
                                           cudf::data_type{cudf::type_id::INT64},
                                           stream,
                                           mr);
      }

      if (!acc_neighbors) {
        acc_neighbors = std::move(neighbors);
        acc_distances = std::move(knn.distances);
        continue;
      }

      // Fold: stack accumulator and this batch part-major and merge them back down to
      // k_join. This is knn_merge_parts with n_parts = 2 -- the same kernel the split
      // design called once over every partial, applied incrementally instead.
      auto const stacked_distances = cudf::concatenate(
        std::vector<cudf::column_view>{acc_distances->view(), knn.distances->view()}, stream, mr);
      auto const stacked_neighbors = cudf::concatenate(
        std::vector<cudf::column_view>{acc_neighbors->view(), neighbors->view()}, stream, mr);

      auto merged   = vss::knn_merge_parts_topk(res,
                                              stacked_distances->view(),
                                              stacked_neighbors->view(),
                                              n_left,
                                              /*n_parts=*/2,
                                              k_join,
                                              stream,
                                              mr);
      acc_neighbors = std::move(merged.neighbors);
      acc_distances = std::move(merged.distances);
    }
  }

  if (!acc_neighbors) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] right table produced no rows to join against");
  }

  // The fold is mode-independent; only which of its candidates survive is not.
  vss::shaped_join_result shaped;
  switch (_request.mode) {
    case vss::vector_join_mode::global_top_k: {
      shaped = vss::shape_global_top_k(
        acc_neighbors->view(), acc_distances->view(), n_left, k_join, k_join, stream, mr);
      break;
    }
    case vss::vector_join_mode::threshold: {
      // The kernel works in distance space. For cosine with a similarity threshold the
      // user's "score >= eps" is the same set as "distance <= 1 - eps"; for a distance
      // threshold it is eps directly.
      auto const max_distance = _request.output_type == vss::vector_join_output_type::similarity
                                  ? static_cast<float>(1.0 - _request.eps)
                                  : static_cast<float>(_request.eps);
      bool truncated          = false;
      shaped                  = vss::shape_threshold(acc_neighbors->view(),
                                    acc_distances->view(),
                                    n_left,
                                    k_join,
                                    max_distance,
                                    truncated,
                                    stream,
                                    mr);
      if (truncated) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream] threshold join truncated: at least one left row "
          "has k=" +
          std::to_string(k_join) +
          " neighbours inside the threshold, so pairs beyond k were never searched for. Raise k "
          "or tighten eps.");
      }
      break;
    }
    case vss::vector_join_mode::per_row_top_k:
    default: {
      shaped = vss::shape_per_row_top_k(
        std::move(acc_neighbors), std::move(acc_distances), n_left, k_join, stream, mr);
      break;
    }
  }

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(3);
  out_cols.push_back(std::move(shaped.left_rows));
  out_cols.push_back(std::move(shaped.neighbors));
  out_cols.push_back(std::move(shaped.distances));
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));

  auto batch = sirius::make_data_batch(std::move(out_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<partitioned_operator_data>(std::move(batches), left_idx);
}

//===----------------------------------------------------------------------===//
// Sink
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::sink(const operator_data& output_data,
                                              rmm::cuda_stream_view /*stream*/)
{
  auto const& part         = dynamic_cast<const partitioned_operator_data&>(output_data);
  auto const partition_idx = part.get_partition_idx();
  for (auto& batch : part.get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      auto* consumer =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (consumer == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream::sink] next operator is not a partition consumer");
      }
      consumer->push_data_batch_partitioned(
        next_port_info.next_operator_port_name, batch, partition_idx);
    }
  }
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_vector_join_stream::per_left_batch_estimate(std::size_t left_idx) const
{
  // Live at once: the accumulator, one batch's partial, and the stacked pair the
  // merge reads (2x), plus the merge output. Six [n_left x k] blocks covers it, with
  // the same 1 MiB floor the split design used. Notably independent of the right
  // batch count -- the split design's merge stage scaled with it.
  auto const n_left = _probe->chunk_rows(left_idx);
  auto const k      = static_cast<std::size_t>(std::max<std::int64_t>(_request.k, 1));
  auto const block  = n_left * k * (sizeof(std::int64_t) + sizeof(float));

  // cuVS tiles the pairwise distances against a bounded internal workspace, so its
  // scratch does not scale with the search shape -- the two figures recorded on the
  // split path (~35 MB L2, ~209 MB cosine, both at 50k x 50k) are workspace sizes, not
  // a function of n. Modelled as a metric-dependent constant on that basis. Leaving it
  // out entirely is what let tasks reserve far less than they used; these figures are
  // observations from one shape, so treat them as a floor to refine, not a derivation.
  auto const cuvs_scratch =
    (_request.metric == "cosine") ? (std::size_t{220} << 20) : (std::size_t{40} << 20);

  // A streamed corpus also holds the staged chunk itself; it is drawn from this task's
  // budget, so it has to be reserved here too.
  std::size_t staged_chunk = 0;
  if (_corpus && _corpus->is_streaming()) { staged_chunk = _max_chunk_bytes; }
  // A streamed probe side holds its chunk for the whole task, so it is live alongside the
  // corpus chunk rather than instead of it.
  if (_probe && _probe->is_streaming()) { staged_chunk += _max_probe_chunk_bytes; }

  return (block * 6) + cuvs_scratch + staged_chunk + (std::size_t{1} << 20);
}

std::size_t sirius_physical_vector_join_stream::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // As in the split design, cuVS's on-demand search scratch is not modelled here.
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_stream::params_to_string() const
{
  return _request.left.table + "(" + _request.left.column + ") x " + _request.right.table + "(" +
         _request.right.column + ") metric=" + _request.metric +
         " k=" + std::to_string(_request.k) +
         " mode=" + std::to_string(static_cast<int>(_request.mode)) + " streaming";
}

}  // namespace sirius::op
