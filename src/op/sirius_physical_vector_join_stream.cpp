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
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"
#include "vss/join_result_shaping.hpp"
#include "vss/knn_merge.hpp"
#include "vss/pinned_column.hpp"
#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include "vss/vector_clustering.hpp"
#include "vss/pinned_column.hpp"
#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>

#include "data/sirius_converter_registry.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <array>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
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
      throw std::runtime_error("[sirius_physical_vector_join_stream] host-tier pin is missing "
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
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] corpus chunk " + std::to_string(i) + " needs " +
        std::to_string(bytes) + " bytes device-side, which exceeds this task's budget");
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
  const cudf::column* centroids)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_STREAM, std::move(types), estimated_cardinality),
    _build_side(std::move(build_side)),
    _probe_side(std::move(probe_side)),
    _request(std::move(request)),
    _scan_manager(scan_manager),
    _centroids(centroids)
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

  const auto* left_pin =
    _probe_side ? nullptr
                : _scan_manager->find_pinned_entry_for_duckdb_table(
                    left.catalog, left.schema, left.table);
  // The corpus comes from the build port on the build path, so only the probe side has to be
  // pinned there.
  const auto* right_pin =
    _build_side ? nullptr
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
    _probe = make_gpu_pinned_chunk_source(
      *left_pin, left.column, vss::pinned_entry_gpu_space(*left_pin));
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
        if (chunk) {
          _max_chunk_bytes = std::max(_max_chunk_bytes, chunk->column_size(col));
        }
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
void sirius_physical_vector_join_stream::ensure_cluster_ranges(
  ::cucascade::memory::memory_space& space, rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> const lock(_op_mutex);
  if (!_chunk_cluster_ranges.empty()) { return; }

  // Read once for the whole join rather than per probe batch: the corpus is fixed, and a
  // chunk's span is what every batch tests against. Deferred to here because staging needs a
  // memory space, which only a running task has. Each chunk is staged and released, so this
  // costs one chunk of device memory whatever the corpus size.
  auto const* pin = _scan_manager->find_pinned_entry_for_duckdb_table(
    _request.right.catalog, _request.right.schema, _request.right.table);
  if (pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] clustering needs the corpus table '" +
      _request.right.table + "' pinned");
  }

  auto const n_chunks =
    vss::pinned_column_chunk_count(*pin, _request.build_cluster_column);
  if (n_chunks != _corpus->num_chunks()) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] cluster column '" + _request.build_cluster_column +
      "' has " + std::to_string(n_chunks) + " chunks but the vector column has " +
      std::to_string(_corpus->num_chunks()) + "; both must come from the same pin");
  }

  std::vector<std::pair<std::int32_t, std::int32_t>> ranges;
  ranges.reserve(n_chunks);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto staged = vss::stage_pinned_column_chunk(
      *pin, _request.build_cluster_column, i, space, stream, batch_telemetry());
    if (staged.view.size() == 0) {
      ranges.emplace_back(1, 0);  // an empty span matches no cluster
      continue;
    }
    auto const lo = cudf::reduce(staged.view,
                                 *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::INT32},
                                 stream);
    auto const hi = cudf::reduce(staged.view,
                                 *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::INT32},
                                 stream);
    stream.synchronize();
    ranges.emplace_back(static_cast<cudf::numeric_scalar<std::int32_t> const&>(*lo).value(stream),
                        static_cast<cudf::numeric_scalar<std::int32_t> const&>(*hi).value(stream));
  }
  _chunk_cluster_ranges = std::move(ranges);
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

  // Which clusters this batch's rows were assigned to. Computed per batch because it depends on
  // the probe rows, and kept on the host as a sorted set so the per-chunk test below is a binary
  // search rather than a device round-trip.
  if (_centroids != nullptr) { ensure_cluster_ranges(*mem_space, stream); }

  std::vector<std::int32_t> wanted;
  if (_centroids != nullptr) {
    vss::assignment_spec assign_spec;
    assign_spec.n_probes = _request.n_probes;
    auto assignment      = vss::assign_to_centroids(res,
                                               staged_probe.view,
                                               _centroids->view(),
                                               dim,
                                               assign_spec,
                                               /*row_id_base=*/0,
                                               metric,
                                               stream,
                                               mr);
    auto const distinct  = cudf::distinct(cudf::table_view{{assignment.cluster_ids->view()}},
                                         std::vector<cudf::size_type>{0},
                                         cudf::duplicate_keep_option::KEEP_ANY,
                                         cudf::null_equality::EQUAL,
                                         cudf::nan_equality::ALL_EQUAL,
                                         stream,
                                         mr);
    auto const& ids      = distinct->get_column(0);
    wanted.resize(static_cast<std::size_t>(ids.size()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(wanted.data(),
                                  ids.view().data<std::int32_t>(),
                                  wanted.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    std::sort(wanted.begin(), wanted.end());
  }

  // A chunk is visited when any wanted cluster falls inside its span. Conservative by design:
  // a span covering ids the chunk does not actually hold costs a needless visit, never a
  // missed neighbour.
  auto chunk_is_wanted = [&](std::size_t j) {
    if (_centroids == nullptr) { return true; }
    auto const [lo, hi] = _chunk_cluster_ranges[j];
    if (lo > hi) { return false; }
    auto const it = std::lower_bound(wanted.begin(), wanted.end(), lo);
    return it != wanted.end() && *it <= hi;
  };

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

  auto prefetched = n_chunks > 0 ? _corpus->stage(0, *mem_space, stage_on) : staged_vector_chunk{};

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
    // Skipping still has to advance `offset`: it is the base that turns a chunk-local neighbour
    // index into a right-table row id, so dropping a chunk without counting its rows would
    // silently shift every id found after it.
    if (!chunk_is_wanted(j)) {
      offset += static_cast<std::int64_t>(_corpus->chunk_rows(j));
      release_staged(prefetched);
      advance(j + 1);
      continue;
    }

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
    auto const stacked_distances =
      cudf::concatenate(std::vector<cudf::column_view>{acc_distances->view(), knn.distances->view()},
                        stream,
                        mr);
    auto const stacked_neighbors =
      cudf::concatenate(std::vector<cudf::column_view>{acc_neighbors->view(), neighbors->view()},
                        stream,
                        mr);

    auto merged = vss::knn_merge_parts_topk(res,
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

  if (!acc_neighbors) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] right table produced no rows to join against");
  }

  // The fold is mode-independent; only which of its candidates survive is not.
  vss::shaped_join_result shaped;
  switch (_request.mode) {
    case vss::vector_join_mode::global_top_k: {
      shaped = vss::shape_global_top_k(acc_neighbors->view(),
                                       acc_distances->view(),
                                       n_left,
                                       k_join,
                                       k_join,
                                       stream,
                                       mr);
      break;
    }
    case vss::vector_join_mode::threshold: {
      // The kernel works in distance space. For cosine with a similarity threshold the
      // user's "score >= eps" is the same set as "distance <= 1 - eps"; for a distance
      // threshold it is eps directly.
      auto const max_distance =
        _request.output_type == vss::vector_join_output_type::similarity
          ? static_cast<float>(1.0 - _request.eps)
          : static_cast<float>(_request.eps);
      bool truncated = false;
      shaped         = vss::shape_threshold(acc_neighbors->view(),
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
  if (_corpus && _corpus->is_streaming()) {
    staged_chunk = _max_chunk_bytes;
  }
  // A streamed probe side holds its chunk for the whole task, so it is live alongside the
  // corpus chunk rather than instead of it.
  if (_probe && _probe->is_streaming()) {
    staged_chunk += _max_probe_chunk_bytes;
  }

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
