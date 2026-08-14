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

#include "vss/sirius_physical_vector_join_materialize.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/pinned_column.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_vector_join_materialize::sirius_physical_vector_join_materialize(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager,
  std::shared_ptr<sirius::vss::materialized_side_buffer> build_side,
  std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager),
    _build_side(std::move(build_side)),
    _probe_side(std::move(probe_side))
{
}

std::vector<std::unique_ptr<cudf::column>>
sirius_physical_vector_join_materialize::build_side_output_columns(
  std::size_t num_output_columns,
  rmm::cuda_stream_view stream,
  ::cucascade::memory::memory_space& space)
{
  // The fold numbered neighbour ids by walking this snapshot, so concatenating in the same
  // order is what makes id i address row i. Re-deriving the order here -- from the pin, or by
  // asking the repository again -- is the bug this shares a handle to avoid: the batches
  // arrive in scan-completion order, which is not the table's row order.
  auto const ids = _build_side->batch_ids();
  auto* repo     = _build_side->repo();
  if (repo == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] build side has no snapshot; the join stage "
      "should have taken it before any row reached this operator");
  }

  // Held until every concatenate below has copied out of them.
  std::vector<cucascade::read_only_data_batch> readers;
  std::vector<std::shared_ptr<cucascade::data_batch>> restaged;
  std::vector<std::shared_ptr<cucascade::memory::reservation>> reservations;
  std::vector<cudf::table_view> tables;
  //! Index of the first output column in the matching entry of `tables`: a borrowed batch is
  //! the whole scan batch, whose column 0 is the vector; a re-staged one holds the output
  //! columns alone.
  std::vector<cudf::size_type> first_output_col;
  readers.reserve(ids.size());
  tables.reserve(ids.size());
  first_output_col.reserve(ids.size());

  // Output columns sit after the vector column, which the corpus scan projects first.
  std::vector<std::size_t> out_cols(num_output_columns);
  for (std::size_t c = 0; c < num_output_columns; ++c) {
    out_cols[c] = c + 1;
  }

  for (auto const id : ids) {
    auto batch = repo->get_data_batch_by_id(id, /*partition_idx=*/0);
    if (!batch) {
      throw std::runtime_error("[sirius_physical_vector_join_materialize] build-side batch " +
                               std::to_string(id) + " is no longer in the repository");
    }
    auto ro = batch->to_read_only();
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      tables.push_back(sirius::get_cudf_table_view(ro));
      first_output_col.push_back(1);
      readers.push_back(std::move(ro));
      continue;
    }

    // Spilled: bring back the output columns alone. The vector column is the bulk of the
    // batch and nothing here reads it.
    const auto* data = ro.get_data();
    if (data == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_materialize] build-side batch has no data representation");
    }
    auto data_rep    = data->cast<cucascade::host_data_representation>().slice(out_cols);
    auto const bytes = data_rep->get_size_in_bytes();
    std::shared_ptr<cucascade::memory::reservation> reservation{
      space.make_reservation_or_null(bytes)};
    if (!reservation) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_materialize] build-side output columns need " +
        std::to_string(bytes) + " bytes device-side, which exceeds the available budget");
    }
    auto const batch_id = sirius::get_next_batch_id();
    auto staged         = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
    {
      auto mut = staged->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), *reservation, stream);
    }
    tables.push_back(sirius::get_cudf_table_view(*staged));
    first_output_col.push_back(0);
    restaged.push_back(std::move(staged));
    reservations.push_back(std::move(reservation));
    readers.push_back(std::move(ro));
  }

  auto const mr = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(num_output_columns);
  for (std::size_t c = 0; c < num_output_columns; ++c) {
    std::vector<cudf::column_view> views;
    views.reserve(tables.size());
    for (std::size_t b = 0; b < tables.size(); ++b) {
      views.push_back(tables[b].column(first_output_col[b] + static_cast<cudf::size_type>(c)));
    }
    cols.push_back(cudf::concatenate(views, stream, mr));
  }
  return cols;
}

std::vector<std::vector<cudf::column_view>>
sirius_physical_vector_join_materialize::probe_side_output_views(
  std::size_t num_output_columns,
  rmm::cuda_stream_view stream,
  ::cucascade::memory::memory_space& space)
{
  auto const ids = _probe_side->batch_ids();
  auto* repo     = _probe_side->repo();
  if (repo == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] probe side has no snapshot; the join stage "
      "should have taken it before any row reached this operator");
  }

  // Output columns sit after the vector column, which the probe scan projects first.
  std::vector<std::size_t> out_cols(num_output_columns);
  for (std::size_t c = 0; c < num_output_columns; ++c) {
    out_cols[c] = c + 1;
  }

  std::vector<std::vector<cudf::column_view>> per_column(num_output_columns);
  for (auto const id : ids) {
    auto batch = repo->get_data_batch_by_id(id, /*partition_idx=*/0);
    if (!batch) {
      throw std::runtime_error("[sirius_physical_vector_join_materialize] probe-side batch " +
                               std::to_string(id) + " is no longer in the repository");
    }
    auto ro = batch->to_read_only();
    cudf::table_view table;
    cudf::size_type first = 0;
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      table = sirius::get_cudf_table_view(ro);
      first = 1;
    } else {
      const auto* data = ro.get_data();
      if (data == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_materialize] probe-side batch has no data representation");
      }
      auto data_rep    = data->cast<cucascade::host_data_representation>().slice(out_cols);
      auto const bytes = data_rep->get_size_in_bytes();
      std::shared_ptr<cucascade::memory::reservation> reservation{
        space.make_reservation_or_null(bytes)};
      if (!reservation) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_materialize] probe-side output columns need " +
          std::to_string(bytes) + " bytes device-side, which exceeds the available budget");
      }
      auto const batch_id = sirius::get_next_batch_id();
      auto staged         = cucascade::data_batch::make(
        batch_id,
        std::move(data_rep),
        telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
      {
        auto mut = staged->to_mutable();
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), *reservation, stream);
      }
      table = sirius::get_cudf_table_view(*staged);
      _probe_restaged.push_back(std::move(staged));
      _probe_reservations.push_back(std::move(reservation));
    }
    for (std::size_t c = 0; c < num_output_columns; ++c) {
      per_column[c].push_back(table.column(first + static_cast<cudf::size_type>(c)));
    }
    _probe_readers.push_back(std::move(ro));
  }
  return per_column;
}

void sirius_physical_vector_join_materialize::ensure_initialized(
  rmm::cuda_stream_view stream, ::cucascade::memory::memory_space& space)
{
  std::lock_guard<std::mutex> lg(_init_mutex);
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_materialize] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  const auto* left_pin =
    _probe_side ? nullptr
                : _scan_manager->find_pinned_entry_for_duckdb_table(
                    left.catalog, left.schema, left.table);
  const auto* right_pin =
    _build_side ? nullptr
                : _scan_manager->find_pinned_entry_for_duckdb_table(
                    right.catalog, right.schema, right.table);
  if ((!_probe_side && left_pin == nullptr) || (!_build_side && right_pin == nullptr)) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] left/right table is no longer pinned");
  }
  // Staged rather than aliased so a HOST-tier pin works: output columns are the small
  // non-vector columns, and the right side is concatenated on device below regardless, so
  // staging does not change the memory profile for a GPU-tier pin (where it is zero-copy).
  if (_probe_side) {
    _left_output_cols = probe_side_output_views(left.output_columns.size(), stream, space);
  } else {
    _left_output_cols.resize(left.output_columns.size());
    for (std::size_t c = 0; c < left.output_columns.size(); ++c) {
      auto staged = vss::stage_pinned_column(
        *left_pin, left.output_columns[c], space, stream, batch_telemetry());
      _left_output_cols[c] = staged.views;
      _staged_left.push_back(std::move(staged));
    }
  }

  // Right output columns concatenated once across batches, so a global right id
  // gathers straight into row i. Small columns (not the vectors), so cheap.
  auto const mr = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> right_cols;
  right_cols.reserve(right.output_columns.size());
  if (_build_side) {
    right_cols = build_side_output_columns(right.output_columns.size(), stream, space);
  } else {
    for (auto const& name : right.output_columns) {
      auto staged = vss::stage_pinned_column(*right_pin, name, space, stream, batch_telemetry());
      right_cols.push_back(cudf::concatenate(staged.views, stream, mr));
    }
  }
  _right_output_concat = std::make_unique<cudf::table>(std::move(right_cols));

  _initialized = true;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::get_next_task_input_data()
{
  // One task per partition (= one left batch): drain its merge outputs.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (_current_partition_index >= repo->num_partitions()) { return nullptr; }

  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (true) {
    auto batch = repo->pop_next_data_batch(_current_partition_index);
    if (!batch) { break; }
    all_batches.push_back(std::move(batch));
  }
  auto const partition_idx = _current_partition_index++;
  if (all_batches.empty()) { return nullptr; }
  return std::make_unique<partitioned_operator_data>(std::move(all_batches), partition_idx);
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_materialize::execute"};

  auto const& input         = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const partition_idx  = input.get_partition_idx();  // = left batch index
  auto const& input_batches = input.get_read_only_batches();

  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (space == nullptr) { space = batch.get_memory_space(); }
  }
  if (input_batches.empty() || space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  ensure_initialized(stream, *space);
  auto const mr = space->get_default_allocator();

  // Merge emits one result batch per partition; concatenate defensively if more.
  std::vector<cudf::column_view> left_row_views;
  std::vector<cudf::column_view> neighbor_views;
  std::vector<cudf::column_view> distance_views;
  for (auto const& ro : input_batches) {
    auto const tv = sirius::get_cudf_table_view(ro);
    left_row_views.push_back(tv.column(0));  // INT32 row index into the left batch
    neighbor_views.push_back(tv.column(1));  // INT64 global right id
    distance_views.push_back(tv.column(2));  // FLOAT32 distance
  }
  std::unique_ptr<cudf::column> left_row_owned;
  std::unique_ptr<cudf::column> neighbor_owned;
  std::unique_ptr<cudf::column> distance_owned;
  cudf::column_view left_row_view = left_row_views.front();
  cudf::column_view neighbor_view = neighbor_views.front();
  cudf::column_view distance_view = distance_views.front();
  if (input_batches.size() > 1) {
    left_row_owned = cudf::concatenate(left_row_views, stream, mr);
    neighbor_owned = cudf::concatenate(neighbor_views, stream, mr);
    distance_owned = cudf::concatenate(distance_views, stream, mr);
    left_row_view  = left_row_owned->view();
    neighbor_view  = neighbor_owned->view();
    distance_view  = distance_owned->view();
  }

  // Left columns gathered by the left row each pair belongs to. This used to repeat every
  // left row k times, which assumed a fixed k per row; threshold and global top-k are ragged
  // by construction, so the join stage now names the left row for each pair instead.
  // Either side can contribute no columns once projection pushdown has narrowed the output --
  // `SELECT count(*)`, or a query reading only the score -- and gathering a table of no columns
  // is not something cudf defines, so the gather is skipped rather than fed an empty table.
  std::vector<cudf::column_view> left_batch_cols;
  left_batch_cols.reserve(_left_output_cols.size());
  for (auto const& per_batch : _left_output_cols) {
    left_batch_cols.push_back(per_batch[partition_idx]);
  }
  std::unique_ptr<cudf::table> left_repeated;
  if (!left_batch_cols.empty()) {
    left_repeated = cudf::gather(cudf::table_view(left_batch_cols),
                                 left_row_view,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
  }

  // Right columns gathered by the global neighbor id.
  std::unique_ptr<cudf::table> right_gathered;
  if (_right_output_concat->num_columns() > 0) {
    right_gathered = cudf::gather(_right_output_concat->view(),
                                  neighbor_view,
                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr);
  }

  // Score: distance, or cosine similarity = max(0, 1 - distance).
  std::unique_ptr<cudf::column> score;
  if (_request.metric == "cosine") {
    cudf::numeric_scalar<float> const lo(0.0F, true, stream);
    cudf::numeric_scalar<float> const hi(2.0F, true, stream);
    auto distance = cudf::clamp(distance_view, lo, hi, stream, mr);
    if (_request.output_type == sirius::vss::vector_join_output_type::similarity) {
      cudf::numeric_scalar<float> const one(1.0F, true, stream);
      score = cudf::binary_operation(one,
                                     distance->view(),
                                     cudf::binary_operator::SUB,
                                     cudf::data_type{cudf::type_id::FLOAT32},
                                     stream,
                                     mr);
    } else {
      score = std::move(distance);
    }
  } else {
    score = std::make_unique<cudf::column>(distance_view, stream, mr);
  }

  // Assemble [left cols..., right cols..., score] — the TVF schema.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  if (left_repeated) {
    for (auto& c : left_repeated->release()) {
      out_cols.push_back(std::move(c));
    }
  }
  if (right_gathered) {
    for (auto& c : right_gathered->release()) {
      out_cols.push_back(std::move(c));
    }
  }
  out_cols.push_back(std::move(score));
  auto out_table = std::make_unique<cudf::table>(std::move(out_cols));

  auto batch = sirius::make_data_batch(std::move(out_table), *space, stream, batch_telemetry());
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_physical_vector_join_materialize::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_materialize::params_to_string() const
{
  return _request.left.table + " x " + _request.right.table + " k=" + std::to_string(_request.k);
}

}  // namespace sirius::op
