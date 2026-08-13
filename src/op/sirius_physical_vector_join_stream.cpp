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
#include "scan_manager/sirius_scan_manager.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"
#include "vss/knn_merge.hpp"
#include "vss/pinned_column.hpp"

#include "data/sirius_converter_registry.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
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

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

/// GPU-tier pin: every chunk is already device-resident, so staging hands back a view.
class gpu_pinned_corpus_source : public corpus_source {
 public:
  gpu_pinned_corpus_source(const scan_manager::pinned_entry& pin,
                           const std::string& column,
                           cucascade::memory::memory_space& space)
    : _views(vss::pinned_column_chunk_views(pin, column, space))
  {
  }

  [[nodiscard]] std::size_t num_chunks() const override { return _views.size(); }
  [[nodiscard]] bool is_streaming() const override { return false; }

  staged_corpus_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& /*space*/,
                            rmm::cuda_stream_view /*stream*/) override
  {
    return staged_corpus_chunk{_views.at(i), nullptr, nullptr};
  }

 private:
  std::vector<cudf::column_view> _views;
};

/// HOST-tier pin: each chunk is copied device-side on demand through the same converter
/// the scan path uses, and freed when the caller drops the returned owner. Peak device
/// memory is therefore set by the chunks in flight, not by the corpus size.
class host_pinned_corpus_source : public corpus_source {
 public:
  host_pinned_corpus_source(const scan_manager::pinned_entry& pin,
                            const std::string& column,
                            const telemetry::batch_telemetry_info& telemetry_info)
    : _pin(pin), _telemetry_info(telemetry_info)
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

  staged_corpus_chunk stage(std::size_t i,
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
    return staged_corpus_chunk{table.column(0), std::move(batch), std::move(reservation)};
  }

 private:
  const scan_manager::pinned_entry& _pin;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
};

}  // namespace

std::unique_ptr<corpus_source> make_gpu_pinned_corpus_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  cucascade::memory::memory_space& space)
{
  return std::make_unique<gpu_pinned_corpus_source>(pin, column, space);
}

std::unique_ptr<corpus_source> make_host_pinned_corpus_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  return std::make_unique<host_pinned_corpus_source>(pin, column, telemetry_info);
}

sirius_physical_vector_join_stream::sirius_physical_vector_join_stream(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_STREAM, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager)
{
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::ensure_initialized_locked()
{
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  const auto* left_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(left.catalog, left.schema, left.table);
  const auto* right_pin =
    _scan_manager->find_pinned_entry_for_duckdb_table(right.catalog, right.schema, right.table);
  if (left_pin == nullptr || right_pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] left or right table is no longer pinned");
  }

  // The probe side stays resident: it is the [M x d] build side and it is small. Only the
  // corpus streams, so only the corpus goes behind the tier-agnostic seam.
  _left_views =
    vss::pinned_column_chunk_views(*left_pin, left.column, vss::pinned_entry_gpu_space(*left_pin));

  if (right_pin->tier == cucascade::memory::Tier::HOST) {
    _corpus = make_host_pinned_corpus_source(*right_pin, right.column, batch_telemetry());
  } else {
    _corpus = make_gpu_pinned_corpus_source(
      *right_pin, right.column, vss::pinned_entry_gpu_space(*right_pin));
  }

  // Row counts per chunk are not known until a chunk is staged, so the neighbor-id base is
  // accumulated while streaming rather than pre-summed. The total comes from the pin.
  _right_total_rows = static_cast<std::int64_t>(right_pin->num_rows);

  _num_left    = _left_views.size();
  _initialized = true;
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_vector_join_stream::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  if (_num_left == 0 || _next_left >= _num_left || _hint_returned) { return std::nullopt; }
  _hint_returned = true;
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_vector_join_stream::all_ports_empty()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  return _next_left >= _num_left;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  ensure_initialized_locked();
  if (_next_left >= _num_left) { return nullptr; }

  auto const left_idx = _next_left++;
  return std::make_unique<vector_join_stream_input>(left_idx, per_left_batch_estimate(left_idx));
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
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

  auto const queries  = vss::list_column_as_dataset_view(_left_views[left_idx], dim);
  auto const n_left   = static_cast<std::int64_t>(queries.extent(0));
  auto const k_join   = std::min<std::int64_t>(_request.k, _right_total_rows);

  raft::device_resources res{stream};
  auto const exact_unexpanded = _request.search_mode == vss::vector_join_search_mode::exact;
  auto const metric =
    vss::join_selection_distance_type_from_metric(_request.metric, exact_unexpanded);

  // Running [n_left x k_join] accumulator. Seeded by the first right batch, then
  // every later batch is folded in and released, so peak device memory does not
  // grow with the number of right batches.
  std::unique_ptr<cudf::column> acc_neighbors;
  std::unique_ptr<cudf::column> acc_distances;

  auto const n_chunks = _corpus->num_chunks();
  std::int64_t offset = 0;  // running base of the current chunk in right-table row space

  for (std::size_t j = 0; j < n_chunks; ++j) {
    // Staged for this iteration only. For a host-tier corpus this is the H2D copy, and
    // `staged.owner` releases the device copy at the end of the iteration -- which is what
    // keeps device memory bounded by the chunk in flight rather than the corpus.
    auto staged           = _corpus->stage(j, *mem_space, stream);
    auto const dataset    = vss::list_column_as_dataset_view(staged.view, dim);
    auto const batch_rows = static_cast<std::int64_t>(dataset.extent(0));
    if (batch_rows == 0) { continue; }

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

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(2);
  out_cols.push_back(std::move(acc_neighbors));
  out_cols.push_back(std::move(acc_distances));
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
  auto const n_left = static_cast<std::size_t>(_left_views[left_idx].size());
  auto const k      = static_cast<std::size_t>(std::max<std::int64_t>(_request.k, 1));
  auto const block  = n_left * k * (sizeof(std::int64_t) + sizeof(float));
  return (block * 6) + (std::size_t{1} << 20);
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
