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

// The stream operator's CAGRA path: each task searches its probe rows in a global index over the
// pinned corpus, in batches sized to the memory beside the index, and re-scores what comes back
// from the FP32 rows. The search and its output are all this file adds; tasks, probe chunks and
// materialization are the operator's as for any other join.

#include "cuda/vss/cagra_index.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/bound_gemm.hpp"
#include "vss/cagra_join_index.hpp"
#include "vss/cagra_partitions.hpp"
#include "vss/join_result_shaping.hpp"
#include "vss/pinned_column.hpp"
#include "vss/sirius_physical_vector_join_stream.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace sirius::op {

namespace {

double seconds_since(std::chrono::steady_clock::time_point start)
{ return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }

bool phase_debug() { return std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr; }

}  // namespace

struct vector_join_index_state {
  vss::cagra_index_target target;
  std::mutex build_mutex;
  std::shared_ptr<const vss::pinned_index_entry> entry;
  /// One search and its re-score at a time. Both are limited by device memory bandwidth, so a
  /// second task in parallel adds nothing to them, and this bounds the device scratch to one
  /// batch's and keeps each phase's time its own.
  std::mutex search_mutex;
  /// Probe rows a search takes at once, from the room beside the index.
  std::atomic<std::int64_t> batch_rows{0};
  /// A partitioned index: every probe row's candidates, found once by the first task.
  std::optional<vss::cagra_partition_answers> partition_answers;
};

void sirius_physical_vector_join_stream::prepare_index_join_locked()
{
  _index_join  = std::make_shared<vector_join_index_state>();
  auto& state  = *_index_join;
  state.target = {_request.right.catalog,
                  _request.right.schema,
                  _request.right.table,
                  _request.right.column,
                  _request.right.table_identity,
                  _request.dim};
  auto& cache  = _sirius_ctx->get_cuvs_index_cache();
  auto const wanted =
    _request.cagra.build_settings_given ? std::optional{_request.cagra.build} : std::nullopt;
  state.entry = vss::find_usable_cagra_index(cache, state.target, *_right_pin, wanted);
  _sirius_ctx->begin_vector_join_index_stats(state.entry != nullptr ? "reused" : "built");
  auto const spaces =
    _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const& space = *spaces.front();
  bool const partitioned =
    state.entry != nullptr
      ? dynamic_cast<vss::cagra_partitioned_index const*>(state.entry->index.get()) != nullptr
      : _request.cagra.build.partitions > 1;
  if (partitioned) {
    // The first task visits every partition; the others then only merge and re-score.
    set_index_join_limits_locked(space.get_available_memory());
    std::lock_guard<std::mutex> admission_lock(_admission->mutex);
    _admission->limit = 1;
  } else if (state.entry != nullptr) {
    set_index_join_limits_locked(space.get_available_memory());
  } else {
    // The first task builds the index, on a reservation of its own, so the others should neither
    // hold memory it needs nor wait on it with theirs. The limit opens when the build is done.
    auto const resident =
      vss::cagra_resident_bytes(_right_total_rows, _request.dim, _request.cagra.build);
    auto const available = space.get_available_memory();
    set_index_join_limits_locked(available > resident ? available - resident : 0);
    std::lock_guard<std::mutex> admission_lock(_admission->mutex);
    _admission->limit = 1;
  }
}

void sirius_physical_vector_join_stream::set_index_join_limits_locked(std::size_t room)
{
  auto& state = *_index_join;
  auto const build =
    state.entry != nullptr ? vss::cagra_index_of(*state.entry).settings() : _request.cagra.build;
  auto const k_join  = std::min<std::int64_t>(_request.k, _right_total_rows);
  auto const per_row = vss::cagra_batch_bytes_per_row(
    _right_total_rows, _request.dim, k_join, build, _request.cagra.search);
  std::int64_t widest = 0;
  for (std::size_t i = 0; _probe && i < _num_left; ++i) {
    widest = std::max(widest, static_cast<std::int64_t>(_probe->chunk_rows(i)));
  }
  // A quarter of the room for one batch's scratch; the task's own output and what materialize
  // stages take from the rest.
  auto const fitting = static_cast<std::int64_t>(room / 4 / std::max<std::size_t>(per_row, 1));
  auto const rows = std::clamp<std::int64_t>(fitting, 1024, std::max<std::int64_t>(widest, 1024));
  state.batch_rows.store(rows);
  if (phase_debug()) {
    std::fprintf(stderr,
                 "[vecjoin-phase] index join: %.2f GiB beside the index, %zu B a probe row, %lld "
                 "rows a search\n",
                 static_cast<double>(room) / 1073741824.0,
                 per_row,
                 static_cast<long long>(rows));
  }
  // One task searches while another hands its answer on; more would only wait for the search.
  auto const need = estimate_for_rows(static_cast<std::size_t>(widest));
  std::lock_guard<std::mutex> admission_lock(_admission->mutex);
  _admission->limit = std::clamp<std::size_t>(room / std::max<std::size_t>(need, 1), 1, 2);
}

std::size_t sirius_physical_vector_join_stream::index_join_estimate(std::size_t n_left) const
{
  auto const k = static_cast<std::size_t>(
    std::max<std::int64_t>(std::min<std::int64_t>(_request.k, _right_total_rows), 1));
  auto const& state  = _index_join;
  auto const build   = state && state->entry != nullptr
                         ? vss::cagra_index_of(*state->entry).settings()
                         : _request.cagra.build;
  auto const per_row = vss::cagra_batch_bytes_per_row(
    _right_total_rows, _request.dim, static_cast<std::int64_t>(k), build, _request.cagra.search);
  auto const batch =
    state && state->batch_rows.load() > 0
      ? std::min<std::size_t>(n_left, static_cast<std::size_t>(state->batch_rows.load()))
      : n_left;
  // The ids and distances of the whole chunk, the left-row column shaping adds and the copy shaping
  // or a filter may make.
  auto const output  = n_left * k * (2 * sizeof(std::int64_t) + 2 * sizeof(float));
  std::size_t staged = 0;
  if (_probe && _probe->is_streaming()) { staged = _max_probe_chunk_bytes; }
  return batch * per_row + output + staged + (std::size_t{64} << 20);
}

vss::shaped_join_result sirius_physical_vector_join_stream::index_join_task(
  vss::dataset_matrix_view queries,
  std::size_t left_idx,
  std::int64_t k_join,
  ::cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (_sirius_ctx == nullptr || !_index_join || _right_pin == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] the index join was not set up");
  }
  auto& state       = *_index_join;
  auto const n_left = static_cast<std::int64_t>(queries.extent(0));
  auto const dim    = _request.dim;
  bool const cosine = _request.metric == "cosine";

  std::shared_ptr<const vss::pinned_index_entry> entry;
  {
    std::lock_guard<std::mutex> build_lock(state.build_mutex);
    auto const entry_is_partitioned = [&] {
      return state.entry != nullptr &&
             dynamic_cast<vss::cagra_partitioned_index const*>(state.entry->index.get()) != nullptr;
    };
    if (state.entry == nullptr || (entry_is_partitioned() && !state.partition_answers)) {
      telemetry::batch_telemetry_info const telemetry_info{};
      vss::cagra_build_report report;
      // Reservations are tracked per thread and this one's holds the task's, so the build, which is
      // charged to its own, runs on a thread of its own while the task waits for it.
      std::exception_ptr build_error;
      bool built_here = false;
      std::thread builder([&] {
        try {
          rmm::cuda_set_device_raii const device{rmm::cuda_device_id{space.get_device_id()}};
          auto const hosts = _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(
            cucascade::memory::Tier::HOST);
          auto& cache = _sirius_ctx->get_cuvs_index_cache();
          if (state.entry == nullptr) {
            state.entry = _request.cagra.build.partitions > 1
                            ? vss::build_partitioned_cagra_index_for_pin("sirius_knn_join",
                                                                         cache,
                                                                         state.target,
                                                                         _right_pin,
                                                                         _request.cagra.build,
                                                                         space,
                                                                         *hosts.front(),
                                                                         telemetry_info,
                                                                         report)
                            : vss::build_cagra_index_for_pin("sirius_knn_join",
                                                             cache,
                                                             state.target,
                                                             _right_pin,
                                                             _request.cagra.build,
                                                             space,
                                                             telemetry_info,
                                                             report);
            built_here  = true;
          }
          if (auto const* parts =
                dynamic_cast<vss::cagra_partitioned_index const*>(state.entry->index.get())) {
            rmm::cuda_stream probe_stream;
            rmm::cuda_stream_view const probe_view = probe_stream;
            std::int64_t probe_rows                = 0;
            for (std::size_t i = 0; i < _num_left; ++i) {
              probe_rows += static_cast<std::int64_t>(_probe->chunk_rows(i));
            }
            auto const wanted_probes =
              _request.cagra.search.partition_probes > 0
                ? _request.cagra.search.partition_probes
                : std::max<std::int64_t>(2, parts->settings().partitions / 2);
            state.partition_answers = vss::cagra_search_partitions(
              *parts,
              [&](std::size_t i, std::function<void(float const*, std::int64_t)> const& with_rows) {
                auto const chunk = _probe->stage(i, space, probe_view);
                probe_view.sync();
                auto const rows = vss::list_column_as_dataset_view(chunk.view, dim);
                with_rows(rows.data_handle(), rows.extent(0));
                probe_view.sync();
              },
              _num_left,
              probe_rows,
              k_join,
              _request.cagra.search,
              std::min(wanted_probes, parts->settings().partitions),
              cosine,
              cache,
              space);
          }
        } catch (...) {
          build_error = std::current_exception();
        }
      });
      builder.join();
      if (build_error) { std::rethrow_exception(build_error); }
      if (built_here) {
        _sirius_ctx->record_vector_join_index_build(
          report.convert_seconds, report.build_seconds, report.peak_bytes, report.resident_bytes);
        _sirius_ctx->record_vector_join_index_moves(0, report.move_seconds);
      }
      if (state.partition_answers) {
        auto const& answers = *state.partition_answers;
        _sirius_ctx->record_vector_join_index_moves(answers.route_seconds, answers.move_seconds);
        _sirius_ctx->record_vector_join_index_task(
          0, answers.search_seconds, 0, answers.batches, 0, 0);
      }
      if (phase_debug()) {
        std::fprintf(stderr,
                     "[vecjoin-phase] cagra build: convert %.3f s, graph %.3f s, peak %.2f GiB, "
                     "resident %.2f GiB\n",
                     report.convert_seconds,
                     report.build_seconds,
                     static_cast<double>(report.peak_bytes) / 1073741824.0,
                     static_cast<double>(report.resident_bytes) / 1073741824.0);
      }
      std::lock_guard<std::mutex> op_lock(_op_mutex);
      set_index_join_limits_locked(space.get_available_memory());
    }
    entry = state.entry;
  }
  auto const& index = vss::cagra_index_of(*entry);
  auto const build  = index.settings();
  if (cosine && !index.rows_unit()) {
    throw duckdb::InvalidInputException(
      std::format("sirius_knn_join: a CAGRA join under metric => 'cosine' needs unit rows, and "
                  "'{}.{}' holds rows further than {} from length 1; normalize them, or join "
                  "under metric => 'l2'",
                  state.target.table,
                  state.target.column,
                  vss::unit_norm_tolerance));
  }

  auto const* partitioned = dynamic_cast<vss::cagra_partitioned_index const*>(entry->index.get());
  // The rows the re-score reads, wherever the pin keeps them.
  auto rows_map = vss::map_pinned_rows(*_right_pin, _request.right.column, dim, space, stream, mr);
  if (!rows_map) {
    throw duckdb::InvalidInputException(
      std::format("sirius_knn_join: the pin of '{}.{}' cannot be read a row at a time (compressed, "
                  "or not plain rows), and a CAGRA join re-scores from it; pin it uncompressed",
                  state.target.table,
                  state.target.column));
  }
  auto const exact_source = rows_map->view();

  auto neighbors_col    = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                    static_cast<cudf::size_type>(n_left * k_join),
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  auto distances_col    = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                                    static_cast<cudf::size_type>(n_left * k_join),
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  auto* const ids       = neighbors_col->mutable_view().data<std::int64_t>();
  auto* const distances = distances_col->mutable_view().data<float>();

  bool const is_half     = build.row_type == vss::cagra_row_type::float16;
  auto const value_bytes = vss::cagra_value_bytes(build.row_type);
  double convert_seconds = 0, search_seconds = 0, rescore_seconds = 0;
  std::int64_t batch_rows = std::max<std::int64_t>(state.batch_rows.load(), 1);
  std::int64_t batches    = 0;
  std::int64_t at         = 0;
  if (partitioned != nullptr) {
    // The partitions were searched by the first task: what is left is each row's candidates,
    // nearest by the walk's own distances, and their FP32 distances.
    auto mark = std::chrono::steady_clock::now();
    std::vector<std::int64_t> merged;
    vss::cagra_merge_partition_answers(
      *state.partition_answers, state.partition_answers->chunk_first[left_idx], n_left, merged);
    CUDF_CUDA_TRY(cudaMemcpyAsync(ids,
                                  merged.data(),
                                  merged.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    vss::exact_distances(queries.data_handle(),
                         nullptr,
                         k_join,
                         ids,
                         distances,
                         n_left * k_join,
                         exact_source,
                         dim,
                         stream,
                         mr);
    vss::cagra_sort_rows(distances, ids, n_left, k_join, stream);
    stream.sync();
    rescore_seconds += seconds_since(mark);
    at = n_left;
  }
  while (at < n_left) {
    auto const m = std::min(batch_rows, n_left - at);
    try {
      auto const* const q_fp32 = queries.data_handle() + at * dim;

      auto mark = std::chrono::steady_clock::now();
      std::optional<rmm::device_buffer> narrowed;
      if (is_half) {
        narrowed.emplace(static_cast<std::size_t>(m * dim) * value_bytes, stream, mr);
      }
      vss::row_norm_range norms;
      // FP32 rows are searched in place, so only a cosine join has a reason to read them here.
      if (is_half || cosine) {
        vss::cagra_convert_rows(q_fp32,
                                m,
                                dim,
                                build.row_type,
                                is_half ? narrowed->data() : nullptr,
                                static_cast<std::size_t>(dim) * value_bytes,
                                norms,
                                stream,
                                mr);
        if (!(norms.max_sq < std::numeric_limits<float>::infinity())) {
          throw duckdb::InvalidInputException(
            "sirius_knn_join: a probe row holds a value past float16's range, a NaN or an "
            "infinity; row_type => 'float32' keeps values float16 cannot hold");
        }
        if (cosine && !vss::norms_are_unit(norms)) {
          throw duckdb::InvalidInputException(
            std::format("sirius_knn_join: a CAGRA join under metric => 'cosine' needs unit rows, "
                        "and a probe row's length is further than {} from 1 (squared lengths {} "
                        "to {}); normalize them, or join under metric => 'l2'",
                        vss::unit_norm_tolerance,
                        norms.min_sq,
                        norms.max_sq));
        }
      }
      stream.sync();
      convert_seconds += seconds_since(mark);

      rmm::device_uvector<std::uint32_t> found(static_cast<std::size_t>(m * k_join), stream, mr);
      auto* const row_distances = distances + at * k_join;
      std::lock_guard<std::mutex> search_lock(state.search_mutex);
      mark = std::chrono::steady_clock::now();
      index.search(is_half ? narrowed->data() : static_cast<void const*>(q_fp32),
                   m,
                   k_join,
                   _request.cagra.search,
                   found.data(),
                   row_distances,
                   stream);
      search_seconds += seconds_since(mark);

      mark                = std::chrono::steady_clock::now();
      auto* const row_ids = ids + at * k_join;
      vss::cagra_widen_ids(found.data(), m * k_join, _right_total_rows, row_ids, stream);
      vss::exact_distances(
        q_fp32, nullptr, k_join, row_ids, row_distances, m * k_join, exact_source, dim, stream, mr);
      vss::cagra_sort_rows(row_distances, row_ids, m, k_join, stream);
      stream.sync();
      rescore_seconds += seconds_since(mark);
    } catch (rmm::out_of_memory const&) {
      // A smaller batch needs less of what the task may not get. The engine's own retry, which
      // restarts the task with a larger reservation, is what remains below that.
      if (m <= 1024) { throw; }
      batch_rows = std::max<std::int64_t>(m / 2, 1024);
      state.batch_rows.store(batch_rows);
      stream.sync();
      continue;
    }
    at += m;
    ++batches;
  }

  auto mark = std::chrono::steady_clock::now();
  if (cosine) {
    // Over unit rows |q - x|^2 = 2 (1 - cos), computed from the differences so that near-duplicates
    // keep their digits; the cosine distance is half of it.
    vss::scale_in_place(distances, n_left * k_join, 0.5f, stream);
  } else {
    vss::sqrt_in_place(distances, n_left * k_join, stream);
  }
  stream.sync();
  rescore_seconds += seconds_since(mark);

  auto const misses = vss::cagra_count_misses(ids, n_left * k_join, stream);
  _sirius_ctx->record_vector_join_index_task(
    convert_seconds, search_seconds, rescore_seconds, batches, n_left, misses);
  if (phase_debug()) {
    std::fprintf(stderr,
                 "[vecjoin-phase] cagra task: %lld rows in %lld batches, convert %.3f s, search "
                 "%.3f s, re-score %.3f s, misses %lld\n",
                 static_cast<long long>(n_left),
                 static_cast<long long>(batches),
                 convert_seconds,
                 search_seconds,
                 rescore_seconds,
                 static_cast<long long>(misses));
  }

  auto shaped = vss::shape_per_row_top_k(
    std::move(neighbors_col), std::move(distances_col), n_left, k_join, stream, mr);
  if (misses > 0) {
    cudf::numeric_scalar<std::int64_t> const zero(0, true, stream);
    auto const kept_mask = cudf::binary_operation(shaped.neighbors->view(),
                                                  zero,
                                                  cudf::binary_operator::GREATER_EQUAL,
                                                  cudf::data_type{cudf::type_id::BOOL8},
                                                  stream,
                                                  mr);
    auto kept            = cudf::apply_boolean_mask(
                             cudf::table_view{
                               {shaped.left_rows->view(), shaped.neighbors->view(), shaped.distances->view()}},
                             kept_mask->view(),
                             stream,
                             mr)
                             ->release();
    shaped.left_rows     = std::move(kept[0]);
    shaped.neighbors     = std::move(kept[1]);
    shaped.distances     = std::move(kept[2]);
  }
  return shaped;
}

}  // namespace sirius::op
