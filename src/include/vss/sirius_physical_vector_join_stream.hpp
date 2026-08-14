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

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "vss/vector_join.hpp"
#include "vss/vector_join_build_side.hpp"

#include "telemetry/data_batch_probe.hpp"

#include <cucascade/data/data_batch.hpp>

#include <cudf/column/column_view.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace cucascade {
class data_batch;
namespace memory {
class reservation;
}  // namespace memory
}  // namespace cucascade

namespace sirius::op {

/**
 * @brief One corpus chunk made device-resident for the duration of a single fold step.
 *
 * @c owner is null when the chunk was already device-resident (GPU-tier pin); otherwise it
 * holds the staged copy alive and releases it when the fold step drops it, which is what
 * bounds device memory to the chunks in flight rather than the whole corpus.
 */
struct staged_vector_chunk {
  cudf::column_view view;
  std::shared_ptr<::cucascade::data_batch> owner;
  /// Draws the staged copy from the task's device budget instead of committing fresh
  /// capacity; must outlive @c owner, so it is released alongside it.
  std::shared_ptr<::cucascade::memory::reservation> reservation;
  /// Held only when @c view points into a batch this chunk does not own -- a build-side batch
  /// that was already device-resident. The shared lock is what stops the downgrade executor
  /// spilling that batch out from under the fold; dropping it ends the borrow.
  std::optional<::cucascade::read_only_data_batch> reader;
};

/**
 * @brief Where the streamed (corpus) side's chunks come from.
 *
 * The fold loop only needs "hand me chunk i, device-resident, and take it back when I am
 * done". Isolating that behind this interface is what lets the corpus live on the GPU, in
 * host memory, or -- later -- behind an ordinary child scan, without the fold changing.
 */
class vector_chunk_source {
 public:
  vector_chunk_source()                                = default;
  vector_chunk_source(const vector_chunk_source&)            = delete;
  vector_chunk_source& operator=(const vector_chunk_source&) = delete;
  virtual ~vector_chunk_source()                       = default;

  [[nodiscard]] virtual std::size_t num_chunks() const = 0;

  /// Make chunk @p i device-resident in @p space. Called once per chunk per left batch.
  virtual staged_vector_chunk stage(std::size_t i,
                                    ::cucascade::memory::memory_space& space,
                                    rmm::cuda_stream_view stream) = 0;

  /// True when staging performs a host-to-device copy, i.e. the data is not resident.
  [[nodiscard]] virtual bool is_streaming() const = 0;

  /// Rows in chunk @p i, and its device footprint, both without staging it. The memory
  /// estimate has to size a task before any copy happens.
  [[nodiscard]] virtual std::size_t chunk_rows(std::size_t i) const = 0;
  [[nodiscard]] virtual std::size_t chunk_bytes(std::size_t i) const = 0;
};

/// GPU-tier pin: chunks are already device-resident, so staging is a no-op view.
std::unique_ptr<vector_chunk_source> make_gpu_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  ::cucascade::memory::memory_space& space);

/// HOST-tier pin: each chunk is copied device-side on demand and released after the fold
/// step, which is what makes the corpus side out-of-core.
std::unique_ptr<vector_chunk_source> make_host_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info);

/// Build phase: chunks are the batches a child scan deposited in the join's build port, taken
/// in the buffer's snapshot order. A batch the downgrade executor has spilled is brought back
/// device-side for the fold step and released after it, exactly as a HOST-tier pin chunk is;
/// one still resident is viewed in place.
std::unique_ptr<vector_chunk_source> make_materialized_chunk_source(
  sirius::vss::build_side_buffer& buffer,
  std::size_t column_index,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info);

/**
 * @brief The input handed to one sirius_physical_vector_join_stream::execute() call.
 *
 * One task per left (query) batch, not per (left, right) pair: the task streams every
 * right batch through a running top-k fold, so the pair grid is walked inside execute()
 * instead of being handed out as separate tasks.
 */
class vector_join_stream_input : public operator_data {
 public:
  vector_join_stream_input(std::size_t left_idx, std::size_t estimated_bytes)
    : _left_idx(left_idx), _estimated_bytes(estimated_bytes)
  {
  }

  [[nodiscard]] operator_data_type get_type() const override { return operator_data_type::BASE; }

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view /*stream*/) override
  {
    _gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  [[nodiscard]] ::cucascade::memory::memory_space* get_gpu_memory_space() const
  {
    return _gpu_memory_space;
  }

  /// Index of this task's left batch; also the output partition, so the materialize
  /// stage can gather that batch's left columns.
  [[nodiscard]] std::size_t left_idx() const { return _left_idx; }

 private:
  std::size_t _left_idx;
  ::cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

/**
 * @brief Streaming exact k-nearest-neighbor vector join: search and merge fused.
 *
 * Replaces the VECTOR_JOIN_SELECT -> VECTOR_JOIN_REDUCE_LOCAL pair. Each task takes
 * one left (query) batch, keeps an `[n_left x k]` top-k accumulator, and folds every
 * right batch into it as it is searched, releasing each right batch's partial
 * immediately. The fold reuses cuVS `knn_merge_parts` with `n_parts = 2` (accumulator
 * + the batch just searched), so no new kernel is needed.
 *
 * Why this shape: the split design emitted one `[n_left x k]` partial per
 * (left batch, right batch) pair through the batch repo, so intermediate data was
 * `n_right_batches` times the final answer and the merge stage had to hold all of a
 * partition's partials at once. Folding in place makes device memory independent of
 * the number of right batches, which is what makes the operator out-of-core; see
 * `docs` note in vector_join.hpp and the measured residency/throughput tradeoff.
 *
 * Parallelism is across left batches: each task owns its own accumulator, so there is
 * no shared state and no lock on the hot path.
 *
 * Output matches what the merge stage used to emit -- `[neighbor_id INT64,
 * distance FLOAT32]` flattened `[n_left * k]`, partitioned by left batch index -- so
 * the materialize stage is unchanged.
 */
class sirius_physical_vector_join_stream : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_STREAM;

  /// @param build_side  When non-null the corpus comes from this operator's build port -- a
  ///                    child scan materialized by the build phase -- instead of a pinned
  ///                    catalog table, and the buffer is the row order both this operator and
  ///                    materialize resolve neighbour ids against.
  sirius_physical_vector_join_stream(
    duckdb::vector<sirius::logical_type> types,
    duckdb::idx_t estimated_cardinality,
    sirius::vss::vector_join_request request,
    sirius::scan_manager::sirius_scan_manager* scan_manager,
    std::shared_ptr<sirius::vss::build_side_buffer> build_side = nullptr);

  [[nodiscard]] const sirius::vss::vector_join_request& request() const { return _request; }

  /// True when the corpus is fed by a child scan rather than resolved from a pin.
  [[nodiscard]] bool has_build_phase() const { return _build_side != nullptr; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  // -----------------------------
  // Source interface
  // -----------------------------
  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /// Streams every right batch through this left batch's running top-k fold and
  /// returns the finished `[n_left * k]` result.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// Routes the finished result to the materialize stage under its left batch index.
  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  /// Resolve both pinned tables and snapshot per-batch views plus right-batch row
  /// offsets. Idempotent; caller holds _op_mutex. On the build path this is a no-op until
  /// @ref build_side_ready_locked, so callers must check that first.
  void ensure_initialized_locked();

  /// Whether the build port is wired and its producing pipeline has finished. Always true on
  /// the pinned path. Caller holds _op_mutex.
  bool build_side_ready_locked();

  /// Peak bytes for one left batch's task: the accumulator, the partial being folded
  /// in, and the stacked pair the merge reads. Independent of the right batch count,
  /// which is the point of the fold.
  [[nodiscard]] std::size_t per_left_batch_estimate(std::size_t left_idx) const;

  /// The corpus row order, shared with materialize. Null on the pinned path, where the
  /// pinned_entry plays the same role.
  std::shared_ptr<sirius::vss::build_side_buffer> _build_side;

  sirius::vss::vector_join_request _request;
  sirius::scan_manager::sirius_scan_manager* _scan_manager;

  std::mutex _op_mutex;
  bool _initialized{false};
  bool _hint_returned{false};
  std::unique_ptr<vector_chunk_source> _probe;
  //! Streamed side, behind the tier-agnostic seam.
  std::unique_ptr<vector_chunk_source> _corpus;
  //! Total right-table rows, used to clamp k the way the plan already does.
  std::int64_t _right_total_rows{0};
  //! Largest corpus chunk in bytes; reserved per task when the corpus is streamed.
  std::size_t _max_chunk_bytes{0};
  std::size_t _max_probe_chunk_bytes{0};
  std::size_t _num_left{0};
  std::size_t _next_left{0};
};

}  // namespace sirius::op
