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
#include "vss/brute_force_threshold.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"
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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

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
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
                            const telemetry::batch_telemetry_info& telemetry_info,
                            duckdb::SiriusContext* ctx)
    : _repo(buffer.repo()),
      _ctx(ctx),
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

    std::shared_ptr<cucascade::memory::reservation> reservation{reserve_or_spill(space, bytes)};
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

  // The device can be full of build-side batches this fold has already read, and nothing but
  // the downgrade executor will move them: a task's own reservation request does not, and a
  // blocking wait for memory that no one is freeing never returns. So ask it directly, the way
  // the pipeline executor does for a short task reservation, until the copy-back fits or
  // there is nothing left it may spill.
  std::unique_ptr<cucascade::memory::reservation> reserve_or_spill(
    cucascade::memory::memory_space& space, std::size_t bytes) const
  {
    auto res = space.make_reservation_or_null(bytes);
    if (res || _ctx == nullptr) { return res; }
    std::unique_ptr<cucascade::memory::reservation> got;
    std::mutex m;
    _ctx->get_downgrade_executor(space.get_id())
      .request_downgrade([&]() {
        std::lock_guard<std::mutex> const lock(m);
        if (!got) {
          auto r = space.make_reservation_or_null(bytes);
          if (r && r->size() >= bytes) { got = std::move(r); }
        }
        return got != nullptr;
      })
      .get();
    return got;
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
  duckdb::SiriusContext* _ctx{nullptr};
  std::vector<std::uint64_t> _batch_ids;
  std::vector<std::size_t> _rows;
  std::vector<std::size_t> _bytes;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
  std::int64_t _dim{0};
};

/// A pushed-down predicate's constant, as the cuDF scalar its column compares against.
std::unique_ptr<cudf::scalar> predicate_scalar(duckdb::Value const& v,
                                               cudf::data_type type,
                                               rmm::cuda_stream_view stream)
{
  switch (type.id()) {
    case cudf::type_id::BOOL8:
      return std::make_unique<cudf::numeric_scalar<bool>>(v.GetValue<bool>(), true, stream);
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<std::int8_t>>(
        v.GetValue<std::int8_t>(), true, stream);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<std::int16_t>>(
        v.GetValue<std::int16_t>(), true, stream);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<std::int32_t>>(
        v.GetValue<std::int32_t>(), true, stream);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<std::int64_t>>(
        v.GetValue<std::int64_t>(), true, stream);
    case cudf::type_id::UINT8:
      return std::make_unique<cudf::numeric_scalar<std::uint8_t>>(
        v.GetValue<std::uint8_t>(), true, stream);
    case cudf::type_id::UINT16:
      return std::make_unique<cudf::numeric_scalar<std::uint16_t>>(
        v.GetValue<std::uint16_t>(), true, stream);
    case cudf::type_id::UINT32:
      return std::make_unique<cudf::numeric_scalar<std::uint32_t>>(
        v.GetValue<std::uint32_t>(), true, stream);
    case cudf::type_id::UINT64:
      return std::make_unique<cudf::numeric_scalar<std::uint64_t>>(
        v.GetValue<std::uint64_t>(), true, stream);
    case cudf::type_id::FLOAT32:
      return std::make_unique<cudf::numeric_scalar<float>>(v.GetValue<float>(), true, stream);
    case cudf::type_id::FLOAT64:
      return std::make_unique<cudf::numeric_scalar<double>>(v.GetValue<double>(), true, stream);
    case cudf::type_id::STRING:
      return std::make_unique<cudf::string_scalar>(v.ToString(), true, stream);
    default:
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] a pushed-down corpus predicate compares a column "
        "of an unsupported device type");
  }
}

cudf::binary_operator predicate_operator(vss::corpus_predicate::op op)
{
  using cmp = vss::corpus_predicate::op;
  switch (op) {
    case cmp::eq: return cudf::binary_operator::EQUAL;
    case cmp::ne: return cudf::binary_operator::NOT_EQUAL;
    case cmp::lt: return cudf::binary_operator::LESS;
    case cmp::le: return cudf::binary_operator::LESS_EQUAL;
    case cmp::gt: return cudf::binary_operator::GREATER;
    case cmp::ge: return cudf::binary_operator::GREATER_EQUAL;
  }
  return cudf::binary_operator::EQUAL;
}

/// The rows of corpus chunk @p j that every pushed-down predicate keeps, compacted into one
/// row-major matrix, with each kept row's corpus row id alongside. A null compares false, as it
/// does in the WHERE clause the predicate came from.
struct filtered_chunk {
  rmm::device_uvector<float> vectors;
  std::unique_ptr<cudf::column> row_ids;  ///< INT64 corpus row of each kept row
  std::int64_t rows{0};
};

filtered_chunk filter_corpus_chunk(const scan_manager::pinned_entry& pin,
                                   std::vector<vss::corpus_predicate> const& preds,
                                   std::size_t j,
                                   vss::dataset_matrix_view vectors,
                                   std::int64_t row_base,
                                   cucascade::memory::memory_space& space,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   telemetry::batch_telemetry_info const& telemetry_info)
{
  std::vector<vss::staged_pinned_chunk> columns;
  std::unique_ptr<cudf::column> mask;
  for (auto const& p : preds) {
    columns.push_back(
      vss::stage_pinned_column_chunk(pin, p.column, j, space, stream, telemetry_info));
    auto const& col   = columns.back().view;
    auto const scalar = predicate_scalar(p.value, col.type(), stream);
    auto keep         = cudf::binary_operation(
      col, *scalar, predicate_operator(p.cmp), cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
    mask = mask ? cudf::binary_operation(mask->view(),
                                         keep->view(),
                                         cudf::binary_operator::LOGICAL_AND,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr)
                : std::move(keep);
  }
  // Compacting a LIST column through cuDF rebuilds its offsets and gathers its child element
  // by element, which cost more than the searched rows it saved. The mask is applied to row
  // indices instead, and the kept rows are copied whole.
  auto const n = static_cast<cudf::size_type>(vectors.extent(0));
  cudf::numeric_scalar<std::int64_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int64_t> const base(row_base, true, stream);
  cudf::numeric_scalar<std::int64_t> const step(1, true, stream);
  auto const local  = cudf::sequence(n, zero, step, stream, mr);
  auto const global = cudf::sequence(n, base, step, stream, mr);
  auto kept         = cudf::apply_boolean_mask(
    cudf::table_view{{local->view(), global->view()}}, mask->view(), stream, mr);
  auto const rows = static_cast<std::int64_t>(kept->num_rows());
  auto const dim  = vectors.extent(1);
  filtered_chunk out{
    rmm::device_uvector<float>(static_cast<std::size_t>(rows * dim), stream, mr), nullptr, rows};
  vss::gather_rows(vectors.data_handle(),
                   dim,
                   kept->get_column(0).view().data<std::int64_t>(),
                   rows,
                   out.vectors.data(),
                   stream);
  out.row_ids = std::move(kept->release()[1]);
  // The staged predicate columns are released on return; the mask read them.
  stream.synchronize();
  return out;
}

/// Whether the GEMM-ranked kernels replace cuVS where they can; SIRIUS_VSS_GEMM_TOPK=0 is the
/// A/B switch back to cuVS for both the top-k and the threshold searches.
bool gemm_search_enabled()
{
  static bool const enabled = [] {
    auto const* v = std::getenv("SIRIUS_VSS_GEMM_TOPK");
    return v == nullptr || std::string_view{v} != "0";
  }();
  return enabled;
}

vss::threshold_join_result threshold_search(raft::device_resources const& res,
                                            vss::dataset_matrix_view dataset,
                                            vss::dataset_matrix_view queries,
                                            float eps,
                                            cuvs::distance::DistanceType metric,
                                            rmm::device_async_resource_ref mr)
{
  return gemm_search_enabled() && vss::gemm_search_supports(metric)
           ? vss::gemm_threshold(res, dataset, queries, eps, metric, mr)
           : vss::brute_force_threshold(res, dataset, queries, eps, metric, mr);
}

/// Ask the downgrade executor to spill until @p bytes of the device are free. A streamed corpus
/// fills the pool with build-side batches this fold has already read; if the next chunk's
/// scratch then does not fit, the allocation fails, and the task is restarted from its first
/// chunk -- repeatedly, since the next attempt meets the same full pool. Spilling first costs a
/// copy of batches no one will read again.
void ensure_device_headroom(duckdb::SiriusContext* ctx,
                            cucascade::memory::memory_space& space,
                            std::size_t bytes)
{
  if (ctx == nullptr || space.get_available_memory() >= bytes) { return; }
  ctx->get_downgrade_executor(space.get_id())
    .request_downgrade([&space, bytes]() { return space.get_available_memory() >= bytes; })
    .get();
}

/// Read locks on borrowed build-side batches, each held only until the searches that read it
/// have finished on the compute stream. A borrow costs no device memory, but a locked batch
/// cannot be spilled, so holding every borrow to the end of the task pinned the whole corpus on
/// the device: past the pool size the first spilled chunk found no room to be copied back.
/// Keeping the last few in flight is what lets a corpus larger than the pool stream at all.
class borrow_window {
 public:
  explicit borrow_window(rmm::cuda_stream_view stream) : _stream(stream) {}
  borrow_window(const borrow_window&)            = delete;
  borrow_window& operator=(const borrow_window&) = delete;
  ~borrow_window()
  {
    while (!_held.empty()) {
      release_front();
    }
  }

  void hold(cucascade::read_only_data_batch reader)
  {
    cudaEvent_t done = nullptr;
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    CUDF_CUDA_TRY(cudaEventRecord(done, _stream.value()));
    _held.push_back(held{done, std::move(reader)});
    while (_held.size() > kInFlight) {
      release_front();
    }
  }

 private:
  // The chunk being searched and the one before it: deeper than that only delays the spill.
  static constexpr std::size_t kInFlight = 2;
  struct held {
    cudaEvent_t done;
    cucascade::read_only_data_batch reader;
  };

  void release_front()
  {
    auto& front = _held.front();
    cudaEventSynchronize(front.done);
    cudaEventDestroy(front.done);
    _held.pop_front();
  }

  rmm::cuda_stream_view _stream;
  std::deque<held> _held;
};

/// Cluster lists: a chunk is a fixed span of layout rows. On the GPU tier it is a view into the
/// lists themselves; on the HOST tier its pinned blocks are copied into one device buffer drawn
/// from the task's budget, the same bargain the host pin makes.
class cluster_lists_chunk_source : public vector_chunk_source {
 public:
  explicit cluster_lists_chunk_source(const vss::cluster_lists& lists) : _lists(lists) {}

  [[nodiscard]] std::size_t num_chunks() const override
  {
    return static_cast<std::size_t>(_lists.num_chunks());
  }
  /// A HOST-tier chunk is copied in and a UINT8 one is widened into a buffer: either way the
  /// fold must size its budget for a staged FP32 chunk.
  [[nodiscard]] bool is_streaming() const override
  {
    return _lists.tier == cucascade::memory::Tier::HOST ||
           _lists.encoding == vss::list_encoding::uint8;
  }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override
  {
    return static_cast<std::size_t>(_lists.rows_in_chunk(static_cast<std::int64_t>(i)));
  }
  [[nodiscard]] std::size_t chunk_bytes(std::size_t i) const override
  {
    if (!is_streaming()) { return 0; }
    auto const staged_u8 =
      _lists.tier == cucascade::memory::Tier::HOST && _lists.encoding == vss::list_encoding::uint8
        ? chunk_rows(i) * _lists.row_bytes()
        : 0;
    return chunk_rows(i) * float_row_bytes() + staged_u8;
  }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) override
  {
    auto const rows    = static_cast<std::int64_t>(chunk_rows(i));
    auto const first   = static_cast<std::int64_t>(i) * _lists.chunk_rows;
    bool const narrow  = _lists.encoding == vss::list_encoding::uint8;
    bool const on_host = _lists.tier == cucascade::memory::Tier::HOST;
    if (!is_streaming()) {
      auto const* data =
        static_cast<const float*>(_lists.device_vectors->data()) + first * _lists.dim;
      return staged_vector_chunk{list_view(data, rows), nullptr, nullptr};
    }

    auto const bytes = chunk_bytes(i);
    std::shared_ptr<cucascade::memory::reservation> reservation{
      space.make_reservation_or_null(bytes)};
    if (!reservation) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] corpus chunk " +
                               std::to_string(i) + " needs " + std::to_string(bytes) +
                               " bytes device-side, which exceeds this task's budget");
    }
    auto const mr = reservation->get_memory_resource();
    auto buffer   = std::make_unique<rmm::device_buffer>(
      static_cast<std::size_t>(rows) * float_row_bytes(), stream, mr);

    // The chunk in its stored encoding, device-side: in place for the GPU tier, copied in block by
    // block for the HOST tier -- a chunk starts on a block boundary, so it is whole blocks plus a
    // short last one.
    std::byte const* stored = nullptr;
    std::optional<rmm::device_buffer> copied;
    if (on_host) {
      if (narrow) {
        copied.emplace(static_cast<std::size_t>(rows) * _lists.row_bytes(), stream, mr);
      }
      auto* out             = static_cast<std::byte*>(narrow ? copied->data() : buffer->data());
      auto const block_rows = _lists.rows_per_block;
      for (std::int64_t r = 0; r < rows; r += block_rows) {
        auto const n = std::min(block_rows, rows - r);
        auto const block =
          _lists.host_vectors->at(static_cast<std::size_t>((first + r) / block_rows));
        CUDF_CUDA_TRY(cudaMemcpyAsync(out + static_cast<std::size_t>(r) * _lists.row_bytes(),
                                      block.data(),
                                      static_cast<std::size_t>(n) * _lists.row_bytes(),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
      }
      if (narrow) { stored = static_cast<std::byte const*>(copied->data()); }
    } else {
      stored = static_cast<std::byte const*>(_lists.device_vectors->data()) +
               static_cast<std::size_t>(first) * _lists.row_bytes();
    }
    if (narrow) {
      vss::widen_shifted_int8(reinterpret_cast<std::int8_t const*>(stored),
                              rows * _lists.dim,
                              static_cast<float*>(buffer->data()),
                              stream);
    }
    // Synchronous like the pin converter: the caller overlaps "host waits on this copy" with
    // the compute it already issued, and the view must be complete before it is searched. It is
    // also what lets the byte staging buffer go at the end of this call.
    stream.synchronize();
    auto view = list_view(static_cast<const float*>(buffer->data()), rows);
    staged_vector_chunk staged{view, nullptr, std::move(reservation)};
    staged.buffer = std::move(buffer);
    return staged;
  }

 private:
  [[nodiscard]] std::size_t float_row_bytes() const
  {
    return static_cast<std::size_t>(_lists.dim) * sizeof(float);
  }

  [[nodiscard]] cudf::column_view list_view(const float* data, std::int64_t rows) const
  {
    auto const n = static_cast<cudf::size_type>(rows);
    cudf::column_view const offsets{
      cudf::data_type{cudf::type_id::INT32}, n + 1, _lists.list_offsets->data(), nullptr, 0};
    cudf::column_view const child{cudf::data_type{cudf::type_id::FLOAT32},
                                  static_cast<cudf::size_type>(rows * _lists.dim),
                                  data,
                                  nullptr,
                                  0};
    return cudf::column_view{
      cudf::data_type{cudf::type_id::LIST}, n, nullptr, nullptr, 0, 0, {offsets, child}};
  }

  const vss::cluster_lists& _lists;
};

}  // namespace

std::unique_ptr<vector_chunk_source> make_materialized_chunk_source(
  sirius::vss::materialized_side_buffer& buffer,
  std::size_t column_index,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info,
  duckdb::SiriusContext* ctx)
{
  return std::make_unique<materialized_chunk_source>(
    buffer, column_index, dim, telemetry_info, ctx);
}

std::unique_ptr<vector_chunk_source> make_gpu_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  cucascade::memory::memory_space& space)
{
  return std::make_unique<gpu_pinned_chunk_source>(pin, column, space);
}

std::unique_ptr<vector_chunk_source> make_cluster_lists_chunk_source(
  const vss::cluster_lists& lists)
{
  return std::make_unique<cluster_lists_chunk_source>(lists);
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
  _right_pin = right_pin;

  // Both sides go behind the same seam. The probe side used to be required GPU-resident on
  // the argument that it is the small one; that is false for any join where both sides are
  // large (rec-sys candidate generation is the motivating case). One task already handles one
  // probe chunk, so streaming the probe is the same change staging the corpus was: stage the
  // chunk this task owns, search the whole corpus against it, release.
  if (_probe_side) {
    auto* port = get_port("default");
    _probe_side->ensure_snapshot(*port->repo);
    _probe = make_materialized_chunk_source(
      *_probe_side, /*column_index=*/0, _request.dim, batch_telemetry(), _sirius_ctx);
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
      *_build_side, /*column_index=*/0, _request.dim, batch_telemetry(), _sirius_ctx);
    for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
      _max_chunk_bytes = std::max(_max_chunk_bytes, _corpus->chunk_bytes(i));
    }
  } else if (_centroids != nullptr && _request.build_cluster_column.empty()) {
    // Lists mode: the corpus is the cluster-ordered copy, and the pin stays behind only as the
    // row space that copy's neighbour ids are reported in.
    _lists =
      _sirius_ctx != nullptr ? vss::find_cluster_lists(*_sirius_ctx, _request.clustering) : nullptr;
    if (_lists == nullptr || _lists->pin != right_pin || _lists->column != right.column ||
        _lists->n_rows != static_cast<std::int64_t>(right_pin->num_rows)) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] clustering '" +
                               _request.clustering + "' has no lists over the current pin of '" +
                               right.table + "." + right.column +
                               "'; run sirius_kmeans_build_lists after pinning");
    }
    _corpus = make_cluster_lists_chunk_source(*_lists);
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
  rmm::device_async_resource_ref mr)
{
  std::lock_guard<std::mutex> const lock(_op_mutex);
  if (_cluster_index_built) { return; }

  _n_clusters = static_cast<std::int64_t>(
    vss::list_column_as_dataset_view(_centroids->view(), _request.dim).extent(0));

  // Lists carry their order on the host already: list c is one range of layout rows, and a
  // chunk is a fixed range of them, so the runs are the overlaps of the two.
  if (_lists != nullptr) {
    if (_lists->n_clusters != _n_clusters) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] lists hold " +
                               std::to_string(_lists->n_clusters) + " clusters but clustering '" +
                               _request.clustering + "' has " + std::to_string(_n_clusters));
    }
    auto const n_chunks = static_cast<std::size_t>(_lists->num_chunks());
    _cluster_rows.assign(static_cast<std::size_t>(_n_clusters), 0);
    _chunk_cluster_runs.assign(n_chunks, {});
    _chunk_row_base.assign(n_chunks, 0);
    for (std::size_t j = 0; j < n_chunks; ++j) {
      _chunk_row_base[j] = static_cast<std::int64_t>(j) * _lists->chunk_rows;
    }
    for (std::int64_t c = 0; c < _n_clusters; ++c) {
      auto const lo                              = _lists->offsets[static_cast<std::size_t>(c)];
      auto const hi                              = _lists->offsets[static_cast<std::size_t>(c) + 1];
      _cluster_rows[static_cast<std::size_t>(c)] = hi - lo;
      for (auto r = lo; r < hi;) {
        auto const j    = r / _lists->chunk_rows;
        auto const base = _chunk_row_base[static_cast<std::size_t>(j)];
        auto const end  = std::min(hi, base + _lists->chunk_rows);
        _chunk_cluster_runs[static_cast<std::size_t>(j)].push_back(
          chunk_cluster_run{static_cast<std::int32_t>(c), r - base, end - base});
        r = end;
      }
    }
    _cluster_index_built = true;
    return;
  }

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
    // The clustering is session state and cuvs::cluster::kmeans::fit is not bit-stable across
    // processes -- roughly one session in six converges to a different centroid set. The probe
    // side is assigned from those centroids at join time, so two approximate runs are only
    // comparable when this hash agrees; without it a re-fit reads as a result regression.
    std::uint64_t centroid_hash = 1469598103934665603ull;
    {
      auto const nvals =
        static_cast<std::size_t>(_n_clusters) * static_cast<std::size_t>(_request.dim);
      std::vector<float> host_centroids(nvals);
      auto const values = _centroids->view().child(cudf::lists_column_view::child_column_index);
      CUDF_CUDA_TRY(cudaMemcpyAsync(host_centroids.data(),
                                    values.data<float>(),
                                    nvals * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      stream.synchronize();
      for (auto const value : host_centroids) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        centroid_hash = (centroid_hash ^ bits) * 1099511628211ull;
      }
    }
    std::fprintf(
      stderr, "[vecjoin] centroids=%016llx\n", static_cast<unsigned long long>(centroid_hash));
    std::fprintf(stderr,
                 "[vecjoin] cluster index: %ld clusters over %ld rows in %zu chunks, %zu runs, "
                 "%zu empty\n",
                 static_cast<long>(_n_clusters),
                 static_cast<long>(row_base),
                 n_chunks,
                 runs,
                 empty);
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

  // A radius join needs no fold. Top-k folds because chunk j+1 can displace chunk j's winners,
  // so a running [n_left, k] merge is unavoidable; "within eps" is independent per chunk, so a
  // chunk's surviving edges are final when produced and the chunks only have to be concatenated.
  // That is why this path accumulates ragged edge lists instead of a fixed-width block, and why
  // it has no k -- and therefore none of the k <= 1024 ceiling that knn_merge_parts imposes.
  bool const radius_join = _request.mode == vss::vector_join_mode::threshold;
  std::vector<std::unique_ptr<cudf::column>> radius_left, radius_neighbors, radius_distances;
  // The kernel works in distance space. For cosine with a similarity threshold the user's
  // "score >= eps" is the same set as "distance <= 1 - eps"; for a distance threshold it is eps
  // directly. Identical to what the shape_threshold path below computes.
  auto const radius_eps = _request.output_type == vss::vector_join_output_type::similarity
                            ? static_cast<float>(1.0 - _request.eps)
                            : static_cast<float>(_request.eps);

  // Rows the exhaustive fold searched; a corpus with none has nothing to join against.
  std::int64_t exhaustive_rows_seen = 0;

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

    ensure_cluster_index(*mem_space, stream, mr);
    phase("cluster index");

    auto const n_probes = std::clamp<std::int64_t>(_request.n_probes, 1, _n_clusters);

    // Each probe row picks its own n_probes nearest centroids, the way an IVF index routes a
    // query. Routing a whole probe cluster through its centroid's nearest centroids kept a
    // run contiguous, but a row at the edge of its cluster has its neighbours next door, and
    // that cost about 3x the probes at equal recall. Contiguity comes back below by sorting
    // the (row, cluster) edges by cluster instead.
    vss::assignment_spec routing;
    routing.n_probes = n_probes;
    auto assignment  = vss::assign_to_centroids(
      res, staged_probe.view, _centroids->view(), dim, routing, 0, metric, stream, mr);
    auto const n_edges = n_left * n_probes;  // row-major [n_left x n_probes]
    phase("assign probe->clusters");

    // A row whose clusters cannot supply k is refused before a single search is issued, rather
    // than after the merge has silently padded its answer out with misses. The edge list is
    // row-major, so a row's clusters are n_probes consecutive entries.
    std::vector<std::int32_t> host_edges(static_cast<std::size_t>(n_edges));
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_edges.data(),
                                  assignment.cluster_ids->view().data<std::int32_t>(),
                                  host_edges.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    std::vector<bool> wanted(static_cast<std::size_t>(_n_clusters), false);
    std::vector<std::int64_t> edge_begin(static_cast<std::size_t>(_n_clusters) + 1, 0);
    for (std::int64_t r = 0; r < n_left; ++r) {
      std::int64_t candidates = 0;
      for (std::int64_t t = 0; t < n_probes; ++t) {
        auto const c = host_edges[static_cast<std::size_t>(r * n_probes + t)];
        if (c < 0 || c >= _n_clusters) {
          throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                                   std::to_string(r) + " was routed to cluster id " +
                                   std::to_string(c) + ", outside the clustering's " +
                                   std::to_string(_n_clusters) + " clusters");
        }
        ++edge_begin[static_cast<std::size_t>(c) + 1];
        auto const rows = _cluster_rows[static_cast<std::size_t>(c)];
        candidates += rows;
        if (rows > 0) { wanted[static_cast<std::size_t>(c)] = true; }
      }
      if (candidates == 0) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                                 std::to_string(r) + " reached no non-empty corpus cluster");
      }
      if (candidates < k_join) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                                 std::to_string(r) + " reaches only " + std::to_string(candidates) +
                                 " corpus rows, fewer than k=" + std::to_string(k_join) +
                                 "; raise n_probes or cluster with fewer, larger clusters");
      }
    }
    for (std::size_t c = 0; c < static_cast<std::size_t>(_n_clusters); ++c) {
      edge_begin[c + 1] += edge_begin[c];
    }
    phase("edges to host");

    // Sorted by cluster, each cluster's rows are one contiguous range of the edge list. That
    // range is what a slice's search gathers its queries from and what its answer folds back
    // through. Only the per-cluster counts are needed to place the ranges, and those are
    // already on the host, so the sorted labels never come back.
    auto const order =
      cudf::sorted_order(cudf::table_view{{assignment.cluster_ids->view()}}, {}, {}, stream, mr);
    auto const sorted      = cudf::gather(cudf::table_view{{assignment.row_ids->view()}},
                                     order->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
    auto const sorted_rows = sorted->get_column(0).view();  // INT64 probe row per edge
    phase("sort edges");

    // The prune, made explicit: a chunk holding no cluster any row wants is never staged, so
    // an out-of-core clustered corpus pays no transfer for the part it skips.
    std::vector<std::size_t> needed_chunks;
    for (std::size_t j = 0; j < _chunk_cluster_runs.size(); ++j) {
      for (auto const& slice : _chunk_cluster_runs[j]) {
        if (wanted[static_cast<std::size_t>(slice.cluster)]) {
          needed_chunks.push_back(j);
          break;
        }
      }
    }

    // A miss -- id -1 at infinite distance -- is what the accumulator is seeded with, so a row's
    // first answer merges against nothing. It can only survive the fold when a row had fewer
    // than k real candidates, which was refused above.
    cudf::numeric_scalar<std::int64_t> const miss_id(-1, true, stream);
    cudf::numeric_scalar<float> const miss_distance(
      std::numeric_limits<float>::infinity(), true, stream);

    // The per-slice query matrix, sized for the busiest cluster and reused by every slice: the
    // slices are issued in stream order, so each one's gather waits for the last one's search.
    std::int64_t max_routed = 0;
    for (std::size_t c = 0; c < static_cast<std::size_t>(_n_clusters); ++c) {
      max_routed = std::max(max_routed, edge_begin[c + 1] - edge_begin[c]);
    }
    rmm::device_uvector<float> routed_queries(
      static_cast<std::size_t>(max_routed * dim), stream, mr);
    auto const* routed_rows = sorted_rows.data<std::int64_t>();

    // Running [n_left x k_join] accumulator in the caller's row order, seeded with misses so
    // a row's first answer merges against nothing. Each slice folds its answer in through the
    // rows it served, so device memory does not grow with n_probes: a row's answers from its
    // n_probes clusters never coexist.
    {
      auto const total = static_cast<cudf::size_type>(n_left * k_join);
      acc_neighbors    = cudf::make_column_from_scalar(miss_id, total, stream, mr);
      acc_distances    = cudf::make_column_from_scalar(miss_distance, total, stream, mr);
    }

    // Staging runs on its own stream so the next needed chunk's H2D overlaps this one's
    // compute, exactly as in the exhaustive fold below.
    std::optional<rmm::cuda_stream> staging_stream;
    if (_corpus->is_streaming()) { staging_stream.emplace(); }
    auto const stage_on = staging_stream ? staging_stream->view() : stream;
    borrow_window borrowed{stream};
    auto release_staged = [&](staged_vector_chunk& chunk) {
      if (chunk.owner) {
        auto mut = chunk.owner->to_mutable();
        mut.rebind_stream(stream);
      }
      if (chunk.buffer) { chunk.buffer->set_stream(stream); }
      if (chunk.reader) { borrowed.hold(std::move(*chunk.reader)); }
      chunk = staged_vector_chunk{};
    };

    // Byte-valued lists on the device, searched by a byte-valued probe side, run on the int8 tensor
    // cores straight from the stored bytes: no chunk is widened or staged, the row norms were
    // computed at build time, and every dot product is exact. Only the probe is converted, once.
    bool int8_search = gemm_search_enabled() && !radius_join && _lists != nullptr &&
                       _lists->encoding == vss::list_encoding::uint8 &&
                       _lists->tier == cucascade::memory::Tier::GPU && _lists->row_sq != nullptr &&
                       (metric == cuvs::distance::DistanceType::L2Expanded ||
                        metric == cuvs::distance::DistanceType::L2SqrtExpanded) &&
                       dim % 4 == 0;
    std::optional<rmm::device_uvector<std::int8_t>> probe_i8, routed_i8;
    std::optional<rmm::device_uvector<std::int32_t>> probe_sq, routed_sq;
    if (int8_search) {
      rmm::device_uvector<unsigned long long> non_bytes(1, stream, mr);
      CUDF_CUDA_TRY(
        cudaMemsetAsync(non_bytes.data(), 0, sizeof(unsigned long long), stream.value()));
      vss::count_non_uint8(queries.data_handle(), n_left * dim, non_bytes.data(), stream);
      unsigned long long non_bytes_host = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(&non_bytes_host,
                                    non_bytes.data(),
                                    sizeof(non_bytes_host),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      stream.synchronize();
      int8_search = non_bytes_host == 0;
    }
    if (int8_search) {
      probe_i8.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
      probe_sq.emplace(static_cast<std::size_t>(n_left), stream, mr);
      routed_i8.emplace(static_cast<std::size_t>(max_routed * dim), stream, mr);
      routed_sq.emplace(static_cast<std::size_t>(max_routed), stream, mr);
      vss::narrow_to_shifted_int8(queries.data_handle(), n_left * dim, probe_i8->data(), stream);
      vss::int8_row_sq_norms(probe_i8->data(), n_left, dim, probe_sq->data(), stream);
    }

    std::int64_t scanned_pairs = 0;
    auto prefetched            = needed_chunks.empty() || int8_search
                                   ? staged_vector_chunk{}
                                   : _corpus->stage(needed_chunks[0], *mem_space, stage_on);

    for (std::size_t ci = 0; ci < needed_chunks.size(); ++ci) {
      auto const j            = needed_chunks[ci];
      auto staged             = std::move(prefetched);
      auto const chunk_base   = _chunk_row_base[j];
      float const* chunk_data = nullptr;
      std::int64_t chunk_n    = 0;
      if (int8_search) {
        chunk_n = _lists->rows_in_chunk(static_cast<std::int64_t>(j));
      } else {
        auto const chunk_view = vss::list_column_as_dataset_view(staged.view, dim);
        chunk_data            = chunk_view.data_handle();
        chunk_n               = chunk_view.extent(0);
      }

      for (auto const& slice : _chunk_cluster_runs[j]) {
        auto const eb = edge_begin[static_cast<std::size_t>(slice.cluster)];
        auto const ee = edge_begin[static_cast<std::size_t>(slice.cluster) + 1];
        if (eb == ee) { continue; }
        // The slice was cut from the cluster column's chunk j; this is the first point at
        // which the vector column's chunk j is resident and its row count exactly known. The
        // two are the same pin's row groups, so a mismatch is a broken invariant rather than
        // a user error -- but it would read past the end of the chunk, so it is checked.
        if (slice.end > chunk_n) {
          throw std::runtime_error(
            "[sirius_physical_vector_join_stream] cluster column chunk " + std::to_string(j) +
            " describes row " + std::to_string(slice.end) + " but the vector column's chunk " +
            "holds " + std::to_string(chunk_n) + "; both must come from the same pin");
        }
        auto const slice_rows = slice.end - slice.begin;
        auto const slice_view =
          raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
            chunk_data == nullptr ? nullptr : chunk_data + slice.begin * dim, slice_rows, dim);
        auto const k_eff = std::min<std::int64_t>(k_join, slice_rows);

        // ONE search per slice: every row routed here searches the same dataset, and the
        // fold's dominant term is per call rather than per pair (X1 measured 278 us/call at
        // 42-96% of runtime). The rows are gathered into one query matrix; a row is gathered
        // once per cluster it visits, so the copies total n_probes probe batches per join --
        // the price of routing rows rather than runs.
        auto const m       = ee - eb;
        auto const* rows_c = routed_rows + eb;
        if (!int8_search) {
          vss::gather_rows(queries.data_handle(), dim, rows_c, m, routed_queries.data(), stream);
        }
        auto const queries_view =
          raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
            routed_queries.data(), m, dim);
        scanned_pairs += slice_rows * m;
        // Neighbour ids come back local to the slice; the slice's own start in corpus row
        // space is the base that makes them corpus row ids, exactly as the chunk offset does
        // in the exhaustive fold.
        auto const id_base = chunk_base + slice.begin;
        // Lists are in cluster order, not pin order; their row map turns a layout row back into
        // the pin row every later stage reads the corpus by.
        auto const* id_map = _lists != nullptr
                               ? static_cast<const std::int64_t*>(_lists->row_ids->data()) + id_base
                               : nullptr;

        if (radius_join) {
          // Same construction as the exhaustive radius path: a slice's in-range pairs are
          // final when produced, so they are appended, never folded, and there is no k. The
          // kernel numbers query rows within the gathered matrix; rows_c maps them back.
          auto edges = threshold_search(res, slice_view, queries_view, radius_eps, metric, mr);
          if (edges.n_edges > 0) {
            auto left = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  static_cast<cudf::size_type>(edges.n_edges),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
            vss::remap_radius_edges(edges.query_rows->view().data<std::int64_t>(),
                                    rows_c,
                                    left->mutable_view().data<std::int32_t>(),
                                    edges.neighbors->mutable_view().data<std::int64_t>(),
                                    edges.n_edges,
                                    id_base,
                                    stream,
                                    id_map);
            radius_left.push_back(std::move(left));
            radius_neighbors.push_back(std::move(edges.neighbors));
            radius_distances.push_back(std::move(edges.distances));
          }
          continue;
        }

        // The GEMM-ranked search, as in the exhaustive fold: it also carries none of cuVS's
        // per-call device synchronization, which is paid here once per slice.
        auto const knn = [&] {
          if (int8_search) {
            vss::gather_bytes(probe_i8->data(), dim, rows_c, m, routed_i8->data(), stream);
            vss::gather_int32(probe_sq->data(), rows_c, m, routed_sq->data(), stream);
            return vss::gemm_int8_topk(
              res,
              static_cast<std::int8_t const*>(_lists->device_vectors->data()) + id_base * dim,
              static_cast<std::int32_t const*>(_lists->row_sq->data()) + id_base,
              slice_rows,
              routed_i8->data(),
              routed_sq->data(),
              m,
              dim,
              k_eff,
              metric == cuvs::distance::DistanceType::L2SqrtExpanded,
              mr);
          }
          return gemm_search_enabled() && vss::gemm_search_supports(metric)
                   ? vss::gemm_topk(res, slice_view, queries_view, k_eff, metric, mr)
                   : vss::brute_force_knn_untrimmed(res,
                                                    vss::brute_force_build(res, slice_view, metric),
                                                    queries_view,
                                                    k_eff,
                                                    mr);
        }();

        // Fold in place: a row is routed to a cluster at most once, so the rows of one slice
        // are distinct and no two threads of the fold write the same accumulator row.
        vss::fold_topk_rows(acc_distances->mutable_view().data<float>(),
                            acc_neighbors->mutable_view().data<std::int64_t>(),
                            k_join,
                            knn.distances->view().data<float>(),
                            knn.neighbors->view().data<std::int64_t>(),
                            knn.k,
                            k_eff,
                            rows_c,
                            m,
                            id_base,
                            stream,
                            id_map);
      }

      // The searches above are issued, not finished. Staging the next needed chunk now runs its
      // H2D while the GPU works on this one.
      prefetched = (ci + 1 < needed_chunks.size() && !int8_search)
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
                   "[vecjoin] batch: %lld probe rows x %ld probes, %zu of %zu corpus chunks "
                   "staged, %lld of %lld probe-row x corpus-row pairs scored (%.2f%%)\n",
                   static_cast<long long>(n_left),
                   static_cast<long>(n_probes),
                   needed_chunks.size(),
                   _chunk_cluster_runs.size(),
                   static_cast<long long>(scanned_pairs),
                   static_cast<long long>(exhaustive_pairs),
                   exhaustive_pairs == 0 ? 0.0
                                         : 100.0 * static_cast<double>(scanned_pairs) /
                                             static_cast<double>(exhaustive_pairs));
    }
    phase("search+fold slices");
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
    borrow_window borrowed{stream};

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
      if (chunk.buffer) { chunk.buffer->set_stream(stream); }
      if (chunk.reader) { borrowed.hold(std::move(*chunk.reader)); }
      chunk = staged_vector_chunk{};
    };

    // Running [n_left x k_join] accumulator, seeded with misses (id -1 at infinite distance) so the
    // first chunk merges against nothing; a miss can only survive for a row with fewer than k
    // corpus rows in total.
    std::unique_ptr<cudf::column> all_rows;
    if (!radius_join) {
      cudf::numeric_scalar<std::int64_t> const miss_id(-1, true, stream);
      cudf::numeric_scalar<float> const miss_distance(
        std::numeric_limits<float>::infinity(), true, stream);
      auto const total = static_cast<cudf::size_type>(n_left * k_join);
      acc_neighbors    = cudf::make_column_from_scalar(miss_id, total, stream, mr);
      acc_distances    = cudf::make_column_from_scalar(miss_distance, total, stream, mr);
      cudf::numeric_scalar<std::int64_t> const zero(0, true, stream);
      cudf::numeric_scalar<std::int64_t> const one(1, true, stream);
      all_rows = cudf::sequence(static_cast<cudf::size_type>(n_left), zero, one, stream, mr);
    }
    std::size_t last_edge_bytes = 0;
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

      if (radius_join) {
        // Pushed-down corpus predicates: only the rows they keep are searched, and those rows
        // carry their corpus ids, so the kernel's local ids map back through them instead of
        // through the chunk offset.
        std::optional<filtered_chunk> kept;
        auto searched = dataset;
        if (!_request.right_predicates.empty()) {
          kept.emplace(filter_corpus_chunk(*_right_pin,
                                           _request.right_predicates,
                                           j,
                                           dataset,
                                           offset,
                                           *mem_space,
                                           stream,
                                           mr,
                                           batch_telemetry()));
          searched = raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
            kept->vectors.data(), kept->rows, dim);
        }
        // The search holds its score tile and the prepared chunk, and its edges grow the result;
        // the last chunk's edge count is the estimate for this one's.
        if (_corpus->is_streaming()) {
          ensure_device_headroom(_sirius_ctx,
                                 *mem_space,
                                 vss::gemm_search_tile_bytes() +
                                   static_cast<std::size_t>(batch_rows) *
                                     static_cast<std::size_t>(dim + 4) * sizeof(float) +
                                   2 * last_edge_bytes);
        }
        auto edges = searched.extent(0) == 0
                       ? vss::threshold_join_result{nullptr, nullptr, nullptr, 0}
                       : threshold_search(res, searched, queries, radius_eps, metric, mr);
        last_edge_bytes =
          static_cast<std::size_t>(edges.n_edges) * (2 * sizeof(std::int64_t) + sizeof(float));
        advance(j + 1);
        release_staged(staged);
        auto const chunk_base = offset;
        offset += batch_rows;
        if (edges.n_edges > 0) {
          if (kept) {
            auto mapped     = cudf::gather(cudf::table_view{{kept->row_ids->view()}},
                                       edges.neighbors->view(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);
            edges.neighbors = std::move(mapped->release().front());
          } else if (chunk_base != 0) {
            // Local dataset-batch rows -> right-table row space, the same shift the top-k path
            // applies to its neighbour ids.
            cudf::numeric_scalar<std::int64_t> const off_scalar(chunk_base, true, stream);
            edges.neighbors = cudf::binary_operation(edges.neighbors->view(),
                                                     off_scalar,
                                                     cudf::binary_operator::ADD,
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     stream,
                                                     mr);
          }
          // shaped_join_result carries left_rows as INT32; the kernel emits INT64.
          radius_left.push_back(cudf::cast(
            edges.query_rows->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr));
          radius_neighbors.push_back(std::move(edges.neighbors));
          radius_distances.push_back(std::move(edges.distances));
        }
        continue;
      }

      auto const k_eff = std::min<std::int64_t>(k_join, batch_rows);

      // Expanded L2 and cosine go through the GEMM-ranked search; the unexpanded `exact` mode keeps
      // cuVS.
      if (_corpus->is_streaming()) {
        ensure_device_headroom(
          _sirius_ctx,
          *mem_space,
          vss::gemm_search_tile_bytes() + static_cast<std::size_t>(batch_rows) *
                                            static_cast<std::size_t>(dim + 4) * sizeof(float));
      }
      auto knn = gemm_search_enabled() && vss::gemm_search_supports(metric)
                   ? vss::gemm_topk(res, dataset, queries, k_eff, metric, mr)
                   : vss::brute_force_knn(res, dataset, queries, k_eff, metric, mr);

      // The search above is issued, not finished. Staging the next chunk now runs its H2D
      // while the GPU works on this one; the host blocks inside the converter, the device
      // does not.
      advance(j + 1);
      release_staged(staged);

      // Fold in place, the chunk's ids shifted to corpus rows by the fold itself. Every probe row
      // is in every chunk's answer, so the rows are the identity. Unlike the knn_merge_parts
      // fold this replaced, a chunk shorter than k simply contributes fewer candidates, and k has
      // no 1024 ceiling.
      auto const chunk_base = offset;
      offset += batch_rows;
      exhaustive_rows_seen += batch_rows;
      vss::fold_topk_rows(acc_distances->mutable_view().data<float>(),
                          acc_neighbors->mutable_view().data<std::int64_t>(),
                          k_join,
                          knn.distances->view().data<float>(),
                          knn.neighbors->view().data<std::int64_t>(),
                          knn.k,
                          k_eff,
                          all_rows->view().data<std::int64_t>(),
                          n_left,
                          chunk_base,
                          stream);
    }
  }

  if (!radius_join && (!acc_neighbors || (_centroids == nullptr && exhaustive_rows_seen == 0))) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] right table produced no rows to join against");
  }

  vss::shaped_join_result shaped;
  if (radius_join) {
    // Concatenate the per-chunk edge lists. No merge and no truncation test: every edge the
    // kernel emitted is inside eps and nothing later can displace it, so the answer is complete
    // by construction rather than complete-if-k-was-big-enough.
    auto const join_cols = [&](std::vector<std::unique_ptr<cudf::column>>& parts,
                               cudf::type_id id) -> std::unique_ptr<cudf::column> {
      if (parts.empty()) { return cudf::make_empty_column(cudf::data_type{id}); }
      if (parts.size() == 1) { return std::move(parts.front()); }
      std::vector<cudf::column_view> views;
      views.reserve(parts.size());
      for (auto const& c : parts) {
        views.push_back(c->view());
      }
      return cudf::concatenate(views, stream, mr);
    };
    shaped.left_rows = join_cols(radius_left, cudf::type_id::INT32);
    shaped.neighbors = join_cols(radius_neighbors, cudf::type_id::INT64);
    shaped.distances = join_cols(radius_distances, cudf::type_id::FLOAT32);
  } else {
    // The fold is mode-independent; only which of its candidates survive is not.
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

  // The GEMM-ranked search holds its score tile, the prepared corpus chunk ([rows x d+pad]) and
  // the prepared probe chunk at once, and a threshold search its edge buffer (1M edges to start).
  // Left out, a task under a small pool reserved a fraction of what it used, ran out mid-fold
  // and was restarted from its first chunk, over and over.
  std::size_t gemm_scratch = 0;
  if (gemm_search_enabled()) {
    auto const row_bytes       = static_cast<std::size_t>(_request.dim + 4) * sizeof(float);
    std::size_t max_chunk_rows = 0;
    for (std::size_t i = 0; _corpus && i < _corpus->num_chunks(); ++i) {
      max_chunk_rows = std::max(max_chunk_rows, _corpus->chunk_rows(i));
    }
    gemm_scratch = vss::gemm_search_tile_bytes() + max_chunk_rows * row_bytes + n_left * row_bytes;
    if (_request.mode == vss::vector_join_mode::threshold) {
      gemm_scratch += (std::size_t{1} << 20) * (2 * sizeof(std::int64_t) + sizeof(float));
    }
  }

  return (block * 6) + std::max(cuvs_scratch, gemm_scratch) + staged_chunk + (std::size_t{1} << 20);
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
