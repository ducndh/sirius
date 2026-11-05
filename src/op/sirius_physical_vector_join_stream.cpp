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

#include "creator/task_creator.hpp"
#include "cuda/vss/brute_force_search.hpp"
#include "cuda/vss/brute_force_threshold.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "cuda/vss/knn_merge.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/block_host_rows.hpp"
#include "vss/bound_gemm.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"
#include "vss/device_rates.hpp"
#include "vss/distance_metric.hpp"
#include "vss/join_result_shaping.hpp"
#include "vss/pinned_column.hpp"
#include "vss/routed_edges.hpp"
#include "vss/size_limits.hpp"
#include "vss/spilled_read.hpp"
#include "vss/staging_shortfall.hpp"
#include "vss/vector_clustering.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

// Mean seed work per (task, nearest cluster) slice, in multiply-adds, from which a task over
// clustered FLOAT16 device lists ranks its nearest clusters on tensor cores instead of seeding
// them. On an RTX A5000 the ranked pass costs 2-5x the seed below ~3e6, 1.03x at 1.0e7, 0.94x at
// 2.7e7 and 0.79-0.84x from 8e7. 2^24 stays within 10 % of the faster method at every point
// measured.
constexpr double kSweep0RankedSliceMacs = 0x1p24;

/// Share of the device one probe block over a scanned corpus may plan for.
constexpr double kScannedBlockDeviceShare = 0.8;

/// Share of the host pool the probe rows of the blocks in flight may plan for. Below the 80 % at
/// which the host's downgrade monitor stops by default, so they do not spill themselves.
constexpr double kBlockRowsHostShare = 0.6;

/// Probe blocks at once over a scanned corpus (SIRIUS_VSS_DEVICE_BLOCK_TASKS, default 1).
std::size_t scanned_block_tasks()
{
  auto const* v = std::getenv("SIRIUS_VSS_DEVICE_BLOCK_TASKS");
  return v != nullptr ? static_cast<std::size_t>(std::max(1.0, std::atof(v))) : 1;
}

/// Pair tasks at once over one block by default: while one reads its chunk the other's kernels run.
constexpr std::size_t kPairTasksAtOnce = 2;

/// Probe rows one threshold search of a pair task takes: its prepared copy of them stays within
/// what a score tile holds, and its pairs leave between searches. SIRIUS_VSS_PAIR_CALL_ROWS sets
/// it.
std::int64_t pair_call_rows(std::int64_t dim)
{
  if (auto const* v = std::getenv("SIRIUS_VSS_PAIR_CALL_ROWS"); v != nullptr) {
    return std::max<std::int64_t>(1, std::atoll(v));
  }
  return std::max<std::int64_t>(
    1,
    static_cast<std::int64_t>(vss::gemm_search_tile_bytes() /
                              (static_cast<std::size_t>(dim + 4) * sizeof(float))));
}

/// Whether a threshold search that threw would fit over fewer probe rows.
bool search_can_shrink(std::exception const& thrown)
{
  return dynamic_cast<rmm::out_of_memory const*>(&thrown) != nullptr ||
         dynamic_cast<vss::threshold_pairs_exceed_device const*>(&thrown) != nullptr;
}

/// Waits for the stream as a scope ends, before the buffers and locks declared ahead of it go.
class sync_stream_on_exit {
 public:
  explicit sync_stream_on_exit(rmm::cuda_stream_view stream) : _stream(stream) {}
  sync_stream_on_exit(sync_stream_on_exit const&)            = delete;
  sync_stream_on_exit& operator=(sync_stream_on_exit const&) = delete;
  ~sync_stream_on_exit() { _stream.synchronize_no_throw(); }

 private:
  rmm::cuda_stream_view _stream;
};

/// A task's place under vector_join_admission: released, it makes room and asks the task creator
/// for this join's next tasks, as a streaming source does when data arrives (schedule() only
/// enqueues, so any thread may call it).
struct admission_ticket {
  explicit admission_ticket(std::shared_ptr<vector_join_admission> admission)
    : _admission(std::move(admission))
  {
  }
  admission_ticket(admission_ticket const&)            = delete;
  admission_ticket& operator=(admission_ticket const&) = delete;
  ~admission_ticket()
  {
    {
      std::lock_guard<std::mutex> lock(_admission->mutex);
      --_admission->in_flight;
    }
    try {
      if (auto p = _admission->pipeline.lock()) {
        auto* creator = p->get_task_creator();
        auto head     = p->get_source();
        if (creator != nullptr && head) { creator->schedule(head.get()); }
      }
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("[sirius_physical_vector_join_stream] re-arming after a task: {}", e.what());
    }
  }

 private:
  std::shared_ptr<vector_join_admission> _admission;
};

/// A host-tier pin chunk as the uncompressed host table the corpus stream slices and sizes. A
/// compressed pin chunk has no per-column size to stage by, so it is refused with what to change.
cucascade::host_data_representation const& uncompressed_host_chunk(
  cucascade::idata_representation const& chunk)
{
  auto const* host = dynamic_cast<cucascade::host_data_representation const*>(&chunk);
  if (host == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] the corpus is pinned with host compression, which "
      "the vector join cannot stream; pin it with compression => false");
  }
  return *host;
}

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
                           const telemetry::batch_telemetry_info& telemetry_info)
    : _pin(pin), _telemetry_info(telemetry_info)
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
  // copy happens, and a probe block can be cut to exact rows.
  [[nodiscard]] std::size_t chunk_bytes(std::size_t i) const override
  {
    auto const& chunk = _pin.host_chunks.at(i);
    return chunk ? uncompressed_host_chunk(*chunk).column_size(_column_index) : 0;
  }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override
  {
    auto const& chunk = _pin.host_chunks.at(i);
    if (!chunk) { return 0; }
    return static_cast<std::size_t>(
      uncompressed_host_chunk(*chunk).get_host_table()->columns.at(_column_index).num_rows);
  }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) override
  {
    return copy_in(i, space, stream, /*own_reservation=*/true);
  }
  staged_vector_chunk stage_for_task(std::size_t i,
                                     cucascade::memory::memory_space& space,
                                     rmm::cuda_stream_view stream) override
  {
    return copy_in(i, space, stream, /*own_reservation=*/false);
  }

 private:
  staged_vector_chunk copy_in(std::size_t i,
                              cucascade::memory::memory_space& space,
                              rmm::cuda_stream_view stream,
                              bool own_reservation)
  {
    auto const& chunk = _pin.host_chunks.at(i);
    if (!chunk) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] host chunk " +
                               std::to_string(i) + " is null");
    }
    // Slice to the vector column alone: the rest of the pinned table is dead weight on
    // the wire and this copy is the operator's bandwidth budget.
    std::array<std::size_t, 1> const cols{_column_index};
    auto data_rep    = uncompressed_host_chunk(*chunk).slice(cols);
    auto const bytes = data_rep->get_size_in_bytes();

    // A chunk staged for the corpus fold gets a reservation of its own: it is copied on the
    // staging stream, which no task reservation covers. A null one is retried once a downgrade
    // has freed memory, unless the chunk could never fit.
    std::shared_ptr<cucascade::memory::reservation> reservation;
    if (own_reservation) {
      reservation = space.make_reservation_or_null(bytes);
      if (!reservation) {
        vss::throw_staging_shortfall(
          space, bytes, "[sirius_physical_vector_join_stream] host chunk " + std::to_string(i));
      }
    }

    auto const batch_id = sirius::get_next_batch_id();
    auto batch          = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));

    {
      auto mut = batch->to_mutable();
      if (reservation) {
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), *reservation, stream);
      } else {
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), &space, stream);
      }
      mut.rebind_stream(stream);  // the converter binds the copy to its own stream
    }
    auto const table = sirius::get_cudf_table_view(*batch);
    return staged_vector_chunk{table.column(0), std::move(batch), std::move(reservation)};
  }

  const scan_manager::pinned_entry& _pin;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
};

/// A column of a batch the engine took to disk, read straight from the batch's file into one
/// device buffer: a flat fixed-width column (cluster labels), or a LIST of one (the vector
/// column) as its offsets and values. The batch stays on disk. Restoring it instead moves it back
/// into the host pool, where with a corpus larger than that pool every probe task's pass cycles
/// the whole corpus disk -> host -> disk, and the restore fails once the pool is committed.
/// Nullopt for any other shape, or nulls, which the full restore handles.
struct spilled_column_layout {
  cucascade::memory::column_metadata const* offsets{nullptr};  // null for a flat column
  cucascade::memory::column_metadata const* values{nullptr};
  std::size_t values_at{0};  // where the values start in the buffer
  std::size_t bytes{0};
};

std::optional<spilled_column_layout> spilled_layout(cucascade::read_only_data_batch const& ro,
                                                    std::size_t column)
{
  auto const* disk = dynamic_cast<cucascade::disk_data_representation const*>(ro.get_data());
  if (disk == nullptr) { return std::nullopt; }
  auto const& meta = disk->get_disk_table().columns.at(column);
  auto const flat  = [](cucascade::memory::column_metadata const& m) {
    auto const type = cudf::data_type{static_cast<cudf::type_id>(m.type_id)};
    return m.children.empty() && cudf::is_fixed_width(type) && !cudf::is_fixed_point(type) &&
           (!m.has_null_mask || m.null_count == 0) && (m.has_data || m.num_rows == 0);
  };
  if (meta.has_null_mask && meta.null_count != 0) { return std::nullopt; }
  if (flat(meta)) { return spilled_column_layout{nullptr, &meta, 0, meta.data_size}; }
  if (static_cast<cudf::type_id>(meta.type_id) != cudf::type_id::LIST ||
      meta.children.size() != 2 || !flat(meta.children[0]) || !flat(meta.children[1]) ||
      static_cast<cudf::type_id>(meta.children[0].type_id) != cudf::type_id::INT32) {
    return std::nullopt;
  }
  auto const values_at = rmm::align_up(meta.children[0].data_size, std::size_t{256});
  return spilled_column_layout{
    &meta.children[0], &meta.children[1], values_at, values_at + meta.children[1].data_size};
}

/// Reads @p layout's buffers from the file of the spilled batch @p ro (held read-locked, which
/// keeps the file in place) into @p buffer and returns the column's view over it.
cudf::column_view read_spilled_column(cucascade::read_only_data_batch const& ro,
                                      std::size_t column,
                                      spilled_column_layout const& layout,
                                      rmm::device_buffer& buffer,
                                      rmm::cuda_stream_view stream)
{
  auto const& path = dynamic_cast<cucascade::disk_data_representation const&>(*ro.get_data())
                       .get_disk_table()
                       .file_path;
  int const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] cannot open spilled batch " +
                             path);
  }
  struct close_on_exit {
    int fd;
    ~close_on_exit() { ::close(fd); }
  } closer{fd};
  auto copy = [&](cucascade::memory::column_metadata const& m, std::size_t at) {
    vss::copy_file_to_device(
      fd, m.data_offset, m.data_size, static_cast<std::byte*>(buffer.data()) + at, stream, path);
  };
  auto const& values = *layout.values;
  if (layout.offsets != nullptr) { copy(*layout.offsets, 0); }
  copy(values, layout.values_at);
  cudf::column_view const values_view{
    cudf::data_type{static_cast<cudf::type_id>(values.type_id)},
    values.num_rows,
    static_cast<std::byte const*>(buffer.data()) + layout.values_at,
    nullptr,
    0};
  if (layout.offsets == nullptr) { return values_view; }
  auto const& meta = dynamic_cast<cucascade::disk_data_representation const&>(*ro.get_data())
                       .get_disk_table()
                       .columns.at(column);
  cudf::column_view const offsets_view{
    cudf::data_type{cudf::type_id::INT32}, layout.offsets->num_rows, buffer.data(), nullptr, 0};
  return cudf::column_view{cudf::data_type{cudf::type_id::LIST},
                           meta.num_rows,
                           nullptr,
                           nullptr,
                           0,
                           0,
                           {offsets_view, values_view}};
}

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
    if (_ctx != nullptr) {
      auto spaces =
        _ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
      if (!spaces.empty()) { _host_space = spaces.front(); }
    }
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
      } else if (ro.get_current_tier() == cucascade::memory::Tier::DISK) {
        // From the file's column metadata, which stays in memory: restoring every spilled batch
        // just to count it would bring a corpus larger than the host pool back at once.
        auto const* data = ro.get_data();
        if (data == nullptr) {
          throw std::runtime_error(
            "[sirius_physical_vector_join_stream] build-side batch has no data representation");
        }
        auto const& col =
          data->cast<cucascade::disk_data_representation>().get_disk_table().columns.at(
            _column_index);
        _bytes.push_back(column_bytes(col));
        _rows.push_back(static_cast<std::size_t>(col.num_rows));
      } else {
        // Rows from the column's own count: its bytes include the list offsets, and a probe
        // block is cut to exact rows.
        auto const& host = host_repr(ro);
        _bytes.push_back(host.column_size(_column_index));
        _rows.push_back(
          static_cast<std::size_t>(host.get_host_table()->columns.at(_column_index).num_rows));
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
    return resident(i, space, stream, /*own_reservation=*/true);
  }
  staged_vector_chunk stage_for_task(std::size_t i,
                                     cucascade::memory::memory_space& space,
                                     rmm::cuda_stream_view stream) override
  {
    return resident(i, space, stream, /*own_reservation=*/false);
  }
  void count_reads() override { _count_reads = true; }

 private:
  void note_read(std::size_t bytes_moved) const
  {
    if (_count_reads && _ctx != nullptr) { _ctx->record_vector_join_corpus_stage(bytes_moved); }
  }

  staged_vector_chunk resident(std::size_t i,
                               cucascade::memory::memory_space& space,
                               rmm::cuda_stream_view stream,
                               bool own_reservation)
  {
    auto batch = fetch(_batch_ids.at(i));
    auto ro    = batch->to_read_only();
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      note_read(0);
      auto const table = sirius::get_cudf_table_view(ro);
      auto const view  = table.column(static_cast<cudf::size_type>(_column_index));
      // Borrowed, not owned: the reader holds the batch alive as well as locked, and `owner`
      // stays null so the fold does not try to rebind a stream on memory it did not allocate.
      return staged_vector_chunk{view, nullptr, nullptr, std::move(ro)};
    }

    // Spilled: copy just the vector column back, the rest of the batch is dead weight on the
    // wire. As on the HOST-tier pin path, a chunk staged for the corpus fold reserves its own
    // copy (spilling first if it must), while the task's probe chunk draws on the task's budget.
    // The slice references the batch's host allocation, so the borrow is held across the copy.
    // A batch the engine took all the way to disk is read from its file, the column alone, and
    // stays there; a shape that read does not take comes back to host through the engine.
    if (ro.get_current_tier() == cucascade::memory::Tier::DISK) {
      if (auto const layout = spilled_layout(ro, _column_index)) {
        note_read(layout->bytes);
        std::shared_ptr<cucascade::memory::reservation> reservation;
        if (own_reservation) {
          reservation = vss::reserve_or_spill(_ctx, space, layout->bytes);
          if (!reservation) {
            vss::throw_staging_shortfall(
              space,
              layout->bytes,
              "[sirius_physical_vector_join_stream] batch " + std::to_string(i));
          }
        }
        auto buffer = std::make_unique<rmm::device_buffer>(
          layout->bytes,
          stream,
          reservation ? reservation->get_memory_resource() : space.get_default_allocator());
        auto const view = read_spilled_column(ro, _column_index, *layout, *buffer, stream);
        staged_vector_chunk out{view, nullptr, std::move(reservation)};
        out.buffer = std::move(buffer);
        return out;
      }
      ro = host_resident(batch, std::move(ro));
    }
    std::array<std::size_t, 1> const cols{_column_index};
    auto data_rep    = host_repr(ro).slice(cols);
    auto const bytes = data_rep->get_size_in_bytes();
    note_read(bytes);

    std::shared_ptr<cucascade::memory::reservation> reservation;
    if (own_reservation) {
      reservation = vss::reserve_or_spill(_ctx, space, bytes);
      if (!reservation) {
        vss::throw_staging_shortfall(
          space, bytes, "[sirius_physical_vector_join_stream] batch " + std::to_string(i));
      }
    }

    auto const batch_id = sirius::get_next_batch_id();
    auto staged         = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(_telemetry_info, batch_id));
    {
      auto mut = staged->to_mutable();
      if (reservation) {
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), *reservation, stream);
      } else {
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), &space, stream);
      }
      mut.rebind_stream(stream);  // the converter binds the copy to its own stream
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

  /// Brings a disk-resident build batch back to the host tier through the engine (in place) and
  /// returns a fresh read lock on it; @p ro is the lock the caller held and is released first.
  cucascade::read_only_data_batch host_resident(const std::shared_ptr<cucascade::data_batch>& batch,
                                                cucascade::read_only_data_batch ro) const
  {
    if (_host_space == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] build-side batch is on disk and no HOST memory "
        "space is available to restore it");
    }
    std::ignore   = cucascade::data_batch::to_idle(std::move(ro));
    auto prepared = pipeline::lock_and_prepare_batch(
      batch, _host_space, ::cuda::stream_ref{rmm::cuda_stream_per_thread.get()});
    if (!prepared) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] could not restore a disk-resident build batch");
    }
    return std::visit([](auto& r) { return std::move(r.ro_lock); }, *prepared);
  }

  /// A spilled column's bytes: its own buffers and its children's (a list's offsets and values).
  static std::size_t column_bytes(cucascade::memory::column_metadata const& col)
  {
    std::size_t bytes = (col.has_data ? col.data_size : 0) + col.null_mask_size;
    for (auto const& child : col.children) {
      bytes += column_bytes(child);
    }
    return bytes;
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
  const cucascade::memory::memory_space* _host_space{nullptr};
  std::vector<std::uint64_t> _batch_ids;
  std::vector<std::size_t> _rows;
  std::vector<std::size_t> _bytes;
  telemetry::batch_telemetry_info _telemetry_info;
  std::size_t _column_index{0};
  std::int64_t _dim{0};
  bool _count_reads{false};
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
  stream.sync();
  return out;
}

/// One chunk of an `earlier_by` column as INT64 into @p out. Integer columns without nulls only: a
/// null key compares false against every other, which the best-match fold has no encoding for.
void copy_earlier_keys(cudf::column_view const& keys,
                       std::int64_t* out,
                       std::string const& name,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  if (!cudf::is_integral_not_bool(keys.type())) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] earlier_by column '" + name +
                             "' must be an integer column");
  }
  if (keys.has_nulls()) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] earlier_by column '" + name +
                             "' holds NULLs, which a best-match join cannot order");
  }
  if (keys.size() == 0) { return; }
  std::unique_ptr<cudf::column> widened;
  auto view = keys;
  if (keys.type().id() != cudf::type_id::INT64) {
    widened = cudf::cast(keys, cudf::data_type{cudf::type_id::INT64}, stream, mr);
    view    = widened->view();
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(out,
                                view.data<std::int64_t>(),
                                static_cast<std::size_t>(view.size()) * sizeof(std::int64_t),
                                cudaMemcpyDeviceToDevice,
                                stream.get()));
  // The caller drops the staged chunk, and this drops the widened copy, once it returns.
  stream.sync();
}

constexpr std::int64_t kHostBlockPairs = std::int64_t{1} << 22;
constexpr std::size_t kHostBlockBytes =
  static_cast<std::size_t>(kHostBlockPairs) *
  (sizeof(std::int32_t) + sizeof(std::int64_t) + sizeof(float));

/// One block of pairs on the host: rows, ids and codes side by side, each with room for
/// kHostBlockPairs. Pinned, because a copy to pageable memory runs at a fraction of the link's
/// rate (2.3 GB/s against about 25 GB/s, RTX A5000).
struct host_pair_block {
  std::byte* mem = nullptr;
  std::int64_t n = 0;
  [[nodiscard]] std::int32_t* rows() const { return reinterpret_cast<std::int32_t*>(mem); }
  [[nodiscard]] std::int64_t* ids() const
  {
    return reinterpret_cast<std::int64_t*>(mem + static_cast<std::size_t>(kHostBlockPairs) * 4);
  }
  [[nodiscard]] float* codes() const
  {
    return reinterpret_cast<float*>(mem + static_cast<std::size_t>(kHostBlockPairs) * 12);
  }
};

/// A task's pairs on the host, in arrival order. Its blocks go back to the operator however the
/// task ends.
class host_pair_queue {
 public:
  explicit host_pair_queue(host_pair_block_pool& pool) : _pool(pool) {}
  ~host_pair_queue()
  {
    while (!_held.empty()) {
      drop_front();
    }
    give_back_spare();
  }
  host_pair_queue(host_pair_queue const&)            = delete;
  host_pair_queue& operator=(host_pair_queue const&) = delete;

  [[nodiscard]] bool empty() const { return _held.empty(); }
  [[nodiscard]] host_pair_block const& front() const { return _held.front(); }
  /// Blocks with room for @p pairs more, all of it or none: false when the budget would not cover
  /// them. Taken before any copy is queued, so a failed pin leaves no copy in flight to a block.
  [[nodiscard]] bool reserve(std::int64_t pairs, std::size_t budget_bytes)
  {
    auto room = (_held.empty() ? std::int64_t{0} : kHostBlockPairs - _held.back().n) +
                static_cast<std::int64_t>(_spare.size()) * kHostBlockPairs;
    while (room < pairs) {
      auto* block = static_cast<std::byte*>(_pool.take(budget_bytes));
      if (block == nullptr) {
        give_back_spare();
        return false;
      }
      _spare.push_back(block);
      room += kHostBlockPairs;
    }
    return true;
  }
  /// The block the next pair goes in, from those reserved.
  [[nodiscard]] host_pair_block& tail()
  {
    if (_held.empty() || _held.back().n == kHostBlockPairs) {
      _held.push_back(host_pair_block{_spare.back(), 0});
      _spare.pop_back();
    }
    return _held.back();
  }
  void drop_front()
  {
    _pool.give_back(_held.front().mem);
    _held.pop_front();
  }

 private:
  void give_back_spare()
  {
    for (auto* block : _spare) {
      _pool.give_back(block);
    }
    _spare.clear();
  }

  host_pair_block_pool& _pool;
  std::deque<host_pair_block> _held;
  std::vector<std::byte*> _spare;
};

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

/// Whether a clustered top-k join over a scanned corpus (no cluster lists) codes each staged chunk
/// to FLOAT16 and runs the bounded search over it. SIRIUS_VSS_SCAN_BOUNDED=0 keeps the FP32 ranked
/// search for every slice.
bool scan_bounded_enabled()
{
  auto const* v = std::getenv("SIRIUS_VSS_SCAN_BOUNDED");
  return v == nullptr || std::string_view{v} != "0";
}

/// Probe rows below which the exhaustive fold keeps cuVS (the per-slice searches of the clustered
/// path always use the GEMM form: cuVS there also syncs the device per call). The GEMM search
/// rewrites the corpus chunk into its prepared layout on every call, which a GEMM over hundreds of
/// queries amortizes and a handful does not: on Vec-H (1024-d, 2.4M rows) one-query searches ran
/// 10-30 ms slower through it, and on SIFT1M it wins from ~1k probes.
constexpr std::int64_t kGemmMinProbeRows = 512;

vss::threshold_join_result threshold_search(raft::device_resources const& res,
                                            vss::dataset_matrix_view dataset,
                                            vss::dataset_matrix_view queries,
                                            float eps,
                                            cuvs::distance::DistanceType metric,
                                            rmm::device_async_resource_ref mr,
                                            std::int64_t min_gemm_rows = 0,
                                            bool count_only            = false)
{
  return gemm_search_enabled() && vss::gemm_search_supports(metric) &&
             queries.extent(0) >= min_gemm_rows
           ? vss::gemm_threshold(res, dataset, queries, eps, metric, mr, count_only)
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
    CUDF_CUDA_TRY(cudaEventRecord(done, _stream.get()));
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
  /// With @p exact_rows, a FLOAT16 chunk is staged from the FP32 rows the lists kept instead of
  /// widened from its half rows: for a search that must score the rows as they are.
  explicit cluster_lists_chunk_source(const vss::cluster_lists& lists,
                                      bool exact_rows                = false,
                                      duckdb::SiriusContext* context = nullptr)
    : _lists(lists),
      _exact_rows(exact_rows && lists.encoding == vss::list_encoding::float16 &&
                  lists.has_exact_rows()),
      _context(context)
  {
  }

  [[nodiscard]] std::size_t num_chunks() const override
  {
    return static_cast<std::size_t>(_lists.num_chunks());
  }
  /// A HOST-tier chunk is copied in and a UINT8 one is widened into a buffer: either way the
  /// fold must size its budget for a staged FP32 chunk.
  [[nodiscard]] bool is_streaming() const override
  {
    return _lists.tier == cucascade::memory::Tier::HOST ||
           _lists.encoding != vss::list_encoding::float32;
  }
  [[nodiscard]] std::size_t chunk_rows(std::size_t i) const override
  {
    return static_cast<std::size_t>(_lists.rows_in_chunk(static_cast<std::int64_t>(i)));
  }
  [[nodiscard]] std::size_t chunk_bytes(std::size_t i) const override
  {
    if (!is_streaming()) { return 0; }
    auto const staged_u8 =
      _lists.tier == cucascade::memory::Tier::HOST && _lists.encoding != vss::list_encoding::float32
        ? chunk_rows(i) * _lists.row_bytes()
        : 0;
    return chunk_rows(i) * float_row_bytes() + staged_u8;
  }

  staged_vector_chunk stage(std::size_t i,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) override
  { return staged_chunk(i, space, stream, /*own_reservation=*/true); }
  staged_vector_chunk stage_for_pair(std::size_t i,
                                     cucascade::memory::memory_space& space,
                                     rmm::cuda_stream_view stream) override
  { return staged_chunk(i, space, stream, /*own_reservation=*/false); }
  void count_reads() override { _count_reads = true; }

 private:
  staged_vector_chunk staged_chunk(std::size_t i,
                                   cucascade::memory::memory_space& space,
                                   rmm::cuda_stream_view stream,
                                   bool own_reservation)
  {
    auto const rows    = static_cast<std::int64_t>(chunk_rows(i));
    auto const first   = static_cast<std::int64_t>(i) * _lists.chunk_rows;
    bool const narrow  = _lists.encoding != vss::list_encoding::float32;
    bool const on_host = _lists.tier == cucascade::memory::Tier::HOST;
    if (!is_streaming()) {
      auto const* data =
        static_cast<const float*>(_lists.device_vectors->data()) + first * _lists.dim;
      return staged_vector_chunk{list_view(data, rows), nullptr, nullptr};
    }

    auto const bytes = chunk_bytes(i);
    if (_count_reads && _context != nullptr) { _context->record_vector_join_corpus_stage(bytes); }
    std::shared_ptr<cucascade::memory::reservation> reservation;
    if (own_reservation) {
      reservation = space.make_reservation_or_null(bytes);
      if (!reservation) {
        vss::throw_staging_shortfall(
          space, bytes, "[sirius_physical_vector_join_stream] corpus chunk " + std::to_string(i));
      }
    }
    auto const mr =
      reservation ? reservation->get_memory_resource() : space.get_default_allocator();
    auto buffer   = std::make_unique<rmm::device_buffer>(
      static_cast<std::size_t>(rows) * float_row_bytes(), stream, mr);
    if (_exact_rows) {
      // Layout rows [first, first + rows) of the kept FP32 rows, which the re-score reads too.
      vss::gather_exact_rows(_lists.exact_source(),
                             first,
                             1,
                             rows,
                             _lists.dim,
                             static_cast<float*>(buffer->data()),
                             stream);
      stream.sync();
      staged_vector_chunk staged{list_view(static_cast<const float*>(buffer->data()), rows),
                                 nullptr,
                                 std::move(reservation)};
      staged.buffer = std::move(buffer);
      return staged;
    }

    // The chunk in its stored encoding, device-side: in place for the GPU tier, copied in block by
    // block for the HOST tier -- a chunk starts on a block boundary, so it is whole blocks plus a
    // short last one.
    std::byte const* stored = nullptr;
    std::optional<rmm::device_buffer> copied;
    // The read lock keeps the engine from spilling the chunk while the copy below is in flight;
    // it is released after the synchronize at the end.
    std::optional<cucascade::read_only_data_batch> lock;
    if (on_host) {
      if (narrow) {
        copied.emplace(static_cast<std::size_t>(rows) * _lists.row_bytes(), stream, mr);
      }
      auto* out = static_cast<std::byte*>(narrow ? copied->data() : buffer->data());
      lock.emplace(vss::lock_host_list_chunk(_lists, i, stream));
      vss::copy_host_list_rows(_lists, *lock, 0, rows, out, stream);
      if (narrow) { stored = static_cast<std::byte const*>(copied->data()); }
    } else {
      stored = static_cast<std::byte const*>(_lists.device_vectors->data()) +
               static_cast<std::size_t>(first) * _lists.row_bytes();
    }
    if (narrow) {
      if (_lists.encoding == vss::list_encoding::float16) {
        vss::widen_float16(reinterpret_cast<std::uint16_t const*>(stored),
                           rows * _lists.dim,
                           static_cast<float*>(buffer->data()),
                           stream);
      } else {
        vss::widen_shifted_int8(reinterpret_cast<std::int8_t const*>(stored),
                                rows * _lists.dim,
                                static_cast<float*>(buffer->data()),
                                stream);
      }
    }
    // Synchronous like the pin converter: the caller overlaps "host waits on this copy" with
    // the compute it already issued, and the view must be complete before it is searched. It is
    // also what lets the byte staging buffer go at the end of this call.
    stream.sync();
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
  bool _exact_rows{false};
  duckdb::SiriusContext* _context{nullptr};
  bool _count_reads{false};
};

/// A corpus chunk for the fold: staged on @p stage_on under a reservation of its own, so its copy
/// overlaps the compute on @p stream. That reservation is outside the one the task was admitted
/// with, so it fails once another task's admission has taken the device's free memory; the
/// chunk then comes from the task's own reservation, on @p stream, which the task's estimate
/// already counts its staged chunks in. Failing instead retried the task with a doubled
/// reservation, which left even less room for staging, until the retry asked for the whole device
/// and waited for it forever (Deep10M self-join on a 2.5 GB device, two tasks admitted at once).
staged_vector_chunk stage_for_fold(vector_chunk_source& source,
                                   std::size_t j,
                                   cucascade::memory::memory_space& space,
                                   rmm::cuda_stream_view stage_on,
                                   rmm::cuda_stream_view stream)
{
  try {
    return source.stage(j, space, stage_on);
  } catch (rmm::out_of_memory const&) {
    return source.stage_for_task(j, space, stream);
  }
}

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
  const vss::cluster_lists& lists, duckdb::SiriusContext* ctx)
{ return std::make_unique<cluster_lists_chunk_source>(lists, /*exact_rows=*/false, ctx); }

std::unique_ptr<vector_chunk_source> make_host_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  return std::make_unique<host_pinned_chunk_source>(pin, column, telemetry_info);
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
  if (_probe_side && !_request.probe_shares_build && !ready("default")) { return false; }
  return true;
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::ensure_initialized_locked()
{
  if (_initialized || _init_error) { return; }
  // The callers are the scheduler's hooks, which run where an exception ends the process (a
  // task's destructor re-evaluates the pipeline through them). A failure is kept instead, and the
  // one task scheduled for it throws it, which fails the query the way any task's error does.
  try {
    initialize_locked();
  } catch (...) {
    _init_error = std::current_exception();
  }
}

void sirius_physical_vector_join_stream::initialize_locked()
{
  if (!build_side_ready_locked()) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_stream] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  auto left_pin = _probe_side ? nullptr
                              : _scan_manager->find_pinned_entry_for_duckdb_table(
                                  left.catalog, left.schema, left.table, left.table_identity);
  // The corpus comes from the build port on the build path, so only the probe side has to be
  // pinned there.
  auto right_pin = _build_side ? nullptr
                               : _scan_manager->find_pinned_entry_for_duckdb_table(
                                   right.catalog, right.schema, right.table, right.table_identity);
  if ((!_probe_side && left_pin == nullptr) || (!_build_side && right_pin == nullptr)) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] left or right table is no longer pinned");
  }
  _right_pin = right_pin;
  _left_pin  = left_pin;

  // Both sides go behind the same seam. The probe side used to be required GPU-resident on
  // the argument that it is the small one; that is false for any join where both sides are
  // large (rec-sys candidate generation is the motivating case). One task already handles one
  // probe chunk, so streaming the probe is the same change staging the corpus was: stage the
  // chunk this task owns, search the whole corpus against it, release.
  if (_probe_side) {
    // A self-join's probe is the corpus side's own batches, in the snapshot both read.
    auto* port = get_port(_request.probe_shares_build ? "build" : "default");
    _probe_side->ensure_snapshot(*port->repo);
    _probe = make_materialized_chunk_source(
      *_probe_side, /*column_index=*/0, _request.dim, batch_telemetry(), _sirius_ctx);
    if (_request.probe_shares_build && std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[vecjoin-phase] self-join: the probe reads the corpus scan's %zu batches\n",
                   _probe->num_chunks());
    }
  } else if (left_pin->tier == cucascade::memory::Tier::HOST) {
    _probe = make_host_pinned_chunk_source(*left_pin, left.column, batch_telemetry());
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
    if (_lists == nullptr || _lists->pin != right_pin.get() ||
        _lists->pin_snapshot.lock() != right_pin->snapshot_identity ||
        _lists->column != right.column ||
        _lists->n_rows != static_cast<std::int64_t>(right_pin->num_rows)) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] clustering '" +
                               _request.clustering + "' has no lists over the current pin of '" +
                               right.table + "." + right.column +
                               "'; run sirius_kmeans_build_lists after pinning");
    }
    _corpus = make_cluster_lists_chunk_source(*_lists, _sirius_ctx);
    for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
      _max_chunk_bytes = std::max(_max_chunk_bytes, _corpus->chunk_bytes(i));
    }
    // Host-tier chunks are engine batches; registering them in the query's repository manager
    // is what makes them visible to the downgrade executor, which only sweeps repositories.
    if (_lists->tier == cucascade::memory::Tier::HOST && _lists_repo == nullptr &&
        _sirius_ctx != nullptr && get_pipeline() != nullptr) {
      auto manager = _sirius_ctx->get_data_repository_manager(get_pipeline()->get_query_id());
      if (manager) {
        auto repo = std::make_unique<cucascade::shared_data_repository>();
        for (auto const& batch : _lists->chunk_batches) {
          repo->add_data_batch(batch);
        }
        _lists_repo = repo.get();
        manager->add_new_repository(get_operator_id(), "lists", std::move(repo));
      }
    }
  } else if (right_pin->tier == cucascade::memory::Tier::HOST) {
    _corpus = make_host_pinned_chunk_source(*right_pin, right.column, batch_telemetry());
    // Sized from the widest chunk, since any one of them may be the one in flight when
    // the reservation is granted.
    auto const& names = right_pin->cache_info.column_names();
    auto const it     = std::find(names.begin(), names.end(), right.column);
    if (it != names.end()) {
      auto const col = static_cast<std::size_t>(std::distance(names.begin(), it));
      for (auto const& chunk : right_pin->host_chunks) {
        if (chunk) {
          _max_chunk_bytes =
            std::max(_max_chunk_bytes, uncompressed_host_chunk(*chunk).column_size(col));
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

  _corpus->count_reads();
  for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
    _max_corpus_chunk_rows = std::max(_max_corpus_chunk_rows, _corpus->chunk_rows(i));
  }
  _num_left = _probe->num_chunks();
  // A probe that stood in for a scalar subquery keeps the subquery's contract: one row. Counted
  // here, raised in execute(), where an exception becomes the query's error.
  if (_request.probe_scalar) {
    _scalar_probe_rows = 0;
    for (std::size_t i = 0; i < _num_left; ++i) {
      _scalar_probe_rows += _probe->chunk_rows(i);
    }
  }
  if (_probe->is_streaming()) {
    for (std::size_t i = 0; i < _num_left; ++i) {
      _max_probe_chunk_bytes = std::max(_max_probe_chunk_bytes, _probe->chunk_bytes(i));
    }
  }
  plan_probe_blocks_locked();
  if (pair_tasks_wanted() && !_blocks.empty()) {
    _pair_schedule.emplace(_blocks.size(), scanned_block_tasks());
    _pair_states.resize(_blocks.size());
    _corpus_chunk_base.assign(_corpus->num_chunks(), 0);
    std::int64_t base = 0;
    for (std::size_t i = 0; i < _corpus->num_chunks(); ++i) {
      _corpus_chunk_base[i] = base;
      base += static_cast<std::int64_t>(_corpus->chunk_rows(i));
    }
  }
  if (!_blocks.empty() && block_rows_on_host() && _host_rows_repo == nullptr &&
      _sirius_ctx != nullptr && get_pipeline() != nullptr) {
    auto manager = _sirius_ctx->get_data_repository_manager(get_pipeline()->get_query_id());
    if (manager) {
      auto repo       = std::make_unique<cucascade::shared_data_repository>();
      _host_rows_repo = repo.get();
      manager->add_new_repository(get_operator_id(), "host_rows", std::move(repo));
    }
  }
  // Lists on the device leave this join's tasks the rest of it, which the engine cannot enlarge by
  // moving anything. As many tasks may hold memory at once as fit there. The materialize stage runs
  // in the same task after the search has freed its scratch, so a task needs the search's estimate;
  // the quarter on top is for what that stage stages once for all tasks, outside any of them.
  _admission->pipeline = get_pipeline();
  if (_pair_schedule) {
    _admission->limit = _pair_tasks_at_once;
  } else if (_lists == nullptr && _build_side != nullptr && !_blocks.empty()) {
    _admission->limit = scanned_block_tasks();
  }
  if (_lists != nullptr && _lists->tier == cucascade::memory::Tier::GPU && _sirius_ctx != nullptr &&
      _num_left > 1) {
    auto const spaces =
      _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (!spaces.empty()) {
      std::size_t need = 0;
      for (std::size_t i = 0; i < _num_left; ++i) {
        need = std::max(need, per_left_batch_estimate(i));
      }
      // Probe blocks are the tasks when there are any.
      if (!_blocks.empty()) {
        need = 0;
        for (auto const& b : _blocks) {
          need = std::max(need, estimate_for_rows(static_cast<std::size_t>(b.rows)));
        }
      }
      need += need / 4;
      auto const room = 0.9 * static_cast<double>(spaces.front()->get_available_memory());
      _admission->limit =
        std::max<std::size_t>(1, static_cast<std::size_t>(room / static_cast<double>(need)));
      if (std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[vecjoin-phase] admission: %zu of %zu tasks at once (%.2f GiB free on the "
                     "device beside the lists, %.2f GiB a task)\n",
                     _admission->limit,
                     num_tasks(),
                     room / 0.9 / 1073741824.0,
                     static_cast<double>(need) / 1073741824.0);
      }
    }
  }
  _initialized = true;
}

void sirius_physical_vector_join_stream::plan_probe_blocks_locked()
{
  _blocks.clear();
  // Lists in host memory cross PCIe in every task, once per sweep, so the probe rows one task
  // searches set how often they are read. Deep100M, 9.6 GB of INT8 lists, 10M probes at 64 probes
  // each on a 24 GB device: one task per ~245k-row probe chunk 86.2 s, ~0.9M-row blocks (a quarter
  // of the device) 62.7 s, 1.25M-row blocks (a third) 59.6 s; lists on the device 53.6 s. Blocks
  // cut and join probe chunks to that size; only top-k answers are split back. A corpus the join
  // scans itself is read by every task too, from wherever the engine left it, the disk tier for
  // one larger than the host pool: SIFT100M from disk, 1M probes, ~5 passes as probe chunks.
  bool const reread = corpus_reread();
  // Clustered lists on the device are read in place, so a task's size costs no re-read there; it
  // sets how many probe rows each (task, cluster) slice carries. Both sweeps pay a fixed cost per
  // slice (a GEMM and a selection, or a seed and its stretches), so pin-chunk tasks spread a large
  // probe side thin: 1M x 1,024-d over 1,024 clusters, sweep 0 ranked on tensor cores took 0.61 s
  // over 9 pin-chunk tasks and 0.23 s over 2 tasks of 500k rows. A probe side of more
  // than one chunk is cut into blocks there too, sized to the room beside the lists (below).
  // SIRIUS_VSS_DEVICE_BLOCKS=0 keeps one task per probe chunk.
  bool const device_lists = _lists != nullptr && _lists->tier == cucascade::memory::Tier::GPU &&
                            _lists->offsets.size() > 2 && _num_left > 1 && [] {
                              auto const* v = std::getenv("SIRIUS_VSS_DEVICE_BLOCKS");
                              return v == nullptr || std::string_view{v} != "0";
                            }();
  // A threshold join holds no [n x k] answer per block, so blocks go where pair tasks do: its pairs
  // then leave chunk by chunk, split by probe chunk, instead of being kept to the block's end.
  bool const pair_tasks = pair_tasks_wanted();
  bool const wanted     = (reread || device_lists) &&
                          (_request.mode != vss::vector_join_mode::threshold || pair_tasks) &&
                          _request.earlier_by.empty() && !_request.probe_scalar && _num_left > 0;
  if (!wanted) { return; }
  auto const k_join = std::max<std::int64_t>(
    1, _right_total_rows > 0 ? std::min<std::int64_t>(_request.k, _right_total_rows) : _request.k);
  std::int64_t block_rows = 0;
  if (pair_tasks) {
    auto const* v = std::getenv("SIRIUS_VSS_PAIR_TASKS");
    _pair_tasks_at_once =
      v != nullptr ? static_cast<std::size_t>(std::max(1, std::atoi(v))) : kPairTasksAtOnce;
  }
  if (auto const* env = std::getenv("SIRIUS_VSS_PROBE_BLOCK_ROWS")) {
    block_rows = std::max<std::int64_t>(0, std::atoll(env));
  } else if (pair_tasks) {
    block_rows = pair_block_rows();
  } else if (_sirius_ctx != nullptr) {
    auto const spaces =
      _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (spaces.empty()) { return; }
    // The estimate is affine in the row count past a million rows; the joined probe matrix is on
    // top of it.
    constexpr std::size_t kA = std::size_t{1} << 20, kB = std::size_t{1} << 21;
    auto const estimate_a = estimate_for_rows(kA, true, true);
    auto const slope      = static_cast<double>(estimate_for_rows(kB, true, true) - estimate_a) / kA;
    auto const per_row =
      slope + (block_rows_on_host() ? 0.0 : static_cast<double>(_request.dim) * sizeof(float));
    auto const fixed = static_cast<double>(estimate_a) - static_cast<double>(kA) * slope;
    // A third of the device, so a few tasks run at once; never less than a whole probe chunk,
    // unless one does not fit in most of the device, which is what makes a small device work.
    auto const device        = static_cast<double>(spaces.front()->get_max_memory());
    std::size_t widest_chunk = 0;
    for (std::size_t i = 0; i < _num_left; ++i) {
      widest_chunk = std::max(widest_chunk, _probe->chunk_rows(i));
    }
    auto const chunk_need = fixed + per_row * static_cast<double>(widest_chunk);
    auto budget = chunk_need <= 0.8 * device ? std::max(device / 3, chunk_need) : 0.8 * device;
    if (_lists == nullptr && _build_side != nullptr) {
      // A scanned corpus is read whole for every block, from disk once it outgrows host memory, and
      // blocks in flight each read it themselves. The rows one read serves set the cost, so one
      // block takes most of the device (113.5M x 1024-d on one NVMe: GPU busy 13 % with ~3 blocks).
      budget = std::max(
        budget,
        std::max(kScannedBlockDeviceShare * device / static_cast<double>(scanned_block_tasks()),
                 chunk_need));
    }
    if (device_lists && !reread) {
      // Beside device lists the room is what is free now (the lists, the pins and the other
      // operators already hold the rest), shared by SIRIUS_VSS_DEVICE_BLOCK_TASKS tasks at once.
      // Default 2 when the FP32 re-score reads device rows: one task's host-side steps (routing
      // edges, slice setup, read-backs) overlap the other's kernels. Default 1 when it reads host
      // memory (a host pin, or FP32 rows kept in host blocks): each row it fetches over PCIe then
      // serves the most probe rows, as host lists' blocks do (10M x 768-d, host pin, 1 thread, p4:
      // 38.9 s at 1 against 46.7 at 2). Only blocks larger than a chunk are worth cutting.
      bool const exact_on_host =
        _lists->exact_vectors || (_lists->exact_pin && _lists->pin != nullptr &&
                                  _lists->pin->tier == cucascade::memory::Tier::HOST);
      auto const* v    = std::getenv("SIRIUS_VSS_DEVICE_BLOCK_TASKS");
      auto const tasks = v != nullptr ? std::max(1.0, std::atof(v)) : (exact_on_host ? 1.0 : 2.0);
      // As the admission cap below counts it: 90 % of the free room, a quarter on top of each
      // task's estimate for the materialize stage, so that `tasks` blocks are admitted together.
      auto const room = 0.9 * static_cast<double>(spaces.front()->get_available_memory());
      budget          = room / (1.25 * tasks);
      if (budget <= chunk_need) { return; }
    }
    block_rows =
      budget > fixed ? static_cast<std::int64_t>((budget - fixed) / per_row) : std::int64_t{0};
    block_rows = std::max<std::int64_t>(block_rows, std::int64_t{1} << 14);
    if (device_lists && !reread && block_rows <= static_cast<std::int64_t>(widest_chunk)) {
      return;
    }
  }
  // A threshold block's limit is its routing edges, one column; a top-k block's its answer.
  block_rows =
    std::min<std::int64_t>(block_rows,
                           static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()) /
                             (pair_tasks ? std::max<std::int64_t>(_request.n_probes, 1) : k_join));
  if (block_rows_on_host()) {
    // A block's routing edges are one column.
    block_rows = std::min<std::int64_t>(
      block_rows,
      static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()) /
        std::max<std::int64_t>(_request.n_probes, 1));
    // The rows live in the host pool. With a disk tier all else there can move out of their way, so
    // they plan on a share of its size; without one nothing there can leave, so on a share of what
    // is free now, the rest of which still has to take what the device spills. Never less than the
    // floor of the budget above: a block that then finds no room keeps its rows on the device.
    if (_sirius_ctx != nullptr) {
      auto& memory      = _sirius_ctx->get_memory_manager();
      auto const hosts  = memory.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
      bool const spills = !memory.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).empty();
      if (!hosts.empty()) {
        auto const room =
          spills ? hosts.front()->get_max_memory()
                 : std::min(hosts.front()->get_max_memory(), hosts.front()->get_available_memory());
        auto const per_task = kBlockRowsHostShare * static_cast<double>(room) /
                              static_cast<double>(scanned_block_tasks());
        block_rows =
          std::min(block_rows,
                   std::max(static_cast<std::int64_t>(
                              per_task / (static_cast<double>(_request.dim) * sizeof(float))),
                            std::int64_t{1} << 14));
      }
    }
  }
  if (block_rows <= 0) { return; }
  // Blocks of even size, so the last is not a sliver that pays for a whole read of the lists alone.
  std::int64_t total = 0;
  for (std::size_t i = 0; i < _num_left; ++i) {
    total += static_cast<std::int64_t>(_probe->chunk_rows(i));
  }
  if (total == 0) { return; }
  auto const n_blocks = (total + block_rows - 1) / block_rows;
  block_rows          = (total + n_blocks - 1) / n_blocks;
  probe_block current;
  for (std::size_t i = 0; i < _num_left; ++i) {
    auto const rows = static_cast<std::int64_t>(_probe->chunk_rows(i));
    for (std::int64_t at = 0; at < rows;) {
      auto const take = std::min(rows - at, block_rows - current.rows);
      current.pieces.push_back(probe_piece{i, at, at + take});
      current.rows += take;
      at += take;
      if (current.rows == block_rows) {
        _blocks.push_back(std::move(current));
        current = probe_block{};
      }
    }
  }
  if (current.rows > 0) { _blocks.push_back(std::move(current)); }
  if (std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
    std::fprintf(stderr,
                 "[vecjoin-phase] probe blocks: %zu of up to %lld rows over %zu probe chunks%s\n",
                 _blocks.size(),
                 static_cast<long long>(block_rows),
                 _num_left,
                 pair_tasks ? ", in pair tasks" : "");
  }
}

bool sirius_physical_vector_join_stream::corpus_reread() const
{
  return (_lists != nullptr && _lists->tier == cucascade::memory::Tier::HOST) ||
         (_lists == nullptr && _build_side != nullptr);
}

bool sirius_physical_vector_join_stream::pair_tasks_wanted() const
{
  return _request.pair_tasks && _request.mode == vss::vector_join_mode::threshold &&
         _request.earlier_by.empty() && !_request.best_only && !_request.probe_scalar &&
         _right_total_rows > 0 && corpus_reread();
}

double sirius_physical_vector_join_stream::pair_block_row_bytes(bool building) const
{
  auto const row    = static_cast<double>(_request.dim) * sizeof(float);
  bool const cosine = _request.metric == "cosine";
  if (_centroids == nullptr) {
    // A cosine block goes on holding only its unit rows.
    return row * (cosine && building ? 2.0 : 1.0);
  }
  auto const probes = static_cast<double>(std::max<std::int64_t>(_request.n_probes, 1));
  // A row's routing edges, sorted (~45 B an edge), then their rows alone; cosine's unit rows;
  // coded lists' probe copy, norms, rounding error and limits.
  auto const edges = probes * (building ? 45.0 : static_cast<double>(sizeof(std::int64_t)));
  auto const coded = _lists != nullptr ? 0.5 * row + 4.0 * sizeof(float) : 0.0;
  return row * (cosine ? 2.0 : 1.0) + edges + coded;
}

std::size_t sirius_physical_vector_join_stream::pair_task_bytes(std::int64_t gathered_rows) const
{
  auto const n        = _max_corpus_chunk_rows;
  auto const row      = static_cast<std::size_t>(_request.dim) * sizeof(float);
  auto const prepared = static_cast<std::size_t>(_request.dim + 4) * sizeof(float);
  auto const gathered = static_cast<std::size_t>(std::max<std::int64_t>(gathered_rows, 0));
  // Unexpanded distances go through cuVS's tiles, a block of distances and of kept indices; the
  // rest through the GEMM's one score tile, with both sides prepared.
  auto const cells = std::min(std::size_t{512} * n, std::size_t{1} << 27);
  auto const cuvs  = cells * (sizeof(float) + sizeof(std::int64_t));
  auto const gemm  = vss::gemm_search_tile_bytes() + n * prepared + gathered * prepared;
  // The chunk, a chunk's unit rows under cosine, its gathered queries, the first edges, and on
  // coded lists the buffer the filter's pairs wait in.
  std::size_t const edges   = (std::size_t{1} << 20) * (2 * sizeof(std::int64_t) + sizeof(float));
  std::size_t const waiting = _lists != nullptr ? (std::size_t{1} << 22) * 16 : 0;
  return _max_chunk_bytes + (_request.metric == "cosine" ? n * row : 0) + gathered * row +
         std::max(cuvs, gemm) + edges + waiting;
}

std::int64_t sirius_physical_vector_join_stream::pair_block_rows()
{
  if (_sirius_ctx == nullptr) { return 0; }
  auto const spaces =
    _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (spaces.empty()) { return 0; }
  auto const device        = static_cast<double>(spaces.front()->get_max_memory());
  std::size_t widest_chunk = 0;
  for (std::size_t i = 0; i < _num_left; ++i) {
    widest_chunk = std::max(widest_chunk, _probe->chunk_rows(i));
  }
  auto const widest = static_cast<double>(widest_chunk);
  // A block over a re-read corpus takes most of the device, for the reason a top-k block does. Out
  // of its share come its pair tasks' working memory and room for the pairs they hold: a task goes
  // on past that only until its next search would not fit. What materialize concatenates once,
  // the corpus's output columns, is about 8 B a row a column.
  auto const share = kScannedBlockDeviceShare * device / static_cast<double>(scanned_block_tasks());
  auto const scratch = static_cast<double>(pair_task_bytes(
    std::min(static_cast<std::int64_t>(widest_chunk), pair_call_rows(_request.dim))));
  if (2.0 * scratch > share && std::getenv("SIRIUS_VSS_PAIR_TASKS") == nullptr) {
    _pair_tasks_at_once = 1;
  }
  auto const fixed    = static_cast<double>(_pair_tasks_at_once) * scratch + device / 8 +
                        static_cast<double>(_right_total_rows) * sizeof(std::int64_t) *
                          static_cast<double>(_request.right.output_columns.size());
  auto const holding  = pair_block_row_bytes(false);
  auto const building = pair_block_row_bytes(true);
  auto rows           = share > fixed ? std::min((share - fixed) / holding, share / building) : 0.0;
  // Never less than a whole probe chunk, unless one does not fit in the share at all.
  if (rows < widest) {
    rows = widest * building + scratch <= share
             ? widest
             : std::max(rows, static_cast<double>(std::int64_t{1} << 14));
  }
  return static_cast<std::int64_t>(rows);
}

void* host_pair_block_pool::take(std::size_t budget_bytes)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_free.empty()) {
      auto* block = _free.back();
      _free.pop_back();
      return block;
    }
    if ((_pinned + 1) * kHostBlockBytes > budget_bytes) { return nullptr; }
    ++_pinned;
  }
  void* block       = nullptr;
  auto const status = cudaMallocHost(&block, kHostBlockBytes);
  if (status != cudaSuccess) {
    std::lock_guard<std::mutex> lock(_mutex);
    --_pinned;
  }
  CUDF_CUDA_TRY(status);
  return block;
}

void host_pair_block_pool::give_back(void* block)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _free.push_back(block);
}

void host_pair_block_pool::release()
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto* block : _free) {
    cudaFreeHost(block);
  }
  _pinned -= _free.size();
  _free.clear();
}

void sirius_physical_vector_join_stream::on_finalize_operator()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  _host_pair_blocks.release();
  auto drain = [](cucascade::shared_data_repository* repo) {
    if (repo == nullptr) { return; }
    for (std::size_t p = 0; p < repo->num_partitions(); ++p) {
      for (auto const id : repo->get_batch_ids(p)) {
        std::ignore = repo->pop_data_batch_by_id(id, p);
      }
    }
  };
  // The lists outlive the query in the session's index cache; only this query's references go.
  drain(_lists_repo);
  _lists_repo = nullptr;
  if (_host_rows_repo != nullptr && _sirius_ctx != nullptr) {
    _sirius_ctx->record_vector_join_host_left(_host_rows_repo->total_size());
  }
  drain(_host_rows_repo);
  if (_host_rows_repo != nullptr && _sirius_ctx != nullptr &&
      std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
    auto const stats = _sirius_ctx->get_vector_join_host_rows_stats();
    std::fprintf(stderr,
                 "[vecjoin-phase] block rows this session: %llu batches, %llu restored from disk, "
                 "%llu blocks on the device\n",
                 static_cast<unsigned long long>(stats.batches_made),
                 static_cast<unsigned long long>(stats.batches_restored),
                 static_cast<unsigned long long>(stats.blocks_on_device));
  }
  _host_rows_repo = nullptr;
  _pair_states.clear();
  if (_sirius_ctx != nullptr && std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
    auto const stats = _sirius_ctx->get_vector_join_pair_stats();
    std::fprintf(
      stderr,
      "[vecjoin-phase] corpus reads this session: %llu chunks staged, %llu of them "
      "moved to the device, %.3f GiB; pair tasks: %llu blocks built, %llu attempts, %llu "
      "finished, %llu cut short\n",
      static_cast<unsigned long long>(stats.corpus_stages),
      static_cast<unsigned long long>(stats.corpus_moved),
      static_cast<double>(stats.corpus_bytes_moved) / 1073741824.0,
      static_cast<unsigned long long>(stats.blocks_built),
      static_cast<unsigned long long>(stats.pair_attempts),
      static_cast<unsigned long long>(stats.pairs_finished),
      static_cast<unsigned long long>(stats.pairs_cut_short));
  }
  _corpus_key.reset();
  _layout_key.reset();
  _layout_key_host    = {};
  _probe_row_base     = {};
  _earlier_keys_built = false;
  // The probe and corpus batches were only ever borrowed by id (every probe task reads the same
  // corpus); the materialize stage holds its own references to the probe batches it still needs.
  for (auto const* name : {"default", "build"}) {
    if (auto* port = try_get_port(name); port != nullptr) { drain(port->repo); }
  }
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
  if (_init_error) {
    if (_init_error_task_issued) { return std::nullopt; }
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  if (_pair_schedule) {
    // A task's release re-arms this operator, so nothing waits here that a running task will not
    // wake: a build about to publish, a block's last pair, or a free place under the cap.
    std::lock_guard<std::mutex> admission_lock(_admission->mutex);
    if (_admission->in_flight >= _admission->limit || !_pair_schedule->peek()) {
      return std::nullopt;
    }
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  if (num_tasks() == 0 || _next_left >= num_tasks()) { return std::nullopt; }
  if (_admission->capped()) {
    // At the cap, a task's release re-arms this operator.
    std::lock_guard<std::mutex> admission_lock(_admission->mutex);
    if (_admission->in_flight >= _admission->limit) { return std::nullopt; }
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  if (_hint_returned) { return std::nullopt; }
  _hint_returned = true;
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_vector_join_stream::all_ports_empty()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  // Work is still to come, it just cannot be scheduled yet.
  if (!build_side_ready_locked()) { return false; }
  ensure_initialized_locked();
  // Not empty until the task carrying the error has failed the query.
  if (_init_error) { return false; }
  if (_pair_schedule) { return _pair_schedule->exhausted(); }
  return _next_left >= num_tasks();
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(_op_mutex);
  if (!build_side_ready_locked()) { return nullptr; }
  ensure_initialized_locked();
  if (_init_error) {
    if (_init_error_task_issued) { return nullptr; }
    _init_error_task_issued = true;
    return std::make_unique<vector_join_stream_input>(0, 0);
  }
  if (_pair_schedule) { return next_pair_step_locked(); }
  if (_next_left >= num_tasks()) { return nullptr; }

  // The probe batch is deliberately NOT carried in the task input as a pipelineable batch: a
  // queued task subscribes to its batches, which removes them from the repository spill tier, and
  // under device pressure (10M self-join through every list) every task then waits for memory
  // that only those batches could free. Left idle in the port, the engine evicts them and the
  // chunk source stages each one back when its task runs.
  std::shared_ptr<void> ticket;
  if (_admission->capped()) {
    {
      std::lock_guard<std::mutex> admission_lock(_admission->mutex);
      if (_admission->in_flight >= _admission->limit) { return nullptr; }
      ++_admission->in_flight;
    }
    ticket = std::make_shared<admission_ticket>(_admission);
  }
  auto const task = _next_left++;
  std::unique_ptr<vector_join_stream_input> input;
  if (_blocks.empty()) {
    input = std::make_unique<vector_join_stream_input>(task, per_left_batch_estimate(task));
  } else {
    auto const& block = _blocks[task];
    input             = std::make_unique<vector_join_stream_input>(
      block.pieces.front().chunk, estimate_for_rows(static_cast<std::size_t>(block.rows)), task);
  }
  input->set_ticket(std::move(ticket));
  return input;
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
  std::shared_ptr<const scan_manager::pinned_entry> pin;
  if (_build_side) {
    // The plan generator projects the corpus scan as [vector, emitted columns..., cluster id],
    // so the cluster column's position is arithmetic rather than a name lookup -- the same
    // arrangement that puts the vector column at 0. A `dim` of 1 gives the source a four-byte
    // row width, which is what one INT32 label per row occupies.
    // With the context, like the corpus source: a label batch the engine took to disk comes
    // back through the host space.
    build_labels =
      make_materialized_chunk_source(*_build_side,
                                     /*column_index=*/1 + _request.right.output_columns.size(),
                                     /*dim=*/1,
                                     batch_telemetry(),
                                     _sirius_ctx);
  } else {
    pin = _scan_manager->find_pinned_entry_for_duckdb_table(_request.right.catalog,
                                                            _request.right.schema,
                                                            _request.right.table,
                                                            _request.right.table_identity);
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
      staged_build = stage_for_fold(*build_labels, j, space, stream, stream);
      labels_view  = staged_build.view;
    } else {
      staged_pin = vss::stage_pinned_column_chunk(
        *pin, _request.build_cluster_column, j, space, stream, batch_telemetry());
      labels_view = staged_pin.view;
    }
    // A pin may store the ids narrowed (compressed materialization keeps the narrowest integer
    // type that holds them), so any integer width is widened back rather than refused.
    std::unique_ptr<cudf::column> widened_labels;
    if (labels_view.type().id() != cudf::type_id::INT32) {
      if (!cudf::is_integral(labels_view.type())) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column '" +
                                 _request.build_cluster_column +
                                 "' must be an integer; sirius_kmeans_assign emits cluster_id "
                                 "as INTEGER");
      }
      widened_labels = cudf::cast(labels_view,
                                  cudf::data_type{cudf::type_id::INT32},
                                  stream,
                                  cudf::get_current_device_resource_ref());
      labels_view    = widened_labels->view();
    }
    auto const rows = static_cast<std::size_t>(labels_view.size());
    if (rows == 0) { continue; }

    labels.resize(rows);
    CUDF_CUDA_TRY(cudaMemcpyAsync(labels.data(),
                                  labels_view.data<std::int32_t>(),
                                  rows * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    stream.sync();
    row_base += static_cast<std::int64_t>(rows);

    // A chunk is read as runs of one cluster each, and a cluster may come as several runs --
    // within a chunk as across chunks, since a slice carries its own chunk and nothing reads
    // across a run. A build phase's batches are ordered spans of an ORDER BY, but the stage that
    // coalesces them into chunks joins spans in the order they finish, so a chunk can be two
    // sorted runs back to back. Only a corpus never put in cluster order is refused: its runs
    // are a few rows long, and the search would visit it a few rows at a time.
    auto const runs_before = _chunk_cluster_runs[j].size();
    std::size_t start      = 0;
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
      runs.push_back(
        chunk_cluster_run{c, static_cast<std::int64_t>(start), static_cast<std::int64_t>(i)});
      _cluster_rows[static_cast<std::size_t>(c)] += static_cast<std::int64_t>(i - start);
      start = i;
    }
    auto const chunk_runs = _chunk_cluster_runs[j].size() - runs_before;
    if (chunk_runs > 1024 && rows < 8 * chunk_runs) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] corpus chunk " + std::to_string(j) +
        " is not stored in cluster order: " + std::to_string(chunk_runs) + " runs over " +
        std::to_string(rows) + " rows. Materialize the corpus with ORDER BY " +
        _request.build_cluster_column);
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
                                    stream.get()));
      stream.sync();
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

void sirius_physical_vector_join_stream::ensure_earlier_keys(
  ::cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::lock_guard<std::mutex> const lock(_op_mutex);
  if (_earlier_keys_built) { return; }
  if (_right_pin == nullptr || _left_pin == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_stream] earlier_by needs both tables pinned");
  }
  auto const bytes = static_cast<std::size_t>(_right_total_rows) * sizeof(std::int64_t);
  // Every task of the query reads them, so they are allocated the way the lists are: on the
  // default stream, outside any one task's.
  auto const persistent    = cudf::get_default_stream();
  auto const persistent_mr = space.get_default_allocator();
  // Row positions are their own keys: nothing is read, and over lists the row map is the keys.
  // A probe task only has to know where its chunk starts, from the chunks' own row counts (the
  // chunk source's are estimates from byte sizes, good for a reservation and not for an order).
  std::unique_ptr<rmm::device_buffer> keys;
  if (_request.key_is_rowid) {
    std::vector<std::int64_t> base{0};
    if (_left_pin->tier == cucascade::memory::Tier::HOST) {
      for (auto const& chunk : _left_pin->host_chunks) {
        auto const& columns = uncompressed_host_chunk(*chunk).get_host_table()->columns;
        base.push_back(base.back() + (columns.empty() ? 0 : columns.front().num_rows));
      }
    } else {
      for (auto const& view :
           vss::pinned_column_chunk_views(*_left_pin, _request.left.column, space)) {
        base.push_back(base.back() + view.size());
      }
    }
    _probe_row_base = std::move(base);
  } else {
    auto const& name    = _request.earlier_by;
    auto const n_chunks = vss::pinned_column_chunk_count(*_right_pin, name);
    keys                = std::make_unique<rmm::device_buffer>(bytes, persistent, persistent_mr);
    std::int64_t row    = 0;
    for (std::size_t j = 0; j < n_chunks; ++j) {
      auto const staged =
        vss::stage_pinned_column_chunk(*_right_pin, name, j, space, stream, batch_telemetry());
      auto const rows = static_cast<std::int64_t>(staged.view.size());
      if (row + rows > _right_total_rows) { break; }
      copy_earlier_keys(
        staged.view, static_cast<std::int64_t*>(keys->data()) + row, name, stream, mr);
      row += rows;
    }
    if (row != _right_total_rows) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] earlier_by column '" + name +
                               "' does not hold the corpus's " + std::to_string(_right_total_rows) +
                               " rows");
    }
  }
  if (_lists != nullptr) {
    std::unique_ptr<rmm::device_buffer> layout;
    void const* layout_keys = _lists->row_ids->data();
    if (keys) {
      layout = std::make_unique<rmm::device_buffer>(bytes, persistent, persistent_mr);
      vss::gather_bytes(keys->data(),
                        sizeof(std::int64_t),
                        static_cast<std::int64_t const*>(_lists->row_ids->data()),
                        _right_total_rows,
                        layout->data(),
                        stream);
      layout_keys = layout->data();
    }
    // Lists keep the pin's row order, so a corpus stored in key order has ascending keys within
    // each list, and row positions always do. Checked rather than assumed: the cut it allows
    // would drop pairs otherwise.
    std::vector<std::int64_t> host(static_cast<std::size_t>(_right_total_rows));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(host.data(), layout_keys, bytes, cudaMemcpyDefault, stream.get()));
    stream.sync();
    bool ascending = true;
    for (std::size_t c = 0; ascending && c + 1 < _lists->offsets.size(); ++c) {
      auto const first = host.begin() + _lists->offsets[c];
      auto const last  = host.begin() + _lists->offsets[c + 1];
      ascending        = std::is_sorted(first, last);
    }
    if (ascending) { _layout_key_host = std::move(host); }
    _layout_key = std::move(layout);
  }
  _corpus_key         = std::move(keys);
  _earlier_keys_built = true;
}

//===----------------------------------------------------------------------===//
// A threshold join's pairs
//===----------------------------------------------------------------------===//
/// What one task has found of a threshold join's answer: pairs on the device, and pieces moved to
/// host memory. Nothing here outlives the task that found them unless the task's output carries it.
struct sirius_physical_vector_join_stream::found_pairs {
  std::vector<std::unique_ptr<cudf::column>> left, neighbors, distances;
  std::vector<std::shared_ptr<::cucascade::data_batch>> offloaded;
  /// Under pair tasks, each offloaded piece's probe chunk.
  std::vector<std::size_t> offloaded_chunks;
  std::int64_t held{0};
  std::int64_t total{0};
};

//===----------------------------------------------------------------------===//
// A probe block's state
//===----------------------------------------------------------------------===//
/// What a task keeps of its probe rows from their load to the answer: the rows themselves, what
/// the answer accumulates, and, while the corpus is searched, @ref search_state. Holds data only:
/// nothing here names a stream, so what operates on it says which stream it runs on.
struct sirius_physical_vector_join_stream::block_state {
  /// What a walk over corpus chunks changes as it goes: the pairs it holds and what it did.
  struct search_run {
    explicit search_run(host_pair_block_pool& pool) : host_pairs(pool) {}

    // The candidate buffer, and the pairs waiting beyond it.
    std::optional<vss::bound_candidates> candidates;
    std::int64_t pending{0};
    std::optional<rmm::device_uvector<float>> gap_limit;
    /// Moved pairs back in the buffer: already counted as passed.
    std::int64_t pairs_back{0};
    std::vector<std::int32_t> host_row_slot;
    host_pair_queue host_pairs;
    std::optional<rmm::device_uvector<float>> moved_codes;

    // The running bound's schedule.
    std::int64_t running_piece{0};
    std::int64_t since_merge{0};

    // What the search did.
    std::int64_t scanned_pairs{0};
    std::int64_t bounded_emitted{0};
    std::int64_t bounded_merges{0};
    std::int64_t bounded_launches{0};
    std::int64_t bounded_padded{0};
    /// Debug only (they synchronize): time spent re-scoring candidates and taking/merging them.
    double rescore_seconds{0};
    double take_seconds{0};
    double trim_seconds{0};
    vss::gap_rescore_stats gap_stats{};
    std::int64_t gap_rescored{0};
    std::int64_t pair_moves{0};
    std::int64_t slice_cuts{0};
    std::int64_t pairs_moved{0};
    std::int64_t budget_flushes{0};
    double host_seconds{0};
    /// Debug only: pairs each stretch passed.
    std::vector<std::int64_t> stretch_pairs;
  };

  /// What a clustered search reads and writes from one corpus chunk to the next, gone before the
  /// answer is shaped. The search's form is decided once for its rows and held here with them.
  struct search_state {
    explicit search_state(host_pair_block_pool& pool) : run(pool) {}

    // Routing: each probe row's clusters, and the corpus chunks they need.
    std::int64_t n_probes{0};
    std::vector<std::int32_t> nearest_cluster;
    std::optional<vss::sorted_edges> routed;
    std::int64_t const* routed_rows{nullptr};
    std::int64_t max_routed{0};
    std::int64_t max_nearest{0};
    std::vector<std::size_t> needed_chunks;
    std::vector<std::size_t> nearest_chunks;

    // The rows the search reads, as the metric wants them.
    bool unit_cosine{false};
    bool direct_cosine{false};
    std::optional<rmm::device_uvector<float>> unit_probe;
    vss::dataset_matrix_view search_queries;
    cuvs::distance::DistanceType search_metric{};
    cuvs::distance::DistanceType ranked_metric{};
    float search_radius{0.f};
    float radius_bound{0.f};

    // The probe coded to INT8 for byte and INT8 lists. The rotated probe lives as long as its
    // codes: freed while the quantize reading it is still queued, its memory went to a concurrent
    // task (one task per probe chunk) and corrupted them.
    bool codes_int8{false};
    bool int8_search{false};
    std::optional<rmm::device_uvector<std::int8_t>> probe_i8;
    std::optional<rmm::device_uvector<std::int32_t>> probe_sq;
    std::optional<rmm::device_uvector<float>> probe_code_error, rotated_probe;

    // The probe coded to FP16, its norms and its rounding error.
    std::optional<rmm::device_uvector<std::uint16_t>> probe_f16;
    std::optional<rmm::device_uvector<float>> probe_sqf, probe_half_error;
    std::optional<rmm::device_uvector<float>> ip_probe_zero_sq, ip_row_zero_sq;
    void const* list_norms{nullptr};

    // The form of the search.
    bool f16_list_no_seed{false};
    std::int64_t running_bound_rows{0};
    bool bounded{false};
    bool scan_f16{false};
    bool f16_bounded{false};
    bool any_bounded{false};
    bool rescore{false};
    bool skip_certain{false};
    bool sqrt_at_end{false};
    bool ranked_sweep0{false};
    bool seed{false};
    bool f16_sweep0{false};
    bool sweep0_bound_from_codes{false};
    bool gap_rescore{false};
    bool gap_stats_on{false};
    bool group_launch{false};
    std::size_t bounded_group{0};
    std::int64_t group_m_max{0};
    std::int64_t group_work_max{0};
    std::int64_t group_slice_rows{0};
    bool hold_codes_pairs{false};
    bool pairs_to_host{false};
    bool budget_flush_refused{false};
    bool pairs_to_host_forced{false};
    bool cut_large_slices{false};
    bool slice_cuts_forced{false};
    std::size_t pair_budget{0};
    bool lists_on_host{false};
    bool keep_chunks{false};
    bool stretch_sweep0{false};
    /// FLOAT16 lists the bounded search does not answer: their chunks staged from the FP32 rows.
    std::optional<cluster_lists_chunk_source> exact_lists_corpus;

    // Each probe row's bound, and the limits derived from it for the filter and the trim. Both
    // filter under a per-row limit derived from the FP32 bound and both sides' rounding.
    std::optional<rmm::device_uvector<float>> code_limit;
    std::optional<rmm::device_uvector<float>> certain_limit;
    std::optional<rmm::device_uvector<float>> bound;

    search_run run;
  };

  std::int64_t n_left{0};

  // The probe rows.
  staged_vector_chunk staged_probe;
  std::optional<rmm::device_uvector<float>> block_probe;
  /// An exhaustive cosine block's rows as its pair tasks search them.
  std::optional<rmm::device_uvector<float>> unit_rows;
  bool probe_rows_on_host{false};
  std::optional<vss::block_host_rows> host_rows;
  vss::dataset_matrix_view queries;

  // Running [n_left x k_join] accumulator. Seeded by the first right batch, then
  // every later batch is folded in and released, so peak device memory does not
  // grow with the number of right batches.
  std::unique_ptr<cudf::column> acc_neighbors;
  std::unique_ptr<cudf::column> acc_distances;

  float radius_scale{1.0f};  ///< unit-row cosine search reports twice the cosine distance

  // A keyed or best-match threshold join's keys, and each probe row's best match so far.
  std::optional<rmm::device_uvector<std::uint32_t>> best_order, best_scratch;
  std::optional<rmm::device_uvector<std::int64_t>> best_ids, probe_keys;
  std::unique_ptr<cudf::column> probe_positions;
  vss::earlier_keys pin_keys{}, layout_keys{};
  /// Whether the fold's ids are layout rows of the lists, and its distances squared, at the end.
  bool best_ids_are_layout{false}, best_take_sqrt{false};
  std::int64_t best_rescored{0};
  /// Set when the lists' keys ascend within each list: no row at or past probe_key_max can be
  /// earlier than any probe row of this task, so each list is searched only up to there.
  std::int64_t const* earlier_cut{nullptr};
  std::int64_t probe_key_max{0};

  /// Rows the exhaustive fold searched; a corpus with none has nothing to join against.
  std::int64_t exhaustive_rows_seen{0};

  std::optional<search_state> search;
};

//===----------------------------------------------------------------------===//
// One task's work on a block
//===----------------------------------------------------------------------===//
/// What a task does with @ref block_state. Everything here runs on this task's stream and
/// allocator, and whatever must outlive it is in the state.
class sirius_physical_vector_join_stream::block_task {
 public:
  block_task(sirius_physical_vector_join_stream& op,
             block_state& state,
             found_pairs& pairs,
             vector_join_stream_input const& input,
             ::cucascade::memory::memory_space& space,
             rmm::cuda_stream_view task_stream)
    : _op(op),
      _request(op._request),
      _lists(op._lists),
      st(state),
      found(pairs),
      join_in(&input),
      mem_space(&space),
      stream(task_stream),
      mr(space.get_default_allocator()),
      res{task_stream},
      left_idx(input.left_idx()),
      block(input.block() ? &op._blocks[*input.block()] : nullptr),
      dim(op._request.dim),
      k_join(std::min<std::int64_t>(op._request.k, op._right_total_rows)),
      metric(vss::join_selection_distance_type_from_metric(
        op._request.metric, op._request.search_mode == vss::vector_join_search_mode::exact_direct)),
      ip(metric == cuvs::distance::DistanceType::InnerProduct),
      radius_join(op._request.mode == vss::vector_join_mode::threshold),
      count_only(radius_join && op._request.count_only),
      keyed(radius_join && !op._request.earlier_by.empty()),
      best_only(radius_join && op._request.best_only && !count_only),
      radius_eps(radius_for_kernel()),
      next_search_room(space.get_max_memory() / 8),
      pair_step(input.pair_step() ? &*input.pair_step() : nullptr)
  {
    // A pair task finds its block's rows loaded.
    n_left = state.n_left;
  }

  // Held for the whole task: every corpus chunk is searched against this probe chunk. A copy
  // staged back from the host is freed on the compute stream (the chunk sources rebind it), so
  // dropping it as the task returns, before the task's stream sync, is ordered behind the searches.
  // A block's pieces are copied into one probe matrix, each chunk dropped once copied. A flat
  // buffer because rows x dim can pass what a list column's child holds (a third of a 186 GB device
  // plans ~4M rows, past 2^31 values at 1,024-d).
  void load_rows()
  {
    // Every pass of a block over a scanned corpus reads the corpus again, so the rows a block holds
    // set the cost, and a block's FP32 rows are its largest part on the device (4 KB of ~10 KB a
    // row at 1,024-d). The filter reads their FP16 copy; only routing, rounding and the re-score
    // read FP32, a piece or a few rows at a time. So they stay on the host, as engine batches in
    // the pool that are idle between those reads and go with this object however the task ends.
    auto const load_start = std::chrono::steady_clock::now();
    if (block != nullptr && _op.block_rows_on_host() && _op._host_rows_repo != nullptr) {
      auto const hosts = _op._sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(
        cucascade::memory::Tier::HOST);
      st.host_rows.emplace(*_op._host_rows_repo,
                           const_cast<cucascade::memory::memory_space&>(*hosts.front()),
                           *_op._sirius_ctx,
                           block->rows,
                           dim,
                           stream);
      st.probe_rows_on_host = st.host_rows->make_room_for_block();
      for (auto piece = block->pieces.begin();
           st.probe_rows_on_host && piece != block->pieces.end();
           ++piece) {
        auto const chunk = _op._probe->stage_for_task(piece->chunk, *mem_space, stream);
        auto const rows  = vss::list_column_as_dataset_view(chunk.view, dim);
        st.probe_rows_on_host =
          st.host_rows->append(rows.data_handle() + piece->begin * dim, piece->end - piece->begin);
      }
      // No room in the pool, even after spilling what the engine may: the search is the same with
      // the rows on the device, where a real shortage is the engine's to retry.
      if (!st.probe_rows_on_host) {
        st.host_rows.reset();
        _op._sirius_ctx->record_vector_join_block_on_device();
        SIRIUS_LOG_WARN(
          "[sirius_physical_vector_join_stream] no room in the host pool for a block of {} probe "
          "rows; keeping them on the device",
          block->rows);
      }
      if (std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr) {
        std::fprintf(
          stderr,
          "[vecjoin-phase] block rows to host    %8.3f s\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count());
      }
    }
    if (block == nullptr) {
      st.staged_probe = _op._probe->stage_for_task(left_idx, *mem_space, stream);
    } else if (!st.probe_rows_on_host) {
      st.block_probe.emplace(static_cast<std::size_t>(block->rows * dim), stream, mr);
      std::size_t at = 0;
      for (auto const& piece : block->pieces) {
        auto const chunk  = _op._probe->stage_for_task(piece.chunk, *mem_space, stream);
        auto const rows   = vss::list_column_as_dataset_view(chunk.view, dim);
        auto const values = static_cast<std::size_t>((piece.end - piece.begin) * dim);
        CUDF_CUDA_TRY(cudaMemcpyAsync(st.block_probe->data() + at,
                                      rows.data_handle() + piece.begin * dim,
                                      values * sizeof(float),
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        stream.sync();
        at += values;
      }
    }
    st.queries = block != nullptr
                   ? raft::make_device_matrix_view<float const, std::int64_t>(
                       st.probe_rows_on_host ? nullptr : st.block_probe->data(), block->rows, dim)
                   : vss::list_column_as_dataset_view(st.staged_probe.view, dim);
    n_left     = static_cast<std::int64_t>(st.queries.extent(0));
    st.n_left  = n_left;
  }

  /// The answer of a probe batch with no rows.
  [[nodiscard]] std::unique_ptr<operator_data> empty_answer()
  {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
    empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
    empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
    batches.push_back(sirius::make_data_batch(std::make_unique<cudf::table>(std::move(empty_cols)),
                                              *mem_space,
                                              stream,
                                              _op.batch_telemetry()));
    return output(std::move(batches));
  }

  /// Refuses, once, what no out-of-memory retry could help.
  void refuse_what_cannot_run()
  {
    // Pair tasks hold no [n x k] answer.
    if (pair_step == nullptr) { vss::column_size(n_left * k_join, "vector join output"); }
    // A top-k task whose estimate exceeds the whole device cannot fit however much memory is
    // released, so it is refused here, once, instead of through every out-of-memory retry of the
    // executor (k = 32,768 over 10,000 probes used to retry for ~5 s before failing). The routing
    // term of a clustered search only paces admission; it is not exact enough to refuse on. The
    // rest is within ~10% of the peak, and pinned tables hold part of the device, so the bar is 90%
    // of it.
    if (_request.mode != vss::vector_join_mode::threshold) {
      auto const need = block != nullptr
                          ? _op.estimate_for_rows(static_cast<std::size_t>(block->rows), false)
                          : _op.per_left_batch_estimate(left_idx, /*with_routing=*/false);
      if (static_cast<double>(need) > 0.9 * static_cast<double>(mem_space->get_max_memory())) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream] k = " + std::to_string(k_join) + " over " +
          std::to_string(n_left) + " probe rows needs ~" +
          std::format("{:.1f}", need / 1073741824.0) +
          " GiB on the device, more than 90% of its (" +
          std::format("{:.1f}", mem_space->get_max_memory() / 1073741824.0) +
          " GiB); ask for a smaller k or join fewer probe rows at a time");
      }
    }

    if (ip && !gemm_search_enabled()) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] an inner-product join runs only through the GEMM "
        "search, which SIRIUS_VSS_GEMM_TOPK=0 turns off");
    }
  }

  /// Sets up what a threshold join's pairs go through, and a keyed join's keys.
  void prepare_answer()
  {
    if (_op._sirius_ctx != nullptr) {
      for (auto const* h : _op._sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(
             cucascade::memory::Tier::HOST)) {
        if (host_space == nullptr ||
            h->get_available_memory() > host_space->get_available_memory()) {
          host_space = h;
        }
      }
    }
    // Pairs stay on the device while closing them (a concatenate, as much again) and the next
    // search still fit in what this task can allocate; past that they move to host. A fixed share
    // of the device moved answers that fit (379M pairs on SIFT1M: 0.44 -> 1.24 s).
    if (auto const* env = std::getenv("SIRIUS_VSS_RADIUS_FLUSH_PAIRS")) {
      flush_pairs = std::max<std::int64_t>(1, std::atoll(env));
    }
    max_pairs =
      static_cast<std::int64_t>(((host_space != nullptr ? host_space->get_max_memory() : 0) +
                                 mem_space->get_max_memory() / 2) /
                                16);
    if (best_only) {
      st.best_order.emplace(static_cast<std::size_t>(n_left), stream, mr);
      st.best_scratch.emplace(static_cast<std::size_t>(n_left), stream, mr);
      st.best_ids.emplace(static_cast<std::size_t>(n_left), stream, mr);
      vss::best_match_reset(st.best_order->data(), st.best_ids->data(), n_left, stream);
    }
    if (keyed) {
      _op.ensure_earlier_keys(*mem_space, stream, mr);
      // This task's probe rows are chunk left_idx of the pinned probe table.
      std::int64_t const* probe_key = nullptr;
      if (_request.key_is_rowid) {
        if (left_idx + 1 >= _op._probe_row_base.size() ||
            _op._probe_row_base[left_idx + 1] - _op._probe_row_base[left_idx] != n_left) {
          throw std::runtime_error("[sirius_physical_vector_join_stream] probe chunk " +
                                   std::to_string(left_idx) + " holds " + std::to_string(n_left) +
                                   " rows, not what the pin records for it");
        }
        auto const first_row = _op._probe_row_base[left_idx];
        cudf::numeric_scalar<std::int64_t> const first(first_row, true, stream),
          step(1, true, stream);
        st.probe_positions =
          cudf::sequence(static_cast<cudf::size_type>(n_left), first, step, stream, mr);
        probe_key        = st.probe_positions->view().data<std::int64_t>();
        st.probe_key_max = first_row + n_left - 1;
      } else {
        st.probe_keys.emplace(static_cast<std::size_t>(n_left), stream, mr);
        auto const staged = vss::stage_pinned_column_chunk(
          *_op._left_pin, _request.earlier_by, left_idx, *mem_space, stream, _op.batch_telemetry());
        if (static_cast<std::int64_t>(staged.view.size()) != n_left) {
          throw std::runtime_error(
            "[sirius_physical_vector_join_stream] earlier_by column '" + _request.earlier_by +
            "' and the probe vectors are chunked differently; both must come from the same pin");
        }
        copy_earlier_keys(staged.view, st.probe_keys->data(), _request.earlier_by, stream, mr);
        probe_key = st.probe_keys->data();
      }
      st.pin_keys = {
        probe_key,
        _op._corpus_key ? static_cast<std::int64_t const*>(_op._corpus_key->data()) : nullptr,
        _request.key_differs};
      if (_lists != nullptr) {
        st.layout_keys = {probe_key,
                          static_cast<std::int64_t const*>(
                            _op._layout_key ? _op._layout_key->data() : _lists->row_ids->data()),
                          _request.key_differs};
      }
      // SIRIUS_VSS_EARLIER_CUT=0 searches the whole lists, the A/B for the cut.
      bool const cut_enabled = [] {
        auto const* v = std::getenv("SIRIUS_VSS_EARLIER_CUT");
        return v == nullptr || std::string_view{v} != "0";
      }();
      if (cut_enabled && !_request.key_differs && !_op._layout_key_host.empty()) {
        if (!_request.key_is_rowid) {
          std::vector<std::int64_t> host(static_cast<std::size_t>(n_left));
          CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(),
                                        st.probe_keys->data(),
                                        host.size() * sizeof(std::int64_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          st.probe_key_max = *std::max_element(host.begin(), host.end());
        }
        st.earlier_cut = _op._layout_key_host.data();
      }
    }
  }

  /// The whole-corpus fold, chunk by chunk.
  void search_corpus()
  {
    auto const n_chunks = _op._corpus->num_chunks();
    std::int64_t offset = 0;  // running base of the current chunk in right-table row space
    // Every exhaustive join is an exact one, and an exact L2 join ranks by the FP32 sum of
    // (q_j - x_j)^2. The GEMM's |q|^2 + |x|^2 - 2 q.x cancels for rows far from the origin and
    // close together, so L2 is computed from the differences (cuVS's Unexpanded form) here.
    // Cosine likewise, on rows normalized in FP32: |q^ - x^|^2 = 2 (1 - cos), so the squared
    // distance is halved at the end and a radius doubled. The GEMM's 1 - q.x / (|q| |x|) loses
    // near-duplicates (cosine distances below about 1e-7) to the rounding of the dot product.
    bool const exhaustive_cosine = _request.metric == "cosine";
    auto const exhaustive_metric =
      exhaustive_cosine
        ? cuvs::distance::DistanceType::L2Unexpanded
        : vss::join_selection_distance_type_from_metric(_request.metric, /*exact_unexpanded=*/true);
    auto const exhaustive_radius = exhaustive_cosine ? 2.f * radius_eps : radius_eps;
    std::optional<rmm::device_uvector<float>> unit_queries;
    if (exhaustive_cosine) {
      unit_queries.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
      vss::normalize_rows(st.queries.data_handle(), n_left, dim, unit_queries->data(), stream);
      st.radius_scale = 0.5f;
    }
    auto const fold_queries =
      exhaustive_cosine
        ? raft::make_device_matrix_view<const float, int64_t>(unit_queries->data(), n_left, dim)
        : st.queries;
    // A chunk's rows as the search reads them: normalized for cosine, into a buffer that lives
    // until the next chunk's (its release is ordered behind the searches on the stream).
    std::optional<rmm::device_uvector<float>> unit_chunk;
    auto searched_rows = [&](vss::dataset_matrix_view rows) {
      if (!exhaustive_cosine || rows.extent(0) == 0) { return rows; }
      unit_chunk.emplace(static_cast<std::size_t>(rows.extent(0) * dim), stream, mr);
      vss::normalize_rows(rows.data_handle(), rows.extent(0), dim, unit_chunk->data(), stream);
      return raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
        unit_chunk->data(), rows.extent(0), dim);
    };

    // Staging runs on its own stream so chunk j+1's H2D overlaps chunk j's compute. The
    // converter host-synchronizes at the end of its copy, so what is actually overlapped is
    // "host blocked on the copy" against "GPU busy with the previous chunk" -- the compute
    // below is issued asynchronously and does not block the host. Only worth a stream when
    // the corpus actually streams; a GPU-tier pin stages nothing.
    std::optional<rmm::cuda_stream> staging_stream;
    if (_op._corpus->is_streaming()) { staging_stream.emplace(); }
    auto const stage_on = staging_stream ? staging_stream->view() : stream;

    auto prefetched = n_chunks > 0 ? stage_for_fold(*_op._corpus, 0, *mem_space, stage_on, stream)
                                   : staged_vector_chunk{};

    auto advance = [&](std::size_t next) {
      prefetched = next < n_chunks
                     ? stage_for_fold(*_op._corpus, next, *mem_space, stage_on, stream)
                     : staged_vector_chunk{};
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
      auto const total = vss::column_size(n_left * k_join, "vector join output");
      st.acc_neighbors = cudf::make_column_from_scalar(miss_id, total, stream, mr);
      st.acc_distances = cudf::make_column_from_scalar(miss_distance, total, stream, mr);
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
          kept.emplace(filter_corpus_chunk(*_op._right_pin,
                                           _request.right_predicates,
                                           j,
                                           dataset,
                                           offset,
                                           *mem_space,
                                           stream,
                                           mr,
                                           _op.batch_telemetry()));
          searched = raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
            kept->vectors.data(), kept->rows, dim);
        }
        // The search holds its score tile and the prepared chunk, and its edges grow the result;
        // the last chunk's edge count is the estimate for this one's.
        if (_op._corpus->is_streaming()) {
          ensure_device_headroom(_op._sirius_ctx,
                                 *mem_space,
                                 vss::gemm_search_tile_bytes() +
                                   static_cast<std::size_t>(batch_rows) *
                                     static_cast<std::size_t>(dim + 4) * sizeof(float) +
                                   2 * last_edge_bytes);
        }
        auto edges = searched.extent(0) == 0
                       ? vss::threshold_join_result{nullptr, nullptr, nullptr, 0}
                       : threshold_search(res,
                                          searched_rows(searched),
                                          fold_queries,
                                          exhaustive_radius,
                                          exhaustive_metric,
                                          mr,
                                          kGemmMinProbeRows,
                                          count_only && !keyed);
        last_edge_bytes =
          static_cast<std::size_t>(edges.n_edges) * (2 * sizeof(std::int64_t) + sizeof(float));
        advance(j + 1);
        release_staged(staged);
        auto const chunk_base = offset;
        offset += batch_rows;
        if (count_only && !keyed) {
          note_radius_part(edges.n_edges);
        } else if (edges.n_edges > 0) {
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
          auto const pairs = static_cast<std::int64_t>(edges.neighbors->size());
          if (best_only) {
            auto const left = cudf::cast(
              edges.query_rows->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
            fold_best(left->view().data<std::int32_t>(),
                      edges.neighbors->view().data<std::int64_t>(),
                      edges.distances->view().data<float>(),
                      pairs,
                      std::numeric_limits<float>::infinity(),
                      st.pin_keys,
                      nullptr);
            continue;
          }
          if (keyed) {
            auto const left = cudf::cast(
              edges.query_rows->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
            keep_keyed(left->view().data<std::int32_t>(),
                       edges.neighbors->view().data<std::int64_t>(),
                       edges.distances->view().data<float>(),
                       pairs);
            continue;
          }
          found.left.push_back(cudf::cast(
            edges.query_rows->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr));
          found.neighbors.push_back(std::move(edges.neighbors));
          found.distances.push_back(std::move(edges.distances));
          note_radius_part(pairs);
        }
        continue;
      }

      auto const k_eff = std::min<std::int64_t>(k_join, batch_rows);

      // Cosine goes through the GEMM-ranked search; L2 (unexpanded, above) through cuVS.
      if (_op._corpus->is_streaming()) {
        ensure_device_headroom(
          _op._sirius_ctx,
          *mem_space,
          vss::gemm_search_tile_bytes() + static_cast<std::size_t>(batch_rows) *
                                            static_cast<std::size_t>(dim + 4) * sizeof(float));
      }
      auto const rows = searched_rows(dataset);
      auto knn = gemm_search_enabled() && vss::gemm_search_supports(exhaustive_metric) &&
                     n_left >= kGemmMinProbeRows
                   ? vss::gemm_topk(res, rows, fold_queries, k_eff, exhaustive_metric, mr)
                   : vss::brute_force_knn(res, rows, fold_queries, k_eff, exhaustive_metric, mr);

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
      st.exhaustive_rows_seen += batch_rows;
      vss::fold_topk_rows(st.acc_distances->mutable_view().data<float>(),
                          st.acc_neighbors->mutable_view().data<std::int64_t>(),
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
    emit_best();
    if (exhaustive_cosine) {
      if (radius_join) {
        for (auto& part : found.distances) {
          vss::scale_in_place(part->mutable_view().data<float>(), part->size(), 0.5f, stream);
        }
      } else {
        vss::scale_in_place(
          st.acc_distances->mutable_view().data<float>(), n_left * k_join, 0.5f, stream);
      }
    }
  }

  /// The task's output, from what the search left in the state.
  [[nodiscard]] std::unique_ptr<operator_data> shape_answer()
  {
    // An empty right table joins to nothing: the answer is empty, as DuckDB's is.
    bool const nothing_searched =
      !radius_join &&
      (!st.acc_neighbors || (_op._centroids == nullptr && st.exhaustive_rows_seen == 0));
    if (nothing_searched && _op._right_total_rows > 0) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] right table produced no rows to join against");
    }

    if (count_only) {
      // One row, the pairs this probe batch found; the aggregate above sums the batches.
      cudf::numeric_scalar<std::int64_t> const count(found.total, true, stream);
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(cudf::make_column_from_scalar(count, 1, stream, mr));
      auto batch = sirius::make_data_batch(
        std::make_unique<cudf::table>(std::move(cols)), *mem_space, stream, _op.batch_telemetry());
      std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
      batches.push_back(std::move(batch));
      return output(std::move(batches));
    }

    vss::shaped_join_result shaped;
    if (nothing_searched) {
      shaped.left_rows = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
      shaped.neighbors = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64});
      shaped.distances = cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32});
    } else if (block != nullptr) {
      // One output per left batch the block touched, shaped from its rows of the accumulator, with
      // left rows counted from the start of that batch.
      std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
      std::vector<std::size_t> partitions;
      std::int64_t first = 0;
      for (auto const& piece : block->pieces) {
        auto const rows = piece.end - piece.begin;
        std::vector<cudf::size_type> const span{
          static_cast<cudf::size_type>(first * k_join),
          static_cast<cudf::size_type>((first + rows) * k_join)};
        auto neighbors = std::make_unique<cudf::column>(
          cudf::slice(st.acc_neighbors->view(), span, stream).front(), stream, mr);
        auto distances = std::make_unique<cudf::column>(
          cudf::slice(st.acc_distances->view(), span, stream).front(), stream, mr);
        auto part =
          _request.mode == vss::vector_join_mode::global_top_k
            ? vss::shape_global_top_k(
                neighbors->view(), distances->view(), rows, k_join, _request.k, stream, mr)
            : vss::shape_per_row_top_k(
                std::move(neighbors), std::move(distances), rows, k_join, stream, mr);
        if (piece.begin > 0) {
          cudf::numeric_scalar<std::int32_t> const shift(
            static_cast<std::int32_t>(piece.begin), true, stream);
          part.left_rows = cudf::binary_operation(part.left_rows->view(),
                                                  shift,
                                                  cudf::binary_operator::ADD,
                                                  cudf::data_type{cudf::type_id::INT32},
                                                  stream,
                                                  mr);
        }
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(std::move(part.left_rows));
        cols.push_back(std::move(part.neighbors));
        cols.push_back(std::move(part.distances));
        batches.push_back(sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                                  *mem_space,
                                                  stream,
                                                  _op.batch_telemetry()));
        partitions.push_back(piece.chunk);
        first += rows;
      }
      return std::make_unique<vector_join_stream_block_output>(
        std::move(batches), std::move(partitions), join_in->ticket());
    } else if (radius_join) {
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
      shaped.left_rows = join_cols(found.left, cudf::type_id::INT32);
      shaped.neighbors = join_cols(found.neighbors, cudf::type_id::INT64);
      shaped.distances = join_cols(found.distances, cudf::type_id::FLOAT32);
    } else {
      // The fold is mode-independent; only which of its candidates survive is not.
      switch (_request.mode) {
        case vss::vector_join_mode::global_top_k: {
          // Each left row is searched to depth k_join (k clamped to the corpus), but the batch
          // keeps the requested k: with k beyond one row's candidates the answer spans several
          // rows.
          shaped = vss::shape_global_top_k(st.acc_neighbors->view(),
                                           st.acc_distances->view(),
                                           n_left,
                                           k_join,
                                           _request.k,
                                           stream,
                                           mr);
          break;
        }
        case vss::vector_join_mode::threshold: {
          // The kernel works in distance space. For cosine with a similarity threshold the
          // user's "score >= eps" is the same set as "distance <= 1 - eps"; for a distance
          // threshold it is eps directly.
          auto const max_distance = radius_eps;
          bool truncated          = false;
          shaped                  = vss::shape_threshold(st.acc_neighbors->view(),
                                                         st.acc_distances->view(),
                                                         n_left,
                                                         k_join,
                                                         max_distance,
                                                         truncated,
                                                         stream,
                                                         mr);
          if (truncated) {
            throw std::runtime_error(
              "[sirius_physical_vector_join_stream] threshold join truncated: at least one left "
              "row "
              "has k=" +
              std::to_string(k_join) +
              " neighbours inside the threshold, so pairs beyond k were never searched for. Raise "
              "k "
              "or tighten eps.");
          }
          break;
        }
        case vss::vector_join_mode::per_row_top_k:
        default: {
          shaped = vss::shape_per_row_top_k(
            std::move(st.acc_neighbors), std::move(st.acc_distances), n_left, k_join, stream, mr);
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

    auto batch =
      sirius::make_data_batch(std::move(out_table), *mem_space, stream, _op.batch_telemetry());
    auto batches = std::move(found.offloaded);
    batches.push_back(std::move(batch));
    return output(std::move(batches));
  }

  /// A block's rows as an exhaustive search's pair tasks read them.
  void prepare_rows_for_pairs()
  {
    if (_request.metric != "cosine") { return; }
    st.unit_rows.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
    vss::normalize_rows(st.queries.data_handle(), n_left, dim, st.unit_rows->data(), stream);
    stream.sync();
    st.block_probe.reset();
    st.queries =
      raft::make_device_matrix_view<const float, std::int64_t>(st.unit_rows->data(), n_left, dim);
    st.radius_scale = 0.5f;
  }

  /// One corpus chunk against the block's rows, in searches of at most pair_call_rows() rows,
  /// from the step's row on. Pairs are the task's own until it returns, so a retry has nothing to
  /// undo. When a search runs out of memory with pairs held, those go out first; with none held it
  /// is run on fewer rows, down to one, past which the shortage is the engine's to retry.
  /// @return where the next task starts, if this one stopped early.
  [[nodiscard]] std::optional<vss::pair_schedule::step> search_pair()
  {
    auto const j = pair_step->chunk;
    auto staged  = _op._corpus->stage_for_pair(j, *mem_space, stream);
    sync_stream_on_exit const quiesce{stream};
    auto const dataset    = vss::list_column_as_dataset_view(staged.view, dim);
    auto const batch_rows = static_cast<std::int64_t>(dataset.extent(0));
    if (static_cast<std::size_t>(batch_rows) != _op._corpus->chunk_rows(j)) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] corpus chunk " +
                               std::to_string(j) + " holds " + std::to_string(batch_rows) +
                               " rows, not the " + std::to_string(_op._corpus->chunk_rows(j)) +
                               " its batch reports");
    }
    if (batch_rows == 0) { return std::nullopt; }
    bool const cosine = _request.metric == "cosine";
    auto const metric_searched =
      cosine ? cuvs::distance::DistanceType::L2Unexpanded
             : vss::join_selection_distance_type_from_metric(_request.metric,
                                                             /*exact_unexpanded=*/true);
    std::optional<rmm::device_uvector<float>> unit_chunk;
    auto corpus = dataset;
    if (cosine) {
      unit_chunk.emplace(static_cast<std::size_t>(batch_rows * dim), stream, mr);
      vss::normalize_rows(dataset.data_handle(), batch_rows, dim, unit_chunk->data(), stream);
      corpus = raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
        unit_chunk->data(), batch_rows, dim);
    }
    auto const base = _op._corpus_chunk_base[j];
    auto call_rows  = first_call_rows();
    for (auto row = pair_step->row; row < n_left;) {
      auto const rows = std::min(call_rows, n_left - row);
      auto const queries =
        raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
          st.queries.data_handle() + row * dim, rows, dim);
      std::optional<vss::threshold_join_result> edges;
      try {
        edges.emplace(threshold_search(res,
                                       corpus,
                                       queries,
                                       cosine ? 2.f * radius_eps : radius_eps,
                                       metric_searched,
                                       mr,
                                       kGemmMinProbeRows,
                                       count_only));
      } catch (std::exception const& shortage) {
        if (!search_can_shrink(shortage)) { throw; }
        stream.sync();
        if (found.held > 0) { return rest_of_chunk(0, row, call_rows); }
        if (rows == 1) { throw; }
        call_rows = rows / 2;
        continue;
      }
      call_rows = rows_for_next_call(call_rows, rows, edges->n_edges);
      if (count_only) {
        note_radius_part(edges->n_edges);
      } else if (edges->n_edges > 0) {
        if (base != 0) {
          cudf::numeric_scalar<std::int64_t> const shift(base, true, stream);
          edges->neighbors = cudf::binary_operation(edges->neighbors->view(),
                                                    shift,
                                                    cudf::binary_operator::ADD,
                                                    cudf::data_type{cudf::type_id::INT64},
                                                    stream,
                                                    mr);
        }
        auto left =
          cudf::cast(edges->query_rows->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
        if (row != 0) {
          cudf::numeric_scalar<std::int32_t> const shift(
            static_cast<std::int32_t>(row), true, stream);
          left = cudf::binary_operation(left->view(),
                                        shift,
                                        cudf::binary_operator::ADD,
                                        cudf::data_type{cudf::type_id::INT32},
                                        stream,
                                        mr);
        }
        auto const pairs = edges->n_edges;
        found.left.push_back(std::move(left));
        found.neighbors.push_back(std::move(edges->neighbors));
        found.distances.push_back(std::move(edges->distances));
        note_radius_part(pairs, /*may_stop=*/true);
      }
      row += rows;
      if (row < n_left && holds_enough_to_stop()) { return rest_of_chunk(0, row, call_rows); }
    }
    return std::nullopt;
  }

  /// What a pair task found, as its output: one batch per probe chunk with pairs in it.
  [[nodiscard]] std::unique_ptr<operator_data> pair_answer()
  {
    if (count_only) {
      cudf::numeric_scalar<std::int64_t> const count(found.total, true, stream);
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(cudf::make_column_from_scalar(count, 1, stream, mr));
      auto batch = sirius::make_data_batch(
        std::make_unique<cudf::table>(std::move(cols)), *mem_space, stream, _op.batch_telemetry());
      std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
      batches.push_back(std::move(batch));
      return output(std::move(batches));
    }
    auto batches   = std::move(found.offloaded);
    auto chunks    = std::move(found.offloaded_chunks);
    bool on_device = false;
    if (!found.left.empty()) {
      for (auto& [chunk, table] : pairs_by_probe_chunk()) {
        batches.push_back(
          sirius::make_data_batch(std::move(table), *mem_space, stream, _op.batch_telemetry()));
        chunks.push_back(chunk);
        on_device = true;
      }
    }
    if (batches.empty()) { return output({}); }
    // Materialize takes its device from a piece on it.
    if (!on_device) {
      std::vector<std::unique_ptr<cudf::column>> empty_cols;
      empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
      empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
      empty_cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
      batches.push_back(
        sirius::make_data_batch(std::make_unique<cudf::table>(std::move(empty_cols)),
                                *mem_space,
                                stream,
                                _op.batch_telemetry()));
      chunks.push_back(chunks.front());
    }
    return std::make_unique<vector_join_stream_block_output>(
      std::move(batches), std::move(chunks), join_in->ticket());
  }

  /// A build task's output: nothing goes on to materialize.
  [[nodiscard]] std::unique_ptr<operator_data> no_answer() { return output({}); }

 protected:
  [[nodiscard]] vss::pair_schedule::step rest_of_chunk(std::size_t run,
                                                       std::int64_t row,
                                                       std::int64_t call_rows) const
  {
    auto rest      = *pair_step;
    rest.run       = run;
    rest.row       = row;
    rest.call_rows = call_rows;
    return rest;
  }

  [[nodiscard]] std::int64_t first_call_rows() const
  { return pair_step->call_rows > 0 ? pair_step->call_rows : pair_call_rows(dim); }

  /// Probe rows for the search after one that found @p pairs over @p rows: a search should leave
  /// about what a task may hold, or the pairs need cutting after the work is done. For this chunk
  /// only: how many pairs a row finds differs from chunk to chunk.
  [[nodiscard]] std::int64_t rows_for_next_call(std::int64_t call_rows,
                                                std::int64_t rows,
                                                std::int64_t pairs) const
  {
    auto const most = static_cast<std::int64_t>(mem_space->get_max_memory() / 8 / 16);
    if (count_only || pairs <= most) { return call_rows; }
    return std::clamp<std::int64_t>(
      static_cast<std::int64_t>(static_cast<double>(rows) * static_cast<double>(most) /
                                static_cast<double>(pairs)),
      1,
      call_rows);
  }

  /// Whether a pair task should end here and let its pairs out: the split by probe chunk and
  /// materialize take as much again as the pairs while the next search needs its room, or a piece
  /// has already gone to the host.
  [[nodiscard]] bool holds_enough_to_stop()
  {
    if (count_only) { return false; }
    if (!found.offloaded.empty()) { return true; }
    if (flush_pairs) { return found.held > *flush_pairs; }
    if (found.held < (std::int64_t{1} << 20)) { return false; }
    auto const held_bytes = static_cast<std::size_t>(found.held) * 16;
    return held_bytes > mem_space->get_max_memory() / 8 ||
           2 * held_bytes + next_search_room > mem_space->get_available_memory(stream);
  }

  /// The pairs held on the device, closed: their parts joined, and scaled to what the join reports.
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> closed_pairs()
  {
    auto concat = [&](std::vector<std::unique_ptr<cudf::column>>& parts) {
      std::unique_ptr<cudf::column> out;
      if (parts.size() == 1) {
        out = std::move(parts.front());
      } else {
        std::vector<cudf::column_view> views;
        for (auto const& c : parts) {
          views.push_back(c->view());
        }
        out = cudf::concatenate(views, stream, mr);
      }
      parts.clear();
      return out;
    };
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(concat(found.left));
    cols.push_back(concat(found.neighbors));
    cols.push_back(concat(found.distances));
    if (st.radius_scale != 1.0f) {
      vss::scale_in_place(
        cols[2]->mutable_view().data<float>(), cols[2]->size(), st.radius_scale, stream);
    }
    return cols;
  }

  /// A pair task's left rows are rows of its block; materialize gathers a probe chunk at a time, so
  /// its pairs split by chunk here, left rows counted from the start of each. Skips chunks with
  /// none.
  [[nodiscard]] std::vector<std::pair<std::size_t, std::unique_ptr<cudf::table>>>
  pairs_by_probe_chunk()
  {
    auto table = std::make_unique<cudf::table>(closed_pairs());
    std::vector<std::pair<std::size_t, std::unique_ptr<cudf::table>>> out;
    auto const shifted = [&](std::unique_ptr<cudf::table> part, std::int64_t by) {
      if (by == 0) { return part; }
      cudf::numeric_scalar<std::int32_t> const shift(static_cast<std::int32_t>(by), true, stream);
      auto cols = part->release();
      cols[0]   = cudf::binary_operation(cols[0]->view(),
                                         shift,
                                         cudf::binary_operator::ADD,
                                         cudf::data_type{cudf::type_id::INT32},
                                         stream,
                                         mr);
      return std::make_unique<cudf::table>(std::move(cols));
    };
    auto const& pieces = block->pieces;
    if (pieces.size() == 1) {
      out.emplace_back(pieces.front().chunk, shifted(std::move(table), pieces.front().begin));
      return out;
    }
    std::vector<std::int32_t> first_rows;
    first_rows.reserve(pieces.size());
    std::int64_t first = 0;
    for (auto const& piece : pieces) {
      first_rows.push_back(static_cast<std::int32_t>(first));
      first += piece.end - piece.begin;
    }
    auto firsts = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(first_rows.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(firsts->mutable_view().data<std::int32_t>(),
                                  first_rows.data(),
                                  first_rows.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    stream.sync();
    auto const above = cudf::upper_bound(cudf::table_view{{firsts->view()}},
                                         cudf::table_view{{table->view().column(0)}},
                                         {cudf::order::ASCENDING},
                                         {cudf::null_order::BEFORE},
                                         stream,
                                         mr);
    cudf::numeric_scalar<std::int32_t> const one(1, true, stream);
    auto const chunk_of    = cudf::binary_operation(above->view(),
                                                    one,
                                                    cudf::binary_operator::SUB,
                                                    cudf::data_type{cudf::type_id::INT32},
                                                    stream,
                                                    mr);
    auto [parted, offsets] = cudf::partition(
      table->view(), chunk_of->view(), static_cast<cudf::size_type>(pieces.size()), stream, mr);
    table.reset();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      if (offsets[i + 1] == offsets[i]) { continue; }
      auto const part = cudf::slice(parted->view(), {offsets[i], offsets[i + 1]}, stream).front();
      out.emplace_back(
        pieces[i].chunk,
        shifted(std::make_unique<cudf::table>(part, stream, mr), pieces[i].begin - first_rows[i]));
    }
    return out;
  }

  // A capped task keeps its admission ticket until the materialize stage is done with the output.
  [[nodiscard]] std::unique_ptr<operator_data> output(
    std::vector<std::shared_ptr<cucascade::data_batch>> batches)
  {
    if (join_in->ticket()) {
      return std::make_unique<vector_join_stream_output>(
        std::move(batches), left_idx, join_in->ticket());
    }
    return std::make_unique<partitioned_operator_data>(
      std::move(batches), left_idx, partition_placement::unpinned(left_idx + 1));
  }

  // The kernel works in distance space. For cosine with a similarity threshold the user's
  // "score >= eps" is the same set as "distance <= 1 - eps"; for a distance threshold it is eps
  // directly. Identical to what the shape_threshold path computes. For an inner product,
  // "q.x >= eps" is D = -2 q.x <= -2 eps, and a negative-inner-product bound "-q.x <= eps" is
  // D <= 2 eps; past FLOAT's range the radius is infinite (every pair) or -infinite (none).
  [[nodiscard]] float radius_for_kernel() const
  {
    if (ip) {
      double const r = _request.output_type == vss::vector_join_output_type::similarity
                         ? -2.0 * _request.eps
                         : 2.0 * _request.eps;
      if (r > static_cast<double>(std::numeric_limits<float>::max())) {
        return std::numeric_limits<float>::infinity();
      }
      if (r < -static_cast<double>(std::numeric_limits<float>::max())) {
        return -std::numeric_limits<float>::infinity();
      }
      return static_cast<float>(r);
    }
    return _request.output_type == vss::vector_join_output_type::similarity
             ? static_cast<float>(1.0 - _request.eps)
             : static_cast<float>(_request.eps);
  }

  [[nodiscard]] bool must_flush()
  {
    if (flush_pairs) { return found.held > *flush_pairs; }
    if (found.held < (std::int64_t{1} << 20)) { return false; }
    auto const held_bytes = static_cast<std::size_t>(found.held) * 16;
    return held_bytes + next_search_room > mem_space->get_available_memory(stream);
  }

  void flush_radius()
  {
    if (found.left.empty()) { return; }
    if (pair_step != nullptr) {
      for (auto& [chunk, table] : pairs_by_probe_chunk()) {
        auto piece =
          sirius::make_data_batch(std::move(table), *mem_space, stream, _op.batch_telemetry());
        piece->to_mutable().convert_to<cucascade::host_data_representation>(
          sirius::converter_registry::get(), host_space, stream);
        found.offloaded.push_back(std::move(piece));
        found.offloaded_chunks.push_back(chunk);
      }
      found.held = 0;
      return;
    }
    auto piece = sirius::make_data_batch(
      std::make_unique<cudf::table>(closed_pairs()), *mem_space, stream, _op.batch_telemetry());
    piece->to_mutable().convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host_space, stream);
    found.offloaded.push_back(std::move(piece));
    SIRIUS_LOG_DEBUG(
      "[sirius_physical_vector_join_stream] left batch {}: {} threshold pairs moved "
      "to host memory (piece {})",
      left_idx,
      found.held,
      found.offloaded.size());
    found.held = 0;
  }

  /// @param may_stop The caller can end the task right after this: its pairs then leave as its
  /// output instead of going to the host, where nothing can move them. A forced flush still goes.
  void note_radius_part(std::int64_t pairs, bool may_stop = false)
  {
    found.held += pairs;
    found.total += pairs;
    if (!count_only && pair_step == nullptr && found.total > max_pairs) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] the pairs within the threshold reach " +
        std::to_string(found.total) + " for one probe batch, more than host and device memory " +
        "can hold as output (~" + std::to_string(max_pairs) + "); tighten the threshold or join " +
        "fewer probe rows at a time");
    }
    if (host_space != nullptr && !count_only && (!may_stop || flush_pairs) && must_flush()) {
      flush_radius();
    }
  }

  void fold_best(std::int32_t const* rows,
                 std::int64_t const* ids,
                 float const* distances,
                 std::int64_t n,
                 float max_distance,
                 vss::earlier_keys keys,
                 float* bound_rows)
  {
    vss::best_match_fold(rows,
                         ids,
                         distances,
                         n,
                         max_distance,
                         keys,
                         st.best_order->data(),
                         st.best_ids->data(),
                         st.best_scratch->data(),
                         n_left,
                         bound_rows,
                         stream);
  }

  // A search that returns finished pairs (exact distances, pin-row ids): the ones the keys keep
  // join the answer, or only its count.
  void keep_keyed(std::int32_t const* rows,
                  std::int64_t const* ids,
                  float const* distances,
                  std::int64_t n)
  {
    auto kept        = vss::take_pairs(rows,
                                       ids,
                                       distances,
                                       n,
                                       st.pin_keys,
                                       std::numeric_limits<float>::infinity(),
                                       nullptr,
                                       false,
                                       stream,
                                       mr);
    auto const pairs = static_cast<std::int64_t>(kept.rows.size());
    if (pairs == 0) { return; }
    if (!count_only) {
      found.left.push_back(
        std::make_unique<cudf::column>(std::move(kept.rows), rmm::device_buffer{}, 0));
      found.neighbors.push_back(
        std::make_unique<cudf::column>(std::move(kept.ids), rmm::device_buffer{}, 0));
      found.distances.push_back(
        std::make_unique<cudf::column>(std::move(kept.distances), rmm::device_buffer{}, 0));
    }
    note_radius_part(pairs);
  }

  void emit_best()
  {
    if (!best_only) { return; }
    auto best = vss::best_match_emit(
      st.best_order->data(),
      st.best_ids->data(),
      n_left,
      st.best_ids_are_layout ? static_cast<std::int64_t const*>(_lists->row_ids->data()) : nullptr,
      st.best_take_sqrt,
      stream,
      mr);
    auto const pairs = static_cast<std::int64_t>(best.rows.size());
    if (pairs == 0) { return; }
    found.left.push_back(
      std::make_unique<cudf::column>(std::move(best.rows), rmm::device_buffer{}, 0));
    found.neighbors.push_back(
      std::make_unique<cudf::column>(std::move(best.ids), rmm::device_buffer{}, 0));
    found.distances.push_back(
      std::make_unique<cudf::column>(std::move(best.distances), rmm::device_buffer{}, 0));
    note_radius_part(pairs);
  }

  sirius_physical_vector_join_stream& _op;
  // Read throughout, and fixed for a task's length.
  sirius::vss::vector_join_request const& _request;
  sirius::vss::cluster_lists const* _lists;
  block_state& st;
  found_pairs& found;
  vector_join_stream_input const* join_in;
  ::cucascade::memory::memory_space* mem_space;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  raft::device_resources res;
  std::size_t left_idx;
  probe_block const* block;
  std::int64_t dim;
  std::int64_t n_left{0};
  // Every mode is served by searching each left row to some depth and then deciding which
  // of those candidates survive. The depth differs: global top-k needs k_global per left
  // row (a single left row may own the entire global answer), and threshold needs a cap on
  // how many in-range neighbours one left row may have, which is what k supplies there.
  std::int64_t k_join;
  cuvs::distance::DistanceType metric;
  // An inner-product join ranks by D = -2 q.x (smaller is closer, like a distance; the factor is
  // exact in FP32), so every fold, merge and bound works on it unchanged. Its scores are
  // computed from the FP32 rows: by the FP32 GEMM, or by the FP32 re-score of an FP16 filter.
  bool ip;
  // A radius join needs no fold. Top-k folds because chunk j+1 can displace chunk j's winners,
  // so a running [n_left, k] merge is unavoidable; "within eps" is independent per chunk, so a
  // chunk's surviving edges are final when produced and the chunks only have to be concatenated.
  // That is why this path accumulates ragged edge lists instead of a fixed-width block, and why
  // it has no k -- and therefore none of the k <= 1024 ceiling that knn_merge_parts imposes.
  bool radius_join;
  // A count-only statement never sees the pairs: each part is counted and dropped, and the task
  // ends with one BIGINT row, the count, in place of the edge lists and host pieces.
  bool count_only;
  // A threshold join asked for the pairs in one order only (earlier_by / other_by) keeps a pair
  // by its two rows' keys; candidate ids are pin rows, or over inexact lists their layout rows.
  bool keyed;
  // A best-match join folds every pair it finds into one (distance, id) per probe row and emits
  // those at the end, through the pieces a plain threshold join's pairs leave by.
  bool best_only;
  float radius_eps;
  // A threshold's answer can outgrow the device. Once the pairs held on it (16 B each) pass a
  // budget they are closed into an output batch and moved to host memory, the way the downgrade
  // executor spills a batch waiting between operators; materialize takes the batches one at a time,
  // so only the host pool bounds the answer. Past host and device together nothing can hold it,
  // and an out-of-memory retry would only re-fold the corpus to the same point.
  ::cucascade::memory::memory_space const* host_space{nullptr};
  std::optional<std::int64_t> flush_pairs;
  std::size_t next_search_room;
  std::int64_t max_pairs{0};
  // Under pair tasks, this task's place in the schedule.
  vss::pair_schedule::step const* pair_step;
};

//===----------------------------------------------------------------------===//
// The clustered search
//===----------------------------------------------------------------------===//
/// The search that walks (probe run x neighbouring cluster) pairs. @ref prepare builds a block's
/// routing and probe copies and settles the search's form, @ref begin_sweep and @ref search_chunk
/// take each pass and each corpus chunk in turn, and @ref finish closes the accumulators. What
/// these share across chunks is in @ref block_state::search_state; here are this task's gather
/// buffers and the chunks it holds staged.
class sirius_physical_vector_join_stream::clustered_search : public block_task {
 public:
  explicit clustered_search(block_task const& task)
    : block_task(task),
      s(st.search.emplace(_op._host_pair_blocks)),
      w(s.run),
      dbg(std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr),
      phase_t0(std::chrono::steady_clock::now()),
      stage_on(stream),
      borrowed{stream},
      building_for_pairs(pair_step != nullptr)
  {
  }

  /// A pair task over a block already prepared, its walk its own.
  clustered_search(block_task const& task, block_state::search_run& own)
    : block_task(task),
      s(*st.search),
      w(own),
      dbg(std::getenv("SIRIUS_VECTOR_JOIN_PHASE_DEBUG") != nullptr),
      phase_t0(std::chrono::steady_clock::now()),
      stage_on(stream),
      borrowed{stream}
  {
  }

  void run()
  {
    prepare();
    for (int pass = radius_join && s.any_bounded ? 1 : 0; pass < (s.any_bounded ? 2 : 1); ++pass) {
      begin_sweep(pass);
      for (std::size_t ci = 0; ci < chunks->size(); ++ci) {
        search_chunk(ci);
      }
      run_stretches();
    }
    finish();
    phase("search+fold slices");
    // What the answer does not read goes before it is shaped, which can take a good part of the
    // device.
    st.search.reset();
  }

  /// A build task's part: everything a pair task reads, written here and never after.
  void prepare_for_pairs()
  {
    prepare();
    begin_sweep(s.any_bounded ? 1 : 0);
    phase("block set up");
    if (_op._sirius_ctx != nullptr) {
      _op._sirius_ctx->record_vector_join_prune(
        0,
        static_cast<std::uint64_t>(n_left) * static_cast<std::uint64_t>(_op._right_total_rows),
        s.needed_chunks.size(),
        _op._chunk_cluster_runs.size());
    }
  }

  /// One chunk of the sweep against the block, from the step's run and row on. As in the exhaustive
  /// search, what it holds is its own until it returns.
  /// @return where the next task starts, if this one stopped early.
  [[nodiscard]] std::optional<vss::pair_schedule::step> run_pair()
  {
    set_sweep(s.any_bounded ? 1 : 0);
    keep_from_plan();
    w.running_piece = s.running_bound_rows;
    if (s.any_bounded) { w.candidates.emplace(std::int64_t{1} << 22, stream, mr); }
    auto const j = (*chunks)[pair_step->chunk];
    // Only a search that gathers its probe rows needs room for them, and only for this chunk's.
    std::int64_t gathered = 0;
    if (!s.any_bounded) {
      auto const& edge_begin = s.routed->edge_begin;
      for (auto const& slice : _op._chunk_cluster_runs[j]) {
        auto const c = static_cast<std::size_t>(slice.cluster);
        gathered     = std::max(gathered, edge_begin[c + 1] - edge_begin[c]);
      }
      gathered = std::min(gathered, pair_call_rows(dim));
    }
    routed_queries.emplace(static_cast<std::size_t>(gathered * dim), stream, mr);
    prefetched = stage_next(j);
    search_chunk(pair_step->chunk);
    if (s.any_bounded) {
      check_group();
      flush();
    }
    stream.sync();
    if (_op._sirius_ctx != nullptr) {
      _op._sirius_ctx->record_vector_join_prune(
        static_cast<std::uint64_t>(w.scanned_pairs), 0, 0, 0);
    }
    return stopped_at;
  }

 private:
  using bounded_slice = vss::bound_slice;

  static constexpr std::size_t kFlushBytesPerPair = 80;
  static constexpr std::size_t kFlushMargin       = std::size_t{1} << 28;
  static constexpr std::int64_t kBackPiecePairs   = std::int64_t{1} << 23;

  void phase(const char* name)
  {
    if (!dbg) { return; }
    stream.sync();
    auto const now = std::chrono::steady_clock::now();
    std::fprintf(
      stderr,
      "[vecjoin-phase] %-22s %8.3f s, device in use %.2f GiB\n",
      name,
      std::chrono::duration<double>(now - phase_t0).count(),
      static_cast<double>(mem_space->get_max_memory() - mem_space->get_available_memory(stream)) /
        1073741824.0);
    phase_t0 = now;
  }

  [[nodiscard]] std::chrono::steady_clock::time_point debug_mark()
  {
    if (!dbg) { return std::chrono::steady_clock::time_point{}; }
    stream.sync();
    return std::chrono::steady_clock::now();
  }

  // One piece of a block's host rows at a time on the device.
  template <typename Use>
  void for_each_host_piece(Use&& use)
  {
    constexpr std::int64_t kPieceRows = std::int64_t{1} << 17;
    for (std::int64_t at = 0; at < n_left; at += kPieceRows) {
      auto const rows = std::min(kPieceRows, n_left - at);
      rmm::device_uvector<float> piece(static_cast<std::size_t>(rows * dim), stream, mr);
      st.host_rows->copy_to_device(at, rows, piece.data());
      use(piece.data(), at, rows);
      stream.sync();
    }
  }

  // A search that needs every FP32 row at hand (rare: see plan_search) gets them back.
  void block_rows_to_device()
  {
    if (!st.probe_rows_on_host) { return; }
    st.block_probe.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
    st.host_rows->copy_to_device(0, n_left, st.block_probe->data());
    s.search_queries =
      raft::make_device_matrix_view<float const, std::int64_t>(st.block_probe->data(), n_left, dim);
    routed_queries->resize(static_cast<std::size_t>(s.max_routed * dim), stream);
    st.host_rows.reset();
    st.probe_rows_on_host = false;
  }

  void prepare()
  {
    route_block();
    if (!building_for_pairs) { seed_accumulator(); }
    // Staging runs on its own stream so the next needed chunk's H2D overlaps this one's
    // compute, exactly as in the exhaustive fold.
    if (_op._corpus->is_streaming() && !building_for_pairs) { staging_stream.emplace(); }
    stage_on = staging_stream ? staging_stream->view() : stream;
    plan_search();
  }

  /// Each probe row's clusters, sorted by cluster, and the corpus chunks they need.
  void route_block()
  {
    _op.ensure_cluster_index(*mem_space, stream, mr);
    phase("cluster index");

    // Lists of unit rows answer a cosine join as L2 over unit vectors, |q - x|^2 = 2 (1 - cos):
    // the probe rows are normalized too, the radius doubled, and the distances halved at the end.
    // Routing below keeps the join's metric; it only picks clusters.
    s.unit_cosine = _lists != nullptr && _lists->unit_rows;
    if (s.unit_cosine && _request.metric != "cosine") {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] cluster lists built with metric => 'cosine' hold "
        "unit rows and answer only cosine joins; rebuild them without it for an " +
        _request.metric + " join");
    }
    s.n_probes = std::clamp<std::int64_t>(_request.n_probes, 1, _op._n_clusters);
    // Every cluster probed is the exact join through lists. A cosine join over lists of rows as
    // they are (not unit rows) then ranks as the exhaustive fold does: the probe and each chunk
    // normalized in FP32, L2 from the differences between them, halved at the end. No coded
    // search takes such lists, so only FP32 slices are affected: FLOAT16 lists stage theirs from
    // the kept FP32 rows (exact_lists_corpus).
    s.direct_cosine =
      !s.unit_cosine && _request.metric == "cosine" && s.n_probes >= _op._n_clusters;
    if (s.unit_cosine || s.direct_cosine) { st.radius_scale = 0.5f; }
    s.search_metric = s.unit_cosine ? cuvs::distance::DistanceType::L2Expanded : metric;
    s.search_radius = s.unit_cosine || s.direct_cosine ? 2.f * radius_eps : radius_eps;
    if (s.unit_cosine || s.direct_cosine) {
      s.unit_probe.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
      vss::normalize_rows(st.queries.data_handle(), n_left, dim, s.unit_probe->data(), stream);
    }
    s.search_queries =
      s.unit_cosine || s.direct_cosine
        ? raft::make_device_matrix_view<const float, int64_t>(s.unit_probe->data(), n_left, dim)
        : st.queries;

    // Rows that a plain FP32 search ranks (FP32 lists, or rows the coded searches below do not
    // take) then rank as the exhaustive fold's do, L2 from the differences; the coded searches
    // keep their own FP32 re-score.
    s.ranked_metric = [&] {
      if (s.n_probes < _op._n_clusters) { return s.search_metric; }
      if (s.direct_cosine) { return cuvs::distance::DistanceType::L2Unexpanded; }
      if (s.search_metric == cuvs::distance::DistanceType::L2Expanded) {
        return cuvs::distance::DistanceType::L2Unexpanded;
      }
      if (s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        return cuvs::distance::DistanceType::L2SqrtUnexpanded;
      }
      return s.search_metric;
    }();

    // Each probe row picks its own n_probes nearest centroids, the way an IVF index routes a
    // query. Routing a whole probe cluster through its centroid's nearest centroids kept a
    // run contiguous, but a row at the edge of its cluster has its neighbours next door, and
    // that cost about 3x the probes at equal recall. Contiguity comes back below by sorting
    // the (row, cluster) edges by cluster instead.
    auto const n_edges = n_left * s.n_probes;  // row-major [n_left x n_probes]
    auto edge_clusters = [&] {
      if (!st.probe_rows_on_host) {
        return vss::nearest_centroid_ids(
          res, st.queries, _op._centroids->view(), dim, s.n_probes, metric, stream, mr);
      }
      rmm::device_uvector<std::int32_t> all(
        static_cast<std::size_t>(vss::column_size(n_edges, "vector join routing")), stream, mr);
      for_each_host_piece([&](float const* rows, std::int64_t at, std::int64_t n) {
        auto const part = vss::nearest_centroid_ids(
          res,
          raft::make_device_matrix_view<float const, std::int64_t>(rows, n, dim),
          _op._centroids->view(),
          dim,
          s.n_probes,
          metric,
          stream,
          mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(all.data() + at * s.n_probes,
                                      part.data(),
                                      part.size() * sizeof(std::int32_t),
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        stream.sync();
      });
      return all;
    }();
    phase("assign probe->clusters");

    // A row whose clusters cannot supply k is refused before a single search is issued, rather
    // than after the merge has silently padded its answer out with misses.
    auto edge_keys = vss::key_routed_edges(
      std::move(edge_clusters), n_left, s.n_probes, _op._cluster_rows, k_join, stream, mr);
    if (edge_keys.refused_row >= 0) {
      auto const r            = edge_keys.refused_row;
      std::int64_t candidates = 0;
      for (auto const c : edge_keys.refused_clusters) {
        if (c < 0 || c >= _op._n_clusters) {
          throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                                   std::to_string(r) + " was routed to cluster id " +
                                   std::to_string(c) + ", outside the clustering's " +
                                   std::to_string(_op._n_clusters) + " clusters");
        }
        candidates += _op._cluster_rows[static_cast<std::size_t>(c)];
      }
      if (candidates == 0) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                                 std::to_string(r) + " reached no non-empty corpus cluster");
      }
      throw std::runtime_error("[sirius_physical_vector_join_stream] probe row " +
                               std::to_string(r) + " reaches only " + std::to_string(candidates) +
                               " corpus rows, fewer than k=" + std::to_string(k_join) +
                               "; raise n_probes or cluster with fewer, larger clusters");
    }
    s.nearest_cluster = std::move(edge_keys.nearest_cluster);
    phase("edges to host");

    // Sorted by cluster, each cluster's rows are one contiguous range of the edge list. That
    // range is what a slice's search gathers its queries from and what its answer folds back
    // through. Within a cluster, the rows for which it is the nearest cluster come first: the
    // bounded search below seeds every row from its nearest cluster before it searches any of the
    // others. Rows then run in probe order, which keeps a tile's probe-row loads close together.
    s.routed.emplace(vss::sort_routed_edges(std::move(edge_keys), stream, mr));
    auto const& edge_begin    = s.routed->edge_begin;
    auto const& nearest_edges = s.routed->nearest_edges;
    std::vector<bool> wanted(static_cast<std::size_t>(_op._n_clusters));
    for (std::size_t c = 0; c < wanted.size(); ++c) {
      wanted[c] = edge_begin[c + 1] > edge_begin[c] && _op._cluster_rows[c] > 0;
    }
    phase("sort edges");

    // The prune, made explicit: a chunk holding no cluster any row wants is never staged, so
    // an out-of-core clustered corpus pays no transfer for the part it skips.
    for (std::size_t j = 0; j < _op._chunk_cluster_runs.size(); ++j) {
      for (auto const& slice : _op._chunk_cluster_runs[j]) {
        if (wanted[static_cast<std::size_t>(slice.cluster)]) {
          s.needed_chunks.push_back(j);
          break;
        }
      }
    }

    // The per-slice query matrix, sized for the busiest cluster and reused by every slice: the
    // slices are issued in stream order, so each one's gather waits for the last one's search.
    for (std::size_t c = 0; c < static_cast<std::size_t>(_op._n_clusters); ++c) {
      s.max_routed = std::max(s.max_routed, edge_begin[c + 1] - edge_begin[c]);
    }
    // Over host rows only sweep 0 gathers FP32 rows, each row's nearest cluster: the filter reads
    // the FP16 copy.
    s.max_nearest = nearest_edges.empty()
                      ? std::int64_t{0}
                      : *std::max_element(nearest_edges.begin(), nearest_edges.end());
    if (!building_for_pairs) {
      routed_queries.emplace(
        static_cast<std::size_t>((st.probe_rows_on_host ? s.max_nearest : s.max_routed) * dim),
        stream,
        mr);
    }
    s.routed_rows = reinterpret_cast<std::int64_t const*>(s.routed->rows.data());
  }

  void seed_accumulator()
  {
    // A miss -- id -1 at infinite distance -- is what the accumulator is seeded with, so a row's
    // first answer merges against nothing. It can only survive the fold when a row had fewer
    // than k real candidates, which was refused above.
    cudf::numeric_scalar<std::int64_t> const miss_id(-1, true, stream);
    cudf::numeric_scalar<float> const miss_distance(
      std::numeric_limits<float>::infinity(), true, stream);

    // Running [n_left x k_join] accumulator in the caller's row order, seeded with misses so
    // a row's first answer merges against nothing. Each slice folds its answer in through the
    // rows it served, so device memory does not grow with n_probes: a row's answers from its
    // n_probes clusters never coexist.
    {
      auto const total = vss::column_size(n_left * k_join, "vector join output");
      st.acc_neighbors = cudf::make_column_from_scalar(miss_id, total, stream, mr);
      st.acc_distances = cudf::make_column_from_scalar(miss_distance, total, stream, mr);
    }
  }

  /// What each task over a block keeps at hand from its plan.
  void keep_from_plan()
  {
    // The FLOAT16 rows' largest rounding error and norm, which the filter limit allows for: the
    // lists' own, or for a scanned corpus the staged chunk's, set as each chunk is coded.
    row_half_error = _lists != nullptr ? _lists->half_error : 0.f;
    row_norm_max   = _lists != nullptr ? _lists->max_row_norm : 0.f;
    // Where the FP32 re-score reads a layout row: the lists' kept rows, or for a scanned corpus the
    // staged chunk, pointed at as each chunk is coded (its candidates are flushed before it goes).
    exact_source = s.rescore && _lists != nullptr ? _lists->exact_source() : vss::exact_rows{};
  }

  /// Settles the search's form, from the join, the lists and the probe's rows, coding the probe on
  /// the way.
  void plan_search()
  {
    auto const& nearest_edges     = s.routed->nearest_edges;
    bool const bound_gemm_enabled = [] {
      auto const* v = std::getenv("SIRIUS_VSS_BOUND_GEMM");
      return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    // Byte-valued lists, searched by a byte-valued probe side, run on the int8 tensor cores
    // straight from the stored bytes: no chunk is widened, the row norms were
    // computed at build time, and every dot product is exact. Only the probe is converted, once.
    // A radius join has only the bounded form of that search.
    bool const l2 = s.search_metric == cuvs::distance::DistanceType::L2Expanded ||
                    s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded;
    // INT8 codes are only ever answered by the bounded search below, which filters with them and
    // re-scores what passes in FP32; decoded on their own they would give approximate distances.
    s.codes_int8 = _lists != nullptr && _lists->encoding == vss::list_encoding::int8;
    if (s.codes_int8 &&
        !(gemm_search_enabled() && bound_gemm_enabled && l2 && _lists->has_exact_rows() &&
          _lists->row_sq != nullptr && vss::bound_filter_int8_supports(dim))) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] int8 cluster lists answer only l2 joins through the "
        "bounded search (vector width a multiple of 16, at most 256, and the FP32 rows kept); "
        "rebuild the lists with storage => 'float16' or 'float32'");
    }
    s.int8_search =
      gemm_search_enabled() && _lists != nullptr &&
      (!radius_join || (bound_gemm_enabled && vss::bound_filter_int8_supports(dim))) &&
      (_lists->encoding == vss::list_encoding::uint8 || s.codes_int8) &&
      _lists->row_sq != nullptr && l2 && dim % 4 == 0;
    code_probe_int8();

    // Bounded search: sweep 0 searches each row's nearest cluster and folds as usual, which gives
    // every row a k-th distance; sweep 1 searches the remaining clusters with a GEMM that keeps
    // only pairs under that bound, so the scores of the other n_probes - 1 clusters never reach
    // memory. Distances stay squared until the end, where the bound is compared.
    // A radius join is the same search with the radius as every row's bound, in one sweep: there
    // is no k-th distance to seed, and a pair under the bound is final when found.
    // Lists on the device can seed sweep 0 (below), which makes even one probe a bounded search.
    // SIRIUS_VSS_SEED=0 turns the seed off (sweep 0 then ranks each row's nearest cluster in full);
    // SIRIUS_VSS_SEED=force seeds INT8 lists on the device past the partition-size test below.
    std::string_view const seed_switch = [] {
      auto const* v = std::getenv("SIRIUS_VSS_SEED");
      return v == nullptr ? std::string_view{} : std::string_view{v};
    }();
    bool const seed_enabled = seed_switch != "0";
    bool const seed_forced  = seed_switch == "force";
    // A seeded sweep 0 is each row's exact join with its nearest cluster, filtered in stretches
    // that double from the seed's sample and merged in between (stretch_sweep0, below), so its cost
    // grows with the number of stretches -- the log of the partition -- while the unseeded pass
    // ranks the whole partition on CUDA cores, linear in it. FLOAT16 lists take the seed at every
    // partition size measured (Deep10M, 1M and 10k probes, ~2.4k-19.5k rows a partition: 0.69-0.92x
    // the unseeded pass). INT8 codes pass about 2k pairs a row a stretch (FLOAT16 about 1k), so for
    // INT8 the seed pays only once a partition is large: per 1M probes 1.04x the unseeded pass at
    // ~2.4k rows a partition, 0.99x at ~4.9k, 0.82x at ~9.8k, 0.84x at ~19.5k (10-30
    // nearest-as-join, RTX A5000). INT8 lists averaging under kSeedInt8Rows rows a partition keep
    // the unseeded pass. Lossless UINT8 rows seed exactly at any size.
    constexpr std::int64_t kSeedInt8Rows = 5'000;
    bool const large_partitions =
      _lists != nullptr && _lists->offsets.size() > 1 &&
      _lists->offsets.back() >=
        static_cast<std::int64_t>(_lists->offsets.size() - 1) * kSeedInt8Rows;
    // A single list -- the exact join's compact layout, no partitioning -- lowers each row's bound
    // as its bounded sweep goes: every slice is cut into pieces whose merge points double from the
    // first piece, and what passes is merged before the next, so after the first pieces the filter
    // runs under a bound near the final one (Deep10M INT8, 10k probes: 0.33 s, against 0.37 s
    // through its best partitioning). On the device the seed's sample bounds the first piece; host
    // lists, with no device rows to sample, start from an unbounded first piece instead. A seeded
    // first piece is cut to the batch, so that what it lets through, about k per sample's worth,
    // stays near 2^24 pairs. All of an unbounded piece's pairs pass, so it is only as long as a
    // first k-th needs: 4k rows, at least 64. A longer one only holds more: the time does not move
    // from 1k to 1M probes. Later pieces pass about k a row.
    // SIRIUS_VSS_RUNNING_BOUND=<rows> sets the first piece for any lists, 0 turns it off.
    // A single list of FLOAT16 rows takes no sample on the device either: its codes give each row's
    // bound after the first piece (see hold_codes_pairs, below). Exact, with no FP32 read for the
    // bound. INT8 and byte codes keep the seed: measured slower without it, at every probe count.
    // The first piece's pairs all pass, so a device too full for their buffer keeps the seed too.
    // SIRIUS_VSS_F16_NO_SEED=0 keeps the seed here too, and setting SIRIUS_VSS_SEED or
    // SIRIUS_VSS_RUNNING_BOUND asks for the schedule they name.
    auto const unbounded_piece_rows = [&] { return std::max<std::int64_t>(4 * k_join, 64); };
    s.f16_list_no_seed =
      _lists != nullptr && _lists->offsets.size() == 2 && !s.int8_search &&
      _lists->encoding == vss::list_encoding::float16 && !radius_join && !ip &&
      st.earlier_cut == nullptr && seed_switch.empty() &&
      std::getenv("SIRIUS_VSS_RUNNING_BOUND") == nullptr &&
      static_cast<std::size_t>(n_left * unbounded_piece_rows()) * 20 + (std::size_t{1} << 28) <=
        mem_space->get_available_memory(stream) &&
      [] {
        auto const* v = std::getenv("SIRIUS_VSS_F16_NO_SEED");
        return v == nullptr || std::string_view{v} != "0";
      }();
    s.running_bound_rows = [&] {
      if (auto const* v = std::getenv("SIRIUS_VSS_RUNNING_BOUND"); v != nullptr) {
        return std::max<std::int64_t>(0, std::atoll(v));
      }
      if (_lists == nullptr || _lists->offsets.size() != 2) { return std::int64_t{0}; }
      constexpr std::int64_t kFirstPiecePairs = std::int64_t{1} << 24;
      auto const rows                         = std::max<std::int64_t>(n_left, 1);
      auto const at_least                     = std::max<std::int64_t>(4 * k_join, 64);
      // The seed is the k-th of max(kSeedSample, 8k) sample rows (seed_rows), so about k / that of
      // a piece's rows pass it.
      auto const sample = std::max<std::int64_t>(vss::kSeedSample, 8 * k_join);
      // The upper bounds give way to at_least for a large k (std::clamp needs lo <= hi).
      return _lists->tier == cucascade::memory::Tier::GPU && !s.f16_list_no_seed
               ? std::clamp<std::int64_t>(
                   kFirstPiecePairs / rows * sample / std::max<std::int64_t>(k_join, 1),
                   at_least,
                   std::max<std::int64_t>(at_least, 10'000))
               : unbounded_piece_rows();
    }();
    // Lists in host memory have no device rows to sample; with the bound lowered as it goes, their
    // sweep starts from an unbounded first piece instead, which serves as the sample.
    // The seed's sample grows with k (seed_rows), so any k is seeded; unseeded, the sweep would
    // rank every row first, a full GEMM and selection.
    bool const seedable =
      s.int8_search && !radius_join &&
      (_lists->tier == cucascade::memory::Tier::GPU || s.running_bound_rows > 0) && seed_enabled &&
      (!s.codes_int8 || large_partitions || s.running_bound_rows > 0 || seed_forced);
    s.bounded = s.int8_search && (s.n_probes > 1 || radius_join || s.codes_int8 || seedable) &&
                vss::bound_filter_int8_supports(dim) && bound_gemm_enabled;
    // FLOAT16 lists that kept their FP32 rows get the same two sweeps with the half-precision
    // GEMM, and an exact answer: sweep 0's nearest-cluster answers are re-scored in FP32 only to
    // give each row a bound (their largest distance) and then dropped, and sweep 1 searches every
    // routed cluster, nearest one included, keeping pairs whose FP16 distance is within the bound
    // plus the FP16 rounding slack. What passes is re-scored in FP32 before it is merged, and ids
    // stay layout rows until the end, where the re-scoring needs them.
    // A scanned corpus has no lists, but every chunk it stages can be coded on the device and
    // searched the same way: sweep 0 ranks each row's nearest cluster in FP32 and re-scores that
    // answer exactly, so its largest distance bounds the row; sweep 1 codes each staged chunk to
    // FLOAT16 and filters every routed cluster under the bound, re-scoring what passes from the
    // chunk's FP32 rows before the chunk is released. Deep10M self-join (k 10, 32 probes) spent 82
    // % of its GPU time in the FP32 GEMM and the top-k selection of the ranked search this
    // replaces. With one probe sweep 0 is the whole search already, and sweep 1 would only repeat
    // it.
    bool const scan_f16_shape = _lists == nullptr && !s.int8_search && gemm_search_enabled() &&
                                bound_gemm_enabled && l2 && !radius_join &&
                                st.earlier_cut == nullptr && s.n_probes > 1 &&
                                vss::bound_filter_f16_supports(dim) && scan_bounded_enabled();
    bool const f16_lists_shape =
      !s.int8_search && gemm_search_enabled() && _lists != nullptr &&
      _lists->encoding == vss::list_encoding::float16 && _lists->has_exact_rows() &&
      (s.search_metric == cuvs::distance::DistanceType::L2Expanded ||
       s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded || ip) &&
      vss::bound_filter_f16_supports(dim) && bound_gemm_enabled;
    // The probe is rounded like the rows, with its rounded norms and its own rounding error. A
    // probe row past half's range (a component of 65520 or more becomes infinite) would make its
    // pairs' filter values infinite or NaN, which no limit admits: its true pairs would be dropped
    // with no error. So the probe rows are held to the rows' own check (kmeans_functions.cpp: a
    // norm under 65504), and a batch with one outside it is searched in FP32 instead: a scanned
    // corpus from its staged FP32 chunks, FLOAT16 lists from their kept FP32 rows.
    bool probe_in_f16_range = true;
    if (!scan_f16_shape) { block_rows_to_device(); }
    if (scan_f16_shape || f16_lists_shape) { probe_in_f16_range = code_probe_f16(); }
    s.scan_f16    = scan_f16_shape && probe_in_f16_range;
    s.f16_bounded = s.scan_f16 || (f16_lists_shape && probe_in_f16_range);
    // FLOAT16 lists the bounded search does not answer (a cosine join over rows as they are, a
    // probe row past half's range, the bounded search switched off): their chunks are staged from
    // the kept FP32 rows, in layout order, so the FP32 search scores what the rows hold. Over the
    // half rows it would be exact for the rounded rows, and report their distances.
    if (_lists != nullptr && _lists->encoding == vss::list_encoding::float16 && !s.f16_bounded) {
      if (!_lists->has_exact_rows()) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream] these float16 cluster lists kept no FP32 rows, "
          "which a " +
          _request.metric +
          " join outside the bounded search reads; rebuild the lists with storage => 'float32'");
      }
      s.exact_lists_corpus.emplace(*_lists, /*exact_rows=*/true);
    }
    s.any_bounded = s.bounded || s.f16_bounded;
    // FP16 and INT8 lists filter with inexact distances: their bounded search keeps layout rows as
    // ids and re-scores every candidate in FP32 from the kept rows. Unseeded, they search the
    // nearest cluster again in sweep 1 (sweep 0 only sets the bound). INT8 filters in code space,
    // under a per-row limit derived from the FP32 bound and both sides' coding errors.
    s.rescore = s.f16_bounded || s.codes_int8;
    // A radius join whose score is never read keeps the filter's verdict where it already proves
    // the pair is inside the radius, and re-scores only the rest.
    s.skip_certain = radius_join && s.rescore && !_request.score_read && !best_only;
    keep_from_plan();
    s.sqrt_at_end =
      s.any_bounded && s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded;
    // Lists on the device run sweep 0 -- each row's nearest cluster, which only sets the bounded
    // sweep's starting bound -- as a bounded search too, a few grouped launches instead of a GEMM
    // and a selection per cluster. Its own bound comes from the first kSeedSample rows of that
    // cluster: their k-th smallest distance, which for INT8 codes and FLOAT16 rows is first raised
    // by both sides' coding or rounding error so it bounds the FP32 distance. What passes is merged
    // (re-scored in FP32 where the rows are inexact), so sweep 1 starts from the nearest cluster's
    // exact k-th and need not search that cluster again. SIRIUS_VSS_SEED=0 keeps the GEMM.
    // The seed is CUDA-core work per probe row (a warp per row against a quarter of its nearest
    // cluster, at most kSeedSample rows); ranking each row's nearest cluster on the tensor cores
    // instead (gemm_f16_topk below, as host lists do) costs a GEMM and a selection per (task,
    // cluster) slice plus a second search of that cluster in sweep 1. So the seed pays while the
    // slices carry little seed work -- few probe rows a cluster, as in a small probe batch -- and
    // the ranked pass once they carry much. A task over clustered FLOAT16 lists on the device ranks
    // when its mean seed work per slice (nearest-cluster probe rows x sample rows x width) reaches
    // SIRIUS_VSS_SWEEP0_SLICE_MACS. SIRIUS_VSS_SWEEP0=seed | ranked forces either;
    // SIRIUS_VSS_SEED=0 still means the grouped CUDA-core ranking (nearest_cluster_topk).
    s.ranked_sweep0 = false;
    if (s.f16_bounded && _lists != nullptr && _lists->tier == cucascade::memory::Tier::GPU &&
        _lists->offsets.size() > 2 && s.running_bound_rows == 0 && !radius_join && !ip &&
        seed_enabled && s.probe_f16) {
      auto const* sw = std::getenv("SIRIUS_VSS_SWEEP0");
      std::string_view const how{sw == nullptr ? "auto" : sw};
      auto const floor    = std::max<std::int64_t>(64, 8 * k_join);
      double seed_macs    = 0;
      std::int64_t slices = 0;
      for (std::size_t c = 0; c + 1 < _lists->offsets.size(); ++c) {
        auto const rows = nearest_edges[c];
        if (rows == 0) { continue; }
        auto const size   = _lists->offsets[c + 1] - _lists->offsets[c];
        auto const sample = std::min<std::int64_t>(
          size,
          std::clamp<std::int64_t>(
            size / 4, floor, std::max<std::int64_t>(floor, vss::kSeedSample)));
        seed_macs += static_cast<double>(rows) * static_cast<double>(sample) * dim;
        ++slices;
      }
      auto const* t          = std::getenv("SIRIUS_VSS_SWEEP0_SLICE_MACS");
      double const threshold = t != nullptr ? std::atof(t) : kSweep0RankedSliceMacs;
      s.ranked_sweep0 = how == "ranked" || (how != "seed" && slices > 0 &&
                                            seed_macs >= threshold * static_cast<double>(slices));
      if (dbg) {
        std::fprintf(stderr,
                     "[vecjoin-phase] sweep-0 rule: %lld probe rows over %lld nearest slices, "
                     "seed work %.3g MACs a slice (ranked from %.3g, SIRIUS_VSS_SWEEP0=%s): %s\n",
                     static_cast<long long>(n_left),
                     static_cast<long long>(slices),
                     slices > 0 ? seed_macs / static_cast<double>(slices) : 0.0,
                     threshold,
                     std::string(how).c_str(),
                     s.ranked_sweep0 ? "ranked" : "seed");
      }
    }
    bool const seed_f16 =
      s.f16_bounded && _lists != nullptr && !radius_join &&
      (_lists->tier == cucascade::memory::Tier::GPU || s.running_bound_rows > 0) && seed_enabled &&
      !s.ranked_sweep0;
    s.seed = (s.bounded && seedable) || seed_f16;
    // Unseeded, sweep 0 ranks each row's nearest cluster on the half-precision tensor cores from
    // the stored rows; only the bound it leaves is used.
    // SIRIUS_VSS_F16_GEMM=0 widens the rows and ranks them with the FP32 GEMM instead.
    // An inner-product join keeps to the searches it was verified on: the seed, the bounded
    // filter, and the FP32 GEMM ranking of an unseeded sweep 0.
    s.f16_sweep0 = s.f16_bounded && !s.scan_f16 && !s.seed && !ip && [] {
      auto const* v = std::getenv("SIRIUS_VSS_F16_GEMM");
      return v == nullptr || std::string_view{v} != "0";
    }();
    // That bound comes from the codes alone, with no FP32 read, across PCIe when the rows are in
    // host memory. Row r's k pairs of sweep 0 are distinct, and the largest of their codes is
    // c_(k), so all k have c(r, s) <= c_(k), hence delta(r, s) <= U_r(c_(k)) (the lemma above
    // gap_trim), which therefore bounds delta*_r and may stand as b_r wherever their re-scored k-th
    // did. It is never below that k-th, so what a re-scored bound keeps it keeps too. U_r holds for
    // sweep 0's codes as for the filter's, in all of its kernels, with N = |q^| + max|x^|:
    //  1. A product of two halves has at most 22 significant bits and is at least 2^-48: exact in
    //     FP32, with no denormal.
    //  2. Any summation tree over d terms has depth at most d - 1, so a dot's error is at most
    //     1.001 d u |q^| |x^|, u = 2^-24, whatever the order.
    //  3. c doubles that, adds the two FP32 norms' errors and the last additions: at most
    //     (d/2 + d/32 + 8) u N^2 with 4 |q^| |x^| <= N^2, inside eta_r = 2 d u N^2. Clamping at 0
    //     only raises c. So |q^ - x^|^2 <= c + eta_r, with room left of about 1.47 d u N^2.
    //  4. |q - x| <= |q^ - x^| + e_r + e by the triangle inequality.
    //  5. The re-score is at most |q - x|^2 (1 + (d/32 + 7) u), one lane's FMA chain, 5 shuffle
    //     additions and the subtraction, which 2^-20 and the room in (3) cover for d >= 16.
    // A row with a miss in sweep 0 has an infinite c_(k), which stays infinite.
    // SIRIUS_VSS_SWEEP0_CODE_BOUND=0 re-scores the k pairs instead.
    s.sweep0_bound_from_codes = s.f16_sweep0 && !radius_join && [] {
      auto const* v = std::getenv("SIRIUS_VSS_SWEEP0_CODE_BOUND");
      return v == nullptr || std::string_view{v} != "0";
    }();
    if (s.f16_sweep0 && !building_for_pairs) {
      routed_f16.emplace(static_cast<std::size_t>(s.max_routed * dim), stream, mr);
      routed_sqf.emplace(static_cast<std::size_t>(s.max_routed), stream, mr);
    }
    if (s.f16_bounded) {
      // The probe was rounded above, and is within half's range.
      if (ip) {
        // The filter kernels compute |q|^2 + |x|^2 - 2 q.x from the norms they are handed: with
        // zeros on both sides that is -2 q.x, the inner-product distance, exactly.
        s.ip_probe_zero_sq.emplace(static_cast<std::size_t>(n_left), stream, mr);
        CUDF_CUDA_TRY(cudaMemsetAsync(
          s.ip_probe_zero_sq->data(), 0, s.ip_probe_zero_sq->size() * sizeof(float), stream.get()));
        if (_lists != nullptr && !_lists->offsets.empty()) {
          s.ip_row_zero_sq.emplace(
            static_cast<std::size_t>(std::max<std::int64_t>(_lists->offsets.back(), 1)),
            stream,
            mr);
          CUDF_CUDA_TRY(cudaMemsetAsync(
            s.ip_row_zero_sq->data(), 0, s.ip_row_zero_sq->size() * sizeof(float), stream.get()));
        }
      }
    }
    // The kernels compare squared distances.
    s.radius_bound = s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded
                       ? s.search_radius * s.search_radius
                       : s.search_radius;
    // Gap-checked re-score (top-k joins over inexact rows only): before a stretch's candidates are
    // re-scored, each row's k-th FP32 distance is bounded from its k best codes and its running
    // top-k, and the candidates past the filter limit of that bound -- which cannot be in the row's
    // top-k -- are dropped without reading their FP32 rows. SIRIUS_VSS_GAP_RESCORE=0 re-scores
    // every candidate; SIRIUS_VSS_GAP_RESCORE_STATS=1 counts what it trims (also under the phase
    // debug).
    s.gap_rescore = s.rescore && !radius_join && [] {
      auto const* v = std::getenv("SIRIUS_VSS_GAP_RESCORE");
      return v == nullptr || std::string_view{v} != "0";
    }();
    s.gap_stats_on = s.gap_rescore && (dbg || [] {
                       auto const* v = std::getenv("SIRIUS_VSS_GAP_RESCORE_STATS");
                       return v != nullptr && std::string_view{v} == "1";
                     }());
    // With the lists on the device every slice's rows stay put, so a group's slices of one probe
    // tile at most are launched together, a few grouped launches in place of one per slice. A
    // Host-tier lists are staged a chunk at a time and released as the sweep moves on, so there
    // each slice launches at once.
    // SIRIUS_VSS_GROUP=0 launches each slice on its own.
    // A scanned chunk stays staged until its group is checked, so its slices group too.
    // With the CUTLASS tile kernel, lists staged from the host group as well: a chunk's slices are
    // launched together before the next chunk's copy starts, and checked before the chunk goes.
    bool const cutlass_tiles = s.f16_bounded && vss::bound_filter_f16_uses_cutlass(dim);
    s.group_launch =
      (s.scan_f16 ||
       (_lists != nullptr && (_lists->tier == cucascade::memory::Tier::GPU || cutlass_tiles))) &&
      [] {
        auto const* v = std::getenv("SIRIUS_VSS_GROUP");
        return v == nullptr || std::string_view{v} != "0";
      }();
    s.bounded_group = s.group_launch ? 256 : 64;
    // Which slices join the grouped launch. Narrow slices (<= 128 probe rows) always do. A wide
    // slice launched alone keeps its corpus tiles in L2 across its probe tiles, which pays when the
    // slice's GEMM is large and costs a launch when it is small; measured 2026-10-05: SIFT1M 10k x
    // 16 lists (156 probes x 977 rows per slice) went 15 -> 8 ms grouped, SIFT100M (98k-row lists)
    // and a 1M-probe self-join over 64 lists (250k probes x 15k rows) lost 3-13 % grouped. So a
    // wide slice joins the group when probes x rows is at most 4M. SIRIUS_VSS_GROUP_M_MAX and
    // SIRIUS_VSS_GROUP_WORK_MAX override. The CUTLASS kernel's grouped launch keeps a slice's rows
    // in L2 itself while they fit, so such a slice joins at any size; past that it reads them again
    // for every probe tile. 1M rows, 256-1,536 wide: slices up to half the L2 ran 0.98-1.03x their
    // lone launches, from 4 MB (6 MB L2) alone won by 1-15 %.
    auto const env_i64 = [](char const* name, std::int64_t dflt) {
      auto const* v = std::getenv(name);
      return v == nullptr ? dflt : std::max<std::int64_t>(0, std::atoll(v));
    };
    s.group_m_max      = env_i64("SIRIUS_VSS_GROUP_M_MAX", 128);
    s.group_work_max   = env_i64("SIRIUS_VSS_GROUP_WORK_MAX", 4'000'000);
    s.group_slice_rows = cutlass_tiles ? vss::bound_filter_f16_group_slice_bytes() / (dim * 2) : 0;
    s.hold_codes_pairs = s.f16_list_no_seed && s.gap_rescore && s.running_bound_rows > 0;
    // A top-k join's pairs wait in the buffer for their FP32 re-score, and a flush needs far more
    // than the buffer holds: the re-score's and the merge's working space, about 64 B a pair beside
    // the buffer's 16. Next to device lists that leave little free, a full buffer cannot flush, and
    // on a corpus stored in cluster order, where a slice's pairs can themselves overflow the
    // buffer, the task runs out of memory on every retry. So when a flush would not fit, the
    // buffer's pairs move to pinned host blocks with no FP32 read and the sweep goes on; they come
    // back at the end of the task in pieces small enough to flush. A slice that alone would need a
    // buffer past what the device has is cut in two instead.
    // Exactness, in the notation of the lemma above gap_trim. A moved pair is not in the
    // accumulator and keeps its code. moved_codes holds row r's k smallest codes among all moved
    // pairs, those trimmed since included. They are k distinct pairs, as a row visits a cluster
    // once and a rollback discards what was not committed, each with delta <= U_r(c_(k)), so
    // min(b_r, U_r(c_(k))) >= delta*_r may stand as b_r, whether or not earlier flushes filled the
    // accumulator: those pairs are not in moved_codes, and they only lower b_r. A pair past its
    // limit has delta > b_r >= delta*_r, strictly past the k-th, and is dropped. One float a row
    // would not do: j pairs of one move could not count with k - j of a later one, the buffer being
    // empty between them. Every pair kept goes through flush as any other does, whose merge takes
    // the k smallest, so the answer is what flushing the whole buffer gives, up to which of several
    // pairs tied at exactly the k-th it keeps.
    // What waits on the host can reach probes x rows x 16 B when limits drop nothing, as in a large
    // group of duplicates, so the blocks have a budget, shared by the join's tasks. Past it the
    // pairs flush where they are if that fits, which the argument above allows: they stay among
    // moved_codes' distinct pairs and leave the buffer. If it does not fit the query fails, with
    // the numbers, instead of pinning until the process is killed.
    // The shape is an FP16 sweep 1 over lists, which starts from a bound its codes gave and holds
    // no pair yet. SIRIUS_VSS_PAIRS_TO_HOST=0 turns it off. For tests: =force moves after every
    // group and cuts every slice that overflows the buffer, =force_no_flush also takes any flush
    // past the budget not to fit, and SIRIUS_VSS_PAIRS_TO_HOST_BUDGET sets the budget in bytes.
    std::string_view const pairs_to_host_setting = [] {
      auto const* v = std::getenv("SIRIUS_VSS_PAIRS_TO_HOST");
      return v == nullptr ? std::string_view{} : std::string_view{v};
    }();
    s.pairs_to_host        = s.sweep0_bound_from_codes && s.gap_rescore && !keyed && !best_only &&
                             st.earlier_cut == nullptr && s.running_bound_rows == 0 &&
                             pairs_to_host_setting != "0";
    s.budget_flush_refused = pairs_to_host_setting == "force_no_flush";
    s.pairs_to_host_forced =
      s.pairs_to_host && (pairs_to_host_setting == "force" || s.budget_flush_refused);
    // Pairs can wait on the host only on that one path. A scanned chunk, for one, stays staged
    // until its group has been checked, and its candidates are re-scored from those rows. But any
    // slice can be cut, each part flushed before the next: a large group of near duplicates passes
    // hundreds of millions of pairs in one slice, more than the device holds beside a large block.
    // The parts are pieces like the running bound's, disjoint rows with a flush between them, which
    // every bounded search already takes.
    s.cut_large_slices = s.any_bounded && pairs_to_host_setting != "0";
    s.slice_cuts_forced =
      s.cut_large_slices && (pairs_to_host_setting == "force" || s.budget_flush_refused);
    s.pair_budget = s.pairs_to_host ? _op.host_pair_budget() : 0;
    if (dbg) {
      std::fprintf(stderr,
                   "[vecjoin-phase] paths: sweep-0 bound from codes %d, pairs to host %d, single "
                   "list without a seed %d\n",
                   static_cast<int>(s.sweep0_bound_from_codes),
                   static_cast<int>(s.pairs_to_host),
                   static_cast<int>(s.hold_codes_pairs));
    }

    w.running_piece = s.running_bound_rows;

    // Lists that did not fit the device are read by the same searches: a sweep that reads the
    // lists in place copies each needed chunk's stored rows in as they are, still compact, and
    // points the search at the copy. Four rows of slack, as on the device.
    s.lists_on_host = _lists != nullptr && _lists->tier == cucascade::memory::Tier::HOST;
    // The per-row norms the searches read (INT32 for byte rows, FP32 for half rows) are copied in
    // behind the rows, so lists on the host keep none of their size on the device.
    // An inner product's filter reads zeros in their place (see probe_filter_sq).
    s.list_norms = _lists == nullptr  ? nullptr
                   : s.ip_row_zero_sq ? static_cast<void const*>(s.ip_row_zero_sq->data())
                   : _lists->encoding == vss::list_encoding::float16
                     ? (_lists->row_sq_f32 ? _lists->row_sq_f32->data() : nullptr)
                     : (_lists->row_sq ? _lists->row_sq->data() : nullptr);

    // Sweep 0 of a bounded search only reads each row's nearest cluster, so only the chunks
    // holding one; with few probe rows that is a small part of what sweep 1 reads.
    if (s.any_bounded) {
      for (auto const j : s.needed_chunks) {
        for (auto const& slice : _op._chunk_cluster_runs[j]) {
          if (nearest_edges[static_cast<std::size_t>(slice.cluster)] > 0) {
            s.nearest_chunks.push_back(j);
            break;
          }
        }
      }
    }

    // Probe blocks are sized to share the device, so only a lone small one keeps chunks: a ~1M-row
    // task that kept them filled the device and ran out of memory in sweep 1.
    bool const small_lone_block = block != nullptr && _op._blocks.size() == 1 &&
                                  _op.estimate_for_rows(static_cast<std::size_t>(block->rows)) <=
                                    mem_space->get_max_memory() / 8;
    s.keep_chunks               = s.lists_on_host && s.any_bounded && !radius_join &&
                                  (block == nullptr || small_lone_block) && [] {
                      auto const* v = std::getenv("SIRIUS_VSS_KEEP_LIST_CHUNKS");
                      return v == nullptr || std::strcmp(v, "0") != 0;
                    }();

    // A seeded sweep 0 over lists on the device is each row's exact join with its nearest cluster,
    // run as the exact join runs: the sample's seed bounds the first stretch of the cluster (the
    // sample itself), and what passes is merged before the next stretch, so each row's bound falls
    // to the k-th of all the cluster rows filtered so far. Each stretch is as long as all those
    // before it, so about k pairs a row pass each one; under the seed alone a cluster's every
    // sample's worth of rows passes about k (Deep10M at 4,096 clusters, 1M probes: 1.7x (FP16) and
    // 2.3x (INT8) the pairs of the unseeded sweep). Stretch i of every cluster is filtered in one
    // round of grouped launches, and the rounds merge in between. SIRIUS_VSS_SEED_TIGHTEN=0 filters
    // each cluster whole under the seed. Lists in host memory, and the running bound, keep their
    // own schedule.
    s.stretch_sweep0 = s.seed && !s.lists_on_host && s.running_bound_rows == 0 && [] {
      auto const* v = std::getenv("SIRIUS_VSS_SEED_TIGHTEN");
      return v == nullptr || std::string_view{v} != "0";
    }();
  }

  /// The probe in the lists' codes, with its norms; a probe with a non-byte value leaves the search
  /// to the FP32 rows.
  void code_probe_int8()
  {
    if (s.int8_search && !s.codes_int8) {
      rmm::device_uvector<unsigned long long> non_bytes(1, stream, mr);
      CUDF_CUDA_TRY(cudaMemsetAsync(non_bytes.data(), 0, sizeof(unsigned long long), stream.get()));
      vss::count_non_uint8(s.search_queries.data_handle(), n_left * dim, non_bytes.data(), stream);
      unsigned long long non_bytes_host = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(&non_bytes_host,
                                    non_bytes.data(),
                                    sizeof(non_bytes_host),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
      s.int8_search = non_bytes_host == 0;
    }
    if (s.int8_search) {
      s.probe_i8.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
      s.probe_sq.emplace(static_cast<std::size_t>(n_left), stream, mr);
      if (!building_for_pairs) {
        routed_i8.emplace(static_cast<std::size_t>(s.max_routed * dim), stream, mr);
        routed_sq.emplace(static_cast<std::size_t>(s.max_routed), stream, mr);
      }
      if (s.codes_int8) {
        // The probe gets the lists' codes too, and its own exact coding error per row -- rotated
        // first, like the rows, when the lists were (the FP32 re-score keeps the unrotated probe).
        s.probe_code_error.emplace(static_cast<std::size_t>(n_left), stream, mr);
        if (_lists->code_rotation) {
          s.rotated_probe.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
          vss::rotate_rows(s.search_queries.data_handle(),
                           n_left,
                           dim,
                           static_cast<float const*>(_lists->code_rotation->data()),
                           s.rotated_probe->data(),
                           stream);
        }
        vss::quantize_rows_int8(
          s.rotated_probe ? s.rotated_probe->data() : s.search_queries.data_handle(),
          n_left,
          dim,
          static_cast<float const*>(_lists->code_offset->data()),
          _lists->code_scale,
          s.probe_i8->data(),
          s.probe_code_error->data(),
          nullptr,
          stream,
          _lists->code_rotation ? vss::rotation_slack(dim) : 0.f);
      } else {
        vss::narrow_to_shifted_int8(
          s.search_queries.data_handle(), n_left * dim, s.probe_i8->data(), stream);
      }
      vss::int8_row_sq_norms(s.probe_i8->data(), n_left, dim, s.probe_sq->data(), stream);
    }
  }

  /// The probe rounded to FP16, with its norms and rounding error. False, with its FP32 rows on the
  /// device, when a row is outside FP16's range.
  [[nodiscard]] bool code_probe_f16()
  {
    s.probe_f16.emplace(static_cast<std::size_t>(n_left * dim), stream, mr);
    s.probe_sqf.emplace(static_cast<std::size_t>(n_left), stream, mr);
    s.probe_half_error.emplace(static_cast<std::size_t>(n_left), stream, mr);
    rmm::device_uvector<unsigned int> max_sq_bits(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(max_sq_bits.data(), 0, sizeof(unsigned int), stream.get()));
    if (st.probe_rows_on_host) {
      for_each_host_piece([&](float const* rows, std::int64_t at, std::int64_t n) {
        vss::narrow_to_float16(rows, n * dim, s.probe_f16->data() + at * dim, stream);
        vss::half_rows_norms(rows,
                             s.probe_f16->data() + at * dim,
                             n,
                             dim,
                             s.probe_sqf->data() + at,
                             s.probe_half_error->data() + at,
                             nullptr,
                             max_sq_bits.data(),
                             stream);
      });
    } else {
      vss::narrow_to_float16(
        s.search_queries.data_handle(), n_left * dim, s.probe_f16->data(), stream);
      vss::half_rows_norms(s.search_queries.data_handle(),
                           s.probe_f16->data(),
                           n_left,
                           dim,
                           s.probe_sqf->data(),
                           s.probe_half_error->data(),
                           nullptr,
                           max_sq_bits.data(),
                           stream);
    }
    unsigned int bits = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &bits, max_sq_bits.data(), sizeof(bits), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    float max_sq = 0;
    std::memcpy(&max_sq, &bits, sizeof(max_sq));
    float const probe_norm_max = std::sqrt(max_sq);
    bool const in_range        = probe_norm_max < 65504.f;
    if (!in_range) {
      SIRIUS_LOG_INFO(
        "[sirius_physical_vector_join_stream] a probe row norm of {} is outside float16's "
        "range; searching this batch of {} rows in FP32",
        probe_norm_max,
        n_left);
      s.probe_f16.reset();
      s.probe_sqf.reset();
      s.probe_half_error.reset();
      block_rows_to_device();
    }
    return in_range;
  }

  void code_limits(float const* bound_rows, float* limit, bool lower)
  {
    if (s.codes_int8) {
      vss::int8_code_limit(bound_rows,
                           s.probe_code_error->data(),
                           _lists->code_error,
                           _lists->code_scale,
                           n_left,
                           limit,
                           stream,
                           lower);
    } else if (s.f16_bounded) {
      vss::float16_bound_limit(bound_rows,
                               s.probe_sqf->data(),
                               s.probe_half_error->data(),
                               row_half_error,
                               row_norm_max,
                               dim,
                               n_left,
                               limit,
                               stream,
                               lower,
                               ip);
    }
  }

  void refresh_code_limit(float const* bound_rows)
  { code_limits(bound_rows, s.code_limit->data(), false); }

  // The probe norms the filter kernels read: zeros for an inner product (see above).
  [[nodiscard]] float const* probe_filter_sq()
  { return s.ip_probe_zero_sq ? s.ip_probe_zero_sq->data() : s.probe_sqf->data(); }

  // The terms the filter limit widens by (code_limits), turned around: a code distance as an
  // FP32 bound, exactly as the seeds raise theirs.
  [[nodiscard]] vss::code_bound_terms code_terms()
  {
    vss::code_bound_terms t{};
    if (s.codes_int8) {
      t.int8        = true;
      t.probe_error = s.probe_code_error->data();
      t.row_error   = _lists->code_error;
      t.scale       = _lists->code_scale;
    } else {
      t.probe_error   = s.probe_half_error->data();
      t.row_error     = row_half_error;
      t.probe_sq      = s.probe_sqf->data();
      t.row_norm      = row_norm_max;
      t.d             = dim;
      t.inner_product = ip;
    }
    return t;
  }

  // Exactness, in the notation of the lemma (sections/3-motiv.tex, "Error-bounded filtering").
  // For probe row r and build row s let c(r, s) be the filter value (the code distance) and
  // delta(r, s) the re-scored FP32 distance. b_r, the row's bound (the k-th re-scored distance it
  // holds, or the seed's), never falls below delta*_r, its true k-th FP32 distance. The lemma's
  // widened limit (code_limits: float16_bound_limit, int8_code_limit)
  //   L_r(b) = ((sqrt(b) + e_r + e)^2 + eta_r)(1 + 2^-20)
  // keeps every pair within b:  delta(r, s) <= b  =>  c(r, s) <= L_r(b).  Its converse, the bound
  // the seeds raise a k-th code distance to (float16_seed_upper_bound, int8_seed_upper_bound),
  //   U_r(c) = (sqrt(c + eta_r) + e_r + e)^2 (1 + 2^-20),
  // gives  c(r, s) <= c  =>  delta(r, s) <= U_r(c),  from the same e_r, e and eta_r (INT8: eta_r
  // = 0, codes exact in int32, and c in units of the codes' scale squared). Let a stretch hold
  // row r's candidates with codes c_(1) <= c_(2) <= ..., none of them a pair r's accumulator
  // holds, and acc_(i) its i-th re-scored distance (acc_(0) = -inf). Each of b_r and
  // max(acc_(k-j), U_r(c_(j))) for j = 0..k has k distinct pairs within it, so
  //   B_r = min(b_r, min_j max(acc_(k-j), U_r(c_(j))))  >=  delta*_r,
  // and a candidate with  c > L_r(B_r)  has delta > B_r >= delta*_r: strictly past the k-th, not
  // in r's top-k under any order of ties. It is dropped without reading its FP32 row.
  // The gap check: the first m k candidates are all that can be in r's top-k when
  // c_(mk) > L_r(B_r), which, as B_r <= U_r(c_(k)), holds whenever
  //   c_(mk) > L_r(U_r(c_(k))) ~ (sqrt(c_(k) + eta_r) + 2 (e_r + e))^2 + eta_r,
  // i.e. "c_(mk) - e_r > c_(k) + e_r" in distance units with e_r + e the error of a pair.
  // Comparing every candidate with L_r(B_r) applies that test at every m at once: it keeps fewer
  // than m k exactly where the check passes for m (2, then 4), and where both fail it still drops
  // what lies past L_r(B_r) instead of re-scoring all. Nothing within B_r is dropped, so the
  // merge keeps the same pairs as without the trim, up to which of several pairs tied at exactly
  // the k-th distance it keeps -- which the filter's appends, in no fixed order, already leave
  // open from one run to the next. (The kernels use 1 + 1e-6 for 1 + 2^-20, in the limits and
  // seeds alike.)
  void gap_trim()
  {
    if (!s.gap_rescore || w.pending == 0) { return; }
    auto const t0 = debug_mark();
    if (!w.gap_limit) { w.gap_limit.emplace(static_cast<std::size_t>(n_left), stream, mr); }
    vss::gap_rescore_bound(w.candidates->rows.data(),
                           w.candidates->distances.data(),
                           w.pending,
                           st.acc_distances->view().data<float>(),
                           n_left,
                           k_join,
                           s.bound->data(),
                           code_terms(),
                           w.gap_limit->data(),
                           stream,
                           mr);
    // In place: each row's limit is computed from its own bound alone.
    code_limits(w.gap_limit->data(), w.gap_limit->data(), false);
    w.pending = vss::keep_within_row_limit(w.candidates->rows.data(),
                                           w.candidates->ids.data(),
                                           w.candidates->distances.data(),
                                           w.pending,
                                           w.gap_limit->data(),
                                           n_left,
                                           k_join,
                                           s.gap_stats_on ? &w.gap_stats : nullptr,
                                           stream,
                                           mr);
    if (dbg) { w.trim_seconds += std::chrono::duration<double>(debug_mark() - t0).count(); }
  }

  // The candidates' FP32 distances, over host probe rows: their rows are brought over a slice of
  // the candidates at a time, each distinct row once, up to kRows of them. The rows and their
  // slots are read on the host, where each slice's copy starts from.
  void rescore_from_host_rows()
  {
    constexpr std::size_t kRescoreBytes = std::size_t{256} << 20;
    auto const row_bytes                = static_cast<std::size_t>(dim) * sizeof(float);
    auto const kRows                    = std::max<std::size_t>(1, kRescoreBytes / row_bytes);
    if (w.host_row_slot.empty()) { w.host_row_slot.assign(static_cast<std::size_t>(n_left), -1); }
    std::vector<std::int32_t> rows(static_cast<std::size_t>(w.pending));
    CUDF_CUDA_TRY(cudaMemcpyAsync(rows.data(),
                                  w.candidates->rows.data(),
                                  rows.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    stream.sync();
    std::vector<std::int32_t> distinct;
    std::vector<float> staged;
    for (std::size_t from = 0; from < rows.size();) {
      distinct.clear();
      auto to = from;
      for (; to < rows.size(); ++to) {
        auto& slot = w.host_row_slot[static_cast<std::size_t>(rows[to])];
        if (slot < 0) {
          if (distinct.size() == kRows) { break; }
          slot = static_cast<std::int32_t>(distinct.size());
          distinct.push_back(rows[to]);
        }
        rows[to] = slot;
      }
      staged.resize(distinct.size() * static_cast<std::size_t>(dim));
      st.host_rows->gather(distinct.data(), distinct.size(), staged.data());
      for (auto const row : distinct) {
        w.host_row_slot[static_cast<std::size_t>(row)] = -1;
      }
      rmm::device_uvector<float> device_rows(staged.size(), stream, mr);
      rmm::device_uvector<std::int32_t> slots(to - from, stream, mr);
      // From pageable vectors, so the copies have completed when the calls return.
      CUDF_CUDA_TRY(cudaMemcpyAsync(device_rows.data(),
                                    staged.data(),
                                    staged.size() * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(slots.data(),
                                    rows.data() + from,
                                    slots.size() * sizeof(std::int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      vss::exact_distances(device_rows.data(),
                           slots.data(),
                           0,
                           w.candidates->ids.data() + from,
                           w.candidates->distances.data() + from,
                           static_cast<std::int64_t>(to - from),
                           exact_source,
                           dim,
                           stream,
                           mr,
                           nullptr,
                           0.f,
                           ip);
      stream.sync();
      from = to;
    }
  }

  void flush()
  {
    if (w.pending == 0) { return; }
    ++w.bounded_merges;
    w.bounded_emitted += w.pending - w.pairs_back;
    w.pairs_back       = 0;
    auto const t0      = debug_mark();
    auto const* c_rows = w.candidates->rows.data();
    auto const* c_ids  = w.candidates->ids.data();
    auto const* c_dist = w.candidates->distances.data();
    auto const done    = [&] {
      CUDF_CUDA_TRY(
        cudaMemsetAsync(w.candidates->count.data(), 0, sizeof(unsigned long long), stream.get()));
      w.pending = 0;
      if (dbg) { w.rescore_seconds += std::chrono::duration<double>(debug_mark() - t0).count(); }
    };
    if (keyed && !best_only) {
      // The pairs the keys drop are dropped before anything is re-scored; the rest are held to
      // the radius as every pair of a threshold join is.
      auto const no_limit = std::numeric_limits<float>::infinity();
      auto const* row_map =
        s.rescore ? static_cast<std::int64_t const*>(_lists->row_ids->data()) : nullptr;
      auto kept = s.rescore ? vss::take_pairs(c_rows,
                                              c_ids,
                                              c_dist,
                                              w.pending,
                                              st.layout_keys,
                                              no_limit,
                                              nullptr,
                                              false,
                                              stream,
                                              mr)
                            : vss::take_pairs(c_rows,
                                              c_ids,
                                              c_dist,
                                              w.pending,
                                              st.pin_keys,
                                              s.radius_bound,
                                              nullptr,
                                              s.sqrt_at_end,
                                              stream,
                                              mr);
      if (s.rescore && kept.rows.size() > 0) {
        auto const n = static_cast<std::int64_t>(kept.rows.size());
        vss::exact_distances(s.search_queries.data_handle(),
                             kept.rows.data(),
                             0,
                             kept.ids.data(),
                             kept.distances.data(),
                             n,
                             exact_source,
                             dim,
                             stream,
                             mr,
                             s.certain_limit ? s.certain_limit->data() : nullptr,
                             s.radius_bound,
                             ip);
        kept = vss::take_pairs(kept.rows.data(),
                               kept.ids.data(),
                               kept.distances.data(),
                               n,
                               {},
                               s.radius_bound,
                               row_map,
                               s.sqrt_at_end,
                               stream,
                               mr);
      }
      if (kept.rows.size() > 0) {
        auto const pairs = static_cast<std::int64_t>(kept.rows.size());
        if (!count_only) {
          found.left.push_back(
            std::make_unique<cudf::column>(std::move(kept.rows), rmm::device_buffer{}, 0));
          found.neighbors.push_back(
            std::make_unique<cudf::column>(std::move(kept.ids), rmm::device_buffer{}, 0));
          found.distances.push_back(
            std::make_unique<cudf::column>(std::move(kept.distances), rmm::device_buffer{}, 0));
        }
        note_radius_part(pairs);
      }
      done();
      return;
    }
    if (best_only) {
      if (s.rescore) {
        // The filter's distances are inexact, so a row's nearest by them is re-scored first: its
        // exact distance bounds the row's best match, and the filter limit of that bound admits
        // few of the row's other pairs. Those are re-scored too, and the rest never are.
        rmm::device_uvector<std::int64_t> winner(static_cast<std::size_t>(n_left), stream, mr);
        vss::best_match_winners(c_rows,
                                c_ids,
                                c_dist,
                                w.pending,
                                st.layout_keys,
                                s.code_limit->data(),
                                st.best_scratch->data(),
                                winner.data(),
                                n_left,
                                stream);
        for (bool const winners : {true, false}) {
          auto taken   = vss::best_match_take(c_rows,
                                              c_ids,
                                              c_dist,
                                              w.pending,
                                              st.layout_keys,
                                              s.code_limit->data(),
                                              winner.data(),
                                              winners,
                                              stream,
                                              mr);
          auto const n = static_cast<std::int64_t>(taken.rows.size());
          if (n == 0) { continue; }
          vss::exact_distances(s.search_queries.data_handle(),
                               taken.rows.data(),
                               0,
                               taken.ids.data(),
                               taken.distances.data(),
                               n,
                               exact_source,
                               dim,
                               stream,
                               mr,
                               nullptr,
                               0.f,
                               ip);
          fold_best(taken.rows.data(),
                    taken.ids.data(),
                    taken.distances.data(),
                    n,
                    s.radius_bound,
                    {},
                    s.bound->data());
          refresh_code_limit(s.bound->data());
          st.best_rescored += n;
        }
      } else {
        fold_best(c_rows, c_ids, c_dist, w.pending, s.radius_bound, st.pin_keys, s.bound->data());
      }
      done();
      return;
    }
    if (!radius_join) { gap_trim(); }
    w.gap_rescored += s.rescore ? w.pending : 0;
    auto const t_rescore = debug_mark();
    if (s.rescore && st.probe_rows_on_host) {
      rescore_from_host_rows();
    } else if (s.rescore) {
      vss::exact_distances(s.search_queries.data_handle(),
                           w.candidates->rows.data(),
                           0,
                           w.candidates->ids.data(),
                           w.candidates->distances.data(),
                           w.pending,
                           exact_source,
                           dim,
                           stream,
                           mr,
                           s.certain_limit ? s.certain_limit->data() : nullptr,
                           s.radius_bound,
                           ip);
    }
    auto const t1 = debug_mark();
    if (dbg) { w.rescore_seconds += std::chrono::duration<double>(t1 - t_rescore).count(); }
    if (radius_join) {
      // FP16 survivors were re-scored above and are held to the radius again in FP32; int8
      // distances are already exact. Their ids stay layout rows until here.
      auto kept = vss::take_within(
        *w.candidates,
        w.pending,
        s.radius_bound,
        s.rescore ? static_cast<std::int64_t const*>(_lists->row_ids->data()) : nullptr,
        s.sqrt_at_end,
        stream,
        mr);
      if (kept.rows.size() > 0) {
        auto const pairs = static_cast<std::int64_t>(kept.rows.size());
        if (!count_only) {
          found.left.push_back(
            std::make_unique<cudf::column>(std::move(kept.rows), rmm::device_buffer{}, 0));
          found.neighbors.push_back(
            std::make_unique<cudf::column>(std::move(kept.ids), rmm::device_buffer{}, 0));
          found.distances.push_back(
            std::make_unique<cudf::column>(std::move(kept.distances), rmm::device_buffer{}, 0));
        }
        note_radius_part(pairs);
      }
      CUDF_CUDA_TRY(
        cudaMemsetAsync(w.candidates->count.data(), 0, sizeof(unsigned long long), stream.get()));
      w.pending = 0;
      if (dbg) { w.take_seconds += std::chrono::duration<double>(debug_mark() - t1).count(); }
      return;
    }
    vss::merge_bound_candidates(st.acc_distances->mutable_view().data<float>(),
                                st.acc_neighbors->mutable_view().data<std::int64_t>(),
                                n_left,
                                k_join,
                                *w.candidates,
                                w.pending,
                                s.bound->data(),
                                stream,
                                mr);
    refresh_code_limit(s.bound->data());
    CUDF_CUDA_TRY(
      cudaMemsetAsync(w.candidates->count.data(), 0, sizeof(unsigned long long), stream.get()));
    w.pending = 0;
    if (dbg) { w.take_seconds += std::chrono::duration<double>(debug_mark() - t1).count(); }
  }

  [[nodiscard]] bool joins_group(std::int64_t m, std::int64_t n)
  {
    return s.group_launch &&
           (m <= s.group_m_max || m * n <= s.group_work_max || n <= s.group_slice_rows);
  }

  void launch_bounded(bounded_slice const& sl)
  {
    if (s.f16_bounded) {
      vss::bound_filter_f16(static_cast<std::uint16_t const*>(sl.x),
                            static_cast<float const*>(sl.x_sq),
                            sl.n,
                            sl.id_base,
                            s.probe_f16->data(),
                            probe_filter_sq(),
                            sl.rows,
                            sl.m,
                            dim,
                            s.code_limit->data(),
                            vss::float16_slack{},
                            *w.candidates,
                            stream);
      return;
    }
    vss::bound_filter_int8(static_cast<std::int8_t const*>(sl.x),
                           static_cast<std::int32_t const*>(sl.x_sq),
                           sl.n,
                           sl.id_base,
                           sl.id_map,
                           s.probe_i8->data(),
                           s.probe_sq->data(),
                           sl.rows,
                           sl.m,
                           dim,
                           s.codes_int8 ? s.code_limit->data() : s.bound->data(),
                           *w.candidates,
                           stream);
  }

  [[nodiscard]] std::int64_t read_count()
  {
    unsigned long long total = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &total, w.candidates->count.data(), sizeof(total), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    return static_cast<std::int64_t>(total);
  }

  void rollback()
  {
    auto const keep = static_cast<unsigned long long>(w.pending);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      w.candidates->count.data(), &keep, sizeof(keep), cudaMemcpyHostToDevice, stream.get()));
  }

  void launch_deferred()
  {
    if (!deferred.empty()) {
      vss::bound_filter_group(
        deferred,
        s.f16_bounded,
        s.f16_bounded ? static_cast<void const*>(s.probe_f16->data())
                      : static_cast<void const*>(s.probe_i8->data()),
        s.f16_bounded ? static_cast<void const*>(probe_filter_sq())
                      : static_cast<void const*>(s.probe_sq->data()),
        dim,
        s.f16_bounded || s.codes_int8 ? s.code_limit->data() : s.bound->data(),
        *w.candidates,
        stream,
        mr);
      deferred.clear();
    }
  }

  // A single FLOAT16 list holds its pairs through the pieces instead of re-scoring between them.
  // Each row's bound falls to B_r of the lemma above gap_trim, from the k smallest codes held and
  // its accumulator: the codes are of k distinct pairs, none in the accumulator, as a piece's
  // rows are new and a rollback discards what was not committed. A pair past its limit has
  // delta > B_r >= delta*_r, strictly past the k-th, and is dropped unread; the limit is applied
  // to the next piece too. Each trim's B_r stands on its own, so seeing a pair in several trims
  // changes nothing, and bounds only fall. All that is left reaches the last flush, or an
  // earlier one when the buffer stays too full. The first piece's bound is infinite, so all its
  // pairs pass, and the buffer has room for them. The same codes and U_r as the seed's bound.
  void hold_trim()
  {
    if (w.pending == 0) { return; }
    auto const t0 = debug_mark();
    w.bounded_emitted += w.pending - w.pairs_back;
    if (!w.gap_limit) { w.gap_limit.emplace(static_cast<std::size_t>(n_left), stream, mr); }
    vss::gap_rescore_bound(w.candidates->rows.data(),
                           w.candidates->distances.data(),
                           w.pending,
                           st.acc_distances->view().data<float>(),
                           n_left,
                           k_join,
                           s.bound->data(),
                           code_terms(),
                           w.gap_limit->data(),
                           stream,
                           mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(s.bound->data(),
                                  w.gap_limit->data(),
                                  static_cast<std::size_t>(n_left) * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
    refresh_code_limit(s.bound->data());
    w.pending    = vss::keep_within_row_limit(w.candidates->rows.data(),
                                              w.candidates->ids.data(),
                                              w.candidates->distances.data(),
                                              w.pending,
                                              s.code_limit->data(),
                                              n_left,
                                              k_join,
                                              nullptr,
                                              stream,
                                              mr);
    w.pairs_back = w.pending;
    rollback();
    if (dbg) { w.trim_seconds += std::chrono::duration<double>(debug_mark() - t0).count(); }
  }

  [[nodiscard]] std::size_t device_room()
  {
    auto const free_bytes = mem_space->get_available_memory(stream);
    return free_bytes > kFlushMargin ? free_bytes - kFlushMargin : std::size_t{0};
  }

  void move_pairs_to_host()
  {
    if (w.pending == 0) { return; }
    auto const t0 = debug_mark();
    w.bounded_emitted += w.pending - w.pairs_back;
    w.pairs_back = 0;
    if (!w.moved_codes) {
      w.moved_codes.emplace(static_cast<std::size_t>(n_left * k_join), stream, mr);
      vss::fill_bound(
        w.moved_codes->data(), n_left * k_join, std::numeric_limits<float>::infinity(), stream);
    }
    vss::code_topk_merge(w.candidates->rows.data(),
                         w.candidates->distances.data(),
                         w.pending,
                         w.moved_codes->data(),
                         n_left,
                         k_join,
                         s.bound->data(),
                         code_terms(),
                         stream,
                         mr);
    refresh_code_limit(s.bound->data());
    w.pending = vss::keep_within_row_limit(w.candidates->rows.data(),
                                           w.candidates->ids.data(),
                                           w.candidates->distances.data(),
                                           w.pending,
                                           s.code_limit->data(),
                                           n_left,
                                           k_join,
                                           nullptr,
                                           stream,
                                           mr);
    if (!w.host_pairs.reserve(w.pending, s.pair_budget)) {
      w.pairs_back    = w.pending;
      auto const need = static_cast<std::size_t>(w.pending) * kFlushBytesPerPair;
      auto const room = device_room();
      if (s.budget_flush_refused || need > room) {
        throw std::runtime_error(
          std::format("[sirius_physical_vector_join_stream] vector join of '{}' and '{}' (k {}, "
                      "{} probes): {} pairs of this task wait on the host and {} more in the "
                      "buffer have nowhere to go. The host budget for waiting pairs is {} MiB "
                      "(half the host pool, less what the join already holds beside it), and a "
                      "flush needs about {} MiB with {} MiB free on the device. A larger host "
                      "pool or a smaller k may let it run.",
                      _request.left.table,
                      _request.right.table,
                      k_join,
                      _request.n_probes,
                      w.pairs_moved,
                      w.pending,
                      s.pair_budget >> 20,
                      need >> 20,
                      room >> 20));
      }
      ++w.budget_flushes;
      rollback();
      flush();
      return;
    }
    for (std::int64_t at = 0; at < w.pending;) {
      auto& block      = w.host_pairs.tail();
      auto const take  = std::min(w.pending - at, kHostBlockPairs - block.n);
      auto const count = static_cast<std::size_t>(take);
      CUDF_CUDA_TRY(cudaMemcpyAsync(block.rows() + block.n,
                                    w.candidates->rows.data() + at,
                                    count * sizeof(std::int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(block.ids() + block.n,
                                    w.candidates->ids.data() + at,
                                    count * sizeof(std::int64_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(block.codes() + block.n,
                                    w.candidates->distances.data() + at,
                                    count * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      block.n += take;
      at += take;
    }
    stream.sync();
    w.pairs_moved += w.pending;
    ++w.pair_moves;
    w.pending = 0;
    rollback();
    if (dbg) { w.host_seconds += std::chrono::duration<double>(debug_mark() - t0).count(); }
  }

  // The task's end: every row's bound is final once the last pairs have moved, and each piece's
  // flush trims against it.
  void bring_pairs_back()
  {
    move_pairs_to_host();
    auto const piece = std::clamp<std::int64_t>(
      static_cast<std::int64_t>(device_room() / kFlushBytesPerPair),
      std::min<std::int64_t>(std::int64_t{1} << 16, w.candidates->capacity()),
      std::min(kBackPiecePairs, w.candidates->capacity()));
    std::int64_t filled = 0;
    auto flush_piece    = [&] {
      if (filled == 0) { return; }
      w.pending    = filled;
      w.pairs_back = filled;
      rollback();
      flush();
      filled = 0;
    };
    while (!w.host_pairs.empty()) {
      auto const& block = w.host_pairs.front();
      for (std::int64_t at = 0; at < block.n;) {
        auto const take  = std::min(block.n - at, piece - filled);
        auto const count = static_cast<std::size_t>(take);
        CUDF_CUDA_TRY(cudaMemcpyAsync(w.candidates->rows.data() + filled,
                                      block.rows() + at,
                                      count * sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(w.candidates->ids.data() + filled,
                                      block.ids() + at,
                                      count * sizeof(std::int64_t),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(w.candidates->distances.data() + filled,
                                      block.codes() + at,
                                      count * sizeof(float),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        stream.sync();
        at += take;
        filled += take;
        if (filled == piece) { flush_piece(); }
      }
      w.host_pairs.drop_front();
    }
    flush_piece();
  }

  void make_room()
  {
    if (s.hold_codes_pairs) {
      auto const before = w.pending;
      hold_trim();
      if (w.pending == before || w.pending > w.candidates->capacity() / 2) { flush(); }
    } else if (s.pairs_to_host &&
               (s.pairs_to_host_forced ||
                static_cast<std::size_t>(w.pending) * kFlushBytesPerPair > device_room())) {
      move_pairs_to_host();
    } else {
      flush();
    }
  }

  // The slice is launched again after each change: the buffer's room, and the bound.
  void replay_slice(bounded_slice const& sl)
  {
    while (true) {
      launch_bounded(sl);
      auto const total = read_count();
      if (total <= w.candidates->capacity()) {
        w.pending = total;
        return;
      }
      auto const slice_pairs = total - w.pending;
      rollback();
      if (w.pending > 0) {
        make_room();
        continue;
      }
      auto const want = slice_pairs + slice_pairs / 4;
      if (s.cut_large_slices && sl.n > 1 &&
          (s.slice_cuts_forced ||
           static_cast<std::size_t>(want) * kFlushBytesPerPair >
             device_room() + static_cast<std::size_t>(w.candidates->capacity()) * 16)) {
        ++w.slice_cuts;
        auto lo = sl;
        lo.n    = sl.n / 2;
        auto hi = sl;
        hi.x    = static_cast<std::byte const*>(sl.x) + static_cast<std::size_t>(lo.n) *
                                                          static_cast<std::size_t>(dim) *
                                                          (s.f16_bounded ? 2 : 1);
        hi.x_sq = static_cast<std::byte const*>(sl.x_sq) + static_cast<std::size_t>(lo.n) * 4;
        hi.n    = sl.n - lo.n;
        hi.id_base += lo.n;
        if (hi.id_map != nullptr) { hi.id_map += lo.n; }
        replay_slice(lo);
        make_room();
        replay_slice(hi);
        return;
      }
      w.candidates.reset();
      w.candidates.emplace(want, stream, mr);
    }
  }

  // The buffer's fill is read back once per group of slices, not per slice: a group that
  // overflowed is rolled back to the count before it and replayed a slice at a time, first
  // after the buffered pairs are merged, which tightens the bound, then into a bigger buffer.
  void check_group()
  {
    if (group.empty()) { return; }
    launch_deferred();
    auto total = read_count();
    if (total > w.candidates->capacity()) {
      rollback();
      for (auto const& sl : group) {
        replay_slice(sl);
      }
      total = w.pending;
    }
    w.pending = total;
    group.clear();
    if (s.pairs_to_host_forced || w.pending > w.candidates->capacity() / 2) { make_room(); }
  }

  void release_staged(staged_vector_chunk& chunk)
  {
    if (chunk.owner) {
      auto mut = chunk.owner->to_mutable();
      mut.rebind_stream(stream);
    }
    if (chunk.buffer) { chunk.buffer->set_stream(stream); }
    if (chunk.reader) { borrowed.hold(std::move(*chunk.reader)); }
    chunk = staged_vector_chunk{};
  }

  [[nodiscard]] std::size_t staged_rows_bytes(std::int64_t rows)
  { return (static_cast<std::size_t>(rows + 4) * _lists->row_bytes() + 255) / 256 * 256; }

  [[nodiscard]] staged_vector_chunk stage_compact(std::size_t j)
  {
    auto const rows = _lists->rows_in_chunk(static_cast<std::int64_t>(j));
    auto const bytes =
      staged_rows_bytes(rows) + (s.list_norms != nullptr ? static_cast<std::size_t>(rows) * 4 : 0);
    if (_op._sirius_ctx != nullptr) { _op._sirius_ctx->record_vector_join_corpus_stage(bytes); }
    std::shared_ptr<cucascade::memory::reservation> reservation;
    if (pair_step == nullptr) {
      reservation = mem_space->make_reservation_or_null(bytes);
      if (!reservation) {
        vss::throw_staging_shortfall(
          *mem_space,
          bytes,
          "[sirius_physical_vector_join_stream] lists chunk " + std::to_string(j));
      }
    }
    auto buffer = std::make_unique<rmm::device_buffer>(
      bytes, stage_on, reservation ? reservation->get_memory_resource() : mr);
    auto* out       = static_cast<std::byte*>(buffer->data());
    auto const lock = vss::lock_host_list_chunk(*_lists, j, stage_on);
    vss::copy_host_list_rows(*_lists, lock, 0, rows, out, stage_on);
    if (s.list_norms != nullptr) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        out + staged_rows_bytes(rows),
        static_cast<std::byte const*>(s.list_norms) +
          static_cast<std::size_t>(static_cast<std::int64_t>(j) * _lists->chunk_rows) * 4,
        static_cast<std::size_t>(rows) * 4,
        cudaMemcpyDefault,
        stage_on.get()));
    }
    // As the chunk sources do: the host waits on the copy while the device runs what the
    // previous chunk issued, and the copy is complete before anything reads it. The read lock
    // ends with it.
    stage_on.sync();
    staged_vector_chunk staged{cudf::column_view{}, nullptr, std::move(reservation)};
    staged.buffer = std::move(buffer);
    return staged;
  }

  [[nodiscard]] bool can_keep()
  { return mem_space->make_reservation_or_null(mem_space->get_max_memory() / 12) != nullptr; }

  void check_chunk_base(std::size_t j, std::int64_t chunk_base)
  {
    if (chunk_base != static_cast<std::int64_t>(j) * _lists->chunk_rows) {
      throw std::runtime_error("[sirius_physical_vector_join_stream] lists chunk " +
                               std::to_string(j) + " starts at row " +
                               std::to_string(static_cast<std::int64_t>(j) * _lists->chunk_rows) +
                               " but its cluster runs at " + std::to_string(chunk_base));
    }
  }

  [[nodiscard]] staged_vector_chunk stage_next(std::size_t j)
  {
    if (copy_compact) {
      if (auto it = kept.find(j); it != kept.end()) {
        auto chunk = std::move(it->second);
        kept.erase(it);
        return chunk;
      }
      return stage_compact(j);
    }
    auto& source = s.exact_lists_corpus ? *s.exact_lists_corpus : *_op._corpus;
    if (direct) { return staged_vector_chunk{}; }
    return pair_step != nullptr ? source.stage_for_pair(j, *mem_space, stream)
                                : stage_for_fold(source, j, *mem_space, stage_on, stream);
  }

  // A sample is a quarter of its cluster, at least eight times k rows (and 64) and at most the
  // larger of that and kSeedSample: a sample near the whole cluster is the sweep again, on CUDA
  // cores. Past kSeedSample rows the seed ranks it through device memory, not shared memory.
  [[nodiscard]] std::int64_t seed_rows(std::int64_t size)
  {
    auto const floor = std::max<std::int64_t>(64, 8 * k_join);
    return std::min(
      size,
      std::clamp<std::int64_t>(size / 4, floor, std::max<std::int64_t>(floor, vss::kSeedSample)));
  }

  void seed_bound()
  {
    std::vector<std::int64_t> first(static_cast<std::size_t>(n_left));
    std::vector<std::int32_t> count(static_cast<std::size_t>(n_left));
    std::int64_t max_count = 0;
    for (std::int64_t r = 0; r < n_left; ++r) {
      auto const c  = static_cast<std::size_t>(s.nearest_cluster[static_cast<std::size_t>(r)]);
      auto const lo = _lists->offsets[c];
      first[static_cast<std::size_t>(r)] = lo;
      count[static_cast<std::size_t>(r)] =
        static_cast<std::int32_t>(seed_rows(_lists->offsets[c + 1] - lo));
      max_count = std::max<std::int64_t>(max_count, count[static_cast<std::size_t>(r)]);
    }
    // Probe rows by nearest cluster, so the kernel reads one cluster's sample at a time.
    std::vector<std::int64_t> order(static_cast<std::size_t>(n_left));
    {
      std::vector<std::int64_t> at(_lists->offsets.size(), 0);
      for (std::int64_t r = 0; r < n_left; ++r) {
        ++at[static_cast<std::size_t>(s.nearest_cluster[static_cast<std::size_t>(r)]) + 1];
      }
      std::partial_sum(at.begin(), at.end(), at.begin());
      for (std::int64_t r = 0; r < n_left; ++r) {
        auto const c = static_cast<std::size_t>(s.nearest_cluster[static_cast<std::size_t>(r)]);
        order[static_cast<std::size_t>(at[c]++)] = r;
      }
    }
    rmm::device_uvector<std::int64_t> order_d(order.size(), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(order_d.data(),
                                  order.data(),
                                  order.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    rmm::device_uvector<std::int64_t> first_d(first.size(), stream, mr);
    rmm::device_uvector<std::int32_t> count_d(count.size(), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(first_d.data(),
                                  first.data(),
                                  first.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(count_d.data(),
                                  count.data(),
                                  count.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    if (s.f16_bounded) {
      vss::seed_bound_f16(static_cast<std::uint16_t const*>(_lists->device_vectors->data()),
                          s.ip_row_zero_sq ? s.ip_row_zero_sq->data()
                                           : static_cast<float const*>(_lists->row_sq_f32->data()),
                          s.probe_f16->data(),
                          probe_filter_sq(),
                          first_d.data(),
                          count_d.data(),
                          order_d.data(),
                          n_left,
                          dim,
                          static_cast<int>(k_join),
                          max_count,
                          s.bound->data(),
                          stream,
                          mr);
      vss::float16_seed_upper_bound(s.bound->data(),
                                    s.probe_sqf->data(),
                                    s.probe_half_error->data(),
                                    _lists->half_error,
                                    _lists->max_row_norm,
                                    dim,
                                    n_left,
                                    stream,
                                    ip);
    } else {
      vss::seed_bound_int8(static_cast<std::int8_t const*>(_lists->device_vectors->data()),
                           static_cast<std::int32_t const*>(_lists->row_sq->data()),
                           s.probe_i8->data(),
                           s.probe_sq->data(),
                           first_d.data(),
                           count_d.data(),
                           order_d.data(),
                           n_left,
                           dim,
                           static_cast<int>(k_join),
                           max_count,
                           s.bound->data(),
                           stream,
                           mr);
      if (s.codes_int8) {
        vss::int8_seed_upper_bound(s.bound->data(),
                                   s.probe_code_error->data(),
                                   _lists->code_error,
                                   _lists->code_scale,
                                   n_left,
                                   stream);
      }
    }
    if (s.rescore) {
      s.code_limit.emplace(static_cast<std::size_t>(n_left), stream, mr);
      refresh_code_limit(s.bound->data());
    }
  }

  // Where each stretch of a cluster of @p size rows ends, in rows from its start: the sample,
  // then doubling; a last stretch under a quarter of the one before it joins that one.
  [[nodiscard]] std::vector<std::int64_t> stretch_ends(std::int64_t size)
  {
    std::vector<std::int64_t> ends;
    for (auto e = seed_rows(size);; e *= 2) {
      if (size - e < e / 4 + 1) {
        ends.push_back(size);
        return ends;
      }
      ends.push_back(e);
    }
  }

  /// Which chunks a pass reads and how: what a task needs at hand of a sweep the block is set for.
  void set_sweep(int which)
  {
    sweep = which;
    if (sweep == 1 && !kept.empty()) {
      for (auto const& entry : kept) {
        kept_first.push_back(entry.first);
      }
      for (auto const j : s.needed_chunks) {
        if (kept.count(j) == 0) { kept_first.push_back(j); }
      }
    }
    chunks = s.any_bounded && sweep == 0 ? &s.nearest_chunks
             : kept_first.empty()        ? &s.needed_chunks
                                         : &kept_first;
    // A sweep that reads the lists in place stages nothing. FP16 lists on the host are copied
    // in compact for sweep 0 as well (and widened on the device when it ranks in FP32), so that
    // copy can be kept.
    direct        = _lists != nullptr &&
                    (s.int8_search || (s.f16_bounded && (sweep == 1 || s.seed)) || s.f16_sweep0);
    widen_compact = s.lists_on_host && s.f16_bounded && sweep == 0 && !direct;
    copy_compact  = (direct && s.lists_on_host) || widen_compact;
    // Sweep 0 over lossy rows only sets each probe row's bound, which sweep 1 re-searches past:
    // every cluster in one launch over the device lists (nearest_cluster_topk) instead of one
    // GEMM and top-k call per cluster, whose ~0.18 ms of host setup dwarfs its arithmetic once
    // each cluster has only tens of probe rows. SIRIUS_VSS_GROUPED_SWEEP0=0 keeps the calls.
    grouped_sweep0 = sweep == 0 && s.rescore && !s.seed && !s.ranked_sweep0 && !radius_join &&
                     !ip && st.earlier_cut == nullptr && k_join <= 64 && !s.lists_on_host &&
                     ((s.int8_search && s.probe_i8) || (s.f16_sweep0 && s.probe_f16)) && [] {
                       auto const* v = std::getenv("SIRIUS_VSS_GROUPED_SWEEP0");
                       return v == nullptr || std::string_view{v} != "0";
                     }();
  }

  /// Sets up a pass over the corpus: which chunks it reads and how, and the bounds it starts from.
  void begin_sweep(int which)
  {
    set_sweep(which);
    auto const& edge_begin    = s.routed->edge_begin;
    auto const& nearest_edges = s.routed->nearest_edges;
    stretches.clear();
    if (!building_for_pairs) {
      prefetched = chunks->empty() ? staged_vector_chunk{} : stage_next((*chunks)[0]);
    }
    if (sweep == 0 && s.seed) {
      s.bound.emplace(static_cast<std::size_t>(n_left), stream, mr);
      if (s.lists_on_host || s.f16_list_no_seed) {
        vss::fill_bound(s.bound->data(), n_left, std::numeric_limits<float>::infinity(), stream);
        if (s.rescore) {
          s.code_limit.emplace(static_cast<std::size_t>(n_left), stream, mr);
          refresh_code_limit(s.bound->data());
        }
      } else {
        seed_bound();
      }
      w.candidates.emplace(
        std::max<std::int64_t>({std::int64_t{1} << 22,
                                4 * n_left * k_join,
                                s.f16_list_no_seed ? n_left * s.running_bound_rows * 5 / 4 : 0}),
        stream,
        mr);
    }
    if (sweep == 1) {
      if (s.seed) {
        check_group();
        flush();
      }
      if (dbg && s.seed) {
        std::string per_stretch;
        for (auto const p : w.stretch_pairs) {
          per_stretch += " " + std::to_string(p);
        }
        std::fprintf(stderr,
                     "[vecjoin-phase] sweep 0: %lld pairs passed, %lld merges, %zu stretches:%s\n",
                     static_cast<long long>(w.bounded_emitted),
                     static_cast<long long>(w.bounded_merges),
                     w.stretch_pairs.size(),
                     per_stretch.c_str());
      }
      phase("nearest-cluster sweep");
      s.bound.emplace(static_cast<std::size_t>(n_left), stream, mr);
      if (radius_join) {
        vss::fill_bound(s.bound->data(), n_left, s.radius_bound, stream);
      } else if (s.rescore && !s.seed) {
        // A scanned corpus re-scored sweep 0's answers chunk by chunk, while each was staged.
        if (_lists != nullptr && !s.sweep0_bound_from_codes) {
          vss::exact_distances(s.search_queries.data_handle(),
                               nullptr,
                               k_join,
                               st.acc_neighbors->view().data<std::int64_t>(),
                               st.acc_distances->mutable_view().data<float>(),
                               n_left * k_join,
                               exact_source,
                               dim,
                               stream,
                               mr,
                               nullptr,
                               0.f,
                               ip);
        }
        vss::row_max_bound(
          st.acc_distances->view().data<float>(), n_left, k_join, s.bound->data(), stream);
        if (s.sweep0_bound_from_codes) {
          vss::float16_seed_upper_bound(s.bound->data(),
                                        s.probe_sqf->data(),
                                        s.probe_half_error->data(),
                                        _lists->half_error,
                                        _lists->max_row_norm,
                                        dim,
                                        n_left,
                                        stream,
                                        false);
        }
        vss::fill_misses(st.acc_distances->mutable_view().data<float>(),
                         st.acc_neighbors->mutable_view().data<std::int64_t>(),
                         n_left * k_join,
                         stream);
      } else {
        vss::kth_distance_bound(
          st.acc_distances->view().data<float>(), n_left, k_join, s.bound->data(), stream);
      }
      if (s.rescore) {
        s.code_limit.emplace(static_cast<std::size_t>(n_left), stream, mr);
        refresh_code_limit(s.bound->data());
      }
      if (s.skip_certain) {
        s.certain_limit.emplace(static_cast<std::size_t>(n_left), stream, mr);
        code_limits(s.bound->data(), s.certain_limit->data(), true);
      }
      // A radius join's buffer only has to amortize the flush that appends it to the result.
      if (!building_for_pairs) {
        w.candidates.emplace(radius_join
                               ? std::int64_t{1} << 22
                               : std::max<std::int64_t>(std::int64_t{1} << 22, 4 * n_left * k_join),
                             stream,
                             mr);
      }
    }
    if (grouped_sweep0) {
      // Each cluster is contiguous in the device lists, so one task group covers it whole.
      std::vector<vss::nearest_task> tasks;
      auto const& offsets = _lists->offsets;
      for (std::size_t c = 0; c + 1 < offsets.size(); ++c) {
        auto const nearest = static_cast<std::int64_t>(nearest_edges[c]);
        auto const n_rows  = offsets[c + 1] - offsets[c];
        if (nearest == 0 || n_rows == 0) { continue; }
        for (std::int64_t e = 0; e < nearest; e += vss::kNearestTaskProbes) {
          tasks.push_back({offsets[c],
                           n_rows,
                           edge_begin[c] + e,
                           static_cast<std::int32_t>(
                             std::min<std::int64_t>(vss::kNearestTaskProbes, nearest - e)),
                           offsets[c]});
        }
        w.scanned_pairs += n_rows * nearest;
      }
      rmm::device_uvector<vss::nearest_task> device_tasks(tasks.size(), stream, mr);
      // From a pageable vector, so the copy has completed when the call returns.
      CUDF_CUDA_TRY(cudaMemcpyAsync(device_tasks.data(),
                                    tasks.data(),
                                    tasks.size() * sizeof(vss::nearest_task),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      bool const f16 = !s.int8_search;
      vss::nearest_cluster_topk(
        device_tasks.data(),
        static_cast<std::int64_t>(tasks.size()),
        f16,
        _lists->device_vectors->data(),
        static_cast<std::int64_t>(_lists->row_bytes()),
        f16 ? static_cast<void const*>(_lists->row_sq_f32->data()) : _lists->row_sq->data(),
        s.routed_rows,
        f16 ? static_cast<void const*>(s.probe_f16->data()) : s.probe_i8->data(),
        f16 ? static_cast<void const*>(s.probe_sqf->data()) : s.probe_sq->data(),
        k_join,
        st.acc_distances->mutable_view().data<float>(),
        st.acc_neighbors->mutable_view().data<std::int64_t>(),
        stream);
    }
  }

  /// Filters or ranks one chunk of the pass against every probe row routed to its clusters, then
  /// stages the next chunk while the device works on this one.
  void search_chunk(std::size_t ci)
  {
    auto const& edge_begin    = s.routed->edge_begin;
    auto const& nearest_edges = s.routed->nearest_edges;
    auto const j              = (*chunks)[ci];
    auto staged               = std::move(prefetched);
    // A pair task's kernels may still be reading the chunk if it fails, and the reader goes with
    // it.
    std::optional<sync_stream_on_exit> quiesce;
    if (pair_step != nullptr) { quiesce.emplace(stream); }
    auto const chunk_base     = _op._chunk_row_base[j];
    float const* chunk_data   = nullptr;
    std::int64_t chunk_n      = 0;
    // Chunk j's stored rows and their norms, where a direct sweep reads them.
    std::byte const* compact     = nullptr;
    std::byte const* chunk_norms = nullptr;
    std::optional<rmm::device_buffer> widened;
    // A cosine chunk normalized for the every-cluster search (direct_cosine).
    std::optional<rmm::device_uvector<float>> unit_chunk;
    if (direct) {
      chunk_n = _lists->rows_in_chunk(static_cast<std::int64_t>(j));
      check_chunk_base(j, chunk_base);
      compact = copy_compact ? static_cast<std::byte const*>(staged.buffer->data())
                             : static_cast<std::byte const*>(_lists->device_vectors->data()) +
                                 static_cast<std::size_t>(chunk_base) * _lists->row_bytes();
      if (s.list_norms != nullptr) {
        chunk_norms = copy_compact ? compact + staged_rows_bytes(chunk_n)
                                   : static_cast<std::byte const*>(s.list_norms) +
                                       static_cast<std::size_t>(chunk_base) * 4;
      }
    } else if (widen_compact) {
      chunk_n = _lists->rows_in_chunk(static_cast<std::int64_t>(j));
      check_chunk_base(j, chunk_base);
      widened.emplace(static_cast<std::size_t>(chunk_n * dim) * sizeof(float), stream, mr);
      vss::widen_float16(static_cast<std::uint16_t const*>(staged.buffer->data()),
                         chunk_n * dim,
                         static_cast<float*>(widened->data()),
                         stream);
      chunk_data = static_cast<float const*>(widened->data());
    } else {
      auto const chunk_view = vss::list_column_as_dataset_view(staged.view, dim);
      chunk_data            = chunk_view.data_handle();
      chunk_n               = chunk_view.extent(0);
      if (s.direct_cosine && chunk_n > 0) {
        unit_chunk.emplace(static_cast<std::size_t>(chunk_n * dim), stream, mr);
        vss::normalize_rows(chunk_data, chunk_n, dim, unit_chunk->data(), stream);
        chunk_data = unit_chunk->data();
      }
    }
    // A scanned chunk read by the re-score: sweep 1's candidates carry corpus rows, which the
    // one block below maps onto the chunk (rows_per_block past any row, so block 0 always, its
    // base chunk_base rows before the chunk); sweep 0's ranked answers carry rows local to
    // their slice, one block per slice.
    std::optional<rmm::device_uvector<float const*>> scan_blocks;
    std::optional<rmm::device_uvector<std::uint16_t>> scan_f16_rows;
    std::optional<rmm::device_uvector<float>> scan_sqf;
    auto scan_source = [&](std::size_t block) {
      vss::exact_rows src;
      src.copy_blocks    = scan_blocks->data() + block;
      src.rows_per_block = std::numeric_limits<std::int64_t>::max();
      return src;
    };
    if (s.scan_f16 && chunk_n > 0) {
      auto const& runs = _op._chunk_cluster_runs[j];
      std::vector<float const*> bases;
      if (sweep == 1) {
        // Unsigned arithmetic: the base may lie before the allocation, and only rows of the
        // chunk are ever added back onto it.
        bases.push_back(
          reinterpret_cast<float const*>(reinterpret_cast<std::uintptr_t>(chunk_data) -
                                         static_cast<std::uintptr_t>(chunk_base) *
                                           static_cast<std::uintptr_t>(dim) * sizeof(float)));
      } else {
        for (auto const& slice : runs) {
          bases.push_back(chunk_data + slice.begin * dim);
        }
      }
      scan_blocks.emplace(bases.size(), stream, mr);
      // From a pageable vector, so the copy has completed when the call returns.
      CUDF_CUDA_TRY(cudaMemcpyAsync(scan_blocks->data(),
                                    bases.data(),
                                    bases.size() * sizeof(float const*),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      if (sweep == 1) {
        // The chunk as the FLOAT16 lists would hold it: rounded rows, their norms, and the
        // largest rounding error and norm, which this chunk's filter limit allows for.
        scan_f16_rows.emplace(static_cast<std::size_t>(chunk_n * dim), stream, mr);
        scan_sqf.emplace(static_cast<std::size_t>(chunk_n), stream, mr);
        rmm::device_uvector<unsigned int> maxima(2, stream, mr);
        CUDF_CUDA_TRY(cudaMemsetAsync(maxima.data(), 0, 2 * sizeof(unsigned int), stream.get()));
        vss::narrow_to_float16(chunk_data, chunk_n * dim, scan_f16_rows->data(), stream);
        vss::half_rows_norms(chunk_data,
                             scan_f16_rows->data(),
                             chunk_n,
                             dim,
                             scan_sqf->data(),
                             nullptr,
                             maxima.data(),
                             maxima.data() + 1,
                             stream);
        unsigned int bits[2] = {0, 0};
        CUDF_CUDA_TRY(
          cudaMemcpyAsync(bits, maxima.data(), sizeof(bits), cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
        float max_sq = 0;
        std::memcpy(&row_half_error, &bits[0], sizeof(float));
        std::memcpy(&max_sq, &bits[1], sizeof(float));
        row_norm_max = std::sqrt(max_sq);
        if (!(row_norm_max < 65504.f)) {
          throw std::runtime_error(
            "[sirius_physical_vector_join_stream] a corpus row norm of " +
            std::to_string(row_norm_max) +
            " is outside float16's range for the bounded search over a scanned corpus; "
            "SIRIUS_VSS_SCAN_BOUNDED=0 searches it in FP32");
        }
        refresh_code_limit(s.bound->data());
        compact      = reinterpret_cast<std::byte const*>(scan_f16_rows->data());
        chunk_norms  = reinterpret_cast<std::byte const*>(scan_sqf->data());
        exact_source = scan_source(0);
      }
    }

    auto const& runs     = grouped_sweep0 ? no_runs : _op._chunk_cluster_runs[j];
    auto const first_run = pair_step != nullptr ? pair_step->run : std::size_t{0};
    for (auto run = first_run; run < runs.size(); ++run) {
      auto const& slice = runs[run];
      if (pair_step != nullptr && holds_enough_to_stop()) {
        stopped_at = rest_of_chunk(run, 0, pair_step->call_rows);
        break;
      }
      auto eb          = edge_begin[static_cast<std::size_t>(slice.cluster)];
      auto ee          = edge_begin[static_cast<std::size_t>(slice.cluster) + 1];
      auto const split = eb + nearest_edges[static_cast<std::size_t>(slice.cluster)];
      if (s.any_bounded && !radius_join && (!s.rescore || s.seed)) {
        (sweep == 0 ? ee : eb) = split;
      }
      if (s.rescore && sweep == 0) { ee = split; }
      if (eb == ee) { continue; }
      // The slice was cut from the cluster column's chunk j; this is the first point at
      // which the vector column's chunk j is resident and its row count exactly known. The
      // two are the same pin's row groups, so a mismatch is a broken invariant rather than
      // a user error -- but it would read past the end of the chunk, so it is checked.
      if (slice.end > chunk_n) {
        throw std::runtime_error("[sirius_physical_vector_join_stream] cluster column chunk " +
                                 std::to_string(j) + " describes row " + std::to_string(slice.end) +
                                 " but the vector column's chunk " + "holds " +
                                 std::to_string(chunk_n) + "; both must come from the same pin");
      }
      auto slice_rows = slice.end - slice.begin;
      if (st.earlier_cut != nullptr) {
        auto const* first = st.earlier_cut + chunk_base + slice.begin;
        slice_rows        = std::lower_bound(first, first + slice_rows, st.probe_key_max) - first;
        if (slice_rows == 0) { continue; }
      }
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
      auto const* rows_c = s.routed_rows + eb;
      if (st.probe_rows_on_host) {
        if (sweep == 0) {
          std::vector<std::int64_t> slice_rows_c(static_cast<std::size_t>(m));
          CUDF_CUDA_TRY(cudaMemcpyAsync(slice_rows_c.data(),
                                        rows_c,
                                        slice_rows_c.size() * sizeof(std::int64_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          std::vector<float> gathered(static_cast<std::size_t>(m * dim));
          st.host_rows->gather(slice_rows_c.data(), slice_rows_c.size(), gathered.data());
          // From a pageable vector, so the copy has completed when the call returns.
          CUDF_CUDA_TRY(cudaMemcpyAsync(routed_queries->data(),
                                        gathered.data(),
                                        gathered.size() * sizeof(float),
                                        cudaMemcpyHostToDevice,
                                        stream.get()));
        }
      } else if (!direct && pair_step == nullptr) {
        vss::gather_rows(
          s.search_queries.data_handle(), dim, rows_c, m, routed_queries->data(), stream);
      }
      auto const queries_view =
        raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
          routed_queries->data(), m, dim);
      w.scanned_pairs += slice_rows * m;
      // Neighbour ids come back local to the slice; the slice's own start in corpus row
      // space is the base that makes them corpus row ids, exactly as the chunk offset does
      // in the exhaustive fold.
      auto const id_base = chunk_base + slice.begin;
      // Lists are in cluster order, not pin order; their row map turns a layout row back into
      // the pin row every later stage reads the corpus by.
      auto const* id_map = _lists != nullptr
                             ? static_cast<const std::int64_t*>(_lists->row_ids->data()) + id_base
                             : nullptr;

      if (sweep == 1 || s.seed) {
        bounded_slice const sl =
          s.f16_bounded
            ? bounded_slice{reinterpret_cast<std::uint16_t const*>(compact) + slice.begin * dim,
                            reinterpret_cast<float const*>(chunk_norms) + slice.begin,
                            slice_rows,
                            id_base,
                            nullptr,
                            rows_c,
                            m}
            : bounded_slice{reinterpret_cast<std::int8_t const*>(compact) + slice.begin * dim,
                            reinterpret_cast<std::int32_t const*>(chunk_norms) + slice.begin,
                            slice_rows,
                            id_base,
                            s.rescore ? nullptr : id_map,
                            rows_c,
                            m};
        if (s.stretch_sweep0 && sweep == 0) {
          // The slice's rows, from row `at` of its cluster, cut where the cluster's stretches
          // end; each piece waits for its stretch's round (after this loop).
          auto const c    = static_cast<std::size_t>(slice.cluster);
          auto const at   = id_base - _lists->offsets[c];
          auto const ends = stretch_ends(_lists->offsets[c + 1] - _lists->offsets[c]);
          if (stretches.size() < ends.size()) { stretches.resize(ends.size()); }
          for (std::size_t i = 0; i < ends.size(); ++i) {
            auto const from = std::max(i == 0 ? std::int64_t{0} : ends[i - 1], at);
            auto const to   = std::min(ends[i], at + slice_rows);
            if (from >= to) { continue; }
            auto const o = from - at;
            auto piece   = sl;
            piece.x      = static_cast<std::byte const*>(sl.x) + o * dim * (s.f16_bounded ? 2 : 1);
            piece.x_sq   = static_cast<std::byte const*>(sl.x_sq) + o * 4;
            piece.n      = to - from;
            piece.id_base += o;
            if (piece.id_map != nullptr) { piece.id_map += o; }
            stretches[i].push_back(piece);
          }
          continue;
        }
        if (s.running_bound_rows > 0) {
          for (std::int64_t o = 0; o < slice_rows;) {
            auto const n = std::min(slice_rows - o, w.running_piece - w.since_merge);
            auto piece   = sl;
            piece.x      = static_cast<std::byte const*>(sl.x) + o * dim * (s.f16_bounded ? 2 : 1);
            piece.x_sq   = static_cast<std::byte const*>(sl.x_sq) + o * 4;
            piece.n      = n;
            piece.id_base += o;
            if (piece.id_map != nullptr) { piece.id_map += o; }
            launch_bounded(piece);
            ++w.bounded_launches;
            w.bounded_padded += ((n + 127) / 128) * ((m + 127) / 128);
            group.push_back(piece);
            o += n;
            w.since_merge += n;
            if (w.since_merge >= w.running_piece) {
              check_group();
              if (s.hold_codes_pairs) {
                hold_trim();
              } else {
                flush();
              }
              w.since_merge   = 0;
              w.running_piece = std::min<std::int64_t>(2 * w.running_piece, std::int64_t{1} << 22);
            } else if (group.size() >= s.bounded_group) {
              check_group();
            }
          }
          continue;
        }
        if (joins_group(m, slice_rows)) {
          deferred.push_back(sl);
        } else {
          launch_bounded(sl);
        }
        ++w.bounded_launches;
        w.bounded_padded += ((slice_rows + 127) / 128) * ((m + 127) / 128);
        group.push_back(sl);
        if (group.size() >= s.bounded_group) { check_group(); }
        continue;
      }

      if (radius_join && pair_step != nullptr) {
        if (!search_slice_of_pair(
              run, slice_view, rows_c, m, id_base, id_map, run == first_run ? pair_step->row : 0)) {
          break;
        }
        continue;
      }

      if (radius_join) {
        // Same construction as the exhaustive radius path: a slice's in-range pairs are
        // final when produced, so they are appended, never folded, and there is no k. The
        // kernel numbers query rows within the gathered matrix; rows_c maps them back.
        auto edges = threshold_search(res,
                                      slice_view,
                                      queries_view,
                                      s.search_radius,
                                      s.ranked_metric,
                                      mr,
                                      0,
                                      count_only && !keyed);
        if (count_only && !keyed) {
          note_radius_part(edges.n_edges);
        } else if (edges.n_edges > 0) {
          auto left =
            cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                      vss::column_size(edges.n_edges, "vector join threshold"),
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
          if (best_only) {
            fold_best(left->view().data<std::int32_t>(),
                      edges.neighbors->view().data<std::int64_t>(),
                      edges.distances->view().data<float>(),
                      edges.n_edges,
                      std::numeric_limits<float>::infinity(),
                      st.pin_keys,
                      nullptr);
            continue;
          }
          if (keyed) {
            keep_keyed(left->view().data<std::int32_t>(),
                       edges.neighbors->view().data<std::int64_t>(),
                       edges.distances->view().data<float>(),
                       edges.n_edges);
            continue;
          }
          found.left.push_back(std::move(left));
          found.neighbors.push_back(std::move(edges.neighbors));
          found.distances.push_back(std::move(edges.distances));
          note_radius_part(edges.n_edges);
        }
        continue;
      }

      // The GEMM-ranked search, as in the exhaustive fold: it also carries none of cuVS's
      // per-call device synchronization, which is paid here once per slice.
      auto const knn = [&] {
        if (s.f16_sweep0) {
          vss::gather_bytes(s.probe_f16->data(),
                            dim * static_cast<std::int64_t>(sizeof(std::uint16_t)),
                            rows_c,
                            m,
                            routed_f16->data(),
                            stream);
          vss::gather_rows(s.probe_sqf->data(), 1, rows_c, m, routed_sqf->data(), stream);
          return vss::gemm_f16_topk(
            res,
            reinterpret_cast<std::uint16_t const*>(compact) + slice.begin * dim,
            reinterpret_cast<float const*>(chunk_norms) + slice.begin,
            slice_rows,
            routed_f16->data(),
            routed_sqf->data(),
            m,
            dim,
            k_eff,
            /*take_sqrt=*/false,
            mr);
        }
        if (s.int8_search) {
          vss::gather_bytes(s.probe_i8->data(), dim, rows_c, m, routed_i8->data(), stream);
          vss::gather_int32(s.probe_sq->data(), rows_c, m, routed_sq->data(), stream);
          return vss::gemm_int8_topk(
            res,
            reinterpret_cast<std::int8_t const*>(compact) + slice.begin * dim,
            reinterpret_cast<std::int32_t const*>(chunk_norms) + slice.begin,
            slice_rows,
            routed_i8->data(),
            routed_sq->data(),
            m,
            dim,
            k_eff,
            s.search_metric == cuvs::distance::DistanceType::L2SqrtExpanded && !s.bounded,
            mr);
        }
        // A re-scored sweep 0 only sets each row's bound, so it keeps the GEMM.
        auto const ranked = s.rescore ? s.search_metric : s.ranked_metric;
        return gemm_search_enabled() && vss::gemm_search_supports(ranked)
                 ? vss::gemm_topk(res, slice_view, queries_view, k_eff, ranked, mr)
                 : vss::brute_force_knn_untrimmed(
                     res, vss::brute_force_build(res, slice_view, ranked), queries_view, k_eff, mr);
      }();

      // Over a scanned corpus this sweep only sets each row's bound, the largest of its k
      // distances, so they are made exact first, read from the slice's rows. The accumulator
      // may then be out of order; only that largest distance is read before it is reset.
      if (s.scan_f16) {
        auto const s = static_cast<std::size_t>(&slice - _op._chunk_cluster_runs[j].data());
        vss::exact_distances(routed_queries->data(),
                             nullptr,
                             knn.k,
                             knn.neighbors->view().data<std::int64_t>(),
                             knn.distances->mutable_view().data<float>(),
                             m * knn.k,
                             scan_source(s),
                             dim,
                             stream,
                             mr,
                             nullptr,
                             0.f,
                             ip);
      }
      // Fold in place: a row is routed to a cluster at most once, so the rows of one slice
      // are distinct and no two threads of the fold write the same accumulator row.
      vss::fold_topk_rows(st.acc_distances->mutable_view().data<float>(),
                          st.acc_neighbors->mutable_view().data<std::int64_t>(),
                          k_join,
                          knn.distances->view().data<float>(),
                          knn.neighbors->view().data<std::int64_t>(),
                          knn.k,
                          k_eff,
                          rows_c,
                          m,
                          id_base,
                          stream,
                          s.rescore ? nullptr : id_map);
    }

    // The searches above are issued, not finished. Staging the next needed chunk now runs its
    // H2D while the GPU works on this one.
    if (s.lists_on_host) { launch_deferred(); }
    prefetched = pair_step == nullptr && ci + 1 < chunks->size() ? stage_next((*chunks)[ci + 1])
                                                                 : staged_vector_chunk{};
    if (s.keep_chunks && sweep == 0 && can_keep()) {
      kept.emplace(j, std::move(staged));
    } else {
      // A group that overflowed replays its slices, which read this chunk's staged rows: check
      // it while they are still there. A scanned chunk's candidates are re-scored from its
      // rows, so they are merged before it goes too.
      if (s.scan_f16 && sweep == 1) {
        check_group();
        flush();
        exact_source = vss::exact_rows{};
      } else if (s.any_bounded && (staged.buffer || staged.owner || staged.reader)) {
        check_group();
      }
      release_staged(staged);
    }
  }

  /// One cluster's rows of this chunk against the block rows routed to it, a part of them at a
  /// time, from @p first_row on. False when the task should stop here, with @ref stopped_at saying
  /// where.
  [[nodiscard]] bool search_slice_of_pair(std::size_t run,
                                          vss::dataset_matrix_view slice_view,
                                          std::int64_t const* rows_c,
                                          std::int64_t m,
                                          std::int64_t id_base,
                                          std::int64_t const* id_map,
                                          std::int64_t first_row)
  {
    auto call_rows = first_call_rows();
    for (auto at = first_row; at < m;) {
      auto const rows =
        std::min({call_rows, m - at, static_cast<std::int64_t>(routed_queries->size()) / dim});
      vss::gather_rows(
        s.search_queries.data_handle(), dim, rows_c + at, rows, routed_queries->data(), stream);
      auto const queries =
        raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
          routed_queries->data(), rows, dim);
      std::optional<vss::threshold_join_result> edges;
      try {
        edges.emplace(threshold_search(
          res, slice_view, queries, s.search_radius, s.ranked_metric, mr, 0, count_only));
      } catch (std::exception const& shortage) {
        if (!search_can_shrink(shortage)) { throw; }
        stream.sync();
        if (found.held > 0) {
          stopped_at = rest_of_chunk(run, at, call_rows);
          return false;
        }
        if (rows == 1) { throw; }
        call_rows = rows / 2;
        continue;
      }
      call_rows = rows_for_next_call(call_rows, rows, edges->n_edges);
      if (count_only) {
        note_radius_part(edges->n_edges);
      } else if (edges->n_edges > 0) {
        auto left =
          cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                    vss::column_size(edges->n_edges, "vector join threshold"),
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr);
        vss::remap_radius_edges(edges->query_rows->view().data<std::int64_t>(),
                                rows_c + at,
                                left->mutable_view().data<std::int32_t>(),
                                edges->neighbors->mutable_view().data<std::int64_t>(),
                                edges->n_edges,
                                id_base,
                                stream,
                                id_map);
        auto const pairs = edges->n_edges;
        found.left.push_back(std::move(left));
        found.neighbors.push_back(std::move(edges->neighbors));
        found.distances.push_back(std::move(edges->distances));
        note_radius_part(pairs, /*may_stop=*/true);
      }
      at += rows;
      if (at < m && holds_enough_to_stop()) {
        stopped_at = rest_of_chunk(run, at, call_rows);
        return false;
      }
    }
    return true;
  }

  void run_stretches()
  {
    // The stretched sweep 0: round i filters stretch i of every cluster, under the bounds the
    // merge after round i - 1 left (the seed for round 0). The device lists stay put, so the
    // pieces' rows are where the slices left them. A group grows past bounded_group slices while
    // even all of its pairs would fit the buffer, so it cannot overflow: with few probe rows a
    // round is a few launches, not one per bounded_group clusters (10k probes, 4,096 clusters:
    // 16 checks a round, each reading the buffer's fill back).
    for (auto const& round : stretches) {
      auto const emitted_before = w.bounded_emitted;
      std::int64_t group_pairs  = 0;  // the group's pairs, were every one to pass
      for (auto const& piece : round) {
        if (joins_group(piece.m, piece.n)) {
          deferred.push_back(piece);
        } else {
          launch_bounded(piece);
        }
        ++w.bounded_launches;
        w.bounded_padded += ((piece.n + 127) / 128) * ((piece.m + 127) / 128);
        group.push_back(piece);
        group_pairs += piece.m * piece.n;
        if (group.size() >= s.bounded_group && group_pairs > w.candidates->capacity() - w.pending) {
          check_group();
          group_pairs = 0;
        }
      }
      check_group();
      flush();
      if (dbg) { w.stretch_pairs.push_back(w.bounded_emitted - emitted_before); }
    }
  }

  /// Takes in what is still waiting, and leaves the accumulators as the answer reads them.
  void finish()
  {
    if (s.any_bounded && !kept.empty()) { check_group(); }
    for (auto& entry : kept) {
      release_staged(entry.second);
    }
    kept.clear();
    if (s.any_bounded) {
      check_group();
      if (w.host_pairs.empty()) {
        flush();
      } else {
        bring_pairs_back();
      }
      // Lists hold layout rows; a scanned corpus's candidates already carry corpus rows.
      if (s.rescore && !radius_join && _lists != nullptr) {
        vss::map_ids(st.acc_neighbors->mutable_view().data<std::int64_t>(),
                     n_left * k_join,
                     static_cast<std::int64_t const*>(_lists->row_ids->data()),
                     stream);
      }
      if (dbg) {
        std::fprintf(stderr,
                     "[vecjoin-phase] bounded: %lld pairs passed, %lld merges, %lld launches, "
                     "%lld 128x128 tiles; re-score %.3f s, take/merge %.3f s\n",
                     static_cast<long long>(w.bounded_emitted),
                     static_cast<long long>(w.bounded_merges),
                     static_cast<long long>(w.bounded_launches),
                     static_cast<long long>(w.bounded_padded),
                     w.rescore_seconds,
                     w.take_seconds);
        if (w.pair_moves > 0 || w.slice_cuts > 0 || w.budget_flushes > 0) {
          std::fprintf(stderr,
                       "[vecjoin-phase] pairs to host: %lld moves, %lld pairs, %lld slice cuts, "
                       "%lld flushes past the budget, %.3f s\n",
                       static_cast<long long>(w.pair_moves),
                       static_cast<long long>(w.pairs_moved),
                       static_cast<long long>(w.slice_cuts),
                       static_cast<long long>(w.budget_flushes),
                       w.host_seconds);
        }
        if (s.f16_bounded) {
          auto const l = vss::f16_filter_launch_counts();
          std::fprintf(stderr,
                       "[vecjoin-phase] FP16 filter launches so far in this process: CUTLASS %lld "
                       "single, %lld grouped; fused %lld single\n",
                       static_cast<long long>(l.cutlass_single),
                       static_cast<long long>(l.cutlass_group),
                       static_cast<long long>(l.fused_single));
        }
        if (best_only) {
          std::fprintf(stderr,
                       "[vecjoin-phase] best match: %lld of those pairs re-scored\n",
                       static_cast<long long>(st.best_rescored));
        }
        if (s.gap_rescore) {
          std::fprintf(
            stderr,
            "[vecjoin-phase] gap re-score: %lld of %lld candidates re-scored (trim %.3f s); "
            "row-stretches %llu: %llu under 2k, %llu trimmed at m=2, %llu at m=4, %llu "
            "fall back; the m-rule would re-score %llu\n",
            static_cast<long long>(w.gap_rescored),
            static_cast<long long>(w.bounded_emitted),
            w.trim_seconds,
            w.gap_stats.rows,
            w.gap_stats.short_rows,
            w.gap_stats.m2_rows,
            w.gap_stats.m4_rows,
            w.gap_stats.fallback_rows,
            w.gap_stats.literal_rescored);
        }
      }
      if (s.gap_stats_on) { vss::add_gap_rescore_totals(w.gap_stats); }
      if (s.sqrt_at_end && !radius_join) {
        vss::sqrt_in_place(st.acc_distances->mutable_view().data<float>(), n_left * k_join, stream);
      }
    }
    if (best_only) {
      st.best_ids_are_layout = s.any_bounded && s.rescore;
      st.best_take_sqrt      = s.sqrt_at_end;
      emit_best();
    }
    if (s.unit_cosine || s.direct_cosine) {
      if (radius_join) {
        for (auto& part : found.distances) {
          vss::scale_in_place(part->mutable_view().data<float>(), part->size(), 0.5f, stream);
        }
      } else {
        vss::scale_in_place(
          st.acc_distances->mutable_view().data<float>(), n_left * k_join, 0.5f, stream);
      }
    }

    // What the pruning did, reported as a value rather than only a print: an approximate join
    // that skipped nothing returns a correct answer, so nothing in the result set distinguishes
    // it from one that pruned hard. This is what a test can assert on.
    auto const exhaustive_pairs = n_left * _op._right_total_rows;
    if (_op._sirius_ctx != nullptr) {
      _op._sirius_ctx->record_vector_join_prune(static_cast<std::uint64_t>(w.scanned_pairs),
                                                static_cast<std::uint64_t>(exhaustive_pairs),
                                                s.needed_chunks.size(),
                                                _op._chunk_cluster_runs.size());
    }

    if (std::getenv("SIRIUS_VECTOR_JOIN_PRUNE_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[vecjoin] batch: %lld probe rows x %ld probes, %zu of %zu corpus chunks "
                   "staged, %lld of %lld probe-row x corpus-row pairs scored (%.2f%%)\n",
                   static_cast<long long>(n_left),
                   static_cast<long>(s.n_probes),
                   s.needed_chunks.size(),
                   _op._chunk_cluster_runs.size(),
                   static_cast<long long>(w.scanned_pairs),
                   static_cast<long long>(exhaustive_pairs),
                   exhaustive_pairs == 0 ? 0.0
                                         : 100.0 * static_cast<double>(w.scanned_pairs) /
                                             static_cast<double>(exhaustive_pairs));
    }
  }

  block_state::search_state& s;
  block_state::search_run& w;
  bool dbg;
  std::chrono::steady_clock::time_point phase_t0;
  // The gather buffers: a slice's probe rows as its search reads them.
  std::optional<rmm::device_uvector<float>> routed_queries;
  std::optional<rmm::cuda_stream> staging_stream;
  rmm::cuda_stream_view stage_on;
  borrow_window borrowed;
  std::optional<rmm::device_uvector<std::int8_t>> routed_i8;
  std::optional<rmm::device_uvector<std::int32_t>> routed_sq;
  float row_half_error{0.f};
  float row_norm_max{0.f};
  std::optional<rmm::device_uvector<std::uint16_t>> routed_f16;
  std::optional<rmm::device_uvector<float>> routed_sqf;
  vss::exact_rows exact_source{};
  std::vector<bounded_slice> group, deferred;
  staged_vector_chunk prefetched;
  // The compact chunks sweep 0 copied in stay on the device for sweep 1 while the budget still
  // has a margin free, and sweep 1 starts with them: lists that nearly fit are read about once.
  std::map<std::size_t, staged_vector_chunk> kept;
  std::vector<std::size_t> kept_first;

  // The pass under way.
  int sweep{0};
  std::vector<std::size_t> const* chunks{nullptr};
  bool direct{false};
  bool widen_compact{false};
  bool copy_compact{false};
  bool grouped_sweep0{false};
  std::vector<chunk_cluster_run> const no_runs;
  // The stretched sweep 0's pieces, by stretch (see stretch_sweep0).
  std::vector<std::vector<bounded_slice>> stretches;
  // A build task leaves out what only a walk over chunks reads.
  bool building_for_pairs{false};
  std::optional<vss::pair_schedule::step> stopped_at;
};

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::next_pair_step_locked()
{
  if (!_pair_schedule->peek()) { return nullptr; }
  {
    std::lock_guard<std::mutex> admission_lock(_admission->mutex);
    if (_admission->in_flight >= _admission->limit) { return nullptr; }
    ++_admission->in_flight;
  }
  auto ticket       = std::make_shared<admission_ticket>(_admission);
  auto const step   = _pair_schedule->take();
  auto const& block = _blocks[step.block];
  std::size_t bytes = 0;
  if (step.kind == vss::pair_schedule::step_kind::build) {
    bytes = static_cast<std::size_t>(static_cast<double>(block.rows) * pair_block_row_bytes(true)) +
            _max_probe_chunk_bytes + vss::gemm_search_tile_bytes();
  } else {
    auto const& state = *_pair_states[step.block];
    bytes             = pair_task_bytes(
      std::min(state.search ? state.search->max_routed : block.rows, pair_call_rows(_request.dim)));
  }
  auto input =
    std::make_unique<vector_join_stream_input>(block.pieces.front().chunk, bytes, step.block);
  input->set_pair_step(step);
  input->set_ticket(std::move(ticket));
  if (_pair_device) { input->set_preferred_device_id(*_pair_device); }
  return input;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::run_build_step(
  vector_join_stream_input const& input,
  ::cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  auto const b = input.pair_step()->block;
  auto st      = std::make_shared<block_state>();
  found_pairs none;
  block_task task{*this, *st, none, input, space, stream};
  task.load_rows();
  task.refuse_what_cannot_run();
  auto chunks = _corpus->num_chunks();
  if (_centroids != nullptr) {
    clustered_search search{task};
    search.prepare_for_pairs();
    chunks = st->search->needed_chunks.size();
  } else {
    task.prepare_rows_for_pairs();
  }
  // Other tasks read all this on their own streams.
  stream.sync();
  inject_pair_oom();
  auto answer = task.no_answer();
  std::lock_guard<std::mutex> lg(_op_mutex);
  if (!_pair_device) { _pair_device = space.get_device_id(); }
  if (!_pair_schedule->built(b, chunks)) { _pair_states[b] = std::move(st); }
  if (_sirius_ctx != nullptr) { _sirius_ctx->record_vector_join_block_built(); }
  return answer;
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::run_pair_step(
  vector_join_stream_input const& input,
  ::cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  auto const& step = *input.pair_step();
  std::shared_ptr<block_state> state;
  {
    std::lock_guard<std::mutex> lg(_op_mutex);
    state = _pair_states[step.block];
  }
  if (_sirius_ctx != nullptr) { _sirius_ctx->record_vector_join_pair_attempt(); }
  std::optional<vss::pair_schedule::step> rest;
  std::unique_ptr<operator_data> answer;
  {
    found_pairs found;
    block_task task{*this, *state, found, input, space, stream};
    task.prepare_answer();
    if (_centroids != nullptr) {
      block_state::search_run run{_host_pair_blocks};
      clustered_search search{task, run};
      rest = search.run_pair();
    } else {
      rest = task.search_pair();
    }
    inject_pair_oom();
    answer = task.pair_answer();
    stream.sync();
  }
  // From here on the pairs are this task's output: a retry resumes after this operator, so the
  // schedule moves on exactly once for them. Nothing below can run out of memory.
  std::lock_guard<std::mutex> lg(_op_mutex);
  if (rest) {
    _pair_schedule->cut_short(*rest);
    if (_sirius_ctx != nullptr) { _sirius_ctx->record_vector_join_pair_cut_short(); }
  } else {
    if (_pair_schedule->pair_done(step.block)) { _pair_states[step.block].reset(); }
    if (_sirius_ctx != nullptr) { _sirius_ctx->record_vector_join_pair_finished(); }
  }
  return answer;
}

void sirius_physical_vector_join_stream::inject_pair_oom()
{
  if (auto const* slow = std::getenv("SIRIUS_VSS_PAIR_SLOW_MS"); slow != nullptr) {
    std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(slow)));
  }
  auto const* v = std::getenv("SIRIUS_VSS_PAIR_OOM_ONE_IN");
  if (v == nullptr) { return; }
  auto const one_in = std::strtoull(v, nullptr, 10);
  if (one_in == 0) { return; }
  // Splitmix64 of a counter: scattered over tasks, and the same run to run for one schedule.
  auto x = (_pair_oom_draws.fetch_add(1) + 1) * 0x9E3779B97F4A7C15ull;
  x      = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x      = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  if (x % one_in == 0) { throw rmm::out_of_memory("injected vector join pair task shortage"); }
}

std::unique_ptr<operator_data> sirius_physical_vector_join_stream::execute(
  const operator_data& input_data, ::cuda::stream_ref stream_ref)
{
  // First, so it also covers what this task frees on the way out.
  vector_join_debug_wall_time const task_wall_time{"stream task"};
  rmm::cuda_stream_view stream{stream_ref};
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_stream::execute"};
  {
    std::lock_guard<std::mutex> lg(_op_mutex);
    if (_init_error) { std::rethrow_exception(_init_error); }
  }

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

  if (_request.probe_scalar && _scalar_probe_rows != 1) {
    throw std::runtime_error(
      _scalar_probe_rows > 1
        ? "More than one row returned by a subquery used as an expression - scalar subqueries can "
          "only return a single row."
        : "sirius vector join: the scalar subquery giving the query vector returned no rows");
  }

  if (auto const& step = join_in->pair_step()) {
    return step->kind == vss::pair_schedule::step_kind::build
             ? run_build_step(*join_in, *mem_space, stream)
             : run_pair_step(*join_in, *mem_space, stream);
  }

  block_state st;
  found_pairs found;
  block_task task{*this, st, found, *join_in, *mem_space, stream};
  task.load_rows();
  // An empty probe batch joins to nothing; the search libraries refuse a zero-row query matrix.
  if (st.n_left == 0) { return task.empty_answer(); }
  task.refuse_what_cannot_run();
  task.prepare_answer();
  // The clustered path replaces the whole-corpus fold below. It is a separate branch rather
  // than a predicate inside it because the two iterate different things: the exhaustive fold
  // walks corpus chunks, while this walks (probe run x neighbouring cluster) pairs.
  if (_centroids != nullptr) {
    clustered_search search{task};
    search.run();
  } else {
    task.search_corpus();
  }
  return task.shape_answer();
}

//===----------------------------------------------------------------------===//
// Sink
//===----------------------------------------------------------------------===//
void sirius_physical_vector_join_stream::sink(const operator_data& output_data,
                                              ::cuda::stream_ref /*stream*/)
{
  auto const& part         = dynamic_cast<const partitioned_operator_data&>(output_data);
  auto const partition_idx = part.get_partition_idx().value_or(0);
  auto const* block        = dynamic_cast<const vector_join_stream_block_output*>(&output_data);
  auto const& out          = part.get_data_batches();
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto const& batch = out[i];
    for (auto& next_port_info : next_port_after_sink) {
      auto* consumer =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (consumer == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_stream::sink] next operator is not a partition consumer");
      }
      consumer->push_data_batch_partitioned(
        next_port_info.next_operator_port_name,
        batch,
        block != nullptr ? block->partitions()[i] : partition_idx);
    }
  }
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_vector_join_stream::per_left_batch_estimate(std::size_t left_idx,
                                                                        bool with_routing) const
{
  auto const n_left = _probe->chunk_rows(left_idx);
  // No reservation holds an output past a column's row limit: refuse here, once, instead of
  // through every out-of-memory retry of the task.
  auto const k_join =
    _right_total_rows > 0 ? std::min<std::int64_t>(_request.k, _right_total_rows) : _request.k;
  vss::column_size(static_cast<std::int64_t>(n_left) * k_join, "vector join output");
  return estimate_for_rows(n_left, with_routing);
}

std::size_t sirius_physical_vector_join_stream::host_pair_budget() const
{
  if (auto const* v = std::getenv("SIRIUS_VSS_PAIRS_TO_HOST_BUDGET"); v != nullptr) {
    return static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
  }
  if (_sirius_ctx == nullptr) { return 0; }
  auto const hosts =
    _sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (hosts.empty()) { return 0; }
  // Half the pool's size is all this join may hold beside the pool, so a block's rows there, as
  // many as may be in flight, come out of it first.
  std::size_t rows_held = 0;
  if (block_rows_on_host()) {
    std::int64_t widest = 0;
    for (auto const& block : _blocks) {
      widest = std::max(widest, block.rows);
    }
    rows_held = static_cast<std::size_t>(widest) * static_cast<std::size_t>(_request.dim) *
                sizeof(float) * scanned_block_tasks();
  }
  auto const half = hosts.front()->get_max_memory() / 2;
  return half > rows_held ? half - rows_held : 0;
}

bool sirius_physical_vector_join_stream::block_rows_on_host() const
{
  auto const* off = std::getenv("SIRIUS_VSS_BLOCK_ROWS_ON_HOST");
  if (off != nullptr && std::string_view{off} == "0") { return false; }
  auto const* bounded = std::getenv("SIRIUS_VSS_BOUND_GEMM");
  return _lists == nullptr && _centroids != nullptr && _request.metric == "l2" &&
         _request.mode != vss::vector_join_mode::threshold && _request.earlier_by.empty() &&
         _request.n_probes > 1 && gemm_search_enabled() && scan_bounded_enabled() &&
         (bounded == nullptr || std::strcmp(bounded, "0") != 0) &&
         vss::bound_filter_f16_supports(_request.dim);
}

std::size_t sirius_physical_vector_join_stream::estimate_for_rows(std::size_t n_left,
                                                                  bool with_routing,
                                                                  bool planning_a_block) const
{
  // Live at once: the accumulator, one batch's partial, and the stacked pair the
  // merge reads (2x), plus the merge output. Six [n_left x k] blocks covers it, with
  // the same 1 MiB floor the split design used. Notably independent of the right
  // batch count -- the split design's merge stage scaled with it.
  auto const k     = static_cast<std::size_t>(std::max<std::int64_t>(
    _right_total_rows > 0 ? std::min<std::int64_t>(_request.k, _right_total_rows) : _request.k, 1));
  auto const block = n_left * k * (sizeof(std::int64_t) + sizeof(float));

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
  // corpus chunk rather than instead of it. A probe block holds its rows joined into one matrix,
  // after the chunks they were cut from are dropped.
  // A block with its rows on the host holds none of this while it searches: the chunks it was cut
  // from are gone before the search starts. It holds a slice of rows for the re-score instead.
  bool const rows_on_host = (planning_a_block || !_blocks.empty()) && block_rows_on_host();
  if (rows_on_host) {
    staged_chunk += std::size_t{256} << 20;
  } else {
    if (_probe && _probe->is_streaming()) { staged_chunk += _max_probe_chunk_bytes; }
    if (!_blocks.empty()) {
      staged_chunk += n_left * static_cast<std::size_t>(_request.dim) * sizeof(float);
    }
  }

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
    // A clustered search over a scanned corpus prepares one slice's probes at a time, a small part
    // of a block (1,024-d, k 10, p32: measured peak ~1.7 GB + 10 KB a probe row).
    auto const prepared_probe_rows =
      _centroids != nullptr && _lists == nullptr ? std::min(n_left, max_chunk_rows) : n_left;
    gemm_scratch =
      vss::gemm_search_tile_bytes() + max_chunk_rows * row_bytes + prepared_probe_rows * row_bytes;
    if (_request.mode == vss::vector_join_mode::threshold) {
      gemm_scratch += (std::size_t{1} << 20) * (2 * sizeof(std::int64_t) + sizeof(float));
    }
  }

  // Clustered search: each probe row's routing edges (~45 B an edge across assignment and the
  // sort that orders them), and on the bounded paths the candidate buffer and its merge.
  std::size_t routed = 0;
  if (with_routing && _centroids != nullptr) {
    auto const probes = static_cast<std::size_t>(std::max<std::int64_t>(_request.n_probes, 1));
    // Sorting them takes more, but on the host rows' path before the FP16 copy is made.
    routed = n_left * probes * (rows_on_host ? sizeof(std::int64_t) : 45);
    // A scanned corpus searched like FLOAT16 lists also holds the probe rounded to half and each
    // staged chunk's half copy and norms.
    bool const scan_bounded = _lists == nullptr && _request.metric == "l2" && probes > 1 &&
                              _request.mode != vss::vector_join_mode::threshold &&
                              scan_bounded_enabled();
    if (_lists != nullptr || scan_bounded) {
      routed += std::max<std::size_t>(std::size_t{1} << 22, 4 * n_left * k) * 16 + n_left * k * 32;
    }
    if (scan_bounded) {
      std::size_t max_chunk_rows = 0;
      for (std::size_t i = 0; _corpus && i < _corpus->num_chunks(); ++i) {
        max_chunk_rows = std::max(max_chunk_rows, _corpus->chunk_rows(i));
      }
      auto const half_row = static_cast<std::size_t>(_request.dim) * sizeof(std::uint16_t);
      routed +=
        max_chunk_rows * (half_row + sizeof(float)) + n_left * (half_row + 4 * sizeof(float));
    }
  }

  return (block * 6) + std::max(cuvs_scratch, gemm_scratch) + staged_chunk + routed +
         (std::size_t{1} << 20);
}

std::size_t sirius_physical_vector_join_stream::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // As in the split design, cuVS's on-demand search scratch is not modelled here.
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string_view sirius_physical_vector_join_stream::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::CONCAT) {
    // Children follow the probe-then-corpus order, and a side read from its pin has no child.
    bool const probe_is_first = _request.probe_from_scan && !_request.probe_shares_build;
    return probe_is_first && children[0].get() == &producer ? "default" : "build";
  }
  if (producer.type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK) {
    return producer.Cast<sirius_physical_passthrough_sink>().union_port_label();
  }
  return sirius_physical_operator::input_port_for(producer);
}

std::string sirius_physical_vector_join_stream::params_to_string() const
{
  return _request.left.table + "(" + _request.left.column + ") x " + _request.right.table + "(" +
         _request.right.column + ") metric=" + _request.metric +
         " k=" + std::to_string(_request.k) +
         " mode=" + std::to_string(static_cast<int>(_request.mode)) +
         (_request.best_only ? " keep=best" : "") +
         (_request.earlier_by.empty()
            ? ""
            : (_request.key_differs ? " other_by=" : " earlier_by=") +
                (_request.key_is_rowid ? std::string("rowid") : _request.earlier_by)) +
         " streaming";
}

}  // namespace sirius::op
