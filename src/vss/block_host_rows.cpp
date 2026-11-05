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

#include "vss/block_host_rows.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "sirius_context.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/memory/column_metadata.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

namespace sirius::vss {

namespace {

constexpr std::size_t kPartBytes = std::size_t{256} << 20;

using blocks = cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

/// Calls @p copy(block pointer, bytes before this range, bytes) for each block a byte range
/// crosses.
template <typename Copy>
void for_each_block_range(blocks const& allocation,
                          std::size_t offset,
                          std::size_t bytes,
                          Copy&& copy)
{
  auto const block_bytes = allocation.block_size();
  std::size_t done       = 0;
  while (bytes > 0) {
    auto const in_block = offset % block_bytes;
    auto const n        = std::min(bytes, block_bytes - in_block);
    copy(allocation.at(offset / block_bytes).data() + in_block, done, n);
    offset += n;
    bytes -= n;
    done += n;
  }
}

}  // namespace

std::unique_ptr<cucascade::memory::reservation> reserve_or_spill(
  duckdb::SiriusContext* ctx, cucascade::memory::memory_space& space, std::size_t bytes)
{
  auto res = space.make_reservation_or_null(bytes);
  if (res || ctx == nullptr) { return res; }
  std::unique_ptr<cucascade::memory::reservation> got;
  std::mutex m;
  ctx->get_downgrade_executor(space.get_id())
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

block_host_rows::block_host_rows(cucascade::shared_data_repository& repository,
                                 cucascade::memory::memory_space& host_space,
                                 duckdb::SiriusContext& ctx,
                                 std::int64_t rows,
                                 std::int64_t dim,
                                 rmm::cuda_stream_view stream)
  : _repository(repository),
    _host_space(host_space),
    _ctx(ctx),
    _rows(rows),
    _dim(dim),
    _row_bytes(static_cast<std::size_t>(dim) * sizeof(float)),
    _rows_per_part(std::max<std::int64_t>(
      1,
      static_cast<std::int64_t>(std::min(kPartBytes, host_space.get_max_memory() / 16) /
                                _row_bytes))),
    _stream(stream)
{
}

block_host_rows::~block_host_rows()
{
  // A task that failed with a batch half written may have copies running into its blocks.
  if (_open) { _stream.synchronize_no_throw(); }
  for (auto const& part : _parts) {
    try {
      std::ignore = _repository.pop_data_batch_by_id(part->get_batch_id());
    } catch (...) {
    }
  }
}

bool block_host_rows::make_room(std::size_t bytes)
{
  // Over the space's limit a reservation is an error rather than a null.
  if (bytes > _host_space.get_max_memory()) { return false; }
  auto const& spiller = _ctx.get_downgrade_executor(_host_space.get_id());
  if (!spiller.has_disk_tier()) { return _host_space.make_reservation_or_null(bytes) != nullptr; }
  return reserve_or_spill(&_ctx, _host_space, bytes) != nullptr;
}

bool block_host_rows::make_room_for_block()
{
  return make_room(static_cast<std::size_t>(_rows) * _row_bytes);
}

bool block_host_rows::append(float const* device_rows, std::int64_t rows)
{
  auto* const host_mr =
    _host_space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (host_mr == nullptr) { return false; }
  while (rows > 0) {
    if (!_open) {
      auto const already = static_cast<std::int64_t>(_parts.size()) * _rows_per_part;
      _open_capacity     = std::min(_rows_per_part, _rows - already);
      _open_rows         = 0;
      try {
        _open =
          host_mr->allocate_multiple_blocks(static_cast<std::size_t>(_open_capacity) * _row_bytes);
      } catch (std::bad_alloc const&) {
        return false;
      }
    }
    auto const n = std::min(rows, _open_capacity - _open_rows);
    for_each_block_range(
      *_open,
      static_cast<std::size_t>(_open_rows) * _row_bytes,
      static_cast<std::size_t>(n) * _row_bytes,
      [&](std::byte* block, std::size_t before, std::size_t bytes) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(block,
                                      reinterpret_cast<std::byte const*>(device_rows) + before,
                                      bytes,
                                      cudaMemcpyDeviceToHost,
                                      _stream.get()));
      });
    device_rows += n * _dim;
    rows -= n;
    _open_rows += n;
    _stream.sync();
    if (_open_rows == _open_capacity) { seal_open_part(); }
  }
  return true;
}

void block_host_rows::seal_open_part()
{
  auto const bytes = static_cast<std::size_t>(_open_capacity) * _row_bytes;
  cucascade::memory::column_metadata meta{};
  meta.type_id     = static_cast<std::int32_t>(cudf::type_id::FLOAT32);
  meta.num_rows    = static_cast<std::int32_t>(_open_capacity * _dim);
  meta.null_count  = 0;
  meta.has_data    = true;
  meta.data_offset = 0;
  meta.data_size   = bytes;
  auto table       = cucascade::memory::host_table_allocation::create(
    std::move(_open), std::vector<cucascade::memory::column_metadata>{meta}, bytes);
  auto batch = cucascade::data_batch::make(
    sirius::get_next_batch_id(),
    std::make_unique<cucascade::host_data_representation>(std::move(table), &_host_space));
  _repository.add_data_batch(batch);
  _parts.push_back(std::move(batch));
  _ctx.record_vector_join_host_batch(_repository.total_size());
}

template <typename Read>
void block_host_rows::with_part(std::size_t part, Read&& read)
{
  // A sweep can find every candidate busy for a moment, and room found can be taken before the next
  // try. Past a few tries the pool holds what cannot move, and the error names it rather than an
  // out-of-memory the engine would retry on the device.
  constexpr int kAttempts = 5;
  for (int attempt = 1;; ++attempt) {
    std::size_t bytes = 0;
    {
      // Exclusive for the read alone. Only this task reads these rows, and a shared lock would
      // leave a gap after the restore in which the downgrade executor could take them straight
      // back.
      auto lock = _parts.at(part)->to_mutable();
      if (lock.get_current_tier() != cucascade::memory::Tier::HOST) {
        try {
          lock.convert_to<cucascade::host_data_representation>(
            sirius::converter_registry::get(), &_host_space, ::cuda::stream_ref{_stream.get()});
          _ctx.record_vector_join_host_restore();
        } catch (std::bad_alloc const&) {
          bytes = lock.get_data()->get_size_in_bytes();
        }
      }
      if (bytes == 0) {
        auto const& table =
          lock.get_data()->cast<cucascade::host_data_representation>().get_host_table();
        read(*table->allocation, table->columns.front().data_offset);
        return;
      }
    }
    // With no lock held: the executor's sweep can block on a batch it has picked.
    if (attempt == kAttempts) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_stream] a probe block's rows on disk need " +
        std::to_string(bytes) + " bytes in a host pool of " +
        std::to_string(_host_space.get_max_memory()) + ", and nothing more can be spilled");
    }
    if (!make_room(bytes)) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  }
}

void block_host_rows::copy_to_device(std::int64_t row, std::int64_t rows, float* device_dst)
{
  while (rows > 0) {
    auto const part    = row / _rows_per_part;
    auto const in_part = row - part * _rows_per_part;
    auto const n       = std::min(rows, _rows_per_part - in_part);
    with_part(static_cast<std::size_t>(part), [&](blocks const& allocation, std::size_t base) {
      for_each_block_range(
        allocation,
        base + static_cast<std::size_t>(in_part) * _row_bytes,
        static_cast<std::size_t>(n) * _row_bytes,
        [&](std::byte const* block, std::size_t before, std::size_t bytes) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<std::byte*>(device_dst) + before,
                                        block,
                                        bytes,
                                        cudaMemcpyHostToDevice,
                                        _stream.get()));
        });
      // The blocks are pinned, so the copies are still running and must end under the lock.
      _stream.sync();
    });
    device_dst += n * _dim;
    row += n;
    rows -= n;
  }
}

template <typename Row>
void block_host_rows::gather_rows(Row const* rows, std::size_t n, float* out)
{
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), std::uint32_t{0});
  if (!std::is_sorted(rows, rows + n)) {
    std::sort(order.begin(), order.end(), [rows](std::uint32_t a, std::uint32_t b) {
      return rows[a] < rows[b];
    });
  }
  auto const part_of = [&](std::size_t i) {
    return static_cast<std::int64_t>(rows[order[i]]) / _rows_per_part;
  };
  for (std::size_t i = 0; i < n;) {
    auto const part = part_of(i);
    with_part(static_cast<std::size_t>(part), [&](blocks const& allocation, std::size_t base) {
      for (; i < n && part_of(i) == part; ++i) {
        auto const in_part = static_cast<std::int64_t>(rows[order[i]]) - part * _rows_per_part;
        auto* dst = reinterpret_cast<std::byte*>(out + static_cast<std::int64_t>(order[i]) * _dim);
        for_each_block_range(allocation,
                             base + static_cast<std::size_t>(in_part) * _row_bytes,
                             _row_bytes,
                             [&](std::byte const* block, std::size_t before, std::size_t bytes) {
                               std::memcpy(dst + before, block, bytes);
                             });
      }
    });
  }
}

void block_host_rows::gather(std::int64_t const* rows, std::size_t n, float* out)
{
  gather_rows(rows, n, out);
}

void block_host_rows::gather(std::int32_t const* rows, std::size_t n, float* out)
{
  gather_rows(rows, n, out);
}

}  // namespace sirius::vss
