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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::vss {

/// A reservation of @p bytes in @p space. Nothing but the downgrade executor moves what fills a
/// space: a task's own request does not, and a blocking wait for memory no one is freeing never
/// returns. So this asks it directly, the way the pipeline executor does for a short task
/// reservation, until the bytes fit or there is nothing left it may spill. Null if they never fit.
std::unique_ptr<cucascade::memory::reservation> reserve_or_spill(
  duckdb::SiriusContext* ctx, cucascade::memory::memory_space& space, std::size_t bytes);

/// One probe block's FP32 rows in the host pool, as engine batches of a fixed number of rows in a
/// repository the downgrade executor sweeps. Between reads they are idle, so the engine counts them
/// and may take them to disk; a read brings a spilled batch back into the pool first. They leave
/// the repository when this object is destroyed, on every path out of a task, so a retried task
/// starts clean and the pool blocks or spill files go with the last reference.
///
/// Row ranges rather than one batch per probe chunk: a chunk can be much smaller than a pool block,
/// and a read of a few rows should not have to restore a whole large chunk.
class block_host_rows {
 public:
  block_host_rows(cucascade::shared_data_repository& repository,
                  cucascade::memory::memory_space& host_space,
                  duckdb::SiriusContext& ctx,
                  std::int64_t rows,
                  std::int64_t dim,
                  rmm::cuda_stream_view stream);
  ~block_host_rows();

  block_host_rows(block_host_rows const&)            = delete;
  block_host_rows& operator=(block_host_rows const&) = delete;

  /// Whether the pool has room for the whole block, spilling for it if a disk tier allows.
  [[nodiscard]] bool make_room_for_block();

  /// Appends @p rows device rows in order, complete when the call returns. False, with no error,
  /// when the pool has no room for them. A host shortage is not reported as rmm::out_of_memory,
  /// which the engine answers by raising a device reservation that cannot help.
  [[nodiscard]] bool append(float const* device_rows, std::int64_t rows);

  /// Copies rows [@p row, @p row + @p rows) to device memory, complete when the call returns.
  void copy_to_device(std::int64_t row, std::int64_t rows, float* device_dst);

  /// Copies @p n rows, in any order and with repeats, to @p out. Each batch is locked once.
  void gather(std::int64_t const* rows, std::size_t n, float* out);
  void gather(std::int32_t const* rows, std::size_t n, float* out);

 private:
  [[nodiscard]] bool make_room(std::size_t bytes);
  void seal_open_part();
  template <typename Read>
  void with_part(std::size_t part, Read&& read);
  template <typename Row>
  void gather_rows(Row const* rows, std::size_t n, float* out);

  cucascade::shared_data_repository& _repository;
  cucascade::memory::memory_space& _host_space;
  duckdb::SiriusContext& _ctx;
  std::int64_t _rows;
  std::int64_t _dim;
  std::size_t _row_bytes;
  std::int64_t _rows_per_part;
  rmm::cuda_stream_view _stream;
  std::vector<std::shared_ptr<cucascade::data_batch>> _parts;
  cucascade::memory::fixed_multiple_blocks_allocation _open;
  std::int64_t _open_capacity{0};
  std::int64_t _open_rows{0};
};

}  // namespace sirius::vss
