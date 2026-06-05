/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Global atomic counter for generating unique data batch IDs.
 *
 * This provides a simple way to generate unique IDs for data batches
 * without requiring a data_repository_manager instance.
 */
inline std::atomic<uint64_t> g_next_batch_id{0};

/**
 * @brief Generate a unique data batch ID.
 */
inline uint64_t get_next_batch_id() { return g_next_batch_id++; }

/**
 * @brief Get a cudf::table_view from a read-only data_batch accessor.
 *
 * Assumes the underlying data_batch contains a gpu_table_representation.
 * The caller MUST already hold the read_only_data_batch (shared lock) — this
 * helper deliberately does NOT internally call batch.to_read_only() because
 * doing so would let a misuse pattern hide a P1 self-deadlock (acquiring a
 * read lock on a batch that the caller is about to upgrade to mutable).
 *
 * @param batch The read-only accessor to extract the table view from.
 * @return cudf::table_view The underlying cudf table view.
 */
inline cudf::table_view get_cudf_table_view(const cucascade::read_only_data_batch& batch)
{
  auto* data = batch.get_data();
  if (data == nullptr) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table_view();
}

/**
 * @brief Get a cudf::table_view from an idle data_batch (convenience overload).
 *
 * Acquires a temporary read-only lock, extracts the table_view, then releases the lock.
 *
 * @warning The returned table_view references GPU memory that is only guaranteed stable while a
 * read-only lock is held. Since this function releases the lock before returning, the view can
 * become dangling if another thread downgrades or mutates the batch concurrently. Only use this
 * overload in contexts where the caller has exclusive ownership of the batch (e.g., diagnostic
 * functions running synchronously within a pipeline task). Prefer the
 * get_cudf_table_view(const read_only_data_batch&) overload when the caller can hold the lock.
 *
 * @param batch The idle data batch to extract the table view from.
 * @return cudf::table_view The underlying cudf table view.
 */
// NOLINTNEXTLINE(readability-non-const-parameter) -- to_read_only() is non-const
inline cudf::table_view get_cudf_table_view(cucascade::data_batch& batch)
{
  auto ro    = batch.to_read_only();
  auto* data = ro.get_data();
  if (data == nullptr) { throw std::runtime_error("data_batch has no data representation"); }
  return data->cast<cucascade::gpu_table_representation>().get_table_view();
}

/**
 * @brief Create a shared_ptr<data_batch> from a cudf::table, recording the writer event.
 *
 * STREAM-LINEAGE: @p writer_stream is REQUIRED. Every data_batch carrying a
 * gpu_table_representation is born with a recorded writer event so
 * cucascade::convert_gpu_to_gpu() can call
 * cudaStreamWaitEvent(reader_stream, writer_event, 0) before peer-copying
 * source buffers. This closes the cross-mempool stream-ordered race in
 * multi-GPU runs.
 *
 * @param table The cudf table (will be moved from).
 * @param memory_space The memory space where the table resides.
 * @param writer_stream The stream on which @p table's data was last written.
 *                      MUST be the actual writer stream — passing the wrong
 *                      stream re-opens the race this contract closes.
 * @return std::shared_ptr<cucascade::data_batch> The new data batch.
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  cudf::table&& table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream)
{
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), memory_space, writer_stream);
  return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(gpu_repr));
}

/**
 * @brief Create a shared_ptr<data_batch> from a unique_ptr<cudf::table>, recording the writer
 * event.
 *
 * @copydoc make_data_batch(cudf::table&&, cucascade::memory::memory_space&,
 *                          rmm::cuda_stream_view)
 */
inline std::shared_ptr<cucascade::data_batch> make_data_batch(
  std::unique_ptr<cudf::table> table,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view writer_stream)
{
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), memory_space, writer_stream);
  return std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(gpu_repr));
}

/**
 * @brief True if any column in @p t is a cuDF DICTIONARY32 column.
 *
 * Used by decode-on-receipt guards to cheaply skip the decode path when the
 * batch carries no scan-emitted dictionary columns (the common case).
 */
inline bool table_has_dictionary(cudf::table_view const& t)
{
  for (auto const& c : t) {
    if (c.type().id() == cudf::type_id::DICTIONARY32) return true;
  }
  return false;
}

/**
 * @brief Decode any DICTIONARY32 columns in @p t to their materialized form
 * (STRING for the scan's string dicts); deep-copy the rest. Returns a
 * fully-owned table.
 *
 * This is the decode-on-receipt primitive: operators and the result boundary
 * that are not (yet) dictionary-aware call this before consuming a batch that
 * may carry scan-emitted DICTIONARY32 columns, so they only ever see the
 * materialized types they already handle. Callers should gate on
 * table_has_dictionary() first to avoid the deep-copy when there's nothing to
 * decode.
 */
inline std::unique_ptr<cudf::table> decode_dictionary_columns(
  cudf::table_view const& t, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(t.num_columns());
  for (auto const& c : t) {
    if (c.type().id() == cudf::type_id::DICTIONARY32) {
      cols.push_back(cudf::dictionary::decode(cudf::dictionary_column_view(c), stream, mr));
    } else {
      cols.push_back(std::make_unique<cudf::column>(c, stream, mr));
    }
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace sirius
