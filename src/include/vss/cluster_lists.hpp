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

#include "vss/kmeans_functions.hpp"

#include <rmm/device_buffer.hpp>

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/**
 * @brief A pinned corpus column rewritten in cluster order: the inverted lists of an IVF index.
 *
 * The clustered join needs every cluster's rows contiguous so that a probe routed to a cluster
 * searches one slice. Getting that order through SQL means handing every row's label back to
 * DuckDB and sorting the corpus there, which at 100M rows is minutes of work around seconds of
 * GPU math. This is the same order built in two streaming passes over the pin instead: label
 * every row, count, then scatter each row straight to its place.
 *
 * Layout row r holds pin row @c row_ids[r], so a neighbour found in the lists is reported in
 * the pin's own row space and every downstream reader of the corpus is unaffected.
 */
/// How the lists hold a vector. The join always searches FP32; an encoding is only ever used
/// where it is lossless, so a staged chunk widens back to exactly the values that were pinned.
enum class list_encoding : std::uint8_t {
  float32,
  /// One byte per component: every value was an integer in [0, 255] (SIFT, BigANN, and other
  /// byte-quantized descriptors stored as FLOAT), held as int8 x - 128. A quarter of the memory
  /// and of the transfer, and exact input for an int8 GEMM.
  uint8,
};

struct cluster_lists {
  const scan_manager::pinned_entry* pin{nullptr};  ///< The pin the lists were built from.
  std::string table;
  std::string column;
  std::int64_t n_rows{0};
  std::int64_t dim{0};
  std::int64_t n_clusters{0};
  /// Rows per staged chunk; only the last chunk may be shorter.
  std::int64_t chunk_rows{0};
  /// List c is layout rows [offsets[c], offsets[c + 1]).
  std::vector<std::int64_t> offsets;
  ::cucascade::memory::Tier tier{::cucascade::memory::Tier::GPU};
  list_encoding encoding{list_encoding::float32};

  /// GPU tier: the whole [n_rows x dim] matrix.
  std::unique_ptr<rmm::device_buffer> device_vectors;
  /// HOST tier: pinned blocks of @c rows_per_block rows each, in layout order. A chunk is a
  /// whole number of blocks, so staging one is a run of block-sized copies.
  ::cucascade::memory::fixed_multiple_blocks_allocation host_vectors;
  std::int64_t rows_per_block{0};

  /// INT64 [n_rows], device-resident on either tier: the pin row of each layout row.
  std::unique_ptr<rmm::device_buffer> row_ids;
  /// UINT8 lists only: INT32 [n_rows] |x - 128|^2 per layout row, device-resident, which is what
  /// the int8 search adds to its dot products in place of a per-query norm pass.
  std::unique_ptr<rmm::device_buffer> row_sq;
  /// INT32 [chunk_rows + 1] offsets 0, dim, 2 dim, ... A LIST view of any chunk borrows a
  /// prefix of these, since every list in the column has the same width.
  std::unique_ptr<rmm::device_buffer> list_offsets;

  /// Stored bytes per row, in the list encoding.
  [[nodiscard]] std::size_t row_bytes() const
  {
    return static_cast<std::size_t>(dim) * (encoding == list_encoding::uint8 ? 1 : sizeof(float));
  }
  [[nodiscard]] std::int64_t num_chunks() const
  {
    return chunk_rows == 0 ? 0 : (n_rows + chunk_rows - 1) / chunk_rows;
  }
  [[nodiscard]] std::int64_t rows_in_chunk(std::int64_t i) const
  {
    return std::min(chunk_rows, n_rows - i * chunk_rows);
  }
};

/// What a lists build did, for the table function to report.
struct cluster_lists_result {
  std::int64_t n_rows{0};
  std::int64_t n_clusters{0};
  std::int64_t min_list{0};
  std::int64_t max_list{0};
  std::int64_t empty_lists{0};
  std::string tier;
  std::string encoding;
};

/// `storage =>` of the build: FLOAT32 always, UINT8 or fail, or the tightest lossless one.
enum class list_storage : std::uint8_t { automatic, float32, uint8 };

/**
 * @brief `sirius_kmeans_build_lists(table, column, clustering)`: build @ref cluster_lists for a
 *        pinned column under a fitted clustering, replacing any lists that clustering had.
 *
 * GPU tier when the pin is GPU-resident and the copy fits the device, HOST tier otherwise.
 */
cluster_lists_result run_kmeans_build_lists(duckdb::SiriusContext& ctx,
                                            const kmeans_assign_request& req,
                                            list_storage storage = list_storage::automatic);

/// The lists built for @p clustering, or nullptr when there are none.
[[nodiscard]] const cluster_lists* find_cluster_lists(duckdb::SiriusContext& ctx,
                                                      const std::string& clustering);

/// Drop the lists built for @p clustering, if any; a re-fit makes them stale.
void erase_cluster_lists(duckdb::SiriusContext& ctx, const std::string& clustering);

/// row_ids[dest[i]] = row_base + i for i in [0, n): the row map of one scattered chunk.
void scatter_row_ids(std::int64_t const* dest,
                     std::int64_t n,
                     std::int64_t row_base,
                     std::int64_t* row_ids,
                     rmm::cuda_stream_view stream);

/// Adds to @p out the number of the @p n values that are not an integer in [0, 255].
void count_non_uint8(float const* values,
                     std::int64_t n,
                     unsigned long long* out,
                     rmm::cuda_stream_view stream);

/// out[i] = in[i] - 128 as int8, for values already known to be integers in [0, 255]. Byte-valued
/// lists are stored this way: L2 does not change under the shift, and int8 is what the
/// tensor-core GEMM multiplies exactly.
void narrow_to_shifted_int8(float const* in,
                            std::int64_t n,
                            std::int8_t* out,
                            rmm::cuda_stream_view stream);

/// out[i] = in[i] + 128 as FP32: the stored bytes back to the pinned values.
void widen_shifted_int8(std::int8_t const* in,
                        std::int64_t n,
                        float* out,
                        rmm::cuda_stream_view stream);

/// |x|^2 per row of shifted int8 rows, exact in int32.
void int8_row_sq_norms(std::int8_t const* x,
                       std::int64_t rows,
                       std::int64_t d,
                       std::int32_t* out,
                       rmm::cuda_stream_view stream);

/// out[i, :] = src[rows[i], :] for rows of @p row_bytes bytes.
void gather_bytes(void const* src,
                  std::int64_t row_bytes,
                  std::int64_t const* rows,
                  std::int64_t m,
                  void* out,
                  rmm::cuda_stream_view stream);

/// out[i] = src[rows[i]].
void gather_int32(std::int32_t const* src,
                  std::int64_t const* rows,
                  std::int64_t m,
                  std::int32_t* out,
                  rmm::cuda_stream_view stream);

/// INT32 offsets 0, dim, ..., n * dim into @p out (n + 1 entries).
void fill_list_offsets(std::int32_t* out,
                       std::int64_t n,
                       std::int64_t dim,
                       rmm::cuda_stream_view stream);

}  // namespace sirius::vss
