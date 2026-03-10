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

// cudf_asof_join.cu — Phase 1 ASOF JOIN implementation (Option A algorithm)
//
// Algorithm:
//   1. Sort right by (eq_keys..., ts) ASC          ← cudf::sorted_order + cudf::gather
//   2. Build left needle table (eq_keys..., ts)
//   3. Binary search: cudf::upper_bound (>=) or cudf::lower_bound (>)
//   4. Custom kernel: match = bound-1, NULL if OOB or eq_key mismatch at boundary
//   5. cudf::gather(sorted_right_payload, match_indices, NULLIFY)
//   6. Convert gathered cudf columns → GPUColumns

#include "../operator/cuda_helper.cuh"
#include "cudf/cudf_utils.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_asof_join.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdint.h>
#include <vector>

namespace duckdb {

// ─────────────────────────────────────────────────────────────────────────────
// CUDA kernel: compute final match indices from binary search bound results
//
// For each left row i:
//   match = bound_indices[i] - 1
//   → -1 if match < 0 (no preceding right row in the entire table)
//   → -1 if sorted_right_eq_key[match] != left_eq_key[i] (partition boundary crossed)
//   → match otherwise (valid match — right row at this position)
//
// The -1 sentinel is interpreted by cudf::gather with NULLIFY as a null output.
// ─────────────────────────────────────────────────────────────────────────────

template <typename KeyType>
__global__ void compute_match_indices_kernel(const int32_t* __restrict__ bound_indices,
                                             const KeyType* __restrict__ right_eq_key,
                                             const KeyType* __restrict__ left_eq_key,
                                             int32_t* __restrict__ match_indices,
                                             int64_t num_left_rows,
                                             bool    has_eq_key)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_left_rows) return;

  int32_t bound = bound_indices[i];
  int32_t match = bound - 1;

  if (match < 0) {
    match_indices[i] = -1;
    return;
  }

  // Partition boundary check: if the candidate right row belongs to a different
  // partition (equality key differs), there is no valid match for this left row.
  if (has_eq_key && right_eq_key[match] != left_eq_key[i]) {
    match_indices[i] = -1;
    return;
  }

  match_indices[i] = match;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: dispatch the match-index kernel with the correct key type
// ─────────────────────────────────────────────────────────────────────────────

static void dispatch_match_kernel(const int32_t*   bound_data,
                                  const void*      right_sorted_eq_key_ptr,
                                  const void*      left_eq_key_ptr,
                                  int32_t*         match_indices,
                                  int64_t          num_left,
                                  bool             has_eq_key,
                                  GPUColumnTypeId  key_type_id)
{
  const int threads = 256;
  const int blocks  = static_cast<int>((num_left + threads - 1) / threads);

  if (!has_eq_key) {
    // No partition key: just compute match = bound - 1, check OOB only
    compute_match_indices_kernel<int64_t>
      <<<blocks, threads>>>(bound_data, nullptr, nullptr, match_indices, num_left, false);
    return;
  }

  switch (key_type_id) {
    case GPUColumnTypeId::INT32:
      compute_match_indices_kernel<int32_t>
        <<<blocks, threads>>>(bound_data,
                              reinterpret_cast<const int32_t*>(right_sorted_eq_key_ptr),
                              reinterpret_cast<const int32_t*>(left_eq_key_ptr),
                              match_indices,
                              num_left,
                              true);
      break;
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::FLOAT64:  // same 8-byte width
      compute_match_indices_kernel<int64_t>
        <<<blocks, threads>>>(bound_data,
                              reinterpret_cast<const int64_t*>(right_sorted_eq_key_ptr),
                              reinterpret_cast<const int64_t*>(left_eq_key_ptr),
                              match_indices,
                              num_left,
                              true);
      break;
    default:
      throw NotImplementedException(
        "cudf_asof_join: unsupported equality key type %d for partition boundary check. "
        "Support for VARCHAR and other types is planned for Phase 2.",
        static_cast<int>(key_type_id));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Functor: retain only positive-or-zero match indices (i.e. valid matches)
// ─────────────────────────────────────────────────────────────────────────────

struct IsValidMatch {
  __device__ bool operator()(int32_t m) const { return m >= 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
// cudf_asof_join — main implementation
// ─────────────────────────────────────────────────────────────────────────────

void cudf_asof_join(vector<shared_ptr<GPUColumn>>& right_eq_keys,
                    shared_ptr<GPUColumn>&          right_ts,
                    vector<shared_ptr<GPUColumn>>& right_payload,
                    vector<shared_ptr<GPUColumn>>& left_eq_keys,
                    shared_ptr<GPUColumn>&          left_ts,
                    vector<shared_ptr<GPUColumn>>& output_right_cols,
                    ExpressionType                  comparison_type,
                    JoinType                        join_type,
                    shared_ptr<GPUColumn>&          out_left_row_indices)
{
  GPUBufferManager* mgr = &GPUBufferManager::GetInstance();
  cudf::set_current_device_resource(mgr->mr);

  const int64_t num_right = static_cast<int64_t>(right_ts->column_length);
  const int64_t num_left  = static_cast<int64_t>(left_ts->column_length);
  const bool    has_eq_key = !right_eq_keys.empty();
  const idx_t   n_output   = output_right_cols.size();

  SIRIUS_LOG_DEBUG("cudf_asof_join: {} right rows, {} left rows, {} output cols, cmp={}",
                   num_right, num_left, n_output, static_cast<int>(comparison_type));

  // Phase 1 supports only these comparison types
  if (comparison_type != ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
      comparison_type != ExpressionType::COMPARE_GREATERTHAN) {
    throw NotImplementedException(
      "cudf_asof_join: comparison type %d not supported in Phase 1. "
      "Only COMPARE_GREATERTHANOREQUALTO (>=) and COMPARE_GREATERTHAN (>) are supported. "
      "LESSTHAN/LESSTHANOREQUALTO support is planned for Phase 2.",
      static_cast<int>(comparison_type));
  }

  // Phase 1 supports at most one equality (partition) key
  if (right_eq_keys.size() > 1) {
    throw NotImplementedException(
      "cudf_asof_join: %zu partition (equality) keys not supported in Phase 1. "
      "Only 0 or 1 partition keys are supported. Multi-key support is planned for Phase 2.",
      right_eq_keys.size());
  }

  // ── Handle empty left side ────────────────────────────────────────────────

  if (num_left == 0) {
    for (idx_t i = 0; i < n_output; i++) {
      output_right_cols[i] = make_shared_ptr<GPUColumn>(
        0, right_payload[i]->data_wrapper.type, nullptr, nullptr);
    }
    return;
  }

  // ── Handle empty right side → all right output columns are NULL ───────────

  if (num_right == 0) {
    for (idx_t i = 0; i < n_output; i++) {
      auto cudf_type   = right_payload[i]->convertToCudfColumn().type();
      auto null_scalar = cudf::make_default_constructed_scalar(cudf_type);
      null_scalar->set_valid_async(false, rmm::cuda_stream_default);
      auto null_col = cudf::make_column_from_scalar(*null_scalar,
                                                    static_cast<cudf::size_type>(num_left));
      output_right_cols[i] =
        make_shared_ptr<GPUColumn>(0, right_payload[i]->data_wrapper.type, nullptr, nullptr);
      output_right_cols[i]->setFromCudfColumn(*null_col, false, nullptr, 0, mgr);
    }
    return;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 1: Sort right table by (eq_keys..., ts) ASC
  // ─────────────────────────────────────────────────────────────────────────

  // Build sort key table: [eq_key_0, ..., ts]
  std::vector<cudf::column_view> right_sort_key_views;
  for (auto& col : right_eq_keys) {
    right_sort_key_views.push_back(col->convertToCudfColumn());
  }
  right_sort_key_views.push_back(right_ts->convertToCudfColumn());
  auto right_sort_key_table = cudf::table_view(right_sort_key_views);

  // Sort orders: all ascending, NULLs last (NULLs sort after all valid values)
  std::vector<cudf::order>      sort_orders(right_sort_key_views.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(right_sort_key_views.size(), cudf::null_order::AFTER);

  // Get sorted permutation indices (int32 column)
  auto sorted_order      = cudf::sorted_order(right_sort_key_table, sort_orders, null_orders);
  auto sorted_order_view = sorted_order->view();

  // Gather sort keys in sorted order → used for binary search + boundary check
  auto right_sorted_keys = cudf::gather(right_sort_key_table, sorted_order_view);

  // Gather payload columns in sorted order → used for final output gather
  std::vector<cudf::column_view> right_payload_views;
  for (auto& col : right_payload) {
    right_payload_views.push_back(col->convertToCudfColumn());
  }
  auto right_payload_table  = cudf::table_view(right_payload_views);
  auto right_sorted_payload = cudf::gather(right_payload_table, sorted_order_view);

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 2: Binary search — find insertion point for each left row
  // ─────────────────────────────────────────────────────────────────────────

  // Build left needle table: [eq_key_0, ..., ts]  (same schema as right sort keys)
  std::vector<cudf::column_view> left_needle_views;
  for (auto& col : left_eq_keys) {
    left_needle_views.push_back(col->convertToCudfColumn());
  }
  left_needle_views.push_back(left_ts->convertToCudfColumn());
  auto left_needle_table = cudf::table_view(left_needle_views);

  // upper_bound for >= : returns first index where sorted_right > left needle
  //   → bound - 1 is the last right row with combined key ≤ left needle
  // lower_bound for >  : returns first index where sorted_right ≥ left needle
  //   → bound - 1 is the last right row with combined key < left needle
  std::unique_ptr<cudf::column> bound_col;
  if (comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
    bound_col =
      cudf::upper_bound(right_sorted_keys->view(), left_needle_table, sort_orders, null_orders);
  } else {
    // COMPARE_GREATERTHAN
    bound_col =
      cudf::lower_bound(right_sorted_keys->view(), left_needle_table, sort_orders, null_orders);
  }
  const int32_t* bound_data = bound_col->view().data<int32_t>();

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 3: Compute final match indices (custom CUDA kernel)
  // ─────────────────────────────────────────────────────────────────────────

  // Allocate device memory for match indices (int32, one per left row)
  int32_t* match_indices =
    mgr->customCudaMalloc<int32_t>(static_cast<size_t>(num_left), 0, 0);

  // Pointer to the sorted right first-eq-key column (for partition boundary check)
  const void*     right_sorted_eq_key_ptr = nullptr;
  const void*     left_eq_key_ptr         = nullptr;
  GPUColumnTypeId key_type_id             = GPUColumnTypeId::INT64;

  if (has_eq_key) {
    // First column of right_sorted_keys is the first (and only, in Phase 1) eq_key
    right_sorted_eq_key_ptr = right_sorted_keys->view().column(0).data<int8_t>();
    left_eq_key_ptr         = left_eq_keys[0]->data_wrapper.data;
    key_type_id             = left_eq_keys[0]->data_wrapper.type.id();
  }

  dispatch_match_kernel(bound_data,
                        right_sorted_eq_key_ptr,
                        left_eq_key_ptr,
                        match_indices,
                        num_left,
                        has_eq_key,
                        key_type_id);

  cudaDeviceSynchronize();

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 4: Gather right payload (INNER JOIN compacts; LEFT JOIN uses NULLIFY)
  // ─────────────────────────────────────────────────────────────────────────

  if (join_type == JoinType::INNER) {
    // Compact match_indices: keep only entries ≥ 0 (valid matches).
    // Simultaneously build valid_left_indices (left row numbers to output).
    auto stream = rmm::cuda_stream_default;
    rmm::device_uvector<int32_t> valid_left_vec(num_left, stream);
    rmm::device_uvector<int32_t> valid_match_vec(num_left, stream);

    // Compact left row indices 0..num_left-1 where match >= 0
    auto end_left = thrust::copy_if(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(static_cast<int32_t>(num_left)),
      match_indices,            // stencil: keep index i when match_indices[i] >= 0
      valid_left_vec.begin(),
      IsValidMatch{});

    // Compact match_indices themselves (values where match >= 0)
    auto end_match = thrust::copy_if(
      rmm::exec_policy(stream),
      match_indices,
      match_indices + num_left,
      valid_match_vec.begin(),
      IsValidMatch{});

    cudaStreamSynchronize(stream.value());

    const int64_t n_valid = static_cast<int64_t>(end_left - valid_left_vec.begin());
    SIRIUS_LOG_DEBUG("cudf_asof_join INNER: {} / {} rows matched", n_valid, num_left);

    if (n_valid == 0) {
      for (idx_t i = 0; i < n_output; i++) {
        output_right_cols[i] =
          make_shared_ptr<GPUColumn>(0, right_payload[i]->data_wrapper.type, nullptr, nullptr);
      }
      out_left_row_indices = nullptr;
      return;
    }

    // Gather right payload using compacted valid match indices (all in-bounds)
    cudf::column_view valid_match_view(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(n_valid),
                                       valid_match_vec.data(),
                                       nullptr, 0);
    auto gathered_payload =
      cudf::gather(right_sorted_payload->view(), valid_match_view,
                   cudf::out_of_bounds_policy::DONT_CHECK);

    auto gathered_cols = gathered_payload->release();
    for (idx_t i = 0; i < n_output; i++) {
      output_right_cols[i] =
        make_shared_ptr<GPUColumn>(0, right_payload[i]->data_wrapper.type, nullptr, nullptr);
      output_right_cols[i]->setFromCudfColumn(*gathered_cols[i], false, nullptr, 0, mgr);
    }

    // Build out_left_row_indices: INT32 cudf column from valid_left_vec
    auto left_idx_col =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                static_cast<cudf::size_type>(n_valid),
                                cudf::mask_state::UNALLOCATED, stream);
    thrust::copy(rmm::exec_policy(stream),
                 valid_left_vec.begin(), valid_left_vec.begin() + n_valid,
                 left_idx_col->mutable_view().data<int32_t>());
    cudaStreamSynchronize(stream.value());

    out_left_row_indices =
      make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr);
    out_left_row_indices->setFromCudfColumn(*left_idx_col, false, nullptr, 0, mgr);

  } else {
    // LEFT JOIN: keep all left rows; non-matched right columns become NULL
    cudf::column_view match_map_view(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(num_left),
                                     match_indices,
                                     nullptr, 0);
    auto gathered_payload =
      cudf::gather(right_sorted_payload->view(), match_map_view,
                   cudf::out_of_bounds_policy::NULLIFY);

    auto gathered_cols = gathered_payload->release();
    for (idx_t i = 0; i < n_output; i++) {
      output_right_cols[i] =
        make_shared_ptr<GPUColumn>(0, right_payload[i]->data_wrapper.type, nullptr, nullptr);
      output_right_cols[i]->setFromCudfColumn(*gathered_cols[i], false, nullptr, 0, mgr);
    }
    out_left_row_indices = nullptr;  // all left rows output, no filtering needed
  }

  SIRIUS_LOG_DEBUG("cudf_asof_join: complete");
}

}  // namespace duckdb
