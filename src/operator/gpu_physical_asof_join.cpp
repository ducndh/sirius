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

#include "operator/gpu_physical_asof_join.hpp"

#include <chrono>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"
#include "log/logging.hpp"

namespace duckdb {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

GPUPhysicalAsOfJoin::GPUPhysicalAsOfJoin(LogicalOperator&               op,
                                         unique_ptr<GPUPhysicalOperator> left,
                                         unique_ptr<GPUPhysicalOperator> right,
                                         vector<JoinCondition>           cond,
                                         const vector<idx_t>&            left_proj_map,
                                         const vector<idx_t>&            right_proj_map,
                                         idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::ASOF_JOIN, op.types, estimated_cardinality),
    conditions(std::move(cond)),
    comparison_type(ExpressionType::INVALID),
    join_type(op.Cast<LogicalComparisonJoin>().join_type)
{
  children.push_back(std::move(left));
  children.push_back(std::move(right));

  // Extract equality (partition) and inequality (timestamp) column indices
  for (auto& c : this->conditions) {
    switch (c.comparison) {
      case ExpressionType::COMPARE_EQUAL:
      case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
        lhs_partition_col_idxs.push_back(c.left->Cast<BoundReferenceExpression>().index);
        rhs_partition_col_idxs.push_back(c.right->Cast<BoundReferenceExpression>().index);
        break;
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      case ExpressionType::COMPARE_GREATERTHAN:
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
      case ExpressionType::COMPARE_LESSTHAN:
        lhs_order_col_idx = c.left->Cast<BoundReferenceExpression>().index;
        rhs_order_col_idx = c.right->Cast<BoundReferenceExpression>().index;
        comparison_type   = c.comparison;
        break;
      default:
        throw NotImplementedException(
          "GPUPhysicalAsOfJoin: unsupported condition comparison %d",
          static_cast<int>(c.comparison));
    }
  }
  D_ASSERT(comparison_type != ExpressionType::INVALID);

  // Resolve left output projection (empty map → output all columns)
  left_projection_map = left_proj_map;
  if (left_projection_map.empty()) {
    for (idx_t i = 0; i < children[0]->GetTypes().size(); i++) {
      left_projection_map.push_back(i);
    }
  }
  n_left_output = left_projection_map.size();

  // Resolve right output projection (empty map → output all columns)
  right_projection_map = right_proj_map;
  if (right_projection_map.empty()) {
    for (idx_t i = 0; i < children[1]->GetTypes().size(); i++) {
      right_projection_map.push_back(i);
    }
  }
  n_right_output = right_projection_map.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sink: accumulate right (build) side rows
// ─────────────────────────────────────────────────────────────────────────────

SinkResultType GPUPhysicalAsOfJoin::Sink(GPUIntermediateRelation& input_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();

  const size_t n_rows = input_relation.columns.empty()
                          ? 0
                          : input_relation.columns[0]->column_length;
  SIRIUS_LOG_DEBUG("AsOf Join Sink: {} rows, {} columns", n_rows, input_relation.column_count);

  // Store a reference to each right-side column (the GPU buffers stay alive)
  right_table = make_shared_ptr<GPUIntermediateRelation>(input_relation.column_count);
  for (size_t i = 0; i < input_relation.column_count; i++) {
    right_table->columns[i] = input_relation.columns[i];
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("AsOf Join Sink time: {:.2f} ms", duration.count() / 1000.0);

  return SinkResultType::FINISHED;
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute: probe left rows against the built right table
// ─────────────────────────────────────────────────────────────────────────────

OperatorResultType GPUPhysicalAsOfJoin::Execute(GPUIntermediateRelation& input_relation,
                                                GPUIntermediateRelation& output_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();

  const size_t n_left  = input_relation.columns.empty()
                           ? 0
                           : input_relation.columns[0]->column_length;
  const size_t n_right = (right_table && !right_table->columns.empty())
                           ? right_table->columns[0]->column_length
                           : 0;
  SIRIUS_LOG_DEBUG("AsOf Join Execute: {} left rows, {} right rows", n_left, n_right);

  D_ASSERT(right_table);
  D_ASSERT(output_relation.columns.size() == n_left_output + n_right_output);

  // ── Build column vectors for cudf_asof_join ──────────────────────────────

  // Left equality (partition) key columns
  vector<shared_ptr<GPUColumn>> left_eq_keys;
  for (auto col_idx : lhs_partition_col_idxs) {
    left_eq_keys.push_back(input_relation.columns[col_idx]);
  }
  // Left timestamp column
  auto left_ts = input_relation.columns[lhs_order_col_idx];

  // Right equality (partition) key columns
  vector<shared_ptr<GPUColumn>> right_eq_keys;
  for (auto col_idx : rhs_partition_col_idxs) {
    right_eq_keys.push_back(right_table->columns[col_idx]);
  }
  // Right timestamp column
  auto right_ts = right_table->columns[rhs_order_col_idx];

  // Right projected payload columns (the right columns we output)
  vector<shared_ptr<GPUColumn>> right_payload;
  for (auto col_idx : right_projection_map) {
    right_payload.push_back(right_table->columns[col_idx]);
  }

  // ── Run the ASOF join ────────────────────────────────────────────────────

  vector<shared_ptr<GPUColumn>> output_right_cols(n_right_output);
  shared_ptr<GPUColumn> out_left_row_indices;  // populated for INNER JOIN only
  cudf_asof_join(right_eq_keys, right_ts, right_payload,
                 left_eq_keys,  left_ts,
                 output_right_cols,
                 comparison_type,
                 join_type,
                 out_left_row_indices);

  // ── Assemble output relation ──────────────────────────────────────────────

  GPUBufferManager* mgr = &GPUBufferManager::GetInstance();

  if (out_left_row_indices) {
    // INNER JOIN: only matched rows are output.  Gather each left column using
    // the valid row indices returned by cudf_asof_join.
    auto row_idx_view = out_left_row_indices->convertToCudfColumn();
    for (idx_t i = 0; i < n_left_output; i++) {
      auto left_col_view = input_relation.columns[left_projection_map[i]]->convertToCudfColumn();
      auto filtered = cudf::gather(cudf::table_view({left_col_view}), row_idx_view);
      auto filtered_cols = filtered->release();
      output_relation.columns[i] =
        make_shared_ptr<GPUColumn>(0, input_relation.columns[left_projection_map[i]]->data_wrapper.type,
                                   nullptr, nullptr);
      output_relation.columns[i]->setFromCudfColumn(*filtered_cols[0], false, nullptr, 0, mgr);
    }
  } else {
    // LEFT JOIN (or INNER with all rows matching): copy left columns directly
    for (idx_t i = 0; i < n_left_output; i++) {
      output_relation.columns[i] = input_relation.columns[left_projection_map[i]];
    }
  }
  // Right columns: gathered output from cudf_asof_join
  for (idx_t i = 0; i < n_right_output; i++) {
    output_relation.columns[n_left_output + i] = output_right_cols[i];
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("AsOf Join Execute time: {:.2f} ms", duration.count() / 1000.0);

  return OperatorResultType::FINISHED;
}

// ─────────────────────────────────────────────────────────────────────────────
// BuildPipelines
// ─────────────────────────────────────────────────────────────────────────────

void GPUPhysicalAsOfJoin::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  // Register this operator in the current (probe) pipeline
  auto& state = meta_pipeline.GetState();
  state.AddPipelineOperator(current, *this);

  // Build side (children[1] = right/quotes) gets its own child meta-pipeline.
  // The data flow is: children[1] source → child pipeline → Sink()
  auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
  child_meta_pipeline.Build(*children[1]);

  // Probe side (children[0] = left/trades) continues in the current pipeline.
  // The data flow is: children[0] source → current pipeline → Execute()
  children[0]->BuildPipelines(current, meta_pipeline);
}

}  // namespace duckdb
