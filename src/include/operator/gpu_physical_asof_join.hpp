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

#include "duckdb/execution/operator/join/physical_asof_join.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "gpu_columns.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

// Forward declaration of the cudf ASOF join implementation (cudf_asof_join.cu).
// Algorithm (Phase 1 — Option A: combined sort + upper_bound):
//   1. Sort right by (eq_keys..., ts) ASC using cudf::sorted_order + gather
//   2. Build left needle table (eq_keys..., ts)
//   3. cudf::upper_bound (>=) or cudf::lower_bound (>) → per-row bound indices
//   4. Custom kernel: match_idx = bound - 1, NULL if OOB or partition boundary crossed
//   5. cudf::gather(right_payload, match_indices, NULLIFY) → right output cols (NULLs for misses)
void cudf_asof_join(
    // Right (build) side — all rows accumulated during Sink
    vector<shared_ptr<GPUColumn>>& right_eq_keys,         // equality key columns (e.g. symbol)
    shared_ptr<GPUColumn>&         right_ts,               // timestamp column (e.g. quote_ts)
    vector<shared_ptr<GPUColumn>>& right_payload,          // projected right output columns
    // Left (probe) side — current batch from Execute
    vector<shared_ptr<GPUColumn>>& left_eq_keys,          // equality key columns (e.g. symbol)
    shared_ptr<GPUColumn>&         left_ts,                // timestamp column (e.g. trade_ts)
    // Right output: one gathered column per right_payload entry
    vector<shared_ptr<GPUColumn>>& output_right_cols,
    ExpressionType                 comparison_type,        // COMPARE_GTE or COMPARE_GT
    JoinType                       join_type,              // INNER (drop misses) or LEFT (NULLs)
    // INNER JOIN only: INT32 GPUColumn of valid left row indices used by Execute() to
    // gather the left output columns.  Set to nullptr for LEFT JOIN.
    shared_ptr<GPUColumn>&         out_left_row_indices);

// GPUPhysicalAsOfJoin: GPU physical operator for ASOF JOIN.
//
// LEFT ASOF JOIN semantics: every left row appears in output; right columns are NULL
// when no matching right row exists.
//
// Pipeline structure (mirrors GPUPhysicalHashJoin):
//   Sink   ← right (build) side: accumulates all right rows into right_table
//   Execute ← left  (probe) side: sorts right, searches, gathers, produces output
class GPUPhysicalAsOfJoin : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::ASOF_JOIN;

 public:
  GPUPhysicalAsOfJoin(LogicalOperator&               op,
                      unique_ptr<GPUPhysicalOperator> left,
                      unique_ptr<GPUPhysicalOperator> right,
                      vector<JoinCondition>           conditions,
                      const vector<idx_t>&            left_projection_map,
                      const vector<idx_t>&            right_projection_map,
                      idx_t                           estimated_cardinality);

  // Join conditions from the logical plan
  vector<JoinCondition> conditions;

  // Column indices in the left input for equality (partition) keys
  vector<idx_t> lhs_partition_col_idxs;
  // Column indices in the right input for equality (partition) keys
  vector<idx_t> rhs_partition_col_idxs;

  // Column index in the left input for the ordering (timestamp) condition
  idx_t lhs_order_col_idx {0};
  // Column index in the right input for the ordering (timestamp) condition
  idx_t rhs_order_col_idx {0};

  // The comparison operator from the inequality condition (>= or >)
  ExpressionType comparison_type {ExpressionType::INVALID};

  // INNER = drop non-matching left rows; LEFT = keep them with NULL right columns
  JoinType join_type {JoinType::INNER};

  // Which left columns to project to output (empty = all)
  vector<idx_t> left_projection_map;
  // Which right columns to project to output (empty = all)
  vector<idx_t> right_projection_map;

  // Effective output column counts (resolved from projection maps in constructor)
  idx_t n_left_output  {0};
  idx_t n_right_output {0};

  // Accumulated right (build) table — populated during Sink, consumed during Execute
  mutable shared_ptr<GPUIntermediateRelation> right_table;

 public:
  // ── Sink interface ────────────────────────────────────────────────────────
  // Receives the right (build) side rows and stores them for the probe phase.
  SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
  bool IsSink() const override { return true; }
  bool ParallelSink() const override { return false; }

  // ── Execute interface ─────────────────────────────────────────────────────
  // Receives left (probe) side rows; searches right_table; writes joined output.
  OperatorResultType Execute(GPUIntermediateRelation& input_relation,
                             GPUIntermediateRelation& output_relation) const override;

  // ── Pipeline construction ─────────────────────────────────────────────────
  // children[1] (right/build) → child meta-pipeline → Sink
  // children[0] (left/probe)  → main pipeline        → Execute
  void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) override;
};

}  // namespace duckdb
