# ASOF JOIN for Sirius — Implementation Roadmap

## Overview

ASOF JOIN is a time-series join that, for each row in the left table, finds the most recent
preceding matching row in the right table. Canonical query:

```sql
-- For each trade, find the last quote before it (same symbol)
SELECT t.symbol, t.ts, t.price, q.bid, q.ask
FROM trades t
ASOF JOIN quotes q
ON t.symbol = q.symbol AND t.ts >= q.ts;
```

DuckDB 1.4 natively parses and plans this as `LogicalAsOfJoin`. The GPU physical operator
(`GPUPhysicalAsOfJoin`) is the missing piece.

**Scope**: old `gpu_processing` path only (no changes to `gpu_execution`/cucascade path until
after GHTC 2025-03-16).

---

## What cudf 26.02 Already Provides

| Primitive | Header | Use |
|-----------|--------|-----|
| `cudf::lower_bound` | `cudf/search.hpp` | Binary search: find first index in sorted right table where `right_ts >= left_ts` — i.e., the insertion point. Subtract 1 to get the last preceding row. |
| `cudf::upper_bound` | `cudf/search.hpp` | Alternative for strict `>` semantics. |
| `cudf::sort_merge_join` | `cudf/join/sort_merge_join.hpp` | Sorted merge join for dense/equi-join cases. Does not directly support inequality, but can be combined with a filter. |
| `cudf::sort` / `cudf::stable_sort` | `cudf/sorting.hpp` | Radix sort for partitions by timestamp. |
| `cudf::hash_partition` | `cudf/partitioning.hpp` | Hash-partition both tables by equality key (e.g., symbol). |
| `cudf::gather` | `cudf/copying.hpp` | Gather right-table rows by matched index. |

**Key finding**: cudf does *not* have a dedicated `asof_join`. We build on `lower_bound` +
`hash_partition` + `sort`. No need to write CUDA kernels from scratch for the baseline.

---

## Corrected Algorithm Design (vs. Gemini plan)

The Gemini plan had two naming errors fixed here:

| Path | When to use | Algorithm |
|------|-------------|-----------|
| **Merge path** | Dense: left and right timestamps interleave frequently (small avg Δt) | `cudf::sort_merge_join` variant — linear scan both sorted arrays together. O(N+M) per partition. |
| **Binary search path** | Sparse: right table has infrequent timestamps relative to left | `cudf::lower_bound` per left row against sorted right partition. O(N log M) per partition. |

For the initial implementation, implement **binary search only** (handles both cases correctly,
just slightly sub-optimal for very dense data). Add density profiler + merge path in a later
iteration.

**TMA note**: TMA (Tensor Memory Accelerator) is designed for matrix/tensor operations, not
general memory transfers. For ASOF JOIN, the real H100 advantage is the 50 MB L2 cache (vs.
40 MB on A100) and NVLink bandwidth. Do not claim TMA in benchmark marketing.

---

## Phase 0: Audit & Planner Hook (1–2 days)

### 0.1 Existing hook point

In `src/plan/gpu_plan_comparison_join.cpp` there is already a commented-out stub:

```cpp
unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalComparisonJoin& op)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_ASOF_JOIN:
      // return PlanAsOfJoin(op);          // <-- just needs implementation
      throw NotImplementedException("Asof join not supported in GPU");
    ...
  }
}
```

### 0.2 Key fields on `LogicalComparisonJoin` for ASOF

```cpp
// From duckdb/include/duckdb/execution/operator/join/physical_asof_join.hpp
class PhysicalAsOfJoin : public PhysicalComparisonJoin {
  vector<unique_ptr<Expression>> lhs_partitions;  // equality key exprs (left side)
  vector<unique_ptr<Expression>> rhs_partitions;  // equality key exprs (right side)
  vector<BoundOrderByNode>       lhs_orders;      // timestamp ordering (left side)
  vector<BoundOrderByNode>       rhs_orders;      // timestamp ordering (right side)
  ExpressionType                 comparison_type; // e.g. COMPARE_GREATERTHANOREQUALTO
  vector<column_t>               right_projection_map; // which right cols to output
};
```

---

## Phase 1: GPUPhysicalAsOfJoin Operator (3–5 days)

### 1.1 Header: `src/include/operator/gpu_physical_asof_join.hpp`

```cpp
#pragma once
#include "gpu_physical_operator.hpp"

namespace duckdb {

// Forward declarations
void cudf_asof_join(
    // Right table (build side — sorted right partitions)
    vector<shared_ptr<GPUColumn>>& right_partition_keys,  // equality keys
    vector<shared_ptr<GPUColumn>>& right_ts,              // timestamp column
    vector<shared_ptr<GPUColumn>>& right_payload,         // projected right columns
    uint64_t num_right_rows,
    // Left table (probe side)
    vector<shared_ptr<GPUColumn>>& left_partition_keys,
    vector<shared_ptr<GPUColumn>>& left_ts,
    uint64_t num_left_rows,
    // Output
    vector<shared_ptr<GPUColumn>>& output_columns,
    ExpressionType comparison_type);                      // >= or >

class GPUPhysicalAsOfJoin : public GPUPhysicalOperator {
 public:
  GPUPhysicalAsOfJoin(LogicalOperator& op,
                      unique_ptr<GPUPhysicalOperator> left,
                      unique_ptr<GPUPhysicalOperator> right);

  // Equality key expressions extracted from LogicalComparisonJoin
  vector<unique_ptr<Expression>> lhs_partitions;
  vector<unique_ptr<Expression>> rhs_partitions;
  // Timestamp ordering
  vector<BoundOrderByNode> lhs_orders;
  vector<BoundOrderByNode> rhs_orders;
  ExpressionType comparison_type;
  vector<column_t> right_projection_map;

  // Built during Sink phase
  mutable shared_ptr<GPUIntermediateRelation> right_table;

 public:
  // Sink: receives right (quotes) table batches and stores them in GPU memory
  SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
  bool IsSink() const override { return true; }
  bool ParallelSink() const override { return false; }  // single-threaded for now

  // Source: probe left (trades) table against the built right table
  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;
  bool IsSource() const override { return true; }

  void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) override;
};

} // namespace duckdb
```

### 1.2 Algorithm sketch: `src/cuda/cudf/cudf_asof_join.cu`

```cpp
// High-level algorithm (binary search path):
//
// SETUP (Sink phase — runs once on right/quotes table):
//   1. Evaluate rhs_partition_keys expressions → GPU columns [sym_R]
//   2. Evaluate rhs_orders expression        → GPU column  [ts_R]
//   3. Hash-partition right table by sym_R   → partition offsets + sorted sym_R, ts_R, payload_R
//      (cudf::hash_partition gives per-key-hash buckets; sort within each bucket by ts_R)
//
// PROBE (GetData phase — runs once on left/trades table):
//   4. Evaluate lhs_partition_keys           → GPU columns [sym_L]
//   5. Evaluate lhs_orders expression        → GPU column  [ts_L]
//   6. For each left row i:
//      a. Look up its partition bucket in the right table (by sym_L[i] hash → same hash fn)
//      b. Run cudf::lower_bound on ts_R[bucket] to find position p
//         where ts_R[p] is the first right timestamp >= ts_L[i]
//      c. The matching right row is at position p-1 (last right ts before left ts)
//         If p == 0, no match → NULL (left outer join semantics)
//   7. Gather right_payload[match_indices] → output right columns
//   8. Concatenate left columns + gathered right columns → output relation

#include <cudf/search.hpp>       // lower_bound, upper_bound
#include <cudf/partitioning.hpp> // hash_partition
#include <cudf/sorting.hpp>      // sort_by_key
#include <cudf/copying.hpp>      // gather

void cudf_asof_join(
    vector<shared_ptr<GPUColumn>>& right_partition_keys,
    vector<shared_ptr<GPUColumn>>& right_ts,
    vector<shared_ptr<GPUColumn>>& right_payload,
    uint64_t num_right,
    vector<shared_ptr<GPUColumn>>& left_partition_keys,
    vector<shared_ptr<GPUColumn>>& left_ts,
    uint64_t num_left,
    vector<shared_ptr<GPUColumn>>& output_columns,
    ExpressionType comparison_type)
{
    auto stream = cudf::get_default_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    // --- Step 1: Hash-partition right table by equality key ---
    // Combine all right partition key columns into a cudf table_view
    auto right_keys_table = build_table_view(right_partition_keys);
    auto right_ts_view    = right_ts[0]->convertToCudfColumn();

    // hash_partition: assigns each row to a bucket [0, num_partitions)
    const int num_partitions = choose_num_partitions(num_right);  // e.g. next power of 2
    auto [right_partitioned, right_partition_offsets] =
        cudf::hash_partition(right_keys_table, /* columns_to_hash= */ {0},
                             num_partitions, cudf::hash_id::HASH_MURMUR3,
                             stream, mr);

    // Sort each partition by timestamp
    // (hash_partition already groups rows by bucket; sort within each group)
    auto right_ts_partitioned = gather_column(right_ts_view, right_partitioned);
    auto right_sorted_order   = sort_within_partitions(right_ts_partitioned,
                                                        right_partition_offsets);
    // right_payload_sorted = gather(right_payload, right_sorted_order)

    // --- Step 2: For each left row, binary search in its right partition ---
    auto left_keys_table = build_table_view(left_partition_keys);
    auto left_ts_view    = left_ts[0]->convertToCudfColumn();

    // Compute which partition bucket each left row maps to (same hash fn)
    auto left_buckets = hash_to_partition(left_keys_table, num_partitions, stream, mr);

    // lower_bound per-partition: for each left row, find insertion point in sorted right ts
    // match_index[i] = right_partition_offsets[bucket[i]] + lower_bound_within_bucket - 1
    // If lower_bound == 0: no match → -1 (NULL)
    // For >= semantics: match = lower_bound - 1
    // For >  semantics: match = upper_bound - 1
    auto match_indices = compute_asof_match_indices(
        left_buckets, left_ts_view,
        right_ts_partitioned, right_partition_offsets,
        comparison_type, stream, mr);

    // --- Step 3: Gather matched right-table rows ---
    // match_indices[i] == -1 → no match (output NULL for right columns)
    auto gathered_right = cudf::gather(right_payload_sorted_view, match_indices,
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       stream, mr);

    // --- Step 4: Assemble output ---
    // output = left columns (pass-through) + gathered right columns
    assemble_output(left_partition_keys, left_ts, gathered_right, output_columns);
}
```

### 1.3 Sink implementation sketch: `src/operator/gpu_physical_asof_join.cpp`

```cpp
SinkResultType GPUPhysicalAsOfJoin::Sink(GPUIntermediateRelation& input_relation) const {
    // Accumulate right table batches into right_table
    // (Right table is typically smaller — the "quotes" lookup table)
    if (!right_table) {
        right_table = make_shared<GPUIntermediateRelation>(input_relation.column_count);
    }
    // Append this batch to right_table (column-wise concat)
    for (size_t i = 0; i < input_relation.column_count; i++) {
        right_table->columns[i] = ConcatGPUColumns(
            right_table->columns[i], input_relation.columns[i]);
    }
    return SinkResultType::NEED_MORE_INPUT;
}

SourceResultType GPUPhysicalAsOfJoin::GetData(GPUIntermediateRelation& output_relation) const {
    // Left table is the pipeline's input; right_table was built during Sink
    // (In the GPU pipeline: GetData is called after left side is fully available)
    // Extract evaluated partition keys and timestamps from right_table
    auto right_pkeys = EvaluateExpressions(rhs_partitions, *right_table);
    auto right_ts    = EvaluateExpressions(rhs_orders,     *right_table);
    auto right_payload = ProjectColumns(*right_table, right_projection_map);

    // Extract from left pipeline input (stored during left sink phase)
    auto left_pkeys  = EvaluateExpressions(lhs_partitions, *left_table);
    auto left_ts     = EvaluateExpressions(lhs_orders,     *left_table);

    cudf_asof_join(right_pkeys, right_ts, right_payload, right_table->num_rows(),
                   left_pkeys,  left_ts,  left_table->num_rows(),
                   output_relation.columns, comparison_type);

    return SourceResultType::FINISHED;
}
```

### 1.4 Planner hook: `src/plan/gpu_plan_comparison_join.cpp`

```cpp
unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::PlanAsOfJoin(LogicalComparisonJoin& op)
{
    // Build left and right child operators
    auto left  = CreatePlan(*op.children[0]);
    auto right = CreatePlan(*op.children[1]);

    auto asof = make_uniq<GPUPhysicalAsOfJoin>(op, std::move(left), std::move(right));

    // Copy partition key + timestamp expressions from the logical operator
    for (auto& expr : op.lhs_partitions) asof->lhs_partitions.push_back(expr->Copy());
    for (auto& expr : op.rhs_partitions) asof->rhs_partitions.push_back(expr->Copy());
    asof->lhs_orders      = op.lhs_orders;
    asof->rhs_orders      = op.rhs_orders;
    asof->comparison_type = op.comparison_type;
    asof->right_projection_map = op.right_projection_map;

    return std::move(asof);
}

// And in CreatePlan():
case LogicalOperatorType::LOGICAL_ASOF_JOIN:
    return PlanAsOfJoin(op);   // <-- replace the throw
```

---

## Phase 2: Density Profiler (1–2 days, after Phase 1 works correctly)

Before running the join, sample the data to choose the algorithm path:

```cpp
struct AsOfJoinDensityProfile {
    double left_avg_dt;   // average time gap between consecutive left rows (per partition)
    double right_avg_dt;  // average time gap between consecutive right rows (per partition)
    double density_ratio; // right_avg_dt / left_avg_dt
    bool use_merge_path;  // density_ratio < MERGE_PATH_THRESHOLD (e.g., 10.0)
};

// Kernel: compute average delta-t from a sorted timestamp column in O(N/256) blocks
AsOfJoinDensityProfile ProfileDensity(
    const cudf::column_view& left_ts,
    const cudf::column_view& right_ts,
    const cudf::column_view& partition_offsets,  // per-partition boundaries
    rmm::cuda_stream_view stream);
```

Route to merge path (dense) or binary search path (sparse) based on `density_ratio`.
Profile overhead should be < 1 ms for 10M rows — sample only first 1000 rows per partition
if needed.

---

## Phase 3: Custom CUDA Kernel Optimizations (1 week, post-correctness)

### 3.1 Coalesced binary search kernel

The cudf `lower_bound` applies the binary search column-by-column. For ASOF join, all left rows
in a partition need to binary-search the *same* sorted right-partition array. A custom kernel
can process multiple left rows per warp with better cache utilization:

```cuda
// One warp per partition; threads within warp collaborate on binary searches
__global__ void asof_binary_search_kernel(
    const int64_t* __restrict__ left_ts,      // sorted left timestamps (per partition)
    const int64_t* __restrict__ right_ts,     // sorted right timestamps (per partition)
    const int32_t* __restrict__ left_offsets, // partition start indices in left_ts
    const int32_t* __restrict__ right_offsets,// partition start indices in right_ts
    int32_t* __restrict__ match_indices,      // output: right row index for each left row
    int32_t num_partitions)
{
    // Each block handles one partition
    int32_t part = blockIdx.x;
    if (part >= num_partitions) return;

    int32_t l_start = left_offsets[part],  l_end = left_offsets[part + 1];
    int32_t r_start = right_offsets[part], r_end = right_offsets[part + 1];

    // Threads in the block cooperatively process left rows
    for (int32_t li = l_start + threadIdx.x; li < l_end; li += blockDim.x) {
        int64_t ts = left_ts[li];
        // Binary search in right_ts[r_start..r_end) for last entry <= ts
        int32_t lo = r_start, hi = r_end;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (right_ts[mid] <= ts) lo = mid + 1;
            else hi = mid;
        }
        match_indices[li] = (lo > r_start) ? (lo - 1) : -1;  // -1 = no match
    }
}
```

### 3.2 H100-specific optimization: larger L2 cache

Load small right-partition arrays into shared memory before searching. The H100's 50 MB L2
(vs. 40 MB on A100) means more partitions stay cache-resident automatically. For small lookup
tables (< few MB), pre-load into shared memory:

```cuda
// If right partition fits in shared memory, load it first
extern __shared__ int64_t shared_right_ts[];
for (int i = threadIdx.x; i < r_size; i += blockDim.x)
    shared_right_ts[i] = right_ts[r_start + i];
__syncthreads();
// ... then binary search shared_right_ts[] instead of global right_ts
```

---

## Phase 4: Benchmark Strategy

### Dataset

**Option A (recommended — immediately available):**
Generate synthetic tick data:
- `trades`: 500M rows, (symbol SMALLINT, ts BIGINT ns-epoch, price DOUBLE)
- `quotes`: 100M rows, (symbol SMALLINT, ts BIGINT ns-epoch, bid DOUBLE, ask DOUBLE)
- 1000 symbols, Poisson-distributed timestamps (λ = 1 tick/ms for quotes)
- Generation script: `scripts/gen_asof_benchmark.py` using cupy/numpy, write to parquet

**Option B (realistic but slower to obtain):**
Uniswap V3 DEX event logs from public Ethereum data (Google BigQuery `bigquery-public-data.crypto_ethereum`).
Join swaps (left) against pool-price-updates (right) to compute USD value at trade time.
~ 500M rows total across all pools.

### Competitors

| System | Mode | Why |
|--------|------|-----|
| DuckDB CPU | Single-node, all cores | Correctness reference + best CPU baseline |
| ClickHouse | Single-node | Has `ASOF JOIN` with `full_sorting_merge` — best optimized CPU ASOF |
| Sirius GPU | Single-node H100 | Target |

Do not benchmark against QuestDB — it is single-threaded C and not a fair comparison point.

### Cost-normalized framing

```
H100 rental cost / hr ≈ $3.00  (Lambda Labs / CoreWeave spot)
Equivalent CPU (128-core, 1TB RAM): r7i.32xlarge ≈ $8.06/hr

Metric: GB/s of joined output (or joins/second) per dollar
```

---

## Correctness Validation

Use DuckDB CPU as the reference:

```sql
-- DuckDB: generate reference output
COPY (
  SELECT t.symbol, t.ts, t.price, q.bid, q.ask
  FROM trades t ASOF JOIN quotes q ON t.symbol = q.symbol AND t.ts >= q.ts
  ORDER BY t.symbol, t.ts
) TO 'reference.parquet' (FORMAT PARQUET);
```

```python
# Compare Sirius GPU output to DuckDB reference
gpu_df = pd.read_parquet('gpu_output.parquet').sort_values(['symbol','ts'])
cpu_df = pd.read_parquet('reference.parquet').sort_values(['symbol','ts'])
assert gpu_df.shape == cpu_df.shape
pd.testing.assert_frame_equal(gpu_df, cpu_df, check_exact=False, rtol=1e-5)
```

---

## File Map

```
src/
  include/operator/
    gpu_physical_asof_join.hpp          # operator declaration
  operator/
    gpu_physical_asof_join.cpp          # Sink / GetData / BuildPipelines
  cuda/cudf/
    cudf_asof_join.cu                   # cudf_asof_join() implementation
  cuda/operator/
    asof_binary_search.cu              # Phase 3 custom kernel
  plan/
    gpu_plan_comparison_join.cpp        # add PlanAsOfJoin() + hook in CreatePlan()

scripts/
  gen_asof_benchmark.py               # synthetic benchmark data generator

docs/
  asof_join_roadmap.md                # this file
```

## Implementation Order

1. `GPUPhysicalAsOfJoin` header + planner hook (throws `NotImplementedException` → compiles)
2. `Sink()`: accumulate right table batches in GPU memory
3. `cudf_asof_join()` using cudf primitives (hash_partition + lower_bound + gather)
4. `GetData()`: wire it up, run correctness test vs DuckDB on small synthetic data
5. Density profiler + merge path (Phase 2)
6. Custom CUDA kernel (Phase 3) — only if profiler shows binary search is the bottleneck
7. Benchmark vs DuckDB + ClickHouse on 500M-row synthetic dataset
