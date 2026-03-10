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
| `cudf::lower_bound` | `cudf/search.hpp` | Binary search: first index where `right_key >= needle` |
| `cudf::upper_bound` | `cudf/search.hpp` | Binary search: first index where `right_key > needle` |
| `cudf::sorted_order` | `cudf/sorting.hpp` | Get sort permutation indices |
| `cudf::sort` / `cudf::stable_sort` | `cudf/sorting.hpp` | Radix sort |
| `cudf::hash_partition` | `cudf/partitioning.hpp` | Hash-partition by equality key (Phase 2) |
| `cudf::gather` | `cudf/copying.hpp` | Gather rows by matched index; NULLIFY for -1 |

---

## Algorithm Design

Two algorithms are defined. Phase 1 implements Option A (simpler, correct). Option B is
the Phase 2 performance optimization.

### Option A — Combined Sort + cudf::upper_bound (Phase 1 implementation)

```
Build (Sink):
  1. Accumulate all right rows into right_table (GPU memory)

Probe (Execute — called once with full left batch):
  2. Sort right_table by (eq_key_0, ..., eq_key_N, ts) ASC  ← cudf::sorted_order + gather
  3. For each left row, search sorted right as a multi-column table:
       bound = cudf::upper_bound(sorted_right_keys, left_needles)  [for >= semantics]
       bound = cudf::lower_bound(sorted_right_keys, left_needles)  [for >  semantics]
  4. match_idx = bound - 1
     if match_idx < 0: no match → NULL
     if sorted_right[match_idx].eq_key != left.eq_key: no match → NULL  (partition boundary)
     Custom CUDA kernel handles steps 4a/4b per row
  5. cudf::gather(sorted_right_payload, match_indices, NULLIFY) → gathered right cols
  6. Output = left columns (pass-through) + gathered right columns
```

**Complexity**: O(N log N) sort + O(M log N) binary search
**Correctness**: Yes (for 0 or 1 equality key; N>1 keys planned for Phase 2)
**Limitation**: Each binary search spans the full right table — no partition isolation

### Option B — Hash Partition + Per-Partition Binary Search (Phase 2 optimization)

```
Build (Sink):
  1. cudf::hash_partition(right_table, [eq_key_cols], num_partitions)
     → partitioned right table + partition offsets
  2. Sort within each partition by ts (sort by partition secondary key)

Probe (Execute):
  3. For each left row:
     a. Compute hash(eq_key) % num_partitions → bucket
     b. Binary search ts within right_table[offsets[bucket]..offsets[bucket+1])
     c. match = result - 1 (last right ts ≤ left ts in same partition)
     d. If result == offsets[bucket]: no match → NULL
  4. cudf::gather(right_payload, match_indices, NULLIFY) → gathered right cols
```

Custom CUDA kernel for step 3b:
```cuda
// One block per partition, threads cooperate on binary searches
__global__ void asof_partition_search_kernel(
    const int64_t* __restrict__ right_ts,      // right timestamps (partitioned, sorted within)
    const int32_t* __restrict__ right_offsets, // partition start indices
    const int64_t* __restrict__ left_ts,       // left timestamps (per partition bucket)
    const int32_t* __restrict__ left_bucket,   // partition bucket per left row
    int32_t* __restrict__ match_indices,       // output: right row index or -1
    int32_t num_partitions)
{
    int32_t part   = blockIdx.x;
    int32_t r_start = right_offsets[part], r_end = right_offsets[part + 1];
    // ... all left rows in this partition binary-search [r_start, r_end)
}
```

**Advantages over Option A**:
- Each thread only searches its partition (smaller range → better L2 cache hit rate)
- Right partitions that fit in shared memory (≤ 48 KB) can be cached → near-zero memory latency
- Scales better for high partition counts (1000+ symbols)

**Phase 2 also adds**: density profiler to choose between binary search and merge-scan paths.

---

## Critique of AM3D Framework (Adaptive Morsel-Driven Density Detector)

The AM3D framework described in early planning documents contains several
**incorrect or impractical claims** that must NOT be implemented as described:

### Real issues:

1. **"Hardware interrupt to abort a running kernel"** — CUDA does NOT support mid-kernel
   preemption from outside the kernel. There is no GPU runtime API for graceful kernel
   interruption. The correct approach is to check a device-side `volatile int* abort_flag`
   periodically _inside_ the kernel (cooperative abort).

2. **"JIT compilation to dispatch a kernel at runtime"** — technically possible via NVRTC,
   but compilation latency (hundreds of ms) would exceed any data-layout benefit on typical
   dataset sizes. This is not practical per-query.

3. **"GPU driver continuously monitors memory hit ratios during execution"** — hardware
   performance counters (CUPTI/nvperf) are not readable by application code _inside_ a running
   kernel. This would require a separate profiling pass, not real-time steering.

4. **"Cost function with fixed C_sequential and C_uncoalesced_fetch scalars"** — these are
   not constants; they vary with L2 hit rate, access stride, and SM occupancy. Treating them
   as fixed scalars in a runtime formula is unsound.

### What SHOULD be done (Phase 2 density profiler):

```cpp
// Sample 8192 rows from left and right (already in VRAM — no PCIe transfer)
// Compute average time delta between consecutive timestamps per partition
// μ_L = avg delta in left table;  μ_R = avg delta in right table
// ρ = μ_R / μ_L  (ratio of right sparsity to left sparsity)
//
// if ρ < MERGE_PATH_THRESHOLD (e.g. 10.0): route to linear merge-scan kernel
// else:                                     route to binary search kernel
//
// This decision is made ONCE before the join, not dynamically during execution.
```

The profiler kernel runs in O(sample_size / 256) blocks and adds < 1 ms overhead.
No JIT, no runtime interrupt, no hardware counter streaming.

---

## Phase 0: Planner Hook (DONE)

The planner hook at `src/plan/gpu_plan_comparison_join.cpp` now calls `PlanAsOfJoin(op)`
instead of throwing. `GPUPhysicalAsOfJoin` is created from the `LogicalComparisonJoin`
conditions by extracting:
- Equality conditions → `lhs_partition_col_idxs`, `rhs_partition_col_idxs`
- Inequality condition → `lhs_order_col_idx`, `rhs_order_col_idx`, `comparison_type`

---

## Phase 1: GPUPhysicalAsOfJoin Operator (IMPLEMENTED)

Uses Option A algorithm. Files:

```
src/
  include/operator/
    gpu_physical_asof_join.hpp        # operator declaration + cudf_asof_join() forward decl
  operator/
    gpu_physical_asof_join.cpp        # Sink / Execute / BuildPipelines
  cuda/cudf/
    cudf_asof_join.cu                 # cudf_asof_join() implementation
  plan/
    gpu_plan_comparison_join.cpp      # PlanAsOfJoin() + hook in CreatePlan()
```

### Phase 1 limitations (to fix in Phase 2):
- Only supports `COMPARE_GREATERTHANOREQUALTO` and `COMPARE_GREATERTHAN`
- Only supports 0 or 1 equality partition key (throws for N > 1)
- No density-based routing (always uses binary search)
- Single-threaded Sink (no parallel accumulation)

---

## Phase 2: Optimization (Post-correctness)

1. **Multi-key boundary check**: extend kernel to check all N equality keys
2. **Option B (hash_partition)**: per-partition binary search with shared memory
3. **Density profiler**: sample-based ρ = μ_R / μ_L, route to merge-scan vs binary search
4. **Merge-scan kernel**: for ρ ≈ 1 (dense interleaved timestamps)
5. **LESSTHAN/LESSTHANOREQUALTO**: sort descending, analogous kernel

---

## Phase 3: Custom CUDA Kernel Optimization

Coalesced binary search within shared-memory-cached right partitions.
See Option B kernel sketch above. Only implement after Phase 2 shows binary search
is the bottleneck (profile with nsys before writing custom kernels).

---

## Phase 4: Benchmark Strategy

### Dataset

**Option A (recommended):** Synthetic tick data (immediately available):
- `trades`: 500M rows (symbol SMALLINT, ts BIGINT ns-epoch, price DOUBLE)
- `quotes`: 100M rows (symbol SMALLINT, ts BIGINT ns-epoch, bid DOUBLE, ask DOUBLE)
- 1000 symbols, Poisson-distributed timestamps (λ = 1 tick/ms for quotes)
- Generation: `scripts/gen_asof_benchmark.py` using cupy/numpy → parquet

**Option B (realistic):** Uniswap V3 DEX event logs from Google BigQuery
`bigquery-public-data.crypto_ethereum`. ~500M rows total.

### Competitors

| System | Mode | Why |
|--------|------|-----|
| DuckDB CPU | Single-node, all cores | Correctness reference + best CPU baseline |
| ClickHouse | Single-node | Has ASOF JOIN with `full_sorting_merge` — best CPU ASOF |
| Sirius GPU | Single-node RTX 6000 | Target |

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
-- DuckDB CPU: generate reference output
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

## Implementation Order

1. [DONE] Planner hook: `PlanAsOfJoin()` + `GPUPhysicalAsOfJoin` stub
2. [DONE] `Sink()`: store right table in GPU memory
3. [DONE] `cudf_asof_join()`: sort + upper_bound + boundary check + gather
4. [DONE] `Execute()`: wire it up, run correctness test vs DuckDB CPU on small data
5. [ ] Multi-key boundary check kernel (Phase 2)
6. [ ] Option B: hash_partition + per-partition binary search (Phase 2)
7. [ ] Density profiler + merge-scan path (Phase 2)
8. [ ] Benchmark vs DuckDB + ClickHouse on 500M-row synthetic dataset
