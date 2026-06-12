# Gate-0 findings on current dev (23c3f72b + ALP gate fix), 2026-06-12, A100

Empirical observations from building/running the harness. Items 1-3 are
upstream-relevant independent of the paper. Repro: `probe` phase +
`results/probe_{default,native}.jsonl`, TPC-H sf5 (30M-row lineitem),
churn = 10 official refresh pairs + 5000-row update stream.

## 1. Native scan ignores deletes — PERSISTENTLY (correctness bug, file-worthy)

`SET enable_gpu_duckdb_native_scan=true`, after INSERT/DELETE on lineitem:

| stage | GPU (native) | CPU truth | verdict |
|---|---|---|---|
| after un-checkpointed writes | 30,029,936 | 29,999,735 | sees inserts, misses deletes |
| after CHECKPOINT (no restart) | 30,029,936 | 29,999,735 | still wrong |
| after CHECKPOINT + engine restart | 30,029,936 | 29,999,735 | **still wrong — cold read of the file** |

The cold-restart case means this is not (only) a cache-invalidation gap:
DuckDB's checkpoint persists deletes for non-rewritten row groups as
serialized version/delete masks in the file, and the native scan walks the
data blocks without applying them. **Any `.duckdb` file that has experienced
sparse DML returns wrong results on the GPU-native path, forever.** Existing
tests never caught it because bulk-loaded benchmark files carry no masks.
Fix directions: (a) parse + apply persisted delete masks in
`duckdb_native_metadata`/decoder (a GPU bitmap AND — cheap), or (b) detect
mask presence per row group and fall back to CPU for those row groups.
(Related: sirius#819 covers the *transactional* slice; this is the on-disk
slice and bites even single-writer, fully-checkpointed files.)

## 2. `SET gpu_execution` defaults TRUE = transparent interception of ALL SQL

Plain SQL (no `CALL gpu_execution`) is routed to the GPU path by default.
Consequences observed: (a) any "CPU baseline" measured without
`SET gpu_execution=false` is silently GPU; (b) an unsupported construct in
intercepted SQL (`count(DISTINCT ...)`) raised
"Distinct aggregates not supported in GPU path yet" as a **FATAL that
invalidated the database** (restart required) instead of falling back.
At minimum the fatal-instead-of-fallback looks like a bug; the default-on
interception deserves a docs callout for benchmarking.

## 3. Constant SELECTs return 0 rows through the CLI

`SELECT 42 AS x` → well-formed result frame, 0 rows (header/type correct).
Queries with a FROM clause are unaffected. Likely the interception path (#2)
mishandling the dummy/constant scan. Stock DuckDB prints the row.

## 4. No scan route serves a consistent stale snapshot (harness design driver)

Under concurrent DML the default route is always FRESH (re-reads through
DuckDB's MVCC-merging scan; correct but pays per query), and the native
route is INCONSISTENT (#1). A merge-on-read experiment therefore cannot get
its pin-epoch snapshot from either route; the harness models the pin as a
frozen `pin_snap` table (immutable epoch copy — what #819's pinned cache
would hold). This is also the cleanest statement of the paper's gap: the
system today offers freshness-by-recompute or inconsistency — nothing in
between.

## 5. ALP/ALPRD codec gate (fixed on this branch, commit 88a0f82e)

`is_supported_fixed_width_codec()` omitted ALP(10)/ALPRD(11) although the
decode kernels exist, are dispatched, and are unit-tested → every
FLOAT/DOUBLE column silently fell back to CPU on native scan. Bit the
harness within minutes (TPC-H `l_quantity`/`l_extendedprice` predicates).
Extract as its own upstream PR with a float-column e2e regression test.
