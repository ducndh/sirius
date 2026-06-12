# Gate-0 iteration 1 — TPC-H sf5, A100, 2026-06-12

Full sweep: 4 churn rates x 4 arms x 15 ticks, official RF1/RF2 refresh
streams + 5K-row update stream. **Integrity: 16/16 cells completed, 120/120
merge verifications pass, 0 fallbacks, 0 errors.** Data:
`results/sweep_tpch_a100.jsonl`, plots `results/sweep_tpch_a100_*.png`.

## Headline numbers (steady-state read latency per query, mean)

| rho | cpu | arm_a (inval+reupload) | arm_b (merge-on-read) | arm_c (periodic fold) |
|---|---|---|---|---|
| 0.001 | 0.035 -> 0.026s | 0.51 -> 0.49s | 0.13 -> 0.16s | 0.14 -> 0.16s |
| 0.03  | 0.031 -> 0.040s | 0.49 -> 0.50s | 0.17 -> 0.18s | 0.17 -> 0.16s |

(x -> y = ticks 0-2 -> ticks 12-14.) arm_a additionally pays ~6s per write
batch (checkpoint 0.02s + engine restart 4.3s + rewarm 1.7s); arm_c pauses
~10s per fold.

Per-query at rho=0.01 (steady): q1 — cpu 0.05s, arm_b 0.24s, arm_a 1.09s.
arm_b split: gpu ~0.12s flat, delta 0.02 -> 0.04s (growing but tiny).

## Verdict vs PASS/KILL: NEITHER yet — the regime is wrong on 3 axes

The harness measured correctly, and the structure is visible (arm_a >>
arm_b > cpu; merge tax grows with delta), but the breaking workload did not
appear because:

1. **The pin is not actually pinned.** arm_b's GPU reads cost the same warm
   and steady (q1 0.28 vs 0.24s) -> `scan_cache_level=table_gpu` produced no
   cache-hits on this path; every read re-decodes/re-uploads ~30M rows. A
   truly GPU-resident pin would read in ~ms, which would (a) make the GPU
   arms competitive and (b) expose the delta term as the dominant growth.
   The documented cached path in this codebase is the PARQUET route
   (`CREATE VIEW ... read_parquet` + table_gpu; pinned-cache PRs #783/#881
   are parquet-path) and the explicit `pin_table(path, tier:='gpu')`
   function. **Iteration 2: export pin_snap to parquet and read it through
   the cached parquet route / pin_table.**
2. **sf5 is too small to motivate GPU at all** — CPU DuckDB wins every query
   (q1 0.05s at 30M rows). Sirius's own benchmarks win at SF >= 20-100.
   **Iteration 2: SF >= 20 (dbgen time ~minutes-to-an-hour; disk fine).**
3. **The delta term doesn't bite yet** — CPU DuckDB chews 13M delta rows in
   40ms. With (1) fixed, delta_s/gpu_s becomes the policy-relevant ratio;
   with (2), absolute deltas grow 4-20x. Optionally add a join query
   (Q3-shape vs a static dim copy) to make per-read work heavier.

## What iteration 1 DID establish (real findings)

- **Current dev pays freshness-by-recompute on every read**: the default GPU
  route re-reads/re-uploads per query (~0.5s/q here) regardless of writes —
  there is no usable GPU-resident pin for native tables under DML. Combined
  with FINDINGS.md #1 (native route permanently wrong after deletes), the
  system today offers *correct-but-recompute-per-query* or *fast-but-wrong* —
  nothing in between. That IS the paper's gap, measured.
- arm_a's fixed ~6s/batch overhead is restart-dominated (4.3s Sirius init),
  i.e., an engine-level invalidation primitive (drop one cache entry instead
  of the process) is worth ~70% of arm_a's refresh cost — relevant to #819's
  design.
- The merge-on-read arms beat invalidate+reupload by ~3x on reads at ALL
  churn rates even with an unpinned GPU term — the ordering arm_b < arm_a is
  robust here, but both lose to CPU at this scale.

## Iteration 2 checklist

- [ ] pin_snap as parquet + `pin_table`/cached-parquet route (verify
      cache-hit: warm >> steady) — the critical fix
- [ ] SF=20 (or 50) regenerate + rerun sweep
- [ ] optional: add join query for heavier reads (excluded from arm_b merge
      set, or merged with static-dim linearity)
- [ ] then re-evaluate PASS/KILL; only after that, GH200.
