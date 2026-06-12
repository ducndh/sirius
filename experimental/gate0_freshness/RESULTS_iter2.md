# Gate-0 iteration 2 — TPC-H sf20, real pin via pin_table, A100, 2026-06-12

Full sweep, official RF1/RF2 + update stream, all GPU arms reading a
column-pruned parquet pin via `CALL pin_table(tier='gpu')`. **Integrity:
16/16 cells, 120/120 merge verifications, 0 fallbacks, 0 errors.** Data:
`results/sweep_tpch_sf20_a100.jsonl` + PNGs.

## Headline: the policy tradeoff is now measurable

Steady-state read latency per query (ticks 0-2 -> 12-14):

| rho | cpu | arm_a | arm_b | arm_c |
|---|---|---|---|---|
| 0.001 | 0.063 -> 0.075 | 0.263 -> 0.265 | 0.300 -> 0.305 | 0.315 -> 0.316 |
| 0.03  | 0.080 -> 0.078 | 0.241 -> 0.247 | 0.294 -> 0.327 | 0.299 -> 0.292 |

Plus per-write-batch refresh: arm_a pays **pin_s ~= 5.3s** (the real
re-upload; export ~6s more is a harness artifact), arm_c pays the same per
fold (every K=5 ticks), arm_b never.

**The cost model has a job now (within the GPU world):**

- Per tick: arm_a = 5.3s + R x 0.25s; arm_b = R x (0.25 + delta_tax)s.
  At our R=3 reads/tick arm_b wins outright; **break-even R ~= 105
  reads per write batch** — the same queries-per-update quantity that decided
  GATE-6, re-derived in the relational domain.
- **arm_b's delta tax grows with accumulated delta**: 0.030 -> 0.067s/query
  at rho=0.03 (54M delta rows by tick 14) — monotone, doubling over the
  horizon. The break-even therefore SHIFTS over time, which is exactly what
  makes recompact timing (arm_c's K) a genuine optimization rather than a
  constant.
- arm_c interpolates as designed: arm_b-level reads + bounded fold pauses.

## The remaining gap to PASS: CPU still wins the reads

DuckDB CPU does these queries in 0.06-0.08s at 120M rows — faster than any
GPU arm (~0.25s, of which a large share is per-call gpu_execution overhead +
result collection, not scan). Single-table TPC-H aggregates at SF=20 simply
do not motivate GPU residency on a 100-core box. Conclusion unchanged from
iter-1 but now isolated: **the missing ingredient is read workload weight,
not pin mechanics.**

## Iteration 3 (the decisive one): join-heavy mergeable reads

Add TPC-H **Q14/Q19-shape queries (lineitem JOIN part)**:
- `part` is untouched by RF1/RF2 -> the query stays LINEAR in lineitem ->
  still mergeable for arm_b (delta arms join delta x part on CPU; or
  pre-pin part on GPU once, never re-pinned).
- Joins at SF>=20 are where Sirius historically beats CPU DuckDB (its own
  benchmark suite) — this gives the GPU arms a reason to exist, making the
  CPU baseline an honest loser on reads and the freshness policy the binding
  decision.
- Requires: load `part` in setup (dbgen -T P, 4M rows at sf20), pin it once,
  add q14 to the query set + merge rules unchanged (linear).
- Optionally: longer horizon (30+ ticks) and/or rho=0.1 cells to stress the
  delta-tax growth; high-read-rate cells (R=20) to bracket the break-even
  empirically.

## Iteration-2 artifacts established along the way

- `pin_table(tier='gpu')` = the real pin (25x warm/steady gap); `table_gpu`
  view caching gave none — worth documenting upstream.
- arm_a no longer needs engine restarts (unpin/pin) — iter-1's 4.3s restart
  was the cost of having NO invalidation primitive on the default route.
- All 240 merged results across both iterations verified exact vs CPU truth:
  the harness-level merge-on-read semantics are sound.
