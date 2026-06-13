# arm_d: granular invalidation results — TPC-H sf20, A100, 2026-06-13

Pin = 16 range-partitioned parquet chunks under one glob `pin_table`; refresh
rewrites only dirty chunks (tracked from the write log) then re-pins.
Comparison cells: arm_d vs arm_a (full re-pin) x clustered writes (official
RF1/RF2 only) vs scattered writes (RF + 5000-row uniform update stream).
**60/60 freshness verifies pass.** Data: `results/armd_sf20.jsonl`.

## Refresh cost per write batch (mean over 15 ticks)

| arm | writes | dirty chunks | export_s | pin_s (re-upload) | projected pin_s* |
|---|---|---|---|---|---|
| arm_a | clustered | full | 6.0-7.6 | 5.4 | - |
| arm_d | clustered | **1.0-1.5 / 16** | **0.8-0.9** | 2.75 | **0.17-0.25** |
| arm_d | scattered | **16 / 16** | 10.3-11.2 | 2.75 | 2.75 (no benefit) |

*projected = pin_s x dirty_fraction = what an engine-level PARTIAL re-pin
primitive would cost. Current `pin_table` re-uploads the whole glob no matter
how few files changed (measured), so the realized GPU-side saving is zero;
the export side is the realized saving today.

Steady read latency is identical across arms (~0.27s/q) — the pin read path
is the same; only refresh policy differs. (Side observation: the 16-file
glob pin uploads 2x faster than the equivalent single file — 2.75 vs 5.4s —
parallel ingest.)

## The three takeaways

1. **Under clustered churn (the official TPC-H refresh shape), granular
   invalidation wins big**: realized refresh 3.6s vs 11.4s total (~3x), and
   with a per-chunk re-pin primitive it would be ~0.2s vs 5.4s (**~25x**).
   That 25x is the concrete, measured value of adding partial re-pin /
   per-granule invalidation to the pin-table design (#819 input).
2. **Under scattered churn, granular invalidation is strictly WORSE than
   full**: a 5000-row uniform update stream (0.004% of the table) dirties
   all 16 chunks every tick — the birthday-paradox effect measured earlier
   at row-group granularity (5K scattered rows dirty 72% of 3,900 row
   groups; the 25x-larger clustered RF batch dirties ~4).
3. Therefore **granularity choice is itself a cost-model decision** driven by
   write clustering, not write volume: dirty-fraction estimation (free from
   DuckDB's version metadata / a write log) must be an input, and the policy
   space is {full, granular} x {when} — strictly richer than iteration 2's
   {re-pin, merge, fold} x {when}.

## Findings ledger update

Finding #6 (count(*) over multi-file pinned entry deadlocks) discovered while
building this — see FINDINGS.md. Query sets now use count(key).
