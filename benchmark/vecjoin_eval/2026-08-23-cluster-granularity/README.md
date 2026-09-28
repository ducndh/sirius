# 2026-08-23 — S1: the clustered fold pays per SPAN-VISIT, not per byte

**Question.** J2 found 256 clusters losing badly to 64 — at 256 the approximate path is slower
than our own exact join past 1 probe. The reading was "cost is per-span, not per-byte", but that
comparison confounded two things: more clusters means **more spans** *and* **fewer bytes per
probe**. Which one costs?

**Answer: span-visits. At identical bytes scored, 16× the span-visits costs 7.4× the time.**

## The control

Hold the scanned fraction constant and vary only the span count. 8/64 and 32/256 both probe
12.5% of the corpus, so both should score the same number of pairs — and `PRUNE_DEBUG` confirms
they do, rather than leaving it assumed.

| clusters | probes | probe runs | span-visits | **pairs scored** | search+merge | total |
|---|---|---|---|---|---|---|
| 64 | 8 | 64 | **512** | **13.34%** | **0.396 s** | 0.455 s |
| 256 | 32 | 256 | **8,192** | **13.01%** | **2.921 s** | 2.969 s |

**Equal work (13.34% vs 13.01% of pairs scored). 16× the span-visits. 7.4× the time.**

`search+merge runs` is ~99% of total in both cases — every setup phase (cluster index, probe
assignment, sort order, gather, scatter) is ≤ 15 ms and irrelevant. The fold *is* the cost.

## Mechanism

A "probe run" is one group of probes sharing a cluster, and there is one per cluster: 64 runs at
64 clusters, 256 at 256. Each run visits `n_probes` corpus spans, so span-visits = clusters ×
probes. Scaling is sublinear in span-visits (16× → 7.4×) but nowhere near flat, so there is real
per-visit overhead on top of the per-byte work.

Per span-visit: **773 µs at 64 clusters vs 357 µs at 256** — the larger spans are individually more
expensive, as expected, but not nearly enough to offset having 16× as many.

## Consequences

1. **Span batching is the lever, and this confirms it is worth building** (QUEUE S1). The fold
   should visit fewer, larger contiguous ranges; coalescing adjacent spans within a run, or
   processing multiple runs against one staged corpus chunk, attacks the term that actually
   dominates.
2. **It explains why our optimum is the opposite of cuVS's.** cuVS wants `n_lists=1024`; we want
   64. Their kernel amortizes many small lists well; ours pays per visit. That asymmetry is now a
   measured property rather than a curiosity.
3. **Any Sirius approx number without its cluster count is meaningless** — a 4× change in cluster
   count at equal recall moved the time 6.5×.

## What this does NOT say

- One corpus (SIFT1M, d=128), one probe set (10k), one k. The d=960 run in
  `../2026-08-23-dimensionality/` did not repeat this control.
- The two configurations reach *different recall* (8/64 ≈ 0.893 from J2; 32/256 unmeasured here).
  The control fixes **work**, not recall — it isolates the cost mechanism, and is not a
  recall-matched comparison.
- Sublinearity (16× visits → 7.4× time) means per-visit overhead is real but not the whole story;
  a batching fix should be expected to recover much of the gap, not all of it.

## Reproduce

```bash
SIRIUS_VECTOR_JOIN_PHASE_DEBUG=1 SIRIUS_VECTOR_JOIN_PRUNE_DEBUG=1 \
  duckdb -unsigned /var/tmp/vj/s1.db < s1_phase_profile.sql
```
Both diagnostics exist because two earlier published diagnoses of this operator turned out wrong
when finally measured (`../../README.md` §3). Use them before forming a theory.
