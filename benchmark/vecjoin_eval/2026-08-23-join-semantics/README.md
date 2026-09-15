# 2026-08-23 — J1: the corpus is an intermediate result (the canonical join case)

**Question.** Every earlier benchmark handed the ANN library a curated corpus it had indexed in
advance — that is vector *search*. In a join the right side is whatever the plan produces. If the
corpus is the output of a filter, no index can pre-exist. Does our position improve there?

**Answer: no. We lose, 1.69 s vs 1.12 s at matched recall — and the reason is entirely our data
plumbing, not our operator.**

## Environment

A100-SXM4-40GB, 64 cores, 120 GB RAM. Sirius `vecjoin-approx-cluster` @ `ae5e3d0d`, `80-real`,
under `../../vecjoin_bench.yaml`. SIFT1M, 10k probes, k=10, L2.
Predicate `id % 10 < 3` → **300,000 of 1,000,000 rows (30%)**, identical on both sides.
Oracle: exact brute force over the *filtered* subset, on GPU — the right answer for *this* query,
not for the unfiltered corpus.

## Results — same query, matched recall

| | stage | time |
|---|---|---|
| **Sirius** | materialize filtered corpus | **1.226 s** |
| | checkpoint | 0.020 s |
| | pin | 0.031 s |
| | **join (exact)** | **0.415 s** |
| | **total** | **1.692 s** — recall **0.9998** |
| **cuVS practitioner path** | export surviving rows | 0.048 s |
| | host→device | 0.108 s |
| | index build | 0.353 s |
| | **search (n_probes=1024, exhaustive)** | **0.598 s** |
| | map ids back to database ids | 0.009 s |
| | **total** | **1.115 s** — recall **0.9998** |

Both land on 0.9998 rather than 1.0 because of ties at equal distance in the oracle's
tie-breaking; both are exact in substance.

cuVS's cheaper recall points, for the curve: 0.569 s @ 0.907 (16 probes), 0.607 s @ 0.994 (64),
0.762 s @ 0.9997 (256).

## The finding: our operator wins, our plumbing loses

Split the same numbers by what they are doing:

| | prepare the corpus | do the join |
|---|---|---|
| Sirius | 1.277 s (materialize + checkpoint + pin) | **0.415 s** |
| cuVS | **0.509 s** (export + h2d + build) | 0.598 s |

**Our join kernel is 1.4× faster than cuVS's exhaustive search. Our corpus preparation is 2.5×
more expensive than theirs — and it is 75% of our total.** Materializing 300k filtered rows
(~150 MB) costs 1.226 s, about 125 MB/s, which is slow enough to suspect the write path rather
than any fundamental limit.

This is the same shape as J0, where `CREATE TABLE ... ORDER BY cluster_id` cost 1.176 s and sank
the approximate path. **Two independent experiments now point at corpus materialization as the
dominant cost.**

## What would flip this result

The join itself already wins. If the filtered corpus did not have to be materialized and pinned —
if the operator could consume a filtered *scan* of an already-pinned base table, pushing the
predicate down the way `sirius_knn_join_rel` already does for the **probe** side (recorded 6.8× on
a filtered probe) — the Sirius total would approach 0.415 s plus scan cost, against cuVS's 1.115 s.
That is a 2.7× win instead of a 1.5× loss, from one missing capability.

**Corpus-side predicate pushdown is now the highest-value engineering item in the project**
(QUEUE S4). It is worth more than any further baseline: it is the difference between losing and
winning the canonical join case.

## J2 — the selectivity sweep, and the finding J1 could not see

Same query, corpus-side predicate swept. cuVS at `n_probes=1024` (exhaustive) throughout, so every
row is matched-recall (0.999–1.000).

| selectivity | Sirius total | cuVS total | winner | **Sirius join** | **cuVS search** | **join winner** |
|---|---|---|---|---|---|---|
| 10% | 0.872 s | 0.489 s | cuVS 1.78× | **0.192 s** | 0.216 s | Sirius 1.12× |
| 30% | 2.026 s | 1.122 s | cuVS 1.81× | **0.263 s** | 0.604 s | Sirius 2.30× |
| 50% | 3.061 s | 1.679 s | cuVS 1.82× | **0.335 s** | 0.978 s | Sirius 2.92× |
| 90% | 4.295 s | 3.833 s | cuVS 1.12× | **0.456 s** | 2.354 s | **Sirius 5.16×** |

**Our join kernel's advantage grows with corpus size — 1.12× at 10% to 5.16× at 90% — because our
scan-and-fold scales far better than cuVS's IVF search (0.192→0.456 s, i.e. 2.4× for 9× the data,
versus 0.216→2.354 s, i.e. 10.9×).** But our materialize grows just as fast (0.344→3.778 s) and
cancels the entire advantage out. cuVS wins every row on total, and by a *shrinking* margin.

### What S4 is worth, quantified

If corpus-side pushdown removed the materialize, leaving pin + join:

| selectivity | Sirius (pin+join) | cuVS | result |
|---|---|---|---|
| 10% | 0.213 s | 0.489 s | **Sirius 2.3×** |
| 30% | 0.284 s | 1.122 s | **Sirius 4.0×** |
| 50% | 0.365 s | 1.679 s | **Sirius 4.6×** |
| 90% | 0.497 s | 3.833 s | **Sirius 7.7×** |

**One missing capability separates losing every row from winning every row by 2–8×.** That is the
strongest argument the project has for a single piece of engineering work.

## What this does NOT say

- **Materialize timing is noisy.** The 30% materialize measured 1.226 s in J1 and 1.732 s in J2 —
  a 40% spread. The conclusion is robust to it (the gap is ~1.8×), but do not quote a single
  materialize figure to three digits.
- cuVS's 90% `h2d` jumped to 0.618 s from 0.124 s at 50%, out of line with the data growth —
  likely device memory pressure. Its 90% total may be slightly pessimistic.
- Only the exact path. The approximate path would add the clustering build measured in J0
  (1.714 s), which at this scale makes it strictly worse.
- The practitioner path here is generous to cuVS in one way and harsh in another: generous because
  the "export" is a numpy slice rather than a real DuckDB→host round trip; harsh because a real
  deployment would keep the index if the filter were stable. Both are noted rather than adjusted.

## Reproduce

```bash
python3 j1_intermediate_corpus.py --selectivity 0.3 --n-probes 1024   # cuVS side
duckdb -unsigned /var/tmp/vj/j1.db < j1_sirius.sql                    # Sirius side (see run_j1_sirius.log)
```

---

# J3 — the many-to-many self-join, one box (the regime the operator is FOR)

**Question.** The 6.64 s vs cuVS 5.73 s near-parity figure — the one result that looked best for
us — was measured across two boxes and carried a ⚠. Does it survive a single-box measurement?

**Answer: it more than survives. At high recall we win, and at exact recall we win by 9.9×.**

## Setup

1M × 1M self-join, 10M output pairs, k=10, L2, one A100. Sirius 256 clusters; cuVS IVF-Flat
n_lists=1024. A self-join is not a search workload: with 1M probes a batch spans a handful of
clusters and pruning can skip, where 10k probes span nearly all of them — which is exactly why the
two regimes disagree.

**Oracle is not self-referential.** cuVS recall is measured against its own exhaustive run
(n_probes=1024), and that run was checked against a cupy brute force on 2000 sampled rows:
**0.9989 agreement**. Sirius recall is against its own exhaustive run. `feedback_green_suite_not_a_gate`
is in memory precisely because a self-consistent oracle hid silent wrong results before.

## Matched-recall comparison (cuVS interpolated between its measured points)

| recall | Sirius | cuVS | winner |
|---|---|---|---|
| 0.550 | 0.411 s | 0.663 s | **Sirius 1.61×** |
| 0.677 | 1.120 s | 1.084 s | cuVS 1.03× |
| 0.876 | 3.852 s | 3.764 s | cuVS 1.02× |
| 0.992 | 14.507 s | 19.768 s | **Sirius 1.36×** |
| **1.000 (exact)** | **27.251 s** | **269.392 s** | **Sirius 9.88×** |

Raw points — Sirius: p1 0.411 @ 0.550 · p4 1.120 @ 0.677 · p16 3.852 @ 0.876 · p64 14.507 @ 0.992
· exact 27.251 @ 1.0 · kmeans fit 0.101 s.
cuVS (build 0.333 s): p1 0.363 @ 0.460 · p4 1.338 @ 0.753 · p16 5.148 @ 0.946 · p64 21.252 @ 0.997
· p256 82.066 @ 0.9999 · p1024 269.392 @ 1.0.

## What this says

**The advantage is at the high-recall end, and it grows.** Parity in the middle band
(0.68–0.88), we win 1.36× at 0.99, and 9.88× at exact. cuVS's IVF cost scales with
`n_probes × queries`, so reaching recall 1.0 means probing every list — 269 s. Our exact path is a
GEMM fold that does not degrade that way: 27 s.

**This is the regime a join operator is for**, and it is the opposite of the search-regime result
(where cuVS is 3–8× ahead). Both are real; they answer different questions. Quote them together or
neither.

**The old cross-box figure was pessimistic about us.** 6.64 s @ 0.873 recorded vs **3.852 s @
0.876** here — the recall reproduces to three decimals and the box is ~1.66× faster (exact
re-timed 27.25 s vs 45.2 s recorded).

## Caveats

- **Sirius's build column is understated here, as in J0.** The reported 0.101 s is `kmeans_fit`
  only; `assign` + `ORDER BY cluster_id` are not in it (J0 measured the full pipeline at 1.714 s
  on a 1M corpus). Build-inclusive, the approximate rows each gain ~1.7 s and cuVS's gain 0.333 s
  — which *widens* our exact-path win, since exact needs no clustering at all.
- Recall for each system is against its own exhaustive run. Cross-checked for cuVS (0.9989 vs
  brute force); Sirius's exact path was separately verified byte-identical to DuckDB `vss_join`
  on 2026-08-22.
- One dataset (SIFT1M), one k, one metric.

---

# J4 — composition: what it costs to let SQL touch the join's output

**Question.** "An ANN library is not a SQL operator" is the composability claim. As a number: what
does each system pay to run an aggregate over 10M join pairs?

**Answer: we pay +1.6%; cuVS pays 0.78 s. Real, but modest — this is not a knockout argument.**

## Sirius — the pairs never leave the GPU

1M × 1M self-join, 10M pairs, exact, one A100.

| shape | time | vs raw join |
|---|---|---|
| J4a raw join, 10M pairs to the client | 26.928 s | — |
| J4b global aggregates in-query | 27.168 s | **+0.24 s (+0.9%)** |
| J4c grouped aggregate (`GROUP BY left_id`) | 27.359 s | **+0.43 s (+1.6%)** |

## cuVS — the round trip before SQL can see the pairs

Search **excluded**; only the round trip is timed. Deliberately generous: `cp.asnumpy` rather than
a row loop, Arrow zero-copy registration rather than `INSERT`, same `memory_limit`.

| stage | time |
|---|---|
| device → host | 0.051 s |
| into DuckDB (Arrow zero-copy) | 0.472 s |
| **round trip subtotal** | **0.523 s** |
| grouped aggregate | 0.260 s |
| **total added over search** | **0.782 s** |

## The honest reading

**Composition costs us 0.43 s and costs cuVS 0.78 s on the same 10M pairs — 1.8× cheaper for us,
in absolute terms.** Both are aggregating identical output, so this part *is* apples-to-apples.

But keep it in proportion: 0.78 s is ~15% of cuVS's 5.26 s search at this scale, and it shrinks
with the result size. **A 10M-pair result is the best case for this argument and it still only
buys ~0.35 s.** Composability is mostly a *qualitative* claim — one SQL statement, predicates that
push down, arbitrary composition with joins and aggregates — and the quantitative round-trip
saving is small. Do not lead with this number.

**And the round trip only exists if the downstream work is SQL.** A user whose next step is numpy
or pandas never pays it — they aggregate the arrays in place. The 0.472 s DuckDB ingest, which
dominates the round trip, is the cost of *entering a database*, not of using a library. State the
assumption explicitly whenever this is quoted.

## Caveats

- Sirius's composition overhead is measured on the **exact** join (26.9 s base); cuVS's round trip
  is measured against an **approximate** search (5.26 s @ recall 0.946). The overheads themselves
  are comparable because both process the same 10M pairs, but the percentages are not.
- `round(distance,3)` over the join output still errors (QUEUE S2), so the projection here is
  aggregates only. A scalar function would have measured that bug rather than composition.


---

## ⚠️ J3 CORRECTED 2026-08-23 (late) — the comparison used cuVS's exact config, not its best

J3's 9.88× is against **cuVS IVF-Flat with every list probed**, which is cuVS's way of reaching
recall 1.0. It is not against cuVS's best index. Once CAGRA was fixed (QUEUE S3 — misconfigured
`build_algo`, not invalid), the self-join looks like this build-inclusive:

| | Sirius exact | cuVS IVF exhaustive | cuVS CAGRA (~0.98) |
|---|---|---|---|
| d=128 | 27.25 s @ **1.0** | 269.39 s @ 1.0 | **6.92 s** |
| d=960 | 120.77 s @ **1.0** | not run | **16.30 s** |

**cuVS CAGRA beats our exact join by 3.9× at d=128 and 7.4× at d=960** — at recall ~0.98, which it
cannot push to 1.0. So J3's number stands as written but must always carry "against cuVS's exact
configuration"; quoting it as "we beat cuVS on the self-join" would be wrong.

CAGRA's recall here is the 0.979 measured on the SIFT 10k query set at the same `itopk`; a
self-join has no packaged ground truth, so it is a proxy rather than a measurement on this exact
workload.
