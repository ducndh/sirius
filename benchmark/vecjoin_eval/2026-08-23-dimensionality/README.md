# 2026-08-23 — E4: does any of this hold at d=960? (GIST1M)

**Question.** Every performance and recall number recorded before this came from SIFT1M at
d=128, and ANN behaviour is strongly dimension-dependent. Which conclusions survive 7.5× the
dimensionality?

**Answer: the one that matters survives and sharpens. Our advantage is EXACT at high recall, not
approximate search — and that is true at both dimensionalities.**

## Setup

GIST1M, 1,000,000 × FLOAT[960], one A100. Sirius `vecjoin-approx-cluster` @ `7a34e809` under
`../../vecjoin_bench.yaml`, 64 clusters. cuVS IVF-Flat, n_lists=1024. k=10, L2.
Recall against the packaged ground truth.

⚠️ **GIST1M ships 1,000 queries, not SIFT's 10,000.** This is a **1k × 1M** regime. Absolute times
are NOT comparable to the SIFT 10k × 1M numbers — compare the *shape* of the curves.

## Results

**Sirius** — fit 0.091 s · **materialize `ORDER BY cluster_id` 10.339 s** · exact **0.263 s** @ 1.0
· approx p1 0.131 · p4 0.172 · p8 **0.313 s @ 0.780** · p16 0.586 s.

**cuVS IVF-Flat** — build 0.89 s · p1 0.0034 @ 0.208 · p4 0.0127 @ 0.474 ·
p16 **0.0513 @ 0.775** · p64 0.2056 @ 0.962 · p256 **0.8126 @ 0.9988**.

| comparison | Sirius | cuVS | winner |
|---|---|---|---|
| search only, recall ~0.775 | 0.313 s | 0.051 s | **cuVS 6.1×** |
| build-inclusive, recall ~0.775 | 10.743 s | 0.941 s | **cuVS 11.4×** |
| search only, **high recall** | **0.263 s** @ 1.0 | 0.813 s @ 0.9988 | **Sirius 3.1×** |
| build-inclusive, **high recall** | **0.263 s** @ 1.0 | 1.703 s @ 0.9988 | **Sirius 6.5×** |

## What generalizes from d=128

1. **The exact-at-high-recall advantage holds.** SIFT self-join: Sirius 9.88× at recall 1.0.
   GIST: **6.5× build-inclusive, 3.1× search-only.** Same mechanism both times — cuVS's IVF must
   probe ever more lists to approach recall 1.0 (p256 costs 16× its p16), while our GEMM fold does
   not degrade that way and needs no build at all for exact.
2. **The approximate path is not competitive, and is worse here.** cuVS is 6.1× ahead search-only
   at matched recall and 11.4× build-inclusive. At d=128 J0 already found approx losing to our own
   exact path one-shot; at d=960 it loses more clearly — exact 0.263 s beats p8's 0.313 s outright
   while also being exact.
3. **Materialize still dominates the build** and scales with corpus bytes: 1.176 s at 512 MB
   (d=128) → 10.339 s at 3.84 GB (d=960), i.e. 8.8× for 7.5× the data. S4 remains the top
   engineering item.
4. **Recall at a matched config is lower**, 0.780 vs 0.893 — GIST is genuinely harder, with weaker
   cluster structure at high dimension. Expected, and it is why the approximate path suffers.

## The confound I nearly reported as a finding

Materialize is 39× the join here versus 3.6× at d=128, which looks like the ratio collapsing with
dimensionality. **It is mostly the 10× smaller probe set**, not dimensionality: fewer probes make
the join cheap while the corpus-side materialize is unchanged. The defensible statement is the one
in point 3 — materialize scales with corpus *bytes* — not a ratio across two datasets with
different probe counts.

## What this does NOT say

- Only 1k probes on GIST. A 10k-probe GIST run would move every Sirius/cuVS ratio and is not
  measured.
- Only the search regime. No GIST self-join (the many-to-many regime where SIFT looked best).
- cuVS's n_lists=1024 was tuned at d=128 and carried over unchanged; a d=960 sweep might favour it
  further.
- CAGRA omitted deliberately — its params are known-broken here (recall ~3e-4, QUEUE S3), and
  carrying a broken configuration into a new dataset would produce a flattering non-result.
