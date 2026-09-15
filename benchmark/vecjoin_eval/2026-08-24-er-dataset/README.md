# 2026-08-24 — a real two-sided vector-join dataset (entity resolution)

**Built and validated. Ready to use once the open bugs are green** — see the gate at the bottom.

Why not SIFT/GIST/GloVe: ../../../design_vecjoin_demo_datasets.md.
Short version — they give one table plus a query *file*, so every "join" measured on them is search
with the query set relabelled, there are no scalar columns to filter on, and the ground truth is
top-k only.

## What is on disk (`/var/tmp/vj/data/parquet/`)

| file | rows | columns |
|---|---|---|
| `abt-buy_a.parquet` | 1,081 | `id`, `vec` FLOAT[384], **`price`**, `name` |
| `abt-buy_b.parquet` | 1,092 | same |
| `abt-buy_labels.parquet` | 9,575 labelled pairs (1,028 positive) | `left_id`, `right_id`, `label` |
| `amazon-google_a.parquet` | 1,363 | `id`, `vec` FLOAT[384], **`price`**, `manufacturer` |
| `amazon-google_b.parquet` | 3,226 | same |
| `amazon-google_labels.parquet` | 11,460 labelled pairs (1,167 positive) | `left_id`, `right_id`, `label` |

Source: `matchbench/{Abt-Buy,Amazon-Google}` on HF — the canonical labelled ER benchmarks.
Encoder: `all-MiniLM-L6-v2`, 384-dim, **L2-normalised**, so `metric => 'cosine'` and cosine
distance = `1 - dot`. Built by `build_er_dataset.py`; CPU, seconds — these are the *semantics*
datasets, SIFT stays the *scale* dataset.

**The `price` column is deliberately excluded from the encoded text** so it remains a usable scalar
predicate. Coverage: amazon-google_b 100%, _a 85%, abt-buy 39–54% — good enough for a range
predicate on amazon-google, thin on abt-buy.

## What this dataset makes possible that SIFT could not

- **Two real relations** with their own schemas — no half-splitting a corpus (which is what put
  cuVS's CAGRA into a degenerate state in I5).
- **A hybrid predicate**: `WHERE a.price BETWEEN … AND <vector-near>` on real attributes instead of
  `id % 10 < 3`. This is I5's last unmeasured shape.
- **Precision / recall / F1 for the join**, against labelled matches — not recall@k against our own
  exhaustive run.
- **A naturally threshold-shaped query** ("all pairs above similarity τ"), which is the mode Andy's
  radius kernel made fast (`ef56272b`).
- **A readable answer**: "these two SKUs are the same product."

## Validation — the embeddings do work (`validate_er.py` → `run_validate_er.log`)

Gold pairs separate clearly from non-matches:

| dataset | gold-pair median sim | labelled non-match median |
|---|---|---|
| abt-buy | 0.760 | 0.633 |
| amazon-google | **0.832** | **0.586** |

Join-style recall of the gold matches, retrieving every pair above τ from the full cross product:

| τ | abt-buy recall (pairs out of 1.18M) | amazon-google recall (pairs out of 4.40M) |
|---|---|---|
| 0.60 | 0.937 (18,066) | 0.933 (11,408) |
| 0.70 | 0.727 (5,858) | 0.832 (5,329) |
| 0.75 | 0.548 (2,794) | 0.733 (3,540) |
| 0.80 | 0.310 (1,110) | 0.591 (2,187) |

**Use amazon-google as the primary demo**: better separation, better F1, and a 4.4M-pair cross
product. **τ = 0.70–0.75** is the useful operating range.

### Two honest caveats

1. **F1 peaks at ~0.44 (amazon-google) and ~0.37 (abt-buy).** Published ER systems reach 0.6–0.9 on
   these — with *trained* matchers and blocking. A generic un-finetuned encoder is a weak matcher.
   That is fine here because **we are demonstrating the join, not the matcher**; do not present
   these F1 numbers as a quality result for the encoder.
2. **Precision in the labelled-set framing is a lower bound.** The labelled candidate set is
   deliberately adversarial — blocked hard negatives — so a low precision there does not mean the
   join is retrieving junk. Pairs outside the labelled set are *unknown*, not known-wrong.

## ⚠️ Gate before using this for anything reportable

**B1 makes `WHERE <score> … GROUP BY` return silently wrong rows**, which is exactly the shape every
query on this dataset needs. Numbers produced now would have to be thrown away.
See ../../QUEUE.md § OPEN BUGS and ../../HANDOFF_S5.md.

## Reproducing the environment

`torch` here is **CPU-only**, in a separate target dir, because the GPU wheel wanted a full CUDA
wheel set (`libcudnn`, then `libcupti`, …) and the tables are thousands of rows:

```bash
pip install --target /var/tmp/vjtools/py sentence-transformers pandas
pip install --target /var/tmp/vjtools/pycpu --index-url https://download.pytorch.org/whl/cpu torch
PYTHONPATH=/var/tmp/vjtools/pycpu:/var/tmp/vjtools/py HF_HOME=/var/tmp/vjtools/hf python3 build_er_dataset.py
```
Raw CSVs are cached at `/var/tmp/vj/er/` (ephemeral — re-fetch from HF, they are ~100 KB each).

---

# The engine harness — correctness and expressiveness across every available baseline

**Ran 2026-08-24.** amazon-google, τ = 0.70 cosine similarity, 1,363 × 3,226 = **4.4M pairs**.
Scripts: `er_baselines.py` (cupy / cuVS / FAISS), `er_sirius.sql`, `er_sirius_hybrid.sql`,
`er_duckdb.sql` (plain + VSS), pgvector inline.

> ⚠️ **The timings below are fixed-cost dominated and must not be quoted as performance.** The
> entire cross product is 3.38 GFLOP ≈ **0.4 ms** of A100 compute. SIFT/GIST/GloVe stay the scale
> datasets. What this harness establishes is *whether each engine can express the query and whether
> it gets the right answer* — which is scale-independent.

## Q1 — "every pair above similarity τ". Reference answer: **5,329 pairs**

| engine | pairs | time | |
|---|---|---|---|
| **Sirius** (`join_mode => 'threshold'`, ported radius kernel) | **5,329** ✅ | 0.162 s | **byte-identical to the reference set** |
| cuVS `brute_force` top-k, k=100 | 5,329 ✅ | 0.001 s | correct *only because k was guessed high enough* |
| **cuVS `brute_force` top-k, k=10** | **3,964** 🔴 | 0.003 s | **SILENTLY INCOMPLETE — 156 rows hit the k bound** |
| cupy cross product + mask | 5,329 ✅ | 0.335 s | the reference |
| FAISS-CPU `IndexFlatIP.range_search` | 5,329 ✅ | 0.144 s | real radius API; agrees 5,329/5,329 |
| pgvector | 5,329 ✅ | 0.523 s | |
| plain DuckDB `array_cosine_similarity` | 5,329 ✅ | 1.641 s | |
| DuckDB VSS + HNSW index | 5,329 ✅ | 1.180 s | **index does not serve a range predicate** — `EXPLAIN` shows `BLOCKWISE_NL_JOIN` over two `SEQ_SCAN`s, no index node, after a 1.110 s build |

## Q2 — the ER question: "products with ≥2 candidates". Reference: **673**

Sirius **673** / 0.071 s · plain DuckDB **673** / 1.643 s · pgvector **673** / 0.542 s. All correct.
(Note this is a `GROUP BY` over a *threshold* join with no `WHERE` on the score, so it dodges B1.)

## Q3 — the HYBRID shape, and the most interesting result here

Add a real scalar predicate: `b.price BETWEEN a.price*0.8 AND a.price*1.25`.

| engine | pairs | time |
|---|---|---|
| plain DuckDB | 1,789 ✅ | **0.177 s** |
| Sirius | **1,790** 🟡 | — |

**DuckDB got 9× faster by adding a predicate** (1.641 s → 0.177 s). `EXPLAIN` shows *why*, and it is
not what "predicate ordering" would suggest — it is stronger:

```
FILTER (array_cosine_similarity ...)          <- the vector op is now just a filter
  PIECEWISE_MERGE_JOIN
    Conditions: price >= (price * 0.8), price <= (price * 1.25)
      SEQ_SCAN price / SEQ_SCAN price
```

**DuckDB made the cheap scalar predicate the JOIN** — a band/range join — and demoted the vector
similarity to a filter over its much smaller output. It is not evaluating predicates in a better
order on the same 4.4M pairs; it is *producing far fewer pairs*.

**Sirius structurally cannot make that choice.** In our design the vector comparison **is** the join
operator, so the planner has no plan available in which a different, cheaper predicate drives the
join. We compute the full vector join and filter afterwards. That is the architectural cost of
modelling similarity as an operator rather than as a predicate, and it is I5's last open shape.

⚠️ **CORRECTED same day — see QUEUE § S7.** An earlier draft of this section said a CPU engine with
a band join "beats a GPU that cannot." **Never measured, and false at this scale.** Sweeping the
band, Sirius wins at every selectivity (41× / 13× / 7.9× / 3.4× as the band tightens from none to
±2%) and is FLAT at 0.040–0.050 s. The finding is about **slope**: DuckDB improves 12× as the
predicate tightens and we improve 0×, so the gap closes. A crossover is plausible at scale but has
**not been observed**.

**And Sirius returns one pair too many.** A float64 reference confirms 1,789, and exactly one pair
sits within 1e-6 of the `price*0.8` boundary — so Sirius's arithmetic flips a boundary comparison.
Low severity (1 in 1,789, boundary-only) but it is a silent CPU/GPU disagreement on a scalar
predicate. Filed as **B7**.

## What this harness establishes, independent of scale

1. **A native radius join is not a convenience, it is a correctness feature.** The universal
   workaround — top-k with k inflated, then filter — returned **3,964 of 5,329 pairs** at k=10 with
   no error and no warning. Nothing tells the user the answer was truncated; 156 rows silently hit
   the bound. Every engine that lacks a radius API has this failure mode.
2. **An HNSW index buys nothing for a range predicate — and cannot.** `EXPLAIN` confirms the plan
   never touches the index. This is not a DuckDB limitation: a graph index answers "the k nearest,
   best effort" by greedy descent whose stopping rule is a candidate-list size (`ef`), not a
   distance bound. To answer "everything within τ" it would have to *certify* it had visited every
   node inside the radius, which the graph cannot do without risking a full traversal. That is the
   same reason cuVS ships no radius API and why pgvector's HNSW serves `ORDER BY … LIMIT k` only.
   ⚠️ Our `EXPLAIN` for the `ORDER BY … LIMIT` shape also showed `SEQ_SCAN`, but that test used a
   scalar subquery which likely blocked index use — so it does NOT establish "index works for
   top-k here", only that it is unused for range.
3. **The hybrid shape is decided by predicate ordering, not vector throughput** (Q3).
4. **Sirius is exact on a metric and dimensionality the ported kernel had never seen** — cosine,
   384-dim — matching all three independent references.

---

# Follow-ups from ducndh's three questions (2026-08-24)

## Q1 — is the ported radius kernel actually in the out-of-core path?

**Yes, structurally and verified on the streaming path.** The port went into the exact path's
corpus-chunk loop — the same loop that stages/prefetches chunks — so it inherits the out-of-core
machinery for free. Better than that, it is the *easy* case: top-k folds across chunks because
chunk *j+1* can displace chunk *j*'s winners; "within eps" is independent per chunk, so the radius
path just **concatenates** each chunk's edges and nothing later can invalidate them.

Verified (`ooc_threshold_check.sql`), SIFT1M, corpus pinned **host-tier** (streams, 9 chunks) vs
GPU-tier:

| | GPU-tier | host-tier (streaming) |
|---|---|---|
| eps=150 | 260,864 | **260,864** ✅ 0.365 s |
| eps=200 | — | **1,789,956** ✅ 0.365 s |

Identical answers across the chunked path.

### Now tested PAST the device limit — it works (`ooc48_radius.sql` → `run_ooc48_radius.log`)

**100,663,296 rows × 128 dims = 48 GiB corpus, 40 GiB device**, corpus host-tier (streams),
probe 10k GPU-tier:

| query | result | time |
|---|---|---|
| exact k=10 (the recorded baseline) | — | **32.414 s** (recorded: 32.39 s) |
| **radius, eps=13.0** | 277,937 pairs | **33.834 s** |
| **radius, eps=13.5** | 1,435,321 pairs | **34.150 s** |

**Correctness: every one of the 87,288 ground-truth pairs within eps=13.5 is present. Zero missing.**
(The packaged ground truth only covers top-10 per probe, so it certifies no *misses*; the other
1.35M pairs are genuine neighbours beyond rank 10 that a top-k answer cannot express at all.)

Two things this settles:
- **Radius out-of-core costs the same as exact top-k out-of-core** — 34.2 s vs 32.4 s, ~5% — and is
  **flat in eps** (33.8 s for 278k pairs vs 34.2 s for 1.44M). At this scale the join is bound by
  streaming 48 GiB from host, not by the fold, exactly as the concatenate-don't-fold design predicts.
- The **unbounded-output backpressure risk** flagged in `~/vecjoin/DISCUSSION_approx_and_threshold.md`
  did not bite at 1.44M pairs. It is still untested at the scale where it would — a genuinely loose
  eps on a 100M-row corpus can produce billions of pairs — so the risk is *unrefuted*, not disproven.

## Q2 — why does an HNSW index buy nothing for a range predicate?

See finding 2 above: it is inherent to graph indexes, not a DuckDB bug. Greedy descent stops on a
candidate-list heuristic, not a distance bound, so it cannot certify it has found *everything*
within τ. Confirmed by `EXPLAIN`: no index node in the plan.

## Q3 — didn't we already implement predicate pushdown?

**Partly — and the hybrid case is a different thing than what we built.**

What exists: `sirius_knn_join_rel` takes the probe side as a *relation*, so a filter on the probe
table is planned as a normal scan+filter **below** the join (the recorded 6.8× on a filtered probe),
and `build_source => 'scan'` lets the corpus side be an unpinned scan.

What does **not** exist: the TVF declares `projection_pushdown = true` and **never**
`filter_pushdown` — verified in `sirius_extension.cpp:2605,2622` — so DuckDB is never offered a
filter to push into the join.

But the hybrid predicate is not a pushdown problem at all. `b.price BETWEEN a.price*0.8 AND
a.price*1.25` **relates both sides**, so it cannot be pushed to either side independently. DuckDB's
win came from choosing a *different join algorithm* (a band join on price, vector similarity as a
filter above it). **So the answer to "is it the SQL interface or the optimizer?" is neither: it is
the operator model.** Making similarity a join operator removes the plan in which something cheaper
drives the join. Fixing it means either teaching the planner to accept a scalar join condition
*alongside* the vector one, or exposing the vector comparison as a predicate the optimizer can
order — both real design work, not a flag.
