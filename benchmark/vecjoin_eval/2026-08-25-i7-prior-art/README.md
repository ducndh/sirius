# 2026-08-25 — I7: prior-art survey. Read before drafting any novelty sentence.

**Done because the handoff said to do it EARLY and cheap: it can invalidate framing after the
numbers are in, which is the expensive time to find out. It did.**

## TL;DR — three claims are dead, four survive

| claim | verdict |
|---|---|
| "first vector join in SQL" | ❌ **DEAD** — DuckDB `vss` ships `vss_join`/`vss_match` |
| "first GPU vector join operator in a SQL engine" | ❌ **DEAD** — MaxVec (ETH, PVLDB 2026) is a two-input-port CPU-GPU vector operator supporting inner/left/lateral/semi/anti join patterns |
| "out-of-core vector join is unaddressed" | ❌ **DEAD** — DiskJoin (SIGMOD 2026) is billion-scale similarity join on one machine + NVMe |
| radius / threshold join | ✅ survives *within GPU+SQL* — see caveat below |
| global top-k over the whole join | ✅ survives — no surveyed system exposes it |
| exact (recall 1.0) as the operating point | ✅ survives as a *position*, and it is now contested — see MaxVec |
| centroid-to-centroid pruning (probe side has cluster structure) | ✅ **strongest surviving technical claim** — a search index cannot precompute it |

## The prior art, with what each one takes from us

### 1. DuckDB `vss` — `vss_join` / `vss_match`  (shipping, in-tree)
Signatures, verbatim from the docs:
```
vss_join(left_table, right_table, left_col, right_col, k, metric := 'l2sq')
vss_match(right_table, left_col, right_col, k, metric := 'l2sq')
```
- Two **different** tables. So "a separate probe side" is not novel *at the SQL surface*.
- **top-k only** — no radius/threshold.
- **Brute force only**: "do not currently make use of the `HNSW` index ... provided as convenience
  utility functions." They are table *macros*, not physical operators.
- CPU, FLOAT vectors only.

**Takes:** the phrase "vector join in SQL". **Leaves:** it is a macro over a cross product, so it
is the thing we should be *beating*, and a fair CPU baseline we have not yet run head-to-head.

### 2. MaxVec / Vec-H — "To GPU or Not to GPU: Vector Search in Relational Engines"
Mageirakos, André, Kabić, Wu, Chronis, Alonso (ETH Zürich), **PVLDB 14(1), May 2026**,
arXiv:2605.15957. Extends the Maximus CPU-GPU engine.
- The vector operator is **binary, two input ports** (query side + data side) and drives
  **inner / left / lateral / semi / anti** join patterns. Same operator shape as ours.
- FAISS + cuVS; exhaustive (exact) *and* indexed (CAGRA/IVF) paths, recall target ≥95%.
- **Non-data-owning indexes**: index structure decoupled from embeddings, GPU fetches only the
  needed embeddings from host on demand — i.e. they already attack larger-than-GPU-memory.
- Claims "the only open-source engine that supports full SQL+VS on a hybrid CPU-GPU platform."
- **top-k only (k=100 in all eight Vec-H queries); radius/threshold not addressed.**

**Their headline is the opposite of ours, and that is the real problem:**
> "with current data-owning vector indexes, executing vector search on a GPU does not pay off,
> even with fast interconnects"

and GPU helps the *relational* operators far more (87% of speedup in the CAGRA cases); CPU-VS +
GPU-relational beat all-CPU on 47 of 48 (query, index) pairs.

**Takes:** novelty of "GPU vector operator inside a SQL engine", and of out-of-core-by-streaming.
**Leaves:** they measure *indexed* VS at ~95% recall; our position is recall **1.0**, where they
have no result. That is now the axis to argue on, and it must be argued against MaxVec, not
against a bare library.

### 3. DiskJoin — SIGMOD 2026 (arXiv 2508.18494)
Chen, Yan, Meliou, Lo. Billion-scale **threshold** similarity join, single machine + NVMe SSD,
approximate (probabilistic pruning), CPU, **standalone — not in a SQL engine**. 50×–1000× vs
alternatives.

**Takes:** "out-of-core vector join is unaddressed", and it is *threshold*-shaped, which is the
mode we were calling uniquely ours. **Leaves:** not exact, not GPU, not composable with SQL.

### 4. SimJoin — "Fast Approximate Similarity Join in Vector Databases"
Xie, Yu, Liu. **PACMMOD 3(3), June 2025.** Threshold join over two datasets, proximity-graph
based. Its framing is the one we independently arrived at:
> "existing approaches ... are selection-based such that they treat each data point as an
> individual query point, but such methods do not fully capitalize on the inherent properties of
> the join operation itself."

**This is our thesis, already published.** It also covers k-similarity join.

**Takes:** the "a join is not N searches" argument as a *novel framing*. **Leaves:** they exploit
join windows between partial results on a proximity graph (CPU); we exploit the probe side's
*cluster structure* on GPU. Different mechanism, same motivating observation — so it must be
**cited as motivation, not claimed as insight.**

### 5. Also in the neighbourhood
- **VecFlow** (PACMMOD, Sep 2025) — GPU filtered-search system.
- **"Efficient Data Access Paths for Mixed Vector-Relational Search"** (DaMoN 2024).
- Classic k-NN join: Böhm & Krebs (KAIS 2003); Yao/Li/Kumar ICDE 2010 (kNN-join in RDBMS
  "almost for free"); Gowanlock hybrid CPU+GPU kNN self-join (JPDC 2021, low-dimensional/exact).

## What a defensible sentence looks like now

Not "the first vector join in SQL". Something closer to:

> a GPU-native **exact** vector join operator that is composable with SQL, exposes **radius** and
> **global top-k** reductions no surveyed engine offers, and prunes using the **probe side's own
> cluster structure** — a join-specific optimization unavailable to a per-query search index.

Every clause there is load-bearing and every one of them is measured except the prior-art-relative
part of the radius claim.

## Caveat on the radius claim, stated precisely
Neither cuVS (no radius API), nor DuckDB `vss_join` (top-k only), nor MaxVec (top-k only) offers a
radius join. But **DiskJoin and SimJoin are both threshold joins.** So the honest form is
"no *GPU* engine and no *SQL* engine exposes a radius join", not "radius joins are new".

## Open, not done here
- No head-to-head against `vss_join` (CPU, brute force) — cheap and we should have it.
- No MaxVec/Vec-H comparison; Vec-H is a benchmark we could adopt rather than invent one.
- SimJoin and DiskJoin have no public head-to-head with us; both are threshold-mode, which is
  exactly the mode we have the fewest numbers for.
