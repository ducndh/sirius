# 2026-08-25 — post-I1 recall/time Pareto, and cuVS at matched recall

**One A100-SXM4-40GB (sm_80). SIFT1M, the PACKAGED 10k query set × 1M base, k=10, recall@10 against
the PACKAGED ground truth** — so a recall here means what a published ANN-benchmark recall means,
and both systems are scored by the same oracle. Sirius times are end-to-end SQL (`Run Time` with
`.mode trash`, warm, min of the last two of three); cuVS times are kernel-only with an explicit
device sync, min of three.

**Tie ceiling:** SIFT1M contains duplicate vectors, so a provably exact answer scores
**recall_id 0.9993**, not 1.0. Our exact join scores 0.99938; cuVS brute force scores 0.9993.

## 1. What I1 bought

I1 (one cuVS call per corpus slice instead of one per probe-run × slice) on this workload:

| clusters / probes | before | after | speedup |
|---|---|---|---|
| 64 / 1 | 0.081 s | 0.081 s | 1.00× |
| 64 / 2 | 0.121 s | 0.090 s | 1.34× |
| 64 / 4 | 0.182 s | 0.112 s | 1.63× |
| 64 / 8 | 0.315 s | 0.142 s | 2.22× |
| 64 / 16 | 0.598 s | 0.202 s | 2.96× |
| 256 / 32 | 3.183 s | 0.486 s | **6.55×** |

*(those are from `../2026-08-25-i1-batching/`, a slightly different session; the Pareto below is a
single clean session and its absolute numbers are a little lower.)*

**The result that matters is not the speedup, it is what it fixed.** Our exact join costs
**0.314 s** at recall 0.99938. Before I1, at 64 clusters:

- `n_probes=8` cost **0.325 s** for recall 0.881 — *more* than the exact join, for less recall.
- so the approximate path was **strictly dominated by our own exact path from p8 upward**, and
  only p1–p4 (recall ≤ 0.77) were worth running at all.

After I1, p8 is 0.122 s and p16 is 0.183 s. **The crossover with our exact path moved from ~p8 to
~p32**, which is the first time the approximate path has been useful in the recall range anyone
would ship (0.88–0.98).

## 2. Sirius Pareto — 64 clusters is still the optimum, now with recall attached

| config | time | recall@10 | | config | time | recall@10 |
|---|---|---|---|---|---|---|
| **exact** | **0.314 s** | **0.99938** | | c1024_p1 | 0.304 s | 0.36955 |
| c64_p1 | 0.071 s | 0.61177 | | c1024_p4 | 0.385 s | 0.48685 |
| c64_p2 | 0.081 s | 0.67993 | | c1024_p16 | 0.645 s | 0.69809 |
| c64_p4 | 0.092 s | 0.77045 | | c1024_p32 | 0.983 s | 0.81910 |
| c64_p8 | 0.122 s | 0.88100 | | c1024_p64 | 1.702 s | 0.91996 |
| c64_p16 | 0.183 s | 0.97554 | | c1024_p128 | 3.139 s | 0.97567 |
| c64_p32 | 0.315 s | 0.99920 | | c1024_p256 | 5.978 s | 0.99601 |
| c64_p64 | 0.558 s | 0.99938 | | c256_p32 | 0.375 s | 0.94915 |

**S1's conclusion survives I1: 64 clusters dominates at every recall level.** c64_p16 and
c1024_p128 reach the same recall (0.9755 / 0.9757); c64 does it **17× faster**. I1 narrowed the
gap (it was ~65× before) but did not close it, so cluster count is still a first-class knob and
**no Sirius approximate number means anything without it**.

Above ~0.99 recall the approximate path buys nothing: c64_p32 is 0.315 s at 0.9992, the exact join
is 0.314 s at 0.99938.

## 3. vs cuVS at matched recall — the accounting decides the winner

Full table in `matched_recall.txt`. Ratio = Sirius / cuVS, so **>1 means cuVS is faster**.

| recall | cuVS opponent | cuVS search | cuVS total | Sirius | amortized | one-shot |
|---|---|---|---|---|---|---|
| 0.935 | IVF-Flat nlist=64 nprobe=4 | 0.202 s | 0.409 s | 0.154 s | **0.76×** | **0.38×** |
| 0.932 | IVF-Flat nlist=1024 nprobe=16 | 0.054 s | 0.320 s | 0.152 s | 2.81× | **0.47×** |
| 0.976 | CAGRA nn_descent deg=32 itopk=64 | 0.008 s | 7.356 s | 0.186 s | 21.9× | **0.03×** |
| 0.9987 | CAGRA nn_descent deg=64 itopk=128 | 0.026 s | 8.484 s | 0.311 s | 12.0× | **0.04×** |
| 0.9990 | IVF-Flat nlist=64 nprobe=16 | 0.815 s | 0.999 s | 0.314 s | **0.38×** | **0.31×** |
| 0.9993 | **brute_force, naive k=10** | 0.347 s | 0.371 s | 0.314 s | **0.91×** | **0.85×** |
| 0.9993 | **brute_force, tuned k=65+trim** | 0.279 s | 0.303 s | 0.314 s | 1.13× | 1.04× |

**Read this way:**

- **Exact vs exact is a tie, and the tie is the honest headline.** Against cuVS's *naive*
  `brute_force(k=10)` we are 1.18× faster end-to-end. But cuVS at k=10 dispatches into `fusedL2Knn`,
  and asking it for **k=65 and trimming — the same trick our own operator applies internally —
  makes it 1.25× faster** (0.347 s → 0.279 s, `cuvs_k65.py`). Against that tuned opponent we are at
  **parity (1.04×)** while additionally planning, pruning and materializing 100k rows through SQL.
  Do not quote the 1.18×without saying the opponent was untuned.
- **One-shot, we win everywhere** — 2.6× over the best IVF-Flat, **25–50× over CAGRA**, whose
  7.3–8.5 s build dwarfs everything.
- **Amortized, CAGRA wins by 7–27×** and nothing we did changes that. Unchanged from the settled
  position: *the win is exactness and build-freedom, not throughput.*
- One genuinely new win: at **recall ~0.935 we now beat cuVS IVF-Flat even amortized** (0.76×) at
  the same cluster count. Before I1 we lost that comparison.

## 4. The biggest remaining defect, now measured: probe→cluster selection

Same algorithm, same cluster count, same n_probes, same corpus, same ground truth:

| n_probes at 64 clusters/lists | cuVS IVF-Flat | **Sirius** |
|---|---|---|
| 1 | 0.6200 | 0.61177 |
| 4 | **0.9352** | **0.77045** |
| 16 | 0.9990 | 0.97554 |

At `n_probes=1` we match cuVS exactly, as we must. **At n_probes=4 cuVS reaches 0.9352 and we
reach 0.7705**; we need roughly **n_probes≈12 to match**, i.e. **~3× the probes for the same
recall.**

That is QUEUE **I1b**, and this is the first clean measurement of it: cuVS assigns each *query
vector* to its n_probes nearest centroids, while we take the probe's own cluster and then that
**centroid's** nearest centroids (`sirius_physical_vector_join_stream.cpp:689/799/855`). Ours is a
coarser choice that ignores where in its cluster the probe actually sits.

**Fixing it is worth about 2.5–3× on the approximate path — more than I1 just delivered**, and it
would move the whole c64 curve left rather than a single point. It is the top optimization item now.

## 5. Trap recorded: cuVS CAGRA's default build is degenerate here

`cagra.IndexParams(...)` defaults to `build_algo='ivf_pq'`. On SIFT1M on this box that produces a
**broken graph**: `"Self-included ratio is low: 0.00%"` and **recall 0.0001–0.0024 at every search
parameter**, with a 25 s build. `build_algo='nn_descent'` gives a healthy graph (0.9227–0.9992).

Every CAGRA number here uses nn_descent. Publishing the default-build numbers would have shown us
beating CAGRA by ~4000×, which is the third time in three days a degenerate opponent nearly became
a headline (`../../../feedback_benchmark_discipline.md` rule 7).

## 6. What this does NOT cover
- One dataset (SIFT1M, d=128), one k (10), one shape (10k × 1M). No GIST1M, no deep-image, no
  many-to-many 1M × 1M, no out-of-core.
- `threshold` and `global top-k` reduction modes are not in this Pareto.
- cuVS times are kernel-only; Sirius times are end-to-end SQL. That handicaps us, deliberately —
  it is the comparison a user actually faces — but it is not a like-for-like kernel comparison.
- No CPU baseline here (DuckDB `vss_join` is the one that matters and is still unrun — see
  `../2026-08-25-i7-prior-art/`).

## Reproduce
```bash
./gen.sh && SIRIUS_VECTOR_JOIN_PRUNE_DEBUG=1 \
  ~/vecjoin/sirius/build/release/duckdb -unsigned /var/tmp/vj/pareto/bench.db < /var/tmp/vj/pareto/pareto.sql
python3 cuvs_packaged.py     # cuVS opponents, same workload, same ground truth
python3 cuvs_k65.py          # the k-dispatch fairness check
```
`gen.sh` builds the corpora **in the measuring session on purpose**: the cluster-ordered corpus is
written with the fitting session's centroids and `cuvs::cluster::kmeans::fit` is not bit-stable
across processes (~1 session in 6), so a separate setup session can silently leave the corpus
ordered by labels the join's centroids no longer produce.
