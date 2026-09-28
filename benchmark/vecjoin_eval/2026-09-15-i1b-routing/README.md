# 2026-09-15 — I1b (per-row routing) landed; recall/time Pareto + cuVS at matched recall

**RTX A5000 24 GB (sm_86), box `dnguyen56-sirius-0`. NOT the A100 of every earlier experiment —
absolute times here are ~1.5× the A100's; compare ratios, not seconds.** SIFT1M, packaged 10k
queries × 1M base, k=10, recall@10 against the packaged ground truth (tie ceiling 0.9993). Sirius
times are end-to-end SQL (`.mode trash`, warm, min of the last two of three; the CLI timer
quantizes at ~10 ms, so 0.100 vs 0.110 is not a real difference); cuVS times are kernel-only with
a device sync, min of three. Branch `vecjoin-approx-cluster` @ **`23efcf13`** on `ducndh/sirius`.

## 1. What I1b is

The clustered path assigned each probe row to ONE home cluster and then visited that centroid's
`n_probes` nearest centroids (`_cluster_neighbors`, computed once from the centroids). Every row in
a cluster therefore searched the same corpus clusters regardless of where in its cluster it sat —
and a row at the edge has its neighbours next door. Now each row picks its own `n_probes` nearest
centroids (`assign_to_centroids` with the real `n_probes`); the (row, cluster) edges are sorted by
cluster so a corpus slice still gets one batched search (I1 is preserved), and each slice's answer
is folded into a fixed `[n_left × k]` accumulator by gather → 2-part merge → scatter, so device
memory does not grow with `n_probes`. The centroid-to-centroid table is deleted.

Gates: `[kmeans]` 17/17, `[vss],[vector_join]` 98/98 (4,127 assertions); new test
`sirius_knn_join approx routes each probe row by its own nearest centroids` checks the answer
against a CPU oracle of "the row's own two nearest clusters" (holds for whatever clustering
k-means converges to) and additionally that 2 probes reach the exact answer where the home
cluster alone does not. `compute-sanitizer --tool memcheck` over `[approx]`: **no invalid
accesses**; the single reported "error" is a benign duplicate `cub::EmptyKernel` registration
(two CUB copies linked in), not a memory fault. `sanitizer_approx.log`.

⚠ Fixture lesson: a first version of the test used groups with ~0.1 jitter, and balanced k-means
split A∪C by **parity of y** instead of by position (two centroids at x=5.1). Groups now carry
spread on every axis. `../../../active_vecjoin_why_approx_is_slower.md` has the routing history.

## 2. Recall at fixed n_probes — the defect, closed

| 64 clusters, n_probes | before I1b (A100, 08-25) | **after I1b (A5000)** | cuVS IVF-Flat nlist=64 (A5000) |
|---|---|---|---|
| 1 | 0.6118 | **0.6118** | 0.6217 |
| 2 | 0.6799 | **0.8014** | — |
| 4 | 0.7705 | **0.9329** | 0.9381 |
| 8 | 0.8810 | **0.9878** | — |
| 16 | 0.9755 | **0.9988** | 0.9991 |
| 32 | 0.9992 | **0.9993** | — |

Recall at fixed `n_probes` is a property of the routing, not the GPU, so the before column (A100)
is comparable. We now match cuVS's IVF-Flat within 0.005 at every probe count; the "~3× the probes
for the same recall" gap is gone. `11_prefix_same_box` (queued for a sibling box) re-runs the
pre-I1b commit on an A5000 for a same-box before/after in seconds too.

## 3. Sirius Pareto (`recall.csv`, `run_sirius.log`)

| config | time | recall@10 | | config | time | recall@10 |
|---|---|---|---|---|---|---|
| **exact** | **0.463 s** | **0.99937** | | c256_p8 | 0.211 s | 0.93267 |
| c64_p1 | 0.100 s | 0.61180 | | c256_p16 | 0.242 s | 0.98294 |
| c64_p2 | 0.100 s | 0.80137 | | c256_p32 | 0.282 s | 0.99756 |
| c64_p4 | 0.110 s | 0.93285 | | c256_p64 | 0.362 s | 0.99922 |
| c64_p8 | 0.141 s | 0.98776 | | c1024_p16 | 0.614 s | 0.92718 |
| c64_p16 | 0.201 s | 0.99876 | | c1024_p64 | 0.654 s | 0.99537 |
| c64_p32 | 0.322 s | 0.99933 | | c1024_p256 | 0.876 s | 0.99934 |

64 clusters still dominates at every recall (S1's conclusion survives a second time). The
approximate path is now useful up to ~0.999: c64_p16 is 0.201 s at 0.9988 against exact 0.463 s.
Above that, exact.

## 4. vs cuVS at matched recall (`matched_recall.txt`, `cuvs.json`, `run_cuvs.log`)

`ratio = Sirius / cuVS`. **amort** counts cuVS search only (index pre-built); **1-shot** counts
cuVS build + search. Sirius is the cheapest config reaching ≥ the cuVS recall.

| cuVS opponent | recall | search | total | Sirius | via | amort | 1-shot |
|---|---|---|---|---|---|---|---|
| IVF-Flat 1024/16 | 0.9321 | 0.116 | 0.547 | 0.110 | c64_p4 | **0.95×** | 0.20× |
| IVF-Flat 64/4 | 0.9381 | 0.438 | 0.738 | 0.141 | c64_p8 | **0.32×** | 0.19× |
| CAGRA nn_descent 32/64 | 0.9767 | 0.016 | 11.649 | 0.141 | c64_p8 | 8.7× | 0.01× |
| IVF-Flat 1024/64 | 0.9952 | 0.455 | 0.885 | 0.201 | c64_p16 | **0.44×** | 0.23× |
| CAGRA nn_descent 64/128 | 0.9987 | 0.055 | 12.831 | 0.201 | c64_p16 | 3.7× | 0.02× |
| IVF-Flat 64/16 | 0.9991 | 1.773 | 2.073 | 0.322 | c64_p32 | **0.18×** | 0.16× |
| CAGRA nn_descent 64/256 | 0.9993 | 0.103 | 12.879 | 0.322 | c64_p32 | 3.1× | 0.03× |
| brute_force k=10 | 0.9993 | 0.643 | 0.879 | 0.463 | exact | 0.72× | 0.53× |
| brute_force k=65+trim (tuned) | 0.9994 | 0.428 | — | 0.463 | exact | 1.08× | — |

**What changed versus 08-25:** before I1b, cuVS IVF-Flat beat us *amortized* at every recall
(0.76×–2.8× in its favour, and we needed 3× the probes). **Now, at every recall ≥ 0.93, the
Sirius end-to-end SQL join is faster than a cuVS IVF-Flat search on a pre-built index** (0.18×–0.95×),
before counting cuVS's build. The only opponent that still wins amortized is CAGRA (3–9× at
0.98–0.999), and CAGRA's build is 11.6–12.9 s here, i.e. 40–90× our whole join. Exact-vs-exact is
still parity against a *tuned* brute force (k=65 dispatch, `run_cuvs_k65.log`): 0.463 s vs
0.428 s — **do not quote 0.72× against the k=10 call**, that opponent is mis-dispatched.

So the accounting fight narrows to one line: **IVF is now ours at any accounting; graph indexes
win amortized and lose one-shot by 30–90×.** J5 (churn) is the experiment that decides how often
"one-shot" is the real accounting.

## 5. What the fix retires

- The "centroid-level pruning that exploits the probe side's own cluster structure" claim, listed
  as the strongest surviving technical claim in `../2026-08-25-i7-prior-art/`. It was the defect.
  Drop it from the draft.
- The X1 cost-model term "we scan 9.1× more corpus for the same recall" (cause 2 in
  `active_vecjoin_why_approx_is_slower.md`). At matched recall the scanned fraction now equals
  IVF's by construction.
- The SQL-level `sirius_kmeans_assign` over the *probe* table in every earlier harness script. The
  operator never read it (the request has no probe-side cluster column) and re-assigns each probe
  batch itself; `gen.sh` here uses the query table directly.

## 6. J5 churn — DONE, see [churn/README.md](churn/README.md)
1% of the corpus replaced per epoch, 10k queries per epoch: Sirius exact **0.486 s** with no
preparation (corpus scanned), cuVS brute force 0.430 s (parity), cuVS IVF-Flat rebuild+search
1.19 s, **CAGRA rebuild+search 9.46 s** — CAGRA needs ≥ 20 batches (~200k queries) per corpus
version before amortization wins. Our own approx path (re-cluster 1.15 s + 0.14 s) is the wrong
tool under churn.

## 6b. GIST1M (d=960) — DONE, see [gist/README.md](gist/README.md)
Exact 0.201 s vs cuVS tuned brute force 0.175 s (parity); IVF-Flat loses amortized at every
recall ≥ 0.83; CAGRA builds in 21 s and wins amortized by only 1.6× at recall 0.99.

## 6c. 1M × 1M self-join — DONE, see [m2m/README.md](m2m/README.md)
Exact parity (45.2 vs 43.7 s). vs IVF-Flat at its best list count ~7× at matched recall under
either accounting (3.2 vs 22.4 s at 0.94). vs CAGRA: parity build-inclusive at 0.9976, CAGRA wins 1.9× at
0.9989 even build-inclusive (one 1M batch amortises its build), 5–9× search-only. 256 clusters overtakes 64 at this probe count.

## 6d. Churn rate sweep — DONE (`churn/0p1`, `churn/10p`, sibling boxes)
Cost per corpus version is flat in the change rate for every system (none maintains an index
incrementally): Sirius exact 0.481/0.486/0.486 s, cuVS IVF-Flat 1.18/1.19/1.19 s, CAGRA
8.5/9.5/8.4 s at 0.1/1/10 %. Only queries-per-version matters.

## 6e. Sibling-box runs — DONE (`prefix/`, `ooc/`, `full_suite.*.log`, `run_vss_join_1k.log`)
- **Full suite GREEN on `23efcf13`**: 32,147,836 assertions / 2,025 cases (sirius-2).
- **Same-box before/after of I1b** (sirius-1, old commit `3c6db00b` built in a worktree):
  recall identical to the A100 columns; time at p1–p4 is +20 ms (0.080→0.100 s, the per-row
  gather/fold has a slightly higher floor), equal at p8, faster from p16 (0.211→0.201) and p32
  (0.362→0.322); at matched recall 0.999 the old routing needs p32 (0.362 s) where the new one
  needs p16 (0.201 s). c256 improves 0.795→0.362 s at 0.999.
- **Out-of-core on the 24 GB A5000** (sirius-4): Sirius exact 8 GiB **8.0 s**, 24 GiB **23.9 s**
  (exceeds the device), recall 1.0 both; cuVS IVF-Flat direct **OOM**; sharded (3×8 GiB) 56.0 s
  GPU-side / 103 s wall, recall 1.0; IVF-PQ under UVM 66.2 s at recall **0.427**; UVM IVF-Flat
  92.9 s. Same ordering as the A100 48 GiB run.
- **DuckDB `vss_join` on 1k queries** (sirius-3): 119.3 s / 117.0 s, i.e. linear in queries
  (10k = 1,197 s); its recall pass is re-running after a query-alias fix.

## 6f. J1 with a VIEW corpus — DONE, see [j1view/README.md](j1view/README.md)
S4 reverses the August loss: corpus streamed from a view, 0.101 / 0.201 / 0.282 / 0.454 s at
10 / 30 / 50 / 90 % selectivity vs cuVS practitioner path 0.299 / 0.879 / 1.637 / 2.872 s
(3–6× ours) and vs our old CTAS+pin recipe 0.715 / 1.661 / 2.668 / 4.560 s.

## 6g. Composability pipeline — DONE, see [compose/README.md](compose/README.md)
filter → join → equi-join → GROUP BY in one statement **0.141 s**; cuVS path 0.125 s with the
corpus resident, **2.634 s** one-shot. CTAS 0.141 s, COPY 0.141 s, round/sqrt over the join
0.202 s — nothing falls to the CPU. Answers identical (97 categories, 25,000 matches).

## 6h. Threshold + global top-k curves — DONE, see [threshold/README.md](threshold/README.md)
Threshold join 0.47 s at eps 150–250 (0.26–12 M pairs), 1.8 s at 80 M pairs, 11.3 s at 379 M
(output-bound) vs FAISS-CPU range_search 52–70 s. Global top-k = its emulation (0.45 s at k=100,
2.2 s at k=1000); k=100k refused by the per-task budget.

## 6i. Full suite GREEN on b21525fd and 54eea872 (sirius-1, sirius-3): 32,148,0xx assertions / 2,029 cases.

## 7. Still open / not covered here
- S7 planner demotion under a selective scalar predicate; approximate threshold via the radius
  kernel (approx mode still emulates); k-means assign row-id recipe; a large two-relation dataset
  with meaning; GPU utilisation sampling (no profiler on the pods).
- DuckDB `vss_join` head-to-head is running (`run_vss_join.sh`); at ~8 cores it is far slower
  than the LATERAL anchor and may hit its 1-hour cap — `vss_join_1k.sql` is the subset fallback
  (rate only; label it as an extrapolation).
- Sirius per-batch host loop over `n_left × n_probes` edges (candidate check) is O(edges) CPU per
  probe batch; invisible at 10k probes, worth a look at 1M × 64.

## Reproduce
```bash
./gen.sh && SIRIUS_VECTOR_JOIN_PRUNE_DEBUG=1 \
  ~/vecjoin/sirius/build/release/duckdb -unsigned /var/tmp/vj/i1b/bench.db < /var/tmp/vj/i1b/pareto.sql
python3 cuvs_packaged.py && python3 cuvs_k65.py
DS=gist-960-euclidean DIM=960 ./gen_ds.sh     # any other packaged dataset
```
