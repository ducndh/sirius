# 2026-08-23 — X1: why is our approximate path slower than cuVS? (QUEUE item P0)

**Question (ducndh).** We built an algorithm for a vector *join*; cuVS did not. So why is our
approximate path slower than theirs? Four candidates that imply completely different work:
workload shape, kernel speed, algorithm choice, implementation detail.

**Answer, in one sentence.** At matched recall in the many-to-many regime we are at parity with
cuVS IVF-Flat, and that parity is the product of two large effects cancelling: **our fold is
~7.6× faster per distance computed, and we compute ~9.1× more distances for the same recall.**

Nothing here is interpolated across boxes: one A100-SXM4-40GB (sm_80), one corpus, the same 
probe *rows* on both sides, one oracle.

## What was run

SIFT1M base (1M × 128) as corpus. Probe batches of **1k / 10k / 100k / 1M rows drawn from that
same corpus**, nested (`ids_1k ⊂ ids_10k ⊂ ids_100k ⊂ ids_1M`) so one 1000-row ground truth
serves every batch size. k=10, L2.

- `gen_x1_probes.py` — the probe sets and a **cupy brute-force** ground truth. The oracle is
  deliberately neither system's own exhaustive run; matched recall *is* the comparison here, so a
  self-referential reference would beg the question.
- `x1_cuvs.py` → `run_x1_cuvs.log` — cuVS IVF-Flat, `n_lists=1024` (its optimum) at all four batch
  sizes, plus `n_lists=64` (our optimum) at 1M probes.
- `gen_x1_sirius_sql.sh` → `run_x1_sirius_c64.log`, `run_x1_sirius_c1024.log` — Sirius at 64
  clusters (our optimum) and at 1024 (cuVS's), all four batch sizes.
- `x1_score.py` / `x1_score_sirius.py` — one scorer for both systems.
- `x1_analyze.py` → `run_x1_analysis.log`; `x1_costmodel.py` → `run_x1_costmodel.log`.

### Two recall numbers, and why

`recall_id` is the usual id-set intersection. **SIFT1M contains exact duplicate vectors** — 1.5%
of sampled rows have one — so equal-distance ties cap it below 1.0 *for a provably exact answer*.
`recall_dist` counts a returned neighbour if its true distance is within tolerance of the k-th
true distance, and is tie-robust.

**Sirius exact scores `recall_dist` = 1.0000 at every batch size while `recall_id` reads 0.9989.**
The long-recorded "exact returns 0.9993/0.999" was never a miss — it is entirely the tie ceiling.
All matching below is on `recall_dist`.

## Result 1 — the gap decays with batch size. The parity hypothesis holds.

Sirius (64 clusters) vs cuVS IVF-Flat (1024 lists), at matched `recall_dist`, cuVS time
interpolated log-linearly between its bracketing points. **>1 = cuVS faster.**

| Sirius point | recall | 1k probes | 10k | 100k | **1M** |
|---|---|---|---|---|---|
| p1 | 0.6796 | 44.3× | 8.9× | 1.5× | **0.81×** |
| p4 | 0.8148 | 60.0× | 8.1× | 2.2× | **1.28×** |
| p8 | 0.9159 | 54.2× | 7.4× | 2.0× | **1.19×** |
| p16 | 0.9807 | 33.7× | 4.4× | 1.2× | **0.75×** |
| **exact** | **1.0000** | 0.86× | **0.40×** | **0.37×** | **0.34×** |

The previously *interpolated* "1.02× parity at 1M probes" is **confirmed by direct measurement**:
the iso-parameter point (`n_lists=1024`, `n_probes=64`, 1M probes) lands at exactly 1.02×.

The exact row is the standing claim, re-measured cleanly: at every batch ≥10k, **our exact join
beats cuVS's own recall-1.0 configuration**, by 2.9× at 1M probes (27.73 s vs 80.90 s).
⚠️ cuVS's `n_probes=256` reaches `recall_dist` 1.0000 *on a 1000-probe sample*; its provably
exhaustive setting (`n_probes = n_lists`) costs 269 s. Quote 80.90 s as "recall 1.0 within
measurement resolution", not as "exhaustive".

## Result 2 — the four hypotheses, decided

`x1_costmodel.py` fits the cost model the source implies. The operator issues one
`vss::brute_force_knn` per (corpus-cluster slice × probe run that wants it), each doing a fresh
`bf::build` over the whole slice:

```
time ≈ C + a·CALLS + g·REREAD + B·PAIRS
C = 51.7 ms/join   a = 278 µs/call   g = 10.6 ns/re-read row   B = 0.040 ns/scored pair
36 points, relative-error weighted, median error 4.8%, max 23%
```

| hypothesis | verdict | the number |
|---|---|---|
| **1. Workload shape** | **confirmed, and it is the biggest single term at small batch** | non-distance work is **93% → 84% → 40% → 5%** of our runtime as the batch grows 1k→1M |
| **2. Kernel speed** | **refuted — we are the faster kernel** | `B` = **6.47 TFLOP/s** = 33% of A100 fp32 peak. cuVS's IVF-Flat kernel measures **0.74 TFLOP/s**. We are **8.5× more FLOP-efficient per distance** |
| **3. Algorithm** | **partly, and it splits in two** | (a) within IVF we scan **2.4–4.1× more corpus for the same recall** (Result 3, ours to fix); (b) CAGRA-vs-IVF-Flat is a separate ~10× that no IVF tuning reaches |
| **4. Implementation** | **confirmed, and now sized** | `a·CALLS` alone is **42–96%** of runtime everywhere except c64 at 1M probes. This is exactly QUEUE **I1** |

Each fitted term names its fix: `C` amortises with a bigger probe side; **`a` is I1(b)** — one
search per slice with all wanting runs concatenated; **`g` is I1(a)** — hoist `bf::build` out of
the per-slice loop, which today recomputes the slice's norms on every call; `B` only moves with a
better kernel, and it is already the best kernel in the comparison.

## Result 3 — NEW, and not previously recorded: we probe clusters, not vectors

At **identical `n_lists` and identical `n_probes`**, cuVS reaches materially higher recall:

| n_lists | n_probes | Sirius recall | cuVS recall | gap |
|---|---|---|---|---|
| 1024 | 1 | 0.4613 | 0.4648 | +0.0035 |
| 1024 | 4 | 0.5686 | 0.7580 | **+0.1894** |
| 1024 | 16 | 0.7566 | 0.9455 | **+0.1889** |
| 1024 | 64 | 0.9393 | 0.9981 | +0.0588 |
| 64 | 1 | 0.6796 | 0.6856 | +0.0060 |
| 64 | 4 | 0.8148 | 0.9473 | **+0.1325** |
| 64 | 8 | 0.9159 | 0.9924 | +0.0765 |

**They agree at `n_probes=1` and diverge above it** — so this is not clustering quality, it is
multi-probe *selection*. Confirmed in source, and it is deliberate:

- `sirius_physical_vector_join_stream.cpp:689` — `brute_force_knn(centers, centers, n_probes)`
  builds, once per join, each **centroid's** nearest centroids.
- `:799` — the probe batch is assigned with `nearest.n_probes = 1`: every probe gets exactly
  **one** cluster.
- `:855` — a probe in cluster `c` visits `_cluster_neighbors[c*n_probes + t]`.

So every probe in a cluster visits the **same** clusters — the ones nearest to *its centroid* —
whereas textbook IVF visits the centroids nearest to *the probe vector itself*. The in-source
comment states the intent: *"This is what a search index cannot precompute: its query side has no
cluster structure, while a join's does."* It is what makes a probe run contiguous, which is what
lets the fold read a slice instead of a gathered copy.

**Priced:** to reach the same recall at the same `n_lists` we need **4.1× more clusters** at 1024
lists (64 vs ~15.6) and **2.35× more** at 64 lists (8 vs ~3.4). Combined with our coarser optimum
(64 lists vs 1024, forced on us by `a`), that is the 9.1× more distances scored.

## The one-sentence decomposition, at 1M probes and recall ≈ 0.916

| | Sirius (64 clusters, p8) | cuVS (1024 lists, ~p14) |
|---|---|---|
| distances scored | 1.25e11 | 1.38e10 (**9.1× fewer**) |
| throughput | 2.55e10 pairs/s | 3.34e9 pairs/s (**7.6× slower**) |
| **time** | **4.90 s** | **4.13 s** |

## Corrections this run forces

1. **"The fold runs at 4.4% of fp32 peak — not GPU-saturated, so tensor cores / concurrent streams
   are open."** ✗ 4.4% is an *end-to-end* rate at 10k probes, where **85% of wall time is not
   scored distances at all**. The fold kernel itself runs at **6.47 TFLOP/s (33% of peak)**, and
   our exact path at 9.23 TFLOP/s (47%). *Why the error happened:* useful-FLOPs ÷ wall-time is a
   kernel efficiency only if wall time is mostly kernel. Divide by the phase, not the total.
2. **"Exact path: 28.8% of fp32 peak."** ✗ That is the pre-reboot slower box (implies a 45.5 s
   self-join). On this box the exact self-join is 27.73 s → **47.3%**.
3. **"At parity with cuVS IVF at 1M probes, so the kernel is fine."** ✓ on the number (measured
   1.02×), ✗ on the reading. The kernel is not "fine", it is **8.5× better**; parity is a fast
   kernel cancelling ~9× more work. Which flips the roadmap: do not tune the fold, cut the work.
4. **"Exact returns recall 0.9993/0.999."** ✗ artifact. `recall_dist` is exactly **1.0000**;
   SIFT1M's duplicate vectors cap any id-set recall.
5. **NEW cause, previously unlisted:** cluster-level probing costs 2.4–4.1× the corpus scanned at
   matched recall. The old decomposition had only fixed cost + IVF-vs-graph.

## What this says to do

- **I1 is confirmed as the right next engineering item**, and it is worth more than it looked:
  `a·CALLS` is 42–96% of runtime across the grid. Do I1(a) (hoist `bf::build`, kills `g`) and
  I1(b) (one search per slice, kills most of `a`).
- **Do not chase the kernel.** It is already the fastest thing in the comparison.
- **Consider per-vector probe assignment** as a second lever, priced at 2.4–4.1×. It trades away
  contiguous probe runs, so it is not free and it interacts with I1 — but it is ours, unlike the
  IVF-vs-graph gap.
- **Lead with exactness, unchanged.** 2.9× ahead of cuVS's recall-1.0 configuration at 1M probes,
  and the exact path needs no clustering at all.

## Traps hit

- `CREATE TABLE AS` **and** `COPY … TO` over the TVF are both rejected as "cannot run on the CPU"
  (issue #19); only the CLI's `.mode csv` + `.output` sink works. A `WHERE left_id IN (SELECT …)`
  filter **is** accepted — unlike a scalar function over an output column (S2).
- Clusterings and GPU pins are **session** state, not database state. A second connection to the
  same `.db` reports "no clustering named cl".
- **The first join of a session pays one-time initialisation.** Without a warm-up the 1k-probe
  point read 162 µs/probe against 8.1 µs/probe at 10k — the artefact landed exactly on the
  smallest batch, the one the fixed-cost question is about.
- A single per-call constant does not fit both cluster counts: each call's `bf::build` is O(slice
  rows), and a 64-cluster slice is 16× a 1024-cluster slice. That is what the `g` term is.

---

# I2 — profiler installed, and what it can and cannot say here

**`ncu` is installed but structurally unusable on this box. `nsys` works and is enough.**

## Install (survives nothing — `/var/tmp` is wiped; re-run after a wipe)

```bash
export MAMBA_ROOT_PREFIX=/var/tmp/vjtools/mamba
micromamba create -y -p /var/tmp/vjtools/prof -c nvidia/label/cuda-12.8.0 nsight-compute   # ncu 2025.1
cd /var/tmp/vjtools && curl -sfLO \
  https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-systems-2025.6.1_2025.6.1.190-1_amd64.deb
dpkg-deb -x nsight-systems-*.deb /var/tmp/vjtools/nsys
export PATH=/var/tmp/vjtools/nsys/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64:$PATH
```

The `nvidia` channel's default label serves 2021-era repodata — `nsight-compute` is only findable
under a versioned label (`nvidia/label/cuda-12.8.0`). `nsight-systems` is not on conda at all;
take the CLI `.deb` from the devtools repo. Neither is on PyPI.

## Why ncu cannot run here — diagnosed, not guessed

```
==ERROR== Profiling failed because a driver resource was unavailable.
```

Not the usual permission error, and **not** driver policy: `/proc/driver/nvidia/params` reports
`RmProfilingAdminOnly: 0`, so the driver *does* permit non-admin profiling. The container is the
blocker, on two counts:

- `capsh --print` → `Current: =` — **no capabilities at all**, and `cap_sys_admin` / `cap_perfmon`
  are dropped from the **bounding** set, so they cannot be regained from inside.
- **`/dev/nvidia-caps/` does not exist.** That directory is how the driver hands a profiling
  capability to an unprivileged container process.

Nothing inside the container fixes this; it needs the container launched with `--cap-add=SYS_ADMIN`
and the caps device exposed. **So: no occupancy, no memory-bandwidth counters, no roofline.**
Any claim needing those stays unmade — which was the point of I2.

## What nsys does deliver, and it is the decision-relevant half

`nsys profile -t cuda` traces kernel launches and durations without performance counters. CPU
sampling is also disabled by the missing capabilities; CUDA tracing is not.

`i2_nsys_attribute.py` → `run_i2_nsys_attribute.log`, from `run_i2_join_trace.log`:

| window | config | span | Σ kernel time | GPU busy | launches | **CALLS** | launches/call |
|---|---|---|---|---|---|---|---|
| 7 | approx 10k × 1M, p8 | 0.440 s | 0.136 s | **31%** | 8,473 | **504** | 16.8 |
| 10 | approx 1M × 1M, p8 | 4.784 s | 8.665 s | **181%** | 60,353 | **560** | 107.8 |
| 12 | exact 1M × 1M | 27.30 s | 38.35 s | **141%** | 47,633 | 56 | 850.6 |

`CALLS` is **measured**, not modelled: `cuvs::neighbors::detail::knn_merge_parts_kernel` fires once
per `brute_force_knn` call. (Spans are ~35% longer than un-profiled at 10k probes and unchanged at
1M — tracing overhead is per-launch, which is itself a launch-bound signature.)

### Three things this settles

1. **X1's model is confirmed by an independent instrument, to 1%.** At 10k probes the model's
   *device-work* terms are `B·PAIRS + g·REREAD` = 0.050 + 0.085 = **0.135 s**; nsys measures GPU
   busy at **0.136 s**. The rest of the span is launch gap, which is the model's `C + a·CALLS`.
2. **CALLS is near-constant across a 100× batch — 504 → 560.** This is the *reason* the gap decays
   with batch size, and it survives a check that could easily have broken it: the probe side IS
   chunked (`_next_left++` over `_probe->chunk_rows`; the 1M probe table is 9 row groups), which
   would have multiplied CALLS by 9. It does not, because the probe side is cluster-**ordered**,
   so each probe chunk contains ~7 clusters rather than all 64. The 7× rise in *launches* is cuVS
   launching data-proportional work inside each call, which `B·PAIRS` already covers.
3. **"Is the fold GPU-saturated?" — it depends entirely on batch size, and neither answer is about
   occupancy.** At 10k probes the GPU is **idle 69% of the time**. At 1M probes kernels *overlap
   across streams* (181%), so the device is oversubscribed already. **Concurrent streams are
   therefore NOT the open lever at large batch** — they are already happening — and at small batch
   the problem is launch gaps, not under-filled kernels.

Point 3 retires the last live piece of the old headroom section: it read "the runs are not filling
the device" off a FLOP percentage. The trace says the large-batch fold already overlaps streams,
and the small-batch problem is that the GPU has nothing queued.

---

# I3 — do the ANN libraries support the other join shapes?

**Everything below was executed, not read off an API listing** (`i3_shape_support.py` →
`run_i3_shape_support.log`). An argument named `prefilter` is not support; a correct answer is.

| shape | cuVS 26.02 | FAISS 1.15.0 (**CPU-only in this env**, `get_num_gpus()=0`) |
|---|---|---|
| A. per-probe top-k | **YES** — recall 1.000 | **YES** — recall 1.000 |
| B. radius / threshold | **NO** — no range/radius entry point anywhere in `cuvs.neighbors` | **YES** — `range_search` on Flat/IVF/HNSW; returned 3,176 pairs, brute force agrees exactly |
| C. global top-k over pairs | **NO** — user-side reduction over `n_probes × k` | **NO** — same |
| D. filtered / predicated | **YES** — `filters.from_bitset` on brute_force/ivf_flat/cagra; every hit inside the filter, recall 1.000 vs filtered truth | **YES** — `IDSelectorBatch`, every hit inside the filter |
| E. many-to-many self-join | **YES — `all_neighbors.build`** (see below) | **PARTIAL** — no join API; the user calls `search(base)` |

## The correction: "neither has join semantics — a user loops" is FALSE

**cuVS ships `cuvs.neighbors.all_neighbors`** — "build an approximate all-neighbors k-NN graph …
finds nearest neighbors for all the training vectors" — which *is* the many-to-many self-join, as
one library call. It takes `algo ∈ {brute_force, ivf_pq, nn_descent}`, an `n_clusters` batching
parameter, and **a host dataset** ("for multi-GPU build"). This was missed because every prior
comparison used `ivf_flat`/`cagra` — *search* indexes benchmarked in a search shape. `all_neighbors`
is cuVS's *join* API, and it exploits the same structural advantage our operator does: the query
set is the dataset.

### Measured, 1M × 1M, k=10, recall by distance against cupy brute force

SIFT1M, 20,000 sampled rows = 200,000 pairs, resolving recall to 5e-6
(`i3_recall_deep.py` → `run_i3_recall_deep.log`):

| | time | recall_dist | misses / 200k |
|---|---|---|---|
| cuVS `all_neighbors` nn_descent (device) | **4.95 s** | 0.999830 | 34 |
| cuVS `all_neighbors` nn_descent, `n_clusters=4`, **host array** | 10.59 s | 0.999940 | 12 |
| cuVS `all_neighbors` brute_force | 34.29 s | **1.000000** | 0 |
| **Sirius exact** | **27.73 s** | **1.000000** | 0 |

GIST1M, d=960, 5,000 sampled rows (`i3_gist_all_neighbors.py` → `run_i3_gist_all_neighbors.log`):

| | time | recall_dist |
|---|---|---|
| cuVS `all_neighbors` nn_descent | 9.86 s | **0.9780** |
| cuVS `all_neighbors` brute_force | 270.1 s | 1.000000 |
| **Sirius exact** (recorded 2026-08-23) | **120.8 s** | **1.000000** |

### What this changes — two headline claims are overstated

1. **"cuVS's own exact configuration is ~10× slower than us on the self-join."** ✗ That 269 s was
   `ivf_flat` with `n_probes = n_lists` — the wrong tool. cuVS's *right* exact tool for a self-join
   is `all_neighbors(brute_force)`. **The real margin is 1.24× at d=128 and 2.24× at d=960**, not
   9.88×. *Why it happened: we benchmarked the library we knew against the shape we cared about,
   instead of asking the library what it offers for that shape.*
2. **"CAGRA cannot reach recall 1.0 at any speed."** ✗ at d=128 — `all_neighbors(nn_descent)`
   reaches **0.99983 in 4.95 s, 5.6× faster than our exact path**. 34 misses in 200,000 pairs is
   recall 1.0 for any practical purpose. It **holds at d=960**, where the same call reaches only
   0.978.

**What survives, and it is still a real claim:** at *provable* exactness we are the fastest option
at both dimensionalities, and our exact GEMM fold beats cuVS's own brute-force all-neighbors by
1.24×/2.24× — an independent confirmation of P0's finding that our kernel is the better kernel.
The margin is single-digit, not an order of magnitude, and **any "no ANN index reaches recall 1.0"
wording must go.**

### What is genuinely unique to us, on this evidence

- **Radius / threshold on the GPU** — cuVS has no radius API at all; FAISS has one but is CPU-only
  here. This is the strongest shape-level differentiator, and it is exactly what I4 benchmarks.
- **Global top-k over pairs** — neither library offers it.
- **A separate probe side in a many-to-many call** — `all_neighbors` is self-join *only*: one
  dataset, no probe/corpus split. Asymmetric many-to-many still means a user loop.
- **In-engine composition and SQL surface** — unchanged, and not something either library targets.

### New evidence gap this opens (NOT run — needs a human decision)

`all_neighbors` with a **host** dataset and `n_clusters > 1` is a library batched path, so the
recorded "cuVS sharded 47.29 s (hand-written)" out-of-core comparison may be against the wrong
opponent too. At 1M rows the host path costs 10.59 s vs 4.95 s on device. **Whether it scales to
the 48 GiB corpus is untested**, and it bears directly on the out-of-core claim.

---

# I4 — the two unbenchmarked reduction modes: threshold and global top-k

They had zero numbers. They now have numbers, **two hard limits and two bugs**. Every result was
checked against an independent brute-force truth, not against the other mode.

## First, a semantics correction

`eps` is **plain L2**, matching DuckDB's `array_distance` — not squared L2. Verified by exact pair
counts against a cupy truth: `eps=100` → 3,445 pairs and `eps=50` → 1,113 pairs, both **exactly**
what brute force says for squared thresholds 10,000 and 2,500. FAISS's `range_search` takes a
**squared** radius, so any comparison must pass `eps**2`.

## Threshold / radius join — correct, 38× faster than the only competitor, and capped

**It is exact where it runs.** Pair counts match brute force exactly at every threshold tested.
Against FAISS-CPU (the only rival with a radius API — cuVS has none, I3), SIFT1M corpus, SIFT's
10k query set, one A100 vs 64 CPU threads:

| eps (plain L2) | max nbrs / probe | Sirius | FAISS-CPU `range_search` | pairs (Sirius / FAISS) | |
|---|---|---|---|---|---|
| 100 | 979 | **1.794 s** | 68.247 s | 20,849 / 20,844 | **38.0× faster** |
| 150 | 3,204 | **REFUSES** | 69.330 s | — / 260,832 | — |

*(The 5-pair difference is a boundary convention: Sirius includes pairs at exactly `eps`, FAISS
excludes them. Sirius matches the independent `<=` truth.)*

**The cap, and it is a hard one.** `join_mode => 'threshold'` is implemented as *top-k then filter*,
so it needs `k ≥ the largest number of neighbours any single probe has inside eps` — and cuVS's
`knn_merge_parts` is **`Unimplemented for k > 1024`**. So the mode can only answer a radius query
where **no probe has more than 1024 neighbours in range**. On SIFT1M that is roughly `eps ≤ 100`
(mean 2.1 neighbours per probe). At `eps = 150` — still only 26 neighbours per probe on average —
one probe has 3,204 and the whole query is refused.

Two things make this better than it sounds and one makes it worse:
- **It fail-closes.** It refuses with an actionable message rather than silently truncating. That
  is the right behaviour and it is what let this be diagnosed at all.
- **The user cannot pick k.** The required k is a property of the data at that eps, unknowable in
  advance; the mean is no guide (2.1 vs a max of 979).
- **Cost is flat in eps.** 1.794 s at `eps=100` is the same as per-row `k=1024` (1.791 s) and as
  `eps=50`. The mode always computes 1024 neighbours per probe to return ~2 — a **~490×
  over-computation** at `eps=100`. That, not the kernel, is why the win over FAISS is 38× and not
  larger.

**So I3's "radius is our differentiator" is only half right.** We are much faster where we can
answer, and we are the only GPU option at all — but FAISS's `range_search` has no such ceiling and
answers queries we reject. Lifting the cap means not going through `knn_merge_parts`.

## Global top-k — correct and cheap, but NOT a speed win

Sirius returns exactly the right 1,000 pairs (max abs difference 0.0004 on squared distances vs a
cupy running top-N over all 10⁹ pairs). SIFT 10k query set × 1M corpus:

| | time |
|---|---|
| Sirius `join_mode => 'global'`, k=1000 | 1.794 s |
| cuVS emulation: `brute_force` k=1000 search 1.433 s + host reduction **0.003 s** | **1.436 s** |
| *(Sirius global k=10: 0.335 s vs per-row k=10: 0.324 s — global costs essentially nothing extra)* | |

**cuVS is 1.25× faster.** The emulation neither library offers as an API turns out to be trivial:
the reduction is 0.003 s of the 1.436 s. **The value of global top-k is API convenience, not
performance** — and the I4 premise that "radius is plausibly where ANN indexes are weakest" is
right about radius and wrong about global top-k. What the number really compares is per-probe
k=1000 throughput, where cuVS's brute force beats our path by 1.25×.

## Two bugs, both in the global-mode plan rewrite

Both surface as `Executor Error: … TopN order index out of range`, from
`src/op/sirius_physical_top_n.cpp:70`. Isolated one factor at a time (`i4h.sql` → `run_i4h.log`):

1. **Omitting `left_output_columns` / `right_output_columns` breaks global mode.** With them the
   identical query works; without them it errors. Everything else — `exact` vs `exact-gemm`,
   `probe_source => 'scan'`, `LIMIT`, `.mode trash`, d=3 vs d=128, corpus 10k vs 1M — makes no
   difference.
2. **Any aggregate above the join breaks global mode**, even *with* output columns:
   `SELECT count(*) FROM sirius_knn_join(… join_mode => 'global' …)` always fails, while
   `SELECT left_id, right_id, distance FROM` the same call succeeds.

Bug 2 is **distinct from S2**: S2 is per-row mode, where aggregates are fine (J4 measured a
`GROUP BY avg`) and scalar functions fail. Here it is the reverse.

**Why this was never caught:** the only test of global mode
(`test_gpu_execution_vector_join_exact_per_row.cpp:436`) passes `left_output_columns` and
`right_output_columns` and selects columns directly — it threads the one path that works. A green
suite of our own fixtures again proves self-consistency rather than correctness.

⚠️ **A trap that nearly produced a meaningless benchmark.** The first global-top-k run used a probe
set drawn *from the corpus*, so every probe was its own zero-distance match and the global top-1000
was 1,000 self-matches — correctness "verified" against a truth that was equally degenerate. Any
global-top-k or radius benchmark needs a probe side **disjoint** from the corpus.

---

# I4b — "is our radius join actually unique / fastest?" (ducndh, 2026-08-24)

**No, and the 38× in §I4 above is against the wrong opponent — the same mistake as I3.**

§I4 compared the threshold join to FAISS-**CPU** because that is what happened to be installed.
The real question is *what is the fastest radius join a person could reach for today*, and the
answer is about 25 lines of cupy — or cuVS's `pairwise_distance` plus a mask. Neither is an index;
both just compute the distances and threshold them, which is **less** work than a top-k.

## The field, measured (`i4b_radius_shootout.py`, `i4b_radius_cpu.py`)

10k probes × 1M SIFT1M corpus, `eps = 100` plain L2, every contender **materialising the 20.8k
index pairs**, GPU contenders with the corpus already resident (as Sirius's is pinned), all warmed.

| method | where | time | vs Sirius |
|---|---|---|---|
| **cuVS `pairwise_distance` + mask** | GPU | **0.345 s** | **5.2× faster** |
| **cupy blocked brute force** | GPU | **0.515 s** | **3.5× faster** |
| **Sirius `join_mode => 'threshold'`** | GPU | **1.794 s** | — |
| FAISS-CPU `IndexIVFFlat` nprobe=64 | 64 thr | 3.57 s **+ 28.1 s build** | 0.50× |
| sklearn `radius_neighbors(brute)` | 64 thr | ~17.9 s ⁽ˢ⁾ | |
| FAISS-CPU `IndexFlatL2` | 64 thr | 66.0 s | |
| scipy `cKDTree.query_ball_point` | 64 thr | ~69.6 s ⁽ˢ⁾ | KD-trees degenerate at d=128 |
| plain DuckDB `array_distance` cross join | 64 thr | ~84 s ⁽ˢ⁾ | no Sirius |
| sklearn `radius_neighbors(kd_tree)` | 64 thr | ~177 s ⁽ˢ⁾ | |
| FAISS-**GPU** `GpuIndexFlatL2` | GPU | **NOT IMPLEMENTED** | `range_search` throws |

⁽ˢ⁾ measured at 1,000 probes and scaled ×10 — labelled, never presented as measured. Every exact
method returns the same 1,349 pairs at 1k probes; Sirius/cupy/cuVS agree on 20,849 at 10k, FAISS
returns 20,844 (it uses a strict `<`, we use `<=`, and `<=` matches the independent truth).

Steady-state cupy is **0.515 s** (5 repeats, min 0.513 / max 0.546). The 1.299 s in the first run
of the harness was cuBLAS warm-up, not the method.

## What is actually true about radius, restated

1. **No library ships a GPU radius *index*.** cuVS has no radius API; FAISS-GPU's `range_search`
   is an unimplemented stub that throws. That part of §I4 stands.
2. **But nobody needs one.** A brute-force GPU scan answers the query 5× faster than we do, with
   no index, no build, no recall question and **no ceiling** — it happily does `eps = 150`, which
   we refuse.
3. **So the radius mode has no performance case at this scale.** Its case is integration: it is in
   SQL, it composes, and it streams a corpus larger than device memory, which the brute-force
   block does not.

## The concrete fix, and it is well-signposted

Our threshold path is `top-k (k=1024) then filter`, so it does *more* work than the brute-force
scan and still hits cuVS's `knn_merge_parts` k≤1024 wall. That the fold itself is not the problem
is easy to show: **Sirius's exact per-row join scans the same 10¹⁰ distances in 0.324 s — 1.6×
FASTER than the cupy scan.** The radius mode should threshold *inside* the fold instead of routing
through a top-k. That removes the 490× over-computation, the k≤1024 ceiling and the 5× deficit in
one change.

## The methodological lesson — twice in two days

I3 compared us to `ivf_flat` when cuVS had `all_neighbors`. I4 compared us to FAISS-CPU when a
GPU brute force was the real opponent. Both times the error was the same shape:
**benchmarking the entry point we already knew instead of asking what the fastest available way to
answer this query is.** The check to run before quoting any margin: *would a competent engineer,
told to do this today, actually use the thing I measured against?*

---

# I4c — the fix was already written. Ported, 2026-08-24.

**§I4b concluded our radius join should threshold inside the fold instead of routing through a
k=1024 top-k. That kernel already existed and had for ten days.**

`dc579e33`, **Andy \<lin383@wisc.edu\>, 2026-08-14**, branch `yayen/sirius-exact-join-threshold`:
`brute_force_threshold` forks cuVS's `tiled_brute_force_knn` and replaces the per-tile `select_k`
with a `copy_if` on the threshold. No `k`, therefore no `knn_merge_parts` ceiling. From his header:
*"Because 'within eps' is independent across column tiles, no cross-tile merge is needed."*
`~/vecjoin/DISCUSSION_approx_and_threshold.md` (ducndh, 2026-08-15) had already judged it:
*"his replaces the search kernel, mine post-processes its output. His is the right layer."*

**Why I missed it:** I searched `src/` on the current branch and the local branch list. The work was
on a contributor branch not present in this clone, reachable only through the fork network. *Rule:
before concluding a fix is unwritten, search the fork network and the project's own design notes,
not just the checked-out tree.*

## Measured — his branch, built and run here (sm_80)

| eps | pairs | our old path | **Andy's branch** | cuVS `pairwise`+mask | cupy |
|---|---|---|---|---|---|
| 100 | 20,849 | 1.794 s | **0.342 s** | 0.345 s | 0.515 s |
| 150 | 260,864 | **REFUSED** | **0.345 s** | 0.343 s | 0.513 s |
| 200 | 1,789,956 | **REFUSED** | **0.345 s** | — | — |

Flat in eps — 0.345 s whether it returns 20k pairs or 1.79M.

## Ported to `vecjoin-approx-cluster` as `ef56272b`

Kernel, header and his 264 lines of unit tests copied **unmodified** (they compile clean against
current cuDF 26.06 / RAFT / CUDA 13.2 — no API drift). The **wiring is new**, because the branches
diverged exactly where his doc predicted: his work lives in the split `select`/`reduce_local`
operators, ours in the fused stream. So the exact path now forks:

- **top-k** — unchanged; folds each corpus chunk into a running `[n_left, k]` accumulator via
  `knn_merge_parts`, because chunk *j+1* can displace chunk *j*'s winners.
- **radius** — calls `brute_force_threshold` per chunk and **concatenates** the ragged edge lists.
  No merge, no truncation test, no `k`. `shape_threshold` remains for the approximate path only.

| eps | our branch before | **after the port** |
|---|---|---|
| 100 | 1.794 s | **0.376 s** (4.8×) |
| 150 | REFUSED | **0.355 s** |
| 200 | REFUSED | **0.355 s** |

All counts match the brute-force truth. Full suite **green: 32,147,501 assertions / 2,018 cases**.

**The obsolete test was replaced, not deleted.** `"threshold join refuses to truncate"` asserted the
limitation the port removes. It is now `"threshold join answers past k without truncating"` and is
*stronger*: an eps admitting thousands of pairs against `k => 4` must return **all** of them,
checked against an exhaustive CPU range query — not merely "did not error".

## What this does to the claim

**Radius goes back on the list, but as capability, not speed.** We are now at *parity* with the
fastest hand-written GPU option (0.355 s vs 0.345 s), not 5.2× behind it, and the ceiling is gone.
The differentiator is that it is in SQL, composes with `GROUP BY`, and streams out-of-core — the
brute-force block does none of that. Do not claim a speed win over a GPU scan; there isn't one.

**Still true from §I4b:** no library ships a GPU radius *index*, and none is needed.
