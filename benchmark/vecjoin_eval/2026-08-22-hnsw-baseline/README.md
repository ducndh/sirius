# 2026-08-22 — HNSW (indexed ANN) baseline for the vector join

**Question.** Everything measured before this was brute-force-vs-brute-force. For a paper whose
contribution is an *approximate* join, "you never compared against an index" is the first
reviewer attack. What does DuckDB's own HNSW index cost, at what recall, on the same workload?

**Reproduce:** `./run.sh` (or `./run.sh plan_probe|join|sweep|anchor|recall`), plus
`python3 hnsw_parallel.py` for the all-cores CPU point.

## Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, cc 8.0 (**not used by this experiment** — CPU only) |
| CPU | 64 cores, 96 GB RAM, **no swap** |
| Indexed baseline | **DuckDB v1.5.4 + `vss` `b833341`** for Sirius; **v1.4.4 + `vss` `9b25336`** for the HNSW runs — see "Which version" below |
| Data | SIFT1M, 1,000,000 × FLOAT[128] corpus; 10,000 packaged queries; packaged top-100 ground truth |
| Shape | 10k probe × 1M corpus, k=10, L2, consumed under `.mode trash` |
| Index | HNSW, `metric='l2sq'`, M=16, ef_construction=128 (vss defaults) |

## Results — DuckDB `HNSW_INDEX_JOIN`, v1.4.4

Index build: **30.3 s wall / 810 s CPU** (parallel). Search is **single-threaded** — see below.

| ef_search | time | recall@10 |
|---|---|---|
| 10 | 1.52 s | 0.7004 |
| 20 | 2.10 s | 0.8309 |
| 40 | 3.29 s | 0.9210 |
| **default (64)** | **4.50 s** | **0.9599** |
| 80 | 5.08 s | 0.9709 |
| 160 | 8.24 s | 0.9914 |
| 320 | 14.60 s | 0.9976 |

All points return exactly 10 neighbours for all 10,000 queries (100,000 rows).

## Results — FAISS HNSW, the *fair* CPU point (`hnsw_parallel.py`)

Same index parameters (M=16, ef_c=128), same data, same k. Index build **18.6 s** (64 threads).

| threads | ef | time | recall@10 | qps |
|---|---|---|---|---|
| 64 | 10 | 0.024 s | 0.7122 | 416,658 |
| 64 | 40 | 0.078 s | 0.9281 | 128,002 |
| **64** | **64** | **0.207 s** | **0.9639** | 48,352 |
| 64 | 160 | 0.256 s | 0.9921 | 39,101 |
| 64 | 320 | 0.533 s | 0.9977 | 18,759 |
| 1 | 64 | 1.376 s | 0.9639 | 7,269 |
| 1 | 320 | 5.492 s | 0.9977 | 1,821 |

**Two findings that change how this project should be positioned.**

1. **DuckDB's operator carries ~3.3× overhead over the library.** At the same thread count (1)
   and the same recall (~0.96), DuckDB is 4.50 s and FAISS is 1.376 s. So the DuckDB number is
   not "what HNSW costs" — it is what HNSW costs *inside DuckDB's single-threaded operator*.
   Quoting DuckDB's 4.50 s as "the CPU index baseline" would overstate our advantage by 3×.
2. **The "252× vs CPU" headline is a statement about brute force, not about CPU.** A parallel
   CPU HNSW does 0.207 s @ recall 0.964, versus the 79.2 s brute-force LATERAL the 252× was
   computed against. Framing built on 252× will not survive a reviewer who runs FAISS.
   ⚠️ **This does not say a CPU index beats us.** 0.207 s is approximate; our 0.314 s is *exact* —
   different quantities. The matched-recall approx-vs-approx evidence (1M×1M self-join) puts
   Sirius within 5–16% of cuVS and FAISS. See "What these numbers do NOT mean".

Parallel speedup is only 1.376/0.207 = **6.6× on 64 cores** — HNSW graph traversal is
memory-latency bound, not compute bound. Times under ~0.3 s here are near the noise floor
(ef=64 measured slower than ef=80); treat the small-ef points as indicative only.

## Results — cuVS on the same A100, the tier-1 GPU baseline (`cuvs_gpu.py`)

Reproduced twice; same workload, same ground truth. Build time is reported separately.

| index | param | build | search | recall@10 |
|---|---|---|---|---|
| IVF-Flat (n_lists=1024) | n_probes=1 | 0.33 s | 0.0043 s | 0.3724 |
| IVF-Flat | n_probes=4 | 0.33 s | 0.0143 s | 0.7008 |
| **IVF-Flat** | **n_probes=16** | **0.33 s** | **0.0536 s** | **0.9311** |
| IVF-Flat | n_probes=64 | 0.33 s | 0.2094 s | 0.9954 |
| IVF-Flat | n_probes=256 | 0.33 s | 0.8122 s | 0.9994 |

**IVF-Flat is the direct analogue of our clustered approximate join** — same algorithm family
(partition the corpus, probe n partitions), same GPU. At recall 0.93 it takes **0.054 s**, ~4×
faster than parallel FAISS HNSW at similar recall, with a 0.33 s build versus 18.6 s for HNSW.
**Do not compare this to Sirius's exact 0.314 s** — see "What these numbers do NOT mean". The
matched-recall approx-vs-approx result on the self-join workload is near-parity (5–16%).

⚠️ **CAGRA numbers in the log are INVALID — do not use them.** Every CAGRA point returns recall
~0.0003 (random), with a cuVS warning about a degenerate IVF-PQ graph build. That is a
misconfiguration in `cuvs_gpu.py`, not a cuVS result. Fix the build params before quoting CAGRA;
IVF-Flat is the more relevant comparison anyway.

## Verification

**Cross-version oracle.** The whole curve was run twice on two independent DuckDB releases:

| | v1.3.2 (`vss ccfa7c9`) | v1.4.4 (`vss 9b25336`) |
|---|---|---|
| index build | 27.6 s | 30.3 s |
| default ef, cold | 4.84 s | 5.11 s |
| default ef, warm | 4.47 s | 4.50 s |
| recall@10 | 0.9593 | 0.9599 |

Two engines a year of development apart agree to ~1% on time and 6e-4 on recall. That is the
check that the *host version does not influence this measurement* — the time is usearch walking
a graph on one core, not DuckDB doing relational work.

**Id mapping.** Brute force on this data reproduces the packaged ground truth exactly (query 0:
ids 932085/934876/561813/708177/706771, distances 232.8712/234.7147/243.9898/255.4604/256.3143).
So recall@10 computed by id-intersection against `_gt.parquet` is sound.

## Which DuckDB version — and why not the one Sirius pins

`HNSW_INDEX_JOIN` is the optimizer rule that rewrites `LATERAL … ORDER BY array_distance … LIMIT k`
into an index join. Measured on upstream's own fixture at realistic scale:

| DuckDB | vss | `HNSW_INDEX_JOIN` |
|---|---|---|
| v1.3.2 | `ccfa7c9` | fires |
| **v1.4.4** | `9b25336` | **fires** ← baseline runs here |
| v1.5.4 | `b833341` | **does not fire** (all shapes, 200k and 1M) |

This is a **known upstream bug, already fixed**, not a deprecation:
[duckdb-vss #80](https://github.com/duckdb/duckdb-vss/issues/80), "HNSW_INDEX_JOIN regression in
DuckDB 1.5.0". DuckDB 1.5.0 changed the lateral top-k into an `arg_min` top-k plan shape, so the
rule's pattern match silently stopped matching — no error, just a `CROSS_PRODUCT`. Fixed on `main`
2026-06-02 ("Fix HNSW lateral-join optimizer for DuckDB 1.5 arg_min top-k plans (#80)"). The
shipped `b833341` is literally the *"Bump to v1.5.0"* merge (PR #79) — i.e. the pre-fix state; the
fix was never rebuilt for the v1.5.x channel, and there is no nightly vss for v1.4.4 or v1.5.4
(both 404; only 1.3.2 has one).

**Not worth building vss from source to close this.** The v1.3.2/v1.4.4 agreement above shows the
host version does not move the number, and a hand-built unreleased extension is a *weaker* paper
baseline than a stock release nobody can question. v1.4.4 is one minor version from Sirius's
v1.5.4.

**The index itself is fine on v1.5.4** — single-vector `ORDER BY array_distance(col, <constant>)
LIMIT k` fires `HNSW_INDEX_SCAN` there. Only the batch/join rewrite is broken.

## What these numbers do NOT mean

- **`HNSW_INDEX_JOIN` is single-threaded**, and additionally ~3.3× slower than the same index in
  FAISS at the same thread count. **Do not quote the 4.50 s as "the CPU index baseline"** — the
  fair CPU number is FAISS at 64 threads (0.207 s @ 0.964). The DuckDB figure answers a
  different, narrower question: what a DuckDB user gets today, in-engine, without leaving SQL.
- **Index build is excluded from the query times.** 30 s is real, and for a one-shot *join* it
  arguably belongs inside the measurement. State the choice; do not bury it.
- **This is a CPU index.** The true competitor for a GPU approximate join is a GPU ANN library
  (cuVS CAGRA / IVF-Flat, FAISS-GPU) on the same A100. **Not yet run** — next experiment.
- **No Sirius number is quoted here, and the exact join is NOT the right comparison.** Every
  baseline in this directory is APPROXIMATE. Sirius's approximate curve on this workload was NOT
  run (the box rebooted and took the build tree with it), so the only Sirius figure available in
  this regime is the *exact* 0.314 s — a different quantity. Reading 0.054 s against 0.314 s
  compares approximate search to exact search and understates our position.
  **The approx-vs-approx comparison exists on the 1M×1M self-join and is near-parity: Sirius
  6.64 s @ 0.873 vs cuVS IVF 5.73 s @ 0.869 and FAISS IVF 6.31 s @ 0.869 — 5–16% ahead, not
  multiples** (`../../active_vector_join_approx_granularity.md`, "The frontier"). Probe-count
  scaling extrapolates Sirius to ~0.066 s in this regime, which is an extrapolation to be
  replaced by a measurement, not cited.

## Traps this experiment walked into (do not repeat)

1. **Upstream's own plan fixture is a false negative.** `hnsw_lateral_join_plan.test` builds a
   **2-row** table; on two rows the rule does not fire even on versions where it demonstrably
   works at 1M. This produced a wrong "v1.4.4 is broken" conclusion that survived until a
   realistic-scale re-test. `plan_probe.sql` now uses 200k rows. **Never probe an optimizer rule
   on a toy table.**
2. **An in-memory DuckDB has no spill target.** The brute-force anchor was run against
   `:memory:` with no `memory_limit` on a 96 GB box with **no swap**, and the box went down
   mid-run. The earlier cpu-baselines experiment had correctly used a file-backed DB for that
   same query; the `:memory:` choice was carried over from the HNSW runs, where it is fine
   (~1.6 GB), into the anchor, where it is not. `bf_anchor.sql` is now file-backed with an
   explicit `memory_limit`, a `temp_directory` to spill into, and `watchdog.sh` to kill on RSS.
3. **Warning signs were misread as engine differences.** The anchor burned 1537 s CPU on a query
   that took 79 s on another version and was allowed to keep running on the theory that "v1.3.2
   is just slower". Check RSS before believing that story.
4. **`pgrep -f <script>` waiters and `pkill -f <script>` self-match** and kill their own shell —
   this bit three times in one session, including mid-file-write. Already recorded in
   `feedback_background_watcher_self_match.md`; it applies to `pkill` too.
5. **A persisted HNSW index is slow to reopen.** Opening a 797 MB DB with a persisted index did
   not finish in 2 minutes. Build in-memory per session instead (30 s), or budget for the load.
6. **`.mode trash`, never `.output /dev/null`.** duckbox formats every row regardless
   (~1.15 µs/row). `../../approx_join_bench.sh` still has this bug in its timed runs; the
   corrected copy is `sirius_approx.sh` in this directory.
