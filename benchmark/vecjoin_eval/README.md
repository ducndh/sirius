# Vector join evaluation harness

Everything behind the evaluation of `sirius_knn_join`: the scripts that generate each benchmark's
SQL, the cuVS / FAISS / DuckDB opponent scripts, the raw logs and result tables, and a README per
experiment with the numbers and the caveats. Each directory is one dated experiment; read its
README first, then the scripts. The write-up that stitches them together is the evaluation-section
draft (ask for the link).

## Layout

| directory | what it measures |
|---|---|
| `2026-09-15-i1b-routing/` | **the current numbers**: SIFT1M recall/time Pareto + cuVS at matched recall (`gen.sh`, `cuvs_packaged.py`, `matched_recall.txt`); GIST1M (`gist/`); 1M×1M self-join (`m2m/`); churn at 0.1/1/10 % (`churn/`); out-of-core on a 24 GB card (`ooc/`); corpus as a VIEW (`j1view/`); end-to-end composability (`compose/`); threshold / global top-k curves (`threshold/`); same-box before/after of per-row routing (`prefix/`); DuckDB `vss_join` (`vss_join*.sql`, `run_vss_join*.log`) |
| `2026-08-25-post-i1-pareto/` | the Pareto before per-row routing (A100) — the "before" column of the routing ablation |
| `2026-08-25-i1-batching/` | one cuVS search per corpus slice: 1.00×–6.55×, answer-equivalent |
| `2026-08-23-batch-sweep/` | probe batch 1k→1M vs cuVS IVF-Flat at matched recall; the cost model `C + a·CALLS + g·REREAD + B·PAIRS` |
| `2026-08-23-join-semantics/` | corpus as an intermediate result (J1), selectivity sweep (J2), composition (J4) — the CTAS+pin recipe that `j1view/` supersedes |
| `2026-08-23-out-of-core/` | 8 / 24 / 48 GiB synthetic corpora vs cuVS sharded, UVM, IVF-PQ and FAISS-CPU (A100) |
| `2026-08-24-er-dataset/` | entity resolution (Amazon-Google, Abt-Buy): threshold join, GROUP BY over it, the hybrid price-band query, F1 |
| `2026-08-23-cluster-granularity/`, `2026-08-23-dimensionality/` | cluster count sweep; GIST1M d=960 |
| `2026-08-22-hnsw-baseline/`, `2026-08-22-cpu-baselines/` | DuckDB HNSW, LATERAL brute force, FAISS HNSW |
| `2026-08-25-i7-prior-art/` | MaxVec, DiskJoin, SimJoin, DuckDB `vss_join`: which claims survive |

## Running anything here

1. Build Sirius for the card you have (`CUDAARCHS=<sm>-real pixi run make release`); every
   Sirius script uses `build/release/duckdb -unsigned`, which auto-loads the extension.
2. Data goes under `/var/tmp/vj/data/parquet/`: `sift-128-euclidean_{base,query,gt}.parquet` and
   `gist-960-euclidean_{base,query,gt}.parquet` (ann-benchmarks HDF5 converted to parquet; `id`,
   `vec FLOAT[d]`, and `query_id, rank, neighbor_id, distance` for the ground truth). The
   entity-resolution parquet files are built by `2026-08-24-er-dataset/build_er_dataset.py`; the
   out-of-core corpora by `2026-08-23-out-of-core/gen_corpus.py <gib>`.
3. cuVS opponents need `cuvs`, `cupy`, `pyarrow` (and `duckdb`, `faiss-cpu` for some scripts) in
   the system Python; cuVS 26.02 was used.
4. A generator (`gen*.sh`) writes SQL under `/var/tmp/vj/<exp>/`; run it through the CLI as the
   experiment README shows, then the matching `*_cuvs.py`. Scripts that set `VECJOIN_EVAL` expect
   it to point at this directory.

## Rules the numbers follow

- Sirius times are end-to-end SQL (`.mode trash`, `.timer on`, warm, min of the last two of
  three); cuVS times are kernel-only with a device sync. The shell timer quantises at 10 ms.
- Recall is always against an external truth (packaged ground truth, or an independent brute
  force), never against our own run, except the 1M×1M self-join where the exact join is the truth.
- Every library comparison is reported under two accountings, amortized (index for free) and
  one-shot (build + search). Opponents are run at their best configuration (CAGRA with
  `nn_descent`, brute force at k=65 trimmed); headlines that failed this rule were retracted and
  are listed in the READMEs.
- Approximate numbers are meaningless without their cluster count; cuVS k-means is not
  bit-stable across processes, so clusterings are built inside the measuring session.
- GPU: RTX A5000 24 GB (sm_86) for everything dated 2026-09-15; A100-40GB for earlier runs.
  Absolute times are not comparable across the two.
