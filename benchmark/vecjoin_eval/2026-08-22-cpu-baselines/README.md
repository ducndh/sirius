# 2026-08-22 — CPU baselines for the exact vector join

**Question.** How fast is the GPU exact join against the CPU implementations of the *same
operator*, on one box, with the answers proven identical?

**Reproduce:** `./run.sh` (or `./run.sh vss_join|lateral|sirius`).

## Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, compute cap 8.0, built `CMAKE_CUDA_ARCHITECTURES=80-real` |
| CPU | 64 cores |
| Sirius | branch `vecjoin-approx-cluster` @ `ae5e3d0d` (fork `ducndh/sirius`) |
| DuckDB | v1.5.4 + `vss` extension (installed from the core repo) |
| Data | SIFT1M, 1,000,000 x FLOAT[128] corpus; 10,000 packaged queries |
| Shape | 10k probe x 1M corpus, k=10, L2, consumed under `.mode trash` |

## Results

| engine | time | CPU time | vs Sirius warm |
|---|---|---|---|
| DuckDB `vss_join` (shipped macro, brute force, **no index**) | **859.2 s** | 6971 s | **2736x** |
| DuckDB hand-written `LATERAL ... ORDER BY ... LIMIT` | **79.2 s** | 714 s | **252x** |
| Sirius GPU exact | **0.517 s** cold / **0.314 s** warm | 0.42 s | — |

## Verification — this is the point of the experiment

All three return the **same answer**, checked as a cross-engine oracle rather than
self-consistency:

| | min | max | rows |
|---|---|---|---|
| `vss_join` | 20.8087 | 361.866 | 100,000 |
| Sirius | 20.8087 | 361.866 | 100,000 |

`run.sh sirius` re-derives the Sirius side and prints the expected values beneath it.

## What the numbers mean — and what they do NOT

- **`vss_join` is 10.8x SLOWER than the hand-written LATERAL.** Counter-intuitive: its
  `min_by(tbl, score, k)` is a bounded heap and *should* beat a per-row sort. The macro
  `struct_pack`s WHOLE ROWS through the aggregate and `unnest`s at `max_depth := 2`, so 128-dim
  vectors ride through the top-k. This direction was predicted wrong before measuring.
- **Therefore the LATERAL is the STRONGER baseline and 252x is the conservative figure to
  quote.** Quoting 2736x against `vss_join` would be a strawman.
- **Neither is an ANN baseline.** `vss_join` does not use the HNSW index. The honest strong
  baseline is `LATERAL ... LIMIT` over an **HNSW index**, which triggers DuckDB's
  `HNSW_INDEX_JOIN` operator. **Not yet run — this is the next experiment.**
- Prior recorded figure was 209x on an **A5000**; 252x here on an **A100**. Same ballpark, but
  never mix boxes in one ratio.

## Traps this experiment walked into (do not repeat)

1. **`vss_join` with the same column name on both sides silently scores 0.0.** Table arguments
   are strings, column arguments are bare identifiers; with `(...,vec,vec,10)` both resolve to
   the inner table. Returns the correct row count in 0.263 s having done no work — a fake ~400x.
   `setup.sql` renames to `qvec`/`ivec` for exactly this reason. **Always check score values.**
2. **Sirius cannot be verified in-engine.** `round(min(distance),4)` over the join errors (see
   the scalar-function composition bug), so verification goes via CSV.
3. **Never run these concurrently.** A Sirius run burns CPU while pinning and inflates a timed
   64-core CPU baseline beside it.
