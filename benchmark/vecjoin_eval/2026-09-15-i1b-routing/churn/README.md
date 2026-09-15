# J5 — churn: what a corpus change costs each system (2026-09-15, RTX A5000)

**Protocol.** SIFT1M base as a mutable DuckDB table `live`; per epoch, 10,000 random rows are deleted
and 10,000 new rows inserted (copies of random survivors shifted by +0.5 in every dimension, ids
≥ 1,000,000), i.e. **1% of the corpus replaced**, corpus size constant at 1M. After each mutation
every system must answer the same packaged 10k queries (k=10). 5 epochs; steady state = epochs 1–4.
Sirius steps are timed SQL statements (`.timer on`, `.mode trash`); cuVS is kernel-only with sync on
the same per-epoch corpora (exported to parquet). Recall: Sirius approx vs Sirius exact on that
epoch's corpus; cuVS vs a cuVS brute-force truth on the same corpus.

⚠ Two harness traps hit and fixed on the way (both in `gen_churn.sh` comments): DuckDB row ids keep
gaps after DELETE while `sirius_kmeans_assign` emits a dense scan index, so a cluster-ordered
corpus built from a mutated table is **silently mislabelled** (first run: recall fine only because
the INSERT had also failed) — rebuild the corpus fresh each epoch; and `REPEATABLE (100+e)` is a
parser error, the seed must be a literal.

## Steady-state cost to answer 10k queries after a 1% corpus change (`summary.txt`)

| system | prepare | answer | **total** | recall |
|---|---|---|---|---|
| **Sirius exact, corpus scanned unpinned** | 0 | 0.486 | **0.486 s** | 1.0 |
| Sirius exact, re-pin + join | 0.040 | 0.453 | 0.493 s | 1.0 |
| Sirius approx c64/p8: fit + assign + cluster-ordered CTAS + pin + join | 1.145 | 0.141 | 1.286 s | 0.988 |
| cuVS brute_force k=65 (exact) | 0.001 | 0.429 | 0.430 s | 1.0 |
| cuVS IVF-Flat 64/8 rebuild + search | 0.307 | 0.880 | 1.186 s | 0.989 |
| cuVS CAGRA nn_descent 32/64 rebuild + search | 9.446 | 0.015 | 9.461 s | 0.978 |

Every SQL path additionally pays **0.98 s** per epoch to materialize the mutated table
(`CREATE OR REPLACE TABLE base AS SELECT … FROM live ORDER BY id; CHECKPOINT`); the cuVS side got
its per-epoch array from that same table via parquet and its load is not counted. The 0.98 s is a
DuckDB cost, not a join cost, and it is the same for all three Sirius rows.

## What it says
- **Under churn the exact join is the cheapest way to get a correct answer, and it needs no
  preparation at all**: 0.486 s with the corpus scanned straight from the table, vs 1.19 s for IVF
  rebuild+search and 9.46 s for CAGRA rebuild+search. cuVS brute force is at parity (0.43 s) — the
  exact-vs-exact tie holds here too.
- **CAGRA's amortized win needs ≥ 20 query batches (≈200k queries) per corpus version**:
  9.446 / (0.486 − 0.015) ≈ 20. Below that rate of reuse, rebuilding a graph loses to simply
  joining. IVF-Flat never wins here: rebuild + search (1.19 s) is 2.4× our exact join.
- **Our own approximate path is the wrong tool under churn**: its preparation (1.15 s, of which
  1.05 s is the cluster-ordered CTAS, not the fit) is 8× its search. Approximate only pays when the
  clustering is reused across many batches — the same amortization argument, one notch cheaper.
- Recall is stable across epochs (0.9877–0.9897), i.e. the fold does not degrade with mutation.

## Files
`gen_churn.sh` → `/var/tmp/vj/churn/churn.sql` · `run_sirius.log/.err` · `churn_cuvs.py` →
`run_cuvs.log`, `cuvs_churn.json` · `recall.csv` · `summary.txt` (the parser that made the table).
