# I6 — composability, end to end (2026-09-15, RTX A5000, commit after b21525fd)

One SQL statement: filter the probe side (a subquery), vector-join it against the corpus, equi-join
the matches back to the corpus for an attribute, aggregate. SIFT1M base as the corpus with a
synthetic `category = id % 97`; probes = packaged queries with `id % 4 = 0` (2,500 rows); k=10,
exact. Against the practitioner path (`compose_cuvs.py`): DuckDB filters and exports the probes,
cuVS brute force (k=65 trimmed) finds the matches, ids are imported back, DuckDB joins and
aggregates. Two accountings for the library: the corpus already resident on the device
(amortized) or exported from the database for this query (one-shot). Answers identical: 97
categories, 25,000 matches.

| form | time |
|---|---|
| **Sirius, one statement** (filter → join → equi-join → GROUP BY) | **0.141 s** |
| cuVS path, corpus already on the device | 0.125 s (filter+export probes 0.007, search 0.105, import+join+aggregate 0.013) |
| cuVS path, corpus exported for this query | **2.634 s** (export 2.508) |
| Sirius `CREATE TABLE AS` over the join (GPU operator under DuckDB's sink) | 0.141 s, then 0.030 s for the equi-join + aggregate over the stored table |
| Sirius `COPY (join) TO parquet` | 0.141 s |
| Sirius `round(distance,1)`, `sqrt(distance)` over the join in the same plan | 0.202 s |

- The whole pipeline in SQL costs the same as the library's bare search when the library is
  handed the corpus for free (0.141 vs 0.125 s), and 19× less when it has to fetch the corpus.
- CTAS, COPY and named functions over the join each cost what the join costs; nothing falls to
  the CPU any more. The CTAS form needed one more fix after B6: DuckDB's order-preserving sinks
  ask their source for a batch index per chunk (`GetPartitionData`), which the spliced GPU
  operator now supplies (`fix(transparent): batch-index protocol`).
- Not measured: GPU utilisation during the pipeline (no profiler on these pods).
