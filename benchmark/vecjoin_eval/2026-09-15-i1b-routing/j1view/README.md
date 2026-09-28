# J1 re-run with the corpus as a VIEW (S4 landed, 2026-09-15, RTX A5000)

The canonical join case: the corpus is the output of a filter (`SELECT id, vec FROM base WHERE
id % 100 < sel`), expressed as a view and passed to the join with `build_source => 'scan'`. The
planner binds and streams it; nothing is materialized or pinned. Compared with the previous
recipe (CTAS + CHECKPOINT + pin + join over the pinned table) and with the cuVS practitioner path
(DuckDB filter → export → host-to-device → brute-force k=65 → map ids back). SIFT1M, 10k packaged
queries, k=10, exact. Sirius = end-to-end SQL (min of last 2 of 3); cuVS stages timed with device
sync, min of 3.

| selectivity | **Sirius, view corpus** | Sirius, CTAS + pin + join | cuVS practitioner path (export + h2d + build + search) | cuVS search only |
|---|---|---|---|---|
| 10 % (100k rows) | **0.101 s** | 0.715 s | 0.299 s | 0.042 s |
| 30 % | **0.201 s** | 1.661 s | 0.879 s | 0.132 s |
| 50 % | **0.282 s** | 2.668 s | 1.637 s | 0.231 s |
| 90 % | **0.454 s** | 4.560 s | 2.872 s | 0.378 s |

- **The August loss is reversed.** J1 on 2026-08-23 lost 1.69 s to 1.12 s at 30 % because the
  corpus had to be materialized and pinned (1.28 s of 1.69). With the view streamed straight into
  the operator the same query is **0.201 s: 4.4× faster than the cuVS path, 8× faster than our
  own CTAS recipe**, and the margin holds at every selectivity (3.0×, 4.4×, 5.8×, 6.3×).
- The join alone (view time) is within 1.2–2.4× of cuVS's *search-only* kernel time, i.e. the
  exact-vs-exact parity of §6.3 with the filter and the streaming included.
- Answers were checked equal against the pinned-table join for the 10 % case in the unit test
  (`sirius_knn_join takes the corpus from a VIEW`); the CSVs here are the full outputs.
