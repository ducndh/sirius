# Threshold (radius) join and global top-k — time vs the knob (2026-09-15, RTX A5000, sirius-4)

SIFT1M, packaged 10k queries × 1M base, exact path. eps values bracket the packaged ground-truth
distance distribution: the median 1st-NN distance is 195, the median 10th-NN 226, the median
50th-NN 244 (`@@@EPS` block in `run_sirius.log`), so eps=150 is a near-duplicate regime and eps=350
returns ~38 pairs per probe. Sirius = end-to-end SQL, min of last 2 of 3. The only baseline with a
radius API is FAISS-CPU `IndexFlatL2.range_search`, 64 threads (`faiss_range.py`); no GPU library
exposes one. Pair counts from `counts.sql` (Sirius) and `faiss_range.json` differ by ≤ 0.01 %,
the `<`/`<=` boundary and float32 rounding on ties.

## Threshold join

| eps | pairs (Sirius) | pairs / probe | **Sirius** | FAISS-CPU range_search | ratio |
|---|---|---|---|---|---|
| 150 | 260,864 | 26 | **0.473 s** | 52.3 s | 110× |
| 200 | 1,789,956 | 179 | **0.483 s** | 52.7 s | 109× |
| 250 | 12,404,959 | 1,240 | **0.604 s** | 55.6 s | 92× |
| 300 | 79,740,869 | 7,974 | **1.83 s** | 58.1 s | 32× |
| 350 | 378,920,485 | 37,892 | **11.3 s** | 69.6 s | 6.2× |

- Up to ~12 M pairs the threshold join costs what the exact top-10 join costs (0.46 s): the
  radius kernel scans the same 10^10 pairs and the output is cheap. From ~80 M pairs the cost is
  the output — 380 M (left, right, distance) rows are 4.5 GB — and the slope is the
  materialization, not the search.
- This is a CPU opponent for a GPU kernel, so the ratio is not the headline; the point is the
  shape (flat until the output dominates) and that the answer is the exact set with no k to guess.

## Global top-k ("the k closest pairs overall")

| k | **Sirius global mode** | emulation: per-row top-k to depth k, then `ORDER BY distance LIMIT k` |
|---|---|---|
| 100 | 0.453 s | 0.453 s |
| 1,000 | 2.21 s | 2.22 s |
| 10,000 | 2.67 s | — |
| 100,000 | refused: per-task budget (the [10k × 100k] candidate block is 12 GB) | — |

- Global mode costs exactly what its emulation costs, because it *is* that plan: every probe is
  searched to depth k so the global answer is bounded, then one relational TOP-N collapses the
  per-batch answers. Its value is the API (no k-per-row to reason about), not speed.
- The bound "one probe may own the whole answer" is what makes k=100k infeasible at 10k probes;
  a tightening global threshold (the τ from the prune-rate study) is the known better algorithm
  and is not implemented.
