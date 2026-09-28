# 1M × 1M self-join after I1b (2026-09-15, RTX A5000)

SIFT1M base joined against itself, k=10 (the self-match is one of the 10 on both sides). Recall on
the first 10k probe rows against Sirius's exact answer for those rows. Sirius times are end-to-end
SQL with `.mode trash` (min of 2); answers exported through the shell in csv mode because
`COPY (... sirius_knn_join ...)` still falls to the CPU stub (bug B6). cuVS is kernel-only with
device sync, every base vector as a query; recall vs a cuVS brute-force truth for the same 10k rows.
`matched_recall.txt` is the full table.

| | time | recall |
|---|---|---|
| Sirius exact | 45.2 s | 1.0 |
| cuVS brute_force k=65 | 43.7 s | 1.0 |
| **Sirius c256 p8** | **3.2 s** | 0.9426 |
| cuVS IVF-Flat best at ≥0.94 (256 lists / 8 probes) | 22.7 s (build 0.34 + search 22.4) | 0.9416 |
| **Sirius c256 p16** | **6.0 s** | 0.9861 |
| cuVS IVF-Flat 256/16 | 44.9 s | 0.9857 |
| **Sirius c64 p16** | 14.9 s | 0.9988 |
| cuVS CAGRA 32/64 (build + search) | 11.0 s (9.6 + 1.5) | 0.9976 |
| cuVS CAGRA 32/64 (search only) | 1.5 s | 0.9976 |
| cuVS IVF-Flat 256/32 | 90.1 s | 0.9978 |

- **Exact-vs-exact is parity for the third time** (45.2 vs 43.7 s).
- **Against IVF-Flat at its own best list count we are ~7× faster at matched recall, under either
  accounting** (3.2 vs 22.4 s at 0.94; 6.0 vs 44.6 s at 0.986). The per-query IVF kernel scales
  with probes × n_probes; our per-slice GEMM does not.
- **Against CAGRA the build-inclusive advantage is gone at this probe count.** At CAGRA's recall
  0.9976 the cheapest Sirius config at or above it is c256 p32 at 11.5 s against CAGRA's 11.0 s
  build + search: parity. At 0.9989 CAGRA wins 1.9× even with its build counted (12.0 vs 22.5 s),
  and search-only it is 5–9× faster. One batch of 1M queries is enough to amortise a 9.6 s build,
  which is exactly the crossover the churn experiment predicted (~200k queries per corpus version).
- At 1M probes **256 clusters overtakes 64** at every recall (3.2 vs 7.5 s at ~0.94), because the
  per-slice overhead is amortised over far more probes; cluster count remains a first-class knob.
