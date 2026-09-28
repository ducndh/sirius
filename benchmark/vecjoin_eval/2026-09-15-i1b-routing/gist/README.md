# GIST1M (d=960) after I1b — Pareto + cuVS at matched recall (2026-09-15, RTX A5000)

Packaged 1k queries × 1M base, k=10, recall@10 vs packaged GT (ceiling 0.9993). Same protocol as
the SIFT run in `..` (`gen_ds.sh` with `DS=gist-960-euclidean DIM=960`); ⚠ 1k queries, not 10k, so
absolute times are NOT comparable to SIFT's. `matched_recall.txt` has the full table; the cuVS
sweep died on the CAGRA `ivf_pq` configs (degenerate graph + crash, as on 08-23) after every
`nn_descent` row was recorded, so `run_cuvs.log` is the source and there is no `cuvs.json`.

| | time | recall |
|---|---|---|
| Sirius exact | **0.201 s** | 0.9990 |
| Sirius c64_p8 | 0.121 s | 0.9438 |
| Sirius c64_p16 | 0.141 s | 0.9924 |
| Sirius c64_p32 | 0.181 s | 0.9992 |
| cuVS brute_force k=10 | 0.583 s search | 0.9992 |
| cuVS brute_force **k=65 tuned** (`run_cuvs_k65.log`) | **0.175 s** search | — |
| cuVS IVF-Flat 64/4 | 0.372 s search + 1.46 s build | 0.8279 |
| cuVS IVF-Flat 64/16 | 1.456 s search + 1.44 s build | 0.9919 |
| cuVS CAGRA nn_descent 64/256 | 0.087 s search + **21.8 s build** | 0.9910 |

- **Exact-vs-exact is parity here too**: 0.201 s vs 0.175 s tuned (1.15×); the 2.9× against the
  k=10 call is the mis-dispatch again — do not quote it.
- **IVF-Flat loses to us amortized at every recall ≥ 0.83** (0.05×–0.37×), and 1-shot by 14–25×.
- **CAGRA at d=960**: build 20.8–21.8 s, best recall 0.991 at itopk=256; amortized it is 1.6× faster
  than our c64_p16 (0.087 vs 0.141 s) — a much smaller margin than at d=128 (3–9×) — and 1-shot
  it loses by ~150×. High dimension hurts the graph more than it hurts the fold.
- 64 clusters is the optimum at d=960 as well; recall at fixed n_probes is ~0.05 lower than on
  SIFT at every point (0.8215 vs 0.9329 at p4), so GIST needs ~2× the probes for the same recall —
  consistent with cuVS IVF-Flat, which shows the same shift (0.8279 vs 0.9381 at nprobe=4).
