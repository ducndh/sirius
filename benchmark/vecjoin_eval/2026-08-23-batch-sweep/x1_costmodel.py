"""Fit the cost model the source implies, to size I1 (batch the per-slice cuVS calls).

`sirius_physical_vector_join_stream.cpp` issues one `vss::brute_force_knn(slice, queries_run, ...)`
per (corpus-cluster slice x probe run that wants it) -- the loop nest is
`for chunk / for slice / for run in wanting`. Each such call does a fresh `bf::build` over the
WHOLE slice (recomputing its norms) and then searches only that run's rows. So:

    time  ~=  C  +  a * CALLS  +  g * REREAD  +  B * PAIRS

    CALLS  = runs * n_probes                        (one cuVS call each)
    REREAD = CALLS * mean_cluster_size              corpus rows re-read by the per-call build
    PAIRS  = probes_n * n_probes * mean_cluster_size   distances actually scored
    C      = per-join setup (assign, sort, gather, labels to host, cluster index)

Each term names a fix:
    C -> nothing to batch, it is amortised by a bigger probe side
    a -> I1(b): one search per slice with all wanting runs concatenated
    g -> I1(a): hoist bf::build out of the per-slice loop; norms are recomputed every call today
    B -> the fold's real throughput; only a better kernel moves it

`runs` is MEASURED from the assignment tables, not assumed -- kmeans clusters are uneven and at
1k probes over 1024 clusters only 622 of them are occupied.

Fitted on RELATIVE error: the measured times span 0.08 s to 21 s, and an unweighted least squares
would fit the four largest points and ignore the small-batch regime the question is about.
"""
import json, numpy as np
from x1_score import OUT

RUNS = {("c64", 1000): 64, ("c64", 10000): 64, ("c64", 100000): 64, ("c64", 1000000): 64,
        ("c1024", 1000): 622, ("c1024", 10000): 1023, ("c1024", 100000): 1024,
        ("c1024", 1000000): 1024}
MEAN_SZ = {"c64": 15625.0, "c1024": 976.6}
NAMES = ["C (per join, ms)", "a (per call, us)", "g (per re-read row, ns)",
         "B (per scored pair, ns)"]
SCALE = [1e3, 1e6, 1e9, 1e9]

rows = []
for tag in ("c64", "c1024"):
    d = json.load(open(f"{OUT}/sirius_x1_{tag}.json"))
    for r in d["rows"]:
        if r["n_probes"] is None:
            continue
        calls = RUNS[(tag, r["probes_n"])] * r["n_probes"]
        reread = calls * MEAN_SZ[tag]
        pairs = r["probes_n"] * r["n_probes"] * MEAN_SZ[tag]
        rows.append((tag, r["probes_n"], r["n_probes"], calls, reread, pairs, r["search_s"]))

X = np.array([[1.0, c, rr, p] for _, _, _, c, rr, p, _ in rows])
y = np.array([t for *_, t in rows])
W = 1.0 / y[:, None]
coef, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
pred = X @ coef
rel = (pred - y) / y

print(f"fit over {len(rows)} points (2 cluster counts x 4 batch sizes x 4-5 n_probes), "
      f"relative-error weighted")
for nm, sc, v in zip(NAMES, SCALE, coef):
    print(f"  {nm:>26} = {v*sc:10.3f}")
print(f"  {'fold throughput':>26} = {2*128/coef[3]/1e12:10.2f} TFLOP/s   (2*d flops per pair)")
print(f"  {'max |relative error|':>26} = {abs(rel).max()*100:9.1f}%   "
      f"median {np.median(abs(rel))*100:.1f}%\n")

print(f"{'tag':>6} {'probes_n':>9} {'n_probes':>9} {'calls':>7} {'meas_s':>8} {'pred_s':>8} "
      f"{'err':>7} | share of predicted time: {'C':>5} {'calls':>6} {'reread':>7} {'pairs':>6}")
for (tag, pn, npr, c, rr, p, t), pv in zip(rows, pred):
    parts = np.array([coef[0], coef[1] * c, coef[2] * rr, coef[3] * p]) / pv * 100
    print(f"{tag:>6} {pn:>9} {npr:>9} {c:>7} {t:>8.3f} {pv:>8.3f} {(pv-t)/t*100:>+6.1f}% |"
          f"{'':25}{parts[0]:>5.0f}%{parts[1]:>6.0f}%{parts[2]:>7.0f}%{parts[3]:>6.0f}%")

print("\nHeadline: share of Sirius's own runtime that is NOT scored distances (C + calls + reread)")
print("  -- i.e. the fraction I1 and a bigger probe side are competing for.")
for tag in ("c64", "c1024"):
    for pn in (1000, 10000, 100000, 1000000):
        sh = [(npr, (1 - coef[3] * p / pv) * 100)
              for (t_, p_, npr, c, rr, p, t), pv in zip(rows, pred) if t_ == tag and p_ == pn]
        print(f"  {tag:>6} probes_n={pn:>9}: " + "  ".join(f"p{n}={s:.0f}%" for n, s in sh))
