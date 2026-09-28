"""I4 — the radius/threshold baseline. FAISS-CPU is the ONLY competitor with a range API
(I3: cuVS has none, and this FAISS build is CPU-only), so this is the comparison that exists.

FAISS `IndexFlatL2` uses SQUARED L2, Sirius's `eps` is PLAIN L2 (it matches DuckDB's
`array_distance` -- verified against brute-force pair counts, 3,445 and 1,113 exactly). So the
radius passed to FAISS is eps**2.
"""
import time, numpy as np, faiss, pyarrow.parquet as pq
from x1_score import Scorer

sc = Scorer()
base = sc.base
faiss.omp_set_num_threads(64)
idx = faiss.IndexFlatL2(base.shape[1])
t0 = time.perf_counter(); idx.add(base); add = time.perf_counter() - t0
print(f"FAISS IndexFlatL2 add {len(base):,} rows: {add:.3f} s, {faiss.omp_get_max_threads()} threads\n")

print(f"{'probes':>8} {'eps':>7} {'faiss_s':>9} {'pairs':>10} {'sirius_s':>9} {'sirius x':>9}")
SIR = {(1000, 50.0): 0.263, (1000, 100.0): 0.213, (10000, 50.0): 1.800, (10000, 100.0): 1.801}
for cnt in (1000, 10000):
    pid = np.asarray(pq.read_table(f"/var/tmp/vj/x1/probe_ids_{cnt}.parquet").column("id"),
                     dtype="int64")
    q = np.ascontiguousarray(base[sc.row_of_id[pid]])
    for eps in (50.0, 100.0):
        idx.range_search(q[:100], eps ** 2)                      # warm
        t0 = time.perf_counter()
        lims, D, I = idx.range_search(q, eps ** 2)
        dt = time.perf_counter() - t0
        s = SIR[(cnt, eps)]
        print(f"{cnt:>8} {eps:>7.0f} {dt:>9.3f} {len(I):>10,} {s:>9.3f} {s/dt:>8.2f}x")

print("\nper-row top-k reference at the same k the threshold path is forced to use:")
for k in (10, 1024):
    idx.search(q[:100], k)
    t0 = time.perf_counter(); idx.search(q, k); dt = time.perf_counter() - t0
    print(f"  FAISS-CPU search k={k:>4}, 10k probes: {dt:>8.3f} s")
