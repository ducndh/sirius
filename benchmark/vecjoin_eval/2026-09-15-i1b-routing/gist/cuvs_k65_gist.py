"""Fairness check: cuVS brute_force at k=10 dispatches into fusedL2Knn, which our own operator
avoids by asking for k=65 and trimming. If that trick is what our exact-path margin is made of,
the opponent must be allowed it too (feedback_benchmark_discipline rule 7)."""
import time
import numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import brute_force

DATA = "/var/tmp/vj/data/parquet"

def load(n):
    t = pq.read_table(f"{DATA}/gist-960-euclidean_{n}.parquet")
    return np.ascontiguousarray(np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype(np.float32))

def timed(fn, reps=5):
    fn(); cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.runtime.deviceSynchronize(); t0 = time.perf_counter()
        fn(); cp.cuda.runtime.deviceSynchronize(); ts.append(time.perf_counter() - t0)
    return min(ts)

d_base, d_q = cp.asarray(load("base")), cp.asarray(load("query"))
idx = brute_force.build(d_base)
for k in (10, 64, 65, 128):
    print(f"cuVS brute_force k={k:<4} search {timed(lambda: brute_force.search(idx, d_q, k)):7.4f}s", flush=True)
