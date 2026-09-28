"""I3 follow-up — cuVS ships an all-pairs self-join. Measure it against our J3 numbers.

`cuvs.neighbors.all_neighbors.build(dataset, k, params)` builds an all-neighbors k-NN graph for a
whole dataset -- which IS the many-to-many self-join, as a single library call. It also takes a
HOST dataset and an `n_clusters` batching parameter, so it is a candidate for the out-of-core
shape as well. Both matter: the project has been recording "neither library has join semantics --
a user loops", and "cuVS needs a hand-written sharded loop out-of-core".

Three configs, because they answer different claims:
  brute_force / n_clusters=1  -> is there an EXACT library self-join? (our exactness claim)
  nn_descent  / n_clusters=1  -> cuVS's best approximate self-join, in core (our J3 claim)
  nn_descent  / n_clusters=4  -> the batched path, on a host array (our out-of-core claim)

Recall is against a cupy brute-force truth over the same 1000 sampled rows X1 used.
"""
import sys, time, numpy as np, cupy as cp
sys.path.insert(0, ".")
from x1_score import Scorer, K
from cuvs.neighbors import all_neighbors as an

sc = Scorer()
base = sc.base
print(f"self-join {len(base):,} x {len(base):,}, k={K}, oracle = cupy brute force "
      f"on {len(sc.smp_row)} sampled rows\n", flush=True)
print(f"{'algo':>12} {'n_clusters':>11} {'where':>7} {'build_s':>9} {'recall_id':>10} {'recall_d':>9}",
      flush=True)

for algo, ncl, host in (("brute_force", 1, False), ("nn_descent", 1, False),
                        ("nn_descent", 4, True), ("ivf_pq", 4, True)):
    try:
        p = an.AllNeighborsParams(algo=algo, n_clusters=ncl,
                                  **({"overlap_factor": 2} if ncl > 1 else {}))
        data = base if host else cp.asarray(base)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        out = an.build(data, K, p)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        ind = out[0] if isinstance(out, (tuple, list)) else out
        got = cp.asarray(ind).get()[sc.smp_row].astype(np.int64)
        r_id, r_d = sc.score(got)
        print(f"{algo:>12} {ncl:>11} {'host' if host else 'device':>7} {dt:>9.3f} "
              f"{r_id:>10.4f} {r_d:>9.4f}", flush=True)
        del out, ind
    except Exception as e:
        print(f"{algo:>12} {ncl:>11} {'host' if host else 'device':>7} "
              f"{'ERROR':>9}  {type(e).__name__}: {str(e)[:60]}", flush=True)
    cp.get_default_memory_pool().free_all_blocks()
