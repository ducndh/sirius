"""I3 — pin down `all_neighbors(nn_descent)`'s recall with enough resolution to matter.

The 1000-row sample used elsewhere can only resolve recall to ~1e-4, and it reported 1.0000 for a
method that is supposed to be approximate. The claim at stake -- "at recall 1.0 Sirius is the
fastest option and no ANN index reaches it at any speed" -- cannot rest on a sample that small.
20,000 rows x 10 neighbours = 200,000 pairs resolves to 5e-6.

Scored by DISTANCE against a cupy brute-force truth, so SIFT1M's duplicate vectors do not
manufacture a fake miss (see gen_x1_probes.py).
"""
import time, numpy as np, cupy as cp, pyarrow as pa, pyarrow.parquet as pq

DATA = "/var/tmp/vj/data/parquet"
K, SAMPLE, SEED = 10, 20_000, 20260823


def flat(col, dim=128):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


tbl = pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet")
base = flat(tbl.column("vec"))
n = len(base)
smp = np.sort(np.random.default_rng(SEED).choice(n, SAMPLE, replace=False))
d_all = cp.asarray(base)
d_s = d_all[cp.asarray(smp)]
sn = (d_s ** 2).sum(1, keepdims=True)
bd = cp.full((SAMPLE, K), cp.inf, dtype=cp.float32)
t0 = time.perf_counter()
for off in range(0, n, 100_000):
    blk = d_all[off:off + 100_000]
    dist = sn + (blk ** 2).sum(1)[None, :] - 2.0 * (d_s @ blk.T)
    cd = cp.sort(dist, axis=1)[:, :K]
    bd = cp.sort(cp.concatenate([bd, cd], 1), axis=1)[:, :K]
    del dist, cd
    cp.get_default_memory_pool().free_all_blocks()
kth = bd[:, K - 1]
print(f"brute-force truth for {SAMPLE:,} rows: {time.perf_counter()-t0:.1f} s", flush=True)

from cuvs.neighbors import all_neighbors as an


def recall_of(ind):
    got = cp.asarray(ind)[cp.asarray(smp)]
    hits = 0
    for lo in range(0, SAMPLE, 2000):
        g = got[lo:lo + 2000]
        q = d_s[lo:lo + 2000]
        nb = d_all[g.reshape(-1)].reshape(g.shape[0], K, -1)
        d = ((nb - q[:, None, :]) ** 2).sum(-1)
        hits += int((d <= kth[lo:lo + 2000, None] * (1 + 1e-4) + 1e-4).sum())
    return hits / (SAMPLE * K)


print(f"\n{'algo':>12} {'n_clusters':>11} {'where':>7} {'build_s':>9} {'recall_dist':>12} {'misses':>9}",
      flush=True)
for algo, ncl, host in (("nn_descent", 1, False), ("nn_descent", 4, True),
                        ("brute_force", 1, False)):
    p = an.AllNeighborsParams(algo=algo, n_clusters=ncl,
                              **({"overlap_factor": 2} if ncl > 1 else {}))
    data = base if host else d_all
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    out = an.build(data, K, p)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    ind = out[0] if isinstance(out, (tuple, list)) else out
    r = recall_of(ind)
    print(f"{algo:>12} {ncl:>11} {'host' if host else 'device':>7} {dt:>9.3f} "
          f"{r:>12.6f} {int(round((1-r)*SAMPLE*K)):>9,}", flush=True)
    del out, ind
    cp.get_default_memory_pool().free_all_blocks()
