"""I3 — does cuVS's all-neighbors self-join hold up at d=960? E4 showed dimensionality flips
several of this project's conclusions, so a SIFT-only finding is not a finding.

Sirius's own GIST1M exact self-join is 120.77 s @ recall 1.0 (recorded, 2026-08-23).
"""
import time, numpy as np, cupy as cp, pyarrow as pa, pyarrow.parquet as pq
from cuvs.neighbors import all_neighbors as an

DATA = "/var/tmp/vj/data/parquet"
K, SAMPLE, SEED, DIM = 10, 5_000, 20260823, 960

col = pq.read_table(f"{DATA}/gist-960-euclidean_base.parquet").column("vec")
if isinstance(col, pa.ChunkedArray):
    col = col.combine_chunks()
base = np.asarray(col.values, dtype="float32").reshape(-1, DIM)
n = len(base)
print(f"GIST1M {n:,} x {DIM} ({base.nbytes/2**30:.2f} GiB)", flush=True)

smp = np.sort(np.random.default_rng(SEED).choice(n, SAMPLE, replace=False))
d_all = cp.asarray(base)
d_s = d_all[cp.asarray(smp)]
sn = (d_s ** 2).sum(1, keepdims=True)
bd = cp.full((SAMPLE, K), cp.inf, dtype=cp.float32)
t0 = time.perf_counter()
for off in range(0, n, 50_000):
    blk = d_all[off:off + 50_000]
    dist = sn + (blk ** 2).sum(1)[None, :] - 2.0 * (d_s @ blk.T)
    bd = cp.sort(cp.concatenate([bd, cp.sort(dist, axis=1)[:, :K]], 1), axis=1)[:, :K]
    del dist
    cp.get_default_memory_pool().free_all_blocks()
kth = bd[:, K - 1]
print(f"brute-force truth for {SAMPLE:,} rows: {time.perf_counter()-t0:.1f} s", flush=True)


def recall_of(ind):
    got = cp.asarray(ind)[cp.asarray(smp)]
    hits = 0
    for lo in range(0, SAMPLE, 500):
        g = got[lo:lo + 500]; q = d_s[lo:lo + 500]
        nb = d_all[g.reshape(-1)].reshape(g.shape[0], K, -1)
        d = ((nb - q[:, None, :]) ** 2).sum(-1)
        hits += int((d <= kth[lo:lo + 500, None] * (1 + 1e-4) + 1e-4).sum())
        del nb, d
        cp.get_default_memory_pool().free_all_blocks()
    return hits / (SAMPLE * K)


print(f"\n{'algo':>12} {'build_s':>9} {'recall_dist':>12} {'misses':>8}", flush=True)
for algo in ("nn_descent", "brute_force"):
    try:
        p = an.AllNeighborsParams(algo=algo, n_clusters=1)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        out = an.build(d_all, K, p)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        ind = out[0] if isinstance(out, (tuple, list)) else out
        r = recall_of(ind)
        print(f"{algo:>12} {dt:>9.3f} {r:>12.6f} {int(round((1-r)*SAMPLE*K)):>8,}", flush=True)
        del out, ind
    except Exception as e:
        print(f"{algo:>12} {'ERROR':>9}  {type(e).__name__}: {str(e)[:70]}", flush=True)
    cp.get_default_memory_pool().free_all_blocks()
