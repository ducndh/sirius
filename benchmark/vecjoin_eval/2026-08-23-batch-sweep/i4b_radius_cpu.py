"""Radius-join contenders that are NOT FAISS: sklearn, scipy, and plain SQL in DuckDB.

Run at 1,000 probes (not 10,000) because the exact CPU methods are ~100x slower than the GPU ones
and a full run would take an hour. Per-probe cost is what is compared; the 10k-probe column is a
LINEAR SCALE-UP and is labelled as such, never presented as measured.
"""
import sys, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, "/var/tmp/vjtools/py")

DATA = "/var/tmp/vj/data/parquet"
EPS = 100.0
NQ = 1000


def flat(t, dim=128):
    c = t.column("vec")
    if isinstance(c, pa.ChunkedArray):
        c = c.combine_chunks()
    return np.ascontiguousarray(np.asarray(c.values, dtype="float32").reshape(-1, dim))


base = flat(pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet"))
qry = flat(pq.read_table(f"{DATA}/sift-128-euclidean_query.parquet"))[:NQ]
print(f"corpus {len(base):,}, probes {len(qry):,}, eps={EPS:g} plain L2\n")
print(f"{'method':<38} {'build_s':>9} {'query_s':>9} {'pairs':>9}  {'->10k probes':>13}")


def rec(name, build, q, pairs, note=""):
    print(f"{name:<38} {build:>9.2f} {q:>9.3f} {pairs:>9,}  {q*10:>12.1f}s  {note}", flush=True)


from sklearn.neighbors import NearestNeighbors
for algo in ("brute", "kd_tree"):
    try:
        nn = NearestNeighbors(radius=EPS, algorithm=algo, n_jobs=-1)
        t0 = time.perf_counter(); nn.fit(base); b = time.perf_counter() - t0
        t0 = time.perf_counter(); ind = nn.radius_neighbors(qry, return_distance=False)
        q = time.perf_counter() - t0
        rec(f"sklearn radius_neighbors ({algo})", b, q, sum(len(x) for x in ind))
    except Exception as e:
        print(f"{'sklearn ' + algo:<38} ERROR {type(e).__name__}: {str(e)[:50]}", flush=True)

from scipy.spatial import cKDTree
t0 = time.perf_counter(); tree = cKDTree(base); b = time.perf_counter() - t0
t0 = time.perf_counter(); res = tree.query_ball_point(qry, EPS, workers=-1)
q = time.perf_counter() - t0
rec("scipy cKDTree query_ball_point", b, q, sum(len(x) for x in res),
    "KD-trees degenerate at d=128")
