"""GPU ANN baseline (cuVS) on the same A100, same workload as the HNSW baseline.

This is the TIER-1 comparison: for a GPU approximate join, a GPU ANN library is the real
competitor. A CPU index is a reference point, not the opponent. Prior cuVS figures in this
project were the 1M x 1M SELF-JOIN on another box and must not be paired with anything here.

Workload: SIFT1M, 10k probes x 1M corpus, k=10, L2, recall@10 vs the packaged ground truth --
identical to hnsw_parallel.py and the DuckDB HNSW runs, so all three sit on one recall axis.

Reports build time separately from search time. For a one-shot JOIN the build arguably belongs
inside the measurement; state the choice rather than burying it.

Deps: cuvs-cu12, cupy-cuda12x, pyarrow, numpy.
"""
import time, numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import cagra, ivf_flat

DATA = "/var/tmp/vj/data/parquet"
K = 10


def load(name):
    t = pq.read_table(f"{DATA}/sift-128-euclidean_{name}.parquet")
    return np.ascontiguousarray(
        np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype("float32"))


base, query = load("base"), load("query")
_gt = pq.read_table(f"{DATA}/sift-128-euclidean_gt.parquet")
_q, _r, _n = (_gt.column(c).to_numpy() for c in ("query_id", "rank", "neighbor_id"))
_m = _r < K
truth = np.full((int(_q.max()) + 1, K), -1, dtype=np.int64)
truth[_q[_m], _r[_m]] = _n[_m]
print(f"base {base.shape} query {query.shape}", flush=True)

d_base, d_query = cp.asarray(base), cp.asarray(query)


def recall(I):
    I = cp.asnumpy(I)
    return sum(len(set(truth[q]) & set(I[q])) for q in range(len(query))) / (len(query) * K)


def timed(fn):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    out = fn()
    cp.cuda.Stream.null.synchronize()
    return out, time.perf_counter() - t0


print(f"{'index':>10} {'param':>16} {'build_s':>9} {'search_s':>9} {'recall@10':>10} {'qps':>10}",
      flush=True)

# --- CAGRA: the GPU-native graph index cuVS is built around
# build_algo MUST be nn_descent. The default (ivf_pq) produces a degenerate graph here --
# "Self-included ratio is low: 0.01%" and recall ~3e-4 -- and on this cuVS build it also
# crashes the process outright. Diagnosed 2026-08-23 (QUEUE S3).
idx, build = timed(lambda: cagra.build(
    cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64, build_algo="nn_descent"),
    d_base))
for itopk in (32, 64, 128, 256):
    sp = cagra.SearchParams(itopk_size=itopk)
    cagra.search(sp, idx, d_query[:100], K)                       # warm
    (_, I), dt = timed(lambda: cagra.search(sp, idx, d_query, K))
    print(f"{'cagra':>10} {'itopk=' + str(itopk):>16} {build:>9.2f} {dt:>9.4f} "
          f"{recall(I):>10.4f} {len(query)/dt:>10.1f}", flush=True)

# --- IVF-Flat: the direct analogue of our clustered approximate join (probe n lists)
idx, build = timed(lambda: ivf_flat.build(ivf_flat.IndexParams(n_lists=1024), d_base))
for nprobe in (1, 4, 16, 64, 256):
    sp = ivf_flat.SearchParams(n_probes=nprobe)
    ivf_flat.search(sp, idx, d_query[:100], K)                    # warm
    (_, I), dt = timed(lambda: ivf_flat.search(sp, idx, d_query, K))
    print(f"{'ivf_flat':>10} {'n_probes=' + str(nprobe):>16} {build:>9.2f} {dt:>9.4f} "
          f"{recall(I):>10.4f} {len(query)/dt:>10.1f}", flush=True)
