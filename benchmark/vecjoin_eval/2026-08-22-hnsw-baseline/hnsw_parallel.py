"""Parallel-CPU HNSW baseline: the same index, searched with all cores instead of one.

DuckDB's HNSW_INDEX_JOIN is single-threaded (user time tracks real time at every ef point), so
timing it against a whole GPU compares ONE CPU core to an A100. HNSW search is embarrassingly
parallel across probe rows, so the honest CPU number uses every core. FAISS is used because it
exposes thread count directly; index parameters mirror DuckDB/usearch defaults (M=16, ef_c=128)
so the single-thread point should land near DuckDB's measured ~4.5 s and validate the swap.

Reports both thread counts so the parallel speedup is measured, not assumed.
Deps: faiss-cpu, pyarrow, numpy. NOT pandas -- it is absent on this box.
"""
import time, numpy as np, pyarrow.parquet as pq, faiss

DATA = "/var/tmp/vj/data/parquet"
K, M, EFC = 10, 16, 128
EFS = [10, 20, 40, 64, 80, 160, 320]
THREADS = [64, 1]


def load(name):
    t = pq.read_table(f"{DATA}/sift-128-euclidean_{name}.parquet")
    return np.ascontiguousarray(
        np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype("float32"))


base, query = load("base"), load("query")
_gt = pq.read_table(f"{DATA}/sift-128-euclidean_gt.parquet")
_q, _r, _n = (_gt.column(c).to_numpy() for c in ("query_id", "rank", "neighbor_id"))
_m = _r < K
truth_arr = np.full((int(_q.max()) + 1, K), -1, dtype=np.int64)
truth_arr[_q[_m], _r[_m]] = _n[_m]
print(f"base {base.shape} query {query.shape}", flush=True)

faiss.omp_set_num_threads(64)
index = faiss.IndexHNSWFlat(base.shape[1], M, faiss.METRIC_L2)
index.hnsw.efConstruction = EFC
t0 = time.perf_counter()
index.add(base)
build = time.perf_counter() - t0
print(f"index build (64 threads): {build:.2f}s", flush=True)

print(f"{'threads':>8} {'ef':>5} {'time_s':>9} {'recall@10':>10} {'qps':>10}", flush=True)
for nt in THREADS:
    faiss.omp_set_num_threads(nt)
    for ef in EFS:
        index.hnsw.efSearch = ef
        index.search(query[:100], K)                      # warm
        t0 = time.perf_counter()
        _, I = index.search(query, K)
        dt = time.perf_counter() - t0
        hits = sum(len(set(truth_arr[q]) & set(I[q])) for q in range(len(query)))
        print(f"{nt:>8} {ef:>5} {dt:>9.3f} {hits/(len(query)*K):>10.4f} {len(query)/dt:>10.1f}",
              flush=True)
