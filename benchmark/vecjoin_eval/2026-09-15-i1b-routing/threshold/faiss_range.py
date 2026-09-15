"""FAISS-CPU range_search on the same workload, 64 threads, the only radius API among the baselines."""
import time, json, numpy as np, pyarrow.parquet as pq
try:
    import faiss
except ImportError:
    raise SystemExit("faiss not installed on this box")
faiss.omp_set_num_threads(64)
def load(n):
    t=pq.read_table(f"/var/tmp/vj/data/parquet/sift-128-euclidean_{n}.parquet"); return np.ascontiguousarray(np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype(np.float32))
B=load("base"); Q=load("query"); idx=faiss.IndexFlatL2(128); idx.add(B)
out={}
for eps in (150,200,250,300,350):
    best=None
    for i in range(3):
        t0=time.perf_counter(); lims,D,I=idx.range_search(Q, float(eps)**2); dt=time.perf_counter()-t0
        best=dt if best is None else min(best,dt)
    out[eps]=dict(time=best, pairs=int(lims[-1])); print(eps, out[eps], flush=True)
json.dump(out, open("/var/tmp/vj/threshold/faiss_range.json","w"), indent=1)
