"""cuVS on the 1M x 1M self-join: every base vector is a query. Recall on the first 10k queries
against a cuVS brute-force truth for those rows. Kernel-only times with device sync."""
import time, json, numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import brute_force, ivf_flat, cagra
K=10
t=pq.read_table("/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet")
B=np.ascontiguousarray(np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype(np.float32)); Bd=cp.asarray(B)
def sync(): cp.cuda.Device().synchronize()
def timed(fn):
    sync(); t0=time.perf_counter(); r=fn(); sync(); return time.perf_counter()-t0, r
_, bf=timed(lambda: brute_force.build(Bd, metric="sqeuclidean"))
_, (d,n)=timed(lambda: brute_force.search(bf, Bd[:10000], 65)); truth=cp.asnumpy(n)[:, :K]
def rec(nb): return float(np.mean([len(set(truth[i]) & set(nb[i]))/K for i in range(len(truth))]))
rows=[]
def run(name, build, search):
    tb, idx = timed(build); ts, (d, n) = timed(lambda: search(idx))
    r=dict(system=name, build=tb, search=ts, total=tb+ts, recall=rec(cp.asnumpy(n)[:10000]))
    rows.append(r); print(f"{name:34s} build {tb:7.3f} search {ts:7.3f} total {tb+ts:7.3f} recall {r['recall']:.4f}", flush=True)
    del idx, d, n; cp.get_default_memory_pool().free_all_blocks()
run("cuVS brute_force k=65 (exact)", lambda: brute_force.build(Bd, metric="sqeuclidean"), lambda i: brute_force.search(i, Bd, 65))
for nl, probes in ((64,(4,8,16,32)),(256,(8,16,32,64))):
    for p in probes:
        run(f"cuVS IVF-Flat {nl}/{p}", lambda: ivf_flat.build(ivf_flat.IndexParams(n_lists=nl, metric="sqeuclidean"), Bd), lambda i,p=p: ivf_flat.search(ivf_flat.SearchParams(n_probes=p), i, Bd, K))
for deg, idg, itopk in ((32,64,64),(32,64,128),(64,128,128),(64,128,256)):
    run(f"cuVS CAGRA nn_descent {deg}/{itopk}", lambda: cagra.build(cagra.IndexParams(graph_degree=deg, intermediate_graph_degree=idg, build_algo="nn_descent", metric="sqeuclidean"), Bd), lambda i,itopk=itopk: cagra.search(cagra.SearchParams(itopk_size=itopk), i, Bd, K))
json.dump(rows, open("/var/tmp/vj/m2m/cuvs_m2m.json","w"), indent=1)
