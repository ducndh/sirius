"""cuVS side of J5: for each epoch's corpus (parquet written by churn.sql), rebuild each index and
search the same 10k packaged queries. Recall vs a cupy brute-force truth on that epoch's corpus."""
import glob, json, time, sys
import numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import brute_force, ivf_flat, cagra
import os
OUT = os.environ.get("OUT", "/var/tmp/vj/churn"); K = 10
def load(path):
    t = pq.read_table(path); ids = t.column("id").to_numpy()
    return ids, np.ascontiguousarray(np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype(np.float32))
def sync(): cp.cuda.Device().synchronize()
def timed(fn):
    sync(); t = time.perf_counter(); r = fn(); sync(); return time.perf_counter() - t, r
_, Q = load("/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet"); Qd = cp.asarray(Q)
rows = []
for path in sorted(glob.glob(f"{OUT}/base_e*.parquet")):
    e = int(path.split("_e")[-1].split(".")[0]); ids, B = load(path); Bd = cp.asarray(B)
    # truth: exact top-10 on this epoch's corpus
    tb, idx = timed(lambda: brute_force.build(Bd, metric="sqeuclidean"))
    ts, (d, n) = timed(lambda: brute_force.search(idx, Qd, 65))
    truth = cp.asnumpy(n)[:, :K]
    def rec(nb): return float(np.mean([len(set(truth[i]) & set(nb[i])) / K for i in range(len(truth))]))
    rows.append(dict(epoch=e, system="cuVS brute_force k=65 (exact)", build=tb, search=ts, recall=rec(cp.asnumpy(n)[:, :K])))
    tb, idx = timed(lambda: ivf_flat.build(ivf_flat.IndexParams(n_lists=64, metric="sqeuclidean"), Bd))
    ts, (d, n) = timed(lambda: ivf_flat.search(ivf_flat.SearchParams(n_probes=8), idx, Qd, K))
    rows.append(dict(epoch=e, system="cuVS IVF-Flat 64/8", build=tb, search=ts, recall=rec(cp.asnumpy(n))))
    tb, idx = timed(lambda: cagra.build(cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64, build_algo="nn_descent", metric="sqeuclidean"), Bd))
    ts, (d, n) = timed(lambda: cagra.search(cagra.SearchParams(itopk_size=64), idx, Qd, K))
    rows.append(dict(epoch=e, system="cuVS CAGRA nn_descent 32/64", build=tb, search=ts, recall=rec(cp.asnumpy(n))))
    for r in rows[-3:]: print(f"e{e} {r['system']:32s} build {r['build']:7.3f} search {r['search']:7.3f} total {r['build']+r['search']:7.3f} recall {r['recall']:.4f}", flush=True)
    del Bd, idx; cp.get_default_memory_pool().free_all_blocks()
json.dump(rows, open(f"{OUT}/cuvs_churn.json", "w"), indent=1)
