"""cuVS practitioner path for J1 at four selectivities: DuckDB filter -> export -> h2d -> brute-force
(k=65 trimmed) -> ids mapped back. Kernel + data movement timed per stage; min of 3."""
import time, json, numpy as np, duckdb, cupy as cp
from cuvs.neighbors import brute_force
def sync(): cp.cuda.Device().synchronize()
con=duckdb.connect(); con.execute("SET memory_limit='32GB'")
con.execute("CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet')")
Q=np.stack(con.execute("SELECT vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet') ORDER BY id").fetchnumpy()["vec"]); Qd=cp.asarray(Q); sync()
out={}
for sel in (10,30,50,90):
    best=None
    for i in range(3):
        t={}; t0=time.perf_counter()
        r=con.execute(f"SELECT id, vec FROM base WHERE id % 100 < {sel} ORDER BY id").fetchnumpy(); ids=r["id"]; B=np.stack(r["vec"])
        t["export"]=time.perf_counter()-t0; t0=time.perf_counter()
        Bd=cp.asarray(B); sync(); t["h2d"]=time.perf_counter()-t0; t0=time.perf_counter()
        idx=brute_force.build(Bd, metric="sqeuclidean"); sync(); t["build"]=time.perf_counter()-t0; t0=time.perf_counter()
        d,n=brute_force.search(idx, Qd, 65); sync(); t["search"]=time.perf_counter()-t0; t0=time.perf_counter()
        mapped=ids[cp.asnumpy(n)[:, :10]]; t["map_ids"]=time.perf_counter()-t0
        t["total"]=sum(t.values())
        if best is None or t["total"]<best["total"]: best=t
        del Bd, idx, d, n; cp.get_default_memory_pool().free_all_blocks()
    out[sel]=best; print(sel, {k:round(v,3) for k,v in best.items()}, flush=True)
json.dump(out, open("/var/tmp/vj/j1view/cuvs_j1.json","w"), indent=1)
