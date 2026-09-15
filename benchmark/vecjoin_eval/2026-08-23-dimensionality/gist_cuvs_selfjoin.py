"""cuVS on the GIST1M self-join (d=960, 1M x 1M) — the last cell of the regime x dimensionality grid.

CAGRA rather than IVF-Flat: S3 established CAGRA is cuVS's best index here (~15x IVF-Flat at
matched recall) once build_algo=nn_descent is set. Reporting IVF-Flat would understate the rival.

CAGRA cannot reach recall 1.0, so the exact comparison is stated as a capability difference rather
than forced into a matched-recall row.
"""
import time, numpy as np, pyarrow as pa, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import cagra
DATA = "/var/tmp/vj/data/parquet"
DIM = 960
t = pq.read_table(f"{DATA}/gist-960-euclidean_base.parquet").column("vec").combine_chunks()
base = np.asarray(t.values, dtype="float32").reshape(-1, DIM)
d_all = cp.asarray(base)
print(f"self-join {len(base):,} x {len(base):,} at d={DIM}", flush=True)
t0 = time.perf_counter()
idx = cagra.build(cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64,
                                    build_algo="nn_descent"), d_all)
cp.cuda.Stream.null.synchronize()
build = time.perf_counter() - t0
print(f"cagra build: {build:.2f} s", flush=True)
for itopk in (64, 128):
    cagra.search(cagra.SearchParams(itopk_size=itopk), idx, d_all[:1000], 10)
    t0 = time.perf_counter()
    _, I = cagra.search(cagra.SearchParams(itopk_size=itopk), idx, d_all, 10)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    print(f"  cagra itopk={itopk:<4} search {dt:8.3f} s  build+search {build+dt:8.3f} s", flush=True)
    del I
    cp.get_default_memory_pool().free_all_blocks()
