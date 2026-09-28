"""I4 — the radius baseline on the SAME probe set Sirius was timed on (SIFT's 10k query set)."""
import time, numpy as np, faiss, pyarrow as pa, pyarrow.parquet as pq
from x1_score import load_base

_, base = load_base()
qc = pq.read_table("/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet").column("vec")
if isinstance(qc, pa.ChunkedArray):
    qc = qc.combine_chunks()
q = np.ascontiguousarray(np.asarray(qc.values, dtype="float32").reshape(-1, 128))
faiss.omp_set_num_threads(64)
idx = faiss.IndexFlatL2(128); idx.add(base)
print(f"FAISS IndexFlatL2, {faiss.omp_get_max_threads()} threads, probes {len(q):,}")
for eps in (100.0, 150.0):
    idx.range_search(q[:100], eps ** 2)
    t0 = time.perf_counter(); lims, D, I = idx.range_search(q, eps ** 2)
    dt = time.perf_counter() - t0
    cnt = np.diff(lims)
    print(f"  eps={eps:>6.0f}  {dt:>8.3f} s  pairs={len(I):>10,}  mean/probe={cnt.mean():>7.2f}  "
          f"max/probe={cnt.max():>6,}")
