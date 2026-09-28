"""I4 — global top-k: verify Sirius's answer and time the library path that has to emulate it.

Neither cuVS nor FAISS has a global-top-k API (I3). The emulation is forced to search **k=N per
probe** -- all N best pairs can belong to one probe, so a smaller per-probe k is not sound -- and
then reduce N*n_probes candidates on the host. Both halves are timed.

Truth is a cupy running top-N over every pair, which is independent of both.
"""
import time, numpy as np, cupy as cp, pyarrow.parquet as pq, pyarrow.csv as pacsv
from cuvs.neighbors import brute_force
from x1_score import Scorer

# The probe side MUST be disjoint from the corpus here. A probe set drawn FROM the corpus makes
# every probe its own zero-distance match, so the global top-1000 is 1000 self-matches and the
# benchmark measures nothing about ranking. Caught 2026-08-23 after running exactly that.
N_GLOBAL, PROBES = 1000, 10_000
sc = Scorer(); base = sc.base
import pyarrow as pa
qc = pq.read_table("/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet").column("vec")
if isinstance(qc, pa.ChunkedArray):
    qc = qc.combine_chunks()
qry = np.asarray(qc.values, dtype="float32").reshape(-1, 128)
d_all = cp.asarray(base); d_q = cp.asarray(qry)
sn = (d_q ** 2).sum(1, keepdims=True)

# --- truth: running global top-N over all PROBES x 1M pairs ---
bd = cp.full(N_GLOBAL, cp.inf, dtype=cp.float32)
t0 = time.perf_counter()
for off in range(0, len(base), 100_000):
    blk = d_all[off:off + 100_000]
    d = (sn + (blk ** 2).sum(1)[None, :] - 2.0 * (d_q @ blk.T)).reshape(-1)
    part = cp.partition(d, N_GLOBAL)[:N_GLOBAL]
    bd = cp.sort(cp.concatenate([bd, part]))[:N_GLOBAL]
    del d, part
    cp.get_default_memory_pool().free_all_blocks()
truth = cp.asnumpy(bd)
print(f"truth: global top-{N_GLOBAL} over {PROBES:,} x {len(base):,} pairs in "
      f"{time.perf_counter()-t0:.2f} s; kth distance^2 = {truth[-1]:.4f}", flush=True)

# --- Sirius's answer ---
t = pacsv.read_csv("/var/tmp/vj/x1/i4_global_q10k_k1000.csv")
sd = np.sort(np.asarray(t.column("distance"), dtype="float64"))
print(f"sirius: {len(sd):,} pairs, distances {sd[0]:.4f} .. {sd[-1]:.4f} (plain L2)")
sd2 = sd ** 2
ok = np.allclose(sd2, np.sort(truth), rtol=1e-4, atol=1e-2)
print(f"        matches truth (as squared L2): {ok}; max abs diff {np.abs(sd2-np.sort(truth)).max():.4f}")

# --- library emulation: per-probe top-N then host reduction ---
idx = brute_force.build(d_all)
brute_force.search(idx, d_q[:100], N_GLOBAL)
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
D, I = brute_force.search(idx, d_q, N_GLOBAL)
cp.cuda.Stream.null.synchronize()
t_search = time.perf_counter() - t0
t0 = time.perf_counter()
flat = cp.asarray(D).reshape(-1)
sel = cp.sort(cp.partition(flat, N_GLOBAL)[:N_GLOBAL])
cp.cuda.Stream.null.synchronize()
t_red = time.perf_counter() - t0
lib = cp.asnumpy(sel)
print(f"\ncuVS brute_force k={N_GLOBAL} search: {t_search:.3f} s + reduction {t_red:.3f} s "
       f"= {t_search+t_red:.3f} s")
print(f"        matches truth: {np.allclose(lib, np.sort(truth), rtol=1e-4, atol=1e-2)}")
SIR = 1.794
print(f"\nSirius global top-{N_GLOBAL}, {PROBES:,} probes: {SIR:.3f} s  -> "
      f"{(t_search+t_red)/SIR:.2f}x  (>1 = Sirius faster)")
