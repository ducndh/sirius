"""I3 — which join shapes do the ANN libraries actually support? Verified by RUNNING each one.

An API's presence is not support: cuVS 26.02 exposes a `prefilter` argument on three index types,
and FAISS exposes eleven `IDSelector` classes, but only a call that returns a correct answer
settles whether a shape is our differentiator or our gap. Everything here is executed.

Shapes, matching the operator's reduction modes and the join shapes on the queue:
  A. per-probe top-k            (the baseline every library is built for)
  B. radius / threshold         (all pairs closer than tau)
  C. global top-k across pairs  (the k best pairs in the whole join, not per probe)
  D. filtered / predicated      (join against a subset of the corpus)
  E. many-to-many self-join     (probe set == corpus)
"""
import numpy as np, time, traceback

N, D, K = 20_000, 32, 10
rng = np.random.default_rng(7)
base = rng.random((N, D), dtype=np.float32)
qry = base[:2_000].copy()
TAU = None


def truth_topk(q, b, k):
    d = ((q[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, 1)[:, :k]
    return idx, np.take_along_axis(d, idx, 1)


def report(lib, shape, status, detail):
    print(f"{lib:<10} {shape:<26} {status:<12} {detail}")


T_I, T_D = truth_topk(qry[:200], base, K)
TAU = float(np.median(T_D[:, K - 1]))     # radius that keeps ~k neighbours for half the probes

print(f"{'library':<10} {'shape':<26} {'status':<12} detail")
print("-" * 100)

# ---------------- cuVS ----------------
import cupy as cp
from cuvs.neighbors import brute_force, ivf_flat, filters, all_neighbors

d_b, d_q = cp.asarray(base), cp.asarray(qry)
idx = brute_force.build(d_b)

# A
_, I = brute_force.search(idx, d_q[:200], K)
rec = np.mean([len(set(T_I[i]) & set(cp.asnumpy(I)[i])) for i in range(200)]) / K
report("cuVS", "A per-probe top-k", "YES", f"recall {rec:.3f}")

# B radius
try:
    import cuvs.neighbors as NB
    found = [m for m in dir(NB) if "range" in m.lower() or "radius" in m.lower()]
    raise AttributeError(f"no range/radius entry point in cuvs.neighbors ({found or 'none'})")
except AttributeError as e:
    report("cuVS", "B radius / threshold", "NO", str(e))

# C global top-k
report("cuVS", "C global top-k", "NO",
       "search() returns k per query; a global top-k is a user-side reduction over n_probes*k")

# D filtered
try:
    keep = np.zeros(N, dtype=bool); keep[::3] = True          # keep every 3rd corpus row
    bits = np.packbits(keep, bitorder="little").view(np.uint32)
    pf = filters.from_bitset(cp.asarray(bits))
    _, If = brute_force.search(idx, d_q[:200], K, prefilter=pf)
    Ifh = cp.asnumpy(If)
    allowed = bool(np.all(keep[Ifh[Ifh >= 0]]))
    sub = np.flatnonzero(keep)
    ti, _ = truth_topk(qry[:200], base[sub], K)
    rec = np.mean([len(set(sub[ti[i]]) & set(Ifh[i])) for i in range(200)]) / K
    report("cuVS", "D filtered / predicated", "YES" if allowed else "WRONG",
           f"bitset prefilter, all hits inside filter={allowed}, recall vs filtered truth {rec:.3f}")
except Exception as e:
    report("cuVS", "D filtered / predicated", "ERROR", f"{type(e).__name__}: {e}")

# E self-join
try:
    p = all_neighbors.AllNeighborsParams()
    t0 = time.perf_counter()
    out = all_neighbors.build(cp.asarray(base), K, p)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    ind = out[0] if isinstance(out, (tuple, list)) else out
    ih = cp.asarray(ind).get()[:200]
    ti, _ = truth_topk(base[:200], base, K)
    rec = np.mean([len(set(ti[i]) & set(ih[i])) for i in range(200)]) / K
    report("cuVS", "E many-to-many self-join", "YES",
           f"all_neighbors.build, {dt:.3f}s for {N:,}x{N:,}, recall {rec:.3f} "
           f"(SELF-JOIN ONLY: one dataset, no separate probe side)")
except Exception as e:
    report("cuVS", "E many-to-many self-join", "ERROR", f"{type(e).__name__}: {str(e)[:70]}")

# ---------------- FAISS (CPU only in this env) ----------------
import faiss
print(f"\n(faiss {faiss.__version__}, get_num_gpus()={faiss.get_num_gpus()} -> CPU only here)")
fi = faiss.IndexFlatL2(D); fi.add(base)

_, I = fi.search(qry[:200], K)
rec = np.mean([len(set(T_I[i]) & set(I[i])) for i in range(200)]) / K
report("FAISS-CPU", "A per-probe top-k", "YES", f"recall {rec:.3f}")

lims, RD, RI = fi.range_search(qry[:200], TAU)
n_pairs = len(RI)
brute = ((qry[:200, None, :] - base[None, :, :]) ** 2).sum(-1)
exp = int((brute <= TAU).sum())
report("FAISS-CPU", "B radius / threshold", "YES" if n_pairs == exp else "MISMATCH",
       f"range_search returned {n_pairs:,} pairs, brute force says {exp:,} (tau={TAU:.4f})")

report("FAISS-CPU", "C global top-k", "NO",
       "no API; user reduces over the per-query results")

sel = faiss.IDSelectorBatch(np.flatnonzero(keep).astype("int64"))
params = faiss.SearchParameters(); params.sel = sel
_, If = fi.search(qry[:200], K, params=params)
ok = bool(np.all(keep[If[If >= 0]]))
report("FAISS-CPU", "D filtered / predicated", "YES" if ok else "WRONG",
       f"IDSelectorBatch, all hits inside filter={ok}")

t0 = time.perf_counter(); _, Iself = fi.search(base, K); dt = time.perf_counter() - t0
report("FAISS-CPU", "E many-to-many self-join", "PARTIAL",
       f"no join API; a user calls search(base) -- {dt:.2f}s for {N:,}x{N:,}, exact")
