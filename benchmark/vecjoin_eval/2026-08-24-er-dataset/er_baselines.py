"""ER vector join across every available engine — CORRECTNESS and EXPRESSIVENESS, not speed.

⚠️ This dataset's whole cross product is 4.4M pairs = 3.4 GFLOP = ~0.4 ms of A100 compute. Any
timing here is ~100% fixed cost and MUST NOT be quoted as a performance result. SIFT/GIST/GloVe
remain the scale datasets. What this harness measures instead, which nothing else in the project
does:

  1. Can the engine EXPRESS the query at all? ("all pairs above similarity tau, per left row")
  2. Does it get the RIGHT ANSWER, scored against labelled gold matches (P/R/F1)?

The query, in words: for each product in A, how many products in B are within similarity tau --
and which pairs are they. That is a threshold join, the shape ER practitioners actually use.

Engines here are the library ones (cuVS, FAISS-CPU, cupy reference). Sirius, DuckDB VSS, plain
DuckDB and pgvector are driven from er_sql.sh / er_pg.sh so each runs in its own process.
"""
import json, sys, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq
# NOTE: do NOT put /var/tmp/vjtools/py on sys.path here. It holds a faiss-GPU wheel whose
# libcudart.so.12 was clobbered by a later nvidia wheel upgrade; the working CPU faiss 1.15.0 is
# the one in ~/.local. Shadowing it costs an OSError at import.

OUT = "/var/tmp/vj/data/parquet"
DS = sys.argv[1] if len(sys.argv) > 1 else "amazon-google"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70      # cosine SIMILARITY floor
DIM = 384


def load(stem):
    t = pq.read_table(f"{OUT}/{stem}.parquet")
    c = t.column("vec")
    if isinstance(c, pa.ChunkedArray):
        c = c.combine_chunks()
    return (np.asarray(t.column("id"), dtype="int64"),
            np.ascontiguousarray(np.asarray(c.values, dtype="float32").reshape(-1, DIM)))


ia, A = load(f"{DS}_a")
ib, B = load(f"{DS}_b")
lab = pq.read_table(f"{OUT}/{DS}_labels.parquet")
ly = np.asarray(lab.column("label"), dtype="int8")
gold = set(zip(np.asarray(lab.column("left_id"), dtype="int64")[ly == 1].tolist(),
               np.asarray(lab.column("right_id"), dtype="int64")[ly == 1].tolist()))
print(f"{DS}: A={len(A):,} B={len(B):,} cross={len(A)*len(B):,} gold={len(gold):,} tau={TAU}")
print(f"⚠ cross product is {2*DIM*len(A)*len(B)/1e9:.2f} GFLOP — timings below are fixed-cost "
      f"dominated, do not quote them as performance\n")

RES = []


def score(name, pairs, seconds, expressed=True, note=""):
    if not expressed:
        print(f"  {name:<34} {'CANNOT EXPRESS':>14}   {note}")
        RES.append(dict(engine=name, expressed=False, note=note))
        return
    got = set(pairs)
    tp = len(got & gold)
    p = tp / max(len(got), 1)
    r = tp / max(len(gold), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"  {name:<34} {seconds:>8.3f}s  pairs={len(got):>7,}  P={p:.3f} R={r:.3f} F1={f1:.3f}"
          f"  {note}")
    RES.append(dict(engine=name, expressed=True, seconds=seconds, pairs=len(got),
                    precision=p, recall=r, f1=f1, note=note))


# ---- reference: cupy full cross product + threshold (also the "hand-written GPU" opponent) ----
import cupy as cp
d_a, d_b = cp.asarray(A), cp.asarray(B)
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
S = d_a @ d_b.T                       # both sides unit-norm -> cosine similarity
gi, gj = cp.nonzero(S >= TAU)
cp.cuda.Stream.null.synchronize()
dt = time.perf_counter() - t0
ref = list(zip(ia[cp.asnumpy(gi)].tolist(), ib[cp.asnumpy(gj)].tolist()))
score("cupy cross product + mask (GPU)", ref, dt, note="reference answer")
REF = set(ref)
del S, gi, gj
cp.get_default_memory_pool().free_all_blocks()

# ---- cuVS: no radius API, so the practitioner path is top-k with k inflated, then filter ----
from cuvs.neighbors import brute_force
idx = brute_force.build(d_b, metric="cosine")
for k in (10, 100, 1000):
    k_eff = min(k, len(B))
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    D, I = brute_force.search(idx, d_a, k_eff)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    Dh, Ih = cp.asnumpy(D), cp.asnumpy(I)
    keep = (1.0 - Dh) >= TAU          # cuVS returns cosine DISTANCE
    pairs = [(int(ia[r]), int(ib[Ih[r, c]]))
             for r, c in zip(*np.nonzero(keep)) if Ih[r, c] >= 0]
    truncated = int((keep[:, -1]).sum())
    note = (f"k={k_eff} INCOMPLETE: {truncated} rows hit the k bound"
            if truncated else f"k={k_eff} complete")
    score(f"cuVS brute_force top-k + filter", pairs, dt, note=note)
    del D, I
    cp.get_default_memory_pool().free_all_blocks()

# ---- FAISS-CPU: has a real range_search ----
import faiss
faiss.omp_set_num_threads(64)
fi = faiss.IndexFlatIP(DIM); fi.add(B)          # inner product on unit vectors == cosine
fi.range_search(A[:16], TAU)
t0 = time.perf_counter()
lims, Dr, Ir = fi.range_search(A, TAU)
dt = time.perf_counter() - t0
pairs = [(int(ia[r]), int(ib[Ir[j]]))
         for r in range(len(A)) for j in range(lims[r], lims[r + 1])]
score("FAISS-CPU IndexFlatIP.range_search", pairs, dt, note="real radius API, exact")

print(f"\nagreement with the cupy reference: "
      f"FAISS {len(set(pairs) & REF)}/{len(REF)}")
json.dump(RES, open(f"/var/tmp/vj/er_baselines_{DS}.json", "w"), indent=1)
