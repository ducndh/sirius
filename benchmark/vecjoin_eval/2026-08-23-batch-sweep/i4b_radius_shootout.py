"""What is actually the fastest widely-available radius (range) join? — ducndh, 2026-08-24.

I4 claimed our radius join is "38x faster than the only competitor". That compared against
FAISS-**CPU** because that is what happened to be installed, which is not a fair reading of "the
competition". The honest opponent is the thing a competent practitioner writes in ~25 lines: a
blocked brute-force distance scan on the GPU, with no index at all -- so no build, no recall
question, and no ceiling.

FAIRNESS RULES, chosen to favour nobody:
  * Every contender MATERIALISES the (probe, corpus) index pairs, as Sirius does -- not just a count.
  * The GPU contenders get the corpus resident on the device BEFORE the timer starts, because
    Sirius's corpus is pinned on the GPU before its timer starts too. (A first, cold run costs an
    extra ~1.0 s of H2D for the 512 MB corpus; that is reported separately, not folded in.)
  * Every contender is warmed once before being timed.
  * eps is PLAIN L2 (the Sirius/DuckDB convention). FAISS takes a SQUARED radius, so it is passed
    eps**2; sklearn and scipy take plain eps.
"""
import sys, time, json, numpy as np, pyarrow as pa, pyarrow.parquet as pq, cupy as cp

DATA = "/var/tmp/vj/data/parquet"
EPS = [float(x) for x in (sys.argv[1:] or [100.0, 150.0])]
SIRIUS = {100.0: 1.794, 150.0: float("nan")}          # measured; 150 is REFUSED (k<=1024 cap)
RES = []


def flat(t, dim=128):
    c = t.column("vec")
    if isinstance(c, pa.ChunkedArray):
        c = c.combine_chunks()
    return np.ascontiguousarray(np.asarray(c.values, dtype="float32").reshape(-1, dim))


base = flat(pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet"))
qry = flat(pq.read_table(f"{DATA}/sift-128-euclidean_query.parquet"))
print(f"corpus {len(base):,} x 128 ({base.nbytes/2**20:.0f} MB), probes {len(qry):,}, "
      f"eps {EPS} (plain L2)\n", flush=True)

d_b = cp.asarray(base)                 # resident before any timer, matching Sirius's pinned corpus
bn = (d_b ** 2).sum(1)
cp.cuda.Stream.null.synchronize()


def rec(name, eps, dt, pairs, note=""):
    RES.append(dict(name=name, eps=eps, seconds=dt, pairs=pairs, note=note))
    s = SIRIUS.get(eps, float("nan"))
    rel = "" if not (s == s) or not (dt == dt) else f"{s/dt:>7.2f}x"
    print(f"  {name:<36} {dt:>9.3f} s  pairs={pairs:>10,}  {rel:>8}  {note}", flush=True)


def cupy_radius(q, eps, qb=4096, cb=250_000):
    tau = eps * eps
    oq, oc = [], []
    for qs in range(0, len(q), qb):
        d_q = q[qs:qs + qb]
        qn = (d_q ** 2).sum(1, keepdims=True)
        for cs in range(0, len(base), cb):
            d = qn + bn[cs:cs + cb][None, :] - 2.0 * (d_q @ d_b[cs:cs + cb].T)
            i, j = cp.nonzero(d <= tau)
            if i.size:
                oq.append(i + qs); oc.append(j + cs)
            del d, i, j
    return cp.concatenate(oq), cp.concatenate(oc)


from cuvs.distance import pairwise_distance


def cuvs_radius(q, eps, qb=4096, cb=250_000):
    tau = eps * eps
    oq, oc = [], []
    for qs in range(0, len(q), qb):
        d_q = q[qs:qs + qb]
        for cs in range(0, len(base), cb):
            w = min(cb, len(base) - cs)
            out = cp.empty((d_q.shape[0], w), dtype=cp.float32)
            pairwise_distance(d_q, d_b[cs:cs + cb], out, metric="sqeuclidean")
            i, j = cp.nonzero(out <= tau)
            if i.size:
                oq.append(i + qs); oc.append(j + cs)
            del out, i, j
    return cp.concatenate(oq), cp.concatenate(oc)


d_q_all = cp.asarray(qry)
for eps in EPS:
    print(f"=== eps = {eps:g}   (Sirius: "
          f"{'REFUSES — k<=1024 cap' if SIRIUS[eps] != SIRIUS[eps] else f'{SIRIUS[eps]:.3f} s'})",
          flush=True)
    for name, fn in (("cupy blocked brute force (GPU)", cupy_radius),
                     ("cuVS pairwise_distance + mask (GPU)", cuvs_radius)):
        fn(d_q_all[:1024], eps); cp.cuda.Stream.null.synchronize()      # warm
        t0 = time.perf_counter()
        a, b_ = fn(d_q_all, eps)
        cp.cuda.Stream.null.synchronize()
        rec(name, eps, time.perf_counter() - t0, int(a.size), "no index, no build")
        del a, b_
        cp.get_default_memory_pool().free_all_blocks()

t0 = time.perf_counter(); _ = cp.asarray(base); cp.cuda.Stream.null.synchronize()
print(f"\n  (cold-start reference: H2D of the {base.nbytes/2**20:.0f} MB corpus = "
      f"{time.perf_counter()-t0:.3f} s, excluded above for every GPU contender)", flush=True)

sys.path.insert(0, "/var/tmp/vjtools/py")
import faiss
faiss.omp_set_num_threads(64)
fi = faiss.IndexFlatL2(128); fi.add(base)
quant = faiss.IndexFlatL2(128); ivf = faiss.IndexIVFFlat(quant, 128, 1024)
t0 = time.perf_counter(); ivf.train(base); ivf.add(base); ivf_build = time.perf_counter() - t0
ivf.nprobe = 64
print()
for eps in EPS:
    print(f"=== eps = {eps:g}  (CPU, 64 threads)", flush=True)
    fi.range_search(qry[:100], eps ** 2)
    t0 = time.perf_counter(); lims, D, I = fi.range_search(qry, eps ** 2)
    rec("FAISS-CPU IndexFlatL2", eps, time.perf_counter() - t0, len(I), "exact")
    ivf.range_search(qry[:100], eps ** 2)
    t0 = time.perf_counter(); lims, D, I = ivf.range_search(qry, eps ** 2)
    rec("FAISS-CPU IndexIVFFlat nprobe=64", eps, time.perf_counter() - t0, len(I),
        f"APPROX, +{ivf_build:.1f}s build")
    rec("FAISS-GPU GpuIndexFlatL2", eps, float("nan"), 0,
        "NOT IMPLEMENTED — range_search throws")

json.dump(RES, open("/var/tmp/vj/x1/radius_shootout.json", "w"), indent=1)
print("\nwrote /var/tmp/vj/x1/radius_shootout.json", flush=True)
