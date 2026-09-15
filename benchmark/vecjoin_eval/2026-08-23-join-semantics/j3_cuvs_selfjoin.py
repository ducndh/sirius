"""J3 — cuVS on the 1M x 1M self-join, the many-to-many regime the clustered join is built for.

This is the regime where the recorded Sirius figure (6.64 s @ 0.873) sat closest to cuVS
(5.73 s @ 0.869) — but those were measured on different boxes, so the comparison carried a ⚠ on
the single result that looks best for us. This puts both on one machine.

A self-join is not a search workload: 1M probes over 1024 clusters means a probe batch spans a
handful of clusters and pruning can actually skip, whereas 10k probes span nearly all of them.
That is a property of the design, which is why the two regimes disagree.

Recall is reported two ways so the oracle is not self-referential:
  * against cuVS's OWN exhaustive run (n_probes = n_lists), and
  * that exhaustive run is itself checked against a cupy brute-force sample, so the reference is
    not "cuVS agrees with cuVS".

Build time is reported separately; for a one-shot join it belongs in the total (see
../../../ref_vector_join_vs_search_framing.md).

Usage: j3_cuvs_selfjoin.py [--n-lists 1024] [--k 10] [--sample 2000]
"""
import argparse, time, numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import ivf_flat

DATA = "/var/tmp/vj/data/parquet"


def flat(col, dim=128):
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lists", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=2000,
                    help="probe rows to verify the exhaustive reference against brute force")
    ap.add_argument("--probes", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    a = ap.parse_args()

    base = flat(pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet").column("vec"))
    print(f"self-join {len(base):,} x {len(base):,}, k={a.k}, n_lists={a.n_lists}", flush=True)
    d_all = cp.asarray(base)

    t0 = time.perf_counter()
    idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_all)
    cp.cuda.Stream.null.synchronize()
    build = time.perf_counter() - t0
    print(f"index build: {build:.3f} s", flush=True)

    # Exhaustive reference: every list probed. Verified against cupy brute force on a sample so
    # the reference is not cuVS checking itself.
    _, I_ref = ivf_flat.search(ivf_flat.SearchParams(n_probes=a.n_lists), idx, d_all, a.k)
    ref = cp.asnumpy(I_ref)
    smp = np.random.default_rng(0).choice(len(base), size=min(a.sample, len(base)), replace=False)
    d_s = cp.asarray(base[smp]); sn = (d_s ** 2).sum(1, keepdims=True)
    bd = cp.full((len(smp), a.k), cp.inf, dtype=cp.float32)
    bi = cp.full((len(smp), a.k), -1, dtype=cp.int64)
    for off in range(0, len(base), 200_000):
        blk = d_all[off:off + 200_000]
        dist = sn + (blk ** 2).sum(1)[None, :] - 2.0 * (d_s @ blk.T)
        ii = cp.argpartition(dist, a.k, axis=1)[:, :a.k]
        cd = cp.take_along_axis(dist, ii, 1); ci = ii.astype(cp.int64) + off
        md = cp.concatenate([bd, cd], 1); mi = cp.concatenate([bi, ci], 1)
        o = cp.argsort(md, axis=1)[:, :a.k]
        bd = cp.take_along_axis(md, o, 1); bi = cp.take_along_axis(mi, o, 1)
        del dist, ii, cd, ci, md, mi
        cp.get_default_memory_pool().free_all_blocks()
    truth_s = cp.asnumpy(bi)
    agree = sum(len(set(truth_s[i]) & set(ref[smp[i]])) for i in range(len(smp))) / (len(smp) * a.k)
    print(f"exhaustive reference vs brute force on {len(smp)} sampled rows: {agree:.4f} "
          f"({'ok' if agree > 0.99 else 'REFERENCE SUSPECT'})", flush=True)

    print(f"\n{'n_probes':>9} {'search_s':>10} {'total_s':>9} {'recall@10':>10}", flush=True)
    for np_ in a.probes:
        ivf_flat.search(ivf_flat.SearchParams(n_probes=np_), idx, d_all[:1000], a.k)   # warm
        t0 = time.perf_counter()
        _, I = ivf_flat.search(ivf_flat.SearchParams(n_probes=np_), idx, d_all, a.k)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        Ih = cp.asnumpy(I)
        rec = sum(len(set(ref[i]) & set(Ih[i])) for i in range(0, len(base), 10)) / \
              (len(range(0, len(base), 10)) * a.k)     # every 10th row: 100k rows is plenty
        print(f"{np_:>9} {dt:>10.3f} {build + dt:>9.3f} {rec:>10.4f}", flush=True)
        del I, Ih
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
