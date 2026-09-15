"""J1 — the canonical join case: the corpus is an INTERMEDIATE RESULT, so no index can pre-exist.

Every benchmark before this handed the ANN library a curated corpus it had indexed in advance.
That is vector search. In a join the right side is whatever the plan produces — here, the rows of
SIFT1M surviving a predicate. An index over the *unfiltered* corpus does not answer this query,
and an index over the filtered corpus cannot exist until the filter has run.

This script measures the PRACTITIONER PATH, which is what someone does today:

    DuckDB filter  ->  export surviving rows to host  ->  build cuVS IVF  ->  search
                   ->  map library-local row ids back to database ids

Every stage is timed, because in a join every stage is real work. The Sirius side is
`j1_sirius.sql`, which expresses the same query as one statement.

The oracle is exact brute force on the FILTERED subset, computed here on GPU, so recall is
measured against the right answer for this query and not against the unfiltered corpus.

Usage: j1_intermediate_corpus.py [--selectivity 0.3] [--n-lists 1024] [--n-probes 16] [--k 10]
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
    ap.add_argument("--selectivity", type=float, default=0.3)
    ap.add_argument("--n-lists", type=int, default=1024)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    base = flat(pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet").column("vec"))
    query = flat(pq.read_table(f"{DATA}/sift-128-euclidean_query.parquet").column("vec"))

    # The predicate. `id % 10 < N` is deterministic and trivially expressible in SQL, so the
    # Sirius side filters the identical rows -- the two systems must answer the SAME query.
    n_keep = int(round(a.selectivity * 10))
    ids = np.arange(len(base), dtype=np.int64)
    mask = (ids % 10) < n_keep
    kept_ids = ids[mask]
    print(f"predicate: id % 10 < {n_keep}  ->  {mask.sum():,} of {len(base):,} rows "
          f"({mask.mean()*100:.1f}%)", flush=True)

    # ---- oracle: exact top-k within the FILTERED subset
    d_q = cp.asarray(query)
    d_qn = (d_q ** 2).sum(1, keepdims=True)
    best_d = cp.full((len(query), a.k), cp.inf, dtype=cp.float32)
    best_i = cp.full((len(query), a.k), -1, dtype=cp.int64)
    TILE, CH = 1000, 200_000
    kept = base[mask]
    for off in range(0, len(kept), CH):
        blk = cp.asarray(kept[off:off + CH])
        bn = (blk ** 2).sum(1)[None, :]
        for lo in range(0, len(query), TILE):
            hi = min(lo + TILE, len(query))
            dist = d_qn[lo:hi] + bn - 2.0 * (d_q[lo:hi] @ blk.T)
            idx = cp.argpartition(dist, a.k, axis=1)[:, :a.k]
            cd = cp.take_along_axis(dist, idx, 1)
            ci = idx.astype(cp.int64) + off
            md = cp.concatenate([best_d[lo:hi], cd], 1)
            mi = cp.concatenate([best_i[lo:hi], ci], 1)
            o = cp.argsort(md, axis=1)[:, :a.k]
            best_d[lo:hi] = cp.take_along_axis(md, o, 1)
            best_i[lo:hi] = cp.take_along_axis(mi, o, 1)
            del dist, idx, cd, ci, md, mi
            cp.get_default_memory_pool().free_all_blocks()
        del blk, bn
        cp.get_default_memory_pool().free_all_blocks()
    truth = cp.asnumpy(best_i)          # positions within `kept`
    del d_q, d_qn, best_d, best_i
    cp.get_default_memory_pool().free_all_blocks()
    print("oracle: exact top-k over the filtered subset computed", flush=True)

    # ---- practitioner path, every stage timed
    t0 = time.perf_counter()
    subset = base[mask]                                   # "export the surviving rows"
    export_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_sub = cp.asarray(subset)
    d_probe = cp.asarray(query)
    cp.cuda.Stream.null.synchronize()
    h2d_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_sub)
    cp.cuda.Stream.null.synchronize()
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, I = ivf_flat.search(ivf_flat.SearchParams(n_probes=a.n_probes), idx, d_probe, a.k)
    cp.cuda.Stream.null.synchronize()
    search_s = time.perf_counter() - t0

    # map library-local positions back to database ids -- the "join back" a library cannot do
    t0 = time.perf_counter()
    _ = kept_ids[cp.asnumpy(I)]
    mapback_s = time.perf_counter() - t0

    Ih = cp.asnumpy(I)
    rec = sum(len(set(truth[i]) & set(Ih[i])) for i in range(len(query))) / (len(query) * a.k)
    total = export_s + h2d_s + build_s + search_s + mapback_s
    print(f"\npractitioner path (cuVS), selectivity {a.selectivity:.0%}:", flush=True)
    for name, v in (("export filtered rows", export_s), ("host->device", h2d_s),
                    ("index build", build_s), ("search", search_s),
                    ("map ids back", mapback_s)):
        print(f"  {name:22s} {v:7.3f} s", flush=True)
    print(f"  {'TOTAL':22s} {total:7.3f} s   recall@{a.k} {rec:.4f}", flush=True)
    print(f"RESULT j1 cuvs {a.selectivity} {export_s:.3f} {h2d_s:.3f} {build_s:.3f} "
          f"{search_s:.3f} {mapback_s:.3f} {total:.3f} {rec:.4f}", flush=True)


if __name__ == "__main__":
    main()
