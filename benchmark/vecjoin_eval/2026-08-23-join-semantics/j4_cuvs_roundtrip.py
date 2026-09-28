"""J4 — what an ANN library pays to let SQL touch its output.

Sirius aggregates 10M join pairs for +1.6% over the raw join, because they never leave the GPU.
A library returns arrays; to run `GROUP BY` over them the pairs must cross device->host and enter
a database first. This measures that round trip.

Deliberately generous to the library, so the number is not a strawman:
  * the search itself is EXCLUDED -- only the round trip is timed,
  * results move device->host with `cp.asnumpy`, no Python-level row loop,
  * they enter DuckDB through Arrow zero-copy registration, not INSERT statements,
  * DuckDB gets the same memory_limit as the Sirius run.

Reported as the cost added on top of a search the library has already done.

Usage: j4_cuvs_roundtrip.py [--n-lists 1024] [--n-probes 16] [--k 10]
"""
import argparse, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq, cupy as cp, duckdb
from cuvs.neighbors import ivf_flat

DATA = "/var/tmp/vj/data/parquet"


def flat(col, dim=128):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lists", type=int, default=1024)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    base = flat(pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet").column("vec"))
    d_all = cp.asarray(base)
    idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_all)

    t0 = time.perf_counter()
    D, I = ivf_flat.search(ivf_flat.SearchParams(n_probes=a.n_probes), idx, d_all, a.k)
    cp.cuda.Stream.null.synchronize()
    search_s = time.perf_counter() - t0
    n_pairs = len(base) * a.k
    print(f"search (EXCLUDED from the round trip): {search_s:.3f} s for {n_pairs:,} pairs",
          flush=True)

    # --- device -> host
    t0 = time.perf_counter()
    I_h = cp.asnumpy(cp.asarray(I)).reshape(-1)
    D_h = cp.asnumpy(cp.asarray(D)).reshape(-1)
    left = np.repeat(np.arange(len(base), dtype=np.int64), a.k)
    cp.cuda.Stream.null.synchronize()
    d2h_s = time.perf_counter() - t0

    # --- into the database, via Arrow zero-copy
    t0 = time.perf_counter()
    tbl = pa.table({"left_id": pa.array(left),
                    "right_id": pa.array(I_h.astype(np.int64)),
                    "distance": pa.array(D_h.astype(np.float32))})
    con = duckdb.connect()
    con.execute("SET memory_limit='32GB'")
    con.execute("SET temp_directory='/var/tmp/ddb_spill'")
    con.register("pairs", tbl)
    con.execute("CREATE TABLE pairs_t AS SELECT * FROM pairs")
    ingest_s = time.perf_counter() - t0

    # --- the aggregate Sirius did inside the join
    t0 = time.perf_counter()
    con.execute("SELECT count(*), avg(distance), min(distance), max(distance) FROM pairs_t").fetchall()
    agg_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    con.execute("SELECT left_id, avg(distance) FROM pairs_t GROUP BY left_id").fetchall()
    grp_s = time.perf_counter() - t0

    rt = d2h_s + ingest_s
    print(f"\nround trip the library must pay before SQL can touch the pairs:", flush=True)
    print(f"  device->host            {d2h_s:8.3f} s", flush=True)
    print(f"  into DuckDB (Arrow)     {ingest_s:8.3f} s", flush=True)
    print(f"  ROUND TRIP SUBTOTAL     {rt:8.3f} s", flush=True)
    print(f"  then global aggregate   {agg_s:8.3f} s", flush=True)
    print(f"  then grouped aggregate  {grp_s:8.3f} s", flush=True)
    print(f"  TOTAL added over search {rt + grp_s:8.3f} s", flush=True)
    print(f"RESULT j4 cuvs {search_s:.3f} {d2h_s:.3f} {ingest_s:.3f} {agg_s:.3f} {grp_s:.3f} "
          f"{rt + grp_s:.3f}", flush=True)


if __name__ == "__main__":
    main()
