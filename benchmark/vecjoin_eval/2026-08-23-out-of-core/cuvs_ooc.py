"""cuVS past the device limit: what a user must actually do when the corpus does not fit.

The out-of-core claim is that Sirius streams both sides while a GPU ANN library needs the corpus
resident. "Needs it resident" is only a fair criticism if we show what the alternative costs, so
this measures BOTH:

  direct  -- hand cuVS the whole corpus and see where it fails (the honest failure point)
  sharded -- split into device-sized shards, build an index per shard, search each, merge top-k.
             This is what a competent user would write, and it is a legitimate baseline. It pays
             repeated index builds and one host->device transfer of the entire corpus per query
             batch, which is exactly the cost Sirius's streaming fold is meant to avoid.

Total wall time INCLUDES index build, because at these sizes the index cannot be kept resident
between queries -- so for a one-shot join the build is not amortizable. Search-only time is
reported alongside so the two accountings are both visible.

Usage: cuvs_ooc.py <gib> [--shard-gib 8] [--n-probes 16] [--n-lists 4096] [--k 10]
"""
import argparse, time, gc, numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import ivf_flat

OOC = "/var/tmp/vj/ooc"
BYTES_PER_VEC = 128 * 4


def flat(col, dim=128):
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def free():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def merge_topk(best_d, best_i, cand_d, cand_i, k):
    if best_d is None:
        return cand_d[:, :k], cand_i[:, :k]
    md = cp.concatenate([best_d, cand_d], 1)
    mi = cp.concatenate([best_i, cand_i], 1)
    order = cp.argsort(md, axis=1)[:, :k]
    return cp.take_along_axis(md, order, 1), cp.take_along_axis(mi, order, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gib", type=float)
    ap.add_argument("--shard-gib", type=float, default=8.0)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--n-lists", type=int, default=4096)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    tag = f"{a.gib:g}gib"

    probe = flat(pq.read_table(f"{OOC}/probe_{tag}.parquet").column("vec"))
    d_probe = cp.asarray(probe)
    gt = pq.read_table(f"{OOC}/gt_{tag}.parquet")
    q, r, nid = (gt.column(c).to_numpy() for c in ("query_id", "rank", "neighbor_id"))
    m = r < a.k
    truth = np.full((int(q.max()) + 1, a.k), -1, dtype=np.int64)
    truth[q[m], r[m]] = nid[m]

    total, free_mem = cp.cuda.Device().mem_info[1], cp.cuda.Device().mem_info[0]
    print(f"corpus {a.gib:g} GiB | device {total/2**30:.1f} GiB total, "
          f"{free_mem/2**30:.1f} GiB free | k={a.k} n_probes={a.n_probes}", flush=True)

    # --- direct: does the whole corpus even fit ON THE DEVICE?
    #
    # Streams parquet batches straight into a preallocated device array. An earlier version built
    # a host list and then np.concatenate'd it, holding the corpus in host RAM TWICE -- 96 GiB
    # for the 48 GiB case on a 120 GB box. That almost certainly took the machine down mid-run.
    # The question here is device capacity; it must not be answered by exhausting the host.
    direct_note = ""
    d_all = None
    try:
        pf0 = pq.ParquetFile(f"{OOC}/corpus_{tag}.parquet")
        n_rows = pf0.metadata.num_rows
        t0 = time.perf_counter()
        d_all = cp.empty((n_rows, 128), dtype=cp.float32)   # fails here if the device is too small
        at = 0
        for b in pf0.iter_batches(batch_size=1_000_000, columns=["vec"]):
            blk = flat(b.column("vec"))
            d_all[at:at + len(blk)] = cp.asarray(blk)
            at += len(blk)
            del blk
        idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_all)
        direct_note = f"fits: h2d+build {time.perf_counter()-t0:.1f}s"
        del idx
    except (cp.cuda.memory.OutOfMemoryError, MemoryError, RuntimeError) as e:
        direct_note = f"does NOT fit: {type(e).__name__}"
    finally:
        del d_all
        free()
    print(f"  direct (whole corpus resident on device): {direct_note}", flush=True)

    # --- sharded: what a user must write once the corpus exceeds the device
    rows_per_shard = int(a.shard_gib * 2**30 / BYTES_PER_VEC)
    best_d = best_i = None
    build_s = search_s = h2d_s = read_s = 0.0
    offset = n_shards = 0
    pf = pq.ParquetFile(f"{OOC}/corpus_{tag}.parquet")

    def shards():
        """Yield host-resident shards, accumulating from parquet batches.

        pyarrow refuses a batch_size of tens of millions ("List index overflow"), so the shard is
        assembled from ordinary batches instead of asked for in one go.
        """
        buf, have = [], 0
        for b in pf.iter_batches(batch_size=1_000_000, columns=["vec"]):
            buf.append(flat(b.column("vec"))); have += len(buf[-1])
            if have >= rows_per_shard:
                yield np.concatenate(buf, 0); buf, have = [], 0
        if buf:
            yield np.concatenate(buf, 0)

    t_all = time.perf_counter()
    for blk in shards():
        # Parquet decode is EXCLUDED from the comparison: Sirius's timed statement likewise
        # excludes its CREATE TABLE + pin_table, which is where its host-side load happens. What
        # is counted for both is the work from host-resident data onwards.
        t0 = time.perf_counter(); d_blk = cp.asarray(blk)
        cp.cuda.Stream.null.synchronize(); h2d_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_blk)
        cp.cuda.Stream.null.synchronize(); build_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        dist, ind = ivf_flat.search(ivf_flat.SearchParams(n_probes=a.n_probes),
                                    idx, d_probe, a.k)
        cp.cuda.Stream.null.synchronize(); search_s += time.perf_counter() - t0

        best_d, best_i = merge_topk(best_d, best_i,
                                    cp.asarray(dist), cp.asarray(ind).astype(cp.int64) + offset,
                                    a.k)
        offset += len(blk); n_shards += 1
        del idx, d_blk, dist, ind, blk
        free()
        print(f"    shard {n_shards}: {offset:,} rows done", flush=True)
    wall = time.perf_counter() - t_all

    I = cp.asnumpy(best_i)
    hits = sum(len(set(truth[i]) & set(I[i])) for i in range(len(probe)))
    rec = hits / (len(probe) * a.k)
    print(f"  sharded: {n_shards} shards of {a.shard_gib:g} GiB", flush=True)
    gpu_s = h2d_s + build_s + search_s
    print(f"    h2d {h2d_s:.2f}s | build {build_s:.2f}s | search {search_s:.2f}s "
          f"| GPU-side total {gpu_s:.2f}s | wall incl. parquet decode {wall:.2f}s "
          f"| recall@{a.k} {rec:.4f}", flush=True)
    print(f"RESULT cuvs {a.gib:g} {n_shards} {h2d_s:.3f} {build_s:.3f} {search_s:.3f} "
          f"{gpu_s:.3f} {wall:.3f} {rec:.4f}", flush=True)


if __name__ == "__main__":
    main()
