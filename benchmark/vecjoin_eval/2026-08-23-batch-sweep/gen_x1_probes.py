"""X1 setup — one nested probe-id sample and one independent ground truth, shared by both systems.

The whole point of X1 is that the two systems are asked the SAME question at four batch sizes, so
the probe sets must be identical rows, not merely identical counts. They are also NESTED
(ids_1k subset of ids_10k subset of ...), which lets a single 1000-row brute-force truth serve as
the recall oracle at every batch size.

The oracle is cupy brute force over the full corpus, NOT either system's own exhaustive run --
recall matching is the whole comparison here, so a self-referential reference would beg the
question. Self-matches are kept (a probe row is in the corpus); both systems see the same thing.
"""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, cupy as cp, time

DATA = "/var/tmp/vj/data/parquet"
OUT = "/var/tmp/vj/x1"
K = 10
SAMPLE = 1000
SEED = 20260823


def flat(col, dim=128):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    tbl = pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet")
    ids = np.asarray(tbl.column("id"), dtype="int64")
    base = flat(tbl.column("vec"))
    n = len(base)
    print(f"base {n:,} x 128, id range [{ids.min()},{ids.max()}]", flush=True)

    perm = np.random.default_rng(SEED).permutation(n)
    for cnt in (1_000, 10_000, 100_000, 1_000_000):
        sel = np.sort(ids[perm[:cnt]])
        pq.write_table(pa.table({"id": pa.array(sel, pa.int64())}), f"{OUT}/probe_ids_{cnt}.parquet")
        print(f"probe_ids_{cnt}: {len(sel):,}", flush=True)

    # recall sample: the first SAMPLE ids of the permutation, so it is a subset of every set above
    smp_row = perm[:SAMPLE]                      # positions into base
    smp_id = ids[smp_row]
    d_all = cp.asarray(base)
    d_s = cp.asarray(base[smp_row])
    sn = (d_s ** 2).sum(1, keepdims=True)
    bd = cp.full((SAMPLE, K), cp.inf, dtype=cp.float32)
    bi = cp.full((SAMPLE, K), -1, dtype=cp.int64)
    t0 = time.perf_counter()
    for off in range(0, n, 200_000):
        blk = d_all[off:off + 200_000]
        dist = sn + (blk ** 2).sum(1)[None, :] - 2.0 * (d_s @ blk.T)
        ii = cp.argpartition(dist, K, axis=1)[:, :K]
        cd = cp.take_along_axis(dist, ii, 1); ci = ii.astype(cp.int64) + off
        md = cp.concatenate([bd, cd], 1); mi = cp.concatenate([bi, ci], 1)
        o = cp.argsort(md, axis=1)[:, :K]
        bd = cp.take_along_axis(md, o, 1); bi = cp.take_along_axis(mi, o, 1)
        del dist, ii, cd, ci, md, mi
        cp.get_default_memory_pool().free_all_blocks()
    print(f"brute force {SAMPLE} x {n:,}: {time.perf_counter()-t0:.2f} s", flush=True)
    truth_rows = cp.asnumpy(bi)                  # positions into base
    truth_dist = cp.asnumpy(bd)                  # squared L2, same order
    truth_ids = ids[truth_rows]                  # -> corpus ids

    pq.write_table(pa.table({
        "left_id": pa.array(np.repeat(smp_id, K), pa.int64()),
        "right_id": pa.array(truth_ids.reshape(-1), pa.int64()),
    }), f"{OUT}/truth_{SAMPLE}.parquet")
    np.save(f"{OUT}/truth_rows.npy", truth_rows)
    np.save(f"{OUT}/truth_dist.npy", truth_dist)
    np.save(f"{OUT}/sample_rows.npy", smp_row)
    np.save(f"{OUT}/perm.npy", perm)
    print(f"truth_{SAMPLE}.parquet written ({SAMPLE*K:,} pairs)", flush=True)
    # SIFT1M base contains exact duplicate vectors (~1.5% of sampled rows have one), so the
    # probe row is not always ranked FIRST -- an equal-distance tie breaks toward the lower index.
    # It is always PRESENT, which is the invariant worth asserting. The same ties put a ceiling
    # below 1.0 on any id-set recall, which is why recall is also scored by distance downstream.
    self_hit = np.mean([smp_row[i] in set(truth_rows[i]) for i in range(SAMPLE)])
    dup = 1.0 - (truth_rows[:, 0] == smp_row).mean()
    print(f"sanity: probe row is inside its own top-{K} for {self_hit:.3f} of the sample "
          f"({'ok' if self_hit > 0.999 else 'SUSPECT'}); "
          f"{dup:.3f} of rows have an exact duplicate that outranks them", flush=True)


if __name__ == "__main__":
    main()
