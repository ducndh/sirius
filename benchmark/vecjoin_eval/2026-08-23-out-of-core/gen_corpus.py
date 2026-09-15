"""Synthesize a corpus larger than GPU memory, plus a probe set and exact ground truth.

Why synthetic: the out-of-core claim is about CAPACITY and throughput past the device limit, not
about recall on a specific distribution. SIFT1M is 512 MB -- three orders of magnitude too small
to reach a 40 GB card's limit, and no public SIFT-like set spans the range we need to bracket.

Honesty constraints this encodes:
  * Vectors are drawn from a GAUSSIAN MIXTURE, not uniform noise. Uniform high-dimensional data
    has no cluster structure, which would flatter any IVF method (every list equally useless) and
    is not what real embeddings look like. A mixture gives genuine clusterability, so IVF-style
    pruning behaves qualitatively like it does on real data.
  * Ground truth is computed EXACTLY on GPU by brute force over the whole corpus, in chunks, so
    recall is real and not "recall against another approximate method".
  * The probe set is drawn from the same mixture, so queries land near clusters as they would in
    a real workload.

Sizes are given in GiB of raw FLOAT32[128] payload: 512 bytes/vector, so 1 GiB = 2,097,152 vectors.

Usage: gen_corpus.py <gib> [--dim 128] [--queries 10000] [--k 10] [--clusters 4096]
Writes /var/tmp/vj/ooc/corpus_<gib>gib.parquet, probe_<gib>gib.parquet, gt_<gib>gib.parquet
"""
import argparse, os, shutil, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq

OUT = "/var/tmp/vj/ooc"
# probe rows per distance tile. The expression builds 2-3 temporaries of
# PROBE_TILE x CORPUS_CHUNK floats, and cupy's pool RETAINS them between tiles, so this
# has to stay small enough that a few of them fit comfortably -- 1000 x 200k x 4 B = 0.8 GB.
PROBE_TILE = 1000
CORPUS_CHUNK = 200_000
BYTES_PER_VEC = 128 * 4


def flat(col, dim):
    """FixedSizeList -> (n, dim) float32 via the underlying value buffer.

    `np.stack(col.to_numpy(zero_copy_only=False))` builds one Python object per row and is ~50x
    slower; at 48 GiB that difference is an hour. Chunked arrays must be combined first.
    """
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def write_vectors(path, gen, n_total, dim, chunk):
    """Stream to parquet in chunks -- a 48 GiB array must never be materialized in host RAM."""
    schema = pa.schema([("id", pa.int64()),
                        ("vec", pa.list_(pa.float32(), dim))])
    w = pq.ParquetWriter(path, schema)
    written = 0
    while written < n_total:
        m = min(chunk, n_total - written)
        block = gen(written, m)
        ids = pa.array(np.arange(written, written + m, dtype=np.int64))
        vecs = pa.FixedSizeListArray.from_arrays(pa.array(block.reshape(-1)), dim)
        w.write_table(pa.Table.from_arrays([ids, vecs], schema=schema))
        written += m
        print(f"    {path.rsplit('/',1)[-1]}: {written}/{n_total} "
              f"({written*BYTES_PER_VEC/2**30:.1f} GiB)", flush=True)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gib", type=float)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-gt", action="store_true",
                    help="write vectors only and restore probe+gt from the JuiceFS backup. "
                         "Valid because generation is deterministic in --seed: the same seed "
                         "yields byte-identical vectors, so a previously computed ground truth "
                         "still applies. Saves the brute-force pass, which dominates runtime.")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    n = int(a.gib * 2**30 / BYTES_PER_VEC)
    rng = np.random.default_rng(a.seed)
    centroids = rng.normal(0, 1, (a.clusters, a.dim)).astype("float32") * 4.0
    tag = f"{a.gib:g}gib"
    print(f"corpus {n:,} x {a.dim} = {n*BYTES_PER_VEC/2**30:.1f} GiB, "
          f"{a.clusters} true clusters", flush=True)

    def gen_corpus(offset, m):
        r = np.random.default_rng(a.seed + 1000 + offset)
        which = r.integers(0, a.clusters, m)
        return (centroids[which] + r.normal(0, 1, (m, a.dim)).astype("float32")).astype("float32")

    def gen_probe(offset, m):
        r = np.random.default_rng(a.seed + 7)
        which = r.integers(0, a.clusters, m)
        return (centroids[which] + r.normal(0, 1, (m, a.dim)).astype("float32")).astype("float32")

    t0 = time.perf_counter()
    write_vectors(f"{OUT}/corpus_{tag}.parquet", gen_corpus, n, a.dim, 2_000_000)
    write_vectors(f"{OUT}/probe_{tag}.parquet", gen_probe, a.queries, a.dim, a.queries)
    print(f"  vectors written in {time.perf_counter()-t0:.1f}s", flush=True)

    backup = "/home/dnguyen56/vecjoin/data_backup/ooc_gt"
    if a.skip_gt:
        for f in (f"probe_{tag}.parquet", f"gt_{tag}.parquet"):
            src = f"{backup}/{f}"
            if not os.path.exists(src):
                raise SystemExit(f"--skip-gt needs {src}; run without it once to create it")
            shutil.copy2(src, f"{OUT}/{f}")
        print(f"  probe+gt restored from {backup} (same --seed => same vectors)", flush=True)
        return

    # Exact ground truth on GPU, streamed over corpus chunks. Keeps a running top-k so the
    # corpus never has to be resident -- the same constraint the experiment is about.
    import cupy as cp
    probe = flat(pq.read_table(f"{OUT}/probe_{tag}.parquet").column("vec"), a.dim)
    d_probe = cp.asarray(probe)
    d_pn = (d_probe ** 2).sum(1, keepdims=True)
    best_d = cp.full((len(probe), a.k), cp.inf, dtype=cp.float32)
    best_i = cp.full((len(probe), a.k), -1, dtype=cp.int64)

    t0, seen = time.perf_counter(), 0
    pf = pq.ParquetFile(f"{OUT}/corpus_{tag}.parquet")
    for batch in pf.iter_batches(batch_size=CORPUS_CHUNK, columns=["vec"]):
        blk = flat(batch.column("vec"), a.dim)
        d_blk = cp.asarray(blk)
        d_bn = (d_blk ** 2).sum(1)[None, :]
        # Tile the PROBES too: the distance block is n_probe x chunk, which at 10k probes and a
        # 500k chunk would be 20 GB on device. Tiling keeps it near PROBE_TILE x chunk x 4 B.
        for lo in range(0, len(probe), PROBE_TILE):
            hi = min(lo + PROBE_TILE, len(probe))
            dist = d_pn[lo:hi] + d_bn - 2.0 * (d_probe[lo:hi] @ d_blk.T)
            idx = cp.argpartition(dist, a.k, axis=1)[:, :a.k]
            cand_d = cp.take_along_axis(dist, idx, 1)
            cand_i = idx.astype(cp.int64) + seen
            merged_d = cp.concatenate([best_d[lo:hi], cand_d], 1)
            merged_i = cp.concatenate([best_i[lo:hi], cand_i], 1)
            order = cp.argsort(merged_d, axis=1)[:, :a.k]
            best_d[lo:hi] = cp.take_along_axis(merged_d, order, 1)
            best_i[lo:hi] = cp.take_along_axis(merged_i, order, 1)
            del dist, idx, cand_d, cand_i, merged_d, merged_i
            cp.get_default_memory_pool().free_all_blocks()
        seen += len(blk)
        del d_blk, d_bn
        cp.get_default_memory_pool().free_all_blocks()
        print(f"    gt: {seen:,}/{n:,}", flush=True)

    gt_i, gt_d = cp.asnumpy(best_i), np.sqrt(np.maximum(cp.asnumpy(best_d), 0))
    nq = len(probe)
    pq.write_table(pa.table({
        "query_id":    pa.array(np.repeat(np.arange(nq, dtype=np.int32), a.k)),
        "rank":        pa.array(np.tile(np.arange(a.k, dtype=np.int32), nq)),
        "neighbor_id": pa.array(gt_i.reshape(-1)),
        "distance":    pa.array(gt_d.reshape(-1).astype("float32")),
    }), f"{OUT}/gt_{tag}.parquet")
    print(f"  exact ground truth in {time.perf_counter()-t0:.1f}s -> gt_{tag}.parquet", flush=True)

    # The corpus is far too big for the 64 GB JuiceFS quota, but probe+gt are a few MB and are
    # the EXPENSIVE half to recompute (brute force over the whole corpus dominates generation).
    # Copy them off the ephemeral overlay so a wipe costs the corpus write, not the ground truth.
    backup = "/home/dnguyen56/vecjoin/data_backup/ooc_gt"
    try:
        os.makedirs(backup, exist_ok=True)
        for f in (f"probe_{tag}.parquet", f"gt_{tag}.parquet"):
            shutil.copy2(f"{OUT}/{f}", f"{backup}/{f}")
        print(f"  probe+gt backed up to {backup} (corpus is regenerable from --seed)", flush=True)
    except OSError as e:
        print(f"  WARNING: could not back up probe/gt: {e}", flush=True)


if __name__ == "__main__":
    main()
