"""The ways cuVS CAN go past the device limit -- because "cuVS needs the corpus resident" is
too strong a claim to put in a paper.

A reviewer who knows the ecosystem will immediately name these. Measuring them is the difference
between a defensible narrow claim ("each strategy costs you something we don't pay") and an
overclaim that collapses under one question.

Strategies measured here, on a corpus larger than the device:

  uvm     -- RMM managed memory. The corpus oversubscribes the device and the driver pages it in
             on demand. cuVS runs UNMODIFIED; the cost is paging traffic, not code.
  ivfpq   -- product quantization. 128 fp32 dims -> `pq_dim` bytes, roughly a 32-64x shrink, so a
             48 GiB corpus becomes ~1 GiB and fits trivially. This is what practitioners actually
             do, and it is the strongest counter to an out-of-core claim. The cost is RECALL:
             distances are approximated from codes, so this is not exact and cannot reach 1.0
             without a refinement pass over the original vectors.
  shard   -- covered by cuvs_ooc.py; listed here only so the comparison is complete.

Reported against the same exact ground truth as everything else in this directory.

Usage: cuvs_strategies.py <gib> [--n-probes 16] [--n-lists 4096] [--pq-dim 32] [--k 10]
"""
import argparse, gc, time, numpy as np, pyarrow.parquet as pq, cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from cuvs.neighbors import ivf_flat, ivf_pq

OOC = "/var/tmp/vj/ooc"


def flat(col, dim=128):
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def load_device(path, dim=128, into=None):
    """Stream parquet into a device array without ever holding the corpus twice in host RAM."""
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    d = cp.empty((n, dim), dtype=cp.float32) if into is None else into
    at = 0
    for b in pf.iter_batches(batch_size=1_000_000, columns=["vec"]):
        blk = flat(b.column("vec"), dim)
        d[at:at + len(blk)] = cp.asarray(blk)
        at += len(blk)
        del blk
    return d


def recall_of(I, truth, k):
    I = cp.asnumpy(I)
    return sum(len(set(truth[i]) & set(I[i])) for i in range(len(I))) / (len(I) * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gib", type=float)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--n-lists", type=int, default=4096)
    ap.add_argument("--pq-dim", type=int, default=32)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--only", default="", help="uvm|ivfpq (default: both)")
    a = ap.parse_args()
    tag = f"{a.gib:g}gib"
    corpus = f"{OOC}/corpus_{tag}.parquet"

    probe_h = flat(pq.read_table(f"{OOC}/probe_{tag}.parquet").column("vec"))
    gt = pq.read_table(f"{OOC}/gt_{tag}.parquet")
    q, r, nid = (gt.column(c).to_numpy() for c in ("query_id", "rank", "neighbor_id"))
    m = r < a.k
    truth = np.full((int(q.max()) + 1, a.k), -1, dtype=np.int64)
    truth[q[m], r[m]] = nid[m]

    dev_total = cp.cuda.Device().mem_info[1] / 2**30
    print(f"corpus {a.gib:g} GiB vs device {dev_total:.1f} GiB | k={a.k}", flush=True)

    want = a.only or "uvm,ivfpq"

    # IVF-PQ compresses the stored INDEX, not the build INPUT: cuvs still wants the raw fp32
    # corpus readable by the builder. At 48 GiB that is 51.5 GB and a plain device allocation
    # OOMs -- which says nothing about IVF-PQ, only about how it was fed. Enabling managed
    # memory FIRST lets the build input oversubscribe, which is the fair test.
    if "ivfpq" in want:
        rmm.reinitialize(managed_memory=True, pool_allocator=False)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print("  (managed memory enabled so the IVF-PQ build input can oversubscribe)", flush=True)

    # ---- IVF-PQ first, while the default allocator is still in place.
    if "ivfpq" in want:
        try:
            d_corpus = load_device(corpus)
            d_probe = cp.asarray(probe_h)
            t0 = time.perf_counter()
            idx = ivf_pq.build(ivf_pq.IndexParams(n_lists=a.n_lists, pq_dim=a.pq_dim), d_corpus)
            cp.cuda.Stream.null.synchronize(); build = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, I = ivf_pq.search(ivf_pq.SearchParams(n_probes=a.n_probes), idx, d_probe, a.k)
            cp.cuda.Stream.null.synchronize(); search = time.perf_counter() - t0
            print(f"  ivfpq  pq_dim={a.pq_dim}: build {build:.2f}s search {search:.4f}s "
                  f"recall@{a.k} {recall_of(I, truth, a.k):.4f}", flush=True)
            print(f"    NOTE: compressed index is ~{128*4/a.pq_dim:.0f}x smaller than the fp32 "
                  f"corpus -- this is why it fits. Recall is bounded by the quantization.",
                  flush=True)
            del idx, d_corpus, I
        except Exception as e:                                    # noqa: BLE001
            print(f"  ivfpq: FAILED {type(e).__name__}: {e}", flush=True)
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ---- UVM last: switching cupy's allocator is process-wide and hard to undo cleanly.
    if "uvm" in want:
        try:
            if "ivfpq" not in want:
                rmm.reinitialize(managed_memory=True, pool_allocator=False)
                cp.cuda.set_allocator(rmm_cupy_allocator)
            print("  uvm: RMM managed memory active (corpus may oversubscribe the device)",
                  flush=True)
            d_corpus = load_device(corpus)
            d_probe = cp.asarray(probe_h)
            t0 = time.perf_counter()
            idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.n_lists), d_corpus)
            cp.cuda.Stream.null.synchronize(); build = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, I = ivf_flat.search(ivf_flat.SearchParams(n_probes=a.n_probes), idx, d_probe, a.k)
            cp.cuda.Stream.null.synchronize(); search = time.perf_counter() - t0
            print(f"  uvm    ivf_flat: build {build:.2f}s search {search:.4f}s "
                  f"recall@{a.k} {recall_of(I, truth, a.k):.4f}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  uvm: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
