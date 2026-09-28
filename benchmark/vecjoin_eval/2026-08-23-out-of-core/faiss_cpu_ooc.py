"""E2 — FAISS-CPU at 48 GiB: the rival that has no device limit at all.

Out-of-core is only a *capability* gap against GPU libraries. A CPU index lives in host RAM, so a
48 GiB corpus is unremarkable to FAISS on a 120 GB box — there is nothing for it to work around.
That makes FAISS-CPU the honest competitor for "GPU speed at host scale", and it was untested.

MEMORY DISCIPLINE (this box has a 120 GiB cgroup ceiling and has gone down mid-run before):
`IndexIVFFlat` keeps its own copy of every vector in the inverted lists, so holding the raw corpus
AND the index would be ~96 GiB. Instead the corpus is streamed from parquet in chunks and each
chunk is freed after `add`, keeping peak at roughly index + one chunk. An RSS guard aborts rather
than letting the machine die: a benchmark that takes the box down measures nothing.

Training uses a sample, which is standard FAISS practice — training on 100M vectors would dominate
the run and is not what anyone does.

Usage: faiss_cpu_ooc.py [--gib 48] [--n-lists 4096] [--n-probes 16] [--k 10]
                        [--train-sample 2000000] [--rss-ceiling-gb 85]
"""
import argparse, gc, os, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq, faiss

OOC = "/var/tmp/vj/ooc"


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def guard(ceiling, where):
    r = rss_gb()
    if r >= ceiling:
        raise SystemExit(f"ABORT at {where}: RSS {r:.1f} GB >= ceiling {ceiling} GB. "
                         f"Lower --train-sample or raise the ceiling deliberately.")
    return r


def flat(col, dim=128):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(col.values, dtype="float32").reshape(-1, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=48)
    ap.add_argument("--n-lists", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--train-sample", type=int, default=2_000_000)
    ap.add_argument("--rss-ceiling-gb", type=float, default=85)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    tag = f"{a.gib:g}gib"
    faiss.omp_set_num_threads(a.threads)

    probe = flat(pq.read_table(f"{OOC}/probe_{tag}.parquet").column("vec"))
    gt = pq.read_table(f"{OOC}/gt_{tag}.parquet")
    q, r, nid = (gt.column(c).to_numpy() for c in ("query_id", "rank", "neighbor_id"))
    m = r < a.k
    truth = np.full((int(q.max()) + 1, a.k), -1, dtype=np.int64)
    truth[q[m], r[m]] = nid[m]

    corpus = f"{OOC}/corpus_{tag}.parquet"
    pf = pq.ParquetFile(corpus)
    n_rows = pf.metadata.num_rows
    print(f"FAISS-CPU, {a.gib:g} GiB corpus ({n_rows:,} rows), {a.threads} threads, "
          f"n_lists={a.n_lists}, RSS ceiling {a.rss_ceiling_gb} GB", flush=True)

    # --- train on a sample
    t0 = time.perf_counter()
    sample, have = [], 0
    for b in pf.iter_batches(batch_size=500_000, columns=["vec"]):
        sample.append(flat(b.column("vec"))); have += len(sample[-1])
        if have >= a.train_sample:
            break
    train = np.concatenate(sample, 0)[:a.train_sample]
    del sample
    guard(a.rss_ceiling_gb, "after loading training sample")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, a.n_lists)
    index.train(train)
    train_s = time.perf_counter() - t0
    del train
    gc.collect()
    print(f"  train ({a.train_sample:,} sampled rows): {train_s:.2f} s  RSS {rss_gb():.1f} GB",
          flush=True)

    # --- add the whole corpus, streamed
    t0 = time.perf_counter()
    added = 0
    pf2 = pq.ParquetFile(corpus)
    for b in pf2.iter_batches(batch_size=1_000_000, columns=["vec"]):
        blk = flat(b.column("vec"))
        index.add(blk)
        added += len(blk)
        del blk, b
        gc.collect()
        # Arrow keeps freed batch buffers in its own pool, so RSS climbs well past what the index
        # holds -- 60M rows showed 66.3 GB against ~31 GB of actual vector data. Hand it back to
        # the OS or the guard trips on the reader, not on FAISS.
        pa.default_memory_pool().release_unused()
        guard(a.rss_ceiling_gb, f"after adding {added:,} rows")
        if added % 20_000_000 == 0:
            print(f"    added {added:,}/{n_rows:,}  RSS {rss_gb():.1f} GB", flush=True)
    add_s = time.perf_counter() - t0
    print(f"  add (streamed): {add_s:.2f} s  ntotal={index.ntotal:,}  RSS {rss_gb():.1f} GB",
          flush=True)

    # --- search
    index.nprobe = a.n_probes
    index.search(probe[:100], a.k)                       # warm
    t0 = time.perf_counter()
    _, I = index.search(probe, a.k)
    search_s = time.perf_counter() - t0
    rec = sum(len(set(truth[i]) & set(I[i])) for i in range(len(probe))) / (len(probe) * a.k)

    build = train_s + add_s
    print(f"\n  build (train+add) {build:8.2f} s", flush=True)
    print(f"  search            {search_s:8.3f} s   recall@{a.k} {rec:.4f}", flush=True)
    print(f"  TOTAL one-shot    {build + search_s:8.2f} s", flush=True)
    print(f"  peak RSS          {rss_gb():8.1f} GB", flush=True)
    print(f"RESULT faiss_cpu {a.gib:g} {train_s:.3f} {add_s:.3f} {search_s:.3f} "
          f"{build + search_s:.3f} {rec:.4f}", flush=True)


if __name__ == "__main__":
    main()
