"""X1, cuVS side — IVF-Flat search time vs PROBE BATCH SIZE at 1k / 10k / 100k / 1M.

P0 asks whether our approximate path's deficit is fixed cost (which shrinks as the batch grows)
or kernel speed (which does not). That can only be read off a curve, and the curve has to be the
same corpus, the same probe ROWS and the same oracle on both sides -- see gen_x1_probes.py.

cuVS is run at its own optimum (n_lists=1024) and, at the largest batch, also at Sirius's optimum
(n_lists=64), so the iso-parameter comparison (X2) comes out of the same run.

Usage: x1_cuvs.py [--n-lists 1024] [--iso-lists 64]
"""
import argparse, json, time, numpy as np, pyarrow.parquet as pq, cupy as cp
from cuvs.neighbors import ivf_flat
from x1_score import Scorer, OUT, K

COUNTS = [1_000, 10_000, 100_000, 1_000_000]
PROBES = [1, 4, 16, 64, 256]


def run_index(sc, base, n_lists, counts, probes, tag, results):
    d_all = cp.asarray(base)
    t0 = time.perf_counter()
    idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=n_lists), d_all)
    cp.cuda.Stream.null.synchronize()
    build = time.perf_counter() - t0
    print(f"\n=== cuVS IVF-Flat n_lists={n_lists} ({tag}) build {build:.3f} s", flush=True)
    print(f"{'probes_n':>9} {'n_probes':>9} {'search_s':>10} {'us/probe':>9} "
          f"{'recall_id':>10} {'recall_d':>9}", flush=True)
    for cnt in counts:
        pid = np.asarray(pq.read_table(f"{OUT}/probe_ids_{cnt}.parquet").column("id"), dtype="int64")
        prow = sc.row_of_id[pid]
        d_q = cp.asarray(base[prow])
        # position of each sampled probe inside THIS probe batch, for scoring
        pos = np.searchsorted(pid, sc.ids[sc.smp_row])
        assert (pid[pos] == sc.ids[sc.smp_row]).all(), "sample not a subset of this probe set"
        for npr in probes:
            ivf_flat.search(ivf_flat.SearchParams(n_probes=npr), idx, d_q[:1000], K)   # warm
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            _, I = ivf_flat.search(ivf_flat.SearchParams(n_probes=npr), idx, d_q, K)
            cp.cuda.Stream.null.synchronize()
            dt = time.perf_counter() - t0
            got = cp.asnumpy(I)[pos].astype(np.int64)       # cuVS returns POSITIONS into d_all
            r_id, r_d = sc.score(got)
            print(f"{cnt:>9} {npr:>9} {dt:>10.4f} {dt/cnt*1e6:>9.2f} {r_id:>10.4f} {r_d:>9.4f}",
                  flush=True)
            results.append(dict(system="cuvs", tag=tag, n_lists=n_lists, probes_n=cnt,
                                n_probes=npr, build_s=build, search_s=dt,
                                recall_id=r_id, recall_dist=r_d))
            del I, got
            cp.get_default_memory_pool().free_all_blocks()
        del d_q
        cp.get_default_memory_pool().free_all_blocks()
    del idx, d_all
    cp.get_default_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lists", type=int, default=1024)
    ap.add_argument("--iso-lists", type=int, default=64)
    ap.add_argument("--out", default=f"{OUT}/cuvs_x1.json")
    a = ap.parse_args()

    sc = Scorer()
    base = sc.base
    print(f"corpus {len(base):,} x 128, k={K}, oracle = cupy brute force on "
          f"{len(sc.smp_row)} sampled probes", flush=True)
    results = []
    run_index(sc, base, a.n_lists, COUNTS, PROBES, "own-optimum", results)
    # iso-parameter (X2): cuVS forced onto Sirius's cluster count, largest batch only
    run_index(sc, base, a.iso_lists, [1_000_000], [1, 2, 4, 8, 16], "iso-64", results)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
