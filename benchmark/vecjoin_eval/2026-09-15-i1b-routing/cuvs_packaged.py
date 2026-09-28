"""cuVS opponents on the SAME workload the Sirius Pareto uses.

Deliberately the packaged SIFT1M query set (10k x 1M) scored against the packaged ground
truth, not a self-join with a self-referential reference -- so a recall here means the same
thing as a recall in the Sirius table next to it.

Two accountings are reported for every index, because they give different winners and the
handoff requires naming which one is quoted:
  search   the index already exists (amortized) -- what an ANN benchmark reports
  total    build + search (one-shot) -- what a JOIN over a corpus you did not pre-index costs
"""
import json, time
import numpy as np
import pyarrow.parquet as pq
import cupy as cp
from cuvs.neighbors import brute_force, ivf_flat, cagra

DATA = "/var/tmp/vj/data/parquet"
K = 10


def load(name):
    t = pq.read_table(f"{DATA}/sift-128-euclidean_{name}.parquet")
    return np.ascontiguousarray(
        np.stack(t.column("vec").to_numpy(zero_copy_only=False)).astype(np.float32))


def ground_truth():
    t = pq.read_table(f"{DATA}/sift-128-euclidean_gt.parquet")
    q = t.column("query_id").to_numpy()
    rk = t.column("rank").to_numpy()
    nb = t.column("neighbor_id").to_numpy()
    keep = rk < K
    q, rk, nb = q[keep], rk[keep], nb[keep]
    order = np.lexsort((rk, q))
    return nb[order].reshape(int(q.max()) + 1, K)


def timed(fn, reps=3):
    fn()
    return min(_time_once(fn) for _ in range(reps))


def _time_once(fn):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    fn()
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def recall(got, ref):
    return sum(len(set(g.tolist()) & set(r.tolist())) for g, r in zip(got, ref)) / ref.size


def main():
    base, queries, gt = load("base"), load("query"), ground_truth()
    print(f"# base {base.shape}, queries {queries.shape}, gt {gt.shape}, k={K}", flush=True)
    d_base, d_q = cp.asarray(base), cp.asarray(queries)
    rows = []

    def record(name, t_build, t_search, nb):
        r = recall(cp.asarray(nb).get(), gt)
        rows.append(dict(name=name, build_s=t_build, search_s=t_search,
                         total_s=t_build + t_search, recall=r))
        print(f"{name:34s} build {t_build:7.3f}s  search {t_search:7.3f}s  "
              f"total {t_build + t_search:7.3f}s  recall {r:.4f}", flush=True)

    rows.clear()
    t0 = time.perf_counter()
    idx = brute_force.build(d_base)
    cp.cuda.runtime.deviceSynchronize()
    t_b = time.perf_counter() - t0
    t_s = timed(lambda: brute_force.search(idx, d_q, K))
    record("cuVS brute_force (exact)", t_b, t_s, brute_force.search(idx, d_q, K)[1])
    del idx

    for nlist in (64, 1024):
        for nprobe in (1, 4, 16, 64, 256):
            if nprobe > nlist:
                continue
            t0 = time.perf_counter()
            index = ivf_flat.build(ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean"), d_base)
            cp.cuda.runtime.deviceSynchronize()
            t_b = time.perf_counter() - t0
            sp = ivf_flat.SearchParams(n_probes=nprobe)
            t_s = timed(lambda: ivf_flat.search(sp, index, d_q, K))
            record(f"cuVS IVF-Flat nlist={nlist} nprobe={nprobe}", t_b, t_s,
                   ivf_flat.search(sp, index, d_q, K)[1])
            del index

    # build_algo defaults to ivf_pq and on this corpus that produces a DEGENERATE graph --
    # "self-included ratio 0.00%", recall 0.0001 at every search parameter. Reporting that would
    # be the exact failure feedback_benchmark_discipline rule 7 exists to prevent, so the opponent
    # is built with nn_descent, which I3 already measured as healthy here.
    for graph_degree, build_algo in ((32, "nn_descent"), (64, "nn_descent"), (64, "ivf_pq")):
        t0 = time.perf_counter()
        index = cagra.build(
            cagra.IndexParams(graph_degree=graph_degree, metric="sqeuclidean",
                              build_algo=build_algo, refinement_rate=2.0), d_base)
        cp.cuda.runtime.deviceSynchronize()
        t_b = time.perf_counter() - t0
        for itopk in (32, 64, 128, 256):
            sp = cagra.SearchParams(itopk_size=itopk)
            t_s = timed(lambda: cagra.search(sp, index, d_q, K))
            record(f"cuVS CAGRA {build_algo} deg={graph_degree} itopk={itopk}", t_b, t_s,
                   cagra.search(sp, index, d_q, K)[1])
        del index

    with open("/var/tmp/vj/i1b/cuvs.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote /var/tmp/vj/i1b/cuvs.json")


main()
