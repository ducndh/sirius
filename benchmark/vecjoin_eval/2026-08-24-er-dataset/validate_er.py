"""Does a cosine threshold join actually recover the labelled matches? Validate before handing off.

Two framings, because they answer different things and only one is the demo claim:
  (a) ER-standard: score only the LABELLED candidate pairs. Precision/recall/F1 vs tau. This is what
      the benchmark's numbers mean.
  (b) JOIN-style: retrieve every pair in the full cross product above tau. Recall of the gold
      positives is the number that matters for us -- "can a vector join find the true matches".
      Precision here is a LOWER BOUND: pairs outside the labelled candidate set are unknown, not
      known-wrong, so a retrieved unlabelled pair is not necessarily a false positive.
"""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

OUT = "/var/tmp/vj/data/parquet"


def vecs(stem, dim=384):
    t = pq.read_table(f"{OUT}/{stem}.parquet")
    c = t.column("vec")
    if isinstance(c, pa.ChunkedArray):
        c = c.combine_chunks()
    return (np.asarray(t.column("id"), dtype="int64"),
            np.asarray(c.values, dtype="float32").reshape(-1, dim))


for ds in ("abt-buy", "amazon-google"):
    ia, A = vecs(f"{ds}_a")
    ib, B = vecs(f"{ds}_b")
    lab = pq.read_table(f"{OUT}/{ds}_labels.parquet")
    li = np.asarray(lab.column("left_id"), dtype="int64")
    ri = np.asarray(lab.column("right_id"), dtype="int64")
    y = np.asarray(lab.column("label"), dtype="int8")
    ra = {int(v): k for k, v in enumerate(ia)}
    rb = {int(v): k for k, v in enumerate(ib)}
    sim_pairs = np.einsum("ij,ij->i", A[[ra[int(x)] for x in li]], B[[rb[int(x)] for x in ri]])
    gold = set(zip(li[y == 1].tolist(), ri[y == 1].tolist()))
    S = A @ B.T                                     # cosine similarity, both sides unit-norm
    print(f"\n=== {ds}: A={len(A):,} B={len(B):,} gold matches={len(gold):,} "
          f"cross product={len(A)*len(B):,}")
    print(f"  gold-pair similarity: median {np.median(sim_pairs[y==1]):.3f} | "
          f"labelled non-match median {np.median(sim_pairs[y==0]):.3f}")
    print(f"  {'tau':>5} | {'(a) labelled-set P / R / F1':>30} | {'(b) join recall':>15} {'pairs out':>10}")
    for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        pred = sim_pairs >= tau
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        gi, gj = np.nonzero(S >= tau)
        got = set(zip(ia[gi].tolist(), ib[gj].tolist()))
        jr = len(got & gold) / max(len(gold), 1)
        print(f"  {tau:>5.2f} | {p:>9.3f} {r:>9.3f} {f1:>9.3f} | {jr:>15.3f} {len(got):>10,}")
