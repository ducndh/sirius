"""Shared scorer for X1 so both systems are graded by identical code.

Two recall numbers, because they answer different questions:

  * `recall_id`   -- classic id-set intersection with the brute-force top-k. SIFT1M contains exact
                     duplicate vectors, so equal-distance ties cap this below 1.0 even for a
                     provably exact answer (measured: 1.5% of rows have a duplicate). Reported
                     because it is what every ANN paper reports.
  * `recall_dist` -- a returned neighbour counts if its TRUE squared distance is within tolerance
                     of the k-th true distance. Tie-robust: an exact answer scores 1.0. This is the
                     number to match two systems on, since a tie ceiling that differs by chance
                     between systems would otherwise look like a quality difference.
"""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

DATA = "/var/tmp/vj/data/parquet"
OUT = "/var/tmp/vj/x1"
K = 10
TOL = 1e-4      # relative; SIFT distances are O(1e4..1e5) so this is far below any real gap


def load_base():
    tbl = pq.read_table(f"{DATA}/sift-128-euclidean_base.parquet")
    col = tbl.column("vec")
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return np.asarray(tbl.column("id"), dtype="int64"), \
           np.asarray(col.values, dtype="float32").reshape(-1, 128)


class Scorer:
    """Grades a system's neighbours for the 1000 sampled probes."""

    def __init__(self):
        self.ids, self.base = load_base()
        self.smp_row = np.load(f"{OUT}/sample_rows.npy")
        self.truth_rows = np.load(f"{OUT}/truth_rows.npy")
        self.truth_dist = np.load(f"{OUT}/truth_dist.npy")
        self.row_of_id = np.full(self.ids.max() + 1, -1, dtype=np.int64)
        self.row_of_id[self.ids] = np.arange(len(self.ids))
        self.kth = self.truth_dist[:, K - 1]
        self.truth_sets = [set(r.tolist()) for r in self.truth_rows]

    def score(self, got_rows):
        """got_rows: (SAMPLE, k) positions into base, -1 for a missing slot."""
        n, k = got_rows.shape
        assert n == len(self.smp_row), (n, len(self.smp_row))
        hits_id = hits_d = 0
        for i in range(n):
            g = got_rows[i][got_rows[i] >= 0]
            hits_id += len(self.truth_sets[i] & set(g.tolist()))
            if len(g):
                d = ((self.base[g] - self.base[self.smp_row[i]]) ** 2).sum(1)
                hits_d += int((d <= self.kth[i] * (1 + TOL) + TOL).sum())
        return hits_id / (n * K), hits_d / (n * K)

    def rows_from_ids(self, arr):
        out = np.full(arr.shape, -1, dtype=np.int64)
        m = arr >= 0
        out[m] = self.row_of_id[arr[m]]
        return out
