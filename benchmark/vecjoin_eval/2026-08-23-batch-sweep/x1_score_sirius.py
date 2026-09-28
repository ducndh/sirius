"""Grade the Sirius X1 CSVs with the SAME scorer the cuVS side used, and pair timings to recalls.

Timings come from the run log's `Run Time (s): real` lines, keyed by the `@@@` marker printed
immediately before each timed statement -- the recall export that follows a timed statement is
untimed and prints no Run Time line, so the pairing is unambiguous.
"""
import json, re, sys, numpy as np, pyarrow.csv as pacsv
from x1_score import Scorer, OUT, K


def parse_log(path):
    """-> [(marker, seconds)] in file order"""
    out, marker = [], None
    for line in open(path):
        m = re.match(r"@@@ (.*)", line.strip())
        if m:
            marker = m.group(1)
            continue
        m = re.search(r"Run Time \(s\): real ([0-9.]+)", line)
        if m and marker is not None:
            out.append((marker, float(m.group(1))))
            marker = None
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "c64"
    global MODES
    MODES = (sys.argv[3].split(",") if len(sys.argv) > 3
             else ["p1", "p2", "p4", "p8", "p16", "exact"])
    log = sys.argv[2] if len(sys.argv) > 2 else f"/var/tmp/vj/x1_sirius_{tag}.log"
    sc = Scorer()
    smp_ids = sc.ids[sc.smp_row]
    order = {int(v): i for i, v in enumerate(smp_ids)}
    times = dict(parse_log(log))
    setup = {k: v for k, v in times.items() if not k.startswith(("approx", "exact"))}
    print(f"setup phases: " + "  ".join(f"{k}={v:.3f}s" for k, v in setup.items()), flush=True)

    rows = []
    print(f"\n{'probes_n':>9} {'mode':>10} {'search_s':>10} {'us/probe':>9} "
          f"{'recall_id':>10} {'recall_d':>9}", flush=True)
    for cnt in (1_000, 10_000, 100_000, 1_000_000):
        for mode in MODES:
            f = f"{OUT}/sir_{tag}_{cnt}_{mode}.csv"
            t = pacsv.read_csv(f)
            li = np.asarray(t.column("left_id"), dtype="int64")
            ri = np.asarray(t.column("right_id"), dtype="int64")
            got = np.full((len(smp_ids), K), -1, dtype=np.int64)
            fill = np.zeros(len(smp_ids), dtype=np.int64)
            for a, b in zip(li, ri):
                i = order[int(a)]
                if fill[i] < K:
                    got[i, fill[i]] = b
                    fill[i] += 1
            assert (fill == K).all(), f"{f}: not all sampled probes got {K} neighbours"
            r_id, r_d = sc.score(sc.rows_from_ids(got))
            key = (f"approx probes_n={cnt} n_probes={mode[1:]}" if mode != "exact"
                   else f"exact probes_n={cnt}")
            dt = times[key]
            print(f"{cnt:>9} {mode:>10} {dt:>10.4f} {dt/cnt*1e6:>9.2f} {r_id:>10.4f} {r_d:>9.4f}",
                  flush=True)
            rows.append(dict(system="sirius", tag=tag, probes_n=cnt, mode=mode,
                             n_probes=(None if mode == "exact" else int(mode[1:])),
                             search_s=dt, recall_id=r_id, recall_dist=r_d))
    json.dump(dict(setup=setup, rows=rows), open(f"{OUT}/sirius_x1_{tag}.json", "w"), indent=1)
    print(f"\nwrote {OUT}/sirius_x1_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
