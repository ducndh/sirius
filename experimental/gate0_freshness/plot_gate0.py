#!/usr/bin/env python3
"""Plot Gate-0 sweep results: read latency vs tick per (arm, rho), plus the
arm comparison at each churn rate. Usage: plot_gate0.py results/sweep.jsonl"""

import json
import sys
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not available — run under the pixi env or pip install it")


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.jsonl"
    recs = load(path)
    # read latency per (rho, arm, tick): mean over queries
    lat = defaultdict(list)
    for r in recs:
        if r.get("event") == "read" and r.get("wall_s") is not None and r.get("tick", -1) >= 0:
            lat[(r["rho"], r["arm"], r["tick"])].append(r["wall_s"])
    rhos = sorted({k[0] for k in lat})
    arms = sorted({k[1] for k in lat})
    fig, axes = plt.subplots(1, max(len(rhos), 1), figsize=(5 * max(len(rhos), 1), 4),
                             squeeze=False)
    for i, rho in enumerate(rhos):
        ax = axes[0][i]
        for arm in arms:
            ticks = sorted(t for (rr, a, t) in lat if rr == rho and a == arm)
            ys = [sum(lat[(rho, arm, t)]) / len(lat[(rho, arm, t)]) for t in ticks]
            ax.plot(ticks, ys, marker="o", label=arm)
        ax.set_title(f"rho={rho}")
        ax.set_xlabel("tick")
        ax.set_ylabel("mean read wall (s)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    box = next((r.get("box") for r in recs if r.get("box")), "?")
    fig.suptitle(f"Gate-0 read latency under churn ({box})")
    fig.tight_layout()
    out = path.replace(".jsonl", "_latency.png")
    fig.savefig(out, dpi=130)
    print(f"-> {out}")

    # arm_b decomposition: gpu vs delta share over ticks
    dec = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("event") == "read" and r.get("mode") == "merge_on_read" and r.get("gpu_s"):
            dec[r["rho"]]["gpu"].append((r["tick"], r["gpu_s"]))
            dec[r["rho"]]["delta"].append((r["tick"], r["delta_s"]))
    if dec:
        fig2, axes2 = plt.subplots(1, len(dec), figsize=(5 * len(dec), 4), squeeze=False)
        for i, rho in enumerate(sorted(dec)):
            ax = axes2[0][i]
            for part in ("gpu", "delta"):
                pts = defaultdict(list)
                for t, v in dec[rho][part]:
                    pts[t].append(v)
                ts = sorted(pts)
                ax.plot(ts, [sum(pts[t]) / len(pts[t]) for t in ts], marker="o", label=part)
            ax.set_title(f"merge-on-read split rho={rho}")
            ax.set_xlabel("tick")
            ax.set_ylabel("s")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        out2 = path.replace(".jsonl", "_morsplit.png")
        fig2.savefig(out2, dpi=130)
        print(f"-> {out2}")


if __name__ == "__main__":
    main()
