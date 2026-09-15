"""X1 analysis — decompose the approximate-path gap into its independent parts.

Three tables:
  1. MATCHED RECALL vs batch size. For each Sirius operating point, the time cuVS needs to reach
     the SAME recall_dist, interpolated log-linearly in time between its two bracketing points.
     If the ratio decays toward 1 as the batch grows, the gap is fixed cost.
  2. ISO-PARAMETER. Same n_lists, same n_probes, both systems. Separates "we do the same work
     slower" from "we do less useful work per probe".
  3. FIXED-COST FIT. search_s = a + b*probes_n per n_probes setting; `a` is the intercept the
     batch sweep exists to measure.
"""
import json, numpy as np
from x1_score import OUT

cu = json.load(open(f"{OUT}/cuvs_x1.json"))
si64 = json.load(open(f"{OUT}/sirius_x1_c64.json"))
si1k = json.load(open(f"{OUT}/sirius_x1_c1024.json"))
COUNTS = [1_000, 10_000, 100_000, 1_000_000]


def cu_curve(cnt, tag="own-optimum", n_lists=1024):
    pts = [(r["recall_dist"], r["search_s"]) for r in cu
           if r["tag"] == tag and r["probes_n"] == cnt and r["n_lists"] == n_lists]
    return sorted(pts)


def interp(pts, rec):
    """cuVS time at `rec`, log-linear in time. None if outside the measured range."""
    rs = [p[0] for p in pts]
    if rec < rs[0] or rec > rs[-1]:
        return None, "extrapolated"
    for i in range(len(pts) - 1):
        r0, t0 = pts[i]; r1, t1 = pts[i + 1]
        if r0 <= rec <= r1:
            if r1 == r0:
                return t0, "exact"
            f = (rec - r0) / (r1 - r0)
            return float(np.exp(np.log(t0) + f * (np.log(t1) - np.log(t0)))), "interp"
    return None, "?"


print("=" * 94)
print("1. MATCHED RECALL — Sirius (64 clusters) vs cuVS IVF-Flat (1024 lists), by probe batch size")
print("=" * 94)
print(f"{'probes_n':>9} {'sirius':>7} {'recall_d':>9} {'sirius_s':>9} {'cuvs_s':>9} "
      f"{'how':>12} {'ratio':>8}  (>1 = cuVS faster)")
for cnt in COUNTS:
    pts = cu_curve(cnt)
    for r in si64["rows"]:
        if r["probes_n"] != cnt:
            continue
        t, how = interp(pts, r["recall_dist"])
        if t is None:
            print(f"{cnt:>9} {r['mode']:>7} {r['recall_dist']:>9.4f} {r['search_s']:>9.3f} "
                  f"{'-':>9} {how:>12} {'-':>8}")
        else:
            print(f"{cnt:>9} {r['mode']:>7} {r['recall_dist']:>9.4f} {r['search_s']:>9.3f} "
                  f"{t:>9.3f} {how:>12} {r['search_s']/t:>8.2f}x")
    print()

print("=" * 94)
print("2. ISO-PARAMETER — identical n_lists and n_probes on both sides")
print("=" * 94)
for n_lists, si, probes in ((1024, si1k, [1, 4, 16, 64]), (64, si64, [1, 2, 4, 8, 16])):
    tag = "own-optimum" if n_lists == 1024 else "iso-64"
    for cnt in ([1_000_000] if n_lists == 64 else COUNTS):
        cus = {r["n_probes"]: r for r in cu
               if r["tag"] == tag and r["probes_n"] == cnt and r["n_lists"] == n_lists}
        if not cus:
            continue
        print(f"\n  n_lists={n_lists}, probes_n={cnt:,}")
        print(f"  {'n_probes':>8} | {'sirius_s':>9} {'recall_d':>9} | {'cuvs_s':>9} "
              f"{'recall_d':>9} | {'time x':>7} {'recall gap':>11}")
        for p in probes:
            s = next((r for r in si["rows"] if r["probes_n"] == cnt and r["n_probes"] == p), None)
            c = cus.get(p)
            if not s or not c:
                continue
            print(f"  {p:>8} | {s['search_s']:>9.3f} {s['recall_dist']:>9.4f} | "
                  f"{c['search_s']:>9.3f} {c['recall_dist']:>9.4f} | "
                  f"{s['search_s']/c['search_s']:>6.2f}x {c['recall_dist']-s['recall_dist']:>+11.4f}")

print("\n" + "=" * 94)
print("3. FIXED COST — least-squares fit search_s = a + b * probes_n  (Sirius, 64 clusters)")
print("=" * 94)
print(f"{'n_probes':>9} {'a (fixed, s)':>13} {'b (per probe, us)':>18} {'a as % of 10k run':>19}")
for p in [1, 2, 4, 8, 16]:
    rows = [r for r in si64["rows"] if r["n_probes"] == p]
    n = np.array([r["probes_n"] for r in rows], float)
    t = np.array([r["search_s"] for r in rows], float)
    b, a = np.polyfit(n, t, 1)
    t10k = next(r["search_s"] for r in rows if r["probes_n"] == 10_000)
    print(f"{p:>9} {a:>13.4f} {b*1e6:>18.3f} {a/t10k*100:>18.1f}%")
print("\n  same fit, cuVS IVF-Flat n_lists=1024:")
print(f"{'n_probes':>9} {'a (fixed, s)':>13} {'b (per probe, us)':>18}")
for p in [1, 4, 16, 64]:
    rows = [r for r in cu if r["tag"] == "own-optimum" and r["n_probes"] == p]
    n = np.array([r["probes_n"] for r in rows], float)
    t = np.array([r["search_s"] for r in rows], float)
    b, a = np.polyfit(n, t, 1)
    print(f"{p:>9} {a:>13.4f} {b*1e6:>18.3f}")
