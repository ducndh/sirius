"""I2 — read the join's time split off a hardware trace, independently of X1's fitted model.

X1 fitted `time ~= C + a*CALLS + g*REREAD + B*PAIRS` and concluded the small-batch regime is
launch-bound. That was a FIT. This checks it against nsys:

  * `cuvs::neighbors::detail::knn_merge_parts_kernel` fires once per `brute_force_knn` call, so it
    MEASURES CALLS instead of assuming `runs * n_probes`.
  * GPU busy time vs wall span says directly whether the device is idle waiting on launches.
  * Summed kernel time can exceed the span: kernels overlap across streams. >100% is concurrency,
    not an error.

ncu is NOT usable on this box -- see README. This is trace-only: no occupancy, no counters.
"""
import sqlite3, sys

DB = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/vj/vj_join.sqlite"
GAP_NS = 30_000_000        # 30 ms of GPU idle separates one statement from the next
MERGE = "knn_merge_parts_kernel"

c = sqlite3.connect(DB)
rows = c.execute(
    "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start").fetchall()
print(f"{len(rows):,} kernel launches in the trace\n")

periods, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > GAP_NS:
        periods.append(cur); cur = [r]
    else:
        cur.append(r)
periods.append(cur)

print(f"{'#':>3} {'span_s':>8} {'kernel_s':>9} {'busy%':>6} {'launches':>9} {'CALLS':>6} "
      f"{'launch/call':>11}  top kernel")
for i, p in enumerate(periods):
    span = (p[-1][1] - p[0][0]) / 1e9
    if span < 0.05:
        continue
    busy = sum(e - s for s, e, _ in p) / 1e9
    calls = sum(1 for _, _, n in p if MERGE in n)
    names = {}
    for s, e, n in p:
        names[n] = names.get(n, 0) + (e - s)
    top = max(names.items(), key=lambda kv: kv[1])
    lpc = f"{len(p)/calls:.1f}" if calls else "-"
    print(f"{i:>3} {span:>8.3f} {busy:>9.3f} {busy/span*100:>5.1f}% {len(p):>9,} {calls:>6,} "
          f"{lpc:>11}  {top[0].split('<')[0][:44]} ({top[1]/1e9/busy*100:.0f}% of busy)")
