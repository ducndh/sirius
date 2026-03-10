#!/usr/bin/env python3
"""
ASOF JOIN Benchmark: Sirius GPU vs DuckDB CPU

Generates synthetic tick data (trades + quotes) and compares ASOF JOIN
performance between DuckDB CPU (Python package) and Sirius GPU (CLI binary).

Algorithm under test (Sirius Phase 1 / Option A):
  Sort right by (symbol, ts) ASC → cudf::upper_bound → custom boundary kernel
  → cudf::gather with NULLIFY

Usage:
  python3 scripts/run_asof_benchmark.py [options]

  --scale N    1=1M/200K  2=10M/2M  3=50M/10M  4=200M/40M  (default: 1 2 3)
  --runs  N    timed repetitions per scale  (default: 3)
  --no-cpu     skip DuckDB CPU baseline
  --no-gpu     skip Sirius GPU run
  --warmup     run one un-timed warmup iteration for GPU (default: true)

Environment:
  Must be run from the repo root or via pixi:
    ~/.pixi/bin/pixi run -e cuda12 python3 scripts/run_asof_benchmark.py
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
SIRIUS_BIN = os.path.join(REPO_DIR, "build", "release", "duckdb")
SIRIUS_EXT = os.path.join(REPO_DIR, "build", "release", "extension", "sirius", "sirius.duckdb_extension")
SIRIUS_LIB = os.path.join(REPO_DIR, ".pixi", "envs", "cuda12", "lib")

# ── Scale factors ──────────────────────────────────────────────────────────
# trades: (symbol INTEGER, ts BIGINT, price DOUBLE)           → 20 bytes/row
# quotes: (symbol INTEGER, ts BIGINT, bid DOUBLE, ask DOUBLE) → 28 bytes/row
SCALES = {
    1: (1_000_000,   200_000,   "1M trades / 200K quotes   (~24 MB)"),
    2: (10_000_000,  2_000_000, "10M trades / 2M quotes    (~240 MB)"),
    3: (50_000_000,  10_000_000,"50M trades / 10M quotes   (~1.2 GB)"),
    4: (200_000_000, 40_000_000,"200M trades / 40M quotes  (~4.8 GB)"),
}

# ── SQL templates ──────────────────────────────────────────────────────────
# 1000 symbols, timestamps uniformly distributed over a 24-hour window
# starting at Unix epoch + 10^12 ns (arbitrary but reproducible)
GEN_TRADES_SQL = """
CREATE OR REPLACE TABLE trades AS
SELECT
    (random() * 1000)::INTEGER                              AS symbol,
    (1_000_000_000_000 + random() * 86_400_000_000_000)::BIGINT AS ts,
    50.0 + random() * 100.0                                AS price
FROM range({n});
"""

GEN_QUOTES_SQL = """
CREATE OR REPLACE TABLE quotes AS
SELECT
    (random() * 1000)::INTEGER                              AS symbol,
    (1_000_000_000_000 + random() * 86_400_000_000_000)::BIGINT AS ts,
    50.0 + random() * 100.0                                AS bid,
    50.5 + random() * 100.0                                AS ask
FROM range({n});
"""

# Benchmark query: COUNT(*) + aggregates so the optimizer can't elide the join
JOIN_QUERY = """\
SELECT COUNT(*) AS matched, AVG(q.bid) AS avg_bid, AVG(q.ask) AS avg_ask
FROM trades t
ASOF JOIN quotes q ON t.symbol = q.symbol AND t.ts >= q.ts\
"""

# ── Helpers ────────────────────────────────────────────────────────────────

def fmt_time(secs: float) -> str:
    if secs < 1:
        return f"{secs * 1000:.0f} ms"
    return f"{secs:.3f} s"


def fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ── CPU benchmark ──────────────────────────────────────────────────────────

def run_cpu_benchmark(trades_pq: str, quotes_pq: str, n_runs: int):
    """
    Load parquet into in-memory DuckDB once, then time N runs of the JOIN.
    Returns (best_time_secs, result_row).
    """
    con = duckdb.connect()
    progress("  [CPU] Loading parquet into memory ...")
    t_load = time.perf_counter()
    con.execute(f"CREATE TABLE trades AS SELECT * FROM '{trades_pq}'")
    con.execute(f"CREATE TABLE quotes AS SELECT * FROM '{quotes_pq}'")
    progress(f"  [CPU] Load done in {fmt_time(time.perf_counter() - t_load)}")

    times = []
    result = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = con.execute(JOIN_QUERY).fetchone()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        progress(f"  [CPU] Run {i+1}/{n_runs}: {fmt_time(elapsed)}"
                 f"  → {result[0]:,} matched rows")

    return min(times), result


# ── GPU benchmark ──────────────────────────────────────────────────────────

def _build_gpu_sql(trades_pq: str, quotes_pq: str, n_runs: int) -> str:
    """
    Build a Sirius duckdb CLI script that:
      1. Loads the extension and parquet data (un-timed)
      2. Runs the JOIN query N times with .timer on (each prints Run Time)
    """
    # GPU execution requires gpu_processing(); direct SQL uses DuckDB's standard
    # physical planner and bypasses the GPU physical plan generator entirely.
    gpu_query = JOIN_QUERY.replace('"', '\\"')  # escape for shell embedding
    lines = [
        # Extension is embedded in binary — no LOAD needed.
        # gpu_buffer_init sets up GPU memory allocator; use conservative defaults.
        "CALL gpu_buffer_init('20 GB', '15 GB');",
        f"CREATE TABLE trades AS SELECT * FROM '{trades_pq}';",
        f"CREATE TABLE quotes AS SELECT * FROM '{quotes_pq}';",
        ".timer on",
    ]
    for _ in range(n_runs):
        lines.append(f'CALL gpu_processing("{gpu_query}");')
    return "\n".join(lines) + "\n"


def _parse_timer_lines(stdout: str) -> list[float]:
    """
    Extract all 'Run Time (s): real <N>' values from DuckDB .timer output.
    """
    times = []
    for line in stdout.splitlines():
        m = re.search(r"Run Time \(s\):\s+real\s+([0-9.]+)", line)
        if m:
            times.append(float(m.group(1)))
    return times


def run_gpu_benchmark(trades_pq: str, quotes_pq: str, n_runs: int, do_warmup: bool):
    """
    Run ASOF JOIN via Sirius GPU binary, parse .timer on output.
    Returns (best_time_secs, result_row).
    """
    if not os.path.exists(SIRIUS_BIN):
        progress(f"  [GPU] Sirius binary not found: {SIRIUS_BIN}  (skipping)")
        return None, None

    env = {
        **os.environ,
        "LD_LIBRARY_PATH": SIRIUS_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }

    def _run(sql: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [SIRIUS_BIN, "-unsigned"],
            input=sql,
            capture_output=True,
            text=True,
            env=env,
        )

    if do_warmup:
        progress("  [GPU] Warm-up run ...")
        warmup_sql = _build_gpu_sql(trades_pq, quotes_pq, 1)
        proc = _run(warmup_sql)
        if proc.returncode != 0:
            progress(f"  [GPU] Warm-up FAILED:\n{proc.stderr[:400]}")
            return None, None
        progress("  [GPU] Warm-up done")

    sql = _build_gpu_sql(trades_pq, quotes_pq, n_runs)

    progress(f"  [GPU] Running {n_runs} timed iterations ...")
    proc = _run(sql)

    if proc.returncode != 0:
        progress(f"  [GPU] Run FAILED (exit {proc.returncode}):")
        progress(proc.stderr[:600])
        return None, None

    times = _parse_timer_lines(proc.stdout)
    if not times:
        progress("  [GPU] WARNING: no timer lines found in output")
        progress("stdout:\n" + proc.stdout[:400])
        return None, None

    # Parse result row: first non-header, non-dashed line after .timer output
    result = None
    for i, t in enumerate(times):
        progress(f"  [GPU] Run {i+1}/{n_runs}: {fmt_time(t)}")

    # Extract last query result from stdout (matched rows)
    for line in proc.stdout.splitlines():
        # DuckDB table output rows look like "│  12345678 │  50.123 │  50.623 │"
        m = re.search(r"[│|]\s*(\d[\d,]*)\s*[│|]", line)
        if m:
            try:
                result = (int(m.group(1).replace(",", "")),)
            except ValueError:
                pass

    if result:
        progress(f"  [GPU] Result: {result[0]:,} matched rows")

    return min(times), result


# ── Data generation ────────────────────────────────────────────────────────

def generate_parquet(n_trades: int, n_quotes: int, tmpdir: str):
    """Generate synthetic tick data and export to parquet files."""
    progress(f"Generating {fmt_rows(n_trades)} trades + {fmt_rows(n_quotes)} quotes ...")
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(GEN_TRADES_SQL.format(n=n_trades))
    con.execute(GEN_QUOTES_SQL.format(n=n_quotes))

    trades_pq = os.path.join(tmpdir, "trades.parquet")
    quotes_pq = os.path.join(tmpdir, "quotes.parquet")
    con.execute(f"COPY trades TO '{trades_pq}' (FORMAT PARQUET, ROW_GROUP_SIZE 1000000)")
    con.execute(f"COPY quotes TO '{quotes_pq}' (FORMAT PARQUET, ROW_GROUP_SIZE 1000000)")
    progress(f"Data generated + exported in {fmt_time(time.perf_counter() - t0)}")
    return trades_pq, quotes_pq


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASOF JOIN benchmark: Sirius GPU vs DuckDB CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scale", type=int, choices=[1, 2, 3, 4], action="append",
                        dest="scales", metavar="N",
                        help="Scale factor(s) to run (1–4). May be repeated. Default: 1 2 3")
    parser.add_argument("--runs",    type=int, default=3,
                        help="Timed repetitions per scale (default: 3)")
    parser.add_argument("--no-cpu",  action="store_true", help="Skip DuckDB CPU baseline")
    parser.add_argument("--no-gpu",  action="store_true", help="Skip Sirius GPU")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip GPU warm-up iteration")
    args = parser.parse_args()

    scales = sorted(set(args.scales)) if args.scales else [1, 2, 3]
    do_warmup = not args.no_warmup

    print()
    print("=" * 65)
    print("  ASOF JOIN Benchmark — Sirius GPU vs DuckDB CPU")
    print("=" * 65)
    print(f"  Query: {JOIN_QUERY}")
    print(f"  Runs per scale: {args.runs}  |  Warmup: {do_warmup}")
    print()

    summary = []  # [(label, cpu_best, gpu_best, n_trades)]

    for scale in scales:
        n_trades, n_quotes, label = SCALES[scale]
        print(f"{'─'*65}")
        print(f"  Scale {scale}: {label}")
        print(f"{'─'*65}")

        with tempfile.TemporaryDirectory() as tmpdir:
            trades_pq, quotes_pq = generate_parquet(n_trades, n_quotes, tmpdir)
            print()

            cpu_best = gpu_best = None
            cpu_rows = gpu_rows = None

            if not args.no_cpu:
                print("  ── DuckDB CPU ──")
                cpu_best, cpu_res = run_cpu_benchmark(trades_pq, quotes_pq, args.runs)
                if cpu_res:
                    cpu_rows = cpu_res[0]
                print()

            if not args.no_gpu:
                print("  ── Sirius GPU ──")
                gpu_best, gpu_res = run_gpu_benchmark(
                    trades_pq, quotes_pq, args.runs, do_warmup)
                if gpu_res:
                    gpu_rows = gpu_res[0]
                print()

        # Per-scale summary
        if cpu_best and gpu_best:
            speedup = cpu_best / gpu_best
            print(f"  → CPU: {fmt_time(cpu_best):>10}   GPU: {fmt_time(gpu_best):>10}"
                  f"   Speedup: {speedup:.1f}×")
        elif cpu_best:
            print(f"  → CPU: {fmt_time(cpu_best):>10}")
        elif gpu_best:
            print(f"  → GPU: {fmt_time(gpu_best):>10}")

        if cpu_rows and gpu_rows and cpu_rows != gpu_rows:
            print(f"  ⚠  Row count mismatch: CPU={cpu_rows:,}  GPU={gpu_rows:,}")
        elif cpu_rows and gpu_rows:
            print(f"  ✓  Row counts match: {cpu_rows:,}")

        summary.append((label, n_trades, cpu_best, gpu_best))
        print()

    # ── Final summary table ───────────────────────────────────────────────
    if len(scales) > 1:
        print("=" * 65)
        print("  RESULTS SUMMARY")
        print("=" * 65)
        hdr = f"  {'Scale':<35} {'CPU':>10} {'GPU':>10} {'Speedup':>9}"
        print(hdr)
        print(f"  {'-'*63}")
        for (label, n_trades, cpu_t, gpu_t) in summary:
            cpu_s   = fmt_time(cpu_t) if cpu_t else "N/A"
            gpu_s   = fmt_time(gpu_t) if gpu_t else "N/A"
            speedup = f"{cpu_t/gpu_t:.1f}×" if (cpu_t and gpu_t) else "N/A"
            n_label = f"{label}"
            print(f"  {n_label:<35} {cpu_s:>10} {gpu_s:>10} {speedup:>9}")
        print()


if __name__ == "__main__":
    main()
