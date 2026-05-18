#!/usr/bin/env python3
"""Run TPC-H 1-22 via Sirius gpu_execution + DuckDB CPU baseline, compare results.

This is a spike-validation driver, not a CI test. Reads queries from
test/tpch_performance/tpch_queries/orig/qN.sql against the configured .duckdb
fixture (default /var/tmp/tpch_sf10.duckdb; override with --db or $TPCH_DB).

Usage:
    python3 test/tpch_smoke_spike.py [--query N] [--batch-size 5368709120]
    python3 test/tpch_smoke_spike.py --allowed-diff 1  # q1 has FP-render rounding DIFF
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DUCKDB = REPO / "build/release/duckdb"
QDIR = REPO / "test/tpch_performance/tpch_queries/orig"
# /tmp is wiped on rental cycle; /var/tmp is overlay-persisted on the rented boxes.
# Override with --db or $TPCH_DB for other fixtures.
DEFAULT_DB = Path(os.environ.get("TPCH_DB", "/var/tmp/tpch_sf10.duckdb"))


def normalize_result(raw: str) -> str:
    """Strip CLI chrome, keep just the box-drawn table block. Hash for compare."""
    lines = []
    in_table = False
    for line in raw.splitlines():
        if (
            line.startswith("┌")
            or line.startswith("│")
            or line.startswith("├")
            or line.startswith("└")
        ):
            in_table = True
            lines.append(line.rstrip())
        elif in_table and line.strip() == "":
            in_table = False
    return "\n".join(lines)


def run_cli(
    sql: str, db: Path, env_extra: dict | None = None, timeout: int = 300
) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        p = subprocess.run(
            [str(DUCKDB), "-unsigned", str(db)],
            input=sql,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        # Surface as a non-zero rc the caller can recognise; preserve any output
        # the child wrote before the timeout fired.
        return (
            -2,
            e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
            (
                e.stderr.decode()
                if isinstance(e.stderr, bytes)
                else (e.stderr or "TIMEOUT")
            ),
            time.time() - t0,
        )
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def query_text(n: int) -> str:
    return (QDIR / f"q{n}.sql").read_text().strip().rstrip(";")


def run_gpu(qsql: str, batch_size: str | None, db: Path) -> tuple[int, str, str, float]:
    settings = [
        "SET sirius_log_level='info';",
        "SET enable_gpu_duckdb_native_scan=true;",
    ]
    if batch_size:
        settings.append(f"SET scan_task_batch_size={batch_size};")
    # gpu_execution is a TABLE FUNCTION — feed the query as a string literal.
    escaped = qsql.replace("'", "''")
    sql = "\n".join(settings) + f"\nCALL gpu_execution('{escaped}');\n"
    return run_cli(sql, db=db)


def run_cpu(qsql: str, db: Path) -> tuple[int, str, str, float]:
    # Disable sirius transparent execution so we get a true DuckDB-CPU baseline.
    # Without this, sirius would intercept and run through its legacy scan path —
    # still correct, but not a pure CPU comparison. Also avoids the static-link
    # sirius teardown SIGSEGV that fires after any transparent-execution-intercepted
    # query.
    sql = "SET gpu_execution=false;\n" + qsql + ";\n"
    return run_cli(sql, db=db)


def summarize_result(out: str) -> tuple[int, str, str]:
    """Return (row_count_in_result_table, first_row, hash) for compare."""
    block = normalize_result(out)
    rows = [ln for ln in block.splitlines() if ln.startswith("│")]
    # Drop header rows (the first two │-prefixed are typically the column names + types).
    data_rows = rows[2:] if len(rows) > 2 else rows
    first = data_rows[0] if data_rows else ""
    h = hashlib.sha256(block.encode()).hexdigest()[:12]
    return len(data_rows), first, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query", type=int, default=0, help="Run a single query 1..22 (0=all)"
    )
    ap.add_argument(
        "--batch-size", default=None, help="e.g. '5368709120' to stress big-batch path"
    )
    ap.add_argument(
        "--skip", type=str, default="", help="Comma-separated list of qNN to skip"
    )
    ap.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="Path to the .duckdb fixture"
    )
    ap.add_argument(
        "--allowed-diff",
        type=str,
        default="",
        help="Comma-separated qNN whose DIFF is expected (e.g. '1' for q1 FP-render rounding)",
    )
    args = ap.parse_args()

    db = args.db
    if not db.exists():
        print(f"--db path does not exist: {db}", file=sys.stderr)
        return 2

    skip = {int(s) for s in args.skip.split(",") if s.strip()}
    allowed_diff = {int(s) for s in args.allowed_diff.split(",") if s.strip()}
    qs = [args.query] if args.query else range(1, 23)

    print(f"DB: {db}  batch_size={args.batch_size or 'default'}")
    print(
        f"{'Q':>4}  {'GPU rc':>6} {'CPU rc':>6}  {'GPU s':>6} {'CPU s':>6}  match  notes"
    )
    print("-" * 78)

    had_unexpected_diff = False
    dumped = 0
    for q in qs:
        if q in skip:
            print(f"q{q:<3}  skipped")
            continue
        qsql = query_text(q)
        rc_gpu, out_gpu, err_gpu, t_gpu = run_gpu(qsql, args.batch_size, db)
        rc_cpu, out_cpu, err_cpu, t_cpu = run_cpu(qsql, db)
        nrows_gpu, first_gpu, h_gpu = summarize_result(out_gpu)
        nrows_cpu, first_cpu, h_cpu = summarize_result(out_cpu)
        # Sirius shutdown segfaults the CLI (-11) AFTER the result is on stdout, so we
        # rely on the parsed table block, not the return code.
        is_match = h_gpu == h_cpu and nrows_gpu > 0
        if is_match:
            label = "OK"
        elif q in allowed_diff:
            label = "DIFF*"  # expected DIFF — does not fail the sweep
        else:
            label = "DIFF"
            had_unexpected_diff = True

        notes = []
        if rc_gpu == -2:
            notes.append("gpu TIMEOUT")
        elif rc_gpu not in (0, -11):
            notes.append(f"gpu rc={rc_gpu}")
        if rc_cpu == -2:
            notes.append("cpu TIMEOUT")
        elif rc_cpu not in (0, -11):
            notes.append(f"cpu rc={rc_cpu}")
        if h_gpu != h_cpu:
            notes.append(f"rows g={nrows_gpu} c={nrows_cpu}")
            notes.append(f"h g={h_gpu} c={h_cpu}")
        # Detect Sirius fallback ("Error in SiriusExecuteQuery" emitted on stdout/stderr).
        if "fallback to DuckDB" in out_gpu or "fallback to DuckDB" in err_gpu:
            notes.append("FALLBACK")
        # Detect "Not implemented" or other clear scan errors.
        for pat in (
            "Not implemented",
            "INTERNAL Error",
            "assertion failure",
            "Segmentation",
        ):
            if pat in (out_gpu + err_gpu):
                notes.append(pat)
                break
        print(
            f"q{q:<3}  {rc_gpu:>6} {rc_cpu:>6}  {t_gpu:6.1f} {t_cpu:6.1f}  {label:<5}  {' / '.join(notes)}"
        )

        # On any non-trivial (unexpected) failure, dump tails for the first 3 problems.
        if (label == "DIFF") and dumped < 3:
            dumped += 1
            print(f"    --- q{q} gpu stdout tail ---")
            for line in out_gpu.splitlines()[-15:]:
                print(f"    {line}")
            print(f"    --- q{q} gpu stderr tail ---")
            for line in err_gpu.splitlines()[-15:]:
                print(f"    {line}")

    return 1 if had_unexpected_diff else 0


if __name__ == "__main__":
    sys.exit(main())
