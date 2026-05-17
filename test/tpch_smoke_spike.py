#!/usr/bin/env python3
"""Run TPC-H 1-22 via Sirius gpu_execution + DuckDB CPU baseline, compare results.

This is a spike-validation driver, not a CI test. Reads queries from
test/tpch_performance/tpch_queries/orig/qN.sql against /tmp/tpch_sf10.duckdb.

Usage:
    python3 test/tpch_smoke_spike.py [--query N] [--batch-size 5Gi]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "build/release/extension/sirius/sirius.duckdb_extension"
DUCKDB = REPO / "build/release/duckdb"
QDIR = REPO / "test/tpch_performance/tpch_queries/orig"
DB = Path("/tmp/tpch_sf10.duckdb")


def normalize_result(raw: str) -> str:
    """Strip CLI chrome, keep just the box-drawn table block. Hash for compare."""
    lines = []
    in_table = False
    for line in raw.splitlines():
        if line.startswith("┌") or line.startswith("│") or line.startswith("├") or line.startswith("└"):
            in_table = True
            lines.append(line.rstrip())
        elif in_table and line.strip() == "":
            in_table = False
    return "\n".join(lines)


def run_cli(sql: str, env_extra: dict | None = None, timeout: int = 300,
            db: Path | None = None) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run(
        [str(DUCKDB), "-unsigned", str(db or DB)],
        input=sql,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def query_text(n: int) -> str:
    return (QDIR / f"q{n}.sql").read_text().strip().rstrip(";")


def run_gpu(qsql: str, batch_size: str | None) -> tuple[int, str, str, float]:
    settings = [
        "SET sirius_log_level='info';",
        "SET enable_gpu_duckdb_native_scan=true;",
    ]
    if batch_size:
        settings.append(f"SET scan_task_batch_size={batch_size};")
    # gpu_execution is a TABLE FUNCTION — feed the query as a string literal.
    escaped = qsql.replace("'", "''")
    sql = "\n".join(settings) + f"\nCALL gpu_execution('{escaped}');\n"
    return run_cli(sql)


def run_cpu(qsql: str) -> tuple[int, str, str, float]:
    # Sirius autoloads on attach and its teardown segfaults the CLI (rc=-11) AFTER the
    # result is already on stdout — so we ignore rc and trust the table block parse.
    return run_cli(qsql + ";\n")


def summarize_result(out: str) -> tuple[int, str, str]:
    """Return (row_count_in_result_table, first_row, hash) for compare."""
    block = normalize_result(out)
    rows = [ln for ln in block.splitlines() if ln.startswith("│")]
    # Drop header rows (the first two │-prefixed are typically the column names + types).
    data_rows = rows[2:] if len(rows) > 2 else rows
    first = data_rows[0] if data_rows else ""
    h = hashlib.sha256(block.encode()).hexdigest()[:12]
    return len(data_rows), first, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=int, default=0, help="Run a single query 1..22 (0=all)")
    ap.add_argument("--batch-size", default=None, help="e.g. '5Gi' to stress big-batch path")
    ap.add_argument("--skip", type=str, default="",
                    help="Comma-separated list of qNN to skip")
    args = ap.parse_args()

    skip = {int(s) for s in args.skip.split(",") if s.strip()}
    qs = [args.query] if args.query else range(1, 23)

    print(f"DB: {DB}  batch_size={args.batch_size or 'default'}")
    print(f"{'Q':>4}  {'GPU rc':>6} {'CPU rc':>6}  {'GPU s':>6} {'CPU s':>6}  match  notes")
    print("-" * 78)

    for q in qs:
        if q in skip:
            print(f"q{q:<3}  skipped")
            continue
        qsql = query_text(q)
        rc_gpu, out_gpu, err_gpu, t_gpu = run_gpu(qsql, args.batch_size)
        rc_cpu, out_cpu, err_cpu, t_cpu = run_cpu(qsql)
        nrows_gpu, first_gpu, h_gpu = summarize_result(out_gpu)
        nrows_cpu, first_cpu, h_cpu = summarize_result(out_cpu)
        # Sirius shutdown segfaults the CLI (-11) AFTER the result is on stdout, so we
        # rely on parsed table block, not return code.
        match = "OK" if (h_gpu == h_cpu and nrows_gpu > 0) else "DIFF"
        notes = []
        if rc_gpu not in (0, -11):
            notes.append(f"gpu rc={rc_gpu}")
        if rc_cpu not in (0, -11):
            notes.append(f"cpu rc={rc_cpu}")
        if h_gpu != h_cpu:
            notes.append(f"rows g={nrows_gpu} c={nrows_cpu}")
            notes.append(f"h g={h_gpu} c={h_cpu}")
        # Detect Sirius fallback ("Error in SiriusExecuteQuery" emitted on stdout/stderr).
        if "fallback to DuckDB" in out_gpu or "fallback to DuckDB" in err_gpu:
            notes.append("FALLBACK")
        # Detect "Not implemented" or other clear scan errors.
        for pat in ("Not implemented", "INTERNAL Error", "assertion failure", "Segmentation"):
            if pat in (out_gpu + err_gpu):
                notes.append(pat)
                break
        print(f"q{q:<3}  {rc_gpu:>6} {rc_cpu:>6}  {t_gpu:6.1f} {t_cpu:6.1f}  {match:<5}  {' / '.join(notes)}")

        # On any non-trivial failure, dump tails for the first 3 problems.
        if (match != "OK") and getattr(main, "_dumped", 0) < 3:
            main._dumped = getattr(main, "_dumped", 0) + 1
            print(f"    --- q{q} gpu stdout tail ---")
            for line in out_gpu.splitlines()[-15:]:
                print(f"    {line}")
            print(f"    --- q{q} gpu stderr tail ---")
            for line in err_gpu.splitlines()[-15:]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
