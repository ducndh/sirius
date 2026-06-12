#!/usr/bin/env python3
"""Gate-0: freshness-under-churn harness for Sirius (stdlib-only).

Measures the cost of keeping GPU-resident table data fresh while a CPU-side
write stream (INSERT/DELETE) churns the base table, under four policies:

  arm_a   invalidate-and-reupload : CHECKPOINT + engine restart + re-warm after
                                    every delta batch. Always fresh; pays full
                                    re-upload per batch.
  arm_b   merge-on-read (sim)     : GPU queries the stale pinned snapshot;
                                    the harness runs the same query over
                                    explicit delta tables on CPU and merges.
                                    Simulates sirius#819's MVCC delta provider
                                    at the harness level. Fresh; read cost
                                    grows with accumulated delta.
  arm_c   periodic-recompact      : arm_b + every K ticks fold the deltas
                                    (CHECKPOINT + restart + re-warm + truncate
                                    delta tables). The K knob is the policy.
  cpu     CPU-only DuckDB         : RateupDB-style baseline; no GPU at all.

plus:

  probe   documents CURRENT dev semantics: does gpu_execution serve stale
          results after un-checkpointed writes? after CHECKPOINT without
          restart? does the table_gpu cache drop on row-count change?

Pass/kill criterion for Gate-0 (see README): there must exist a realistic
churn regime where arm_b read latency degrades monotonically with accumulated
delta AND arm_a is worse — otherwise a fixed policy suffices and the cost
model has no job.

All box-specific values live in the config JSON (see config.a100.json /
config.gh200.json). Results are JSONL, one record per measurement, keyed by
(churn rate, tick, arm, query) and carrying granule statistics so the same
data can later feed the general cost model.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

SENTINEL = "G0_SENTINEL_DONE"
ROWS_PER_ROW_GROUP = 122880  # DuckDB row-group granule (config-overridable)
ROWS_PER_VECTOR = 2048  # DuckDB vector granule

# ---------------------------------------------------------------------------
# Engine wrapper: persistent duckdb CLI (Sirius statically linked)
# ---------------------------------------------------------------------------


class EngineCrash(Exception):
    pass


class SiriusEngine:
    """Drives a persistent duckdb CLI subprocess over a pipe.

    Persistence matters: the Sirius scan cache (table_gpu) lives in-process,
    so killing the process is the only guaranteed full cache invalidation —
    which is exactly what arm_a/arm_c exploit.
    """

    def __init__(self, cfg, db_path, tag=""):
        self.cfg = cfg
        self.db_path = db_path
        self.tag = tag
        self.proc = None
        self.last_engine_times = []  # 'real' seconds parsed from .timer output

    def start(self, init_sql=None, timeout=300.0):
        repo_root = self.cfg["repo_root"]
        binary = os.path.join(repo_root, self.cfg["duckdb_binary"])
        ld = os.path.join(repo_root, self.cfg["ld_library_path"])
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
        t0 = time.monotonic()
        # stdbuf: the CLI block-buffers stdout on a pipe; without line
        # buffering the sentinel never arrives and the driver deadlocks.
        self.proc = subprocess.Popen(
            ["stdbuf", "-oL", "-eL", binary, "-unsigned", self.db_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        setup = [".headers off", ".mode list", ".separator |", ".timer on"]
        for line in setup:
            self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        for stmt in init_sql or []:
            self.sql(stmt, timeout=timeout)
        return time.monotonic() - t0

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(".quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def sql(self, query, timeout=600.0):
        """Run one statement; return (rows, wall_seconds).

        rows: list of '|'-separated text rows. Raises EngineCrash on process
        death and RuntimeError on SQL errors (with full output attached).
        """
        if self.proc is None or self.proc.poll() is not None:
            raise EngineCrash(f"engine not running (tag={self.tag})")
        q = query.strip().rstrip(";")
        self.last_engine_times = []
        t0 = time.monotonic()
        self.proc.stdin.write(q + ";\n")
        self.proc.stdin.write(f"SELECT '{SENTINEL}';\n")
        self.proc.stdin.flush()
        rows, errors = [], []
        deadline = t0 + timeout
        while True:
            if time.monotonic() > deadline:
                self.proc.kill()
                raise EngineCrash(f"timeout after {timeout}s on: {q[:120]}")
            line = self.proc.stdout.readline()
            if line == "":  # EOF: process died (segfault etc.)
                raise EngineCrash(f"engine died during: {q[:120]}")
            line = line.rstrip("\n")
            if SENTINEL in line:
                break
            m = re.match(r"Run Time \(s\): real ([0-9.]+)", line)
            if m:
                self.last_engine_times.append(float(m.group(1)))
                continue
            if re.search(r"Error|error:|Exception|FATAL", line):
                errors.append(line)
                continue
            if line:
                rows.append(line)
        wall = time.monotonic() - t0
        if errors:
            raise RuntimeError(f"SQL error on [{q[:120]}]: " + " // ".join(errors[:5]))
        return rows, wall

    def gpu_sql(self, query, timeout=600.0):
        q = query.strip().rstrip(";").replace("'", "''")
        return self.sql(f"CALL gpu_execution('{q}')", timeout=timeout)


def gpu_mem_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

# Decomposable-aggregate query set: every query must be mergeable as
#   result(main_stale) (+) result(delta_ins) (-) result(delta_del)
# which restricts aggregates to SUM/COUNT (AVG derived). This mirrors what a
# real merge-on-read operator could do cheaply and keeps arm_b honest.

QUERIES = {
    "q_count": "SELECT count(*) AS c FROM {t}",
    "q_sum_filter": (
        "SELECT sum(price * (1 - disc)) AS s, count(*) AS c FROM {t} WHERE k < {kmax}"
    ),
    "q_group": (
        "SELECT grp, sum(qty) AS s, count(*) AS c FROM {t} GROUP BY grp ORDER BY grp"
    ),
}


def parse_rows(query_key, rows):
    """Parse '|'-separated rows into {group_key: [sums...]} ; scalar queries
    use the fixed key '_'. NULL columns (empty string) parse as 0."""
    out = {}

    def num(x):
        if x == "" or x == "NULL":
            return 0.0
        return float(x)

    for r in rows:
        parts = r.split("|")
        if query_key == "q_group":
            out[parts[0]] = [num(p) for p in parts[1:]]
        else:
            out["_"] = [num(p) for p in parts]
    return out


def merge_results(stale, ins, dele):
    """result = stale + ins - dele, per group key, per aggregate column."""
    keys = set(stale) | set(ins) | set(dele)
    out = {}
    for k in keys:
        ncols = max(
            len(stale.get(k, [])), len(ins.get(k, [])), len(dele.get(k, []))
        )

        def col(d, i):
            v = d.get(k)
            return v[i] if v is not None and i < len(v) else 0.0

        out[k] = [col(stale, i) + col(ins, i) - col(dele, i) for i in range(ncols)]
    # drop groups that net to zero count (fully-deleted groups)
    return {k: v for k, v in out.items() if any(abs(x) > 1e-9 for x in v)}


def results_match(a, b, rtol=1e-6):
    if set(a) != set(b):
        return False
    for k in a:
        for x, y in zip(a[k], b[k]):
            if abs(x - y) > rtol * max(1.0, abs(x), abs(y)):
                return False
    return True


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------

SCHEMA_COLS = "id BIGINT, grp INTEGER, k INTEGER, qty DOUBLE, price DOUBLE, disc DOUBLE, flag SMALLINT, ts BIGINT"
GEN_EXPR = (
    "SELECT range AS id, (range % 25)::INTEGER AS grp, "
    "(hash(range) % 1000000)::INTEGER AS k, "
    "1.0 + (hash(range + 7) % 50)::DOUBLE AS qty, "
    "(hash(range + 13) % 10000)::DOUBLE / 100.0 AS price, "
    "(hash(range + 17) % 10)::DOUBLE / 100.0 AS disc, "
    "(range % 3)::SMALLINT AS flag, range AS ts "
    "FROM range({lo}, {hi})"
)


def phase_setup(cfg, args):
    """Create the master DB template: churn table [0, N) + incoming reservoir
    [N, N + N*incoming_frac). Checkpointed and closed clean. Sweep cells copy
    this file so every cell starts identical."""
    data_dir = cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    master = os.path.join(data_dir, "gate0_master.duckdb")
    for p in (master, master + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    n = cfg["scale"]["n_rows"]
    n_inc = int(n * cfg["scale"]["incoming_frac"])
    eng = SiriusEngine(cfg, master, tag="setup")
    eng.start()
    print(f"[setup] generating churn table: {n:,} rows", flush=True)
    eng.sql(f"CREATE TABLE churn ({SCHEMA_COLS})")
    eng.sql("INSERT INTO churn " + GEN_EXPR.format(lo=0, hi=n), timeout=3600)
    print(f"[setup] generating incoming reservoir: {n_inc:,} rows", flush=True)
    eng.sql(f"CREATE TABLE incoming ({SCHEMA_COLS})")
    eng.sql("INSERT INTO incoming " + GEN_EXPR.format(lo=n, hi=n + n_inc), timeout=3600)
    # delta bookkeeping tables for arm_b/arm_c (empty in the template)
    eng.sql(f"CREATE TABLE delta_ins ({SCHEMA_COLS})")
    eng.sql(f"CREATE TABLE delta_del ({SCHEMA_COLS})")
    eng.sql("CHECKPOINT")
    eng.stop()
    size = os.path.getsize(master)
    print(f"[setup] master template ready: {master} ({size / 1e9:.2f} GB)")
    return master


# ---------------------------------------------------------------------------
# Churn application
# ---------------------------------------------------------------------------


class ChurnState:
    """Tracks the live id window. FIFO mode: delete oldest batch, insert
    newest batch (clustered writes — append-heavy realism). Uniform mode:
    delete a uniform random sample (adversarial scatter), insert newest."""

    def __init__(self, n_rows, mode):
        self.low = 0  # smallest live id (fifo)
        self.hi = n_rows  # next id to insert from the reservoir
        self.n0 = n_rows
        self.mode = mode

    def tick_sql(self, batch, track_delta):
        """Returns (list of SQL stmts, descriptor). track_delta also records
        the written rows into delta_ins/delta_del for merge-on-read arms."""
        stmts = []
        if self.mode == "fifo":
            del_pred = f"id >= {self.low} AND id < {self.low + batch}"
            self.low += batch
        elif self.mode == "uniform":
            stmts.append(
                "CREATE OR REPLACE TEMP TABLE _victims AS "
                f"SELECT id FROM churn USING SAMPLE {batch} ROWS (system, {self.hi % 1000})"
            )
            del_pred = "id IN (SELECT id FROM _victims)"
        else:
            raise ValueError(self.mode)
        ins_pred = f"id >= {self.hi} AND id < {self.hi + batch}"
        self.hi += batch
        if track_delta:
            stmts.append(f"INSERT INTO delta_del SELECT * FROM churn WHERE {del_pred}")
            stmts.append(f"INSERT INTO delta_ins SELECT * FROM incoming WHERE {ins_pred}")
        stmts.append(f"DELETE FROM churn WHERE {del_pred}")
        stmts.append(f"INSERT INTO churn SELECT * FROM incoming WHERE {ins_pred}")
        return stmts, {"del_pred": del_pred, "ins_pred": ins_pred}


def granule_stats(eng, n0):
    """Dirty-granule statistics from the delta tables. id ≈ insertion order ≈
    physical order for our generated table, so id // rows_per_granule
    approximates the granule index (documented approximation)."""
    out = {}
    for g, label in ((ROWS_PER_ROW_GROUP, "rowgroup"), (ROWS_PER_VECTOR, "vector")):
        rows, _ = eng.sql(
            f"SELECT count(DISTINCT id // {g}) FROM delta_del", timeout=120
        )
        dirty_del = int(rows[0]) if rows else 0
        out[f"dirty_{label}s_del"] = dirty_del
        out[f"total_{label}s"] = (n0 + g - 1) // g
    rows, _ = eng.sql("SELECT count(*) FROM delta_ins", timeout=120)
    out["delta_ins_rows"] = int(rows[0]) if rows else 0
    rows, _ = eng.sql("SELECT count(*) FROM delta_del", timeout=120)
    out["delta_del_rows"] = int(rows[0]) if rows else 0
    return out


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------


class Log:
    def __init__(self, path, base):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
        self.base = base

    def rec(self, **kw):
        d = dict(self.base)
        d.update(kw)
        self.f.write(json.dumps(d) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def fresh_cell_db(cfg, master, cell_tag):
    db = os.path.join(cfg["data_dir"], f"gate0_{cell_tag}.duckdb")
    for p in (db, db + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(master, db)
    return db


def queries_for(cfg):
    kmax = int(1000000 * cfg["workload"]["filter_selectivity"])
    return {
        name: QUERIES[name].replace("{kmax}", str(kmax))
        for name in cfg["workload"]["queries"]
    }


def run_reads(eng, qset, table, mode, log, timeout=600.0, **tags):
    """One read pass: each query once, via GPU or CPU. Returns parsed results."""
    parsed = {}
    for qname, qtpl in qset.items():
        q = qtpl.replace("{t}", table)
        try:
            if mode == "gpu":
                rows, wall = eng.gpu_sql(q, timeout=timeout)
            else:
                rows, wall = eng.sql(q, timeout=timeout)
            eng_t = eng.last_engine_times[-1] if eng.last_engine_times else None
            parsed[qname] = parse_rows(qname, rows)
            log.rec(event="read", query=qname, mode=mode, wall_s=wall,
                    engine_s=eng_t, error=None, **tags)
        except (RuntimeError, EngineCrash) as e:
            log.rec(event="read", query=qname, mode=mode, wall_s=None,
                    engine_s=None, error=str(e)[:300], **tags)
            if isinstance(e, EngineCrash):
                raise
    return parsed


# ---------------------------------------------------------------------------
# Probe phase: document current dev semantics
# ---------------------------------------------------------------------------


def phase_probe(cfg, master, log):
    """Empirically answer, on current dev:
    P1: after warming the table_gpu cache, do un-checkpointed writes change
        GPU results? (stale serve / fresh / error / crash)
    P2: does CHECKPOINT (no restart) change GPU results?
    P3: does engine restart change GPU results? (it must — cache is in-proc)
    Each step diffs GPU vs CPU on q_count.
    """
    db = fresh_cell_db(cfg, master, "probe")
    eng = SiriusEngine(cfg, db, tag="probe")
    eng.start(init_sql=cfg["engine_settings"])
    qset = queries_for(cfg)
    state = ChurnState(cfg["scale"]["n_rows"], "fifo")
    batch = max(1, int(cfg["scale"]["n_rows"] * 0.001))

    def snap(stage):
        out = {}
        for mode in ("gpu", "cpu"):
            try:
                if mode == "gpu":
                    rows, _ = eng.gpu_sql(qset["q_count"].replace("{t}", "churn"))
                else:
                    rows, _ = eng.sql(qset["q_count"].replace("{t}", "churn"))
                out[mode] = rows[0] if rows else "EMPTY"
            except (RuntimeError, EngineCrash) as e:
                out[mode] = f"ERROR: {str(e)[:160]}"
                if isinstance(e, EngineCrash):
                    eng.stop()
                    eng.start(init_sql=cfg["engine_settings"])
        verdict = "FRESH" if out.get("gpu") == out.get("cpu") else "STALE_OR_WRONG"
        if str(out.get("gpu", "")).startswith("ERROR"):
            verdict = "ERROR"
        log.rec(event="probe", stage=stage, gpu=out.get("gpu"),
                cpu=out.get("cpu"), verdict=verdict)
        print(f"[probe] {stage:34s} gpu={out.get('gpu')} cpu={out.get('cpu')} -> {verdict}",
              flush=True)
        return out

    snap("baseline_warm")  # warms the cache
    snap("cache_hit_check")  # second read: should hit cache, same answer
    for stmt, _ in [(s, None) for s in state.tick_sql(batch, track_delta=False)[0]]:
        eng.sql(stmt, timeout=600)
    snap("P1_after_uncheckpointed_writes")
    eng.sql("CHECKPOINT", timeout=600)
    snap("P2_after_checkpoint_no_restart")
    eng.stop()
    eng.start(init_sql=cfg["engine_settings"])
    snap("P3_after_engine_restart")
    eng.stop()


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def run_arm(cfg, master, arm, rho, log):
    """Run one (arm, rho) cell: T ticks of [write batch -> reads]."""
    n0 = cfg["scale"]["n_rows"]
    ticks = cfg["workload"]["ticks"]
    batch = max(1, int(n0 * rho))
    mode = cfg["churn"]["mode"]
    track_delta = arm in ("arm_b", "arm_c")
    recompact_every = cfg.get("recompact_every", 5)
    cell = f"{arm}_rho{rho}"
    db = fresh_cell_db(cfg, master, cell)
    qset = queries_for(cfg)
    eng = SiriusEngine(cfg, db, tag=cell)
    start_s = eng.start(init_sql=cfg["engine_settings"])
    log.rec(event="engine_start", arm=arm, rho=rho, wall_s=start_s)
    state = ChurnState(n0, mode)
    read_mode = "cpu" if arm == "cpu" else "gpu"

    # initial warm (the pin epoch) — timed: this is the upload+decode cost
    if read_mode == "gpu":
        warm = run_reads(eng, qset, "churn", "gpu", log, event_tag="warm",
                         arm=arm, rho=rho, tick=-1)
        log.rec(event="gpu_mem", arm=arm, rho=rho, tick=-1, mib=gpu_mem_used_mib())

    for tick in range(ticks):
        # -- write batch --
        stmts, _ = state.tick_sql(batch, track_delta)
        t0 = time.monotonic()
        for s in stmts:
            eng.sql(s, timeout=1800)
        log.rec(event="write", arm=arm, rho=rho, tick=tick, rows=batch,
                wall_s=time.monotonic() - t0)

        # -- policy action --
        if arm == "arm_a":
            t0 = time.monotonic()
            eng.sql("CHECKPOINT", timeout=1800)
            ckpt_s = time.monotonic() - t0
            eng.stop()
            restart_s = eng.start(init_sql=cfg["engine_settings"])
            rewarm = run_reads(eng, qset, "churn", "gpu", log, event_tag="rewarm",
                               arm=arm, rho=rho, tick=tick)
            log.rec(event="refresh", arm=arm, rho=rho, tick=tick,
                    checkpoint_s=ckpt_s, restart_s=restart_s)
        elif arm == "arm_c" and tick > 0 and tick % recompact_every == 0:
            t0 = time.monotonic()
            eng.sql("CHECKPOINT", timeout=1800)
            eng.sql("DELETE FROM delta_ins", timeout=600)
            eng.sql("DELETE FROM delta_del", timeout=600)
            ckpt_s = time.monotonic() - t0
            eng.stop()
            restart_s = eng.start(init_sql=cfg["engine_settings"])
            run_reads(eng, qset, "churn", "gpu", log, event_tag="rewarm",
                      arm=arm, rho=rho, tick=tick)
            log.rec(event="recompact", arm=arm, rho=rho, tick=tick,
                    checkpoint_s=ckpt_s, restart_s=restart_s)

        # -- reads --
        try:
            if arm in ("arm_b", "arm_c"):
                # stale GPU snapshot + CPU delta queries + harness merge
                for qname, qtpl in qset.items():
                    q_main = qtpl.replace("{t}", "churn")
                    t0 = time.monotonic()
                    rows, w_gpu = eng.gpu_sql(q_main, timeout=600)
                    stale = parse_rows(qname, rows)
                    t1 = time.monotonic()
                    ri, _ = eng.sql(qtpl.replace("{t}", "delta_ins"), timeout=600)
                    rd, _ = eng.sql(qtpl.replace("{t}", "delta_del"), timeout=600)
                    t2 = time.monotonic()
                    merged = merge_results(
                        stale, parse_rows(qname, ri), parse_rows(qname, rd)
                    )
                    t3 = time.monotonic()
                    log.rec(event="read", query=qname, mode="merge_on_read",
                            arm=arm, rho=rho, tick=tick,
                            wall_s=t3 - t0, gpu_s=t1 - t0,
                            delta_s=t2 - t1, merge_s=t3 - t2, error=None)
                    # correctness: merged must equal CPU truth
                    if tick % cfg["workload"].get("verify_every", 1) == 0:
                        rt, _ = eng.sql(q_main, timeout=600)
                        truth = parse_rows(qname, rt)
                        ok = results_match(merged, truth)
                        log.rec(event="verify", query=qname, arm=arm, rho=rho,
                                tick=tick, ok=ok)
                        if not ok:
                            print(f"[{cell}] VERIFY FAIL {qname} tick={tick}",
                                  flush=True)
            else:
                run_reads(eng, qset, "churn", read_mode, log,
                          arm=arm, rho=rho, tick=tick)
        except EngineCrash as e:
            log.rec(event="crash", arm=arm, rho=rho, tick=tick, error=str(e)[:300])
            print(f"[{cell}] ENGINE CRASH tick={tick}: {e}", flush=True)
            eng.stop()
            eng.start(init_sql=cfg["engine_settings"])

        # -- per-tick stats --
        if track_delta:
            try:
                log.rec(event="granules", arm=arm, rho=rho, tick=tick,
                        **granule_stats(eng, n0))
            except (RuntimeError, EngineCrash):
                pass
        log.rec(event="gpu_mem", arm=arm, rho=rho, tick=tick,
                mib=gpu_mem_used_mib())
        print(f"[{cell}] tick {tick + 1}/{ticks} done", flush=True)

    eng.stop()
    os.remove(db)  # cells are throwaway copies; master is the template


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True,
                    choices=["setup", "probe", "arm_a", "arm_b", "arm_c", "cpu", "sweep"])
    ap.add_argument("--rho", type=float, default=None,
                    help="churn fraction per tick (single-arm phases)")
    ap.add_argument("--ticks", type=int, default=None, help="override config ticks")
    ap.add_argument("--out", default=None, help="JSONL output path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if cfg.get("repo_root") in (None, "", "auto"):
        cfg["repo_root"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    if args.ticks:
        cfg["workload"]["ticks"] = args.ticks

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "results", f"{args.phase}.jsonl")
    base = {
        "box": cfg["box"]["name"],
        "interconnect": cfg["box"]["interconnect"],
        "n_rows": cfg["scale"]["n_rows"],
        "churn_mode": cfg["churn"]["mode"],
        "phase": args.phase,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log = Log(out, base)

    master = os.path.join(cfg["data_dir"], "gate0_master.duckdb")
    if args.phase == "setup":
        phase_setup(cfg, args)
        return
    if not os.path.exists(master):
        sys.exit(f"master template missing ({master}) — run --phase setup first")

    if args.phase == "probe":
        phase_probe(cfg, master, log)
    elif args.phase in ("arm_a", "arm_b", "arm_c", "cpu"):
        rho = args.rho if args.rho is not None else cfg["churn"]["rhos"][0]
        run_arm(cfg, master, args.phase, rho, log)
    elif args.phase == "sweep":
        for rho in cfg["churn"]["rhos"]:
            for arm in cfg.get("sweep_arms", ["cpu", "arm_a", "arm_b", "arm_c"]):
                print(f"=== sweep cell: {arm} rho={rho} ===", flush=True)
                try:
                    run_arm(cfg, master, arm, rho, log)
                except Exception as e:  # cell isolation: log and continue
                    log.rec(event="cell_abort", arm=arm, rho=rho, error=str(e)[:300])
                    print(f"=== cell ABORTED: {arm} rho={rho}: {e}", flush=True)
    log.close()
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
