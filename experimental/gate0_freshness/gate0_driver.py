#!/usr/bin/env python3
"""Gate-0: freshness-under-churn harness for Sirius (stdlib-only).

Measures the cost of keeping GPU-resident table data fresh while a CPU-side
write stream (INSERT/DELETE/UPDATE) churns the base table, under four
policies:

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

Datasets:

  tpch    TPC-H with the OFFICIAL refresh streams: dbgen -U generates
          lineitem.tbl.uN / orders.tbl.uN (RF1 inserts) and delete.N orderkey
          lists (RF2 deletes). One refresh pair churns SF*1500 orders = 0.1%
          of the orders table, so rho maps to refresh pairs/tick. An optional
          UPDATE stream (sampled l_quantity/l_extendedprice changes) is
          modeled as delete(old)+insert(new), HANA-style. Reads = Q1, Q6
          (lineitem-linear, decomposable -> mergeable for arm_b) + count.
  synth   deterministic 8-column table; churn shapes `fifo` (clustered,
          sliding window) and `uniform` (scattered deletes — adversarial for
          granular invalidation; dirty-granule fraction follows 1-(1-1/G)^U).
          This is the controlled-clustering instrument; tpch is the
          recognizable realistic workload.

Pass/kill criterion for Gate-0 (see README): there must exist a realistic
churn regime where arm_b read latency degrades monotonically with accumulated
delta AND arm_a is worse — otherwise a fixed policy suffices and the cost
model has no job.

All box-specific values live in the config JSON. Results are JSONL, one
record per measurement, keyed by (dataset, churn rate, tick, arm, query) and
carrying granule statistics so the same data can later feed the general cost
model.
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
ROWS_PER_ROW_GROUP = 122880  # DuckDB row-group granule
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
            ["stdbuf", "-oL", "-eL", binary, "-unsigned", "-init", "/dev/null",
             self.db_path],
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
        death and RuntimeError on SQL errors (with output attached).
        """
        if self.proc is None or self.proc.poll() is not None:
            raise EngineCrash(f"engine not running (tag={self.tag})")
        q = query.strip().rstrip(";")
        self.last_engine_times = []
        self.last_fallback = False
        t0 = time.monotonic()
        self.proc.stdin.write(q + ";\n")
        # NB: .print, not SELECT '...': on current dev, constant SELECTs
        # (no FROM) return 0 rows through this CLI, so a SELECT sentinel
        # never prints and the driver would hang (observed 2026-06-12).
        self.proc.stdin.write(f".print {SENTINEL}\n")
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
            if "fallback to DuckDB" in line:
                # Sirius completed the query on CPU; rows still follow.
                # Tagged, not fatal — but arm_b merges become invalid (the
                # fallback result is FRESH, not the stale snapshot).
                self.last_fallback = True
                continue
            if re.match(r"^[=\-─━┌┐└┘├┤│╞╪╡\s]+$", line):
                continue  # table-border decoration some print paths emit
            if re.search(r"Error|error:|Exception|FATAL", line):
                errors.append(line)
                continue
            if line:
                rows.append(line)
        wall = time.monotonic() - t0
        if errors:
            msg = f"SQL error on [{q[:120]}]: " + " // ".join(errors[:5])
            # a FATAL invalidates the whole database; only a restart recovers
            if "invalidated" in msg or "FATAL" in msg:
                raise EngineCrash(msg)
            raise RuntimeError(msg)
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
# Result parsing / merge (decomposable aggregates only)
# ---------------------------------------------------------------------------
# Every mergeable query must satisfy: result = Q(stale) + Q(delta_ins)
# - Q(delta_del), per group, per aggregate column. That holds for any query
# LINEAR in the churned table (it appears once) with SUM/COUNT aggregates
# (AVG derived). This mirrors what a real merge-on-read operator could do
# cheaply and keeps arm_b honest.


def parse_rows(key_cols, rows):
    """'|'-separated rows -> {group_key: [aggregate floats]}.
    key_cols=0 -> single key '_'. NULL/empty parse as 0."""
    out = {}

    def num(x):
        if x in ("", "NULL"):
            return 0.0
        return float(x)

    for r in rows:
        parts = r.split("|")
        key = "|".join(parts[:key_cols]) if key_cols else "_"
        out[key] = [num(p) for p in parts[key_cols:]]
    return out


def merge_results(stale, ins, dele):
    keys = set(stale) | set(ins) | set(dele)
    out = {}
    for k in keys:
        ncols = max(len(stale.get(k, [])), len(ins.get(k, [])), len(dele.get(k, [])))

        def col(d, i):
            v = d.get(k)
            return v[i] if v is not None and i < len(v) else 0.0

        out[k] = [col(stale, i) + col(ins, i) - col(dele, i) for i in range(ncols)]
    # drop groups whose aggregates net to zero (fully-deleted groups)
    return {k: v for k, v in out.items() if any(abs(x) > 1e-6 for x in v)}


def results_match(a, b, rtol=1e-6):
    if set(a) != set(b):
        return False
    for k in a:
        for x, y in zip(a[k], b[k]):
            if abs(x - y) > rtol * max(1.0, abs(x), abs(y)):
                return False
    return True


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SynthDataset:
    """Deterministic 8-column table + incoming reservoir. Controlled churn
    shapes: fifo (clustered) / uniform (scattered)."""

    table = "churn"
    SCHEMA = ("id BIGINT, grp INTEGER, k INTEGER, qty DOUBLE, price DOUBLE, "
              "disc DOUBLE, flag SMALLINT, ts BIGINT")
    GEN = ("SELECT range AS id, (range % 25)::INTEGER AS grp, "
           "(hash(range) % 1000000)::INTEGER AS k, "
           "1.0 + (hash(range + 7) % 50)::DOUBLE AS qty, "
           "(hash(range + 13) % 10000)::DOUBLE / 100.0 AS price, "
           "(hash(range + 17) % 10)::DOUBLE / 100.0 AS disc, "
           "(range % 3)::SMALLINT AS flag, range AS ts FROM range({lo}, {hi})")

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg["dataset"]
        self.n0 = d["n_rows"]
        self.mode = d.get("churn_shape", "fifo")
        self.low = 0
        self.hi = self.n0

    def queries(self):
        kmax = int(1000000 * self.cfg["dataset"].get("filter_selectivity", 0.1))
        return {
            "q_count": {"sql": "SELECT count(*) FROM {t}", "key_cols": 0},
            "q_sum_filter": {
                "sql": f"SELECT sum(price * (1 - disc)), count(*) FROM {{t}} WHERE k < {kmax}",
                "key_cols": 0,
            },
            "q_group": {
                "sql": "SELECT grp, sum(qty), count(*) FROM {t} GROUP BY grp ORDER BY grp",
                "key_cols": 1,
            },
        }

    def setup(self, eng, log):
        n_inc = int(self.n0 * self.cfg["dataset"].get("incoming_frac", 0.6))
        print(f"[setup] synth churn table: {self.n0:,} rows", flush=True)
        eng.sql(f"CREATE TABLE churn ({self.SCHEMA})")
        eng.sql("INSERT INTO churn " + self.GEN.format(lo=0, hi=self.n0), timeout=3600)
        eng.sql(f"CREATE TABLE incoming ({self.SCHEMA})")
        eng.sql("INSERT INTO incoming " + self.GEN.format(lo=self.n0, hi=self.n0 + n_inc),
                timeout=3600)
        eng.sql(f"CREATE TABLE delta_ins ({self.SCHEMA})")
        eng.sql(f"CREATE TABLE delta_del ({self.SCHEMA})")

    def reset_state(self):
        self.low = 0
        self.hi = self.n0

    def tick_stmts(self, tick, rho, track_delta):
        batch = max(1, int(self.n0 * rho))
        stmts = []
        if self.mode == "fifo":
            del_pred = f"id >= {self.low} AND id < {self.low + batch}"
            self.low += batch
        elif self.mode == "uniform":
            stmts.append(
                "CREATE OR REPLACE TEMP TABLE _victims AS SELECT id FROM churn "
                f"USING SAMPLE {batch} ROWS (reservoir, {tick + 1})"
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
        return stmts, batch

    # id ~ insertion order ~ physical order, so id // granule approximates the
    # granule index (documented approximation).
    granule_expr = "id"
    granule_scale = 1
    # content-sensitive: sum(id) shifts under equal-size FIFO churn where
    # count(*) cannot distinguish stale from fresh.
    probe_query = "SELECT count(*), sum(id) FROM churn"
    # columns the query set touches — the pin exports/pins only these
    pin_cols = ["id", "grp", "k", "qty", "price", "disc"]


class TpchDataset:
    """TPC-H base tables + OFFICIAL refresh streams (dbgen -U).

    Tick = N refresh pairs: RF1 inserts orders.tbl.uK + lineitem.tbl.uK,
    RF2 deletes the delete.K orderkeys (orders + cascading lineitems), plus an
    optional UPDATE stream on sampled lineitem rows (delete+insert model).
    One pair churns SF*1500 orders = 0.1% of orders, so
    pairs_per_tick = round(rho / 0.001). Reads are lineitem-linear ->
    mergeable for arm_b. RF2 deletes cluster at the low-orderkey end (oldest
    data), matching the spec's sliding-window intent.
    """

    table = "lineitem"
    LCOLS = ("l_orderkey BIGINT, l_partkey BIGINT, l_suppkey BIGINT, "
             "l_linenumber INTEGER, l_quantity DOUBLE, l_extendedprice DOUBLE, "
             "l_discount DOUBLE, l_tax DOUBLE, l_returnflag VARCHAR, "
             "l_linestatus VARCHAR, l_shipdate DATE, l_commitdate DATE, "
             "l_receiptdate DATE, l_shipinstruct VARCHAR, l_shipmode VARCHAR, "
             "l_comment VARCHAR")
    OCOLS = ("o_orderkey BIGINT, o_custkey BIGINT, o_orderstatus VARCHAR, "
             "o_totalprice DOUBLE, o_orderdate DATE, o_orderpriority VARCHAR, "
             "o_clerk VARCHAR, o_shippriority INTEGER, o_comment VARCHAR")

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg["dataset"]
        self.sf = d["sf"]
        self.refresh_sets = d["refresh_sets"]
        self.update_rows_per_tick = d.get("update_rows_per_tick", 0)
        self.next_set = 1

    def queries(self):
        return {
            "q_count": {"sql": "SELECT count(*) FROM {t}", "key_cols": 0},
            "q1": {
                "sql": ("SELECT l_returnflag, l_linestatus, sum(l_quantity), "
                        "sum(l_extendedprice), sum(l_extendedprice * (1 - l_discount)), "
                        "count(*) FROM {t} WHERE l_shipdate <= DATE '1998-09-02' "
                        "GROUP BY l_returnflag, l_linestatus "
                        "ORDER BY l_returnflag, l_linestatus"),
                "key_cols": 2,
            },
            "q6": {
                "sql": ("SELECT sum(l_extendedprice * l_discount), count(*) FROM {t} "
                        "WHERE l_shipdate >= DATE '1994-01-01' "
                        "AND l_shipdate < DATE '1995-01-01' "
                        "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"),
                "key_cols": 0,
            },
        }

    def _dbgen(self):
        dbgen_dir = os.path.join(self.cfg["repo_root"], "test_datasets", "tpch-dbgen")
        if not os.path.isfile(os.path.join(dbgen_dir, "dbgen")):
            zip_path = os.path.join(self.cfg["repo_root"], "test_datasets", "tpch-dbgen.zip")
            subprocess.run(["unzip", "-n", "-q", zip_path],
                           cwd=os.path.dirname(zip_path), check=True)
        gen_dir = os.path.join(self.cfg["data_dir"], f"tpch_sf{self.sf}_gen")
        os.makedirs(gen_dir, exist_ok=True)
        env = dict(os.environ, DSS_PATH=gen_dir)
        if not os.path.isfile(os.path.join(gen_dir, "lineitem.tbl")):
            print(f"[setup] dbgen -s {self.sf} (base tables)...", flush=True)
            subprocess.run(["./dbgen", "-f", "-q", "-s", str(self.sf), "-T", "L"],
                           cwd=dbgen_dir, env=env, check=True)
            subprocess.run(["./dbgen", "-f", "-q", "-s", str(self.sf), "-T", "O"],
                           cwd=dbgen_dir, env=env, check=True)
        if not os.path.isfile(os.path.join(gen_dir, f"delete.{self.refresh_sets}")):
            print(f"[setup] dbgen -U {self.refresh_sets} (refresh streams)...", flush=True)
            subprocess.run(["./dbgen", "-f", "-q", "-s", str(self.sf),
                            "-U", str(self.refresh_sets)],
                           cwd=dbgen_dir, env=env, check=True)
        return gen_dir

    def _load_tbl(self, eng, table, cols_ddl, path, extra_cols="", extra_vals=""):
        names = [c.strip().split(" ")[0] for c in cols_ddl.split(",")]
        collist = ", ".join(names)
        # .tbl files end each row with a trailing '|' -> read positionally and
        # select only the real columns.
        coldefs = ", ".join(f"'c{i}': '{c.strip().split(' ')[1]}'"
                            for i, c in enumerate(cols_ddl.split(",")))
        sel = ", ".join(f"c{i}" for i in range(len(names)))
        eng.sql(
            f"INSERT INTO {table} SELECT {extra_vals}{sel} FROM read_csv('{path}', "
            f"delim='|', header=false, columns={{{coldefs}, 'trail': 'VARCHAR'}}, "
            f"null_padding=true)",
            timeout=3600,
        )

    def setup(self, eng, log):
        gen = self._dbgen()
        print("[setup] loading TPC-H base tables...", flush=True)
        eng.sql(f"CREATE TABLE lineitem ({self.LCOLS})")
        eng.sql(f"CREATE TABLE orders ({self.OCOLS})")
        self._load_tbl(eng, "lineitem", self.LCOLS, os.path.join(gen, "lineitem.tbl"))
        self._load_tbl(eng, "orders", self.OCOLS, os.path.join(gen, "orders.tbl"))
        print("[setup] loading refresh streams...", flush=True)
        eng.sql(f"CREATE TABLE rf_lineitem (rf_set INTEGER, {self.LCOLS})")
        eng.sql(f"CREATE TABLE rf_orders (rf_set INTEGER, {self.OCOLS})")
        eng.sql("CREATE TABLE rf_delete (rf_set INTEGER, orderkey BIGINT)")
        for i in range(1, self.refresh_sets + 1):
            self._load_tbl(eng, "rf_lineitem", self.LCOLS,
                           os.path.join(gen, f"lineitem.tbl.u{i}"),
                           extra_vals=f"{i}, ")
            self._load_tbl(eng, "rf_orders", self.OCOLS,
                           os.path.join(gen, f"orders.tbl.u{i}"),
                           extra_vals=f"{i}, ")
            # delete.N: one orderkey per line (also trailing '|')
            eng.sql(
                f"INSERT INTO rf_delete SELECT {i}, c0 FROM read_csv("
                f"'{os.path.join(gen, f'delete.{i}')}', delim='|', header=false, "
                f"columns={{'c0': 'BIGINT', 'trail': 'VARCHAR'}}, null_padding=true)",
                timeout=600,
            )
            if i % 50 == 0:
                print(f"[setup] refresh sets loaded: {i}/{self.refresh_sets}", flush=True)
        eng.sql(f"CREATE TABLE delta_ins ({self.LCOLS})")
        eng.sql(f"CREATE TABLE delta_del ({self.LCOLS})")

    def reset_state(self):
        self.next_set = 1

    def tick_stmts(self, tick, rho, track_delta):
        pairs = max(1, round(rho / 0.001))
        a, b = self.next_set, self.next_set + pairs - 1
        if b > self.refresh_sets:
            raise RuntimeError(
                f"refresh streams exhausted (need set {b}, have {self.refresh_sets}) "
                f"— regenerate with a larger dataset.refresh_sets")
        self.next_set = b + 1
        del_keys = f"(SELECT orderkey FROM rf_delete WHERE rf_set BETWEEN {a} AND {b})"
        lcols = ", ".join(c.strip().split(" ")[0] for c in self.LCOLS.split(","))
        ocols = ", ".join(c.strip().split(" ")[0] for c in self.OCOLS.split(","))
        stmts = []
        if track_delta:
            stmts.append(f"INSERT INTO delta_del SELECT * FROM lineitem "
                         f"WHERE l_orderkey IN {del_keys}")
            stmts.append(f"INSERT INTO delta_ins SELECT {lcols} FROM rf_lineitem "
                         f"WHERE rf_set BETWEEN {a} AND {b}")
        # RF2: delete oldest orders + cascading lineitems
        stmts.append(f"DELETE FROM lineitem WHERE l_orderkey IN {del_keys}")
        stmts.append(f"DELETE FROM orders WHERE o_orderkey IN {del_keys}")
        # RF1: insert new orders + lineitems
        stmts.append(f"INSERT INTO orders SELECT {ocols} FROM rf_orders "
                     f"WHERE rf_set BETWEEN {a} AND {b}")
        stmts.append(f"INSERT INTO lineitem SELECT {lcols} FROM rf_lineitem "
                     f"WHERE rf_set BETWEEN {a} AND {b}")
        # UPDATE stream (not in the TPC-H spec): sampled rows, modeled as
        # delete(old)+insert(new) for the delta bookkeeping, HANA-style.
        u = self.update_rows_per_tick
        if u > 0:
            stmts.append(
                "CREATE OR REPLACE TEMP TABLE _upd AS SELECT l_orderkey, l_linenumber "
                f"FROM lineitem USING SAMPLE {u} ROWS (reservoir, {tick + 1})")
            if track_delta:
                stmts.append("INSERT INTO delta_del SELECT l.* FROM lineitem l "
                             "JOIN _upd u USING (l_orderkey, l_linenumber)")
            stmts.append("UPDATE lineitem SET l_quantity = l_quantity + 1, "
                         "l_extendedprice = l_extendedprice * 1.01 "
                         "WHERE EXISTS (SELECT 1 FROM _upd u WHERE "
                         "u.l_orderkey = lineitem.l_orderkey AND "
                         "u.l_linenumber = lineitem.l_linenumber)")
            if track_delta:
                stmts.append("INSERT INTO delta_ins SELECT l.* FROM lineitem l "
                             "JOIN _upd u USING (l_orderkey, l_linenumber)")
        approx_rows = pairs * self.sf * 1500 * 4 + u  # ~4 lineitems/order
        return stmts, approx_rows

    # lineitem physical order ~ orderkey order, ~4 lines/order: orderkey*4
    # approximates row position for dirty-granule estimation.
    granule_expr = "l_orderkey"
    granule_scale = 4
    probe_query = "SELECT count(*), sum(l_orderkey) FROM lineitem"
    pin_cols = ["l_orderkey", "l_quantity", "l_extendedprice", "l_discount",
                "l_returnflag", "l_linestatus", "l_shipdate"]


def make_dataset(cfg):
    kind = cfg["dataset"]["kind"]
    if kind == "synth":
        return SynthDataset(cfg)
    if kind == "tpch":
        return TpchDataset(cfg)
    raise ValueError(kind)


def init_settings(cfg, arm=None):
    """Engine session settings.

    `SET gpu_execution=false` is mandatory everywhere: on current dev it
    defaults to TRUE = transparent GPU interception of ALL SQL, which would
    (a) make the cpu baseline secretly GPU, (b) route harness bookkeeping
    through the GPU path (observed: count(DISTINCT) -> FATAL, database
    invalidated). The harness invokes GPU explicitly via CALL gpu_execution.

    arm_b/arm_c additionally need the duckdb-native scan route: it reads the
    checkpointed FILE blocks only, so the GPU result is genuinely the
    pin-epoch snapshot (un-checkpointed writes live in WAL/memory) and the
    delta merge is sound. On the default CPU-source route, gpu_execution
    re-reads through DuckDB's MVCC-merging scan -> always fresh -> the merge
    would double-count (observed in smoke: verify failures).
    """
    # iteration 2: the pin is a parquet export + CALL pin_table (the only
    # route with real GPU residency — measured 25x warm/steady gap; the
    # table_gpu view route gave none). Native-scan + checkpoint-threshold
    # workarounds from iteration 1 are no longer needed: the pin file is
    # immutable by construction and auto-checkpoint of the live WAL is
    # harmless (deltas live in tables, not WAL state).
    return ["SET gpu_execution=false"] + list(cfg["engine_settings"])


# ---------------------------------------------------------------------------
# Logging / cell management
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


def master_path(cfg):
    d = cfg["dataset"]
    tag = d["kind"] + (f"_sf{d['sf']}" if d["kind"] == "tpch" else "")
    return os.path.join(cfg["data_dir"], f"gate0_master_{tag}.duckdb")


def fresh_cell_db(cfg, cell_tag):
    db = os.path.join(cfg["data_dir"], f"gate0_{cell_tag}.duckdb")
    for p in (db, db + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(master_path(cfg), db)
    return db


def granule_stats(eng, ds):
    """Dirty-granule statistics from the delta tables (approximate: position
    inferred from the dataset's monotone key — see dataset docstrings)."""
    out = {}
    expr = f"{ds.granule_expr} * {ds.granule_scale}"
    for g, label in ((ROWS_PER_ROW_GROUP, "rowgroup"), (ROWS_PER_VECTOR, "vector")):
        rows, _ = eng.sql(
            f"SELECT count(DISTINCT ({expr}) // {g}) FROM delta_del", timeout=300)
        out[f"dirty_{label}s_del"] = int(rows[0]) if rows else 0
    for t in ("delta_ins", "delta_del"):
        rows, _ = eng.sql(f"SELECT count(*) FROM {t}", timeout=300)
        out[f"{t}_rows"] = int(rows[0]) if rows else 0
    return out


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_setup(cfg):
    """Create the master DB template, checkpointed and closed clean. Sweep
    cells copy this file so every cell starts identical."""
    os.makedirs(cfg["data_dir"], exist_ok=True)
    master = master_path(cfg)
    for p in (master, master + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    ds = make_dataset(cfg)
    eng = SiriusEngine(cfg, master, tag="setup")
    eng.start()
    ds.setup(eng, None)
    eng.sql("CHECKPOINT", timeout=3600)
    eng.stop()
    print(f"[setup] master ready: {master} ({os.path.getsize(master) / 1e9:.2f} GB)")


def run_reads(eng, ds, mode, log, timeout=600.0, table=None, **tags):
    parsed = {}
    for qname, qd in ds.queries().items():
        q = qd["sql"].replace("{t}", table or ds.table)
        try:
            if mode == "gpu":
                rows, wall = eng.gpu_sql(q, timeout=timeout)
            else:
                rows, wall = eng.sql(q, timeout=timeout)
            eng_t = eng.last_engine_times[-1] if eng.last_engine_times else None
            parsed[qname] = parse_rows(qd["key_cols"], rows)
            log.rec(event="read", query=qname, mode=mode, wall_s=wall,
                    engine_s=eng_t, fallback=eng.last_fallback, error=None,
                    **tags)
        except (RuntimeError, EngineCrash) as e:
            log.rec(event="read", query=qname, mode=mode, wall_s=None,
                    engine_s=None, error=str(e)[:300], **tags)
            if isinstance(e, EngineCrash):
                raise
    return parsed


def phase_probe(cfg, log):
    arm = None  # settings: base only
    """Empirically answer, on current dev:
    P1: after warming the table_gpu cache, do un-checkpointed writes change
        GPU results? (stale serve / fresh / error / crash)
    P2: does CHECKPOINT (no restart) change GPU results?
    P3: does engine restart change GPU results? (it must — cache is in-proc)
    Each step diffs GPU vs CPU on count(*).
    """
    ds = make_dataset(cfg)
    db = fresh_cell_db(cfg, "probe")
    eng = SiriusEngine(cfg, db, tag="probe")
    eng.start(init_sql=init_settings(cfg, arm))
    count_q = ds.probe_query

    def snap(stage):
        out = {}
        for mode in ("gpu", "cpu"):
            try:
                rows, wall = (eng.gpu_sql(count_q) if mode == "gpu"
                              else eng.sql(count_q))
                out[mode] = rows[0] if rows else "EMPTY"
                out[mode + "_s"] = round(wall, 4)
            except (RuntimeError, EngineCrash) as e:
                out[mode] = f"ERROR: {str(e)[:160]}"
                if isinstance(e, EngineCrash):
                    eng.stop()
                    eng.start(init_sql=init_settings(cfg, arm))
        verdict = "FRESH" if out.get("gpu") == out.get("cpu") else "STALE_OR_WRONG"
        if str(out.get("gpu", "")).startswith("ERROR"):
            verdict = "ERROR"
        log.rec(event="probe", stage=stage, verdict=verdict, **out)
        print(f"[probe] {stage:36s} gpu={out.get('gpu')} ({out.get('gpu_s')}s) "
              f"cpu={out.get('cpu')} -> {verdict}", flush=True)

    snap("baseline_warm")
    snap("cache_hit_check")
    stmts, _ = ds.tick_stmts(0, cfg["churn"]["rhos"][0], track_delta=False)
    for s in stmts:
        eng.sql(s, timeout=1800)
    snap("P1_after_uncheckpointed_writes")
    eng.sql("CHECKPOINT", timeout=1800)
    snap("P2_after_checkpoint_no_restart")
    eng.stop()
    eng.start(init_sql=init_settings(cfg, arm))
    snap("P3_after_engine_restart")
    eng.stop()
    os.remove(db)


def run_arm(cfg, arm, rho, log):
    """Run one (arm, rho) cell: T ticks of [write batch -> policy -> reads]."""
    ds = make_dataset(cfg)
    ds.reset_state()
    ticks = cfg["workload"]["ticks"]
    track_delta = arm in ("arm_b", "arm_c")
    recompact_every = cfg.get("recompact_every", 5)
    cell = f"{arm}_rho{rho}"
    db = fresh_cell_db(cfg, cell)
    eng = SiriusEngine(cfg, db, tag=cell)
    start_s = eng.start(init_sql=init_settings(cfg, arm))
    log.rec(event="engine_start", arm=arm, rho=rho, wall_s=start_s)
    read_mode = "cpu" if arm == "cpu" else "gpu"

    # Pin model (iteration 2): the pinned GPU copy = a column-pruned parquet
    # export of the table at pin epoch, pinned via CALL pin_table(tier='gpu')
    # — the only route with real GPU residency (measured 25x warm/steady
    # gap). Immutable by construction, so the delta merge is sound regardless
    # of scan-route semantics (iteration 1 showed no route serves a
    # consistent stale snapshot under concurrent DML). All GPU arms share
    # this read mechanism; they differ only in WHEN they re-pin.
    # export_s is a harness artifact (a real system re-uploads from table
    # pages); pin_s + the rewarm read = the true re-upload/decode cost.
    gpu_table = "pin_v" if arm != "cpu" else ds.table
    pin_epoch = {"n": 0}

    def make_pin():
        pin_epoch["n"] += 1
        e = pin_epoch["n"]
        path = os.path.join(cfg["data_dir"], f"pin_{cell}_{e}.parquet")
        cols = ", ".join(ds.pin_cols)
        t0 = time.monotonic()
        eng.sql(f"COPY (SELECT {cols} FROM {ds.table}) TO '{path}' (FORMAT parquet)",
                timeout=3600)
        t1 = time.monotonic()
        if e > 1:
            try:
                eng.sql("CALL unpin_table('pin_v')", timeout=600)
            except RuntimeError:
                pass
        eng.sql("CREATE OR REPLACE VIEW pin_v AS SELECT * FROM "
                f"read_parquet('{path}')", timeout=600)
        eng.sql(f"CALL pin_table('{path}', tier='gpu', name='pin_v')", timeout=1800)
        t2 = time.monotonic()
        old = os.path.join(cfg["data_dir"], f"pin_{cell}_{e - 1}.parquet")
        if os.path.exists(old):
            os.remove(old)
        return t1 - t0, t2 - t1  # export_s (artifact), pin_s

    if read_mode == "gpu":
        export_s, pin_s = make_pin()
        log.rec(event="pin_epoch", arm=arm, rho=rho, tick=-1,
                export_s=export_s, pin_s=pin_s)
        # warm read = pin materialization + decode (the true upload cost)
        run_reads(eng, ds, "gpu", log, table=gpu_table, event_tag="warm",
                  arm=arm, rho=rho, tick=-1)
        log.rec(event="gpu_mem", arm=arm, rho=rho, tick=-1, mib=gpu_mem_used_mib())

    for tick in range(ticks):
        stmts, batch_rows = ds.tick_stmts(tick, rho, track_delta)
        t0 = time.monotonic()
        for s in stmts:
            eng.sql(s, timeout=1800)
        log.rec(event="write", arm=arm, rho=rho, tick=tick, rows=batch_rows,
                wall_s=time.monotonic() - t0)

        if arm == "arm_a":
            # invalidate + re-upload after every write batch: unpin -> re-pin
            # (no engine restart needed; iteration 1's 4.3s restart was an
            # artifact of having no invalidation primitive on that route)
            export_s, pin_s = make_pin()
            run_reads(eng, ds, "gpu", log, table=gpu_table, event_tag="rewarm",
                      arm=arm, rho=rho, tick=tick)
            log.rec(event="refresh", arm=arm, rho=rho, tick=tick,
                    export_s=export_s, pin_s=pin_s)
        elif arm == "arm_c" and tick > 0 and tick % recompact_every == 0:
            t0 = time.monotonic()
            eng.sql("DELETE FROM delta_ins", timeout=600)
            eng.sql("DELETE FROM delta_del", timeout=600)
            trunc_s = time.monotonic() - t0
            export_s, pin_s = make_pin()  # re-pin at the new epoch
            run_reads(eng, ds, "gpu", log, table=gpu_table, event_tag="rewarm",
                      arm=arm, rho=rho, tick=tick)
            log.rec(event="recompact", arm=arm, rho=rho, tick=tick,
                    trunc_s=trunc_s, export_s=export_s, pin_s=pin_s)

        try:
            if arm in ("arm_b", "arm_c"):
                for qname, qd in ds.queries().items():
                    q_main = qd["sql"].replace("{t}", gpu_table)
                    q_truth = qd["sql"].replace("{t}", ds.table)
                    t0 = time.monotonic()
                    rows, _ = eng.gpu_sql(q_main, timeout=600)
                    stale = parse_rows(qd["key_cols"], rows)
                    stale_was_fallback = eng.last_fallback
                    t1 = time.monotonic()
                    ri, _ = eng.sql(qd["sql"].replace("{t}", "delta_ins"), timeout=600)
                    rd, _ = eng.sql(qd["sql"].replace("{t}", "delta_del"), timeout=600)
                    t2 = time.monotonic()
                    merged = merge_results(stale,
                                           parse_rows(qd["key_cols"], ri),
                                           parse_rows(qd["key_cols"], rd))
                    t3 = time.monotonic()
                    log.rec(event="read", query=qname, mode="merge_on_read",
                            arm=arm, rho=rho, tick=tick, wall_s=t3 - t0,
                            gpu_s=t1 - t0, delta_s=t2 - t1, merge_s=t3 - t2,
                            fallback=stale_was_fallback, error=None)
                    if stale_was_fallback:
                        # fallback result is fresh -> merge math invalid;
                        # logged above, skip the (would-be-spurious) verify
                        continue
                    if tick % cfg["workload"].get("verify_every", 1) == 0:
                        rt, _ = eng.sql(q_truth, timeout=600)
                        truth = parse_rows(qd["key_cols"], rt)
                        ok = results_match(merged, truth)
                        log.rec(event="verify", query=qname, arm=arm, rho=rho,
                                tick=tick, ok=ok,
                                detail=None if ok else
                                {"merged": {k: merged[k] for k in list(merged)[:4]},
                                 "truth": {k: truth[k] for k in list(truth)[:4]}})
                        if not ok:
                            print(f"[{cell}] VERIFY FAIL {qname} tick={tick}", flush=True)
            else:
                run_reads(eng, ds, read_mode, log, table=gpu_table,
                          arm=arm, rho=rho, tick=tick)
        except EngineCrash as e:
            log.rec(event="crash", arm=arm, rho=rho, tick=tick, error=str(e)[:300])
            print(f"[{cell}] ENGINE CRASH tick={tick}: {e}", flush=True)
            eng.stop()
            eng.start(init_sql=init_settings(cfg, arm))

        if track_delta:
            try:
                log.rec(event="granules", arm=arm, rho=rho, tick=tick,
                        **granule_stats(eng, ds))
            except (RuntimeError, EngineCrash):
                pass
        log.rec(event="gpu_mem", arm=arm, rho=rho, tick=tick, mib=gpu_mem_used_mib())
        print(f"[{cell}] tick {tick + 1}/{ticks} done", flush=True)

    eng.stop()
    for p in (db, db + ".wal",
              os.path.join(cfg["data_dir"], f"pin_{cell}_{pin_epoch['n']}.parquet")):
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True,
                    choices=["setup", "probe", "arm_a", "arm_b", "arm_c", "cpu", "sweep"])
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if cfg.get("repo_root") in (None, "", "auto"):
        cfg["repo_root"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.ticks:
        cfg["workload"]["ticks"] = args.ticks

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(
        here, "results", f"{cfg['dataset']['kind']}_{args.phase}.jsonl")
    base = {
        "box": cfg["box"]["name"],
        "interconnect": cfg["box"]["interconnect"],
        "dataset": cfg["dataset"]["kind"],
        "churn_shape": cfg["dataset"].get("churn_shape", "rf"),
        "phase": args.phase,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log = Log(out, base)

    if args.phase == "setup":
        phase_setup(cfg)
        return
    if not os.path.exists(master_path(cfg)):
        sys.exit(f"master template missing ({master_path(cfg)}) — run --phase setup")

    if args.phase == "probe":
        phase_probe(cfg, log)
    elif args.phase in ("arm_a", "arm_b", "arm_c", "cpu"):
        rho = args.rho if args.rho is not None else cfg["churn"]["rhos"][0]
        run_arm(cfg, args.phase, rho, log)
    elif args.phase == "sweep":
        for rho in cfg["churn"]["rhos"]:
            for arm in cfg.get("sweep_arms", ["cpu", "arm_a", "arm_b", "arm_c"]):
                print(f"=== sweep cell: {arm} rho={rho} ===", flush=True)
                try:
                    run_arm(cfg, arm, rho, log)
                except Exception as e:
                    log.rec(event="cell_abort", arm=arm, rho=rho, error=str(e)[:300])
                    print(f"=== cell ABORTED: {arm} rho={rho}: {e}", flush=True)
    log.close()
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
