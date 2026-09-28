#!/usr/bin/env bash
# Reproduce the 2026-08-22 HNSW (indexed ANN) baseline.
#   ./run.sh            # everything, in the right order, one job at a time
#   ./run.sh plan_probe # just the which-version-fires-the-rule check
#   ./run.sh join | sweep | anchor | recall
#
# Requires SIFT1M ingested (../../README.md section 1). Does NOT require a built Sirius
# tree except for `recall`, which only needs any DuckDB to do the set intersection.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet
GT=/var/tmp/vj/data/parquet/sift-128-euclidean_gt.parquet
DDB_DIR=${DDB_DIR:-/var/tmp/ddb_old}
TARGET=${1:-all}

[ -f "$DATA" ] || { echo "MISSING $DATA -- ingest SIFT1M first (../../README.md)"; exit 1; }
nproc | sed 's/^/cpu cores: /'
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

# The HNSW_INDEX_JOIN rewrite fires on DuckDB v1.3.2 and NOT on v1.4.1 or v1.5.4 -- see the
# README. The baseline is run on the version where it works, which is the baseline's best
# configuration; `anchor` re-runs brute force on that SAME version so the speedup is a
# within-version ratio.
fetch() { # fetch <version>
  local v=$1 d="$DDB_DIR/v$v"
  [ -x "$d/duckdb" ] && return
  mkdir -p "$d"
  curl -sL -o "$d/d.zip" "https://github.com/duckdb/duckdb/releases/download/v${v}/duckdb_cli-linux-amd64.zip"
  ( cd "$d" && unzip -oq d.zip && rm -f d.zip )
}
fetch 1.4.4
DDB="$DDB_DIR/v1.4.4/duckdb"

recall_for() { # recall_for <csv> <label>
  "$DDB" :memory: -c "
    CREATE TABLE got AS SELECT * FROM read_csv('$1');
    CREATE TABLE truth AS SELECT query_id, neighbor_id FROM read_parquet('$GT') WHERE rank < 10;
    SELECT '$2' AS run,
      round((SELECT count(*) FROM got g JOIN truth t ON g.qid=t.query_id AND g.iid=t.neighbor_id)::DOUBLE
            / (SELECT count(*) FROM truth), 4) AS recall_at_10,
      (SELECT count(*) FROM got) AS rows_returned,
      (SELECT min(c) FROM (SELECT count(*) c FROM got GROUP BY qid)) AS min_k;"
}

if [ "$TARGET" = all ] || [ "$TARGET" = plan_probe ]; then
  echo "### plan probe -- which versions fire HNSW_INDEX_JOIN"
  for v in 1.3.2 1.4.4 1.5.4; do
    fetch $v 2>/dev/null || { echo "v$v: unavailable"; continue; }
    printf 'v%s: ' "$v"
    ( cd "$DDB_DIR/v$v" && ./duckdb :memory: < "$HERE/plan_probe.sql" ) 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g' \
      | grep -oE 'HNSW_INDEX_JOIN|CROSS_PRODUCT' | sort -u | tr '\n' ' '
    echo
  done
fi

# One job at a time: these are 64-core CPU timings and any neighbour inflates them.
if [ "$TARGET" = all ] || [ "$TARGET" = join ]; then
  echo "### hnsw index join, default ef_search"
  "$DDB" :memory: < "$HERE/hnsw_join.sql" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -E '===|Run Time|HNSW_INDEX_JOIN|CROSS_PRODUCT'
  recall_for /var/tmp/hnsw_default.csv "ef_search=default(64)"
fi

if [ "$TARGET" = all ] || [ "$TARGET" = sweep ]; then
  echo "### ef_search sweep"
  bash "$HERE/gen_sweep.sh" > /var/tmp/hnsw_sweep.sql
  "$DDB" :memory: < /var/tmp/hnsw_sweep.sql 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -E '===|Run Time'
  for ef in 10 20 40 80 160 320; do recall_for "/var/tmp/hnsw_ef$ef.csv" "ef_search=$ef"; done
fi

if [ "$TARGET" = all ] || [ "$TARGET" = anchor ]; then
  echo "### brute-force LATERAL on the SAME version (v1.3.2)"
  "$DDB" :memory: < "$HERE/bf_anchor.sql" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -E '===|Run Time'
fi

if [ "$TARGET" = recall ]; then
  recall_for /var/tmp/hnsw_default.csv "ef_search=default(64)"
  for ef in 10 20 40 80 160 320; do
    [ -f "/var/tmp/hnsw_ef$ef.csv" ] && recall_for "/var/tmp/hnsw_ef$ef.csv" "ef_search=$ef"
  done
fi
