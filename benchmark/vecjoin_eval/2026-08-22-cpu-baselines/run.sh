#!/usr/bin/env bash
# Reproduce the 2026-08-22 CPU-baseline comparison end to end.
#   ./run.sh            # all three
#   ./run.sh vss_join   # one
# Requires: SIFT1M ingested (see ../../README.md section 1) and a built Sirius tree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIRIUS=${SIRIUS:-$HOME/vecjoin/sirius}
DUCKDB=${DUCKDB:-$SIRIUS/build/release/duckdb}
DB=${DB:-/var/tmp/baselines.db}
DATA=/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet

[ -f "$DATA" ] || { echo "MISSING $DATA -- ingest SIFT1M first (../../README.md)"; exit 1; }
[ -x "$DUCKDB" ] || { echo "MISSING $DUCKDB -- build Sirius first"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

want() { [ $# -eq 0 ] || [ "${1:-}" = "${TARGET:-}" ]; }
TARGET=${1:-all}

if [ "$TARGET" = all ] || [ "$TARGET" = vss_join ] || [ "$TARGET" = lateral ]; then
  rm -f "$DB"; "$DUCKDB" "$DB" < "$HERE/setup.sql" > /dev/null
fi
# One GPU/CPU job at a time: a Sirius run burns CPU while pinning and would inflate a
# timed 64-core CPU baseline running alongside it. Never parallelise these.
if [ "$TARGET" = all ] || [ "$TARGET" = vss_join ]; then
  echo "### vss_join"; "$DUCKDB" "$DB" < "$HERE/vss_join.sql"
fi
if [ "$TARGET" = all ] || [ "$TARGET" = lateral ]; then
  echo "### lateral";  "$DUCKDB" "$DB" < "$HERE/lateral.sql"
fi
if [ "$TARGET" = all ] || [ "$TARGET" = sirius ]; then
  echo "### sirius"
  rm -f /var/tmp/sirf.db*
  (cd "$SIRIUS" && pixi run build/release/duckdb -unsigned /var/tmp/sirf.db) < "$HERE/sirius.sql"
  echo "### sirius verify (cross-engine oracle)"
  rm -f /var/tmp/sird.db* /var/tmp/sirius_dist.csv
  (cd "$SIRIUS" && SIRIUS_LOG_BACKEND=noop pixi run build/release/duckdb -unsigned /var/tmp/sird.db) \
    < "$HERE/verify_sirius.sql" > /dev/null
  "$DUCKDB" :memory: -c "SELECT round(min(distance),4) min_d, round(max(distance),4) max_d, count(*) n FROM read_csv('/var/tmp/sirius_dist.csv');"
  echo "expected: 20.8087 | 361.866 | 100000"
fi
