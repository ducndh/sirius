#!/usr/bin/env bash
# Sirius over a corpus that does not fit the GPU -- the out-of-core claim, measured.
#
# The point is not only speed. cuvs_ooc.py has to split the corpus by hand, build an index per
# shard, and merge top-k across shards (see that file). Here the same workload is ONE SQL
# statement with the corpus arriving from a child scan, never pinned. Both numbers matter:
# the wall time, and the fact that the second column of the comparison is user-written code.
#
# Exact mode is measured first because it is iso-recall by construction (1.0), which sidesteps
# the recall-matching problem that makes approximate-vs-approximate comparisons delicate.
#
# Usage: sirius_ooc.sh <gib> [<gib> ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OOC=/var/tmp/vj/ooc
SIRIUS=${SIRIUS:-$HOME/vecjoin/sirius}
DUCKDB=${DUCKDB:-$SIRIUS/build/release/duckdb}
EXT=${EXT:-$SIRIUS/build/release/extension/sirius/sirius.duckdb_extension}
K=${K:-10}

[ -x "$DUCKDB" ] || { echo "MISSING $DUCKDB -- build Sirius for the CURRENT arch first"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

for GIB in "$@"; do
  TAG="${GIB}gib"
  CORPUS="$OOC/corpus_$TAG.parquet"
  [ -f "$CORPUS" ] || { echo "MISSING $CORPUS -- run gen_corpus.py $GIB first"; continue; }
  DB=/var/tmp/vj/ooc_$TAG.db
  SQL=/var/tmp/vj/ooc_$TAG.sql
  rm -f "$DB" "$DB".wal "$OOC/sirius_${TAG}_"*.csv

  # The corpus is pinned to the HOST tier, not the GPU: it lives in host memory and streams to
  # the device chunk by chunk. That is what "out-of-core" means here -- the corpus exceeds GPU
  # memory, not host memory. (Measured 2026-08-23: the EXACT path still requires the corpus
  # pinned to SOME tier and refuses an unpinned child scan -- "right table 'corpus' must be
  # pinned". Only the APPROX path was freed from pinning, in commit edecf159. The probe side is
  # small, so it goes on the GPU.)
  cat > "$SQL" <<EOF
-- no explicit LOAD: this duckdb IS the Sirius build and auto-loads the extension. A second
-- LOAD re-runs init, tries to allocate a second GPU pool, throws, and later kmeans/cuVS calls
-- die with cudaErrorIllegalAddress. See vecjoin_bench.yaml.
-- Cap DuckDB's buffer pool. It defaults to ~80% of RAM, and Sirius SEPARATELY pins the whole
-- corpus to the host tier, so on a 48 GiB corpus the two together can exceed the 120 GiB cgroup
-- ceiling. Measured 2026-08-23: without this the loader reached 98.9 GB RSS with 23 GB left.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE corpus AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$CORPUS');
CREATE TABLE probe  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$OOC/probe_$TAG.parquet');
CHECKPOINT;
SELECT count(*) AS corpus_rows FROM corpus;
SELECT * FROM pin_table(name => 'corpus', tier => 'host', format => 'duckdb');
SELECT * FROM pin_table(name => 'probe',  tier => 'gpu',  format => 'duckdb');
.print === TIMED sirius exact streaming, corpus host-pinned (streams to GPU), $GIB GiB
.timer on
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec',
  probe_source => 'scan', metric => 'l2', k => $K, search_mode => 'exact-gemm');
.timer off
.mode csv
.headers on
.output $OOC/sirius_${TAG}_exact.csv
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec',
  probe_source => 'scan', metric => 'l2', k => $K, search_mode => 'exact-gemm');
.output
EOF

  # Log into the experiment directory (git-backed), NOT /var/tmp. A previous run's numbers were
  # read from a /var/tmp log that the next reboot erased, leaving the figures with no artifact
  # behind them -- which this directory's own rule forbids.
  LOG="$HERE/run_${TAG}.log"
  echo "########## Sirius out-of-core, $GIB GiB (log: $LOG)"
  SIRIUS_CONFIG_FILE=${SIRIUS_CONFIG_FILE:-$HERE/../../vecjoin_bench.yaml} \
    SIRIUS_LOG_BACKEND=noop "$DUCKDB" -unsigned "$DB" < "$SQL" 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | tee "$LOG" | grep -E '===|Run Time|corpus_rows|Error|error' || true

  # Recall against the exact ground truth. Exact mode must come back at 1.0; anything else means
  # the streaming fold dropped corpus chunks, which is the failure this experiment must catch.
  if [ -f "$OOC/sirius_${TAG}_exact.csv" ]; then
    /var/tmp/ddb_old/v1.4.4/duckdb :memory: -c "
      SELECT round(count(t.neighbor_id)::DOUBLE / (SELECT count(*) FROM read_csv('$OOC/sirius_${TAG}_exact.csv', header=true)), 4) AS recall_at_$K,
             (SELECT count(*) FROM read_csv('$OOC/sirius_${TAG}_exact.csv', header=true)) AS rows_returned
      FROM read_csv('$OOC/sirius_${TAG}_exact.csv', header=true) g
      LEFT JOIN (SELECT query_id, neighbor_id FROM read_parquet('$OOC/gt_$TAG.parquet') WHERE rank < $K) t
        ON g.left_id = t.query_id AND g.right_id = t.neighbor_id;" 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
  fi
done
