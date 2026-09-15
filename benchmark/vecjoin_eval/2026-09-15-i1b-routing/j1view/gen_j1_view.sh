#!/usr/bin/env bash
# J1 re-run with the corpus as a VIEW (S4): the right side is 'SELECT ... FROM base WHERE id % 10 < 3'
# (30 % of SIFT1M), never materialized, never pinned. Sirius exact, build_source => 'scan'. The
# cuVS practitioner path is j1_cuvs.py (export surviving rows -> h2d -> build -> search -> map back).
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/j1view}; DATA=/var/tmp/vj/data/parquet
{
echo "SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');"
for sel in 10 30 50 90; do
  echo "CREATE OR REPLACE VIEW corpus_v$sel AS SELECT id, vec FROM base WHERE id % 100 < $sel;"
  Q="SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus_v$sel','vec', probe_source => 'scan', build_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm')"
  echo ".print @@@RUN view_$sel"; echo ".mode trash"; echo ".timer on"; for i in 1 2 3; do echo "$Q;"; done; echo ".timer off"
  echo ".mode csv"; echo ".headers on"; echo ".output $OUT/rows_view_$sel.csv"; echo "$Q;"; echo ".output"; echo ".mode trash"
  # the old way, for the same selectivity: materialize + checkpoint + pin + join
  echo ".print @@@RUN ctas_$sel"; echo ".timer on"
  echo "CREATE OR REPLACE TABLE corpus_t$sel AS SELECT id, vec FROM base WHERE id % 100 < $sel;"
  echo "CHECKPOINT;"
  echo "SELECT * FROM pin_table(name => 'corpus_t$sel', tier => 'gpu', format => 'duckdb');"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus_t$sel','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
  echo ".timer off"
  echo "SELECT * FROM unpin_table('corpus_t$sel');"
done
} > "$OUT/j1view.sql"
echo "wrote $OUT/j1view.sql"
