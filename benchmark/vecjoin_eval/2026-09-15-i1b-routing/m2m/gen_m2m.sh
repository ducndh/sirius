#!/usr/bin/env bash
# 1M x 1M self-join (SIFT1M base against itself), k=10, after I1b. Recall is measured on the first
# 10k probe rows against Sirius's own exact answer for those rows (a 1M x 1M exact ground truth is
# not needed: recall is a per-row property). Answers go to parquet via COPY, never through the
# shell renderer (duckbox costs 1.15 us/row; 10M rows would be the whole measurement).
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/m2m}; DATA=/var/tmp/vj/data/parquet
{
echo "SET memory_limit='40GB'; SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
for C in 64 256; do
  echo "SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c$C', n_clusters => $C);"
  echo "CREATE TABLE asg$C AS SELECT * FROM sirius_kmeans_assign('base','vec','c$C', n_probes => 1);"
  echo "CREATE TABLE corpus$C AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg$C a ON b.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"
  echo "SELECT * FROM pin_table(name => 'corpus$C', tier => 'gpu', format => 'duckdb');"
done
emit() {  # tag sql
  echo ".print @@@RUN $1"; echo ".mode trash"; echo ".timer on"; echo "$2;"; echo "$2;"; echo ".timer off"
  # COPY over the join falls to the CPU stub (bug B6), so the answer goes out through the shell in
  # csv mode (0.22 us/row, ~2 s for 10M rows) -- never duckbox.
  echo ".mode csv"; echo ".headers on"; echo ".output $OUT/rows_$1.csv"; echo "$2;"; echo ".output"; echo ".mode trash"
}
emit exact "SELECT left_id, right_id FROM sirius_knn_join('base','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm')"
for spec in "64:4 8 16 32" "256:8 16 32 64"; do
  C="${spec%%:*}"; for p in ${spec#*:}; do
    emit "c${C}_p${p}" "SELECT left_id, right_id FROM sirius_knn_join('base','vec','corpus$C','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c$C', cluster_column => 'cluster_id', n_probes => $p)"
  done
done
echo ".print @@@RECALL"
echo "CREATE TABLE truth AS SELECT left_id, right_id FROM read_csv('$OUT/rows_exact.csv', header=true) WHERE left_id < 10000;"
echo ".mode csv"; echo ".headers on"; echo ".output $OUT/recall.csv"
echo "SELECT 'exact' AS tag, count(*) AS hits, round(count(*)::DOUBLE/100000,5) AS recall_at_10 FROM truth"
for spec in "64:4 8 16 32" "256:8 16 32 64"; do
  C="${spec%%:*}"; for p in ${spec#*:}; do
    echo "UNION ALL SELECT 'c${C}_p${p}', count(*), round(count(*)::DOUBLE/100000,5) FROM (SELECT left_id, right_id FROM read_csv('$OUT/rows_c${C}_p${p}.csv', header=true) WHERE left_id < 10000) a JOIN truth t ON t.left_id=a.left_id AND t.right_id=a.right_id"
  done
done
echo ";"; echo ".output"
} > "$OUT/m2m.sql"
echo "wrote $OUT/m2m.sql"
