#!/usr/bin/env bash
# Recall/time Pareto for the clustered approximate join AFTER I1b (per-row routing: each probe row
# is assigned to its own n_probes nearest centroids, instead of its home cluster's nearest
# centroids). Same workload and scoring as ../2026-08-25-post-i1-pareto: packaged SIFT1M 10k x 1M,
# k=10, recall@10 against the packaged ground truth.
#
# Differences from that script: the probe side is the query table itself. The operator assigns
# probes to centroids per batch; the SQL-level sirius_kmeans_assign over the queries that the old
# script ran was never read by the join.
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/i1b}
DATA=/var/tmp/vj/data/parquet
mkdir -p "$OUT"

{
echo "SET memory_limit='32GB';"
echo "SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
echo "CREATE TABLE gt    AS SELECT * FROM read_parquet('$DATA/sift-128-euclidean_gt.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'base',  tier => 'gpu', format => 'duckdb');"
echo "SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');"

for C in 64 256 1024; do
  echo ".print @@@FIT c$C"
  echo ".timer on"
  echo "SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c$C', n_clusters => $C);"
  echo ".timer off"
  echo "CREATE TABLE asg$C AS SELECT * FROM sirius_kmeans_assign('base','vec','c$C', n_probes => 1);"
  echo "CREATE TABLE corpus$C AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg$C a ON b.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"
  echo "SELECT * FROM pin_table(name => 'corpus$C', tier => 'gpu', format => 'duckdb');"
done

emit() {  # $1 tag  $2 sql
  echo ".print @@@RUN $1"
  echo ".mode trash"; echo ".timer on"
  echo "$2"; echo "$2"; echo "$2"
  echo ".timer off"
  echo ".mode csv"; echo ".headers on"; echo ".output $OUT/rows_$1.csv"
  echo "$2"
  echo ".output"; echo ".mode duckbox"
}

EX="SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
emit exact "$EX"

SPECS=${SPECS:-"64:1 2 4 8 16 32 64|256:1 2 4 8 16 32 64 128|1024:1 4 16 32 64 128 256"}
IFS='|' read -ra SPECARR <<< "$SPECS"
for spec in "${SPECARR[@]}"; do
  C="${spec%%:*}"; PS="${spec#*:}"
  for p in $PS; do
    emit "c${C}_p${p}" "SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus$C','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c$C', cluster_column => 'cluster_id', n_probes => $p);"
  done
done

echo "CREATE TABLE gt10 AS SELECT query_id, neighbor_id FROM gt WHERE rank < 10;"
echo "CREATE TABLE recall (tag VARCHAR, rows BIGINT, hits BIGINT);"
TAGS="exact"
for spec in "${SPECARR[@]}"; do
  C="${spec%%:*}"; PS="${spec#*:}"
  for p in $PS; do TAGS="$TAGS c${C}_p${p}"; done
done
for t in $TAGS; do
  echo "INSERT INTO recall SELECT '$t', (SELECT count(*) FROM read_csv('$OUT/rows_$t.csv', header=true)), count(*) FROM read_csv('$OUT/rows_$t.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;"
done
echo ".print @@@RECALL"
echo ".mode csv"; echo ".headers on"; echo ".output $OUT/recall.csv"
echo "SELECT tag, rows, hits, round(hits::DOUBLE / 100000, 5) AS recall_at_10 FROM recall;"
echo ".output"
} > "$OUT/pareto.sql"
echo "wrote $OUT/pareto.sql ($(grep -c . "$OUT/pareto.sql") lines)"
