#!/usr/bin/env bash
# Emit the X1 Sirius sweep as ONE SQL script.
#
# Three things this script exists to get right:
#
# 1. ONE SESSION. Clusterings and GPU pins are session state, not database state: a second
#    connection to the same .db reports "no clustering named cl" and "right table must be pinned".
# 2. NO SINK ABOVE THE OPERATOR. Neither `CREATE TABLE AS` nor `COPY ... TO` works over the TVF --
#    both are rejected as "cannot run on the CPU" (issue #19). Only the CLI's own `.mode csv` +
#    `.output` consumes the result set directly. A `WHERE left_id IN (SELECT ...)` filter IS
#    accepted (unlike a scalar function over an output column, S2), so recall exports carry only
#    the 1000 sampled probes instead of the full 10M rows.
# 3. A WARM-UP JOIN BEFORE ANY TIMING. The first join of a session pays one-time initialisation.
#    Measured 2026-08-23: without it the 1k-probe point reads 162 us/probe against 8.1 us/probe at
#    10k -- i.e. the smallest batch, the one the fixed-cost question is about, was the one carrying
#    an artefact. All probe tables are therefore built up-front and a throwaway join runs first.
#
# Timed runs use `.mode trash`; duckbox formats every row it is handed, ~11 s of shell rendering on
# a 10M-row result.
set -euo pipefail
X1=/var/tmp/vj/x1
DATA=/var/tmp/vj/data/parquet
NC=${NC:-64}
PROBES=${PROBES:-"1 2 4 8 16"}
COUNTS=${COUNTS:-"1000 10000 100000 1000000"}
TAG=${TAG:-c$NC}

echo "SET memory_limit='32GB';"
echo "SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE smp AS SELECT DISTINCT left_id AS id FROM read_parquet('$X1/truth_1000.parquet');"
for n in $COUNTS; do
  echo "CREATE TABLE pid_$n AS SELECT id FROM read_parquet('$X1/probe_ids_$n.parquet');"
done
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
echo ".print @@@ fit n_clusters=$NC"
echo ".timer on"
echo "SELECT * FROM sirius_kmeans_fit('base','vec', name => 'cl', n_clusters => $NC);"
echo ".timer off"
echo ".print @@@ assign+order corpus"
echo ".timer on"
echo "CREATE TABLE asg AS SELECT * FROM sirius_kmeans_assign('base','vec','cl', n_probes => 1);"
echo "CREATE TABLE corpus AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg a ON b.rowid = a.row_id ORDER BY a.cluster_id;"
echo "CHECKPOINT;"
echo ".timer off"
echo "SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');"

# --- every probe table built before anything is timed ---
for n in $COUNTS; do
  [ "$n" = "1000000" ] && continue      # the 1M probe set IS the corpus; reuse it, do not copy it
  echo "CREATE TABLE qraw_$n AS SELECT b.id, b.vec FROM base b SEMI JOIN pid_$n p ON b.id = p.id;"
  echo "CHECKPOINT;"
  echo "SELECT * FROM pin_table(name => 'qraw_$n', tier => 'gpu', format => 'duckdb');"
  echo "CREATE TABLE qa_$n AS SELECT * FROM sirius_kmeans_assign('qraw_$n','vec','cl', n_probes => 1);"
  echo "CREATE TABLE probe_$n AS SELECT q.id, q.vec, a.cluster_id FROM qraw_$n q JOIN qa_$n a ON q.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"
  echo "SELECT * FROM unpin_table('qraw_$n');"
  echo "SELECT * FROM pin_table(name => 'probe_$n', tier => 'gpu', format => 'duckdb');"
done

warm=$(echo $COUNTS | tr ' ' '\n' | sort -n | head -1)
WP=$([ "$warm" = "1000000" ] && echo corpus || echo probe_$warm)
echo ".print @@@warmup discarded"
echo ".mode trash"
echo "SELECT left_id, right_id FROM sirius_knn_join('$WP','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
echo "SELECT left_id, right_id FROM sirius_knn_join('$WP','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'cl', cluster_column => 'cluster_id', n_probes => 4);"
echo ".mode duckbox"

for n in $COUNTS; do
  P=$([ "$n" = "1000000" ] && echo corpus || echo probe_$n)
  A="'$P','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10"
  for p in $PROBES; do
    echo ".print @@@ approx probes_n=$n n_probes=$p"
    echo ".timer on"; echo ".mode trash"
    echo "SELECT left_id, right_id FROM sirius_knn_join($A, search_mode => 'approx', clustering => 'cl', cluster_column => 'cluster_id', n_probes => $p);"
    echo ".mode duckbox"; echo ".timer off"
    echo ".mode csv"; echo ".output $X1/sir_${TAG}_${n}_p${p}.csv"
    echo "SELECT left_id, right_id FROM sirius_knn_join($A, search_mode => 'approx', clustering => 'cl', cluster_column => 'cluster_id', n_probes => $p) WHERE left_id IN (SELECT id FROM smp);"
    echo ".output"; echo ".mode duckbox"
  done
  echo ".print @@@ exact probes_n=$n"
  echo ".timer on"; echo ".mode trash"
  echo "SELECT left_id, right_id FROM sirius_knn_join($A, search_mode => 'exact-gemm');"
  echo ".mode duckbox"; echo ".timer off"
  echo ".mode csv"; echo ".output $X1/sir_${TAG}_${n}_exact.csv"
  echo "SELECT left_id, right_id FROM sirius_knn_join($A, search_mode => 'exact-gemm') WHERE left_id IN (SELECT id FROM smp);"
  echo ".output"; echo ".mode duckbox"
done
