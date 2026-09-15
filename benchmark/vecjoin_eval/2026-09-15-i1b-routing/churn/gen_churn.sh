#!/usr/bin/env bash
# J5 -- churn. The corpus mutates between query batches (1% of rows replaced per epoch), so every
# system must re-prepare before answering the same 10k packaged queries again. Per epoch, Sirius
# pays (a) join only, exact, corpus scanned unpinned; (b) re-pin + exact join; (c) re-fit 64
# clusters + assign + cluster-ordered CTAS + pin + approx join (p8). cuVS (churn_cuvs.py, same
# corpora via parquet) pays index build + search. This is the accounting cuVS's amortized number
# assumes away.
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/churn}; DATA=/var/tmp/vj/data/parquet; EPOCHS=${EPOCHS:-5}; NREP=${NREP:-10000}  # rows replaced per epoch
{
echo "SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE live  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE base  AS SELECT id, vec FROM live ORDER BY id;"
echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');"
echo ".mode trash"
for e in $(seq 0 $((EPOCHS-1))); do
  echo ".print @@@EPOCH $e"
  if [ "$e" -gt 0 ]; then
    echo ".print @@@STEP mutate"
    echo ".timer on"
    echo "DELETE FROM live WHERE id IN (SELECT id FROM live USING SAMPLE reservoir($NREP ROWS) REPEATABLE ($e));"
    echo "INSERT INTO live SELECT 1000000 + $e*$NREP + (row_number() OVER ()) - 1, list_transform(vec, x -> x + 0.5)::FLOAT[128] FROM (SELECT vec FROM live USING SAMPLE reservoir($NREP ROWS) REPEATABLE ($((100+e))));"
    echo ".timer off"
    # every system needs the mutated corpus as one contiguous table: DuckDB rowids keep gaps after a
    # DELETE, and sirius_kmeans_assign's row_id is a dense scan index, so the corpus is rebuilt fresh
    echo ".print @@@STEP materialize"; echo ".timer on"
    echo "SELECT * FROM unpin_table('base');"
    echo "CREATE OR REPLACE TABLE base AS SELECT id, vec FROM live ORDER BY id;"
    echo "CHECKPOINT;"
    echo ".timer off"
  fi
  echo "COPY (SELECT id, vec FROM base ORDER BY id) TO '$OUT/base_e$e.parquet' (FORMAT parquet);"
  # (a) exact, corpus scanned, no preparation at all
  echo ".print @@@STEP a_exact_scan"; echo ".timer on"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', build_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
  echo ".timer off"
  # (b) re-pin, then exact join over the pin
  echo ".print @@@STEP b_pin"; echo ".timer on"
  echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
  echo ".timer off"
  echo ".print @@@STEP b_exact_pinned"; echo ".timer on"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
  echo ".timer off"
  # (c) re-cluster: fit + assign + cluster-ordered corpus + pin, then approx join
  echo ".print @@@STEP c_fit"; echo ".timer on"
  echo "SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c64', n_clusters => 64);"
  echo ".timer off"
  echo ".print @@@STEP c_assign_ctas"; echo ".timer on"
  echo "CREATE OR REPLACE TABLE asg AS SELECT * FROM sirius_kmeans_assign('base','vec','c64', n_probes => 1);"
  [ "$e" -gt 0 ] && echo "SELECT * FROM unpin_table('corpus');"
  echo "CREATE OR REPLACE TABLE corpus AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg a ON b.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"
  echo ".timer off"
  echo ".print @@@STEP c_pin"; echo ".timer on"
  echo "SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');"
  echo ".timer off"
  echo ".print @@@STEP c_approx_p8"; echo ".timer on"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);"
  echo ".timer off"
  # recall of (c) against (a) on this epoch's corpus
  echo ".mode csv"; echo ".headers on"
  echo ".output $OUT/exact_e$e.csv"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');"
  echo ".output $OUT/approx_e$e.csv"
  echo "SELECT left_id, right_id FROM sirius_knn_join('query','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);"
  echo ".output"; echo ".mode trash"
done
echo ".print @@@RECALL"; echo ".mode csv"; echo ".headers on"; echo ".output $OUT/recall.csv"
for e in $(seq 0 $((EPOCHS-1))); do
  echo "SELECT $e AS epoch, (SELECT count(*) FROM read_csv('$OUT/approx_e$e.csv', header=true)) AS rows, count(*) AS hits, round(count(*)::DOUBLE/100000,5) AS recall FROM read_csv('$OUT/approx_e$e.csv', header=true) a JOIN read_csv('$OUT/exact_e$e.csv', header=true) x ON x.left_id=a.left_id AND x.right_id=a.right_id;"
done
echo ".output"
} > "$OUT/churn.sql"
echo "wrote $OUT/churn.sql ($(grep -c . $OUT/churn.sql) lines)"
