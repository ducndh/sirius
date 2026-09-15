#!/usr/bin/env bash
# Gap D: threshold (radius) join and global top-k, time and answer size vs the knob, SIFT1M 10k x 1M,
# exact path (the radius kernel). Opponent for the radius join: FAISS-CPU range_search when the
# module is present (no GPU library exposes a radius search); for global top-k: the emulation
# "per-row top-k then ORDER BY LIMIT" in the same engine, which is the only alternative.
# eps values are chosen from the exact 10-NN distance distribution: p50 of the 1st-NN (near-dup
# regime) up to p50 of the 10th-NN (a "k~10 on average" regime).
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/threshold}; DATA=/var/tmp/vj/data/parquet; mkdir -p $OUT
{
echo "SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
echo "CREATE TABLE gt AS SELECT * FROM read_parquet('$DATA/sift-128-euclidean_gt.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'base',  tier => 'gpu', format => 'duckdb');"
echo "SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');"
echo ".print @@@EPS quantiles of the packaged ground-truth distances"
echo "SELECT rank, round(quantile_cont(distance, 0.5),1) AS p50, round(quantile_cont(distance, 0.9),1) AS p90 FROM gt WHERE rank IN (0, 4, 9, 49) GROUP BY rank ORDER BY rank;"
for eps in 150 200 250 300 350; do
  Q="SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => $eps, search_mode => 'exact-gemm')"
  echo ".print @@@RUN threshold_eps$eps"; echo ".mode trash"; echo ".timer on"; for i in 1 2 3; do echo "$Q;"; done; echo ".timer off"
  echo ".mode csv"; echo ".headers on"; echo ".output $OUT/pairs_eps$eps.csv"; echo "SELECT count(*) AS pairs, count(DISTINCT left_id) AS probes_with_match FROM ($Q);"; echo ".output"; echo ".mode trash"
done
for k in 100 1000 10000 100000; do
  Q="SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => $k, search_mode => 'exact-gemm')"
  echo ".print @@@RUN global_k$k"; echo ".mode trash"; echo ".timer on"; for i in 1 2 3; do echo "$Q;"; done; echo ".timer off"
  # the emulation: per-row top-k to the same depth, then a relational top-N on the score
  E2="SELECT left_id, right_id FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', k => $k, search_mode => 'exact-gemm') ORDER BY distance LIMIT $k"
  if [ $k -le 1000 ]; then echo ".print @@@RUN global_emul_k$k"; echo ".timer on"; for i in 1 2 3; do echo "$E2;"; done; echo ".timer off"; fi
done
} > $OUT/threshold.sql
echo "wrote $OUT/threshold.sql"
