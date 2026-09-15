#!/usr/bin/env bash
# I6 -- composability, end to end. One SQL statement: filter the probe side (a subquery), vector-join
# it against the corpus, equi-join the matches back to the corpus for an attribute, aggregate.
# Against the practitioner path in compose_cuvs.py (DuckDB filter -> export -> cuVS -> import ->
# DuckDB join + aggregate). Also the CTAS form, which needed the GPU-under-sink splice.
# SIFT1M: corpus = base with a synthetic category (id % 97); probes = packaged queries with
# id % 4 = 0 (2,500 rows). k = 10, exact.
set -euo pipefail
OUT=${OUT:-/var/tmp/vj/compose}; DATA=/var/tmp/vj/data/parquet
{
echo "SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill';"
echo "CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec, (id % 97)::INTEGER AS category FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec, (id % 4 = 0) AS flagged FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
echo "CHECKPOINT;"
echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
PIPE="SELECT b.category, count(*) AS matches FROM sirius_knn_join_rel((SELECT id, vec FROM query WHERE flagged), 'vec', 'base', 'vec', search_mode => 'exact-gemm', metric => 'l2', k => 10, left_output_columns => ['id'], right_output_columns => ['id']) vj JOIN base b ON b.id = vj.right_id GROUP BY b.category ORDER BY matches DESC, b.category"
echo ".print @@@RUN pipeline"; echo ".mode trash"; echo ".timer on"
for i in 1 2 3; do echo "$PIPE;"; done
echo ".timer off"; echo ".mode csv"; echo ".headers on"; echo ".output $OUT/pipeline_sirius.csv"; echo "$PIPE;"; echo ".output"; echo ".mode trash"
echo ".print @@@RUN ctas"; echo ".timer on"
echo "CREATE OR REPLACE TABLE matches AS SELECT left_id, right_id, distance FROM sirius_knn_join_rel((SELECT id, vec FROM query WHERE flagged), 'vec', 'base', 'vec', search_mode => 'exact-gemm', metric => 'l2', k => 10, left_output_columns => ['id'], right_output_columns => ['id']);"
echo ".timer off"
echo "SELECT count(*) AS ctas_rows FROM matches;"
echo ".print @@@RUN ctas_then_aggregate"; echo ".timer on"
echo "SELECT b.category, count(*) AS matches FROM matches m JOIN base b ON b.id = m.right_id GROUP BY b.category ORDER BY matches DESC, b.category;"
echo ".timer off"
echo ".print @@@RUN copy"; echo ".timer on"
echo "COPY (SELECT left_id, right_id, distance FROM sirius_knn_join_rel((SELECT id, vec FROM query WHERE flagged), 'vec', 'base', 'vec', search_mode => 'exact-gemm', metric => 'l2', k => 10, left_output_columns => ['id'], right_output_columns => ['id'])) TO '$OUT/matches_copy.parquet' (FORMAT parquet);"
echo ".timer off"
echo ".print @@@RUN math_over_join"; echo ".timer on"
echo "SELECT count(*), round(avg(sqrt(distance)), 3) FROM sirius_knn_join_rel((SELECT id, vec FROM query WHERE flagged), 'vec', 'base', 'vec', search_mode => 'exact-gemm', metric => 'l2', k => 10) WHERE round(distance, 1) > 0;"
echo ".timer off"
} > "$OUT/compose.sql"
echo "wrote $OUT/compose.sql"
