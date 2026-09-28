-- Sirius GPU exact join, same shape. Run with the sirius build's duckdb + -unsigned.
CREATE TABLE items   AS SELECT id, vec::FLOAT[128] AS vec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE queries AS SELECT id, vec::FLOAT[128] AS vec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'items',   tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'queries', tier => 'gpu', format => 'duckdb');
.mode trash
.timer on
.print === TIMED sirius exact, run 1 then warm
SELECT distance FROM sirius_knn_join('queries','vec','items','vec', k => 10, metric => 'l2');
SELECT distance FROM sirius_knn_join('queries','vec','items','vec', k => 10, metric => 'l2');
