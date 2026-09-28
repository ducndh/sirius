-- Verification goes through CSV because a scalar function over the join's output
-- (round/abs) breaks the plan -- see the composition bug in the memory file. Aggregates
-- alone work; `round(min(distance),4)` does not, so the check cannot be done in-engine.
CREATE TABLE items   AS SELECT id, vec::FLOAT[128] AS vec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE queries AS SELECT id, vec::FLOAT[128] AS vec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'items',   tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'queries', tier => 'gpu', format => 'duckdb');
.mode csv
.headers on
.output /var/tmp/sirius_dist.csv
SELECT distance FROM sirius_knn_join('queries','vec','items','vec', k => 10, metric => 'l2');
.output
