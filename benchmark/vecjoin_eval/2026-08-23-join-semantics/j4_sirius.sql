-- J4, Sirius side: the join's output is CONSUMED BY THE QUERY, not returned to a client.
--
-- The composability claim is that a join is an operator other operators can sit on top of. The
-- test is an aggregate over the join result: for a library the 10M pairs must cross device->host
-- and enter a database before a GROUP BY can touch them; here they never leave the GPU.
--
-- Aggregates only, deliberately: `round(distance,3)` over this output still errors (QUEUE S2), so
-- a scalar function in the projection would measure that bug rather than composition.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
.timer on
.print @@@ J4a raw join, 10M pairs materialized to the client (the baseline shape)
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('base','vec','base','vec', k => 10, metric => 'l2');
.print @@@ J4b SAME join, aggregated in-query -- 10M pairs never leave the GPU
SELECT count(*), avg(distance), min(distance), max(distance)
  FROM sirius_knn_join('base','vec','base','vec', k => 10, metric => 'l2');
.print @@@ J4c grouped aggregate over the join output
SELECT left_id, avg(distance) AS d
  FROM sirius_knn_join('base','vec','base','vec', k => 10, metric => 'l2')
 GROUP BY left_id;
