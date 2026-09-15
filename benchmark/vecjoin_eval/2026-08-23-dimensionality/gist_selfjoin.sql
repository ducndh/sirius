-- The many-to-many regime at d=960: does J3's exact-path win survive high dimensionality?
-- 1M x 1M self-join, 10M output pairs. J3 measured 9.88x over cuVS at recall 1.0 on SIFT (d=128);
-- E4 measured 6.5x in the SEARCH regime at d=960. This is the missing cell: many-to-many AND
-- high-dimensional, where our GEMM fold does the most work per pair.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base AS SELECT id, vec::FLOAT[960] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/gist-960-euclidean_base.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
.timer on
.mode trash
.print @@@ EXACT self-join 1M x 1M d=960
SELECT left_id, right_id FROM sirius_knn_join('base','vec','base','vec', k => 10, metric => 'l2');
