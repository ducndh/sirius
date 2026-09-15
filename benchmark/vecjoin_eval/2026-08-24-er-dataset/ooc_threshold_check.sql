SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE users    AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE products AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
.print @@@ GPU-tier corpus (baseline): eps=150 -> expect 260864
SELECT * FROM pin_table(name => 'products', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'users', tier => 'gpu', format => 'duckdb');
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('products','vec','users','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 150.0, search_mode => 'exact-gemm');
.timer off
SELECT * FROM unpin_table('users');
.print @@@ HOST-tier corpus (streams, multi-chunk = the out-of-core path)
SELECT * FROM pin_table(name => 'users', tier => 'host', format => 'duckdb');
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('products','vec','users','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 150.0, search_mode => 'exact-gemm');
.timer off
.print @@@ HOST-tier, eps=200 -> expect 1789956
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('products','vec','users','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 200.0, search_mode => 'exact-gemm');
.timer off
