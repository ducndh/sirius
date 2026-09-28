SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');
.print @@@EPS 150
SELECT count(*) AS pairs FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 150, search_mode => 'exact-gemm');
.print @@@EPS 200
SELECT count(*) AS pairs FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 200, search_mode => 'exact-gemm');
.print @@@EPS 250
SELECT count(*) AS pairs FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 250, search_mode => 'exact-gemm');
.print @@@EPS 300
SELECT count(*) AS pairs FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 300, search_mode => 'exact-gemm');
.print @@@EPS 350
SELECT count(*) AS pairs FROM sirius_knn_join('query','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 350, search_mode => 'exact-gemm');
