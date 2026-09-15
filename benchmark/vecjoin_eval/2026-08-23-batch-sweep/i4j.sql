SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE q10k AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'q10k', tier => 'gpu', format => 'duckdb');
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.print @@@ GLOBAL q10k k=10
.timer on
SELECT left_id, right_id, distance FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, join_mode => 'global', search_mode => 'exact-gemm', left_output_columns => ['id'], right_output_columns => ['id']);
.print @@@ GLOBAL q10k k=1000
SELECT left_id, right_id, distance FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 1000, join_mode => 'global', search_mode => 'exact-gemm', left_output_columns => ['id'], right_output_columns => ['id']);
.print @@@ PER-ROW q10k k=10 reference
SELECT left_id, right_id FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.timer off
.mode csv
.output /var/tmp/vj/x1/i4_global_q10k_k1000.csv
SELECT left_id, right_id, distance FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 1000, join_mode => 'global', search_mode => 'exact-gemm', left_output_columns => ['id'], right_output_columns => ['id']);
.output
.mode duckbox
.print @@@ THRESHOLD q10k eps=100 and eps=150
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 1024, join_mode => 'threshold', eps => 100.0, search_mode => 'exact-gemm');
SELECT left_id, right_id FROM sirius_knn_join('q10k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 1024, join_mode => 'threshold', eps => 150.0, search_mode => 'exact-gemm');
.timer off
.mode duckbox
