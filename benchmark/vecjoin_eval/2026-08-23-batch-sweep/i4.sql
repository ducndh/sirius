SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE pid1k AS SELECT id FROM read_parquet('/var/tmp/vj/x1/probe_ids_1000.parquet');
CREATE TABLE pid10k AS SELECT id FROM read_parquet('/var/tmp/vj/x1/probe_ids_10000.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');
CREATE TABLE p1k AS SELECT b.id, b.vec FROM base b SEMI JOIN pid1k p ON b.id = p.id;
CREATE TABLE p10k AS SELECT b.id, b.vec FROM base b SEMI JOIN pid10k p ON b.id = p.id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'p1k', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'p10k', tier => 'gpu', format => 'duckdb');
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.mode duckbox
.print @@@ THRESHOLD -- correctness first: does eps mean squared L2, and are the counts exact?
.timer on
SELECT count(*) AS n FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 20000.0, k => 8000, search_mode => 'exact-gemm');
SELECT count(*) AS n FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 30000.0, k => 8000, search_mode => 'exact-gemm');
SELECT count(*) AS n FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 40000.0, k => 8000, search_mode => 'exact-gemm');
.timer off
.print @@@ THRESHOLD -- the k the data actually needs vs a k a user would guess
.timer on
SELECT count(*) AS n_k100 FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 20000.0, k => 100, search_mode => 'exact-gemm');
.timer off
.print @@@ THRESHOLD timed, 1k probes, k=8000
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 20000.0, k => 8000, search_mode => 'exact-gemm');
SELECT left_id, right_id FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 40000.0, k => 8000, search_mode => 'exact-gemm');
.timer off
.mode duckbox
.print @@@ GLOBAL TOP-K, 1k probes
.timer on
SELECT count(*) AS g10 FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => 10, search_mode => 'exact-gemm');
SELECT count(*) AS g1000 FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => 1000, search_mode => 'exact-gemm');
.timer off
.print @@@ GLOBAL TOP-K, 10k probes
.timer on
SELECT count(*) AS g10 FROM sirius_knn_join('p10k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => 10, search_mode => 'exact-gemm');
SELECT count(*) AS g1000 FROM sirius_knn_join('p10k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => 1000, search_mode => 'exact-gemm');
.timer off
.print @@@ GLOBAL TOP-K correctness -- export the k=1000 pairs
.mode csv
.output /var/tmp/vj/x1/i4_global_1k_k1000.csv
SELECT left_id, right_id, distance FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'global', k => 1000, search_mode => 'exact-gemm');
.output
.mode duckbox
.print @@@ THRESHOLD correctness -- export eps=20000
.mode csv
.output /var/tmp/vj/x1/i4_thresh_1k_20000.csv
SELECT left_id, right_id FROM sirius_knn_join('p1k','vec','base','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 20000.0, k => 8000, search_mode => 'exact-gemm');
.output
.mode duckbox
