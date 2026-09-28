SET memory_limit='48GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE corpus AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/ooc/corpus_48gib.parquet');
CREATE TABLE probe  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/ooc/probe_48gib.parquet');
CHECKPOINT;
SELECT count(*) AS corpus_rows FROM corpus;
SELECT * FROM pin_table(name => 'corpus', tier => 'host', format => 'duckdb');
SELECT * FROM pin_table(name => 'probe',  tier => 'gpu',  format => 'duckdb');
.print @@@ A exact k=10 baseline (recorded 32.39 s @ recall 1.0)
.timer on
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.mode duckbox
.timer off
.print @@@ B RADIUS join out-of-core, eps=13.0
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 13.0, search_mode => 'exact-gemm');
.timer off
.print @@@ C RADIUS join out-of-core, eps=13.5
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 13.5, search_mode => 'exact-gemm');
.timer off
.print @@@ D export eps=13.5 pairs for correctness vs the packaged ground truth
.mode csv
.output /var/tmp/vj/ooc48_pairs.csv
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', join_mode => 'threshold', eps => 13.5, search_mode => 'exact-gemm');
.output
.mode duckbox
