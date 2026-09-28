SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE g_corpus (id INTEGER, vec FLOAT[3]);
INSERT INTO g_corpus SELECT i, [sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);
CREATE TABLE g_probe (id INTEGER, vec FLOAT[3]);
INSERT INTO g_probe SELECT i, [sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(32) t(i);
CHECKPOINT;
SELECT * FROM pin_table(name => 'g_probe',  tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'g_corpus', tier => 'gpu', format => 'duckdb');
.print @@@ M0 baseline: exact, output_columns, no probe_source, duckbox
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global', left_output_columns => ['id'], right_output_columns => ['id']) LIMIT 2;
.print @@@ M1 change ONLY search_mode -> exact-gemm
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', search_mode => 'exact-gemm', metric => 'l2', k => 12, join_mode => 'global', left_output_columns => ['id'], right_output_columns => ['id']) LIMIT 2;
.print @@@ M2 change ONLY: drop output_columns
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global') LIMIT 2;
.print @@@ M3 change ONLY: add probe_source => scan
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', probe_source => 'scan', search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global', left_output_columns => ['id'], right_output_columns => ['id']) LIMIT 2;
.print @@@ M4 change ONLY: remove LIMIT
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global', left_output_columns => ['id'], right_output_columns => ['id']);
.print @@@ M5 change ONLY: .mode trash
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('g_probe','vec','g_corpus','vec', search_mode => 'exact', metric => 'l2', k => 12, join_mode => 'global', left_output_columns => ['id'], right_output_columns => ['id']);
.mode duckbox
.print @@@ M6 done
