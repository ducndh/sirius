SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CREATE TABLE gt    AS SELECT * FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_gt.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base',  tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');
.print @@@FIT c64
.timer on
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c64', n_clusters => 64);
.timer off
CREATE TABLE asg64 AS SELECT * FROM sirius_kmeans_assign('base','vec','c64', n_probes => 1);
CREATE TABLE corpus64 AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg64 a ON b.rowid = a.row_id ORDER BY a.cluster_id;
CREATE TABLE qa64 AS SELECT * FROM sirius_kmeans_assign('query','vec','c64', n_probes => 1);
CREATE TABLE probe64 AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN qa64 a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus64', tier => 'gpu', format => 'duckdb');
.print @@@FIT c256
.timer on
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c256', n_clusters => 256);
.timer off
CREATE TABLE asg256 AS SELECT * FROM sirius_kmeans_assign('base','vec','c256', n_probes => 1);
CREATE TABLE corpus256 AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg256 a ON b.rowid = a.row_id ORDER BY a.cluster_id;
CREATE TABLE qa256 AS SELECT * FROM sirius_kmeans_assign('query','vec','c256', n_probes => 1);
CREATE TABLE probe256 AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN qa256 a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus256', tier => 'gpu', format => 'duckdb');
.print @@@FIT c1024
.timer on
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c1024', n_clusters => 1024);
.timer off
CREATE TABLE asg1024 AS SELECT * FROM sirius_kmeans_assign('base','vec','c1024', n_probes => 1);
CREATE TABLE corpus1024 AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg1024 a ON b.rowid = a.row_id ORDER BY a.cluster_id;
CREATE TABLE qa1024 AS SELECT * FROM sirius_kmeans_assign('query','vec','c1024', n_probes => 1);
CREATE TABLE probe1024 AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN qa1024 a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus1024', tier => 'gpu', format => 'duckdb');
.print @@@RUN exact
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_exact.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.output
.mode duckbox
.print @@@RUN c64_p1
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 1);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p1.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 1);
.output
.mode duckbox
.print @@@RUN c64_p2
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 2);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 2);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 2);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p2.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 2);
.output
.mode duckbox
.print @@@RUN c64_p4
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 4);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p4.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 4);
.output
.mode duckbox
.print @@@RUN c64_p8
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p8.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);
.output
.mode duckbox
.print @@@RUN c64_p16
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 16);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p16.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 16);
.output
.mode duckbox
.print @@@RUN c64_p32
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 32);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p32.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 32);
.output
.mode duckbox
.print @@@RUN c64_p64
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 64);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c64_p64.csv
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 64);
.output
.mode duckbox
.print @@@RUN c256_p1
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 1);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p1.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 1);
.output
.mode duckbox
.print @@@RUN c256_p2
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 2);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 2);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 2);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p2.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 2);
.output
.mode duckbox
.print @@@RUN c256_p4
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 4);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p4.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 4);
.output
.mode duckbox
.print @@@RUN c256_p8
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 8);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 8);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 8);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p8.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 8);
.output
.mode duckbox
.print @@@RUN c256_p16
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 16);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p16.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 16);
.output
.mode duckbox
.print @@@RUN c256_p32
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 32);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p32.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 32);
.output
.mode duckbox
.print @@@RUN c256_p64
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 64);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p64.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 64);
.output
.mode duckbox
.print @@@RUN c256_p128
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 128);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 128);
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 128);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c256_p128.csv
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 128);
.output
.mode duckbox
.print @@@RUN c1024_p1
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 1);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 1);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p1.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 1);
.output
.mode duckbox
.print @@@RUN c1024_p4
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 4);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 4);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p4.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 4);
.output
.mode duckbox
.print @@@RUN c1024_p16
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 16);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 16);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p16.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 16);
.output
.mode duckbox
.print @@@RUN c1024_p32
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 32);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 32);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p32.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 32);
.output
.mode duckbox
.print @@@RUN c1024_p64
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 64);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 64);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p64.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 64);
.output
.mode duckbox
.print @@@RUN c1024_p128
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 128);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 128);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 128);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p128.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 128);
.output
.mode duckbox
.print @@@RUN c1024_p256
.mode trash
.timer on
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 256);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 256);
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 256);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/pareto/rows_c1024_p256.csv
SELECT left_id, right_id FROM sirius_knn_join('probe1024','vec','corpus1024','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c1024', cluster_column => 'cluster_id', n_probes => 256);
.output
.mode duckbox
CREATE TABLE gt10 AS SELECT query_id, neighbor_id FROM gt WHERE rank < 10;
CREATE TABLE recall (tag VARCHAR, rows BIGINT, hits BIGINT);
INSERT INTO recall SELECT 'exact', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_exact.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_exact.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p1', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p1.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p1.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p2', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p2.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p2.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p4', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p4.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p4.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p8', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p8.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p8.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p16', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p16.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p16.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p32', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p32.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p32.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c64_p64', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p64.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c64_p64.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p1', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p1.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p1.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p2', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p2.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p2.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p4', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p4.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p4.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p8', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p8.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p8.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p16', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p16.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p16.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p32', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p32.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p32.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p64', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p64.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p64.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c256_p128', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p128.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c256_p128.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p1', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p1.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p1.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p4', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p4.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p4.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p16', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p16.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p16.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p32', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p32.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p32.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p64', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p64.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p64.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p128', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p128.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p128.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
INSERT INTO recall SELECT 'c1024_p256', (SELECT count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p256.csv', header=true)), count(*) FROM read_csv('/var/tmp/vj/pareto/rows_c1024_p256.csv', header=true) a JOIN gt10 g ON g.query_id = a.left_id AND g.neighbor_id = a.right_id;
.print @@@RECALL
.mode csv
.headers on
.output /var/tmp/vj/pareto/recall.csv
SELECT tag, rows, hits, round(hits::DOUBLE / 100000, 5) AS recall_at_10 FROM recall;
.output
