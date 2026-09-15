-- E4 — GIST1M (d=960) so no claim rests on a single dimensionality.
-- ANN behaviour is strongly dimension-dependent, and every number recorded before this came from
-- SIFT1M at d=128. 960 dims is 7.5x the distance work per pair, weaker cluster structure, and a
-- different balance between our GEMM fold and IVF probing.
-- NOTE: GIST1M ships 1,000 queries, not SIFT's 10,000 -- this is a 1k x 1M regime. Do not compare
-- its absolute times to the SIFT 10k x 1M numbers; compare the SHAPE of the curves.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base  AS SELECT id, vec::FLOAT[960] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/gist-960-euclidean_base.parquet');
CREATE TABLE query AS SELECT id, vec::FLOAT[960] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/gist-960-euclidean_query.parquet');
CREATE TABLE gt    AS SELECT * FROM read_parquet('/var/tmp/vj/data/parquet/gist-960-euclidean_gt.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base',  tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');
.timer on
.print @@@ FIT
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c', n_clusters => 64);
.timer off
CREATE TABLE asg AS SELECT * FROM sirius_kmeans_assign('base','vec','c', n_probes => 1);
.timer on
.print @@@ MATERIALIZE ORDER BY cluster_id
CREATE TABLE corpus AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg a ON b.rowid = a.row_id ORDER BY a.cluster_id;
.timer off
CHECKPOINT;
SELECT * FROM unpin_table('base');
SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');
CREATE TABLE qasg AS SELECT * FROM sirius_kmeans_assign('query','vec','c', n_probes => 1);
CREATE TABLE probe AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN qasg a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
.timer on
.mode trash
.print @@@ EXACT
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'exact-gemm');
.print @@@ APPROX p1
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => 1);
.print @@@ APPROX p4
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => 4);
.print @@@ APPROX p8
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => 8);
.print @@@ APPROX p16
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => 16);
.timer off
.mode csv
.headers on
.output /var/tmp/vj/gist_p8.csv
SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => 8);
.output
.mode duckbox
.print @@@ RECALL p8 vs packaged ground truth
SELECT round(count(g.neighbor_id)::DOUBLE / (SELECT count(*) FROM read_csv('/var/tmp/vj/gist_p8.csv', header=true)), 4) AS recall_at_10
  FROM read_csv('/var/tmp/vj/gist_p8.csv', header=true) r
  LEFT JOIN gt g ON g.query_id = r.left_id AND g.rank < 10 AND g.neighbor_id = r.right_id;
