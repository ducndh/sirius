-- S1 — where does the clustered fold's time go, and why do 256 clusters lose to 64?
--
-- J2/J0 established that cost tracks SPAN COUNT rather than bytes scanned: at 256 clusters the
-- approximate path is slower than our own exact join past 1 probe, while at 64 clusters 8 probes
-- beats exact. If per-span setup dominates, batching spans is the lever; if the scan itself
-- dominates, it is not. SIRIUS_VECTOR_JOIN_PHASE_DEBUG reports the per-phase split.
--
-- Held constant across the two cluster counts: the FRACTION of the corpus probed. 8/64 = 12.5%
-- and 32/256 = 12.5%, so both scan the same expected bytes and differ only in span count.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'base',  tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c64', n_clusters => 64);
CREATE TABLE asg64 AS SELECT * FROM sirius_kmeans_assign('base','vec','c64', n_probes => 1);
CREATE TABLE corpus64 AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg64 a ON b.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus64', tier => 'gpu', format => 'duckdb');
CREATE TABLE q64 AS SELECT * FROM sirius_kmeans_assign('query','vec','c64', n_probes => 1);
CREATE TABLE probe64 AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN q64 a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
.timer on
.mode trash
.print @@@ 64 clusters / 8 probes  (12.5% of corpus, 8 spans)
SELECT left_id, right_id FROM sirius_knn_join('probe64','vec','corpus64','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c64', cluster_column => 'cluster_id', n_probes => 8);
.mode duckbox
.timer off
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c256', n_clusters => 256);
CREATE TABLE asg256 AS SELECT * FROM sirius_kmeans_assign('base','vec','c256', n_probes => 1);
CREATE TABLE corpus256 AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg256 a ON b.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus256', tier => 'gpu', format => 'duckdb');
CREATE TABLE q256 AS SELECT * FROM sirius_kmeans_assign('query','vec','c256', n_probes => 1);
CREATE TABLE probe256 AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN q256 a ON q.rowid = a.row_id ORDER BY a.cluster_id;
CHECKPOINT;
.timer on
.mode trash
.print @@@ 256 clusters / 32 probes (12.5% of corpus, 32 spans -- SAME bytes, 4x the spans)
SELECT left_id, right_id FROM sirius_knn_join('probe256','vec','corpus256','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => 'c256', cluster_column => 'cluster_id', n_probes => 32);
