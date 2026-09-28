LOAD vss;
SET memory_limit='48GB';
.timer on
.print === load tables
CREATE TABLE items2   AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE queries2 AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
.print === TIMED index build (M=16, ef_construction=128, metric l2sq)
CREATE INDEX hnsw_ivec ON items2 USING HNSW (ivec) WITH (metric='l2sq');
.print === PLAN CHECK: HNSW_INDEX_JOIN must appear
EXPLAIN SELECT q.qid, t.iid, t.d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.mode trash
.print === TIMED hnsw index join, default ef_search, run 1
SELECT q.qid, t.iid, t.d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.print === TIMED hnsw index join, default ef_search, run 2 (warm)
SELECT q.qid, t.iid, t.d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.timer off
.mode csv
.headers on
.output /var/tmp/hnsw_default.csv
SELECT q.qid AS qid, t.iid AS iid, t.d AS d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.output
