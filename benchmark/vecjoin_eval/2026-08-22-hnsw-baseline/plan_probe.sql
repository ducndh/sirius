-- Which DuckDB/vss releases actually fire HNSW_INDEX_JOIN.
--
-- MUST use a realistically sized table. Upstream's own fixture (test/sql/hnsw/
-- hnsw_lateral_join_plan.test) uses TWO rows, and on a two-row table the rule does not fire on
-- versions where it demonstrably works at scale -- that false negative cost this experiment a
-- wrong conclusion about v1.4.4 once already. 200k rows is enough and builds in a few seconds.
INSTALL vss; LOAD vss;
SET memory_limit='32GB';
SELECT extension_version AS vss_version FROM duckdb_extensions() WHERE extension_name='vss';
CREATE TABLE items2 AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet') LIMIT 200000;
CREATE TABLE queries2 AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet') LIMIT 500;
CREATE INDEX hidx ON items2 USING HNSW (ivec) WITH (metric='l2sq');
EXPLAIN SELECT * FROM queries2 q, LATERAL
  (SELECT * FROM items2 i ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
