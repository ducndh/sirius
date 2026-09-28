-- DuckDB's own vss_join (CPU, brute-force table macro) on the SAME workload as the Sirius Pareto:
-- packaged SIFT1M 10k queries x 1M base, k=10, recall@10 against the packaged ground truth.
-- File-backed DB + memory_limit + temp_directory, as the LATERAL anchor required (README trap 4).
LOAD vss;
SET memory_limit='48GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE OR REPLACE TABLE items   AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE OR REPLACE TABLE queries AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CREATE OR REPLACE TABLE gt10 AS SELECT query_id, neighbor_id
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_gt.parquet') WHERE rank < 10;
.print === shape probe (100 queries)
CREATE OR REPLACE TABLE q100 AS SELECT * FROM queries LIMIT 100;
DESCRIBE SELECT * FROM vss_join(q100, items, qvec, ivec, 10);
SELECT count(*) FROM vss_join(q100, items, qvec, ivec, 10);
.print === TIMED vss_join 10k x 1M k=10 (run 1 of 2)
.mode trash
.timer on
SELECT left_tbl.qid, right_tbl.iid FROM vss_join(queries, items, qvec, ivec, 10);
.print === TIMED vss_join 10k x 1M k=10 (run 2 of 2)
SELECT left_tbl.qid, right_tbl.iid FROM vss_join(queries, items, qvec, ivec, 10);
.timer off
.mode duckbox
.print === recall@10 vs packaged ground truth
CREATE OR REPLACE TABLE ans AS SELECT left_tbl.qid AS qid, right_tbl.iid AS iid FROM vss_join(queries, items, qvec, ivec, 10);
SELECT count(*) AS rows, (SELECT count(*) FROM ans a JOIN gt10 g ON g.query_id=a.qid AND g.neighbor_id=a.iid) AS hits,
       round(hits::DOUBLE/100000, 5) AS recall_at_10 FROM ans;
