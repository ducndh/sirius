-- Shared setup for the CPU-baseline comparison. Plain DuckDB (no Sirius).
-- Distinct column names on the two sides are LOAD-BEARING: vss_join takes column
-- IDENTIFIERS (not strings), and if both sides use the same name the macro resolves
-- both to the inner table and scores every pair 0.0 -- right row count, zero work.
INSTALL vss;
LOAD vss;
CREATE OR REPLACE TABLE items2   AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE OR REPLACE TABLE queries2 AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
