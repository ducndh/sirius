-- Brute-force LATERAL used as the WITHIN-VERSION anchor for the HNSW speedup.
--
-- SAFETY (learned the hard way -- this run took the box down once, see README trap 4):
--   * file-backed DB, NOT :memory:. An in-memory DuckDB has no spill target, so exceeding RAM
--     is fatal rather than graceful. This box has 96 GB and NO swap.
--   * explicit memory_limit, so an overrun is a DuckDB error instead of an OOM kill.
--   * explicit temp_directory, so there is somewhere to spill to.
--   * run over a SUBSET (:n_queries) -- the purpose is the v1.3.2-vs-v1.5.4 ratio, which a
--     subset measures just as well as the full 10k, at a tenth of the risk.
-- Pass the subset size by generating this file through run.sh, which substitutes NQ.
SET memory_limit='24GB';
SET temp_directory='/var/tmp/ddb_spill';
.timer on
CREATE OR REPLACE TABLE items2   AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE OR REPLACE TABLE queries2 AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet') LIMIT NQ;
.mode trash
.print === TIMED bf lateral, NQ probes x 1M, k=10, projection (qid,iid,d)
SELECT q.qid, t.iid, t.d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
