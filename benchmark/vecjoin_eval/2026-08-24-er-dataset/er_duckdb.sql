-- Plain DuckDB (no Sirius) and DuckDB VSS on the same ER threshold join. Reference: 5,329 pairs.
INSTALL vss; LOAD vss;
SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill'; SET threads=64;
SET hnsw_enable_experimental_persistence=true;
CREATE TABLE pa AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_a.parquet');
CREATE TABLE pb AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_b.parquet');
CHECKPOINT;
.print @@@ D1 plain SQL threshold join (expect 5329)
.timer on
SELECT count(*) AS pairs FROM pa a, pb b WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70;
.timer off
.print @@@ D2 the ER question: products with >=2 candidates (expect 673)
.timer on
SELECT count(*) AS products_with_2plus FROM (
  SELECT a.id FROM pa a, pb b WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70
  GROUP BY a.id HAVING count(*) >= 2);
.timer off
.print @@@ D3 HYBRID: threshold AND a real price range predicate
.timer on
SELECT count(*) AS pairs FROM pa a, pb b
WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70
  AND a.price IS NOT NULL AND b.price IS NOT NULL
  AND b.price BETWEEN a.price * 0.8 AND a.price * 1.25;
.timer off
.print @@@ D4 build an HNSW index and try to use it for the RANGE predicate
.timer on
CREATE INDEX pb_hnsw ON pb USING HNSW (vec) WITH (metric = 'cosine');
.timer off
.timer on
SELECT count(*) AS pairs FROM pa a, pb b WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70;
.timer off
