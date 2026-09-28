INSTALL vss; LOAD vss;
SET memory_limit='32GB'; SET temp_directory='/var/tmp/ddb_spill'; SET threads=64;
SET hnsw_enable_experimental_persistence=true;
CREATE TABLE pa AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_a.parquet');
CREATE TABLE pb AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_b.parquet');
CREATE INDEX pb_hnsw ON pb USING HNSW (vec) WITH (metric = 'cosine');
CHECKPOINT;
.print @@@ E1 does the HNSW index get used for a RANGE predicate?
EXPLAIN SELECT count(*) FROM pa a, pb b WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70;
.print @@@ E2 does it get used for ORDER BY ... LIMIT k (the shape HNSW is built for)?
EXPLAIN SELECT b.id FROM pb b ORDER BY array_cosine_distance(b.vec, (SELECT vec FROM pa LIMIT 1)) LIMIT 5;
.print @@@ E3 hybrid -- is the cheap price predicate evaluated before the vector op?
EXPLAIN SELECT count(*) FROM pa a, pb b
WHERE array_cosine_similarity(a.vec, b.vec) >= 0.70
  AND a.price IS NOT NULL AND b.price IS NOT NULL
  AND b.price BETWEEN a.price * 0.8 AND a.price * 1.25;
