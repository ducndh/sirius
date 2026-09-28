-- ER threshold join in Sirius. Also the first exercise of the ported radius kernel (ef56272b) on
-- COSINE and on a 384-dim dataset -- both new to it. Reference answer: 5,329 pairs at tau=0.70.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE pa AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_a.parquet');
CREATE TABLE pb AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_b.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'pa', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'pb', tier => 'gpu', format => 'duckdb');
.print @@@ Q1 threshold join, cosine similarity floor 0.70 (reference 5329)
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity');
.timer off
.print @@@ Q2 export the pairs for F1 scoring
.mode csv
.output /var/tmp/vj/er_sirius_pairs.csv
SELECT left_id, right_id FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity');
.output
.mode duckbox
.print @@@ Q3 per-left-row match count -- the ER question ("how many candidates per product")
.timer on
SELECT count(*) AS products_with_2plus FROM (
  SELECT left_id FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity')
  GROUP BY left_id HAVING count(*) >= 2);
.timer off
.print @@@ Q4 HYBRID: threshold join AND a real scalar range predicate on price
.timer on
SELECT count(*) AS pairs FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity', left_output_columns => ['id','price'], right_output_columns => ['id','price']);
.timer off
