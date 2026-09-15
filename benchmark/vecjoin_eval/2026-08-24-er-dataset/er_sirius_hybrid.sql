-- Can Sirius express the HYBRID shape -- threshold join AND a scalar range predicate relating the
-- two sides' price columns? Plain DuckDB answers it in 0.177 s (1,789 pairs) by filtering on the
-- cheap scalar FIRST. This is I5's last unmeasured shape.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE pa AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_a.parquet');
CREATE TABLE pb AS SELECT id, vec::FLOAT[384] AS vec, price FROM read_parquet('/var/tmp/vj/data/parquet/amazon-google_b.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'pa', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'pb', tier => 'gpu', format => 'duckdb');
.print @@@ H1 hybrid via a WHERE over the join output (expect 1789)
SELECT count(*) AS pairs FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity', left_output_columns => ['id','price'], right_output_columns => ['id','price']) WHERE right_price BETWEEN left_price * 0.8 AND left_price * 1.25;
.print @@@ H2 what are the output column names?
SELECT * FROM sirius_knn_join('pa','vec','pb','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.70, output_type => 'similarity', left_output_columns => ['id','price'], right_output_columns => ['id','price']) LIMIT 2;
