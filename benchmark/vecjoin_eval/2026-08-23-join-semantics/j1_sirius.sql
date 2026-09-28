-- J1, Sirius side: the same query the practitioner path in j1_intermediate_corpus.py answers.
-- The corpus is an INTERMEDIATE RESULT (rows surviving `id % 10 < 3`), so no index can pre-exist.
-- Every stage is timed, because in a join every stage is real work -- the practitioner path is
-- charged for its export and index build, so we are charged for our materialize and pin.
SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CHECKPOINT;
SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');
.timer on
.print @@@ S1 materialize the filtered corpus (the intermediate result)
CREATE TABLE filtered AS SELECT id, vec FROM base WHERE id % 10 < 3;
CHECKPOINT;
.print @@@ S2 pin it
SELECT * FROM pin_table(name => 'filtered', tier => 'gpu', format => 'duckdb');
.print @@@ S3 the join itself, exact
.mode trash
SELECT left_id, right_id FROM sirius_knn_join('query','vec','filtered','vec', k => 10, metric => 'l2');
.timer off
.mode csv
.headers on
.output /var/tmp/vj/j1_sirius.csv
SELECT left_id, right_id FROM sirius_knn_join('query','vec','filtered','vec', k => 10, metric => 'l2');
.output
