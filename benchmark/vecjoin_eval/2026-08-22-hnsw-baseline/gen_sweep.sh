#!/usr/bin/env bash
# Emit the ef_search sweep SQL: build the index ONCE, then vary ef_search per run.
# ef_search is a query-time knob, so one session covers the whole curve.
cat <<'EOF'
LOAD vss;
SET memory_limit='48GB';
CREATE TABLE items2   AS SELECT id AS iid, vec::FLOAT[128] AS ivec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet');
CREATE TABLE queries2 AS SELECT id AS qid, vec::FLOAT[128] AS qvec
  FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet');
CREATE INDEX hnsw_ivec ON items2 USING HNSW (ivec) WITH (metric='l2sq');
EOF
for ef in 10 20 40 80 160 320; do
cat <<EOF
SET hnsw_ef_search = $ef;
.mode trash
.timer on
.print === TIMED ef_search=$ef
SELECT q.qid, t.iid, t.d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.timer off
.mode csv
.headers on
.output /var/tmp/hnsw_ef$ef.csv
SELECT q.qid AS qid, t.iid AS iid, t.d AS d FROM queries2 q, LATERAL
  (SELECT i.iid, array_distance(q.qvec, i.ivec) AS d FROM items2 i
   ORDER BY array_distance(q.qvec, i.ivec) LIMIT 10) t;
.output
EOF
done
