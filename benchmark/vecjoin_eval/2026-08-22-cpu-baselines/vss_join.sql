-- DuckDB's shipped vector join. Brute force; does NOT use the HNSW index.
LOAD vss;
.print === VERIFY (expect min 20.8087 max 361.866 n 100000)
SELECT round(min(score),4) AS min_s, round(max(score),4) AS max_s, count(*) AS n
  FROM vss_join('queries2','items2',qvec,ivec,10);
.mode trash
.timer on
.print === TIMED vss_join 10k x 1M k=10
SELECT score FROM vss_join('queries2','items2',qvec,ivec,10);
