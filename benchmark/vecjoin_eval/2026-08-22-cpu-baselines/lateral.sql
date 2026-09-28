-- Hand-written LATERAL: the tuned CPU baseline. Full sort per left row.
.mode trash
.timer on
.print === TIMED LATERAL 10k x 1M k=10
SELECT d FROM queries2 q, LATERAL (
  SELECT array_distance(q.qvec, i.ivec) AS d FROM items2 i ORDER BY d LIMIT 10);
