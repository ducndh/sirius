-- Copyright 2025 Sirius Contributors
--
-- ASOF JOIN integration tests.
-- Run against Sirius GPU DuckDB and plain DuckDB 1.4.4 to cross-validate.
-- Expected results are documented inline.

-- ============================================================
-- Test 1: No partition key — timestamp-only ASOF JOIN
-- ============================================================
CREATE TABLE t1_ts (id INTEGER, ts BIGINT, val DOUBLE);
INSERT INTO t1_ts VALUES (1, 10, 1.0), (2, 20, 2.0), (3, 30, 3.0), (4, 5, 0.5);

CREATE TABLE t2_ts (ts BIGINT, price DOUBLE);
INSERT INTO t2_ts VALUES (3, 30.0), (12, 120.0), (22, 220.0);

-- Expected (ORDER BY id):
--   id=1, ts=10, price=30.0   (last t2 ts before 10 is 3)
--   id=2, ts=20, price=120.0  (last t2 ts before 20 is 12)
--   id=3, ts=30, price=220.0  (last t2 ts before 30 is 22)
--   id=4, ts=5,  price=30.0   (last t2 ts before 5  is 3)
SELECT t1_ts.id, t1_ts.ts, t2_ts.price
FROM t1_ts ASOF JOIN t2_ts ON t1_ts.ts >= t2_ts.ts
ORDER BY t1_ts.id;

DROP TABLE t1_ts;
DROP TABLE t2_ts;

-- ============================================================
-- Test 2: Single partition key (symbol) + timestamp
-- ============================================================
CREATE TABLE trades (symbol INTEGER, ts BIGINT, price DOUBLE);
INSERT INTO trades VALUES
    (1, 100, 10.5),
    (1, 200, 11.0),
    (2, 150, 20.0),
    (2, 300, 21.0);

CREATE TABLE quotes (symbol INTEGER, ts BIGINT, bid DOUBLE, ask DOUBLE);
INSERT INTO quotes VALUES
    (1,  50, 10.0, 10.2),
    (1, 120, 10.4, 10.6),
    (1, 180, 10.8, 11.0),
    (2, 100, 19.8, 20.2),
    (2, 250, 20.5, 20.8);

-- Expected (ORDER BY symbol, ts):
--   (1, 100, 10.5, 10.0, 10.2)   -- quote at ts=50 is last before trade at ts=100
--   (1, 200, 11.0, 10.8, 11.0)   -- quote at ts=180 is last before trade at ts=200
--   (2, 150, 20.0, 19.8, 20.2)   -- quote at ts=100 is last before trade at ts=150
--   (2, 300, 21.0, 20.5, 20.8)   -- quote at ts=250 is last before trade at ts=300
SELECT t.symbol, t.ts, t.price, q.bid, q.ask
FROM trades t ASOF JOIN quotes q
ON t.symbol = q.symbol AND t.ts >= q.ts
ORDER BY t.symbol, t.ts;

DROP TABLE trades;
DROP TABLE quotes;

-- ============================================================
-- Test 3: No match — all left timestamps precede all right timestamps
-- ============================================================
CREATE TABLE early_trades (symbol INTEGER, ts BIGINT, price DOUBLE);
INSERT INTO early_trades VALUES (1, 5, 10.0), (1, 10, 11.0), (2, 8, 20.0);

CREATE TABLE late_quotes (symbol INTEGER, ts BIGINT, bid DOUBLE);
INSERT INTO late_quotes VALUES (1, 100, 50.0), (1, 200, 55.0), (2, 150, 60.0);

-- Expected: all bid values are NULL (no preceding quote exists)
SELECT e.symbol, e.ts, l.bid
FROM early_trades e ASOF JOIN late_quotes l
ON e.symbol = l.symbol AND e.ts >= l.ts
ORDER BY e.symbol, e.ts;

DROP TABLE early_trades;
DROP TABLE late_quotes;

-- ============================================================
-- Test 4: Exact timestamp match (ts = ts, not just ts > ts)
-- ============================================================
CREATE TABLE exact_left (id INTEGER, ts BIGINT);
INSERT INTO exact_left VALUES (1, 10), (2, 20), (3, 15);

CREATE TABLE exact_right (ts BIGINT, val INTEGER);
INSERT INTO exact_right VALUES (10, 100), (20, 200), (15, 150);

-- Expected (ORDER BY id): (1,10,100), (2,20,200), (3,15,150)
-- Exact timestamp matches should be included for >=
SELECT l.id, l.ts, r.val
FROM exact_left l ASOF JOIN exact_right r
ON l.ts >= r.ts
ORDER BY l.id;

DROP TABLE exact_left;
DROP TABLE exact_right;

-- ============================================================
-- Test 5: Empty right table → all NULL right columns
-- ============================================================
CREATE TABLE probe5 (id INTEGER, ts BIGINT);
INSERT INTO probe5 VALUES (1, 10), (2, 20);

CREATE TABLE build5 (ts BIGINT, val INTEGER);
-- intentionally empty

-- Expected: both rows have NULL val
SELECT p.id, p.ts, b.val
FROM probe5 p ASOF JOIN build5 b
ON p.ts >= b.ts
ORDER BY p.id;

DROP TABLE probe5;
DROP TABLE build5;
