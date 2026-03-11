-- Smoke tests for gpu_graph_bfs, gpu_graph_pagerank, gpu_graph_wcc
-- Run with: LD_LIBRARY_PATH=".pixi/envs/cuda12/lib:$LD_LIBRARY_PATH" ./build/release/duckdb < test/sql/test_graph_functions.sql

LOAD 'build/release/extension/sirius/sirius.duckdb_extension';

-- Build a small directed graph:
--   A -> B -> C -> D
--        B -> E
--   F -> G  (separate component)
CREATE TABLE edges (src VARCHAR, dst VARCHAR);
INSERT INTO edges VALUES
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'), ('F', 'G');

-- -----------------------------------------------------------------------
-- BFS from A (unlimited depth)
-- Expected: A=0, B=1, C=2, D=3, E=2  (F and G not reachable from A)
-- -----------------------------------------------------------------------
SELECT 'BFS unlimited' AS test;
SELECT vertex, distance, predecessor
FROM gpu_graph_bfs('edges', 'src', 'dst', 'A')
ORDER BY distance, vertex;

-- BFS with depth_limit=1 — only A(0) and B(1)
SELECT 'BFS depth_limit=1' AS test;
SELECT vertex, distance
FROM gpu_graph_bfs('edges', 'src', 'dst', 'A', 1)
ORDER BY distance, vertex;

-- BFS from F — only F(0) and G(1)
SELECT 'BFS from F' AS test;
SELECT vertex, distance
FROM gpu_graph_bfs('edges', 'src', 'dst', 'F')
ORDER BY distance, vertex;

-- -----------------------------------------------------------------------
-- PageRank — all vertices should have positive scores summing to ~1
-- -----------------------------------------------------------------------
SELECT 'PageRank' AS test;
SELECT vertex, ROUND(CAST(pagerank AS DOUBLE), 6) AS pr
FROM gpu_graph_pagerank('edges', 'src', 'dst')
ORDER BY pr DESC;

SELECT ROUND(CAST(SUM(pagerank) AS DOUBLE), 4) AS total_pr_should_be_1
FROM gpu_graph_pagerank('edges', 'src', 'dst');

-- -----------------------------------------------------------------------
-- WCC — should produce 2 components: {A,B,C,D,E} and {F,G}
-- -----------------------------------------------------------------------
SELECT 'WCC' AS test;
SELECT component, COUNT(*) AS size
FROM gpu_graph_wcc('edges', 'src', 'dst')
GROUP BY component
ORDER BY size DESC;

DROP TABLE edges;
