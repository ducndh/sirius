/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_gpu_execution_vector_join_exact_per_row.cpp
 * @brief End-to-end tests for the sirius_knn_join() table function, scoped to the
 *        exact per-row top-k vector join (search_mode exact / exact-gemm). Both sides
 *        must be pinned.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>
#include <vector>

using VectorJoinFixture = sirius::test::GpuExecutionFixture;

namespace {

// Sorted rows from a query that must succeed. Sorting both sides lets us compare
// result sets independent of the arrival order of the join's output rows.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto& mat = r->Cast<duckdb::MaterializedQueryResult>();
  return sirius::test::GpuExecutionFixture::collect_rows(mat, /*sort=*/true);
}

// Single FLOAT scalar from a one-cell query that must succeed (e.g. min/max checks).
float single_float(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return r->GetValue(0, 0).GetValue<float>();
}

// Assert a query fails, and that its error mentions `needle`.
void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

}  // namespace

// -----------------------------------------------------------------------------
// Adversarial large-magnitude L2: `exact` (unexpanded) must be bit-correct; this
// is precisely the input the `exact` mode exists to handle.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact L2 on large-magnitude vectors",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE vj_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vj_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] FROM range(50000) "
    "t(i);");
  run_ok("CREATE TABLE vj_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO vj_probe VALUES "
    "(0, [49800.0, 49801.0, 49802.0]), "
    "(1, [49890.0, 49891.0, 49892.0]), "
    "(2, [49810.0, 49811.0, 49812.0]), "
    "(3, [49780.0, 49781.0, 49782.0]), "
    "(4, [49820.0, 49821.0, 49822.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'vj_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'vj_probe', tier => 'gpu', format => 'duckdb');");

  // k=9 lands on complete, tie-free shells around each probe ({P, P+-1..P+-4} at
  // 0, sqrt3, 2*sqrt3, ...), so the top-k SET is unambiguous. Compare ids only:
  // CPU array_distance vs GPU cuVS agree on membership, but their float distance
  // strings can differ in the last digit.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM vj_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM vj_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 9) n;");
  con->Query("SET gpu_execution = true;");

  // sirius_knn_join in exact mode must match the reference neighbor set exactly.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'vj_probe','vec','vj_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 9);");
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('vj_probe');");
  run_ok("SELECT * FROM unpin_table('vj_corpus');");
}

// -----------------------------------------------------------------------------
// Well-conditioned corpus: exact and exact-gemm must agree neighbor-for-neighbor,
// and both must match the CPU exact reference. This is the GEMM happy-path check.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact and exact-gemm agree on well-conditioned L2",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Magnitudes ~1, directions spread over the sphere -> GEMM is well-conditioned.
  run_ok("CREATE TABLE gemm_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gemm_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(50000) t(i);");
  run_ok("CREATE TABLE gemm_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO gemm_probe SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'gemm_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'gemm_probe', tier => 'gpu', format => 'duckdb');");

  // CPU exact reference (ids only: distances match to ~1e-6, but the neighbor
  // SET is what must be identical, and these directions are tie-free at k=10).
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM gemm_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM gemm_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 10) n;");
  con->Query("SET gpu_execution = true;");

  auto join_ids = [&](const std::string& mode) {
    return ok_rows(*con,
                   "SELECT left_id, right_id FROM sirius_knn_join("
                   "'gemm_probe','vec','gemm_corpus','vec', "
                   "search_mode => '" +
                     mode + "', metric => 'l2', k => 10);");
  };

  auto exact_ids      = join_ids("exact");
  auto exact_gemm_ids = join_ids("exact-gemm");

  // exact-gemm agrees with exact, and both match the CPU reference.
  REQUIRE(exact_ids == reference);
  REQUIRE(exact_gemm_ids == exact_ids);

  run_ok("SELECT * FROM unpin_table('gemm_probe');");
  run_ok("SELECT * FROM unpin_table('gemm_corpus');");
}

// -----------------------------------------------------------------------------
// Self-join (left table == right table, the dedup shape) with k larger than the
// table: k is lowered to the row count, so every row pairs with all rows.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - self-join with k larger than the table",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE sj (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO sj VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0]), "
    "(3, [1.0, 1.0, 0.0]), (4, [2.0, 3.0, 4.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sj', tier => 'gpu', format => 'duckdb');");

  // k = 10 is more than the 5 rows, so it is lowered to 5: every left row pairs
  // with all 5 right rows (including itself at distance 0) -- the full 5x5 set.
  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'sj','vec','sj','vec', search_mode => 'exact', metric => 'l2', k => 10);");

  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con, "SELECT a.id, b.id FROM sj a, sj b;");
  con->Query("SET gpu_execution = true;");

  REQUIRE(joined.size() == 25);
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('sj');");
}

// -----------------------------------------------------------------------------
// Regular self-join: k smaller than the table, so real per-row top-k selection
// happens. Geometric spacing (x = 1,2,4,...) makes every pairwise distance
// distinct, so the top-k is tie-free and matches the CPU reference exactly.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - self-join per-row top-k with k below the table size",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE dedup (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO dedup VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [2.0, 0.0, 0.0]), (2, [4.0, 0.0, 0.0]), "
    "(3, [8.0, 0.0, 0.0]), (4, [16.0, 0.0, 0.0]), (5, [32.0, 0.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'dedup', tier => 'gpu', format => 'duckdb');");

  // Each row's 3 nearest (itself + 2 closest) are unambiguous because all gaps
  // differ; compare the id pairs against DuckDB's own per-row top-3.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM dedup p, LATERAL ("
                           "  SELECT c.id FROM dedup c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 3) n;");
  con->Query("SET gpu_execution = true;");

  auto joined =
    ok_rows(*con,
            "SELECT left_id, right_id FROM sirius_knn_join("
            "'dedup','vec','dedup','vec', search_mode => 'exact', metric => 'l2', k => 3);");

  REQUIRE(joined.size() == 18);  // 6 rows x 3 neighbors
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('dedup');");
}

// -----------------------------------------------------------------------------
// Multi-batch right table: under the test's 1 MB block size a ~300k-row corpus
// spans several pinned batches, so reduce_local merges partials across batches
// (n_parts > 1) -- the cross-batch path a single-batch corpus never runs.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact L2 across multiple right batches",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE mb_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO mb_corpus SELECT i, [i::float, (i+1)::float, (i+2)::float] "
    "FROM range(300000) t(i);");
  run_ok("CREATE TABLE mb_probe (id INTEGER, vec FLOAT[3]);");
  // Probes spread across the corpus so the winning neighbors come from different
  // batches -- the merge has to pick each probe's near shell out of far partials
  // contributed by every other batch.
  run_ok(
    "INSERT INTO mb_probe VALUES "
    "(0, [150000.0, 150001.0, 150002.0]), "
    "(1, [37000.0, 37001.0, 37002.0]), "
    "(2, [260000.0, 260001.0, 260002.0]), "
    "(3, [150090.0, 150091.0, 150092.0]), "
    "(4, [90000.0, 90001.0, 90002.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'mb_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'mb_probe', tier => 'gpu', format => 'duckdb');");

  // k=9 lands on complete, tie-free shells ({P, P+-1..P+-4}); compare neighbor ids
  // to DuckDB's own per-row top-9.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM mb_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM mb_corpus c "
                           "  ORDER BY array_distance(p.vec, c.vec) LIMIT 9) n;");
  con->Query("SET gpu_execution = true;");

  auto joined = ok_rows(*con,
                        "SELECT left_id, right_id FROM sirius_knn_join("
                        "'mb_probe','vec','mb_corpus','vec', "
                        "search_mode => 'exact', metric => 'l2', k => 9);");
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('mb_probe');");
  run_ok("SELECT * FROM unpin_table('mb_corpus');");
}

// -----------------------------------------------------------------------------
// Cosine: correctness on direction-varied vectors, both output types return the
// same neighbor set, and the clamp keeps distance >= 0 and similarity <= 1.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - exact cosine matches CPU and both outputs stay in range",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  // Varied directions so cosine is discriminative (the [i,i+1,i+2] corpus is all
  // parallel -> every pair ~1, degenerate for cosine).
  run_ok("CREATE TABLE cos_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO cos_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(50000) t(i);");
  run_ok("CREATE TABLE cos_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO cos_probe SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(5) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'cos_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'cos_probe', tier => 'gpu', format => 'duckdb');");

  // True cosine top-5 per probe from DuckDB.
  con->Query("SET gpu_execution = false;");
  auto reference = ok_rows(*con,
                           "SELECT p.id, n.id FROM cos_probe p, LATERAL ("
                           "  SELECT c.id, c.vec FROM cos_corpus c "
                           "  ORDER BY array_cosine_distance(p.vec, c.vec) LIMIT 5) n;");
  con->Query("SET gpu_execution = true;");

  // The neighbor set is the same whether the score is reported as distance or
  // similarity -- output_type only changes the score column, not the ranking.
  auto dist_ids = ok_rows(*con,
                          "SELECT left_id, right_id FROM sirius_knn_join("
                          "'cos_probe','vec','cos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 5, "
                          "output_type => 'distance');");
  auto sim_ids  = ok_rows(*con,
                         "SELECT left_id, right_id FROM sirius_knn_join("
                          "'cos_probe','vec','cos_corpus','vec', "
                          "search_mode => 'exact', metric => 'cosine', k => 5, "
                          "output_type => 'similarity');");
  REQUIRE(dist_ids == reference);
  REQUIRE(sim_ids == reference);

  // Clamp holds on the self-matches (probe i == corpus i): distance stays >= 0
  // (would read a hair below 0 without the floor) and similarity stays <= 1.
  REQUIRE(single_float(*con,
                       "SELECT min(distance) FROM sirius_knn_join("
                       "'cos_probe','vec','cos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 5, "
                       "output_type => 'distance');") >= 0.0F);
  REQUIRE(single_float(*con,
                       "SELECT max(similarity) FROM sirius_knn_join("
                       "'cos_probe','vec','cos_corpus','vec', "
                       "search_mode => 'exact', metric => 'cosine', k => 5, "
                       "output_type => 'similarity');") <= 1.0F);

  run_ok("SELECT * FROM unpin_table('cos_probe');");
  run_ok("SELECT * FROM unpin_table('cos_corpus');");
}

// -----------------------------------------------------------------------------
// L2 has no natural similarity, so the combination is rejected at bind.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - l2 with similarity output is rejected",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE rej (id INTEGER, vec FLOAT[3]);");
  run_ok("INSERT INTO rej VALUES (0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'rej', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('rej','vec','rej','vec', "
               "metric => 'l2', output_type => 'similarity');",
               "only meaningful for metric => 'cosine'");

  run_ok("SELECT * FROM unpin_table('rej');");
}

// -----------------------------------------------------------------------------
// Out-of-core: a HOST-tier corpus is streamed chunk-by-chunk through the fused
// operator's running top-k instead of being GPU-resident, and must produce exactly
// what the GPU-resident corpus produces. Multiple right batches, so the fold is the
// thing under test. Streaming-path only -- the split operators require GPU residency.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - host-tier corpus matches a GPU-tier corpus",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  setenv("SIRIUS_VECTOR_JOIN_STREAMING", "1", 1);

  run_ok("CREATE TABLE oc_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO oc_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE oc_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO oc_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(64) t(i);");
  run_ok("CHECKPOINT;");

  const std::string join_sql =
    "SELECT left_id, right_id FROM sirius_knn_join("
    "'oc_probe','vec','oc_corpus','vec', search_mode => 'exact', metric => 'l2', k => 8);";

  run_ok("SELECT * FROM pin_table(name => 'oc_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'oc_corpus', tier => 'gpu', format => 'duckdb');");
  auto const gpu_tier_rows = ok_rows(*con, join_sql);
  REQUIRE(gpu_tier_rows.size() == 64 * 8);
  run_ok("SELECT * FROM unpin_table('oc_corpus');");

  run_ok("SELECT * FROM pin_table(name => 'oc_corpus', tier => 'host', format => 'duckdb');");
  auto const host_tier_rows = ok_rows(*con, join_sql);

  REQUIRE(host_tier_rows == gpu_tier_rows);

  run_ok("SELECT * FROM unpin_table('oc_corpus');");
  run_ok("SELECT * FROM unpin_table('oc_probe');");
  unsetenv("SIRIUS_VECTOR_JOIN_STREAMING");
}

// -----------------------------------------------------------------------------
// Global top-k: the k closest pairs over the whole join, not k per left row. The CPU
// reference is an exhaustive cross join ordered by distance, which is the definition.
// The corpus spans several right batches so the fold, the per-batch depth bound and
// the TOP_N above materialize are all exercised together.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - global top-k matches an exhaustive CPU ranking",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE g_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO g_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE g_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO g_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(32) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'g_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'g_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM g_probe p, g_corpus c "
                                 "ORDER BY array_distance(p.vec, c.vec) LIMIT 12;");
  con->Query("SET gpu_execution = true;");

  auto const joined = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'g_probe','vec','g_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 12, join_mode => 'global', "
                              "left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(joined.size() == 12);
  REQUIRE(joined == reference);

  run_ok("SELECT * FROM unpin_table('g_corpus');");
  run_ok("SELECT * FROM unpin_table('g_probe');");
}

// -----------------------------------------------------------------------------
// Threshold (range) join: every pair within eps, which is ragged by construction --
// left rows contribute anywhere from zero to many pairs. This is the case the old
// fixed-k output contract could not represent at all, so it is also the regression
// test for materialize gathering by an explicit left row instead of repeating.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold join matches an exhaustive CPU range query",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE t_corpus (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO t_corpus SELECT i, [(i%100)::float, (i/100)::float] FROM range(10000) t(i);");
  run_ok("CREATE TABLE t_probe (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO t_probe VALUES (0,[10.0,10.0]),(1,[50.5,50.5]),(2,[99.0,99.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 't_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 't_corpus', tier => 'gpu', format => 'duckdb');");

  con->Query("SET gpu_execution = false;");
  auto const reference = ok_rows(*con,
                                 "SELECT p.id, c.id FROM t_probe p, t_corpus c "
                                 "WHERE array_distance(p.vec, c.vec) <= 2.0;");
  con->Query("SET gpu_execution = true;");

  auto const joined = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'t_probe','vec','t_corpus','vec', search_mode => 'exact', "
                              "metric => 'l2', k => 64, join_mode => 'threshold', eps => 2.0, "
                              "left_output_columns => ['id'], right_output_columns => ['id']);");
  REQUIRE(joined == reference);
  // Ragged by construction: this is not a whole multiple of the probe row count.
  REQUIRE(joined.size() % 3 != 0);

  run_ok("SELECT * FROM unpin_table('t_corpus');");
  run_ok("SELECT * FROM unpin_table('t_probe');");
}

// -----------------------------------------------------------------------------
// A threshold join searches each left row only to depth k, so an eps loose enough to
// admit more than k neighbours cannot be answered completely. It must say so rather
// than return a silently truncated result set.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - threshold join refuses to truncate",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE tt_corpus (id INTEGER, vec FLOAT[2]);");
  run_ok(
    "INSERT INTO tt_corpus SELECT i, [(i%100)::float, (i/100)::float] FROM range(10000) t(i);");
  run_ok("CREATE TABLE tt_probe (id INTEGER, vec FLOAT[2]);");
  run_ok("INSERT INTO tt_probe VALUES (0,[50.0,50.0]);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'tt_probe',  tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'tt_corpus', tier => 'gpu', format => 'duckdb');");

  // eps => 50 admits thousands of pairs; k => 4 cannot represent them.
  expect_error(*con,
               "SELECT left_id, right_id FROM sirius_knn_join("
               "'tt_probe','vec','tt_corpus','vec', search_mode => 'exact', metric => 'l2', "
               "k => 4, join_mode => 'threshold', eps => 50.0, "
               "left_output_columns => ['id'], right_output_columns => ['id']);",
               "truncated");

  run_ok("SELECT * FROM unpin_table('tt_corpus');");
  run_ok("SELECT * FROM unpin_table('tt_probe');");
}

// -----------------------------------------------------------------------------
// Query-side partitioning: the probe side no longer has to be device-resident either.
// A task owns one probe chunk and searches the whole corpus against it, so both sides
// can exceed VRAM at once -- the shape every rec-sys candidate-generation join has.
// Answers must not depend on where either side was pinned.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(VectorJoinFixture,
                 "sirius_knn_join - host-tier probe side matches a GPU-tier probe side",
                 "[integration][gpu_execution][array][vss][vector_join]")
{
  run_ok("CREATE TABLE qp_corpus (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO qp_corpus SELECT i, "
    "[sin(i)::float, cos(i*1.3)::float, sin(i*0.7)::float] FROM range(60000) t(i);");
  run_ok("CREATE TABLE qp_probe (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO qp_probe SELECT i, "
    "[sin(i*2.1)::float, cos(i*0.9)::float, sin(i*1.7)::float] FROM range(50000) t(i);");
  run_ok("CHECKPOINT;");

  const std::string join_sql =
    "SELECT left_id, right_id FROM sirius_knn_join("
    "'qp_probe','vec','qp_corpus','vec', search_mode => 'exact', metric => 'l2', k => 8, "
    "left_output_columns => ['id'], right_output_columns => ['id']);";

  run_ok("SELECT * FROM pin_table(name => 'qp_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'qp_probe',  tier => 'gpu', format => 'duckdb');");
  auto const gpu_probe_rows = ok_rows(*con, join_sql);
  REQUIRE(gpu_probe_rows.size() == 50000 * 8);
  run_ok("SELECT * FROM unpin_table('qp_probe');");

  // Probe streamed, corpus resident.
  run_ok("SELECT * FROM pin_table(name => 'qp_probe', tier => 'host', format => 'duckdb');");
  REQUIRE(ok_rows(*con, join_sql) == gpu_probe_rows);
  run_ok("SELECT * FROM unpin_table('qp_corpus');");

  // Both sides streamed: neither has to fit device memory.
  run_ok("SELECT * FROM pin_table(name => 'qp_corpus', tier => 'host', format => 'duckdb');");
  REQUIRE(ok_rows(*con, join_sql) == gpu_probe_rows);

  run_ok("SELECT * FROM unpin_table('qp_corpus');");
  run_ok("SELECT * FROM unpin_table('qp_probe');");
}
