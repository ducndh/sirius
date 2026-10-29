/*
 * Copyright 2025, Sirius Contributors.
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
 * @file test_gpu_execution_vector_join_sql_rewrite.cpp
 * @brief Plain-SQL vector joins (no sirius_knn_join call) run as the vector-join operator: each
 *        shape must run on the GPU -- without the rewrite DuckDB's plan for it falls back to the
 *        CPU -- and return what DuckDB returns for the same statement.
 */

#include "duckdb/main/config.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <vector>

using SqlRewriteFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Rows of @p sql, sorted.
std::vector<std::vector<std::string>> sorted_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  auto rows = sirius::test::GpuExecutionFixture::collect_rows(
    r->Cast<duckdb::MaterializedQueryResult>(), /*sort=*/true);
  return rows;
}

/// Equal rows, numbers within 1e-4 relative: the GPU scores pairs with a GEMM and DuckDB
/// directly, and the two round differently in the last places.
bool rows_match(const std::vector<std::vector<std::string>>& a,
                const std::vector<std::vector<std::string>>& b)
{
  if (a.size() != b.size()) { return false; }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) { return false; }
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      if (a[i][j] == b[i][j]) { continue; }
      char* ea       = nullptr;
      char* eb       = nullptr;
      double const x = std::strtod(a[i][j].c_str(), &ea);
      double const y = std::strtod(b[i][j].c_str(), &eb);
      if (*ea != '\0' || *eb != '\0') { return false; }
      if (std::fabs(x - y) > 1e-4 * std::max(1.0, std::fabs(y))) { return false; }
    }
  }
  return true;
}

/// Run @p sql on the GPU (asserting it ran there, one rebind and one execution) and on DuckDB,
/// and require the same rounded rows.
void require_gpu_matches_duckdb(duckdb::Connection& con, const std::string& sql)
{
  con.Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(con, sql);
  con.Query("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(con);
  auto const got    = sorted_rows(con, sql);
  auto const after  = sirius::test::get_transparent_execution_stats(con);
  UNSCOPED_INFO("query: " << sql);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(got.size() == expected.size());
  // Every query here leads with the row's ids, so both sides sort into the same order.
  REQUIRE(rows_match(got, expected));
}

}  // namespace

// Vectors from a hash, so no two corpus rows share a distance to a probe (ties would make a
// top-k answer a matter of tie-breaking), and a small category column to join and filter on.
class SqlRewriteTables {
 public:
  explicit SqlRewriteTables(SqlRewriteFixture& f) : _f(f)
  {
    // The database's optimizer mask is shared, and an earlier test in a full run can leave it
    // cleared; compressed materialization then wraps small-range columns in functions the GPU
    // plan does not translate, and the queries here run on DuckDB instead. Pin the mask Sirius
    // publishes at load while these tables exist.
    auto& disabled  = duckdb::DBConfig::GetConfig(*f.con->context).options.disabled_optimizers;
    _saved_disabled = disabled;
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    auto const vec = [](const std::string& seed) {
      return "list_transform(range(8), lambda d: ((hash(i * 100 + d + " + seed +
             ") % 2000)::FLOAT / 1000.0 - 1.0))::FLOAT[8]";
    };
    f.run_ok("CREATE TABLE sr_corpus AS SELECT i::INTEGER AS id, (i % 20)::INTEGER AS cat, " +
             vec("0") + " AS vec FROM range(20000) t(i);");
    f.run_ok("CREATE TABLE sr_probe AS SELECT i::INTEGER AS id, (i % 20)::INTEGER AS cat, " +
             vec("77777777") + " AS vec FROM range(30) t(i);");
    f.run_ok("CREATE TABLE sr_one AS SELECT 0::INTEGER AS id, " + vec("55555555") +
             " AS vec FROM range(1) t(i);");
    f.run_ok(
      "CREATE TABLE sr_cats AS SELECT i::INTEGER AS cat, 'c' || i AS name FROM range(20) t(i);");
    f.run_ok("CHECKPOINT;");
    for (auto const* t : {"sr_corpus", "sr_probe", "sr_one", "sr_cats"}) {
      f.run_ok(std::string("SELECT * FROM pin_table(name => '") + t +
               "', tier => 'gpu', format => 'duckdb');");
    }
  }
  ~SqlRewriteTables()
  {
    for (auto const* t : {"sr_corpus", "sr_probe", "sr_one", "sr_cats"}) {
      _f.con->Query(std::string("SELECT * FROM unpin_table('") + t + "');");
    }
    duckdb::DBConfig::GetConfig(*_f.con->context).options.disabled_optimizers = _saved_disabled;
  }

 private:
  SqlRewriteFixture& _f;
  std::set<duckdb::OptimizerType> _saved_disabled;
};

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // l2 distance bound, the distance read back in the select list.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_distance(p.vec, c.vec) AS d FROM sr_probe p, "
    "sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  // cosine similarity bound, with the distance form of the same pair in the select list.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_cosine_distance(p.vec, c.vec) AS d FROM sr_probe p, sr_corpus c "
    "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.8;");
  // strict comparison and the constant on the left.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe p, sr_corpus c "
                             "WHERE 0.9 > array_distance(p.vec, c.vec);");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "count(*) over a threshold join is counted inside the join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // The aggregate directly over the join: the join counts, the aggregate sums the batches.
  std::string const counted =
    "SELECT count(*) FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;";
  require_gpu_matches_duckdb(*con, counted);
  REQUIRE(con->ExtractPlan(counted)->ToString().find("sum_no_overflow") != std::string::npos);
  // The explicit call, a cosine threshold, against the same count on DuckDB.
  std::string const called =
    "SELECT count(*) FROM sirius_knn_join('sr_probe','vec','sr_corpus','vec', metric => "
    "'cosine', join_mode => 'threshold', eps => 0.8);";
  REQUIRE(con->ExtractPlan(called)->ToString().find("sum_no_overflow") != std::string::npos);
  run_ok("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con,
                                    "SELECT count(*) FROM sr_probe p, sr_corpus c "
                                    "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.8;");
  run_ok("SET gpu_execution = true;");
  auto const got = sorted_rows(*con, called);
  REQUIRE(got == expected);
  // Shapes the pushdown must leave alone still answer: a second aggregate, groups, a filter
  // above the join.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*), min(array_distance(p.vec, c.vec)) FROM sr_probe "
                             "p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.cat, count(*) FROM sr_probe p, sr_corpus c WHERE "
                             "array_distance(p.vec, c.vec) <= 0.9 GROUP BY p.cat;");
  std::string const filtered =
    "SELECT count(*) FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9 "
    "AND p.id <> c.id;";
  require_gpu_matches_duckdb(*con, filtered);
  REQUIRE(con->ExtractPlan(filtered)->ToString().find("sum_no_overflow") == std::string::npos);
  // Over the lists: the count equals the number of pairs the same approximate join returns.
  run_ok("SET vector_join_probes = 2;");
  auto const approx_rows = sorted_rows(
    *con,
    "SELECT p.id, c.id FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;");
  auto const approx_count = sorted_rows(*con, counted);
  run_ok("RESET vector_join_probes;");
  REQUIRE(approx_count.size() == 1);
  REQUIRE(approx_count[0][0] == std::to_string(approx_rows.size()));
  // Enough probe rows for the kernel to score them in several tiles, so the count crosses tile
  // boundaries (a 512 MB score tile holds 6 400 rows against this corpus).
  run_ok(
    "CREATE TABLE sr_probe_big AS SELECT i::INTEGER AS id, list_transform(range(8), lambda "
    "d: ((hash(i * 100 + d + 31337) % 2000)::FLOAT / 1000.0 - 1.0))::FLOAT[8] AS vec FROM "
    "range(15000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_probe_big', tier => 'gpu', format => 'duckdb');");
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe_big p, sr_corpus c WHERE "
                             "array_distance(p.vec, c.vec) <= 1.2;");
  con->Query("SELECT * FROM unpin_table('sr_probe_big');");
  run_ok("DROP TABLE sr_probe_big;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a range on the similarity or the distance runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // DuckDB folds two bounds on one expression into a BETWEEN. The join takes the bound that
  // makes it a threshold join and the other is applied to the score it returns.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p, sr_corpus c "
                             "WHERE array_cosine_similarity(p.vec, c.vec) BETWEEN 0.7 AND 0.9;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, array_distance(p.vec, c.vec) AS d FROM sr_probe p, "
    "sr_corpus c WHERE array_distance(p.vec, c.vec) BETWEEN 0.5 AND 0.9;");
  // The upper bound sits above the join, on the score the select list reads.
  require_gpu_matches_duckdb(
    *con,
    "SELECT id, cid FROM (SELECT p.id, c.id AS cid, array_cosine_similarity(p.vec, c.vec) AS s "
    "FROM sr_probe p JOIN sr_corpus c ON array_cosine_similarity(p.vec, c.vec) >= 0.7) WHERE s < "
    "0.9;");
  // Strict on both sides.
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM sr_probe p, sr_corpus c "
                             "WHERE array_distance(p.vec, c.vec) > 0.5 AND array_distance(p.vec, "
                             "c.vec) < 0.9;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join keeps the corpus-side join and filter",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id, k.name FROM sr_probe p, sr_corpus c, sr_cats k WHERE c.cat = k.cat AND "
    "k.cat < 5 AND array_cosine_similarity(p.vec, c.vec) >= 0.7;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain-SQL threshold join under a <> join condition runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB makes `p.cat <> c.cat` the join and the similarity a filter above it; the join keeps
  // nearly every pair, so it becomes a filter over the vector join. `c.cat < 10` is pushed into
  // the corpus scan, and from there into the operator.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON array_cosine_similarity(p.vec, c.vec) "
    ">= 0.7 WHERE p.cat <> c.cat AND c.cat < 10;");
  // A filtered probe streams in as a relation; a bare pinned one is read from its pin.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, c.id FROM (SELECT * FROM sr_probe WHERE id < 20) p JOIN sr_corpus c ON "
    "array_cosine_similarity(p.vec, c.vec) >= 0.7 WHERE p.cat <> c.cat;");
  // One range condition too, as a self-join counts each pair once; it keeps about half the pairs.
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, b.id FROM sr_corpus a JOIN sr_corpus b ON array_distance(a.vec, b.vec) <= 0.6 "
    "WHERE a.id < b.id AND a.cat <> b.cat;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL top-k with a join DuckDB moved into it runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB's join order puts the join with sr_cats inside the LATERAL's subquery side.
  require_gpu_matches_duckdb(
    *con,
    "SELECT k.name, count(*), round(avg(n.d), 4) FROM sr_probe p, LATERAL (SELECT c.cat, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 5) n JOIN sr_cats k ON "
    "n.cat = k.cat WHERE k.cat <> 3 GROUP BY k.name;");
  // A join on a probe column pushed in makes the LATERAL correlated on that column too.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, k.name, n.id FROM sr_probe p, LATERAL (SELECT c.id, c.cat, "
    "array_cosine_similarity(p.vec, c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n JOIN "
    "sr_cats k ON k.cat = p.cat WHERE n.cat <> p.cat;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL LIMIT 1 runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // DuckDB decorrelates LIMIT 1 without the list and unnest: max(similarity) when only the score
  // is selected, arg_max(value, similarity) otherwise.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, round(t.s, 5) FROM sr_probe p, LATERAL (SELECT array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 1) t;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, t.id, round(t.d, 5) FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, "
    "c.vec) AS d FROM sr_corpus c WHERE c.cat = 3 ORDER BY d LIMIT 1) t;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "LATERAL top-k runs as the per-row vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  // A filter inside the subquery restricts the corpus before the top-k, as LATERAL's does.
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c WHERE c.cat = 3 ORDER BY s DESC LIMIT 4) n;");
  // Counting the matches reads none of the join's columns, the probe's vector included.
  require_gpu_matches_duckdb(
    *con,
    "SELECT count(*) FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "ORDER BY distance to a scalar-subquery vector runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT id, array_cosine_distance(vec, (SELECT vec FROM sr_one)) AS d "
                             "FROM sr_corpus ORDER BY d LIMIT 25;");
  // Every row's distance, joined and ranked afterwards.
  require_gpu_matches_duckdb(
    *con,
    "SELECT r.id, k.name, r.d FROM sr_cats k JOIN (SELECT id, cat, array_distance(vec, (SELECT vec "
    "FROM sr_one)) AS d FROM sr_corpus) r ON k.cat = r.cat WHERE k.cat = 7 ORDER BY r.d, r.id "
    "LIMIT 30;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a scalar subquery with more than one row still fails",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  auto r = con->Query(
    "SELECT id FROM sr_corpus ORDER BY array_distance(vec, (SELECT vec FROM sr_probe)) LIMIT 3;");
  REQUIRE(r);
  REQUIRE(r->HasError());
  REQUIRE(r->GetError().find("More than one row") != std::string::npos);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a vector read above the join leaves the plan to DuckDB",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // c.vec itself is selected, which the operator does not emit: no rewrite, and the CPU answer.
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con,
                                    "SELECT p.id, c.vec FROM sr_probe p, sr_corpus c "
                                    "WHERE array_distance(p.vec, c.vec) <= 0.6;");
  con->Query("SET gpu_execution = true;");
  auto const got = sorted_rows(*con,
                               "SELECT p.id, c.vec FROM sr_probe p, sr_corpus c "
                               "WHERE array_distance(p.vec, c.vec) <= 0.6;");
  REQUIRE(got == expected);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a top-k rewritten under a correlated subquery keeps DuckDB's answer",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // Vec-H q2's shape, scaled down: the top-k joins beside a correlated subquery that repeats the
  // outer join, under a filter that prunes columns by position. The rewritten operator emits the
  // columns the top-k did, in the same positions, so nothing above it is re-laid out.
  require_gpu_matches_duckdb(
    *con,
    "SELECT r.id, k.name, c.cat, r.d FROM sr_cats k, sr_corpus c, (SELECT id, array_distance(vec, "
    "(SELECT vec FROM sr_one)) AS d FROM sr_corpus ORDER BY d LIMIT 40) r WHERE k.cat = c.cat AND "
    "c.id = r.id AND length(k.name) >= 2 AND c.id >= (SELECT min(c2.id) FROM sr_cats k2, sr_corpus "
    "c2 WHERE k2.cat = c2.cat AND length(k2.name) >= 2 AND c2.cat = c.cat) ORDER BY r.d, r.id;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a vector join in a plan Sirius declines elsewhere still runs on DuckDB",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // The correlated scalar subquery plans as a SINGLE join, which the GPU does not run. The
  // rewritten join has no CPU form, so the rewrite must be undone and DuckDB answer as usual.
  auto const sql =
    "SELECT r.id, k.name, r.d FROM sr_cats k, (SELECT id, cat, array_distance(vec, (SELECT vec "
    "FROM "
    "sr_one)) AS d FROM sr_corpus ORDER BY d LIMIT 40) r WHERE k.cat = r.cat AND r.id = (SELECT "
    "min(c2.id) FROM sr_corpus c2 WHERE c2.cat = k.cat AND c2.id >= r.id) ORDER BY r.d, r.id;";
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con, sql);
  con->Query("SET gpu_execution = true;");
  auto const got = sorted_rows(*con, sql);
  REQUIRE(got.size() == expected.size());
  REQUIRE(rows_match(got, expected));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "plain SQL searches through cluster lists that answer exactly",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  // FLOAT32 lists (these values are not bytes) answer exactly once every cluster is probed, but
  // hold the same bytes as the pin, so by cost the rewrite keeps brute force: the prune
  // statistics, which only the clustered search records, stay put. Forced through the lists they
  // move, and the answers stay DuckDB's. The lists hold rows as they are, not unit rows, so a
  // cosine join keeps the exact GEMM either way.
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_corpus','vec', name => 'sr_c', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('sr_corpus','vec','sr_c');");
  auto const unforced = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive ==
        unforced.pairs_exhaustive);
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;");
  auto const topk = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(topk.pairs_exhaustive > before.pairs_exhaustive);
  CHECK(topk.pairs_scored - before.pairs_scored == topk.pairs_exhaustive - before.pairs_exhaustive);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p, sr_corpus c "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");
  auto const radius = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(radius.pairs_exhaustive > topk.pairs_exhaustive);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id FROM sr_probe p, LATERAL (SELECT c.id, array_cosine_similarity(p.vec, "
    "c.vec) AS s FROM sr_corpus c ORDER BY s DESC LIMIT 4) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive ==
        radius.pairs_exhaustive);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a band join can run as a vector join with the band as a filter",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Two range conditions between the sides (a band) next to a distance threshold;
  // SIRIUS_VSS_BAND_VECTOR_FIRST=1 takes the vector join first with the band as a filter whatever
  // the estimate says, on the GPU, with DuckDB's answer.
  SqlRewriteTables tables(*this);
  ::setenv("SIRIUS_VSS_BAND_VECTOR_FIRST", "1", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_BAND_VECTOR_FIRST"); }
  } unset_band;
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
                             "c.cat BETWEEN p.cat - 2 AND p.cat + 2 "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a self-join DuckDB shares as a materialized CTE is still rewritten",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // The same relation on both sides (probe joined to its category, twice): DuckDB's common-subplan
  // pass turns it into a CTE scanned twice, which carries no vectors to search until the rewrite
  // puts it back in place.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT a.id, b.id FROM sr_probe a JOIN sr_cats ca ON a.cat = ca.cat "
                             "JOIN sr_probe b ON array_distance(a.vec, b.vec) <= 1.5 "
                             "JOIN sr_cats cb ON b.cat = cb.cat WHERE a.id < b.id;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a band join's own statistics decide whether the vector join goes first",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // cat is 0..19, so a +-2 band keeps about 4/19 of the pairs: vector join first, on the GPU. id
  // is 0..19,999, so a band one wide keeps about 1/20,000: left to DuckDB's band join, same answer.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
                             "c.cat BETWEEN p.cat - 2 AND p.cat + 2 "
                             "WHERE array_distance(p.vec, c.vec) <= 0.9;");

  auto const narrow =
    "SELECT p.id, c.id FROM sr_probe p JOIN sr_corpus c ON "
    "c.id BETWEEN p.id * 600 AND p.id * 600 + 1 WHERE array_distance(p.vec, c.vec) <= 3.0;";
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con, narrow);
  con->Query("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(*con, narrow);
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 0, 1, 0);
  REQUIRE(!expected.empty());
  REQUIRE(rows_match(got, expected));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "SET vector_join_probes makes a plain SQL join approximate through the lists",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  SqlRewriteTables tables(*this);
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_corpus','vec', name => 'sr_c', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('sr_corpus','vec','sr_c');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clustering;");
    }
  } reset{*con};
  // Ids are row positions, so the table function's answer reads as the plain SQL one does.
  auto const topk_sql =
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM sr_corpus c ORDER BY d LIMIT 7) n;";
  auto const radius_sql =
    "SELECT p.id, c.id FROM sr_probe p, sr_corpus c WHERE array_distance(p.vec, c.vec) <= 0.9;";
  auto const tvf = [](std::string const& args) {
    return "sirius_knn_join('sr_probe','vec','sr_corpus','vec', search_mode => 'approx', "
           "clustering => 'sr_c', n_probes => 2, " +
           args + ")";
  };
  auto const topk_tvf =
    sorted_rows(*con, "SELECT left_id, right_id, distance FROM " + tvf("k => 7") + ";");
  auto const radius_tvf = sorted_rows(
    *con, "SELECT left_id, right_id FROM " + tvf("join_mode => 'threshold', eps => 0.9") + ";");

  run_ok("SET vector_join_probes = 2;");
  for (auto const& [sql, expected] :
       {std::pair{topk_sql, topk_tvf}, std::pair{radius_sql, radius_tvf}}) {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto const got    = sorted_rows(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    sirius::test::require_transparent_execution_delta(
      before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
    REQUIRE(rows_match(got, expected));
  }
  // Two of eight clusters drop some true neighbours, so the approximate answer is not DuckDB's.
  con->Query("SET gpu_execution = false;");
  auto const exact = sorted_rows(*con, topk_sql);
  con->Query("SET gpu_execution = true;");
  CHECK_FALSE(rows_match(topk_tvf, exact));

  // Naming the clustering selects the same lists; a name that is not this column's is ignored.
  run_ok("SET vector_join_clustering = 'sr_c';");
  CHECK(rows_match(sorted_rows(*con, topk_sql), topk_tvf));
  run_ok("SET vector_join_clustering = 'no_such_clustering';");
  CHECK(rows_match(sorted_rows(*con, topk_sql), topk_tvf));

  // Every cluster probed, or the setting back at 0, is the exact answer again.
  run_ok("SET vector_join_probes = 8;");
  require_gpu_matches_duckdb(*con, topk_sql);
  run_ok("RESET vector_join_probes;");
  require_gpu_matches_duckdb(*con, topk_sql);
  require_gpu_matches_duckdb(*con, radius_sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "SET vector_join_probes with no lists builds them inside the query",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // No clustering exists for the corpus: the setting has the rewrite fit one and write its lists
  // inside the statement, then search ceil(c * sqrt(N)) of them: 4,000 rows get sqrt(4,000) = 63
  // clusters (no more, which would leave a cluster under 256 rows), and c = 0.25 is
  // ceil(0.25 * 7.94) = 2 of those. The lists
  // stay, so the table function over the auto-built clustering returns the same answer, and a c
  // that reaches every cluster (c >= sqrt(N)) is DuckDB's.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_fresh AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_fresh_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_fresh', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit() { con.Query("RESET vector_join_probes;"); }
  } reset{*con};
  auto const sql =
    "SELECT p.id, n.id, n.d FROM sr_fresh_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_fresh c ORDER BY d LIMIT 5) n;";
  run_ok("SET vector_join_probes = 0.25;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(*con, sql);
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
  auto centroids = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_sr_fresh_vec');");
  REQUIRE_FALSE(centroids->HasError());
  auto const n_clusters = centroids->GetValue(0, 0).GetValue<std::int64_t>();
  REQUIRE(n_clusters == 63);
  auto const tvf = sorted_rows(
    *con,
    "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_fresh_probe','vec','sr_fresh',"
    "'vec', probe_source => 'scan', search_mode => 'approx', "
    "clustering => '__sirius_auto_sr_fresh_vec', n_probes => 2, k => 5);");
  REQUIRE(rows_match(got, tvf));
  run_ok("SET vector_join_probes = " + std::to_string(n_clusters) + ";");
  require_gpu_matches_duckdb(*con, sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "vector_join_clusters sets the count of an in-query build, and probes follow it",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // An explicit vector_join_clusters builds that many lists and searches c of them, a count of
  // clusters rounded up: c = 2 is 2 of 252, c = 2.5 is 3. With the setting back at 0 the kept
  // lists are Sirius's own, and c is a multiple of the square root of their count:
  // c = 0.25 is ceil(0.25 * sqrt(252)) = 4 of 252, and c = 16 (past sqrt(252) = 15.9) all of them.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_count AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_count_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_count', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clusters;");
    }
  } reset{*con};
  auto const sql =
    "SELECT p.id, n.id, n.d FROM sr_count_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_count c ORDER BY d LIMIT 5) n;";
  auto const tvf = [&](int probes) {
    return sorted_rows(
      *con,
      "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_count_probe','vec','sr_count',"
      "'vec', probe_source => 'scan', search_mode => 'approx', "
      "clustering => '__sirius_auto_sr_count_vec', n_probes => " +
        std::to_string(probes) + ", k => 5);");
  };
  run_ok("SET vector_join_clusters = 252;");
  run_ok("SET vector_join_probes = 2;");
  auto const fixed = sorted_rows(*con, sql);
  auto centroids   = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
      "sirius_kmeans_centroids('__sirius_auto_sr_count_vec');");
  REQUIRE_FALSE(centroids->HasError());
  REQUIRE(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 252);
  auto const two = tvf(2), three = tvf(3), four = tvf(4);
  REQUIRE_FALSE(rows_match(two, four));
  REQUIRE_FALSE(rows_match(two, three));
  CHECK(rows_match(fixed, two));
  run_ok("SET vector_join_probes = 2.5;");
  CHECK(rows_match(sorted_rows(*con, sql), three));

  run_ok("RESET vector_join_clusters;");
  run_ok("SET vector_join_probes = 0.25;");
  CHECK(rows_match(sorted_rows(*con, sql), four));
  run_ok("SET vector_join_probes = 16;");
  require_gpu_matches_duckdb(*con, sql);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "vector_join_probes = c visits ceil(c * sqrt(N)) of the N clusters Sirius sized",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // 4,096 clusters of 65,536 rows, built under vector_join_clusters (c = 8 is then 8 clusters)
  // and kept under Sirius's own name. With vector_join_clusters back at 0 the rewrite takes N
  // from the lists: c = 0.5 visits ceil(0.5 * sqrt(4096)) = 32 of them, c = 64 (= sqrt(4096))
  // every one, the exact answer. A negative c is refused and leaves the setting as it was; c = 0 is
  // the exact join, with no lists searched approximately. The rows are not bytes, so the lists are
  // FLOAT16 with their FP32 rows, every candidate re-scored: no ties to break differently from
  // DuckDB.
  SqlRewriteTables tables(*this);
  auto const floats = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: ((hash(i * 100 + d + " + seed +
           ") % 100000) / 977.0)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_c4k AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(65536) t(i);");
  run_ok("CREATE TABLE sr_c4k_probe AS SELECT i::INTEGER AS id, " + floats("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_c4k', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clusters;");
    }
  } reset{*con};
  auto const sql =
    "SELECT p.id, n.id, n.d FROM sr_c4k_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_c4k c ORDER BY d LIMIT 5) n;";
  // The rewrite's "searching ... through X of the Y clusters" lines while @p run runs.
  auto const probe_lines = [](auto&& run) {
    sirius::test::scoped_recording_log_sink log{"info"};
    run();
    std::vector<std::string> lines;
    for (auto const& r : log.records()) {
      if (r.message.find("(vector_join_probes") != std::string::npos) {
        lines.push_back(r.message);
      }
    }
    return lines;
  };
  auto const require_through = [](std::vector<std::string> const& lines, std::string const& what) {
    REQUIRE_FALSE(lines.empty());
    for (auto const& line : lines) {
      UNSCOPED_INFO("log: " << line);
      CHECK(line.find("searching 'sr_c4k' through " + what + " clusters") != std::string::npos);
    }
  };

  run_ok("SET vector_join_clusters = 4096;");
  run_ok("SET vector_join_probes = 8;");
  require_through(probe_lines([&] { (void)sorted_rows(*con, sql); }), "8 of the 4096");
  auto centroids = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM sirius_kmeans_centroids('__sirius_auto_sr_c4k_vec');");
  REQUIRE_FALSE(centroids->HasError());
  REQUIRE(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 4096);
  run_ok("RESET vector_join_clusters;");

  run_ok("SET vector_join_probes = 0.5;");
  std::vector<std::vector<std::string>> half;
  require_through(probe_lines([&] { half = sorted_rows(*con, sql); }), "32 of the 4096");
  auto const tvf32 = sorted_rows(
    *con,
    "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_c4k_probe','vec','sr_c4k','vec', "
    "probe_source => 'scan', search_mode => 'approx', clustering => '__sirius_auto_sr_c4k_vec', "
    "n_probes => 32, k => 5);");
  CHECK(rows_match(half, tvf32));

  run_ok("SET vector_join_probes = 64;");
  require_through(probe_lines([&] { require_gpu_matches_duckdb(*con, sql); }), "4096 of the 4096");

  for (auto const* bad : {"-1", "-0.5", "'nan'"}) {
    auto const r = con->Query(std::string("SET vector_join_probes = ") + bad + ";");
    REQUIRE(r->HasError());
    UNSCOPED_INFO("SET vector_join_probes = " << bad << ": " << r->GetError());
    CHECK(r->GetError().find("vector_join_probes must be a number >= 0") != std::string::npos);
  }
  auto const kept = con->Query("SELECT current_setting('vector_join_probes');");
  REQUIRE_FALSE(kept->HasError());
  CHECK(kept->GetValue(0, 0).GetValue<double>() == 64.0);

  run_ok("SET vector_join_probes = 0;");
  CHECK(probe_lines([&] { require_gpu_matches_duckdb(*con, sql); }).empty());
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a large batch of pairs reads a host-pinned corpus's columns chunk by chunk",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // 30,000 probe rows x top-10 is 300,000 values per corpus column, past what a task gathers
  // straight from host memory, so each task copies in the corpus chunks its neighbours fall in
  // (three row groups here) on the reservation it was admitted with. Both columns of a pair
  // must come from the same corpus row, and a few probe rows' pairs must be DuckDB's.
  SqlRewriteTables tables(*this);
  run_ok(
    "CREATE TABLE sr_wide AS SELECT i::INTEGER AS id, (i * 7 + 3)::BIGINT AS tag, "
    "[sin(i)::FLOAT, cos(i * 1.3)::FLOAT, sin(i * 0.7)::FLOAT, cos(i * 0.31)::FLOAT]::FLOAT[4] "
    "AS vec FROM range(300000) t(i);");
  run_ok(
    "CREATE TABLE sr_wide_probe AS SELECT i::INTEGER AS id, "
    "[sin(i * 2.1)::FLOAT, cos(i * 0.9)::FLOAT, sin(i * 1.7)::FLOAT, cos(i * 0.53)::FLOAT]"
    "::FLOAT[4] AS vec FROM range(30000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_wide', tier => 'host', format => 'duckdb');");
  std::string const pairs =
    " FROM sr_wide_probe p, LATERAL (SELECT c.id, c.tag, array_distance(p.vec, c.vec) AS d "
    "FROM sr_wide c ORDER BY d LIMIT 10) n";
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto const got    = sorted_rows(
    *con,
    "SELECT count(*), sum((n.tag - n.id * 7 - 3) * (n.tag - n.id * 7 - 3)), min(n.id) >= 0, "
       "max(n.id) < 300000" +
      pairs + ";");
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
  REQUIRE(got.size() == 1);
  CHECK(got[0][0] == "300000");
  CHECK(got[0][1] == "0");
  CHECK(got[0][2] == "true");
  CHECK(got[0][3] == "true");
  require_gpu_matches_duckdb(*con, "SELECT p.id, n.id, n.tag" + pairs + " WHERE p.id < 20;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a join over a pinned corpus with no lists can build them inside the query",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // With no lists, SIRIUS_VSS_ACCESS_PATH=lists has the rewrite write the exact join's compact
  // layout first -- one list of codes, no clustering to fit -- (as SIRIUS_VSS_BUILD_IN_QUERY=1 does
  // when the cost model says it pays), then search it whole: DuckDB's answer, and the list stays
  // for later queries under its own name, apart from the approximate join's partitions.
  // Byte values and a width of 16, so the codes are UINT8 and searched exactly.
  SqlRewriteTables tables(*this);
  auto const bytes = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
           ") % 256)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_bytes AS SELECT i::INTEGER AS id, " + bytes("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_bytes_probe AS SELECT i::INTEGER AS id, " + bytes("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_bytes', tier => 'gpu', format => 'duckdb');");
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_bytes_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_bytes c ORDER BY d LIMIT 5) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive > before.pairs_exhaustive);
  auto centroids = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_codes_sr_bytes_vec');");
  REQUIRE_FALSE(centroids->HasError());
  CHECK(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 1);

  // An approximate join does not take the single list for its partitions: it builds its own.
  run_ok("SET vector_join_probes = 2;");
  struct reset_on_exit {
    duckdb::Connection& c;
    ~reset_on_exit() { c.Query("RESET vector_join_probes;"); }
  } reset_probes{*con};
  auto const approx = con->Query(
    "SELECT count(*) FROM (SELECT p.id, n.id FROM sr_bytes_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_bytes c ORDER BY d LIMIT 5) n);");
  REQUIRE_FALSE(approx->HasError());
  CHECK(approx->GetValue(0, 0).GetValue<std::int64_t>() == 30 * 5);
  auto partitions = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_sr_bytes_vec');");
  REQUIRE_FALSE(partitions->HasError());
  CHECK(partitions->GetValue(0, 0).GetValue<std::int64_t>() > 1);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "lists built inside the query over float vectors answer exactly",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Values that are not bytes: the lists the query builds are FLOAT16 with their FP32 rows, and
  // every candidate is re-scored in FP32, so the answer is DuckDB's.
  SqlRewriteTables tables(*this);
  auto const floats = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: ((hash(i * 100 + d + " + seed +
           ") % 100000) / 977.0)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_floats AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_floats_probe AS SELECT i::INTEGER AS id, " + floats("9999") +
         " AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_floats', tier => 'gpu', format => 'duckdb');");
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct unset_on_exit {
    ~unset_on_exit() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  } unset_access_path;
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_floats_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_floats c ORDER BY d LIMIT 5) n;");
  CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive > before.pairs_exhaustive);
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a probe row outside float16's range is searched in FP32 over float16 lists",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Probe row 7 has a component of 70,000, which float16 rounds to infinity: the bounded FP16
  // search gave its pairs NaN filter values, which no limit admits, and dropped every one of them
  // without an error. Its batch is searched in FP32 instead, from the kept FP32 rows, so each
  // answer is DuckDB's: through the exact join's single FLOAT16 list, through partitions all
  // probed, and for the inner product (which refused such a probe). The corpus's first component
  // is 15 * id, under 60,000 (inside float16's range, so its lists build), which keeps row 7's
  // neighbours far apart; the rest are not bytes, so the lists are FLOAT16.
  SqlRewriteTables tables(*this);
  run_ok(
    "CREATE TABLE sr_wide16 AS SELECT i::INTEGER AS id, list_transform(range(16), lambda d: "
    "(CASE WHEN d = 0 THEN i * 15.0 ELSE (hash(i * 100 + d) % 100000) / 977.0 END)::FLOAT)"
    "::FLOAT[16] AS vec FROM range(4000) t(i);");
  run_ok(
    "CREATE TABLE sr_wide16_probe AS SELECT i::INTEGER AS id, list_transform(range(16), lambda "
    "d: (CASE WHEN i = 7 AND d = 0 THEN 70000.0 ELSE (hash(i * 100 + d + 9999) % 100000) / 977.0 "
    "END)::FLOAT)::FLOAT[16] AS vec FROM range(30) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_wide16', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      ::unsetenv("SIRIUS_VSS_ACCESS_PATH");
      con.Query("RESET vector_join_probes;");
    }
  } reset{*con};
  auto const lateral =
    "SELECT p.id, n.id, n.d FROM sr_wide16_probe p, LATERAL (SELECT c.id, "
    "array_distance(p.vec, c.vec) AS d FROM sr_wide16 c ORDER BY d LIMIT 5) n;";
  // Row 7 is 10,015 or more from every row, the others mostly under 10,600 from a few hundred.
  auto const threshold =
    "SELECT p.id, c.id FROM sr_wide16_probe p, sr_wide16 c WHERE "
    "array_distance(p.vec, c.vec) <= 10600.0;";
  auto const lateral_ip =
    "SELECT p.id, n.id, n.s FROM sr_wide16_probe p, LATERAL (SELECT c.id, "
    "array_inner_product(p.vec, c.vec) AS s FROM sr_wide16 c ORDER BY s DESC LIMIT 5) n;";
  auto const fell_back = [](auto&& run) {
    sirius::test::scoped_recording_log_sink log{"info"};
    run();
    for (auto const& r : log.records()) {
      if (r.message.find("outside float16's range; searching this batch") != std::string::npos) {
        return true;
      }
    }
    return false;
  };
  auto const has_rows_of_7 = [&](std::string const& sql) {
    con->Query("SET gpu_execution = false;");
    auto const rows = sorted_rows(*con, sql);
    con->Query("SET gpu_execution = true;");
    return std::any_of(rows.begin(), rows.end(), [](auto const& r) { return r[0] == "7"; });
  };
  REQUIRE(has_rows_of_7(lateral));
  REQUIRE(has_rows_of_7(threshold));

  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  CHECK(fell_back([&] { require_gpu_matches_duckdb(*con, lateral); }));
  auto codes = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_codes_sr_wide16_vec');");
  REQUIRE_FALSE(codes->HasError());
  CHECK(codes->GetValue(0, 0).GetValue<std::int64_t>() == 1);
  CHECK(fell_back([&] { require_gpu_matches_duckdb(*con, threshold); }));
  CHECK(fell_back([&] { require_gpu_matches_duckdb(*con, lateral_ip); }));
  ::unsetenv("SIRIUS_VSS_ACCESS_PATH");

  // Partitions of the corpus, every one probed: exact too.
  run_ok("SET vector_join_probes = 1000;");
  CHECK(fell_back([&] { require_gpu_matches_duckdb(*con, lateral); }));
  auto partitions = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM "
    "sirius_kmeans_centroids('__sirius_auto_sr_wide16_vec');");
  REQUIRE_FALSE(partitions->HasError());
  CHECK(partitions->GetValue(0, 0).GetValue<std::int64_t>() > 1);
  CHECK(fell_back([&] { require_gpu_matches_duckdb(*con, threshold); }));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a probe that is a pinned table is read from its pin",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // A self-join over a pinned table reads both sides from the pin; a probe with a filter is still
  // scanned. DuckDB's answers either way.
  SqlRewriteTables tables(*this);
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, n.id, n.d FROM sr_probe a, LATERAL (SELECT b.id, array_distance(a.vec, b.vec) "
    "AS d FROM sr_probe b ORDER BY d LIMIT 3) n;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT a.id, n.id FROM sr_corpus a, LATERAL (SELECT b.id, array_cosine_similarity(a.vec, "
    "b.vec) AS s FROM sr_corpus b ORDER BY s DESC LIMIT 2) n WHERE a.id < 50;");
  // The 30 self-pairs at distance 0, then the closest pair in both orders.
  require_gpu_matches_duckdb(*con,
                             "SELECT a.id, b.id, array_distance(a.vec, b.vec) AS d FROM sr_probe "
                             "a, sr_probe b ORDER BY d LIMIT 32;");
  require_gpu_matches_duckdb(
    *con,
    "SELECT p.id, n.id, n.d FROM sr_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM sr_corpus c WHERE c.cat = 3 ORDER BY d LIMIT 4) n;");
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a plain-SQL LATERAL over rows far from the origin is DuckDB's",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // Rows [i, i+1, i+2] of width 3 (no codes), so the rewrite takes the exhaustive search. Ranked
  // by |q|^2 + |x|^2 - 2 q.x (about 7.4e9 a term in FP32, against shells 3 apart) the probe's own
  // row 49,800 was missed and rows from 49,794 to 49,811 came back instead of 49,796 to 49,804.
  run_ok(
    "CREATE TABLE sr_far AS SELECT i::INTEGER AS id, [i::FLOAT, (i + 1)::FLOAT, (i + 2)::FLOAT]"
    "::FLOAT[3] AS vec FROM range(50000) t(i);");
  run_ok(
    "CREATE TABLE sr_far_probe AS SELECT * FROM (VALUES (0, [49800.0, 49801.0, 49802.0]), "
    "(1, [49890.0, 49891.0, 49892.0]), (2, [49810.0, 49811.0, 49812.0]), (3, [49780.0, "
    "49781.0, 49782.0]), (4, [49820.0, 49821.0, 49822.0])) t(id, vec);");
  run_ok(
    "CREATE TABLE sr_far_p AS SELECT id::INTEGER AS id, vec::FLOAT[3] AS vec FROM "
    "sr_far_probe;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_far', tier => 'gpu', format => 'duckdb');");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, n.id, n.d FROM sr_far_p p, LATERAL (SELECT c.id, "
                             "array_distance(p.vec, c.vec) AS d FROM sr_far c ORDER BY d LIMIT "
                             "9) n;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id FROM sr_far_p p, sr_far c WHERE "
                             "array_distance(p.vec, c.vec) <= 6.0;");
  run_ok("SELECT * FROM unpin_table('sr_far');");
}

namespace {

/// The "[vector_join_rewrite] ..." lines logged while @p run runs.
template <typename F>
std::vector<std::string> rewrite_log(F&& run)
{
  sirius::test::scoped_recording_log_sink log{"info"};
  run();
  std::vector<std::string> lines;
  for (auto const& r : log.records()) {
    if (r.message.find("[vector_join_rewrite]") != std::string::npos) {
      lines.push_back(r.message);
    }
  }
  return lines;
}

bool any_line_has(std::vector<std::string> const& lines, std::string const& needle)
{
  return std::any_of(lines.begin(), lines.end(), [&](std::string const& l) {
    return l.find(needle) != std::string::npos;
  });
}

}  // namespace

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a DELETE or INSERT on a pinned corpus leaves the vector join to DuckDB",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // The pin holds the rows as they were pinned, a deleted row included and an inserted one
  // missing; ordinary scans reconcile that with DuckDB's visibility, but the vector join reads the
  // pinned rows as they are. It returned the deleted row 5 and missed the inserted 9999. Now the
  // rewrite leaves such a statement to DuckDB, the table function refuses to read the pin, and
  // the table function over a scan (which applies the visibility) answers.
  auto const floats = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: ((hash(i * 100 + d + " + seed +
           ") % 100000) / 977.0)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_dml AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_dml_probe AS SELECT id, vec FROM sr_dml WHERE id IN (5, 6, 7);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_dml', tier => 'gpu', format => 'duckdb');");
  struct unpin_on_exit {
    duckdb::Connection& con;
    ~unpin_on_exit() { con.Query("SELECT * FROM unpin_table('sr_dml');"); }
  } unpin{*con};
  auto const lateral =
    "SELECT p.id, n.id, n.d FROM sr_dml_probe p, LATERAL (SELECT c.id, array_distance(p.vec, "
    "c.vec) AS d FROM sr_dml c ORDER BY d LIMIT 2) n;";
  auto const threshold =
    "SELECT p.id, c.id FROM sr_dml_probe p, sr_dml c WHERE array_distance(p.vec, c.vec) <= 40.0;";
  auto const on_duckdb = [&](std::string const& sql) {
    con->Query("SET gpu_execution = false;");
    auto rows = sorted_rows(*con, sql);
    con->Query("SET gpu_execution = true;");
    return rows;
  };
  CHECK(any_line_has(rewrite_log([&] { require_gpu_matches_duckdb(*con, lateral); }),
                     "rewritten to sirius_knn_join_rel"));

  run_ok("DELETE FROM sr_dml WHERE id = 5;");
  run_ok("INSERT INTO sr_dml SELECT 9999, vec FROM sr_dml WHERE id = 7;");
  for (auto const* sql : {lateral, threshold}) {
    UNSCOPED_INFO("query: " << sql);
    auto const expected = on_duckdb(sql);
    std::vector<std::vector<std::string>> got;
    auto const lines = rewrite_log([&] { got = sorted_rows(*con, sql); });
    CHECK(rows_match(got, expected));
    CHECK(any_line_has(lines, "left to DuckDB: the pin of 'sr_dml' no longer holds its rows"));
  }
  auto const after = on_duckdb(lateral);
  REQUIRE(std::any_of(after.begin(), after.end(), [](auto const& r) { return r[1] == "9999"; }));

  auto const pinned_tvf = con->Query(
    "SELECT * FROM sirius_knn_join('sr_dml_probe','vec','sr_dml','vec', probe_source => 'scan', "
    "k => 2);");
  REQUIRE(pinned_tvf->HasError());
  UNSCOPED_INFO("error: " << pinned_tvf->GetError());
  CHECK(pinned_tvf->GetError().find("the pin of right table 'sr_dml' no longer holds the "
                                    "table's rows") != std::string::npos);
  CHECK(pinned_tvf->GetError().find("build_source => 'scan'") != std::string::npos);
  auto const scanned_tvf =
    sorted_rows(*con,
                "SELECT left_id, right_id, distance FROM sirius_knn_join('sr_dml_probe','vec',"
                "'sr_dml','vec', probe_source => 'scan', build_source => 'scan', k => 2);");
  CHECK(rows_match(scanned_tvf, after));

  // A pin taken after a CHECKPOINT answers as DuckDB does too: from its rows, or, while the
  // deleted rows are still in storage (and so in the pin), by leaving the statement to DuckDB.
  run_ok("SELECT * FROM unpin_table('sr_dml');");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_dml', tier => 'gpu', format => 'duckdb');");
  CHECK(rows_match(sorted_rows(*con, lateral), after));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "lists cached from an earlier pin are not used for a re-pinned table",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite]")
{
  // The rewrite keeps the lists it builds for later statements, and found them by table, column
  // and row count, so a table changed while unpinned and pinned again with as many rows was
  // searched through the lists of its old rows. Lists now serve only the pin (generation) they
  // were built from.
  auto const floats = [](const std::string& seed) {
    return "list_transform(range(16), lambda d: ((hash(i * 100 + d + " + seed +
           ") % 100000) / 977.0)::FLOAT)::FLOAT[16]";
  };
  run_ok("CREATE TABLE sr_rp AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE sr_rp_probe AS SELECT id, vec FROM sr_rp WHERE id IN (5, 6, 7);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_rp', tier => 'gpu', format => 'duckdb');");
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      ::unsetenv("SIRIUS_VSS_ACCESS_PATH");
      con.Query("SELECT * FROM unpin_table('sr_rp');");
    }
  } reset{*con};
  auto const lateral =
    "SELECT p.id, n.id, n.d FROM sr_rp_probe p, LATERAL (SELECT c.id, array_distance(p.vec, "
    "c.vec) AS d FROM sr_rp c ORDER BY d LIMIT 2) n;";
  auto const built = [](std::vector<std::string> const& lines) {
    return any_line_has(lines, "built lists '__sirius_auto_codes_sr_rp_vec'");
  };
  CHECK(built(rewrite_log([&] { require_gpu_matches_duckdb(*con, lateral); })));
  CHECK_FALSE(built(rewrite_log([&] { require_gpu_matches_duckdb(*con, lateral); })));

  // The same table, as many rows, other vectors: a fresh table, so no deleted rows in its pin.
  run_ok("SELECT * FROM unpin_table('sr_rp');");
  run_ok("DROP TABLE sr_rp;");
  run_ok("CREATE TABLE sr_rp AS SELECT i::INTEGER AS id, " + floats("777") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_rp', tier => 'gpu', format => 'duckdb');");
  CHECK(built(rewrite_log([&] { require_gpu_matches_duckdb(*con, lateral); })));

  // An UPDATE while unpinned, then a re-pin: DuckDB updates a vector as a delete and an insert,
  // so the pin holds the old row too, and the statement is DuckDB's.
  run_ok("SELECT * FROM unpin_table('sr_rp');");
  run_ok(
    "UPDATE sr_rp SET vec = list_transform(range(16), lambda d: 1000.0::FLOAT)::FLOAT[16] "
    "WHERE id = 6;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_rp', tier => 'gpu', format => 'duckdb');");
  con->Query("SET gpu_execution = false;");
  auto const expected = sorted_rows(*con, lateral);
  con->Query("SET gpu_execution = true;");
  std::vector<std::vector<std::string>> got;
  auto const lines = rewrite_log([&] { got = sorted_rows(*con, lateral); });
  CHECK(rows_match(got, expected));
  CHECK(
    std::none_of(got.begin(), got.end(), [](auto const& r) { return r[0] == "6" && r[1] == "6"; }));
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "cosine ranks near-duplicates on rows normalized in FP32",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite][cosine]")
{
  // Row i is the probe with its first component moved by (2000 - i) * 1e-4, so the nearest six
  // are rows 1999 down to 1994, at cosine distances of about 2.5e-10 to 9e-9. The GEMM's
  // 1 - q.x / (|q| |x|) rounds those to 0 and returned rows 1985..1999 in no order (DuckDB's own
  // array_cosine_distance, the same formula, returned 1980..1989). An exact cosine join ranks the
  // rows normalized in FP32 by L2 between them: here every exhaustive and every-cluster search.
  run_ok(
    "CREATE TABLE sr_cos AS SELECT i::INTEGER AS id, list_transform(range(16), lambda d: (CASE "
    "WHEN d = 0 THEN 1.0 + (2000 - i) * 0.0001 ELSE 0.5 + d * 0.07 END)::FLOAT)::FLOAT[16] AS "
    "vec FROM range(2000) t(i);");
  run_ok(
    "CREATE TABLE sr_cos_probe AS SELECT 0::INTEGER AS id, list_transform(range(16), lambda d: "
    "(CASE WHEN d = 0 THEN 1.0 ELSE 0.5 + d * 0.07 END)::FLOAT)::FLOAT[16] AS vec;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'sr_cos', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM pin_table(name => 'sr_cos_probe', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      ::unsetenv("SIRIUS_VSS_ACCESS_PATH");
      ::unsetenv("SIRIUS_VSS_SQL_REWRITE");
      con.Query("SELECT * FROM unpin_table('sr_cos_probe');");
      con.Query("SELECT * FROM unpin_table('sr_cos');");
    }
  } reset{*con};

  // The cosine distance of row i, in double from the rows as stored.
  auto const oracle = [](int i) {
    std::array<double, 16> q{}, x{};
    for (int d = 0; d < 16; ++d) {
      q[d] = static_cast<float>(d == 0 ? 1.0 : 0.5 + d * 0.07);
      x[d] = d == 0 ? static_cast<float>(1.0 + (2000 - i) * 0.0001) : q[d];
    }
    double nq = 0, nx = 0, s = 0;
    for (int d = 0; d < 16; ++d) {
      nq += q[d] * q[d];
      nx += x[d] * x[d];
    }
    for (int d = 0; d < 16; ++d) {
      auto const diff = q[d] / std::sqrt(nq) - x[d] / std::sqrt(nx);
      s += diff * diff;
    }
    return s / 2;
  };
  std::vector<std::string> const nearest{"1994", "1995", "1996", "1997", "1998", "1999"};
  // The ids of a result's rows (column @p id_col), sorted, and each row's distance near the
  // oracle's (column @p d_col, when given): within 10%, the FP32 normalization's rounding.
  auto const ids_of = [&](std::vector<std::vector<std::string>> const& rows,
                          std::size_t id_col,
                          std::optional<std::size_t> d_col) {
    std::vector<std::string> ids;
    for (auto const& r : rows) {
      ids.push_back(r[id_col]);
      if (d_col) {
        auto const want = oracle(std::stoi(r[id_col]));
        auto const got  = std::stod(r[*d_col]);
        UNSCOPED_INFO("row " << r[id_col] << ": " << got << " vs " << want);
        CHECK(std::fabs(got - want) <= 0.1 * want);
      }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  };
  auto const tvf = [&](std::string const& extra) {
    return sorted_rows(*con,
                       "SELECT right_id, distance FROM sirius_knn_join('sr_cos_probe','vec',"
                       "'sr_cos','vec', metric => 'cosine', output_type => 'distance'" +
                         extra + ");");
  };
  auto const lateral =
    "SELECT n.id, n.d FROM sr_cos_probe p, LATERAL (SELECT c.id, array_cosine_distance(p.vec, "
    "c.vec) AS d FROM sr_cos c ORDER BY d LIMIT 6) n;";

  // The exhaustive search: top-k, and a threshold between the 6th and 7th distances.
  CHECK(ids_of(tvf(", k => 6"), 0, 1) == nearest);
  char eps_text[32];
  std::snprintf(eps_text, sizeof eps_text, "%.17g", (oracle(1994) + oracle(1993)) / 2);
  std::string const eps{eps_text};
  CHECK(ids_of(tvf(", join_mode => 'threshold', eps => " + eps), 0, 1) == nearest);
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "brute", 1);
  CHECK(ids_of(sorted_rows(*con, lateral), 0, 1) == nearest);
  // A FLOAT bound: against a DOUBLE one DuckDB compares the distance cast to DOUBLE, which the
  // rewrite does not match, and the statement stays DuckDB's.
  auto const within =
    "SELECT p.id, c.id, array_cosine_distance(p.vec, c.vec) FROM sr_cos_probe p, sr_cos c WHERE "
    "array_cosine_distance(p.vec, c.vec) <= " +
    eps + "::FLOAT;";
  CHECK(ids_of(sorted_rows(*con, within), 1, 2) == nearest);
  // The coded path (FLOAT16 unit rows, every candidate re-scored in FP32), unchanged.
  ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1);
  CHECK(ids_of(sorted_rows(*con, lateral), 0, 1) == nearest);
  ::unsetenv("SIRIUS_VSS_ACCESS_PATH");

  // Every cluster probed: FP32 lists of the rows as they are, and of unit rows.
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_cos','vec', name => 'sr_cos_c', n_clusters => 4);");
  run_ok(
    "SELECT * FROM sirius_kmeans_build_lists('sr_cos','vec','sr_cos_c', storage => "
    "'float32');");
  CHECK(ids_of(tvf(", k => 6, search_mode => 'approx', clustering => 'sr_cos_c', n_probes => 4"),
               0,
               1) == nearest);
  CHECK(ids_of(tvf(", join_mode => 'threshold', eps => " + eps +
                   ", search_mode => 'approx', clustering => 'sr_cos_c', n_probes => 4"),
               0,
               1) == nearest);
  run_ok("SELECT * FROM sirius_kmeans_fit('sr_cos','vec', name => 'sr_cos_u', n_clusters => 4);");
  run_ok(
    "SELECT * FROM sirius_kmeans_build_lists('sr_cos','vec','sr_cos_u', storage => "
    "'float32', metric => 'cosine');");
  CHECK(ids_of(tvf(", k => 6, search_mode => 'approx', clustering => 'sr_cos_u', n_probes => 4"),
               0,
               1) == nearest);

  // With the rewrite off, the physical planner's own vector top-k join answers.
  ::setenv("SIRIUS_VSS_SQL_REWRITE", "0", 1);
  CHECK(ids_of(sorted_rows(*con, lateral), 0, 1) == nearest);
}

namespace {

/// A pinned table of near-duplicates: 200 groups of four rows at offsets 0, 1, 3 and 7 along the
/// group's own direction (so every row's nearest member is unique, with a clear gap), then 100
/// rows on their own. Groups and singletons are random 16-d directions, far apart in cosine.
void create_dedup_table(SqlRewriteFixture& fx, std::string const& name, std::string const& key_type)
{
  auto const comp = [](std::string const& seed) {
    return "((hash(g * 100 + d + " + seed + ") % 2000)::FLOAT / 1000.0 - 1.0)";
  };
  fx.run_ok("CREATE TABLE " + name + " AS SELECT (g * 4 + j)::" + key_type +
            " AS id, list_transform(range(16), lambda d: (" + comp("0") +
            " + 0.05 * ([0, 1, 3, 7])[j + 1] * " + comp("777") +
            ")::FLOAT)::FLOAT[16] AS vec FROM range(200) a(g), range(4) b(j) UNION ALL "
            "SELECT (800 + g)::" +
            key_type + " AS id, list_transform(range(16), lambda d: " + comp("4242") +
            "::FLOAT)::FLOAT[16] AS vec FROM range(100) a(g);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a LATERAL best match within a cap runs as the best-match threshold join",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite][best_match]")
{
  // The deduplication query: per row the nearest earlier row (s.id < r.id), kept when its
  // similarity reaches 0.97. DuckDB plans the predicate as a join with the correlated columns and
  // pushes the cap into the subquery; the rewrite reads it as the operator's best-match threshold
  // join, keep => 'best' with earlier_by => 'id' (other_by for s.id <> r.id).
  create_dedup_table(*this, "sr_dd", "INTEGER");
  run_ok("SELECT * FROM pin_table(name => 'sr_dd', tier => 'gpu', format => 'duckdb');");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clusters;");
      con.Query("SELECT * FROM unpin_table('sr_dd');");
    }
  } reset{*con};
  auto const dedup = [](std::string const& cmp,
                        std::string const& limit = "1",
                        std::string const& cap   = " WHERE m.sim >= 0.97") {
    return "SELECT r.id, m.id, m.sim FROM sr_dd r, LATERAL (SELECT s.id, "
           "array_cosine_similarity(r.vec, s.vec) AS sim FROM sr_dd s WHERE s.id " +
           cmp + " r.id ORDER BY array_cosine_distance(r.vec, s.vec) LIMIT " + limit + ") m" + cap +
           ";";
  };
  auto const tvf = [&](std::string const& key, std::string const& extra = "") {
    return sorted_rows(*con,
                       "SELECT left_id, right_id, similarity FROM sirius_knn_join('sr_dd','vec',"
                       "'sr_dd','vec', metric => 'cosine', join_mode => 'threshold', eps => 0.97, "
                       "keep => 'best', " +
                         key + " => 'id'" + extra + ");");
  };

  for (auto const& [cmp, key] : {std::pair{std::string{"<"}, std::string{"earlier_by"}},
                                 std::pair{std::string{"<>"}, std::string{"other_by"}}}) {
    UNSCOPED_INFO("s.id " << cmp << " r.id");
    auto const sql = dedup(cmp);
    std::vector<std::vector<std::string>> got;
    auto const lines = rewrite_log([&] { require_gpu_matches_duckdb(*con, sql); });
    CHECK(any_line_has(lines, "LATERAL best match"));
    got = sorted_rows(*con, sql);
    // Most group rows have a member within the cap (588 earlier ones, 788 with <>).
    CHECK(got.size() > (cmp == "<" ? 500u : 700u));
    CHECK(rows_match(got, tvf(key)));
  }

  // Approximate: the same lists under both, so the same answer.
  run_ok("SET vector_join_clusters = 16;");
  run_ok("SET vector_join_probes = 4;");
  auto const approx = sorted_rows(*con, dedup("<"));
  CHECK(rows_match(approx,
                   tvf("earlier_by",
                       ", search_mode => 'approx', clustering => '__sirius_auto_sr_dd_vec_cosine', "
                       "n_probes => 4")));
  run_ok("RESET vector_join_probes;");
  run_ok("RESET vector_join_clusters;");

  // What the operator does not answer stays DuckDB's: k > 1, no cap, a cap that is strict.
  for (auto const& sql : {dedup("<", "3"),
                          dedup("<", "1", ""),
                          dedup("<>", "3", ""),
                          dedup("<", "1", " WHERE m.sim > 0.97")}) {
    UNSCOPED_INFO("query: " << sql);
    con->Query("SET gpu_execution = false;");
    auto const expected = sorted_rows(*con, sql);
    con->Query("SET gpu_execution = true;");
    std::vector<std::vector<std::string>> got;
    auto const lines = rewrite_log([&] { got = sorted_rows(*con, sql); });
    CHECK_FALSE(any_line_has(lines, "LATERAL best match"));
    CHECK(rows_match(got, expected));
  }
}

TEST_CASE_METHOD(SqlRewriteFixture,
                 "a LATERAL best match the operator cannot key stays DuckDB's",
                 "[integration][gpu_execution][array][vss][vector_join][sql_rewrite][best_match]")
{
  // A key that is not an integer, and a side that is not pinned.
  create_dedup_table(*this, "sr_ddf", "DOUBLE");
  create_dedup_table(*this, "sr_ddu", "INTEGER");
  run_ok("SELECT * FROM pin_table(name => 'sr_ddf', tier => 'gpu', format => 'duckdb');");
  struct unpin_on_exit {
    duckdb::Connection& con;
    ~unpin_on_exit() { con.Query("SELECT * FROM unpin_table('sr_ddf');"); }
  } unpin{*con};
  for (auto const* table : {"sr_ddf", "sr_ddu"}) {
    auto const sql =
      std::string("SELECT r.id, m.id FROM ") + table +
      " r, LATERAL (SELECT s.id, array_cosine_similarity(r.vec, s.vec) AS sim FROM " + table +
      " s WHERE s.id < r.id ORDER BY array_cosine_distance(r.vec, s.vec) LIMIT 1) m "
      "WHERE m.sim >= 0.97;";
    UNSCOPED_INFO("query: " << sql);
    con->Query("SET gpu_execution = false;");
    auto const expected = sorted_rows(*con, sql);
    con->Query("SET gpu_execution = true;");
    std::vector<std::vector<std::string>> got;
    auto const lines = rewrite_log([&] { got = sorted_rows(*con, sql); });
    CHECK_FALSE(any_line_has(lines, "LATERAL best match"));
    CHECK(rows_match(got, expected));
    CHECK(expected.size() > 500u);
  }
}
