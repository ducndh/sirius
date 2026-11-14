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
 * @file test_gpu_execution_vector_join_uncached.cpp
 * @brief A vector join over a table that is not pinned gets the codes and lists a pinned one
 *        gets, built by the join over the corpus it scans, and answers as the pinned path does,
 *        pair by pair; a pinned table changed since its pin is scanned the same way, its kept
 *        lists dropped.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using UncachedFixture = sirius::test::GpuExecutionFixture;

namespace {

std::vector<std::vector<std::string>> sorted_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return sirius::test::GpuExecutionFixture::collect_rows(*r, /*sort=*/true);
}

/// Equal rows, numbers within 1e-4 relative, for comparing against DuckDB's own answer.
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

struct gpu_run {
  std::vector<std::vector<std::string>> rows;
  std::vector<std::string> log;
};

/// @p sql on the GPU (one rebind, one execution), with the vector join's log lines.
gpu_run run_on_gpu(duckdb::Connection& con, const std::string& sql)
{
  UNSCOPED_INFO("query: " << sql);
  gpu_run out;
  sirius::test::scoped_recording_log_sink sink{"info"};
  auto const before = sirius::test::get_transparent_execution_stats(con);
  out.rows          = sorted_rows(con, sql);
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(con), 1, 0, 1);
  for (auto const& r : sink.records()) {
    if (r.message.find("[vector_join_rewrite]") != std::string::npos ||
        r.message.find("[sirius_physical_vector_join_stream]") != std::string::npos) {
      out.log.push_back(r.message);
    }
  }
  return out;
}

std::vector<std::vector<std::string>> on_duckdb(duckdb::Connection& con, const std::string& sql)
{
  con.Query("SET gpu_execution = false;");
  auto rows = sorted_rows(con, sql);
  con.Query("SET gpu_execution = true;");
  return rows;
}

bool any_line_has(std::vector<std::string> const& lines, std::string const& needle)
{
  return std::any_of(lines.begin(), lines.end(), [&](std::string const& l) {
    return l.find(needle) != std::string::npos;
  });
}

/// FLOAT[32] rows that are not bytes, so the codes are FLOAT16 with an FP32 copy.
std::string floats(const std::string& seed)
{
  return "list_transform(range(32), lambda d: ((hash(i * 100 + d + " + seed +
         ") % 100000) / 977.0 - 51.0)::FLOAT)::FLOAT[32]";
}

/// SIRIUS_VSS_ACCESS_PATH=lists for the scope: tables this small cost less by brute force, and
/// these tests are about the lists.
struct force_lists {
  force_lists() { ::setenv("SIRIUS_VSS_ACCESS_PATH", "lists", 1); }
  ~force_lists() { ::unsetenv("SIRIUS_VSS_ACCESS_PATH"); }
  force_lists(force_lists const&)            = delete;
  force_lists& operator=(force_lists const&) = delete;
};

}  // namespace

TEST_CASE_METHOD(UncachedFixture,
                 "an unpinned corpus is searched through codes the join builds, as a pinned one",
                 "[integration][gpu_execution][array][vss][vector_join][uncached]")
{
  run_ok("CREATE TABLE uc_corpus AS SELECT i::INTEGER AS id, " + floats("0") +
         " AS vec FROM range(6000) t(i);");
  run_ok("CREATE TABLE uc_probe AS SELECT i::INTEGER AS id, " + floats("4242") +
         " AS vec FROM range(120) t(i);");
  run_ok("CHECKPOINT;");
  force_lists forced;
  struct unpin_on_exit {
    duckdb::Connection& con;
    ~unpin_on_exit() { con.Query("SELECT * FROM unpin_table('uc_corpus');"); }
  } unpin{*con};

  std::vector<std::string> const queries = {
    "SELECT p.id, n.id, n.d FROM uc_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM uc_corpus c ORDER BY d LIMIT 5) n;",
    "SELECT p.id, n.id, n.d FROM uc_probe p, LATERAL (SELECT c.id, array_cosine_distance(p.vec, "
    "c.vec) AS d FROM uc_corpus c ORDER BY d LIMIT 5) n;",
    "SELECT p.id, c.id, array_distance(p.vec, c.vec) FROM uc_probe p, uc_corpus c WHERE "
    "array_distance(p.vec, c.vec) <= 150.0;",
    "SELECT p.id, c.id, array_cosine_similarity(p.vec, c.vec) FROM uc_probe p, uc_corpus c "
    "WHERE array_cosine_similarity(p.vec, c.vec) >= 0.4;",
    "SELECT left_id, right_id, distance FROM sirius_knn_join('uc_probe', 'vec', 'uc_corpus', "
    "'vec', probe_source => 'scan', build_source => 'scan', k => 5);",
    "SELECT left_id, right_id, similarity FROM sirius_knn_join('uc_probe', 'vec', 'uc_corpus', "
    "'vec', probe_source => 'scan', build_source => 'scan', k => 5, metric => 'cosine');",
  };
  // The pinned form of the table function reads the corpus from its pin.
  auto pinned_form = [](std::string sql) {
    auto const at = sql.find(", build_source => 'scan'");
    if (at != std::string::npos) { sql.erase(at, std::string(", build_source => 'scan'").size()); }
    return sql;
  };

  std::vector<gpu_run> uncached;
  for (auto const& sql : queries) {
    auto run = run_on_gpu(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    CHECK(any_line_has(run.log, "built 1 float16 lists over the 6000 scanned corpus rows"));
    REQUIRE_FALSE(run.rows.empty());
    // The table function runs on the GPU only; its plain-SQL forms above are checked against DuckDB.
    if (sql.find("sirius_knn_join") == std::string::npos) {
      CHECK(rows_match(run.rows, on_duckdb(*con, sql)));
    }
    uncached.push_back(std::move(run));
  }

  run_ok("SELECT * FROM pin_table(name => 'uc_corpus', tier => 'gpu', format => 'duckdb');");
  for (std::size_t i = 0; i < queries.size(); ++i) {
    auto const cached = run_on_gpu(*con, pinned_form(queries[i]));
    UNSCOPED_INFO("query: " << queries[i]);
    CHECK_FALSE(any_line_has(cached.log, "scanned corpus rows"));
    CHECK(any_line_has(cached.log, "through the lists of clustering '__sirius_auto_codes_uc_corpus"));
    // The same codes, the same bound and the same FP32 re-score: the same pairs, the same scores.
    CHECK(cached.rows == uncached[i].rows);
  }
}

TEST_CASE_METHOD(UncachedFixture,
                 "vector_join_probes takes effect on an unpinned corpus, as on a pinned one",
                 "[integration][gpu_execution][array][vss][vector_join][uncached]")
{
  run_ok("CREATE TABLE ua_corpus AS SELECT i::INTEGER AS id, " + floats("11") +
         " AS vec FROM range(2000) t(i);");
  run_ok("CREATE TABLE ua_probe AS SELECT i::INTEGER AS id, " + floats("977") +
         " AS vec FROM range(200) t(i);");
  run_ok("CHECKPOINT;");
  struct reset_on_exit {
    duckdb::Connection& con;
    ~reset_on_exit()
    {
      con.Query("RESET vector_join_probes;");
      con.Query("RESET vector_join_clusters;");
      con.Query("SELECT * FROM unpin_table('ua_corpus');");
    }
  } reset{*con};
  run_ok("SET vector_join_clusters = 16;");
  run_ok("SET vector_join_probes = 2;");

  std::vector<std::string> const queries = {
    "SELECT p.id, n.id, n.d FROM ua_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM ua_corpus c ORDER BY d LIMIT 5) n;",
    "SELECT p.id, n.id, n.d FROM ua_probe p, LATERAL (SELECT c.id, array_cosine_distance(p.vec, "
    "c.vec) AS d FROM ua_corpus c ORDER BY d LIMIT 5) n;",
    "SELECT left_id, right_id, distance FROM sirius_knn_join('ua_probe', 'vec', 'ua_corpus', "
    "'vec', probe_source => 'scan', build_source => 'scan', k => 5);",
  };
  std::vector<gpu_run> uncached;
  for (auto const& sql : queries) {
    auto const before = sirius::test::get_vector_join_prune_stats(*con);
    auto run          = run_on_gpu(*con, sql);
    auto const after  = sirius::test::get_vector_join_prune_stats(*con);
    UNSCOPED_INFO("query: " << sql);
    CHECK(any_line_has(run.log, "built 16 float16 lists over the 2000 scanned corpus rows"));
    CHECK(any_line_has(run.log, "searching 2 of them"));
    // Two clusters of sixteen: about an eighth of the pairs an exhaustive join scores.
    auto const scored     = after.pairs_scored - before.pairs_scored;
    auto const exhaustive = after.pairs_exhaustive - before.pairs_exhaustive;
    CHECK(exhaustive > 0);
    CHECK(scored * 2 < exhaustive);
    uncached.push_back(std::move(run));
  }

  run_ok("SELECT * FROM pin_table(name => 'ua_corpus', tier => 'gpu', format => 'duckdb');");
  for (std::size_t i = 0; i < queries.size(); ++i) {
    auto sql        = queries[i];
    auto const scan = sql.find(", build_source => 'scan'");
    if (scan != std::string::npos) {
      sql.erase(scan, std::string(", build_source => 'scan'").size());
    }
    auto const cached = run_on_gpu(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    CHECK(any_line_has(cached.log, "searching 'ua_corpus' through 2 of the 16 clusters"));
    // At 128 training rows a cluster every row trains, in the same order either way, so the fit,
    // the lists and the answers are the same.
    CHECK(cached.rows == uncached[i].rows);
  }
}

TEST_CASE_METHOD(UncachedFixture,
                 "a changed pinned corpus drops its lists and the join rebuilds them from a scan",
                 "[integration][gpu_execution][array][vss][vector_join][uncached]")
{
  run_ok("CREATE TABLE ud_corpus AS SELECT i::INTEGER AS id, " + floats("5") +
         " AS vec FROM range(4000) t(i);");
  run_ok("CREATE TABLE ud_probe AS SELECT id, vec FROM ud_corpus WHERE id IN (5, 6, 7, 8);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ud_corpus', tier => 'gpu', format => 'duckdb');");
  force_lists forced;
  struct unpin_on_exit {
    duckdb::Connection& con;
    ~unpin_on_exit() { con.Query("SELECT * FROM unpin_table('ud_corpus');"); }
  } unpin{*con};
  auto const lateral =
    "SELECT p.id, n.id, n.d FROM ud_probe p, LATERAL (SELECT c.id, array_distance(p.vec, c.vec) "
    "AS d FROM ud_corpus c ORDER BY d LIMIT 3) n;";
  auto const threshold =
    "SELECT p.id, c.id FROM ud_probe p, ud_corpus c WHERE array_distance(p.vec, c.vec) <= 120.0;";
  auto const kept_lists = [&] {
    auto r = con->Query(
      "SELECT count(*) FROM sirius_kmeans_centroids('__sirius_auto_codes_ud_corpus_vec');");
    return !r->HasError();
  };

  auto const cached = run_on_gpu(*con, lateral);
  CHECK(any_line_has(cached.log, "built lists '__sirius_auto_codes_ud_corpus_vec'"));
  CHECK(rows_match(cached.rows, on_duckdb(*con, lateral)));
  REQUIRE(kept_lists());

  // An INSERT: the pin misses the new row. The statement still runs on the GPU, over a scan.
  run_ok("INSERT INTO ud_corpus SELECT 9999, vec FROM ud_corpus WHERE id = 7;");
  for (auto const* sql : {lateral, threshold}) {
    auto const run = run_on_gpu(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    CHECK(rows_match(run.rows, on_duckdb(*con, sql)));
    CHECK(any_line_has(run.log, "lists over the 4001 scanned corpus rows"));
    CHECK(std::any_of(run.rows.begin(), run.rows.end(), [](auto const& r) {
      return r[0] == "7" && r[1] == "9999";
    }));
  }
  CHECK_FALSE(kept_lists());

  // An UPDATE and a DELETE as well. Sirius refuses an UPDATE on a pinned table, so it is made
  // unpinned and the table pinned again after a CHECKPOINT; DuckDB updates a vector as a delete
  // and an insert, so the new pin still holds the old row 6 and the deleted row 8.
  run_ok("SELECT * FROM unpin_table('ud_corpus');");
  run_ok(
    "UPDATE ud_corpus SET vec = list_transform(range(32), lambda d: 1000.0::FLOAT)::FLOAT[32] "
    "WHERE id = 6;");
  run_ok("DELETE FROM ud_corpus WHERE id = 8;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'ud_corpus', tier => 'gpu', format => 'duckdb');");
  for (auto const* sql : {lateral, threshold}) {
    auto const run = run_on_gpu(*con, sql);
    UNSCOPED_INFO("query: " << sql);
    CHECK(rows_match(run.rows, on_duckdb(*con, sql)));
    CHECK(any_line_has(run.log, "scanned corpus rows"));
    CHECK(std::none_of(run.rows.begin(), run.rows.end(), [](auto const& r) {
      return (r[0] == "6" && r[1] == "6") || r[1] == "8";
    }));
  }
}
