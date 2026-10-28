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
 * @file test_gpu_execution_vector_join_inner_product.cpp
 * @brief Inner-product vector joins: sirius_knn_join(metric => 'ip') and plain SQL ordered or
 *        bounded on array_inner_product / array_negative_inner_product run on the GPU and return
 *        what DuckDB returns, through every search that accepts them (the GEMM over the pinned
 *        rows, FLOAT16 lists filtered and re-scored in FP32, FP32 lists), and the searches that
 *        do not accept them refuse.
 */

#include "duckdb/main/config.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using InnerProductFixture = sirius::test::GpuExecutionFixture;

namespace {

std::vector<std::vector<std::string>> sorted_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return sirius::test::GpuExecutionFixture::collect_rows(
    r->Cast<duckdb::MaterializedQueryResult>(), /*sort=*/true);
}

/// Equal rows, numbers within 1e-4 relative (of at least 1): the GPU and DuckDB sum the inner
/// product in different orders.
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
      if (*ea != '\0' || *eb != '\0') {
        UNSCOPED_INFO("row " << i << " col " << j << ": " << a[i][j] << " vs " << b[i][j]);
        return false;
      }
      if (std::fabs(x - y) > 1e-4 * std::max(1.0, std::fabs(y))) {
        UNSCOPED_INFO("row " << i << " col " << j << ": " << a[i][j] << " vs " << b[i][j]);
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<std::string>> on_duckdb(duckdb::Connection& con, const std::string& sql)
{
  con.Query("SET gpu_execution = false;");
  auto rows = sorted_rows(con, sql);
  con.Query("SET gpu_execution = true;");
  return rows;
}

/// @p sql on the GPU (asserting it ran there) and on DuckDB: the same rows.
void require_gpu_matches_duckdb(duckdb::Connection& con, const std::string& sql)
{
  auto const expected = on_duckdb(con, sql);
  auto const before   = sirius::test::get_transparent_execution_stats(con);
  auto const got      = sorted_rows(con, sql);
  auto const after    = sirius::test::get_transparent_execution_stats(con);
  UNSCOPED_INFO("query: " << sql);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  REQUIRE(got.size() == expected.size());
  REQUIRE(rows_match(got, expected));
}

void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

struct scoped_env {
  std::string name;
  scoped_env(std::string n, const char* value) : name(std::move(n))
  {
    ::setenv(name.c_str(), value, 1);
  }
  ~scoped_env() { ::unsetenv(name.c_str()); }
};

}  // namespace

// Rows of 16 components from a hash, scaled per row by 0.05 to 10: the inner product's order is
// then far from cosine's (norms vary 200x) and from l2's, and no two corpus rows tie on a probe.
class InnerProductTables {
 public:
  explicit InnerProductTables(InnerProductFixture& f) : _f(f)
  {
    auto& disabled  = duckdb::DBConfig::GetConfig(*f.con->context).options.disabled_optimizers;
    _saved_disabled = disabled;
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    auto const vec = [](const std::string& seed) {
      return "list_transform(range(16), lambda d: (((hash(i * 100 + d + " + seed +
             ") % 2000)::FLOAT / 1000.0 - 1.0) * (0.05 + (hash(i + " + seed +
             " + 424242) % 1000)::FLOAT / 100.0))::FLOAT)::FLOAT[16]";
    };
    f.run_ok("CREATE TABLE ip_corpus AS SELECT i::INTEGER AS id, " + vec("0") +
             " AS vec FROM range(20000) t(i);");
    f.run_ok("CREATE TABLE ip_probe AS SELECT i::INTEGER AS id, " + vec("77777777") +
             " AS vec FROM range(40) t(i);");
    f.run_ok("CHECKPOINT;");
    for (auto const* t : {"ip_corpus", "ip_probe"}) {
      f.run_ok(std::string("SELECT * FROM pin_table(name => '") + t +
               "', tier => 'gpu', format => 'duckdb');");
    }
  }
  ~InnerProductTables()
  {
    for (auto const* t : {"ip_corpus", "ip_probe"}) {
      _f.con->Query(std::string("SELECT * FROM unpin_table('") + t + "');");
    }
    duckdb::DBConfig::GetConfig(*_f.con->context).options.disabled_optimizers = _saved_disabled;
  }

 private:
  InnerProductFixture& _f;
  std::set<duckdb::OptimizerType> _saved_disabled;
};

namespace {

std::string const kLateralIp =
  "SELECT p.id, n.id, n.s FROM ip_probe p, LATERAL (SELECT c.id, array_inner_product(p.vec, "
  "c.vec) AS s FROM ip_corpus c ORDER BY s DESC LIMIT 7) n;";
std::string const kLateralNegIp =
  "SELECT p.id, n.id, n.d FROM ip_probe p, LATERAL (SELECT c.id, "
  "array_negative_inner_product(p.vec, c.vec) AS d FROM ip_corpus c ORDER BY d LIMIT 5) n;";

}  // namespace

TEST_CASE_METHOD(InnerProductFixture,
                 "inner-product data differs from cosine and l2",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  // Guards the tests below: on rows of equal norm an inner-product join is a cosine join.
  InnerProductTables tables(*this);
  auto const ids = [&](const std::string& fn, const std::string& order) {
    return on_duckdb(*con,
                     "SELECT p.id, n.id FROM ip_probe p, LATERAL (SELECT c.id FROM ip_corpus c "
                     "ORDER BY " +
                       fn + "(p.vec, c.vec) " + order + " LIMIT 7) n;");
  };
  auto const ip  = ids("array_inner_product", "DESC");
  auto const cos = ids("array_cosine_similarity", "DESC");
  auto const l2  = ids("array_distance", "ASC");
  REQUIRE(ip.size() == 40 * 7);
  CHECK(ip != cos);
  CHECK(ip != l2);
}

TEST_CASE_METHOD(InnerProductFixture,
                 "plain-SQL LATERAL top-k on the inner product runs as the vector join",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  InnerProductTables tables(*this);
  // Brute force over the pinned FP32 rows (the GEMM search).
  {
    scoped_env brute("SIRIUS_VSS_ACCESS_PATH", "brute");
    require_gpu_matches_duckdb(*con, kLateralIp);
    require_gpu_matches_duckdb(*con, kLateralNegIp);
    // LIMIT 1 decorrelates without the list, as max / arg_max of the score.
    require_gpu_matches_duckdb(*con,
                               "SELECT p.id, t.id, t.s FROM ip_probe p, LATERAL (SELECT c.id, "
                               "array_inner_product(p.vec, c.vec) AS s FROM ip_corpus c ORDER BY "
                               "s DESC LIMIT 1) t;");
  }
  // Through the exact join's single list, built inside the query: FLOAT16 codes filtered under a
  // Cauchy-Schwarz slack and re-scored from the FP32 rows.
  {
    scoped_env lists("SIRIUS_VSS_ACCESS_PATH", "lists");
    auto const before = sirius::test::get_vector_join_prune_stats(*con);
    require_gpu_matches_duckdb(*con, kLateralIp);
    CHECK(sirius::test::get_vector_join_prune_stats(*con).pairs_exhaustive >
          before.pairs_exhaustive);
    require_gpu_matches_duckdb(*con, kLateralNegIp);
    auto centroids = con->Query(
      "SELECT count(DISTINCT cluster_id) FROM "
      "sirius_kmeans_centroids('__sirius_auto_codes_ip_corpus_vec_ip');");
    REQUIRE_FALSE(centroids->HasError());
    CHECK(centroids->GetValue(0, 0).GetValue<std::int64_t>() == 1);
  }
  // Under vector_join_probes the rewrite partitions the corpus inside the query (FLOAT16 lists of
  // its own, apart from an l2 join's) and routes each probe row by the inner product; with every
  // cluster probed that is exact.
  run_ok("SET vector_join_probes = 8;");
  run_ok("SET vector_join_clusters = 8;");
  struct reset_on_exit {
    duckdb::Connection& c;
    ~reset_on_exit()
    {
      c.Query("RESET vector_join_probes;");
      c.Query("RESET vector_join_clusters;");
    }
  } reset{*con};
  require_gpu_matches_duckdb(*con, kLateralIp);
  auto partitions = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM sirius_kmeans_centroids('__sirius_auto_ip_corpus_vec_ip');");
  REQUIRE_FALSE(partitions->HasError());
  CHECK(partitions->GetValue(0, 0).GetValue<std::int64_t>() == 8);
}

TEST_CASE_METHOD(InnerProductFixture,
                 "plain-SQL global top-k and thresholds on the inner product",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  InnerProductTables tables(*this);
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id, array_inner_product(p.vec, c.vec) AS s FROM "
                             "ip_probe p, ip_corpus c ORDER BY s DESC LIMIT 20;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id, array_negative_inner_product(p.vec, c.vec) AS d "
                             "FROM ip_probe p, ip_corpus c ORDER BY d LIMIT 20;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id, array_inner_product(p.vec, c.vec) AS s FROM "
                             "ip_probe p, ip_corpus c WHERE array_inner_product(p.vec, c.vec) >= "
                             "150.0;");
  require_gpu_matches_duckdb(*con,
                             "SELECT p.id, c.id, array_negative_inner_product(p.vec, c.vec) AS d "
                             "FROM ip_probe p, ip_corpus c WHERE "
                             "array_negative_inner_product(p.vec, c.vec) <= -150.0;");
  // Strict, and a negative bound on the inner product (most pairs pass).
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM ip_probe p, ip_corpus c WHERE "
                             "array_negative_inner_product(p.vec, c.vec) < -100.0;");
  require_gpu_matches_duckdb(*con,
                             "SELECT count(*) FROM ip_probe p, ip_corpus c WHERE "
                             "array_inner_product(p.vec, c.vec) > -2.5;");
}

TEST_CASE_METHOD(InnerProductFixture,
                 "sirius_knn_join metric => 'ip' matches DuckDB through every search",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  InnerProductTables tables(*this);
  auto const expected     = on_duckdb(*con, kLateralIp);
  auto const expected_neg = on_duckdb(*con, kLateralNegIp);
  auto const join         = [&](const std::string& extra, std::int64_t k = 7) {
    return sorted_rows(*con,
                       "SELECT left_id, right_id, similarity FROM sirius_knn_join('ip_probe', "
                       "'vec', 'ip_corpus', 'vec', metric => 'ip', k => " +
                         std::to_string(k) + extra + ");");
  };
  auto const via = [&](const std::string& clustering) {
    return ", search_mode => 'approx', clustering => '" + clustering + "', n_probes => 8";
  };
  // The pinned FP32 rows.
  CHECK(rows_match(join(""), expected));
  // The negative inner product, as output_type => 'distance'.
  CHECK(rows_match(sorted_rows(*con,
                               "SELECT left_id, right_id, distance FROM sirius_knn_join("
                               "'ip_probe', 'vec', 'ip_corpus', 'vec', metric => 'ip', k => 5, "
                               "output_type => 'distance');"),
                   expected_neg));

  // One list (the exact join's layout, its bound lowered as it goes), FLOAT16 and FP32.
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip1', n_clusters => 1);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip1', storage => "
         "'float16');");
  CHECK(rows_match(join(", search_mode => 'approx', clustering => 'ip1', n_probes => 1"),
                   expected));
  // The single list in host memory: its first piece is unbounded and serves as the sample.
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip1h', n_clusters => 1);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip1h', storage => "
         "'float16', tier => 'host');");
  CHECK(rows_match(join(", search_mode => 'approx', clustering => 'ip1h', n_probes => 1"),
                   expected));
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip1f', n_clusters => 1);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip1f', storage => "
         "'float32');");
  CHECK(rows_match(join(", search_mode => 'approx', clustering => 'ip1f', n_probes => 1"),
                   expected));

  // Eight clusters, all probed: exact. FLOAT16 on the device (seeded), FLOAT16 unseeded (sweep 0
  // ranked by the FP32 GEMM), FLOAT16 in host memory, and FP32.
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip8', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip8', storage => "
         "'float16');");
  CHECK(rows_match(join(via("ip8")), expected));
  {
    scoped_env unseeded("SIRIUS_VSS_SEED", "0");
    CHECK(rows_match(join(via("ip8")), expected));
  }
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip8h', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip8h', storage => "
         "'float16', tier => 'host');");
  CHECK(rows_match(join(via("ip8h")), expected));
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ip8f', n_clusters => 8);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ip8f', storage => "
         "'float32');");
  CHECK(rows_match(join(via("ip8f")), expected));

  // Approximate: two of eight clusters, routed by the inner product to their centroids. Every
  // probe row gets k pairs, each scored exactly: its score is DuckDB's for that pair.
  auto const approx = sorted_rows(
    *con,
    "SELECT left_id, right_id, similarity FROM sirius_knn_join('ip_probe', 'vec', 'ip_corpus', "
    "'vec', metric => 'ip', k => 7, search_mode => 'approx', clustering => 'ip8', n_probes => 2);");
  REQUIRE(approx.size() == 40 * 7);
  std::string pairs;
  for (auto const& row : approx) {
    pairs += (pairs.empty() ? "(" : ", (") + row[0] + ", " + row[1] + ")";
  }
  auto const scored =
    on_duckdb(*con,
              "SELECT p.id, c.id, array_inner_product(p.vec, c.vec) FROM ip_probe p, ip_corpus c "
              "WHERE (p.id, c.id) IN (VALUES " +
                pairs + ");");
  CHECK(rows_match(approx, scored));

  // Global top-k and a threshold, over the pin and over the FLOAT16 list.
  for (std::string const extra :
       {std::string{}, std::string{", search_mode => 'approx', clustering => 'ip1', n_probes => 1"}}) {
    CHECK(rows_match(join(", join_mode => 'global'" + extra, 20),
                     on_duckdb(*con,
                               "SELECT p.id, c.id, array_inner_product(p.vec, c.vec) AS s FROM "
                               "ip_probe p, ip_corpus c ORDER BY s DESC LIMIT 20;")));
    CHECK(rows_match(join(", eps => 150.0" + extra),
                     on_duckdb(*con,
                               "SELECT p.id, c.id, array_inner_product(p.vec, c.vec) AS s FROM "
                               "ip_probe p, ip_corpus c WHERE array_inner_product(p.vec, c.vec) "
                               ">= 150.0;")));
  }
}

TEST_CASE_METHOD(InnerProductFixture,
                 "sirius_knn_join metric => 'ip' refuses what it cannot answer exactly",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  InnerProductTables tables(*this);
  run_ok("SELECT * FROM sirius_kmeans_fit('ip_corpus', 'vec', name => 'ipi8', n_clusters => 1);");
  run_ok("SELECT * FROM sirius_kmeans_build_lists('ip_corpus', 'vec', 'ipi8', storage => "
         "'int8');");
  expect_error(*con,
               "SELECT count(*) FROM sirius_knn_join('ip_probe', 'vec', 'ip_corpus', 'vec', "
               "metric => 'ip', k => 5, search_mode => 'approx', clustering => 'ipi8', n_probes "
               "=> 1);",
               "int8 cluster lists answer only l2 joins");
  expect_error(*con,
               "SELECT count(*) FROM sirius_knn_join('ip_probe', 'vec', 'ip_corpus', 'vec', "
               "metric => 'ip', eps => 1.0, keep => 'best');",
               "not supported for metric => 'ip'");
}
