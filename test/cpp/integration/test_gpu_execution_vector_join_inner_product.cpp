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
// With `bytes`, the components are whole numbers in [0, m) for a per-row m in 1..256 instead
// (UINT8 lists hold them losslessly; norms vary ~250x, and inner products are whole numbers, so
// rows can tie). <prefix>_corpus has 20,000 rows, <prefix>_probe 40; both pinned unless `pin` is
// false.
class InnerProductTables {
 public:
  explicit InnerProductTables(InnerProductFixture& f,
                              std::string prefix = "ip",
                              bool bytes         = false,
                              bool pin           = true)
    : _f(f), _prefix(std::move(prefix)), _pin(pin)
  {
    auto& disabled  = duckdb::DBConfig::GetConfig(*f.con->context).options.disabled_optimizers;
    _saved_disabled = disabled;
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    auto const vec = [bytes](const std::string& seed) {
      if (bytes) {
        return "list_transform(range(16), lambda d: (hash(i * 100 + d + " + seed +
               ") % (1 + hash(i + " + seed + " + 424242) % 256))::FLOAT)::FLOAT[16]";
      }
      return "list_transform(range(16), lambda d: (((hash(i * 100 + d + " + seed +
             ") % 2000)::FLOAT / 1000.0 - 1.0) * (0.05 + (hash(i + " + seed +
             " + 424242) % 1000)::FLOAT / 100.0))::FLOAT)::FLOAT[16]";
    };
    f.run_ok("CREATE TABLE " + _prefix + "_corpus AS SELECT i::INTEGER AS id, " + vec("0") +
             " AS vec FROM range(20000) t(i);");
    f.run_ok("CREATE TABLE " + _prefix + "_probe AS SELECT i::INTEGER AS id, " + vec("77777777") +
             " AS vec FROM range(40) t(i);");
    f.run_ok("CHECKPOINT;");
    if (_pin) {
      for (auto const& t : {_prefix + "_corpus", _prefix + "_probe"}) {
        f.run_ok("SELECT * FROM pin_table(name => '" + t + "', tier => 'gpu', format => 'duckdb');");
      }
    }
  }
  ~InnerProductTables()
  {
    if (_pin) {
      for (auto const& t : {_prefix + "_corpus", _prefix + "_probe"}) {
        _f.con->Query("SELECT * FROM unpin_table('" + t + "');");
      }
    }
    duckdb::DBConfig::GetConfig(*_f.con->context).options.disabled_optimizers = _saved_disabled;
  }

 private:
  InnerProductFixture& _f;
  std::string _prefix;
  bool _pin;
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

namespace {

/// "(l, r), (l, r), ..." of the first two columns, for an IN (VALUES ...) list.
std::string pair_list(const std::vector<std::vector<std::string>>& rows)
{
  std::string pairs;
  for (auto const& row : rows) {
    pairs += (pairs.empty() ? "(" : ", (") + row[0] + ", " + row[1] + ")";
  }
  return pairs;
}

/// DuckDB's own inner product (or negative inner product) of every (probe id, corpus id) pair in
/// the first two columns of @p rows: an approximate join's pairs must carry exactly these scores.
std::vector<std::vector<std::string>> duckdb_scores(duckdb::Connection& con,
                                                    const std::string& prefix,
                                                    const std::vector<std::vector<std::string>>& rows,
                                                    const std::string& fn = "array_inner_product")
{
  if (rows.empty()) { return {}; }
  return on_duckdb(con,
                   "SELECT p.id, c.id, " + fn + "(p.vec, c.vec) FROM " + prefix + "_probe p, " +
                     prefix + "_corpus c WHERE (p.id, c.id) IN (VALUES " + pair_list(rows) + ");");
}

/// The fraction of @p exact's (probe id, corpus id) pairs that @p got also returned.
double pair_recall(const std::vector<std::vector<std::string>>& got,
                   const std::vector<std::vector<std::string>>& exact)
{
  std::set<std::pair<std::string, std::string>> have;
  for (auto const& row : got) {
    have.emplace(row[0], row[1]);
  }
  std::size_t hit = 0;
  for (auto const& row : exact) {
    hit += have.count({row[0], row[1]});
  }
  return exact.empty() ? 1.0 : static_cast<double>(hit) / static_cast<double>(exact.size());
}

/// @p rows with column @p col removed (ties: rows that tie on the score may be either corpus row).
std::vector<std::vector<std::string>> without_column(std::vector<std::vector<std::string>> rows,
                                                     std::size_t col)
{
  for (auto& row : rows) {
    row.erase(row.begin() + static_cast<std::ptrdiff_t>(col));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

bool lists_exist(duckdb::Connection& con, const std::string& clustering)
{
  auto r = con.Query("SELECT count(*) FROM sirius_kmeans_centroids('" + clustering + "');");
  return r && !r->HasError();
}

std::string lateral_on(const std::string& prefix,
                       const std::string& fn,
                       const std::string& order,
                       int k,
                       bool with_corpus_id = true)
{
  return "SELECT p.id, " + std::string(with_corpus_id ? "n.id, " : "") + "n.s FROM " + prefix +
         "_probe p, LATERAL (SELECT c.id, " + fn + "(p.vec, c.vec) AS s FROM " + prefix +
         "_corpus c ORDER BY s " + order + " LIMIT " + std::to_string(k) + ") n;";
}

std::string global_on(const std::string& prefix, const std::string& fn, const std::string& order)
{
  return "SELECT p.id, c.id, " + fn + "(p.vec, c.vec) AS s FROM " + prefix + "_probe p, " +
         prefix + "_corpus c ORDER BY s " + order + " LIMIT 20;";
}

std::string threshold_on(const std::string& prefix, const std::string& fn, const std::string& bound)
{
  return "SELECT p.id, c.id, " + fn + "(p.vec, c.vec) AS s FROM " + prefix + "_probe p, " +
         prefix + "_corpus c WHERE " + fn + "(p.vec, c.vec) " + bound + ";";
}

}  // namespace

TEST_CASE_METHOD(InnerProductFixture,
                 "inner-product global top-k and thresholds over FP32, FLOAT16 and UINT8 lists",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  auto const knn = [&](const std::string& prefix,
                       const std::string& select,
                       const std::string& extra) {
    return sorted_rows(*con,
                       "SELECT " + select + " FROM sirius_knn_join('" + prefix + "_probe', 'vec', '" +
                         prefix + "_corpus', 'vec', metric => 'ip'" + extra + ");");
  };
  auto const via = [](const std::string& clustering, int n_probes) {
    return ", search_mode => 'approx', clustering => '" + clustering +
           "', n_probes => " + std::to_string(n_probes);
  };
  auto const lists = [&](const std::string& prefix,
                         const std::string& name,
                         const std::string& storage,
                         int clusters) {
    run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_corpus', 'vec', name => '" + name +
           "', n_clusters => " + std::to_string(clusters) + ");");
    run_ok("SELECT * FROM sirius_kmeans_build_lists('" + prefix + "_corpus', 'vec', '" + name +
           "', storage => '" + storage + "');");
  };

  SECTION("FP32 and FLOAT16 lists (rows of varied norm)")
  {
    InnerProductTables tables(*this, "ipg");
    auto const global_ip  = on_duckdb(*con, global_on("ipg", "array_inner_product", "DESC"));
    auto const global_neg = on_duckdb(*con, global_on("ipg", "array_negative_inner_product", "ASC"));
    auto const thresh_ip  = on_duckdb(*con, threshold_on("ipg", "array_inner_product", ">= 150.0"));
    auto const thresh_neg =
      on_duckdb(*con, threshold_on("ipg", "array_negative_inner_product", "<= -150.0"));
    REQUIRE(global_ip.size() == 20);
    REQUIRE(thresh_ip.size() >= 10);
    REQUIRE(thresh_neg.size() == thresh_ip.size());
    // One FP32 list; eight FP32 / FLOAT16 clusters, every one probed (exact).
    lists("ipg", "ipg32", "float32", 1);
    lists("ipg", "ipg32x8", "float32", 8);
    lists("ipg", "ipg16x8", "float16", 8);
    for (auto const& [clustering, n_probes] :
         std::vector<std::pair<std::string, int>>{{"ipg32", 1}, {"ipg32x8", 8}, {"ipg16x8", 8}}) {
      UNSCOPED_INFO("lists " << clustering);
      auto const v = via(clustering, n_probes);
      CHECK(rows_match(knn("ipg", "left_id, right_id, similarity", ", k => 20, join_mode => 'global'" + v),
                       global_ip));
      CHECK(rows_match(knn("ipg",
                           "left_id, right_id, distance",
                           ", k => 20, join_mode => 'global', output_type => 'distance'" + v),
                       global_neg));
      CHECK(rows_match(knn("ipg", "left_id, right_id, similarity", ", eps => 150.0" + v), thresh_ip));
      // output_type => 'distance': eps bounds the negative inner product.
      CHECK(rows_match(
        knn("ipg", "left_id, right_id, distance", ", eps => -150.0, output_type => 'distance'" + v),
        thresh_neg));
    }
    // Approximate, two of eight FP32 clusters: every pair scored exactly.
    auto const approx = knn("ipg", "left_id, right_id, similarity", ", k => 7" + via("ipg32x8", 2));
    REQUIRE(approx.size() == 40 * 7);
    CHECK(rows_match(approx, duckdb_scores(*con, "ipg", approx)));

    // Plain SQL forced through lists finds the FP32 list (one cluster) and searches it, rather
    // than writing FLOAT16 codes of its own.
    scoped_env forced("SIRIUS_VSS_ACCESS_PATH", "lists");
    require_gpu_matches_duckdb(*con, lateral_on("ipg", "array_inner_product", "DESC", 7));
    require_gpu_matches_duckdb(*con, global_on("ipg", "array_inner_product", "DESC"));
    require_gpu_matches_duckdb(*con, global_on("ipg", "array_negative_inner_product", "ASC"));
    require_gpu_matches_duckdb(*con, threshold_on("ipg", "array_inner_product", ">= 150.0"));
    require_gpu_matches_duckdb(*con,
                               threshold_on("ipg", "array_negative_inner_product", "<= -150.0"));
    CHECK_FALSE(lists_exist(*con, "__sirius_auto_codes_ipg_corpus_vec_ip"));
  }

  SECTION("UINT8 lists (byte-valued rows)")
  {
    InnerProductTables tables(*this, "ipu", /*bytes=*/true);
    lists("ipu", "ipu8", "uint8", 1);
    lists("ipu", "ipu8x8", "uint8", 8);
    // Inner products of byte rows are whole numbers, so rows can tie: the top-k answers are
    // compared on (probe, score), and the threshold sits half-way between two whole numbers.
    auto const per_row =
      on_duckdb(*con, lateral_on("ipu", "array_inner_product", "DESC", 7, /*with_corpus_id=*/false));
    auto const global = without_column(on_duckdb(*con, global_on("ipu", "array_inner_product", "DESC")), 1);
    auto const thirtieth = on_duckdb(*con,
                                     "SELECT array_inner_product(p.vec, c.vec) AS s FROM ipu_probe "
                                     "p, ipu_corpus c ORDER BY s DESC LIMIT 1 OFFSET 29;");
    REQUIRE(thirtieth.size() == 1);
    auto const eps    = std::to_string(std::strtod(thirtieth[0][0].c_str(), nullptr) - 0.5);
    auto const thresh = on_duckdb(*con, threshold_on("ipu", "array_inner_product", ">= " + eps));
    REQUIRE(thresh.size() >= 30);
    for (auto const& [clustering, n_probes] :
         std::vector<std::pair<std::string, int>>{{"ipu8", 1}, {"ipu8x8", 8}}) {
      UNSCOPED_INFO("lists " << clustering);
      auto const v = via(clustering, n_probes);
      CHECK(rows_match(knn("ipu", "left_id, similarity", ", k => 7" + v), per_row));
      CHECK(rows_match(
        without_column(knn("ipu", "left_id, right_id, similarity", ", k => 20, join_mode => 'global'" + v), 1),
        global));
      CHECK(rows_match(knn("ipu", "left_id, right_id, similarity", ", eps => " + eps + v), thresh));
      CHECK(rows_match(knn("ipu", "left_id, right_id, distance", ", eps => -" + eps +
                                                                    ", output_type => 'distance'" + v),
                       on_duckdb(*con,
                                 threshold_on("ipu", "array_negative_inner_product", "<= -" + eps))));
    }
    auto const approx = knn("ipu", "left_id, right_id, similarity", ", k => 7" + via("ipu8x8", 2));
    REQUIRE(approx.size() == 40 * 7);
    CHECK(rows_match(approx, duckdb_scores(*con, "ipu", approx)));

    // Plain SQL never searches UINT8 lists for an inner product (the int8 tensor-core dot does not
    // survive the shift by 128): forced through lists, it writes FLOAT16 codes of its own.
    scoped_env forced("SIRIUS_VSS_ACCESS_PATH", "lists");
    auto const sql_per_row = lateral_on("ipu", "array_inner_product", "DESC", 7, false);
    {
      auto const before = sirius::test::get_transparent_execution_stats(*con);
      auto const got    = sorted_rows(*con, sql_per_row);
      auto const after  = sirius::test::get_transparent_execution_stats(*con);
      sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
      CHECK(rows_match(got, per_row));
    }
    CHECK(lists_exist(*con, "__sirius_auto_codes_ipu_corpus_vec_ip"));
    require_gpu_matches_duckdb(*con, threshold_on("ipu", "array_inner_product", ">= " + eps));
  }
}

TEST_CASE_METHOD(InnerProductFixture,
                 "plain-SQL approximate inner-product join at partial probe counts",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  // SET vector_join_probes makes the rewrite partition the corpus inside the query (an l2 k-means
  // fit, FLOAT16 lists of its own) and route each probe row to the clusters of largest inner
  // product with their centroid. Every pair it returns is scored exactly, and recall against
  // DuckDB's exact answer rises with the probe count, to 1 with every cluster probed.
  InnerProductTables tables(*this, "ipa");
  auto const lateral   = lateral_on("ipa", "array_inner_product", "DESC", 7);
  auto const exact     = on_duckdb(*con, lateral);
  auto const exact_thr = on_duckdb(*con, threshold_on("ipa", "array_inner_product", ">= 150.0"));
  REQUIRE(exact.size() == 40 * 7);
  run_ok("SET vector_join_clusters = 16;");
  struct reset_on_exit {
    duckdb::Connection& c;
    ~reset_on_exit()
    {
      c.Query("RESET vector_join_probes;");
      c.Query("RESET vector_join_clusters;");
    }
  } reset{*con};
  double previous = 0.0;
  for (int const probes : {1, 2, 4, 16}) {
    UNSCOPED_INFO("vector_join_probes = " << probes);
    run_ok("SET vector_join_probes = " + std::to_string(probes) + ";");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto const got    = sorted_rows(*con, lateral);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(got.size() == 40 * 7);
    CHECK(rows_match(got, duckdb_scores(*con, "ipa", got)));
    auto const recall = pair_recall(got, exact);
    UNSCOPED_INFO("recall " << recall);
    CHECK(recall >= previous);
    if (probes == 1) { CHECK(recall < 0.95); }
    if (probes == 16) { CHECK(recall == 1.0); }
    previous = recall;
    // The global top-k and the threshold under the same setting: exact scores; a threshold pair
    // DuckDB would not return never appears.
    auto const global = sorted_rows(*con, global_on("ipa", "array_inner_product", "DESC"));
    REQUIRE(global.size() == 20);
    CHECK(rows_match(global, duckdb_scores(*con, "ipa", global)));
    auto const thr = sorted_rows(*con, threshold_on("ipa", "array_inner_product", ">= 150.0"));
    CHECK(rows_match(thr, duckdb_scores(*con, "ipa", thr)));
    CHECK(pair_recall(exact_thr, thr) == 1.0);
    if (probes == 16) { CHECK(rows_match(thr, exact_thr)); }
  }
  auto partitions = con->Query(
    "SELECT count(DISTINCT cluster_id) FROM sirius_kmeans_centroids('__sirius_auto_ipa_corpus_vec_ip');");
  REQUIRE_FALSE(partitions->HasError());
  CHECK(partitions->GetValue(0, 0).GetValue<std::int64_t>() == 16);
}

TEST_CASE_METHOD(InnerProductFixture,
                 "inner-product joins over a scanned (unpinned) corpus",
                 "[integration][gpu_execution][array][vss][vector_join][inner_product]")
{
  // Nothing pinned: plain SQL takes both sides from scans, and so does the table function with
  // probe_source / build_source => 'scan'. The corpus is searched by the FP32 GEMM, chunk by chunk.
  InnerProductTables tables(*this, "ips", /*bytes=*/false, /*pin=*/false);
  require_gpu_matches_duckdb(*con, lateral_on("ips", "array_inner_product", "DESC", 7));
  require_gpu_matches_duckdb(*con, lateral_on("ips", "array_negative_inner_product", "ASC", 5));
  require_gpu_matches_duckdb(*con, global_on("ips", "array_inner_product", "DESC"));
  require_gpu_matches_duckdb(*con, threshold_on("ips", "array_inner_product", ">= 150.0"));
  require_gpu_matches_duckdb(*con, threshold_on("ips", "array_negative_inner_product", "<= -150.0"));
  auto const tvf = [&](const std::string& corpus, const std::string& extra) {
    return sorted_rows(*con,
                       "SELECT left_id, right_id, similarity FROM sirius_knn_join('ips_probe', "
                       "'vec', '" + corpus + "', 'vec', metric => 'ip', probe_source => 'scan', "
                       "build_source => 'scan'" + extra + ");");
  };
  auto const exact = on_duckdb(*con, lateral_on("ips", "array_inner_product", "DESC", 7));
  CHECK(rows_match(tvf("ips_corpus", ", k => 7"), exact));
  CHECK(rows_match(tvf("ips_corpus", ", k => 20, join_mode => 'global'"),
                   on_duckdb(*con, global_on("ips", "array_inner_product", "DESC"))));
  CHECK(rows_match(tvf("ips_corpus", ", eps => 150.0"),
                   on_duckdb(*con, threshold_on("ips", "array_inner_product", ">= 150.0"))));

  // An approximate join over a scanned, cluster-ordered copy (the cluster ids a column of it):
  // every cluster probed is exact; two of eight are scored exactly.
  run_ok("SELECT * FROM pin_table(name => 'ips_corpus', tier => 'gpu', format => 'duckdb');");
  run_ok("SELECT * FROM sirius_kmeans_fit('ips_corpus', 'vec', name => 'ips8', n_clusters => 8);");
  run_ok("CREATE TABLE ips_asg AS SELECT * FROM sirius_kmeans_assign('ips_corpus', 'vec', 'ips8', "
         "n_probes => 1);");
  run_ok("CREATE TABLE ips_ordered AS SELECT r.id, r.vec, a.cluster_id FROM ips_corpus r JOIN "
         "ips_asg a ON r.rowid = a.row_id ORDER BY a.cluster_id;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM unpin_table('ips_corpus');");
  auto const clustered = [&](int n_probes) {
    return tvf("ips_ordered",
               ", k => 7, search_mode => 'approx', clustering => 'ips8', cluster_column => "
               "'cluster_id', right_output_columns => ['id'], n_probes => " +
                 std::to_string(n_probes));
  };
  CHECK(rows_match(clustered(8), exact));
  auto const approx = clustered(2);
  REQUIRE(approx.size() == 40 * 7);
  CHECK(rows_match(approx, duckdb_scores(*con, "ips", approx)));
}
