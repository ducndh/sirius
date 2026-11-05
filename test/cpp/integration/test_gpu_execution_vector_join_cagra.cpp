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

// sirius_knn_join(search_mode => 'index'): answers checked against a float64 brute force over the
// rows read back from DuckDB, refusals, and the index's reuse.

#include "vss/cagra_settings.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int k_dim = 32;

using rows_t = std::vector<std::vector<double>>;

/// Rows with some structure, as the neighbour graph expects: a Gaussian around one of 16 centres.
std::string gaussian_rows_sql(int rows, int dim, int seed, bool unit)
{
  auto const seed_a = std::to_string(seed), seed_b = std::to_string(seed + 1);
  auto const raw = "list_transform(range(" + std::to_string(dim) + "), lambda j: hash(i % 16, j, " +
                   seed_a + ") % 1000 / 1000.0 + (hash(i, j, " + seed_b +
                   ") % 1000 / 1000.0 - 0.5) * 0.25)";
  auto const out =
    unit ? std::string{"list_transform(v, lambda x: x / sqrt(list_aggregate(list_transform(v, "
                       "lambda y: y * y), 'sum')))"}
         : std::string{"v"};
  return "SELECT i::BIGINT AS id, (" + out + ")::FLOAT[" + std::to_string(dim) +
         "] AS vec FROM (SELECT i, " + raw + " AS v FROM range(" + std::to_string(rows) + ") t(i))";
}

std::map<std::int64_t, std::vector<double>> read_rows(duckdb::Connection& con,
                                                      std::string const& table)
{
  auto r = con.Query("SELECT id, vec FROM " + table + " ORDER BY id");
  REQUIRE_FALSE(r->HasError());
  std::map<std::int64_t, std::vector<double>> out;
  for (duckdb::idx_t i = 0; i < r->RowCount(); ++i) {
    std::vector<double> row;
    auto const vec = r->Collection().GetValue(1, i);
    for (auto const& v : duckdb::ArrayValue::GetChildren(vec)) {
      row.push_back(v.GetValue<double>());
    }
    out.emplace(r->Collection().GetValue(0, i).GetValue<std::int64_t>(), std::move(row));
  }
  return out;
}

double l2(std::vector<double> const& a, std::vector<double> const& b)
{
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(s);
}

double cosine_distance(std::vector<double> const& a, std::vector<double> const& b)
{
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return 1.0 - ab / std::sqrt(aa * bb);
}

struct pair_row {
  std::int64_t probe, corpus;
  double score;
};

std::vector<pair_row> join_rows(duckdb::Connection& con, std::string const& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  std::vector<pair_row> out;
  for (duckdb::idx_t i = 0; i < r->RowCount(); ++i) {
    out.push_back({r->Collection().GetValue(0, i).GetValue<std::int64_t>(),
                   r->Collection().GetValue(1, i).GetValue<std::int64_t>(),
                   r->Collection().GetValue(2, i).GetValue<double>()});
  }
  return out;
}

void expect_error(duckdb::Connection& con, std::string const& sql, std::string const& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

std::string index_event(duckdb::Connection& con)
{
  auto r = con.Query("SELECT index_event FROM sirius_vector_join_index_stats()");
  REQUIRE_FALSE(r->HasError());
  REQUIRE(r->RowCount() == 1);
  return r->Collection().GetValue(0, 0).ToString();
}

std::string join_sql(std::string const& options, std::string const& probe = "probe")
{
  return "SELECT left_id, right_id, distance FROM sirius_knn_join('" + probe +
         "', 'vec', 'corpus', 'vec', search_mode => 'index', left_output_columns => ['id'], "
         "right_output_columns => ['id']" +
         (options.empty() ? "" : ", " + options) + ")";
}

/// Every pair's score against a float64 brute force, every probe row's count, and the recall.
template <typename DistanceFn>
double check_answer(std::vector<pair_row> const& pairs,
                    std::map<std::int64_t, std::vector<double>> const& probe,
                    std::map<std::int64_t, std::vector<double>> const& corpus,
                    int k,
                    DistanceFn distance,
                    double relative_tolerance,
                    bool allow_short = false)
{
  std::map<std::int64_t, std::vector<pair_row>> by_probe;
  for (auto const& p : pairs) {
    by_probe[p.probe].push_back(p);
  }
  REQUIRE(by_probe.size() == probe.size());
  std::int64_t hits = 0;
  for (auto const& [id, q] : probe) {
    auto const& got = by_probe.at(id);
    if (allow_short) {
      REQUIRE(got.size() <= static_cast<std::size_t>(k));
    } else {
      REQUIRE(got.size() == static_cast<std::size_t>(k));
    }
    std::vector<std::pair<double, std::int64_t>> all;
    all.reserve(corpus.size());
    for (auto const& [cid, x] : corpus) {
      all.emplace_back(distance(q, x), cid);
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::set<std::int64_t> truth;
    for (int i = 0; i < k; ++i) {
      truth.insert(all[static_cast<std::size_t>(i)].second);
    }
    std::set<std::int64_t> seen;
    for (auto const& p : got) {
      REQUIRE(seen.insert(p.corpus).second);
      auto const oracle = distance(q, corpus.at(p.corpus));
      CHECK(std::abs(p.score - oracle) <= relative_tolerance * std::max(oracle, 1e-3));
      hits += truth.contains(p.corpus) ? 1 : 0;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(probe.size() * k);
}

class CagraJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  void make_tables(bool unit, int dim = k_dim, int corpus_rows = 5000, int probe_rows = 300)
  {
    run_ok("CREATE TABLE corpus AS " + gaussian_rows_sql(corpus_rows, dim, 7, unit));
    run_ok("CREATE TABLE probe AS " + gaussian_rows_sql(probe_rows, dim, 11, unit));
    run_ok("CHECKPOINT;");
    probe_rows_  = read_rows(*con, "probe");
    corpus_rows_ = read_rows(*con, "corpus");
    run_ok("SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');");
    run_ok("SELECT * FROM pin_table(name => 'probe', tier => 'gpu', format => 'duckdb');");
  }
  std::map<std::int64_t, std::vector<double>> probe_rows_, corpus_rows_;
};

}  // namespace

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: L2, FP16 and FP32 rows, against float64",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  auto const& probe  = probe_rows_;
  auto const& corpus = corpus_rows_;
  std::map<std::pair<std::int64_t, std::int64_t>, double> first_distances;
  for (auto const* row_type : {"float16", "float32"}) {
    UNSCOPED_INFO("row_type " << row_type);
    auto const pairs = join_rows(
      *con, join_sql(std::string("k => 10, itopk => 64, row_type => '") + row_type + "'"));
    REQUIRE(index_event(*con) == "built");
    auto const recall = check_answer(pairs, probe, corpus, 10, l2, 2e-6);
    CHECK(recall >= 0.85);
    // Distances come from the FP32 rows whichever rows the walk read, so they agree to the bit.
    for (auto const& p : pairs) {
      auto const [it, fresh] = first_distances.emplace(std::pair{p.probe, p.corpus}, p.score);
      if (!fresh) { CHECK(it->second == p.score); }
    }
  }
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: cosine over unit rows, against float64",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(true);
  auto const& probe  = probe_rows_;
  auto const& corpus = corpus_rows_;
  auto const pairs =
    join_rows(*con, join_sql("k => 10, metric => 'cosine', output_type => 'distance'"));
  // Unit rows are unit to about 1e-7, which is the relative error of (q - x)^2 / 2 as a distance.
  CHECK(check_answer(pairs, probe, corpus, 10, cosine_distance, 3e-5) >= 0.85);

  auto const similarity = join_rows(
    *con,
    "SELECT left_id, right_id, similarity FROM sirius_knn_join('probe', 'vec', 'corpus', 'vec', "
    "search_mode => 'index', metric => 'cosine', k => 3, left_output_columns => ['id'], "
    "right_output_columns => ['id'])");
  REQUIRE(similarity.size() == probe.size() * 3);
  for (auto const& p : similarity) {
    CHECK(std::abs(p.score - (1.0 - cosine_distance(probe.at(p.probe), corpus.at(p.corpus)))) <=
          1e-5);
  }
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: answers a self-join, a host pin, a relation and any width",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  auto const& probe  = probe_rows_;
  auto const& corpus = corpus_rows_;

  SECTION("self-join")
  {
    auto const pairs = join_rows(*con, join_sql("k => 5", "corpus"));
    CHECK(pairs.size() == corpus.size() * 5);
    std::int64_t self = 0;
    for (auto const& p : pairs) {
      self += p.probe == p.corpus ? 1 : 0;
    }
    CHECK(self >= static_cast<std::int64_t>(corpus.size() * 95 / 100));
  }
  SECTION("host pin")
  {
    run_ok("SELECT * FROM unpin_table('corpus');");
    run_ok("SELECT * FROM pin_table(name => 'corpus', tier => 'host', format => 'duckdb');");
    auto const pairs = join_rows(*con, join_sql("k => 10"));
    CHECK(check_answer(pairs, probe, corpus, 10, l2, 2e-6) >= 0.85);
  }
  SECTION("relational probe")
  {
    auto const pairs = join_rows(*con,
                                 "SELECT left_id, right_id, distance FROM sirius_knn_join_rel("
                                 "(SELECT id, vec FROM probe WHERE id < 100), 'vec', 'corpus', "
                                 "'vec', search_mode => 'index', k => 10, "
                                 "left_output_columns => ['id'], right_output_columns => ['id'])");
    CHECK(pairs.size() == 1000);
  }
  SECTION("k past itopk, and k past the table")
  {
    auto const pairs = join_rows(*con, join_sql("k => 100, itopk => 16"));
    CHECK(pairs.size() == probe.size() * 100);
  }
  SECTION("a probe table with no rows")
  {
    run_ok("CREATE TABLE no_rows (id BIGINT, vec FLOAT[32]);");
    run_ok("SELECT * FROM pin_table(name => 'no_rows', tier => 'gpu', format => 'duckdb');");
    CHECK(join_rows(*con, join_sql("k => 5", "no_rows")).empty());
  }
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: a width whose rows need padding",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false, 12, 3000, 100);
  auto const& probe  = probe_rows_;
  auto const& corpus = corpus_rows_;
  for (auto const* row_type : {"float16", "float32"}) {
    UNSCOPED_INFO("row_type " << row_type);
    auto const pairs =
      join_rows(*con, join_sql(std::string("k => 10, row_type => '") + row_type + "'"));
    CHECK(check_answer(pairs, probe, corpus, 10, l2, 2e-6, /*allow_short=*/true) >= 0.7);
  }
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: refusals",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  auto const join = [](std::string const& options) {
    return "SELECT * FROM sirius_knn_join('probe', 'vec', 'corpus', 'vec', search_mode => "
           "'index', " +
           options + ")";
  };
  expect_error(*con, join("eps => 0.5"), "threshold");
  expect_error(*con, join("join_mode => 'threshold'"), "threshold");
  expect_error(*con, join("join_mode => 'global'"), "top-k per row");
  expect_error(*con, join("metric => 'ip'"), "metric");
  expect_error(*con, join("k => 0"), "k >= 1");
  expect_error(*con, join("eps => 0.5, keep => 'best'"), "threshold");
  expect_error(*con, join("n_probes => 4"), "n_probes");
  expect_error(*con, join("build_source => 'scan'"), "build_source");
  expect_error(*con, join("itopk => 0"), "itopk must be >= 1");
  expect_error(*con, join("graph_degree => 64, intermediate_graph_degree => 32"), "intermediate");
  expect_error(*con, join("row_type => 'int8'"), "row_type");
  expect_error(*con, join("build_algo => 'ace'"), "build_algo");
  expect_error(*con, join("index_type => 'hnsw'"), "index_type");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('probe', 'vec', 'corpus', 'vec', itopk => 32)",
               "only applies under search_mode");

  // Rows that are not unit cannot be read as cosine by the L2 walk, on the corpus or the probe.
  expect_error(*con, join("metric => 'cosine'"), "unit rows");

  // A table that does not fit is refused by its numbers before anything is allocated.
  expect_error(*con,
               join("graph_degree => 100000000000, intermediate_graph_degree => 100000000000"),
               "does not fit");

  run_ok("CREATE TABLE tiny (id BIGINT, vec FLOAT[32]);");
  run_ok("INSERT INTO tiny SELECT * FROM (" + gaussian_rows_sql(10, 32, 3, false) + ");");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'tiny', tier => 'gpu', format => 'duckdb');");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('probe', 'vec', 'tiny', 'vec', search_mode => "
               "'index', k => 3)",
               "more rows than its intermediate graph degree");
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: one index per column, reused until it cannot answer",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  auto const first = join_rows(*con, join_sql("k => 10, itopk => 32"));
  CHECK(index_event(*con) == "built");

  // Any search settings over the same index, and answers that repeat.
  auto const again = join_rows(*con, join_sql("k => 10, itopk => 32"));
  CHECK(index_event(*con) == "reused");
  auto const by_ids = [](std::vector<pair_row> v) {
    std::sort(v.begin(), v.end(), [](auto const& a, auto const& b) {
      return std::tie(a.probe, a.corpus) < std::tie(b.probe, b.corpus);
    });
    return v;
  };
  auto const a = by_ids(first), b = by_ids(again);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].corpus == b[i].corpus);
    CHECK(a[i].score == b[i].score);
  }
  static_cast<void>(join_rows(*con, join_sql("k => 5, itopk => 128, search_width => 2")));
  CHECK(index_event(*con) == "reused");

  // Unit rows or not is a property of the rows: the same index answers the other metric, if unit.
  static_cast<void>(join_rows(*con, join_sql("k => 5, metric => 'l2'")));
  CHECK(index_event(*con) == "reused");

  // A build setting asks for the index built that way, and replaces the one the column had.
  static_cast<void>(join_rows(*con,
                              join_sql("k => 5, graph_degree => 16, "
                                       "intermediate_graph_degree => 32")));
  CHECK(index_event(*con) == "built");
  static_cast<void>(join_rows(*con, join_sql("k => 5")));
  CHECK(index_event(*con) == "reused");

  // Built ahead of the statement, then reused by it.
  run_ok(
    "SELECT * FROM sirius_create_ann_index('corpus', 'vec', index_type => 'cagra', "
    "graph_degree => 24, intermediate_graph_degree => 48, nn_descent_niter => 6);");
  static_cast<void>(join_rows(*con, join_sql("k => 5")));
  CHECK(index_event(*con) == "reused");
  static_cast<void>(join_rows(*con,
                              join_sql("k => 5, graph_degree => 24, "
                                       "intermediate_graph_degree => 48, "
                                       "nn_descent_niter => 6")));
  CHECK(index_event(*con) == "reused");

  // Pinned again, the row positions it answers in are gone: rebuilt, not trusted.
  run_ok("SELECT * FROM unpin_table('corpus');");
  run_ok("SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');");
  static_cast<void>(join_rows(*con, join_sql("k => 5")));
  CHECK(index_event(*con) == "built");

  run_ok("SELECT * FROM sirius_drop_ann_index('corpus', 'vec');");
  static_cast<void>(join_rows(*con, join_sql("k => 5")));
  CHECK(index_event(*con) == "built");
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: the phases' times are reported",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  static_cast<void>(join_rows(*con, join_sql("k => 10")));
  auto r = con->Query(
    "SELECT build_s, build_convert_s, search_s, rescore_s, probe_rows, misses, index_bytes FROM "
    "sirius_vector_join_index_stats()");
  REQUIRE_FALSE(r->HasError());
  REQUIRE(r->RowCount() == 1);
  CHECK(r->Collection().GetValue(0, 0).GetValue<double>() > 0.0);
  CHECK(r->Collection().GetValue(1, 0).GetValue<double>() > 0.0);
  CHECK(r->Collection().GetValue(2, 0).GetValue<double>() > 0.0);
  CHECK(r->Collection().GetValue(3, 0).GetValue<double>() > 0.0);
  CHECK(r->Collection().GetValue(4, 0).GetValue<std::int64_t>() == 300);
  CHECK(r->Collection().GetValue(5, 0).GetValue<std::int64_t>() == 0);
  // The rows at FP16, and a graph of 32 ids a row.
  CHECK(r->Collection().GetValue(6, 0).GetValue<std::int64_t>() >= 5000 * (32 * 2 + 32 * 4));
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "sirius_knn_join index: a task out of memory is retried to the same answer",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  auto const expected = join_rows(*con, join_sql("k => 10"));
  run_ok("SET sirius_test_inject_gpu_task_oom=2");
  auto const retried = join_rows(*con, join_sql("k => 10"));
  run_ok("SET sirius_test_inject_gpu_task_oom=0");
  CHECK(retried.size() == expected.size());
}

TEST_CASE_METHOD(CagraJoinFixture,
                 "SET vector_join_index = 'cagra': a plain top-k join runs through the index",
                 "[integration][gpu_execution][array][vss][vector_join][cagra]")
{
  make_tables(false);
  run_ok("SET vector_join_index = 'cagra';");
  auto const pairs = join_rows(*con,
                               "SELECT p.id, n.id, n.d FROM probe p, LATERAL (SELECT c.id, "
                               "array_distance(p.vec, c.vec) AS d FROM corpus c ORDER BY d "
                               "LIMIT 10) n");
  CHECK(index_event(*con) == "built");
  CHECK(check_answer(pairs, probe_rows_, corpus_rows_, 10, l2, 2e-6) >= 0.85);
}

TEST_CASE("cagra search scratch follows itopk and width", "[integration][vss][vector_join][cagra]")
{
  using sirius::vss::cagra_search_scratch_per_row;
  // Shared memory holds the walk at low itopk and the table is reset every iteration.
  CHECK(cagra_search_scratch_per_row(1000000, 32, 10, {64, 1}) == 0);
  CHECK(cagra_search_scratch_per_row(1000000, 32, 10, {512, 1}) == 0);
  // Past itopk 512 the search keeps a table per row: 262,144 rows here asked cuVS for 4.3 GB.
  CHECK(cagra_search_scratch_per_row(100000000, 32, 10, {1024, 8}) >= 16384);
  // A wide search outgrows the small hash, and more of it takes more.
  auto const wide  = cagra_search_scratch_per_row(1000000, 32, 10, {512, 128});
  auto const wider = cagra_search_scratch_per_row(1000000, 32, 10, {512, 256});
  CHECK(wide > 0);
  CHECK(wider >= wide);
}
