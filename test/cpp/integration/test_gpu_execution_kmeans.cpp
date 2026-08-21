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
 * @file test_gpu_execution_kmeans.cpp
 * @brief End-to-end tests for sirius_kmeans_fit() / sirius_kmeans_assign(), the pair that
 *        clusters a pinned vector column and emits the assignment edge list that a
 *        cluster-ordered copy of the table is built from.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

using KMeansFixture = sirius::test::GpuExecutionFixture;

namespace {

std::unique_ptr<duckdb::MaterializedQueryResult> query_ok(duckdb::Connection& con,
                                                          const std::string& sql)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO("query error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
  return std::unique_ptr<duckdb::MaterializedQueryResult>(
    static_cast<duckdb::MaterializedQueryResult*>(r.release()));
}

std::int64_t scalar_i64(duckdb::Connection& con, const std::string& sql)
{
  auto r = query_ok(con, sql);
  return r->GetValue(0, 0).GetValue<std::int64_t>();
}

// Sorted rows from a query that must succeed, so two result sets compare independent of the
// order rows happen to arrive in.
std::vector<std::vector<std::string>> ok_rows(duckdb::Connection& con, const std::string& sql)
{
  auto r = query_ok(con, sql);
  return sirius::test::GpuExecutionFixture::collect_rows(*r, /*sort=*/true);
}

// (probe id, distance rounded) pairs, sorted. Rounding happens here rather than in SQL because a
// projection over the join falls back to the CPU, which cannot execute the rewrite target.
std::vector<std::pair<std::string, long long>> distance_multiset(duckdb::Connection& con,
                                                                 const std::string& sql)
{
  auto const rows = ok_rows(con, sql);
  std::vector<std::pair<std::string, long long>> out;
  out.reserve(rows.size());
  for (auto const& r : rows) {
    out.emplace_back(r.at(0), std::llround(std::stod(r.at(1)) * 1000.0));
  }
  std::sort(out.begin(), out.end());
  return out;
}

void expect_error(duckdb::Connection& con, const std::string& sql, const std::string& needle)
{
  auto r = con.Query(sql);
  REQUIRE(r);
  REQUIRE(r->HasError());
  UNSCOPED_INFO("error was: " << r->GetError());
  REQUIRE(r->GetError().find(needle) != std::string::npos);
}

// Two tight, well-separated groups in 3-D: ids 0-199 near the origin, ids 200-399 near
// (100,100,100). Any sane 2-means splits them, so cluster membership is checkable without
// depending on which label each group happens to get.
void create_two_group_table(KMeansFixture& fixture, const std::string& table)
{
  fixture.run_ok("CREATE TABLE " + table + " (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + table +
                 " SELECT i, [(i % 5)::float, (i % 3)::float, (i % 7)::float] "
                 "FROM range(200) t(i);");
  fixture.run_ok("INSERT INTO " + table +
                 " SELECT 200 + i, "
                 "[100.0 + (i % 5), 100.0 + (i % 3), 100.0 + (i % 7)]::FLOAT[3] "
                 "FROM range(200) t(i);");
  fixture.run_ok("CHECKPOINT;");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + table +
                 "', tier => 'gpu', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_fit reports the knobs it resolved",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmr_corpus");

  auto r = query_ok(*con,
                    "SELECT n_clusters, dim, train_rows, n_rows FROM sirius_kmeans_fit("
                    "'kmr_corpus','vec', name => 'kmr_c', n_clusters => 2);");
  REQUIRE(r->RowCount() == 1);
  CHECK(r->GetValue(0, 0).GetValue<std::int64_t>() == 2);
  CHECK(r->GetValue(1, 0).GetValue<std::int64_t>() == 3);
  CHECK(r->GetValue(3, 0).GetValue<std::int64_t>() == 400);
  // The default sample is 256 rows per cluster, which the 400-row table cannot supply.
  CHECK(r->GetValue(2, 0).GetValue<std::int64_t>() == 400);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_fit defaults n_clusters to sqrt(n_rows)",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kma_corpus");

  auto const n = scalar_i64(*con,
                            "SELECT n_clusters FROM sirius_kmeans_fit("
                            "'kma_corpus','vec', name => 'kma_auto');");
  CHECK(n == 20);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign emits one edge per row and separates the two groups",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmb_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmb_corpus','vec', name => 'kmb_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmb_asg AS SELECT * FROM sirius_kmeans_assign('kmb_corpus','vec','kmb_c', "
    "n_probes => 1);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg;") == 400);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kmb_asg;") == 400);
  CHECK(scalar_i64(*con, "SELECT min(row_id) FROM kmb_asg;") == 0);
  CHECK(scalar_i64(*con, "SELECT max(row_id) FROM kmb_asg;") == 399);
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg WHERE cluster_id NOT IN (0,1);") == 0);
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmb_asg WHERE distance < 0;") == 0);

  // Each separated group lands wholly in one cluster, and the two land in different ones.
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id < 200;") ==
        1);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id >= 200;") ==
        1);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT cluster_id) FROM kmb_asg;") == 2);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign repeats each row once per probe",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmc_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmc_corpus','vec', name => 'kmc_c', n_clusters => 4);");
  run_ok(
    "CREATE TABLE kmc_asg AS SELECT * FROM sirius_kmeans_assign('kmc_corpus','vec','kmc_c', "
    "n_probes => 3);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmc_asg;") == 1200);
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kmc_asg;") == 400);
  // A row never takes the same cluster twice.
  CHECK(scalar_i64(*con,
                   "SELECT count(*) FROM (SELECT row_id, cluster_id FROM kmc_asg "
                   "GROUP BY row_id, cluster_id HAVING count(*) > 1);") == 0);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign caps probes at the cluster count",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmd_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmd_corpus','vec', name => 'kmd_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmd_asg AS SELECT * FROM sirius_kmeans_assign('kmd_corpus','vec','kmd_c', "
    "n_probes => 10);");

  CHECK(scalar_i64(*con, "SELECT count(*) FROM kmd_asg;") == 800);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign radius mode keeps every row and varies the edge count",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kme_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kme_corpus','vec', name => 'kme_c', n_clusters => 4);");
  run_ok(
    "CREATE TABLE kme_asg AS SELECT * FROM sirius_kmeans_assign('kme_corpus','vec','kme_c', "
    "radius_factor => 0.05, max_probes => 4);");

  // Every row keeps its nearest centroid, so no row can drop out...
  CHECK(scalar_i64(*con, "SELECT count(DISTINCT row_id) FROM kme_asg;") == 400);
  // ...and the two groups are far enough apart that a 5% radius admits nothing across them,
  // which is what makes this fewer edges than a fixed n_probes => 4 would produce.
  CHECK(scalar_i64(*con, "SELECT count(*) FROM kme_asg;") < 1600);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign row ids line up with the table's rowid",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmf_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmf_corpus','vec', name => 'kmf_c', n_clusters => 2);");
  run_ok(
    "CREATE TABLE kmf_asg AS SELECT * FROM sirius_kmeans_assign('kmf_corpus','vec','kmf_c', "
    "n_probes => 1);");

  // The whole cluster-ordering flow rests on the assignment's row_id addressing the same row
  // the table's rowid does; the ids were built to make that checkable.
  con->Query("SET gpu_execution = false;");
  auto const mismatches = scalar_i64(*con,
                                     "SELECT count(*) FROM kmf_corpus c JOIN kmf_asg a ON c.rowid "
                                     "= a.row_id WHERE c.id <> a.row_id;");
  con->Query("SET gpu_execution = true;");
  CHECK(mismatches == 0);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans functions reject bad arguments",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmg_corpus");

  expect_error(*con,
               "SELECT * FROM sirius_kmeans_fit('kmg_corpus','id', name => 'x');",
               "must be a FLOAT[N] array column");
  expect_error(*con, "SELECT * FROM sirius_kmeans_fit('kmg_corpus','vec');", "'name'");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_fit('kmg_corpus','vec', name => 'x', n_iters => 0);",
               "n_iters must be >= 1");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_assign('kmg_corpus','vec','no_such_clustering');",
               "no clustering named");
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_assign('kmg_corpus','vec','x', n_probes => 0);",
               "n_probes must be >= 1");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign rejects a clustering trained on a different width",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmh_corpus");
  run_ok("CREATE TABLE kmh_wide (id INTEGER, vec FLOAT[4]);");
  run_ok("INSERT INTO kmh_wide SELECT i, [i::float, 1, 2, 3] FROM range(100) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'kmh_wide', tier => 'gpu', format => 'duckdb');");

  run_ok("SELECT * FROM sirius_kmeans_fit('kmh_corpus','vec', name => 'kmh_c', n_clusters => 2);");
  expect_error(
    *con, "SELECT * FROM sirius_kmeans_assign('kmh_wide','vec','kmh_c');", "is over FLOAT[3]");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_kmeans_assign matches a CPU-computed nearest centroid",
                 "[integration][gpu_execution][array][vss][kmeans]")
{
  create_two_group_table(*this, "kmo_corpus");
  run_ok("SELECT * FROM sirius_kmeans_fit('kmo_corpus','vec', name => 'kmo_c', n_clusters => 4);");
  run_ok("CREATE TABLE kmo_cent AS SELECT * FROM sirius_kmeans_centroids('kmo_c');");
  run_ok(
    "CREATE TABLE kmo_asg AS SELECT * FROM sirius_kmeans_assign('kmo_corpus','vec','kmo_c', "
    "n_probes => 1);");

  // The oracle is DuckDB on the CPU: pivot the centroids back to one row each, compute every
  // (row, centroid) distance in SQL, and keep each row's argmin. Nothing here shares code with
  // the GPU path, so agreement is evidence the assignment is right rather than self-consistent.
  con->Query("SET gpu_execution = false;");
  auto const disagreements =
    scalar_i64(*con,
               "WITH cent AS ("
               "  SELECT cluster_id,"
               "         max(CASE WHEN dim_index = 0 THEN value END) AS c0,"
               "         max(CASE WHEN dim_index = 1 THEN value END) AS c1,"
               "         max(CASE WHEN dim_index = 2 THEN value END) AS c2"
               "  FROM kmo_cent GROUP BY cluster_id),"
               "ranked AS ("
               "  SELECT t.rowid AS row_id, k.cluster_id,"
               "         row_number() OVER (PARTITION BY t.rowid ORDER BY"
               "           (t.vec[1]-k.c0)*(t.vec[1]-k.c0) + (t.vec[2]-k.c1)*(t.vec[2]-k.c1)"
               "         + (t.vec[3]-k.c2)*(t.vec[3]-k.c2), k.cluster_id) AS rn"
               "  FROM kmo_corpus t CROSS JOIN cent k),"
               "oracle AS (SELECT row_id, cluster_id FROM ranked WHERE rn = 1)"
               "SELECT count(*) FROM oracle o JOIN kmo_asg a USING (row_id)"
               "  WHERE o.cluster_id <> a.cluster_id;");
  con->Query("SET gpu_execution = true;");

  CHECK(disagreements == 0);
}

// -----------------------------------------------------------------------------
// The approximate join. Its first gate is not recall but exactness: probing every
// cluster skips nothing, so the answer must match the exhaustive join it is built on.
// -----------------------------------------------------------------------------

namespace {

// A corpus stored in cluster order, plus a probe table, plus a clustering over both.
//
// @p corpus_rows sets how many pinned chunks the corpus lands in: batching follows DuckDB row
// groups (122,880 rows), so anything under that is a single chunk.
void create_clustered_join(KMeansFixture& fixture,
                           duckdb::Connection& con,
                           const std::string& prefix,
                           int n_clusters,
                           int corpus_rows = 4000)
{
  auto const corpus = prefix + "_corpus";
  auto const probe  = prefix + "_probe";
  auto const clust  = prefix + "_c";

  fixture.run_ok("CREATE TABLE " + prefix + "_raw (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + prefix +
                 "_raw SELECT i, "
                 "[(i % 97)::float, ((i * 7) % 89)::float, ((i * 13) % 83)::float] "
                 "FROM range(" +
                 std::to_string(corpus_rows) + ") t(i);");
  fixture.run_ok("CREATE TABLE " + probe + " (id INTEGER, vec FLOAT[3]);");
  fixture.run_ok("INSERT INTO " + probe +
                 " SELECT i, "
                 "[((i * 3) % 97)::float, ((i * 11) % 89)::float, ((i * 5) % 83)::float] "
                 "FROM range(200) t(i);");
  fixture.run_ok("CHECKPOINT;");

  // Cluster the raw table, then materialize a copy ordered by cluster. The order is not an
  // optimization: the join reads each chunk's cluster runs, and a chunk whose labels are not
  // non-decreasing is refused rather than silently answered from a partial range.
  fixture.run_ok("SELECT * FROM pin_table(name => '" + prefix +
                 "_raw', tier => 'gpu', format => 'duckdb');");
  fixture.run_ok("SELECT * FROM sirius_kmeans_fit('" + prefix + "_raw','vec', name => '" + clust +
                 "', n_clusters => " + std::to_string(n_clusters) + ");");
  fixture.run_ok("CREATE TABLE " + prefix + "_asg AS SELECT * FROM sirius_kmeans_assign('" +
                 prefix + "_raw','vec','" + clust + "', n_probes => 1);");
  fixture.run_ok("CREATE TABLE " + corpus + " AS SELECT r.id, r.vec, a.cluster_id FROM " + prefix +
                 "_raw r JOIN " + prefix + "_asg a ON r.rowid = a.row_id ORDER BY a.cluster_id;");
  fixture.run_ok("CHECKPOINT;");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + corpus +
                 "', tier => 'gpu', format => 'duckdb');");
  fixture.run_ok("SELECT * FROM pin_table(name => '" + probe +
                 "', tier => 'gpu', format => 'duckdb');");
}

}  // namespace

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx probing every cluster equals the exhaustive join",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  constexpr int n_clusters = 8;
  create_clustered_join(*this, *con, "apx", n_clusters);

  // Compared on DISTANCES, not neighbour ids. Equidistant corpus rows are interchangeable
  // answers, and the two paths visit clusters in different orders, so they break those ties
  // differently -- an id comparison reports that as a failure and undercuts by exactly the tie
  // multiplicity. The distance multiset is what "same answer" actually means here.
  auto const exhaustive =
    distance_multiset(*con,
                      "SELECT left_id, distance FROM sirius_knn_join("
                      "'apx_probe','vec','apx_corpus','vec', "
                      "search_mode => 'exact-gemm', metric => 'l2', k => 5);");

  // n_probes == n_clusters wants every cluster, so nothing is skipped and the approximate path
  // must reproduce the exhaustive answer. This is what catches a mistake in the neighbour-id
  // base, which pruning would otherwise hide as a plausible-looking wrong answer.
  auto const before    = sirius::test::get_vector_join_prune_stats(*con);
  auto const probe_all = distance_multiset(*con,
                                           "SELECT left_id, distance FROM sirius_knn_join("
                                           "'apx_probe','vec','apx_corpus','vec', "
                                           "search_mode => 'approx', metric => 'l2', k => 5, "
                                           "clustering => 'apx_c', cluster_column => 'cluster_id', "
                                           "n_probes => " +
                                             std::to_string(n_clusters) + ");");

  REQUIRE(probe_all == exhaustive);

  // The other half of the gate. Reproducing the exhaustive answer is only evidence of a correct
  // neighbour-id base if this run really did visit every cluster -- and it is the same counter
  // the pruning test asserts is small, so asserting it is maximal here is what stops that test
  // passing on a counter that is simply always zero.
  auto const after            = sirius::test::get_vector_join_prune_stats(*con);
  auto const scored           = after.pairs_scored - before.pairs_scored;
  auto const exhaustive_pairs = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive_pairs > 0);
  CHECK(scored == exhaustive_pairs);
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx returns fewer, still-valid neighbours when pruning",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apy", 8);

  // One cluster per probe row prunes hard. Recall is expected to drop -- that is the trade --
  // but every pair returned must still be a real corpus row, and every probe row must be
  // answered, which is what separates "approximate" from "broken".
  auto const before = sirius::test::get_vector_join_prune_stats(*con);
  auto const pruned = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'apy_probe','vec','apy_corpus','vec', "
                              "search_mode => 'approx', metric => 'l2', k => 5, "
                              "clustering => 'apy_c', cluster_column => 'cluster_id', "
                              "n_probes => 1);");
  CHECK(pruned.size() == 200 * 5);

  // What "approximate" has to mean. Every assertion below this one holds just as well for a run
  // that scored the whole corpus and called itself pruned -- which is exactly what the
  // chunk-granularity version did while passing this test. One cluster of eight is ~12.5% of the
  // corpus; the bound is loose because k-means does not make clusters equal.
  auto const after      = sirius::test::get_vector_join_prune_stats(*con);
  auto const scored     = after.pairs_scored - before.pairs_scored;
  auto const exhaustive = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive > 0);
  CHECK(scored * 4 < exhaustive);

  // Counted here rather than in SQL: an aggregate directly over the join falls back to the CPU,
  // which cannot execute the rewrite target at all.
  std::set<std::string> answered;
  std::set<std::string> neighbours;
  for (auto const& row : pruned) {
    answered.insert(row.at(0));
    neighbours.insert(row.at(1));
  }
  // Every probe row is answered...
  CHECK(answered.size() == 200);

  // ...and every neighbour is a real corpus row, which is what a mis-based neighbour id would
  // break: pruning changes which rows come back, never whether they exist.
  auto const corpus_ids = ok_rows(*con, "SELECT id FROM apy_corpus;");
  std::set<std::string> valid;
  for (auto const& row : corpus_ids) {
    valid.insert(row.at(0));
  }
  for (auto const& n : neighbours) {
    CHECK(valid.count(n) == 1);
  }
}

// -----------------------------------------------------------------------------
// The same two gates against a corpus of more than one pinned chunk.
//
// Every clustered test above uses a corpus small enough to arrive as one chunk, and
// that is the one shape in which the clustered path cannot be wrong about chunks: it
// staged chunk 0 and treated a cluster's row range as an offset into it, which is
// correct for one chunk and a read past the end of it for two. Pinned chunks follow
// DuckDB row groups (122,880 rows), so the row count below is the whole difference --
// the clustering, the queries and the assertions are the ones already used above.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx spans more than one corpus chunk",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  constexpr int n_clusters  = 8;
  constexpr int corpus_rows = 300000;  // > 2 row groups
  create_clustered_join(*this, *con, "apm", n_clusters, corpus_rows);

  auto const exhaustive =
    distance_multiset(*con,
                      "SELECT left_id, distance FROM sirius_knn_join("
                      "'apm_probe','vec','apm_corpus','vec', "
                      "search_mode => 'exact-gemm', metric => 'l2', k => 5);");

  auto const before    = sirius::test::get_vector_join_prune_stats(*con);
  auto const probe_all = distance_multiset(*con,
                                           "SELECT left_id, distance FROM sirius_knn_join("
                                           "'apm_probe','vec','apm_corpus','vec', "
                                           "search_mode => 'approx', metric => 'l2', k => 5, "
                                           "clustering => 'apm_c', cluster_column => 'cluster_id', "
                                           "n_probes => " +
                                             std::to_string(n_clusters) + ");");

  REQUIRE(probe_all == exhaustive);

  // The premise of this test, asserted rather than assumed: if DuckDB's row-group size ever
  // changes, or the generator stops producing 300,000 rows, this silently becomes another
  // single-chunk test and stops covering the thing it exists for. The counter accumulates one
  // corpus-chunk count per probe batch, and 200 probe rows are a single batch, so the delta is
  // the corpus's chunk count.
  auto const after = sirius::test::get_vector_join_prune_stats(*con);
  CHECK(after.chunks_available - before.chunks_available > 1);

  // Pruning across chunks is where a neighbour id is most easily mis-based: an id is local to
  // the slice it came from, and turning it back into a corpus row now takes the chunk's base
  // as well as the slice's own start. A wrong base still returns k plausible ids per probe
  // row, so what catches it is that every id must name a real corpus row.
  auto const pruned = ok_rows(*con,
                              "SELECT left_id, right_id FROM sirius_knn_join("
                              "'apm_probe','vec','apm_corpus','vec', "
                              "search_mode => 'approx', metric => 'l2', k => 5, "
                              "clustering => 'apm_c', cluster_column => 'cluster_id', "
                              "n_probes => 1);");
  CHECK(pruned.size() == 200 * 5);

  std::set<std::string> answered;
  std::set<std::string> neighbours;
  for (auto const& row : pruned) {
    answered.insert(row.at(0));
    neighbours.insert(row.at(1));
  }
  CHECK(answered.size() == 200);

  auto const corpus_ids = ok_rows(*con, "SELECT id FROM apm_corpus;");
  std::set<std::string> valid;
  for (auto const& row : corpus_ids) {
    valid.insert(row.at(0));
  }
  for (auto const& n : neighbours) {
    CHECK(valid.count(n) == 1);
  }
}

// -----------------------------------------------------------------------------
// A corpus that is not stored in cluster order is refused, not answered.
//
// The join reads each chunk's labels as a run per cluster. An unordered corpus makes
// that reading wrong rather than merely slow -- a cluster would be a partial range,
// and the rows outside it would never be scored while the answer still looked whole.
//
// The refusal is thrown at execution, not at bind, because it needs the corpus labels
// on the device. If DuckDB turns that into a CPU fallback the query still fails, but
// with the GPU-only-TVF message instead of this one -- in which case it is the needle
// that is wrong here, not the refusal.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx refuses a corpus not in cluster order",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apu", 8);

  // Same rows, same clustering, only the storage order differs.
  run_ok("CREATE TABLE apu_shuffled AS SELECT * FROM apu_corpus ORDER BY id;");
  run_ok("CHECKPOINT;");
  run_ok("SELECT * FROM pin_table(name => 'apu_shuffled', tier => 'gpu', format => 'duckdb');");

  expect_error(*con,
               "SELECT left_id, distance FROM sirius_knn_join("
               "'apu_probe','vec','apu_shuffled','vec', "
               "search_mode => 'approx', metric => 'l2', k => 5, "
               "clustering => 'apu_c', cluster_column => 'cluster_id', n_probes => 2);",
               "not stored in cluster order");
}

// -----------------------------------------------------------------------------
// The approximate join no longer needs the corpus pinned.
//
// Clustering used to refuse build_source => 'scan' outright, so an approximate join
// meant materializing a cluster-ordered copy AND pinning it -- and a pin of a column
// subset makes that table's other columns unusable in the same query. The cluster ids
// now ride along as a column of the corpus scan, in the same batches the fold
// searches, which is the only way they can mean the same row order the neighbour ids
// are resolved against.
//
// Verified against the pin path rather than against a recomputed oracle: the two must
// answer identically, which is what isolates "where the bytes came from" from "what
// the search did". Compared on distances because the build side's batch order is a
// race -- any order is correct as long as every stage uses the same one, but two
// orders break ties between equidistant rows differently.
// -----------------------------------------------------------------------------
TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join approx takes the corpus from a child scan",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_clustered_join(*this, *con, "apb", 8);

  auto const pinned = distance_multiset(*con,
                                        "SELECT left_id, distance FROM sirius_knn_join("
                                        "'apb_probe','vec','apb_corpus','vec', "
                                        "search_mode => 'approx', metric => 'l2', k => 5, "
                                        "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                        "n_probes => 2, right_output_columns => ['id']);");

  auto const before  = sirius::test::get_vector_join_prune_stats(*con);
  auto const scanned = distance_multiset(*con,
                                         "SELECT left_id, distance FROM sirius_knn_join("
                                         "'apb_probe','vec','apb_corpus','vec', "
                                         "search_mode => 'approx', metric => 'l2', k => 5, "
                                         "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                         "n_probes => 2, right_output_columns => ['id'], "
                                         "build_source => 'scan');");
  auto const after   = sirius::test::get_vector_join_prune_stats(*con);

  REQUIRE(scanned == pinned);

  // Agreeing with the pin path is not enough on its own: an unpruned build-phase run would
  // agree too, and more of the time. Two clusters of eight is ~25% of the corpus.
  auto const scored     = after.pairs_scored - before.pairs_scored;
  auto const exhaustive = after.pairs_exhaustive - before.pairs_exhaustive;
  CHECK(exhaustive > 0);
  CHECK(scored * 2 < exhaustive);

  // The point of the change. Nothing above proves the pin was unused -- it was still there.
  run_ok("SELECT * FROM unpin_table('apb_corpus');");
  auto const unpinned = distance_multiset(*con,
                                          "SELECT left_id, distance FROM sirius_knn_join("
                                          "'apb_probe','vec','apb_corpus','vec', "
                                          "search_mode => 'approx', metric => 'l2', k => 5, "
                                          "clustering => 'apb_c', cluster_column => 'cluster_id', "
                                          "n_probes => 2, right_output_columns => ['id'], "
                                          "build_source => 'scan');");
  REQUIRE(unpinned == pinned);

  // The cluster ids are a column of that scan, so a cluster_column the catalog does not have
  // has to be caught at bind: the plan generator would throw instead, and a throw there reads
  // as "this query cannot run on the GPU" and is reported as something unrelated.
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apb_probe','vec','apb_corpus','vec', "
               "search_mode => 'approx', metric => 'l2', k => 5, clustering => 'apb_c', "
               "cluster_column => 'not_a_column', build_source => 'scan');",
               "not found in table");
}

TEST_CASE_METHOD(KMeansFixture,
                 "sirius_knn_join rejects approx without a usable clustering",
                 "[integration][gpu_execution][array][vss][kmeans][approx]")
{
  create_two_group_table(*this, "apz_corpus");

  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2);",
               "needs clustering =>");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2, clustering => 'nope');",
               "requires cluster_column =>");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "k => 2, clustering => 'nope', cluster_column => 'c');",
               "only applies under search_mode");
  expect_error(*con,
               "SELECT * FROM sirius_knn_join('apz_corpus','vec','apz_corpus','vec', "
               "search_mode => 'approx', k => 2, clustering => 'nope', "
               "cluster_column => 'cluster_id');",
               "no clustering named");
}
