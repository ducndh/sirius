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

#include <string>
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
  CHECK(scalar_i64(*con,
                   "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id < 200;") == 1);
  CHECK(scalar_i64(*con,
                   "SELECT count(DISTINCT cluster_id) FROM kmb_asg WHERE row_id >= 200;") == 1);
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
  auto const mismatches = scalar_i64(
    *con,
    "SELECT count(*) FROM kmf_corpus c JOIN kmf_asg a ON c.rowid = a.row_id WHERE c.id <> a.row_id;");
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
  expect_error(*con,
               "SELECT * FROM sirius_kmeans_assign('kmh_wide','vec','kmh_c');",
               "is over FLOAT[3]");
}
