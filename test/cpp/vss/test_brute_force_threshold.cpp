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

// test
#include <catch.hpp>

// sirius
#include <cuda/vss/brute_force_search.hpp>
#include <cuda/vss/brute_force_threshold.hpp>
#include <cuda/vss/cudf_raft_interop.hpp>

#include <vss/bound_gemm.hpp>
#include <vss/scoped_device_resource.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// raft / rmm / cuvs
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cuvs/distance/distance.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::vss::brute_force_threshold;
using sirius::vss::list_column_as_dataset_view;
using sirius::vss::threshold_join_result;
using Metric = cuvs::distance::DistanceType;

// Build a Sirius-style ARRAY<FLOAT>[dim] column (cudf LIST with a contiguous,
// uniform FLOAT32 values child) so list_column_as_dataset_view can wrap it.
std::unique_ptr<cudf::column> make_fixed_size_float_list(std::vector<float> const& values,
                                                         cudf::size_type n_rows,
                                                         cudf::size_type dim)
{
  auto child = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n_rows * dim, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(child->mutable_view().data<float>(),
             values.data(),
             sizeof(float) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<int32_t> offsets(n_rows + 1);
  for (cudf::size_type i = 0; i <= n_rows; ++i) {
    offsets[i] = i * dim;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  return cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), 0, rmm::device_buffer{});
}

template <typename T>
std::vector<T> to_host(cudf::column_view const& col)
{
  std::vector<T> out(col.size());
  if (!out.empty()) {
    cudaMemcpy(out.data(), col.data<T>(), sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
  }
  return out;
}

// The result is an unordered ragged edge list; fold it into query -> sorted
// (neighbor, distance) so tests can assert on it deterministically.
using edge_map = std::map<int64_t, std::map<int64_t, float>>;

edge_map collect_edges(threshold_join_result const& r)
{
  auto const q = to_host<int64_t>(r.query_rows->view());
  auto const n = to_host<int64_t>(r.neighbors->view());
  auto const d = to_host<float>(r.distances->view());
  REQUIRE(q.size() == static_cast<std::size_t>(r.n_edges));
  REQUIRE(n.size() == q.size());
  REQUIRE(d.size() == q.size());

  edge_map edges;
  for (std::size_t i = 0; i < q.size(); ++i) {
    edges[q[i]][n[i]] = d[i];
  }
  return edges;
}

}  // namespace

TEST_CASE("brute_force_threshold keeps every L2 neighbor within the radius", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // Dataset row i is [i, i, i]; the single query is the origin, so the distance
  // to row i is i * sqrt(3). eps = 4.0 keeps rows 0, 1, 2 (0, 1.73, 3.46) and
  // drops row 3 (5.196) onward.
  constexpr cudf::size_type n_rows = 8;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> data(n_rows * dim);
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    data[i * dim + 0] = static_cast<float>(i);
    data[i * dim + 1] = static_cast<float>(i);
    data[i * dim + 2] = static_cast<float>(i);
  }
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const q{0.0f, 0.0f, 0.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  // Same expected result whether L2 is computed unexpanded (no GEMM) or expanded (GEMM).
  auto metric = Metric::L2SqrtUnexpanded;
  SECTION("L2 unexpanded (no GEMM)") { metric = Metric::L2SqrtUnexpanded; }
  SECTION("L2 expanded (GEMM)") { metric = Metric::L2SqrtExpanded; }

  auto r = brute_force_threshold(res, dataset_view, query_view, /*eps=*/4.0f, metric, mr);
  stream.sync();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 3);
  REQUIRE(edges.size() == 1);
  auto const& q0 = edges.at(0);
  REQUIRE(q0.size() == 3);
  REQUIRE(q0.count(0) == 1);
  REQUIRE(q0.count(1) == 1);
  REQUIRE(q0.count(2) == 1);
  REQUIRE(q0.at(0) == Approx(0.0f).margin(1e-3));
  REQUIRE(q0.at(1) == Approx(std::sqrt(3.0f)).margin(1e-3));
  REQUIRE(q0.at(2) == Approx(2.0f * std::sqrt(3.0f)).margin(1e-3));
}

TEST_CASE("brute_force_threshold cosine radius ignores magnitude", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // Row 0 aligned with the query, row 1 at 45 deg, row 2 aligned but large.
  // Cosine distance: rows 0 and 2 are 0 (magnitude ignored), row 1 is ~0.293.
  // A 0.1 cutoff keeps the two aligned rows and drops the off-axis one.
  constexpr cudf::size_type dim    = 2;
  constexpr cudf::size_type n_rows = 3;
  std::vector<float> const data{1.0f, 0.0f, 1.0f, 1.0f, 9.0f, 0.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const q{1.0f, 0.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  auto r =
    brute_force_threshold(res, dataset_view, query_view, /*eps=*/0.1f, Metric::CosineExpanded, mr);
  stream.sync();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 2);
  auto const& q0 = edges.at(0);
  REQUIRE(q0.count(0) == 1);
  REQUIRE(q0.count(2) == 1);
  REQUIRE(q0.count(1) == 0);
  REQUIRE(q0.at(0) == Approx(0.0f).margin(1e-4));
  REQUIRE(q0.at(2) == Approx(0.0f).margin(1e-4));
}

TEST_CASE("brute_force_threshold emits a ragged, per-query-variable edge list", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // 1-D points 0..5. Query 1.0 has neighbors {0,1,2} within 1.5; query 5.0 has
  // {4,5}. Different counts per query exercises the ragged output.
  constexpr cudf::size_type dim    = 1;
  constexpr cudf::size_type n_rows = 6;
  std::vector<float> const data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const queries{1.0f, 5.0f};
  auto query_col  = make_fixed_size_float_list(queries, 2, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  // Force tiny tiles too, so the same result must come out of the tiled append path.
  std::size_t tile_rows = 0;
  std::size_t tile_cols = 0;
  SECTION("auto tile size") {}
  SECTION("forced small tiles (1x2)")
  {
    tile_rows = 1;
    tile_cols = 2;
  }

  auto r = brute_force_threshold(res,
                                 dataset_view,
                                 query_view,
                                 /*eps=*/1.5f,
                                 Metric::L2SqrtExpanded,
                                 mr,
                                 tile_rows,
                                 tile_cols);
  stream.sync();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 5);
  REQUIRE(edges.at(0).size() == 3);  // query 1.0 -> {0,1,2}
  REQUIRE(edges.at(0).count(0) == 1);
  REQUIRE(edges.at(0).count(1) == 1);
  REQUIRE(edges.at(0).count(2) == 1);
  REQUIRE(edges.at(1).size() == 2);  // query 5.0 -> {4,5}
  REQUIRE(edges.at(1).count(4) == 1);
  REQUIRE(edges.at(1).count(5) == 1);
}

TEST_CASE("brute_force_threshold returns no edges when nothing is within the radius", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  constexpr cudf::size_type dim    = 1;
  constexpr cudf::size_type n_rows = 3;
  std::vector<float> const data{0.0f, 1.0f, 2.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const q{100.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  auto r =
    brute_force_threshold(res, dataset_view, query_view, /*eps=*/1.0f, Metric::L2SqrtExpanded, mr);
  stream.sync();

  REQUIRE(r.n_edges == 0);
  REQUIRE(r.query_rows->size() == 0);
  REQUIRE(r.neighbors->size() == 0);
  REQUIRE(r.distances->size() == 0);
}

namespace {

using host_rows = std::vector<float>;

std::unique_ptr<cudf::column> device_rows(host_rows const& rows, std::int64_t d)
{
  return make_fixed_size_float_list(
    rows, static_cast<cudf::size_type>(rows.size() / d), static_cast<cudf::size_type>(d));
}

/// sum_j (q_j - x_j)^2 over the rows as stored, in FP64: what an FP32 sum approximates, in any
/// order.
double true_squared(float const* q, float const* x, std::int64_t d)
{
  double s = 0;
  for (std::int64_t j = 0; j < d; ++j) {
    auto const e = static_cast<double>(q[j]) - static_cast<double>(x[j]);
    s += e * e;
  }
  return s;
}

/// Every pair against the oracle: inside the radius by more than the differences' own rounding it
/// must be there, outside by as much it must not, and a reported distance must be within rounding
/// of the true one. Pairs within rounding of the radius may go either way. Rounding includes what
/// a sum of squares under FP32's normal range loses, 2^-126 a term at most.
void check_against_oracle(threshold_join_result const& r,
                          host_rows const& queries,
                          host_rows const& dataset,
                          std::int64_t d,
                          double squared_radius,
                          bool sqrt_distance)
{
  auto const m = static_cast<std::int64_t>(queries.size()) / d;
  auto const n = static_cast<std::int64_t>(dataset.size()) / d;
  auto const u = std::ldexp(1.0, -24);
  // Twice the bound of any order of summation, for a sum's own rounding and a square root's.
  auto const rel = 2.0 * static_cast<double>(d + 8) * u;
  std::map<std::pair<std::int64_t, std::int64_t>, float> got;
  if (r.n_edges > 0) {
    auto const q = to_host<int64_t>(r.query_rows->view());
    auto const c = to_host<int64_t>(r.neighbors->view());
    auto const v = to_host<float>(r.distances->view());
    for (std::size_t i = 0; i < q.size(); ++i) {
      got[{q[i], c[i]}] = v[i];
    }
    REQUIRE(got.size() == q.size());
  }
  std::int64_t missed = 0, extra = 0, wrong_distance = 0;
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      auto const truth = true_squared(queries.data() + i * d, dataset.data() + j * d, d);
      auto const it    = got.find({i, j});
      if (it == got.end()) {
        missed += truth < squared_radius * (1.0 - rel) - 1e-37;
        continue;
      }
      extra += truth > squared_radius * (1.0 + rel) + 1e-37;
      auto const reported = sqrt_distance ? static_cast<double>(it->second) * it->second
                                          : static_cast<double>(it->second);
      wrong_distance += std::fabs(reported - truth) > rel * truth + 1e-37;
    }
  }
  CHECK(missed == 0);
  CHECK(extra == 0);
  CHECK(wrong_distance == 0);
}

struct screen_run {
  threshold_join_result result;
  bool worthwhile;
};

screen_run run_screen(host_rows const& queries,
                      host_rows const& dataset,
                      std::int64_t d,
                      float eps,
                      Metric metric,
                      raft::device_resources const& res,
                      std::unique_ptr<cudf::column>& q_col,
                      std::unique_ptr<cudf::column>& x_col)
{
  q_col           = device_rows(queries, d);
  x_col           = device_rows(dataset, d);
  auto q          = list_column_as_dataset_view(q_col->view(), d);
  auto x          = list_column_as_dataset_view(x_col->view(), d);
  bool worthwhile = true;
  auto r          = sirius::vss::screened_threshold(
    res, x, q, eps, metric, cudf::get_current_device_resource_ref(), false, &worthwhile);
  raft::resource::sync_stream(res);
  return {std::move(r), worthwhile};
}

/// Rows of a Gaussian cloud around (centre, ..., centre).
host_rows cloud(std::int64_t rows, std::int64_t d, float centre, float spread, unsigned seed)
{
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.f, spread);
  host_rows out(static_cast<std::size_t>(rows * d));
  for (auto& v : out) {
    v = centre + normal(gen);
  }
  return out;
}

host_rows unit_rows(host_rows rows, std::int64_t d)
{
  for (std::size_t i = 0; i < rows.size(); i += static_cast<std::size_t>(d)) {
    double s = 0;
    for (std::int64_t j = 0; j < d; ++j) {
      s += static_cast<double>(rows[i + j]) * rows[i + j];
    }
    auto const inv = 1.0 / std::sqrt(s);
    for (std::int64_t j = 0; j < d; ++j) {
      rows[i + j] = static_cast<float>(rows[i + j] * inv);
    }
  }
  return rows;
}

}  // namespace

TEST_CASE("screened_threshold answers as the differences do, at the radius", "[vss]")
{
  raft::device_resources res;
  struct shape {
    std::int64_t d;
    float centre, spread;
    bool screens;
  };
  auto const sh = GENERATE(shape{3, 49800.f, 40.f, false},
                           shape{3, 2000.f, 0.5f, false},
                           shape{96, 49800.f, 4.f, false},
                           shape{96, 0.f, 1.f, true},
                           shape{1152, 0.f, 1.f, true},
                           shape{128, 1e6f, 300.f, false});
  auto const m  = std::int64_t{600};
  auto const n  = std::int64_t{1500};
  auto const q  = cloud(m, sh.d, sh.centre, sh.spread, 1);
  auto const x  = cloud(n, sh.d, sh.centre, sh.spread, 2);
  // The radius on a real pair, so some pair sits on the boundary to the last bit; about one pair in
  // 4,096 is inside it, under what the screen gives up at.
  std::vector<double> squared;
  squared.reserve(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      squared.push_back(true_squared(q.data() + i * sh.d, x.data() + j * sh.d, sh.d));
    }
  }
  auto const rank = squared.size() / 4096;
  std::nth_element(squared.begin(), squared.begin() + rank, squared.end());
  auto const eps = static_cast<float>(std::sqrt(squared[rank]));
  std::unique_ptr<cudf::column> qc, xc;
  INFO("d " << sh.d << " centre " << sh.centre);
  auto sq = run_screen(q, x, sh.d, eps, Metric::L2SqrtUnexpanded, res, qc, xc);
  CHECK(sq.worthwhile == sh.screens);
  check_against_oracle(sq.result, q, x, sh.d, static_cast<double>(eps) * eps, true);
  auto l2 = run_screen(q, x, sh.d, eps * eps, Metric::L2Unexpanded, res, qc, xc);
  check_against_oracle(l2.result, q, x, sh.d, static_cast<double>(eps * eps), false);
}

TEST_CASE("screened_threshold keeps cosine near-duplicates the GEMM's form rounds to zero", "[vss]")
{
  raft::device_resources res;
  // Unit rows, each corpus row a probe moved by about 1e-4 in a few components: squared distances
  // near 1e-7, cosine distances under the rounding of a dot product.
  auto const d = std::int64_t{128};
  auto const m = std::int64_t{600};
  auto q       = unit_rows(cloud(m, d, 0.f, 1.f, 3), d);
  auto x       = q;
  std::mt19937 gen(4);
  std::normal_distribution<float> tiny(0.f, 1e-4f);
  for (std::size_t i = 0; i < x.size(); i += static_cast<std::size_t>(d)) {
    for (std::int64_t j = 0; j < 8; ++j) {
      x[i + j] += tiny(gen);
    }
  }
  x = unit_rows(std::move(x), d);
  std::vector<double> near;
  for (std::int64_t i = 0; i < m; ++i) {
    near.push_back(true_squared(q.data() + i * d, x.data() + i * d, d));
  }
  std::sort(near.begin(), near.end());
  auto const eps = static_cast<float>(near[near.size() / 2]);
  REQUIRE(eps < 1e-6f);
  std::unique_ptr<cudf::column> qc, xc;
  auto r = run_screen(q, x, d, eps, Metric::L2Unexpanded, res, qc, xc);
  check_against_oracle(r.result, q, x, d, eps, false);
  CHECK(r.result.n_edges >= m / 3);
  CHECK(r.result.n_edges <= 2 * m / 3);
}

TEST_CASE("screened_threshold leaves to the differences what it cannot screen", "[vss]")
{
  raft::device_resources res;
  auto const d = std::int64_t{4};
  auto const m = std::int64_t{600};
  auto const n = std::int64_t{2000};
  std::unique_ptr<cudf::column> qc, xc;

  SECTION("a norm past FP32's range, with differences inside it")
  {
    // Components near 3e19 square to 9e38, past FP32's largest; differences of 1e15 square to 1e30.
    auto q = cloud(m, d, 0.f, 1e15f, 5);
    auto x = cloud(n, d, 0.f, 1e15f, 6);
    for (auto* rows : {&q, &x}) {
      for (auto& v : *rows) {
        v += 3e19f;
      }
    }
    auto r = run_screen(q, x, d, 1.5e15f, Metric::L2SqrtUnexpanded, res, qc, xc);
    CHECK_FALSE(r.worthwhile);
    check_against_oracle(r.result, q, x, d, 1.5e15 * 1.5e15, true);
    CHECK(r.result.n_edges > 0);
  }
  SECTION("a NaN in a row joins nothing and takes no other row with it")
  {
    auto q         = cloud(m, d, 0.f, 1.f, 7);
    auto x         = cloud(n, d, 0.f, 1.f, 8);
    auto const nan = std::numeric_limits<float>::quiet_NaN();
    auto q_clean   = q;
    q[3 * d + 1]   = nan;
    x[5 * d + 2]   = nan;
    auto r         = run_screen(q, x, d, 0.5f, Metric::L2SqrtUnexpanded, res, qc, xc);
    CHECK_FALSE(r.worthwhile);
    auto const rows = to_host<int64_t>(r.result.query_rows->view());
    auto const cols = to_host<int64_t>(r.result.neighbors->view());
    for (std::size_t i = 0; i < rows.size(); ++i) {
      CHECK(rows[i] != 3);
      CHECK(cols[i] != 5);
    }
    std::int64_t expected = 0;
    for (std::int64_t i = 0; i < m; ++i) {
      for (std::int64_t j = 0; j < n; ++j) {
        if (i == 3 || j == 5) { continue; }
        auto const t = true_squared(q.data() + i * d, x.data() + j * d, d);
        expected += t <= 0.25 * (1.0 - 1e-5);
      }
    }
    CHECK(r.result.n_edges >= expected);
  }
  SECTION("a radius past the rows' spread passes everything, and stays right")
  {
    auto const q = cloud(m, d, 49800.f, 1.f, 9);
    auto const x = cloud(n, d, 49800.f, 1.f, 10);
    auto r       = run_screen(q, x, d, 3.f, Metric::L2SqrtUnexpanded, res, qc, xc);
    CHECK_FALSE(r.worthwhile);
    check_against_oracle(r.result, q, x, d, 9.0, true);
  }
  SECTION("a tile past the pass limit hands its rows on, after one that screened")
  {
    // 200,000 rows make a tile of 671 probe rows. The first tile's rows reach nothing; each row of
    // the second from 700 on reaches 300 rows, past the limit of about 1 pair in 1,024 of a tile.
    // Its score tile alone is more than the small pool an earlier test can leave current holds.
    static rmm::mr::cuda_memory_resource device;
    sirius::vss::scoped_current_device_resource const from_device{device};
    auto const big = std::int64_t{200000};
    auto x         = cloud(big, d, 0.f, 0.001f, 11);
    for (std::int64_t j = 300; j < big; ++j) {
      x[static_cast<std::size_t>(j * d)] += 100.f + static_cast<float>(j);
    }
    auto q = cloud(1200, d, 0.f, 0.001f, 12);
    for (std::int64_t i = 0; i < 700; ++i) {
      q[static_cast<std::size_t>(i * d + 1)] += 1e6f;
    }
    auto r = run_screen(q, x, d, 1.f, Metric::L2SqrtUnexpanded, res, qc, xc);
    CHECK_FALSE(r.worthwhile);
    CHECK(r.result.n_edges == 500 * 300);
    auto const rows = to_host<int64_t>(r.result.query_rows->view());
    CHECK(*std::min_element(rows.begin(), rows.end()) == 700);
    CHECK(*std::max_element(rows.begin(), rows.end()) == 1199);
  }
}

TEST_CASE("screened_threshold counts what it would emit", "[vss]")
{
  raft::device_resources res;
  auto const d = std::int64_t{32};
  auto const q = cloud(600, d, 0.f, 1.f, 13);
  auto const x = cloud(3000, d, 0.f, 1.f, 14);
  auto qc      = device_rows(q, d);
  auto xc      = device_rows(x, d);
  auto qv      = list_column_as_dataset_view(qc->view(), d);
  auto xv      = list_column_as_dataset_view(xc->view(), d);
  auto mr      = cudf::get_current_device_resource_ref();
  auto const full =
    sirius::vss::screened_threshold(res, xv, qv, 6.f, Metric::L2SqrtUnexpanded, mr, false);
  auto const count =
    sirius::vss::screened_threshold(res, xv, qv, 6.f, Metric::L2SqrtUnexpanded, mr, true);
  raft::resource::sync_stream(res);
  CHECK(count.n_edges == full.n_edges);
}

TEST_CASE("the GEMM's score stays within the screen's bound of the true distance", "[vss]")
{
  // The bound's second step, on this GPU's cuBLAS: gemm_threshold over [-2x, |x|^2] reports s + Q,
  // which step 2 of the argument above screened_threshold bounds around the true D by
  // 3.03 (d + 1) u (A + B)^2.
  raft::device_resources res;
  struct shape {
    std::int64_t d, m, n;
    float centre, spread;
  };
  auto const sh = GENERATE(shape{3, 600, 2000, 49800.f, 40.f},
                           shape{8, 600, 2000, 2000.f, 1.f},
                           shape{96, 600, 2000, 49800.f, 4.f},
                           shape{96, 1024, 4096, 1.f, 1.f},
                           shape{128, 600, 2000, 1e6f, 300.f},
                           shape{1024, 600, 1500, 0.f, 1.f},
                           shape{1152, 600, 1500, 0.f, 0.03f});
  auto const q  = cloud(sh.m, sh.d, sh.centre, sh.spread, 15);
  auto const x  = cloud(sh.n, sh.d, sh.centre, sh.spread, 16);
  auto qc       = device_rows(q, sh.d);
  auto xc       = device_rows(x, sh.d);
  auto all      = sirius::vss::gemm_threshold(res,
                                         list_column_as_dataset_view(xc->view(), sh.d),
                                         list_column_as_dataset_view(qc->view(), sh.d),
                                         1e30f,
                                         Metric::L2Expanded,
                                         cudf::get_current_device_resource_ref());
  raft::resource::sync_stream(res);
  REQUIRE(all.n_edges == sh.m * sh.n);
  auto const rows = to_host<int64_t>(all.query_rows->view());
  auto const cols = to_host<int64_t>(all.neighbors->view());
  auto const dist = to_host<float>(all.distances->view());
  auto const u    = std::ldexp(1.0, -24);
  double worst    = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto const* qr = q.data() + rows[i] * sh.d;
    auto const* xr = x.data() + cols[i] * sh.d;
    double a = 0, b = 0;
    for (std::int64_t j = 0; j < sh.d; ++j) {
      a += static_cast<double>(qr[j]) * qr[j];
      b += static_cast<double>(xr[j]) * xr[j];
    }
    auto const reach = std::sqrt(a) + std::sqrt(b);
    auto const bound = 3.03 * static_cast<double>(sh.d + 1) * u * reach * reach;
    // The report is clamped at 0 and rounded once more, so a distance of 0 only shows the side
    // below; the +u covers the addition that made it.
    auto const seen = std::fabs(static_cast<double>(dist[i]) - true_squared(qr, xr, sh.d));
    worst           = std::max(worst, seen / (bound + u * reach * reach));
  }
  INFO("d " << sh.d << " centre " << sh.centre << ": worst error / bound " << worst);
  CHECK(worst < 1.0);
  WARN("d " << sh.d << " centre " << sh.centre << ": worst error / bound " << worst);
}

namespace {

struct topk_fold {
  host_rows distances;
  std::vector<int64_t> ids;
  bool worthwhile{true};
};

/// The accumulator after screening @p dataset in chunks of @p chunk_rows, each chunk under the
/// running bound its predecessors left.
topk_fold run_topk_fold(host_rows const& queries,
                        host_rows const& dataset,
                        std::int64_t d,
                        std::int64_t k,
                        std::int64_t chunk_rows,
                        Metric metric,
                        raft::device_resources const& res)
{
  auto const m      = static_cast<std::int64_t>(queries.size()) / d;
  auto const n      = static_cast<std::int64_t>(dataset.size()) / d;
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const mr     = cudf::get_current_device_resource_ref();
  auto q_col        = device_rows(queries, d);
  auto q            = list_column_as_dataset_view(q_col->view(), d);
  rmm::device_uvector<float> acc_d(static_cast<std::size_t>(m * k), stream, mr);
  rmm::device_uvector<int64_t> acc_n(static_cast<std::size_t>(m * k), stream, mr);
  sirius::vss::fill_misses(acc_d.data(), acc_n.data(), m * k, stream);
  topk_fold out;
  for (std::int64_t base = 0; base < n; base += chunk_rows) {
    auto const rows = std::min(chunk_rows, n - base);
    host_rows const piece(dataset.begin() + base * d, dataset.begin() + (base + rows) * d);
    auto x_col = device_rows(piece, d);
    sirius::vss::screened_topk_fold(res,
                                    list_column_as_dataset_view(x_col->view(), d),
                                    q,
                                    std::min(k, rows),
                                    metric,
                                    {acc_d.data(), acc_n.data(), m, k, nullptr, base, nullptr},
                                    mr,
                                    &out.worthwhile);
    raft::resource::sync_stream(res);
  }
  out.distances.resize(acc_d.size());
  out.ids.resize(acc_n.size());
  cudaMemcpy(
    out.distances.data(), acc_d.data(), acc_d.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(out.ids.data(), acc_n.data(), acc_n.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
  return out;
}

/// Every row against the FP64 truth over the rows as stored: its k reported rows, by their true
/// distances, within the differences' rounding of the k nearest, and each reported distance within
/// that of its row's true distance.
void check_topk_against_oracle(topk_fold const& got,
                               std::vector<double> const& truth,
                               host_rows const& queries,
                               host_rows const& dataset,
                               std::int64_t d,
                               std::int64_t k,
                               bool sqrt_distance)
{
  auto const m            = static_cast<std::int64_t>(queries.size()) / d;
  auto const n            = static_cast<std::int64_t>(dataset.size()) / d;
  auto const rel          = 2.0 * static_cast<double>(d + 8) * std::ldexp(1.0, -24);
  std::int64_t wrong_rows = 0, wrong_distance = 0, misses = 0, unordered = 0;
  std::vector<double> row(static_cast<std::size_t>(n)), mine(static_cast<std::size_t>(k));
  for (std::int64_t i = 0; i < m; ++i) {
    std::copy_n(truth.begin() + i * n, n, row.begin());
    std::partial_sort(row.begin(), row.begin() + k, row.end());
    for (std::int64_t j = 0; j < k; ++j) {
      auto const id = got.ids[static_cast<std::size_t>(i * k + j)];
      if (id < 0 || id >= n) {
        ++misses;
        mine[static_cast<std::size_t>(j)] = std::numeric_limits<double>::infinity();
        continue;
      }
      auto const t                      = truth[static_cast<std::size_t>(i * n + id)];
      mine[static_cast<std::size_t>(j)] = t;
      auto const v = static_cast<double>(got.distances[static_cast<std::size_t>(i * k + j)]);
      wrong_distance += std::fabs((sqrt_distance ? v * v : v) - t) > rel * t + 1e-37;
      if (j > 0) {
        unordered += got.distances[static_cast<std::size_t>(i * k + j)] <
                     got.distances[static_cast<std::size_t>(i * k + j - 1)];
      }
    }
    std::sort(mine.begin(), mine.end());
    for (std::int64_t j = 0; j < k; ++j) {
      if (mine[static_cast<std::size_t>(j)] >
          row[static_cast<std::size_t>(j)] * (1.0 + rel) + 1e-37) {
        ++wrong_rows;
        break;
      }
    }
  }
  CHECK(misses == 0);
  CHECK(wrong_rows == 0);
  CHECK(wrong_distance == 0);
  CHECK(unordered == 0);
}

std::vector<double> all_true_squared(host_rows const& queries,
                                     host_rows const& dataset,
                                     std::int64_t d)
{
  auto const m = static_cast<std::int64_t>(queries.size()) / d;
  auto const n = static_cast<std::int64_t>(dataset.size()) / d;
  std::vector<double> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      out[static_cast<std::size_t>(i * n + j)] =
        true_squared(queries.data() + i * d, dataset.data() + j * d, d);
    }
  }
  return out;
}

struct hard_table {
  std::string name;
  std::int64_t d;
  host_rows q, x;
  bool screens;  ///< whether the screen is expected to decide most rows
};

host_rows uniform(std::int64_t rows, std::int64_t d, float scale, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(0.f, scale);
  host_rows out(static_cast<std::size_t>(rows * d));
  for (auto& v : out) {
    v = u(gen);
  }
  return out;
}

host_rows lattice(std::int64_t rows, std::int64_t d, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> u(0, 2);
  host_rows out(static_cast<std::size_t>(rows * d));
  for (auto& v : out) {
    v = static_cast<float>(u(gen));
  }
  return out;
}

/// Tables that cancel, overflow, tie or underflow, with enough probe rows to reach the GEMM.
std::vector<hard_table> hard_tables()
{
  auto const m = std::int64_t{600};
  auto const n = std::int64_t{3000};
  std::vector<hard_table> t;
  t.push_back({"plain 96-d", 96, cloud(m, 96, 0.f, 1.f, 21), cloud(n, 96, 0.f, 1.f, 22), true});
  t.push_back({"plain 16-d", 16, cloud(m, 16, 0.f, 1.f, 23), cloud(n, 16, 0.f, 1.f, 24), true});
  t.push_back(
    {"partly cancelled", 16, cloud(m, 16, 100.f, 1.f, 25), cloud(n, 16, 100.f, 1.f, 26), true});
  t.push_back({"far from the origin, 3-d",
               3,
               cloud(m, 3, 49800.f, 40.f, 27),
               cloud(n, 3, 49800.f, 40.f, 28),
               false});
  t.push_back({"far from the origin, 96-d",
               96,
               cloud(m, 96, 49800.f, 4.f, 29),
               cloud(n, 96, 49800.f, 4.f, 30),
               false});
  t.push_back({"close together near 2,000",
               3,
               cloud(m, 3, 2000.f, 0.5f, 31),
               cloud(n, 3, 2000.f, 0.5f, 32),
               false});
  t.push_back(
    {"huge norm", 16, cloud(m, 16, 8000.f, 10.f, 33), cloud(n, 16, 8000.f, 10.f, 34), false});
  t.push_back({"half denormals", 16, uniform(m, 16, 5e-5f, 35), uniform(n, 16, 5e-5f, 36), true});
  t.push_back({"below half", 16, uniform(m, 16, 1e-9f, 37), uniform(n, 16, 1e-9f, 38), true});
  t.push_back(
    {"squares under FP32", 16, uniform(m, 16, 1e-21f, 39), uniform(n, 16, 1e-21f, 40), true});
  {
    auto q = uniform(m, 16, 3e-5f, 41);
    auto x = uniform(n, 16, 3e-5f, 42);
    for (std::size_t i = 0; i < q.size(); i += 16) {
      q[i] = 1.f;
    }
    for (std::size_t i = 0; i < x.size(); i += 16) {
      x[i] = 1.f;
    }
    t.push_back({"denormals beside a one", 16, std::move(q), std::move(x), false});
  }
  {
    auto q = cloud(m, 16, 0.f, 1.f, 43);
    auto x = cloud(n, 16, 0.f, 1.f, 44);
    for (std::size_t i = q.size() / 2; i < q.size(); ++i) {
      q[i] += 9000.f;
    }
    for (std::size_t i = x.size() / 2; i < x.size(); ++i) {
      x[i] += 9000.f;
    }
    t.push_back({"mixed scale", 16, std::move(q), std::move(x), false});
  }
  t.push_back({"lattice ties", 16, lattice(m, 16, 45), lattice(n, 16, 46), false});
  {
    auto x = cloud(n, 16, 0.f, 1.f, 47);
    for (std::size_t i = 16 * 5; i < x.size(); ++i) {
      x[i] = x[i % (16 * 5)];
    }
    t.push_back({"all duplicates", 16, cloud(m, 16, 0.f, 1.f, 48), std::move(x), false});
  }
  {
    // A tenth of the probe rows have more duplicates than the screen selects; the rest screen.
    auto q = cloud(m, 16, 0.f, 1.f, 49);
    auto x = cloud(n, 16, 0.f, 1.f, 50);
    for (std::int64_t c = 0; c < 60; ++c) {
      for (std::int64_t copy = 0; copy < 40; ++copy) {
        std::copy_n(q.begin() + c * 16, 16, x.begin() + (c * 40 + copy) * 16);
      }
    }
    t.push_back({"some rows duplicated", 16, std::move(q), std::move(x), true});
  }
  {
    auto q = cloud(m, 16, 0.f, 1.f, 51);
    auto x = cloud(n, 16, 0.f, 1.f, 52);
    std::fill_n(q.begin(), 16 * 50, 0.f);
    std::fill_n(x.begin(), 16 * 50, 0.f);
    t.push_back({"zero rows", 16, std::move(q), std::move(x), true});
  }
  {
    auto q = cloud(m, 4, 0.f, 1e15f, 53);
    auto x = cloud(n, 4, 0.f, 1e15f, 54);
    for (auto* rows : {&q, &x}) {
      for (auto& v : *rows) {
        v += 3e19f;
      }
    }
    t.push_back({"norm past FP32's range", 4, std::move(q), std::move(x), false});
  }
  return t;
}

}  // namespace

TEST_CASE("screened_topk_fold answers as the differences do on hard tables", "[vss]")
{
  raft::device_resources res;
  for (auto const& table : hard_tables()) {
    auto const truth = all_true_squared(table.q, table.x, table.d);
    auto const n     = static_cast<std::int64_t>(table.x.size()) / table.d;
    for (auto const k : {std::int64_t{1}, std::int64_t{10}}) {
      for (auto const chunk : {n, std::int64_t{700}}) {
        for (auto const metric : {Metric::L2Unexpanded, Metric::L2SqrtUnexpanded}) {
          INFO(table.name << ": k " << k << ", chunks of " << chunk << ", "
                          << (metric == Metric::L2Unexpanded ? "squared" : "Euclidean"));
          auto const got = run_topk_fold(table.q, table.x, table.d, k, chunk, metric, res);
          check_topk_against_oracle(
            got, truth, table.q, table.x, table.d, k, metric == Metric::L2SqrtUnexpanded);
          if (table.screens) { CHECK(got.worthwhile); }
          if (table.name == "norm past FP32's range") { CHECK_FALSE(got.worthwhile); }
        }
      }
    }
  }
}

TEST_CASE("screened_topk_fold takes chunks shorter than k and a k past the selection", "[vss]")
{
  raft::device_resources res;
  auto const d     = std::int64_t{16};
  auto const q     = cloud(600, d, 0.f, 1.f, 61);
  auto const x     = cloud(3000, d, 0.f, 1.f, 62);
  auto const truth = all_true_squared(q, x, d);
  SECTION("chunks of 7 rows, k 10: rows have no bound until the second chunk")
  {
    auto const got = run_topk_fold(q, x, d, 10, 7, Metric::L2Unexpanded, res);
    check_topk_against_oracle(got, truth, q, x, d, 10, false);
  }
  SECTION("k 200, past what a row needs at any chunk size")
  {
    auto const got = run_topk_fold(q, x, d, 200, 700, Metric::L2Unexpanded, res);
    check_topk_against_oracle(got, truth, q, x, d, 200, false);
  }
  SECTION("k = every row of the only chunk")
  {
    auto const small = host_rows(x.begin(), x.begin() + 40 * d);
    auto const got   = run_topk_fold(q, small, d, 40, 40, Metric::L2Unexpanded, res);
    check_topk_against_oracle(got, all_true_squared(q, small, d), q, small, d, 40, false);
  }
}

TEST_CASE("screened_topk_fold merges every row of a round past one grid of threads", "[vss]")
{
  // 3,000 rows make a tile of 44,739 probe rows, so at k 100 a round of four tiles holds about 18M
  // pairs: more than 65,535 blocks of 256 threads. A row past them must still reach the answer.
  static rmm::mr::cuda_memory_resource device;
  sirius::vss::scoped_current_device_resource const from_device{device};
  raft::device_resources res;
  auto const d = std::int64_t{4};
  auto const m = std::int64_t{200000};
  auto const n = std::int64_t{3000};
  auto const k = std::int64_t{100};
  auto const q = cloud(m, d, 0.f, 1.f, 71);
  auto const x = cloud(n, d, 0.f, 1.f, 72);
  auto qc      = device_rows(q, d);
  auto xc      = device_rows(x, d);
  auto qv      = list_column_as_dataset_view(qc->view(), d);
  auto xv      = list_column_as_dataset_view(xc->view(), d);
  auto mr      = cudf::get_current_device_resource_ref();
  auto const a = sirius::vss::exact_topk(res, xv, qv, k, Metric::L2Unexpanded, mr);
  raft::resource::sync_stream(res);
  auto const ids  = to_host<int64_t>(a.neighbors->view());
  auto const dist = to_host<float>(a.distances->view());
  REQUIRE(static_cast<std::int64_t>(ids.size()) == m * k);
  std::int64_t misses = 0;
  for (auto const id : ids) {
    misses += id < 0;
  }
  CHECK(misses == 0);
  auto const rel          = 2.0 * static_cast<double>(d + 8) * std::ldexp(1.0, -24);
  std::int64_t wrong_rows = 0;
  std::vector<double> row(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < m; i += 997) {
    for (std::int64_t j = 0; j < n; ++j) {
      row[static_cast<std::size_t>(j)] = true_squared(q.data() + i * d, x.data() + j * d, d);
    }
    std::nth_element(row.begin(), row.begin() + (k - 1), row.end());
    auto const kth = row[static_cast<std::size_t>(k - 1)];
    wrong_rows +=
      static_cast<double>(dist[static_cast<std::size_t>(i * k + k - 1)]) > kth * (1.0 + rel);
  }
  CHECK(wrong_rows == 0);
}

TEST_CASE("screened_threshold answers as the differences do on hard tables", "[vss]")
{
  raft::device_resources res;
  for (auto const& table : hard_tables()) {
    INFO(table.name);
    auto squared    = all_true_squared(table.q, table.x, table.d);
    auto const rank = squared.size() / 4096;
    std::nth_element(squared.begin(), squared.begin() + rank, squared.end());
    auto const eps = static_cast<float>(std::sqrt(squared[rank]));
    std::unique_ptr<cudf::column> qc, xc;
    auto const root =
      run_screen(table.q, table.x, table.d, eps, Metric::L2SqrtUnexpanded, res, qc, xc);
    check_against_oracle(
      root.result, table.q, table.x, table.d, static_cast<double>(eps) * eps, true);
    auto const plain =
      run_screen(table.q, table.x, table.d, eps * eps, Metric::L2Unexpanded, res, qc, xc);
    check_against_oracle(
      plain.result, table.q, table.x, table.d, static_cast<double>(eps * eps), false);
  }
}

TEST_CASE("exact_topk through the screen equals one through the differences", "[vss]")
{
  raft::device_resources res;
  auto const d = std::int64_t{32};
  auto const q = cloud(600, d, 0.f, 1.f, 63);
  auto const x = cloud(20000, d, 0.f, 1.f, 64);
  auto qc      = device_rows(q, d);
  auto xc      = device_rows(x, d);
  auto qv      = list_column_as_dataset_view(qc->view(), d);
  auto xv      = list_column_as_dataset_view(xc->view(), d);
  auto mr      = cudf::get_current_device_resource_ref();
  auto const a = sirius::vss::exact_topk(res, xv, qv, 10, Metric::L2Unexpanded, mr);
  auto const b = sirius::vss::brute_force_knn(res, xv, qv, 10, Metric::L2Unexpanded, mr);
  raft::resource::sync_stream(res);
  CHECK(a.k == 10);
  auto const ad = to_host<float>(a.distances->view());
  auto const bd = to_host<float>(b.distances->view());
  REQUIRE(ad.size() == bd.size());
  std::size_t apart = 0;
  for (std::size_t i = 0; i < ad.size(); ++i) {
    apart += std::fabs(ad[i] - bd[i]) > 1e-5f * std::max(ad[i], 1.f);
  }
  CHECK(apart == 0);
}
