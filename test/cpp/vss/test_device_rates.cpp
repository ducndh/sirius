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

#include <catch.hpp>
#include <vss/access_path_cost.hpp>
#include <vss/device_rates.hpp>

#include <cmath>
#include <limits>

using sirius::vss::access_path_cost;
using sirius::vss::device_rates;
using sirius::vss::device_scale;

TEST_CASE("measure_device_rates measures every resource of the current GPU", "[vss]")
{
  device_rates r;
  REQUIRE(sirius::vss::measure_device_rates(r));
  WARN("fp32 " << r.fp32 << " flop/s, f16 " << r.f16 << ", int8 " << r.int8 << " op/s, pcie "
               << r.pcie << " B/s, device " << r.hbm << " B/s");
  // Sanity bounds that any GPU this builds for clears and none exceeds by far.
  CHECK(r.fp32 > 1e12);
  CHECK(r.fp32 < 2e14);
  CHECK(r.f16 >= r.fp32 * 0.5);
  CHECK(r.int8 >= r.f16 * 0.5);
  CHECK(r.pcie > 1e9);
  CHECK(r.pcie < 1e12);
  CHECK(r.hbm > 5e10);
  CHECK(r.hbm < 2e13);
}

TEST_CASE("the access path cost model scales each rate by its own resource", "[vss]")
{
  access_path_cost const a5000{device_scale{}};
  CHECK(a5000.fp32_gemm == 8.5e12);
  CHECK(a5000.int8_gemm == 77e12);
  CHECK(a5000.pcie == 21e9);

  // Twice the tensor throughput, the same everything else: only the lists' compute side moves.
  access_path_cost const tensor2{device_scale{.f16 = 2, .int8 = 2}};
  CHECK(tensor2.int8_gemm == 2 * 77e12);
  CHECK(tensor2.f16_gemm == 2 * 41e12);
  CHECK(tensor2.fp32_gemm == 8.5e12);
  CHECK(tensor2.sweep0 == 0.125);

  sirius::vss::access_path_shape s;
  s.probe_rows           = 10000;
  s.corpus_rows          = 1e8;
  s.dim                  = 128;
  s.corpus_on_device     = false;
  s.list_bytes_per_value = 1;
  s.lists_on_device      = true;
  CHECK(tensor2.brute(s) == a5000.brute(s));
  CHECK(tensor2.lists(s) < a5000.lists(s));
}

TEST_CASE("the exact join's one list costs a read of the rows, not a clustering", "[vss]")
{
  access_path_cost const cost{device_scale{}};
  sirius::vss::access_path_shape s;
  s.probe_rows           = 1000;
  s.corpus_rows          = 1e7;
  s.dim                  = 96;
  s.corpus_on_device     = true;
  s.list_bytes_per_value = 1;
  s.lists_on_device      = true;
  s.n_clusters           = 1;

  // Deep10M pinned on the device, built as a statement builds it: 0.055 s measured (2026-10-21).
  CHECK(cost.codes(s) > 0.035);
  CHECK(cost.codes(s) < 0.075);
  // So 1,000 probe rows already repay it: build + search 0.05-0.1 s against brute force 0.32 s.
  CHECK(cost.codes(s) + cost.lists(s) < 0.5 * cost.brute(s));
  // A clustering of even 64 partitions costs far more to fit, assign and write.
  s.n_clusters = 64;
  CHECK(cost.build(s) > 10 * cost.codes(s));
  s.n_clusters = 1;

  // Cosine lists copy their unit rows to the host in FP32: 0.185 s measured.
  s.unit_rows            = true;
  s.list_bytes_per_value = 2;
  CHECK(cost.codes(s) > 0.15);
  CHECK(cost.codes(s) < 0.23);
  s.unit_rows            = false;
  s.list_bytes_per_value = 1;

  // SIFT100M pinned in host memory: two PCIe copies of the rows, 3.99 s measured.
  s.corpus_rows      = 1e8;
  s.dim              = 128;
  s.corpus_on_device = false;
  CHECK(cost.codes(s) > 3.4);
  CHECK(cost.codes(s) < 4.6);
  // Twice the PCIe bandwidth halves it; tensor throughput does not move it.
  access_path_cost const pcie2{device_scale{.pcie = 2}};
  access_path_cost const tensor2{device_scale{.f16 = 2, .int8 = 2}};
  CHECK(pcie2.codes(s) < 0.51 * cost.codes(s));
  CHECK(tensor2.codes(s) == cost.codes(s));
}

TEST_CASE("lists a statement builds for itself get more clusters for a larger probe side", "[vss]")
{
  using sirius::vss::choose_cluster_count;
  using sirius::vss::scaled_probes;
  access_path_cost const cost{device_scale{}};
  sirius::vss::access_path_shape s;
  s.corpus_rows          = 1e8;
  s.dim                  = 128;
  s.list_bytes_per_value = 1;
  s.lists_on_device      = true;

  // A few thousand probe rows cannot repay a larger build: sqrt(rows) clusters, at most 1,024.
  // c = 0.5 visits ceil(0.5 * sqrt(1024)) = 16 of them.
  s.probe_rows     = 1e4;
  auto const small = choose_cluster_count(cost, s, 0.5);
  CHECK(small.n_clusters == 1024);
  CHECK(small.probes == 16);

  // A self-join can, and its probes follow the square root of the count.
  s.probe_rows    = 1e8;
  auto const self = choose_cluster_count(cost, s, 0.5);
  CHECK(self.n_clusters > 1024);
  CHECK(self.n_clusters <= 16 * 1024);
  CHECK(self.probes == scaled_probes(0.5, self.n_clusters));

  std::int64_t last = 0;
  for (double const m : {1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}) {
    s.probe_rows = m;
    auto const c = choose_cluster_count(cost, s, 0.5).n_clusters;
    CHECK(c >= last);
    last = c;
  }

  // A c that visits every cluster of every count asks for the exact answer, which more clusters
  // only delay.
  s.probe_rows = 1e8;
  CHECK(choose_cluster_count(cost, s, 1e6).n_clusters == 1024);
  CHECK(choose_cluster_count(cost, s, std::numeric_limits<double>::infinity()).n_clusters == 1024);

  // No count leaves a cluster fewer than 256 rows.
  s.corpus_rows = 1e5;
  CHECK(choose_cluster_count(cost, s, 0.25).n_clusters == 316);

  // n_p = min(ceil(c * sqrt(N)), N), at least 1 for c > 0, on lists Sirius sized itself.
  CHECK(scaled_probes(0.5, 1024) == 16);
  CHECK(scaled_probes(0.5, 4096) == 32);
  CHECK(scaled_probes(0.5, 16384) == 64);
  CHECK(scaled_probes(5.0 / 32, 2048) == 8);
  CHECK(scaled_probes(0.07, 10000) == 7);  // 0.07 * 100 is 7.000000000000001 in doubles
  CHECK(scaled_probes(0.55, 10000) == 55);
  CHECK(scaled_probes(0.555, 10000) == 56);
  CHECK(scaled_probes(1e-9, 4096) == 1);
  CHECK(scaled_probes(32, 4096) == 2048);
  CHECK(scaled_probes(64, 4096) == 4096);
  CHECK(scaled_probes(1e9, 4096) == 4096);
  CHECK(scaled_probes(std::numeric_limits<double>::infinity(), 4096) == 4096);
  CHECK(scaled_probes(0.5, 512) == 12);
  CHECK(scaled_probes(0, 4096) == 0);
  CHECK(scaled_probes(-1, 4096) == 0);
  CHECK(scaled_probes(std::numeric_limits<double>::quiet_NaN(), 4096) == 0);

  // On lists the user sized, c counts clusters, rounded up.
  using sirius::vss::literal_probes;
  CHECK(literal_probes(2, 252) == 2);
  CHECK(literal_probes(2.5, 252) == 3);
  CHECK(literal_probes(0.1, 252) == 1);
  CHECK(literal_probes(1000, 252) == 252);
  CHECK(literal_probes(0, 252) == 0);

  // The two sides of the trade: the build grows with the count, the search falls.
  s.corpus_rows       = 1e8;
  s.probe_rows        = 1e7;
  s.n_clusters        = 1024;
  auto const build_1k = cost.build(s), search_1k = cost.probed(s, 16);
  s.n_clusters = 16384;
  CHECK(cost.build(s) > build_1k);
  CHECK(cost.probed(s, 64) < search_1k);
}
