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

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <vss/bound_gemm.hpp>
#include <vss/cluster_fold.hpp>
#include <vss/cluster_lists.hpp>
#include <vss/exact_rows.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

// rotate_rows promises each output is one fmaf chain over k = 0..d-1 in order: a row's codes
// never depend on the rows it was batched with, and rotation_slack bounds that chain's error.
// The CPU's fmaf is the same correctly rounded operation, so the two must agree bit for bit.
TEST_CASE("rotate_rows is an in-order fmaf chain per output whatever the batch", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  std::mt19937 gen(3);
  std::normal_distribution<float> normal(0.f, 3.f);
  for (std::int64_t const d : {16, 48, 96, 128, 160, 256, 100, 1040, 1536}) {
    INFO("d " << d);
    std::int64_t const rows = d > 256 ? 130 : 1000;  // not a multiple of any tile
    auto const r            = sirius::vss::random_rotation(d, 11);
    std::vector<float> x(static_cast<std::size_t>(rows * d));
    for (auto& v : x) {
      v = normal(gen);
    }
    rmm::device_uvector<float> dx(x.size(), stream), dr(r.size(), stream), doubt(x.size(), stream);
    REQUIRE(cudaMemcpy(dx.data(), x.data(), x.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dr.data(), r.data(), r.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    // The whole batch, and the same rows as a 37-row batch from an odd offset.
    sirius::vss::rotate_rows(dx.data(), rows, d, dr.data(), doubt.data(), stream);
    std::vector<float> got(x.size());
    stream.sync();
    REQUIRE(cudaMemcpy(got.data(), doubt.data(), got.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    std::int64_t const off = 71, part = 37;
    sirius::vss::rotate_rows(dx.data() + off * d, part, d, dr.data(), doubt.data(), stream);
    std::vector<float> got_part(static_cast<std::size_t>(part * d));
    stream.sync();
    REQUIRE(
      cudaMemcpy(got_part.data(), doubt.data(), got_part.size() * 4, cudaMemcpyDeviceToHost) ==
      cudaSuccess);
    std::int64_t mismatches = 0;
    for (std::int64_t i = 0; i < rows; ++i) {
      for (std::int64_t j = 0; j < d; ++j) {
        float acc = 0.f;
        for (std::int64_t k = 0; k < d; ++k) {
          acc = std::fmaf(x[i * d + k], r[j * d + k], acc);
        }
        mismatches += std::memcmp(&acc, &got[i * d + j], sizeof(float)) != 0;
        if (i >= off && i < off + part) {
          mismatches += std::memcmp(&acc, &got_part[(i - off) * d + j], sizeof(float)) != 0;
        }
      }
    }
    REQUIRE(mismatches == 0);

    // The range taken while rotating, never writing the rows, is column_min_max's over them.
    rmm::device_uvector<unsigned int> lo(d, stream), hi(d, stream), lo2(d, stream), hi2(d, stream);
    for (auto* v : {&lo, &lo2}) {
      REQUIRE(cudaMemsetAsync(v->data(), 0xff, d * 4, stream.get()) == cudaSuccess);
    }
    for (auto* v : {&hi, &hi2}) {
      REQUIRE(cudaMemsetAsync(v->data(), 0, d * 4, stream.get()) == cudaSuccess);
    }
    sirius::vss::rotate_rows(dx.data(), rows, d, dr.data(), doubt.data(), stream);
    sirius::vss::column_min_max(doubt.data(), rows, d, lo.data(), hi.data(), stream);
    if (sirius::vss::rotate_rows_min_max(
          dx.data(), rows, d, dr.data(), lo2.data(), hi2.data(), stream)) {
      std::vector<unsigned int> a(d), b(d), c(d), e(d);
      stream.sync();
      REQUIRE(cudaMemcpy(a.data(), lo.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(b.data(), lo2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(c.data(), hi.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(e.data(), hi2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(a == b);
      REQUIRE(c == e);

      // Asked to keep the rows as well, it writes exactly what rotate_rows writes, and the range.
      rmm::device_uvector<float> kept(x.size(), stream);
      REQUIRE(cudaMemsetAsync(lo2.data(), 0xff, d * 4, stream.get()) == cudaSuccess);
      REQUIRE(cudaMemsetAsync(hi2.data(), 0, d * 4, stream.get()) == cudaSuccess);
      REQUIRE(sirius::vss::rotate_rows_min_max(
        dx.data(), rows, d, dr.data(), lo2.data(), hi2.data(), stream, kept.data()));
      std::vector<float> got_kept(x.size());
      stream.sync();
      REQUIRE(
        cudaMemcpy(got_kept.data(), kept.data(), got_kept.size() * 4, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
      REQUIRE(cudaMemcpy(b.data(), lo2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(e.data(), hi2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(std::memcmp(got_kept.data(), got.data(), got.size() * 4) == 0);
      REQUIRE(a == b);
      REQUIRE(c == e);
    } else {
      REQUIRE(d % 16 != 0);  // the fused path covers every width a multiple of 16
    }
  }
}

namespace {

constexpr double kU = 5.9604644775390625e-8;

// || Q Q^T - I || by plain loops on the host, independent of the device kernels.
sirius::vss::orthogonality_defect host_defect(std::vector<double> const& q, std::int64_t d)
{
  sirius::vss::orthogonality_defect out;
  double sum = 0;
  for (std::int64_t i = 0; i < d; ++i) {
    for (std::int64_t j = 0; j < d; ++j) {
      double dot = 0;
      for (std::int64_t k = 0; k < d; ++k) {
        dot += q[i * d + k] * q[j * d + k];
      }
      double const e = dot - (i == j ? 1.0 : 0.0);
      out.max_abs    = std::max(out.max_abs, std::abs(e));
      sum += e * e;
    }
  }
  out.frobenius = std::sqrt(sum);
  return out;
}

template <typename T>
std::vector<T> to_host(T const* p, std::size_t n, rmm::cuda_stream_view stream)
{
  std::vector<T> out(n);
  stream.sync();
  REQUIRE(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}

// The allowance rotation_slack documents for a basis's defect, in the norm that bounds the
// spectral one.
double allowed_defect(std::int64_t d) { return 4.0 * std::sqrt(static_cast<double>(d)) * kU; }

}  // namespace

TEST_CASE("random_rotation_f64_device is orthogonal and the same for the same seed", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  for (std::int64_t const d : {16, 64, 80, 128, 200, 512, 1024}) {
    INFO("d " << d);
    auto const q      = sirius::vss::random_rotation_f64_device(d, 7, stream, mr);
    auto const again  = sirius::vss::random_rotation_f64_device(d, 7, stream, mr);
    auto const other  = sirius::vss::random_rotation_f64_device(d, 8, stream, mr);
    auto const n      = static_cast<std::size_t>(d * d);
    auto const host_q = to_host(q.data(), n, stream);
    auto const host_a = to_host(again.data(), n, stream);
    auto const host_o = to_host(other.data(), n, stream);
    REQUIRE(std::memcmp(host_q.data(), host_a.data(), n * sizeof(double)) == 0);
    REQUIRE(std::memcmp(host_q.data(), host_o.data(), n * sizeof(double)) != 0);
    auto const defect = sirius::vss::rotation_defect(q.data(), d, stream, mr);
    CHECK(defect.frobenius < 1e-11);
    if (d <= 512) {
      // The device's measure agrees with plain loops on the host.
      auto const on_host = host_defect(host_q, d);
      CHECK(on_host.frobenius < 1e-11);
      CHECK(std::abs(on_host.frobenius - defect.frobenius) < 1e-12);
      CHECK(std::abs(on_host.max_abs - defect.max_abs) < 1e-12);
    }
    // What rotate_rows reads, against what rotation_slack allows, as the host's construction.
    rmm::device_uvector<float> narrow(n, stream);
    sirius::vss::narrow_rotation(q.data(), d, narrow.data(), stream);
    auto const f = sirius::vss::rotation_defect(narrow.data(), d, stream, mr);
    CHECK(f.frobenius <= allowed_defect(d));
    auto const host = sirius::vss::random_rotation(d, 7);
    rmm::device_uvector<float> dh(n, stream);
    REQUIRE(cudaMemcpy(dh.data(), host.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    auto const fh = sirius::vss::rotation_defect(dh.data(), d, stream, mr);
    CHECK(fh.frobenius <= allowed_defect(d));
  }
}

TEST_CASE("structured_rotate_rows is within its slack of the exact orthogonal map", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::mt19937 gen(5);
  std::normal_distribution<float> normal(0.f, 1.f);
  for (std::int64_t const d :
       {16, 32, 48, 80, 112, 144, 176, 208, 240, 256, 512, 768, 1024, 1536, 2048, 3072}) {
    INFO("d " << d);
    REQUIRE(sirius::vss::structured_rotation_supports(d));
    int a  = 0;
    auto m = d;
    while (m % 2 == 0) {
      m /= 2;
      ++a;
    }
    auto const signs        = sirius::vss::structured_rotation_signs(d, 3);
    std::int64_t const rows = 9 + std::max<std::int64_t>(3, 4000 / d);
    std::vector<float> x(static_cast<std::size_t>(rows * d));
    for (std::int64_t r = 0; r < rows; ++r) {
      auto* row = x.data() + r * d;
      for (std::int64_t j = 0; j < d; ++j) {
        row[j] = normal(gen);
      }
      // Rows a rounding argument should worry about: one huge component, huge and tiny mixed,
      // a constant row, one set bit, a row of the map's own signs.
      if (r == 0) { row[d / 3] = 1e6f; }
      if (r == 1) {
        for (std::int64_t j = 0; j < d; ++j) {
          row[j] *= j % 2 == 0 ? 1e4f : 1e-4f;
        }
      }
      if (r == 2) { std::fill(row, row + d, 1.f); }
      if (r == 3) {
        std::fill(row, row + d, 0.f);
        row[d - 1] = 1.f;
      }
      if (r == 4) {
        for (std::int64_t j = 0; j < d; ++j) {
          row[j] = (signs[j >> 5] >> (j & 31)) & 1u ? -1.f : 1.f;
        }
      }
      if (r == 5) { std::fill(row, row + d, 0.f); }
    }
    rmm::device_uvector<float> dx(x.size(), stream), out(x.size(), stream);
    rmm::device_uvector<std::uint32_t> ds(signs.size(), stream);
    REQUIRE(cudaMemcpy(dx.data(), x.data(), x.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds.data(), signs.data(), signs.size() * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    sirius::vss::structured_rotate_rows(dx.data(), rows, d, ds.data(), out.data(), stream);
    auto const whole = to_host(out.data(), x.size(), stream);

    // The small factor, as documented; the Walsh matrix by its popcount formula.
    std::vector<double> small(static_cast<std::size_t>(m * m), 0.0);
    for (std::int64_t q = 0; q < m; ++q) {
      for (std::int64_t t = 0; t < m; ++t) {
        small[q * m + t] = m == 1   ? 1.0
                           : m == 3 ? (q == t ? 1.0 : 0.0) - 2.0 / 3.0
                                    : (q == 0 ? std::sqrt(1.0 / m) : std::sqrt(2.0 / m)) *
                                        std::cos(M_PI * (2 * t + 1) * q / (2.0 * m));
      }
    }
    double const scale = std::pow(2.0, -0.5 * a);
    auto const groups  = std::int64_t{1} << a;
    for (std::int64_t r = 0; r < rows; ++r) {
      std::vector<double> exact(static_cast<std::size_t>(d), 0.0),
        mid(static_cast<std::size_t>(d), 0.0);
      double norm2 = 0;
      for (std::int64_t j = 0; j < d; ++j) {
        double const v = (signs[j >> 5] >> (j & 31)) & 1u ? -x[r * d + j] : x[r * d + j];
        norm2 += v * v;
        auto const g = j / m, t = j % m;
        for (std::int64_t q = 0; q < m; ++q) {
          mid[g * m + q] += small[q * m + t] * v;
        }
      }
      for (std::int64_t i = 0; i < groups; ++i) {
        for (std::int64_t g = 0; g < groups; ++g) {
          double const sign =
            __builtin_popcountll(static_cast<unsigned long long>(i & g)) % 2 ? -1.0 : 1.0;
          for (std::int64_t q = 0; q < m; ++q) {
            exact[i * m + q] += sign * scale * mid[g * m + q];
          }
        }
      }
      double err2 = 0;
      for (std::int64_t j = 0; j < d; ++j) {
        double const e = exact[j] - whole[r * d + j];
        err2 += e * e;
      }
      INFO("row " << r);
      CHECK(std::sqrt(err2) <= sirius::vss::structured_rotation_slack(d) * std::sqrt(norm2));
    }

    // A row's result does not depend on what it is batched with.
    for (std::int64_t const off : {std::int64_t{1}, rows / 2}) {
      auto const part = std::min<std::int64_t>(rows - off, 5);
      sirius::vss::structured_rotate_rows(
        dx.data() + off * d, part, d, ds.data(), out.data(), stream);
      auto const got = to_host(out.data(), static_cast<std::size_t>(part * d), stream);
      REQUIRE(std::memcmp(got.data(), whole.data() + off * d, got.size() * 4) == 0);
    }

    // The map itself, row by row over the identity.
    if (d <= 1024) {
      std::vector<float> id(static_cast<std::size_t>(d * d), 0.f);
      for (std::int64_t i = 0; i < d; ++i) {
        id[i * d + i] = 1.f;
      }
      rmm::device_uvector<float> did(id.size(), stream), image(id.size(), stream);
      REQUIRE(cudaMemcpy(did.data(), id.data(), id.size() * 4, cudaMemcpyHostToDevice) ==
              cudaSuccess);
      sirius::vss::structured_rotate_rows(did.data(), d, d, ds.data(), image.data(), stream);
      auto const defect = sirius::vss::rotation_defect(image.data(), d, stream, mr);
      CHECK(defect.frobenius <= allowed_defect(d));
    }
  }
  CHECK(sirius::vss::structured_rotation_supports(15360));
  CHECK_FALSE(sirius::vss::structured_rotation_supports(1040));  // 16 x 65
  CHECK_FALSE(sirius::vss::structured_rotation_supports(100));
}

namespace {

// c(r, s), delta(r, s) and each probe row's error over every pair of @p probes x @p rows, the
// inputs of the two inequalities the INT8 bounded search rests on.
struct coded_pairs {
  std::vector<std::int64_t> code_distance;
  std::vector<float> rescored, pair_probe_error;
  std::vector<std::int32_t> pair_row;
  float row_error{0}, scale{1};
};

// Codes built as the list build does: the range over the rows, one scale, each side's error from
// quantize_rows_int8; the re-score by the kernel a join runs.
coded_pairs code_and_rescore(std::vector<float> const& rows,
                             std::vector<float> const& probes,
                             std::int64_t d,
                             char rotation,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const n  = static_cast<std::int64_t>(rows.size()) / d;
  auto const m  = static_cast<std::int64_t>(probes.size()) / d;
  auto const up = [&](std::vector<float> const& v) {
    rmm::device_uvector<float> out(v.size(), stream);
    REQUIRE(cudaMemcpy(out.data(), v.data(), v.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    return out;
  };
  auto const d_rows = up(rows), d_probes = up(probes);
  rmm::device_uvector<float> r_rows(rows.size(), stream), r_probes(probes.size(), stream);
  auto const signs = sirius::vss::structured_rotation_signs(d, 17);
  rmm::device_uvector<std::uint32_t> d_signs(signs.size(), stream);
  REQUIRE(cudaMemcpy(d_signs.data(), signs.data(), signs.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  std::optional<rmm::device_uvector<float>> dense;
  if (rotation == 'd') {
    auto const q = sirius::vss::random_rotation(d, 19);
    dense.emplace(q.size(), stream);
    REQUIRE(cudaMemcpy(dense->data(), q.data(), q.size() * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  sirius::vss::code_rotation_ref const ref{
    rotation == 's'   ? static_cast<void const*>(d_signs.data())
    : rotation == 'd' ? static_cast<void const*>(dense->data())
                      : nullptr,
    rotation == 's'};
  float const* rows_in   = d_rows.data();
  float const* probes_in = d_probes.data();
  if (rotation != '0') {
    sirius::vss::apply_code_rotation(ref, d_rows.data(), n, d, r_rows.data(), stream);
    sirius::vss::apply_code_rotation(ref, d_probes.data(), m, d, r_probes.data(), stream);
    rows_in   = r_rows.data();
    probes_in = r_probes.data();
  }
  auto const host_rows = to_host(rows_in, rows.size(), stream);
  std::vector<float> offset(static_cast<std::size_t>(d));
  float half_range = 0;
  for (std::int64_t j = 0; j < d; ++j) {
    float lo = INFINITY, hi = -INFINITY;
    for (std::int64_t r = 0; r < n; ++r) {
      lo = std::min(lo, host_rows[r * d + j]);
      hi = std::max(hi, host_rows[r * d + j]);
    }
    offset[j]  = 0.5f * (lo + hi);
    half_range = std::max(half_range, 0.5f * (hi - lo));
  }
  coded_pairs out;
  out.scale           = half_range > 0.f ? half_range / 127.f : 1.f;
  auto const d_offset = up(offset);
  rmm::device_uvector<std::int8_t> c_rows(rows.size(), stream), c_probes(probes.size(), stream);
  rmm::device_uvector<float> probe_error(static_cast<std::size_t>(m), stream);
  rmm::device_uvector<unsigned int> error_bits(1, stream);
  REQUIRE(cudaMemsetAsync(error_bits.data(), 0, 4, stream.get()) == cudaSuccess);
  auto const slack = sirius::vss::code_rotation_slack(ref, d);
  sirius::vss::quantize_rows_int8(rows_in,
                                  n,
                                  d,
                                  d_offset.data(),
                                  out.scale,
                                  c_rows.data(),
                                  nullptr,
                                  error_bits.data(),
                                  stream,
                                  slack);
  sirius::vss::quantize_rows_int8(probes_in,
                                  m,
                                  d,
                                  d_offset.data(),
                                  out.scale,
                                  c_probes.data(),
                                  probe_error.data(),
                                  nullptr,
                                  stream,
                                  slack);
  auto const row_codes   = to_host(c_rows.data(), rows.size(), stream);
  auto const probe_codes = to_host(c_probes.data(), probes.size(), stream);
  auto const row_error   = to_host(error_bits.data(), 1, stream);
  std::memcpy(&out.row_error, row_error.data(), 4);
  auto const probe_errors = to_host(probe_error.data(), static_cast<std::size_t>(m), stream);
  auto const pairs        = static_cast<std::size_t>(n * m);
  std::vector<std::int64_t> ids(pairs);
  out.code_distance.resize(pairs);
  out.pair_probe_error.resize(pairs);
  out.pair_row.resize(pairs);
  for (std::int64_t q = 0; q < m; ++q) {
    for (std::int64_t r = 0; r < n; ++r) {
      std::int64_t c = 0;
      for (std::int64_t j = 0; j < d; ++j) {
        auto const t = static_cast<std::int64_t>(probe_codes[q * d + j]) - row_codes[r * d + j];
        c += t * t;
      }
      auto const i            = static_cast<std::size_t>(q * n + r);
      ids[i]                  = r;
      out.code_distance[i]    = c;
      out.pair_probe_error[i] = probe_errors[q];
      out.pair_row[i]         = static_cast<std::int32_t>(q);
    }
  }
  rmm::device_uvector<std::int64_t> d_ids(pairs, stream);
  rmm::device_uvector<std::int32_t> d_pair_row(pairs, stream);
  REQUIRE(cudaMemcpy(d_ids.data(), ids.data(), pairs * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_pair_row.data(), out.pair_row.data(), pairs * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  rmm::device_uvector<float> distances(pairs, stream);
  rmm::device_uvector<float const*> blocks(1, stream);
  float const* block = d_rows.data();
  REQUIRE(cudaMemcpy(blocks.data(), &block, sizeof(block), cudaMemcpyHostToDevice) == cudaSuccess);
  sirius::vss::exact_rows source;
  source.copy_blocks    = blocks.data();
  source.rows_per_block = n;
  sirius::vss::exact_distances(d_probes.data(),
                               d_pair_row.data(),
                               0,
                               d_ids.data(),
                               distances.data(),
                               static_cast<std::int64_t>(pairs),
                               source,
                               d,
                               stream,
                               mr);
  out.rescored = to_host(distances.data(), pairs, stream);
  return out;
}

// delta <= b => c <= L(b), taking b = delta itself, the tightest b that holds it, and
// c <= c' => delta <= U(c'), taking c' = the pair's own c rounded up as a filter writes it.
void check_code_inequalities(coded_pairs const& p,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<std::int64_t>(p.rescored.size());
  rmm::device_uvector<float> bound(p.rescored.size(), stream), errors(p.rescored.size(), stream),
    limit(p.rescored.size(), stream);
  REQUIRE(
    cudaMemcpy(bound.data(), p.rescored.data(), p.rescored.size() * 4, cudaMemcpyHostToDevice) ==
    cudaSuccess);
  REQUIRE(cudaMemcpy(errors.data(),
                     p.pair_probe_error.data(),
                     p.pair_probe_error.size() * 4,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  sirius::vss::int8_code_limit(
    bound.data(), errors.data(), p.row_error, p.scale, n, limit.data(), stream);
  auto const limits = to_host(limit.data(), limit.size(), stream);
  std::vector<float> code_up(p.rescored.size());
  for (std::size_t i = 0; i < code_up.size(); ++i) {
    auto f = static_cast<float>(p.code_distance[i]);
    if (static_cast<std::int64_t>(static_cast<double>(f)) < p.code_distance[i]) {
      f = std::nextafter(f, INFINITY);
    }
    code_up[i] = f;
  }
  REQUIRE(cudaMemcpy(bound.data(), code_up.data(), code_up.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  sirius::vss::int8_seed_upper_bound(bound.data(), errors.data(), p.row_error, p.scale, n, stream);
  auto const uppers      = to_host(bound.data(), bound.size(), stream);
  std::size_t over_limit = 0, over_upper = 0;
  for (std::size_t i = 0; i < limits.size(); ++i) {
    over_limit += std::isfinite(limits[i]) && static_cast<double>(p.code_distance[i]) >
                                                std::floor(static_cast<double>(limits[i]));
    over_upper += p.rescored[i] > uppers[i];
  }
  CHECK(over_limit == 0);
  CHECK(over_upper == 0);
}

// Rows that take a code exactly, so that no coding error hides the rounding terms: every sum of
// odd squares past 2^24 rounds down at each addition, the most a re-score can understate, and
// large components far from the origin put the rounding where the rows' norm is largest.
std::vector<float> lattice_rows(std::int64_t rows,
                                std::int64_t d,
                                int value,
                                std::int64_t odd_every)
{
  std::vector<float> out(static_cast<std::size_t>(rows * d));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t j = 0; j < d; ++j) {
      out[r * d + j] =
        static_cast<float>(value - (odd_every > 0 && r % 2 == 1 && j % odd_every == 0));
    }
  }
  return out;
}

}  // namespace

TEST_CASE("INT8 code limits hold every pair the FP32 re-score has within a bound", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::mt19937 gen(29);
  std::normal_distribution<float> normal(0.f, 1.f);
  SECTION("rows a re-score rounds most, no coding error")
  {
    // A row of all 127 and one of all -127 fix scale 1 and offset 0, so the others code exactly.
    for (std::int64_t const d : {3072, 8192, 15360}) {
      INFO("d " << d);
      auto rows      = lattice_rows(4, d, -126, 0);
      auto const add = lattice_rows(2, d, 127, 0);
      std::copy(add.begin(), add.begin() + d, rows.begin());
      auto const low = lattice_rows(1, d, -127, 0);
      std::copy(low.begin(), low.end(), rows.begin() + d);
      auto const probes = lattice_rows(3, d, 127, 0);
      auto const p      = code_and_rescore(rows, probes, d, '0', stream, mr);
      REQUIRE(p.scale == 1.f);
      // 253^2 is odd: past 2^24 each lane's sum loses 1 at each addition, the case the rounding
      // terms are for, reached once a lane holds more than 262 terms.
      auto const t253 = static_cast<double>(p.rescored[2]);
      auto const c253 = static_cast<double>(p.code_distance[2]);
      if (d > 262 * 32) {
        CHECK(t253 < c253 * (1 - 16.8 * kU));
      } else {
        CHECK(t253 <= c253);
      }
      check_code_inequalities(p, stream, mr);
    }
  }
  SECTION("gaussian, far from the origin and one wide component, under each rotation")
  {
    for (std::int64_t const d : {512, 768, 1024, 1536, 3072, 15360}) {
      for (char const rotation : {'0', 's', 'd'}) {
        if (rotation == 'd' && d > 1536) { continue; }
        INFO("d " << d << " rotation " << rotation);
        for (int const shape : {0, 1, 2}) {
          INFO("shape " << shape);
          auto const n = std::int64_t{64}, m = std::int64_t{8};
          std::vector<float> rows(static_cast<std::size_t>(n * d)),
            probes(static_cast<std::size_t>(m * d));
          auto const fill = [&](std::vector<float>& v) {
            for (std::size_t i = 0; i < v.size(); ++i) {
              auto const g = normal(gen);
              v[i] = shape == 0   ? g
                     : shape == 1 ? 8000.f + 0.25f * std::rint(4.f * g)
                                  : g * (i % static_cast<std::size_t>(d) == 0 ? 3000.f : 1.f);
            }
          };
          fill(rows);
          fill(probes);
          check_code_inequalities(
            code_and_rescore(rows, probes, d, rotation, stream, mr), stream, mr);
        }
      }
    }
  }
}

// Not a check: prints what each construction costs and how orthogonal it comes out.
TEST_CASE("rotation construction report", "[.rotation_report]")
{
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const seconds = [](auto&& work) {
    auto const t0 = std::chrono::steady_clock::now();
    work();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  std::printf("d kind seconds f64_max f64_fro f32_max f32_fro allowed slack\\n");
  for (std::int64_t const d : {512, 768, 1024, 1536, 3072}) {
    auto const n    = static_cast<std::size_t>(d * d);
    auto const seed = 0x5151'7a7e'0000'0000ull + static_cast<std::uint64_t>(d);
    std::vector<double> host_q;
    auto const t_host = seconds([&] { host_q = sirius::vss::random_rotation_f64(d, seed); });
    rmm::device_uvector<double> dq(n, stream);
    rmm::device_uvector<float> df(n, stream);
    REQUIRE(cudaMemcpy(dq.data(), host_q.data(), n * 8, cudaMemcpyHostToDevice) == cudaSuccess);
    sirius::vss::narrow_rotation(dq.data(), d, df.data(), stream);
    auto const h64 = sirius::vss::rotation_defect(dq.data(), d, stream, mr);
    auto const h32 = sirius::vss::rotation_defect(df.data(), d, stream, mr);
    std::printf("%ld host %.4f %.3e %.3e %.3e %.3e %.3e %.3e\\n",
                (long)d,
                t_host,
                h64.max_abs,
                h64.frobenius,
                h32.max_abs,
                h32.frobenius,
                allowed_defect(d),
                sirius::vss::rotation_slack(d));
    for (int run = 0; run < 3; ++run) {
      stream.sync();
      std::optional<rmm::device_uvector<double>> q;
      auto const t_gpu = seconds([&] {
        q.emplace(sirius::vss::random_rotation_f64_device(d, seed, stream, mr));
        sirius::vss::narrow_rotation(q->data(), d, df.data(), stream);
        stream.sync();
      });
      if (run < 2) {
        std::printf("%ld gpu %.4f\\n", (long)d, t_gpu);
        continue;
      }
      auto const g64 = sirius::vss::rotation_defect(q->data(), d, stream, mr);
      auto const g32 = sirius::vss::rotation_defect(df.data(), d, stream, mr);
      std::printf("%ld gpu %.4f %.3e %.3e %.3e %.3e %.3e %.3e\\n",
                  (long)d,
                  t_gpu,
                  g64.max_abs,
                  g64.frobenius,
                  g32.max_abs,
                  g32.frobenius,
                  allowed_defect(d),
                  sirius::vss::rotation_slack(d));
    }
    auto const signs = sirius::vss::structured_rotation_signs(d, seed);
    rmm::device_uvector<std::uint32_t> ds(signs.size(), stream);
    REQUIRE(cudaMemcpy(ds.data(), signs.data(), signs.size() * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    std::vector<float> id(n, 0.f);
    for (std::int64_t i = 0; i < d; ++i) {
      id[i * d + i] = 1.f;
    }
    rmm::device_uvector<float> did(n, stream), image(n, stream);
    REQUIRE(cudaMemcpy(did.data(), id.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    sirius::vss::structured_rotate_rows(did.data(), d, d, ds.data(), image.data(), stream);
    auto const s32 = sirius::vss::rotation_defect(image.data(), d, stream, mr);
    std::printf("%ld structured 0 - - %.3e %.3e %.3e %.3e\\n",
                (long)d,
                s32.max_abs,
                s32.frobenius,
                allowed_defect(d),
                sirius::vss::structured_rotation_slack(d));
    std::fflush(stdout);
  }
}

// Sweep 0 only sets a bound the join then re-scores, so a join's answer cannot show this kernel
// ranking wrongly: it is checked directly, at widths that fit shared memory whole and that do not.
TEST_CASE("nearest_cluster_topk ranks like plain loops at any row width", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  std::mt19937 gen(9);
  std::uniform_int_distribution<int> code(-40, 40);
  for (std::int64_t const d : {64, 2048, 2064, 3072, 6144}) {
    INFO("d " << d);
    std::int64_t const rows = 400, probes = 37, k = 10;
    std::vector<std::int8_t> x(static_cast<std::size_t>(rows * d)),
      q(static_cast<std::size_t>(probes * d));
    for (auto& v : x) {
      v = static_cast<std::int8_t>(code(gen));
    }
    for (auto& v : q) {
      v = static_cast<std::int8_t>(code(gen));
    }
    auto const sq = [&](std::int8_t const* r) {
      std::int32_t out = 0;
      for (std::int64_t j = 0; j < d; ++j) {
        out += r[j] * r[j];
      }
      return out;
    };
    std::vector<std::int32_t> x_sq(rows), q_sq(probes);
    for (std::int64_t r = 0; r < rows; ++r) {
      x_sq[r] = sq(x.data() + r * d);
    }
    for (std::int64_t r = 0; r < probes; ++r) {
      q_sq[r] = sq(q.data() + r * d);
    }
    // Two slices of the rows; probe rows routed to them through a shuffled list.
    std::vector<std::int64_t> routed(probes);
    std::iota(routed.begin(), routed.end(), std::int64_t{0});
    std::shuffle(routed.begin(), routed.end(), gen);
    std::int64_t const cut = 137, first_part = 21;
    std::vector<sirius::vss::nearest_task> tasks;
    auto const add = [&](std::int64_t off, std::int64_t n, std::int64_t b, std::int64_t e) {
      for (auto i = b; i < e; i += sirius::vss::kNearestTaskProbes) {
        tasks.push_back({off,
                         n,
                         i,
                         static_cast<std::int32_t>(
                           std::min<std::int64_t>(sirius::vss::kNearestTaskProbes, e - i)),
                         off});
      }
    };
    add(0, cut, 0, first_part);
    add(cut, rows - cut, first_part, probes);
    rmm::device_uvector<std::int8_t> dx(x.size(), stream), dq(q.size(), stream);
    rmm::device_uvector<std::int32_t> dxs(x_sq.size(), stream), dqs(q_sq.size(), stream);
    rmm::device_uvector<std::int64_t> droute(routed.size(), stream),
      acc_n(static_cast<std::size_t>(probes * k), stream);
    rmm::device_uvector<float> acc_d(static_cast<std::size_t>(probes * k), stream);
    rmm::device_uvector<sirius::vss::nearest_task> dt(tasks.size(), stream);
    auto const up = [](void* to, void const* from, std::size_t bytes) {
      REQUIRE(cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    };
    up(dx.data(), x.data(), x.size());
    up(dq.data(), q.data(), q.size());
    up(dxs.data(), x_sq.data(), x_sq.size() * 4);
    up(dqs.data(), q_sq.data(), q_sq.size() * 4);
    up(droute.data(), routed.data(), routed.size() * 8);
    up(dt.data(), tasks.data(), tasks.size() * sizeof(sirius::vss::nearest_task));
    sirius::vss::nearest_cluster_topk(dt.data(),
                                      static_cast<std::int64_t>(tasks.size()),
                                      false,
                                      dx.data(),
                                      d,
                                      dxs.data(),
                                      droute.data(),
                                      dq.data(),
                                      dqs.data(),
                                      k,
                                      acc_d.data(),
                                      acc_n.data(),
                                      stream);
    auto const got_d  = to_host(acc_d.data(), acc_d.size(), stream);
    auto const got_n  = to_host(acc_n.data(), acc_n.size(), stream);
    std::size_t wrong = 0;
    for (std::int64_t e = 0; e < probes; ++e) {
      auto const probe = routed[e];
      auto const lo    = e < first_part ? std::int64_t{0} : cut;
      auto const hi    = e < first_part ? cut : rows;
      std::vector<std::int64_t> all;
      for (auto r = lo; r < hi; ++r) {
        std::int64_t dist = 0;
        for (std::int64_t j = 0; j < d; ++j) {
          auto const t = static_cast<std::int64_t>(q[probe * d + j]) - x[r * d + j];
          dist += t * t;
        }
        all.push_back(dist);
      }
      std::sort(all.begin(), all.end());
      for (std::int64_t i = 0; i < k; ++i) {
        wrong += static_cast<std::int64_t>(got_d[probe * k + i]) != all[i];
        wrong += got_n[probe * k + i] < lo || got_n[probe * k + i] >= hi;
      }
    }
    CHECK(wrong == 0);
  }
}
