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

#include "vss/device_rates.hpp"

#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace sirius::vss {

namespace {

// What measure_device_rates reaches on the RTX A5000 (sm_86, 64 SMs, PCIe 4.0 x16) the cost
// constants were fitted on; three runs agreed to 0.1%.
constexpr device_rates kReference{
  .fp32 = 2.796e13, .f16 = 1.2208e14, .int8 = 2.4418e14, .pcie = 2.671e10, .hbm = 6.752e11};

device_scale measure_scale(int device)
{
  if (auto const* v = std::getenv("SIRIUS_VSS_DEVICE_RATES");
      v != nullptr && std::strcmp(v, "reference") == 0) {
    return {};
  }
  device_rates r;
  if (!measure_device_rates(r)) {
    SIRIUS_LOG_WARN(
      "[device_rates] GPU {}: the microbenchmarks failed; the vector join's cost model keeps the "
      "A5000's rates",
      device);
    return {};
  }
  // The A5000 itself measures up to ~10% apart between processes (its boost clock under a 200 W
  // cap), which is inside the model's own ~15%, so a ratio that close is not acted on.
  auto ratio = [](double measured, double reference) {
    double const x = measured / reference;
    return std::abs(x - 1.0) <= 0.15 ? 1.0 : x;
  };
  device_scale const s{.fp32 = ratio(r.fp32, kReference.fp32),
                       .f16  = ratio(r.f16, kReference.f16),
                       .int8 = ratio(r.int8, kReference.int8),
                       .pcie = ratio(r.pcie, kReference.pcie),
                       .hbm  = ratio(r.hbm, kReference.hbm)};
  SIRIUS_LOG_INFO(
    "[device_rates] GPU {}: fp32 {:.3g} flop/s, f16 {:.3g}, int8 {:.3g} op/s, pcie {:.3g} B/s, "
    "device {:.3g} B/s; over the A5000: fp32 {:.2f} f16 {:.2f} int8 {:.2f} pcie {:.2f} hbm {:.2f}",
    device,
    r.fp32,
    r.f16,
    r.int8,
    r.pcie,
    r.hbm,
    s.fp32,
    s.f16,
    s.int8,
    s.pcie,
    s.hbm);
  return s;
}

}  // namespace

device_scale const& current_device_scale()
{
  static std::mutex mutex;
  static std::map<int, device_scale> by_device;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) { device = 0; }
  std::lock_guard<std::mutex> const lock(mutex);
  auto it = by_device.find(device);
  if (it == by_device.end()) { it = by_device.emplace(device, measure_scale(device)).first; }
  return it->second;
}

}  // namespace sirius::vss
