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

#pragma once

#include "duckdb_table_identity.hpp"
#include "vss/cagra_settings.hpp"
#include "vss/cuvs_index_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory
namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager
namespace sirius::telemetry {
struct batch_telemetry_info;
}  // namespace sirius::telemetry

namespace sirius::vss {

class cagra_index;

/// The table and column one CAGRA index covers: there is one index per table and column.
struct cagra_index_target {
  std::string catalog;
  std::string schema;
  std::string table;
  std::string column;
  sirius::duckdb_table_identity identity;
  std::int64_t dim{0};
};

/// The index on @p target's column that can answer over @p pin: built from that pin, and when
/// @p wanted is given, with those settings. Null when there is none.
[[nodiscard]] std::shared_ptr<const pinned_index_entry> find_usable_cagra_index(
  cuvs_index_cache const& cache,
  cagra_index_target const& target,
  scan_manager::pinned_entry const& pin,
  std::optional<cagra_build_settings> const& wanted);

[[nodiscard]] cagra_index const& cagra_index_of(pinned_index_entry const& entry);

/// What a build over @p n_rows needs on the device at its peak, the finished index and the build's
/// scratch: read before committing to anything, so a table that cannot fit is refused by name.
[[nodiscard]] std::size_t cagra_build_peak_bytes(std::int64_t n_rows,
                                                 std::int64_t dim,
                                                 cagra_build_settings const& settings,
                                                 std::size_t staged_chunk_bytes);

/// Throws InvalidInputException naming the numbers when a build over @p pin cannot fit beside what
/// is on the device now. @p fn prefixes the message.
void check_cagra_build_fits(std::string const& fn,
                            cagra_index_target const& target,
                            scan_manager::pinned_entry const& pin,
                            cagra_build_settings const& settings,
                            cucascade::memory::memory_space const& space);

struct cagra_build_report {
  double convert_seconds{0};
  double build_seconds{0};
  std::size_t peak_bytes{0};
  std::size_t resident_bytes{0};
};

/// Builds @p target's index from @p pin, under a reservation of its own, on a stream the cache
/// entry keeps, and replaces whatever index the column had. Rows reach the device a chunk at a time
/// whatever the pin's tier, and are converted there.
[[nodiscard]] std::shared_ptr<const pinned_index_entry> build_cagra_index_for_pin(
  std::string const& fn,
  cuvs_index_cache& cache,
  cagra_index_target const& target,
  std::shared_ptr<const scan_manager::pinned_entry> const& pin,
  cagra_build_settings const& settings,
  cucascade::memory::memory_space& space,
  telemetry::batch_telemetry_info const& telemetry_info,
  cagra_build_report& report);

}  // namespace sirius::vss
