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

#pragma once

#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::vss {

/// @p rows as a cuDF column size. Join outputs are sized rows x k, or by the pairs a threshold
/// keeps, and a cuDF column holds at most 2^31 - 1 rows: a narrowing cast past that wraps to a
/// negative or short size, and the kernels then write past the buffer.
inline cudf::size_type column_size(std::int64_t rows, char const* what)
{
  if (rows < 0 || rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::overflow_error(std::string{what} + ": " + std::to_string(rows) +
                              " rows exceed a column's limit of 2^31 - 1; ask for a smaller k");
  }
  return static_cast<cudf::size_type>(rows);
}

}  // namespace sirius::vss
