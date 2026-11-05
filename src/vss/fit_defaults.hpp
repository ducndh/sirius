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

#include <cstdint>

namespace sirius::vss {

/// A statement's fit is paid once and its recall is paid per probe. Past these two, more training
/// rows and more iterations gain tenths of a recall point at any fixed probe count for several
/// times the fit, on narrow and wide rows alike, so they lose unless the join runs for many
/// multiples of the fit. train_rows and n_iters in SQL ask for more.
inline constexpr std::int64_t default_train_rows_per_cluster = 128;
inline constexpr std::int32_t default_fit_iterations         = 5;

}  // namespace sirius::vss
