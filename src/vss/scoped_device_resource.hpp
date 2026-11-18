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

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <stdexcept>
#include <string>

namespace sirius::vss {

/// Checks that allocations on the current device already go through @p mr, and throws if not.
/// Called before any cuVS call whose internal scratch must come out of a Sirius reservation
/// rather than the ambient default, since cuVS allocates from the current device resource and
/// takes no allocator argument.
///
/// It does not change the current resource: that is one process-wide slot per device, and
/// cudf::get_current_device_resource_ref() returns a reference into it that other threads keep
/// and allocate through without a lock, so replacing the slot's resource while a query runs can
/// leave them allocating through a moved-from resource. After startup the slot holds the GPU
/// memory space's allocator, which is what every caller passes.
inline void require_current_device_resource(rmm::device_async_resource_ref mr, char const* what)
{
  if (mr == cudf::get_current_device_resource_ref()) { return; }
  throw std::logic_error(std::string{what} +
                         ": the current device resource is not the memory resource this call "
                         "allocates from, and cuVS allocates its scratch from the current one");
}

}  // namespace sirius::vss
