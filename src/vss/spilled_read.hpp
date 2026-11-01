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

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::vss {

namespace detail {

/// O_DIRECT on by default; SIRIUS_VSS_SPILL_DIRECT=0 reads through the page cache as before.
inline bool spill_direct_enabled()
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_VSS_SPILL_DIRECT");
    return v == nullptr || std::string_view{v} != "0";
  }();
  return on;
}

/// A second descriptor for the same open file with O_DIRECT, or -1 where the filesystem refuses it
/// (tmpfs, some network filesystems): those keep the buffered path.
struct direct_fd {
  int fd{-1};
  explicit direct_fd(int buffered_fd)
  {
    if (buffered_fd < 0 || !spill_direct_enabled()) { return; }
    auto const path = "/proc/self/fd/" + std::to_string(buffered_fd);
    fd              = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  }
  ~direct_fd()
  {
    if (fd >= 0) { ::close(fd); }
  }
  direct_fd(direct_fd const&)            = delete;
  direct_fd& operator=(direct_fd const&) = delete;
};

}  // namespace detail

/**
 * @brief Copies @p size bytes at @p offset of the open file @p fd to device memory @p dst.
 *
 * Reads go through two pinned staging pieces per thread, reused across calls, so the read of one
 * piece overlaps the asynchronous copy of the other: a spilled batch's column comes back without a
 * pageable bounce or a host-pool allocation. Returns once the last copy has landed.
 *
 * The file is read with O_DIRECT where the filesystem allows it. A spilled corpus is larger than
 * host memory and is read once per pass, so the page cache only costs: buffered reads went out as
 * readahead-sized (~80 KB) requests one at a time (1.3 GB/s on an NVMe that serves 5.3 GB/s to a
 * single 8 MiB direct reader), and the cached pages count against the process's memory limit. The
 * direct read covers the requested range rounded out to 4 KiB; only the requested bytes are copied.
 */
namespace detail {

inline void copy_file_to_device_impl(int fd,
                                     std::size_t offset,
                                     std::size_t size,
                                     void* dst,
                                     rmm::cuda_stream_view stream,
                                     std::string const& what,
                                     bool allow_direct)
{
  constexpr std::size_t kPiece = std::size_t{8} << 20;
  constexpr std::size_t kAlign = 4096;  // covers 512 B and 4 KiB logical blocks
  struct staging {
    std::array<void*, 2> host{nullptr, nullptr};
    std::array<cudaEvent_t, 2> landed{nullptr, nullptr};
    ~staging()
    {
      for (std::size_t i = 0; i < 2; ++i) {
        if (landed[i] != nullptr) { cudaEventDestroy(landed[i]); }
        if (host[i] != nullptr) { cudaFreeHost(host[i]); }
      }
    }
  };
  thread_local staging buffers;
  for (std::size_t i = 0; i < 2; ++i) {
    if (buffers.host[i] == nullptr) {
      // cudaMallocHost is page-aligned, which O_DIRECT needs of the buffer.
      CUDF_CUDA_TRY(cudaMallocHost(&buffers.host[i], kPiece));
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&buffers.landed[i], cudaEventDisableTiming));
    }
  }
  if (size == 0) { return; }

  direct_fd direct{allow_direct ? fd : -1};
  bool const use_direct = direct.fd >= 0;
  // The file range read: exactly [offset, offset + size) when buffered, rounded out to kAlign when
  // direct (a direct read may run past end of file; it then returns short, which is fine as long
  // as the requested bytes are in).
  auto const end         = offset + size;
  std::size_t const base = use_direct ? offset / kAlign * kAlign : offset;

  std::array<bool, 2> pending{false, false};
  std::size_t piece = 0;
  for (std::size_t at = base; at < end; at += kPiece, ++piece) {
    auto const slot = piece % 2;
    if (pending[slot]) { CUDF_CUDA_TRY(cudaEventSynchronize(buffers.landed[slot])); }
    auto* host = static_cast<std::byte*>(buffers.host[slot]);
    // Bytes of this piece that are needed, and how much to ask the file for.
    auto const need = std::min(kPiece, end - at);
    auto const ask  = use_direct ? std::min(kPiece, (need + kAlign - 1) / kAlign * kAlign) : need;
    std::size_t got = 0;
    while (got < need) {
      auto const r =
        ::pread(use_direct ? direct.fd : fd, host + got, ask - got, static_cast<off_t>(at + got));
      if (r < 0 && errno == EINTR) { continue; }
      if (r < 0 && errno == EINVAL && use_direct && got == 0 && piece == 0) {
        // The filesystem accepted O_DIRECT at open but not this request: read buffered instead.
        copy_file_to_device_impl(fd, offset, size, dst, stream, what, false);
        return;
      }
      if (r <= 0) {
        // The other piece's copy may still be queued; the next call must not refill it under it.
        stream.synchronize();
        throw std::runtime_error("reading spilled batch " + what + " failed");
      }
      got += static_cast<std::size_t>(r);
    }
    // Copy the requested part of this piece: a direct first piece starts before offset.
    auto const from = std::max(at, offset);
    auto const to   = std::min(at + need, end);
    CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<std::byte*>(dst) + (from - offset),
                                  host + (from - at),
                                  to - from,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaEventRecord(buffers.landed[slot], stream.value()));
    pending[slot] = true;
  }
  stream.synchronize();
}

}  // namespace detail

/**
 * @brief Copies @p size bytes at @p offset of the open file @p fd to device memory @p dst (see
 * detail::copy_file_to_device_impl: pinned double-buffered, O_DIRECT where the filesystem allows).
 */
inline void copy_file_to_device(int fd,
                                std::size_t offset,
                                std::size_t size,
                                void* dst,
                                rmm::cuda_stream_view stream,
                                std::string const& what)
{
  detail::copy_file_to_device_impl(fd, offset, size, dst, stream, what, true);
}

}  // namespace sirius::vss
