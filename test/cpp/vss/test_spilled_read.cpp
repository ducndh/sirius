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

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <vss/spilled_read.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

// A file of pseudo-random bytes under dir, removed on scope exit.
struct temp_file {
  std::string path;
  std::vector<std::uint8_t> bytes;
  temp_file(std::filesystem::path const& dir, std::size_t n)
  {
    path = (dir / ("sirius_spilled_read_" + std::to_string(::getpid()) + ".bin")).string();
    bytes.resize(n);
    std::mt19937_64 gen(42);
    for (auto& b : bytes) {
      b = static_cast<std::uint8_t>(gen());
    }
    auto const fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    REQUIRE(fd >= 0);
    REQUIRE(::write(fd, bytes.data(), n) == static_cast<ssize_t>(n));
    ::close(fd);
  }
  ~temp_file() { std::filesystem::remove(path); }
};

std::vector<std::uint8_t> read_back(int fd, std::size_t offset, std::size_t size)
{
  void* d = nullptr;
  REQUIRE(cudaMalloc(&d, size == 0 ? 1 : size) == cudaSuccess);
  auto const stream = rmm::cuda_stream_view{};
  sirius::vss::copy_file_to_device(fd, offset, size, d, stream, "test");
  std::vector<std::uint8_t> out(size);
  if (size > 0) { REQUIRE(cudaMemcpy(out.data(), d, size, cudaMemcpyDeviceToHost) == cudaSuccess); }
  cudaFree(d);
  return out;
}

void check_ranges(std::filesystem::path const& dir)
{
  // Larger than two 8 MiB pieces and not a multiple of 4 KiB, so the last direct read runs short.
  constexpr std::size_t n = (std::size_t{8} << 20) * 2 + 3 * 4096 + 123;
  temp_file const f{dir, n};
  auto const fd = ::open(f.path.c_str(), O_RDONLY);
  REQUIRE(fd >= 0);
  struct range {
    std::size_t offset, size;
  };
  // Aligned and unaligned starts, sizes inside one piece, across a piece edge, to end of file.
  for (auto const r : {range{0, n},
                       range{0, 1},
                       range{1, 4095},
                       range{4095, 2},
                       range{4096, 4096},
                       range{12345, (std::size_t{8} << 20) + 7},
                       range{(std::size_t{8} << 20) - 3, 6},
                       range{n - 1, 1},
                       range{n - 5000, 5000},
                       range{777, n - 777}}) {
    INFO("dir " << dir << " offset " << r.offset << " size " << r.size);
    auto const got = read_back(fd, r.offset, r.size);
    REQUIRE(got.size() == r.size);
    CHECK(
      std::equal(got.begin(), got.end(), f.bytes.begin() + static_cast<std::ptrdiff_t>(r.offset)));
  }
  ::close(fd);
}

}  // namespace

// The direct path (a regular filesystem) and the buffered fallback (tmpfs refuses O_DIRECT) must
// both return exactly the requested bytes, whatever the alignment of the range.
TEST_CASE("copy_file_to_device returns the requested bytes, direct or buffered",
          "[vss][spilled_read][downgrade_disk]")
{
  auto const* tmp = std::getenv("TMPDIR");
  check_ranges(tmp != nullptr ? std::filesystem::path{tmp}
                              : std::filesystem::temp_directory_path());
  if (std::filesystem::exists("/dev/shm")) { check_ranges("/dev/shm"); }
}
