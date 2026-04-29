/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

// Smoke tests for `duckdb_db_io_object`. WIP — exercises path-based
// construction against a tempfile.

// test
#include <catch.hpp>

// sirius
#include <io/duckdb_db_io_object.hpp>

// standard library
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace sirius;
using namespace sirius::io;

namespace {

class tempfile {
 public:
  tempfile()
  {
    _path = std::tmpnam(nullptr);
    std::ofstream(_path, std::ios::binary | std::ios::trunc) << "0123456789";  // 10 bytes
  }
  ~tempfile()
  {
    if (!_path.empty()) std::remove(_path.c_str());
  }
  tempfile(tempfile const&)            = delete;
  tempfile& operator=(tempfile const&) = delete;

  std::string const& path() const { return _path; }

 private:
  std::string _path;
};

}  // namespace

TEST_CASE("duckdb_db_io_object - path-based ctor reads size", "[io][duckdb_db_io_object]")
{
  tempfile tf;
  duckdb_db_io_object obj{tf.path()};
  REQUIRE(obj.raw_file_cache_id() == tf.path());
  REQUIRE(obj.size() == 10);
}

TEST_CASE("duckdb_db_io_object - rejects empty path", "[io][duckdb_db_io_object]")
{
  REQUIRE_THROWS_AS(duckdb_db_io_object{std::string{}}, std::invalid_argument);
}

TEST_CASE("duckdb_db_io_object - throws on missing file", "[io][duckdb_db_io_object]")
{
  REQUIRE_THROWS_AS(duckdb_db_io_object{"/this/path/does/not/exist/sirius_test"},
                    std::runtime_error);
}

// TODO end-to-end:
//   - Open a small DuckDB connection backed by a real .db file
//   - Construct duckdb_db_io_object from the BlockManager
//   - Register byte ranges with prefetching_cache, read via cache.read_ranges
//   - Compare bytes against direct file read
