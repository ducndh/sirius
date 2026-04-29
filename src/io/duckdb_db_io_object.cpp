/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "io/duckdb_db_io_object.hpp"

#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {

std::size_t stat_file_size(std::string const& path)
{
  struct ::stat st{};
  if (::stat(path.c_str(), &st) != 0) {
    throw std::runtime_error("[duckdb_db_io_object] stat() failed for '" + path +
                             "': " + std::strerror(errno));
  }
  if (st.st_size < 0) {
    throw std::runtime_error("[duckdb_db_io_object] stat() returned negative size for '" + path +
                             "'");
  }
  return static_cast<std::size_t>(st.st_size);
}

}  // namespace

duckdb_db_io_object::duckdb_db_io_object(std::string absolute_path)
  : _absolute_path(std::move(absolute_path))
{
  if (_absolute_path.empty()) {
    throw std::invalid_argument("[duckdb_db_io_object] absolute_path must be non-empty");
  }
  _size_bytes = stat_file_size(_absolute_path);
}

}  // namespace sirius::io
