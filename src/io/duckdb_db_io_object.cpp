/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "io/duckdb_db_io_object.hpp"

#include <duckdb/storage/block_manager.hpp>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>

namespace sirius::io {

namespace {

std::size_t stat_file_size(std::string const& path)
{
  struct ::stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    throw std::runtime_error(
      "[duckdb_db_io_object] stat() failed for '" + path + "': " + std::strerror(errno));
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

duckdb_db_io_object::duckdb_db_io_object(duckdb::BlockManager& block_manager)
  : duckdb_db_io_object(block_manager.GetDB().GetCatalog().GetName())
{
  // WIP: this ctor delegates to the path-based ctor via a BlockManager → path
  // lookup. The above line uses the catalog name as a stand-in; it's almost
  // certainly wrong for non-default attached databases. Once we know which
  // BlockManager API exposes the underlying file path, swap to that.
  // TODO: replace with `block_manager.GetMetadataManager().GetMetadataFile()`
  // or equivalent path accessor before wiring into direct_block_scan.
}

}  // namespace sirius::io
