/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "io/types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace sirius::io {

//===----------------------------------------------------------------------===//
// duckdb_db_io_object
//
// `sirius_io_object` subclass wrapping a DuckDB `.db` file. Provides file
// identity to the prefetching cache so DuckDB-native byte-range reads can
// flow through the same sirius_io machinery that backs parquet / iceberg
// scans (PR #675).
//
// Identity:
//   - `raw_file_cache_id()` returns the resolved absolute file path. Stable
//     across the lifetime of the BlockManager. This is what the cache uses
//     to dedupe inserts and key the LRU.
//   - `size()` returns the on-disk file size at construction time. A live
//     DuckDB session can grow the file (BlockManager allocates new blocks on
//     write); `direct_block_scan` is a read-only OLAP path so freezing the
//     size at construction is correct for our use case.
//
// Block translation:
//   The walker that produced this object's byte ranges already converted
//   block_id → byte offset using `block_layout_info` (block_start +
//   block_id × alloc_size). This object does NOT hold the block layout — it's
//   just an identity + size handle. Translation lives upstream in
//   `direct_block_scan.cpp`.
//
//===---------------------------------------------------------------------===//
// WIP STATUS (PR H2, 2026-04-29)
//===---------------------------------------------------------------------===//
//
// In this skeleton:
//   ✅ duckdb_db_io_object class compiles + smoke-tests
//
// TODO (intentionally deferred):
//   ❌ Swap inside `direct_block_scan.cpp`: today the data path mmaps the
//      .db file and routes through `transfer_blocks_bulk_h2d` (bounce ring).
//      The substrate swap is to register byte ranges with
//      `prefetching_cache::insert(db_io_object, metadata, ranges)` and read
//      via `cache.read_ranges(...)` instead. That swap is the ACTUAL
//      perf-relevant H2 work; this scaffold is just the io_object plumbing
//      to make the swap callable.
//   ❌ Fallback policy: integrated devices (GH200) still want to skip the
//      cache entirely (system memory == device memory). Today's bounce
//      ring already has `device_is_integrated()` bypass; the sirius_io
//      path needs equivalent behavior — likely a config knob on
//      `direct_block_scan` rather than the io_object itself.
//   ❌ Live-write coherency: DuckDB transactions can grow the .db file
//      mid-query. Today's mmap path remaps on size change; sirius_io's
//      cached `size()` does not. Read-only OLAP is fine; flag for any
//      future write-aware caller.
//===----------------------------------------------------------------------===//
class duckdb_db_io_object : public sirius_io_object {
 public:
  /// Construct from a resolved absolute path. Reads file size via stat().
  /// Throws on stat failure (file missing, permission denied).
  ///
  /// TODO: a BlockManager-based ctor is the natural API for the
  /// `direct_block_scan` integration but the right BlockManager accessor
  /// for the underlying file path isn't settled yet (DuckDB's BufferManager
  /// → BlockManager → filesystem path traversal is private). Settle the
  /// API before wiring into direct_block_scan, then add an overload here.
  explicit duckdb_db_io_object(std::string absolute_path);

  ~duckdb_db_io_object() override = default;

  duckdb_db_io_object(duckdb_db_io_object const&)            = delete;
  duckdb_db_io_object& operator=(duckdb_db_io_object const&) = delete;
  duckdb_db_io_object(duckdb_db_io_object&&)                 = delete;
  duckdb_db_io_object& operator=(duckdb_db_io_object&&)      = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override
  {
    return _absolute_path;
  }

  [[nodiscard]] std::size_t size() const noexcept override { return _size_bytes; }

 private:
  std::string _absolute_path;
  std::size_t _size_bytes = 0;
};

}  // namespace sirius::io
