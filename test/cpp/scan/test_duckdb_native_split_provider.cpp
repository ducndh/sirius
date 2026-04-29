/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

// Smoke tests for `duckdb_native_split_provider`. These exercise constructor
// validation and (TODO) a minimal end-to-end walk against a synthetic in-memory
// DuckDB table.
//
// WIP STATUS (PR H1, 2026-04-29):
//   ✅ Constructor argument validation (null storage / null context / empty
//      row_groups / parallel-vector mismatch)
//   ❌ End-to-end walk against an in-memory DataTable + assertion on the
//      emitted split's metadata. Pattern is `make_test_db_and_connection`
//      (see test/cpp/scan/test_metadata_gpu_scan_operators.cpp). Pending the
//      full provider impl (parallel dispatch + batching).

// test
#include <catch.hpp>

// sirius
#include <op/scan/duckdb_native_scan_info.hpp>
#include <scan_manager/duckdb_native_split_provider.hpp>
#include <scan_manager/split_connector.hpp>

// standard library
#include <memory>
#include <stdexcept>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;
using namespace sirius::scan_manager;

TEST_CASE("duckdb_native_split_provider - rejects null storage",
          "[scan_manager][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  info.storage = nullptr;
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider - rejects null context",
          "[scan_manager][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  // storage is uninitialized but the null check on storage runs first; to
  // exercise the context check we need storage non-null. We can't fabricate
  // a real DataTable here, so use a sentinel non-null pointer — the ctor
  // doesn't dereference it during validation.
  info.storage = reinterpret_cast<duckdb::DataTable*>(0x1);
  info.context = nullptr;
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider - rejects empty row_groups",
          "[scan_manager][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  info.storage = reinterpret_cast<duckdb::DataTable*>(0x1);
  info.context = reinterpret_cast<duckdb::ClientContext*>(0x1);
  info.row_groups.clear();
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider - rejects mismatched row_groups/row_group_starts",
          "[scan_manager][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  info.storage          = reinterpret_cast<duckdb::DataTable*>(0x1);
  info.context          = reinterpret_cast<duckdb::ClientContext*>(0x1);
  info.row_groups       = {reinterpret_cast<duckdb::RowGroup*>(0x1),
                           reinterpret_cast<duckdb::RowGroup*>(0x2)};
  info.row_group_starts = {0};
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider - rejects mismatched projected_cols/projected_types",
          "[scan_manager][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  info.storage          = reinterpret_cast<duckdb::DataTable*>(0x1);
  info.context          = reinterpret_cast<duckdb::ClientContext*>(0x1);
  info.row_groups       = {reinterpret_cast<duckdb::RowGroup*>(0x1)};
  info.row_group_starts = {0};
  info.projected_cols   = {projected_column{}, projected_column{}};
  info.projected_types  = {};
  REQUIRE_THROWS_AS(duckdb_native_split_provider{std::move(info)}, std::invalid_argument);
}

// TODO end-to-end test:
//   - make_test_db_and_connection with a small table
//   - Build duckdb_native_scan_info from table's storage/row_groups/context
//   - Construct provider, run start() against a thread_pool + split_connector
//   - Pull the split via get_next_split(), downcast to split_payload, assert
//     metadata.viable, metadata.row_groups parallel to input, etc.
