/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 *
 * SQL functions:
 *
 *   CALL gpu_preload_region('<table>(<col1>,<col2>,...)[; <table2>(...)]', '<encoded|decoded>');
 *   CALL gpu_release_region();
 *
 * gpu_preload_region walks every row group of every named table and either
 *   ENCODED: bulk-stages every block touched by the listed columns into one
 *            device buffer (block_id -> offset map kept on host), or
 *   DECODED: invokes gpu_decode_table per row group, retaining the resulting
 *            cudf::columns keyed by (RowGroup*, storage_col_idx).
 *
 * The result is hung off SiriusContext::active_region. The scan task consults
 * it on every batch; if the table-being-scanned is preloaded, it short-circuits
 * the H2D (encoded) or the entire decode (decoded). Released by an explicit
 * gpu_release_region() call or process exit.
 *
 * No pruning, no eviction, no admission policy. Per the experiment plan, this
 * is a per-query staging area, not a cache.
 */

#include "op/scan/gpu_preload_region.hpp"

#include "op/scan/direct_block_scan.hpp"
#include "op/scan/scan_region.hpp"
#include "sirius_context.hpp"

#include <cuda/scan/gpu_native_decode.cuh>

#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <log/logging.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

namespace {

// One parsed spec entry: a table name and the list of column names it touches.
struct table_spec {
  std::string table;
  std::vector<std::string> columns;
};

// Parse "tbl1(col1,col2,col3); tbl2(colA,colB)" into table_spec entries.
std::vector<table_spec> parse_specs(const std::string& spec)
{
  std::vector<table_spec> out;
  size_t i = 0;
  auto skip_ws = [&] {
    while (i < spec.size() && std::isspace(static_cast<unsigned char>(spec[i]))) ++i;
  };
  while (i < spec.size()) {
    skip_ws();
    if (i >= spec.size()) break;
    // Read table name up to '('.
    size_t name_start = i;
    while (i < spec.size() && spec[i] != '(' && spec[i] != ';') ++i;
    if (i >= spec.size() || spec[i] != '(') {
      throw InvalidInputException(
        "gpu_preload_region: expected '(' after table name in spec '" + spec + "'");
    }
    std::string tbl(spec.data() + name_start, i - name_start);
    // Trim trailing whitespace from table name.
    while (!tbl.empty() && std::isspace(static_cast<unsigned char>(tbl.back()))) tbl.pop_back();
    ++i;  // consume '('
    // Read column list up to ')'.
    std::vector<std::string> cols;
    while (i < spec.size() && spec[i] != ')') {
      // Read one column name (up to ',' or ')').
      size_t col_start = i;
      while (i < spec.size() && spec[i] != ',' && spec[i] != ')') ++i;
      std::string col(spec.data() + col_start, i - col_start);
      // Trim whitespace.
      while (!col.empty() && std::isspace(static_cast<unsigned char>(col.front()))) col.erase(0, 1);
      while (!col.empty() && std::isspace(static_cast<unsigned char>(col.back()))) col.pop_back();
      if (!col.empty()) cols.push_back(std::move(col));
      if (i < spec.size() && spec[i] == ',') ++i;  // consume ','
    }
    if (i >= spec.size() || spec[i] != ')') {
      throw InvalidInputException(
        "gpu_preload_region: expected ')' to close column list in spec '" + spec + "'");
    }
    ++i;  // consume ')'
    if (cols.empty()) {
      throw InvalidInputException(
        "gpu_preload_region: table '" + tbl + "' has no columns listed");
    }
    out.push_back({std::move(tbl), std::move(cols)});
    skip_ws();
    if (i < spec.size() && spec[i] == ';') {
      ++i;
      continue;
    }
    if (i < spec.size()) {
      throw InvalidInputException(
        "gpu_preload_region: unexpected trailing content in spec '" + spec + "'");
    }
  }
  if (out.empty()) {
    throw InvalidInputException("gpu_preload_region: empty spec '" + spec + "'");
  }
  return out;
}

sirius::op::scan::scan_region::mode parse_mode(const std::string& s)
{
  std::string m = s;
  std::transform(
    m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::tolower(c); });
  if (m == "encoded" || m == "enc") return sirius::op::scan::scan_region::mode::ENCODED;
  if (m == "decoded" || m == "dec") return sirius::op::scan::scan_region::mode::DECODED;
  throw InvalidInputException(
    "gpu_preload_region: mode must be 'encoded' or 'decoded', got '" + s + "'");
}

// Walk the row-group tree the same way gpu_native_scan_global_state does, so
// preloaded RowGroup* pointers match the ones the scan task will see.
std::vector<RowGroup*> walk_row_groups(DataTable& storage)
{
  std::vector<RowGroup*> out;
  auto& rg_collection = *storage.GetRowGroupCollectionRef();
  auto rg_tree        = rg_collection.GetRowGroupsDirect();
  auto node           = rg_tree->GetRootSegment();
  while (node) {
    out.push_back(&node->GetNode());
    node = rg_tree->GetNextSegment(*node);
  }
  return out;
}

// Resolve column names against the table catalog.  Returns parallel vectors of
// (storage_col_idx, logical_type).  Missing columns throw.
void resolve_columns(TableCatalogEntry& table,
                     const std::vector<std::string>& col_names,
                     std::vector<idx_t>& out_idx,
                     std::vector<LogicalType>& out_types)
{
  out_idx.reserve(col_names.size());
  out_types.reserve(col_names.size());
  for (auto const& col_name : col_names) {
    std::string mut = col_name;  // GetColumnIndex takes string& (non-const)
    auto idx = table.GetColumnIndex(mut, /*if_exists=*/true);
    if (idx.index == DConstants::INVALID_INDEX) {
      throw InvalidInputException("gpu_preload_region: column '" + col_name +
                                  "' not found in table '" + table.name + "'");
    }
    out_idx.push_back(idx.index);
    out_types.push_back(table.GetColumn(col_name).GetType());
  }
}

struct PreloadBindData : public TableFunctionData {
  std::string spec;
  sirius::op::scan::scan_region::mode mode;
  bool finished = false;
};

unique_ptr<FunctionData> PreloadBind(ClientContext& context,
                                     TableFunctionBindInput& input,
                                     vector<LogicalType>& return_types,
                                     vector<string>& names)
{
  auto data  = make_uniq<PreloadBindData>();
  data->spec = input.inputs[0].ToString();
  data->mode = parse_mode(input.inputs[1].ToString());
  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("ok");
  return std::move(data);
}

void PreloadFunction(ClientContext& context, TableFunctionInput& input, DataChunk& output)
{
  auto& data = input.bind_data->CastNoConst<PreloadBindData>();
  if (data.finished) return;

  using clock = std::chrono::steady_clock;
  auto t0     = clock::now();

  auto specs = parse_specs(data.spec);

  auto sirius_ctx_sp = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx_sp) {
    throw InvalidInputException("gpu_preload_region: SiriusContext not registered");
  }
  auto& sirius_ctx = *sirius_ctx_sp;

  auto& mem_mgr   = sirius_ctx.get_memory_manager();
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw InvalidInputException("gpu_preload_region: no GPU memory space available");
  }
  auto* gpu_space = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto mr         = gpu_space->get_default_allocator();
  auto stream     = cudf::get_default_stream();

  auto region = std::make_shared<sirius::op::scan::scan_region>();
  region->m   = data.mode;

  auto& catalog = Catalog::GetCatalog(context, INVALID_CATALOG);

  size_t total_rgs       = 0;
  size_t total_blocks    = 0;
  size_t total_bytes_pool = 0;

  for (auto const& sp : specs) {
    auto& tbl_entry =
      catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, sp.table)
        .Cast<TableCatalogEntry>();
    auto& duck_tbl = tbl_entry.Cast<DuckTableEntry>();
    auto& storage  = duck_tbl.GetStorage();

    std::vector<idx_t> col_idxs;
    std::vector<LogicalType> col_types;
    resolve_columns(tbl_entry, sp.columns, col_idxs, col_types);

    auto rgs = walk_row_groups(storage);

    sirius::op::scan::preloaded_table& pt = region->tables[&storage];

    if (data.mode == sirius::op::scan::scan_region::mode::ENCODED) {
      // One pin pass over ALL row groups for ALL listed columns, then bulk H2D.
      // Result: one rmm::device_buffer + a unified block_id->offset map.
      std::vector<sirius::op::scan::column_scan_result> col_scans(col_idxs.size());
      for (size_t ci = 0; ci < col_idxs.size(); ++ci) {
        col_scans[ci] = sirius::op::scan::direct_block_scan_column_range(
          storage, StorageIndex(col_idxs[ci]), context, rgs);
      }
      auto staged = sirius::cuda::scan::stage_blocks_bulk_h2d(col_scans, stream, mr);
      stream.synchronize();
      pt.block_offsets = std::move(staged.first);
      pt.encoded_pool  = std::move(staged.second);
      total_blocks += pt.block_offsets.size();
      total_bytes_pool += pt.encoded_pool.size();
    } else {
      // DECODED: per row group, decode every listed column into a cudf::column,
      // store keyed by (RowGroup*, storage_col_idx). The scan task reconstructs
      // the per-batch table by concatenating per-rg slices.
      pt.decoded_columns.reserve(rgs.size());
      for (auto* rg : rgs) {
        std::vector<duckdb::RowGroup*> single_rg;
        single_rg.push_back(rg);
        std::vector<sirius::op::scan::column_scan_result> col_scans(col_idxs.size());
        for (size_t ci = 0; ci < col_idxs.size(); ++ci) {
          col_scans[ci] = sirius::op::scan::direct_block_scan_column_range(
            storage, StorageIndex(col_idxs[ci]), context, single_rg);
        }
        auto tbl = sirius::cuda::scan::gpu_decode_table(col_scans, col_types, stream, mr);
        // gpu_decode_table syncs the stream internally. Release columns and
        // keep them keyed per-(rg, col).
        auto cols = tbl->release();  // vector<unique_ptr<cudf::column>> in spec order
        if (cols.size() != col_idxs.size()) {
          throw std::runtime_error(
            "gpu_preload_region: gpu_decode_table returned wrong column count");
        }
        auto& by_col = pt.decoded_columns[rg];
        for (size_t ci = 0; ci < col_idxs.size(); ++ci) {
          by_col.emplace(col_idxs[ci], std::move(cols[ci]));
        }
      }
    }

    total_rgs += rgs.size();
  }

  // Replace any previously active region. The shared_ptr drop frees device
  // memory once no scan task still holds a snapshot.
  sirius_ctx.set_active_region(region);

  auto t1     = clock::now();
  auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  const char* mode_str =
    data.mode == sirius::op::scan::scan_region::mode::ENCODED ? "encoded" : "decoded";
  SIRIUS_LOG_INFO(
    "[gpu_preload_region] mode={} {} table(s), {} rgs total, {} blocks, pool_bytes={} | took {} ms",
    mode_str,
    specs.size(),
    total_rgs,
    total_blocks,
    total_bytes_pool,
    ms);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct ReleaseBindData : public TableFunctionData {
  bool finished = false;
};

unique_ptr<FunctionData> ReleaseBind(ClientContext& context,
                                     TableFunctionBindInput& input,
                                     vector<LogicalType>& return_types,
                                     vector<string>& names)
{
  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("ok");
  return make_uniq<ReleaseBindData>();
}

void ReleaseFunction(ClientContext& context, TableFunctionInput& input, DataChunk& output)
{
  auto& data = input.bind_data->CastNoConst<ReleaseBindData>();
  if (data.finished) return;
  auto sirius_ctx_sp = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx_sp) {
    sirius_ctx_sp->set_active_region(nullptr);
    SIRIUS_LOG_INFO("[gpu_preload_region] active region released");
  }
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

}  // namespace

void RegisterGpuPreloadRegionFunctions(CatalogTransaction& transaction, Catalog& catalog)
{
  TableFunction preload("gpu_preload_region",
                        {LogicalType::VARCHAR, LogicalType::VARCHAR},
                        PreloadFunction,
                        PreloadBind);
  CreateTableFunctionInfo preload_info(preload);
  catalog.CreateTableFunction(transaction, preload_info);

  TableFunction release("gpu_release_region", {}, ReleaseFunction, ReleaseBind);
  CreateTableFunctionInfo release_info(release);
  catalog.CreateTableFunction(transaction, release_info);
}

}  // namespace duckdb
