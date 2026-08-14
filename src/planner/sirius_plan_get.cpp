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

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/segment/uncompressed.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/string_stats.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/sirius_physical_vector_join_materialize.hpp"
#include "vss/sirius_physical_vector_join_reduce_local.hpp"
#include "vss/sirius_physical_vector_join_select.hpp"
#include "vss/sirius_physical_vector_join_stream.hpp"
#include "vss/vector_join.hpp"

#include <string_view>

#include <cstdlib>

#include <memory>
#include <unordered_set>

namespace sirius::planner {

namespace {

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner
// boundary. The source vector is drained; size and order are preserved, with a
// null slot wherever from_duckdb declines an unsupported shape (a fallback
// signal) — matching the prior bulk-translation null-skip semantics.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}

//! A scan of the corpus table projected to its vector column, built the way DuckDB builds a
//! base-table scan (`GetScanFunction` mints the same bind data its binder would) rather than
//! bound from SQL. The vector join's LogicalGet has no children -- it is a table function, and
//! a table-in-out surface is a separate piece of work -- so the build phase's input is
//! constructed here instead. The projection is the vector column alone, which is why the fold
//! can take column 0 without a name lookup.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_corpus_scan(
  duckdb::ClientContext& context, const sirius::vss::vector_join_request& req)
{
  auto& entry_base = duckdb::Catalog::GetEntry(
    context, duckdb::CatalogType::TABLE_ENTRY, req.right.catalog, req.right.schema, req.right.table);
  auto& table_entry = entry_base.Cast<duckdb::DuckTableEntry>();

  auto const& columns = table_entry.GetColumns();
  auto const names    = columns.GetColumnNames();
  auto const types    = columns.GetColumnTypes();

  auto const it = std::find(names.begin(), names.end(), req.right.column);
  if (it == names.end()) {
    throw duckdb::InternalException("sirius_knn_join: corpus column '" + req.right.column +
                                    "' vanished between bind and plan");
  }
  auto const vec_idx = static_cast<duckdb::idx_t>(std::distance(names.begin(), it));

  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  auto scan_function = table_entry.GetScanFunction(context, bind_data);

  // Vector column first, then the columns the join emits. Materialize concatenates those in
  // this same batch order to resolve neighbour ids, so they have to travel in the same batches
  // the fold numbered -- not be re-read from the table afterwards.
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(vec_idx);
  duckdb::vector<duckdb::LogicalType> projected_types{types[vec_idx]};
  for (auto const& out_col : req.right.output_columns) {
    auto const out_it = std::find(names.begin(), names.end(), out_col);
    if (out_it == names.end()) {
      throw duckdb::InternalException("sirius_knn_join: corpus output column '" + out_col +
                                      "' vanished between bind and plan");
    }
    auto const idx = static_cast<duckdb::idx_t>(std::distance(names.begin(), out_it));
    column_ids.emplace_back(idx);
    projected_types.push_back(types[idx]);
  }

  return duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(projected_types),
    scan_function,
    std::move(bind_data),
    sirius::from_duckdb_vec(types),
    std::move(column_ids),
    duckdb::vector<std::size_t>{},
    names,
    /*table_filters=*/nullptr,
    table_entry.GetStorage().GetTotalRows(),
    duckdb::ExtraOperatorInfo{},
    duckdb::vector<duckdb::Value>{},
    table_entry.GetVirtualColumns());
}

}  // namespace

duckdb::unique_ptr<duckdb::TableFilterSet> create_table_filter_set(
  duckdb::TableFilterSet& table_filters, const duckdb::vector<duckdb::ColumnIndex>& column_ids)
{
  // create the table filter map
  auto table_filter_set = duckdb::make_uniq<duckdb::TableFilterSet>();
  for (auto& table_filter : table_filters.filters) {
    // find the relative column index from the absolute column index into the table
    duckdb::optional_idx column_index;
    for (std::size_t i = 0; i < column_ids.size(); i++) {
      if (table_filter.first == column_ids[i].GetPrimaryIndex()) {
        column_index = i;
        break;
      }
    }
    if (!column_index.IsValid()) {
      throw duckdb::InternalException("Could not find column index for table filter");
    }
    table_filter_set->filters[column_index.GetIndex()] = std::move(table_filter.second);
  }
  return table_filter_set;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalGet& op)
{
  // sirius_knn_join produces its join result from two pinned tables, so it gets its own leaf
  // builder rather than being routed through the scan path.
  if (op.function.name == "sirius_knn_join") { return create_plan_knn_join(op); }

  auto column_ids = op.GetColumnIds();

  // Only GPU-route known table scan functions; all others (pragma, system catalog
  // functions, etc.) must fall back to CPU.
  static const std::unordered_set<std::string> kSupportedScanFunctions = {
    "seq_scan", "parquet_scan", "read_parquet", "sirius_read_parquet"};
  if (kSupportedScanFunctions.find(op.function.name) == kSupportedScanFunctions.end()) {
    throw duckdb::NotImplementedException("Table function '{}' is not supported in Sirius",
                                          op.function.name);
  }

  if (!op.children.empty()) {
    throw duckdb::NotImplementedException("Table Input Output functions are not supported yet");
  }

  if (!op.projected_input.empty()) {
    throw duckdb::InternalException(
      "LogicalGet::project_input can only be set for table-in-out functions");
  }

  // Plan-time probe for the duckdb-native seq_scan path: strings at/over
  // StringUncompressed::GetStringBlockLimit (a per-value limit) live in overflow
  // blocks the GPU string decoder cannot resolve. Refuse HERE, where the throw still
  // becomes a clean CPU fallback — the walker's refusal at pipeline conversion
  // surfaces as a mid-query error with none. Conservative for DICT_FSST, which
  // inlines strings up to 16 KiB (see prepare_duckdb_native_walk).
  if (op.function.name == "seq_scan" && op.bind_data) {
    auto* table_scan_bind = dynamic_cast<duckdb::TableScanBindData*>(op.bind_data.get());
    if (table_scan_bind != nullptr && table_scan_bind->table.IsDuckTable()) {
      auto& table   = table_scan_bind->table.Cast<duckdb::DuckTableEntry>();
      auto& storage = table.GetStorage();
      auto const block_size =
        storage.GetAttached().GetStorageManager().GetBlockManager().GetBlockSize();
      auto const overflow_limit = duckdb::StringUncompressed::GetStringBlockLimit(block_size);
      for (auto const& col_idx : column_ids) {
        if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() || col_idx.IsVirtualColumn() ||
            col_idx.IsEmptyColumn()) {
          continue;
        }
        auto const primary = col_idx.GetPrimaryIndex();
        if (primary >= op.returned_types.size() ||
            op.returned_types[primary].id() != duckdb::LogicalTypeId::VARCHAR) {
          continue;
        }
        auto stats = table.GetStatistics(context, primary);
        if (!stats || !duckdb::StringStats::HasMaxStringLength(*stats) ||
            duckdb::StringStats::MaxStringLength(*stats) >= overflow_limit) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: varchar column %llu may contain strings at/over the "
            "overflow-block limit (%llu bytes); overflow strings are not GPU-decodable",
            static_cast<unsigned long long>(primary),
            static_cast<unsigned long long>(overflow_limit));
        }
      }

      // Sentinel columns (rowid/virtual/empty/field) have no storage backing,
      // which makes a pin unservable for this scan.
      bool has_unservable_column = false;
      for (auto const& col_idx : column_ids) {
        if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() || col_idx.IsVirtualColumn() ||
            col_idx.IsEmptyColumn()) {
          has_unservable_column = true;
          break;
        }
      }

      sirius::scan_manager::pinned_entry const* entry = nullptr;
      if (context.registered_state) {
        if (auto sirius_state =
              context.registered_state->Get<duckdb::SiriusContext>("sirius_state")) {
          entry = sirius_state->get_scan_manager().find_pinned_entry_for_duckdb_table(
            table.ParentCatalog().GetName(), table.ParentSchema().name, table.name);
        }
      }
      if (entry != nullptr && entry->mvcc != nullptr) {
        // Cache-or-CPU guards: while this table is MVCC-pinned, a GPU plan
        // either serves exactly from the pinned cache (DELETE keep-masks) or is
        // refused HERE, where the throw still becomes a clean CPU fallback. The
        // disk-native path is MVCC-blind, and the pin's checkpoint suppression
        // makes its snapshot increasingly stale — so scans the pin cannot serve
        // never fall through to it.
        auto const n_cache = entry->mvcc->n_cache();
        // (a) snapshot-too-old: this transaction opened before the pin, so
        // the cache's base image is from its future.
        auto const start_time =
          duckdb::DuckTransaction::Get(context, table.ParentCatalog()).start_time;
        if (start_time < entry->mvcc->v_base) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: transaction snapshot (%llu) predates the pinned cache "
            "snapshot (%llu) for table '%s'",
            static_cast<unsigned long long>(start_time),
            static_cast<unsigned long long>(entry->mvcc->v_base),
            table.name);
        }
        // (b) insert-present: committed rows beyond the pinned prefix, or
        // this transaction's own uncommitted appends (transaction-local
        // storage) — not served from the cache until the insert-delta
        // reader lands.
        if (static_cast<std::size_t>(storage.GetTotalRows()) > n_cache) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has rows beyond the %llu pinned at pin time; "
            "post-pin inserts are not served from the cache yet",
            table.name,
            static_cast<unsigned long long>(n_cache));
        }
        if (duckdb::LocalStorage::Get(context, storage.GetAttached()).GetStorage(storage)) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has uncommitted appends in this transaction; "
            "transaction-local inserts are not served from the cache",
            table.name);
        }
        bool pin_serves = !has_unservable_column;
        if (pin_serves && !column_ids.empty()) {
          pin_serves = !entry->cache_info.column_projection_for(column_ids).empty();
        }
        if (!pin_serves) {
          // (d) column-mismatch: the scan would fall through to the MVCC-blind
          // disk-native read, so it always declines. The disabled clean-table
          // relaxation below lets a table that provably matches its
          // last-checkpointed image fall through instead (#1160).
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' is MVCC-pinned and the pin cannot serve the "
            "requested columns",
            table.name);
        }
#if 0
        // Disabled — these guards walk every row group of the table at plan
        // time, per query; #1160 tracks running the checks from the scan
        // manager at execution time instead. With this block off, (d) above
        // has no clean-table relaxation and (c) is unguarded: post-pin UPDATE
        // chains serve stale cached values (UPDATE support is #1162; pin_table
        // still refuses tables that already carry update chains).
        std::vector<duckdb::storage_t> projected;
        for (auto const& col_idx : column_ids) {
          if (col_idx.HasPrimaryIndex() && !col_idx.IsRowIdColumn() &&
              !col_idx.IsVirtualColumn() && !col_idx.IsEmptyColumn()) {
            projected.push_back(col_idx.GetPrimaryIndex());
          }
        }
        if (!pin_serves) {
          // (d) column-mismatch: the scan falls through to the disk-native
          // read, so it must pass the same exactness check as an unpinned
          // scan. Guards (a)/(b) already excluded post-pin inserts and
          // transaction-local appends.
          auto& txn = duckdb::DuckTransaction::Get(context, table.ParentCatalog());
          if (sirius::op::scan::check_native_read_mvcc_state(
                storage, projected, duckdb::TransactionData(txn)) !=
              sirius::op::scan::native_read_mvcc_state::exact) {
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' is MVCC-pinned, the pin cannot serve the "
              "requested columns, and the table has diverged from its last-checkpointed "
              "image",
              table.name);
          }
        } else if (sirius::op::scan::any_update_chains(storage, projected, n_cache)) {
          // (c) update-present on a column the cache would serve: update
          // chains version values in place, invisibly to the DELETE
          // keep-masks — the cached values would be stale.
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has in-memory update chains on a scanned "
            "column; updated values are not served from the cache",
            table.name);
        }
#endif
      }
#if 0
      // Disabled — plan-time MVCC guards for duckdb-native scans of unpinned
      // tables: the exactness walk touches every row group of the table, per
      // query; #1160 tracks running it from the scan manager at execution
      // time instead. With this block off, the disk-native read of an
      // unpinned table is MVCC-blind (#1143): uncheckpointed deletes and
      // update chains are served silently, and committed-but-uncheckpointed
      // inserts fail loudly at execution.
      if (entry == nullptr || entry->mvcc == nullptr) {
        // No MVCC-pinned cache for this table: the plan is the disk-native
        // read, which applies no visibility filtering — refuse any state it
        // would misread HERE, where the throw still becomes a clean CPU
        // fallback. Residual race: a row committing between this check and
        // the scan's metadata capture is read unmasked; the prepare-time
        // keep-masks planned in #1143 close it.
        std::vector<duckdb::storage_t> projected;
        for (auto const& col_idx : column_ids) {
          if (col_idx.HasPrimaryIndex() && !col_idx.IsRowIdColumn() &&
              !col_idx.IsVirtualColumn() && !col_idx.IsEmptyColumn()) {
            projected.push_back(col_idx.GetPrimaryIndex());
          }
        }
        if (duckdb::LocalStorage::Get(context, storage.GetAttached()).GetStorage(storage)) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has uncommitted appends in this transaction; "
            "the disk-native read cannot see transaction-local rows",
            table.name);
        }
        auto& txn = duckdb::DuckTransaction::Get(context, table.ParentCatalog());
        switch (sirius::op::scan::check_native_read_mvcc_state(
          storage, projected, duckdb::TransactionData(txn))) {
          case sirius::op::scan::native_read_mvcc_state::has_update_chains:
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' has in-memory update chains on a scanned "
              "column; the disk-native read would return stale values",
              table.name);
          case sirius::op::scan::native_read_mvcc_state::has_invisible_rows:
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' has rows not visible to this transaction "
              "(uncheckpointed deletes or in-flight inserts); the disk-native read is "
              "MVCC-blind",
              table.name);
          case sirius::op::scan::native_read_mvcc_state::exact: break;
        }
      }
#endif
    }
  }

  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  if (!op.table_filters.filters.empty()) {
    table_filters = create_table_filter_set(op.table_filters, column_ids);
  }

  if (op.function.dependency) { op.function.dependency(dependencies, op.bind_data.get()); }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> filter;
  auto& projection_ids = op.projection_ids;

  // With FILTER_PUSHDOWN enabled, filters from WHERE clauses are pushed into table_filters.
  // Since we don't pass filters to the DuckDB table function (they're applied by Sirius),
  // we need to ensure all filter columns are included in BOTH column_ids and projection_ids.
  // We track the original projection_ids so we can project back after filtering.
  duckdb::vector<std::size_t> original_projection_ids = projection_ids;

  // Save the original types before we modify projection_ids, because modifying projection_ids
  // might affect the types when we call ResolveOperatorTypes()
  duckdb::vector<duckdb::LogicalType> original_types = op.types;

  if (table_filters) {
    for (auto& entry : table_filters->filters) {
      // entry.first is the column index in the table_filters (after remapping by
      // create_table_filter_set) We need to ensure this column is in projection_ids so it gets
      // scanned by DuckDB

      bool found_in_projection = false;
      for (std::size_t j = 0; j < projection_ids.size(); j++) {
        if (projection_ids[j] == entry.first) {
          found_in_projection = true;
          break;
        }
      }

      if (!found_in_projection) { projection_ids.push_back(entry.first); }
    }
  }

  // Handle cases where table function doesn't support pushdown for specific column types
  if (table_filters && op.function.supports_pushdown_type) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::unordered_set<std::size_t> to_remove;
    for (auto& entry : table_filters->filters) {
      auto column_id = column_ids[entry.first].GetPrimaryIndex();
      auto& type     = op.returned_types[column_id];

      // If the table function doesn't support pushdown for this column type,
      // create a separate filter operator for it
      if (!op.function.supports_pushdown_type(*op.bind_data, column_id)) {
        std::size_t column_id_filter = entry.first;
        auto column = duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column_id_filter);
        select_list.push_back(entry.second->ToExpression(*column));
        to_remove.insert(entry.first);
      }
    }
    for (auto& col : to_remove) {
      table_filters->filters.erase(col);
    }

    if (!select_list.empty()) {
      duckdb::vector<duckdb::LogicalType> filter_types;
      for (auto& c : projection_ids) {
        auto column_id = column_ids[c].GetPrimaryIndex();
        filter_types.push_back(op.returned_types[column_id]);
      }
      // sirius_physical_filter owns a single expression; AND-merge predicates when there are many.
      duckdb::unique_ptr<duckdb::Expression> combined;
      if (select_list.size() > 1) {
        auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
          duckdb::ExpressionType::CONJUNCTION_AND);
        for (auto& expr : select_list) {
          conjunction->children.push_back(std::move(expr));
        }
        combined = std::move(conjunction);
      } else {
        combined = std::move(select_list[0]);
      }
      filter =
        duckdb::make_uniq<sirius::op::sirius_physical_filter>(sirius::from_duckdb_vec(filter_types),
                                                              sirius::ast::from_duckdb(*combined),
                                                              op.estimated_cardinality);
    }
  }
  op.ResolveOperatorTypes();
  // create the table scan node
  if (!op.function.projection_pushdown) {
    // function does not support projection pushdown
    auto node = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
      sirius::from_duckdb_vec(op.returned_types),
      op.function,
      std::move(op.bind_data),
      sirius::from_duckdb_vec(op.returned_types),
      column_ids,
      duckdb::vector<duckdb::column_t>(),
      op.names,
      std::move(table_filters),
      op.estimated_cardinality,
      std::move(op.extra_info),
      std::move(op.parameters),
      std::move(op.virtual_columns));
    node->named_parameters = std::move(op.named_parameters);
    // first check if an additional projection is necessary
    if (column_ids.size() == op.returned_types.size()) {
      bool projection_necessary = false;
      for (std::size_t i = 0; i < column_ids.size(); i++) {
        if (column_ids[i].GetPrimaryIndex() != i) {
          projection_necessary = true;
          break;
        }
      }
      if (!projection_necessary) {
        // a projection is not necessary if all columns have been requested in-order
        // in that case we just return the node
        if (filter) {
          filter->children.push_back(std::move(node));
          return filter;
        }
        return std::move(node);
      }
    }
    // push a projection on top that does the projection
    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    for (std::size_t i = 0; i < column_ids.size(); ++i) {
      auto& column_id = column_ids[i];
      if (column_id.IsVirtualColumn()) {
        throw duckdb::NotImplementedException("Virtual columns require projection pushdown");
      } else {
        auto col_id = column_id.GetPrimaryIndex();
        auto type   = op.returned_types[col_id];
        types.push_back(type);
        // The Sirius scan emits exactly the column_ids columns, in order, at
        // positions 0..M-1 (build_scan_plan with empty projection_ids) — unlike
        // DuckDB's native full-width scan this branch was modeled on.  So
        // reference the column by its position i in the scan output, not by its
        // original parquet index col_id, which can exceed the M-column width.
        expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, i));
      }
    }
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> scan_child;
    if (filter) {
      filter->children.push_back(std::move(node));
      scan_child = std::move(filter);
    } else {
      scan_child = std::move(node);
    }
    return push_projection(std::move(scan_child),
                           sirius::from_duckdb_vec(types),
                           translate_expressions(std::move(expressions)),
                           op.estimated_cardinality);
  }

  auto node = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(original_types),  // Use original types, not modified
    op.function,
    std::move(op.bind_data),
    sirius::from_duckdb_vec(op.returned_types),
    column_ids,
    op.projection_ids,
    op.names,
    std::move(table_filters),
    op.estimated_cardinality,
    std::move(op.extra_info),
    std::move(op.parameters),
    std::move(op.virtual_columns));
  node->named_parameters = std::move(op.named_parameters);
  node->dynamic_filters  = op.dynamic_filters;
  if (op.dynamic_filters) {
    node->sirius_dynamic_filters = get_or_create_dynamic_filter_channel(op.dynamic_filters.get());
    SIRIUS_LOG_INFO("[sirius_plan_get] LogicalGet has dynamic_filters attached (channel key={}).",
                    static_cast<void const*>(op.dynamic_filters.get()));
  }
  if (filter) {
    filter->children.push_back(std::move(node));
    return filter;
  }
  return std::move(node);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan_knn_join(duckdb::LogicalGet& op)
{
  if (!op.children.empty()) {
    throw duckdb::NotImplementedException("sirius_knn_join does not take table inputs");
  }
  op.ResolveOperatorTypes();

  auto const& bind_data = op.bind_data->Cast<sirius::vss::SiriusVectorJoinBindData>();

  // The search stage reads both pinned tables directly, so it needs the scan
  // manager to resolve them at execution time.
  auto sirius_state = context.registered_state
                        ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                        : nullptr;
  if (!sirius_state) {
    throw duckdb::InternalException(
      "sirius_knn_join requires the Sirius context to be initialized");
  }
  auto& scan_manager = sirius_state->get_scan_manager();

  // If k is larger than the number of rows in the right table, lower it to that row count.
  auto req = bind_data.req;
  {
    const auto* right_pin = scan_manager.find_pinned_entry_for_duckdb_table(
      req.right.catalog, req.right.schema, req.right.table);
    if (right_pin != nullptr) {
      auto const right_rows = static_cast<std::int64_t>(right_pin->num_rows);
      if (req.k > right_rows) { req.k = right_rows; }
    }
  }

  // Three-stage pipeline: select (per-pair top-k) → reduce_local (per-left-batch
  // reduction) → materialize (gather output columns + score into the TVF rows).
  // The select/reduce stages carry a [neighbor_id BIGINT, distance FLOAT] schema;
  // only materialize emits the TVF's declared columns.
  // Per-pair partials, before any left row is resolved: [neighbor_id, distance].
  auto partial_types = []() {
    duckdb::vector<sirius::logical_type> t;
    t.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    t.push_back(sirius::logical_type::make(sirius::type_id::FLOAT));
    return t;
  };

  // What the join stage hands to materialize: [left_row, neighbor_id, distance].
  //
  // left_row is carried explicitly rather than implied by position. Under per-row top-k the
  // result is exactly k rows per left row and the left index can be recovered as row/k, which
  // is what materialize used to do. Threshold and global top-k break that: their output is
  // ragged by construction, so position carries no information and the left row has to travel
  // with the pair. Making it explicit for every mode is what keeps one materialize.
  auto joined_types = []() {
    duckdb::vector<sirius::logical_type> t;
    t.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    t.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    t.push_back(sirius::logical_type::make(sirius::type_id::FLOAT));
    return t;
  };

  // The fused streaming operator folds each right batch into a running top-k instead
  // of emitting a partial per (left, right) pair for a separate reduce stage. Both
  // paths emit the same [neighbor_id, distance] schema partitioned by left batch, so
  // materialize is shared and the two can be compared directly on the same query.
  //
  // Streaming is the default. The split path is kept for A/B comparison only and is
  // known to be wrong whenever the right table pins to more than one chunk: reduce_local
  // merges each partial on its own instead of folding the partition, and materialize's
  // fixed-k slicing then attributes neighbours to the wrong left rows. Select it with
  // SIRIUS_VECTOR_JOIN_STREAMING=0, and only on a single-chunk right table.
  const char* streaming_env = std::getenv("SIRIUS_VECTOR_JOIN_STREAMING");
  bool const use_streaming =
    streaming_env == nullptr || std::string_view{streaming_env} != "0";

  // The build phase's corpus row order, minted here so the fold and materialize resolve
  // neighbour ids against one list rather than each deriving one of their own.
  std::shared_ptr<sirius::vss::build_side_buffer> build_side =
    req.build_from_scan ? std::make_shared<sirius::vss::build_side_buffer>() : nullptr;

  if (build_side && !use_streaming) {
    throw duckdb::NotImplementedException(
      "sirius_knn_join: build_source => 'scan' needs the streaming operator; the split path "
      "(SIRIUS_VECTOR_JOIN_STREAMING=0) reads the corpus from a pin only");
  }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> join_stage;
  if (use_streaming) {
    auto stream_op = duckdb::make_uniq<sirius::op::sirius_physical_vector_join_stream>(
      joined_types(), op.estimated_cardinality, req, &scan_manager, build_side);
    if (req.build_from_scan) {
      stream_op->children.push_back(make_corpus_scan(context, req));
    }
    join_stage = std::move(stream_op);
  } else {
    auto selection = duckdb::make_uniq<sirius::op::sirius_physical_vector_join_select>(
      partial_types(), op.estimated_cardinality, req, &scan_manager);

    auto reduce_local = duckdb::make_uniq<sirius::op::sirius_physical_vector_join_reduce_local>(
      joined_types(), op.estimated_cardinality, req.k);

    reduce_local->children.push_back(std::move(selection));
    join_stage = std::move(reduce_local);
  }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> node =
    duckdb::make_uniq<sirius::op::sirius_physical_vector_join_materialize>(
      sirius::from_duckdb_vec(op.returned_types),
      op.estimated_cardinality,
      req,
      &scan_manager,
      build_side);
  node->children.push_back(std::move(join_stage));

  // Global top-k finishes above materialize rather than inside the join. The join stage
  // searches every left row to depth k, which bounds the candidates correctly (one left row
  // can own at most k of the global winners) but leaves one top-k per left batch. Ranking the
  // materialized rows by score with an ordinary TOP_N collapses those into a single answer,
  // and the plan generator's post-pass wraps any TOP_N with MERGE_TOP_N, which is exactly the
  // cross-partition merge this needs — so left-side partitioning is not a restriction here.
  if (req.mode == sirius::vss::vector_join_mode::global_top_k) {
    auto const score_idx = op.returned_types.size() - 1;  // materialize emits score last
    auto const ascending =
      req.output_type == sirius::vss::vector_join_output_type::similarity ? false : true;

    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.emplace_back(ascending ? duckdb::OrderType::ASCENDING : duckdb::OrderType::DESCENDING,
                        duckdb::OrderByNullType::NULLS_LAST,
                        duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                          op.returned_types[score_idx], score_idx));

    auto top_n = duckdb::make_uniq<sirius::op::sirius_physical_top_n>(
      sirius::from_duckdb_vec(op.returned_types),
      std::move(orders),
      static_cast<std::size_t>(req.k),
      /*offset=*/std::size_t{0},
      /*dynamic_filter=*/nullptr,
      op.estimated_cardinality);
    top_n->children.push_back(std::move(node));
    node = std::move(top_n);
  }

  auto column_ids  = op.GetColumnIds();
  bool is_identity = column_ids.size() == op.returned_types.size();
  for (std::size_t i = 0; is_identity && i < column_ids.size(); ++i) {
    if (!column_ids[i].HasPrimaryIndex() || column_ids[i].GetPrimaryIndex() != i) {
      is_identity = false;
    }
  }
  if (is_identity) { return node; }

  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  for (auto const& column_id : column_ids) {
    if (!column_id.HasPrimaryIndex()) {
      throw duckdb::NotImplementedException("sirius_knn_join: virtual/rowid columns unsupported");
    }
    auto const col_id = column_id.GetPrimaryIndex();
    auto const type   = op.returned_types[col_id];
    types.push_back(type);
    expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, col_id));
  }
  return push_projection(std::move(node),
                         sirius::from_duckdb_vec(types),
                         translate_expressions(std::move(expressions)),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner
