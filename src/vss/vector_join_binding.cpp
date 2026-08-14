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

#include "vss/vector_join_binding.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/storage/data_table.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <limits>

namespace sirius::vss {

std::int64_t resolve_vector_join_side(duckdb::ClientContext& context,
                                      duckdb::SiriusContext& sirius_ctx,
                                      const std::string& label,
                                      const std::string& table_arg,
                                      const std::string& column_arg,
                                      const std::string& schema_name,
                                      const std::vector<std::string>& out_cols,
                                      bool require_pin,
                                      vector_join_side& side,
                                      duckdb::vector<duckdb::LogicalType>& out_types,
                                      duckdb::vector<duckdb::string>& out_names,
                                      std::uint64_t& out_num_rows)
{
  side.column = column_arg;

  // Resolve the table + vector column against the catalog.
  auto const qname          = duckdb::QualifiedName::Parse(table_arg);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_name;
  auto& entry_base          = duckdb::Catalog::GetEntry(
    context, duckdb::CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  auto& entry  = entry_base.Cast<duckdb::DuckTableEntry>();
  side.catalog = entry.ParentCatalog().GetName();
  side.schema  = entry.ParentSchema().name;
  side.table   = entry.name;  // catalog-resolved name (matches query-side derivation)

  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();

  auto type_of = [&](const std::string& col) -> const duckdb::LogicalType& {
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      if (schema_names[i] == col) { return schema_types[i]; }
    }
    throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + col +
                                  "' not found in table '" + side.table + "'");
  };

  auto const& vec_type = type_of(side.column);
  if (vec_type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(vec_type).id() != duckdb::LogicalTypeId::FLOAT) {
    throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + side.column +
                                  "' must be a FLOAT[N] array column");
  }
  auto const dim = static_cast<std::int64_t>(duckdb::ArrayType::GetSize(vec_type));

  // A side the operator reads straight out of a pin can only emit what the pin holds. A side
  // fed by the build phase is read by an ordinary scan, which serves the table pinned or not --
  // the pin is a cache in front of it, not the source -- so the catalog decides the columns and
  // an absent pin is not an error. That is what stops a column-subset pin from making the rest
  // of the table unusable in the same query.
  const auto* pin = sirius_ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    side.catalog, side.schema, side.table);
  if (pin == nullptr && require_pin) {
    throw duckdb::BinderException("sirius_knn_join: " + label + " table '" + side.table +
                                  "' must be pinned");
  }
  auto const emittable = [&](const std::string& col) {
    if (pin == nullptr || !require_pin) { return true; }
    auto const& pinned_names = pin->cache_info.column_names();
    return std::ranges::find(pinned_names.begin(), pinned_names.end(), col) != pinned_names.end();
  };

  if (out_cols.empty()) {
    for (auto const& name : schema_names) {
      if (emittable(name)) { side.output_columns.push_back(name); }
    }
  } else {
    for (auto const& col : out_cols) {
      bool const in_catalog =
        std::ranges::find(schema_names.begin(), schema_names.end(), col) != schema_names.end();
      if (!in_catalog) {
        throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + col +
                                      "' not found in table '" + side.table + "'");
      }
      if (!emittable(col)) {
        throw duckdb::BinderException(
          "sirius_knn_join: " + label + " output column '" + col + "' is not pinned on table '" +
          side.table + "'; pin it (pin_table cols => [...]) or omit the output_columns list");
      }
      side.output_columns.push_back(col);
    }
  }

  // Row count for the cardinality estimate and the plan's k clamp. The pin's count is the
  // authority when the operator reads the pin; otherwise the table's own.
  out_num_rows = (pin != nullptr && require_pin)
                   ? static_cast<std::uint64_t>(pin->num_rows)
                   : static_cast<std::uint64_t>(entry.GetStorage().GetTotalRows());

  for (auto const& col : side.output_columns) {
    out_types.push_back(type_of(col));
    out_names.push_back(label + "_" + col);
  }
  return dim;
}

std::int64_t resolve_relational_probe_side(
  const duckdb::vector<duckdb::LogicalType>& input_types,
  const duckdb::vector<duckdb::string>& input_names,
  const std::string& column_arg,
  const std::vector<std::string>& out_cols,
  vector_join_side& side,
  duckdb::vector<duckdb::LogicalType>& out_types,
  duckdb::vector<duckdb::string>& out_names)
{
  side.column = column_arg;
  // catalog/schema/table stay empty: there is no table behind this side, and every path that
  // would look one up is the pinned path, which this side never takes.

  auto index_of = [&](const std::string& col) -> std::size_t {
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      if (input_names[i] == col) { return i; }
    }
    throw duckdb::BinderException("sirius_knn_join_rel: column '" + col +
                                  "' is not produced by the probe relation");
  };

  auto const& vec_type = input_types[index_of(side.column)];
  if (vec_type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(vec_type).id() != duckdb::LogicalTypeId::FLOAT) {
    throw duckdb::BinderException("sirius_knn_join_rel: probe column '" + side.column +
                                  "' must be a FLOAT[N] array column");
  }

  if (out_cols.empty()) {
    // Everything the relation produces, minus the vector column: it is the join's input, not
    // usually something the caller wants echoed back, and it is by far the widest column.
    for (auto const& name : input_names) {
      if (name != side.column) { side.output_columns.push_back(name); }
    }
  } else {
    side.output_columns = out_cols;
  }

  for (auto const& col : side.output_columns) {
    out_types.push_back(input_types[index_of(col)]);
    out_names.push_back("left_" + col);
  }
  return static_cast<std::int64_t>(duckdb::ArrayType::GetSize(vec_type));
}

std::uint64_t estimate_vector_join_cardinality(const vector_join_request& req,
                                               std::uint64_t left_rows,
                                               std::uint64_t right_rows)
{
  // create_plan_knn_join lowers k to the right-table row count; asking for more
  // neighbours than the corpus holds cannot produce more pairs than exist.
  auto const k = std::min(static_cast<std::uint64_t>(std::max<std::int64_t>(req.k, 0)), right_rows);

  // Global top-k finishes in a TOP_N above materialize, which cuts the whole
  // result to k rows regardless of how many left rows fed it.
  if (req.mode == vector_join_mode::global_top_k) { return k; }

  // Per-row emits exactly k rows per left row. Threshold searches to the same
  // depth and then drops pairs outside eps, so this is its ceiling, not its
  // expectation — no selectivity is knowable at bind time.
  constexpr auto max_rows = std::numeric_limits<std::uint64_t>::max();
  if (k != 0 && left_rows > max_rows / k) { return max_rows; }
  return left_rows * k;
}

std::vector<std::string> parse_output_columns(const duckdb::Value& v, const std::string& key)
{
  std::vector<std::string> out;
  for (auto const& c : duckdb::ListValue::GetChildren(v)) {
    out.push_back(c.ToString());
  }
  if (out.empty()) {
    throw duckdb::BinderException("sirius_knn_join: " + key +
                                  " cannot be empty; omit it to default to the pinned columns");
  }
  return out;
}

}  // namespace sirius::vss
