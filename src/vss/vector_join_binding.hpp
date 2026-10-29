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

#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/bound_statement.hpp"
#include "vss/vector_join.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class DuckTableEntry;
class SiriusContext;
}  // namespace duckdb

namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/// TableFunction serialize callback for the vector-join table functions: the whole request,
/// so a copied plan keeps what the bind (or the plain-SQL rewrite) decided instead of re-binding.
void serialize_vector_join_bind_data(duckdb::Serializer& serializer,
                                     const duckdb::optional_ptr<duckdb::FunctionData> bind_data,
                                     const duckdb::TableFunction& function);

/// The matching deserialize callback.
duckdb::unique_ptr<duckdb::FunctionData> deserialize_vector_join_bind_data(
  duckdb::Deserializer& deserializer, duckdb::TableFunction& function);

/// Resolve one join side (left or right) at bind time, returning its vector dimensionality
/// and, through @p out_num_rows, its row count.
///
/// @param require_pin  True for a side the operator reads out of a pinned table, where the pin
///                     also decides which columns can be emitted. False for a side fed by the
///                     build phase: its scan reads the table whether or not a pin happens to be
///                     caching it, so columns come from the catalog and no pin is required.
/// Record that a statement writes to the table pinned as generation @p pin_generation (see
/// SiriusContext's prepare hooks, which already refuse an UPDATE on a pinned table). Recorded
/// when the statement is prepared, so a write that rolls back still counts.
void note_pinned_table_written(std::uint64_t pin_generation);

/// Why the vector join cannot read @p table's rows out of its pin @p pin for @p context's
/// transaction, or nullopt when it can. A duckdb pin is the table's last-checkpointed image as
/// it was pinned, deleted rows included and later inserts excluded; ordinary scans reconcile it
/// with DuckDB's visibility on every query, but the vector join reads the pinned rows as they
/// are. So this asks what the scans' own guards ask (sirius_plan_get.cpp, prepare_for_query): a
/// transaction no older than the pin, no rows inserted (in this transaction or committed past the
/// pinned prefix), and, from the scans' keep-mask walk, no pinned row this transaction cannot
/// see; and no statement written to the table since the pin. That last one stands in for the
/// scans' checkpoint check: a CHECKPOINT folds a write into the image (and can compact deleted
/// rows away) so it leaves no trace to walk, but without a write it changes none of the table's
/// rows, and a pin then serves however many checkpoints follow. A parquet pin is immutable and
/// always serves. The mask walk's clean verdict is kept per pin for as long as no commit lands.
[[nodiscard]] std::optional<std::string> pinned_rows_mismatch(
  duckdb::ClientContext& context,
  sirius::scan_manager::pinned_entry const& pin,
  duckdb::DuckTableEntry& table);

/// Bind `SELECT * FROM <view>` for a view-typed join side: the names and types the join can
/// emit, and (in the planner) the logical plan to stream.
duckdb::BoundStatement bind_view_select(duckdb::ClientContext& context,
                                        const vector_join_side& side);

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
                                      std::uint64_t& out_num_rows);

/// Resolve the probe side of the relational surface against the input relation's schema
/// instead of the catalog. There is no table to pin and no pin to restrict the columns, so the
/// relation's own columns are what can be emitted.
std::int64_t resolve_relational_probe_side(const duckdb::vector<duckdb::LogicalType>& input_types,
                                           const duckdb::vector<duckdb::string>& input_names,
                                           const std::string& column_arg,
                                           const std::vector<std::string>& out_cols,
                                           vector_join_side& side,
                                           duckdb::vector<duckdb::LogicalType>& out_types,
                                           duckdb::vector<duckdb::string>& out_names);

/// Pull a LIST(VARCHAR) named parameter into a string vector; throws if empty.
std::vector<std::string> parse_output_columns(const duckdb::Value& v, const std::string& key);

/// Rows the join emits, from the two pinned row counts and the request. Exact for
/// per-row and global top-k; an upper bound for threshold mode, whose selectivity
/// is data-dependent.
std::uint64_t estimate_vector_join_cardinality(const vector_join_request& req,
                                               std::uint64_t left_rows,
                                               std::uint64_t right_rows);

}  // namespace sirius::vss
