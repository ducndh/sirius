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

#include "duckdb/main/database.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/open_file_info.hpp"
#include "expression_evaluator/expression_evaluator_strategy.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

// Forward-declare CUDA profiler API functions (linked via libcudart).
extern "C" int cudaProfilerStart();
extern "C" int cudaProfilerStop();
#include "data/sirius_converter_registry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/column_list.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
// #include "from_substrait.hpp"
#ifdef SIRIUS_ENABLE_LEGACY
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#endif
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/result/host_table_chunk_reader.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "pin_table.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "sirius_interface.hpp"
#include "sirius_sql_rewrite.hpp"
#include "util/segfault_backtrace.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/distance_metric.hpp"
#include "vss/ivf_flat_index.hpp"
#include "vss/kmeans_functions.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_join_binding.hpp"
#include "vss/vector_search.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cmath>

// PinTableFunction routes parquet reads through the per-GPU sirius_ioctx
// instead of cudf's bundled file_source factory (which uses kvikio internally
// and binds to a single CUDA context). This is mandatory in multi-GPU
// configurations (enforced by sirius_config::enforce_sirius_datasource_for_multi_gpu()).
// Single-GPU users may still opt out via use_sirius_datasource=false; the
// pin pipeline always routes through sirius_ioctx when one is available.
//
// Ordering rule: include uring_reactor LAST among sirius headers — liburing.h
// transitively pulled by uring_reactor.hpp defines a BLOCK_SIZE preprocessor
// macro that collides with the BLOCK_SIZE static member in
// <blockingconcurrentqueue.h> (used by pipeline / duckdb
// connection_manager). All consumers of blockingconcurrentqueue.h must
// precede this include.
#include "io/s3/sirius_httpfs.hpp"     // sirius::io::s3::sirius_httpfs
#include "io/types.hpp"                // sirius::io::sirius_ioctx
#include "io/uring/uring_reactor.hpp"  // sirius::io::uring_io_object

#include <algorithm>
#include <cstdlib>
#include <unordered_map>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY = "pinned_memory_size";
#ifdef SIRIUS_ENABLE_LEGACY
bool SiriusExtension::buffer_is_initialized = false;
#endif

constexpr std::string QUERY_LABEL_PARAM_KEY = "query_label";

namespace {

unique_ptr<QueryResult> run_internal_cpu_fallback_query(ClientContext& context,
                                                        Connection& connection,
                                                        const string& query,
                                                        const string& gpu_error = "")
{
  // S3 CPU fallback is not supported. Sirius reads s3:// only on the GPU path
  // (sirius_read_parquet -> describe_parquet -> cuDF via s3_ioctx); DuckDB's CPU
  // read_parquet has no S3 filesystem, so a query that reads s3:// cannot fall
  // back to CPU. Surface a clear error (with the underlying GPU cause) instead
  // of replaying a query that would fail anyway. Local / non-s3 queries are
  // unaffected and fall through to the normal DuckDB CPU replay below.
  if (sirius::references_sirius_owned_s3_parquet(query)) {
    throw std::runtime_error(
      "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, and "
      "Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
      gpu_error);
  }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return connection.Query(query); }

  // CpuFallbackGuard marks this replay so sirius_httpfs refuses to serve s3://
  // data reached indirectly (e.g. through a view) to the CPU plan — the
  // string-level references_sirius_owned_s3_parquet check above only catches a
  // literal read_parquet('s3://') in the query text.
  duckdb::SiriusContext::InternalQueryGuard guard(*sirius_ctx);
  duckdb::SiriusContext::CpuFallbackGuard cpu_fallback_guard(*sirius_ctx);
  return connection.Query(query);
}

// Bind callback for the sirius_read_parquet table function — a thin forwarder.
// It resolves the URI to the connection's scan_manager and probes the parquet
// footer through describe_parquet (footer-only, no full-file download), then
// hands the inferred schema back to DuckDB. Bind data carries the URI and
// footer row count so the cardinality callback can expose a real estimate to
// the optimizer; the pipeline converter still reads the URI from parameters[0].
unique_ptr<FunctionData> SiriusReadParquetBind(ClientContext& context,
                                               TableFunctionBindInput& input,
                                               vector<LogicalType>& return_types,
                                               vector<string>& names)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
    throw std::runtime_error("sirius_read_parquet expects a single non-null parquet URI");
  }
  auto const uri = input.inputs[0].GetValue<std::string>();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error("sirius_read_parquet: Sirius is not initialized on this connection");
  }

  auto bind_result = sirius_ctx->get_scan_manager().describe_parquet(uri);
  return_types     = std::move(bind_result.return_types);
  names            = std::move(bind_result.names);
  return make_uniq<SiriusReadParquetBindData>(uri, bind_result.total_num_rows);
}

// Execute callback for sirius_read_parquet. The real scan runs through the
// Sirius GPU pipeline; this table function is an internal rewrite target for
// read_parquet('s3://...') inside gpu_execution, NOT a user-facing function.
// A direct DuckDB (CPU) execution is rejected cleanly — query S3 Parquet via
// read_parquet('s3://...'), which the bind-time rewrite routes here for the GPU
// path. S3 has no CPU fallback: if GPU execution fails, an s3:// query errors
// (S3 CPU fallback is not supported) rather than replaying on the CPU.
void SiriusReadParquetFunction(ClientContext&, TableFunctionInput&, DataChunk&)
{
  throw std::runtime_error(
    "sirius_read_parquet is an internal rewrite target; query S3 Parquet with "
    "read_parquet('s3://...') inside gpu_execution()");
}

}  // namespace

unique_ptr<NodeStatistics> SiriusReadParquetCardinality(ClientContext&,
                                                        FunctionData const* bind_data_p)
{
  if (bind_data_p == nullptr) { return nullptr; }
  auto const* typed = dynamic_cast<SiriusReadParquetBindData const*>(bind_data_p);
  if (typed == nullptr) { return nullptr; }
  return make_uniq<NodeStatistics>(typed->total_num_rows, typed->total_num_rows);
}

struct SiriusTableFunctionData : public TableFunctionData {
  SiriusTableFunctionData() = default;
  shared_ptr<::sirius::sirius_prepared_statement_data> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<::sirius::sirius_interface> sirius_iface;
  string query;
  // Pre-rewrite query used for the CPU fallback of LOCAL (non-s3) reads. The GPU
  // path runs the rewritten `query` (read_parquet('s3://…') ->
  // sirius_read_parquet('s3://…')), which throws if executed on the CPU, so a
  // fallback replays this original instead. For local / non-s3 queries the
  // rewrite is a no-op, so this equals `query` and replays normally. For s3://
  // queries there is no CPU fallback: run_internal_cpu_fallback_query detects the
  // s3:// read and raises a clear "S3 CPU fallback is not supported" error.
  string cpu_fallback_query;
  bool enable_optimizer;
  bool finished   = false;
  bool plan_error = false;
  // Real error message from a failed GPU plan generation. The CPU fallback path
  // surfaces it so the true cause (e.g. the unsupported operator) is preserved
  // instead of a generic placeholder.
  string plan_error_message;
  //! Original options from the connection
  ClientConfig original_config;
  set<OptimizerType> original_disabled_optimizers;

  void PrepareConnection(ClientContext& context)
  {
    // First collect original options
    original_config              = context.config;
    original_disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;

    // The user might want to disable the optimizer of the new connection
    context.config.enable_optimizer = enable_optimizer;
    // We want for sure to disable the internal compression optimizations.
    // These are DuckDB specific, no other system implements these. Also,
    // respect the user's settings if they chose to disable any specific optimizers.
    //
    // The InClauseRewriter optimization converts large `IN` clauses to a
    // "mark join" against a `ColumnDataCollection`, which may not make
    // sense in other systems and would complicate the conversion to Substrait.
    set<OptimizerType> disabled_optimizers =
      DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
    disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
    // STATISTICS_PROPAGATION is now enabled: the GPU_VALUES source operator
    // handles the COLUMN_DATA_SCAN / EXPRESSION_GET / DUMMY_SCAN sources that
    // this optimizer produces (e.g. folding count(*), MIN, MAX to constants).
#ifdef DEBUG
    disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
#endif
    // disabled_optimizers.insert(OptimizerType::MATERIALIZED_CTE);
    // If error(varchar) gets implemented in substrait this can be removed
    // context.config.scalar_subquery_error_on_multiple_rows = false;
    DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;
  }

  // Reset configuration
  void CleanupConnection(ClientContext& context) const
  {
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
    context.config                                           = original_config;
  }

  unique_ptr<LogicalOperator> ExtractPlan(ClientContext& context)
  {
    PrepareConnection(context);
    unique_ptr<LogicalOperator> plan;
    try {
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(query);

      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      D_ASSERT(planner.plan);

      plan = std::move(planner.plan);

      if (context.config.enable_optimizer) {
        Optimizer optimizer(*planner.binder, context);
        plan = optimizer.Optimize(std::move(plan));
      }

      // After optimization, refresh types before column binding resolution
      // to ensure types are consistent (some optimizers may have set stale types)
      plan->ResolveOperatorTypes();

      ColumnBindingResolver resolver;
      ColumnBindingResolver::Verify(*plan);
      resolver.VisitOperator(*plan);
    } catch (...) {
      CleanupConnection(context);
      throw;
    }

    CleanupConnection(context);
    return plan;
  }
};

#ifdef SIRIUS_ENABLE_LEGACY
struct GPUTableFunctionData : public TableFunctionData {
  GPUTableFunctionData() = default;
  shared_ptr<Relation> plan;
  shared_ptr<GPUPreparedStatementData> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<GPUContext> gpu_context;
  string query;
  bool enable_optimizer;
  bool finished   = false;
  bool plan_error = false;
  //! Original options from the connection
  ClientConfig original_config;
  set<OptimizerType> original_disabled_optimizers;

  void PrepareConnection(ClientContext& context)
  {
    // First collect original options
    original_config              = context.config;
    original_disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;

    // The user might want to disable the optimizer of the new connection
    context.config.enable_optimizer = enable_optimizer;
    // We want for sure to disable the internal compression optimizations.
    // These are DuckDB specific, no other system implements these. Also,
    // respect the user's settings if they chose to disable any specific optimizers.
    //
    // The InClauseRewriter optimization converts large `IN` clauses to a
    // "mark join" against a `ColumnDataCollection`, which may not make
    // sense in other systems and would complicate the conversion to Substrait.
    set<OptimizerType> disabled_optimizers =
      DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
    disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
    // STATISTICS_PROPAGATION folds ungrouped MIN/MAX aggregates into constant
    // expressions using partition statistics, producing EXPRESSION_GET + DUMMY_SCAN.
    // The GPU pipeline cannot schedule COLUMN_DATA_SCAN sources, so disable this
    // to keep the query on the scan -> aggregate path where the GPU can execute it.
    disabled_optimizers.insert(OptimizerType::STATISTICS_PROPAGATION);
#ifdef DEBUG
    disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
#endif
    // disabled_optimizers.insert(OptimizerType::MATERIALIZED_CTE);
    // If error(varchar) gets implemented in substrait this can be removed
    // context.config.scalar_subquery_error_on_multiple_rows = false;
    DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;
  }

  // Reset configuration
  void CleanupConnection(ClientContext& context) const
  {
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
    context.config                                           = original_config;
  }

  unique_ptr<LogicalOperator> ExtractPlan(ClientContext& context)
  {
    PrepareConnection(context);
    unique_ptr<LogicalOperator> plan;
    try {
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(query);

      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      D_ASSERT(planner.plan);

      plan = std::move(planner.plan);

      if (context.config.enable_optimizer) {
        Optimizer optimizer(*planner.binder, context);
        plan = optimizer.Optimize(std::move(plan));
      }

      // After optimization, refresh types before column binding resolution
      // to ensure types are consistent (some optimizers may have set stale types)
      plan->ResolveOperatorTypes();

      ColumnBindingResolver resolver;
      ColumnBindingResolver::Verify(*plan);
      resolver.VisitOperator(*plan);
    } catch (...) {
      CleanupConnection(context);
      throw;
    }

    CleanupConnection(context);
    return plan;
  }
};

void do_nothing_context(ClientContext*) {}

static unique_ptr<GPUPhysicalOperator> GPUGeneratePhysicalPlan(
  ClientContext& context,
  GPUContext& gpu_context,
  unique_ptr<LogicalOperator>& logical_plan,
  Connection& new_conn)
{
  GPUPhysicalPlanGenerator physical_planner = GPUPhysicalPlanGenerator(context, gpu_context);
  auto physical_plan                        = physical_planner.CreatePlan(std::move(logical_plan));
  return physical_plan;
}

// The result of the GPUProcessingBind function is a unique pointer to a FunctionData object.
// This result of this function is used as an argument to the GPUProcessingFunction function (data_p
// argument), which is called to execute the table function.
unique_ptr<FunctionData> SiriusExtension::GPUProcessingBind(ClientContext& context,
                                                            TableFunctionBindInput& input,
                                                            vector<LogicalType>& return_types,
                                                            vector<string>& names)
{
  auto result              = make_uniq<GPUTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;
  result->gpu_context      = make_uniq<GPUContext>(context);
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_processing cannot be called with a NULL parameter");
  }

  // Parse the query just to get the result type information and to create preparedstatmement data
  auto statements = result->conn->context->ParseStatements(result->query);
  Planner planner(context);
  auto statement_type = statements[0]->type;
  planner.CreatePlan(std::move(statements[0]));
  D_ASSERT(planner.plan);

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  // generate physical plan from the logical plan
  unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
  SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
  if (buffer_is_initialized) {
    try {
      auto gpu_physical_plan =
        GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
      auto gpu_prepared    = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared),
                                                                    std::move(gpu_physical_plan));
      result->gpu_prepared = gpu_prepared;
    } catch (std::exception& e) {
      ErrorData error(e);
      SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan: {}", error.RawMessage());
      result->plan_error = true;
    }
  } else {
    result->gpu_prepared = nullptr;
  }

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

void SiriusExtension::GPUProcessingFunction(ClientContext& context,
                                            TableFunctionInput& data_p,
                                            DataChunk& output)
{
  auto& data = (GPUTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  if (!data.res) {
    auto start = std::chrono::high_resolution_clock::now();
    if (!buffer_is_initialized) {
      printf("\033[1;31m");
      printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n");
      printf("\033[0m");
      printf(
        "=============================================\nError in GPUExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
    } else if (data.plan_error) {
      printf(
        "=============================================\nError in GPUExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
    } else {
      data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        printf(
          "=============================================\nError in GPUExecuteQuery, fallback to "
          "DuckDB\n=============================================\n");
        data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = data.res->Fetch();
  if (result_chunk == nullptr) {
    output.SetCardinality(0);
    return;
  }

  output.Reference(*result_chunk);
  return;
}

static void RegisterLegacyGPUFunctions(CatalogTransaction& transaction, Catalog& catalog)
{
  TableFunction gpu_processing("gpu_processing",
                               {LogicalType::VARCHAR},
                               SiriusExtension::GPUProcessingFunction,
                               SiriusExtension::GPUProcessingBind);
  gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
  CreateTableFunctionInfo gpu_processing_info(gpu_processing);
  catalog.CreateTableFunction(transaction, gpu_processing_info);
}
#endif  // SIRIUS_ENABLE_LEGACY

static unique_ptr<sirius::op::sirius_physical_operator> SiriusGeneratePhysicalPlan(
  ClientContext& context, unique_ptr<LogicalOperator>& logical_plan)
{
  sirius::planner::sirius_physical_plan_generator physical_planner =
    sirius::planner::sirius_physical_plan_generator(context);
  auto physical_plan = physical_planner.create_plan(std::move(logical_plan));
  return physical_plan;
}

// The result of the GPUExecutionBind function is a unique pointer to a FunctionData object.
// This result of this function is used as an argument to the GPUExecutionFunction function (data_p
// argument), which is called to execute the table function.
unique_ptr<FunctionData> SiriusExtension::GPUExecutionBind(ClientContext& context,
                                                           TableFunctionBindInput& input,
                                                           vector<LogicalType>& return_types,
                                                           vector<string>& names)
{
  auto result              = make_uniq<SiriusTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;

  std::optional<std::string> query_label = std::nullopt;
  // take any query_label that was set using sirius_set_query_label SQL call.
  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      sirius_ctx) {
    query_label = sirius_ctx->take_pending_query_label();
  }
  // however, give precedence to a query_label that was set inline in with
  // gpu_execution SQL call.
  if (auto it = input.named_parameters.find(QUERY_LABEL_PARAM_KEY);
      it != input.named_parameters.end() && not it->second.IsNull()) {
    query_label = it->second.ToString();
  }

  result->sirius_iface = make_uniq<::sirius::sirius_interface>(context, std::move(query_label));

  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Route Sirius-owned remote parquet reads through Sirius's own bind:
  // read_parquet('s3://…') -> sirius_read_parquet('s3://…'). DuckDB core has no
  // S3 filesystem, so without this rewrite the s3:// bind fails before Sirius
  // ever runs. Local paths and non-s3 calls are left untouched.
  //
  // Capture the pre-rewrite query first: a CPU fallback of a LOCAL read must
  // replay the original read_parquet (not the rewritten sirius_read_parquet,
  // which throws off the GPU). An s3:// query has no CPU fallback — the fallback
  // path detects it and raises a clear error instead of replaying.
  result->cpu_fallback_query = result->query;
  result->query              = sirius::rewrite_sirius_owned_remote_parquet_calls(result->query);

  // Parse the query just to get the result type information and to create PreparedStatementData
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  auto statement_type = parser.statements[0]->type;
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  // generate physical plan from the logical plan
  unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
  SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
  try {
    auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
    SIRIUS_LOG_DEBUG("Done generating sirius physical plan");
    auto gpu_prepared = make_shared_ptr<::sirius::sirius_prepared_statement_data>(
      std::move(prepared), std::move(sirius_physical_plan));
    result->gpu_prepared = gpu_prepared;
  } catch (std::exception& e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("Error in SiriusGeneratePhysicalPlan: {}", error.RawMessage());
    if (duckdb_fallback_enabled(context)) {
      result->plan_error         = true;
      result->plan_error_message = error.RawMessage();
    } else {
      throw std::runtime_error("Error in SiriusGeneratePhysicalPlan: " + error.RawMessage());
      return nullptr;
    }
  }

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

void SiriusExtension::GPUExecutionFunction(ClientContext& context,
                                           TableFunctionInput& data_p,
                                           DataChunk& output)
{
  auto& data = (SiriusTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  if (!data.res) {
    auto start = std::chrono::high_resolution_clock::now();
    if (data.plan_error) {
      print_cpu_fallback_banner();
      data.res = run_internal_cpu_fallback_query(
        context, *data.conn, data.cpu_fallback_query, data.plan_error_message);
    } else {
      data.res =
        data.sirius_iface->sirius_execute_query(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        if (duckdb_fallback_enabled(context)) {
          SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", data.res->GetError());
          print_cpu_fallback_banner();
          data.res = run_internal_cpu_fallback_query(
            context, *data.conn, data.cpu_fallback_query, data.res->GetError());
        } else {
          throw std::runtime_error("SiriusExecuteQuery error: " + data.res->GetError());
          return;
        }
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = data.res->Fetch();
  if (result_chunk == nullptr) {
    output.SetCardinality(0);
    return;
  }

  output.Reference(*result_chunk);
  return;
}

[[maybe_unused]] static unique_ptr<LogicalOperator> OptimizePlan(ClientContext& context,
                                                                 Planner& planner,
                                                                 Connection& new_conn)
{
  unique_ptr<LogicalOperator> plan;
  plan = std::move(planner.plan);

  Optimizer optimizer(*planner.binder, context);
  plan = optimizer.Optimize(std::move(plan));
  SIRIUS_LOG_DEBUG("Query plan:\n{}", plan->ToString());

  ColumnBindingResolver resolver;
  resolver.Verify(*plan);
  resolver.VisitOperator(*plan);

  plan->ResolveOperatorTypes();

  return plan;
}

#ifdef SIRIUS_ENABLE_LEGACY
struct GPUBufferInitFunctionData : public TableFunctionData {
  GPUBufferInitFunctionData() {}
  bool finished = false;
  size_t cache_size;
  size_t processing_size;
  size_t pinned_memory_size;
};

unique_ptr<FunctionData> SiriusExtension::GPUBufferInitBind(ClientContext& context,
                                                            TableFunctionBindInput& input,
                                                            vector<LogicalType>& return_types,
                                                            vector<string>& names)
{
  auto result = make_uniq<GPUBufferInitFunctionData>();

  string gpu_cache_size      = input.inputs[0].ToString();
  string gpu_processing_size = input.inputs[1].ToString();
  string pinned_memory_size("0 GB");  // Default size of pinned memory
  if (input.named_parameters.find(PINNED_MEMORY_PARAM_KEY) != input.named_parameters.end()) {
    // If the pinned memory size is specified in the arguments then use that
    pinned_memory_size = input.named_parameters[PINNED_MEMORY_PARAM_KEY].ToString();
  }

  // parsing 2GB or 2GiB to size_t
  //  Function to parse size strings like "2GB" or "2GiB" to size_t
  auto parse_size = [](const string& size_str) -> size_t {
    size_t result     = 0;
    size_t multiplier = 1;
    string num_part;
    string unit_part;

    size_t i = 0;
    // Skip any whitespace between number and unit
    while (i < size_str.length() && isspace(size_str[i])) {
      i++;
    }

    // Find where the number ends and unit begins
    while (i < size_str.length() && (isdigit(size_str[i]) || size_str[i] == '.')) {
      num_part += size_str[i];
      i++;
    }

    // Skip any whitespace between number and unit
    while (i < size_str.length() && isspace(size_str[i])) {
      i++;
    }

    // Extract unit part
    unit_part = size_str.substr(i);

    // Convert number part to double
    double num_value = stod(num_part);

    // Determine multiplier based on unit
    if (unit_part == "B") {
      multiplier = 1;
    } else if (unit_part == "KB" || unit_part == "KiB") {
      multiplier = 1024;
    } else if (unit_part == "MB" || unit_part == "MiB") {
      multiplier = 1024 * 1024;
    } else if (unit_part == "GB" || unit_part == "GiB") {
      multiplier = 1024 * 1024 * 1024;
    } else if (unit_part == "TB" || unit_part == "TiB") {
      multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    } else {
      throw InvalidInputException("Invalid format");
    }

    result = (size_t)(num_value * multiplier);
    return result;
  };

  // Parse the input sizes
  result->cache_size         = parse_size(gpu_cache_size);
  result->processing_size    = parse_size(gpu_processing_size);
  result->pinned_memory_size = parse_size(pinned_memory_size);

  auto type = LogicalType(LogicalTypeId::BOOLEAN);
  return_types.emplace_back(type);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::GPUBufferInitFunction(ClientContext& context,
                                            TableFunctionInput& data_p,
                                            DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUBufferInitFunctionData>();
  if (data.finished) { return; }

  size_t cache_size         = data.cache_size;
  size_t processing_size    = data.processing_size;
  size_t pinned_memory_size = data.pinned_memory_size;
  if (pinned_memory_size == 0) { pinned_memory_size = std::max(cache_size, processing_size); }

  if (!buffer_is_initialized) {
    SIRIUS_LOG_DEBUG(
      "GPU Buffer Manager initialized with args: Cache Size - {}, Processing Size - {}, Pinned Mem "
      "Size - {}\n",
      cache_size,
      processing_size,
      pinned_memory_size);
    GPUBufferManager* gpuBufferManager =
      &(GPUBufferManager::GetInstance(cache_size, processing_size, pinned_memory_size));
    buffer_is_initialized = true;
  } else {
    SIRIUS_LOG_WARN("GPUBufferManager already initialized");
  }
  data.finished = true;
}
#endif  // SIRIUS_ENABLE_LEGACY

static unique_ptr<FunctionData> ProfilerBind(ClientContext& context,
                                             TableFunctionBindInput& input,
                                             vector<LogicalType>& return_types,
                                             vector<string>& names)
{
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("ok");
  return nullptr;
}

struct PinTableFunctionData : public TableFunctionData {
  PinTableArgs args;
  bool finished = false;
};

namespace {

// Resolve the kept-column logical indices for a pin: the positions of the
// user-requested @p cols within @p schema_names (preserving requested order), or
// identity over all columns when @p cols is absent/empty.
std::vector<std::size_t> resolve_pin_kept_indices(
  std::vector<std::string> const& schema_names, std::optional<std::vector<std::string>> const& cols)
{
  std::vector<std::size_t> keep;
  if (cols && !cols->empty()) {
    std::unordered_map<std::string, std::size_t> pos;
    pos.reserve(schema_names.size());
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      pos.emplace(schema_names[i], i);
    }
    for (auto const& c : *cols) {
      auto it = pos.find(c);
      if (it == pos.end()) {
        throw InvalidInputException("pin_table: column '" + c + "' not found in table schema");
      }
      keep.push_back(it->second);
    }
  } else {
    keep.resize(schema_names.size());
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      keep[i] = i;
    }
  }
  return keep;
}

// Build the table_info for a parquet pin. It drives the ingestible read and is the
// source cache_entry_info::from reads to build the pinned entry's cache descriptor:
// it carries the FULL schema in names/returned_types (build_scan_plan indexes them
// by primary index) plus projection_ids when a column subset is pinned, from which
// cache_entry_info::from derives the column_ids-aligned names the gather needs.
std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> build_parquet_pin_info(
  sirius::scan_manager::sirius_scan_manager& scan_mgr,
  std::vector<std::string> const& file_paths,
  std::optional<std::vector<std::string>> const& cols,
  std::size_t batch_size,
  vector<LogicalType>& pinned_column_types)
{
  using sirius::op::scan::parquet_ingestible_table_info;
  auto desc = scan_mgr.describe_parquet(file_paths.front());
  std::vector<std::string> schema_names(desc.names.begin(), desc.names.end());
  auto keep            = resolve_pin_kept_indices(schema_names, cols);
  bool const is_subset = cols && !cols->empty();

  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths = file_paths;
  info->returned_types      = sirius::from_duckdb_vec(desc.return_types);  // full schema
  info->names               = desc.names;                                  // full schema
  for (auto idx : keep) {
    info->column_ids.emplace_back(duckdb::ColumnIndex(static_cast<duckdb::idx_t>(idx)));
    // Pin-time DuckDB type of each pinned column, in column_ids (batch-column)
    // order. Taken from the native DuckDB schema rather than round-tripped
    // through sirius::logical_type: the zone-map capture keys its type
    // allowlist on exact LogicalType identity (e.g. timestamp units).
    pinned_column_types.push_back(desc.return_types[idx]);
  }
  if (is_subset) {
    // Non-empty projection_ids forces scan_plan::is_projected() so the cudf reader
    // projects to exactly the pinned columns (identity into column_ids).
    for (std::size_t k = 0; k < keep.size(); ++k) {
      info->projection_ids.push_back(static_cast<duckdb::idx_t>(k));
    }
  }
  info->scan_output_arity      = keep.size();
  info->approximate_batch_size = batch_size;
  return info;
}

// A duckdb-native pin is a positional snapshot of the table's last-checkpointed
// disk image: a later checkpoint would compact tombstoned rows and flush transient
// appends, silently shifting rowids out from under the cache (and compressing the
// in-memory transient segments the query-time insert delta reads). Suppress both
// WAL auto-checkpoint triggers — the size threshold and the entry count — before
// the pin's metadata walk snapshots the on-disk row groups. The DBConfig is shared
// by every attached database, so this covers them all. Idempotent, and deliberately
// not restored on unpin (or when a later pin step fails): a restore would need pin
// refcounting to be safe against other pins taken meanwhile. Manual CHECKPOINT —
// and DETACH of the pinned database, whose close runs a shutdown checkpoint —
// while pinned are outside the supported contract.
void suppress_auto_checkpoint_for_pin(ClientContext& context)
{
  auto& config                       = DBConfig::GetConfig(context);
  config.options.checkpoint_wal_size = NumericLimits<idx_t>::Maximum();
  config.SetOptionByName("wal_autocheckpoint_entries", Value::UBIGINT(0));
}

// Resolve an attached duckdb table from the catalog and build its table_info for a
// pin. The .db must be ATTACHed. The pin captures the table's (catalog, schema,
// table) name from the resolved DuckTableEntry; that name tuple must match what a
// later query's scan derives, because it is the cache identity
// (cache_entry_info::can_serve_with_columns) — not the DataTable* pointer. A single
// info serves both the read and the cached entry.
std::unique_ptr<sirius::op::scan::duckdb_native_ingestible_table_info> build_duckdb_pin_info(
  ClientContext& context,
  std::string const& table_ref,
  std::string const& schema_override,
  std::optional<std::vector<std::string>> const& cols,
  std::size_t batch_size,
  vector<LogicalType>& pinned_column_types)
{
  using sirius::op::scan::duckdb_native_ingestible_table_info;

  // 'table_ref' is the (optionally schema/catalog-qualified) table name. Resolve it
  // through the catalog honoring the client's search path — so a bare name picks up
  // the current/USE'd database — yielding the same DataTable* a query-time scan binds.
  auto const qname          = duckdb::QualifiedName::Parse(table_ref);
  std::string const catalog = qname.catalog;  // empty => search path
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_override;
  std::string const& table  = qname.name;

  // Non-template catalog lookup + Cast (mirroring the pipeline converter). The
  // templated Catalog::GetEntry<DuckTableEntry> would ODR-use DuckTableEntry::Name
  // (a static constexpr inherited from TableCatalogEntry), emitting a duplicate
  // symbol against libduckdb_static at link time.
  auto& entry_base =
    duckdb::Catalog::GetEntry(context, duckdb::CatalogType::TABLE_ENTRY, catalog, schema, table);
  auto& entry         = entry_base.Cast<duckdb::DuckTableEntry>();
  auto& storage       = entry.GetStorage();
  auto const& columns = entry.GetColumns();
  auto schema_names   = columns.GetColumnNames();  // logical order
  auto schema_types   = columns.GetColumnTypes();  // logical order

  auto keep            = resolve_pin_kept_indices(schema_names, cols);
  auto const canonical = storage.GetAttached().GetStorageManager().GetDBPath();

  // Update chains version values in place, invisibly to the DELETE
  // keep-masks — a pin would serve stale values to every query until the
  // chains are folded away. Refuse loudly (the transient-rows case already
  // fails the same way); CHECKPOINT folds the chains into the base data.
  {
    std::vector<duckdb::storage_t> pinned_storage_cols(keep.begin(), keep.end());
    if (sirius::op::scan::any_update_chains(
          storage, pinned_storage_cols, static_cast<std::size_t>(storage.GetTotalRows()))) {
      throw InvalidInputException(
        "pin_table: table '%s' has in-memory update chains on a pinned column; run CHECKPOINT "
        "before pinning",
        table_ref);
    }
  }

  auto info     = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage = &storage;
  info->context = &context;
  info->db_path = canonical;
  // Qualified-name identity for the pin cache — derived from the resolved
  // DuckTableEntry so it matches the query-side derivation (the pipeline converter).
  info->catalog_name           = entry.ParentCatalog().GetName();
  info->schema_name            = entry.ParentSchema().name;
  info->table_name             = entry.name;
  info->approximate_batch_size = batch_size;
  // Full-schema names (logical order) so column_names() can derive the
  // column_ids-aligned view; the decoder itself ignores names.
  info->names.assign(schema_names.begin(), schema_names.end());
  info->returned_types = sirius::from_duckdb_vec(schema_types);
  for (auto col : keep) {
    info->column_ids.emplace_back(duckdb::ColumnIndex(static_cast<duckdb::idx_t>(col)));
    // Exact pin-time DuckDB type per pinned column (see build_parquet_pin_info).
    pinned_column_types.push_back(schema_types[col]);
    sirius::op::scan::projected_column pc;
    pc.is_rowid    = false;
    pc.storage_idx = duckdb::StorageIndex(static_cast<duckdb::idx_t>(col));
    info->projected_cols.push_back(pc);
    auto t = sirius::from_duckdb(schema_types[col]);
    info->projected_types.push_back(t);
    info->output_types.push_back(t);
  }
  return info;
}

}  // namespace

unique_ptr<FunctionData> SiriusExtension::PinTableBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<PinTableFunctionData>();

  // The positional path is optional: parquet uses it (file/glob); duckdb takes no
  // positional (its table is named by 'name' and resolved from the catalog).
  if (!input.inputs.empty() && !input.inputs[0].IsNull()) {
    result->args.path = input.inputs[0].ToString();
  }

  auto tier_it = input.named_parameters.find("tier");
  if (tier_it == input.named_parameters.end() || tier_it->second.IsNull()) {
    throw BinderException("pin_table requires a 'tier' named parameter");
  }
  result->args.tier = tier_it->second.ToString();
  if (result->args.tier != "gpu" && result->args.tier != "host") {
    throw NotImplementedException("pin_table tier='" + result->args.tier +
                                  "' is not supported (only 'gpu' and 'host')");
  }

  auto name_it = input.named_parameters.find("name");
  if (name_it == input.named_parameters.end() || name_it->second.IsNull()) {
    throw BinderException("pin_table requires a 'name' named parameter");
  }
  result->args.name = name_it->second.ToString();

  auto cols_it = input.named_parameters.find("cols");
  if (cols_it != input.named_parameters.end() && !cols_it->second.IsNull()) {
    vector<string> cols;
    for (auto& val : ListValue::GetChildren(cols_it->second)) {
      if (val.IsNull()) {
        throw BinderException("pin_table 'cols' list cannot contain NULL entries");
      }
      cols.push_back(val.ToString());
    }
    result->args.cols = std::move(cols);
  }

  // Resolve the source format: an explicit 'format' parameter, else inferred from
  // the path extension (.parquet -> parquet, .db/.duckdb -> duckdb).
  auto to_lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  auto ends_with = [](std::string const& s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  auto format_it = input.named_parameters.find("format");
  if (format_it != input.named_parameters.end() && !format_it->second.IsNull()) {
    result->args.format = to_lower(format_it->second.ToString());
    if (result->args.format != "parquet" && result->args.format != "duckdb") {
      throw BinderException("pin_table 'format' must be 'parquet' or 'duckdb', got '" +
                            result->args.format + "'");
    }
  } else if (!result->args.path.empty()) {
    auto lowered = to_lower(result->args.path);
    if (ends_with(lowered, ".parquet")) {
      result->args.format = "parquet";
    } else if (ends_with(lowered, ".db") || ends_with(lowered, ".duckdb")) {
      result->args.format = "duckdb";
    } else {
      throw BinderException("pin_table: cannot infer format from path '" + result->args.path +
                            "'; pass format => 'parquet' or 'duckdb'");
    }
  } else {
    throw BinderException("pin_table: provide a positional path (parquet) or format => 'duckdb'");
  }

  if (result->args.format == "parquet") {
    if (result->args.path.empty()) {
      throw BinderException("pin_table: format 'parquet' requires a positional path argument");
    }
  } else {
    // duckdb: 'name' is the (optionally qualified) table to pin, resolved from the
    // catalog — no path needed. 'schema' is a SQL reserved word, so the optional
    // schema override is the 'schema_name' parameter.
    auto schema_it = input.named_parameters.find("schema_name");
    if (schema_it != input.named_parameters.end() && !schema_it->second.IsNull()) {
      result->args.schema = schema_it->second.ToString();
    }
  }

  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::PinTableFunction(ClientContext& context,
                                       TableFunctionInput& data_p,
                                       DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<PinTableFunctionData>();
  if (data.finished) { return; }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("pin_table requires the Sirius context to be initialized");
  }

  // The read is driven by sirius::materialize_all_batches (pin_table.cpp), which
  // round-robins the materialized batches across all GPU memory spaces so a pin
  // distributes its chunks evenly. For tier='host' each materialized GPU table is
  // then converted to a host_data_representation (via the GPU<->HOST converter) so
  // the pinned data lives in pinned host memory.
  auto& memory_manager = sirius_ctx->get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw InvalidInputException("pin_table: no GPU memory space available");
  }

  // For host tier, build a target_gpu_id -> NUMA-local host memory_space map.
  // Each round-robin GPU's host conversion should pin its data on the host
  // memory_space whose NUMA node matches the GPU. Fall back to host_spaces[0]
  // when the GPU's NUMA node is unknown or no matching host space exists.
  std::unordered_map<int, cucascade::memory::memory_space*> host_space_by_gpu;
  if (data.args.tier == "host") {
    auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (host_spaces.empty()) {
      throw InvalidInputException("pin_table: no HOST memory space available");
    }
    auto* fallback_host = const_cast<cucascade::memory::memory_space*>(host_spaces[0]);
    auto const& topo    = sirius_ctx->get_config().get_hw_topology();
    for (auto const* gpu_space : gpu_spaces) {
      int const gpu_id = gpu_space->get_device_id();
      int numa_node    = -1;
      if (static_cast<size_t>(gpu_id) < topo.gpus.size()) {
        numa_node = topo.gpus[gpu_id].numa_node;
      }
      cucascade::memory::memory_space* picked = fallback_host;
      if (numa_node >= 0) {
        for (auto* hs : host_spaces) {
          if (hs->get_device_id() == numa_node) {
            picked = const_cast<cucascade::memory::memory_space*>(hs);
            break;
          }
        }
      }
      host_space_by_gpu[gpu_id] = picked;
    }
  }

  auto& scan_mgr = sirius_ctx->get_scan_manager();
  std::size_t const batch_size =
    sirius_ctx->get_config().get_operator_params().scan_task_batch_size;

  // materialize_all_batches round-robins reads across these GPUs and reports the
  // per-batch placement; insert_pinned_entry wants non-const memory_space*.
  std::vector<cucascade::memory::memory_space*> gpu_spaces_mut;
  gpu_spaces_mut.reserve(gpu_spaces.size());
  for (auto const* s : gpu_spaces) {
    gpu_spaces_mut.push_back(const_cast<cucascade::memory::memory_space*>(s));
  }

  // Build the ingestible (drives the metadata walk + decode) from one table_info.
  // duckdb-native has no standalone reader, so both formats go through their
  // gpu_ingestible — one read path.
  std::shared_ptr<sirius::op::scan::gpu_ingestible> ingestible;
  // Pin-time DuckDB types of the pinned columns, in column_ids (batch-column)
  // order — the zone-map capture keys its type allowlist on these exact types.
  vector<LogicalType> pinned_column_types;
  // The pin transaction's MVCC fence on the pinned table's own AttachedDatabase;
  // meaningful only for format == "duckdb" (see duckdb_mvcc_metadata::v_base).
  transaction_t duckdb_pin_v_base = 0;

  if (data.args.format == "duckdb") {
    auto info = build_duckdb_pin_info(
      context, data.args.name, data.args.schema, data.args.cols, batch_size, pinned_column_types);
    // After the catalog resolution (so a bad table name fails without side
    // effects) but before make_ingestible snapshots the on-disk row groups.
    suppress_auto_checkpoint_for_pin(context);
    // Not the default database's counter: each AttachedDatabase has its own MVCC
    // start_time domain, and pins usually target an ATTACHed .db (the catalog
    // resolved by build_duckdb_pin_info), so read the fence off that catalog's
    // DuckTransaction.
    auto& pinned_catalog = Catalog::GetCatalog(context, info->catalog_name);
    duckdb_pin_v_base    = DuckTransaction::Get(context, pinned_catalog).start_time;
    ingestible           = sirius::op::scan::make_ingestible(std::move(info));
  } else {  // parquet
    auto& fs   = FileSystem::GetFileSystem(context);
    auto files = fs.GlobFiles(data.args.path);
    std::vector<std::string> file_paths;
    file_paths.reserve(files.size());
    for (auto& f : files) {
      file_paths.push_back(f.path);
    }
    if (file_paths.empty()) {
      throw InvalidInputException("pin_table: no parquet files matched path: " + data.args.path);
    }
    auto info =
      build_parquet_pin_info(scan_mgr, file_paths, data.args.cols, batch_size, pinned_column_types);
    ingestible = sirius::op::scan::make_ingestible(std::move(info));
  }

  if (!sirius_ctx->get_config().get_operator_params().enable_pinned_zone_map_pruning) {
    pinned_column_types.clear();
  }

  // Build the cache descriptor (table identity + column layout) from the
  // ingestible; it is stored on the pinned entry in place of the heavyweight
  // ingestible_table_info and drives later cache-hit matching + the gather.
  auto cache_info = sirius::scan_manager::cache_entry_info::from(ingestible->table_info());

  if (data.args.tier == "host") {
    // Stream each batch GPU->host: materialize one batch on its round-robin GPU, convert it
    // to a pinned host_data_representation on that GPU's NUMA-local host space, then free the
    // GPU table before materializing the next. Peak GPU residency stays at ~one batch, so the
    // whole table never needs to fit in GPU memory. On multi-GPU the chunks land round-robin
    // across NUMA nodes; the cached-serve path then reads each chunk back on a NUMA-local GPU.
    auto mat = sirius::materialize_pin_to_host(
      *ingestible, gpu_spaces_mut, host_space_by_gpu, *scan_mgr.io_ctx(), pinned_column_types);
    // entry.memory_space is metadata only; each host_chunk carries its own per-GPU
    // NUMA-local memory_space. Pass a representative (the first GPU's host space).
    int const first_gpu_id          = gpu_spaces_mut[0]->get_device_id();
    auto* representative_host_space = host_space_by_gpu.at(first_gpu_id);
    scan_mgr.insert_pinned_entry_host(data.args.name,
                                      std::move(cache_info),
                                      std::move(mat.host_chunks),
                                      *representative_host_space,
                                      std::move(pinned_column_types),
                                      std::move(mat.chunk_stats));
    if (data.args.format == "duckdb") {
      sirius::scan_manager::duckdb_mvcc_metadata mvcc;
      mvcc.v_base                   = duckdb_pin_v_base;
      mvcc.base_row_count_per_chunk = std::move(mat.base_row_count_per_chunk);
      scan_mgr.attach_mvcc_metadata(data.args.name, std::move(mvcc));
    }
  } else {
    // GPU tier: materialize every batch as a GPU-resident cudf::table (with its GPU
    // placement) and pin them in place.
    auto mat = sirius::materialize_all_batches(
      *ingestible, gpu_spaces_mut, *scan_mgr.io_ctx(), pinned_column_types);
    auto base_row_count_per_chunk = std::move(mat.base_row_count_per_chunk);
    scan_mgr.insert_pinned_entry(data.args.name,
                                 std::move(cache_info),
                                 std::move(mat.tables),
                                 std::move(mat.chunk_memory_spaces),
                                 std::move(pinned_column_types),
                                 std::move(mat.chunk_stats));
    if (data.args.format == "duckdb") {
      sirius::scan_manager::duckdb_mvcc_metadata mvcc;
      mvcc.v_base                   = duckdb_pin_v_base;
      mvcc.base_row_count_per_chunk = std::move(base_row_count_per_chunk);
      scan_mgr.attach_mvcc_metadata(data.args.name, std::move(mvcc));
    }
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct UnpinTableFunctionData : public TableFunctionData {
  std::string name;
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::UnpinTableBind(ClientContext& context,
                                                         TableFunctionBindInput& input,
                                                         vector<LogicalType>& return_types,
                                                         vector<string>& names)
{
  auto result = make_uniq<UnpinTableFunctionData>();

  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException("unpin_table requires a non-null name argument");
  }
  result->name = input.inputs[0].ToString();

  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::UnpinTableFunction(ClientContext& context,
                                         TableFunctionInput& data_p,
                                         DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<UnpinTableFunctionData>();
  if (data.finished) { return; }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("unpin_table requires the Sirius context to be initialized");
  }
  sirius_ctx->get_scan_manager().remove_pinned_entry(data.name);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct ProfilerFunctionData : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<GlobalTableFunctionState> ProfilerInit(ClientContext& context,
                                                         TableFunctionInitInput& input)
{
  return make_uniq<ProfilerFunctionData>();
}

static void ProfilerStartFunction(ClientContext& context,
                                  TableFunctionInput& data_p,
                                  DataChunk& output)
{
  auto& data = data_p.global_state->Cast<ProfilerFunctionData>();
  if (data.finished) return;
  cudaProfilerStart();
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

static void ProfilerStopFunction(ClientContext& context,
                                 TableFunctionInput& data_p,
                                 DataChunk& output)
{
  auto& data = data_p.global_state->Cast<ProfilerFunctionData>();
  if (data.finished) return;
  cudaProfilerStop();
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct SiriusSetQueryLabelData : public TableFunctionData {
  std::string label;
  bool finished = false;
};

static unique_ptr<FunctionData> SiriusSetQueryLabelBind(ClientContext& context,
                                                        TableFunctionBindInput& input,
                                                        vector<LogicalType>& return_types,
                                                        vector<string>& names)
{
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException("sirius_set_query_label requires a non-NULL VARCHAR argument");
  }
  auto result   = make_uniq<SiriusSetQueryLabelData>();
  result->label = input.inputs[0].ToString();
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("ok");
  return std::move(result);
}

static void SiriusSetQueryLabelFunction(ClientContext& context,
                                        TableFunctionInput& data_p,
                                        DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<SiriusSetQueryLabelData>();
  if (data.finished) { return; }

  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      sirius_ctx) {
    sirius_ctx->set_pending_query_label(data.label);
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct CreateAnnIndexData : public TableFunctionData {
  std::string table_name;
  std::string column_name;
  std::string index_name;                ///< management name (for a future drop_ann_index)
  std::string index_type  = "ivf_flat";  ///< lowercased; only "ivf_flat" supported today
  std::string metric      = "l2";        ///< lowercased; one of l2 / cosine
  std::string schema_name = "main";
  int64_t n_lists         = 0;  ///< IVF-Flat list count; 0 = choose a default at build time
  bool finished           = false;
};

static unique_ptr<FunctionData> SiriusCreateAnnIndexBind(ClientContext& context,
                                                         TableFunctionBindInput& input,
                                                         vector<LogicalType>& return_types,
                                                         vector<string>& names)
{
  auto result = make_uniq<CreateAnnIndexData>();

  if (input.inputs.size() < 2 || input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
    throw BinderException(
      "sirius_create_ann_index requires two non-NULL positional arguments: table and column");
  }
  result->table_name  = input.inputs[0].ToString();
  result->column_name = input.inputs[1].ToString();

  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_create_ann_index: named parameter '" + kv.first +
                            "' cannot be NULL");
    }
    if (key == "name") {
      result->index_name = kv.second.ToString();
    } else if (key == "metric") {
      result->metric = StringUtil::Lower(kv.second.ToString());
    } else if (key == "index_type") {
      result->index_type = StringUtil::Lower(kv.second.ToString());
    } else if (key == "n_lists") {
      result->n_lists = kv.second.GetValue<int64_t>();
    } else if (key == "schema_name") {
      result->schema_name = kv.second.ToString();
    }
  }

  if (result->index_type != "ivf_flat") {
    throw BinderException("sirius_create_ann_index: unsupported index_type '" + result->index_type +
                          "'; only 'ivf_flat' is supported");
  }
  if (result->metric != "l2" && result->metric != "cosine") {
    throw BinderException("sirius_create_ann_index: metric must be one of 'l2', 'cosine', got '" +
                          result->metric + "'");
  }
  if (result->n_lists < 0) {
    throw BinderException("sirius_create_ann_index: n_lists must be >= 0");
  }

  // Default the management name from the index identity when not given.
  if (result->index_name.empty()) {
    result->index_name =
      result->table_name + "_" + result->column_name + "_" + result->metric + "_ann";
  }

  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

// Map the (already-validated) index_type string to its cache index_kind.
static sirius::vss::index_kind ann_index_kind_from_type(const std::string& index_type)
{
  if (index_type == "ivf_flat") { return sirius::vss::index_kind::ivf_flat; }
  throw InvalidInputException("sirius_create_ann_index: unsupported index_type '" + index_type +
                              "'");
}

// Default IVF-Flat list count
static std::uint32_t default_ivf_n_lists(int64_t n_rows)
{
  auto const approx = static_cast<std::uint32_t>(std::sqrt(static_cast<double>(n_rows)));
  std::uint32_t n   = approx == 0 ? 1u : approx;
  return n > 1024u ? 1024u : n;
}

static void SiriusCreateAnnIndexFunction(ClientContext& context,
                                         TableFunctionInput& data_p,
                                         DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<CreateAnnIndexData>();
  if (data.finished) { return; }

  nvtx3::scoped_range nvtx_range{"SiriusCreateAnnIndexFunction"};

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException(
      "sirius_create_ann_index requires the Sirius context to be initialized");
  }

  // --- Resolve the vector column's fixed dimensionality from the catalog. ---
  auto const qname          = QualifiedName::Parse(data.table_name);
  std::string const catalog = qname.catalog;  // empty => search path
  std::string const schema  = !qname.schema.empty() ? qname.schema : data.schema_name;
  std::string const& table  = qname.name;
  auto& entry_base = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, catalog, schema, table);
  auto& entry      = entry_base.Cast<DuckTableEntry>();
  auto& entry_catalog     = entry.ParentCatalog().GetName();
  auto& entry_schema      = entry.ParentSchema().name;
  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();

  std::size_t col_idx = schema_names.size();
  for (std::size_t i = 0; i < schema_names.size(); ++i) {
    if (schema_names[i] == data.column_name) {
      col_idx = i;
      break;
    }
  }
  if (col_idx == schema_names.size()) {
    throw InvalidInputException("sirius_create_ann_index: column '" + data.column_name +
                                "' not found in table '" + data.table_name + "'");
  }
  auto const& col_type = schema_types[col_idx];
  if (col_type.id() != LogicalTypeId::ARRAY ||
      ArrayType::GetChildType(col_type).id() != LogicalTypeId::FLOAT) {
    throw InvalidInputException("sirius_create_ann_index: column '" + data.column_name +
                                "' must be a FLOAT[N] array column");
  }
  auto const dim    = static_cast<int64_t>(ArrayType::GetSize(col_type));
  auto const metric = sirius::vss::ann_distance_type_from_metric(data.metric);

  // Get the vector column onto a single GPU as one contiguous column for one cuVS index
  auto& memory_manager = sirius_ctx->get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw InvalidInputException("sirius_create_ann_index: no GPU memory space available");
  }
  auto* target_space   = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  int const target_gpu = target_space->get_device_id();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_gpu}};

  auto& scan_mgr = sirius_ctx->get_scan_manager();
  const auto* pin =
    scan_mgr.find_pinned_entry_for_duckdb_table(entry_catalog, entry_schema, entry.name);
  if (pin == nullptr || pin->tier != cucascade::memory::Tier::GPU) {
    throw InvalidInputException("sirius_create_ann_index: table '" + data.table_name +
                                "' must be pinned on the GPU tier before building an index");
  }

  // Collect the vector column's GPU chunks as views:
  // a full coalesce of a large dataset overflows cudf's 2^31-element per-column limit
  // in the LIST child. The chunked builder feeds cuVS one chunk at a time via ivf_flat::extend.
  auto chunk_views = sirius::vss::pinned_column_chunk_views(*pin, data.column_name, *target_space);

  int64_t n_rows = 0;
  for (auto const& v : chunk_views) {
    n_rows += static_cast<int64_t>(v.size());
  }
  if (n_rows <= 0) { throw InvalidInputException("sirius_create_ann_index: empty vector column"); }

  std::uint32_t n_lists =
    data.n_lists > 0 ? static_cast<std::uint32_t>(data.n_lists) : default_ivf_n_lists(n_rows);
  if (static_cast<int64_t>(n_lists) > n_rows) { n_lists = static_cast<std::uint32_t>(n_rows); }

  // Reserve the index footprint (heuristic, over-estimated to cover build-time
  // scratch): ~2x the stored vectors + 2x centroids + 1 MiB slack
  std::size_t const vec_bytes =
    static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(dim) * sizeof(float);
  std::size_t const centroid_bytes =
    static_cast<std::size_t>(n_lists) * static_cast<std::size_t>(dim) * sizeof(float);
  std::size_t const footprint = vec_bytes * 2 + centroid_bytes * 2 + (std::size_t{1} << 20);

  auto& index_cache = sirius_ctx->get_cuvs_index_cache();
  // Drop any existing index on this (table, column, metric) before reserving the
  // new one, so its GPU reservation is freed first.
  index_cache.erase_by_column(entry.name, data.column_name, metric);
  auto reservation = index_cache.reserve_index_memory(footprint, target_gpu);
  if (!reservation) {
    auto const avail = target_space->get_available_memory();
    throw InvalidInputException(
      "sirius_create_ann_index: not enough free GPU memory to build the index for '" + entry.name +
      "." + data.column_name + "': need ~" + std::to_string(footprint >> 20) + " MiB, only ~" +
      std::to_string(avail >> 20) + " MiB free on GPU " + std::to_string(target_gpu));
  }

  // Build IVF-Flat through the reservation's resource, then pin it
  auto handle = sirius::vss::build_ivf_flat_index_from_chunks(
    chunk_views, dim, n_lists, metric, reservation->get_memory_resource());
  reservation->shrink_to_fit();

  sirius::vss::index_metadata meta;
  meta.kind           = ann_index_kind_from_type(data.index_type);
  meta.table_name     = entry.name;  // catalog-resolved name (matches query-side derivation)
  meta.column_name    = data.column_name;
  meta.dim            = dim;
  meta.num_rows       = n_rows;
  meta.n_lists        = static_cast<int64_t>(n_lists);
  meta.metric         = metric;
  meta.reserved_bytes = reservation->size();
  index_cache.insert(data.index_name, std::move(meta), std::move(handle), std::move(reservation));

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct SiriusVectorSearchBindData : public TableFunctionData {
  sirius::vss::vector_search_request req;
  // Output column types + trailing distance, for the host_table_chunk_reader.
  duckdb::vector<sirius::logical_type> reader_types;
};

struct SiriusVectorSearchGlobalState : public GlobalTableFunctionState {
  std::unique_ptr<cucascade::host_data_representation> host_repr;
  std::unique_ptr<sirius::op::result::host_table_chunk_reader> reader;
};

// Pull the float components out of the query argument, accepting either a
// FLOAT[] ARRAY (e.g. [1,2,3]::FLOAT[3]) or a LIST of numbers.
static std::vector<float> vector_search_query_floats(const Value& query)
{
  auto const id                         = query.type().id();
  const duckdb::vector<Value>* children = nullptr;
  if (id == LogicalTypeId::ARRAY) {
    children = &ArrayValue::GetChildren(query);
  } else if (id == LogicalTypeId::LIST) {
    children = &ListValue::GetChildren(query);
  } else {
    throw BinderException(
      "sirius_knn_search: query (3rd argument) must be a FLOAT array, e.g. [..]::FLOAT[N]");
  }
  std::vector<float> out;
  out.reserve(children->size());
  for (auto const& child : *children) {
    if (child.IsNull()) {
      throw BinderException("sirius_knn_search: query vector must not contain NULLs");
    }
    out.push_back(child.GetValue<float>());
  }
  return out;
}

static unique_ptr<FunctionData> SiriusVectorSearchBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<SiriusVectorSearchBindData>();
  auto& req   = result->req;

  // Required params
  if (input.inputs.size() < 3 || input.inputs[0].IsNull() || input.inputs[1].IsNull() ||
      input.inputs[2].IsNull()) {
    throw BinderException(
      "sirius_knn_search requires three non-NULL positional arguments: table, column, query");
  }
  req.table_name  = input.inputs[0].ToString();
  req.column_name = input.inputs[1].ToString();
  req.query       = vector_search_query_floats(input.inputs[2]);

  // Optional params' default values
  req.metric              = "l2";
  req.k                   = 10;
  req.use_index           = true;
  req.n_probes            = 0;
  std::string schema_name = "main";
  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_knn_search: named parameter '" + kv.first + "' cannot be NULL");
    }
    if (key == "k") {
      req.k = kv.second.GetValue<int64_t>();
    } else if (key == "metric") {
      req.metric = StringUtil::Lower(kv.second.ToString());
    } else if (key == "use_index") {
      req.use_index = kv.second.GetValue<bool>();
    } else if (key == "n_probes") {
      req.n_probes = kv.second.GetValue<int64_t>();
    } else if (key == "schema_name") {
      schema_name = kv.second.ToString();
    } else if (key == "output_columns") {
      for (auto const& c : ListValue::GetChildren(kv.second)) {
        req.output_columns.push_back(c.ToString());
      }
    }
  }
  if (req.k <= 0) { throw BinderException("sirius_knn_search: k must be >= 1"); }
  if (req.n_probes < 0) { throw BinderException("sirius_knn_search: n_probes must be >= 0"); }
  if (req.metric != "l2" && req.metric != "cosine") {
    throw BinderException("sirius_knn_search: metric must be one of 'l2', 'cosine', got '" +
                          req.metric + "'");
  }

  // Resolve the vector column's dimensionality and each output column's type
  // from the catalog so the return schema and the host reader agree.
  auto const qname          = QualifiedName::Parse(req.table_name);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_name;
  auto& entry_base =
    Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  auto& entry             = entry_base.Cast<DuckTableEntry>();
  req.catalog             = entry.ParentCatalog().GetName();
  req.schema              = entry.ParentSchema().name;
  req.table_name          = entry.name;  // catalog-resolved name (matches query-side derivation)
  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();

  // Default output_columns to every base-table column (in schema order) when omitted.
  if (req.output_columns.empty()) {
    req.output_columns.assign(schema_names.begin(), schema_names.end());
  }

  auto type_of = [&](const std::string& col) -> const LogicalType& {
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      if (schema_names[i] == col) { return schema_types[i]; }
    }
    throw BinderException("sirius_knn_search: column '" + col + "' not found in table '" +
                          req.table_name + "'");
  };

  auto const& vec_type = type_of(req.column_name);
  if (vec_type.id() != LogicalTypeId::ARRAY ||
      ArrayType::GetChildType(vec_type).id() != LogicalTypeId::FLOAT) {
    throw BinderException("sirius_knn_search: column '" + req.column_name +
                          "' must be a FLOAT[N] array column");
  }
  req.dim = static_cast<int64_t>(ArrayType::GetSize(vec_type));
  if (static_cast<int64_t>(req.query.size()) != req.dim) {
    throw BinderException("sirius_knn_search: query has " + std::to_string(req.query.size()) +
                          " elements but column '" + req.column_name + "' is FLOAT[" +
                          std::to_string(req.dim) + "]");
  }

  for (auto const& col : req.output_columns) {
    return_types.push_back(type_of(col));
    names.push_back(col);
  }
  return_types.push_back(LogicalType::FLOAT);
  names.push_back("distance");

  result->reader_types = sirius::from_duckdb_vec(return_types);
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> SiriusVectorSearchInit(ClientContext& context,
                                                                   TableFunctionInitInput& input)
{
  nvtx3::scoped_range nvtx_range{"SiriusVectorSearchInit"};
  auto& bind_data = input.bind_data->Cast<SiriusVectorSearchBindData>();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("sirius_knn_search requires the Sirius context to be initialized");
  }

  auto state       = make_uniq<SiriusVectorSearchGlobalState>();
  state->host_repr = sirius::vss::run_vector_search(*sirius_ctx, bind_data.req);
  state->reader    = std::make_unique<sirius::op::result::host_table_chunk_reader>(
    context, *state->host_repr, bind_data.reader_types);
  return std::move(state);
}

static void SiriusVectorSearchFunction(ClientContext& context,
                                       TableFunctionInput& data_p,
                                       DataChunk& output)
{
  auto& state = data_p.global_state->Cast<SiriusVectorSearchGlobalState>();
  state.reader->get_next_chunk(output);
}

//! Resolve a (table, column) pair against the catalog and return the vector column's
//! dimensionality, writing the catalog-resolved names back through the out-params so they match
//! the identity the scan manager pinned the table under.
static int64_t ResolveVectorColumn(ClientContext& context,
                                   const std::string& fn,
                                   const std::string& table_arg,
                                   const std::string& schema_arg,
                                   const std::string& column_name,
                                   std::string& out_catalog,
                                   std::string& out_schema,
                                   std::string& out_table)
{
  auto const qname          = QualifiedName::Parse(table_arg);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_arg;
  auto& entry_base =
    Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  auto& entry = entry_base.Cast<DuckTableEntry>();
  out_catalog = entry.ParentCatalog().GetName();
  out_schema  = entry.ParentSchema().name;
  out_table   = entry.name;

  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();
  for (std::size_t i = 0; i < schema_names.size(); ++i) {
    if (schema_names[i] != column_name) { continue; }
    auto const& type = schema_types[i];
    if (type.id() != LogicalTypeId::ARRAY ||
        ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
      throw BinderException(fn + ": column '" + column_name + "' must be a FLOAT[N] array column");
    }
    return static_cast<int64_t>(ArrayType::GetSize(type));
  }
  throw BinderException(fn + ": column '" + column_name + "' not found in table '" + table_arg +
                        "'");
}

struct KMeansFitData : public TableFunctionData {
  sirius::vss::kmeans_fit_request req;
  bool finished = false;
};

static unique_ptr<FunctionData> SiriusKMeansFitBind(ClientContext& context,
                                                    TableFunctionBindInput& input,
                                                    vector<LogicalType>& return_types,
                                                    vector<string>& names)
{
  auto result = make_uniq<KMeansFitData>();
  auto& req   = result->req;

  if (input.inputs.size() < 2 || input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
    throw BinderException(
      "sirius_kmeans_fit requires two non-NULL positional arguments: table and column");
  }
  auto const table_arg = input.inputs[0].ToString();
  req.column           = input.inputs[1].ToString();

  std::string schema_name = "main";
  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_kmeans_fit: named parameter '" + kv.first + "' cannot be NULL");
    }
    if (key == "name") {
      req.name = kv.second.ToString();
    } else if (key == "n_clusters") {
      req.spec.n_clusters = kv.second.GetValue<int64_t>();
    } else if (key == "train_rows") {
      req.spec.train_rows = kv.second.GetValue<int64_t>();
    } else if (key == "n_iters") {
      req.spec.n_iters = kv.second.GetValue<int32_t>();
    } else if (key == "seed") {
      req.spec.seed = static_cast<uint64_t>(kv.second.GetValue<int64_t>());
    } else if (key == "metric") {
      req.metric = StringUtil::Lower(kv.second.ToString());
    } else if (key == "schema_name") {
      schema_name = kv.second.ToString();
    }
  }
  if (req.name.empty()) {
    throw BinderException("sirius_kmeans_fit requires a 'name' named parameter to store the "
                          "clustering under");
  }
  if (req.spec.n_clusters < 0) {
    throw BinderException("sirius_kmeans_fit: n_clusters must be >= 0 (0 = auto)");
  }
  if (req.spec.train_rows < 0) {
    throw BinderException("sirius_kmeans_fit: train_rows must be >= 0 (0 = auto)");
  }
  if (req.spec.n_iters < 1) { throw BinderException("sirius_kmeans_fit: n_iters must be >= 1"); }
  if (req.metric != "l2" && req.metric != "cosine") {
    throw BinderException("sirius_kmeans_fit: metric must be one of 'l2', 'cosine', got '" +
                          req.metric + "'");
  }

  req.dim = ResolveVectorColumn(
    context, "sirius_kmeans_fit", table_arg, schema_name, req.column, req.catalog, req.schema,
    req.table);

  return_types = {LogicalType::BIGINT,
                  LogicalType::BIGINT,
                  LogicalType::BIGINT,
                  LogicalType::BIGINT};
  names        = {"n_clusters", "dim", "train_rows", "n_rows"};
  return std::move(result);
}

static void SiriusKMeansFitFunction(ClientContext& context,
                                    TableFunctionInput& data_p,
                                    DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<KMeansFitData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("sirius_kmeans_fit requires the Sirius context to be initialized");
  }

  auto const fit = sirius::vss::run_kmeans_fit(*sirius_ctx, data.req);
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(fit.n_clusters));
  output.SetValue(1, 0, Value::BIGINT(fit.dim));
  output.SetValue(2, 0, Value::BIGINT(fit.train_rows));
  output.SetValue(3, 0, Value::BIGINT(fit.n_rows));
  data.finished = true;
}

struct KMeansAssignBindData : public TableFunctionData {
  sirius::vss::kmeans_assign_request req;
  duckdb::vector<sirius::logical_type> reader_types;
};

struct KMeansAssignGlobalState : public GlobalTableFunctionState {
  std::unique_ptr<cucascade::host_data_representation> host_repr;
  std::unique_ptr<sirius::op::result::host_table_chunk_reader> reader;
};

static unique_ptr<FunctionData> SiriusKMeansAssignBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<KMeansAssignBindData>();
  auto& req   = result->req;

  if (input.inputs.size() < 3 || input.inputs[0].IsNull() || input.inputs[1].IsNull() ||
      input.inputs[2].IsNull()) {
    throw BinderException("sirius_kmeans_assign requires three non-NULL positional arguments: "
                          "table, column and clustering");
  }
  auto const table_arg = input.inputs[0].ToString();
  req.column           = input.inputs[1].ToString();
  req.clustering       = input.inputs[2].ToString();

  std::string schema_name = "main";
  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_kmeans_assign: named parameter '" + kv.first +
                            "' cannot be NULL");
    }
    if (key == "n_probes") {
      req.spec.n_probes = kv.second.GetValue<int64_t>();
    } else if (key == "radius_factor") {
      req.spec.radius_factor = kv.second.GetValue<double>();
    } else if (key == "max_probes") {
      req.spec.max_probes = kv.second.GetValue<int64_t>();
    } else if (key == "schema_name") {
      schema_name = kv.second.ToString();
    }
  }
  if (req.spec.n_probes < 1) { throw BinderException("sirius_kmeans_assign: n_probes must be >= 1"); }
  if (req.spec.radius_factor < 0.0) {
    throw BinderException("sirius_kmeans_assign: radius_factor must be >= 0 (0 = fixed n_probes)");
  }
  if (req.spec.max_probes < 0) {
    throw BinderException("sirius_kmeans_assign: max_probes must be >= 0 (0 = n_probes)");
  }

  req.dim = ResolveVectorColumn(context,
                                "sirius_kmeans_assign",
                                table_arg,
                                schema_name,
                                req.column,
                                req.catalog,
                                req.schema,
                                req.table);

  return_types = {LogicalType::BIGINT, LogicalType::INTEGER, LogicalType::FLOAT};
  names        = {"row_id", "cluster_id", "distance"};

  result->reader_types = sirius::from_duckdb_vec(return_types);
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> SiriusKMeansAssignInit(ClientContext& context,
                                                                   TableFunctionInitInput& input)
{
  auto& bind_data = input.bind_data->Cast<KMeansAssignBindData>();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException(
      "sirius_kmeans_assign requires the Sirius context to be initialized");
  }

  auto state       = make_uniq<KMeansAssignGlobalState>();
  state->host_repr = sirius::vss::run_kmeans_assign(*sirius_ctx, bind_data.req);
  state->reader    = std::make_unique<sirius::op::result::host_table_chunk_reader>(
    context, *state->host_repr, bind_data.reader_types);
  return std::move(state);
}

static void SiriusKMeansAssignFunction(ClientContext& context,
                                       TableFunctionInput& data_p,
                                       DataChunk& output)
{
  auto& state = data_p.global_state->Cast<KMeansAssignGlobalState>();
  state.reader->get_next_chunk(output);
}

struct KMeansCentroidsBindData : public TableFunctionData {
  std::string clustering;
  duckdb::vector<sirius::logical_type> reader_types;
};

static unique_ptr<FunctionData> SiriusKMeansCentroidsBind(ClientContext& context,
                                                          TableFunctionBindInput& input,
                                                          vector<LogicalType>& return_types,
                                                          vector<string>& names)
{
  auto result = make_uniq<KMeansCentroidsBindData>();
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException(
      "sirius_kmeans_centroids requires one non-NULL positional argument: the clustering name");
  }
  result->clustering = input.inputs[0].ToString();

  return_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::FLOAT};
  names        = {"cluster_id", "dim_index", "value"};

  result->reader_types = sirius::from_duckdb_vec(return_types);
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> SiriusKMeansCentroidsInit(
  ClientContext& context, TableFunctionInitInput& input)
{
  auto& bind_data = input.bind_data->Cast<KMeansCentroidsBindData>();
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException(
      "sirius_kmeans_centroids requires the Sirius context to be initialized");
  }
  auto state       = make_uniq<KMeansAssignGlobalState>();
  state->host_repr = sirius::vss::run_kmeans_centroids(*sirius_ctx, bind_data.clustering);
  state->reader    = std::make_unique<sirius::op::result::host_table_chunk_reader>(
    context, *state->host_repr, bind_data.reader_types);
  return std::move(state);
}

static void SiriusKMeansCentroidsFunction(ClientContext& context,
                                          TableFunctionInput& data_p,
                                          DataChunk& output)
{
  auto& state = data_p.global_state->Cast<KMeansAssignGlobalState>();
  state.reader->get_next_chunk(output);
}

using sirius::vss::parse_output_columns;
using sirius::vss::resolve_vector_join_side;
using sirius::vss::SiriusVectorJoinBindData;
using sirius::vss::vector_join_mode;
using sirius::vss::vector_join_output_type;
using sirius::vss::vector_join_search_mode;
//! Shared bind for both surfaces. `relational` selects the one whose probe side is a bound
//! subquery rather than a table name: its vector column and output columns are resolved against
//! the input relation's schema, and it needs no pin on that side at all.
static unique_ptr<FunctionData> VectorJoinBindImpl(ClientContext& context,
                                                   TableFunctionBindInput& input,
                                                   vector<LogicalType>& return_types,
                                                   vector<string>& names,
                                                   bool relational)
{
  auto result = make_uniq<SiriusVectorJoinBindData>();
  auto& req   = result->req;
  // The relational form's probe is the child relation, so it always runs the scan path on that
  // side; the caller has no say and there is no `probe_source` to set.
  req.probe_from_scan   = relational;
  result->probe_is_relation = relational;

  // Required positional params. The relational form's first argument is the subquery, which
  // reaches the bind as an empty placeholder with the relation's schema in input_table_*.
  if (input.inputs.size() < 4 || (!relational && input.inputs[0].IsNull()) ||
      input.inputs[1].IsNull() || input.inputs[2].IsNull() || input.inputs[3].IsNull()) {
    throw BinderException(
      "sirius_knn_join requires four non-NULL positional arguments: "
      "left_table, left_column, right_table, right_column");
  }
  auto const left_table   = relational ? std::string{} : input.inputs[0].ToString();
  auto const left_column  = input.inputs[1].ToString();
  auto const right_table  = input.inputs[2].ToString();
  auto const right_column = input.inputs[3].ToString();

  // Optional params' default values
  req.metric               = "l2";
  req.k                    = 10;
  req.search_mode          = vector_join_search_mode::exact_gemm;
  req.n_clusters           = 0;
  req.n_probes             = 1;
  req.eps                  = 0.0;
  bool join_mode_is_set    = false;
  bool output_type_is_set  = false;
  std::string left_schema  = "main";
  std::string right_schema = "main";
  std::vector<std::string> left_out_cols;
  std::vector<std::string> right_out_cols;
  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_knn_join: named parameter '" + kv.first + "' cannot be NULL");
    }
    if (key == "k") {
      req.k = kv.second.GetValue<int64_t>();
    } else if (key == "n_clusters") {
      req.n_clusters = kv.second.GetValue<int64_t>();
    } else if (key == "n_probes") {
      req.n_probes = kv.second.GetValue<int64_t>();
    } else if (key == "eps") {
      req.eps = kv.second.GetValue<double>();
    } else if (key == "metric") {
      req.metric = StringUtil::Lower(kv.second.ToString());
    } else if (key == "search_mode") {
      auto const mode = StringUtil::Lower(kv.second.ToString());
      if (mode == "exact") {
        req.search_mode = vector_join_search_mode::exact;
      } else if (mode == "exact-gemm") {
        req.search_mode = vector_join_search_mode::exact_gemm;
      } else if (mode == "approx") {
        req.search_mode = vector_join_search_mode::approx;
      } else {
        throw BinderException(
          "sirius_knn_join: search_mode must be 'exact', 'exact-gemm', or 'approx'");
      }
    } else if (key == "join_mode") {
      auto const jm = StringUtil::Lower(kv.second.ToString());
      if (jm == "global") {
        req.mode = vector_join_mode::global_top_k;
      } else if (jm == "per-row") {
        req.mode = vector_join_mode::per_row_top_k;
      } else if (jm == "threshold") {
        req.mode = vector_join_mode::threshold;
      } else {
        throw BinderException(
          "sirius_knn_join: join_mode must be 'global', 'per-row', or 'threshold'");
      }
      join_mode_is_set = true;
    } else if (key == "output_type") {
      auto const out_type = StringUtil::Lower(kv.second.ToString());
      if (out_type == "similarity") {
        req.output_type = vector_join_output_type::similarity;
      } else if (out_type == "distance") {
        req.output_type = vector_join_output_type::distance;
      } else {
        throw BinderException("sirius_knn_join: output_type must be 'similarity' or 'distance'");
      }
      output_type_is_set = true;
    } else if (key == "probe_source") {
      if (relational) {
        throw BinderException(
          "sirius_knn_join_rel: probe_source does not apply; the probe side is the relation "
          "passed as the first argument");
      }
      auto const src = StringUtil::Lower(kv.second.ToString());
      if (src == "scan") {
        req.probe_from_scan = true;
      } else if (src != "pin") {
        throw BinderException("sirius_knn_join: probe_source must be 'pin' or 'scan'");
      }
    } else if (key == "build_source") {
      auto const src = StringUtil::Lower(kv.second.ToString());
      if (src == "scan") {
        req.build_from_scan = true;
      } else if (src != "pin") {
        throw BinderException("sirius_knn_join: build_source must be 'pin' or 'scan'");
      }
    } else if (key == "left_schema_name") {
      left_schema = kv.second.ToString();
    } else if (key == "right_schema_name") {
      right_schema = kv.second.ToString();
    } else if (key == "left_output_columns") {
      left_out_cols = parse_output_columns(kv.second, "left_output_columns");
    } else if (key == "right_output_columns") {
      right_out_cols = parse_output_columns(kv.second, "right_output_columns");
    }
  }
  if (req.k < 0) { throw BinderException("sirius_knn_join: k must be >= 0"); }
  if (req.n_clusters < 0) { throw BinderException("sirius_knn_join: n_clusters must be >= 0"); }
  if (req.n_probes < 1) { throw BinderException("sirius_knn_join: n_probes must be >= 1"); }
  // Approximate search is unimplemented: nothing downstream reads search_mode beyond the
  // exact/exact-gemm distinction, nor n_clusters/n_probes at all, so accepting them ran an
  // exhaustive search while the query said otherwise -- results indistinguishable from an
  // approximate run that happened to be perfect. Refusing is the only honest answer until
  // there is a real approximate path.
  if (req.search_mode == vector_join_search_mode::approx) {
    throw BinderException(
      "sirius_knn_join: search_mode => 'approx' is not implemented; it would run an exhaustive "
      "search and report itself as approximate. Use 'exact' or 'exact-gemm'");
  }
  if (req.n_clusters > 0 || req.n_probes > 1) {
    throw BinderException(
      "sirius_knn_join: n_clusters and n_probes only mean anything under an approximate search, "
      "which is not implemented; they are ignored, so setting them is refused rather than "
      "silently disregarded");
  }
  if (req.eps < 0.0) { throw BinderException("sirius_knn_join: eps must be >= 0"); }
  if (req.metric != "l2" && req.metric != "cosine") {
    throw BinderException("sirius_knn_join: metric must be one of 'l2', 'cosine', got '" +
                          req.metric + "'");
  }
  if (!join_mode_is_set) {
    req.mode = req.eps > 0.0 ? vector_join_mode::threshold : vector_join_mode::per_row_top_k;
  }
  if (!output_type_is_set) {
    req.output_type = req.metric == "cosine" ? vector_join_output_type::similarity
                                             : vector_join_output_type::distance;
  }
  // Similarity is only well-defined for cosine (1 - distance). L2 has no natural similarity.
  if (req.metric == "l2" && req.output_type == vector_join_output_type::similarity) {
    throw BinderException(
      "sirius_knn_join: output_type => 'similarity' is only meaningful for metric => 'cosine'; "
      "use output_type => 'distance' for l2");
  }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("sirius_knn_join requires the Sirius context to be initialized");
  }

  auto const left_dim =
    relational ? sirius::vss::resolve_relational_probe_side(input.input_table_types,
                                                            input.input_table_names,
                                                            left_column,
                                                            left_out_cols,
                                                            req.left,
                                                            return_types,
                                                            names)
               : resolve_vector_join_side(context,
                                          *sirius_ctx,
                                          "left",
                                          left_table,
                                          left_column,
                                          left_schema,
                                          left_out_cols,
                                          /*require_pin=*/!req.probe_from_scan,
                                          req.left,
                                          return_types,
                                          names,
                                          result->left_rows);
  auto const right_dim = resolve_vector_join_side(context,
                                                  *sirius_ctx,
                                                  "right",
                                                  right_table,
                                                  right_column,
                                                  right_schema,
                                                  right_out_cols,
                                                  /*require_pin=*/!req.build_from_scan,
                                                  req.right,
                                                  return_types,
                                                  names,
                                                  result->right_rows);
  if (left_dim != right_dim) {
    throw BinderException("sirius_knn_join: left is FLOAT[" + std::to_string(left_dim) +
                          "] but right is FLOAT[" + std::to_string(right_dim) +
                          "]; both sides must share the same vector dimensionality");
  }
  req.dim = left_dim;

  // The split path (SIRIUS_VECTOR_JOIN_STREAMING=0) merges each right chunk's partial on
  // its own instead of folding the whole partition, and materialize's fixed-k slicing then
  // attributes neighbours to the wrong left rows. The result passes every downstream check,
  // so refuse it here rather than return scrambled rows. Bind time, not plan time: the plan
  // generator's exceptions are caught and turned into a CPU fallback, where this table
  // function refuses again with an unrelated message.
  auto pinned_entry_for = [&](const sirius::vss::vector_join_side& side) {
    return sirius_ctx->get_scan_manager().find_pinned_entry_for_duckdb_table(
      side.catalog, side.schema, side.table);
  };
  auto pinned_chunk_count = [&](const sirius::vss::vector_join_side& side) -> std::size_t {
    const auto* pin = pinned_entry_for(side);
    if (pin == nullptr) { return 0; }
    return pin->tier == cucascade::memory::Tier::HOST ? pin->host_chunks.size()
                                                      : pin->chunk_memory_spaces.size();
  };

  const char* streaming_env = std::getenv("SIRIUS_VECTOR_JOIN_STREAMING");
  if (streaming_env != nullptr && std::string_view{streaming_env} == "0") {
    auto const right_chunks = pinned_chunk_count(req.right);
    if (right_chunks > 1) {
      throw BinderException(
        "sirius_knn_join: SIRIUS_VECTOR_JOIN_STREAMING=0 selects the split path, which does "
        "not merge across right batches; table '" +
        req.right.table + "' is pinned in " + std::to_string(right_chunks) +
        " chunks. Unset the variable to use the streaming operator.");
    }
  }


  return_types.push_back(LogicalType::FLOAT);
  names.push_back(req.output_type == vector_join_output_type::similarity ? "similarity"
                                                                         : "distance");
  return std::move(result);
}

static unique_ptr<NodeStatistics> SiriusVectorJoinCardinality(ClientContext&,
                                                              FunctionData const* bind_data_p)
{
  auto const* typed = dynamic_cast<SiriusVectorJoinBindData const*>(bind_data_p);
  if (typed == nullptr) { return nullptr; }
  // A relational probe's row count is the child's, which the bind cannot see and this callback
  // is not given. Declining lets LogicalGet fall back to children[0]'s estimate -- the input
  // count rather than the output's, so short by k, but derived from the actual filtered
  // subquery instead of an unfiltered table. create_plan_knn_join applies the k itself, where
  // the child's estimate is in hand.
  if (typed->probe_is_relation) { return nullptr; }
  auto const rows = sirius::vss::estimate_vector_join_cardinality(
    typed->req, typed->left_rows, typed->right_rows);
  // Sound as a maximum in every mode: threshold only ever drops pairs from the
  // same per-row top-k candidate set the other modes emit in full.
  return make_uniq<NodeStatistics>(rows, rows);
}

//! Stamp the vector join's real output cardinality onto its LogicalGet.
//!
//! `LogicalGet::EstimateCardinality` consults the function's cardinality callback, which is
//! handed only the bind data. That is enough for the name-taking form, whose row counts are
//! known at bind. It is not enough for `sirius_knn_join_rel`, whose probe is a subquery: the
//! bind cannot see the child's estimate, so the callback declines and DuckDB falls back to
//! `children[0]->EstimateCardinality()` -- the join's INPUT count, when the join emits k rows
//! per input row. Everything planned above the operator is then sized an order of magnitude
//! low at k=10.
//!
//! Stamping it here fixes that, because `has_estimated_cardinality` takes precedence over both
//! the callback and the child fallback.
static void stamp_vector_join_cardinality(ClientContext& context, LogicalOperator& op)
{
  for (auto& child : op.children) {
    stamp_vector_join_cardinality(context, *child);
  }
  if (op.type != LogicalOperatorType::LOGICAL_GET) { return; }
  auto& get = op.Cast<LogicalGet>();
  if (get.function.name != "sirius_knn_join_rel" || get.children.empty() || !get.bind_data) {
    return;
  }
  auto const* bind = dynamic_cast<SiriusVectorJoinBindData const*>(get.bind_data.get());
  if (bind == nullptr) { return; }

  auto const k = static_cast<idx_t>(std::max<std::int64_t>(bind->req.k, 0));
  if (bind->req.mode == vector_join_mode::global_top_k) {
    get.SetEstimatedCardinality(k);
    return;
  }
  auto const child_rows = get.children[0]->EstimateCardinality(context);
  if (k != 0 && child_rows > NumericLimits<idx_t>::Maximum() / k) {
    get.SetEstimatedCardinality(NumericLimits<idx_t>::Maximum());
    return;
  }
  get.SetEstimatedCardinality(child_rows * k);
}

//! Runs before every built-in optimizer, so JOIN_ORDER sees the stamped value. The child has
//! not been optimized yet at this point -- a probe-side predicate is still a LogicalFilter
//! above the scan rather than pushed into it -- so the count feeding this is rougher than the
//! final one. It is still the k factor that dominates the error, which is what this fixes.
static void SiriusPreOptimizeVectorJoin(OptimizerExtensionInput& input,
                                        unique_ptr<LogicalOperator>& plan)
{
  if (plan) { stamp_vector_join_cardinality(input.context, *plan); }
}

//! Runs after them, when the child's estimate reflects the pushed-down filter. Too late to
//! change a join order, but it is what EXPLAIN and the physical planner read, and it overwrites
//! the rougher pre-pass value.
static void SiriusPostOptimizeVectorJoin(OptimizerExtensionInput& input,
                                         unique_ptr<LogicalOperator>& plan)
{
  if (plan) { stamp_vector_join_cardinality(input.context, *plan); }
}

static unique_ptr<FunctionData> SiriusVectorJoinBind(ClientContext& context,
                                                     TableFunctionBindInput& input,
                                                     vector<LogicalType>& return_types,
                                                     vector<string>& names)
{
  return VectorJoinBindImpl(context, input, return_types, names, /*relational=*/false);
}

static unique_ptr<FunctionData> SiriusVectorJoinRelBind(ClientContext& context,
                                                        TableFunctionBindInput& input,
                                                        vector<LogicalType>& return_types,
                                                        vector<string>& names)
{
  return VectorJoinBindImpl(context, input, return_types, names, /*relational=*/true);
}

// Execute callback
static void SiriusVectorJoinFunction(ClientContext&, TableFunctionInput&, DataChunk&)
{
  throw std::runtime_error(
    "sirius_knn_join is an internal rewrite target executed on the GPU; it cannot run on the CPU");
}

// Present so the binder treats the relational form as taking a table; the operator never runs
// on the CPU, so reaching this is the same error as above.
static OperatorResultType SiriusVectorJoinInOutFunction(ExecutionContext&,
                                                        TableFunctionInput&,
                                                        DataChunk&,
                                                        DataChunk&)
{
  throw std::runtime_error(
    "sirius_knn_join_rel is an internal rewrite target executed on the GPU; it cannot run on "
    "the CPU");
}

void SiriusExtension::RegisterGPUFunctions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);

#ifdef SIRIUS_ENABLE_LEGACY
  TableFunction gpu_buffer_init("gpu_buffer_init",
                                {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                GPUBufferInitFunction,
                                GPUBufferInitBind);
  gpu_buffer_init.named_parameters[PINNED_MEMORY_PARAM_KEY] = LogicalType::VARCHAR;
  CreateTableFunctionInfo gpu_buffer_init_info(gpu_buffer_init);
  catalog.CreateTableFunction(transaction, gpu_buffer_init_info);

  RegisterLegacyGPUFunctions(transaction, catalog);
#endif

  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              SiriusExtension::GPUExecutionBind);
  gpu_execution.named_parameters["enable_optimizer"]    = LogicalType::BOOLEAN;
  gpu_execution.named_parameters[QUERY_LABEL_PARAM_KEY] = LogicalType::VARCHAR;
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);

  // Sirius-owned S3 parquet entry point. gpu_execution rewrites
  // read_parquet('s3://...') to this table function so the bind runs through
  // Sirius's footer-only S3 path instead of DuckDB's native read_parquet.
  // Registered so the rewrite's output binds, but INTERNAL — not a public
  // surface: users query S3 Parquet with read_parquet('s3://...'), not this.
  TableFunction sirius_read_parquet("sirius_read_parquet",
                                    {LogicalType::VARCHAR},
                                    SiriusReadParquetFunction,
                                    SiriusReadParquetBind);
  sirius_read_parquet.cardinality         = SiriusReadParquetCardinality;
  sirius_read_parquet.projection_pushdown = true;
  sirius_read_parquet.filter_pushdown     = true;
  sirius_read_parquet.filter_prune        = true;
  CreateTableFunctionInfo sirius_read_parquet_info(sirius_read_parquet);
  catalog.CreateTableFunction(transaction, sirius_read_parquet_info);

  TableFunction set_query_label("sirius_set_query_label",
                                {LogicalType::VARCHAR},
                                SiriusSetQueryLabelFunction,
                                SiriusSetQueryLabelBind);
  CreateTableFunctionInfo set_query_label_info(set_query_label);
  catalog.CreateTableFunction(transaction, set_query_label_info);

  // Profiler control functions for nsys --capture-range=cudaProfilerApi
  TableFunction profiler_start(
    "profiler_start", {}, ProfilerStartFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_start_info(profiler_start);
  catalog.CreateTableFunction(transaction, profiler_start_info);

  TableFunction profiler_stop(
    "profiler_stop", {}, ProfilerStopFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_stop_info(profiler_stop);
  catalog.CreateTableFunction(transaction, profiler_stop_info);

  // pin_table takes either a positional path (parquet) or no positional (duckdb,
  // where 'name' is the catalog table reference) — register both arities as a set.
  TableFunctionSet pin_table_set("pin_table");
  auto add_pin_table_overload = [&](vector<LogicalType> positional_args) {
    TableFunction pin_table(
      "pin_table", std::move(positional_args), PinTableFunction, PinTableBind);
    pin_table.named_parameters["tier"]        = LogicalType::VARCHAR;
    pin_table.named_parameters["name"]        = LogicalType::VARCHAR;
    pin_table.named_parameters["cols"]        = LogicalType::LIST(LogicalType::VARCHAR);
    pin_table.named_parameters["format"]      = LogicalType::VARCHAR;
    pin_table.named_parameters["schema_name"] = LogicalType::VARCHAR;
    pin_table_set.AddFunction(std::move(pin_table));
  };
  add_pin_table_overload({LogicalType::VARCHAR});
  add_pin_table_overload({});
  CreateTableFunctionInfo pin_table_info(pin_table_set);
  catalog.CreateTableFunction(transaction, pin_table_info);

  TableFunction unpin_table(
    "unpin_table", {LogicalType::VARCHAR}, UnpinTableFunction, UnpinTableBind);
  CreateTableFunctionInfo unpin_table_info(unpin_table);
  catalog.CreateTableFunction(transaction, unpin_table_info);

  // sirius_create_ann_index(table, column, name=>, metric=>, index_type=>, n_lists=>,
  // schema_name=>)
  TableFunction create_ann_index("sirius_create_ann_index",
                                 {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                 SiriusCreateAnnIndexFunction,
                                 SiriusCreateAnnIndexBind);
  create_ann_index.named_parameters["name"]        = LogicalType::VARCHAR;
  create_ann_index.named_parameters["metric"]      = LogicalType::VARCHAR;
  create_ann_index.named_parameters["index_type"]  = LogicalType::VARCHAR;
  create_ann_index.named_parameters["n_lists"]     = LogicalType::BIGINT;
  create_ann_index.named_parameters["schema_name"] = LogicalType::VARCHAR;
  CreateTableFunctionInfo create_ann_index_info(create_ann_index);
  catalog.CreateTableFunction(transaction, create_ann_index_info);

  // sirius_knn_search(table, column, query, k =>, output_columns =>, metric =>,
  // use_index =>, n_probes =>, schema_name =>)
  TableFunction vector_search("sirius_knn_search",
                              {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::ANY},
                              SiriusVectorSearchFunction,
                              SiriusVectorSearchBind,
                              SiriusVectorSearchInit);
  vector_search.named_parameters["k"]              = LogicalType::BIGINT;
  vector_search.named_parameters["output_columns"] = LogicalType::LIST(LogicalType::VARCHAR);
  vector_search.named_parameters["metric"]         = LogicalType::VARCHAR;
  vector_search.named_parameters["use_index"]      = LogicalType::BOOLEAN;
  vector_search.named_parameters["n_probes"]       = LogicalType::BIGINT;
  vector_search.named_parameters["schema_name"]    = LogicalType::VARCHAR;
  CreateTableFunctionInfo vector_search_info(vector_search);
  catalog.CreateTableFunction(transaction, vector_search_info);

  // sirius_kmeans_fit(table, column, name =>, n_clusters =>, train_rows =>, n_iters =>,
  //   seed =>, metric =>, schema_name =>)
  TableFunction kmeans_fit("sirius_kmeans_fit",
                           {LogicalType::VARCHAR, LogicalType::VARCHAR},
                           SiriusKMeansFitFunction,
                           SiriusKMeansFitBind);
  kmeans_fit.named_parameters["name"]        = LogicalType::VARCHAR;
  kmeans_fit.named_parameters["n_clusters"]  = LogicalType::BIGINT;
  kmeans_fit.named_parameters["train_rows"]  = LogicalType::BIGINT;
  kmeans_fit.named_parameters["n_iters"]     = LogicalType::INTEGER;
  kmeans_fit.named_parameters["seed"]        = LogicalType::BIGINT;
  kmeans_fit.named_parameters["metric"]      = LogicalType::VARCHAR;
  kmeans_fit.named_parameters["schema_name"] = LogicalType::VARCHAR;
  CreateTableFunctionInfo kmeans_fit_info(kmeans_fit);
  catalog.CreateTableFunction(transaction, kmeans_fit_info);

  // sirius_kmeans_assign(table, column, clustering, n_probes =>, radius_factor =>,
  //   max_probes =>, schema_name =>)
  TableFunction kmeans_assign(
    "sirius_kmeans_assign",
    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
    SiriusKMeansAssignFunction,
    SiriusKMeansAssignBind,
    SiriusKMeansAssignInit);
  kmeans_assign.named_parameters["n_probes"]      = LogicalType::BIGINT;
  kmeans_assign.named_parameters["radius_factor"] = LogicalType::DOUBLE;
  kmeans_assign.named_parameters["max_probes"]    = LogicalType::BIGINT;
  kmeans_assign.named_parameters["schema_name"]   = LogicalType::VARCHAR;
  CreateTableFunctionInfo kmeans_assign_info(kmeans_assign);
  catalog.CreateTableFunction(transaction, kmeans_assign_info);

  // sirius_kmeans_centroids(clustering)
  TableFunction kmeans_centroids("sirius_kmeans_centroids",
                                 {LogicalType::VARCHAR},
                                 SiriusKMeansCentroidsFunction,
                                 SiriusKMeansCentroidsBind,
                                 SiriusKMeansCentroidsInit);
  CreateTableFunctionInfo kmeans_centroids_info(kmeans_centroids);
  catalog.CreateTableFunction(transaction, kmeans_centroids_info);

  // sirius_knn_join(left_table, left_column, right_table, right_column, k =>, metric =>,
  //   search_mode =>, join_mode =>, n_clusters =>, n_probes =>, eps =>, output_type =>,
  //   left_schema_name =>, right_schema_name =>,
  //   left_output_columns =>, right_output_columns =>)
  TableFunction vector_join(
    "sirius_knn_join",
    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
    SiriusVectorJoinFunction,
    SiriusVectorJoinBind);
  vector_join.named_parameters["k"]                    = LogicalType::BIGINT;
  vector_join.named_parameters["metric"]               = LogicalType::VARCHAR;
  vector_join.named_parameters["search_mode"]          = LogicalType::VARCHAR;
  vector_join.named_parameters["join_mode"]            = LogicalType::VARCHAR;
  vector_join.named_parameters["n_clusters"]           = LogicalType::BIGINT;
  vector_join.named_parameters["n_probes"]             = LogicalType::BIGINT;
  vector_join.named_parameters["eps"]                  = LogicalType::DOUBLE;
  vector_join.named_parameters["output_type"]          = LogicalType::VARCHAR;
  vector_join.named_parameters["left_schema_name"]     = LogicalType::VARCHAR;
  vector_join.named_parameters["right_schema_name"]    = LogicalType::VARCHAR;
  vector_join.named_parameters["left_output_columns"]  = LogicalType::LIST(LogicalType::VARCHAR);
  vector_join.named_parameters["right_output_columns"] = LogicalType::LIST(LogicalType::VARCHAR);
  vector_join.named_parameters["build_source"]         = LogicalType::VARCHAR;
  vector_join.named_parameters["probe_source"]         = LogicalType::VARCHAR;
  vector_join.cardinality                              = SiriusVectorJoinCardinality;
  // Lets DuckDB narrow column_ids to what the query reads. create_plan_knn_join drops the rest
  // before the corpus's output columns are concatenated, which is where the cost is.
  vector_join.projection_pushdown                      = true;
  CreateTableFunctionInfo vector_join_info(vector_join);
  catalog.CreateTableFunction(transaction, vector_join_info);

  // Relational surface: the probe side is a subquery rather than a table name, so a predicate
  // on it is DuckDB's to push into the scan before this ever binds, and the probe can be any
  // relation -- a filtered table, a join, a CTE. Registered under its own name because a
  // function with a TABLE parameter cannot have overloads.
  //   sirius_knn_join_rel(TABLE (SELECT ... FROM users WHERE region = 3), 'vec',
  //                       'items', 'vec', k => 10)
  TableFunction vector_join_rel(
    "sirius_knn_join_rel",
    {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
    SiriusVectorJoinFunction,
    SiriusVectorJoinRelBind);
  vector_join_rel.named_parameters = vector_join.named_parameters;
  vector_join_rel.cardinality          = SiriusVectorJoinCardinality;
  vector_join_rel.projection_pushdown  = true;
  vector_join_rel.in_out_function  = SiriusVectorJoinInOutFunction;
  CreateTableFunctionInfo vector_join_rel_info(vector_join_rel);
  catalog.CreateTableFunction(transaction, vector_join_rel_info);
}

static void SetUsePinMemory(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_PIN_MEM_FOR_CPU_PROCESSING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CPU_PROCESSING to {}",
                   Config::USE_PIN_MEM_FOR_CPU_PROCESSING);
}

static void SetUsePinMemoryForCaching(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_PIN_MEM_FOR_CACHING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CACHING to {}", Config::USE_PIN_MEM_FOR_CACHING);
}

static void SetUseCudfExpr(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_CUDF_EXPR = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUDF_EXPR to {}", Config::USE_CUDF_EXPR);
}

static void ApplyExpressionEvaluatorStrategy(const std::string& value)
{
  sirius::expression_evaluator_strategy parsed;
  if (!sirius::string_to_strategy(value, parsed)) {
    throw InvalidInputException(
      "Invalid expression_evaluator_strategy '{}'. Valid values: materialize, ast_interpret, "
      "ast_jit",
      value);
  }
  Config::EXPRESSION_EVALUATOR_STRATEGY = parsed;
  SIRIUS_LOG_DEBUG("Updated config EXPRESSION_EVALUATOR_STRATEGY to {}",
                   sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY));
}

static void SetExpressionEvaluatorStrategy(ClientContext& context, SetScope scope, Value& parameter)
{
  ApplyExpressionEvaluatorStrategy(StringValue::Get(parameter));
}

// Deprecated alias for `expression_evaluator_strategy`. Kept so existing
// `SET expression_executor_strategy=...` statements keep working; remove in a future release.
static void SetExpressionExecutorStrategyDeprecated(ClientContext& context,
                                                    SetScope scope,
                                                    Value& parameter)
{
  SIRIUS_LOG_WARN(
    "The 'expression_executor_strategy' setting is deprecated; use "
    "'expression_evaluator_strategy' instead.");
  ApplyExpressionEvaluatorStrategy(StringValue::Get(parameter));
}

static void SetUseCustomTopN(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_CUSTOM_TOP_N = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUSTOM_TOP_N to {}", Config::USE_CUSTOM_TOP_N);
}

static void SetUseOptTableScan(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_OPT_TABLE_SCAN = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_OPT_TABLE_SCAN to {}", Config::USE_OPT_TABLE_SCAN);
}

static void SetOptTableScanNumStreams(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS = IntegerValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_NUM_CUDA_STREAMS to {}",
                   Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
}

static void SetOptTableScanMemcpySize(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE to {}",
                   Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE);
}

static void SetPrintGPUTableMaxRows(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::PRINT_GPU_TABLE_MAX_ROWS = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config PRINT_GPU_TABLE_MAX_ROWS to {}",
                   Config::PRINT_GPU_TABLE_MAX_ROWS);
}

static void SetEnableFallbackCheck(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::ENABLE_FALLBACK_CHECK = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_FALLBACK_CHECK to {}", Config::ENABLE_FALLBACK_CHECK);
}

static void SetEnableDuckdbFallback(ClientContext& /*context*/,
                                    SetScope /*scope*/,
                                    Value& /*parameter*/)
{
  // No process-global mirror is kept.  DuckDB stores the value per-context and it
  // is read via duckdb_fallback_enabled() -> ClientContext::TryGetCurrentSetting,
  // so `SET enable_duckdb_fallback = ...` on one connection stays scoped to that
  // connection instead of leaking to every other connection (and, across the test
  // binary, to later test cases that create their own database).
}

static void SetEnableRegexJitImpl(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::ENABLE_REGEX_JIT_IMPL = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_REGEX_JIT_IMPL to {}", Config::ENABLE_REGEX_JIT_IMPL);
}

static void SetModifiedPipeline(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::MODIFIED_PIPELINE = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MODIFIED_PIPELINE to {}", Config::MODIFIED_PIPELINE);
}

static sirius::operator_params* get_operator_params(ClientContext& context)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    SIRIUS_LOG_DEBUG("SiriusContext not available; operator_params SET ignored");
    return nullptr;
  }
  return &sirius_ctx->get_config().get_operator_params();
}

static void SetDefaultScanTaskBatchSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->scan_task_batch_size = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config SCAN_TASK_BATCH_SIZE to {}", params->scan_task_batch_size);
}

static void SetDefaultScanTaskVarcharSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->default_scan_task_varchar_size = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config DEFAULT_SCAN_TASK_VARCHAR_SIZE to {}",
                   params->default_scan_task_varchar_size);
}

static void SetMaxSortPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->max_sort_partition_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_SORT_PARTITION_BYTES to {}",
                   params->max_sort_partition_bytes);
}

static void SetMaxSortPartitionMemoryFraction(ClientContext& context,
                                              SetScope scope,
                                              Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  const double fraction = parameter.GetValue<double>();
  if (fraction < 0.0 || fraction > 1.0) {
    throw InvalidInputException(
      "max_sort_partition_memory_fraction must be between 0.0 and 1.0, got {}", fraction);
  }
  params->max_sort_partition_memory_fraction = fraction;
  SIRIUS_LOG_DEBUG("Updated config MAX_SORT_PARTITION_MEMORY_FRACTION to {}",
                   params->max_sort_partition_memory_fraction);
}

static void SetHashPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->hash_partition_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config HASH_PARTITION_BYTES to {}", params->hash_partition_bytes);
}

static void SetConcatBatchBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->concat_batch_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config CONCAT_BATCH_BYTES to {}", params->concat_batch_bytes);
}

static void SetSortSampleBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->sort_sample_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config SORT_SAMPLE_BYTES to {}", params->sort_sample_bytes);
}

static void SetLogBackend(ClientContext& context, SetScope scope, Value& parameter)
{
  auto backend = StringValue::Get(parameter);
  if (backend != "duckdb" && backend != "spdlog" && backend != "noop") {
    throw InvalidInputException("Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)",
                                backend);
  }
  Config::LOG_BACKEND = std::move(backend);
  install_configured_log_sink(context.db.get());
  SIRIUS_LOG_DEBUG("Updated config LOG_BACKEND to {}", Config::LOG_BACKEND);
}

static void SetLogLevel(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_LEVEL = StringValue::Get(parameter);
  // Only re-targets the current sink; no rebuild (a no-op for the duckdb backend).
  auto parsed_level = sirius::log::string_to_enum(Config::LOG_LEVEL);
  sirius::log::get_sink()->set_level(parsed_level.value_or(sirius::log::level::info));
  if (!parsed_level) {
    SIRIUS_LOG_WARN("Unknown log level '{}', defaulting to info", Config::LOG_LEVEL);
  }
  SIRIUS_LOG_DEBUG("Updated config LOG_LEVEL to {}", Config::LOG_LEVEL);
}

static void SetLogDir(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_DIR = StringValue::Get(parameter);
  // log_dir only affects the spdlog backend; rebuild it when that one is active.
  if (Config::LOG_BACKEND == "spdlog") { install_configured_log_sink(context.db.get()); }
  SIRIUS_LOG_DEBUG("Updated config LOG_DIR to {}", Config::LOG_DIR);
}

static void SetLogFlushSeconds(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_FLUSH_SECONDS = IntegerValue::Get(parameter);
  // The flush interval is fixed at spdlog-sink construction, so rebuild it (only
  // the spdlog backend uses it).
  if (Config::LOG_BACKEND == "spdlog") { install_configured_log_sink(context.db.get()); }
  SIRIUS_LOG_DEBUG("Updated config LOG_FLUSH_SECONDS to {}", Config::LOG_FLUSH_SECONDS);
}

static void SetMaxBuildHashTableBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->max_build_hash_table_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_BUILD_HASH_TABLE_BYTES to {}",
                   params->max_build_hash_table_bytes);
}

static void SetMarkJoinBuildSwitchRatio(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  const double ratio = parameter.GetValue<double>();
  if (ratio < 0.0) {
    throw InvalidInputException("mark_join_build_switch_ratio must be >= 0.0, got {}", ratio);
  }
  params->mark_join_build_switch_ratio = ratio;
  SIRIUS_LOG_DEBUG("Updated config MARK_JOIN_BUILD_SWITCH_RATIO to {}",
                   params->mark_join_build_switch_ratio);
}

static void SetEnableGpuExecution(ClientContext& context, SetScope scope, Value& parameter)
{
  SIRIUS_LOG_DEBUG("Updated gpu_execution to {}", BooleanValue::Get(parameter));
}

static void SetEnableDynamicFilterPushdown(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->enable_dynamic_filter_pushdown = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_DYNAMIC_FILTER_PUSHDOWN to {}",
                   params->enable_dynamic_filter_pushdown);
}

static void SetEnableDynamicZoneMapFilter(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->enable_dynamic_zone_map_filter = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_DYNAMIC_ZONE_MAP_FILTER to {}",
                   params->enable_dynamic_zone_map_filter);
}

static void SetDynamicFilterDomainCoverageThreshold(ClientContext& context,
                                                    SetScope scope,
                                                    Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  const double threshold = parameter.GetValue<double>();
  if (threshold <= 0.0) {
    throw InvalidInputException("dynamic_filter_domain_coverage_threshold must be > 0.0, got %f",
                                threshold);
  }
  params->dynamic_filter_domain_coverage_threshold = threshold;
  SIRIUS_LOG_DEBUG("Updated config DYNAMIC_FILTER_DOMAIN_COVERAGE_THRESHOLD to {}",
                   params->dynamic_filter_domain_coverage_threshold);
}

static void SetDynamicFilterKeepThreshold(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  const double threshold = parameter.GetValue<double>();
  if (threshold < 0.0 || threshold > 1.0) {
    throw InvalidInputException("dynamic_filter_keep_threshold must be in [0.0, 1.0], got %f",
                                threshold);
  }
  params->dynamic_filter_keep_threshold = threshold;
  SIRIUS_LOG_DEBUG("Updated config DYNAMIC_FILTER_KEEP_THRESHOLD to {}",
                   params->dynamic_filter_keep_threshold);
}

static void SetEnablePinnedZoneMapPruning(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->enable_pinned_zone_map_pruning = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_PINNED_ZONE_MAP_PRUNING to {}",
                   params->enable_pinned_zone_map_pruning);
}

void SiriusExtension::InitialGPUConfigs(DBConfig& config)
{
  // Add in config option for gpu buffer manager
  config.AddExtensionOption("use_pin_memory",
                            "Whether or not the buffer manager is initialized with pinned memory",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CPU_PROCESSING),
                            SetUsePinMemory);

  config.AddExtensionOption(
    "use_pin_memory_for_caching",
    "Whether or not the cache buffer is allocated with pinned host memory instead of GPU memory",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CACHING),
    SetUsePinMemoryForCaching);

  // Add in config option for expression executor
  config.AddExtensionOption("use_cudf_expr",
                            "Whether or not cudf is used to evaluate expressions",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_CUDF_EXPR),
                            SetUseCudfExpr);

  config.AddExtensionOption(
    "expression_evaluator_strategy",
    "Strategy for the expression_evaluator: 'materialize', 'ast_interpret', or "
    "'ast_jit'",
    LogicalType::VARCHAR,
    Value(std::string(sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY))),
    SetExpressionEvaluatorStrategy);

  // Deprecated alias for `expression_evaluator_strategy`; remove in a future release.
  config.AddExtensionOption(
    "expression_executor_strategy",
    "[DEPRECATED - use expression_evaluator_strategy] Strategy for the expression_evaluator: "
    "'materialize', 'ast_interpret', or 'ast_jit'",
    LogicalType::VARCHAR,
    Value(std::string(sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY))),
    SetExpressionExecutorStrategyDeprecated);

  // Add in config option for top-N
  config.AddExtensionOption("use_custom_top_n",
                            "Whether or not custom kernel is used to evalaute top n",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_CUSTOM_TOP_N),
                            SetUseCustomTopN);

  // Add in config options for custom table scan
  config.AddExtensionOption("use_opt_table_scan",
                            "Whether or not the optional table scan is used",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_OPT_TABLE_SCAN),
                            SetUseOptTableScan);
  config.AddExtensionOption("opt_table_scan_num_streams",
                            "The number of cuda streams to use in the optional table scan",
                            LogicalType::INTEGER,
                            Value::INTEGER(Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS),
                            SetOptTableScanNumStreams);
  config.AddExtensionOption("opt_table_scan_memcpy_size",
                            "The memcpy size (in bytes) used by the optional table scan",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE),
                            SetOptTableScanMemcpySize);

  // Add in config options for printing gpu table
  config.AddExtensionOption("print_gpu_table_max_rows",
                            "Maximal amount of rows to render when printing gpu table",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(Config::PRINT_GPU_TABLE_MAX_ROWS),
                            SetPrintGPUTableMaxRows);

  // Add in config options for duckdb fallback checking
  config.AddExtensionOption("enable_fallback_check",
                            "Whether to enable fallback checking",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::ENABLE_FALLBACK_CHECK),
                            SetEnableFallbackCheck);

  config.AddExtensionOption(
    "enable_duckdb_fallback",
    "Whether to enable fallback to duckdb execution after an error is detected",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(true),  // literal default: never seed from a process-global that a
                           // prior connection's SET may have mutated (that leaked the
                           // fallback policy into every freshly-created database).
    SetEnableDuckdbFallback);

  // TEST ONLY: when non-empty, transparent GPU execution fails at runtime with that
  // message after plan generation succeeds, to exercise the CPU fallback path. No
  // setter — the value is read via TryGetCurrentSetting in PhysicalSiriusExecution.
  config.AddExtensionOption(
    "sirius_test_inject_transparent_gpu_error",
    "TEST ONLY: force transparent GPU execution to fail at runtime with this message",
    LogicalType::VARCHAR,
    Value(""));

  // Add in config options for special JIT implementation for regex
  config.AddExtensionOption(
    "enable_regex_jit_impl",
    "Whether to use special JIT implementation for particular regex evaluation",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::ENABLE_REGEX_JIT_IMPL),
    SetEnableRegexJitImpl);

  // Add in config options for modified pipeline
  config.AddExtensionOption("modified_pipeline",
                            "Whether to use modified pipeline for GPU execution",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::MODIFIED_PIPELINE),
                            SetModifiedPipeline);

  // Add in config options for duckdb scan task
  // Default batch size
  config.AddExtensionOption("scan_task_batch_size",
                            "The default batch size for a duckdb scan task",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.scan_task_batch_size),
                            SetDefaultScanTaskBatchSize);
  // Default varchar size for estimating rows per batch
  config.AddExtensionOption(
    "default_scan_task_varchar_size",
    "The default varchar size for estimating rows per batch in a duckdb scan task",
    LogicalType::UBIGINT,
    Value::UBIGINT(sirius::operator_params{}.default_scan_task_varchar_size),
    SetDefaultScanTaskVarcharSize);

  // Add in config option for sort partition size
  config.AddExtensionOption("max_sort_partition_bytes",
                            "Maximum bytes per sort partition (0 = auto based on "
                            "max_sort_partition_memory_fraction of GPU memory)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.max_sort_partition_bytes),
                            SetMaxSortPartitionBytes);
  config.AddExtensionOption(
    "max_sort_partition_memory_fraction",
    "Fraction of available GPU memory per sort partition when max_sort_partition_bytes is 0",
    LogicalType::DOUBLE,
    Value::DOUBLE(sirius::operator_params{}.max_sort_partition_memory_fraction),
    SetMaxSortPartitionMemoryFraction);

  // Logging configuration
  config.AddExtensionOption("sirius_log_backend",
                            "Logging backend for Sirius (duckdb, spdlog, noop)",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_BACKEND),
                            SetLogBackend);
  config.AddExtensionOption("sirius_log_level",
                            "Log level for Sirius (trace, debug, info, warn, error, critical, off)",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_LEVEL),
                            SetLogLevel);
  config.AddExtensionOption("sirius_log_dir",
                            "Directory for Sirius log files",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_DIR),
                            SetLogDir);
  config.AddExtensionOption("sirius_log_flush_seconds",
                            "Interval in seconds between automatic log flushes",
                            LogicalType::INTEGER,
                            Value::INTEGER(Config::LOG_FLUSH_SECONDS),
                            SetLogFlushSeconds);

  config.AddExtensionOption("hash_partition_bytes",
                            "Target size in bytes per hash partition",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.hash_partition_bytes),
                            SetHashPartitionBytes);

  config.AddExtensionOption("concat_batch_bytes",
                            "Target size for concat operator",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.concat_batch_bytes),
                            SetConcatBatchBytes);

  config.AddExtensionOption("sort_sample_bytes",
                            "Target bytes to sample before computing sort partition boundaries",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.sort_sample_bytes),
                            SetSortSampleBytes);

  config.AddExtensionOption("max_build_hash_table_bytes",
                            "Maximum size a build-side table can be where it will create a "
                            "reusable hash table for hash joins (i.e. BUILD_PROBE mode)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.max_build_hash_table_bytes),
                            SetMaxBuildHashTableBytes);

  config.AddExtensionOption(
    "mark_join_build_switch_ratio",
    "For STANDARD-mode MARK joins, build on the left/output side via cudf::mark_join when the "
    "right (probe) side has at least this many times more rows than the left side (0 disables). "
    "Hardware-dependent — recalibrate per GPU.",
    LogicalType::DOUBLE,
    Value::DOUBLE(sirius::operator_params{}.mark_join_build_switch_ratio),
    SetMarkJoinBuildSwitchRatio);

  config.AddExtensionOption(
    "gpu_execution",
    "Whether to transparently intercept SQL queries and execute them on GPU",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(true),
    SetEnableGpuExecution);

  config.AddExtensionOption(
    "enable_dynamic_filter_pushdown",
    "Wire dynamic table-filter pushdown: an eligible BUILD_PROBE hash-join build publishes a "
    "runtime membership filter (raw IN-list for 1-12 supported build rows; otherwise a hash "
    "IN-list if it fits the smallest probe-GPU L2, or a Bloom) into the probe-side scan to drop "
    "non-matching rows before the join (on by default)",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(sirius::operator_params{}.enable_dynamic_filter_pushdown),
    SetEnableDynamicFilterPushdown);

  config.AddExtensionOption(
    "enable_dynamic_zone_map_filter",
    "Additionally emit a runtime zone-map (build-key min/max) at the probe scan: parquet scans use "
    "it for read-time row-group pruning, while duckdb-native scans apply it row-wise post-decode; "
    "requires enable_dynamic_filter_pushdown (off by default)",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(sirius::operator_params{}.enable_dynamic_zone_map_filter),
    SetEnableDynamicZoneMapFilter);

  config.AddExtensionOption(
    "dynamic_filter_domain_coverage_threshold",
    "Skip publishing a key's dynamic filters when the hash-join build covers at least this "
    "fraction of the key's domain; >= 1.0 effectively disables the gate",
    LogicalType::DOUBLE,
    Value::DOUBLE(sirius::operator_params{}.dynamic_filter_domain_coverage_threshold),
    SetDynamicFilterDomainCoverageThreshold);

  config.AddExtensionOption(
    "dynamic_filter_keep_threshold",
    "Disable a probe scan's post-decode dynamic filtering once a measured split keeps more than "
    "this fraction of its rows (too unselective to repay the mask kernel); in [0.0, 1.0], 1.0 "
    "keeps filtering always on",
    LogicalType::DOUBLE,
    Value::DOUBLE(sirius::operator_params{}.dynamic_filter_keep_threshold),
    SetDynamicFilterKeepThreshold);

  config.AddExtensionOption(
    "enable_pinned_zone_map_pruning",
    "Skip pinned-table chunks whose pin-time min/max statistics prove the scan's pushed-down "
    "filter matches no rows; also gates the statistics capture during CALL pin_table, so a table "
    "pinned while off carries no zone maps until re-pinned with the flag on",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(sirius::operator_params{}.enable_pinned_zone_map_pruning),
    SetEnablePinnedZoneMapPruning);
}

static void LoadInternal(ExtensionLoader& loader)
{
  sirius::util::install_segfault_backtrace_handler();

  auto& db           = loader.GetDatabaseInstance();
  auto& config       = DBConfig::GetConfig(db);
  auto callback      = make_shared_ptr<duckdb::SiriusContextExtensionCallback>();
  auto* callback_ptr = callback.get();
  config.GetCallbackManager().Register(std::move(callback));

  duckdb::OptimizerExtension vector_join_cardinality;
  vector_join_cardinality.pre_optimize_function = duckdb::SiriusPreOptimizeVectorJoin;
  vector_join_cardinality.optimize_function     = duckdb::SiriusPostOptimizeVectorJoin;
  config.GetCallbackManager().Register(std::move(vector_join_cardinality));

  // The ctor already installed the db-independent backend; reinstall now that the
  // DatabaseInstance exists so the duckdb backend (which needs it) is built and an
  // unknown backend name is reported here rather than swallowed by the ctor.
  install_configured_log_sink(&db);

  sirius::converter_registry::initialize();
  SiriusExtension::InitialGPUConfigs(config);
  SiriusExtension::RegisterGPUFunctions(db);

  // Register the s3:// FileSystem so DuckDB's native read_parquet('s3://') binds
  // by reading the parquet footer through Sirius's routed REST ioctx. This makes
  // the transparent form work — SET gpu_execution=true; SELECT ... FROM
  // read_parquet('s3://...') — with the captured scan run on GPU. sirius_httpfs
  // is read-only and GPU-only: it serves the bind-time footer read, never a CPU
  // data path (a query that reads s3:// and fails on GPU still surfaces a clear
  // "S3 CPU fallback is not supported" error; local reads fall back to CPU).
  db.GetFileSystem().RegisterSubSystem(make_uniq<sirius::io::s3::sirius_httpfs>());

  // Register optimizer extension for transparent GPU execution.
  // Pre-hook disables incompatible optimizers; post-hook captures the plan.
  OptimizerExtension opt_ext;
  opt_ext.pre_optimize_function = sirius::transparent::sirius_pre_optimizer_hook;
  opt_ext.optimize_function     = sirius::transparent::sirius_optimizer_hook;
  OptimizerExtension::Register(config, std::move(opt_ext));

  // Register SiriusContext on connections that were opened before the extension
  // was loaded (e.g. when loaded via LOAD in Python or the CLI).
  for (auto& ctx : ConnectionManager::Get(db).GetConnectionList()) {
    callback_ptr->OnConnectionOpened(*ctx);
  }
}

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }

std::string SiriusExtension::Name() { return "Sirius	Extension"; }

std::string SiriusExtension::Version() const
{
#ifdef EXT_VERSION_SIRIUS
  return EXT_VERSION_SIRIUS;
#else
  return "";
#endif
}

}  // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius, loader) { duckdb::LoadInternal(loader); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
