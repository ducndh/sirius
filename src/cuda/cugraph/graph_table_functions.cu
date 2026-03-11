/*
 * Graph analytics table function implementations (BFS, PageRank, WCC).
 * Spawns an out-of-process Python cuGraph worker and communicates via
 * binary protocol + CUDA IPC for zero-copy GPU result transfer.
 */

#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"
#include "cugraph/vertex_renumber.cuh"

#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace duckdb {

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr int32_t ALGO_BFS      = 0;
static constexpr int32_t ALGO_PAGERANK = 1;
static constexpr int32_t ALGO_WCC      = 2;

// ── Helper: find the graph_worker.py script ────────────────────────────────

static std::string FindWorkerScript() {
  // Look relative to the extension binary: ../../../scripts/graph_worker.py
  // Also try common locations
  const char* candidates[] = {
    "scripts/graph_worker.py",
    "../scripts/graph_worker.py",
    "/home/cc/sirius-dev/scripts/graph_worker.py",
  };
  for (auto* path : candidates) {
    if (access(path, R_OK) == 0) { return path; }
  }
  throw InvalidInputException("gpu_graph: cannot find scripts/graph_worker.py");
}

// ── Helper: write all bytes to fd ──────────────────────────────────────────

static void write_all(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len > 0) {
    ssize_t n = ::write(fd, p, len);
    if (n <= 0) { throw IOException("gpu_graph: write to worker failed"); }
    p += n; len -= n;
  }
}

// ── Helper: read all bytes from fd ─────────────────────────────────────────

static void read_all(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  while (len > 0) {
    ssize_t n = ::read(fd, p, len);
    if (n <= 0) { throw IOException("gpu_graph: read from worker failed"); }
    p += n; len -= n;
  }
}

// ── Worker process management ──────────────────────────────────────────────

struct WorkerProcess {
  pid_t pid        = -1;
  int   to_child   = -1;   // write end → child stdin
  int   from_child = -1;   // read end  ← child stdout

  ~WorkerProcess() { cleanup(); }

  void spawn(const std::string& script_path) {
    int pipe_in[2], pipe_out[2];
    if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
      throw IOException("gpu_graph: pipe() failed");
    }

    pid = fork();
    if (pid < 0) { throw IOException("gpu_graph: fork() failed"); }

    if (pid == 0) {
      // Child process
      close(pipe_in[1]);
      close(pipe_out[0]);
      dup2(pipe_in[0], STDIN_FILENO);
      dup2(pipe_out[1], STDOUT_FILENO);
      close(pipe_in[0]);
      close(pipe_out[1]);

      // Set up cuGraph Python environment.
      // Look for sirius-graph pixi env, then sirius-dev pixi env.
      const char* env_candidates[] = {
        "/home/cc/sirius-graph/.pixi/envs/cuda12",
        "/home/cc/sirius-dev/.pixi/envs/cuda12",
      };
      const char* env_path = nullptr;
      for (auto* candidate : env_candidates) {
        std::string python = std::string(candidate) + "/bin/python3";
        if (access(python.c_str(), X_OK) == 0) { env_path = candidate; break; }
      }
      if (!env_path) { _exit(127); }

      std::string python_bin = std::string(env_path) + "/bin/python3";
      std::string ld_path = std::string(env_path) + "/lib";
      // Prepend to existing LD_LIBRARY_PATH
      const char* old_ld = getenv("LD_LIBRARY_PATH");
      if (old_ld) { ld_path += std::string(":") + old_ld; }

      setenv("CONDA_PREFIX", env_path, 1);
      setenv("PYTHONHOME", env_path, 1);
      unsetenv("PYTHONPATH");  // avoid system Python contamination
      setenv("LD_LIBRARY_PATH", ld_path.c_str(), 1);

      execl(python_bin.c_str(), "python3", script_path.c_str(), nullptr);
      _exit(127);  // exec failed
    }

    // Parent
    close(pipe_in[0]);
    close(pipe_out[1]);
    to_child   = pipe_in[1];
    from_child = pipe_out[0];
  }

  void send_ack() {
    uint32_t ack = 1;
    write_all(to_child, &ack, 4);
  }

  void cleanup() {
    if (to_child >= 0) { close(to_child); to_child = -1; }
    if (from_child >= 0) { close(from_child); from_child = -1; }
    if (pid > 0) {
      int status;
      waitpid(pid, &status, 0);
      pid = -1;
    }
  }
};

// ── IPC result array ───────────────────────────────────────────────────────

struct IPCArray {
  cudaIpcMemHandle_t handle;
  uint32_t           elem_size;
  uint32_t           num_elements;
  void*              mapped_ptr = nullptr;

  void open() {
    cudaError_t err = cudaIpcOpenMemHandle(&mapped_ptr, handle,
                                           cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      throw IOException("gpu_graph: cudaIpcOpenMemHandle failed: %s",
                        cudaGetErrorString(err));
    }
  }

  // Download to host
  std::vector<char> download() {
    size_t nbytes = static_cast<size_t>(elem_size) * num_elements;
    std::vector<char> host(nbytes);
    cudaError_t err = cudaMemcpy(host.data(), mapped_ptr, nbytes,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw IOException("gpu_graph: cudaMemcpy D2H failed: %s",
                        cudaGetErrorString(err));
    }
    return host;
  }

  void close_handle() {
    if (mapped_ptr) {
      cudaIpcCloseMemHandle(mapped_ptr);
      mapped_ptr = nullptr;
    }
  }
};

// ── Core: run algorithm via worker process ─────────────────────────────────

struct GraphResult {
  uint32_t                  num_rows;
  std::vector<IPCArray>     arrays;
  std::vector<std::vector<char>> host_data;  // downloaded arrays
};

static GraphResult RunGraphWorker(int32_t                     algo,
                                  const sirius::RenumberedGraph& graph,
                                  const char*                 params_16bytes)
{
  auto script = FindWorkerScript();
  WorkerProcess worker;
  worker.spawn(script);

  // Send header: algo(4) + num_edges(4) + num_vertices(4) + params(16) = 28 bytes
  uint32_t header[3] = {
    static_cast<uint32_t>(algo),
    static_cast<uint32_t>(graph.num_edges),
    static_cast<uint32_t>(graph.num_vertices)
  };
  write_all(worker.to_child, header, 12);
  write_all(worker.to_child, params_16bytes, 16);

  // Send edge arrays (int32)
  write_all(worker.to_child, graph.src_ids.data(), graph.num_edges * sizeof(int32_t));
  write_all(worker.to_child, graph.dst_ids.data(), graph.num_edges * sizeof(int32_t));

  // Close write end so worker sees EOF after reading edges
  // (worker reads fixed number of bytes, but closing is good practice)
  // Don't close yet — we need to send ACK later.

  // Read response header: status(4) + num_rows(4) + num_arrays(4)
  uint32_t resp[3];
  read_all(worker.from_child, resp, 12);

  if (resp[0] != 0) {
    // Error: read message
    char errbuf[4096] = {};
    ssize_t n = ::read(worker.from_child, errbuf, sizeof(errbuf) - 1);
    worker.cleanup();
    throw InvalidInputException("gpu_graph worker error: %s",
                                n > 0 ? errbuf : "unknown");
  }

  GraphResult result;
  result.num_rows = resp[1];
  uint32_t num_arrays = resp[2];
  result.arrays.resize(num_arrays);

  // Read IPC handles
  for (uint32_t i = 0; i < num_arrays; i++) {
    read_all(worker.from_child, &result.arrays[i].elem_size, 4);
    read_all(worker.from_child, &result.arrays[i].handle, 64);
    result.arrays[i].num_elements = result.num_rows;
  }

  // Open IPC handles and download data
  result.host_data.resize(num_arrays);
  for (uint32_t i = 0; i < num_arrays; i++) {
    result.arrays[i].open();
    result.host_data[i] = result.arrays[i].download();
    result.arrays[i].close_handle();
  }

  // Send ACK so worker can free GPU memory
  worker.send_ack();
  worker.cleanup();

  return result;
}

// =============================================================================
// gpu_graph_bfs(table, src_col, dst_col, source [, depth_limit])
// =============================================================================

struct GPUGraphBFSFunctionData : public TableFunctionData {
  std::string table_name, src_col, dst_col, source_vertex_name;
  int32_t     depth_limit   = -1;

  sirius::RenumberedGraph graph;
  GraphResult             result;
  idx_t                   output_offset = 0;
  bool                    finished      = false;
};

unique_ptr<FunctionData> SiriusExtension::GPUGraphBFSBind(ClientContext&          context,
                                                          TableFunctionBindInput& input,
                                                          vector<LogicalType>&    return_types,
                                                          vector<string>&         names)
{
  auto data                = make_uniq<GPUGraphBFSFunctionData>();
  data->table_name         = input.inputs[0].ToString();
  data->src_col            = input.inputs[1].ToString();
  data->dst_col            = input.inputs[2].ToString();
  data->source_vertex_name = input.inputs[3].ToString();
  if (input.inputs.size() >= 5) {
    data->depth_limit = input.inputs[4].GetValue<int32_t>();
  }

  return_types.emplace_back(LogicalType::VARCHAR); names.emplace_back("vertex");
  return_types.emplace_back(LogicalType::INTEGER); names.emplace_back("distance");
  return_types.emplace_back(LogicalType::VARCHAR); names.emplace_back("predecessor");

  return std::move(data);
}

void SiriusExtension::GPUGraphBFSFunction(ClientContext&      context,
                                          TableFunctionInput& data_p,
                                          DataChunk&          output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUGraphBFSFunctionData>();
  if (data.finished) { return; }

  // First call: renumber graph and run BFS via worker
  if (data.output_offset == 0) {
    data.graph = sirius::renumber_from_table(context, data.table_name,
                                             data.src_col, data.dst_col);
    int32_t source_id = data.graph.get_vertex_id(data.source_vertex_name);
    if (source_id == -1) {
      throw InvalidInputException("gpu_graph_bfs: source vertex '%s' not found",
                                  data.source_vertex_name.c_str());
    }

    // Pack BFS params: source(4) + depth_limit(4) + padding(8) = 16 bytes
    char params[16] = {};
    memcpy(params, &source_id, 4);
    memcpy(params + 4, &data.depth_limit, 4);

    data.result = RunGraphWorker(ALGO_BFS, data.graph, params);
  }

  const idx_t total = static_cast<idx_t>(data.result.num_rows);
  if (data.output_offset >= total) { data.finished = true; return; }

  const idx_t batch = std::min(static_cast<idx_t>(STANDARD_VECTOR_SIZE),
                               total - data.output_offset);

  // Arrays: [0]=vertex_ids, [1]=distances, [2]=predecessors (all int32)
  const int32_t* vertex_ids   = reinterpret_cast<const int32_t*>(data.result.host_data[0].data());
  const int32_t* distances    = reinterpret_cast<const int32_t*>(data.result.host_data[1].data());
  const int32_t* predecessors = reinterpret_cast<const int32_t*>(data.result.host_data[2].data());

  for (idx_t i = 0; i < batch; i++) {
    const idx_t idx = data.output_offset + i;
    const int32_t vid  = vertex_ids[idx];
    const int32_t pred = predecessors[idx];

    output.data[0].SetValue(i, Value(data.graph.id_to_vertex[vid]));
    output.data[1].SetValue(i, Value::INTEGER(distances[idx]));
    if (pred < 0) {
      output.data[2].SetValue(i, Value(LogicalType::VARCHAR));  // NULL
    } else {
      output.data[2].SetValue(i, Value(data.graph.id_to_vertex[pred]));
    }
  }

  output.SetCardinality(batch);
  data.output_offset += batch;
  if (data.output_offset >= total) { data.finished = true; }
}

// =============================================================================
// gpu_graph_pagerank(table, src_col, dst_col [, damping, tolerance, max_iterations])
// =============================================================================

struct GPUGraphPageRankFunctionData : public TableFunctionData {
  std::string table_name, src_col, dst_col;
  float       damping_factor = 0.85f;
  float       tolerance      = 1e-6f;
  int32_t     max_iterations = 500;

  sirius::RenumberedGraph graph;
  GraphResult             result;
  idx_t                   output_offset = 0;
  bool                    finished      = false;
};

unique_ptr<FunctionData> SiriusExtension::GPUGraphPageRankBind(ClientContext&          context,
                                                               TableFunctionBindInput& input,
                                                               vector<LogicalType>&    return_types,
                                                               vector<string>&         names)
{
  auto data        = make_uniq<GPUGraphPageRankFunctionData>();
  data->table_name = input.inputs[0].ToString();
  data->src_col    = input.inputs[1].ToString();
  data->dst_col    = input.inputs[2].ToString();
  if (input.named_parameters.count("damping")) {
    data->damping_factor = input.named_parameters.at("damping").GetValue<float>();
  }
  if (input.named_parameters.count("tolerance")) {
    data->tolerance = input.named_parameters.at("tolerance").GetValue<float>();
  }
  if (input.named_parameters.count("max_iterations")) {
    data->max_iterations = input.named_parameters.at("max_iterations").GetValue<int32_t>();
  }

  return_types.emplace_back(LogicalType::VARCHAR); names.emplace_back("vertex");
  return_types.emplace_back(LogicalType::FLOAT);   names.emplace_back("pagerank");

  return std::move(data);
}

void SiriusExtension::GPUGraphPageRankFunction(ClientContext&      context,
                                               TableFunctionInput& data_p,
                                               DataChunk&          output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUGraphPageRankFunctionData>();
  if (data.finished) { return; }

  if (data.output_offset == 0) {
    data.graph = sirius::renumber_from_table(context, data.table_name,
                                             data.src_col, data.dst_col);
    // Pack params: damping(4f) + tolerance(4f) + max_iter(4i) + padding(4) = 16 bytes
    char params[16] = {};
    memcpy(params,     &data.damping_factor, 4);
    memcpy(params + 4, &data.tolerance, 4);
    memcpy(params + 8, &data.max_iterations, 4);

    data.result = RunGraphWorker(ALGO_PAGERANK, data.graph, params);
  }

  const idx_t total = static_cast<idx_t>(data.result.num_rows);
  if (data.output_offset >= total) { data.finished = true; return; }

  const idx_t batch = std::min(static_cast<idx_t>(STANDARD_VECTOR_SIZE),
                               total - data.output_offset);

  const int32_t* vertex_ids = reinterpret_cast<const int32_t*>(data.result.host_data[0].data());
  const float*   pageranks  = reinterpret_cast<const float*>(data.result.host_data[1].data());

  for (idx_t i = 0; i < batch; i++) {
    const idx_t idx = data.output_offset + i;
    output.data[0].SetValue(i, Value(data.graph.id_to_vertex[vertex_ids[idx]]));
    output.data[1].SetValue(i, Value::FLOAT(pageranks[idx]));
  }

  output.SetCardinality(batch);
  data.output_offset += batch;
  if (data.output_offset >= total) { data.finished = true; }
}

// =============================================================================
// gpu_graph_wcc(table, src_col, dst_col)
// =============================================================================

struct GPUGraphWCCFunctionData : public TableFunctionData {
  std::string table_name, src_col, dst_col;

  sirius::RenumberedGraph graph;
  GraphResult             result;
  idx_t                   output_offset = 0;
  bool                    finished      = false;
};

unique_ptr<FunctionData> SiriusExtension::GPUGraphWCCBind(ClientContext&          context,
                                                          TableFunctionBindInput& input,
                                                          vector<LogicalType>&    return_types,
                                                          vector<string>&         names)
{
  auto data        = make_uniq<GPUGraphWCCFunctionData>();
  data->table_name = input.inputs[0].ToString();
  data->src_col    = input.inputs[1].ToString();
  data->dst_col    = input.inputs[2].ToString();

  return_types.emplace_back(LogicalType::VARCHAR); names.emplace_back("vertex");
  return_types.emplace_back(LogicalType::VARCHAR); names.emplace_back("component");

  return std::move(data);
}

void SiriusExtension::GPUGraphWCCFunction(ClientContext&      context,
                                          TableFunctionInput& data_p,
                                          DataChunk&          output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUGraphWCCFunctionData>();
  if (data.finished) { return; }

  if (data.output_offset == 0) {
    data.graph = sirius::renumber_from_table(context, data.table_name,
                                             data.src_col, data.dst_col);
    char params[16] = {};
    data.result = RunGraphWorker(ALGO_WCC, data.graph, params);
  }

  const idx_t total = static_cast<idx_t>(data.result.num_rows);
  if (data.output_offset >= total) { data.finished = true; return; }

  const idx_t batch = std::min(static_cast<idx_t>(STANDARD_VECTOR_SIZE),
                               total - data.output_offset);

  const int32_t* vertex_ids = reinterpret_cast<const int32_t*>(data.result.host_data[0].data());
  const int32_t* labels     = reinterpret_cast<const int32_t*>(data.result.host_data[1].data());

  for (idx_t i = 0; i < batch; i++) {
    const idx_t idx = data.output_offset + i;
    const int32_t vid  = vertex_ids[idx];
    const int32_t comp = labels[idx];
    output.data[0].SetValue(i, Value(data.graph.id_to_vertex[vid]));
    // Map component label (vertex id) back to name
    if (comp >= 0 && comp < data.graph.num_vertices) {
      output.data[1].SetValue(i, Value(data.graph.id_to_vertex[comp]));
    } else {
      output.data[1].SetValue(i, Value(std::to_string(comp)));
    }
  }

  output.SetCardinality(batch);
  data.output_offset += batch;
  if (data.output_offset >= total) { data.finished = true; }
}

}  // namespace duckdb
