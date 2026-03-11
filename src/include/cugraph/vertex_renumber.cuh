#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare DuckDB types to avoid pulling in all of duckdb.hpp here
namespace duckdb {
class ClientContext;
}

namespace duckdb {
namespace sirius {

// Result of renumbering a VARCHAR edge list into contiguous int32 vertex IDs.
// The mapping is bidirectional so results can be mapped back to original names.
struct RenumberedGraph {
  std::vector<int32_t> src_ids;   // Edge sources (renumbered)
  std::vector<int32_t> dst_ids;   // Edge destinations (renumbered)
  int32_t              num_vertices = 0;
  int32_t              num_edges    = 0;

  std::unordered_map<std::string, int32_t> vertex_to_id;
  std::vector<std::string>                 id_to_vertex;  // index = id

  int32_t get_vertex_id(const std::string& name) const
  {
    auto it = vertex_to_id.find(name);
    return it != vertex_to_id.end() ? it->second : -1;
  }

  const std::string& get_vertex_name(int32_t id) const { return id_to_vertex[id]; }
};

// Read src_col and dst_col from a DuckDB table and build a RenumberedGraph.
// Runs on CPU via a DuckDB Connection; suitable for demo-scale graphs.
RenumberedGraph renumber_from_table(duckdb::ClientContext& context,
                                    const std::string&     table_name,
                                    const std::string&     src_col_name,
                                    const std::string&     dst_col_name);

}  // namespace sirius
}  // namespace duckdb
