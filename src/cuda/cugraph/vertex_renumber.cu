#include "cugraph/vertex_renumber.cuh"

#include "duckdb.hpp"

namespace duckdb {
namespace sirius {

RenumberedGraph renumber_from_table(ClientContext&     context,
                                    const std::string& table_name,
                                    const std::string& src_col_name,
                                    const std::string& dst_col_name)
{
  RenumberedGraph result;

  // Open a connection on the same database
  Connection conn(*context.db);

  // Fetch all edges; quote identifiers to handle reserved words / uppercase
  const std::string query = "SELECT \"" + src_col_name + "\", \"" + dst_col_name +
                            "\" FROM \"" + table_name + "\"";

  auto query_result = conn.Query(query);
  if (query_result->HasError()) {
    throw InvalidInputException("gpu_graph: failed to read table '%s': %s",
                                table_name,
                                query_result->GetError());
  }

  // Helper: assign an integer ID to a vertex, or return existing ID
  auto intern = [&](const std::string& name) -> int32_t {
    auto [it, inserted] = result.vertex_to_id.emplace(name, (int32_t)result.id_to_vertex.size());
    if (inserted) { result.id_to_vertex.push_back(name); }
    return it->second;
  };

  // Iterate result chunks
  while (true) {
    auto chunk = query_result->Fetch();
    if (!chunk || chunk->size() == 0) { break; }

    for (idx_t row = 0; row < chunk->size(); row++) {
      const std::string src = chunk->data[0].GetValue(row).ToString();
      const std::string dst = chunk->data[1].GetValue(row).ToString();
      result.src_ids.push_back(intern(src));
      result.dst_ids.push_back(intern(dst));
    }
  }

  result.num_vertices = static_cast<int32_t>(result.id_to_vertex.size());
  result.num_edges    = static_cast<int32_t>(result.src_ids.size());
  return result;
}

}  // namespace sirius
}  // namespace duckdb
