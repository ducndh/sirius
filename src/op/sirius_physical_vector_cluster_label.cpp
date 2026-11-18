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

#include "op/sirius_physical_vector_cluster_label.hpp"

#include "data/data_batch_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "telemetry/nvtx.hpp"
#include "vss/cluster_lists.hpp"
#include "vss/vector_clustering.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/device_resources.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

sirius_physical_vector_cluster_label::sirius_physical_vector_cluster_label(
  duckdb::vector<sirius::logical_type> types,
  std::size_t vector_column,
  std::size_t label_index,
  const cudf::column* centroids,
  std::int64_t dim,
  cuvs::distance::DistanceType metric,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::VECTOR_CLUSTER_LABEL, std::move(types), estimated_cardinality),
    _vector_column(vector_column),
    _label_index(label_index),
    _centroids(centroids),
    _dim(dim),
    _metric(metric)
{
  if (_centroids == nullptr) {
    throw invalid_input_exception(
      "[sirius_physical_vector_cluster_label] a clustering's centroids are required.");
  }
}

std::unique_ptr<operator_data> sirius_physical_vector_cluster_label::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_vector_cluster_label::execute"};
  auto& input        = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto input_batches = input.get_read_only_batches();
  rmm::cuda_stream_view const task_stream{stream.get()};

  // Built here, on the task's thread and stream: device memory is accounted to the thread that
  // allocates it, and the handle's workspace is tied to the stream.
  raft::device_resources res{task_stream};
  std::optional<vss::centroid_labeler> labeler;

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());
  for (auto& input_ro : input_batches) {
    auto const input_view = sirius::get_cudf_table_view(input_ro);
    auto& mem             = *input_ro.get_memory_space();
    auto const mr         = mem.get_default_allocator();
    if (_vector_column >= static_cast<std::size_t>(input_view.num_columns()) ||
        _label_index > static_cast<std::size_t>(input_view.num_columns())) {
      throw invalid_input_exception(
        "[sirius_physical_vector_cluster_label] the batch does not have the planned columns.");
    }
    auto const rows = static_cast<std::size_t>(input_view.num_rows());
    rmm::device_uvector<std::int32_t> ids(rows, task_stream, mr);
    if (rows > 0) {
      if (!labeler) { labeler.emplace(res, _centroids->view(), _dim, _metric, mr); }
      labeler->label(input_view.column(static_cast<cudf::size_type>(_vector_column)), ids.data());
    }
    auto const label_bytes = rows * sizeof(std::int32_t);
    std::shared_ptr<cudf::column> labels =
      std::make_shared<cudf::column>(std::move(ids), rmm::device_buffer{0, task_stream, mr}, 0);

    // The vectors in half precision, when asked for: the same list offsets over an INT16 child of
    // half bits.
    std::shared_ptr<cudf::column> half_values;
    std::shared_ptr<cudf::column> half_offsets;
    std::shared_ptr<cudf::column> half_errors;
    std::optional<cudf::column_view> half_vectors;
    if (_store_half && rows > 0) {
      auto const vectors = input_view.column(static_cast<cudf::size_type>(_vector_column));
      cudf::lists_column_view const lists{vectors};
      auto const values = lists.child();
      auto const n      = static_cast<std::size_t>(rows) * static_cast<std::size_t>(_dim);
      if (vectors.offset() != 0 || values.type().id() != cudf::type_id::FLOAT32 ||
          static_cast<std::size_t>(values.size()) != n || vectors.has_nulls()) {
        throw invalid_input_exception(
          "[sirius_physical_vector_cluster_label] half-precision storage needs a whole batch of "
          "FLOAT arrays of the clustering's width, without nulls.");
      }
      rmm::device_uvector<std::int16_t> half(n, task_stream, mr);
      rmm::device_uvector<float> norms(rows, task_stream, mr);
      rmm::device_uvector<unsigned int> worst(1, task_stream, mr);
      CUDF_CUDA_TRY(cudaMemsetAsync(worst.data(), 0, sizeof(unsigned int), task_stream.get()));
      auto* const bits = reinterpret_cast<std::uint16_t*>(half.data());
      vss::narrow_to_float16(
        values.data<float>(), static_cast<std::int64_t>(n), bits, task_stream);
      rmm::device_uvector<float> errors(_keep_exact ? rows : 0, task_stream, mr);
      vss::half_rows_norms(values.data<float>(),
                           bits,
                           static_cast<std::int64_t>(rows),
                           _dim,
                           norms.data(),
                           _keep_exact ? errors.data() : nullptr,
                           worst.data(),
                           nullptr,
                           task_stream);
      if (_keep_exact) {
        half_errors = std::make_shared<cudf::column>(
          std::move(errors), rmm::device_buffer{0, task_stream, mr}, 0);
      } else {
        unsigned int worst_bits = 0;
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          &worst_bits, worst.data(), sizeof(worst_bits), cudaMemcpyDeviceToHost, task_stream.get()));
        task_stream.sync();
        float worst_error = 0.f;
        std::memcpy(&worst_error, &worst_bits, sizeof(float));
        if (!(worst_error == 0.f)) {
          throw invalid_input_exception(
            "[sirius_physical_vector_cluster_label] a row is " + std::to_string(worst_error) +
            " from its half-precision rounding; half-precision storage is only for values that "
            "survive it.");
        }
      }
      half_values = std::make_shared<cudf::column>(
        std::move(half), rmm::device_buffer{0, task_stream, mr}, 0);
      half_vectors.emplace(cudf::data_type{cudf::type_id::LIST},
                           vectors.size(),
                           nullptr,
                           nullptr,
                           0,
                           0,
                           std::vector<cudf::column_view>{lists.offsets(), half_values->view()});
    } else if (_store_half) {
      // No rows, and still the type every other batch of the corpus has: one offset, as a list
      // column's rows + 1, which is what the join's readers size by.
      rmm::device_uvector<std::int32_t> first_offset(1, task_stream, mr);
      CUDF_CUDA_TRY(
        cudaMemsetAsync(first_offset.data(), 0, sizeof(std::int32_t), task_stream.get()));
      half_offsets = std::make_shared<cudf::column>(
        std::move(first_offset), rmm::device_buffer{0, task_stream, mr}, 0);
      half_values = std::make_shared<cudf::column>(
        rmm::device_uvector<std::int16_t>(0, task_stream, mr), rmm::device_buffer{0, task_stream, mr}, 0);
      half_vectors.emplace(cudf::data_type{cudf::type_id::LIST},
                           0,
                           nullptr,
                           nullptr,
                           0,
                           0,
                           std::vector<cudf::column_view>{half_offsets->view(), half_values->view()});
    }

    std::optional<std::uint32_t> side_number;
    std::unique_ptr<cudf::column> positions;
    std::vector<cudf::column_view> cols;
    cols.reserve(static_cast<std::size_t>(input_view.num_columns()) + 3);
    for (std::size_t i = 0; i < static_cast<std::size_t>(input_view.num_columns()); ++i) {
      if (i == _label_index) { cols.push_back(labels->view()); }
      cols.push_back(half_vectors && i == _vector_column
                       ? *half_vectors
                       : input_view.column(static_cast<cudf::size_type>(i)));
    }
    if (_label_index == static_cast<std::size_t>(input_view.num_columns())) {
      cols.push_back(labels->view());
    }
    // Last, so no other column moves: the rows as they came, and their rounding errors.
    if (_keep_exact) {
      if (!half_errors) {
        half_errors = std::make_shared<cudf::column>(
          rmm::device_uvector<float>(0, task_stream, mr), rmm::device_buffer{0, task_stream, mr}, 0);
      }
      if (_exact_side) {
        if (!_sort_by_label) {
          throw invalid_input_exception(
            "[sirius_physical_vector_cluster_label] FP32 rows kept beside the batch need the batch "
            "to leave sorted by label.");
        }
        // The rows go to the side as this input batch; what stays is where each one is there.
        if (rows > 0) { side_number = _exact_side->next_batch(); }
        cudf::numeric_scalar<std::int64_t> const first{
          vss::exact_rows_side::position(side_number.value_or(0), 0), true, task_stream, mr};
        positions = cudf::sequence(static_cast<cudf::size_type>(rows), first, task_stream, mr);
        cols.push_back(half_errors->view());
        cols.push_back(positions->view());
      } else {
        cols.push_back(input_view.column(static_cast<cudf::size_type>(_vector_column)));
        cols.push_back(half_errors->view());
      }
    }
    cudf::table_view out_view(cols);

    if (_sort_by_label) {
      auto const order =
        cudf::sorted_order(cudf::table_view({labels->view()}), {}, {}, task_stream, mr);
      auto sorted = cudf::gather(
        out_view, order->view(), cudf::out_of_bounds_policy::DONT_CHECK, task_stream, mr);
      output_batches.push_back(
        sirius::make_data_batch(std::move(sorted), mem, stream, batch_telemetry()));
      if (side_number) {
        // The input batch itself, seen as its vector column: no copy, and it goes to the tiers
        // in the order it came. Last, so a task that fails before this leaves no batch behind.
        struct side_owner {
          cucascade::read_only_data_batch input_lock;
        };
        auto const vectors    = input_view.column(static_cast<cudf::size_type>(_vector_column));
        auto const side_bytes = rows * static_cast<std::size_t>(_dim) * sizeof(float) +
                                (rows + 1) * sizeof(std::int32_t);
        auto side = sirius::make_data_batch_from_view(cudf::table_view({vectors}),
                                                      side_owner{std::move(input_ro)},
                                                      side_bytes,
                                                      mem,
                                                      stream,
                                                      batch_telemetry());
        _exact_side->put(*side_number, side, [&] {
          auto const pipeline = get_pipeline();
          auto manager        = _exact_side_context != nullptr && pipeline != nullptr
                                  ? _exact_side_context->get_data_repository_manager(
                                      pipeline->get_query_id())
                                  : nullptr;
          if (!manager || _exact_side_reader == nullptr) {
            throw invalid_input_exception(
              "[sirius_physical_vector_cluster_label] no repository manager to hold the FP32 "
              "rows kept beside a half-precision corpus.");
          }
          auto repo  = std::make_unique<cucascade::shared_data_repository>();
          auto* held = repo.get();
          manager->add_new_repository(
            _exact_side_reader->get_operator_id(), "exact_rows", std::move(repo));
          return held;
        });
      }
      continue;
    }

    // The labels are this operator's; the other columns stay the input batch's, held read-only
    // for as long as the output batch lives.
    struct label_owner {
      std::shared_ptr<cudf::column> labels;
      std::shared_ptr<cudf::column> half_values;
      std::shared_ptr<cudf::column> half_offsets;
      std::shared_ptr<cudf::column> half_errors;
      cucascade::read_only_data_batch input_lock;
    };
    auto const bytes = input_ro.get_data()->get_size_in_bytes() + label_bytes +
                       (half_values ? half_values->alloc_size() : 0) +
                       (half_errors ? half_errors->alloc_size() : 0);
    label_owner owner{std::move(labels),
                      std::move(half_values),
                      std::move(half_offsets),
                      std::move(half_errors),
                      std::move(input_ro)};
    output_batches.push_back(sirius::make_data_batch_from_view(
      out_view, std::move(owner), bytes, mem, stream, batch_telemetry()));
  }
  return std::make_unique<pipelineable_operator_data>(output_batches, input.get_position());
}

}  // namespace op
}  // namespace sirius
