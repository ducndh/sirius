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

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace sirius::vss {

/**
 * @brief One join side, materialized as an ordered list of batches that every stage agrees on.
 *
 * A neighbour id is a position in the corpus row order, and two operators in two different
 * pipelines resolve it: the fold writes ids, materialize turns them back into rows. Against a
 * pinned table both read the same immutable @c pinned_entry, so they cannot disagree about
 * what position 4,700,013 means. A build phase has no such pre-existing list — each stage
 * would otherwise ask the repository separately, and any difference in the two answers is a
 * silently wrong join rather than a failure.
 *
 * So the plan generator mints one of these per join and hands the same handle to every stage.
 * The first caller to need it takes the snapshot; the order recorded there *is* the corpus row
 * order for the rest of the query. This is the same move the plan generator already makes for
 * dynamic filters, where one channel object is handed to the producing join and the consuming
 * scan rather than letting each rediscover it.
 */
class materialized_side_buffer {
 public:
  /// Take the snapshot if it has not been taken, and return it. The first caller wins; later
  /// callers get that same list even if the repository has changed underneath.
  const std::vector<std::uint64_t>& ensure_snapshot(::cucascade::shared_data_repository& repo)
  {
    std::lock_guard<std::mutex> lg(_mutex);
    if (!_taken) {
      _repo      = &repo;
      _batch_ids = repo.get_batch_ids(/*partition_idx=*/0);
      // Any order is a correct corpus order -- what matters is that every stage uses the same
      // one -- so reversing it must not change a single output row. That makes this switch a
      // deterministic test for the property: whether the batches arrive in the table's order
      // is a race, so a test that waits for them to disagree passes for the wrong reason on
      // most runs. Reversing forces the disagreement that the race only sometimes produces.
      const char* reverse = std::getenv("SIRIUS_VECTOR_JOIN_REVERSE_BUILD_ORDER");
      if (reverse != nullptr && std::string_view{reverse} == "1") {
        std::reverse(_batch_ids.begin(), _batch_ids.end());
      }
      _taken = true;
    }
    return _batch_ids;
  }

  [[nodiscard]] bool has_snapshot() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _taken;
  }

  /// The snapshot. Empty until some stage has called @ref ensure_snapshot.
  [[nodiscard]] std::vector<std::uint64_t> batch_ids() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _batch_ids;
  }

  /// The repository the snapshot was taken from; null before the first snapshot.
  [[nodiscard]] ::cucascade::shared_data_repository* repo() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _repo;
  }

  /// The snapshot is these batches of @p repo, in this order (see exact_rows_side).
  void adopt(::cucascade::shared_data_repository& repo, std::vector<std::uint64_t> batch_ids)
  {
    std::lock_guard<std::mutex> lg(_mutex);
    _repo      = &repo;
    _batch_ids = std::move(batch_ids);
    _taken     = true;
  }

 private:
  mutable std::mutex _mutex;
  ::cucascade::shared_data_repository* _repo{nullptr};
  std::vector<std::uint64_t> _batch_ids;
  bool _taken{false};
};

/// The FP32 rows a corpus stored in half precision keeps beside itself (see
/// sirius_physical_vector_cluster_label::store_half_beside()): batches of the vector column alone,
/// in the order the scan delivered them. They are written once and never sorted; each corpus row
/// carries its batch's number here and its row within it through the sort instead, eight bytes.
/// The batches live in a repository of the query's own, where the engine's downgrade moves them
/// between tiers as it does any other batch, and the join reads single rows back from them.
class exact_rows_side {
 public:
  static constexpr std::uint64_t kNoBatch = std::numeric_limits<std::uint64_t>::max();

  /// Where row @p row of batch @p number is, as one value.
  [[nodiscard]] static std::int64_t position(std::uint32_t number, std::int64_t row)
  {
    return (static_cast<std::int64_t>(number) << 32) | row;
  }

  /// A batch's number, taken before the batch exists so that its rows can carry it. A number
  /// whose batch never arrives (its task failed and ran again under another) stays empty.
  [[nodiscard]] std::uint32_t next_batch()
  {
    std::lock_guard<std::mutex> lg(_mutex);
    _batch_ids.push_back(kNoBatch);
    return static_cast<std::uint32_t>(_batch_ids.size() - 1);
  }

  /// Stores batch @p number. @p make_repo makes the repository, once, for the first batch.
  template <typename MakeRepo>
  void put(std::uint32_t number,
           const std::shared_ptr<::cucascade::data_batch>& batch,
           MakeRepo&& make_repo)
  {
    std::lock_guard<std::mutex> lg(_mutex);
    if (_repo == nullptr) { _repo = make_repo(); }
    _repo->add_data_batch(batch);
    _batch_ids[number] = batch->get_batch_id();
  }

  /// Each number's batch id, kNoBatch for an empty one.
  [[nodiscard]] std::vector<std::uint64_t> batch_ids() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _batch_ids;
  }

  [[nodiscard]] ::cucascade::shared_data_repository* repo() const
  {
    std::lock_guard<std::mutex> lg(_mutex);
    return _repo;
  }

 private:
  mutable std::mutex _mutex;
  ::cucascade::shared_data_repository* _repo{nullptr};
  std::vector<std::uint64_t> _batch_ids;
};

}  // namespace sirius::vss
