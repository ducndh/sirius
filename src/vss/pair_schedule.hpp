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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace sirius::vss {

/**
 * @brief Which task a join over probe blocks and corpus chunks runs next.
 *
 * Each block gets one build task, then one pair task per corpus chunk its rows need. Handing a step
 * out and finishing it are separate calls, so a task that is retried keeps its step and nothing is
 * handed out twice. A pair task that stops early reports where, and the rest of its chunk is handed
 * out before the block's next chunk. Blocks start in order, as many at once as the schedule allows,
 * and a later block starts only when an earlier one has finished. Holds no lock: the operator does.
 */
class pair_schedule {
 public:
  enum class step_kind : std::uint8_t { build, pair };

  struct step {
    step_kind kind{step_kind::build};
    std::size_t block{0};
    std::size_t chunk{0};  ///< among the chunks the block's pair tasks cover, in the order they do
    std::size_t run{0};    ///< cluster run of that chunk to start from; 0 without clusters
    std::int64_t row{0};   ///< probe row to start from, in that run or in the block
    /// Probe rows to search at once if an earlier part of this chunk found out; 0 if none has.
    std::int64_t call_rows{0};
  };

  pair_schedule(std::size_t blocks, std::size_t blocks_at_once)
    : _blocks(blocks), _at_once(blocks_at_once == 0 ? 1 : blocks_at_once)
  {
  }

  /// The step @ref take would hand out, if one can run now.
  [[nodiscard]] std::optional<step> peek() const
  {
    for (auto b = _first_open; b < _next_build; ++b) {
      auto const& p = _blocks[b];
      if (p.done || !p.built) { continue; }
      if (!p.resume.empty()) { return p.resume.front(); }
      if (p.next_chunk < p.chunks) { return step{step_kind::pair, b, p.next_chunk, 0, 0, 0}; }
    }
    if (_next_build < _blocks.size() && _open < _at_once) {
      return step{step_kind::build, _next_build, 0, 0, 0, 0};
    }
    return std::nullopt;
  }

  /// Hands out what @ref peek names, which must be something.
  step take()
  {
    auto const next = *peek();
    auto& p         = _blocks[next.block];
    if (next.kind == step_kind::build) {
      ++_next_build;
      ++_open;
      return next;
    }
    if (!p.resume.empty()) {
      p.resume.pop_front();
    } else {
      ++p.next_chunk;
    }
    ++p.in_flight;
    return next;
  }

  /// The block's build task finished; its pair tasks will cover @p chunks chunks. True when that
  /// finishes the block, which then has nothing to run.
  bool built(std::size_t block, std::size_t chunks)
  {
    auto& p  = _blocks[block];
    p.built  = true;
    p.chunks = chunks;
    return finish_if_done(block);
  }

  /// A pair task finished all it was given.
  /// @return true when it was the block's last.
  bool pair_done(std::size_t block)
  {
    --_blocks[block].in_flight;
    return finish_if_done(block);
  }

  /// A pair task stopped early; @p rest is where the next one starts.
  void cut_short(step const& rest)
  {
    auto& p = _blocks[rest.block];
    --p.in_flight;
    p.resume.push_back(rest);
  }

  /// Nothing is left to hand out, now or later, unless a pair task under way stops early.
  [[nodiscard]] bool exhausted() const
  {
    if (_next_build < _blocks.size()) { return false; }
    for (auto b = _first_open; b < _blocks.size(); ++b) {
      auto const& p = _blocks[b];
      if (!p.done && (!p.built || !p.resume.empty() || p.next_chunk < p.chunks)) { return false; }
    }
    return true;
  }

 private:
  struct block_progress {
    bool built{false};
    bool done{false};
    std::size_t chunks{0};
    std::size_t next_chunk{0};
    std::size_t in_flight{0};
    std::deque<step> resume;
  };

  bool finish_if_done(std::size_t block)
  {
    auto& p = _blocks[block];
    if (p.done || !p.built || p.in_flight > 0 || !p.resume.empty() || p.next_chunk < p.chunks) {
      return false;
    }
    p.done = true;
    --_open;
    while (_first_open < _blocks.size() && _blocks[_first_open].done) {
      ++_first_open;
    }
    return true;
  }

  std::vector<block_progress> _blocks;
  std::size_t _at_once;
  std::size_t _open{0};
  std::size_t _next_build{0};
  std::size_t _first_open{0};
};

}  // namespace sirius::vss
