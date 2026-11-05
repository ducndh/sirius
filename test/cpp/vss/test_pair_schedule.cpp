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

#include "vss/pair_schedule.hpp"

#include <catch.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <tuple>
#include <vector>

using sirius::vss::pair_schedule;
using step_kind = pair_schedule::step_kind;

TEST_CASE("pair_schedule builds a block before its pairs and starts the next when it ends",
          "[vss][vector_join][pair_tasks]")
{
  pair_schedule schedule{2, 1};
  REQUIRE(schedule.peek()->kind == step_kind::build);
  CHECK_FALSE(schedule.exhausted());
  auto const build_0 = schedule.take();
  CHECK(build_0.block == 0);
  // Nothing can start while the only block allowed at once is being built.
  CHECK_FALSE(schedule.peek().has_value());
  CHECK_FALSE(schedule.exhausted());

  CHECK_FALSE(schedule.built(0, 2));
  auto const pair_0 = schedule.take();
  auto const pair_1 = schedule.take();
  CHECK((pair_0.kind == step_kind::pair && pair_0.chunk == 0));
  CHECK((pair_1.kind == step_kind::pair && pair_1.chunk == 1));
  CHECK_FALSE(schedule.peek().has_value());

  CHECK_FALSE(schedule.pair_done(0));
  CHECK_FALSE(schedule.peek().has_value());
  CHECK(schedule.pair_done(0));

  auto const build_1 = schedule.take();
  CHECK((build_1.kind == step_kind::build && build_1.block == 1));
  CHECK(schedule.exhausted() == false);
  CHECK(schedule.built(1, 0));
  CHECK(schedule.exhausted());
  CHECK_FALSE(schedule.peek().has_value());
}

TEST_CASE("pair_schedule hands out the rest of a chunk before the block's next chunk",
          "[vss][vector_join][pair_tasks]")
{
  pair_schedule schedule{1, 1};
  std::ignore = schedule.take();
  CHECK_FALSE(schedule.built(0, 2));
  auto first = schedule.take();
  CHECK(first.chunk == 0);
  auto rest = first;
  rest.run  = 3;
  rest.row  = 40;
  schedule.cut_short(rest);
  auto const resumed = schedule.take();
  CHECK((resumed.chunk == 0 && resumed.run == 3 && resumed.row == 40));
  CHECK_FALSE(schedule.pair_done(0));
  auto const second = schedule.take();
  CHECK((second.chunk == 1 && second.run == 0 && second.row == 0));
  // The last chunk's task is under way, so nothing is left to hand out, but it may yet stop early.
  CHECK(schedule.exhausted());
  rest     = second;
  rest.row = 7;
  schedule.cut_short(rest);
  CHECK_FALSE(schedule.exhausted());
  std::ignore = schedule.take();
  CHECK(schedule.pair_done(0));
  CHECK(schedule.exhausted());
}

TEST_CASE("pair_schedule covers each row of each chunk once whatever stops early",
          "[vss][vector_join][pair_tasks]")
{
  std::mt19937 random{20261105};
  for (int trial = 0; trial < 200; ++trial) {
    std::size_t const blocks        = 1 + random() % 5;
    std::size_t const at_once       = 1 + random() % 3;
    std::size_t const in_flight_max = 1 + random() % 4;
    std::int64_t const rows         = 1 + random() % 20;
    pair_schedule schedule{blocks, at_once};
    std::vector<pair_schedule::step> running;
    std::vector<std::size_t> chunks_of(blocks);
    std::map<std::tuple<std::size_t, std::size_t, std::int64_t>, int> covered;
    std::size_t finished_blocks = 0;
    std::size_t open_blocks     = 0;
    for (;;) {
      bool const start = running.size() < in_flight_max && schedule.peek().has_value();
      if (start && (running.empty() || random() % 2 == 0)) {
        auto const next = schedule.take();
        if (next.kind == step_kind::build) { REQUIRE(++open_blocks <= at_once); }
        running.push_back(next);
        continue;
      }
      if (running.empty()) { break; }
      auto const i    = random() % running.size();
      auto const step = running[i];
      running.erase(running.begin() + static_cast<std::ptrdiff_t>(i));
      if (step.kind == step_kind::build) {
        chunks_of[step.block] = random() % 4;
        if (schedule.built(step.block, chunks_of[step.block])) {
          ++finished_blocks;
          --open_blocks;
        }
        continue;
      }
      // Rows [step.row, stop) of this chunk are this task's, however many a retry takes.
      auto const stop = step.row + 1 + static_cast<std::int64_t>(random() % rows);
      for (auto row = step.row; row < stop && row < rows; ++row) {
        ++covered[{step.block, step.chunk, row}];
      }
      if (stop < rows) {
        auto rest = step;
        rest.row  = stop;
        schedule.cut_short(rest);
      } else if (schedule.pair_done(step.block)) {
        ++finished_blocks;
        --open_blocks;
      }
    }
    REQUIRE(schedule.exhausted());
    CHECK(finished_blocks == blocks);
    std::size_t expected = 0;
    for (auto const chunks : chunks_of) {
      expected += chunks * static_cast<std::size_t>(rows);
    }
    CHECK(covered.size() == expected);
    for (auto const& [cell, times] : covered) {
      CHECK(times == 1);
    }
  }
}
