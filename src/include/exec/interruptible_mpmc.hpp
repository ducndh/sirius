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

#pragma once

#include <blockingconcurrentqueue.h>

#include <atomic>
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>

namespace sirius::exec {

// Type trait to detect std::shared_ptr
template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// Type trait to detect std::unique_ptr
template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename T, typename D>
struct is_unique_ptr<std::unique_ptr<T, D>> : std::true_type {};

// Concept requiring T to be either shared_ptr or unique_ptr
template <typename T>
concept smart_pointer = is_shared_ptr<T>::value || is_unique_ptr<T>::value;

template <smart_pointer T>
class interruptible_mpmc {
  using value_type   = typename T::element_type;
  using pointer_type = T;

 private:
  // The underlying high-performance queue
  duckdb_moodycamel::BlockingConcurrentQueue<pointer_type> queue;

  // Atomic flag to manage the shutdown state
  std::atomic<bool> _is_active{true};

  // interrupt() wakes blocked consumers by enqueueing one null token per thread parked in pop();
  // every read skips tokens, and _wake_tokens counts the ones still queued so is_empty() ignores
  // them. Without the tokens a consumer only saw the flag when its 10 ms timed wait ran out, and
  // stopping the three executors at the end of every query cost ~30 ms.
  std::atomic<int> _waiters{0};
  std::atomic<std::int64_t> _wake_tokens{0};

 public:
  interruptible_mpmc() = default;
  // Delete copy/move to prevent unsafe duplication of the internal queue
  interruptible_mpmc(const interruptible_mpmc&)            = delete;
  interruptible_mpmc& operator=(const interruptible_mpmc&) = delete;

  [[nodiscard]] bool is_open() const noexcept { return _is_active.load(std::memory_order_relaxed); }

  /**
   * \brief Pushes an item into the queue.
   * \return Returns false if the queue has been stopped/interrupted.
   */
  template <typename... Args>
  [[nodiscard]] bool emplace(Args&&... args)
  {
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(std::make_unique<value_type>(std::forward<Args>(args)...));
    return true;
  }

  bool push(pointer_type item)
  {
    assert(item != nullptr);
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(std::move(item));
    return true;
  }

  /**
   * \brief Blocks waiting for an item.
   * \return Returns std::nullopt if the queue is interrupted (shutdown).
   */
  pointer_type pop()
  {
    pointer_type item = nullptr;
    _waiters.fetch_add(1);
    while (_is_active.load()) {
      if (queue.wait_dequeue_timed(item, 10000)) {
        if (item) {
          _waiters.fetch_sub(1);
          return std::move(item);
        }
        _wake_tokens.fetch_sub(1);
      }
    }
    _waiters.fetch_sub(1);
    return nullptr;
  }

  /**
   * \brief Attempts to pop without blocking.
   * \return Returns nullptr if the queue is empty.
   */
  pointer_type try_pop()
  {
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {
      if (item) { return std::move(item); }
      _wake_tokens.fetch_sub(1);
    }
    return nullptr;
  }

  /**
   * Interrupts the queue.
   * \brief Sets the active flag to false.
   * Consumers parked in pop() are woken by a null token each and return nullptr.
   */
  void interrupt()
  {
    _is_active.store(false);
    for (int n = _waiters.load(); n > 0; --n) {
      _wake_tokens.fetch_add(1);
      queue.enqueue(nullptr);
    }
  }

  void drain()
  {
    pointer_type item = nullptr;
    while (queue.try_dequeue(item)) {
      if (!item) { _wake_tokens.fetch_sub(1); }
    }
  }

  /**
   * \brief Returns true if the queue is approximately empty.
   *
   * Uses size_approx() from the underlying concurrent queue, which may
   * transiently over- or under-count in the presence of concurrent producers
   * and consumers. Safe for assertions in quiescent states (e.g. after drain).
   */
  [[nodiscard]] bool is_empty() const noexcept
  {
    return static_cast<std::int64_t>(queue.size_approx()) <= _wake_tokens.load();
  }

  /**
   * Resets the queue state to active (useful for restarting workers).
   */
  void reactivate() { _is_active.store(true, std::memory_order_relaxed); }
};

}  // namespace sirius::exec
