#pragma once
#include <chrono>
#include <cstdint>
namespace ninfer::runtime {
// Monotonic elapsed time; does not claim wall-clock timestamps or GPU start.
template <typename Clock>
[[nodiscard]] constexpr std::int64_t issue58_queue_wait_ms(
    typename Clock::time_point enqueued, typename Clock::time_point removed) noexcept {
    if (removed < enqueued) { return 0; }
    return std::chrono::duration_cast<std::chrono::milliseconds>(removed - enqueued).count();
}
} // namespace ninfer::runtime
