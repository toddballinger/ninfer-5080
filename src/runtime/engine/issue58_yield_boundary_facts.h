#pragma once
// Issue #58: pure, device-independent eligibility fence.
// Necessary condition only: never a license to free, move or reuse GPU state.
#include <cstddef>
#include <cstdint>

namespace ninfer::runtime::issue58 {
enum class YieldBackend : std::uint8_t { Ordinary, Mtp, Unsupported };
struct YieldBoundaryFacts {
    std::uint32_t lane = 0;
    std::uint32_t sequence_lane = 0;
    std::uint32_t capacity = 0;
    bool active = false;
    bool pending_none = false;
    bool prefill_absent = false;
    bool kv_present = false;
    bool retained = false;
    std::uint32_t ledger_frontier = 0;
    std::size_t ledger_size = 0;
    std::size_t prefix_size = 0;
    std::uint32_t execution_frontier = 0;
    std::uint32_t text_kv_valid = 0;
    bool backend_kv_present = false;
    std::uint32_t mtp_kv_valid = 0;
    YieldBackend backend = YieldBackend::Unsupported;
};
[[nodiscard]] constexpr bool at_resolved_yield_boundary(
    const YieldBoundaryFacts& f) noexcept {
    if (f.lane >= f.capacity || f.sequence_lane != f.lane ||
        !f.active || !f.pending_none || !f.prefill_absent ||
        !f.kv_present || f.retained || f.ledger_frontier == 0 ||
        f.ledger_frontier != f.ledger_size ||
        f.ledger_frontier != f.prefix_size ||
        f.execution_frontier != f.ledger_frontier - 1 ||
        f.text_kv_valid != f.execution_frontier) return false;
    if (f.backend == YieldBackend::Mtp)
        return f.backend_kv_present &&
               f.mtp_kv_valid == f.execution_frontier;
    return f.backend == YieldBackend::Ordinary;
}
} // namespace ninfer::runtime::issue58
