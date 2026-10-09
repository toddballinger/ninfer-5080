#pragma once
// Issue #58: fail-closed, host-only suspension accounting token.
// This is an ownership protocol scaffold, not KV migration or lane release.
// It deliberately exposes no method for changing physical lane occupancy.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace ninfer::runtime::issue58 {
struct SuspendedOwnershipCharge {
    std::uint64_t request_id = 0;
    std::uint32_t original_lane = 0;
    std::size_t text_kv_bytes = 0;
    std::size_t backend_kv_bytes = 0;
    std::size_t recurrent_bytes = 0;
    std::size_t hidden_bytes = 0;
    std::size_t host_offload_bytes = 0;
};


struct ChargeTotals {
    std::size_t resident_device_bytes = 0;
    std::size_t host_offload_bytes = 0;
};
// Explicit overflow-safe accounting. Never claim resident pages are freed
// merely because their metadata is represented in a suspended token.
[[nodiscard]] constexpr std::optional<ChargeTotals> checked_charge_totals(
    const SuspendedOwnershipCharge& c) noexcept {
    std::size_t resident = 0;
    const std::size_t parts[] = {
        c.text_kv_bytes, c.backend_kv_bytes, c.recurrent_bytes, c.hidden_bytes
    };
    for (const auto part : parts) {
        if (part > std::numeric_limits<std::size_t>::max() - resident)
            return std::nullopt;
        resident += part;
    }
    if (c.request_id == 0) return std::nullopt;
    return ChargeTotals{resident, c.host_offload_bytes};
}

// A reservation for resources that must stay charged while suspended.
// Sole ownership of a metadata record DOES NOT transfer device allocations.
// A future implementation must attach true KV/linear-state owners and an
// atomic release/restore transaction before enabling lane reuse.
class SuspendedOwnershipToken final {
public:
    SuspendedOwnershipToken() = delete;
    explicit SuspendedOwnershipToken(SuspendedOwnershipCharge charge) noexcept
        : charge_(charge) {}
    SuspendedOwnershipToken(const SuspendedOwnershipToken&) = delete;
    SuspendedOwnershipToken& operator=(const SuspendedOwnershipToken&) = delete;
    SuspendedOwnershipToken(SuspendedOwnershipToken&&) noexcept = default;
    SuspendedOwnershipToken& operator=(SuspendedOwnershipToken&&) noexcept = default;
    [[nodiscard]] const SuspendedOwnershipCharge& charge() const noexcept {
        return charge_;
    }
    // Deliberately NO release_lane(), resume_lane(), or GPU-ownership claim.
private:
    SuspendedOwnershipCharge charge_;
};
} // namespace ninfer::runtime::issue58
