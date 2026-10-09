#pragma once
// Issue #58: fail-closed, host-only suspension accounting token.
// This is an ownership protocol scaffold, not KV migration or lane release.
// It deliberately exposes no method for changing physical lane occupancy.
#include <cstddef>
#include <cstdint>
#include <memory>
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
