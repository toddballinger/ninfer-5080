#pragma once
#include <cstdint>

namespace ninfer::runtime {
enum class Issue58KvDenial : std::uint8_t {
    InvalidOldEntitlement, InvalidReclaimableEntitlement,
    ExceedsLogicalCapacity, InsufficientPhysicalPages, None
};

[[nodiscard]] constexpr Issue58KvDenial issue58_kv_denial(
    std::uint32_t old_pages, std::uint32_t reclaimable_pages,
    std::uint32_t new_pages, std::uint32_t entitled_pages,
    std::uint32_t logical_capacity, std::uint32_t physical_pages) noexcept {
    if (old_pages > entitled_pages) return Issue58KvDenial::InvalidOldEntitlement;
    if (reclaimable_pages > entitled_pages - old_pages)
        return Issue58KvDenial::InvalidReclaimableEntitlement;
    if (new_pages > logical_capacity) return Issue58KvDenial::ExceedsLogicalCapacity;
    const std::uint32_t committed = entitled_pages - old_pages - reclaimable_pages;
    if (committed > physical_pages || new_pages > physical_pages - committed)
        return Issue58KvDenial::InsufficientPhysicalPages;
    return Issue58KvDenial::None;
}
[[nodiscard]] constexpr const char* issue58_kv_denial_name(Issue58KvDenial reason) noexcept {
    switch (reason) {
    case Issue58KvDenial::InvalidOldEntitlement: return "invalid_old_entitlement";
    case Issue58KvDenial::InvalidReclaimableEntitlement: return "invalid_reclaimable_entitlement";
    case Issue58KvDenial::ExceedsLogicalCapacity: return "exceeds_logical_capacity";
    case Issue58KvDenial::InsufficientPhysicalPages: return "insufficient_physical_pages";
    case Issue58KvDenial::None: return "none";
    }
    return "unknown";
}
} // namespace ninfer::runtime
