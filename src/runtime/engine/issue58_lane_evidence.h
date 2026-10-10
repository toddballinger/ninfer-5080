#pragma once
// Diagnostic-only classification of observed per-lane feasibility.
#include <cstdint>
namespace ninfer::runtime {
enum class Issue58LaneEvidence : std::uint8_t {
    Occupied, DirectAdmittable, RetainedEvictionAdmittable, NotAdmittable
};
[[nodiscard]] constexpr Issue58LaneEvidence issue58_lane_evidence(
    bool occupied, bool direct_fit, bool after_retained_eviction_fit) noexcept {
    if (occupied) return Issue58LaneEvidence::Occupied;
    if (direct_fit) return Issue58LaneEvidence::DirectAdmittable;
    if (after_retained_eviction_fit) return Issue58LaneEvidence::RetainedEvictionAdmittable;
    return Issue58LaneEvidence::NotAdmittable;
}
[[nodiscard]] constexpr const char* issue58_lane_evidence_name(Issue58LaneEvidence value) noexcept {
    switch (value) {
    case Issue58LaneEvidence::Occupied: return "occupied";
    case Issue58LaneEvidence::DirectAdmittable: return "direct_admittable";
    case Issue58LaneEvidence::RetainedEvictionAdmittable: return "retained_eviction_admittable";
    case Issue58LaneEvidence::NotAdmittable: return "not_admittable";
    }
    return "unknown";
}
} // namespace ninfer::runtime
