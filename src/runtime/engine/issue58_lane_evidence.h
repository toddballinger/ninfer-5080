#pragma once
// Observational only. Cached feasibility outcomes, never active probes.
#include <cstdint>
namespace ninfer::runtime {
enum class Issue58LaneEvidence : std::uint8_t {
    Occupied, PlanUnavailable, DirectAdmittable, RetainedEvictionAdmittable, NotAdmittable
};
[[nodiscard]] constexpr Issue58LaneEvidence issue58_lane_evidence(
    bool occupied, bool cached_plan, bool direct_fit, bool after_retained_eviction_fit) noexcept {
    if (occupied) return Issue58LaneEvidence::Occupied;
    if (!cached_plan) return Issue58LaneEvidence::PlanUnavailable;
    if (direct_fit) return Issue58LaneEvidence::DirectAdmittable;
    if (after_retained_eviction_fit) return Issue58LaneEvidence::RetainedEvictionAdmittable;
    return Issue58LaneEvidence::NotAdmittable;
}
[[nodiscard]] constexpr const char* issue58_lane_evidence_name(Issue58LaneEvidence value) noexcept {
    switch (value) {
    case Issue58LaneEvidence::Occupied: return "occupied";
    case Issue58LaneEvidence::PlanUnavailable: return "plan_unavailable";
    case Issue58LaneEvidence::DirectAdmittable: return "direct_admittable";
    case Issue58LaneEvidence::RetainedEvictionAdmittable: return "retained_eviction_admittable";
    case Issue58LaneEvidence::NotAdmittable: return "not_admittable";
    }
    return "unknown";
}
} // namespace ninfer::runtime
