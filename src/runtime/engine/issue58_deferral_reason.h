#pragma once

// Issue #58: observational admission-deferral reason labels.
// Does not decide eligibility or change scheduling.
#include <cstdint>

namespace ninfer::runtime {

enum class Issue58DeferralReason : std::uint8_t {
    ShortLanePolicyHold,
    NoVacantLane,
    VacantLaneNotAdmittable,
};

[[nodiscard]] constexpr Issue58DeferralReason issue58_deferral_reason(
    bool short_lane_policy_hold, bool has_vacant_lane) noexcept {
    if (short_lane_policy_hold) { return Issue58DeferralReason::ShortLanePolicyHold; }
    return has_vacant_lane ? Issue58DeferralReason::VacantLaneNotAdmittable
                           : Issue58DeferralReason::NoVacantLane;
}

[[nodiscard]] constexpr const char* issue58_deferral_reason_name(
    Issue58DeferralReason reason) noexcept {
    switch (reason) {
    case Issue58DeferralReason::ShortLanePolicyHold: return "short_lane_policy_hold";
    case Issue58DeferralReason::NoVacantLane: return "no_vacant_lane";
    case Issue58DeferralReason::VacantLaneNotAdmittable:
        return "vacant_lane_not_admittable";
    }
    return "unknown";
}

} // namespace ninfer::runtime
