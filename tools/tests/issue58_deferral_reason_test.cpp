#include "runtime/engine/issue58_deferral_reason.h"
#include <cassert>
#include <string_view>
using ninfer::runtime::issue58_deferral_reason;
using ninfer::runtime::issue58_deferral_reason_name;
int main() {
    assert(std::string_view(issue58_deferral_reason_name(issue58_deferral_reason(true, false))) == "short_lane_policy_hold");
    assert(std::string_view(issue58_deferral_reason_name(issue58_deferral_reason(true, true))) == "short_lane_policy_hold");
    assert(std::string_view(issue58_deferral_reason_name(issue58_deferral_reason(false, false))) == "no_vacant_lane");
    assert(std::string_view(issue58_deferral_reason_name(issue58_deferral_reason(false, true))) == "vacant_lane_not_admittable");
    return 0;
}
