#include "runtime/engine/issue58_lane_evidence.h"
#include <cassert>
#include <string_view>
using namespace ninfer::runtime;
int main() {
    assert(issue58_lane_evidence(true, false, false, false) == Issue58LaneEvidence::Occupied);
    assert(issue58_lane_evidence(false, false, false, false) == Issue58LaneEvidence::PlanUnavailable);
    assert(issue58_lane_evidence(false, true, true, false) == Issue58LaneEvidence::DirectAdmittable);
    assert(issue58_lane_evidence(false, true, false, true) == Issue58LaneEvidence::RetainedEvictionAdmittable);
    assert(issue58_lane_evidence(false, true, false, false) == Issue58LaneEvidence::NotAdmittable);
    assert(std::string_view(issue58_lane_evidence_name(Issue58LaneEvidence::NotAdmittable)) == "not_admittable");
}
