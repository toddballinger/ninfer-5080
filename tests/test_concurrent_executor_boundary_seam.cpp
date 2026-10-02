#include "runtime/engine/concurrent_executor.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

using BoundaryObservation = ninfer::runtime::BoundaryObservation;
using Observer            = std::function<void(const BoundaryObservation&)>;
using LanePair            = std::pair<std::uint32_t, std::uint64_t>;

// Drives the exact production seam helper (boundary_seam_helper), not a standalone
// validator: non-MTP or null-observer rounds emit nothing; a MTP round with an installed
// observer emits one observation per active (lane, request_id) pair, in order, tagged with
// that round index. The captured observations are the helper's real output, so the identity
// checks below verify what the production seam actually would emit.
std::vector<BoundaryObservation> run_helper(bool mtp_gated, bool install_observer,
                                             const std::vector<LanePair> &lanes,
                                             std::uint64_t round_index) {
    std::vector<BoundaryObservation> observed;
    const Observer observer =
        install_observer
            ? Observer{[&observed](const BoundaryObservation &o) { observed.push_back(o); }}
            : Observer{};
    std::span<const LanePair> s(lanes.data(), lanes.size());
    ninfer::runtime::boundary_seam_helper(mtp_gated, observer, s, round_index);
    return observed;
}

} // namespace

int main() {
    int failures = 0;
    const auto check = [&failures](bool ok, const char *message) {
        if (!ok) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    const std::vector<LanePair> two_lane   = {{3, 1}, {7, 2}};
    const std::vector<LanePair> single     = {{3, 1}};

    {
        // (1) Non-MTP backend: the seam is frozen off, so even an installed observer
        //     receives nothing (the seam never fires for any non-MTP backend).
        const auto none = run_helper(/*mtp_gated=*/false, /*install=*/true, two_lane, 4);
        check(none.empty(), "non-MTP round fired the seam observer (must be inert)");
    }

    {
        // (2) MTP + installed observer: one observation per active lane, in order, carrying
        //     that lane's (request_id, lane, round_index).
        auto got = run_helper(/*mtp_gated=*/true, /*install=*/true, two_lane, 5);
        check(got.size() == 2, "MTP round did not emit one observation per active lane");
        if (got.size() == 2) {
            check(got[0].request_id == 1 && got[0].lane == 3 && got[0].round_index == 5,
                  "first emission did not carry (request_id=1, lane=3, round=5)");
            check(got[1].request_id == 2 && got[1].lane == 7 && got[1].round_index == 5,
                  "second emission did not carry (request_id=2, lane=7, round=5)");
        }
    }

    {
        // (3) Repeated MTP rounds: identical request/lane identity across strictly increasing
        //     rounds (the single active request/lane preserved with increasing round_index) -
        //     the M1 identity invariant the seam is designed to prove, now via the helper's
        //     real output.
        std::vector<BoundaryObservation> all;
        for (std::uint64_t r : {1, 2, 3, 4}) {
            const auto round_obs = run_helper(true, true, single, r);
            check(round_obs.size() == 1, "a stable MTP round did not emit exactly one obs");
            all.insert(all.end(), round_obs.begin(), round_obs.end());
        }
        check(all.size() == 4, "repeated rounds did not accumulate one obs per round");
        check(ninfer::runtime::BoundarySeamTestAccess::ValidRoundTrip(
                  std::span<const BoundaryObservation>(all.data(), all.size())),
              "repeated rounds lost request/lane identity or round monotonicity");
    }

    {
        // (4) Null observer: the helper is a strict no-op even on a MTP round with active
        //     lanes - no emission, no crash.
        const auto nop = run_helper(true, /*install=*/false, two_lane, 6);
        check(nop.empty(), "null-observer round emitted an observation (must be no-op)");

        // (4b) MTP round with no active lanes: no emission.
        const auto empty = run_helper(true, true, std::vector<LanePair>{}, 6);
        check(empty.empty(), "MTP round with no active lanes emitted an observation");
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "M1 boundary seam (production helper): all checks passed\n";
    return 0;
}