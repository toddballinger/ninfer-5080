// Build standalone: c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_yield_boundary_test.cpp -o /tmp/issue58-fence-test
#include "runtime/engine/issue58_yield_boundary_facts.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
using namespace ninfer::runtime::issue58;
constexpr YieldBoundaryFacts good() {
    return {.lane=1,.sequence_lane=1,.capacity=2,.active=true,
      .pending_none=true,.prefill_absent=true,.kv_present=true,
      .retained=false,.ledger_frontier=101,.ledger_size=101,
      .prefix_size=101,.execution_frontier=100,.text_kv_valid=100,
      .backend_kv_present=true,.mtp_kv_valid=100,.backend=YieldBackend::Mtp};
}
constexpr bool all_cases() {
    auto f=good();
    if (!at_resolved_yield_boundary(f)) return false;
    f.active=false; if(at_resolved_yield_boundary(f))return false; f=good();
    f.pending_none=false; if(at_resolved_yield_boundary(f))return false; f=good();
    f.prefill_absent=false; if(at_resolved_yield_boundary(f))return false; f=good();
    f.kv_present=false; if(at_resolved_yield_boundary(f))return false; f=good();
    f.retained=true; if(at_resolved_yield_boundary(f))return false; f=good();
    f.lane=2; if(at_resolved_yield_boundary(f))return false; f=good();
    f.sequence_lane=0; if(at_resolved_yield_boundary(f))return false; f=good();
    f.ledger_frontier=0; if(at_resolved_yield_boundary(f))return false; f=good();
    f.ledger_size=100; if(at_resolved_yield_boundary(f))return false; f=good();
    f.prefix_size=100; if(at_resolved_yield_boundary(f))return false; f=good();
    f.execution_frontier=99; if(at_resolved_yield_boundary(f))return false; f=good();
    f.execution_frontier=std::numeric_limits<std::uint32_t>::max();
    if(at_resolved_yield_boundary(f))return false; f=good();
    f.text_kv_valid=99; if(at_resolved_yield_boundary(f))return false; f=good();
    f.backend_kv_present=false; if(at_resolved_yield_boundary(f))return false; f=good();
    f.mtp_kv_valid=99; if(at_resolved_yield_boundary(f))return false; f=good();
    f.backend=YieldBackend::Unsupported;
    if(at_resolved_yield_boundary(f))return false; f=good();
    f.backend=YieldBackend::Ordinary;
    if(!at_resolved_yield_boundary(f))return false;
    f.backend_kv_present=false;
    if(!at_resolved_yield_boundary(f))return false;
    return true;
}
static_assert(all_cases(), "Issue58 boundary must fail closed");
int main() { assert(all_cases()); }
