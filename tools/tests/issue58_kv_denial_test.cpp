#include "runtime/engine/issue58_kv_denial.h"
#include <cassert>
#include <string_view>
using namespace ninfer::runtime;
int main() {
    assert(issue58_kv_denial(0,0,3,0,8,8)==Issue58KvDenial::None);
    assert(issue58_kv_denial(9,0,1,8,12,12)==Issue58KvDenial::InvalidOldEntitlement);
    assert(issue58_kv_denial(2,7,1,8,12,12)==Issue58KvDenial::InvalidReclaimableEntitlement);
    assert(issue58_kv_denial(2,0,9,8,8,12)==Issue58KvDenial::ExceedsLogicalCapacity);
    assert(issue58_kv_denial(2,0,5,8,12,8)==Issue58KvDenial::InsufficientPhysicalPages);
    assert(issue58_kv_denial(2,2,5,8,12,8)==Issue58KvDenial::None);
    assert(std::string_view(issue58_kv_denial_name(Issue58KvDenial::InsufficientPhysicalPages))=="insufficient_physical_pages");
}
