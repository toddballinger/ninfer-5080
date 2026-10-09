#include "runtime/engine/issue58_kv_allocation_plan.h"
#include <type_traits>
using namespace ninfer::runtime::issue58;
static_assert(!std::is_copy_constructible_v<KvHostBackup>);
static_assert(!std::is_copy_constructible_v<KvAllocationCapturePlan>);
static_assert(std::is_move_constructible_v<KvAllocationCapturePlan>);
// This is intentionally a compile-only contract. Real pool construction
// requires GPU memory, so the adapter is not executed by this test.
int main() { return 0; }
