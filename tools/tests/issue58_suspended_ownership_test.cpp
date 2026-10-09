#include "runtime/engine/issue58_suspended_ownership.h"
#include <cassert>
#include <type_traits>
#include <utility>
using namespace ninfer::runtime::issue58;
static_assert(!std::is_default_constructible_v<SuspendedOwnershipToken>);
static_assert(!std::is_copy_constructible_v<SuspendedOwnershipToken>);
static_assert(!std::is_copy_assignable_v<SuspendedOwnershipToken>);
static_assert(std::is_nothrow_move_constructible_v<SuspendedOwnershipToken>);
static_assert(std::is_nothrow_move_assignable_v<SuspendedOwnershipToken>);
int main() {
    SuspendedOwnershipToken original({.request_id=42,.original_lane=1,
       .text_kv_bytes=100,.backend_kv_bytes=200,.recurrent_bytes=300,
       .hidden_bytes=400,.host_offload_bytes=500});
    auto moved=std::move(original);
    assert(moved.charge().request_id==42);
    assert(moved.charge().original_lane==1);
    assert(moved.charge().text_kv_bytes==100);
    assert(moved.charge().backend_kv_bytes==200);
    assert(moved.charge().recurrent_bytes==300);
    assert(moved.charge().hidden_bytes==400);
    assert(moved.charge().host_offload_bytes==500);
}
