#include "runtime/engine/issue58_suspended_ownership.h"
#include <cassert>
#include <limits>
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
    assert(original.charge().request_id==0);
    assert(!checked_charge_totals(original.charge()).has_value());
    assert(moved.charge().request_id==42);
    assert(moved.charge().original_lane==1);
    assert(moved.charge().text_kv_bytes==100);
    assert(moved.charge().backend_kv_bytes==200);
    assert(moved.charge().recurrent_bytes==300);
    assert(moved.charge().hidden_bytes==400);
    assert(moved.charge().host_offload_bytes==500);
    const auto totals = checked_charge_totals(moved.charge());
    assert(totals.has_value());
    assert(totals->resident_device_bytes == 1000);
    assert(totals->host_offload_bytes == 500);
    SuspendedOwnershipToken replacement({.request_id=99,.text_kv_bytes=1});
    replacement = std::move(moved);
    assert(moved.charge().request_id == 0);
    assert(replacement.charge().request_id == 42);
    assert(checked_charge_totals(replacement.charge())->resident_device_bytes == 1000);
    auto* same_token = &replacement;
    replacement = std::move(*same_token);  // exercise runtime self-move guard
    assert(replacement.charge().request_id == 42);
    auto bad = replacement.charge();
    bad.text_kv_bytes = std::numeric_limits<std::size_t>::max();
    assert(!checked_charge_totals(bad).has_value());
    bad = replacement.charge();
    bad.request_id = 0;
    assert(!checked_charge_totals(bad).has_value());
}
