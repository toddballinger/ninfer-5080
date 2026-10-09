#include "runtime/engine/issue58_transfer_gate.h"
#include <cassert>
#include <type_traits>
using namespace ninfer::runtime::issue58;
static_assert(!std::is_copy_constructible_v<SuspensionTransaction>);
static_assert(std::is_nothrow_move_constructible_v<SuspensionTransaction>);
constexpr TransferProof complete{
    .resolved_boundary=true,.cuda_quiesced=true,
    .text_kv_backed_up=true,.backend_kv_backed_up=true,
    .recurrent_backed_up=true,.hidden_backed_up=true,
    .allocator_release_committed=true,.allocator_restore_committed=true};
int main() {
    SuspensionTransaction valid;
    assert(valid.quiesce(complete));
    assert(valid.record_backup(complete,true));
    assert(valid.record_release(complete));
    assert(valid.record_restore(complete));
    assert(valid.stage()==TransferStage::Restored);
    assert(!valid.record_release(complete));
    assert(valid.stage()==TransferStage::Failed);
    SuspensionTransaction no_fence;
    auto proof=complete; proof.resolved_boundary=false;
    assert(!no_fence.quiesce(proof));
    assert(no_fence.failure()==TransferFailure::FenceNotSatisfied);
    assert(!no_fence.quiesce(complete));
    SuspensionTransaction pending;
    proof=complete; proof.cuda_quiesced=false;
    assert(!pending.quiesce(proof));
    assert(pending.failure()==TransferFailure::PendingCudaWork);
    SuspensionTransaction premature;
    assert(!premature.record_release(complete));
    assert(premature.failure()==TransferFailure::InvalidOrder);
    SuspensionTransaction no_kv;
    assert(no_kv.quiesce(complete));
    proof=complete; proof.backend_kv_backed_up=false;
    assert(!no_kv.record_backup(proof,true));
    assert(no_kv.failure()==TransferFailure::NoCompleteKvBackup);
    SuspensionTransaction no_linear;
    assert(no_linear.quiesce(complete));
    proof=complete; proof.recurrent_backed_up=false;
    assert(!no_linear.record_backup(proof,true));
    assert(no_linear.failure()==TransferFailure::NoRecurrentBackup);
    SuspensionTransaction no_hidden;
    assert(no_hidden.quiesce(complete));
    proof=complete; proof.hidden_backed_up=false;
    assert(!no_hidden.record_backup(proof,true));
    assert(no_hidden.failure()==TransferFailure::NoHiddenBackup);
    SuspensionTransaction no_release;
    assert(no_release.quiesce(complete));
    assert(no_release.record_backup(complete,true));
    proof=complete; proof.allocator_release_committed=false;
    assert(!no_release.record_release(proof));
    assert(no_release.failure()==TransferFailure::AllocationNotReleased);
    SuspensionTransaction no_restore;
    assert(no_restore.quiesce(complete));
    assert(no_restore.record_backup(complete,true));
    assert(no_restore.record_release(complete));
    proof=complete; proof.allocator_restore_committed=false;
    assert(!no_restore.record_restore(proof));
    assert(no_restore.failure()==TransferFailure::AllocationNotRestored);
    SuspensionTransaction cancelled;
    cancelled.cancel();
    assert(!cancelled.quiesce(complete));
    assert(cancelled.stage()==TransferStage::Failed);
}
