#pragma once
// Issue58: conservative suspension transaction gate. No GPU transfer occurs here.
// External code must never reclaim a physical lane solely from this object.
#include <cstdint>
#include <utility>
namespace ninfer::runtime::issue58 {
enum class TransferStage : std::uint8_t {
    Active, Quiesced, BackedUp, Released, Restored, Failed
};
enum class TransferFailure : std::uint8_t {
    None, InvalidOrder, FenceNotSatisfied, PendingCudaWork,
    NoCompleteKvBackup, NoRecurrentBackup, NoHiddenBackup,
    AllocationNotReleased, AllocationNotRestored, Cancelled
};
struct TransferProof {
    bool resolved_boundary = false;
    bool cuda_quiesced = false;
    bool text_kv_backed_up = false;
    bool backend_kv_backed_up = false;
    bool recurrent_backed_up = false;
    bool hidden_backed_up = false;
    bool allocator_release_committed = false;
    bool allocator_restore_committed = false;
};
// Host-only state machine specifying the minimum external proof obligations.
// Methods do not access memory or change allocations. Any failed step remains
// fail-closed; a scheduler must never interpret these stages as live readiness.
class SuspensionTransaction final {
public:
    SuspensionTransaction() = default;
    SuspensionTransaction(const SuspensionTransaction&) = delete;
    SuspensionTransaction& operator=(const SuspensionTransaction&) = delete;
    // Prevent a second transaction object from inheriting a live stage.
    SuspensionTransaction(SuspensionTransaction&&) = delete;
    SuspensionTransaction& operator=(SuspensionTransaction&&) = delete;
    [[nodiscard]] TransferStage stage() const noexcept { return stage_; }
    [[nodiscard]] TransferFailure failure() const noexcept { return failure_; }
    [[nodiscard]] bool quiesce(const TransferProof& p) noexcept {
        if (stage_ != TransferStage::Active) return fail(TransferFailure::InvalidOrder);
        if (!p.resolved_boundary) return fail(TransferFailure::FenceNotSatisfied);
        if (!p.cuda_quiesced) return fail(TransferFailure::PendingCudaWork);
        stage_=TransferStage::Quiesced; return true;
    }
    [[nodiscard]] bool record_backup(const TransferProof& p, bool mtp) noexcept {
        if (stage_ != TransferStage::Quiesced) return fail(TransferFailure::InvalidOrder);
        if (!p.text_kv_backed_up || (mtp && !p.backend_kv_backed_up))
            return fail(TransferFailure::NoCompleteKvBackup);
        if (!p.recurrent_backed_up) return fail(TransferFailure::NoRecurrentBackup);
        if (!p.hidden_backed_up) return fail(TransferFailure::NoHiddenBackup);
        stage_=TransferStage::BackedUp; return true;
    }
    [[nodiscard]] bool record_release(const TransferProof& p) noexcept {
        if (stage_ != TransferStage::BackedUp) return fail(TransferFailure::InvalidOrder);
        if (!p.allocator_release_committed)
            return fail(TransferFailure::AllocationNotReleased);
        stage_=TransferStage::Released; return true;
    }
    [[nodiscard]] bool record_restore(const TransferProof& p) noexcept {
        if (stage_ != TransferStage::Released) return fail(TransferFailure::InvalidOrder);
        if (!p.allocator_restore_committed)
            return fail(TransferFailure::AllocationNotRestored);
        stage_=TransferStage::Restored; return true;
    }
    void cancel() noexcept { stage_=TransferStage::Failed; failure_=TransferFailure::Cancelled; }
private:
    [[nodiscard]] bool fail(TransferFailure error) noexcept {
        stage_=TransferStage::Failed; failure_=error; return false;
    }
    TransferStage stage_=TransferStage::Active;
    TransferFailure failure_=TransferFailure::None;
};
} // namespace ninfer::runtime::issue58
