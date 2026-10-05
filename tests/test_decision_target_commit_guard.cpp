#include "targets/qwen3_6/impl/runtime/decision_commit_guard.h"
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using ninfer::targets::qwen3_6::detail::require_decision_commit_ready;
using ninfer::targets::qwen3_6::detail::decision_commit_kv_extents;

namespace {
struct Prefix { std::size_t count = 0; std::size_t size() const { return count; } };
struct KV { bool backend = true; };
// Structural host fixture: the production guard reads these same SequenceState members.
struct Sequence {
    std::optional<KV> kv = KV{};
    bool retained = true;
    bool decision_commit_consumed = false;
    std::uint32_t execution_frontier = 5, ledger_frontier = 6;
    std::vector<ninfer::TokenId> ledger = {1, 2, 3, 4, 5, 6};
    Prefix prefix_identity{6};
    std::uint32_t text_kv_valid = 5, mtp_kv_valid = 5;
    bool tail_hidden_valid = true;
};
} // namespace

int main() {
    // A one-output decision at E=63 retains 63 mapped Text/MTP tokens;
    // terminal resolution cancels unmapped MTP growth entitlement. The
    // commit needs Text=64 (one page) and MTP=66 (two pages).
    if (decision_commit_kv_extents(63, 3, 4096) !=
            std::pair<std::uint32_t, std::uint32_t>{64, 66} ||
        decision_commit_kv_extents(62, 3, 4096) !=
            std::pair<std::uint32_t, std::uint32_t>{63, 65} ||
        decision_commit_kv_extents(4095, 3, 4096) !=
            std::pair<std::uint32_t, std::uint32_t>{4096, 4096}) {
        throw std::runtime_error("decision commit KV extents differ from MTP decode");
    }
    Sequence sequence;
    const auto check = [&] {
        require_decision_commit_ready(0, 42, 1, 100, true, true, sequence, true, 128);
    };
    check(); // The first target-authoritative commit may enter device work.
    sequence.decision_commit_consumed = true; // Production sets this after successful resolution.
    try {
        check();
        throw std::runtime_error("duplicate target commit was admitted");
    } catch (const std::logic_error& error) {
        if (std::string(error.what()) !=
            "decision commit requires an unconsumed retained MTP frontier") throw;
    }
    sequence.decision_commit_consumed = false; // Fresh lane re-admission resets authority.
    check();
    std::cout << "decision-target-commit-guard: first admitted, duplicate rejected, reset admitted\n";
}
