#pragma once
#include "ninfer/types.h"
#include <exception>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
// Worker-owned result remains unpublished until target readiness settles.
template <class Commit, class Success, class Failure>
void settle_decision_commit(DecisionResult result, bool mtp, Commit&& commit,
                            Success&& success, Failure&& failure) {
    if (mtp) {
        try {
            for (const auto& field : result.fields) {
                // Trie suffix_tokens counts ambiguity probes, not a deterministic
                // extension from E. Its selected full path starts at E.
                if (!field.candidate_token_paths.empty()) {
                    if (field.winner_index < 0 ||
                        static_cast<std::size_t>(field.winner_index) >=
                            field.candidate_token_paths.size()) {
                        throw std::logic_error("selected trie path is invalid");
                    }
                    const auto& path = field.candidate_token_paths[field.winner_index];
                    if (path.empty()) {
                        throw std::logic_error("selected trie path is empty");
                    }
                    commit(std::span<const TokenId>(path));
                    break;
                }
                // M1: a depth-1 winner with a nonzero deterministic suffix
                // belongs after E and must not be installed at E.
                if (field.winner_token >= 0 && field.suffix_tokens == 0) {
                    const TokenId winner = field.winner_token;
                    commit(std::span<const TokenId>(&winner, 1));
                    break;
                }
            }
        } catch (...) {
            failure(std::current_exception());
            return;
        }
    }
    // Success (including the no-eligible-field case: the target is left
    // untouched) is the existing publication/commit contract: the worker-owned
    // result is published exactly once, after the commit settles (or is bypassed).
    success(std::move(result));
}
} // namespace ninfer::runtime
