#pragma once
#include "ninfer/types.h"
#include <exception>
#include <utility>

namespace ninfer::runtime {
// Worker-owned result remains unpublished until target readiness settles.
template <class Commit, class Success, class Failure>
void settle_decision_commit(DecisionResult result, bool mtp, Commit&& commit,
                            Success&& success, Failure&& failure) {
    if (mtp) {
        try {
            for (const auto& field : result.fields) {
                // Only a field whose winner was scored at the retained frontier E
                // (winner_token >= 0 AND suffix_tokens == 0) commits at E. A
                // depth-1 field scored after a nonzero deterministic suffix belongs
                // after E; a trie or no-winner field has no singular token at E.
                // Scanning in program order means a later one-token field can never
                // skip a trie and commit where none is eligible.
                if (field.winner_token >= 0 && field.suffix_tokens == 0) {
                    commit(field.winner_token);
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
