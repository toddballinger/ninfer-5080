// Issue-55 M1: production commit-selection gate with host callbacks.
// Binding rule: only a field with winner_token >= 0 AND suffix_tokens == 0
// (winner scored at the retained frontier E) may be committed at E. Fields
// scored after a nonzero deterministic suffix are not eligible; trie fields
// (winner_token == -1) and no-winner fields are ineligible. A later
// one-token field must never skip a trie and install a suffix-scored winner.
// When no field is eligible, the target is left untouched and the result is
// still published once (success path) - the existing publication contract.
#include "runtime/engine/decision_commit_publication.h"
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using runtime::settle_decision_commit;
namespace {
struct FieldCase {
    TokenId winner_token = -1; // -1: no winner / trie
    std::uint32_t suffix_tokens = 0;
    bool trie = false; // multi-token trie field (winner_token forced to -1)
};

DecisionResult make_result(const std::vector<FieldCase>& fields) {
    DecisionResult result;
    for (const FieldCase& f : fields) {
        DecisionFieldResult field;
        field.winner_token = f.winner_token;
        field.suffix_tokens = f.suffix_tokens;
        if (f.trie) {
            field.candidate_token_paths = {{1, 2}, {3, 4}};
            field.winner_token = -1;
        }
        result.fields.push_back(field);
    }
    return result;
}
} // namespace

int main() {
    int cases = 0;
    auto run = [&](const std::vector<FieldCase>& fields, bool mtp,
                   const std::vector<TokenId>& expected) {
        DecisionResult result = make_result(fields);
        const std::size_t field_count = fields.size();
        std::vector<TokenId> commits;
        int successes = 0, failures = 0;
        settle_decision_commit(std::move(result), mtp,
                               [&](TokenId winner) { commits.push_back(winner); },
                               [&](DecisionResult ready) {
                                   ++successes;
                                   if (ready.fields.size() != field_count)
                                       throw std::runtime_error("result lost");
                               },
                               [&](std::exception_ptr) { ++failures; });
        if (commits != expected || successes != 1 || failures != 0)
            throw std::runtime_error("commit gate mismatch");
        ++cases;
    };

    // Eligible one-token field at E: commits.
    run({{42, 0, false}}, true, {42});
    // Winner 0 is legal: commits.
    run({{0, 0, false}}, true, {0});
    // No-winner field (winner -1 at E): no commit, result still published once.
    run({{-1, 0, false}}, true, {});
    // Mixed: first field no-winner, second eligible one-token: first eligible in
    // program order commits (no-winner is ineligible, then 7).
    run({{-1, 0, false}, {7, 0, false}}, true, {7});
    // First eligible field in program order wins; later ones are ignored.
    run({{5, 0, false}, {9, 0, false}}, true, {5});
    // EMPTY program: no commit, result still published once (fields may be
    // zero; the gate's success path is unconditional).
    run({}, true, {});
    // One-token field SCORED AFTER a 3-token suffix: not eligible (it belongs
    // after E), no commit, result still published once (the existing
    // publication contract).
    run({{42, 3, false}}, true, {});

    // Mixed trie / one-token: the trie field has winner_token -1 (ineligible);
    // the later one-token field was scored after a nonzero deterministic suffix.
    // Neither is eligible: the one-token field must NOT skip the trie and
    // install its (after-E) winner at E.
    run({{0, 0, true}, {77, 2, false}}, true, {});
    // Same shape, but the later one-token field is eligible (scored at E):
    // the trie is still skipped (trie fields carry no singular token at E);
    // the eligible one-token field commits.
    run({{0, 0, true}, {77, 0, false}}, true, {77});
    // A trie field alone: ineligible, no commit, result published once.
    run({{0, 0, true}}, true, {});
    // Non-MTP backend: the gate never commits (target entry point is
    // MTP-only); the result is still published exactly once.
    run({{42, 0, false}}, false, {});
    run({{42, 3, false}}, false, {});

    std::cout << "decision-commit-gate: " << cases << " production gate cases passed\n";
}