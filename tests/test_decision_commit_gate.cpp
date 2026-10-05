// Issue-55 M1/M2: production selection gate and program-order path publication.
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
            field.winner_index = 1;
            field.suffix_tokens = 3; // Trie ambiguity probes are not a deterministic extension.
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
                               [&](std::span<const TokenId> path) { commits.insert(commits.end(), path.begin(), path.end()); },
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

    // Selected trie path is candidate_token_paths[winner_index], even though
    // winner_token is -1 and ambiguity probes report a nonzero suffix count.
    run({{0, 0, true}}, true, {3, 4});
    run({{0, 0, true}, {77, 0, false}}, true, {3, 4});
    run({{0, 0, true}, {77, 2, false}}, true, {3, 4});
    run({{-1, 0, false}, {0, 0, true}}, true, {3, 4});
    // An earlier eligible M1 field retains program-order priority.
    run({{42, 0, false}, {0, 0, true}}, true, {42});
    // Non-MTP backend: the gate never commits (target entry point is
    // MTP-only); the result is still published exactly once.
    run({{42, 0, false}}, false, {});
    run({{42, 3, false}}, false, {});
    run({{0, 0, true}}, false, {});

    // Malformed selected metadata must fail locally before target mutation or
    // result publication; it may not fall through to a later M1 winner.
    {
        auto invalid = make_result({{0, 0, true}, {77, 0, false}});
        invalid.fields.front().winner_index = 2;
        int commits = 0, successes = 0, failures = 0;
        settle_decision_commit(std::move(invalid), true,
            [&](std::span<const TokenId>) { ++commits; },
            [&](DecisionResult) { ++successes; },
            [&](std::exception_ptr) { ++failures; });
        if (commits || successes || failures != 1)
            throw std::runtime_error("invalid selected path was published");
        ++cases;
    }

    std::cout << "decision-commit-gate: " << cases << " production gate cases passed\n";
}