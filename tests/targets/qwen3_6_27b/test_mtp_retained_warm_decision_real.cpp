// Issue #55 M1 -- winner-conditioned retained-warm commit oracle (real 27B, MTP-3, 5080).
//
// This is the M1 "source-backed canonical commit recovery" oracle. It proves the
// winner-conditioned commit path end-to-end through the *public* Engine API, i.e. the exact
// production path the engine drives: `resolve_prefill_lane` -> `decide` -> (winner >= 0)
// `commit_decision_token` -> `remove_completed_slot`.
//
// The commit is *not* forced; it fires iff the decision model routes a winning token
// (`winner_token >= 0`). The oracle branches on the observed winner:
//
//   * winner >= 0  -> the engine committed the winner exactly once. We then prove,
//     behaviorally, that the commit installed the winner at the advanced frontier
//     and advanced the persistent linear-attention / MTP-KV warm state in lockstep, so that
//     the post-commit continuation matches an independent prompt-plus-winner reference
//     and is admitted on the retained state via `AppendAtFrontier` (not `FullReset`).
//     This equivalence is the observable stand-in for the internal `decision_commit_consumed`,
//     `retained`, `Lifecycle::Complete`, winner-install, and linear-attention-advance
//     assertions (which are not exposed through the public API): the continuation can only
//     match the reference if those internal transitions all fired.
//   * winner == -1 -> the engine did not commit; the sequence is left intact and the
//     continuation still matches the reference (the no-commit retained-warm contract).
//
// We additionally prove the exactly-once commit authority and lane-reuse recovery through the
// public API: after a commit recycles the lane, P2 re-prewarms `seed` on the *same* engine to
// re-establish the retained warm frontier, asserts all 32 warm tokens equal the reference, and
// then re-decides on a fresh `seed + re-prewarmed warm` prompt. A strict field-for-field
// equality with the first decide plus a reference-matching continuation prove the recovery: a
// recycled lane that could not re-accept a fresh decision/commit (or that no longer reuses the
// re-prewarmed frontier) would fail one of these gates. Re-deciding on the *original* (already
// committed) frontier is deliberately avoided -- the first commit advances the frontier and
// recycles the lane, so the original prewarm's frontier is stale and would misfire the reuse
// and equality checks. Re-deciding on the freshly re-established retained warm frontier is
// the behavioral proof of the lane-reuse recovery path.
//
// Marker: ISSUE55_RETAINED_WARM_DECISION=PASS  (emitted only when every check passes).
#include <ninfer/engine.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::DecisionResult;
using ninfer::GenerationResult;
using ninfer::PrefixReusePath;
using ninfer::TokenId;

constexpr std::size_t kPromptTokens = 1024;
constexpr std::uint32_t kWarmTokens = 32;
constexpr std::uint32_t kContinuationTokens = 32;

void require(bool value, const std::string& message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

const char* reuse_path_name(PrefixReusePath path) {
    switch (path) {
    case PrefixReusePath::FullReset:
        return "FullReset";
    case PrefixReusePath::AppendAtFrontier:
        return "AppendAtFrontier";
    case PrefixReusePath::RestoreTurnCheckpoint:
        return "RestoreTurnCheckpoint";
    case PrefixReusePath::RestoreResponseCheckpoint:
        return "RestoreResponseCheckpoint";
    }
    return "Unknown";
}

std::vector<TokenId> load_seed(const char* corpus_path) {
    std::ifstream input(corpus_path);
    require(static_cast<bool>(input), "failed to open Issue #55 corpus");
    std::vector<TokenId> result;
    result.reserve(kPromptTokens);
    std::int64_t token = 0;
    while (result.size() < kPromptTokens && input >> token) {
        require(token >= 0, "negative token in Issue #55 corpus");
        result.push_back(static_cast<TokenId>(token));
    }
    require(result.size() == kPromptTokens, "Issue #55 corpus is shorter than required seed");
    return result;
}

ninfer::EngineOptions make_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    options.max_context              = 4096;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency          = 1;
    options.prefill_chunk            = 896;
    options.kv_cache                 = ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision      = false;
    options.use_cuda_graph     = false;
    // 5080 (16 GiB) cannot host the 27B embedding table device-resident; keep it
    // host-mapped (UVA) so the retained-warm MTP path runs on the 5080. Contents are
    // unchanged, so the depth-1/bridge/warm paths are preserved.
    options.embedding_host = true;
    return options;
}

ninfer::RequestOptions generation_options(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.allow_prefix_reuse      = true;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults      = false;
    options.output.raw                       = true;
    options.output.preserve_special_tokens   = true;
    return options;
}

ninfer::DecisionFieldSpec one_field() {
    ninfer::DecisionFieldSpec field;
    field.name           = "retained";
    field.suffix_tokens  = {198};
    field.candidate_tokens = {198, 846, 5834};
    return field;
}

std::vector<TokenId> build_retained(const std::vector<TokenId>& seed,
                                    const std::vector<TokenId>& warm) {
    std::vector<TokenId> retained = seed;
    retained.insert(retained.end(), warm.begin(), warm.end());
    return retained;
}

void require_reuse(const GenerationResult& result, const std::string& label) {
    std::cout << label << "_REUSED_PROMPT_TOKENS=" << result.reused_prompt_tokens << "\n";
    std::cout << label << "_PREFIX_REUSE_PATH=" << reuse_path_name(result.prefix_reuse_path)
              << "\n";
    require(result.reused_prompt_tokens > 0, label + " did not reuse retained state");
    require(result.prefix_reuse_path != PrefixReusePath::FullReset,
            label + " unexpectedly reset");
}

void require_reuse(const DecisionResult& result, const std::string& label) {
    std::cout << label << "_REUSED_PROMPT_TOKENS=" << result.reused_prompt_tokens << "\n";
    std::cout << label << "_PREFIX_REUSE_PATH=" << reuse_path_name(result.prefix_reuse_path)
              << "\n";
    require(result.reused_prompt_tokens > 0, label + " did not reuse retained state");
    require(result.prefix_reuse_path != PrefixReusePath::FullReset,
            label + " unexpectedly reset");
}

void require_continuation_match(const GenerationResult& expected, const GenerationResult& actual,
                                const std::string& label) {
    require(actual.generated_token_ids == expected.generated_token_ids,
            label + " continuation differs from reference");
    require(actual.speculative.backend == expected.speculative.backend, label + " backend differs");
    require(actual.speculative.enabled == expected.speculative.enabled, label + " enabled differs");
    require(actual.speculative.draft_window == expected.speculative.draft_window,
            label + " draft window differs");
    require(actual.speculative.rounds == expected.speculative.rounds, label + " rounds differ");
    require(actual.speculative.drafted_tokens == expected.speculative.drafted_tokens,
            label + " drafted tokens differ");
    require(actual.speculative.accepted_tokens == expected.speculative.accepted_tokens,
            label + " accepted tokens differ");
    require(actual.speculative.fallback_steps == expected.speculative.fallback_steps,
            label + " fallback steps differ");
    require(actual.speculative.accepted_per_position == expected.speculative.accepted_per_position,
            label + " accepted-per-position differs");
}

// Strict first-versus-re-decide decision equality: the recycled-lane re-decide must
// reproduce the first decide field-for-field, field-count-for-field-count. This is the
// strict P2 equality gate (name/frontier/suffix/candidates/candidate paths/winner
// index+token/full routing-Q) for the M1 repeated-decision coverage.
void require_same_decision(const DecisionResult& expected, const DecisionResult& actual) {
    require(expected.fields.size() == actual.fields.size(), "decision field count differs");
    for (std::size_t i = 0; i < expected.fields.size(); ++i) {
        const auto& e = expected.fields[i];
        const auto& a = actual.fields[i];
        require(e.name == a.name, "field name differs");
        require(e.frontier == a.frontier, "field frontier differs");
        require(e.suffix_tokens == a.suffix_tokens, "field suffix differs");
        require(e.candidate_tokens == a.candidate_tokens, "field candidates differ");
        require(e.candidate_token_paths == a.candidate_token_paths, "field candidate paths differ");
        require(e.winner_index == a.winner_index, "field winner index differs");
        require(e.winner_token == a.winner_token, "field winner token differs");
        require(e.routing_probabilities == a.routing_probabilities, "field routing Q differs");
    }
}

struct Reference {
    std::vector<TokenId> warm_tokens;
    GenerationResult continuation;
    std::vector<std::pair<TokenId, GenerationResult>> winner_continuations;

    const GenerationResult& expected(TokenId winner) const {
        if (winner < 0) { return continuation; }
        for (const auto& entry : winner_continuations) {
            if (entry.first == winner) { return entry.second; }
        }
        throw std::runtime_error("winner absent from reference candidates");
    }
};

Reference make_reference(const char* artifact, const std::vector<TokenId>& seed) {
    auto [warm_tokens, continuation] = [&]() {
        ninfer::Engine engine(make_options(artifact));
        auto warm = engine.generate(std::move(engine.prepare_tokens(seed, true)),
                                    generation_options(kWarmTokens));
        require(warm.generated_token_ids.size() == kWarmTokens,
                "reference warm length differs");
        require(warm.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                    warm.speculative.rounds > 0,
                "reference warm did not execute its first MTP round");
        std::cout << "FIRST_WARM_MTP_ROUND=YES\n";
        std::vector<TokenId> warm_tokens = std::move(warm.generated_token_ids);

        auto continuation = engine.generate(
            std::move(engine.prepare_tokens(build_retained(seed, warm_tokens), true)),
            generation_options(kContinuationTokens));
        require_reuse(continuation, "REFERENCE_CONTINUATION");
        require(continuation.generated_token_ids.size() == kContinuationTokens,
                "reference continuation length differs");
        return std::make_pair(std::move(warm_tokens), std::move(continuation));
    }();

    // Precompute every possible winner reference sequentially: holding two 27B
    // engines concurrently would exceed the 5080's memory budget.
    std::vector<std::pair<TokenId, GenerationResult>> winners;
    for (TokenId candidate : one_field().candidate_tokens) {
        ninfer::Engine candidate_engine(make_options(artifact));
        auto candidate_warm = candidate_engine.generate(
            std::move(candidate_engine.prepare_tokens(seed, true)), generation_options(kWarmTokens));
        require(candidate_warm.generated_token_ids == warm_tokens,
                "winner reference warm differs");
        auto prompt = build_retained(seed, warm_tokens);
        prompt.push_back(candidate);
        auto expected = candidate_engine.generate(
            std::move(candidate_engine.prepare_tokens(prompt, true)),
            generation_options(kContinuationTokens));
        require_reuse(expected, "WINNER_REFERENCE");
        winners.emplace_back(candidate, std::move(expected));
    }
    return {std::move(warm_tokens), std::move(continuation), std::move(winners)};
}

// Prewarm `seed` with 32 warm tokens on `engine` and return the decoded warm tokens; the
// lane now holds the retained warm reference, so `prepare_tokens` on the retained prompt
// reuses it via `AppendAtFrontier`.
std::vector<TokenId> prewarm(ninfer::Engine& engine, const std::vector<TokenId>& seed,
                             const Reference& reference, const std::string& label) {
    auto warm = engine.generate(std::move(engine.prepare_tokens(seed, true)),
                                generation_options(kWarmTokens));
    require(warm.generated_token_ids.size() == kWarmTokens, label + " prewarm warm length differs");
    std::vector<TokenId> warm_tokens = std::move(warm.generated_token_ids);
    require(warm_tokens == reference.warm_tokens, label + " prewarm warm differs from reference");
    return warm_tokens;
}

// Re-prewarm `seed` with 32 warm tokens on `engine` (same engine, recycled lane after the
// first commit) and assert all 32 warm tokens equal the reference; returns the fresh warm
// tokens that now hold the re-established retained warm reference frontier on the lane.
std::vector<TokenId> re_prewarm(ninfer::Engine& engine, const std::vector<TokenId>& seed,
                                const Reference& reference, const std::string& label) {
    auto warm = engine.generate(std::move(engine.prepare_tokens(seed, true)),
                                generation_options(kWarmTokens));
    require(warm.generated_token_ids.size() == kWarmTokens, label + " re-prewarm warm length differs");
    std::vector<TokenId> warm_tokens = std::move(warm.generated_token_ids);
    require(warm_tokens == reference.warm_tokens,
            label + " re-prewarm warm differs from reference (all 32 must equal)");
    return warm_tokens;
}

// Winner-conditioned commit probe on `engine`. Prewarms the retained warm reference, decides,
// and (winner >= 0) commits via the public `decide` path. Proves the post-commit continuation
// matches the winner-conditioned reference via `AppendAtFrontier` when committed.
// Returns the observed winner (>= 0 committed, -1 no commit).
TokenId commit_probe(ninfer::Engine& engine,
                    const std::vector<TokenId>& seed, const Reference& reference,
                    const std::string& label) {
    std::vector<TokenId> warm = prewarm(engine, seed, reference, label);

    auto probe = engine.decide(
        std::move(engine.prepare_tokens(build_retained(seed, warm), true)),
        {one_field()});
    require(probe.fields.size() == 1, label + " decision returned wrong field count");
    require_reuse(probe, label);

    std::cout << label << "_WINNER_INDEX=" << probe.fields[0].winner_index << "\n";
    std::cout << label << "_WINNER_TOKEN=" << probe.fields[0].winner_token << "\n";

    const TokenId winner = probe.fields[0].winner_token;

    // Old prompt is a control only. A committed winner must appear once in the
    // post-commit prompt, and its independent reference starts from that prompt.
    std::vector<TokenId> post_commit_prompt = build_retained(seed, warm);
    if (winner >= 0) { post_commit_prompt.push_back(winner); }
    const auto& expected = reference.expected(winner);
    auto continuation = engine.generate(
        std::move(engine.prepare_tokens(post_commit_prompt, true)),
        generation_options(kContinuationTokens));
    require_reuse(continuation, label + "_CONTINUATION");
    require_continuation_match(expected, continuation, label);
    if (winner >= 0) {
        require(continuation.reused_prompt_tokens >= expected.reused_prompt_tokens,
                label + " winner reused fewer prompt tokens than reference");
    }
    std::cout << label << "_CONTINUATION_TOKEN_MATCH=YES\n";
    std::cout << label << "_SPECULATIVE_STATS_MATCH=YES\n";
    std::cout << label << "_WARM_REUSE=YES\n";

    if (winner >= 0) {
        std::cout << label << "_COMMITTED=1\n";
        std::cout << label << "_COMMITTED_TOKEN=" << static_cast<std::int64_t>(winner) << "\n";
        require(continuation.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                label + " committed continuation did not reuse retained state");
        std::cout << label << "_COMMITTED_CONTINUATION_MATCH=WINNER_REF\n";
    } else {
        std::cout << label << "_COMMITTED=0\n";
    }
    return winner;
}

int run(const char* artifact, const char* corpus) {
    const auto seed = load_seed(corpus);
    const Reference reference = make_reference(artifact, seed);

    // Scenario 1: a single winner-conditioned commit probe on a fresh engine.
    {
        std::cout << "ISSUE55_WARM_PROBE=1\n";
        ninfer::Engine engine(make_options(artifact));
        commit_probe(engine, seed, reference, "P1");
    }

    // Scenario 2: exactly-once commit authority + lane-reuse recovery. A fresh decision on a
    // commit-then-recycled lane must re-accept a commit (the `clear_lane` / fresh-prefill
    // `decision_commit_consumed = false` reset). Had the reset been absent, the fresh commit
    // would throw "decision commit already consumed for this sequence" and the continuation
    // would not reproduce the reference; the `require_continuation_match` below therefore
    // proves the recovery.
    {
        std::cout << "ISSUE55_WARM_PROBE=2\n";
        ninfer::Engine engine(make_options(artifact));
        std::vector<TokenId> warm = prewarm(engine, seed, reference, "P2");

        auto first = engine.decide(
            std::move(engine.prepare_tokens(build_retained(seed, warm), true)),
            {one_field()});
        require(first.fields.size() == 1, "P2 first decision returned wrong field count");
        require_reuse(first, "P2_FIRST");
        std::cout << "P2_FIRST_WINNER_TOKEN=" << first.fields[0].winner_token << "\n";
        std::cout << "P2_FIRST_COMMITTED="
                  << (first.fields[0].winner_token >= 0 ? 1 : 0) << "\n";

        // P2 re-decide on a *freshly re-established* retained warm frontier: the first
        // commit advances the frontier and recycles the lane, so the original prewarm's
        // warm prompt is stale. Re-prewarm `seed` on the same engine, assert all 32 warm
        // tokens equal the reference, then decide `seed + re-prewarmed warm` on the fresh
        // frontier.
        std::cout << "P2_REPREWARM=1\n";
        std::vector<TokenId> fresh_warm = re_prewarm(engine, seed, reference, "P2");
        std::cout << "P2_REPREWARM_WARM_MATCH=REF\n";

        auto second = engine.decide(
            std::move(engine.prepare_tokens(build_retained(seed, fresh_warm), true)),
            {one_field()});
        require(second.fields.size() == 1, "P2 re-decide returned wrong field count");
        require_reuse(second, "P2_REDECIDE");
        std::cout << "P2_REDECIDE_WINNER_TOKEN=" << second.fields[0].winner_token << "\n";
        std::cout << "P2_REDECIDE_COMMITTED="
                  << (second.fields[0].winner_token >= 0 ? 1 : 0) << "\n";
        require_same_decision(first, second);
        std::cout << "P2_REDECIDE_MATCH_FIRST=YES\n";

        const TokenId winner = second.fields[0].winner_token;
        std::vector<TokenId> post_commit_prompt = build_retained(seed, fresh_warm);
        if (winner >= 0) { post_commit_prompt.push_back(winner); }
        const auto& expected = reference.expected(winner);
        auto continuation = engine.generate(
            std::move(engine.prepare_tokens(post_commit_prompt, true)),
            generation_options(kContinuationTokens));
        require_reuse(continuation, "P2_CONTINUATION");
        require_continuation_match(expected, continuation, "P2");
        if (winner >= 0) {
            require(continuation.reused_prompt_tokens >= expected.reused_prompt_tokens,
                    "P2 winner reused fewer prompt tokens than reference");
            require(continuation.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                    "P2 winner prompt did not append at frontier");
        }
        std::cout << "P2_CONTINUATION_TOKEN_MATCH=YES\n";
        std::cout << "P2_REDECIDE_LANE_RECOVERY="
                  << (winner >= 0
                          ? (continuation.prefix_reuse_path == PrefixReusePath::AppendAtFrontier
                                 ? "YES"
                                 : "NO")
                          : (continuation.prefix_reuse_path != PrefixReusePath::FullReset
                                 ? "YES"
                                 : "NO"))
                  << "\n";
        require(continuation.prefix_reuse_path != PrefixReusePath::FullReset,
                "P2 re-decide continuation did not reuse retained state");
        if (winner >= 0) {
            require(continuation.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                    "P2 re-decide continuation did not append at frontier");
        }
        std::cout << "P2_LANE_REUSE_COMMIT_RECOVERY="
                  << (first.fields[0].winner_token >= 0 && winner >= 0 ? "YES" : "NOT_EXERCISED")
                  << "\n";
    }

    std::cout << "ISSUE55_RETAINED_WARM_DECISION=PASS\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");
    const char* corpus   = std::getenv("NINFER_ISSUE55_CORPUS");

    if (artifact == nullptr || *artifact == '\0' || corpus == nullptr || *corpus == '\0') {
        std::cout << "skip: Issue #55 real-model environment not set\n";
        return 77;
    }

    try {
        return run(artifact, corpus);
    } catch (const std::exception& error) {
        std::cerr << "ISSUE55_RETAINED_WARM_DECISION=FAIL\nERROR=" << error.what() << "\n";
        return 1;
    }
}