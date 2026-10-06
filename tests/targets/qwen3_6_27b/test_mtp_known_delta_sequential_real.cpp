// Issue #55 Phase 3 M1 baseline: deterministic known spans committed through the
// already-qualified sequential exact-target path. This is deliberately not the
// Phase-3 optimization: it establishes the correctness/timing baseline that a
// batched target-delta implementation must beat while preserving retained MTP state.
//
// Engine lifetime: candidate and reference Engines each load ~12 GiB of weights,
// and only ~3 GiB of host memory is free once the candidate Engine is live. The
// candidate Engine is therefore destroyed at the end of its scope; its observed
// tokens, reuse evidence, and commit timing are retained as host data in that
// scope so the independent reference Engine is constructed only after the
// candidate has been torn down, with no production behavior change.
#include "core/device.h"
#include "targets/registry.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ninfer {
struct BoundInstanceReader {
    static void* read(const Engine& engine) { return engine.bound_model_instance(); }
};
} // namespace ninfer

namespace {

using namespace ninfer;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

EngineOptions options(const char* artifact) {
    EngineOptions result;
    result.artifact_path = artifact;
    result.max_context = 4096;
    result.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
    result.max_concurrency = 1;
    result.prefill_chunk = 896;
    result.kv_cache = KvCacheStorage::Int4Group64;
    result.speculative.backend = SpeculativeBackend::Mtp;
    result.speculative.draft_tokens = 3;
    result.speculative.proposal_head = ProposalHead::Optimized;
    result.enable_vision = false;
    result.use_cuda_graph = false;
    result.embedding_host = true;
    return result;
}

RequestOptions generation_options(std::uint32_t outputs) {
    RequestOptions result;
    result.execution.requested_output_tokens = outputs;
    result.execution.allow_prefix_reuse = true;
    result.execution.sampling.temperature = 0.0F;
    result.stop.include_model_defaults = false;
    result.output.raw = true;
    result.output.preserve_special_tokens = true;
    return result;
}

std::vector<TokenId> trunk_tokens() {
    std::vector<TokenId> tokens;
    tokens.reserve(64);
    for (std::int32_t i = 0; i < 64; ++i) {
        tokens.push_back(static_cast<TokenId>(1000 + 53 * i));
    }
    return tokens;
}

std::vector<TokenId> known_span(std::size_t n) {
    std::vector<TokenId> tokens;
    tokens.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        tokens.push_back(static_cast<TokenId>(12000 + 17 * static_cast<std::int32_t>(i)));
    }
    return tokens;
}

void run_span(const char* artifact, std::size_t span) {
    std::vector<TokenId> trunk = trunk_tokens();
    trunk.pop_back();
    const std::vector<TokenId> known = known_span(span);
    std::vector<TokenId> continuation_prompt;

    // Candidate span: the candidate Engine is destroyed at scope exit, so its
    // weight load is released before the reference Engine below is constructed.
    std::vector<TokenId> observed_tokens;
    long commit_us = 0;
    uint32_t observed_reused_prompt_tokens = 0;
    PrefixReusePath observed_prefix_reuse_path = PrefixReusePath::FullReset;
    {
        Engine engine(options(artifact));
        auto first = engine.generate(engine.prepare_tokens(trunk, true), generation_options(2));
        require(first.generated_token_ids.size() == 2,
                "failed to establish shared anchor and replacement position");
        require(first.generated_token_ids[1] != known[0],
                "candidate sampled replacement anchor equals known[0]");
        trunk.push_back(first.generated_token_ids[0]);
        continuation_prompt = trunk;
        continuation_prompt.insert(continuation_prompt.end(), known.begin(), known.end());

        auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
            BoundInstanceReader::read(engine));
        require(instance != nullptr && instance->program != nullptr,
                "Engine did not bind Qwen3.8 program");

        auto& program = *instance->program;
        const auto started = std::chrono::steady_clock::now();
        program.commit_decision_tokens(0, known);
        const auto finished = std::chrono::steady_clock::now();
        commit_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        finished - started)
                        .count();

        auto observed = engine.generate(
            engine.prepare_tokens(continuation_prompt, true), generation_options(8));
        require(observed.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                "known span was not retained through AppendAtFrontier");
        require(observed.reused_prompt_tokens == continuation_prompt.size(),
                "known span retained frontier does not exactly match the exact prompt");
        observed_reused_prompt_tokens = observed.reused_prompt_tokens;
        observed_prefix_reuse_path = observed.prefix_reuse_path;
        observed_tokens = std::move(observed).generated_token_ids;
    }

    // Reference span: constructed only after the candidate Engine has been
    // destroyed, so the reference weight load no longer competes with a live
    // candidate load for the ~12 GiB of host weight data.
    //
    // Matched sequential FORCE: both Engines freshly prefill the same 63-token
    // shared prefix and sample the same anchor at position 63. The candidate
    // decodes one unforced successor at position 64 and replaces it with known[0]
    // through commit_decision_tokens; the independent control resolves the shared
    // prefill anchor non-terminal and forces known[0] at position 64. Subsequent
    // known tokens are likewise forced in one-token target-only rounds. One
    // final unforced target-only round executes known.back() and samples the
    // successor, bringing the execution frontier through the full prompt.
    // reference never cold-prefills the trunk+known continuation prompt.
    // The identical trunk+known prompt is then readmitted with identical eight-
    // output options (AppendAtFrontier) on both Engines, and the two generated
    // continuations plus the exact reuse identities must match exactly.
    std::vector<TokenId> expected_tokens;
    {
        Engine reference(options(artifact));
        auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
            BoundInstanceReader::read(reference));
        require(instance != nullptr && instance->program != nullptr,
                "reference Engine did not bind Qwen3.8 program");

        auto& program = *instance->program;
        // Prefill the shared prefix, confirm the sampled final trunk token,
        // then leave the lane Active and decode-ready for FORCE at position 64.
        std::vector<TokenId> shared_prefix(trunk.begin(), trunk.end() - 1);
        auto prompt = instance->loaded->frontend.prepare_tokens(shared_prefix, true);
        runtime::ResolvedExecutionOptions execution;
        // Direct Program decode rounds bypass Engine's output-budget scheduler.
        // Entitle the full forced span plus the final tail round up front: the
        // plan reserves prompt + requested_outputs - 1 text KV tokens (and the
        // corresponding MTP backend extent). A two-output plan only covers
        // the initial anchor and its successor, not this sequential control.
        execution.requested_output_tokens = static_cast<std::uint32_t>(known.size()) + 2;
        execution.allow_prefix_reuse = true;
        execution.sampling.temperature = 0.0F;
        auto base = program.plan_request_base(prompt, execution);
        auto plan = program.plan_request_for_lane(0, prompt, base);
        const auto begin = plan.summary();
        require(begin.reusable_prompt_tokens == 0,
                "reference trunk plan must be a fresh full reset");
        require(begin.prompt_tokens == shared_prefix.size(),
                "reference prefill summary contradicts the trunk prompt");
        auto step = program.start_prefill_lane(0, std::move(prompt), std::move(plan),
                                               runtime::TransientRegion{});
        std::uint32_t processed = step.processed_prompt_tokens;
        while (!step.complete) {
            step = program.advance_prefill_lane(0);
            processed += step.processed_prompt_tokens;
        }
        require(step.summary.prefix_reuse_path == PrefixReusePath::FullReset &&
                    step.summary.reused_prompt_tokens == 0,
                "reference trunk prefill was not a full reset");
        require(processed == step.summary.prompt_tokens - step.summary.reused_prompt_tokens,
                "executed reference prefill token count contradicts reuse summary");
        require(step.round.tokens.size() == 1,
                "reference prefill anchor absent");
        require(step.round.tokens[0] == trunk.back(),
                "independent prefill anchor differs from candidate trunk token");
        program.resolve_prefill_lane(0, false);
        // The unforced anchor is at 63; the forced successor is at 64.
        const std::size_t first_forced_position = step.summary.prompt_tokens + 1;
        require(first_forced_position == trunk.size(),
                "independent FORCE cannot install known[0] at candidate replacement position");

        // Install the known span at its actual positions: each known token is
        // forced at its real ledger position (trunk index 64+i) by one target-only
        // round. These rounds are non-terminal: the final known token is still
        // the unexecuted ledger tail until the following target-only round.
        for (std::size_t i = 0; i < known.size(); ++i) {
            const std::array<std::uint32_t, 1> lane{0};
            const std::array<runtime::RoundBudget, 1> budget{{
                {.generated_tokens_remaining = 1,
                 .forced_token = known[i]},
            }};
            const auto round = program.decode_batch(lane, budget);
            require(round.row_counts.size() == 1 && round.row_counts[0] == 1,
                    "known-span force round did not license exactly one token");
            const TokenId installed = round.tokens[round.row_counts[0] - 1];
            require(installed == known[i],
                    "forced reference round did not install the exact known token");
            const std::array<std::uint32_t, 1> accepted{1};
            const std::array<std::uint8_t, 1> terminal{0};
            const std::array<std::uint8_t, 1> cancelled{0};
            program.resolve_pending_batch(lane, accepted, terminal, cancelled);
        }

        // FORCE installs the successor but executes the previous anchor. Run
        // known.back() as that anchor once, just as decision commit executes
        // its selected token and retains the sampled successor at prompt end.
        const std::array<std::uint32_t, 1> lane{0};
        const std::array<runtime::RoundBudget, 1> tail_budget{{
            {.generated_tokens_remaining = 1}}};
        const auto tail = program.decode_batch(lane, tail_budget);
        require(tail.row_counts.size() == 1 && tail.row_counts[0] == 1,
                "reference final known token did not execute in a target-only round");
        const std::array<std::uint32_t, 1> accepted{1};
        const std::array<std::uint8_t, 1> terminal{1};
        const std::array<std::uint8_t, 1> cancelled{0};
        program.resolve_pending_batch(lane, accepted, terminal, cancelled);
        require(program.has_retained_lane(0),
                "final known-span resolution left the reference lane non-retained");

        // Readmit the identical trunk+known prompt with identical eight-output
        // options (AppendAtFrontier on the retained trunk+known frontier).
        auto continuation = instance->loaded->frontend.prepare_tokens(continuation_prompt, true);
        auto continuation_options = generation_options(8);
        runtime::ResolvedExecutionOptions continuation_execution;
        continuation_execution.requested_output_tokens = 8;
        continuation_execution.allow_prefix_reuse = true;
        continuation_execution.sampling.temperature = 0.0F;
        auto continuation_base = program.plan_request_base(continuation, continuation_execution);
        auto continuation_plan =
            program.plan_request_for_lane(0, continuation, continuation_base);
        const auto continuation_summary = continuation_plan.summary();
        require(continuation_summary.reusable_prompt_tokens == continuation_prompt.size(),
                "reference reuse base is not exactly the continuation frontier");
        require(continuation_summary.prompt_tokens == continuation_prompt.size(),
                "reference re-admission prompt summary contradicts the exact prompt");
        auto reference_step = program.start_prefill_lane(0, std::move(continuation),
                                                         std::move(continuation_plan),
                                                         runtime::TransientRegion{});
        while (!reference_step.complete) {
            reference_step = program.advance_prefill_lane(0);
        }
        require(reference_step.summary.prefix_reuse_path == PrefixReusePath::AppendAtFrontier &&
                    reference_step.summary.reused_prompt_tokens == continuation_prompt.size(),
                "reference begin did not exactly reuse the continuation frontier");
        require(reference_step.round.tokens.size() == 1,
                "reference continuation prefill did not license its first output");
        expected_tokens.push_back(reference_step.round.tokens[0]);
        program.resolve_prefill_lane(0, false);

        // A one-output budget guarantees zero MTP draft extent, so each
        // direct decode licenses exactly one target-only successor. The returned
        // token span remains padded to the MTP row stride (draft_window + 1);
        // row_counts, not tokens.size(), reports the licensed output count. Keep
        // the lane active through six rounds and terminate on the seventh.
        const std::array<std::uint32_t, 1> ref_lane{0};
        const std::array<std::uint32_t, 1> ref_accepted{1};
        const std::array<std::uint8_t, 1> ref_cancelled{0};
        for (std::uint32_t remaining = 7; remaining != 0; --remaining) {
            const std::array<runtime::RoundBudget, 1> ref_budget{{
                {.generated_tokens_remaining = 1}}};
            const auto ref_round = program.decode_batch(ref_lane, ref_budget);
            require(ref_round.row_counts.size() == 1 && ref_round.row_counts[0] == 1 &&
                        ref_round.tokens.size() == ref_round.row_stride &&
                        ref_round.row_stride >= 1,
                    "reference continuation did not decode one remaining output");
            expected_tokens.push_back(ref_round.tokens[0]);
            const std::array<std::uint8_t, 1> ref_terminal{
                static_cast<std::uint8_t>(remaining == 1)};
            program.resolve_pending_batch(ref_lane, ref_accepted, ref_terminal, ref_cancelled);
        }
        require(expected_tokens.size() == 8,
                "reference continuation did not capture eight outputs");
        require(observed_tokens == expected_tokens,
                "known-span continuation differs from matched sequential FORCE reference");
    }

    require(observed_reused_prompt_tokens == continuation_prompt.size() &&
                observed_prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
            "exact reused-identity comparison not satisfied");

    std::cout << "ISSUE55_PHASE3_BASELINE_SPAN=" << span << "\n"
              << "ISSUE55_PHASE3_SEQUENTIAL_COMMIT_US=" << commit_us << "\n"
              << "ISSUE55_PHASE3_APPEND_AT_FRONTIER=PASS\n"
              << "ISSUE55_PHASE3_REFERENCE_CONSTRUCTION=MATCHED_SEQUENTIAL_FORCE_ONE_TOKEN_TARGET_ONLY_ROUNDS_ACTUAL_POSITIONS_NO_COLD_SUFFIX_PREFILL\n"
              << "ISSUE55_PHASE3_ANCHOR_APPEND=NO_ACTUAL_KNOWN_POSITIONS_ONLY\n"
              << "ISSUE55_PHASE3_EXACT_REUSE_IDENTITY=PASS (reused_prompt_tokens="
              << observed_reused_prompt_tokens << "; exact-equality assertion)\n"
              << "ISSUE55_PHASE3_REFERENCE_MATCH=YES\n";
}

int run(const char* artifact) {
    constexpr std::array<std::size_t, 5> spans{1, 4, 8, 32, 128};
    for (const std::size_t span : spans) {
        run_span(artifact, span);
    }
    std::cout << "ISSUE55_PHASE3_M1_SEQUENTIAL_BASELINE=PASS\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_QWEN3_8_27B_DECISION_WEIGHTS is not set\n";
        return 77;
    }
    try {
        return run(artifact);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
