// Issue #55 Phase 3 M2 bounded exploration: determine whether the existing
// AppendAtFrontier suffix-prefill machinery already provides the target/MTP
// batched known-delta primitive. This is a mechanism-selection fixture, not a
// production planner or final Phase-3 acceptance test.
//
// The candidate first creates the same retained replacement frontier used by the
// accepted M1 baseline. It then readmits prefix+known_span and requires the
// planner to reuse exactly the retained prefix and execute exactly N suffix
// tokens through AppendAtFrontier. Because N<=128 and prefill_chunk=896, the
// known target delta is one suffix-prefill chunk rather than N decode rounds.
// Continuation output is compared with a fresh independent full-prompt Engine.
#include "core/device.h"
#include "targets/registry.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
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

runtime::ResolvedExecutionOptions direct_options(std::uint32_t outputs) {
    runtime::ResolvedExecutionOptions result;
    result.requested_output_tokens = outputs;
    result.allow_prefix_reuse = true;
    result.sampling.temperature = 0.0F;
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

template <class Program>
std::vector<TokenId> collect_eight_outputs(Program& program,
                                           runtime::PrefillStepResult step) {
    while (!step.complete) {
        step = program.advance_prefill_lane(0);
    }
    require(step.round.tokens.size() == 1,
            "append-prefill did not license the first continuation token");

    std::vector<TokenId> out;
    out.reserve(8);
    out.push_back(step.round.tokens[0]);
    program.resolve_prefill_lane(0, false);

    const std::array<std::uint32_t, 1> lane{0};
    const std::array<std::uint32_t, 1> accepted{1};
    const std::array<std::uint8_t, 1> cancelled{0};
    for (std::uint32_t remaining = 7; remaining != 0; --remaining) {
        const std::array<runtime::RoundBudget, 1> budget{{
            {.generated_tokens_remaining = 1},
        }};
        const auto round = program.decode_batch(lane, budget);
        require(round.row_counts.size() == 1 && round.row_counts[0] == 1 &&
                    round.tokens.size() == round.row_stride && round.row_stride >= 1,
                "append-prefill continuation did not decode one output");
        out.push_back(round.tokens[0]);
        const std::array<std::uint8_t, 1> terminal{
            static_cast<std::uint8_t>(remaining == 1)};
        program.resolve_pending_batch(lane, accepted, terminal, cancelled);
    }
    require(out.size() == 8, "append-prefill continuation did not capture eight outputs");
    return out;
}

void run_span(const char* artifact, std::size_t span, long sequential_us) {
    std::vector<TokenId> trunk = trunk_tokens();
    trunk.pop_back();
    const std::vector<TokenId> known = known_span(span);
    std::vector<TokenId> exact_prompt;
    std::vector<TokenId> observed;
    long append_prefill_us = 0;
    std::uint32_t reused = 0;
    std::uint32_t processed = 0;

    {
        Engine candidate(options(artifact));
        auto initial =
            candidate.generate(candidate.prepare_tokens(trunk, true), generation_options(2));
        require(initial.generated_token_ids.size() == 2,
                "failed to establish retained replacement frontier");
        require(initial.generated_token_ids[1] != known[0],
                "sampled replacement anchor equals known[0]; fixture is not discriminating");

        trunk.push_back(initial.generated_token_ids[0]);
        exact_prompt = trunk;
        exact_prompt.insert(exact_prompt.end(), known.begin(), known.end());

        auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
            BoundInstanceReader::read(candidate));
        require(instance != nullptr && instance->program != nullptr,
                "candidate Engine did not bind Qwen3.8 program");
        auto& program = *instance->program;

        auto prompt = instance->loaded->frontend.prepare_tokens(exact_prompt, true);
        auto execution = direct_options(8);
        auto base_plan = program.plan_request_base(prompt, execution);
        auto plan = program.plan_request_for_lane(0, prompt, base_plan);
        const auto summary = plan.summary();

        require(summary.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                "candidate did not select AppendAtFrontier");
        require(summary.reusable_prompt_tokens == trunk.size(),
                "candidate reused more or less than the exact retained prefix");
        require(summary.prompt_tokens == exact_prompt.size(),
                "candidate plan prompt size mismatch");
        require(summary.prompt_tokens - summary.reusable_prompt_tokens == known.size(),
                "candidate suffix work is not exactly the known span");
        reused = summary.reusable_prompt_tokens;

        const auto started = std::chrono::steady_clock::now();
        auto step = program.start_prefill_lane(
            0, std::move(prompt), std::move(plan), runtime::TransientRegion{});
        processed += step.processed_prompt_tokens;
        while (!step.complete) {
            step = program.advance_prefill_lane(0);
            processed += step.processed_prompt_tokens;
        }
        const auto finished = std::chrono::steady_clock::now();
        append_prefill_us =
            std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count();

        require(processed == known.size(),
                "AppendAtFrontier replayed tokens outside the known delta");
        observed = collect_eight_outputs(program, std::move(step));
    }

    std::vector<TokenId> expected;
    {
        Engine reference(options(artifact));
        auto result =
            reference.generate(reference.prepare_tokens(exact_prompt, true), generation_options(8));
        require(result.prefix_reuse_path == PrefixReusePath::FullReset,
                "fresh independent reference unexpectedly reused resident state");
        require(result.generated_token_ids.size() == 8,
                "fresh independent reference did not return eight outputs");
        expected = std::move(result.generated_token_ids);
    }

    require(observed == expected,
            "AppendAtFrontier batched suffix continuation differs from independent target reference");

    const double speedup =
        append_prefill_us > 0 ? static_cast<double>(sequential_us) / append_prefill_us : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "ISSUE55_PHASE3_M2_SPAN=" << span << "\n"
              << "ISSUE55_PHASE3_M2_SEQUENTIAL_BASELINE_US=" << sequential_us << "\n"
              << "ISSUE55_PHASE3_M2_APPEND_PREFILL_US=" << append_prefill_us << "\n"
              << "ISSUE55_PHASE3_M2_SPEEDUP_VS_SEQUENTIAL=" << speedup << "\n"
              << "ISSUE55_PHASE3_M2_REUSED_PREFIX_TOKENS=" << reused << "\n"
              << "ISSUE55_PHASE3_M2_EXECUTED_SUFFIX_TOKENS=" << processed << "\n"
              << "ISSUE55_PHASE3_M2_NO_OLD_PREFIX_REPLAY=PASS\n"
              << "ISSUE55_PHASE3_M2_APPEND_AT_FRONTIER=PASS\n"
              << "ISSUE55_PHASE3_M2_INDEPENDENT_REFERENCE_MATCH=YES\n";
}

int run(const char* artifact) {
    constexpr std::array<std::size_t, 5> spans{1, 4, 8, 32, 128};
    constexpr std::array<long, 5> m1_us{25495, 99810, 199316, 802715, 3233573};
    for (std::size_t i = 0; i < spans.size(); ++i) {
        run_span(artifact, spans[i], m1_us[i]);
    }
    std::cout << "ISSUE55_PHASE3_M2_APPEND_PREFILL_EXPLORATION=PASS\n";
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
