// Issue #55 Phase 3 M1 baseline: deterministic known spans committed through the
// already-qualified sequential exact-target path. This is deliberately not the
// Phase-3 optimization: it establishes the correctness/timing baseline that a
// batched target-delta implementation must beat while preserving retained MTP state.
#include "core/device.h"
#include "targets/registry.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <chrono>
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
    const std::vector<TokenId> trunk = trunk_tokens();
    const std::vector<TokenId> known = known_span(span);

    Engine engine(options(artifact));
    auto first = engine.generate(engine.prepare_tokens(trunk, true), generation_options(1));
    require(first.generated_token_ids.size() == 1,
            "failed to establish one-token retained frontier");

    auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
        BoundInstanceReader::read(engine));
    require(instance != nullptr && instance->program != nullptr,
            "Engine did not bind Qwen3.8 program");

    auto& program = *instance->program;
    const auto started = std::chrono::steady_clock::now();
    program.commit_decision_tokens(0, known);
    const auto finished = std::chrono::steady_clock::now();
    const auto commit_us =
        std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count();

    std::vector<TokenId> continuation_prompt = trunk;
    continuation_prompt.insert(continuation_prompt.end(), known.begin(), known.end());

    auto observed = engine.generate(
        engine.prepare_tokens(continuation_prompt, true), generation_options(8));
    require(observed.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
            "known span was not retained through AppendAtFrontier");
    require(observed.reused_prompt_tokens >= continuation_prompt.size(),
            "known span retained frontier is shorter than the exact prompt");

    Engine reference(options(artifact));
    auto expected = reference.generate(
        reference.prepare_tokens(continuation_prompt, true), generation_options(8));
    require(observed.generated_token_ids == expected.generated_token_ids,
            "known-span continuation differs from independent target reference");

    std::cout << "ISSUE55_PHASE3_BASELINE_SPAN=" << span << "\n"
              << "ISSUE55_PHASE3_SEQUENTIAL_COMMIT_US=" << commit_us << "\n"
              << "ISSUE55_PHASE3_APPEND_AT_FRONTIER=PASS\n"
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
