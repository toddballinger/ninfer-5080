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

using ninfer::DecisionFieldResult;
using ninfer::DecisionFieldSpec;
using ninfer::DecisionResult;
using ninfer::GenerationResult;
using ninfer::PrefixReusePath;
using ninfer::SpeculativeStats;
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

    require(
        static_cast<bool>(input),
        "failed to open Issue #55 corpus");

    std::vector<TokenId> result;
    result.reserve(kPromptTokens);

    std::int64_t token = 0;

    while (
        result.size() < kPromptTokens &&
        input >> token) {

        require(
            token >= 0,
            "negative token in Issue #55 corpus");

        result.push_back(
            static_cast<TokenId>(token));
    }

    require(
        result.size() == kPromptTokens,
        "Issue #55 corpus is shorter than required seed");

    return result;
}

ninfer::EngineOptions make_options(
    const char* artifact) {

    ninfer::EngineOptions options;

    options.artifact_path = artifact;

    options.max_context = 4096;

    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);

    options.max_concurrency = 1;

    options.prefill_chunk = 896;

    options.kv_cache =
        ninfer::KvCacheStorage::Int4Group64;

    options.speculative.backend =
        ninfer::SpeculativeBackend::Mtp;

    options.speculative.draft_tokens = 3;

    options.speculative.proposal_head =
        ninfer::ProposalHead::Optimized;

    options.enable_vision = false;
    options.use_cuda_graph = false;

    return options;
}

ninfer::RequestOptions generation_options(
    std::uint32_t tokens) {

    ninfer::RequestOptions options;

    options.execution.requested_output_tokens =
        tokens;

    options.execution.allow_prefix_reuse = true;

    options.execution.sampling.temperature = 0.0F;

    options.stop.include_model_defaults = false;

    options.output.raw = true;

    options.output.preserve_special_tokens = true;

    return options;
}

DecisionFieldSpec one_field() {
    DecisionFieldSpec field;

    field.name = "retained";

    field.suffix_tokens = {
        198,
    };

    field.candidate_tokens = {
        198,
        846,
        5834,
    };

    return field;
}

void require_reuse(
    const DecisionResult& result,
    const std::string& label) {

    std::cout
        << label
        << "_REUSED_PROMPT_TOKENS="
        << result.reused_prompt_tokens
        << "\n";

    std::cout
        << label
        << "_PREFIX_REUSE_PATH="
        << reuse_path_name(
            result.prefix_reuse_path)
        << "\n";

    require(
        result.reused_prompt_tokens > 0,
        label + " did not reuse retained state");

    require(
        result.prefix_reuse_path !=
            PrefixReusePath::FullReset,
        label + " unexpectedly reset");
}

void require_reuse(
    const GenerationResult& result,
    const std::string& label) {

    std::cout
        << label
        << "_REUSED_PROMPT_TOKENS="
        << result.reused_prompt_tokens
        << "\n";

    std::cout
        << label
        << "_PREFIX_REUSE_PATH="
        << reuse_path_name(
            result.prefix_reuse_path)
        << "\n";

    require(
        result.reused_prompt_tokens > 0,
        label + " did not reuse retained state");

    require(
        result.prefix_reuse_path !=
            PrefixReusePath::FullReset,
        label + " unexpectedly reset");
}

void require_same_field(
    const DecisionFieldResult& expected,
    const DecisionFieldResult& actual) {

    require(
        expected.name == actual.name,
        "field name differs");

    require(
        expected.frontier == actual.frontier,
        "field frontier differs");

    require(
        expected.suffix_tokens ==
            actual.suffix_tokens,
        "field suffix differs");

    require(
        expected.candidate_tokens ==
            actual.candidate_tokens,
        "field candidates differ");

    require(
        expected.candidate_token_paths ==
            actual.candidate_token_paths,
        "field candidate paths differ");

    require(
        expected.winner_index ==
            actual.winner_index,
        "field winner index differs");

    require(
        expected.winner_token ==
            actual.winner_token,
        "field winner token differs");

    require(
        expected.routing_probabilities ==
            actual.routing_probabilities,
        "field routing Q differs");
}

void require_same_decision(
    const DecisionResult& expected,
    const DecisionResult& actual) {

    require(
        expected.fields.size() ==
            actual.fields.size(),
        "decision field count differs");

    for (std::size_t i = 0;
         i < expected.fields.size();
         ++i) {

        require_same_field(
            expected.fields[i],
            actual.fields[i]);
    }
}

void require_same_speculative(
    const SpeculativeStats& expected,
    const SpeculativeStats& actual) {

    require(
        expected.backend == actual.backend,
        "speculative backend differs");

    require(
        expected.enabled == actual.enabled,
        "speculative enabled differs");

    require(
        expected.draft_window ==
            actual.draft_window,
        "draft window differs");

    require(
        expected.rounds == actual.rounds,
        "speculative rounds differ");

    require(
        expected.drafted_tokens ==
            actual.drafted_tokens,
        "drafted token count differs");

    require(
        expected.accepted_tokens ==
            actual.accepted_tokens,
        "accepted token count differs");

    require(
        expected.fallback_steps ==
            actual.fallback_steps,
        "fallback steps differ");

    require(
        expected.accepted_per_position ==
            actual.accepted_per_position,
        "accepted-per-position differs");
}

struct Reference {
    std::vector<TokenId> warm_tokens;
    GenerationResult continuation;
};

Reference make_reference(
    const char* artifact,
    const std::vector<TokenId>& seed) {

    ninfer::Engine engine(
        make_options(artifact));

    auto warm =
        engine.generate(
            engine.prepare_tokens(
                seed,
                true),
            generation_options(
                kWarmTokens));

    require(
        warm.generated_token_ids.size() ==
            kWarmTokens,
        "reference warm length differs");

    std::vector<TokenId> retained =
        seed;

    retained.insert(
        retained.end(),
        warm.generated_token_ids.begin(),
        warm.generated_token_ids.end());

    auto continuation =
        engine.generate(
            engine.prepare_tokens(
                retained,
                true),
            generation_options(
                kContinuationTokens));

    require_reuse(
        continuation,
        "CONTROL_CONTINUATION");

    require(
        continuation.generated_token_ids.size() ==
            kContinuationTokens,
        "reference continuation length differs");

    return {
        std::move(warm.generated_token_ids),
        std::move(continuation),
    };
}

void run_probe_scenario(
    const char* artifact,
    const std::vector<TokenId>& seed,
    const Reference& reference,
    int probe_count) {

    ninfer::Engine engine(
        make_options(artifact));

    auto warm =
        engine.generate(
            engine.prepare_tokens(
                seed,
                true),
            generation_options(
                kWarmTokens));

    require(
        warm.generated_token_ids ==
            reference.warm_tokens,
        "warm sequence differs from reference");

    std::vector<TokenId> retained =
        seed;

    retained.insert(
        retained.end(),
        warm.generated_token_ids.begin(),
        warm.generated_token_ids.end());

    const auto before =
        engine.runtime_stats();

    DecisionResult first;

    for (int i = 0;
         i < probe_count;
         ++i) {

        auto result =
            engine.decide(
                engine.prepare_tokens(
                    retained,
                    true),
                {one_field()});

        require(
            result.fields.size() == 1,
            "decision returned wrong field count");

        const std::string label =
            "P" +
            std::to_string(probe_count) +
            "_PROBE_" +
            std::to_string(i);

        require_reuse(
            result,
            label);

        std::cout
            << label
            << "_WINNER_INDEX="
            << result.fields[0].winner_index
            << "\n";

        std::cout
            << label
            << "_WINNER_TOKEN="
            << result.fields[0].winner_token
            << "\n";

        if (i == 0) {
            first = result;

        } else {
            require_same_decision(
                first,
                result);

            std::cout
                << label
                << "_MATCH_FIRST=YES\n";
        }
    }

    const auto after_probes =
        engine.runtime_stats();

    require(
        after_probes.committed_decode_tokens ==
            before.committed_decode_tokens,
        "probe changed committed decode count");

    require(
        after_probes.decode_rounds ==
            before.decode_rounds,
        "probe changed decode round count");

    std::cout
        << "P"
        << probe_count
        << "_COMMITTED_DECODE_DELTA="
        << (
            after_probes.committed_decode_tokens -
            before.committed_decode_tokens
        )
        << "\n";

    std::cout
        << "P"
        << probe_count
        << "_DECODE_ROUNDS_DELTA="
        << (
            after_probes.decode_rounds -
            before.decode_rounds
        )
        << "\n";

    auto continuation =
        engine.generate(
            engine.prepare_tokens(
                retained,
                true),
            generation_options(
                kContinuationTokens));

    require_reuse(
        continuation,
        "P" +
            std::to_string(probe_count) +
            "_CONTINUATION");

    require(
        continuation.generated_token_ids ==
            reference.continuation.generated_token_ids,
        "post-probe continuation differs from control");

    require_same_speculative(
        reference.continuation.speculative,
        continuation.speculative);

    std::cout
        << "P"
        << probe_count
        << "_CONTINUATION_TOKEN_MATCH=YES\n";

    std::cout
        << "P"
        << probe_count
        << "_SPECULATIVE_STATS_MATCH=YES\n";

    std::cout
        << "P"
        << probe_count
        << "_WARM_REUSE=YES\n";
}

int run(
    const char* artifact,
    const char* corpus) {

    const auto seed =
        load_seed(corpus);

    const Reference reference =
        make_reference(
            artifact,
            seed);

    run_probe_scenario(
        artifact,
        seed,
        reference,
        1);

    run_probe_scenario(
        artifact,
        seed,
        reference,
        2);

    std::cout
        << "ISSUE55_RETAINED_WARM_DECISION=PASS\n";

    return 0;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    const char* corpus =
        std::getenv(
            "NINFER_ISSUE55_CORPUS");

    if (
        artifact == nullptr ||
        *artifact == '\0' ||
        corpus == nullptr ||
        *corpus == '\0') {

        std::cout
            << "skip: Issue #55 real-model environment not set\n";

        return 77;
    }

    try {
        return run(
            artifact,
            corpus);

    } catch (const std::exception& error) {
        std::cerr
            << "ISSUE55_RETAINED_WARM_DECISION=FAIL\n"
            << "ERROR="
            << error.what()
            << "\n";

        return 1;
    }
}
