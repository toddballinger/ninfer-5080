#include <ninfer/engine.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::DecisionFieldResult;
using ninfer::DecisionFieldSpec;
using ninfer::DecisionResult;
using ninfer::TokenId;

constexpr float kProbabilityTolerance = 1.0e-6F;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

DecisionFieldSpec field(
    std::string name,
    std::vector<TokenId> suffix,
    std::vector<TokenId> candidates) {

    DecisionFieldSpec result;
    result.name             = std::move(name);
    result.suffix_tokens    = std::move(suffix);
    result.candidate_tokens = std::move(candidates);
    return result;
}

ninfer::EngineOptions options_for(
    const char* artifact,
    ninfer::SpeculativeBackend backend) {

    ninfer::EngineOptions options;

    options.artifact_path   = artifact;
    options.max_context     = 4096;
    options.kv_capacity     =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk   = 896;
    options.kv_cache        =
        ninfer::KvCacheStorage::Int4Group64;

    options.speculative.backend = backend;

    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head =
            ninfer::ProposalHead::Optimized;
    }

    options.enable_vision   = false;
    options.use_cuda_graph  = false;

    return options;
}

DecisionResult run(
    const char* artifact,
    ninfer::SpeculativeBackend backend) {

    ninfer::Engine engine(
        options_for(artifact, backend));

    // Identical fixture to the established Engine decision test.
    // Frontier 63 also exercises the cross-page B suffix.
    std::vector<TokenId> trunk(63, 198);

    auto prompt =
        engine.prepare_tokens(
            std::move(trunk),
            true);

    std::vector<DecisionFieldSpec> fields;

    fields.push_back(
        field(
            "A1",
            {198},
            {198, 846, 5834}));

    fields.push_back(
        field(
            "B",
            {5834, 198},
            {198, 846, 5834}));

    fields.push_back(
        field(
            "A2",
            {198},
            {198, 846, 5834}));

    return engine.decide(
        std::move(prompt),
        std::move(fields));
}

float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    if (a.size() != b.size()) {
        return INFINITY;
    }

    float result = 0.0F;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result =
            std::max(
                result,
                std::fabs(a[i] - b[i]));
    }

    return result;
}

void compare_field(
    const DecisionFieldResult& ordinary,
    const DecisionFieldResult& mtp,
    std::size_t index) {

    require(
        ordinary.name == mtp.name,
        "field name differs between ordinary and MTP");

    require(
        ordinary.frontier == mtp.frontier,
        "decision frontier differs between ordinary and MTP");

    require(
        ordinary.suffix_tokens == mtp.suffix_tokens,
        "suffix accounting differs between ordinary and MTP");

    require(
        ordinary.candidate_tokens == mtp.candidate_tokens,
        "candidate echo differs between ordinary and MTP");

    require(
        ordinary.winner_index == mtp.winner_index,
        "winner index differs between ordinary and MTP");

    require(
        ordinary.winner_token == mtp.winner_token,
        "winner token differs between ordinary and MTP");

    const float error =
        max_abs_error(
            ordinary.routing_probabilities,
            mtp.routing_probabilities);

    std::cout
        << "FIELD_" << index
        << "_ORDINARY_WINNER="
        << ordinary.winner_token
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_MTP_WINNER="
        << mtp.winner_token
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_MAX_ABS_Q_ERROR="
        << std::setprecision(10)
        << error
        << "\n";

    require(
        error <= kProbabilityTolerance,
        "routing Q differs between ordinary and MTP");
}

int exercise(const char* artifact) {
    const DecisionResult ordinary =
        run(
            artifact,
            ninfer::SpeculativeBackend::None);

    const DecisionResult mtp =
        run(
            artifact,
            ninfer::SpeculativeBackend::Mtp);

    require(
        ordinary.prompt.prompt_tokens ==
            mtp.prompt.prompt_tokens,
        "prompt token count differs");

    require(
        ordinary.fields.size() ==
            mtp.fields.size(),
        "field count differs");

    require(
        ordinary.fields.size() == 3,
        "oracle fixture did not return three fields");

    for (std::size_t i = 0;
         i < ordinary.fields.size();
         ++i) {

        compare_field(
            ordinary.fields[i],
            mtp.fields[i],
            i);
    }

    std::cout
        << "ORDINARY_PROMPT_TOKENS="
        << ordinary.prompt.prompt_tokens
        << "\n";

    std::cout
        << "MTP_PROMPT_TOKENS="
        << mtp.prompt.prompt_tokens
        << "\n";

    std::cout
        << "FIELD_COUNT="
        << ordinary.fields.size()
        << "\n";

    std::cout
        << "PROBABILITY_TOLERANCE="
        << kProbabilityTolerance
        << "\n";

    std::cout
        << "TARGET_AUTHORITATIVE_Q_MATCH=YES\n";

    std::cout
        << "WINNER_MATCH=YES\n";

    std::cout
        << "ORDINARY_VS_MTP_DECISION_EQUIVALENCE=PASS\n";

    return 0;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr ||
        *artifact == '\0') {

        std::cout
            << "skip: "
               "NINFER_QWEN3_8_27B_DECISION_WEIGHTS "
               "is not set\n";

        return 77;
    }

    try {
        return exercise(artifact);

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
