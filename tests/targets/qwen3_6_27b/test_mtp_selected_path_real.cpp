#include <ninfer/engine.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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
RequestOptions generation_options() {
    RequestOptions result;
    result.execution.requested_output_tokens = 16;
    result.execution.allow_prefix_reuse = true;
    result.execution.sampling.temperature = 0.0F;
    result.stop.include_model_defaults = false;
    result.output.raw = true;
    result.output.preserve_special_tokens = true;
    return result;
}
CompiledDecisionPlan plan(Engine& engine, const std::vector<std::string>& candidates) {
    StructuredDecisionSchema schema;
    FiniteChoice choice;
    choice.label = "m2_route";
    for (std::size_t i = 0; i < candidates.size(); ++i)
        choice.choices.push_back(SemanticValue::string("semantic-" + std::to_string(i)));
    const SemanticNodeId node = schema.add_finite_choice(std::move(choice));
    DecisionModelPresentation presentation;
    presentation.set_finite_choice(node, FiniteChoicePresentation{" route: ", candidates});
    return engine.compile_decision_plan(schema, presentation);
}
int run(const char* artifact) {
    const std::vector<TokenId> prompt(63, 198);
    const std::vector<std::vector<std::string>> candidate_sets = {
        {"local alpine ridge north", "local alpine valley south",
         "remote coastal ridge east", "remote coastal valley west"},
        {"alpha northern mountain road", "alpha northern river trail",
         "beta southern mountain path", "beta southern river route"},
        {"group red apple orchard north", "group red berry orchard south",
         "group blue apple valley east", "group blue berry valley west"}
    };
    for (const auto& candidates : candidate_sets) {
        std::vector<TokenId> selected;
        std::vector<TokenId> observed;
        {
            Engine engine(options(artifact));
            CompiledDecisionPlan compiled;
            try { compiled = plan(engine, candidates); }
            catch (const std::invalid_argument&) { continue; }
            DecisionResult result = engine.decide(engine.prepare_tokens(prompt, true), compiled);
            require(result.fields.size() == 1, "decision field count mismatch");
            const auto& field = result.fields.front();
            if (field.candidate_token_paths.size() != candidates.size()) continue;
            require(field.winner_token == -1 && field.winner_index >= 0 &&
                    static_cast<std::size_t>(field.winner_index) < field.candidate_token_paths.size(),
                    "invalid trie winner");
            selected = field.candidate_token_paths[field.winner_index];
            require(selected.size() > 1, "selected trie path is not multi-token");
            std::vector<TokenId> continuation_prompt = prompt;
            continuation_prompt.insert(continuation_prompt.end(), selected.begin(), selected.end());
            auto continuation = engine.generate(engine.prepare_tokens(continuation_prompt, true),
                                                generation_options());
            require(continuation.prefix_reuse_path == PrefixReusePath::AppendAtFrontier &&
                    continuation.reused_prompt_tokens >= continuation_prompt.size(),
                    "selected path was not retained exactly once at advanced frontier");
            observed = std::move(continuation.generated_token_ids);
            require(observed.size() == 16, "selected continuation length mismatch");
        }
        // Independent target execution of the exact selected path: fresh model
        // state, not the decision's restored probe logits or sampled anchor.
        Engine reference(options(artifact));
        std::vector<TokenId> reference_prompt = prompt;
        reference_prompt.insert(reference_prompt.end(), selected.begin(), selected.end());
        auto expected = reference.generate(reference.prepare_tokens(reference_prompt, true),
                                           generation_options());
        require(observed == expected.generated_token_ids,
                "selected continuation differs from independent target reference");
        std::cout << "ISSUE55_M2_SELECTED_PATH=PASS\n"
                  << "ISSUE55_M2_SELECTED_TOKENS=" << selected.size() << "\n"
                  << "ISSUE55_M2_REFERENCE_MATCH=YES\n"
                  << "ISSUE55_M2_RETAINED_FRONTIER=YES\n";
        return 0;
    }
    throw std::runtime_error("no candidate set compiled to a multi-token trie");
}
}
int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_QWEN3_8_27B_DECISION_WEIGHTS is not set\n";
        return 77;
    }
    try { return run(artifact); }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
