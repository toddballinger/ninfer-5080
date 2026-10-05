// Issue-55 M1: a raw depth-1 (non-trie) decision field with a ZERO-token suffix is
// scored directly at the retained execution frontier E and its winning token is
// committed *exactly* at E through the production Engine selection + publication
// path, replacing a runtime anchor that is DISCRIMINATING from the selected winner.
//
// Discriminating anchor/winner construction (Issue-55 proof hardening): the fixture
// uses a 62-token trunk of DISTINCT in-domain token ids, which the model traverses
// in a real prefill + one greedy decode round. The sampled anchor at E is the
// model's top-1 greedy token; it is observed through the ProgramImpl lane after that
// runtime round. The eligible zero-suffix field then offers two deterministic
// in-domain candidates (198, 846) that EXCLUDE the observed anchor (guarded below),
// so the field winner -- the best of those candidates at the actual frontier E --
// must differ from the anchor. That makes the committed replacement at E observable
// and the proof discriminating: the all-198 degenerate case (anchor 198 == winner
// 198) could not distinguish a correct at-E replacement from any no-op or E+1
// off-by-one placement.
//
// The retained boundary E is established through the ACTUAL Engine runtime path
// (real prefill + one greedy decode token), not tokenization alone: a pure
// prepare_tokens prompt never runs that path, so its ProgramImpl lane is empty
// (frontier 0, not retained) and can neither prove the actual runtime position nor
// satisfy the production decision-commit guard (which requires a retained, unconsumed,
// Complete lane at execution_frontier != 0). The fixture then proves the selected
// continuation's ACTUAL runtime position (not just the public result echo) through a
// test-only inspector (friend of the family Program) that reaches the ProgramImplCore
// sequence state bound to the Engine's actual model instance: execution/ledger
// frontier, retained flag, lifecycle, and the exactly-once decision-commit latch. The
// exact post-commit target/MTP KV valid frontiers (E + 1) are asserted as equalities
// (not lower bounds), so a replacement or off-by-one frontier regression fails.
// The zero-suffix field is the eligible field; the later one-suffix companion is
// scored/reported but, by M1 program-order publication, must NOT commit at E.
//
// The Engine's bound instance is obtained through a private friend fixture hook;
// the impl program/variant headers are included (as in
// the matched-force fixture) so the inspector can read the complete runtime state.
#include "core/device.h"
#include "targets/registry.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"
#include <ninfer/engine.h>
#include <ninfer/types.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace ninfer {
// Defined only in this white-box test translation unit; Engine grants friendship
// without declaring a production-visible accessor type or public member.
struct BoundInstanceReader {
    static void* read(const Engine& engine) { return engine.bound_model_instance(); }
};
} // namespace ninfer

// Test-only runtime-state inspector (mirrors the Issue55MatchedForceInspector pattern).
// Defined in the family namespace so it matches the friend declared on the family
// Program, letting it read the private ProgramImpl indirection. It traverses the
// Engine's bound instance -> Program -> impl_ -> sequences[lane] / requests[lane].
namespace ninfer::targets::qwen3_6 {
struct Issue55SelectedCommitInspector {
    using Program = qwen3_6_27b::Package::Program;
    struct Image {
        std::uint32_t execution_frontier;
        std::uint32_t ledger_frontier;
        std::vector<TokenId> ledger;
        std::uint32_t text_kv_valid;
        std::uint32_t mtp_kv_valid;
        std::uint32_t mtp_draft_count;
        bool retained;
        bool decision_commit_consumed;
        std::uint64_t lifecycle;     // numeric enum value of the lane's request lifecycle
        bool lifecycle_is_active;    // request lifecycle == Lifecycle::Active
        bool lifecycle_is_complete;  // request lifecycle == Lifecycle::Complete
        bool has_kv_backend;
    };

    static Image read(const Program& p, std::uint32_t lane = 0) {
        const auto& s = p.impl_->sequences[lane];
        const auto& r = p.impl_->requests[lane];
        using namespace detail::qwen3_6_27b_runtime;
        Image image;
        image.execution_frontier       = s.execution_frontier;
        image.ledger_frontier          = s.ledger_frontier;
        image.ledger                   = s.ledger;
        image.text_kv_valid            = s.text_kv_valid;
        image.mtp_kv_valid             = s.mtp_kv_valid;
        image.mtp_draft_count          = s.mtp_draft_count;
        image.retained                 = s.retained;
        image.decision_commit_consumed = s.decision_commit_consumed;
        image.lifecycle                = static_cast<std::uint64_t>(r.lifecycle);
        image.lifecycle_is_active      = (r.lifecycle == Lifecycle::Active);
        image.lifecycle_is_complete    = (r.lifecycle == Lifecycle::Complete);
        image.has_kv_backend           = s.kv && s.kv->backend.has_value();
        return image;
    }
};  // struct Issue55SelectedCommitInspector
}    // namespace ninfer::targets::qwen3_6

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::KvCapacityPolicy;
using ninfer::KvCacheStorage;
using ninfer::PreparedPrompt;
using ninfer::SpeculativeBackend;
using ninfer::ProposalHead;
using ninfer::DecisionFieldSpec;
using ninfer::DecisionResult;
using ninfer::DecisionFieldResult;
using ninfer::TokenId;

EngineOptions mtp_options(const char* artifact) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 4096;
    options.kv_capacity =
        KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk   = 896;
    options.kv_cache        =
        KvCacheStorage::Int4Group64;
    options.speculative.backend     =
        SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head =
        ProposalHead::Optimized;
    options.enable_vision   = false;
    options.use_cuda_graph  = false;
    options.embedding_host  = true;
    return options;
}

// Field 0: zero-suffix, eligible commit-at-E field (winner committed exactly at E).
// Field 1: one-suffix companion, scored at E but reported at E+1 -> ineligible.
// The candidates (198, 846) exclude the observed runtime anchor, so the selected
// winner is DISCRIMINATING from the anchor; a degenerate anchor inside the
// candidate set is rejected by the caller before any decision runs.
std::vector<DecisionFieldSpec> make_fields(TokenId anchor) {
    std::vector<DecisionFieldSpec> fields;

    DecisionFieldSpec z;
    z.name           = "zero";
    z.suffix_tokens  = {};                     // depth-1, zero-suffix (the relaxed path)
    z.candidate_tokens = {198, 846};           // two distinct in-domain candidates, anchor excluded
    fields.push_back(std::move(z));

    DecisionFieldSpec o;
    o.name           = "one";
    o.suffix_tokens  = {198};                  // one deterministic in-domain suffix token
    o.candidate_tokens = {198, 846};
    fields.push_back(std::move(o));

    return fields;
}

int run(const char* artifact) {
    Engine engine(mtp_options(artifact));

    // Real prefill executes the 62-token trunk (distinct in-domain tokens) and
    // samples one unexecuted anchor at E. The terminal retained lane has execution
    // frontier E=62, ledger E+1.
    constexpr std::size_t trunk_tokens = 62;
    std::vector<TokenId> trunk;
    for (std::size_t i = 0; i < trunk_tokens; ++i) {
        // Distinct in-domain ids (248077 domain): 1000+53*i stays in-domain for all i.
        trunk.push_back(static_cast<TokenId>(1000 + 53 * static_cast<std::int32_t>(i)));
    }
    // Expected retained prefix (captured before trunk moves into prepare_tokens).
    const std::vector<TokenId> trunk_pattern = trunk;
    auto prepared = engine.prepare_tokens(std::move(trunk), true);

    // Reach the Engine's ACTUAL bound model instance through the test-only
    // BoundInstanceReader (private friend access), then the Program bound to it,
    // and read its complete runtime state.
    auto* instance = static_cast<ninfer::targets::Qwen3_6_27BInstance*>(
        ninfer::BoundInstanceReader::read(engine));
    if (instance == nullptr || instance->program == nullptr) {
        std::cerr << "FAIL: Engine did not bind the Qwen3.6 27B instance/program\n";
        return 1;
    }
    auto& program = *instance->program;

    // The real Engine prefill + terminal decode round: this is the runtime path that
    // populates the ProgramImpl lane; the output token is the unexecuted anchor at E.
    ninfer::RequestOptions generation_options;
    generation_options.execution.requested_output_tokens = 1;
    generation_options.execution.allow_prefix_reuse      = true;
    generation_options.execution.sampling.temperature    = 0.0F;
    generation_options.stop.include_model_defaults       = false;
    generation_options.output.raw                        = true;
    generation_options.output.preserve_special_tokens    = true;
    ninfer::GenerationResult generated =
        engine.generate(std::move(prepared), generation_options);
    if (generated.generated_token_ids.size() != 1) {
        std::cerr << "FAIL: real Engine runtime path did not license exactly one output token\n";
        return 1;
    }
    const TokenId generated_token = generated.generated_token_ids.front();

    auto before =
        ninfer::targets::qwen3_6::Issue55SelectedCommitInspector::read(program, 0);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "BEFORE_EXEC_FRONTIER=" << before.execution_frontier << "\n";
    std::cout << "BEFORE_LEDGER_FRONTIER=" << before.ledger_frontier << "\n";
    std::cout << "BEFORE_TEXT_KV=" << before.text_kv_valid << "\n";
    std::cout << "BEFORE_MTP_KV=" << before.mtp_kv_valid << "\n";
    std::cout << "BEFORE_RETAINED=" << before.retained << "\n";
    std::cout << "BEFORE_LIFECYCLE=" << before.lifecycle << "\n";
    std::cout << "BEFORE_LIFECYCLE_IS_COMPLETE=" << before.lifecycle_is_complete << "\n";
    std::cout << "BEFORE_COMMIT_CONSUMED=" << before.decision_commit_consumed << "\n";

    const auto E = before.execution_frontier;
    auto ledger_prefix_equals = [&](const std::vector<TokenId>& ledger, const std::vector<TokenId>& pattern) {
        if (pattern.size() > ledger.size()) return false;
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            if (ledger[i] != pattern[i]) return false;
        }
        return true;
    };
    if (E != trunk_tokens || before.ledger_frontier != E + 1 ||
        before.ledger.size() != E + 1 || before.ledger[E] != generated_token ||
        !ledger_prefix_equals(before.ledger, trunk_pattern)) {
        std::cerr << "FAIL: retained trunk/anchor does not match actual E\n";
        return 1;
    }
    if (before.text_kv_valid != E || before.mtp_kv_valid != E) {
        std::cerr << "FAIL: pre-decision KV frontiers are not at E\n";
        return 1;
    }
    if (!before.retained) {
        std::cerr << "FAIL: pre-decision lane is not retained\n";
        return 1;
    }
    if (before.lifecycle_is_complete == false) {
        std::cerr << "FAIL: pre-decision lane is not at the terminal (Complete) boundary\n";
        return 1;
    }
    if (before.decision_commit_consumed) {
        std::cerr << "FAIL: commit latch already set before any commit\n";
        return 1;
    }

    // The selected zero-suffix winner must be DISCRIMINATING from the runtime anchor:
    // the model's greedy anchor is by definition the global top-1 token, while the
    // field winner is the best of the field's candidates, which exclude that anchor.
    if (generated_token == 198 || generated_token == 846) {
        std::cerr << "FAIL: greedy anchor " << generated_token
                  << " is inside the field candidate set {198, 846}; the fixture is "
                     "degenerate (anchor cannot be discriminated from the winner)\n";
        return 1;
    }
    // Supply exactly the executed prefix [0,E), not the unexecuted sampled anchor
    // at ledger[E]. A decision over E+1 tokens scores at a different frontier.
    std::vector<TokenId> decision_prompt(before.ledger.begin(),
                                         before.ledger.begin() + E);
    const DecisionResult result =
        engine.decide(engine.prepare_tokens(std::move(decision_prompt), true),
                      make_fields(generated_token));

    if (result.fields.size() != 2) {
        std::cerr << "FAIL: expected two raw fields, got "
                  << result.fields.size() << "\n";
        return 1;
    }
    const DecisionFieldResult& zero = result.fields[0];
    const DecisionFieldResult& one  = result.fields[1];

    // ---- Public result assertions (the relaxed raw path) ----
    if (zero.name != "zero" || one.name != "one") {
        std::cerr << "FAIL: field program order not preserved\n";
        return 1;
    }
    if (zero.frontier != E) {
        std::cerr << "FAIL: zero-suffix field did not score at actual retained frontier\n";
        return 1;
    }
    if (zero.suffix_tokens != 0) {
        std::cerr << "FAIL: zero-suffix field must report suffix_tokens=0\n";
        return 1;
    }
    if (zero.winner_token < 0) {
        std::cerr << "FAIL: zero-suffix field has no winner token\n";
        return 1;
    }
    if (one.suffix_tokens != 1) {
        std::cerr << "FAIL: one-suffix companion must report suffix_tokens=1\n";
        return 1;
    }
    if (one.frontier != E) {
        std::cerr << "FAIL: one-suffix companion did not score at actual retained frontier\n";
        return 1;
    }
    std::cout << "ONE_SUFFIX_FRONTIER=" << one.frontier << "\n";
    std::cout << "ZERO_WINNER=" << zero.winner_token << "\n";
    std::cout << "BEFORE_ANCHOR_AT_E=" << before.ledger[E] << "\n";

    // Discriminating proof: the selected winner at the ACTUAL frontier E is a
    // different token than the unexecuted anchor being replaced at E.
    if (zero.winner_token == generated_token) {
        std::cerr << "FAIL: selected winner " << zero.winner_token
                  << " equals the runtime anchor " << generated_token
                  << "; the actual-position proof is not discriminating\n";
        return 1;
    }
    std::cout << "DISCRIMINATING_WINNER_DIFFERS_FROM_ANCHOR=PASS\n";

    // ---- Runtime-state assertions (actual position binding) ----
    const auto after =
        ninfer::targets::qwen3_6::Issue55SelectedCommitInspector::read(program, 0);
    std::cout << "AFTER_EXEC_FRONTIER=" << after.execution_frontier << "\n";
    std::cout << "AFTER_LEDGER_FRONTIER=" << after.ledger_frontier << "\n";
    std::cout << "AFTER_TEXT_KV=" << after.text_kv_valid << "\n";
    std::cout << "AFTER_MTP_KV=" << after.mtp_kv_valid << "\n";
    std::cout << "AFTER_RETAINED=" << after.retained << "\n";
    std::cout << "AFTER_COMMIT_CONSUMED=" << after.decision_commit_consumed << "\n";
    std::cout << "AFTER_LEADER_BACK=" << (after.ledger.empty() ? -1 :
        after.ledger.back()) << "\n";
    std::cout << "AFTER_LIFECYCLE=" << after.lifecycle << "\n";

    // Commit replaces the unexecuted anchor at E; the target-only round appends
    // its successor at E+1.
    if (after.decision_commit_consumed == false) {
        std::cerr << "FAIL: zero-suffix commit at E did not fire (latch not set)\n";
        return 1;
    }
    if (after.ledger_frontier != E + 2 || after.ledger.size() != E + 2 ||
        after.execution_frontier != E + 1) {
        std::cerr << "FAIL: post-commit ledger/frontiers do not match E+1 successor\n";
        return 1;
    }
    const TokenId at_E = after.ledger[E];
    std::cout << "AFTER_TOKEN_AT_E=" << at_E << "\n";
    if (!std::equal(before.ledger.begin(), before.ledger.begin() + E,
                    after.ledger.begin()) || at_E != zero.winner_token) {
        std::cerr << "FAIL: committed token at E is not the zero-suffix winner\n";
        return 1;
    }
    // The commit ran at measured E, not E+1 which the one-suffix field would have
    // claimed. The M1 program-order publication rule commits only the first eligible
    // zero-suffix field; the later one-suffix companion is scored/reported but must NOT
    // commit at E. The exact post-commit target/MTP KV valid frontiers are E + 1
    // (the one-output decision plan materializes text through E + 1 and MTP through
    // E + 1 at a zero draft extent): they are asserted as EXACT equalities, so any
    // replacement or off-by-one frontier regression fails.
    if (after.text_kv_valid != E + 1) {
        std::cerr << "FAIL: post-commit text-KV frontier is not exactly E+1 (got "
                  << after.text_kv_valid << ", expected " << E + 1 << ")\n";
        return 1;
    }
    if (after.mtp_kv_valid != E + 1) {
        std::cerr << "FAIL: post-commit mtp-KV frontier is not exactly E+1 (got "
                  << after.mtp_kv_valid << ", expected " << E + 1 << ")\n";
        return 1;
    }
    // commit_decision_token: set Active; the commit round then resolves terminal (1) which
    // resolves to the terminal boundary -> retained + Complete (M1 program-order publication).
    if (after.retained == false) {
        std::cerr << "FAIL: commit must leave the lane at the retained terminal boundary\n";
        return 1;
    }
    if (after.lifecycle_is_complete == false) {
        std::cerr << "FAIL: post-decision lane must be at the terminal (Complete) boundary\n";
        return 1;
    }

    std::cout << "ZERO_SUFFIX_SCORED_AT_E=PASS\n";
    std::cout << "COMMIT_EXACTLY_AT_E=PASS\n";
    std::cout << "ONE_SUFFIX_INELIGIBLE_AT_E=PASS\n";
    std::cout << "ACTUAL_POSITION_BOUNDING=PASS\n";
    std::cout << "EXACT_KV_FRONTIER_E1=PASS\n";
    std::cout << "M1D_SELECTED_COMMIT_ACTUAL_POSITION=PASS\n";

    return 0;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr || *artifact == '\0') {
        std::cout
            << "skip: NINFER_QWEN3_8_27B_DECISION_WEIGHTS is not set\n";
        return 77;
    }

    try {
        return run(artifact);
    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";
        return 1;
    }
}