// GPU-only: invokes the actual target commit entry point twice on the same lane.
#include "core/device.h"
#include "targets/registry.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_8_27B_DECISION_WEIGHTS unset\n";
        return 77;
    }
    try {
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 4096;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.max_concurrency = 1;
        options.prefill_chunk = 896;
        options.kv_cache = ninfer::KvCacheStorage::Int4Group64;
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        options.enable_vision = false;
        options.use_cuda_graph = false;
        options.embedding_host = true;
        ninfer::DeviceContext device(options.device);
        auto constructed = ninfer::targets::construct_target(options, device);
        auto* target = std::get_if<std::unique_ptr<ninfer::targets::Qwen3_6_27BInstance>>(
            &constructed.active);
        if (target == nullptr || !*target) {
            throw std::runtime_error("artifact did not select Qwen3.6 27B");
        }
        auto& instance = **target;
        auto& program = *instance.program;
        std::vector<ninfer::TokenId> trunk(63, 198);
        auto prepared = instance.loaded->frontend.prepare_tokens(trunk, true);
        ninfer::runtime::ResolvedExecutionOptions execution;
        execution.sampling.temperature = 0.0F;
        execution.sampling.top_p = 1.0F;
        execution.requested_output_tokens = 1;
        execution.allow_prefix_reuse = false;
        auto base = program.plan_request_base(prepared, execution);
        auto plan = program.plan_request_for_lane(0, prepared, base);
        const auto summary = plan.summary();
        bool transient_active = false;
        try {
            ninfer::runtime::TransientRegion transient;
            if (summary.transient_bytes != 0) {
                instance.request_memory.activate(summary.transient_bytes,
                                                 summary.transient_alignment);
                transient = instance.request_memory.region();
                transient_active = true;
            }
            auto step = program.start_prefill_lane(0, std::move(prepared),
                                                   std::move(plan), transient);
            while (!step.complete) { step = program.advance_prefill_lane(0); }
            if (step.round.tokens.size() != 1) {
                throw std::runtime_error("prefill must license one anchor");
            }
            program.resolve_prefill_lane(0, true);
            if (transient_active) {
                instance.request_memory.deactivate();
                transient_active = false;
            }
        } catch (...) {
            program.abort_lane(0);
            if (transient_active) { instance.request_memory.deactivate(); }
            throw;
        }
        if (!program.has_retained_lane(0)) {
            throw std::runtime_error("prefill did not retain the target lane");
        }
        // The prefill anchor occupies position 63; commit replaces it with the winner.
        constexpr ninfer::TokenId winner = 198;
        program.commit_decision_token(0, winner);
        if (!program.has_retained_lane(0)) {
            throw std::runtime_error("first commit did not retain lane");
        }
        std::cout << "ISSUE55_TARGET_FIRST_COMMIT=PASS\n";
        try {
            program.commit_decision_token(0, winner);
        } catch (const std::logic_error& error) {
            if (std::string(error.what()) !=
                "decision commit requires an unconsumed retained MTP frontier") {
                throw;
            }
            if (!program.has_retained_lane(0)) {
                throw std::runtime_error("duplicate evicted retained lane");
            }
            // This exact guard throws before bind_sequence_kv, bridge, decode or fold.
            std::cout << "ISSUE55_TARGET_DUPLICATE_NO_FOLD=PASS\n";
            return 0;
        }
        throw std::runtime_error("duplicate target commit was accepted");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
