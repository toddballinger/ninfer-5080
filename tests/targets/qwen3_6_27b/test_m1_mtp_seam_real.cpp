// Live M1 boundary proof: the recorder is installed before public Engine construction.
#include "ninfer/engine.h"
#include "runtime/engine/concurrent_executor.h"
#include "runtime/engine/engine_boundary_observer_registry.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace {
struct Recorder {
    std::mutex mutex;
    std::vector<ninfer::runtime::BoundaryObservation> observations;
    void record(const ninfer::runtime::BoundaryObservation& o) {
        std::lock_guard lock(mutex);
        observations.push_back(o);
    }
};

ninfer::EngineOptions options_for(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk = 1024;
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision = true;
    return options;
}

int exercise(const char* artifact, const char* label) {
    auto recorder = std::make_shared<Recorder>();
    const auto token = ninfer::BoundaryObserverRegistry::register_observer(
        [recorder](const ninfer::runtime::BoundaryObservation& o) { recorder->record(o); });
    bool ok = true;
    {
        // One Engine, one Engine-owned Program and executor, one submitted request.
        // The worker joins on Engine destruction before observation inspection/unregister.
        ninfer::Engine engine(options_for(artifact));
        const auto load = engine.load_summary();
        ok &= load.target == "qwen3_6_27b";
        ok &= engine.options().speculative.backend == ninfer::SpeculativeBackend::Mtp;
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 16;
        request.execution.sampling.temperature = 0.0F;
        request.execution.allow_prefix_reuse = false;
        request.stop.include_model_defaults = false;
        const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198, 198, 198};
        const auto result = engine.generate(engine.prepare_tokens(prompt, false), request);
        ok &= result.speculative.rounds >= 2;
        ok &= !result.generated_token_ids.empty();
        ok &= result.reused_prompt_tokens == 0;
        std::cout << "[M1] " << label << " target=" << load.target
                  << " backend=Mtp rounds=" << result.speculative.rounds
                  << " reused_prompt_tokens=" << result.reused_prompt_tokens << '\n';
    }
    std::vector<ninfer::runtime::BoundaryObservation> records;
    {
        std::lock_guard lock(recorder->mutex);
        records = recorder->observations;
    }
    ninfer::BoundaryObserverRegistry::unregister(token);
    // Every record came from the actual Engine-owned executor: no synthetic invocation.
    ok &= records.size() >= 2;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& o = records[i];
        std::cout << "[M1] " << label << " observation request_id=" << o.request_id
                  << " lane=" << o.lane << " round_index=" << o.round_index << '\n';
        if (i != 0) {
            ok &= o.request_id == records[0].request_id;
            ok &= o.lane == records[0].lane;
            ok &= o.round_index > records[i - 1].round_index;
        }
    }
    if (!ok) { std::cerr << "[M1] " << label << " live seam proof FAILED\n"; return 1; }
    std::cout << "[M1] " << label << " live seam proof PASS: " << records.size()
              << " real observations; one Engine/Program/request/lane; no prefix replay\n";
    return 0;
}
} // namespace

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4 = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if ((!groupwise || !*groupwise) && (!nvfp4 || !*nvfp4)) {
        std::cout << "skip: set NINFER_QWEN3_6_27B_WEIGHTS or NINFER_QWEN3_6_27B_NVFP4_WEIGHTS\n";
        return 77;
    }
    int failures = 0;
    if (groupwise && *groupwise) failures += exercise(groupwise, "groupwise");
    if (nvfp4 && *nvfp4) failures += exercise(nvfp4, "nvfp4");
    return failures ? 1 : 0;
}
