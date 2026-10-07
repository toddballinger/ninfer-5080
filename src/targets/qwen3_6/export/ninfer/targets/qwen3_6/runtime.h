#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

struct Issue55MatchedForceInspector; // test-only friend, no FORCE API
struct Issue55SelectedCommitInspector; // test-only friend, no decision API

enum class TextPhase {
    Prefill,
    Verify,
};

struct DecisionProbeResult {
    std::vector<float> probabilities;
    std::int32_t winner_index = -1;
    TokenId winner_token      = -1;

    std::uint32_t frontier      = 0;
    std::uint32_t suffix_tokens = 0;

    double capture_seconds = 0.0;
    double suffix_seconds  = 0.0;
    double score_seconds   = 0.0;
    double restore_seconds = 0.0;
};

// One residual finite-choice probe beneath a temporary shared decision
// frontier. Spans are borrowed only for the synchronous call.
struct DecisionWaveProbeSpec {
    std::span<const TokenId> suffix_tokens;
    std::span<const TokenId> candidate_tokens;
};

// Result of one target-private shared-frontier transaction.
struct DecisionWaveProbeResult {
    std::vector<DecisionProbeResult> probes;

    std::uint32_t frontier = 0;
    std::uint32_t shared_prefix_tokens = 0;

    // Actual deterministic target traversals performed by the wave:
    // shared prefix once + every residual sibling suffix.
    std::uint32_t executed_suffix_tokens = 0;

    // Work the replay baseline would perform:
    // complete shared-prefix + residual path for every sibling.
    std::uint32_t replay_equivalent_suffix_tokens = 0;

    double capture_seconds       = 0.0;
    double shared_prefix_seconds = 0.0;
    double restore_seconds       = 0.0;
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    // Issue-55 Phase-2 M1: read-only view of the target's speculative backend, so the
    // engine can gate the exactly-once decision commit to MTP programs (the only backend
    // with an MTP commit frame) without touching the engine's target-private header.
    [[nodiscard]] SpeculativeBackend speculative_backend() const noexcept;

    [[nodiscard]] DecisionProbeResult
    decision_probe_lane(std::uint32_t lane,
                        std::span<const TokenId> suffix_tokens,
                        std::span<const TokenId> candidate_tokens);

    [[nodiscard]] DecisionWaveProbeResult
    decision_probe_wave_lane(
        std::uint32_t lane,
        std::span<const TokenId> shared_prefix_tokens,
        std::span<const DecisionWaveProbeSpec> probes);

    void commit_decision_token(std::uint32_t lane, TokenId winner_token);
    void commit_decision_tokens(std::uint32_t lane, std::span<const TokenId> selected_path);

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    // Issue #55: single-shot diagnostic observer, off by default. Supports only
    // eager layer0/slot0/absolute cache position 62; image read requires sync.
    void set_gdn_mix_observer(std::int32_t target_position, std::int32_t target_layer);
    [[nodiscard]] std::uint8_t* gdn_capture_image() const noexcept;
    [[nodiscard]] std::int64_t gdn_capture_bytes() const noexcept;
    [[nodiscard]] bool gdn_mix_observer_armed() const noexcept;
    [[nodiscard]] bool gdn_mix_observer_fired() const noexcept;
    [[nodiscard]] std::int32_t gdn_observer_target_position() const noexcept;
    [[nodiscard]] std::int32_t gdn_observer_target_layer() const noexcept;
    [[nodiscard]] std::int32_t gdn_observer_gidx() const noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;
    friend struct Issue55MatchedForceInspector;
    friend struct Issue55SelectedCommitInspector; // test-only, no decision API

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
