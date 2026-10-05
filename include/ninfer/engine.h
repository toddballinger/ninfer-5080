#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class DecisionHandle {
public:
    DecisionHandle() noexcept;
    ~DecisionHandle();

    DecisionHandle(DecisionHandle&&) noexcept;
    DecisionHandle& operator=(DecisionHandle&&) noexcept;

    DecisionHandle(const DecisionHandle&)            = delete;
    DecisionHandle& operator=(const DecisionHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;

    DecisionResult wait(const CancellationView& cancellation = {});

private:
    class Impl;
    explicit DecisionHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class CompiledDecisionPlan {
public:
    CompiledDecisionPlan() noexcept;
    ~CompiledDecisionPlan();

    CompiledDecisionPlan(const CompiledDecisionPlan&) noexcept;
    CompiledDecisionPlan& operator=(const CompiledDecisionPlan&) noexcept;

    CompiledDecisionPlan(CompiledDecisionPlan&&) noexcept;
    CompiledDecisionPlan& operator=(CompiledDecisionPlan&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t field_count() const noexcept;

private:
    class Impl;

    explicit CompiledDecisionPlan(
        std::shared_ptr<const Impl> impl) noexcept;

    std::shared_ptr<const Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // Compile model-agnostic decision semantics once against the active
    // target/tokenizer. The resulting plan is immutable, cheap to copy and
    // reusable across requests handled by this exact Engine instance.
    //
    // Current finite-decision execution requires
    // EngineOptions::speculative.backend == SpeculativeBackend::None.
    [[nodiscard]] CompiledDecisionPlan
    compile_decision_plan(
        const StructuredDecisionSchema& schema,
        const DecisionModelPresentation& presentation) const;

    // Execute an already-compiled structured decision plan.
    [[nodiscard]] DecisionHandle
    submit_decision(
        PreparedPrompt prompt,
        const CompiledDecisionPlan& plan,
        std::chrono::steady_clock::time_point pending_deadline = {});

    DecisionResult
    decide(
        PreparedPrompt prompt,
        const CompiledDecisionPlan& plan,
        const CancellationView& cancellation = {});

    // Convenience typed finite-choice path. Boolean fields use canonical
    // false/true candidates; Enum values are both semantic values and
    // model-facing candidate text. Whole-path tokenization may lower a field
    // to a depth-1 choice or a multi-token trie.
    [[nodiscard]] DecisionHandle
    submit_decision(PreparedPrompt prompt, std::vector<DecisionFieldInput> fields,
                    std::chrono::steady_clock::time_point pending_deadline = {});

    DecisionResult decide(PreparedPrompt prompt, std::vector<DecisionFieldInput> fields,
                          const CancellationView& cancellation = {});

    // Raw depth-1 token path retained for parity tests, diagnostics and
    // lower-level callers that already own token IDs. Product code should
    // prefer DecisionFieldInput or a compiled semantic plan.
    [[nodiscard]] DecisionHandle
    submit_decision(PreparedPrompt prompt, std::vector<DecisionFieldSpec> fields,
                    std::chrono::steady_clock::time_point pending_deadline = {});

    DecisionResult decide(PreparedPrompt prompt, std::vector<DecisionFieldSpec> fields,
                          const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;

    // Concrete finite-decision scorer resource envelope for this Engine.
    // This is a workspace bound, not an arbitrary product-level K constant.
    [[nodiscard]] DecisionCapacitySummary decision_capacity() const;

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;

    // White-box fixture hook; not part of the public Engine interface.
    [[nodiscard]] void* bound_model_instance() const;
    friend struct BoundInstanceReader;
};

} // namespace ninfer
