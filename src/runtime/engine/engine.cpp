#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/concurrent_executor.h"
#include "runtime/engine/engine_boundary_observer_registry.h"
#include "targets/registry.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {
struct ObserverState {
    std::mutex mutex;
    BoundaryObserverRegistry::Token next_token = 1;
    BoundaryObserverRegistry::Token active_token = 0;
    std::shared_ptr<const BoundaryObserverRegistry::Callback> callback;
};

ObserverState& observer_state() {
    static ObserverState state;
    return state;
}
} // namespace

BoundaryObserverRegistry::Token BoundaryObserverRegistry::register_observer(Callback callback) {
    if (!callback) { throw std::invalid_argument("boundary observer must not be empty"); }
    auto owned = std::make_shared<const Callback>(std::move(callback));
    auto& state = observer_state();
    std::lock_guard lock(state.mutex);
    if (state.active_token != 0) {
        throw std::invalid_argument("boundary observer already registered");
    }
    if (state.next_token == 0) { throw std::overflow_error("boundary observer tokens exhausted"); }
    const Token token = state.next_token++;
    state.callback = std::move(owned);
    state.active_token = token;
    return token;
}

std::shared_ptr<const BoundaryObserverRegistry::Callback> BoundaryObserverRegistry::snapshot() {
    auto& state = observer_state();
    std::lock_guard lock(state.mutex);
    return state.callback;
}

void BoundaryObserverRegistry::unregister(Token token) {
    auto& state = observer_state();
    std::lock_guard lock(state.mutex);
    if (token == 0 || token != state.active_token) {
        throw std::invalid_argument("boundary observer token does not match active registration");
    }
    state.callback.reset();
    state.active_token = 0;
}

namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

// Installs the process-local M1 boundary-seam test observer (if one is
// registered) onto an Engine-owned executor; no ownership transfer, no
// public API, and no scheduler change: when nothing is registered the
// seam remains inert (null observer => the seam emits nothing).
void install_boundary_observer(
    std::function<void(const runtime::BoundaryObservation&)>& executor_observer) {
    auto callback = BoundaryObserverRegistry::snapshot();
    if (callback) {
        // The executor owns this shared snapshot independently of unregister().
        executor_observer = [callback = std::move(callback)](
                                const runtime::BoundaryObservation& o) { (*callback)(o); };
    }
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using Executor9  = runtime::ConcurrentExecutor<targets::Qwen3_5_9BInstance>;
    using Executor27 = runtime::ConcurrentExecutor<targets::Qwen3_6_27BInstance>;
    using Executor35 = runtime::ConcurrentExecutor<targets::Qwen3_6_35BA3BInstance>;
    using Executor =
        std::variant<std::monostate, std::unique_ptr<Executor9>, std::unique_ptr<Executor27>,
                     std::unique_ptr<Executor35>>;

    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        auto constructed  = targets::construct_target(options, device);
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        executor          = std::visit(
            [&](auto& target_ptr) -> Executor {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                // M1: install the private boundary-seam test observer (no-op when
                // unregistered) on the Engine's own executor before any request can
                // be submitted; the executor's worker then emits the post-commit
                // seam records directly into it.
                std::function<void(const runtime::BoundaryObservation&)> observer;
                install_boundary_observer(observer);
                if constexpr (std::is_same_v<Instance, targets::Qwen3_5_9BInstance>) {
                    auto executor = std::make_unique<Executor9>(*target_ptr, options);
                    executor->boundary_observer(std::move(observer));
                    return executor;
                } else if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                    auto executor = std::make_unique<Executor27>(*target_ptr, options);
                    executor->boundary_observer(std::move(observer));
                    return executor;
                } else {
                    auto executor = std::make_unique<Executor35>(*target_ptr, options);
                    executor->boundary_observer(std::move(observer));
                    return executor;
                }
            },
            active);
    }

    ~Impl() noexcept {
        executor.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Executor executor;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;

            GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& executor) -> GenerationHandle {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                auto submission = executor->submit(std::move(prompt.impl_->value), prompt_summary,
                                                   prepare_seconds, std::move(resolved_options),
                                                   pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->executor);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    return submit(std::move(prompt), std::move(options)).wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> MemorySummary {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->memory_summary();
            }
        },
        impl_->executor);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> RuntimeStats {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->runtime_stats();
            }
        },
        impl_->executor);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (!std::is_same_v<Executor, std::monostate>) {
                executor->reset_memory_peaks();
            }
        },
        impl_->executor);
}

} // namespace ninfer
