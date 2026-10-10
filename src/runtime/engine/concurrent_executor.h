#include "decision_execution.h"
#pragma once

#include "ninfer/targets/qwen3_6/runtime.h"

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/decision_routing.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/issue58_deferral_reason.h"
#include "runtime/engine/issue58_lane_evidence.h"
#include "runtime/engine/issue58_queue_timing.h"
#include "runtime/engine/request_memory.h"
#include "runtime/engine/decision_execution.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, DeviceContext& device, const EngineOptions& options)
        : instance_(instance), device_(device), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          admission_capacity_(instance.program->admission_capacity()) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_ = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    class DecisionSubmission {
    public:
        DecisionSubmission() noexcept = default;

        ~DecisionSubmission() { reset(); }

        DecisionSubmission(DecisionSubmission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)),
              request_(std::move(other.request_)) {}

        DecisionSubmission& operator=(DecisionSubmission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        DecisionSubmission(const DecisionSubmission&)            = delete;
        DecisionSubmission& operator=(const DecisionSubmission&) = delete;

        DecisionResult wait(const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent decision submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_decision(
                std::exchange(request_, nullptr),
                cancellation);
        }

    private:
        DecisionSubmission(ConcurrentExecutor& owner,
                           std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
            request->issue58_enqueued = Clock::now();
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    DecisionSubmission
    submit_decision(targets::qwen3_6::PreparedPrompt prompt,
                    PromptSummary prompt_summary,
                    double prepare_seconds,
                    ResolvedRequestOptions options,
                    DecisionExecutionProgram program,
                    Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();

        if (pending_deadline == Clock::time_point{}) {
            pending_deadline =
                submitted + pending_timeout_;
        }

        if (submitted >= pending_deadline) {
            throw RequestError(
                RequestErrorKind::QueueTimeout,
                "decision request expired before submission");
        }

        std::uint64_t request_id = 0;

        {
            std::lock_guard lock(
                queue_mutex_);

            if (stopping_ || failed_) {
                throw RequestError(
                    RequestErrorKind::Unavailable,
                    "inference engine is unavailable");
            }

            if (outstanding_ >=
                max_outstanding_) {

                throw RequestError(
                    RequestErrorKind::Overloaded,
                    "inference request queue is full");
            }

            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;

        try {
            auto output =
                instance_.loaded->frontend
                    .make_output_session(
                        prompt,
                        options.stop,
                        options.output);

            request =
                std::make_shared<Request>(
                    request_id,
                    std::move(prompt),
                    std::move(output),
                    prompt_summary,
                    prepare_seconds,
                    std::move(options),
                    pending_deadline,
                    submitted,
                    std::move(program));

        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(
                queue_mutex_);

            if (stopping_ || failed_) {
                --outstanding_;

                throw RequestError(
                    RequestErrorKind::Unavailable,
                    "inference engine is unavailable");
            }

            pending_.push_back(
                request);
            request->issue58_enqueued = Clock::now();
        }

        queue_cv_.notify_one();

        return DecisionSubmission(
            *this,
            std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

private:
    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    DecisionResult wait_for_decision(std::shared_ptr<Request> request,
                                     const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;

        for (;;) {
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(
                    lock,
                    std::chrono::milliseconds(10),
                    [&] { return request->done; });
                done = request->done;
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (!done) { continue; }

            if (caller_error != nullptr) {
                std::rethrow_exception(caller_error);
            }

            std::lock_guard lock(request->mutex);

            if (request->error != nullptr) {
                std::rethrow_exception(request->error);
            }

            return std::move(request->decision_result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time,
                DecisionExecutionProgram finite_program = {})
            : id(request_identity), prompt(std::move(input)), output(std::move(output_session)),
              prompt_summary(summary), prepare_seconds(frontend_seconds),
              options(std::move(request_options)), deadline(limit), submitted(submit_time),
              decision_program(std::move(finite_program)) {}

        [[nodiscard]] bool is_decision() const noexcept {
            return !decision_program.empty();
        }

        const std::uint64_t id;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        Clock::time_point issue58_enqueued{};
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;

        DecisionExecutionProgram decision_program;
        DecisionResult decision_result;
        std::size_t reasoning_close_index = 0;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        bool issue58_long_at_admission = false;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        if (request->is_decision()) {
            complete_error(
                request,
                std::make_exception_ptr(
                    RequestError(RequestErrorKind::Cancelled,
                                 "decision request cancelled")));
            return;
        }

        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void complete_decision_success(const std::shared_ptr<Request>& request,
                                   DecisionResult result) {
        release_planning_state(request);
        request->prompt = {};

        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->decision_result = std::move(result);
            request->done            = true;
        }

        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    bool run_decision_request(const std::shared_ptr<Request>& request) {
        if (!request->lane) {
            throw std::logic_error(
                "decision request has no lane");
        }

        const std::uint32_t lane =
            *request->lane;

        DecisionResult result;

        result.prompt =
            request->prompt_summary;

        result.prepare_seconds =
            request->prepare_seconds;

        if (request->begin) {
            result.reused_prompt_tokens =
                request->begin->
                    reused_prompt_tokens;

            result.prefix_reuse_path =
                request->begin->
                    prefix_reuse_path;
        }

        result.fields.reserve(
            request->decision_program
                .nodes.size());

        const auto selected_field =
            [&](std::size_t node_index)
                -> const DecisionExecutionVariant& {

            const DecisionExecutionNode& node =
                request->decision_program
                    .nodes[node_index];

            std::size_t variant_index = 0;

            if (node.parent_result_index) {
                const std::size_t parent_index =
                    *node.parent_result_index;

                if (parent_index >=
                    result.fields.size()) {

                    throw std::logic_error(
                        "decision dependency parent result is unavailable");
                }

                const std::int32_t parent_winner =
                    result.fields[parent_index]
                        .winner_index;

                if (parent_winner < 0) {
                    throw std::logic_error(
                        "decision dependency parent has no valid winner");
                }

                variant_index =
                    static_cast<std::size_t>(
                        parent_winner);

                if (variant_index >=
                    node.variants.size()) {

                    throw std::logic_error(
                        "decision dependency winner is outside the compiled child variants");
                }
            }

            return node.variants[
                variant_index];
        };

        const auto append_result =
            [&](const DecisionFieldSpec& field,
                auto probe,
                std::uint32_t executed_suffix_tokens) {

            DecisionFieldResult field_result;

            field_result.name =
                field.name;

            field_result.type =
                field.type;

            field_result.candidate_values =
                field.candidate_values;

            field_result.candidate_tokens =
                field.candidate_tokens;

            field_result.routing_probabilities =
                std::move(
                    probe.probabilities);

            field_result.winner_index =
                probe.winner_index;

            field_result.winner_token =
                probe.winner_token;

            if (!field_result
                     .candidate_values.empty()) {

                if (field_result
                        .candidate_values.size() !=
                    field_result
                        .candidate_tokens.size()) {

                    throw std::logic_error(
                        "decision candidate value/token metadata size mismatch");
                }

                if (field_result
                        .winner_index < 0 ||
                    static_cast<std::size_t>(
                        field_result
                            .winner_index) >=
                        field_result
                            .candidate_values.size()) {

                    throw std::logic_error(
                        "decision winner index is outside candidate metadata");
                }

                field_result.selected_value =
                    field_result
                        .candidate_values[
                            static_cast<
                                std::size_t>(
                                    field_result
                                        .winner_index)];
            }

            field_result.frontier =
                probe.frontier;

            field_result.suffix_tokens =
                probe.suffix_tokens;

            field_result.executed_suffix_tokens =
                executed_suffix_tokens;

            field_result.capture_seconds =
                probe.capture_seconds;

            field_result.suffix_seconds =
                probe.suffix_seconds;

            field_result.score_seconds =
                probe.score_seconds;

            field_result.restore_seconds =
                probe.restore_seconds;

            result.fields.push_back(
                std::move(
                    field_result));
        };

        const auto append_trie_result =
            [&](const DecisionExecutionVariant& field)
                -> bool {

            if (!field.trie_plan.has_value()) {
                throw std::logic_error(
                    "trie decision execution requested without a trie plan");
            }

            const DecisionTriePlan& trie =
                *field.trie_plan;

            const std::size_t candidate_count =
                trie.candidate_token_paths.size();

            if (candidate_count < 2 ||
                candidate_count >
                    static_cast<std::size_t>(
                        std::numeric_limits<
                            std::int32_t>::max()) ||
                field.candidate_values.size() !=
                    candidate_count) {

                throw std::logic_error(
                    "trie decision metadata is inconsistent");
            }

            DecisionFieldResult field_result;

            field_result.name =
                field.name;

            field_result.type =
                field.type;

            field_result.candidate_values =
                field.candidate_values;

            field_result.candidate_token_paths =
                trie.candidate_token_paths;

            // There is no truthful singular token for a general
            // multi-token semantic candidate.
            field_result.winner_token = -1;

            std::vector<double> probability_products(
                candidate_count,
                1.0);

            bool frontier_initialized = false;

            std::uint64_t executed_suffix_tokens = 0;

            for (const DecisionTrieProbe& trie_probe :
                 trie.probes) {

                // Each D1 probe restores the retained frontier before
                // returning, so cancellation is safe between probes.
                if (request->cancelled.load(
                        std::memory_order_acquire)) {

                    return false;
                }

                auto scored =
                    instance_.program->
                        decision_probe_lane(
                            lane,
                            trie_probe.suffix_tokens,
                            trie_probe.candidate_tokens);

                if (scored.probabilities.size() !=
                    trie_probe.candidate_tokens.size()) {

                    throw std::logic_error(
                        "trie probe returned the wrong probability count");
                }

                if (trie_probe
                        .descendant_candidate_indices
                        .size() !=
                    trie_probe
                        .candidate_tokens
                        .size()) {

                    throw std::logic_error(
                        "trie probe descendant mapping is inconsistent");
                }

                if (scored.suffix_tokens !=
                    trie_probe.suffix_tokens.size()) {

                    throw std::logic_error(
                        "trie probe reported an unexpected suffix length");
                }

                if (!frontier_initialized) {
                    field_result.frontier =
                        scored.frontier;

                    frontier_initialized = true;

                } else if (
                    field_result.frontier !=
                    scored.frontier) {

                    throw std::logic_error(
                        "trie probes disagreed on retained frontier");
                }

                field_result.suffix_tokens =
                    std::max(
                        field_result.suffix_tokens,
                        scored.suffix_tokens);

                executed_suffix_tokens +=
                    static_cast<std::uint64_t>(
                        scored.suffix_tokens);

                field_result.capture_seconds +=
                    scored.capture_seconds;

                field_result.suffix_seconds +=
                    scored.suffix_seconds;

                field_result.score_seconds +=
                    scored.score_seconds;

                field_result.restore_seconds +=
                    scored.restore_seconds;

                apply_decision_routing_branch(
                    probability_products,
                    scored.probabilities,
                    trie_probe
                        .descendant_candidate_indices);

                consume_service_work(
                    request,
                    static_cast<std::uint64_t>(
                        scored.suffix_tokens));

                if (request->cancelled.load(
                        std::memory_order_acquire)) {

                    return false;
                }
            }

            if (!frontier_initialized) {
                throw std::logic_error(
                    "trie decision executed no ambiguity probes");
            }

            if (executed_suffix_tokens >
                static_cast<std::uint64_t>(
                    std::numeric_limits<
                        std::uint32_t>::max())) {

                throw std::overflow_error(
                    "trie executed suffix accounting overflow");
            }

            field_result.executed_suffix_tokens =
                static_cast<std::uint32_t>(
                    executed_suffix_tokens);

            DecisionRoutingFinal routing =
                finalize_decision_routing(
                    probability_products);

            field_result.routing_probabilities =
                std::move(
                    routing.routing_probabilities);

            field_result.winner_index =
                routing.winner_index;

            field_result.selected_value =
                field_result
                    .candidate_values[
                        static_cast<std::size_t>(
                            routing.winner_index)];

            result.fields.push_back(
                std::move(
                    field_result));

            return true;
        };

        try {
            std::size_t node_index = 0;

            while (node_index <
                   request->decision_program
                       .nodes.size()) {

                if (request->cancelled.load(
                        std::memory_order_acquire)) {

                    complete_cancelled(
                        request);

                    return true;
                }

                const DecisionExecutionNode& node =
                    request->decision_program
                        .nodes[node_index];

                bool used_shared_wave = false;

                if (node.parent_result_index) {
                    const std::size_t parent_index =
                        *node.parent_result_index;

                    std::size_t group_end =
                        node_index + 1;

                    while (
                        group_end <
                            request->decision_program
                                .nodes.size()) {

                        const DecisionExecutionNode&
                            candidate_node =
                                request->decision_program
                                    .nodes[group_end];

                        if (!candidate_node
                                 .parent_result_index ||
                            *candidate_node
                                 .parent_result_index !=
                                parent_index) {

                            break;
                        }

                        ++group_end;
                    }

                    const std::size_t sibling_count =
                        group_end - node_index;

                    bool sibling_group_has_trie =
                        false;

                    for (std::size_t index =
                             node_index;
                         index < group_end;
                         ++index) {

                        if (selected_field(index)
                                .trie_plan
                                .has_value()) {

                            sibling_group_has_trie =
                                true;
                            break;
                        }
                    }

                    if (sibling_count >= 2 &&
                        !sibling_group_has_trie) {
                        std::vector<
                            const DecisionFieldSpec*>
                            selected;

                        selected.reserve(
                            sibling_count);

                        for (std::size_t index =
                                 node_index;
                             index < group_end;
                             ++index) {

                            selected.push_back(
                                &selected_field(
                                    index));
                        }

                        std::size_t common =
                            selected.front()
                                ->suffix_tokens.size();

                        for (std::size_t field_index = 1;
                             field_index <
                                 selected.size();
                             ++field_index) {

                            common =
                                std::min(
                                    common,
                                    selected[field_index]
                                        ->suffix_tokens
                                        .size());

                            std::size_t matched = 0;

                            while (
                                matched < common &&
                                selected.front()
                                        ->suffix_tokens[
                                            matched] ==
                                    selected[field_index]
                                        ->suffix_tokens[
                                            matched]) {

                                ++matched;
                            }

                            common = matched;
                        }

                        bool residuals_nonempty =
                            common != 0;

                        if (residuals_nonempty) {
                            for (const auto* field :
                                 selected) {

                                if (field
                                        ->suffix_tokens
                                        .size() <= common) {

                                    residuals_nonempty =
                                        false;
                                    break;
                                }
                            }
                        }

                        // Materialization is profitable exactly when at least
                        // two siblings share one or more deterministic tokens.
                        // Scheduler service accounting remains conservatively
                        // replay-equivalent in V2-C2.
                        if (residuals_nonempty) {
                            std::vector<
                                targets::qwen3_6::
                                    DecisionWaveProbeSpec>
                                probes;

                            probes.reserve(
                                selected.size());

                            for (const auto* field :
                                 selected) {

                                probes.push_back(
                                    targets::qwen3_6::
                                        DecisionWaveProbeSpec{
                                            std::span<
                                                const TokenId>(
                                                field
                                                    ->suffix_tokens
                                                    .data() +
                                                    common,
                                                field
                                                    ->suffix_tokens
                                                    .size() -
                                                    common),
                                            std::span<
                                                const TokenId>(
                                                field
                                                    ->candidate_tokens
                                                    .data(),
                                                field
                                                    ->candidate_tokens
                                                    .size()),
                                        });
                            }

                            const auto shared_prefix =
                                std::span<
                                    const TokenId>(
                                    selected.front()
                                        ->suffix_tokens
                                        .data(),
                                    common);

                            auto wave =
                                instance_.program->
                                    decision_probe_wave_lane(
                                        lane,
                                        shared_prefix,
                                        probes);

                            if (wave.probes.size() !=
                                selected.size()) {

                                throw std::logic_error(
                                    "shared decision wave returned the wrong probe count");
                            }

                            // Shared transaction timing is charged exactly
                            // once across the public per-field timings.
                            wave.probes.front()
                                .capture_seconds +=
                                    wave.capture_seconds;

                            wave.probes.front()
                                .suffix_seconds +=
                                    wave.shared_prefix_seconds;

                            wave.probes.back()
                                .restore_seconds +=
                                    wave.restore_seconds;

                            // The target has fully restored the retained
                            // frontier before returning. A cancellation which
                            // arrived during the atomic wave is therefore safe
                            // to honor here.
                            if (request->cancelled.load(
                                    std::memory_order_acquire)) {

                                complete_cancelled(
                                    request);

                                return true;
                            }

                            for (std::size_t field_index = 0;
                                 field_index <
                                     selected.size();
                                 ++field_index) {

                                const std::size_t
                                    logical_suffix_tokens =
                                        selected[
                                            field_index]
                                            ->suffix_tokens
                                            .size();

                                const std::size_t
                                    executed_suffix_tokens =
                                        field_index == 0
                                            ? logical_suffix_tokens
                                            : logical_suffix_tokens -
                                                  common;

                                append_result(
                                    *selected[
                                        field_index],
                                    std::move(
                                        wave.probes[
                                            field_index]),
                                    static_cast<
                                        std::uint32_t>(
                                            executed_suffix_tokens));

                                // Keep scheduler/service accounting
                                // replay-equivalent for V2-C2. Actual target
                                // work is lower; planner reduction is a later
                                // optimization.
                                consume_service_work(
                                    request,
                                    static_cast<
                                        std::uint64_t>(
                                            selected[
                                                field_index]
                                                ->suffix_tokens
                                                .size()));
                            }

                            node_index =
                                group_end;

                            used_shared_wave =
                                true;
                        }
                    }
                }

                if (used_shared_wave) {
                    continue;
                }

                const DecisionExecutionVariant& field =
                    selected_field(
                        node_index);

                if (field.trie_plan.has_value()) {
                    if (!append_trie_result(
                            field)) {

                        complete_cancelled(
                            request);

                        return true;
                    }

                    ++node_index;
                    continue;
                }

                auto probe =
                    instance_.program->
                        decision_probe_lane(
                            lane,
                            field.suffix_tokens,
                            field.candidate_tokens);

                append_result(
                    field,
                    std::move(probe),
                    static_cast<std::uint32_t>(
                        field.suffix_tokens.size()));

                consume_service_work(
                    request,
                    static_cast<std::uint64_t>(
                        field.suffix_tokens.size()));

                ++node_index;
            }

            result.total_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(
                    Clock::now() -
                    request->submitted)
                    .count();

            complete_decision_success(
                request,
                std::move(result));

            return true;

        } catch (...) {
            // Decision probes normally restore the retained frontier before
            // returning. If any decision execution path nevertheless throws,
            // discard the lane rather than allowing a potentially partial
            // temporary frontier/state snapshot to participate in later
            // prefix reuse.
            instance_.program->abort_lane(
                lane);

            complete_error(
                request,
                std::current_exception());

            return true;
        }
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            runtime::RoundBudget round_budget = request->budget->round_budget();

            if (request->output.in_reasoning()) {
                const auto reasoning_budget = request->output.reasoning_budget();
                if (reasoning_budget) {
                    const std::uint32_t used = request->output.reasoning_tokens();

                    if (used < *reasoning_budget) {
                        round_budget.generated_tokens_remaining =
                            std::min(round_budget.generated_tokens_remaining,
                                     *reasoning_budget - used);
                    } else {
                        const auto close_tokens = request->output.reasoning_close_tokens();
                        if (request->reasoning_close_index >= close_tokens.size()) {
                            throw std::logic_error(
                                "reasoning close marker was exhausted while still reasoning");
                        }

                        round_budget.generated_tokens_remaining =
                            std::min<std::uint32_t>(
                                round_budget.generated_tokens_remaining, 1U);
                        round_budget.forced_token =
                            close_tokens[request->reasoning_close_index];
                    }
                }
            }

            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = round_budget;
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        consume_service_work(request, 1);
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            instance_.program->abort_lane(lane);
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }

        if (request->is_decision()) {
            const std::uint32_t lane = *request->lane;

            // Terminal resolution retains exactly the executed prompt frontier.
            // The sampled prefill token is intentionally not committed.
            instance_.program->resolve_prefill_lane(lane, true);

            (void)run_decision_request(request);
            remove_completed_slot(lane);
            return;
        }

        if (resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    // Issue #58: optional non-preemptive latency isolation. Do not let two large
    // generations occupy both C2 slots; preserve one slot for shorter work.
    // This avoids KV/MTP checkpointing and never aborts an admitted generation.
    [[nodiscard]] bool reserve_short_lane_enabled() const noexcept {
        const char* value = std::getenv("NINFER_SHORT_LANE_RESERVE");
        return max_concurrency_ == 2 && value && value[0] == '1' && value[1] == '\0';
    }

    [[nodiscard]] static bool is_long_generation(const RequestPlanSummary& plan) noexcept {
        // A smaller threshold may be selected for bounded fairness tests.
        std::uint32_t threshold = 8192;
        if (const char* value = std::getenv("NINFER_LONG_OUTPUT_THRESHOLD")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0' && parsed >= 256 && parsed <= 65536) {
                threshold = static_cast<std::uint32_t>(parsed);
            }
        }
        return plan.effective_output_tokens > threshold;
    }

    [[nodiscard]] static bool trusted_short_hint(const Request& request) noexcept {
        const char* flag = std::getenv("NINFER_TRUST_SHORT_OPERATION_HINT");
        return flag && flag[0] == '1' && flag[1] == '\0' &&
               request.options.execution.ninfer_short_operation;
    }

    [[nodiscard]] static bool classified_long(const Request& request,
                                              const RequestPlanSummary& summary) noexcept {
        // Explicit trusted hint affects scheduler classification only. It does
        // not change actual token limits, reservation, model or KV accounting.
        return !trusted_short_hint(request) && is_long_generation(summary);
    }

    [[nodiscard]] bool long_lane_guard(const std::shared_ptr<Request>& candidate) const {
        if (!reserve_short_lane_enabled() || !candidate->base_plan ||
            !classified_long(*candidate, candidate->base_plan->summary())) { return false; }
        for (const auto& active : slots_) {
            if (active && active->issue58_long_at_admission) { return true; }
        }
        return false;
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request,
                        std::array<Issue58LaneEvidence, kMaximumConcurrency>* evidence = nullptr) {
        // The reservation must be enforced on the shared admission path,
        // before either FIFO or protected-backfill can choose a lane.
        if (evidence) { evidence->fill(Issue58LaneEvidence::PlanUnavailable); }
        if (long_lane_guard(request)) { return std::nullopt; }
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                if (evidence) { (*evidence)[lane] = Issue58LaneEvidence::Occupied; }
                continue;
            }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const bool direct_fit = instance_.program->can_admit_lane(lane, plan);
            if (evidence) { (*evidence)[lane] = issue58_lane_evidence(false, true, direct_fit, false); }
            if (direct_fit && (!selected || reuse > selected_reuse)) {
                selected       = LaneChoice{.lane = lane};
                selected_reuse = reuse;
            }
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const bool eviction_fit = instance_.program->can_admit_lane_after_retained_eviction(lane, plan);
            if (evidence) {
                (*evidence)[lane] = issue58_lane_evidence(false, true, false, eviction_fit);
            }
            if (eviction_fit && (!selected || reuse > selected_reuse)) {
                selected = LaneChoice{
                    .lane           = lane,
                    .evict_retained = true,
                };
                selected_reuse = reuse;
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        if (choice.evict_retained) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    instance_.program->evict_retained_lane(retained_lane);
                    invalidate_lane_plans(retained_lane);
                }
            }
            if (!instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        // Measured from completed queue insertion to removal for selected admission.
        // This is not GPU-start latency, TTFT or a wall-clock admission timestamp.
        if (const char* trace = std::getenv("NINFER_ADMISSION_TRACE");
            trace && trace[0] == '1' && trace[1] == '\0') {
            const auto dequeue_time = Clock::now();
            std::fprintf(stderr,
                "[ADMISSION-QUEUE-TIMING] id=%llu lane=%u queue_wait_ms=%lld "
                "event=selected_for_admission\n",
                static_cast<unsigned long long>(request->id), lane,
                static_cast<long long>(issue58_queue_wait_ms<Clock>(
                    request->issue58_enqueued, dequeue_time)));
        }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        clear_protection_if_head(request);

        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->issue58_long_at_admission = classified_long(*request, summary);
            if (const char* trace = std::getenv("NINFER_ADMISSION_CLASS_TRACE");
                trace && trace[0] == '1' && trace[1] == '\0') {
                // Classification observability only. No prompt, output, or identity logged.
                std::fprintf(stderr,
                    "[ADMISSION-CLASS] id=%llu effective_max_output=%u long=%u lane=%u\n",
                    static_cast<unsigned long long>(request->id),
                    static_cast<unsigned>(summary.effective_output_tokens),
                    static_cast<unsigned>(request->issue58_long_at_admission), lane);
            }
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            target_started                = true;
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient);
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::array<Issue58LaneEvidence, kMaximumConcurrency> head_lane_evidence{};
            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head, &head_lane_evidence);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            // Policy-blocked long FIFO head: admit one waiting short request on the
            // vacant lane instead of letting the protection/drain logic strand it.
            // Physical lane and KV admission checks still run in find_admission_lane.
            if (long_lane_guard(head)) {
                for (std::size_t i = 1; i < queued.size(); ++i) {
                    const auto& candidate = queued[i];
                    if (candidate->cancelled.load(std::memory_order_acquire) ||
                        Clock::now() >= candidate->deadline) { continue; }
                    try {
                        ensure_base_plan(candidate);
                        if (!classified_long(*candidate, candidate->base_plan->summary())) {
                            if (auto lane = find_admission_lane(candidate)) {
                                return admit_planned_request(candidate, *lane,
                                                             BackfillClass::None, 0);
                            }
                        }
                    } catch (...) {
                        (void)remove_pending_error(candidate, std::current_exception());
                        control_progress = true;
                    }
                }
                // Deliberate policy hold: physical admission is possible, so
                // the frozen-incumbent protection invariant does not apply.
                // Recheck on each completed GPU unit; do not enter Drain.
                // Observational only: policy guard blocks this FIFO head.
                if (const char* trace = std::getenv("NINFER_ADMISSION_TRACE");
                    trace && trace[0] == '1' && trace[1] == '\0') {
                    const auto now = Clock::now();
                    if (last_admission_trace_ == Clock::time_point{} ||
                        now - last_admission_trace_ >= std::chrono::seconds(10)) {
                        last_admission_trace_ = now;
                        std::fprintf(stderr, "[ADMISSION-DEFERRAL] head=%llu queued=%zu reason=%s\n",
                            static_cast<unsigned long long>(head->id), queued.size(),
                            issue58_deferral_reason_name(issue58_deferral_reason(true, false)));
                    }
                }
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const ActiveAdmissionSet active = active_admission_set();
            // Issue #58: bounded opt-in admission trace; never alters allocation or scheduling.
            // One snapshot per 10s while a FIFO head is blocked. Enable only for a
            // controlled diagnostic process: NINFER_ADMISSION_TRACE=1.
            if (const char* trace = std::getenv("NINFER_ADMISSION_TRACE");
                trace != nullptr && trace[0] == '1') {
                const auto now = Clock::now();
                if (last_admission_trace_ == Clock::time_point{} ||
                    now - last_admission_trace_ >= std::chrono::seconds(10)) {
                    last_admission_trace_ = now;
                    std::uint64_t used_main = 0, used_backend = 0, used_lanes = 0;
                    for (const ActiveAdmissionSnapshot& a : active.span()) {
                        used_main += a.resources.main_kv_pages;
                        used_backend += a.resources.backend_kv_pages;
                        used_lanes += a.resources.active_lanes;
                    }
                    const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        head->deadline - now).count();
                    bool has_vacant_lane = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        if (slots_[lane] == nullptr) { has_vacant_lane = true; break; }
                    }
                    const char* deferral_reason = issue58_deferral_reason_name(
                        issue58_deferral_reason(false, has_vacant_lane));
                    std::fprintf(stderr,
                        "[ADMISSION-TRACE] head=%llu queued=%zu active=%zu reason=%s "
                        "head_pages_main=%llu head_pages_backend=%llu "
                        "used_pages_main=%llu used_pages_backend=%llu used_lanes=%llu "
                        "capacity_pages_main=%llu capacity_pages_backend=%llu capacity_lanes=%llu "
                        "deadline_remaining_ms=%lld protection_epoch=%llu protection_phase=%s\n",
                        static_cast<unsigned long long>(head->id), queued.size(), active.size,
                        deferral_reason,
                        static_cast<unsigned long long>(head_base.admission.main_kv_pages),
                        static_cast<unsigned long long>(head_base.admission.backend_kv_pages),
                        static_cast<unsigned long long>(used_main),
                        static_cast<unsigned long long>(used_backend),
                        static_cast<unsigned long long>(used_lanes),
                        static_cast<unsigned long long>(admission_capacity_.main_kv_pages),
                        static_cast<unsigned long long>(admission_capacity_.backend_kv_pages),
                        static_cast<unsigned long long>(admission_capacity_.active_lanes),
                        static_cast<long long>(remaining_ms),
                        static_cast<unsigned long long>(protection_ ? protection_->epoch_id : 0),
                        !protection_ ? "none" :
                            protection_->phase == ProtectionPhase::Drain ? "drain" : "open");
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        std::fprintf(stderr, "[ADMISSION-LANE] head=%llu lane=%u result=%s\n",
                            static_cast<unsigned long long>(head->id), lane,
                            issue58_lane_evidence_name(head_lane_evidence[lane]));
                    }
                }
            }
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    Clock::time_point last_admission_trace_{};

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const bool forced_reasoning_close =
                membership.budgets[row].forced_token >= 0;

            const OutputDecision decision = request->output.preview(
                row_tokens, request->budget->remaining(), request->budget->limit_reason(),
                !forced_reasoning_close);
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count)) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();

            if (!cancelled[row] && membership.budgets[row].forced_token >= 0 &&
                accepted[row] == 1) {
                ++request->reasoning_close_index;
            }

            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published));
            if (terminal[row]) {
                complete_success(request, finish_reasons[row]);
                remove_completed_slot(lane);
            }
        }
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        publish_runtime_stats();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                if (prefill_lane_) {
                    if (!membership.empty() && !previous_unit_was_decode) {
                        run_decode_round(membership);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
            } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const AdmissionResources admission_capacity_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
