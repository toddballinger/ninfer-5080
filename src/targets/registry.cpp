#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/launcher/speculative_round.h"
#include "ops/launcher/scalar.h"
#include "ops/launcher/gqa_attention.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/common/int8_proj_launch.h"
#include "ops/linear/q4/q4_launch.h"
#include "runtime/engine/kv_capacity.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    // Capture the host-resident token-embedding bytes (nonzero only under --embed-cpu) from the
    // host-mapped materialization, before the load plan is moved into the loaded model.
    std::uint64_t embed_host_weight_bytes = 0;
    if (options.embed_cpu) {
        const auto& objects = reader.objects();
        for (const auto& placement : load_plan.materialization().host_mapped_objects) {
            if (object_name(objects.at(placement.object.index)) == "text/token_embedding") {
                embed_host_weight_bytes = placement.bytes;
            }
        }
    }

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();

    // Account for the Qwen3.8 Q4/Q5 small-T CUDA kernel residency before
    // resolving the explicit KV/runtime capacity. Without this, lazy module
    // loading can consume the final few MiB only after startup is complete.
    if (target_key == Qwen3_6_27B::qwen3_8_target_key) {
        const auto mem_diag = [](const char* label) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::fprintf(stderr,
                         "[PREWARM-MEM] %-32s %12zu B  %9.4f MiB free\n",
                         label,
                         free_bytes,
                         static_cast<double>(free_bytes) / (1024.0 * 1024.0));
        };

        device.synchronize();
        mem_diag("before prewarm");

        ops::detail::q4_q5_attn_input_small_t_prewarm();
        device.synchronize();
        mem_diag("after q4q5 attention");

        ops::detail::q4_q5_gdn_input_independent_prewarm();
        device.synchronize();
        mem_diag("after q4q5 gdn");

        ops::detail::speculative_round_prewarm();
        device.synchronize();
        mem_diag("after speculative");

        ops::detail::scalar_prewarm();
        device.synchronize();
        mem_diag("after scalar");

        ops::detail::q4_small_t_mma_prewarm();
        device.synchronize();
        mem_diag("after q4 small-t linear");

        ops::detail::q5_rowsplit_mma_prewarm();
        device.synchronize();
        mem_diag("after q5 rowsplit mma");

        ops::detail::gqa_q4_prefill_prewarm();
        device.synchronize();
        mem_diag("after gqa q4 prefill");

        ops::detail::int8_proj_prewarm();
        device.synchronize();
        mem_diag("after int8 projection");

        ops::detail::q4_rowsplit_mma_prewarm();
        device.synchronize();
        mem_diag("after q4 rowsplit mma");
    }

    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.embed_host_weight_bytes = embed_host_weight_bytes;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_5_9B::LoadedQwen3_5_9B(std::unique_ptr<Qwen3_5_9B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_5_9B::make_frontend(*model, options)) {}

LoadedQwen3_5_9B::~LoadedQwen3_5_9B() = default;

Qwen3_5_9BInstance::Qwen3_5_9BInstance(std::unique_ptr<LoadedQwen3_5_9B> stable_loaded,
                                       runtime::KvCapacityResolution resolution,
                                       Qwen3_5_9B::SequencePlan sequence_plan,
                                       DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_5_9B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_5_9BInstance::~Qwen3_5_9BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_5_9B::model_id) {
        return construct_registered<Qwen3_5_9B, LoadedQwen3_5_9B, Qwen3_5_9BInstance>(
            options, device, reader, load_start, Qwen3_5_9B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
