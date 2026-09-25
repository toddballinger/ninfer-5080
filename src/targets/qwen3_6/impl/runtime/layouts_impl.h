#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/swa.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const std::int32_t linear_state_slots =
        LinearStateSlots::state_slot_count(plan.max_concurrency);
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads,
                     .attention_head_dim        = TextConfig::head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .mtp_kv_dtype              = plan.kv_dtype,
                     .mtp_kv_quant_group        = plan.kv_quant_group,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     .linear_attention =
                         {
                             .layers         = TextConfig::gdn_layers(),
                             .conv_channels  = TextConfig::convolution_dim,
                             .conv_width     = TextConfig::gdn_conv_state_width,
                             .value_heads    = TextConfig::gdn_value_heads,
                             .value_head_dim = TextConfig::gdn_value_head_dim,
                             .key_head_dim   = TextConfig::gdn_key_head_dim,
                             .slot_count     = linear_state_slots,
                             .conv_dtype     = DType::BF16,
                         },
                 });
    const auto persistent_diag = [&](const char* stage) {
        const std::size_t bytes = builder.finish(kArenaAlign, stage);
        std::fprintf(stderr,
                     "[PERSIST-DIAG] %-28s %12zu B  %9.4f MiB\n",
                     stage, bytes,
                     static_cast<double>(bytes) / (1024.0 * 1024.0));
    };

    persistent_diag("after decoder");

    if (plan.speculative_backend != SpeculativeBackend::None) {
        const GdnReplayRecordSpec full_replay_spec{
            .layers          = TextConfig::gdn_layers(),
            .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
            .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
            .conv_channels   = TextConfig::convolution_dim,
            .qk_heads        = TextConfig::gdn_key_heads,
            .value_heads     = TextConfig::gdn_value_heads,
            .key_dim         = TextConfig::gdn_key_head_dim,
            .value_dim       = TextConfig::gdn_value_head_dim,
        };

        if (plan.use_cuda_graph) {
            // CUDA graph capture is deliberately single-stream. Keep the
            // complete ReplaySSM transaction on device so graph capture has
            // no dependency on the host-transfer stream.
            out.replay_records =
                plan_gdn_replay_records(builder, full_replay_spec);
        } else {
            // Memory-saving non-graph path: double-buffer two ReplaySSM
            // layers through pinned host storage.
            GdnReplayRecordSpec scratch_spec = full_replay_spec;
            scratch_spec.layers = 2;

            GdnReplayRecordSpec host_slot_spec = scratch_spec;
            host_slot_spec.layers = 1;

            LayoutBuilder replay_host_builder;
            out.replay_host_layout =
                plan_gdn_replay_records(replay_host_builder, host_slot_spec);

            const std::size_t replay_host_layer_stride =
                replay_host_builder.finish(
                    kArenaAlign, "ReplaySSM packed host layer");

            out.replay_host_bytes =
                replay_host_layer_stride *
                static_cast<std::size_t>(TextConfig::gdn_layers());

            out.replay_records =
                plan_gdn_replay_records(builder, scratch_spec);
        }

        std::fprintf(stderr,
                     "[REPLAY-LAYOUT] scratch bytes=%zu "
                     "conv=(%zu,%zu) key=(%zu,%zu) "
                     "value=(%zu,%zu) gate=(%zu,%zu)\\n",
                     out.replay_records->gate.region.offset +
                         out.replay_records->gate.region.bytes -
                         out.replay_records->conv.region.offset,
                     out.replay_records->conv.region.offset,
                     out.replay_records->conv.region.bytes,
                     out.replay_records->key.region.offset,
                     out.replay_records->key.region.bytes,
                     out.replay_records->value.region.offset,
                     out.replay_records->value.region.bytes,
                     out.replay_records->gate.region.offset,
                     out.replay_records->gate.region.bytes);
    }
    persistent_diag("after replay");

    if constexpr (Variant::supports_dflash) {
        if (plan.features.masked_draft()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();

            // Preserve the legacy DFlash BF16-K/BF16-V cyclic cache.
            // DFlash2's 2048 sliding-attention kernel requires BF16-K/FP16-V.
            const DType local_value_dtype =
                plan.speculative_backend == SpeculativeBackend::DFlash2 ? DType::FP16
                                                                        : DType::BF16;

            dflash.local = plan_cyclic_kv_cache(
                builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                static_cast<std::int32_t>(plan.max_concurrency), local_value_dtype);

            if constexpr (DFlashConfig::full_layers != 0) {
                PagedKVPoolSpec full_pool{
                    .page_group_count      = physical_pages,
                    .logical_page_capacity = logical_pages,
                    .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                    .plane_order           = PagedKVPlaneOrder::HeadMajor,
                    .planes =
                        {
                            {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                            {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                        },
                };

                dflash.full = qwen3_6::PagedKVCacheLayout{
                    .pool        = plan_paged_kv_pool(builder, full_pool),
                    .layers      = DFlashConfig::full_layers,
                    .max_context = plan.capacity,
                    .kv_heads    = DFlashConfig::kv_heads,
                    .head_dim    = DFlashConfig::head_dim,
                    .dtype       = DType::BF16,
                    .quant_group = 0,
                };
            }

            dflash.prefill_projected = add_tensor(
                builder, DType::FP32, {DFlashConfig::hidden, effective_prefill_chunk},
                "DFlash prefill projected accumulator");
            dflash.prefill_positions =
                add_tensor(builder, DType::I32, {effective_prefill_chunk},
                           "DFlash prefill target positions");
            dflash.pending_features =
                add_tensor(builder, DType::BF16,
                           {DFlashConfig::feature_rows,
                            static_cast<std::int32_t>(plan.draft_window + 1U),
                            static_cast<std::int32_t>(plan.max_concurrency)},
                           "DFlash pending target features");
        }
    }

    persistent_diag("before round");

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden         = TextConfig::hidden,
                                         .output_rows    = TextConfig::output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend});
    persistent_diag("after round begin");

    // Only the final normalized prefill hidden column must survive the
    // workspace lifetime. Full-chunk normalized hidden now lives in WorkspaceArena.
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, 1}, "step prefill hidden tail");
    persistent_diag("after prefill hidden");

    qwen3_6::complete_round_state_layout(builder, out.round);
    persistent_diag("after round complete");

    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    persistent_diag("after token counts");

    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    persistent_diag("after sampling config");

    out.tail_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "tail hidden");
    persistent_diag("after tail hidden");

    out.rewrite_checkpoint_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "rewrite checkpoint hidden");
    persistent_diag("after checkpoint hidden");

    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

std::uint32_t resolved_vision_token_limit(const SequencePlanImpl& plan) {
    constexpr std::uint32_t kFrontendMergedLimit = 32768;
    std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
    if (plan.vision_max_tokens != 0) { merged = std::min(merged, plan.vision_max_tokens); }
    return merged;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();

        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        if (phase == qwen3_6::TextPhase::Prefill) {
            std::fprintf(stderr,
                "[ATTN-DIAG] projection tensors   %12zu B  %8.4f MiB\n",
                layout.peak_bytes(1), layout.peak_bytes(1) / 1048576.0);
        }

        const std::size_t projection_scratch =
            Variant::attention_projection_workspace_capacity_bytes(
                plan.weights_profile, phase, first, last);
        scratch(layout, projection_scratch);
        if (phase == qwen3_6::TextPhase::Prefill) {
            std::fprintf(stderr,
                "[ATTN-DIAG] projection scratch   %12zu B  %8.4f MiB  raw=%zu\n",
                layout.peak_bytes(1), layout.peak_bytes(1) / 1048576.0,
                projection_scratch);
        }

        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        if (phase == qwen3_6::TextPhase::Prefill) {
            std::fprintf(stderr,
                "[ATTN-DIAG] attention results    %12zu B  %8.4f MiB\n",
                layout.peak_bytes(1), layout.peak_bytes(1) / 1048576.0);
        }

        const std::size_t gqa_scratch =
            ops::gqa_attention_workspace_capacity_bytes(
                TextConfig::query_heads, plan.kv_dtype, envelope,
                batch_size, min_width, max_width);
        scratch(layout, gqa_scratch);
        if (phase == qwen3_6::TextPhase::Prefill) {
            std::fprintf(stderr,
                "[ATTN-DIAG] GQA scratch           %12zu B  %8.4f MiB  raw=%zu\n",
                layout.peak_bytes(1), layout.peak_bytes(1) / 1048576.0,
                gqa_scratch);
        }

        const std::size_t output_scratch =
            Variant::attention_output_projection_workspace_capacity_bytes(
                plan.weights_profile, phase, first, last);
        scratch(layout, output_scratch);
        if (phase == qwen3_6::TextPhase::Prefill) {
            std::fprintf(stderr,
                "[ATTN-DIAG] output proj scratch   %12zu B  %8.4f MiB  raw=%zu\n",
                layout.peak_bytes(1), layout.peak_bytes(1) / 1048576.0,
                output_scratch);
        }
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            // The projected/convolved prefill buffers are only needed through the column
            // extraction into q/k/v. They are dead before the recurrent GDN stage, so model
            // that lifetime explicitly instead of carrying ~35 MiB of chunk-896 temporaries
            // into the recurrent workspace peak.
            auto conv_scope = layout.scope();
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    std::fprintf(stderr,
        "[PREFILL-STAGE] roots             %12zu B  %8.4f MiB\n",
        finish(text_prefill), finish(text_prefill) / 1048576.0);

    attention_stage(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill,
                    1, 1, chunk, text_envelope);
    std::fprintf(stderr,
        "[PREFILL-STAGE] after attention   %12zu B  %8.4f MiB\n",
        finish(text_prefill), finish(text_prefill) / 1048576.0);

    gdn_stage(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill,
              GdnWorkspacePath::Prefill, 1, 1, chunk);
    std::fprintf(stderr,
        "[PREFILL-STAGE] after GDN         %12zu B  %8.4f MiB\n",
        finish(text_prefill), finish(text_prefill) / 1048576.0);

    post_mixer_stage(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill);
    std::fprintf(stderr,
        "[PREFILL-STAGE] after postmixer   %12zu B  %8.4f MiB\n",
        finish(text_prefill), finish(text_prefill) / 1048576.0);

    // Normalized target hidden remains live through sampling/MTP but is
    // workspace-backed. GDN remains the dominant workspace peak.
    matrix(text_prefill, DType::BF16, TextConfig::hidden, chunk);

    scratch(text_prefill,
            ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    std::fprintf(stderr,
        "[PREFILL-STAGE] after sampling    %12zu B  %8.4f MiB\n",
        finish(text_prefill), finish(text_prefill) / 1048576.0);

    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);

        // Full normalized target hidden stays live while MTP consumes it.
        matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);

        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                                    TextConfig::query_heads, plan.kv_dtype, text_envelope, batch,
                                    width, width));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.masked_draft()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t width, std::int32_t batch,
                                                     bool compact_input) {
                const auto tokens = width * batch;
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                if constexpr (DFlashConfig::coherent_selector) {
                    const auto local_width = std::min(width, DFlashConfig::local_capacity);
                    (void)workspace_recipe::dflash_context<DFlashConfig>(layout,
                                                                         local_width * batch);
                    scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                        batch, local_width, local_width));
                    return finish(layout);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                if constexpr (DFlashConfig::coherent_selector) {
                    const auto prepare = [&] {
                        (void)workspace_recipe::dflash2_branch<DFlashConfig>(layout, width, batch);
                        scratch(layout,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    width, width, batch, batch));
                    };
                    {
                        auto attention = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, DFlashConfig::query_size, tokens);
                        matrix(layout, DType::BF16, DFlashConfig::kv_size, tokens);
                        matrix(layout, DType::BF16, DFlashConfig::kv_size, tokens);
                        matrix(layout, DType::BF16, DFlashConfig::query_size, tokens);
                        scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                            {DFlashConfig::head_dim, DFlashConfig::query_heads,
                                             DFlashConfig::kv_heads},
                                            DFlashConfig::local_capacity, {0, plan.capacity}, width,
                                            width, batch));
                        scratch(layout,
                                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                    DFlashConfig::query_size, width, width, batch, batch));
                    }
                    {
                        auto mlp = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, DFlashConfig::intermediate, tokens);
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                            DFlashConfig::hidden, tokens, tokens));
                        scratch(layout,
                                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                    DFlashConfig::intermediate, width, width, batch, batch));
                    }
                    const auto mask_columns = drafts * batch;
                    matrix(layout, DType::BF16, DFlashConfig::hidden, mask_columns);
                    matrix(layout, DType::FP32, 16, mask_columns);
                    if (plan.proposal_head == ProposalHead::Optimized) {
                        scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                            QType::Q4G64_F16S, Variant::draft_head_rows,
                                            DFlashConfig::hidden, batch, batch));
                    } else {
                        // Full proposal heads share the same public input and top-k contract.
                        for (const auto qtype : {QType::W8G32_F16S,
                                                 QType::FP8_E4M3FN_ROW_BF16S,
                                                 QType::Q4G64_F16S}) {
                            scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                                qtype, TextConfig::output_rows,
                                                DFlashConfig::hidden, batch, batch));
                        }
                    }
                    matrix(layout, DType::BF16, 256, mask_columns);
                    return finish(layout);
                }
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes({0, plan.capacity}, width,
                                                                       width, batch),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch)));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::query_size, tokens, tokens));
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                        QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                        DFlashConfig::hidden, tokens, tokens));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::intermediate, tokens, tokens));
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                }
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    DFlashConfig::coherent_selector
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              TextConfig::token_domain, {false}, batch, batch)
                        : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                              TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round =
                    std::max({out.dflash_round, finish(target), accept,
                              dflash_context_capacity(verify, batch, true), proposal});
            }
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = resolved_vision_token_limit(plan);
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            merged, std::min(merged, kFrontendSegmentLimit));
    }

    std::fprintf(stderr,
        "[WORKSPACE-DIAG] text_prefill    %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] ordinary_round  %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] mtp_prefill     %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] mtp_round       %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] dflash_context  %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] dflash_round    %12zu B  %8.4f MiB\n"
        "[WORKSPACE-DIAG] vision_encode   %12zu B  %8.4f MiB\n",
        out.text_prefill,   out.text_prefill   / 1048576.0,
        out.ordinary_round, out.ordinary_round / 1048576.0,
        out.mtp_prefill,    out.mtp_prefill    / 1048576.0,
        out.mtp_round,      out.mtp_round      / 1048576.0,
        out.dflash_context, out.dflash_context / 1048576.0,
        out.dflash_round,   out.dflash_round   / 1048576.0,
        out.vision_encode,  out.vision_encode  / 1048576.0);

    out.capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                             out.dflash_context, out.dflash_round, out.vision_encode});
    return out;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    if (options.vision_max_tokens > 32768) {
        throw std::invalid_argument("vision_max_tokens must be in [0,32768]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (options.speculative.backend != DFlashConfig::backend) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument("masked draft window must be in [1,15]");
        }
        break;
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->vision_max_tokens  = inputs.vision_max_tokens;
    impl->draft_window        = inputs.draft_window;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->device              = inputs.device;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->features.vision) {
        const std::uint32_t merged = resolved_vision_token_limit(*impl);
        impl->request_transient_capacity_bytes =
            schedule::VisionContext::output_transient_bytes(merged);
    }
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            // 48 MiB, not 12. The original figure under-estimated the graph driving state on
            // this machine by 2.7x: with --max-context 2048 the preparation step consumed
            // 33,423,360 B (31.9 MiB) against a 12 MiB allowance, which aborted startup with
            // "CUDA Graph preparation consumed ... exceeding the planned allowance".
            // Re-measure with the --log-stats-interval-ms summary if the graph topology
            // changes; the observed figure is reported as "CUDA Graph memory <observed> /
            // <allowance>".
            impl->graph_allowance_bytes = checked_mul(48ULL * kMiB, impl->max_concurrency,
                                                      "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    // 48/256 MiB, not 12/82. The allowance does not scale with context length
                    // while the graph preparation cost does: measured 31.9 MiB at 2048 context
                    // and 191,819,776 B (182.9 MiB) at 65536 context, the latter against the
                    // 82 MiB branch and aborting startup. Raised so graph mode survives the
                    // contexts this device can actually hold. Note the side effect: the
                    // allowance is subtracted during capacity planning, so a larger figure
                    // leaves slightly less room for --kv-capacity auto.
                    return (final_visible <= 4096 ? 48ULL : 256ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 48ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    validate_target_options(device, options);

    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .capacity            = options.max_context,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .vision_max_tokens   = options.vision_max_tokens,
        .draft_window        = options.speculative.draft_tokens,
        .speculative_backend = options.speculative.backend,
        .kv_dtype =
            options.kv_cache == KvCacheStorage::BFloat16
                ? DType::BF16
                : options.kv_cache == KvCacheStorage::Int8Group64 ? DType::I8 : DType::U8,
        .kv_quant_group =
            options.kv_cache == KvCacheStorage::BFloat16 ? 0 : qwen3_6::kKvQuantGroup,
        .proposal_head  = options.speculative.proposal_head,
        .features       = qwen3_6::startup_features(options),
        .use_cuda_graph = options.use_cuda_graph,
        .device         = options.device,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_6::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
