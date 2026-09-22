#include "targets/qwen3_5_9b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                      : block.qtype == QType::Q6G64_F16S ? 16
                                                                         : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 24576, 4096);
    out.down    = materialized_weight(materialized, plan.down, 4096, 12288);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection);
    return SplitAttentionProjectionPayload{
        .query_key  = materialized_weight(materialized, split->query_key, 5120, 4096),
        .gate_value = materialized_weight(materialized, split->gate_value, 5120, 4096),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection);
    return SplitGdnInputProjectionPayload{
        .query_key = materialized_weight(materialized, split->query_key, 4096, 4096),
        .value_z   = materialized_weight(materialized, split->value_z, 8192, 4096),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {4096});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {5120, 4096}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {5120, 4096}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {4096, 4096});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {32});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {32});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_projection", NumericFormat::BF16, {32, 4096});
            target.gdn.b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/b_projection", NumericFormat::BF16, {32, 4096});
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 4096}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {8192, 4096}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {4096, 4096});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {4096});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {24576, 4096});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {4096, 12288});
    }
}

RuntimeModelView build_runtime(BindingPlan const& plan,
                               artifact::MaterializedArtifact& materialized) {
    RuntimeModelView view;
    view.weights_arena = &materialized.device_arena();
    view.features      = plan.features;

    auto& full_layers = view.full_layers;
    auto& gdn_layers  = view.gdn_layers;

    view.token_embedding = materialized_weight(materialized, plan.token_embedding, 248320, 4096);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(
                materialized, source.input_norm, NumericFormat::BF16, {4096});
            target.projection = load_attention_projection(source.attention, materialized);
            target.query_norm = artifact::materialized_tensor(
                materialized, source.attention.query_norm, NumericFormat::BF16, {256});
            target.key_norm = artifact::materialized_tensor(
                materialized, source.attention.key_norm, NumericFormat::BF16, {256});
            target.output = materialized_weight(materialized, source.attention.output, 4096, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                materialized, source.post_attention_norm, NumericFormat::BF16, {4096});
            target.post_mixer = load_mlp(source.mlp, materialized);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(
                materialized, source.input_norm, NumericFormat::BF16, {4096});
            target.projection.a_log = artifact::materialized_tensor(
                materialized, source.gdn.a_log, NumericFormat::FP32, {32});
            target.projection.dt_bias = artifact::materialized_tensor(
                materialized, source.gdn.dt_bias, NumericFormat::FP32, {32});
            target.projection.control_projection = SplitGdnControlProjectionPayload{
                .a_projection = artifact::materialized_weight(
                    materialized, source.gdn.a_projection, NumericFormat::BF16, 32, 4096),
                .b_projection = artifact::materialized_weight(
                    materialized, source.gdn.b_projection, NumericFormat::BF16, 32, 4096),
            };
            target.projection.input_projection =
                load_gdn_input_projection(source.gdn, materialized);
            target.convolution = artifact::materialized_tensor(
                materialized, source.gdn.convolution, NumericFormat::BF16, {8192, 4});
            target.norm = artifact::materialized_tensor(materialized, source.gdn.norm,
                                                        NumericFormat::BF16, {128});
            target.output = materialized_weight(materialized, source.gdn.output, 4096, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                materialized, source.post_attention_norm, NumericFormat::BF16, {4096});
            target.post_mixer = load_mlp(source.mlp, materialized);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    view.final_norm  = artifact::materialized_tensor(materialized, plan.final_norm,
                                                     NumericFormat::BF16, {4096});
    view.output_head = materialized_weight(materialized, plan.output_head, 248320, 4096);

    if (plan.features.optimized_proposal()) {
        auto& proposal     = view.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(
            materialized, plan.draft_head, NumericFormat::Q4G64_F16S, 131072, 4096);
        proposal.token_ids = artifact::materialized_tensor(
            materialized, plan.draft_head_token_ids, NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = view.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            materialized, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 4096, 8192);
        mtp.embedding_norm = artifact::materialized_tensor(materialized, plan.mtp.embedding_norm,
                                                           NumericFormat::BF16, {4096});
        mtp.hidden_norm = artifact::materialized_tensor(materialized, plan.mtp.hidden_norm,
                                                        NumericFormat::BF16, {4096});
        mtp.input_norm = artifact::materialized_tensor(materialized, plan.mtp.input_norm,
                                                       NumericFormat::BF16, {4096});
        mtp.attention.packed = artifact::materialized_weight(
            materialized, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, 10240, 4096);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, 4096);
        mtp.attention.key         = row_view(mtp.attention.packed, 4096, 1024);
        mtp.attention.output_gate = row_view(mtp.attention.packed, 5120, 4096);
        mtp.attention.value       = row_view(mtp.attention.packed, 9216, 1024);
        mtp.query_norm = artifact::materialized_tensor(materialized, plan.mtp.query_norm,
                                                       NumericFormat::BF16, {256});
        mtp.key_norm = artifact::materialized_tensor(materialized, plan.mtp.key_norm,
                                                     NumericFormat::BF16, {256});
        mtp.output = artifact::materialized_weight(
            materialized, plan.mtp.output, NumericFormat::W8G32_F16S, 4096, 4096);
        mtp.post_attention_norm = artifact::materialized_tensor(
            materialized, plan.mtp.post_attention_norm, NumericFormat::BF16, {4096});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, materialized);
        mtp.final_norm = artifact::materialized_tensor(materialized, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {4096});
    }

    if (plan.features.vision) {
        auto& vision  = view.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            materialized, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2 = artifact::materialized_weight(
            materialized, plan.vision_merger_fc2, NumericFormat::W8G32_F16S, 4096, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(
            materialized, plan.vision_merger_fc2_bias, NumericFormat::BF16, {4096});
    }

    return view;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    if (weights_profile != WeightsProfile::GroupwiseInt) {
        throw std::invalid_argument("qwen3_5_9b: only GroupwiseInt is supported");
    }

    BindingPlan out;
    out.features = features;

    // Frontend resources
    out.frontend = qwen3_6::bind_frontend_resources(binder);

    // Token embedding
    const artifact::TensorPlacement embedding_placement =
        features.embedding_host ? artifact::TensorPlacement::HostMapped
                           : artifact::TensorPlacement::Device;
    out.token_embedding = bind_weight(binder, "text/token_embedding",
                                      NumericFormat::Q6G64_F16S, {248320, 4096},
                                      embedding_placement);

    // Text layers
    bind_text_layers(binder, out);

    // Final norm
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm",
                                                   NumericFormat::BF16, {4096});

    // Output head
    out.output_head = bind_weight(binder, "text/output_head",
                                   NumericFormat::Q6G64_F16S, {248320, 4096});

    // Draft head
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 4096}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);

    // MTP
    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {4096, 8192});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {4096});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {4096});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {4096});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {10240, 4096});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {4096, 4096});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {4096});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {24576, 4096}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {4096, 12288}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {4096});

    // Vision backbone
    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone = qwen3_6::bind_vision_backbone(binder, vision_placement);

    // Vision merger
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2 = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {4096, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {4096}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    // Materialization plan
    ArtifactLoadPlan plan;
    plan.bindings = out;
    plan.materialization = binder.finish();

    return plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan,
                                  artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      frontend(qwen3_6::take_frontend_resources(backing, plan.frontend)),
      runtime(build_runtime(plan, backing)) {}

} // namespace ninfer::targets::qwen3_5_9b::detail