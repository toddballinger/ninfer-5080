#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
DynamicConvPrepareRoute resolve_route(int tokens) {
    // Complete cold-cache Graph timings favor split-8 R16 through 48 columns.
    // R32 halves split scratch and avoids the repeated wide tile above that range.
    if (tokens <= 48) return {16, (tokens + 7) / 8 * 8, 8};
    return {32, tokens <= 96 ? 32 : 64, 4};
}

std::size_t capacity(DynamicConvPrepareRoute route, int tokens) {
    return static_cast<std::size_t>(1280) * tokens * route.split_k * sizeof(float);
}

void execute_bf16(DynamicConvPrepareRoute route, const Tensor& residual,
                  const Tensor& norm, float eps, const Tensor& base,
                  const Weight& weight, Tensor& prepared, Tensor& finish,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int tokens = residual.ne[1] * residual.ne[2];

    float* partial =
        static_cast<float*>(
            workspace.alloc_bytes(capacity(route, tokens)).data);

    rmsnorm(residual, norm, eps, false, prepared, stream);

    bf16_dynamic_grouped_conv_prepare_partial_launch(
        route, prepared, weight, partial, stream);

    bf16_dynamic_grouped_conv_prepare_reduce_launch(
        route, base, partial, prepared, finish, stream);
}

void execute_q6(const Tensor& residual, const Tensor& norm, float eps,
                const Tensor& base, const Weight& weight,
                Tensor& prepared, Tensor& finish,
                WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int tokens = residual.ne[1] * residual.ne[2];

    DeviceSpan projected_storage =
        workspace.alloc_bytes(
            static_cast<std::size_t>(1280) * tokens *
            sizeof(std::uint16_t));

    Tensor projected(
        projected_storage.data, DType::BF16, {1280, tokens});

    rmsnorm(residual, norm, eps, false, prepared, stream);

    linear(
        prepared.view({5120, tokens}),
        weight,
        projected,
        stream);

    bf16_dynamic_grouped_conv_prepare_materialized_reduce_launch(
        base, projected, prepared, finish, stream);
}
} // namespace

std::size_t bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(int min_width, int max_width,
                                                                       int min_batch,
                                                                       int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid W/B interval");
    std::size_t maximum = 0;
    for (int w = min_width; w <= max_width; ++w)
        for (int b = min_batch; b <= max_batch; ++b)
            maximum = std::max(maximum, capacity(resolve_route(w * b), w * b));
    return maximum;
}

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base, const Weight& weight,
                                                Tensor& prepared, Tensor& finish,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    if (weight.qtype == QType::BF16_CTRL) {
        execute_bf16(
            resolve_route(residual.ne[1] * residual.ne[2]),
            residual, norm, eps, base, weight,
            prepared, finish, workspace, stream);
        return;
    }

    if (weight.qtype == QType::Q6G64_F16S) {
        execute_q6(
            residual, norm, eps, base, weight,
            prepared, finish, workspace, stream);
        return;
    }

    throw std::logic_error(
        "dynamic grouped conv prepare: unsupported projection qtype");
}
} // namespace ninfer::ops::detail
