#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <int Width>
__global__
__launch_bounds__(Width <= 8 ? 128 : 256, 4) void dynamic_grouped_conv_prepare_reduce_kernel(
    const __nv_bfloat16* base, const float* partial, __nv_bfloat16* prepared, __nv_bfloat16* finish,
    int batch_size, int splits) {
    __shared__ float projected[4][Width];
    __shared__ float normalized[Width][16];
    const int tid = threadIdx.x, group = blockIdx.x, batch = blockIdx.y;
    const int tokens = Width * batch_size;
    if (tid < 4 * Width) {
        const int coefficient = tid / Width, position = tid % Width,
                  row = coefficient * 320 + group;
        float sum     = 0;
        for (int split = 0; split < splits; ++split)
            sum += partial[(static_cast<std::int64_t>(split) * tokens + batch * Width + position) *
                               1280 +
                           row];
        projected[coefficient][position] = sum;
    }
    const int position = tid / 16, channel = tid % 16, hidden = group * 16 + channel;
    const std::int64_t offset = static_cast<std::int64_t>(batch * Width + position) * 5120 + hidden;
    if (position < Width) { normalized[position][channel] = __bfloat162float(prepared[offset]); }
    __syncthreads();
    if (tid < 2 * Width) {
        const int tap = tid / Width, pos = tid % Width;
        finish[((batch * Width + pos) * 2 + tap) * 320 + group] =
            __float2bfloat16_rn(projected[2 + tap][pos]);
    }
    if (position < Width) {
        float value = (__bfloat162float(base[hidden]) + projected[0][position]) *
                      normalized[position][channel];
        if (position > 0)
            value = fmaf(__bfloat162float(base[5120 + hidden]) + projected[1][position],
                         normalized[position - 1][channel], value);
        prepared[offset] = __float2bfloat16_rn(value);
    }
}


template <int Width>
__global__
__launch_bounds__(Width <= 8 ? 128 : 256, 4)
void dynamic_grouped_conv_prepare_materialized_reduce_kernel(
    const __nv_bfloat16* base, const __nv_bfloat16* coefficients,
    __nv_bfloat16* prepared, __nv_bfloat16* finish, int batch_size) {

    __shared__ float projected[4][Width];
    __shared__ float normalized[Width][16];

    const int tid = threadIdx.x;
    const int group = blockIdx.x;
    const int batch = blockIdx.y;

    if (tid < 4 * Width) {
        const int coefficient = tid / Width;
        const int position = tid % Width;
        const int row = coefficient * 320 + group;
        const int token = batch * Width + position;

        projected[coefficient][position] =
            __bfloat162float(
                coefficients[static_cast<std::int64_t>(token) * 1280 + row]);
    }

    const int position = tid / 16;
    const int channel = tid % 16;
    const int hidden = group * 16 + channel;

    const std::int64_t offset =
        static_cast<std::int64_t>(batch * Width + position) * 5120 + hidden;

    if (position < Width) {
        normalized[position][channel] =
            __bfloat162float(prepared[offset]);
    }

    __syncthreads();

    if (tid < 2 * Width) {
        const int tap = tid / Width;
        const int pos = tid % Width;

        finish[((batch * Width + pos) * 2 + tap) * 320 + group] =
            __float2bfloat16_rn(projected[2 + tap][pos]);
    }

    if (position < Width) {
        float value =
            (__bfloat162float(base[hidden]) + projected[0][position]) *
            normalized[position][channel];

        if (position > 0) {
            value = fmaf(
                __bfloat162float(base[5120 + hidden]) + projected[1][position],
                normalized[position - 1][channel],
                value);
        }

        prepared[offset] = __float2bfloat16_rn(value);
    }
}

template <int W>
void launch_materialized(const Tensor& base, const Tensor& projected,
                         Tensor& prepared, Tensor& finish,
                         cudaStream_t stream) {
    const dim3 grid(320, prepared.ne[2]);

    dynamic_grouped_conv_prepare_materialized_reduce_kernel<W>
        <<<grid, W <= 8 ? 128 : 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(base.data),
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<__nv_bfloat16*>(prepared.data),
            static_cast<__nv_bfloat16*>(finish.data),
            prepared.ne[2]);

    CUDA_CHECK(cudaGetLastError());
}

template <int W>
void launch(DynamicConvPrepareRoute route, const Tensor& base, const float* partial,
            Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const dim3 grid(320, prepared.ne[2]);
    dynamic_grouped_conv_prepare_reduce_kernel<W><<<grid, W <= 8 ? 128 : 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(base.data), partial,
        static_cast<__nv_bfloat16*>(prepared.data), static_cast<__nv_bfloat16*>(finish.data),
        prepared.ne[2], route.split_k);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream) {
#define WIDTH(W)                                                                                   \
    case W:                                                                                        \
        return launch<W>(route, base, partial, prepared, finish, stream)
    switch (prepared.ne[1]) {
        WIDTH(2);
        WIDTH(3);
        WIDTH(4);
        WIDTH(5);
        WIDTH(6);
        WIDTH(7);
        WIDTH(8);
        WIDTH(9);
        WIDTH(10);
        WIDTH(11);
        WIDTH(12);
        WIDTH(13);
        WIDTH(14);
        WIDTH(15);
        WIDTH(16);
    }
#undef WIDTH
    throw std::logic_error("dynamic grouped conv prepare: invalid production width");
}

void bf16_dynamic_grouped_conv_prepare_materialized_reduce_launch(
    const Tensor& base, const Tensor& projected, Tensor& prepared, Tensor& finish,
    cudaStream_t stream) {
#define WIDTH(W)                                                                                   \
    case W:                                                                                        \
        return launch_materialized<W>(base, projected, prepared, finish, stream)
    switch (prepared.ne[1]) {
        WIDTH(2);
        WIDTH(3);
        WIDTH(4);
        WIDTH(5);
        WIDTH(6);
        WIDTH(7);
        WIDTH(8);
        WIDTH(9);
        WIDTH(10);
        WIDTH(11);
        WIDTH(12);
        WIDTH(13);
        WIDTH(14);
        WIDTH(15);
        WIDTH(16);
    }
#undef WIDTH
    throw std::logic_error(
        "dynamic grouped conv prepare: invalid materialized production width");
}
} // namespace ninfer::ops::detail
