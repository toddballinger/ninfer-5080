#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void q4_small_t_mma_prewarm();

// RTX 5080 semantic port of upstream bb844c43 for Q4 A16 [4096,5120].
// These concrete launchers keep the fork's centralized dispatch and
// strided-output contract instead of importing upstream's per-shape layer.
void launch_q4_4096_5120_ksplit_c8(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_4096_5120_ksplit_c16(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_4096_5120_ksplit_c24(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_4096_5120_ksplit_c32(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
// Qwen3.8-27B exact T=4 tensor-core paths.
void launch_q4_qwen38_down_t4(const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void launch_q4_qwen38_gdn_out_t4(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_qwen38_head_t4(const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);

void q4_rowsplit_mma_prewarm();

void launch_q4_mma_r32_c32_wide(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r32_c32(
    const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

void launch_q4_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c56(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64_endpoint(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c104_bounded(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c120_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
