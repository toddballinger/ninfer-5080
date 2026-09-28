// Platform stub for the NVFP4 TMA kernels — MSVC only.
//
// ops/linear/nvfp4/nvfp4_w4a4_tma.cu and
// ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu cannot be compiled with MSVC.
// Both pass TMA descriptors by value into a __global__ function as __grid_constant__
// parameters, and MSVC's ABI rejects by-value parameters whose alignment exceeds 64
// while CUtensorMap requires 128. The alignment comes from the CUDA type itself and
// cannot be lowered, __grid_constant__ forbids references, and splitting the aggregate
// per descriptor does not help because each individual CUtensorMap is already
// over-aligned. See the "error C2719" section of docs/WINDOWS_5070TI.md for the full
// measured matrix and the pointer-based alternative.
//
// This file replaces those two translation units on MSVC so that the rest of the engine
// builds and runs. Every entry point reports the gap loudly where it is reached instead
// of the build silently dropping NVFP4 support or, worse, launching a misaligned
// tensormap. Linux builds compile the real implementations and never see this file.

#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_tma_unavailable() {
    throw std::runtime_error(
        "NVFP4 TMA kernels are not built on this platform. MSVC cannot pass the 128-byte "
        "aligned CUtensorMap descriptors by value into a __global__ function (error "
        "C2719), so ops/linear/nvfp4/nvfp4_w4a4_tma.cu and its LinearSwiGLU sibling are "
        "excluded from the Windows build. Load a non-NVFP4 weight artifact, or build the "
        "Linux target.");
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                     float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

} // namespace ninfer::ops::detail
