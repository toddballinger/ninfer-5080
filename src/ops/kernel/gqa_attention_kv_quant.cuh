#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// __builtin_memcpy is a GCC/Clang builtin and does not exist under MSVC. memcpy on a
// compile-time-constant size lowers to the same single load/store pair on both
// compilers, so use the portable spelling.
#include <cstring>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

// INT4-G64 stores two signed 4-bit codes per byte.
inline constexpr int kGqaKvQ4CodeExtent = kGqaKvQuantHeadDim / 2;

// INT2-G64 stores four 2-bit codes per byte.
inline constexpr int kGqaKvQ2CodeExtent = kGqaKvQuantHeadDim / 4;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_q4_code_index(int physical_page, int kv_head,
                                                              int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQ4CodeExtent, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_q2_code_index(int physical_page, int kv_head,
                                                              int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQ2CodeExtent, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 2);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Signed symmetric INT4-G64 codec. The per-group FP16 scale is absmax/7,
// mirroring the INT8-G64 absmax/127 contract while keeping equal positive
// and negative representable magnitude.
__device__ __forceinline__ std::int8_t gqa_kv_quant_q4_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ std::uint8_t gqa_kv_pack_q4(std::int8_t q0, std::int8_t q1) {
    const std::uint8_t lo = static_cast<std::uint8_t>(q0) & 0x0fu;
    const std::uint8_t hi = static_cast<std::uint8_t>(q1) & 0x0fu;
    return static_cast<std::uint8_t>(lo | (hi << 4));
}

__device__ __forceinline__ std::int8_t gqa_kv_unpack_q4_low(std::uint8_t packed) {
    const int q = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ std::int8_t gqa_kv_unpack_q4_high(std::uint8_t packed) {
    const int q = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    return static_cast<std::int8_t>(q);
}


// Symmetric 2-bit KV codec for the experimental MTP cache.
// Reconstruction levels are {-1, -1/3, +1/3, +1} * scale.
// The per-G64 FP16 scale is the group's absolute maximum.
__device__ __forceinline__ std::uint8_t gqa_kv_quant_q2_code(float x, float scale) {
    if (scale == 0.0f) { return 1u; }

    const float y = x / scale;

    if (y < -0.6666666667f) return 0u;
    if (y <  0.0f)          return 1u;
    if (y <  0.6666666667f) return 2u;
    return 3u;
}

__device__ __forceinline__ std::uint8_t gqa_kv_pack_q2(
    std::uint8_t q0, std::uint8_t q1, std::uint8_t q2, std::uint8_t q3) {
    return static_cast<std::uint8_t>(
        (q0 & 0x03u) |
        ((q1 & 0x03u) << 2) |
        ((q2 & 0x03u) << 4) |
        ((q3 & 0x03u) << 6));
}

__device__ __forceinline__ std::int8_t gqa_kv_unpack_q2_i8(std::uint8_t q) {
    switch (q & 0x03u) {
    case 0: return static_cast<std::int8_t>(-3);
    case 1: return static_cast<std::int8_t>(-1);
    case 2: return static_cast<std::int8_t>( 1);
    default: return static_cast<std::int8_t>(3);
    }
}

__device__ __forceinline__ float gqa_kv_dequant_q2_code(std::uint8_t q, float scale) {
    constexpr float one_third = 0.3333333333333333f;
    switch (q & 0x03u) {
    case 0: return -scale;
    case 1: return -one_third * scale;
    case 2: return  one_third * scale;
    default: return scale;
    }
}


// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Dequantize 8 consecutive signed INT4 codes from 4 packed bytes.
__device__ __forceinline__ int4 gqa_kv_dequant_q4x8_from(const std::uint8_t* codes4, float s) {
    std::uint32_t raw = 0;
    memcpy(&raw, codes4, sizeof(raw));

    unsigned values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint8_t b = static_cast<std::uint8_t>(raw >> (8 * i));
        const float x0 = static_cast<float>(gqa_kv_unpack_q4_low(b)) * s;
        const float x1 = static_cast<float>(gqa_kv_unpack_q4_high(b)) * s;
        values[i] = pack_bf16x2(x0, x1);
    }

    return make_int4(static_cast<int>(values[0]), static_cast<int>(values[1]),
                     static_cast<int>(values[2]), static_cast<int>(values[3]));
}

// Dequantize 8 consecutive Q2 values from two packed bytes.
// Each 2-bit code reconstructs to {-1, -1/3, +1/3, +1} * scale.
__device__ __forceinline__ int4 gqa_kv_dequant_q2x8_from(
    const std::uint8_t* codes2, float scale) {

    std::uint16_t raw = 0;
    memcpy(&raw, codes2, sizeof(raw));

    unsigned values[4];

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int bit0 = pair * 4;
        const std::uint8_t q0 =
            static_cast<std::uint8_t>((raw >> bit0) & 0x03u);
        const std::uint8_t q1 =
            static_cast<std::uint8_t>((raw >> (bit0 + 2)) & 0x03u);

        values[pair] =
            pack_bf16x2(
                gqa_kv_dequant_q2_code(q0, scale),
                gqa_kv_dequant_q2_code(q1, scale));
    }

    return make_int4(
        static_cast<int>(values[0]),
        static_cast<int>(values[1]),
        static_cast<int>(values[2]),
        static_cast<int>(values[3]));
}


} // namespace ninfer::ops
