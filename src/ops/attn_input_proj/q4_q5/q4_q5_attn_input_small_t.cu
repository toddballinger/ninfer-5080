#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <std::int32_t Rows, std::int32_t Split, std::int32_t Hidden, std::int32_t FullSlabs>
struct AttnInputGeometry {
    static constexpr std::int32_t kParentRows = Rows;
    static constexpr std::int32_t kSplitRow   = Split;
    static constexpr std::int32_t kHidden     = Hidden;
    static constexpr std::int32_t kFullSlabs  = FullSlabs;
};

using AttnInputGeometry27 =
    AttnInputGeometry<7168, 6144, 5120, 5>;
using AttnInputGeometry9 =
    AttnInputGeometry<5120, 4096, 4096, 4>;

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Geometry>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kSplitRow   = Geometry::kSplitRow;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kSplitRow   = Geometry::kSplitRow;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], kParentRows, kHidden, cols,
            weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    const bool full = (kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0 &&
                      x.ne[1] != 4;
    if (full) {
        launch_q4_simt<Geometry, Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Geometry, Schedule, false>(x, weight, q, key, stream);
    }
}

template <class Geometry>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv<Geometry>(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        launch_q4_simt_route<Geometry, Q4AttnSimtR8C4Schedule>(x, weight, q, key, stream);
        return;
    case 8:
    case 16:
        launch_q4_simt_route<Geometry, Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
        return;
    default:
        // Correctness-first Q4/Q4 fallback for wider token batches.
        //
        // The grouped attention pair kernels are qualified for the
        // production Q4/Q5 pair, but are not safe when gate/value is
        // also Q4. The generic Q4 RowSplit SIMT kernel already supports
        // split output at the 6144-row seam and arbitrary positive T.
        launch_q4_simt_route<
            Geometry,
            Q4AttnSimtR8C8Schedule>(
                x, weight, q, key, stream);
        return;
    }
}

template <class Geometry>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kSplitRow   = Geometry::kSplitRow;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, false, true, kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kSplitRow   = Geometry::kSplitRow;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr std::int32_t kFullSlabs  = Geometry::kFullSlabs;
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, kFullSlabs, kHidden, true,
                                        kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), kParentRows,
                                        gate.ne[0], kHidden, x.ne[1], weight.padded_shape[1],
                                        kFullSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
    case 3:
    case 4:
        launch_q5_split4<Geometry, 4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<Geometry, 5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<Geometry, 6>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
    }
}

template <class Geometry, int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr std::int32_t kParentRows = Geometry::kParentRows;
    constexpr std::int32_t kSplitRow   = Geometry::kSplitRow;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr std::int32_t kFullSlabs  = Geometry::kFullSlabs;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), kParentRows, gate.ne[0], kHidden, cols,
        weight.padded_shape[1], kFullSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<Geometry>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact<Geometry>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt<Geometry, 4>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
}


Weight q4_rowsplit_row_view(const Weight& parent, std::int32_t row_begin,
                            std::int32_t row_count) {
    if (parent.qtype != QType::Q4G64_F16S ||
        parent.layout != QuantLayout::RowSplit ||
        parent.group != 64 ||
        row_begin < 0 ||
        row_count <= 0 ||
        row_begin + row_count > parent.n) {
        throw std::invalid_argument(
            "Q4 attention independent MMA row view is invalid");
    }

    const std::int64_t groups =
        static_cast<std::int64_t>(parent.padded_shape[1]) / parent.group;

    Weight view = parent;

    view.qdata =
        static_cast<const std::uint8_t*>(parent.qdata) +
        static_cast<std::int64_t>(row_begin) * groups * 32;

    view.scales =
        static_cast<const std::uint8_t*>(parent.scales) +
        static_cast<std::int64_t>(row_begin) * groups * 2;

    view.qhigh = nullptr;
    view.high_plane_bytes = 0;

    view.n = row_count;
    view.shape[0] = row_count;
    view.padded_shape[0] = row_count;

    return view;
}

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                     Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4<Geometry>(x, query_key_weight, q, k, stream);

    if (gate_value_weight.qtype == QType::Q4G64_F16S) {
        launch_q4<Geometry>(
            x, gate_value_weight, gate, v, stream);
        return;
    }

    if (gate_value_weight.qtype == QType::Q5G64_F16S) {
        launch_q5<Geometry>(
            x, gate_value_weight, gate, v, stream);
        return;
    }

    throw std::invalid_argument(
        "attention gate/value weight must be "
        "Q4G64_F16S or Q5G64_F16S");
}

} // namespace


void q4_q4_attn_input_independent_mma_launch(
    const Tensor& x, const Weight& query_key_weight,
    const Weight& gate_value_weight, Tensor& q, Tensor& gate,
    Tensor& k, Tensor& v, cudaStream_t stream) {

    if (x.ne[1] <= 16) {
        throw std::invalid_argument(
            "Q4 attention independent MMA requires T > 16");
    }

    if (query_key_weight.qtype != QType::Q4G64_F16S ||
        gate_value_weight.qtype != QType::Q4G64_F16S) {
        throw std::invalid_argument(
            "Q4 attention independent MMA requires Q4/Q4 weights");
    }

    if (query_key_weight.n != q.ne[0] + k.ne[0] ||
        gate_value_weight.n != gate.ne[0] + v.ne[0] ||
        q.ne[1] != x.ne[1] ||
        gate.ne[1] != x.ne[1] ||
        k.ne[1] != x.ne[1] ||
        v.ne[1] != x.ne[1]) {
        throw std::invalid_argument(
            "Q4 attention independent MMA geometry mismatch");
    }

    const Weight q_weight =
        q4_rowsplit_row_view(query_key_weight, 0, q.ne[0]);
    const Weight k_weight =
        q4_rowsplit_row_view(query_key_weight, q.ne[0], k.ne[0]);

    const Weight gate_weight =
        q4_rowsplit_row_view(gate_value_weight, 0, gate.ne[0]);
    const Weight value_weight =
        q4_rowsplit_row_view(gate_value_weight, gate.ne[0], v.ne[0]);

    // Use the already-qualified generic Q4 RowSplit tensor-core kernel.
    // R64/C128 is the production large-T route selected by generic Q4
    // dispatch for the corresponding long-prefill geometries.
    launch_q4_mma_r64_c128(x, q_weight, q, stream);
    launch_q4_mma_r64_c128(x, k_weight, k, stream);
    launch_q4_mma_r64_c128(x, gate_weight, gate, stream);
    launch_q4_mma_r64_c128(x, value_weight, v, stream);
}

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    switch (x.ne[0]) {
    case 5120:
        launch_geometry<AttnInputGeometry27>(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                             stream);
        return;
    case 4096:
        launch_geometry<AttnInputGeometry9>(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                            stream);
        return;
    default:
        throw std::invalid_argument("attention Q4/Q5 split-output: unsupported input width");
    }
}


void q4_q5_attn_input_small_t_prewarm() {
    cudaFuncAttributes attr{};

    // Qwen3.8 / Geometry27 Q4 attention-input path.
    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        q4_rowsplit_gemm_simt_kernel<
            Q4AttnSimtR8C4Schedule,
            false,
            true,
            AttnInputGeometry27::kSplitRow>));

    // Qwen3.8 / Geometry27 Q5 attention-input path.
    //
    // T=2/3/4 now share this kTt=4 specialization, with runtime t
    // controlling the number of active columns.
    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        q5_rowsplit_gemm_simt_split4_kernel<
            Q5RowSplitSimtSchedule,
            4,
            AttnInputGeometry27::kFullSlabs,
            AttnInputGeometry27::kHidden,
            true,
            AttnInputGeometry27::kSplitRow>));
}


} // namespace ninfer::ops::detail
