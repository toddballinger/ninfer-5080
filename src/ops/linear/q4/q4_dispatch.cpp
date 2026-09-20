#include "ops/linear/q4/q4_dispatch.h"

#include <stdexcept>
#include <map>
#include <tuple>
#include <mutex>
#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail {

namespace {

using Q4GeometryKey =
    std::tuple<std::int32_t, std::int32_t, std::int32_t>; // N,K,T

std::map<Q4GeometryKey, std::uint64_t> q4_geometry_counts;
std::mutex q4_geometry_mutex;

void print_q4_geometry_histogram() {
    std::lock_guard<std::mutex> lock(q4_geometry_mutex);

    std::fprintf(stderr, "\nQ4 geometry histogram (N,K,T):\n");

    for (const auto& [key, count] : q4_geometry_counts) {
        const auto [n, k, t] = key;

        std::fprintf(
            stderr,
            "  N=%d K=%d T=%d  calls=%llu\n",
            n, k, t,
            static_cast<unsigned long long>(count));
    }
}

struct Q4GeometryProfileAtExit {
    Q4GeometryProfileAtExit() {
        std::atexit(print_q4_geometry_histogram);
    }
};

Q4GeometryProfileAtExit q4_geometry_profile_at_exit;

} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q4 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 4096:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }

            // RTX 5080 retune of upstream bb844c43.
            //
            // K-split loses to the existing SIMT routes at T=2..4,
            // but wins decisively from T=5 onward.
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 8) { return launch_q4_4096_5120_ksplit_c8; }
            if (t <= 16) { return launch_q4_4096_5120_ksplit_c16; }
            if (t <= 24) { return launch_q4_4096_5120_ksplit_c24; }
            if (t <= 32) { return launch_q4_4096_5120_ksplit_c32; }

            if (t <= 64) { return launch_q4_mma_r32_c32_wide; }

            // R32C32 wins through the five-tile T=160 endpoint.
            // At T=161+ its sixth tile makes R64C128 materially faster.
            if (t <= 160) { return launch_q4_mma_r32_c32; }

            // R64C128 remains best through its two-tile T=256 endpoint.
            if (t <= 256) { return launch_q4_mma_r64_c128; }

            // Crossing 256 forces another R64C128 column tile, making
            // R32C32 faster through its measured T=288 endpoint.
            if (t <= 288) { return launch_q4_mma_r32_c32; }

            // 289+ is effectively tied or favors the incumbent, so keep
            // the established route rather than extending a marginal band.
            return launch_q4_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 34816:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 131072:
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 8) { return launch_q4_draft_head_small_t; }
            return launch_q4_mma_r64_c128;

        // Qwen3.8-27B vocabulary projection [248320,5120].
        // The K=5120 GEMV supports arbitrary output rows; larger T uses
        // the generic row-split SIMT/MMA kernels.
        case 248320:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t == 4) { return launch_q4_qwen38_head_t4; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;

        default:
            break;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 4) { return launch_q4_qwen38_down_t4; }
            // Qwen3.8-27B MLP down projection [5120,17408].
            // Generic row-split SIMT is valid for small T, including decode;
            // use MMA for larger token counts.
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        }
        break;

    case 6144:
        if (n == 5120) {
            if (t == 4) { return launch_q4_qwen38_gdn_out_t4; }
            // Qwen3.8-27B GDN output [5120,6144].
            // Generic SIMT handles even T=1; MMA handles larger token counts.
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        }
        break;

    case 2048:
        if (n == 131072) {
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 20) { return launch_q4_draft_head_small_t; }
            if (t <= 32) { return launch_q4_mma_r64_c32; }
            if (t <= 48) { return launch_q4_mma_r64_c48; }
            if (t <= 56) { return launch_q4_mma_r64_c56; }
            if (t <= 63) { return launch_q4_mma_r64_c72; }
            if (t == 64) { return launch_q4_mma_r64_c64_endpoint; }
            if (t <= 72) { return launch_q4_mma_r64_c72; }
            if (t <= 80) { return launch_q4_mma_r64_c80; }
            if (t <= 96) { return launch_q4_mma_r64_c96; }
            if (t <= 104) { return launch_q4_mma_r64_c104_bounded; }
            if (t <= 111) { return launch_q4_mma_r64_c112_partial; }
            if (t == 112) { return launch_q4_mma_r64_c112; }
            if (t <= 119) { return launch_q4_mma_r64_c120_partial; }
            if (t == 120) { return launch_q4_mma_r64_c120; }
            return launch_q4_mma_r64_c128;
        }
        break;
    case 1152:
        if (t < 4 || t > 131072 || (t % 4) != 0) { break; }
        switch (n) {
        case 3456:
            if (t <= 36) { return launch_q4_simt_r8_c4; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        case 4304:
            if (t == 4) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t == 12) { return launch_q4_simt_r8_c4; }
            if (t <= 24) { return launch_q4_simt_r8_c8; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    case 4096:
        switch (n) {
        case 24576:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 131072:
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q4 linear: unsupported shape or T");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_q4_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q4 linear: unsupported policy");
}

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lock(q4_geometry_mutex);
        ++q4_geometry_counts[
            Q4GeometryKey{
                static_cast<std::int32_t>(w.n),
                static_cast<std::int32_t>(w.k),
                static_cast<std::int32_t>(x.ne[1])
            }
        ];
    }

    const Q4Launch launch = select_q4_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
