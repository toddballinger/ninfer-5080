#include "ninfer/ops/dynamic_grouped_conv.h"

#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int H = 5120;
constexpr int N = 1280;
constexpr int G = 320;
constexpr int W = 8;
constexpr int B = 1;
constexpr int T = W * B;
constexpr float EPS = 1.0e-6F;

constexpr ReductionCriterion kKernelPrepared{
    3.6e-3, 5.0e-3, 6.0e-3
};

constexpr ReductionCriterion kKernelFinish{
    3.2e-3, 4.0e-3, 4.5e-3
};

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

float pattern(std::uint32_t index, std::uint32_t seed, float scale) {
    const std::uint32_t x = mix32(index ^ seed * 0x9e3779b9U);
    int v = static_cast<int>((x >> 8) & 0xffU) - 128;
    if (v == 0) v = (index & 1U) ? -1 : 1;
    return static_cast<float>(v) * scale;
}

struct Reference {
    std::vector<double> prepared;
    std::vector<double> finish;
};

Reference make_reference(
    const std::vector<std::uint16_t>& residual,
    const std::vector<std::uint16_t>& norm,
    const std::vector<std::uint16_t>& base,
    std::span<const float> projection) {

    std::vector<double> normalized(static_cast<std::size_t>(H) * T);

    for (int token = 0; token < T; ++token) {
        const std::size_t off = static_cast<std::size_t>(token) * H;

        double ss = 0.0;
        for (int h = 0; h < H; ++h) {
            const double x = bf16_to_f32(residual[off + h]);
            ss += x * x;
        }

        const double inv =
            1.0 / std::sqrt(ss / static_cast<double>(H) + EPS);

        for (int h = 0; h < H; ++h) {
            normalized[off + h] =
                static_cast<double>(bf16_to_f32(residual[off + h])) *
                inv *
                static_cast<double>(bf16_to_f32(norm[h]));
        }
    }

    std::vector<double> coeff(static_cast<std::size_t>(N) * T);

    for (int token = 0; token < T; ++token) {
        const double* x =
            normalized.data() + static_cast<std::size_t>(token) * H;

        for (int row = 0; row < N; ++row) {
            const float* w =
                projection.data() + static_cast<std::size_t>(row) * H;

            double sum = 0.0;
            for (int h = 0; h < H; ++h)
                sum += static_cast<double>(w[h]) * x[h];

            coeff[static_cast<std::size_t>(token) * N + row] = sum;
        }
    }

    Reference out;
    out.prepared.resize(static_cast<std::size_t>(H) * T);
    out.finish.resize(static_cast<std::size_t>(G) * 2 * T);

    for (int token = 0; token < T; ++token) {
        const int pos = token % W;
        const std::size_t noff = static_cast<std::size_t>(token) * H;

        for (int group = 0; group < G; ++group) {
            const double d0 =
                coeff[static_cast<std::size_t>(token) * N + group];

            const double d1 =
                coeff[static_cast<std::size_t>(token) * N + G + group];

            for (int tap = 0; tap < 2; ++tap) {
                out.finish[
                    (static_cast<std::size_t>(token) * 2 + tap) * G + group] =
                    coeff[static_cast<std::size_t>(token) * N +
                          (2 + tap) * G + group];
            }

            for (int c = 0; c < 16; ++c) {
                const int h = group * 16 + c;

                double value =
                    (static_cast<double>(bf16_to_f32(base[h])) + d0) *
                    normalized[noff + h];

                if (pos > 0) {
                    value +=
                        (static_cast<double>(bf16_to_f32(base[H + h])) + d1) *
                        normalized[noff - H + h];
                }

                out.prepared[noff + h] = value;
            }
        }
    }

    return out;
}

void print_stats(const char* label,
                 const std::vector<double>& actual,
                 const std::vector<double>& reference) {
    const ReductionStats s =
        compute_reduction_stats(
            actual.data(), reference.data(),
            static_cast<std::int64_t>(actual.size()));

    std::cout
        << "Q6_PREPARE_STATS kind=" << label
        << " rel_l2=" << s.relative_l2
        << " rmse=" << s.root_mean_squared_error
        << " max_abs=" << s.maximum_absolute_error
        << " max_ref=" << s.maximum_absolute_reference
        << '\n';
}

int run() {
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(H) * T);
    std::vector<std::uint16_t> norm(H);
    std::vector<std::uint16_t> base(static_cast<std::size_t>(H) * 4);
    std::vector<float> source(static_cast<std::size_t>(N) * H);

    for (int token = 0; token < T; ++token)
        for (int h = 0; h < H; ++h)
            residual[static_cast<std::size_t>(token) * H + h] =
                f32_to_bf16(
                    pattern(
                        static_cast<std::uint32_t>(token * H + h),
                        101U, 1.0F / 192.0F));

    for (int h = 0; h < H; ++h) {
        norm[h] =
            f32_to_bf16(
                0.875F +
                static_cast<float>((h * 17 + 11) % 65) *
                    (0.25F / 64.0F));
    }

    for (int slab = 0; slab < 4; ++slab)
        for (int h = 0; h < H; ++h) {
            const float center =
                slab == 0 ? 0.75F :
                slab == 1 ? -0.125F :
                slab == 2 ? 0.625F : 0.1875F;

            base[static_cast<std::size_t>(slab) * H + h] =
                f32_to_bf16(
                    center +
                    pattern(
                        static_cast<std::uint32_t>(slab * H + h),
                        211U, 1.0F / 8192.0F));
        }

    for (int row = 0; row < N; ++row)
        for (int h = 0; h < H; ++h) {
            const float x =
                pattern(
                    static_cast<std::uint32_t>(row * H + h),
                    307U, 1.0F / 8192.0F);

            source[static_cast<std::size_t>(row) * H + h] =
                bf16_to_f32(f32_to_bf16(x));
        }

    auto packed =
        quantized_weight::pack_q6_row_split(source, N, H);

    const Reference dequant_reference =
        make_reference(residual, norm, base, packed.dequant);

    const Reference source_reference =
        make_reference(residual, norm, base, source);

    DeviceBuffer residual_d = to_device(residual);
    DeviceBuffer norm_d = to_device(norm);
    DeviceBuffer base_d = to_device(base);

    DeviceBuffer weight_d(packed.payload.size());
    weight_d.copy_from_host(
        packed.payload.data(), packed.payload.size());

    GuardedDeviceBuffer prepared_d(
        static_cast<std::size_t>(H) * T * sizeof(std::uint16_t));

    GuardedDeviceBuffer finish_d(
        static_cast<std::size_t>(G) * 2 * T * sizeof(std::uint16_t));

    prepared_d.fill(0xff);
    finish_d.fill(0xff);

    Tensor residual_t(residual_d.p, DType::BF16, {H, W, B});
    Tensor norm_t(norm_d.p, DType::BF16, {H});
    Tensor base_t(base_d.p, DType::BF16, {H, 2, 2});
    Tensor prepared_t(prepared_d.data(), DType::BF16, {H, W, B});
    Tensor finish_t(finish_d.data(), DType::BF16, {G, 2, W, B});

    const Weight weight =
        packed.device_weight(weight_d.p);

    const std::size_t capacity =
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            W, W, B, B);

    WorkspaceArena workspace(capacity);
    workspace.reset_peak();

    ops::rmsnorm_dynamic_grouped_conv_prepare(
        residual_t, norm_t, EPS, base_t, weight,
        prepared_t, finish_t, workspace, nullptr);

    cuda_synchronize();

    const auto prepared_actual =
        from_device_bf16(
            prepared_d.data(), static_cast<std::size_t>(H) * T);

    const auto finish_actual =
        from_device_bf16(
            finish_d.data(), static_cast<std::size_t>(G) * 2 * T);

    int failures = 0;

    failures += verify_reduction(
        "Q6 prepare prepared-vs-dequant",
        prepared_actual,
        std::span<const double>(
            dequant_reference.prepared.data(),
            dequant_reference.prepared.size()),
        kKernelPrepared);

    failures += verify_reduction(
        "Q6 prepare finish-vs-dequant",
        finish_actual,
        std::span<const double>(
            dequant_reference.finish.data(),
            dequant_reference.finish.size()),
        kKernelFinish);

    print_stats(
        "prepared_vs_dequant",
        prepared_actual,
        dequant_reference.prepared);

    print_stats(
        "finish_vs_dequant",
        finish_actual,
        dequant_reference.finish);

    print_stats(
        "prepared_vs_bf16_source",
        prepared_actual,
        source_reference.prepared);

    print_stats(
        "finish_vs_bf16_source",
        finish_actual,
        source_reference.finish);

    failures += prepared_d.verify_guards("Q6 prepare prepared");
    failures += finish_d.verify_guards("Q6 prepare finish");

    if (workspace.used() != 0 ||
        workspace.peak_used() > capacity) {
        std::cerr
            << "Q6 prepare workspace invalid"
            << " peak=" << workspace.peak_used()
            << " capacity=" << capacity
            << '\n';
        ++failures;
    }

    std::cout
        << "Q6_PREPARE_WORKSPACE"
        << " peak=" << workspace.peak_used()
        << " public_capacity=" << capacity
        << '\n';

    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }

        const int failures = run();

        std::cout
            << (failures == 0 ? "OK" : "FAIL")
            << " Q6 rmsnorm_dynamic_grouped_conv_prepare\n";

        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr
            << "Q6 dynamic grouped conv prepare test: "
            << e.what()
            << '\n';
        return 1;
    }
}
