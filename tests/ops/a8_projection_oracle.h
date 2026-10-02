#pragma once

// tests/ops/a8_projection_oracle.h
//
// Independent CPU oracle for the Issue #18 A8 (int8, K=5120, G=80, group=64)
// projection test. Everything below is computed from the TEST fixture alone;
// the production act_quant_g64 / int8_quantize_rn routes are deliberately NOT
// used, so the oracle stays independent of the production implementation.
//
// Weight side: Q4/Q5 row-split weights (QuantSpec W8G32_F16S, qmax=127,
// group_size=64; logical K=5120 => 80 logical groups) are decoded with
// ninfer::test::quantized_weight::logical_weight_fp64, which performs an
// independent signed decode of the packed codes multiplied by the exact FP64
// value of the stored FP16 row/group scale. K padding (physical K=5120 is
// already group-aligned, so padded == logical) is honored by referencing only
// logical K=5120.
//
// Input side: BF16 inputs are supplied by the caller as their exact FP32
// values (a BF16 bit pattern is exactly representable in FP32). The input
// tensor is [tokens, K] token-major (length tokens*K, exactly as the
// gather_rows sample buffer is laid out in
// ninfer::test::input_projection::gather_rows). Each token's (64-input group)
// is quantized independently in FLOAT: scale = maxabs/127.0f (0 for a zero
// group), code = independent ties-to-even nearest rounding of x*(127.0f/maxabs)
// computed in float, saturated into [-127, 127]. The float scale is promoted
// to double for the FP64 references. Each sampled token's quantization is
// computed once and shared by every output row (the test samples the same
// activation rows across all output rows).
//
// Layout: x is [tokens, K] token-major, length tokens*K; actual is the
// gather_rows sample buffer [sampleRows, tokens], row-major, length
// sampleRows*tokens, where each sampled row contains all tokens
// (actual[ri*tokens + token]). There is deliberately no rows*tokens*K
// requirement on the actual buffer.
//
// Bound (frozen): u = 2^-24, gamma160 = 160u/(1-160u),
// Efp32 = gamma160 * S + 160*FLT_TRUE_MIN, where
//   S  = sum over 160 channels of |w| * |code| * scale,
//   Eq = sum over 160 channels of |w| * |bf16-input - code*scale|.
// Bp(x) = 0.5 ulp of x as a BF16 value (conservative upper neighbor at
// power-of-two boundaries; 2^-134 for zero/subnormal).
//   Bp row bound (vs A8 reference) = Efp32 + Bp(A8 reference)
//   Bo row bound (vs original BF16) = Eq + Efp32 + Bp(A8 reference)
//
// Metrics on the deterministic unique sample set (31 row picks x up to 15
// unique token picks): normalized relative L2 uses the REFERENCE ENERGY
// (sum(ref*ref)) as its denominator; when the reference energy is zero, 0 is
// returned if the error energy is zero, else the absolute L2. maxAbs and the
// gross-violation count (with its denominator) are reported separately for
// (a) actual vs A8 bound and (b) actual vs original-BF16 bound. The
// quantization loss (A8 reference vs original-BF16 reference) relL2/maxAbs
// is reported independently, with the original-BF16 reference energy as
// denominator. The aggregate result is a failure when any gross violation is
// observed. No sample payload values are printed.

#include "ops/quantized_weight.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace test {
namespace a8_oracle {

// A8 shape: K=5120 logical inputs, 80 groups of 64 per token row.
constexpr std::int32_t kA8K = 5120;
constexpr std::int32_t kA8Groups = 80;
constexpr std::int32_t kA8GroupSize = 64;
constexpr std::int32_t kA8SampleRows = 31;
constexpr std::int32_t kA8TokenPicks = 15;

// Half-ULP of |value| considered as a BF16 number.
inline double halfBF16ULP(double value) {
    const double a = std::abs(value);
    // BF16 subnormals have a fixed 2^-133 step; half-step is 2^-134.
    // At the smallest normal and every exact power of two, the upper-neighbor
    // spacing is 2^(ilogb(a)-7), so this also covers the boundary correctly.
    if (a < std::ldexp(1.0, -126)) { return std::ldexp(1.0, -134); }
    const int exp = std::ilogb(a);
    return 0.5 * std::ldexp(1.0, exp - 7);
}

// Independent ties-to-even (RNE) nearest-integer rounding in DOUBLE.
inline std::int64_t rne_nearest(double x) {
    const double f  = std::floor(x);
    const double fr = x - f;
    std::int64_t n = static_cast<std::int64_t>(f);
    if (fr > 0.5) {
        ++n;
    } else if (fr < 0.5) {
        // floor is already the nearest integer for a fractional part below .5.
    } else { // exact tie: round to even
        if ((n & 1LL) != 0) { ++n; }
    }
    return n;
}

// Independent float-input RNE: classify the FP32 product p exactly with
// frexp (1.5 significand == exact .5 tie; 1.0 means an integer, which can
// only arise from a 1.5 tie in float rounding), then ties-to-even in double
// on the exact double reconstruction of p.
inline std::int64_t rne_nearest_f32(float p) {
    // Every FP32 value is exactly representable in FP64; nearest-even applied
    // to that exact value makes the half comparison exact, without reusing a
    // production quantizer.
    return rne_nearest(static_cast<double>(p));
}

// Quantize one token (K=5120 inputs, 80 logical groups) exactly as the A8
// contract prescribes, working entirely in FLOAT: per 64-input group,
// scale = maxabs/127.0f (0 for a zero group), code = independent RNE of
// x * (127.0f/maxabs) in float, saturated into [-127, 127]. The group scale
// is promoted to double for the FP64 references.
inline void quantize_input_token(const float* x32, std::vector<double>& scale,
                                 std::vector<std::int8_t>& codes) {
    std::fill(codes.begin(), codes.end(), std::int8_t(0));
    std::fill(scale.begin(), scale.end(), 0.0);
    for (std::int32_t g = 0; g < kA8Groups; ++g) {
        const float* base = x32 + static_cast<std::size_t>(g) * kA8GroupSize;
        float maxabs = 0.0F;
        for (std::int32_t i = 0; i < kA8GroupSize; ++i) {
            maxabs = std::fmaxf(maxabs, std::fabsf(base[i]));
        }
        if (maxabs == 0.0F) { continue; }
        const float s  = maxabs / 127.0F;
        const float inv = 127.0F / maxabs;
        scale[static_cast<std::size_t>(g)] = static_cast<double>(s);
        for (std::int32_t i = 0; i < kA8GroupSize; ++i) {
            const std::int64_t q  = rne_nearest_f32(base[i] * inv);
            const std::int64_t q2 = std::clamp(q, std::int64_t(-127), std::int64_t(127));
            codes[static_cast<std::size_t>(g) * kA8GroupSize + static_cast<std::size_t>(i)] =
                static_cast<std::int8_t>(q2);
        }
    }
}

// Deterministic row picks: Bresenham spacing, unique, covering both ends;
// 31 fixed picks (min(31, rows)).
inline std::vector<std::int32_t> sample_rows(std::int32_t rows, std::int32_t picks) {
    if (rows <= 0 || picks <= 0) { throw std::invalid_argument("a8 oracle: positive extents"); }
    const std::int32_t m = std::min(rows, picks);
    std::vector<std::int32_t> v;
    v.reserve(static_cast<std::size_t>(m) + 2);
    for (std::int32_t i = 0; i < m; ++i) {
        const std::int32_t x = (m == 1) ? 0
                                        : static_cast<std::int32_t>((static_cast<std::int64_t>(rows - 1) * i) / (m - 1));
        if (v.empty() || v.back() != x) { v.push_back(x); }
    }
    if (v.empty() || v.front() != 0) { v.insert(v.begin(), 0); }
    const std::int32_t last = rows - 1;
    if (v.back() != last) { v.push_back(last); }
    return v;
}

// Deterministic token picks: the valid required picks {0, 1, T/4, T/2,
// 3T/4, T-2, T-1} always come first; remaining budget is filled with the
// next increasing unique indices. Result is unique and always covers both
// ends when T permits.
inline std::vector<std::int32_t> token_picks(std::int32_t tokens, std::int32_t maxPicks) {
    if (tokens <= 0 || maxPicks <= 0) { throw std::invalid_argument("a8 oracle: positive extents"); }
    std::vector<std::int32_t> required = {0, 1, tokens / 4, tokens / 2,
                                          (3 * tokens) / 4, tokens - 2, tokens - 1};
    std::vector<std::int32_t> v;
    for (std::int32_t r : required) {
        if (r >= 0 && r < tokens &&
            std::find(v.begin(), v.end(), r) == v.end() &&
            static_cast<std::int32_t>(v.size()) < maxPicks) {
            v.push_back(r);
        }
    }
    for (std::int32_t t = 0;
         t < tokens && static_cast<std::int32_t>(v.size()) < maxPicks &&
                       static_cast<std::int32_t>(v.size()) < std::min(maxPicks, tokens);
         ++t) {
        if (std::find(v.begin(), v.end(), t) == v.end()) { v.push_back(t); }
    }
    return v;
}

struct A8OracleMetrics {
    // Actual vs A8 reference (quantized-input FP64 reference).
    double act_vs_a8_rel_l2     = 0.0;
    double act_vs_a8_max_abs    = 0.0;
    // Actual vs original-BF16 reference (unquantized-input FP64 reference).
    double act_vs_orig_rel_l2   = 0.0;
    double act_vs_orig_max_abs  = 0.0;
    // Quantization loss: A8 reference vs original-BF16 reference.
    double a8_vs_orig_rel_l2    = 0.0;
    double a8_vs_orig_max_abs   = 0.0;
    // Gross violations (|error| above the bound) with denominators, reported
    // separately for the two actual-vs-reference comparisons.
    std::int64_t act_vs_a8_gross     = 0;
    std::int64_t act_vs_a8_gross_den = 0;
    std::int64_t act_vs_orig_gross     = 0;
    std::int64_t act_vs_orig_gross_den = 0;
    std::int64_t samples = 0;

    // Fails when any gross violation is observed.
    bool ok() const { return act_vs_a8_gross == 0 && act_vs_orig_gross == 0; }
};

// Normalized relative L2 over the REFERENCE ENERGY (sum(ref*ref)): when the
// reference energy is zero, return 0 if the error energy is zero, else the
// absolute L2; otherwise sqrt(error energy / reference energy).
inline double normalized_l2(double e2, double ref_energy) {
    if (ref_energy <= 0.0) { return e2 <= 0.0 ? 0.0 : std::sqrt(e2); }
    return std::sqrt(e2 / ref_energy);
}

// Core metric computation. Vectors are 1:1 per sampled (row, token) point,
// laid out as the gather_rows sample buffer:
//  actual  = device int8 projection outputs (FP64), [sampleRows, tokens]
//  ref_a8  = A8 reference (weights x quantized inputs)
//  ref_orig = original-BF16 reference (weights x exact BF16 inputs)
//  a8_bound / orig_bound = per-point frozen bounds.
inline A8OracleMetrics compute_metrics(const std::vector<double>& actual,
                                       const std::vector<double>& ref_a8,
                                       const std::vector<double>& ref_orig,
                                       const std::vector<double>& a8_bound,
                                       const std::vector<double>& orig_bound) {
    if (actual.size() != ref_a8.size() || actual.size() != ref_orig.size() ||
        actual.size() != a8_bound.size() || actual.size() != orig_bound.size()) {
        throw std::invalid_argument("a8 oracle: mismatched vector lengths");
    }
    A8OracleMetrics m;
    const std::size_t n = actual.size();
    m.samples = static_cast<std::int64_t>(n);
    if (n == 0) { return m; }

    m.act_vs_a8_gross_den = static_cast<std::int64_t>(n);
    m.act_vs_orig_gross_den = static_cast<std::int64_t>(n);

    double e2_a8 = 0.0, e2_o = 0.0, e2_q = 0.0, e_a8 = 0.0, e_o = 0.0, e_q = 0.0;
    double ref_energy_a8 = 0.0, ref_energy_o = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ea = actual[i] - ref_a8[i];
        const double eo = actual[i] - ref_orig[i];
        const double eq = ref_a8[i] - ref_orig[i];
        e2_a8 += ea * ea;
        e2_o  += eo * eo;
        e2_q  += eq * eq;
        ref_energy_a8 += ref_a8[i] * ref_a8[i];
        ref_energy_o  += ref_orig[i] * ref_orig[i];
        e_a8 = std::fmax(e_a8, std::abs(ea));
        e_o  = std::fmax(e_o, std::abs(eo));
        e_q  = std::fmax(e_q, std::abs(eq));
        if (std::abs(ea) > a8_bound[i]) { ++m.act_vs_a8_gross; }
        if (std::abs(eo) > orig_bound[i]) { ++m.act_vs_orig_gross; }
    }
    m.act_vs_a8_rel_l2     = normalized_l2(e2_a8, ref_energy_a8);
    m.act_vs_a8_max_abs    = e_a8;
    m.act_vs_orig_rel_l2   = normalized_l2(e2_o, ref_energy_o);
    m.act_vs_orig_max_abs  = e_o;
    m.a8_vs_orig_rel_l2    = normalized_l2(e2_q, ref_energy_o);
    m.a8_vs_orig_max_abs   = e_q;
    return m;
}

// The single stable aggregate metric line for one A8 projection run. Prints
// metric values only; no sample payload values.
inline std::string metrics_line(const A8OracleMetrics& m) {
    std::string s;
    s += "a8: samples=";
    s += std::to_string(m.samples);
    s += " act_vs_a8 relL2=";
    s += std::to_string(m.act_vs_a8_rel_l2);
    s += " maxAbs=";
    s += std::to_string(m.act_vs_a8_max_abs);
    s += " gross=";
    s += std::to_string(m.act_vs_a8_gross);
    s += "/";
    s += std::to_string(m.act_vs_a8_gross_den);
    s += " act_vs_orig relL2=";
    s += std::to_string(m.act_vs_orig_rel_l2);
    s += " maxAbs=";
    s += std::to_string(m.act_vs_orig_max_abs);
    s += " gross=";
    s += std::to_string(m.act_vs_orig_gross);
    s += "/";
    s += std::to_string(m.act_vs_orig_gross_den);
    s += " quant relL2=";
    s += std::to_string(m.a8_vs_orig_rel_l2);
    s += " maxAbs=";
    s += std::to_string(m.a8_vs_orig_max_abs);
    s += " FAIL=";
    s += (m.ok() ? "0" : "1");
    s += '\n';
    return s;
}

// Complete A8 oracle: independently decode the Q4/Q5 weight, independently
// quantize the BF16 input (exact FP32 values, float quantizer), and compute
// the frozen bound and aggregate metrics over the 31 row picks x up to 15
// unique token picks.
//
//   weight: PackedWeight of QType Q4G64_F16S / Q5G64_F16S (W8G32 spec:
//           qmax=127, group=64), logical K=5120.
//   actual: device int8 projection outputs as the gather_rows sample buffer:
//           [sampleRows, tokens] row-major, length sampleRows*tokens, each
//           sampled row all tokens.
//   x:      BF16 inputs as exact FP32 values, [tokens, K] token-major,
//           length tokens*K. The same x is shared by every sampled row.
//
// Returns the aggregate metrics; ok()==false means at least one gross
// bound violation was observed.
inline A8OracleMetrics compute_a8_oracle(const quantized_weight::PackedWeight& weight,
                                         const std::vector<double>& actual,
                                         const std::vector<float>& x,
                                         std::int32_t weight_row_offset,
                                         std::int32_t rows, std::int32_t tokens) {
    if (weight.weight.qtype != QType::Q4G64_F16S && weight.weight.qtype != QType::Q5G64_F16S) {
        throw std::invalid_argument("a8 oracle: weight must be Q4/Q5 row-split (W8G32 spec)");
    }
    if (weight.weight.shape[1] != kA8K) {
        throw std::invalid_argument("a8 oracle: logical K must be 5120");
    }
    if (rows < kA8SampleRows || tokens <= 0 || weight_row_offset < 0 ||
        weight_row_offset + rows > weight.weight.shape[0]) {
        throw std::invalid_argument("a8 oracle: invalid extents");
    }
    if (static_cast<std::size_t>(actual.size()) !=
        static_cast<std::size_t>(std::min(kA8SampleRows, rows)) * static_cast<std::size_t>(tokens)) {
        throw std::invalid_argument("a8 oracle: actual size mismatch (sampleRows*tokens)");
    }
    if (static_cast<std::size_t>(x.size()) !=
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(kA8K)) {
        throw std::invalid_argument("a8 oracle: input size mismatch (tokens*K)");
    }

    const double u        = std::ldexp(1.0, -24);
    const double gamma160 = 160.0 * u / (1.0 - 160.0 * u);
    const double fmin     = static_cast<double>(FLT_TRUE_MIN);

    const std::vector<std::int32_t> row_picks = sample_rows(rows, kA8SampleRows);
    const std::vector<std::int32_t> tok_picks = token_picks(tokens, kA8TokenPicks);

    // Exact independent weight decode (signed codes x exact FP64 stored FP16
    // scale) over logical K=5120 for every sampled row.
    std::vector<double> wrow(row_picks.size() * static_cast<std::size_t>(kA8K));
    for (std::size_t ri = 0; ri < row_picks.size(); ++ri) {
        const std::int32_t grow = weight_row_offset + row_picks[ri];
        for (std::int32_t c = 0; c < kA8K; ++c) {
            wrow[ri * static_cast<std::size_t>(kA8K) + static_cast<std::size_t>(c)] =
                quantized_weight::logical_weight_fp64(weight, grow, c);
        }
    }

    // Independent float quantizer per token; a token's codes/scales are
    // shared by every output row, mirroring the test fixture (same
    // activation sampled across all output rows).
    std::vector<double> tscale(tokens * static_cast<std::size_t>(kA8Groups));
    std::vector<std::int8_t> tcodes(tokens * static_cast<std::size_t>(kA8K));
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<double> sc(kA8Groups);
        std::vector<std::int8_t> cd(kA8K);
        quantize_input_token(x.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(kA8K),
                             sc, cd);
        std::copy(sc.begin(), sc.end(),
                  tscale.begin() + static_cast<std::size_t>(t) * static_cast<std::size_t>(kA8Groups));
        std::copy(cd.begin(), cd.end(),
                  tcodes.begin() + static_cast<std::size_t>(t) * static_cast<std::size_t>(kA8K));
    }

    const std::size_t sample_points = row_picks.size() * tok_picks.size();
    std::vector<double> sampled_actual(sample_points);
    std::vector<double> a8(sample_points);
    std::vector<double> orig(sample_points);
    std::vector<double> a8b(sample_points);
    std::vector<double> origb(sample_points);
    for (std::size_t ri = 0; ri < row_picks.size(); ++ri) {
        for (std::size_t ti = 0; ti < tok_picks.size(); ++ti) {
            const std::int32_t tp = tok_picks[ti];
            const std::size_t idx = ri * tok_picks.size() + ti;
            const std::size_t actual_idx = ri * static_cast<std::size_t>(tokens) +
                                           static_cast<std::size_t>(tp);
            sampled_actual[idx] = actual[actual_idx];
            double S = 0.0, Eq = 0.0, a8acc = 0.0, origacc = 0.0;
            for (std::int32_t g = 0; g < kA8Groups; ++g) {
                const std::size_t cb = static_cast<std::size_t>(tp) * static_cast<std::size_t>(kA8K) +
                                   static_cast<std::size_t>(g) * kA8GroupSize;
                const double scale = tscale[static_cast<std::size_t>(tp) * static_cast<std::size_t>(kA8Groups) +
                                             static_cast<std::size_t>(g)];
                for (std::int32_t i = 0; i < kA8GroupSize; ++i) {
                    const std::size_t c = static_cast<std::size_t>(g) * kA8GroupSize +
                                          static_cast<std::size_t>(i);
                    const double w    = wrow[ri * static_cast<std::size_t>(kA8K) + c];
                    const double code = static_cast<double>(tcodes[cb + static_cast<std::size_t>(i)]);
                    const double xv   = static_cast<double>(
                        x[static_cast<std::size_t>(tp) * static_cast<std::size_t>(kA8K) + c]);
                    S += std::abs(w) * std::abs(code) * scale;
                    Eq += std::abs(w) * std::abs(xv - code * scale);
                    a8acc += w * code * scale;
                    origacc += w * xv;
                }
            }
            a8[idx]   = a8acc;
            orig[idx] = origacc;
            const double efp32 = gamma160 * S + 160.0 * fmin;
            const double bu    = halfBF16ULP(a8acc);
            a8b[idx]   = efp32 + bu;
            origb[idx] = Eq + efp32 + bu;
        }
    }
    return compute_metrics(sampled_actual, a8, orig, a8b, origb);
}

} // namespace a8_oracle
} // namespace test
} // namespace ninfer