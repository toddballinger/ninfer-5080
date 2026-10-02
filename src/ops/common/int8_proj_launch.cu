#include "ops/common/int8_proj_launch.h"

#include "ops/common/act_quant_g64.h"
#include "ops/common/int8_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <mutex>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {
namespace {

    constexpr std::int32_t kWideRowThreshold = 2048;

    using Q4Wide   = Int8RowSplitGemmSchedule<Int8GemmCodec::Q4, 128, 128, 32, 64, 3, 1>;
    using Q4Narrow = Int8RowSplitGemmSchedule<Int8GemmCodec::Q4, 64, 64, 16, 64, 3, 1>;
    using Q5Wide   = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 128, 128, 32, 64, 3, 1>;
    using Q5Narrow = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 64, 64, 16, 64, 3, 1>;

// Opt-in aggregate counters; storage and output are private to this translation unit.
// Fixed capacity, no event payload. Overflow invalidates full-coverage claims.
struct TraceKey {
    std::int32_t family, qtype, rows, k, padded_k, tile_tokens, wide, full;
    bool operator==(const TraceKey& o) const noexcept {
        return family == o.family && qtype == o.qtype && rows == o.rows &&
               k == o.k && padded_k == o.padded_k &&
               tile_tokens == o.tile_tokens && wide == o.wide && full == o.full;
    }
};
struct TraceTable {
    struct Entry { TraceKey key; std::int64_t count; };
    static constexpr std::size_t cap = 4096;
    std::mutex mutex;
    std::int64_t invocations[3]{};
    std::int64_t overflow{};
    Entry jobs[cap]{};
    Entry gemms[cap]{};
    std::size_t job_size{}, gemm_size{};
};
TraceTable& trace_table() {
    static TraceTable table;
    return table;
}
void trace_summary() {
    auto& t = trace_table();
    std::lock_guard<std::mutex> lock(t.mutex);
    std::fprintf(stderr, "[int8-proj-trace] launcher_invocations attn=%lld gdn=%lld\n",
                 static_cast<long long>(t.invocations[1]),
                 static_cast<long long>(t.invocations[2]));
    for (std::size_t i = 0; i < t.job_size; ++i) {
        const auto& e = t.jobs[i];
        const auto& k = e.key;
        std::fprintf(stderr,
            "[int8-proj-trace] job fam=%d qtype=%d rows=%d K=%d count=%lld\n",
            k.family, k.qtype, k.rows, k.k, static_cast<long long>(e.count));
    }
    for (std::size_t i = 0; i < t.gemm_size; ++i) {
        const auto& e = t.gemms[i];
        const auto& k = e.key;
        std::fprintf(stderr,
            "[int8-proj-trace] gemm fam=%d qtype=%d rows=%d K=%d paddedK=%d tileTokens=%d wide=%d full=%d count=%lld\n",
            k.family, k.qtype, k.rows, k.k, k.padded_k, k.tile_tokens,
            k.wide, k.full, static_cast<long long>(e.count));
    }
    std::fprintf(stderr,
        "[int8-proj-trace] overflow=%lld complete_coverage=%d\n",
        static_cast<long long>(t.overflow), t.overflow == 0 ? 1 : 0);
}
bool trace_armed() {
    // Static local initialization is once-only and thread-safe. Construct the
    // table before registering its exit reader, ensuring valid lifetime.
    static const bool armed = [] {
        const char* env = std::getenv("NINFER_INT8_PROJ_TRACE");
        if (!env || std::strcmp(env, "1") != 0) return false;
        (void)trace_table();
        return std::atexit(trace_summary) == 0;
    }();
    return armed;
}
void trace_record(bool gemm, const TraceKey& key) {
    if (!trace_armed()) return;
    auto& t = trace_table();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto* entries = gemm ? t.gemms : t.jobs;
    auto& size = gemm ? t.gemm_size : t.job_size;
    for (std::size_t i = 0; i < size; ++i) {
        if (entries[i].key == key) { ++entries[i].count; return; }
    }
    if (size == TraceTable::cap) { ++t.overflow; return; }
    entries[size++] = {key, 1};
}
    template <class Cfg, bool Full>
    void launch_job(std::int32_t family, const Int8ProjJob& job, const std::int8_t* xq,
                    const float* xs, std::int32_t tokens, std::int32_t padded_k,
                    std::int32_t exact_k, cudaStream_t stream) {
        static const bool configured = [] {
            CUDA_CHECK(cudaFuncSetAttribute(
                int8_rowsplit_gemm_kernel<Cfg, Full, Int8GemmEpilogue::Store>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSharedBytes));
            return true;
        }();
        (void)configured;

        const dim3 grid(static_cast<unsigned>(div_up(job.rows, Cfg::kBlockRows)),
                        static_cast<unsigned>(div_up(tokens, Cfg::kBlockCols)));
        // GEMM job x tile at the actual kernel-launch site: exact dimensions,
        // Wide/Narrow by real rows, Full-or-tail by the real Full parameter.
        int8_rowsplit_gemm_kernel<Cfg, Full, Int8GemmEpilogue::Store>
            <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(
            xq, xs, job.codes, job.high, job.scales, job.out, job.rows, tokens,
            padded_k, job.out_row_stride);
        CUDA_CHECK(cudaGetLastError());
        // Count a successfully enqueued GEMM, not an attempted launch.
        trace_record(true, {family, static_cast<std::int32_t>(job.q5), job.rows,
                            exact_k, padded_k, tokens,
                            static_cast<std::int32_t>(job.rows >= kWideRowThreshold),
                            Full ? 1 : 0});
    }

    template <class Cfg>
    void launch_sized(std::int32_t family, const Int8ProjJob& job, const std::int8_t* xq,
                      const float* xs, std::int32_t tokens, std::int32_t padded_k,
                      std::int32_t exact_k, cudaStream_t stream) {
        const bool full = (tokens % Cfg::kBlockCols) == 0 &&
                          (job.rows % Cfg::kBlockRows) == 0;
        if (full) {
            launch_job<Cfg, true>(family, job, xq, xs, tokens, padded_k, exact_k, stream);
        } else {
            launch_job<Cfg, false>(family, job, xq, xs, tokens, padded_k, exact_k, stream);
        }
    }
    } // namespace

    void int8_proj_prewarm() {
        // Q4 wide.
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q4Wide, true, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q4Wide::kSharedBytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q4Wide, false, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q4Wide::kSharedBytes));
        // Q4 narrow.
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q4Narrow, true, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q4Narrow::kSharedBytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q4Narrow, false, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q4Narrow::kSharedBytes));
        // Q5 wide.
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q5Wide, true, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q5Wide::kSharedBytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q5Wide, false, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q5Wide::kSharedBytes));
        // Q5 narrow.
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q5Narrow, true, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q5Narrow::kSharedBytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            int8_rowsplit_gemm_kernel<Q5Narrow, false, Int8GemmEpilogue::Store>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q5Narrow::kSharedBytes));
    }

    void int8_proj_launch(const Tensor& x, const Int8ProjJob* jobs, int job_count,
                           Int8ProjFamily family, const Int8ProjWorkspace& scratch,
                           cudaStream_t stream) {
        const bool tracing = trace_armed();
        const std::int32_t fam = static_cast<std::int32_t>(family);
        const std::int32_t k        = x.ne[0];
        const std::int32_t padded_k = int8_proj_padded_k(k);
        const std::int32_t tile     = int8_proj_token_tile(x.ne[1]);
        if (tracing) {
            auto& counters = trace_table();
            {
                std::lock_guard<std::mutex> lock(counters.mutex);
                ++counters.invocations[fam];
            }
            // One submitted job per wrapper call, independent of token tiles.
            for (int i = 0; i < job_count; ++i) {
                const auto& job = jobs[i];
                trace_record(false, {fam, static_cast<std::int32_t>(job.q5),
                                     job.rows, k, 0, 0, 0, 0});
            }
        }

        for (std::int32_t offset = 0; offset < x.ne[1]; offset += tile) {
            const std::int32_t count = std::min(tile, x.ne[1] - offset);
            const Tensor x_slice     = x.slice(1, offset, count);
            act_quant_g64_launch(static_cast<const __nv_bfloat16*>(x_slice.data),
                                 scratch.codes, scratch.scales, k, count, padded_k,
                                 stream);
            for (int i = 0; i < job_count; ++i) {
                Int8ProjJob job = jobs[i];
                job.out += static_cast<std::int64_t>(offset) * job.out_row_stride;
                const bool wide = job.rows >= kWideRowThreshold;
                if (job.q5) {
                    if (wide) {
                        launch_sized<Q5Wide>(fam, job, scratch.codes, scratch.scales,
                                            count, padded_k, k, stream);
                    } else {
                        launch_sized<Q5Narrow>(fam, job, scratch.codes, scratch.scales,
                                               count, padded_k, k, stream);
                    }
                } else {
                    if (wide) {
                        launch_sized<Q4Wide>(fam, job, scratch.codes, scratch.scales,
                                             count, padded_k, k, stream);
                    } else {
                        launch_sized<Q4Narrow>(fam, job, scratch.codes, scratch.scales,
                                               count, padded_k, k, stream);
                    }
                }
            }
        }
    }

    } // namespace ninfer::ops::detail
