// Standalone CUDA occupancy diagnostic probe for the Q4 (INT4-G64) GQA decode
// kernel templates (NInfer issue #47: RTX 5080 128K Q4-GQA decode qualification).
//
// WHAT IT MEASURES
//   Instantiates, in this translation unit, the two Q4 GQA decode specializations
//   the NInfer RTX 5080 production path selects at a 131072-token (128K) context
//   for the Gqa27 target (Qwen3.8-27B class: 24 q-heads / 4 KV-heads,
//   DecodeSplits=85), exactly as src/ops/launcher/gqa_attention_decode.cu does:
//
//     T=1 cached   gqa_attention_decode_q4_tiled_kernel<Gqa27, 1, 8, 2, 32,
//                  false, false, false, GqaCachedInput>
//     T=4 append   gqa_attention_decode_q4_tiled_kernel<Gqa27, 4, 8, 2, 32,
//                  false, false, true,  GqaAppendInput>
//
//   (Template order: Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock,
//   DynamicArena, MultiBatch, Masked, InputType. Both are 256 threads with
//   __launch_bounds__(256, 2), key block 32, no dynamic arena, single batch;
//   the cached route is unmasked, the append route is masked.)
//
//   For each specialization it reports:
//     * device properties (name, CC, SM count, VRAM, per-block shared memory);
//     * cudaFuncGetAttributes (CUDA 13.3 cudaFuncAttributes fields):
//       numRegs (32-bit registers per thread, already reflecting the
//       __launch_bounds__(256,2) ptxas clamp), sharedSizeBytes (static smem;
//       the pre-13 name staticSmemBytes is renamed in CUDA 13),
//       maxDynamicSharedSizeBytes, maxThreadsPerBlock. The block's thread
//       count is the explicit launch size (WarpsPerCta x 32 = 256); the
//       cudaFuncAttributes of CUDA 13.3 has no per-kernel thread-count field;
//     * cudaOccupancyMaxActiveBlocksPerMultiprocessor at blockSize 256 with
//       dynamicSMemSize 0 (the production launch; the driver-admitted maximum
//       co-resident CTAs per SM - the launch bounds only guarantee a minimum
//       of 2, this result is the maximum ceiling the driver admits);
//     * theoretical wave counts for split counts 85/64/52/42: the production
//       grid is (KVHeads, splits, 1) at batch 1, so blocks = 4 x splits and
//       waves = ceil(blocks / (SM count x max blocks per SM)).
//
//   All CUDA calls are return-code checked; any nonzero error aborts with a
//   diagnostic and nonzero exit. Output is machine-readable key=value pairs
//   under '#' comment lines; parse the kernel blocks directly.
//
// BUILD (match the production ptxas context: same toolkit, same real arch, no
// -Xptxas codegen overrides; -lineinfo is a no-op for the measured attributes):
//
//   nvcc -O3 -std=c++20 -arch=sm_120a -I <repo>/src \
//        tools/probe/issue47_occupancy_probe.cu -o issue47_occupancy_probe
//
//   Target: RTX 5080 / sm_120a. The probe prints the driver-reported CUDA
//   version at run time; the report must record the toolkit as `nvcc --version`
//   reports it (the installed directory name and the documentation are not
//   the authority). If a different toolkit or a virtual-arch '-gencode' mix is
//   used, the register/occupancy numbers are for that other ptxas context, not
//   the production one; record the exact command line in the report.
//   (-Xptxas informational flags such as -v do not change codegen; codegen
//   -changing -Xptxas overrides must be avoided to match the production build.)
//
// RUN:
//   issue47_occupancy_probe [--splits 85,64,52,42]
//
//   --splits: comma list of split counts to wave-solve (default 85,64,52,42;
//   85 is the production active split count at 131072 tokens for Gqa27).
//
// NON-RDC vs RDC COMPILE MODE (the probe compiles the kernel in its own single TU)
//   This probe measures the NON-RDC, single-TU build only. Both modes run ptxas
//   over the per-TU PTX, so for identical source + arch + toolkit the per-TU
//   cubin (hence register counts, static shared memory, and the driver-admitted
//   CTA count per SM) is expected to be the same in either mode - but that
//   equivalence is NOT established by this probe and must not be reported as
//   measured; if the qualification gate needs the production RDC build
//   (CUDA_SEPARABLE_COMPILATION ON + device link), probe that build separately
//   and diff the numbers.
//
//   non-RDC (nvcc default; this probe's design):
//     * a single TU defines the kernel, so &kernel<spec> is a real module
//       symbol; no device link is involved;
//     * with -arch=sm_120a, nvcc ptxas's the per-TU PTX to cubin and embeds it;
//       on an RTX 5080 (sm_120a) the driver loads that cubin directly (no JIT);
//     * cudaFuncGetAttributes / cudaOccupancyMaxActiveBlocksPerMultiprocessor
//       resolve the function pointer straight out of the embedded module.
//
//   RDC (-rdc=true + device link; the production ninfer_ops archive is built
//   with CUDA_SEPARABLE_COMPILATION ON / CUDA_RESOLVE_DEVICE_SYMBOLS ON):
//     * per-TU device code is a relocatable ELF; cudart --device-link (with
//       --internal-node-list when resolving device symbols) finalizes the cubin;
//     * the host-side &kernel<spec> pointer points at the RDC stub, and the
//       runtime maps it to the real kernel through the demangled symbol in the
//       device symbol table (this is the mechanism the production launcher's
//       cudaFuncSetAttribute on the 128K Q4 kernels relies on); attribute
//       queries on such a stub then hit the device-linked cubin.
//
//   The two modes diverge in codegen only if the ptxas input or flags differ
//   (different -arch, different -Xptxas, different toolkit version). The Q4 GQA
//   kernel has no relocatable-code-sensitive feature - unlike the setmaxnreg
//   warp-specialized nvfp4 TMA kernels, which is why the production build
//   compiles that one target in non-RDC.
//
// PROVENANCE (state in the report alongside this probe's output)
//   * the exact nvcc command line and the CUDA toolkit exactly as `nvcc
//     --version` reports it (the report must cite that command's output; the
//     probe prints the driver-reported version at run time, which is the
//     driver, not the toolkit, and is not a substitute);
//   * the repo commit the kernel headers were read from - export
//     NINFER_REPO_REV=<sha> to have the probe print it, otherwise it prints
//     "unknown" and the report must state the commit explicitly;
//   * the probe's own commit (NINFER_PROBE_REV=<sha>);
//   * hardware: the probe prints the detected device name/CC/SM count/VRAM; the
//     qualification target is the RTX 5080 16 GB, sm_120a, 84 SMs - verify the
//     printed sm_count matches the qualification host before using the numbers.

#include <cuda_runtime.h>

#include "ops/kernel/gqa_attention_decode_q4.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#define PROBE_CUDA_CHECK(expr)                                                                     \
    do {                                                                                           \
        const cudaError_t probe_err__ = (expr);                                                    \
        if (probe_err__ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error %d (%s) at %s:%d: %s\n", (int)probe_err__,           \
                         cudaGetErrorString(probe_err__), __FILE__, __LINE__, #expr);              \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace ninfer_probe {

using Geometry = ninfer::ops::Gqa27Geometry;
using Cached   = ninfer::ops::GqaCachedInput;
using Append   = ninfer::ops::GqaAppendInput;

struct KernelDescriptor {
    const char* label;
    const char* input_type;
    int tokens;
    int tokens_field; // production tokens argument for this route
    bool masked;
    int kv_heads              = Geometry::KVHeads;
    int q_heads               = Geometry::QHeads;
    int decode_splits_cap     = Geometry::DecodeSplits;
    int warps_per_cta         = 8;
    int min_blocks_sm         = 2; // __launch_bounds__(256, 2)
    int key_block             = 32;
    bool dynamic_arena        = false;
    bool multi_batch          = false;
    const void* func          = nullptr;
};

// Mirror of the production gqa_small_t_active_splits<Geometry, true> policy for
// window > 16390 (the 128K arm): div_up(window, 480 / DecodeSplitScale) clamped
// to [4, DecodeSplits]; Gqa27 has DecodeSplitScale == 1.
int active_splits_policy(int window)
{
    int splits = (window + 480 - 1) / 480;
    splits = std::max(splits, 4);
    return std::min(splits, Geometry::DecodeSplits);
}

// Theoretical wave count: blocks = KVHeads x splits (batch 1); capacity =
// SM count x driver-admitted max co-resident blocks per SM.
long long waves(int blocks, int sm_count, int blocks_per_sm)
{
    const long long capacity = std::max(1, sm_count * blocks_per_sm);
    return (blocks + capacity - 1) / capacity;
}

// Geometry defaults (kv_heads, q_heads, decode_splits_cap, warps, min blocks,
// key block, arena, multi-batch) come from the struct's default member
// initializers; only the per-instantiation fields are set here.
KernelDescriptor make_descriptor(const char* label, const char* input_type, int tokens,
                                 int tokens_field, bool masked, const void* func)
{
    KernelDescriptor kd;
    kd.label = label;
    kd.input_type = input_type;
    kd.tokens = tokens;
    kd.tokens_field = tokens_field;
    kd.masked = masked;
    kd.func = func;
    return kd;
}

} // namespace ninfer_probe

namespace {

using namespace ninfer_probe;

void print_header(const cudaDeviceProp& p, int driver_version, const char* repo_rev,
                  const char* probe_rev, const std::string& splits_spec)
{
    char timebuf[64] = "";
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm gm{};
#ifdef _WIN32
    gmtime_s(&gm, &t);
#else
    gmtime_r(&t, &gm);
#endif
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", &gm);

    // Driver API version encodes major*1000 + minor*10 (+ patch*1): 13030 => 13.3.0.
    const int major = driver_version / 1000;
    const int minor = (driver_version % 1000) / 10;
    const int patch = (driver_version % 1000) % 10;

    std::printf("# ninfer issue47 Q4-GQA decode occupancy probe (Gqa27, 128K)\n");
    std::printf("# generated_utc=%s\n", timebuf);
    std::printf("# repo_commit=%s\n", repo_rev);
    std::printf("# probe_commit=%s\n", probe_rev);
    std::printf("# driver_api_version=%d\n", driver_version);
    std::printf("# cuda_version_decoded=%d.%d.%d\n", major, minor, patch);
    std::printf("# device_name=%s\n", p.name);
    std::printf("# device_cc=%d.%d\n", p.major, p.minor);
    std::printf("# sm_count=%d\n", p.multiProcessorCount);
    std::printf("# total_vram_bytes=%llu\n", (unsigned long long)p.totalGlobalMem);
    std::printf("# max_threads_per_block=%d\n", p.maxThreadsPerBlock);
    std::printf("# max_shared_mem_per_block=%zu\n", p.sharedMemPerBlock);
    std::printf("# shared_mem_per_sm=%zu\n", p.sharedMemPerMultiprocessor);
    std::printf("# max_shared_mem_per_block_optin=%zu\n", p.sharedMemPerBlockOptin);
    std::printf("# warp_size=%d\n", p.warpSize);
    std::printf("# context_window=131072\n");
    std::printf("# splits_active_at_131072_production=%d\n", active_splits_policy(131072));
    std::printf("# splits_list=%s\n", splits_spec.c_str());
    std::printf("# compile_contract=non-RDC single-TU, -arch=sm_120a, -std=c++20, -O3; "
                "record the exact command line and the `nvcc --version` output in the report\n");
    std::fflush(stdout);
}

void print_kernel(const KernelDescriptor& kd, int sm_count, const std::vector<int>& splits)
{
    cudaFuncAttributes attr;
    PROBE_CUDA_CHECK(cudaFuncGetAttributes(&attr, kd.func));

    const int block_size = kd.warps_per_cta * 32; // 8 warps x 32 = 256 threads
    int blocks_per_sm = 0;
    // CUDA 13.3 signature: (int*, const void*, int blockSize, size_t
    // dynamicSMemSize). dynamicSMemSize is 0: both probed specializations use
    // no dynamic shared memory (DynamicArena=false), matching the production
    // launch.
    PROBE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kd.func,
                                                                   block_size,
                                                                   static_cast<size_t>(0)));

    std::printf("# kernel: %s\n", kd.label);
    std::printf("name=gqa_attention_decode_q4_tiled_kernel\n");
    std::printf("geometry=Gqa27\n");
    std::printf("q_heads=%d\n", kd.q_heads);
    std::printf("kv_heads=%d\n", kd.kv_heads);
    std::printf("decode_splits_cap=%d\n", kd.decode_splits_cap);
    std::printf("tokens=%d\n", kd.tokens);
    std::printf("tokens_arg_production=%d\n", kd.tokens_field);
    std::printf("masked=%d\n", kd.masked ? 1 : 0);
    std::printf("multi_batch=%d\n", kd.multi_batch ? 1 : 0);
    std::printf("input_type=%s\n", kd.input_type);
    std::printf("warps_per_cta=%d\n", kd.warps_per_cta);
    std::printf("min_blocks_per_sm_launch_bounds=%d\n", kd.min_blocks_sm);
    std::printf("key_block=%d\n", kd.key_block);
    std::printf("dynamic_arena=%d\n", kd.dynamic_arena ? 1 : 0);
    std::printf("launch_block_size=%d\n", block_size);
    std::printf("num_regs_per_thread=%d\n", attr.numRegs);
    std::printf("static_smem_bytes=%llu\n", (unsigned long long)attr.sharedSizeBytes);
    std::printf("max_threads_per_block=%d\n", attr.maxThreadsPerBlock);
    std::printf("max_dynamic_smem_bytes=%d\n", attr.maxDynamicSharedSizeBytes);
    std::printf("local_size_bytes=%llu\n", (unsigned long long)attr.localSizeBytes);
    std::printf("max_blocks_per_sm=%d\n", blocks_per_sm);
    std::printf("regs_per_cta_words=%lld\n", (long long)attr.numRegs * block_size);
    std::printf("regs_per_cta_bytes=%lld\n", 4LL * attr.numRegs * block_size);
    // Derived, not a driver attribute: the per-SM 32-bit register file size is
    // not exposed by the runtime API; this is the capacity implied by the
    // driver-admitted CTA count and the measured per-CTA register demand.
    std::printf("implied_sm_register_file_bytes=%lld\n",
                4LL * blocks_per_sm * attr.numRegs * block_size);
    std::printf("batch=1\n");
    std::printf("grid_x_kv_heads=%d\n", kd.kv_heads);
    std::fflush(stdout);

    for (const int s : splits) {
        const int blocks = kd.kv_heads * s; // grid (kv_heads, s, 1), batch 1
        const long long w = waves(blocks, sm_count, blocks_per_sm);
        std::printf("waves_split_%d=%lld\n", s, w);
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv)
{
    // Parse the split list (default 85,64,52,42).
    std::string splits_spec = "85,64,52,42";
    std::vector<int> splits{85, 64, 52, 42};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--splits") != 0 || i + 1 >= argc) { continue; }
        splits_spec = argv[++i];
        splits.clear();
        std::istringstream ss(splits_spec);
        std::string item;
        bool bad = false;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) { continue; }
            char* end = nullptr;
            const long v = std::strtol(item.c_str(), &end, 10);
            if (end == item.c_str() || (end != nullptr && *end != '\0') || v <= 0 ||
                v > 1000000) {
                bad = true;
                break;
            }
            splits.push_back(static_cast<int>(v));
        }
        if (bad || splits.empty()) {
            std::fprintf(stderr, "invalid --splits value '%s'\n", splits_spec.c_str());
            return 2;
        }
    }

    const char* repo_rev  = std::getenv("NINFER_REPO_REV");
    const char* probe_rev = std::getenv("NINFER_PROBE_REV");

    // Create the primary context on the default device (the only context work
    // this probe does; it never launches kernels or allocates device memory).
    PROBE_CUDA_CHECK(cudaFree(0));

    int driver_version = 0;
    PROBE_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    cudaDeviceProp prop;
    PROBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    print_header(prop, driver_version, repo_rev != nullptr ? repo_rev : "unknown",
                 probe_rev != nullptr ? probe_rev : "unknown", splits_spec);

    // The two production 128K specializations (the &kernel<spec> idiom mirrors
    // src/ops/launcher/gqa_attention_decode.cu, which runs the same
    // cudaFuncSetAttribute/cudaFuncGetAttributes pattern on these kernels in
    // both the RDC ninfer_ops and non-RDC ninfer_nvfp4_tma archives; see the
    // file header for the non-RDC vs RDC discussion).
    const KernelDescriptor t1_cached = make_descriptor(
        "q4_tiled_T1_cached (Gqa27, 128K, production)", "GqaCachedInput", 1, 1, false,
        reinterpret_cast<const void*>(
            &ninfer::ops::gqa_attention_decode_q4_tiled_kernel<
                Geometry, 1, 8, 2, 32, false, false, false, Cached>));
    const KernelDescriptor t4_append = make_descriptor(
        "q4_tiled_T4_masked_append (Gqa27, 128K, production)", "GqaAppendInput", 4, 4, true,
        reinterpret_cast<const void*>(
            &ninfer::ops::gqa_attention_decode_q4_tiled_kernel<
                Geometry, 4, 8, 2, 32, false, false, true, Append>));

    for (const auto& kd : {t1_cached, t4_append}) {
        print_kernel(kd, prop.multiProcessorCount, splits);
    }

    std::printf("# done\n");
    std::fflush(stdout);
    return 0;
}