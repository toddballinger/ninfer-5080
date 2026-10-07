// Issue #55 Phase 3 M2 bounded exploration: determine whether the existing
// AppendAtFrontier suffix-prefill machinery already provides the target/MTP
// batched known-delta primitive. This is a mechanism-selection fixture, not a
// production planner or final Phase-3 acceptance test.
//
// The candidate first creates the same retained replacement frontier used by the
// accepted M1 baseline. It then readmits prefix+known_span and requires the
// planner to reuse exactly the retained prefix and execute exactly N suffix
// tokens through AppendAtFrontier. Because N<=128 and prefill_chunk=896, the
// known target delta is one suffix-prefill chunk rather than N decode rounds.
// Continuation output is compared with a fresh independent full-prompt Engine.
//
// Issue #55: when NINFER_ISSUE55_GDN_OBSERVER_IMAGE is set and the span is 8,
// the permanent minimum P62 single-shot GDN-mix formation observer is armed on
// both the candidate AppendAtFrontier engine and the independent FullReset
// reference engine (layer0/gidx0/absolute position 62) and each capture image
// is decoded, format-verified, and persisted; a malformed capture fails the
// fixture.
#include "core/device.h"
#include "targets/registry.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
struct BoundInstanceReader {
    static void* read(const Engine& engine) { return engine.bound_model_instance(); }
};
} // namespace ninfer

// Fixed-schema P62 observer decoder. Rejects malformed metadata before any
// payload byte is read.
namespace ninfer::targets::qwen3_6 {
struct GdnMixInspector {
    static constexpr std::size_t header_bytes = 44, record_bytes = 40;
    struct Record {
        std::string key;
        std::size_t key_bytes, offset, bytes;
        std::uint32_t slot, dtype;
        std::array<std::uint32_t, 4> shape;
    };
    struct Route {
        std::uint32_t magic, phase, target_position, target_gidx, layers,
            record_count, payload_bytes, width, batch, slot, slot_count;
    };
    static std::uint32_t word(const std::vector<std::uint8_t>& image, std::size_t off) {
        if (off > image.size() || 4 > image.size() - off)
            throw std::runtime_error("truncated gdn observer word");
        std::uint32_t v;
        std::memcpy(&v, image.data() + off, 4);
        return v;
    }
    static Route decode_header(const std::vector<std::uint8_t>& image) {
        if (image.size() < 8192 || image.size() > (1U << 24))
            throw std::runtime_error("invalid gdn observer image size");
        Route r{word(image, 0), word(image, 4), word(image, 8), word(image, 12),
                word(image, 16), word(image, 20), word(image, 24), word(image, 28),
                word(image, 32), word(image, 36), word(image, 40)};
        if (r.magic != 0x53455253U || r.target_position != 62 || r.target_gidx != 0 ||
            r.batch != 1 || r.slot != 0 || r.width == 0 || r.width > 896 ||
            r.slot_count == 0 || r.record_count > 32 || r.payload_bytes != image.size() - 8192)
            throw std::runtime_error("invalid gdn observer header");
        return r;
    }
    static std::vector<Record> decode_records(const std::vector<std::uint8_t>& image,
                                              std::uint32_t count) {
        const auto route = decode_header(image);
        if (count != route.record_count) throw std::runtime_error("gdn record count mismatch");
        std::vector<Record> out;
        std::size_t next_key = 4096, next_value = 8192;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::size_t base = header_bytes + record_bytes * i;
            const auto kb = word(image, base), ko = word(image, base + 4);
            Record r{};
            r.key_bytes = kb;
            r.bytes = word(image, base + 8);
            r.offset = word(image, base + 12);
            r.slot = word(image, base + 16);
            r.dtype = word(image, base + 20);
            for (std::size_t d = 0; d < 4; ++d) r.shape[d] = word(image, base + 24 + 4 * d);
            if (kb == 0 || ko != next_key || ko > 8192 || kb > 8192 - ko ||
                r.offset != next_value || r.offset > image.size() ||
                r.bytes > image.size() - r.offset || r.slot != 0 || r.dtype > 2)
                throw std::runtime_error("invalid gdn observer record offsets");
            std::size_t expected = r.dtype == 0 ? 2 : 4;
            for (auto dim : r.shape) {
                if (dim == 0 || expected > image.size() / dim)
                    throw std::runtime_error("invalid gdn observer record shape");
                expected *= dim;
            }
            if (expected != r.bytes) throw std::runtime_error("gdn record shape/bytes mismatch");
            r.key.assign(reinterpret_cast<const char*>(image.data() + ko), kb);
            next_key += kb;
            next_value += r.bytes;
            out.push_back(std::move(r));
        }
        if (next_value != image.size()) throw std::runtime_error("gdn payload end mismatch");
        return out;
    }
};
} // namespace ninfer::targets::qwen3_6

namespace {

using namespace ninfer;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

EngineOptions options(const char* artifact) {
    EngineOptions result;
    result.artifact_path = artifact;
    result.max_context = 4096;
    result.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
    result.max_concurrency = 1;
    result.prefill_chunk = 896;
    result.kv_cache = KvCacheStorage::Int4Group64;
    result.speculative.backend = SpeculativeBackend::Mtp;
    result.speculative.draft_tokens = 3;
    result.speculative.proposal_head = ProposalHead::Optimized;
    result.enable_vision = false;
    result.use_cuda_graph = false;
    result.embedding_host = true;
    return result;
}

RequestOptions generation_options(std::uint32_t outputs) {
    RequestOptions result;
    result.execution.requested_output_tokens = outputs;
    result.execution.allow_prefix_reuse = true;
    result.execution.sampling.temperature = 0.0F;
    result.stop.include_model_defaults = false;
    result.output.raw = true;
    result.output.preserve_special_tokens = true;
    return result;
}

runtime::ResolvedExecutionOptions direct_options(std::uint32_t outputs) {
    runtime::ResolvedExecutionOptions result;
    result.requested_output_tokens = outputs;
    result.allow_prefix_reuse = true;
    result.sampling.temperature = 0.0F;
    return result;
}

std::vector<TokenId> trunk_tokens() {
    std::vector<TokenId> tokens;
    tokens.reserve(64);
    for (std::int32_t i = 0; i < 64; ++i) {
        tokens.push_back(static_cast<TokenId>(1000 + 53 * i));
    }
    return tokens;
}

std::vector<TokenId> known_span(std::size_t n) {
    std::vector<TokenId> tokens;
    tokens.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        tokens.push_back(static_cast<TokenId>(12000 + 17 * static_cast<std::int32_t>(i)));
    }
    return tokens;
}

template <class Program>
std::vector<TokenId> collect_eight_outputs(Program& program,
                                           runtime::PrefillStepResult step) {
    while (!step.complete) {
        step = program.advance_prefill_lane(0);
    }
    require(step.round.tokens.size() == 1,
            "append-prefill did not license the first continuation token");

    std::vector<TokenId> out;
    out.reserve(8);
    out.push_back(step.round.tokens[0]);
    program.resolve_prefill_lane(0, false);

    const std::array<std::uint32_t, 1> lane{0};
    const std::array<std::uint32_t, 1> accepted{1};
    const std::array<std::uint8_t, 1> cancelled{0};
    for (std::uint32_t remaining = 7; remaining != 0; --remaining) {
        const std::array<runtime::RoundBudget, 1> budget{{
            {.generated_tokens_remaining = 1},
        }};
        const auto round = program.decode_batch(lane, budget);
        require(round.row_counts.size() == 1 && round.row_counts[0] == 1 &&
                    round.tokens.size() == round.row_stride && round.row_stride >= 1,
                "append-prefill continuation did not decode one output");
        out.push_back(round.tokens[0]);
        const std::array<std::uint8_t, 1> terminal{
            static_cast<std::uint8_t>(remaining == 1)};
        program.resolve_pending_batch(lane, accepted, terminal, cancelled);
    }
    require(out.size() == 8, "append-prefill continuation did not capture eight outputs");
    return out;
}

void run_span(const char* artifact, std::size_t span, long sequential_us,
              bool& continuation_mismatch) {
    std::vector<TokenId> trunk = trunk_tokens();
    trunk.pop_back();
    const std::vector<TokenId> known = known_span(span);
    std::vector<TokenId> exact_prompt;
    std::vector<TokenId> observed;
    long append_prefill_us = 0;
    std::uint32_t reused = 0;
    std::uint32_t processed = 0;
    const char* observer_path = std::getenv("NINFER_ISSUE55_GDN_OBSERVER_IMAGE");
    const bool observe = span == 8 && observer_path != nullptr && *observer_path != '\0';

    {
        Engine candidate(options(artifact));
        auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
            BoundInstanceReader::read(candidate));
        require(instance != nullptr && instance->program != nullptr,
                "candidate Engine did not bind Qwen3.8 program");
        auto& program = *instance->program;
        if (observe) {
            program.set_gdn_mix_observer(62, 0);
            std::cout << "ISSUE55_GDN_MIX_OBSERVER=ARMED role=candidate "
                         "route=layer0_gidx0_slot0 position=62\n";
        }
        auto initial =
            candidate.generate(candidate.prepare_tokens(trunk, true), generation_options(2));
        require(initial.generated_token_ids.size() == 2,
                "failed to establish retained replacement frontier");
        require(initial.generated_token_ids[1] != known[0],
                "sampled replacement anchor equals known[0]; fixture is not discriminating");

        trunk.push_back(initial.generated_token_ids[0]);
        exact_prompt = trunk;
        exact_prompt.insert(exact_prompt.end(), known.begin(), known.end());

        auto prompt = instance->loaded->frontend.prepare_tokens(exact_prompt, true);
        auto execution = direct_options(8);
        auto base_plan = program.plan_request_base(prompt, execution);
        auto plan = program.plan_request_for_lane(0, prompt, base_plan);
        const auto summary = plan.summary();

        require(summary.reusable_prompt_tokens == trunk.size(),
                "candidate reused more or less than the exact retained prefix");
        require(summary.prompt_tokens == exact_prompt.size(),
                "candidate plan prompt size mismatch");
        require(summary.prompt_tokens - summary.reusable_prompt_tokens == known.size(),
                "candidate suffix work is not exactly the known span");
        reused = summary.reusable_prompt_tokens;

        const auto started = std::chrono::steady_clock::now();
        auto step = program.start_prefill_lane(
            0, std::move(prompt), std::move(plan), runtime::TransientRegion{});
        require(step.summary.prefix_reuse_path == PrefixReusePath::AppendAtFrontier,
                "candidate did not select AppendAtFrontier");
        require(step.summary.reused_prompt_tokens == trunk.size(),
                "candidate begin summary reused more or less than the exact retained prefix");
        require(step.summary.prompt_tokens == exact_prompt.size(),
                "candidate begin summary prompt size mismatch");
        processed += step.processed_prompt_tokens;
        while (!step.complete) {
            step = program.advance_prefill_lane(0);
            processed += step.processed_prompt_tokens;
        }
        const auto finished = std::chrono::steady_clock::now();
        append_prefill_us =
            std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count();
        if (observe) {
            using Runtime = targets::qwen3_6::GdnMixInspector;
            CUDA_CHECK(cudaDeviceSynchronize());
            require(program.gdn_mix_observer_fired(), "P62 gdn observer did not fire");
            const auto bytes = program.gdn_capture_bytes();
            require(bytes >= 8192 && bytes <= (1 << 24), "invalid gdn capture extent");
            std::vector<std::uint8_t> host(static_cast<std::size_t>(bytes));
            std::memcpy(host.data(), program.gdn_capture_image(), host.size());
            const auto route = Runtime::decode_header(host);
            const auto records = Runtime::decode_records(host, route.record_count);
            require(route.phase == 9 && records.size() == 21,
                    "unexpected P62 formation route or record count");
            const std::array<const char*, 21> required_keys{
                "positions", "lane_slot", "valid_columns", "x_entry", "conv_pre", "rec_pre",
                "conv_weight", "h", "g", "beta", "record_conv", "q", "k", "v", "gate",
                "conv_post", "record_key", "record_value", "record_gate", "out", "rec_post"};
            for (std::size_t i = 0; i < required_keys.size(); ++i)
                require(records[i].key == required_keys[i], "P62 observer key mismatch");
            std::cout << "ISSUE55_GDN_MIX_OBSERVER=FIRED role=candidate position="
                      << route.target_position
                      << " gidx=" << route.target_gidx
                      << " phase_action=" << route.phase
                      << " width=" << route.width
                      << " batch=" << route.batch
                      << " slot=" << route.slot
                      << " slot_count=" << route.slot_count
                      << " records=" << route.record_count
                      << " bytes=" << host.size() << '\n';
            for (const auto& r : records)
                std::cout << "ISSUE55_GDN_MIX_OBSERVER_RECORD=" << r.key
                          << " value_offset=" << r.offset
                          << " value_bytes=" << r.bytes
                          << " dtype=" << r.dtype
                          << " shape=" << r.shape[0] << ','
                          << r.shape[1] << ','
                          << r.shape[2] << ','
                          << r.shape[3] << '\n';
            std::ofstream output(observer_path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(host.data()), host.size());
            output.close();
            require(static_cast<bool>(output), "failed to persist gdn capture image");
            std::cout << "ISSUE55_GDN_MIX_OBSERVER_IMAGE=" << observer_path << '\n';
        }

        require(processed == known.size(),
                "AppendAtFrontier replayed tokens outside the known delta");
        observed = collect_eight_outputs(program, std::move(step));
    }

    std::vector<TokenId> expected;
    {
        Engine reference(options(artifact));
        auto* instance = static_cast<targets::Qwen3_6_27BInstance*>(
            BoundInstanceReader::read(reference));
        require(instance != nullptr && instance->program != nullptr,
                "reference Engine did not bind Qwen3.8 program");
        auto& program = *instance->program;
        if (observe) {
            program.set_gdn_mix_observer(62, 0);
            std::cout << "ISSUE55_GDN_MIX_OBSERVER=ARMED role=reference "
                         "route=layer0_gidx0_slot0 position=62\n";
        }
        auto result =
            reference.generate(reference.prepare_tokens(exact_prompt, true), generation_options(8));
        require(result.prefix_reuse_path == PrefixReusePath::FullReset,
                "fresh independent reference unexpectedly reused resident state");
        require(result.generated_token_ids.size() == 8,
                "fresh independent reference did not return eight outputs");
        if (observe) {
            std::string observer_ref_path = std::string(observer_path) + "_reference";
            using Runtime = targets::qwen3_6::GdnMixInspector;
            CUDA_CHECK(cudaDeviceSynchronize());
            require(program.gdn_mix_observer_fired(),
                    "reference P62 gdn observer did not fire");
            const auto bytes = program.gdn_capture_bytes();
            require(bytes >= 8192 && bytes <= (1 << 24),
                    "invalid reference gdn capture extent");
            std::vector<std::uint8_t> host(static_cast<std::size_t>(bytes));
            std::memcpy(host.data(), program.gdn_capture_image(), host.size());
            const auto route = Runtime::decode_header(host);
            const auto records = Runtime::decode_records(host, route.record_count);
            require(route.phase == 1 && records.size() == 17,
                    "unexpected reference P62 formation route or record count");
            const std::array<const char*, 17> required_keys{
                "positions", "lane_slot", "valid_columns", "x_entry", "conv_pre", "rec_pre",
                "conv_weight", "h", "g", "beta", "q", "k", "v", "gate", "conv_post",
                "out", "rec_post"};
            for (std::size_t i = 0; i < required_keys.size(); ++i)
                require(records[i].key == required_keys[i],
                        "reference P62 observer key mismatch");
            require(records[0].key == "positions" && records[0].bytes == 4 &&
                        Runtime::word(host, records[0].offset) == 62,
                    "reference positions metadata is not truthful");
            require(records[1].key == "lane_slot" && records[1].bytes == 4 &&
                        Runtime::word(host, records[1].offset) == 0,
                    "reference lane slot metadata must be slot0");
            require(records[2].key == "valid_columns" && records[2].bytes == 4 &&
                        Runtime::word(host, records[2].offset) == route.width,
                    "reference valid columns metadata is not truthful");
            std::cout << "ISSUE55_GDN_MIX_OBSERVER=FIRED role=reference position="
                      << route.target_position
                      << " gidx=" << route.target_gidx
                      << " phase_action=" << route.phase
                      << " width=" << route.width
                      << " batch=" << route.batch
                      << " slot=" << route.slot
                      << " slot_count=" << route.slot_count
                      << " records=" << route.record_count
                      << " bytes=" << host.size() << '\n';
            for (const auto& r : records)
                std::cout << "ISSUE55_GDN_MIX_OBSERVER_REFERENCE_RECORD=" << r.key
                          << " value_offset=" << r.offset
                          << " value_bytes=" << r.bytes
                          << " dtype=" << r.dtype
                          << " shape=" << r.shape[0] << ','
                          << r.shape[1] << ','
                          << r.shape[2] << ','
                          << r.shape[3] << '\n';
            std::ofstream output(observer_ref_path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(host.data()), host.size());
            output.close();
            require(static_cast<bool>(output),
                    "failed to persist reference gdn capture image");
            std::cout << "ISSUE55_GDN_MIX_OBSERVER_IMAGE_CANDIDATE=" << observer_path
                      << '\n';
            std::cout << "ISSUE55_GDN_OBSERVER_IMAGE_REFERENCE=" << observer_ref_path
                      << '\n';
        }
        expected = std::move(result.generated_token_ids);
    }

    // The only non-fatal check in the fixture: the original continuation
    // mismatch is reported per span and re-emitted once after all spans have
    // run (original terminal expected-mismatch behavior).
    if (observed != expected) {
        std::cout << "M2_CONTINUATION_MISMATCH=" << span << '\n';
        continuation_mismatch = true;
    }

    const double speedup =
        append_prefill_us > 0 ? static_cast<double>(sequential_us) / append_prefill_us : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "ISSUE55_PHASE3_M2_SPAN=" << span << "\n"
              << "ISSUE55_PHASE3_M2_SEQUENTIAL_BASELINE_US=" << sequential_us << "\n"
              << "ISSUE55_PHASE3_M2_APPEND_PREFILL_US=" << append_prefill_us << "\n"
              << "ISSUE55_PHASE3_M2_SPEEDUP_VS_SEQUENTIAL=" << speedup << "\n"
              << "ISSUE55_PHASE3_M2_REUSED_PREFIX_TOKENS=" << reused << "\n"
              << "ISSUE55_PHASE3_M2_EXECUTED_SUFFIX_TOKENS=" << processed << "\n"
              << "ISSUE55_PHASE3_M2_NO_OLD_PREFIX_REPLAY=PASS\n"
              << "ISSUE55_PHASE3_M2_APPEND_AT_FRONTIER=PASS\n"
              << "ISSUE55_PHASE3_M2_INDEPENDENT_REFERENCE_MATCH="
              << (observed == expected ? "YES" : "NO") << '\n';
}

int run(const char* artifact) {
    constexpr std::array<std::size_t, 5> spans{1, 4, 8, 32, 128};
    constexpr std::array<long, 5> m1_us{25495, 99810, 199316, 802715, 3233573};
    bool continuation_mismatch = false;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        run_span(artifact, spans[i], m1_us[i], continuation_mismatch);
    }
    if (continuation_mismatch) {
        throw std::runtime_error(
            "AppendAtFrontier batched suffix continuation differs from independent target "
            "reference");
    }
    std::cout << "ISSUE55_PHASE3_M2_APPEND_PREFILL_EXPLORATION=PASS\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_QWEN3_8_27B_DECISION_WEIGHTS is not set\n";
        return 77;
    }
    try {
        return run(artifact);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
