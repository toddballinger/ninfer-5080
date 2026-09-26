#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    std::uint32_t vision_max_tokens = 0; // 0 => automatic Vision budget
    bool embedding_host = false;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

// Build the standalone-CLI EngineOptions from a parsed CLI Options value, applying the
// exact handoff used by apps/cli/main.cpp. The load-progress callback is the single
// production field that cannot be derived from the CLI; callers without a renderer
// (e.g. tests) pass std::nullopt to leave load_progress at its default.
[[nodiscard]] EngineOptions engine_options_from_cli(
    const Options& cli, std::optional<LoadProgress> load_progress = std::nullopt);


} // namespace ninfer::cli
