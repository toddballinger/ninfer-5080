#include "cli/options.h"
#include "serve/serve_options.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_35b_a3b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_35b_a3b_vision_planner_test
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/layouts_impl.h"
#include "targets/qwen3_6/impl/runtime/vision_context_impl.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::cli::Options;
using ninfer::serve::ServeOptions;
namespace Runtime = ninfer::targets::qwen3_6::detail::qwen3_6_35b_a3b_vision_planner_test;

int check(bool condition, std::string_view message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

Options cli_parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

ServeOptions serve_parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

std::unique_ptr<Runtime::SequencePlanImpl>
build_test_plan(std::uint32_t capacity, std::uint32_t vision_max_tokens) {
    Runtime::SequencePlanningInputs inputs{
        .weights_profile =
            ninfer::targets::qwen3_6_35b_a3b::detail::WeightsProfile::GroupwiseInt,
        .capacity            = capacity,
        .max_concurrency     = 1,
        .prefill_chunk       = std::min(std::uint32_t{1024}, capacity),
        .vision_max_tokens   = vision_max_tokens,
        .draft_window        = 0,
        .speculative_backend = ninfer::SpeculativeBackend::None,
        .kv_dtype            = ninfer::DType::BF16,
        .kv_quant_group      = 0,
        .proposal_head       = ninfer::ProposalHead::Full,
        .features            = {.vision = true},
        .use_cuda_graph      = false,
        .device              = 0,
    };
    const std::uint32_t logical_pages =
        1U + (capacity - 1U) / static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
    return Runtime::build_sequence_candidate(inputs, logical_pages);
}

} // namespace

int main() {
    int failures = 0;

    const Options cli_default =
        cli_parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures += check(cli_default.vision_max_tokens == 0,
                      "CLI omission must preserve the automatic Vision budget");
    failures += check(
        ninfer::cli::engine_options_from_cli(cli_default).vision_max_tokens == 0,
        "CLI automatic Vision budget must reach EngineOptions");

    const Options cli_capped = cli_parse(
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "1024"});
    failures += check(cli_capped.vision_max_tokens == 1024,
                      "CLI must parse --vision-max-tokens");
    failures += check(
        ninfer::cli::engine_options_from_cli(cli_capped).vision_max_tokens == 1024,
        "CLI Vision cap must reach EngineOptions");

    const Options cli_boundary = cli_parse(
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32768"});
    failures += check(cli_boundary.vision_max_tokens == 32768,
                      "CLI must accept the 32768 Vision boundary");

    bool cli_over_cap_rejected = false;
    try {
        (void)cli_parse(
            {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32769"});
    } catch (const std::invalid_argument&) {
        cli_over_cap_rejected = true;
    }
    failures += check(cli_over_cap_rejected, "CLI must reject a Vision cap above 32768");

    const ServeOptions serve_default = serve_parse({"ninfer-serve", "model.ninfer"});
    const ServeOptions serve_capped =
        serve_parse({"ninfer-serve", "model.ninfer", "--vision-max-tokens", "1024"});
    const ServeOptions serve_boundary =
        serve_parse({"ninfer-serve", "model.ninfer", "--vision-max-tokens", "32768"});
    failures += check(serve_default.vision_max_tokens == cli_default.vision_max_tokens,
                      "CLI and serve automatic Vision budgets must match");
    failures += check(serve_capped.vision_max_tokens == cli_capped.vision_max_tokens,
                      "CLI and serve explicit Vision caps must match");
    failures += check(serve_boundary.vision_max_tokens == cli_boundary.vision_max_tokens,
                      "CLI and serve Vision boundary values must match");

    struct ResolverCase {
        std::uint32_t capacity;
        std::uint32_t cap;
        std::uint32_t expected;
    };
    const std::vector<ResolverCase> resolver_cases{
        {2048, 0, 2048},
        {65536, 0, 32768},
        {32768, 0, 32768},
        {1024, 8192, 1024},
        {65536, 1024, 1024},
        {32768, 256, 256},
    };
    for (const ResolverCase& test : resolver_cases) {
        auto plan = build_test_plan(test.capacity, test.cap);
        failures += check(
            Runtime::resolved_vision_token_limit(*plan) == test.expected,
            "production resolved_vision_token_limit returned an unexpected budget");
        failures += check(plan->workspace.vision_encode > 0,
                          "production plan must reserve a Vision workspace");
        failures += check(plan->request_transient_capacity_bytes > 0,
                          "production plan must reserve request Vision transient storage");
    }

    auto automatic = build_test_plan(65536, 0);
    auto capped     = build_test_plan(65536, 1024);
    auto same_limit = build_test_plan(1024, 8192);
    failures += check(
        Runtime::resolved_vision_token_limit(*automatic) == 32768 &&
            Runtime::resolved_vision_token_limit(*capped) == 1024 &&
            Runtime::resolved_vision_token_limit(*same_limit) == 1024,
        "production plans must expose the expected resolved Vision limits");
    failures += check(capped->workspace.vision_encode < automatic->workspace.vision_encode,
                      "a smaller resolved limit must shrink the production Vision workspace");
    failures += check(capped->request_transient_capacity_bytes <
                          automatic->request_transient_capacity_bytes,
                      "a smaller resolved limit must shrink request Vision transient storage");
    failures += check(capped->workspace.vision_encode == same_limit->workspace.vision_encode,
                      "equal production-resolved limits must produce equal Vision workspaces");
    failures += check(capped->request_transient_capacity_bytes ==
                          same_limit->request_transient_capacity_bytes,
                      "equal production-resolved limits must produce equal transient storage");

    auto cli_plan =
        build_test_plan(cli_capped.max_context, cli_capped.vision_max_tokens);
    failures += check(Runtime::resolved_vision_token_limit(*cli_plan) == 1024,
                      "the production planner must consume the parsed CLI Vision cap");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
