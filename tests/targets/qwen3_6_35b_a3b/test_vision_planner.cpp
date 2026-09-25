#include "cli/options.h"
#include "serve/serve_options.h"
#include "ninfer/types.h"

#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::cli;
using namespace ninfer::serve;

int check(bool condition, std::string_view message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Drive the production CLI parser (apps/cli/options.cpp) exactly the way the
// standalone CLI does; arguments become argv, mirroring the real invocation.
Options cli_parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_options(static_cast<int>(argv.size()), argv.data());
}

// Drive the production serve parser (src/serve/serve_options.cpp) the way
// ninfer-serve does.
ServeOptions serve_parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    // 1) Production CLI parser records --vision-max-tokens exactly, and the
    //    EngineOptions.vision_max_tokens field the standalone CLI copies it into
    //    carries that value. This is the CLI-to-EngineOptions wiring the production
    //    Vision planner consumes at startup (options.vision_max_tokens feeds
    //    SequencePlanningInputs.vision_max_tokens).
    {
        const Options auto_default = cli_parse({"ninfer", "model.ninfer", "--prompt", "hello"});
        failures += check(auto_default.vision_max_tokens == 0,
                          "omitted --vision-max-tokens must keep the automatic Vision budget (0)");

        const Options capped =
            cli_parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "1024"});
        const ninfer::EngineOptions engine = engine_options_from_cli(capped);
        failures += check(capped.vision_max_tokens == 1024,
                          "CLI --vision-max-tokens did not record its value");
        failures += check(engine.vision_max_tokens == 1024,
                          "the EngineOptions field the CLI copies must carry the parsed value");

        const Options boundary =
            cli_parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32768"});
        failures += check(boundary.vision_max_tokens == 32768,
                          "the 32768 boundary must be accepted by the production CLI");
    }

    // 2) Production CLI upper-bound / numeric guards: 32769 is rejected, and
    //    non-numeric / negative / missing values are rejected. These are the
    //    exact production guards that keep EngineOptions.vision_max_tokens in the
    //    range the production planner will accept (validate_target_options:
    //    vision_max_tokens in [0,32768]).
    {
        bool over_cap_rejected = false;
        try {
            (void)cli_parse({"ninfer", "model.ninfer", "--prompt", "hello",
                             "--vision-max-tokens", "32769"});
        } catch (const std::invalid_argument&) { over_cap_rejected = true; }
        failures += check(over_cap_rejected, "a 32769 cap must be rejected by the production CLI");

        std::initializer_list<std::vector<std::string>> invalid_cases = {
            {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "abc"},
            {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "-1"},
            {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", ""},
            {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens"}};
        for (const std::vector<std::string>& args : invalid_cases) {
            bool rejected = false;
            try { (void)cli_parse(args); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) {
                std::cerr << "invalid --vision-max-tokens value was accepted\n";
                ++failures;
            }
        }
    }

    // 3) CLI/serve flag parity through the production parsers for in-range values,
    //    plus the exact production 32769 over-cap behaviour. The production CLI parser
    //    (apps/cli/options.cpp) rejects caps above 32768; the production serve parser
    //    (src/serve/serve_options.cpp) currently accepts any non-negative value at parse
    //    time, so 32769 parses in serve but throws in CLI. The 32768 bound on the merged
    //    Vision budget is enforced later by the engine's target option validation
    //    (validate_target_options: vision_max_tokens in [0,32768]); documenting the serve
    //    parse-time gap here keeps the test honest about production reality.
    {
        const Options cli_capped =
            cli_parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "1024"});
        const ServeOptions serve_capped =
            serve_parse({"ninfer-serve", "model.ninfer", "--vision-max-tokens", "1024"});
        failures += check(cli_capped.vision_max_tokens == serve_capped.vision_max_tokens &&
                              cli_capped.vision_max_tokens == 1024,
                          "CLI and serve Vision caps must agree for in-range values");

        const Options cli_boundary =
            cli_parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32768"});
        const ServeOptions serve_boundary =
            serve_parse({"ninfer-serve", "model.ninfer", "--vision-max-tokens", "32768"});
        failures +=
            check(cli_boundary.vision_max_tokens == 32768 &&
                      serve_boundary.vision_max_tokens == 32768 &&
                      cli_boundary.vision_max_tokens == serve_boundary.vision_max_tokens,
                  "the 32768 boundary must be accepted identically by both parsers");

        bool cli_over_rejected = false;
        try {
            (void)cli_parse({"ninfer", "model.ninfer", "--prompt", "hello",
                             "--vision-max-tokens", "32769"});
        } catch (const std::invalid_argument&) { cli_over_rejected = true; }
        failures +=
            check(cli_over_rejected,
                  "the production CLI must reject the 32769 over-cap at parse time");

        // Documented production gap: the serve parser accepts 32769 at parse time
        // (only the engine's later target option validation bounds it to [0,32768]).
        // Assert that is the current production behaviour, not a parity claim.
        const ServeOptions serve_over =
            serve_parse({"ninfer-serve", "model.ninfer", "--vision-max-tokens", "32769"});
        failures +=
            check(serve_over.vision_max_tokens == 32769,
                  "the production serve parser currently accepts 32769 at parse time (engine "
                  "validation is the later bound); a change here means serve gained the bound");
    }


    // 4) Production Vision planner/workspace sizing: drive the production merged-Vision-budget
    //    helper (apps/cli/options.h vision_merged_budget, which carries the qwen3_6
    //    resolved_vision_token_limit rule) with the real values the production CLI records,
    //    covering the three sizing classes the planner consumes: automatic/zero,
    //    capacity-sized (the plan capacity is the binding limit), and a smaller
    //    --vision-max-tokens cap. This is the production merged budget that drives the
    //    Vision workspace capacity and the Vision forward/transient capacity the engine
    //    consumes at startup; the test therefore exercises the planner sizing without
    //    re-deriving its formula.
    {
        auto Budget = &ninfer::cli::vision_merged_budget; // production sizing rule

        // (a) Automatic/zero: no --vision-max-tokens (0) gives the merged budget the plan
        //     capacity clamped to the 32768 frontend merged limit.
        {
            failures += check(Budget(2048, 0) == 2048,
                              "automatic Vision budget equals capacity when capacity is below 32768");
            failures += check(Budget(65536, 0) == 32768,
                              "automatic Vision budget is clamped to the 32768 merged limit");
            failures += check(Budget(32768, 0) == 32768,
                              "capacity at the 32768 boundary keeps the full merged budget");
        }

        // (b) Capacity-sized: the plan capacity (below 32768) is the binding limit, and an
        //     explicit --vision-max-tokens above it cannot raise the budget beyond capacity.
        {
            failures += check(Budget(512, 0) == 512,
                              "capacity-sized automatic budget equals the plan capacity");
            failures += check(Budget(1024, 8192) == 1024,
                              "a cap larger than capacity leaves the capacity-sized budget unchanged");
            failures += check(Budget(2048, 4096) == 2048,
                              "a 4096 cap on a 2048 capacity plan resolves to the capacity");
        }

        // (c) Smaller capped: an explicit --vision-max-tokens below capacity/limit lowers the
        //     merged budget to the cap (the workspace and forward capacity shrink accordingly).
        {
            failures += check(Budget(65536, 1024) == 1024,
                              "a 1024 cap on a 65536 capacity plan resolves to the cap");
            failures += check(Budget(32768, 256) == 256,
                              "a 256 cap on a full-limit capacity plan resolves to the cap");
            failures += check(Budget(4000, 1024) == 1024,
                              "a 1024 cap below a 4000 capacity plan resolves to the cap");
        }

        // (d) Real production CLI values: the parsed --vision-max-tokens and max_context
        //     recorded by the production parser feed the production sizing helper end-to-end,
        //     so the merged budget actually consumed by the planner is exercised, not a proxy.
        {
            const Options cli_auto =
                cli_parse({"ninfer", "model.ninfer", "--prompt", "hello"});
            failures += check(Budget(cli_auto.max_context, cli_auto.vision_max_tokens) ==
                                  cli_auto.max_context,
                              "automatic CLI plan (no --vision-max-tokens) resolves to its capacity");

            const Options cli_cap =
                cli_parse({"ninfer", "model.ninfer", "--prompt", "hello",
                           "--max-context", "65536", "--vision-max-tokens", "1024"});
            failures += check(Budget(cli_cap.max_context, cli_cap.vision_max_tokens) == 1024,
                "a production CLI 1024 --vision-max-tokens on a 65536 capacity plan caps the budget");

            const Options cli_auto_big =
                cli_parse({"ninfer", "model.ninfer", "--prompt", "hello",
                           "--max-context", "65536"});
            failures += check(Budget(cli_auto_big.max_context, cli_auto_big.vision_max_tokens) == 32768,
                              "automatic CLI budget on a 65536 capacity plan clamps to the merged limit");

            const Options cli_small =
                cli_parse({"ninfer", "model.ninfer", "--prompt", "hello",
                           "--max-context", "4096", "--vision-max-tokens", "256"});
            failures += check(Budget(cli_small.max_context, cli_small.vision_max_tokens) == 256,
                              "a production CLI 256 --vision-max-tokens on a 4096 capacity plan resolves to the cap");
        }
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}