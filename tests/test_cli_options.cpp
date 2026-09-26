#include "cli/options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::cli;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    // Baseline: an omitted --vision-max-tokens leaves the shared automatic Vision budget in
    // place, matching the serve default of 0.
    const Options defaults = parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures +=
        check(defaults.vision_max_tokens == 0, "omitted --vision-max-tokens is not 0 by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");

    const Options capped =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "1024"});
    failures +=
        check(capped.vision_max_tokens == 1024, "--vision-max-tokens did not record its value");

    const Options auto_cap =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "0"});
    failures += check(auto_cap.vision_max_tokens == 0,
                      "an explicit 0 must preserve the automatic Vision budget");

    const Options boundary =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32768"});
    failures += check(boundary.vision_max_tokens == 32768,
                      "the 32768 family Vision cap was rejected at its upper bound");

    // Values beyond the family merged-Vision cap are rejected exactly as in the serve planner.
    bool over_cap_rejected = false;
    try {
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "32769"});
    } catch (const std::invalid_argument&) {
        over_cap_rejected = true;
    }
    failures += check(over_cap_rejected,
                      "a --vision-max-tokens value above 32768 was not rejected");

    // Non-numeric, negative, empty, and missing values all must fail.
    const std::vector<std::vector<std::string>> invalid_values = {
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "abc"},
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", "-1"},
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens", ""},
        {"ninfer", "model.ninfer", "--prompt", "hello", "--vision-max-tokens"},
    };
    for (const std::vector<std::string>& arguments : invalid_values) {
        bool rejected = false;
        try {
            parse(arguments);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        failures += check(rejected, "--vision-max-tokens invalid value was accepted");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}