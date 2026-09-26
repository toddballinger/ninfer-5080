#include "cli/options.h"

#include <iostream>
#include <string>

namespace {

using namespace ninfer::cli;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    const std::string usage = usage_text("ninfer");
    failures += check(usage.find("--vision-max-tokens") != std::string::npos,
                      "CLI help omits --vision-max-tokens");
    failures +=
        check(usage.find("caps merged Vision tokens") != std::string::npos,
              "CLI help omits the --vision-max-tokens semantics explanation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}