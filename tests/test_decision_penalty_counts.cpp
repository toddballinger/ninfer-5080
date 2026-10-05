#include "targets/qwen3_6/impl/runtime/decision_commit_guard.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::replace_decision_sample_count;

int main() {
    // Positive-temperature, frequency-penalty fixture: the sampler increments
    // each pending successor once; a commit replaces only that contribution.
    constexpr float temperature = 0.8F, frequency_penalty = 0.5F;
    std::array<int, 16> counts{};
    auto sample = [&](TokenId id) { ++counts.at(id); };
    auto commit = [&](TokenId anchor, TokenId winner) {
        if (temperature > 0 && frequency_penalty != 0) {
            replace_decision_sample_count(anchor, winner,
                [&](TokenId id) { if (--counts.at(id) < 0) throw std::runtime_error("missing pending sample"); },
                [&](TokenId id) { ++counts.at(id); });
        }
    };
    auto penalized_logit = [&](TokenId id) { return 4.0F - frequency_penalty * counts.at(id); };
    sample(3);
    commit(3, 7); // M1: initial sampled anchor replaced.
    if (counts[3] != 0 || counts[7] != 1) throw std::runtime_error("M1 replacement failed");
    sample(7); // target-only successor equals previously committed winner.
    commit(7, 7); // M2 second selected token repeats the winner.
    if (counts[7] != 2 || penalized_logit(7) != 3.0F)
        throw std::runtime_error("repeated selected token lost frequency penalty");
    sample(7);
    commit(7, 9); // distinct winner must preserve both earlier 7 occurrences.
    if (counts[7] != 2 || counts[9] != 1 || penalized_logit(7) != 3.0F)
        throw std::runtime_error("previous selected history was erased");
    sample(7);
    commit(7, 7); // repeat token after a distinct winner.
    if (counts[7] != 3 || counts[9] != 1 || penalized_logit(7) != 2.5F)
        throw std::runtime_error("third selected occurrence was lost");
    std::cout << "decision-penalty-counts: M1 and repeated M2 positive-temperature penalty preserved\n";
}
