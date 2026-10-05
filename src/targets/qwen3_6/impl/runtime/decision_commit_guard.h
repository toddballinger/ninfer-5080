#pragma once
#include "ninfer/types.h"
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {
// Replace only the pending sample, preserving previously committed M2 winners.
template <class Decrement, class Increment>
void replace_decision_sample_count(TokenId anchor, TokenId winner,
                                   Decrement&& decrement, Increment&& increment) {
    decrement(anchor);
    increment(winner);
}

// Text executes the winner at E; the zero-extent MTP round still reserves
// through E + draft_window. Match decode_mtp_batch's materialize_sequence_kv.
constexpr std::pair<std::uint32_t, std::uint32_t> decision_commit_kv_extents(
    std::uint32_t frontier, std::uint32_t draft_window, std::uint32_t capacity) {
    return {frontier + 1U, frontier +
            (draft_window < capacity - frontier ? draft_window : capacity - frontier)};
}

// Shared by the real target entry point and the deterministic host guard test.
// Keep the guard before any device work: a consumed commit must not touch the lane.
template <class Sequence>
void require_decision_commit_ready(std::uint32_t lane, TokenId winner,
                                   std::uint32_t max_concurrency, std::int32_t token_domain,
                                   bool mtp_backend, bool mtp_decode,
                                   const Sequence& sequence, bool request_complete,
                                   std::uint32_t capacity) {
    if (lane >= max_concurrency || winner < 0 || winner >= token_domain) {
        throw std::invalid_argument("decision commit lane or winner is invalid");
    }
    if (!mtp_backend || !mtp_decode || !sequence.retained || !sequence.kv ||
        !sequence.kv->backend || sequence.decision_commit_consumed || !request_complete ||
        sequence.execution_frontier == 0 || sequence.execution_frontier >= capacity ||
        sequence.ledger_frontier != sequence.execution_frontier + 1U ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.text_kv_valid != sequence.execution_frontier ||
        sequence.mtp_kv_valid != sequence.execution_frontier ||
        !sequence.tail_hidden_valid) {
        throw std::logic_error("decision commit requires an unconsumed retained MTP frontier");
    }
}
} // namespace ninfer::targets::qwen3_6::detail
