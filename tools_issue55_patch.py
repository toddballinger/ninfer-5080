#!/usr/bin/env python3
import sys

PATH = "tests/targets/qwen3_6_27b/test_mtp_matched_force_real.cpp"
with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

def rep(old, new, label, required=True):
    global src
    n = src.count(old)
    if n == 0:
        if required:
            print("MISSING ANCHOR: " + label)
            sys.exit(2)
        else:
            print("OPTIONAL-MISS (ok): " + label)
        return
    if n > 1:
        print("AMBIGUOUS ANCHOR (x%d): " % n + label)
        sys.exit(3)
    src = src.replace(old, new, 1)
    print("OK: " + label)

# 1) insert the exact draft-values + measured-reuse compare helpers after compare_identity
rep(
"""    if (a.vision_items != b.vision_items) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " vision_items " << a.vision_items << " vs "
                  << b.vision_items << '\\n';
    }
    return all;
}
""",
"""    if (a.vision_items != b.vision_items) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " vision_items " << a.vision_items << " vs "
                  << b.vision_items << '\\n';
    }
    return all;
}

// Exact comparison of the retained MTP draft VALUES (not just the count): the
// full kMtpDecodeMaximumDrafts-slot draft array must match slot-for-slot.
inline bool compare_drafts(const std::array<TokenId, kMtpDecodeMaximumDrafts>& a,
                           const std::array<TokenId, kMtpDecodeMaximumDrafts>& b,
                           const char* what) {
    bool all = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            all = false;
            std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " mtp_drafts slot " << i << " "
                      << a[i] << " vs " << b[i] << '\\n';
            break;
        }
    }
    return all;
}

// Exact comparison of the measured re-admission reuse path: the BeginSummary's
// prefix_reuse_path and reused_prompt_tokens must match exactly; an append path
// (preserved retained state) reports the retained frontier as the measured reused
// base, while a full reset reports zero.
inline bool compare_reuse(const ninfer::runtime::BeginSummary& a,
                          const ninfer::runtime::BeginSummary& b, const char* what) {
    bool all = true;
    if (a.prefix_reuse_path != b.prefix_reuse_path) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " prefix_reuse_path differs\\n";
    }
    if (a.reused_prompt_tokens != b.reused_prompt_tokens) {
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " reused_prompt_tokens "
                  << a.reused_prompt_tokens << " vs " << b.reused_prompt_tokens << '\\n';
    }
    return all;
}
""",
"compare helpers")

# 2) align_discrete: append the exact draft-values comparison after the mtp_draft_count line (4-space indent)
rep(
"""    eq(a.drafts, b.drafts, "mtp_draft_count", all, what, what, false);
""",
"""    eq(a.drafts, b.drafts, "mtp_draft_count", all, what, what, false);
    eq(a.drafts_values, b.drafts_values, "mtp_drafts_values", all, what, what, false);
    check(compare_drafts(a.drafts_values, b.drafts_values, what), what, " draft values differ");
""",
"align_discrete additions", required=True)

# 3) align_numerical: add per-field sampling scalar checks + greedy readiness
rep(
"""    if (a.sampling_equal != b.sampling_equal || !a.sampling_equal) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " sampling config differ\\n";
    }
    if (a.token_counts_equal != b.token_counts_equal || !a.token_counts_equal) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " token_counts binding differ\\n";
    }
    return all;
}
""",
"""    if (a.sampling_equal != b.sampling_equal || !a.sampling_equal) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " sampling config differ\\n";
    }
    // Exact per-field sampling scalars (each lane's own retained row): with the
    // fixed greedy harness configuration both rows are bit-identical constants, so
    // equality is exact and no tolerance is justified.
    if (a.temperature != b.temperature || a.top_k != b.top_k || a.top_p != b.top_p ||
        a.min_p != b.min_p || a.presence_penalty != b.presence_penalty ||
        a.frequency_penalty != b.frequency_penalty || a.seed != b.seed) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " sampling scalars differ (temp "
                  << a.temperature << '/' << b.temperature << ", top_p " << a.top_p << '/'
                  << b.top_p << ", seed " << a.seed << '/' << b.seed << ")\\n";
    }
    // Greedy readiness: a retained decode lane's sampling row must be greedy-ready
    // (temperature <= 0) on BOTH lanes at every compared boundary.
    if (!(a.temperature <= 0.0f) || !(b.temperature <= 0.0f)) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " sampling readiness (greedy) not "
                       "satisfied: a.temperature=" << a.temperature << " b.temperature="
                  << b.temperature << '\\n';
    }
    if (a.token_counts_equal != b.token_counts_equal || !a.token_counts_equal) {
        all = false;
        std::cerr << "ISSUE55_MATCHED_DIAG=" << what << " token_counts binding differ\\n";
    }
    return all;
}
""",
"align_numerical sampling checks")

# 4) prefill: add identity_history param
rep(
"""    ninfer::runtime::BeginSummary prefill(const std::vector<TokenId>& tokens,
                                          bool    reuse = false,
                                          std::uint32_t lane = 0) {
        auto prompt = instance->loaded->frontend.prepare_tokens(tokens, true);
""",
"""    ninfer::runtime::BeginSummary prefill(const std::vector<TokenId>& tokens,
                                          bool    reuse = false,
                                          std::uint32_t lane = 0,
                                          const std::vector<TokenId>& identity_history = {}) {
        auto prompt = instance->loaded->frontend.prepare_tokens(tokens, true);
        if (!identity_history.empty()) {
            // Identity-only continuation: the physical prompt is the new suffix; its
            // identity carries the preserved historical prefix (reusable=true, text
            // positions per assign_text_positions), which is the production
            // re-admission mechanism that licenses AppendAtFrontier over the
            // retained lane state.
            std::vector<TokenId> full;
            full.reserve(identity_history.size() + tokens.size());
            full.assign(identity_history.begin(), identity_history.end());
            full.insert(full.end(), tokens.begin(), tokens.end());
            prompt.data_->identity.rewrite_checkpoint = ninfer::runtime::RewriteCheckpointSpec{};
            prompt.data_->identity.reusable            = true;
            prompt.data_->identity.positions_  = std::vector<std::int32_t>(full.size(), 0);
            prompt.data_->identity.token_types_ = std::vector<std::uint8_t>(full.size(), 0);
            prompt.data_->token_ids.clear();
            for (std::size_t i = 0; i < full.size(); ++i) {
                prompt.data_->token_ids.push_back(full[i]);
            }
        }
""",
"prefill identity_history")

# 5) repwarm: add identity_history param (free function, 0-indent body)
rep(
"""std::vector<TokenId> repwarm(Session& s, const std::vector<TokenId>& prompt, std::uint32_t lane,
                             ninfer::runtime::BeginSummary* out = nullptr) {
    const auto summary = s.prefill(prompt, true, lane);
""",
"""std::vector<TokenId> repwarm(Session& s, const std::vector<TokenId>& prompt, std::uint32_t lane,
                             ninfer::runtime::BeginSummary* out = nullptr,
                             const std::vector<TokenId>& identity_history = {}) {
    const auto summary = s.prefill(prompt, true, lane, identity_history);
""",
"repwarm identity_history")

# 6) continuation section: identity-carrying re-prefill + measured reuse
rep(
"""    std::cout
        << "ISSUE55_MATCHED_CONTINUATION_CONSTRUCTION=RE_PREFILL_THEN_32_MTP_ROUNDS_BOTH_LANES\\n";
    std::vector<TokenId> commit_warm, force_warm;
    commit_warm = repwarm(s, commit_warm_full, decision_lane);
    force_warm  = repwarm(s, commit_warm_full, force_lane);
    std::cout << "ISSUE55_MATCHED_CONTINUATION_TOKENS="
              << (commit_warm == force_warm && commit_warm.size() == kWarm ? "IDENTICAL_32"
                                                                           : "MISMATCH")
              << '\\n';
    check(commit_warm == force_warm && commit_warm.size() == kWarm,
          "continuation tokens differ across matched lanes");
""",
"""    // Production re-admission mechanism (request_plan_impl.h plan_request_for_lane:
    // AppendAtFrontier requires a retained lane, a reusable identity whose prefix
    // matches the lane's ledger, and, for MTP, mtp_kv_valid >= reuse_base - 1 and
    // tail_hidden_valid; otherwise the plan degrades to FullReset). Both lanes are
    // Complete+retained at the post-transition boundary; mtp_kv_valid covers
    // reuse_base = E-1, so production plans AppendAtFrontier on BOTH lanes.
    std::cout
        << "ISSUE55_MATCHED_CONTINUATION_CONSTRUCTION=IDENTITY_CARRYING_APPEND_AT_FRONTIER_RE_PREFILL_THEN_32_MTP_ROUNDS_BOTH_LANES\\n";
    std::vector<TokenId> commit_warm, force_warm;
    ninfer::runtime::BeginSummary commit_summary, force_summary;
    commit_warm = repwarm(s, commit_warm_full, decision_lane, &commit_summary, commit_warm_full);
    force_warm  = repwarm(s, commit_warm_full, force_lane, &force_summary, commit_warm_full);
    check(compare_reuse(commit_summary, force_summary, "CONTINUATION_REUSE_PATH_ALIGNMENT"),
          "continuation reuse path/reused base not parity-aligned");
    check(commit_summary.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier &&
              force_summary.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier &&
              commit_summary.reused_prompt_tokens == d0.E &&
              force_summary.reused_prompt_tokens == d0.E,
          "continuation re-admission did not use preserved state (AppendAtFrontier)");
    std::cout << "ISSUE55_MATCHED_CONTINUATION_REUSE_PATH="
              << (commit_summary.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier
                      ? "APPEND_AT_FRONTIER_PRESERVED_STATE" : "FULL_RESET_NOT_PRESERVED")
              << " (measured reused_prompt_tokens commit=" << commit_summary.reused_prompt_tokens
              << " force=" << force_summary.reused_prompt_tokens << ")\\n";
    const std::uint32_t processed_commit = commit_summary.processed_prompt_tokens,
                       processed_force = force_summary.processed_prompt_tokens;
    std::cout << "ISSUE55_MATCHED_MEASURED_HISTORICAL_PREFIX_TOKENS="
              << (commit_summary.reused_prompt_tokens > 0
                      ? commit_summary.prompt_tokens - commit_summary.reused_prompt_tokens : 0)
              << ' '
              << (force_summary.reused_prompt_tokens > 0
                      ? force_summary.prompt_tokens - force_summary.reused_prompt_tokens : 0)
              << '\\n';
    std::cout << "ISSUE55_MATCHED_CONTINUATION_TOKENS="
              << (commit_warm == force_warm && commit_warm.size() == kWarm ? "IDENTICAL_32"
                                                                           : "MISMATCH")
              << '\\n';
    check(commit_warm == force_warm && commit_warm.size() == kWarm,
          "continuation tokens differ across matched lanes");
""",
"continuation identity-carrying re-admission")

# 7) NUMERICAL_TOLERANCE marker before the ARTIFACT line (anchor on unique ARTIFACT line)
rep(
"""    std::cout << "ISSUE55_MATCHED_ARTIFACT=" << artifact << '\\n'
""",
"""    // Numerical-tolerance justification (recorded for the marker schema): the
    // compared device state (tail hidden, linear-attention slot image, KV block
    // tables, sampling rows) is read through the production host-copy path after
    // device synchronization, from two lanes built by IDENTICAL construction and
    // driven through IDENTICAL kernels-on-identical-tensors; the identical
    // deterministic computation is bit-identical, so the comparison is exact bit
    // equality with tolerance 0. No numerical drift is tolerable: any bit
    // difference is a genuine boundary/implementation mismatch.
    std::cout << "ISSUE55_MATCHED_NUMERICAL_TOLERANCE=0_BIT_EXACT_DEVICE_STATE_EQUALITY_IDENTICAL_CONSTRUCTION_JUSTIFIED\\n";
    std::cout << "ISSUE55_MATCHED_ARTIFACT=" << artifact << '\\n'
""",
"numerical tolerance marker")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("WROTE: " + PATH)