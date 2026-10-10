# Issue #27 — Independent review and release gate (2026-10-10)

**Decision: HOLD. Implementation cycle frozen; documentation-only review update.** Neither PR #66 nor dependent PR #67 is authorised to merge, deploy or change a running service. Issue #27 remains open. This is a cumulative review of both PRs, **not** a claim that production E2E was exercised.

## Frozen reference points

- PR #66, base `main`, head `a7fe8418b280332ad0220443bb27e489cf344adb` at review: immutable request tool declarations, host-only schema validation with duplicate JSON key rejection, strict Qwen XML candidate parsing (duplicate parameter rejection), universal terminal parsing, buffered content.
- PR #67, base PR #66 branch, reviewed source head `1b7ea267a7fd2ab7d6a8009f5f122c0c2dc9ccf6` **before documentation commit**: production response-projection seam, CMake linkage, synthetic CPU tests, terminal callback byte accounting.
- Prior merged PR #64 (M0 characterization corpus) and PR #65 (M0B desired contract) remain intact. Do not treat M0 unsafe behavior as an accepted security outcome.
- Host report for PR #67 code head: projection suite RC0 (failures=0), M0 parser RC0, M1A validator RC0, `generation_service.cpp` CPU syntax RC0, `git diff --check` RC0. Evidence is **reported by OpenClaw**, not independently rerun here. Documentation-only commits made after this reference SHA do not constitute newly tested code.
- No CUDA, GPU, model loading, HTTP E2E, deployed restart, or production service changes.

## Independent source review

### Confirmed structural properties

- `GenerationService::run` constructs a request-owned `ToolOutputPolicy`; non-stream path calls `project_tool_output`; stream path owns `ProjectedContentStream` and calls its finalisation, which uses the same `project_impl` implementation.
- Content `feed()` returns no bytes until full output is available. The production sink publishes projected visible text once on terminal validation. This **changes incremental content streaming to terminal-only content delivery**, a significant compatibility/latency trade-off.
- The final streaming callback is accounted for in `GenerationOutcome.streamed_content_bytes` only when a content callback received nonempty terminal visible text. `http_server.cpp::unstreamed_content` trims by this count in Chat Completions and Anthropic Messages handling; inspect Responses encoder separately.
- Strict parser suppresses malformed content containing full `<tool_call>` markers and refuses repeated XML parameter names before JSON normalization; validator rejects duplicate JSON object keys and enforces a constrained JSON Schema subset. Tool authorization is request-scope, not model-scope.

### Open or insufficiently proved release gates

1. **Incomplete/alternate marker handling.** The standalone test currently accepts `Safe <tool_` as visible text, while the documented M1B acceptance matrix requires suppressing potentially unsafe incomplete prefixes. A model can also produce alternate tag spellings. This is an **explicit specification-vs-test conflict**; define exact plain-prose/quoted-marker policy and add independent negative fixtures before asserting marker-suppression completeness. Do not relax tests silently.
2. **Reasoning channel is not atomically gated.** `ProjectedContentSink::publish` immediately forwards `OutputChannel::Reasoning` through `on_reasoning`, while content is terminal-only. This does not itself create a callable tool, but is not an all-channels sanitization guarantee and requires quoted `</think>` / channel-boundary semantics review.
3. **Streaming semantics and finish reasons.** Normal content is delayed until completion, while token accounting and finish reason are determined by `GenerationResult`. Verify the actual client expectations for `OutputLimit`, partial tool responses, zero-content terminal messages, streaming tool event ordering and cancellation/disconnect. CPU projection fixtures alone do not prove network event compatibility.
4. **HTTP disconnect E2E and Responses API.** `http_server.cpp` uses `unstreamed_content` to avoid duplicate terminal emission for Chat Completions and Anthropic; `responses_http.cpp` uses a different encoder (`content_delta` then `encoder->finish`) and requires explicit parity tests. None of these were run as real network/model E2E in the recorded host evidence.
5. **Schema breadth / recovery scope.** Validator only supports a deliberately narrow JSON Schema subset, so client-defined schemas using other keywords may be rejected; Issue #27's broader recovery goals (alternate markup, quoted reasoning closes, malformed later-call salvage, compatibility shapes, diagnostics) are NOT complete. Atomic rejection is safer than silent recovery, but is not a complete implementation of the parent issue.
6. **Output buffer cost.** Buffering all generated content and copying authoritative final content has additional host memory/latency consequences. Benchmark and bound before production adoption.

## Release decision and dependency order

- **PR #66:** HOLD as draft; its integration requires the PR #67 seam and release-gate decision; do not merge it independently as a complete production fix.
- **PR #67:** HOLD as draft; dependent base must be handled deliberately. This review contains documentation only; it is not a code/security approval.
- **Issue #27:** remain open. Preserve completed M0/M0B and host-validated M1A/M1B milestones separately from unfulfilled parent scope.
- **Approval authority:** explicit human approval required for each merge and any production deployment. No implied permission from host-only green tests or from this documentation review.
- **Work discipline:** freeze code changes here as requested; record a bounded follow-up for residual contract issues. Future quick-win one-shots must be isolated, independently testable and must stop/reclassify when new architectural/security dependencies arise.

## Re-review triggers

A later independent review must (a) reconcile marker-prefix/quote policy with tests, (b) verify HTTP protocol and reasoning-channel compatibility, (c) check output memory/latency, (d) confirm code + CI/worktree hashes, (e) make explicit merge recommendation with residual risks and rollback, and (f) obtain human authorisation before any merge or deployment.
