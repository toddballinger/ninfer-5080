# Issue #27 M0B — fail-closed tool-output contract (DESIGN ONLY)

Status: **proposed contract / awaiting human review**, not evidence that current NInfer enforces it. TASK_ID `ISSUE27_M0B_OUTPUT_SAFETY_CONTRACT`. Depends on merged PR #64, M0 current-behavior corpus. This milestone **does not change production source or install passing safety tests**.

## Verified boundaries on main (2026-10-10)

- `src/serve/request.h`: `GenerationRequest::tools` carries `ToolDefinition {name, description, parameters_json, definition_json, strict}`; `tool_choice` includes Auto/None/Required/Named; `tool_name_max_length` is request-specific.
- `src/serve/tool_call_parser.h`: `parse_qwen_tool_call_output(text, max_tool_name_length)` cannot validate declared names or schemas because it has no request argument. `ToolCallStreamFilter::finish(false)` flushes buffered text including malformed markup. Existing M0 tests intentionally pin this.
- `src/serve/generation_service.cpp`: `ServiceOutputSink::publish` publishes non-reasoning ordinary content through `ToolCallStreamFilter::feed`; `GenerationService::run` parses `outcome.text` after generation and transfers candidate calls into `outcome.tool_calls`. Currently `ServiceOutputSink::finish(is_tool_call_response)` uses parser success, not a request-schema validator verdict. **Stream callback bytes already emitted cannot be retracted.**
- All output projections, translated wire response paths and the lifetime of originating request metadata require call-site verification before M1. Approximate line numbers are not a stable interface.

## Proposed trust-boundary flow

```text
GenerationRequest (caller-declared tools, parameter schemas, choice, wire client)
     | immutable output-validation policy derived from request
     v
Generated raw tokens/reasoning (canonical history, cache, generated token IDs)
     | candidate-only parse; preserve raw bytes privately
     v
Candidate tool blocks + ordinary spans + parse diagnostics
     | validate exact declared name, tool choice, JSON args and supported schema
     | reject malformed, ambiguous and unsupported constraints; no repair by invention
     v
Validated projection (visible safe text, complete validated tool calls, diagnostics)
     +--> nonstream OpenAI / Anthropic translation
     +--> stream buffer/commit state machine; only reversible buffers before validation
```

The generation engine, tokenization, prefix-cache keys, raw generated IDs and canonical ChatTurn/history MUST remain unmodified. Safe response projection is a separate object, never reinserted as model history implicitly.

## Security contract decisions (proposed, M1 to implement)

**Allowlist and choice.** A candidate is callable only if the exact case-sensitive name matches exactly one tool declared by the originating request, tool choice permits it, and name length complies with that client's verified limit. `None`, absent/empty tool definitions, name mismatch or duplicate definitions produce **zero callable tools**. For `Named`, enforce exact designated tool. `Required` cannot justify synthesising a missing/invalid call.

**Schema policy.** Parse candidate arguments as a JSON object. Validate required/type/additionalProperties and all constraints actually advertised as supported. If an input schema includes an unsupported keyword, fail closed for that candidate or reject request up front (consistent, explicit API diagnostic); never silently strip a constraint. Define a supported JSON Schema subset and strict-mode mapping *before* M1. Reject schema-invalid calls regardless of syntactic parser success. Duplicate parameter keys are ambiguous: propose reject with diagnostic (no last-wins execution); final policy requires review.

**Recovery and raw-markup suppression.** A fully delimited earlier candidate may survive a later malformed/truncated candidate **only** if it independently passes allowlist/schema checks. Incomplete or rejected blocks are not returned as tools or ordinary content. If a safe parser cannot unambiguously separate ordinary text from unsafe markup, suppress the ambiguous span (possibly all content), rather than leak markup, repair a nonexistent tool or make an unsafe call.

**Quoted reasoning and marker recognition.** Quoted `"</think>"` must not close the reasoning channel; quoted or escaped tool blocks must not become executable. Implement/test the actual reasoning splitter separately; parser-only tests cannot prove this invariant. Alternate XML spellings remain unsupported until their exact grammar is specified and verified.

**Streaming commitment.** For tool-capable generations, emitted stream chunks must be a stable prefix of the final **validated** projection; no premature callable tool or unsafe markup. Candidate tool blocks remain privately buffered through closure + validation. On length stop, disconnect/cancellation or malformed EOS, suppress uncommitted markup; preserve any already safe committed prefix. Streaming and nonstreaming semantically agree on calls, arguments, finish reasons and safe content. A conservative entire-response buffer is allowed initially if streaming latency would otherwise violate the safety boundary.

**Observable diagnostics.** Record reason codes without revealing secrets or schema payloads: `UNDECLARED_TOOL`, `TOOL_CHOICE_DENIED`, `NAME_LENGTH`, `ARGUMENT_JSON_INVALID`, `ARGUMENT_SCHEMA_INVALID`, `SCHEMA_UNSUPPORTED`, `DUPLICATE_PARAMETER`, `INCOMPLETE_TOOL_REGION`, `AMBIGUOUS_MARKUP`, `REASONING_MARKER_AMBIGUOUS`, `CANCELLED_UNCOMMITTED`. Diagnostics must not itself expose an unsafe callable capability.

## M1 interface sketch (not implementation)

```cpp
// DESIGN SKETCH, NOT COMPILE-READY
struct ToolOutputPolicy {
    // immutable, copied/derived from originating GenerationRequest:
    // declared tool name → parsed/supported schema; choice; client name limit
};
struct ValidatedProjection {
    std::string safe_visible_text;
    std::vector<ToolCall> validated_calls;
    std::vector<RejectionDiagnostic> diagnostics;
    bool withheld_unsafe_region;
};
ValidatedProjection validate_candidate_output(
    const ParsedCandidateOutput&, const ToolOutputPolicy&);
```

Prepare the policy **before** inference/streaming; preserve ownership and lifetime across `prepare`, `run`, `ServiceOutputSink` and protocol translators. M1 must verify there is no path emitting unvalidated `outcome.tool_calls` or flushing unsafe `ToolCallStreamFilter` bytes in either stream/nonstream.

## Independently specified desired fixtures

`tests/fixtures/issue27_m0b_desired_contract.json` is **data only**, not executable C++ and not an assertion that current runtime is compliant. Rows have literal desired emitted safe text/calls, safety classifications, finish-event conditions and independent expected diagnostics. `expected_red_on_current_runtime` indicates a presently unimplemented or unproven behavior; it is NOT CI xfail and should not be silently converted into passing characterization output. Preserve merged M0 corpus unchanged.

## M0B pass / future M1 acceptance

M0B pass: independently authored contract + parseable JSON fixture, review of exact request→candidate→projection boundaries, explicit unresolved schema/stream policies, host-only JSON validation and diff checks. For M1 only, turn desired-contract rows into a separate executable suite after API review. M1 must additionally prove nonstream/stream (all split boundaries, bytewise, cancellation, length stop), OpenAI/Anthropic wire parity, no undeclared callable tool, no raw unsafe marker leakage, no premature call emission, and unchanged canonical token/history/prefix-cache identity.

### Deferred integration and compatibility

- End-to-end OpenClaw native tool-loop stress, tool choice, long names and finish reasons.
- Client routes with query strings and later system/developer messages (Strata compatibility cases).
- Unicode, JSON, macros, BPE and retained-prefix semantics from Issue #27 upstream fold-in.
- Captured independent upstream/fork fixtures, beyond hand-authored design cases.
- Recovery/syntax rules for alternate Qwen/Claude XML, and refusal/reasoning transformations.

### Hard stop / outstanding design approvals

1. Choose supported JSON Schema vocabulary and strict-mode behavior; unsupported constraints must fail closed.
2. Confirm `GenerationRequest` metadata reaches the terminal output and streaming translation layers immutably.
3. Decide whether any ordinary content may be committed before all later tool candidates have been validated; otherwise buffer until end.
4. Resolve duplicate-parameter semantics and ambiguity recovery policy without inventing capabilities.
5. Prove no changes to raw generation, canonical messages/history or cache identity.
6. Review behavior of `finish(false)` and streaming callback timing before enabling M1.

No runtime source edits, full CUDA builds, GPU tests, service changes, merge or automatic Issue #27 closure are authorised by this M0B artifact.
