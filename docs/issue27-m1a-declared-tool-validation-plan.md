# Issue #27 M1A — caller-declared tool and argument validation

Status: **implementation plan, NOT runtime implementation or validation**. TASK_ID `ISSUE27_M1A_DECLARED_TOOL_VALIDATION`. Branch starts from M0B squash merge `318bc8a1378d3d02302a2708831235db13520b75`. Merge PR #64 = M0 baseline, PR #65 = M0B desired contract. This document is the bounded engineering specification for the **next implementation PR**.

## Objective and safety invariant

No generated candidate may reach `GenerationOutcome::tool_calls` unless its exact name is permitted by the *originating request*, its arguments parse as a JSON **object**, and each constraint in its declared tool schema is enforced or explicitly rejected. No undeclared/invalid callable tool. Never treat M0 current-behavior tests as proof of security.

**Important integration caveat:** this invariant cannot be satisfied by the currently exposed `parse_qwen_tool_call_output(text,max_name_length)` alone. The implementation must retain immutable request metadata through `GenerationService::prepare` to `run`; do not trust just `prepared.tool_capable`.

## Verified source interface map

- `src/serve/request.h`: `ToolDefinition{name, description, parameters_json, definition_json, strict}`; `GenerationRequest.tools`, `tool_choice`, `tool_name_max_length`.
- `src/serve/generation_service.h`: `PreparedRequest` currently exposes `tool_capable` and `tool_name_max_length` but **not** declared name/schema map; `GenerationOutcome::tool_calls` is the response projection.
- `src/serve/generation_service.cpp`: after wait, `parse_qwen_tool_call_output(outcome.text, prepared.tool_name_max_length)`; if parser marks tool response it moves calls into `outcome.tool_calls`. `ServiceOutputSink` may already have streamed content. This is an emission trust boundary.
- `src/serve/tool_call_parser.cpp`: existing function-name grammar accepts alphanumeric/_/- subject to client length; JSON parameters are flattened and duplicate names currently last-win; malformed later calls fall back to original raw text.
- `tests/test_tool_call_parser.cpp`: merged M0 characterization corpus (DO NOT rewrite unsafe baseline assertions as desired security tests).
- `tests/fixtures/issue27_m0b_desired_contract.json`: independent *design fixtures*, many expected-red or unproven, **not executable CI proof**.

## Implementation slices — execute sequentially, independent review at each gate

### M1A.1 Pure host-only validator (first PR slice)

Add a small validator API in the serve layer; prefer a separate `src/serve/tool_call_validation.{h,cpp}` if the build can include it host-only. Example interface (design, adapt to current compilation conventions):

```cpp
struct ToolValidationPolicy {
    // Immutable snapshot of declared ToolDefinition names, parameters JSON,
    // tool_choice, name-length limit, and verified schema capabilities.
};
struct RejectedToolDiagnostic { std::string code; std::string detail_safe; };
struct ValidatedToolOutput {
    std::vector<ToolCall> accepted;
    std::vector<RejectedToolDiagnostic> rejected;
};
ToolValidationPolicy prepare_tool_validation_policy(const GenerationRequest&);
ValidatedToolOutput validate_candidate_calls(
    const std::vector<ToolCall>& candidates, const ToolValidationPolicy&);
```

- Exact case-sensitive full-name match against the originating request, no wildcard/syntax-only permission. Duplicate names or invalid policy → fail closed.
- Respect `None`, `Named`, `Auto`, `Required`. `Required` never synthesises a missing call.
- Name length uses the request's verified client limit.
- Parse arguments as a JSON object, not arbitrary JSON/string; reject malformed/non-object input.
- Schema support must be explicit and conservative. **Proposed minimum**: `type` (object,string,number,integer,boolean,null,array), `properties`, `required`, `additionalProperties` (boolean), recursively for nested objects; `items` for homogeneous arrays. Unsupported assertions (e.g. `unevaluatedProperties`, `oneOf`, `anyOf`, `pattern`, `format`, `$ref`, unknown validation keywords) → fail closed, not ignore.
- Default `parameters_json` shape and strict-mode semantics must be confirmed against actual wire adapters **before** choosing request rejection versus candidate rejection. For schema-less tools, define a documented conservative rule before writing code.
- Return redacted deterministic rejection categories without serialising private tool arguments.

### M1A.2 Integration at origin-of-request boundary (second PR slice)

Carry the immutable policy in `PreparedRequest`, created while `GenerationRequest` still exists. Validate candidate calls **before** assigning `GenerationOutcome::tool_calls`; ensure all wire translators rely only on validated output. Caller metadata should not be mutable by generated model text.

**Streaming danger:** `ServiceOutputSink::feed` already emits irreversible deltas. M1A must not pretend the pure validator also solves streaming safety. For tool-capable streams, either conservatively buffer until validation or keep the integrated stream path gated as incomplete; do not merge a patch that allows a callable tool or unsafe markup to escape through an unvalidated stream. The complete streaming/recovery rewrite may be separately scoped M1B, but enforce the no-unsafe-emission invariant at the boundary first.

### M1A.3 Deterministic tests (green after implementation)

Standalone host-only tests with literal independent expectations: declared acceptance, undeclared denial, empty declarations, `None`, `Named` mismatch, case mismatch, duplicate declarations, 64/128 name bounds; missing required property, wrong type, extra property forbidden, malformed/non-object JSON, unsupported schema keyword, nested object and homogeneous array. Verify candidate ID/name/args unchanged on acceptance, no accepted call on any rejection; names not converted or repaired.

Add response-boundary tests proving that request-supplied declared schemas travel into validation and tool calls do not bypass it. Compare streaming/non-streaming callable output; failing streaming safety is a **blocker** for landing M1A integration, not an optional follow-up.

### M1A.4 Review / evidence / stop

Compile and test *host-only* standalone validator and parser, no GPU. Record exact compiler command, return code, test output, `git diff --check`, modified files, base/head commits, and independent expected-output oracle. Run full CMake only if a genuine CPU-only target has been proven: **do not accidentally trigger CUDA compilation**.

STOP and return a design blocker if: available JSON Schema semantics cannot be enforced or safely rejected; `tool_choice` semantics diverge across clients; the validation policy cannot be propagated to `run`; already-streamed unsafe bytes cannot be protected; any change would mutate token/history/prefix-cache identity. Do not weaken policy to obtain passing tests.

## One-shot division of labour

**ChatGPT:** research, inspect code, prepare this spec, GitHub-safe bounded edits and PR review. **OpenClaw:** verify actual build linkage/CPU-only test route, compile/execute, identify host issues; if local worker tool formatting fails, do not loop equivalent attempts, return exact GitHub-editable change to ChatGPT. Keep LOCAL engineering strikes separate from model-output failures. No production deploy, service reload, CUDA/GPU use, auto-merge, or Issue #27 closure.

**Milestone handoff target:** next draft PR with a tested isolated validator plus safe output-boundary integration, or a consolidated explicit security blocker; do not claim M1A complete from this planning artifact alone.
