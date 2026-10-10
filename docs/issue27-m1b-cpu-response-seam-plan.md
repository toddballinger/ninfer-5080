# Issue #27 M1B — CPU-only response-projection seam

Status: **dependent draft plan, not a tested runtime fix**. TASK_ID: `ISSUE27_M1B_CPU_RESPONSE_SEAM`. Based on M1A PR #66 HEAD `a7fe8418b280332ad0220443bb27e489cf344adb`. Merge M1A **not** authorised; M1B must land after it and must not silently become a production deployment.

## Problem and chosen design

`GenerationService::run` currently combines `GenerationHandle::wait`, parsing/validation, response-output callbacks and final `GenerationOutcome`. Constructing a real Engine needs a model artifact. CPU syntax checks and isolated validator/parser tests cannot prove that `run` applies identical validated output semantics to streams and nonstreams, especially for tool-disabled requests, cancellation and finish reasons.

**Do not accept untested integration risk or add an artificial mock Engine/GenerationHandle.** Instead extract the *actual response projection and streaming publication path* into an injection-friendly production function/class that accepts synthetic generated text and policy. Both the real `run` and CPU-only tests MUST call the exact same production implementation. Treat this as a correctness dependency, not a testing-only duplicate algorithm.

## Proposed bounded extraction

1. Add `src/serve/tool_output_projection.{h,cpp}` (or a host-only header if linkage requires). Define a policy snapshot of `declared_tools`, `declared_tool_choice`, `tool_name_max_length`; use owned copies or a const reference with an explicitly safe call lifetime.
2. Define a function such as:
   ```cpp
   struct ProjectedToolOutput {
       std::string visible_text;
       std::vector<ToolCall> validated_calls;
       // optional safe diagnostics
   };
   ProjectedToolOutput project_tool_output(
       const std::string& generated_content, const ToolOutputPolicy& policy);
   ```
   Internally call the *strict* parser and schema validator already in PR #66; reject any ambiguous/invalid call atomically. Do not touch token IDs, canonical history, engine state or prefix-cache identity.
3. Add a CPU-exercisable stream publisher/collector that buffers generated Content-channel deltas and releases **only the validated projection** at finalisation. Reasoning events require explicit trusted-channel and cancellation tests; never mistake quote text or markup for an executable tool.
4. Replace the inline projection and buffer logic in `GenerationService::run` with calls to this **same production helper**. The only engine-specific operation should be obtaining generated deltas/final content; no fake GenerationHandle or model-load path. If `run` retains unsafe logic not included in tested helper, M1B is not complete.
5. Add `tests/test_tool_output_projection.cpp` with synthetic text, split-chunk sequences and recording callbacks. Compile with `g++ -std=c++20 -I src -I include -I third_party ...` and link just the host parser/helper sources required. No Engine, CUDA, model artifact, HTTP server or service restart.
6. Verify OpenAI/Anthropic translation handoff through inspection plus bounded protocol-level CPU-only tests where possible. Do not claim full HTTP E2E without a genuine interface; record any remaining gap.

## Discriminating acceptance matrix

- `tools=[]` and `tool_choice=none`: valid-looking Qwen markup rejected, zero callable tools, zero unsafe visible marker bytes; stream/nonstream parity.
- Declared tool allowed + schema-valid arguments: exactly one validated call, stable ID, arguments; optional safe preceding prose preserved.
- Undeclared name, invalid JSON/schema, duplicate JSON keys and duplicate Qwen XML parameters: zero callable tools and zero unsafe markup.
- Valid call followed by duplicate, truncated or malformed call: fail closed without premature first-call commitment.
- Partial marker fragments such as `<tool_`, `<tool_call`, bare `<function=`, nested/alternate malformed tags, quoted marker and end-of-response without closure: neither callable execution nor unsafe visible content.
- All two-chunk split boundaries and bytewise chunks, including marker boundaries; never invoke visible callback before commit point for tool-capable and tool-disabled requests.
- Cancellation while a marker is buffered, OutputLimit/stop/normal finish reason and retry/error unwind: no unsanitised buffer flush; accepted response status/usage semantics unchanged.
- Regression: plain text without markup still exactly preserved (avoid false censorship); distinct object scopes may repeat JSON keys without duplicate-key false positive.
- Canonical engine token/history/cache identity unchanged; no mutation to `GenerationResult.generated_token_ids`, `PromptInput`, model history or cache state.

## Residual risks / stop gates

M1B cannot prove real model decoding, Engine wait/admission and real network disconnect semantics without their own seams. Record them honestly. Reject any plan that tests a *copy* of production logic while the live path remains different. If strict parser permits marker-prefix leakage, fix production helper and red tests before finalisation. If protocol adapters emit content/calls through another path, bring those paths into review or stop.

## Operating contract

ChatGPT first prepares the repo artifacts. OpenClaw MAIN is authorised for bounded host-only engineering/validation and deterministic retries, with LOCAL-WORKER first and SOL escalation only after two genuine engineering failures. Avoid repeated malformed tool invocations. **No intermediate Telegram or ChatGPT approval requests** for work inside the authorised scope. Stop only at a validated draft PR for independent review or consolidated architecture/security blocker. Never run CUDA/GPU, modify live services, merge, deploy, force-push or close Issue #27.
