# Issue #58: client-neutral short-operation admission hint (experimental)

## Status

Opt-in feature validated in PR #61; this does not constitute approval to deploy C2 in production. Default behavior unchanged. This documents an **optional trusted-caller optimization**, not a standard OpenAI / Anthropic protocol field or a reliable inference about generated length.

Under `--max-concurrency 2`, `NINFER_SHORT_LANE_RESERVE=1` isolates one lane from **two simultaneously admitted long-classified generations**. The baseline classifier is `effective_output_tokens > NINFER_LONG_OUTPUT_THRESHOLD` (default 8192). Most logged requests with `requested_output_tokens=16384` actually complete under 1024 tokens, but *that cannot be known at admission*. Never classify solely from post-hoc completion length.

## Optional wire extension

Only **OpenAI Chat Completions** (`POST /v1/chat/completions`) presently parses a top-level boolean `ninfer_short_operation`. It is NOT automatically emitted by OpenClaw, Pi, OpenCode, DeepSeek Harness, or Open WebUI. Requests without it continue existing behavior. Other API surfaces (Responses and Anthropic Messages) do not yet accept the explicit hint; do not advertise cross-protocol compatibility.

Example of a trusted application knowingly marking a bounded short operation:

```json
{"model":"local-model","messages":[{"role":"user","content":"Summarize in one sentence."}],"max_tokens":16384,"ninfer_short_operation":true}
```

Set **both** `NINFER_SHORT_LANE_RESERVE=1` and `NINFER_TRUST_SHORT_OPERATION_HINT=1` to honor the hint for admission class. Otherwise, the hint is inert. This changes **only scheduler classification**, not the generation limit, planned output tokens, physical KV/MTP requirements, stop policy, model weights, or admission feasibility checks. C1 always behaves as before. Optional `NINFER_ADMISSION_CLASS_TRACE=1` logs request IDs, declared effective output and classification without prompt content.

## Trust and limitations

This flag does **not** authenticate clients: any party able to submit requests to the exposed endpoint can supply the boolean. Do not enable the trust flag on a shared/public endpoint where clients are untrusted. A mistakenly or maliciously marked long generation can occupy the reserved lane and re-create two-lane starvation; without real preemption the server cannot prevent that. A small *intent* is not a hard generation bound. Genuine bounded long requests should NOT carry the hint. No automatic heuristic can guarantee accuracy from output budgets alone.

## Existing Brain evidence

2026-10-09 final supervised RTX 5080 C2 GPU test, threshold 1024: long A (1536 output allowance) HTTP200 TTFT 0.204s; hinted 16384-budget request HTTP200 TTFT 0.200s, finish=stop; long B HTTP200 TTFT 28.303s and finish=length. Invalid hint types (string, number, array) each returned HTTP400. Tests passed, original C1 restored. Report `/tmp/issue58-c2-CYj73Q`. This verifies the specific admission scenario, *not* under-load fairness against a malicious or misclassified hint, production C2, or universal client support.

## Release gates

- Build, HTTP invalid-type rejection, hinted admission behavior and supervised restoration passed on Brain (2026-10-09). Explicit-hint-off negative integration check, absent-hint interoperability, and adversarial falsely-hinted long traffic still warrant wider regression coverage prior to production deployment.
- Integrate one **known-bounded** short-operation class from a trusted client. Do **not** tag all agent/tool requests or all 16K-budget requests as short.
- Re-run controlled C2 checks with restoration watchdog; leave C1 production unchanged until separate approval.
- Long term: correct client-neutral time slicing / KV+MTP checkpointing and resume makes hint less necessary; Issue #58 stays open.
