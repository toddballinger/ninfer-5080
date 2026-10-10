# Issue #26 — deterministic host-only OpenClaw prefix opportunity campaign

**Milestone:** CPU-only synthetic fixture generator and exact-identity upper-bound analysis. No NInfer cache implementation or diagnostic capture, and **not a measured baseline**.

Base: `2b2d8dcf7f1a6a6ef208ad4d7a6ab5cd158594d3` (after #70 merge).

## Source mapping and important limitation

`src/targets/qwen3_6/impl/runtime/prefix_identity.h/.cpp` establishes resident prefix identity for prepared prompt token types, three position axes and Vision items, including content digests, image grid, timestamps and span boundaries; `prefix_matches` also checks token IDs. `tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp` provides model-backed prefix-reuse checks (but requires engine/artifact). The host-only campaign cannot execute those checks or reconstruct real cache residency, GDN/KV consistency or tool-template tokenization.

`tools/analysis/issue26_prefix_opportunity.py` uses synthetic token identities with deterministic content labels, positions and Vision digest fields. This is a **deliberately simplified proxy** for the richer runtime identity rules, not a drop-in reference implementation for `ResidentPrefixIdentity`. Full Vision grid/span/timestamp and content equality remain a separate integration gate.

## Scenarios

Named scenarios cover successive tool calls, aborted request/retry, compaction, rewritten assistant history, a new user turn, two alternating conversations, a subagent return, 118,784-token fixed history with 32-token append, same/changed Vision digest and changed position identity. All are synthetic; no real conversation or personal data is loaded.

Comparison arms:
- **Resident-only LCP:** longest identical prefix against immediately preceding prompt. This is a simplistic lower-effort reference, *not* the actual NInfer rolling-tool cache algorithm.
- **Bounded snapshot LCP:** maximum exact-prefix overlap across current resident and at most six prior full-prompt identity snapshots, FIFO latest-first. This is an **idealized opportunity upper bound**; it does not represent valid GPU KV/GDN snapshots, restoration time or space.
- Report unmatched token counts under each conceptual arm; **never label them computed prefill tokens or TTFT measurements**.

Zero-checkpoint mode is a negative control. Tests verify changed Vision digest and positions invalidate reuse beyond common text, and that the 118K-plus-small-suffix case exposes a 32-token unmatched tail.

## Host validation handover

Branch `automation/chatgpt/ISSUE26_SYNTHETIC_PREFIX_CAMPAIGN`. Expected files: this doc, `tools/analysis/issue26_prefix_opportunity.py` and `tests/host/test_issue26_prefix_opportunity.py`.

Run from repository root:

```bash
python3 -B -m unittest -v tests/host/test_issue26_prefix_opportunity.py
python3 -m py_compile tools/analysis/issue26_prefix_opportunity.py tests/host/test_issue26_prefix_opportunity.py
python3 tools/analysis/issue26_prefix_opportunity.py --checkpoints 6 > /tmp/issue26-prefix-opportunity.json
python3 tools/analysis/issue26_prefix_opportunity.py --checkpoints 0 > /tmp/issue26-prefix-no-checkpoints.json
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expect **eight unit tests**, two valid JSON reports with `synthetic=true` and `measured_runtime=false`; 11 named scenarios. Record exact HEAD, Python version, return codes, test count, scenario counts, and relevant 118K/alternating-conversation fields. Do not paste unnecessarily huge full JSON reports into PR comments.

If a fixture is deterministically faulty, MAIN can perform at most two bounded repairs within these three files, rerunning all gates. If blocker is architectural or involves runtime/GPU access, stop and give one evidence report. OpenClaw should use MAIN's functioning terminal path and not loop on invalid LOCAL-WORKER tool-call formats.

Return `ISSUE26_VALIDATED_DRAFT` or `ISSUE26_BLOCKED`; post reproducible evidence to the draft PR. **No auto-merge; no Issue #26 closure.**

## Next measurement gate, not authorised by this PR

Phase 1 runtime work requires a separately approved *representative* OpenClaw request campaign, read-only event diagnostics (`total_prompt_tokens`, actual `reused_prompt_tokens`, `prefill`, `append_frontier`, checkpoint restore/reset counts, cache path), TTFT p50/p90 and prepare/prefill/restore timings. Compare 64K–128K retained history under the canonical Qwen3.8-27B model before attributing possible gains to checkpointing. Avoid creating a new cache architecture until those measured misses are established.

**Hard exclusions:** production 8080 service changes/restarts, CUDA work, model loading, GPU profiling, host checkpoint storage experiments, real traffic exfiltration, production mutation, and any inference that synthetic upper bounds prove a speedup.

## Host validation and independent review — 2026-10-10

MAIN validated the executable-code HEAD `bf454c91d374831467ea54a82bc85c30de7c23fa` on Python 3.14.4: eight unit tests RC0; `py_compile` RC0; six- and zero-checkpoint CLI campaigns RC0 producing valid JSON with 11 scenarios each; diff/scope checks RC0. Selected synthetic outputs: 118K-plus-suffix unmatched tail 32 tokens; alternating conversations yield 139,360 *additional potential* reusable tokens under six idealized snapshots versus zero in the no-snapshot control. Evidence: https://github.com/toddballinger/ninfer-5080/pull/71#issuecomment-6097149909.

Independent review inspected the full three-file PR diff and model semantics: synthetic token identity includes content, position and Vision digest; snapshots are compared using exact longest-common-prefix identity; there are no modifications to NInfer runtime. The analysis intentionally **does not** measure actual cache hits, KV/GDN snapshot validity, prefill, TTFT, storage pressure or wall-clock speedup. The synthetic 139,360-token upper bound must not be treated as measured performance improvement. Suitable for merge as a reusable research fixture only, subject to explicit operator approval. Keep parent #26 open for real instrumented campaigns.
