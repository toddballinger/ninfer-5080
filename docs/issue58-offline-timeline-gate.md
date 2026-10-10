# Issue #58 — offline request-event timeline evidence gate

**Status: ChatGPT-authored CPU-only tool and tests; host validation pending.** This is a narrow supplement to existing #58 forensics, not a new scheduler or a production C2 fix.

Base `main`: `95b5c5e886046b23583c39e409209a77d79922c6`.

## Avoiding duplicate implementation

Current main already has `tools/bench/issue58_admission_forensics.py`, `tests/test_issue58_admission_forensics.py`, server trace tools, scheduler experiments, reversible-yield ownership safeguards and the operational handover `docs/ISSUE58_CONTINUATION_HANDOVER.md`. The existing forensics tool provides TTFT percentiles, completed/failed totals, occupancy sample summaries and slow outliers. It correctly warns that **TTFT includes queue + prefill + initial decode**, not actual queue duration.

This first milestone adds one missing diagnostic: a deterministic per-request, per-server-instance ordering and completeness audit for the **existing** `request_start`, `request_done`, `request_error` and `request_rejected` JSONL events, without guessing when admission actually occurred.

`src/serve/request_log.cpp` emits `timestamp_unix_ms` and `server_instance_id` in the JSON event base. `request_start` is a *logged event*, and by itself is NOT a proven queue-enter or actual admission timestamp. Event files may be split or incomplete. The timeline tool treats absent admission timestamps and deferral reasons as **unknown/null**. No synthetic timing values are represented as measured admission waits.

## How to use

Read-only offline input: saved NInfer server `*.jsonl` paths supplied explicitly by the operator. Do not collect new production data for this milestone.

```bash
python3 tools/bench/issue58_event_timeline.py /path/to/saved/server/capture.jsonl
```

Output schema `issue58-offline-event-timeline-v1` includes per-request start/terminal timestamps, event ordering, `start_to_terminal_ms`, event source line, missing/duplicate starts, missing/multiple terminals, terminal-before-start, unusable event count, and explicit null `queue_wait_ms`, `admission_unix_ms` and `admission_deferral_reason`. A server-instance identifier separates request IDs across captures; if missing, the file path is the fallback namespace. This can surface incomplete evidence but **cannot resolve queue causality** from fields that do not exist.

## Validation handover for OpenClaw MAIN

Branch `automation/chatgpt/ISSUE58_EVENT_TIMELINE_GATE`. Changed scope exactly these three files: `tools/bench/issue58_event_timeline.py`, `tests/test_issue58_event_timeline.py`, and this document.

Run from repo root, without server, CUDA, model, sudo or production writes:

```bash
python3 -B -m unittest -v tests/test_issue58_event_timeline.py
python3 -m py_compile tools/bench/issue58_event_timeline.py tests/test_issue58_event_timeline.py
python3 -B -m unittest -v tests/test_issue58_admission_forensics.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected **9 new tests**, plus 3 existing forensic regression tests. Record Python version, exact HEAD, each RC, test count, changed-file list and reproducibility. Optional synthetic example CLI using a *temporary, locally created* JSONL with paired start/done events is permitted; don't read real C1 data or switch services.

Allow at most two bounded repairs for deterministic host-only failures, then rerun all checks. On unclear event contracts, unexpected source mismatch or task expansion, stop and report a single blocker. Post final results to draft PR as `ISSUE58_TIMELINE_VALIDATED_DRAFT` or `ISSUE58_TIMELINE_BLOCKED`. MAIN should perform basic Python execution directly rather than cycling through malfunctioning LOCAL-WORKER tools.

**Hard boundaries:** no scheduler implementation, no CUDA/GPU/candidate C2 rollout, no model loading, no production service restart, no reading protected production captures without separate authorization, no new admission schema fields in runtime, no auto-merge, no closing #58. Preserve proven C1 on port 8080.

## Next gated engineering question

To separate queue wait from prefill, #58 must instrument independently observed arrival/queue-enter, admission attempt, actual admission, admission-deferral reason and relevant lane/KV eligibility; then reproduce the prior long-output-plus-short-request starvation case on a **separately authorised** supervised candidate environment. Existing #58 safety/ownership work should be reused, not supplanted. A timeline assembled from old `request_start` events cannot prove whether starvation is caused by KV eligibility, lane occupancy, or queue policy.
