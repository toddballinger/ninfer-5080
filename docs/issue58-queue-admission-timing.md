# Issue #58 — monotonic queue insertion to selected-admission timing

## Scope
Adds an off-by-default `[ADMISSION-QUEUE-TIMING]` event for a successfully selected FIFO/backfill request, recording `id`, `lane`, `queue_wait_ms`, and `event=selected_for_admission`. Uses existing `NINFER_ADMISSION_TRACE=1` switch. Trace is OFF by default.

The queue timestamp is taken immediately after `pending_.push_back(request)` **while holding `queue_mutex_`**. This covers both regular and decision submissions. The endpoint is taken after `erase_pending(request)` succeeds in `admit_planned_request`, before release of planning state, lane ownership and prefill. Thus the measurement is **queue residence until selected admission**, not API request-to-response delay, full service wait, GPU start, prefill start, time-to-first-token, or Unix timestamp. `Clock::steady_clock` avoids wall-clock adjustments; the difference is rounded down to integer milliseconds. There is no additional polling or mutation of scheduler choice.

No timing event is emitted for cancelled, expired, rejected or errored requests. The duration excludes pre-enqueue frontend preparation, and cannot alone explain whether a long head was blocked by KV, lane or policy. Existing #72–#75 trace events continue to provide context; this event is not automatically correlated with all rejected probes.

### Four-file scope
- `src/runtime/engine/concurrent_executor.h`
- `src/runtime/engine/issue58_queue_timing.h`
- `tools/tests/issue58_queue_timing_test.cpp`
- `docs/issue58-queue-admission-timing.md`

Base main: `1c9447c19e90770a6791432ad52e290d8aaed0e7`.

## Host-only validation for OpenClaw MAIN

Check exact PR HEAD, clean worktree, diff and scope. Run:
```bash
ccache c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_queue_timing_test.cpp -o /tmp/issue58_queue_timing_test
/tmp/issue58_queue_timing_test
git diff --check main...HEAD
git diff --name-only main...HEAD
```
**Required:** actual `src/runtime/engine/engine.cpp` GCC C++20 `-fsyntax-only` integration with existing CUDA 13.4 headers and project definitions; no nvcc/CUDA compilation/GPU work. Review both enqueue sites, lock ordering and admission-erase lifetime. Verify no changes to scheduling policy, physical admission predicates, queue order or eviction behavior. If integration discovers a syntax/semantics defect, make at most two corrections in the four-file scope and rerun all gates. Post exact RCs, compiler versions, SHA, source invariant review, limitations to PR.

**Hard prohibitions:** No production C1 trace enablement, restart, port 8080 change, C2 run, GPU/device execution, model reload, deployment, automatic merge or issue closure.

This is diagnostic readiness only; a separate approved candidate C2 experiment and fairness/TTFT measurement remain mandatory.
