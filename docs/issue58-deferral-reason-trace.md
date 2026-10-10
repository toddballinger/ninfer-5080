# Issue #58 — opt-in observational admission-deferral reasons

**Scope:** diagnostic-only extension of existing `NINFER_ADMISSION_TRACE=1` in `src/runtime/engine/concurrent_executor.h`. Never changes admission eligibility, policy, queue order, deadline, KV allocation, or GPU execution. The default trace setting remains OFF.

Current main baseline: `d0e093c05a52da6ede4b7a7389583443a448c2f3` (PR #72 merged). Source context: `find_admission_lane()`, `long_lane_guard()`, `try_admit_one()`. These already distinguish a policy hold on a long FIFO head, no free physical lane, and no feasible candidate lane. Current legacy `[ADMISSION-TRACE]` reports aggregate used/capacity but cannot prove the particular cause of failure from those totals.

## What was added

Pure host C++ helper `issue58_deferral_reason.h` classifies only **observable** facts:
- `short_lane_policy_hold`: `long_lane_guard(head)` was true when the long head was deferred; short candidate admission attempts remain unchanged.
- `no_vacant_lane`: FIFO head could not be admitted and all slot pointers were occupied.
- `vacant_lane_not_admittable`: at least one slot pointer was free but neither normal nor retained-eviction lane feasibility selected a lane. This is **NOT synonymous with insufficient KV capacity**: plan geometry, retained continuation, reservation or other eligibility can be involved.

The existing opt-in 10-second `[ADMISSION-TRACE]` gains a `reason=` field. The policy guard uses a similarly throttled `[ADMISSION-DEFERRAL]` event, emitted **after** the existing short-backfill attempt fails. The same per-executor timestamp rate limit applies. The event records request id and queue length only; no prompt, output, credentials or payload.

The runtime does **not** separately know when a request entered the queue or how long admission was deferred; this first milestone does not invent those timestamps. No reason is emitted if the tracing flag is unset. Classification is observational, not a root-cause or fairness guarantee.

## Host validation (OpenClaw MAIN only)

Exact branch `automation/chatgpt/ISSUE58_DEFERRAL_REASON_TRACE`; four files:
- `src/runtime/engine/issue58_deferral_reason.h`
- `src/runtime/engine/concurrent_executor.h`
- `tools/tests/issue58_deferral_reason_test.cpp`
- `docs/issue58-deferral-reason-trace.md`

From repository root, on Brain, build a **standalone CPU-only executable**, no runtime rebuild:

```bash
c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_deferral_reason_test.cpp -o /tmp/issue58_deferral_reason_test
/tmp/issue58_deferral_reason_test
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Use ccache C++ compiler if configured. Test checks four combinations of the policy guard and slot vacancy. Further **targeted C++ syntax validation of `concurrent_executor.h` against configured project headers** is useful if available without a full CUDA build; record exact command and RC. No production binary or CUDA compile is authorised. If the target syntax check is unavailable, report it as NOT RUN—not PASS—and preserve an integration compile gate prior to any staged deployment.

OpenClaw MAIN may correct at most two deterministic, tightly scoped compile defects in these four files and must rerun every attempted gate. Post SHA, toolchain versions, compile/test RC, exact changed file list and limitations to the draft PR. No auto merge or Issue #58 closure. Use MAIN's own reliable terminal execution; do not loop failing LOCAL-WORKER tool calls. Return `ISSUE58_REASON_VALIDATED_DRAFT` or `ISSUE58_REASON_BLOCKED`.

## Remaining engineering and rollout gates

This only labels observable deferral categories. To establish why `vacant_lane_not_admittable` occurs, a later independently reviewed change must capture per-lane plan-specific feasibility and exact resource/retained-state constraints. Actual queue wait and per-request admission timestamps also remain unmeasured. A supervised *candidate* C2 experiment, with production C1 preserved and explicit approval, is required before any latency/fairness conclusion.

**Exclusions:** No CUDA/GPU tests, service restarts, port 8080 changes, model loading, production trace enablement, scheduler policy changes, request content logging, preemption, yield, memory ownership changes, or C2 rollout.
