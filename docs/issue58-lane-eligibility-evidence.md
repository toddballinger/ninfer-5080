# Issue #58 — per-lane admission evidence (bounded milestone)

## Goal and source contract
Capture **the outcomes of lane eligibility checks already performed** in `ConcurrentExecutor::find_admission_lane`, without extra admission attempts or changing scheduling. The prior #73 opt-in trace reports `vacant_lane_not_admittable` but not which lane was occupied, direct-fit rejected, or retained-eviction-fit rejected.

Adds `[ADMISSION-LANE] head=<id> lane=<id> result=<classification>` inside the existing 10-second `NINFER_ADMISSION_TRACE=1` emission. The classifier distinguishes `occupied`, `direct_admittable`, `retained_eviction_admittable`, `not_admittable`, and `plan_unavailable`. The request-head result is captured from existing probes; **no extra `can_admit_lane` or eviction probes** are introduced by the trace. The result is only emitted when the FIFO head was blocked and tracing is explicitly enabled. No request content or user data.

`not_admittable` means both existing admissibility checks failed for that lane; **it does not prove KV resource shortage or scheduler starvation root cause**. These predicates include state beyond raw total capacity. This is not a queue-time instrument and does not yield actual TTFT savings. If the long-lane policy guard returns early, #73's existing policy trace applies instead; do not manufacture lane-probe results for probes never executed.

## Scope, exclusions and review

Base: `1bdd69e685075519ea4629cb4de7477ccf6b4776`. Four files: `src/runtime/engine/concurrent_executor.h`, `src/runtime/engine/issue58_lane_evidence.h`, `tools/tests/issue58_lane_evidence_test.cpp`, and this document.

This PR must not alter queue order, selection, resource reservation, classification policy, actual KV eviction or scheduler fairness. Production C1 on port 8080 is protected. No deployment, GPU execution, CUDA compilation, service restart, model loading, enablement of tracing on production or automatic merge.

## One-shot host validation for OpenClaw MAIN

Fetch exact draft PR HEAD and verify a clean worktree. From repo root run:

```bash
ccache c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_lane_evidence_test.cpp -o /tmp/issue58_lane_evidence_test
/tmp/issue58_lane_evidence_test
git diff --check main...HEAD
git diff --name-only main...HEAD
```

**Required integration gate:** run GCC C++20 `-fsyntax-only` for actual translation unit `src/runtime/engine/engine.cpp` using the same existing CUDA 13.4 header configuration that validated #73, without nvcc, full build, GPU work or production actions. Capture exact compiler command, exit status, warnings, GCC/ccache version and HEAD. Inspect actual code paths to confirm `find_admission_lane` is using exactly the original direct-fit and retained-eviction predicates and that telemetry does not force extra probes. Optionally run existing #73 classifier test.

Perform at most **two** narrowly scoped corrections within four files for reproducible compiler/fixture defects; rerun all gates after amendments. Do not interpret standalone header tests as integration proof. Post final evidence to the draft PR; return `ISSUE58_LANE_VALIDATED_DRAFT` or `ISSUE58_LANE_BLOCKED`. Leave PR draft, no merge or issue closure. No Telegram approvals for routine host validation.

## Subsequent decision gates
After this source milestone, a **separately authorised** supervised candidate trace can distinguish per-lane observed denial, retained-state behavior and long-lane policy paths on C2 with rollback/production C1 safety. Specific rejection causes inside `can_admit_lane` and actual queue-entry/admission timestamps remain future work. Issue #58 must remain open until end-to-end starvation is corrected and qualified.
