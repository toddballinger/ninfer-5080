# Issue #58 — lane-starvation implementation contract (2026-10-09)

## Reproduced failure

Two sustained generations occupy both C2 decode lanes. A third, 256-token request is queued for >100 seconds even though its base KV reservation is only 6 pages, with 508 of 2048 main-KV pages unreserved. Trace: `active=2`, `used_lanes=2`, `capacity_lanes=2`, `head_pages_main=6`, `used_pages_main=1540`, `protection_phase=open`. C1 restored HTTP200 after diagnostic. See Issue #58 and Brain `/tmp/issue58-c2-jekQZ1/`.

This is **non-preemptive lane monopolization**, not a lack of total KV page capacity in this repro. FIFO tweaks/backfilling, larger KV allocation, and shorter pending-timeout values cannot admit the third request while both lanes are occupied.

## Existing implementation limitation

`src/runtime/engine/concurrent_executor.h` owns active `slots_[lane]` and calls `Program::decode_batch()` plus `resolve_pending_batch()`. Lanes are removed on completion or abort. `find_admission_lane()` only considers empty lanes. No public suspend/checkpoint/reinstate-lane API was found in the repository search. Calling `abort_lane()` to yield is **not equivalent to suspension** and would lose the active generation's KV/state. Never silently truncate/cancel a user's generation as a scheduling fix.

## Required implementation

1. Add explicit runtime-level *reversible active-lane suspension*, not just admission heuristics. Define a checkpoint contract retaining all state needed to continue exactly: model KV and backend residency/ownership; accepted/generated token and output history; MTP draft/target alignment; recurrent/vision/attention and prefix-sharing ownership as applicable; RNG/sampling state; in-flight batch and output-publication boundaries. Make clear which state can live in host RAM, and bound transfers by latency/throughput.
2. Only yield at a completed decode/resolve boundary when there is a *feasible queued head* blocked solely by active-lane capacity. Keep the head request's original deadline and ensure fairness for yielding active requests. Account for memory held by suspended requests or explicitly offload/free it; never double-count released pages.
3. Re-admit and restore a suspended request without replaying already streamed output, changing completion reason, corrupting token state, or losing the consumer/cancellation linkage. Preserve clean abort/error paths and shutdown draining.
4. Keep feature **off by default** until unit, CPU mock, and real GPU regression gates pass. Do not expose to production C1 without operator opt-in.

## Tests and acceptance gates

- Deterministic mock: 2 long active requests + delayed short request; short TTFT bounded despite occupied lanes, without cancelling/shortening either long response.
- Resume exactness: same generated tokens and final completion as uninterrupted baseline with matched RNG and deterministic decoding. Include MTP3, vision input, tool calls and prefix cache boundaries.
- Pressure/failure: multiple yielded requests, cancellation/timeout of queued and suspended requests, shutdown, disconnect, and resource exhaustion; no leaked slots/pages or deadlocks.
- Brain smoke: original replicated workload with `NINFER_ADMISSION_TRACE=1`, record short-request TTFT versus >100s baseline and model final output integrity. C2 full corpus run only after these gates; retain C1 as production throughout development.
- Do not close #58 or merge as scheduler-fix merely because instrumentation/tests compile.

## Current state

PR #60 now contains both the admission trace and an **opt-in, non-preemptive long-lane isolation mitigation** (`NINFER_SHORT_LANE_RESERVE=1`; default long-output threshold 8192, test override `NINFER_LONG_OUTPUT_THRESHOLD=1024`). Brain C2 GPU results: isolated short TTFT 0.234s versus previous 100s timeout; fairness regression 3/3 complete; overloaded mixed arrivals 10/10 complete with maximum short TTFT 17.76s; two-long A/B completion 29.381s policy off versus 33.466s policy on (+13.9% wall time). Production C1 restored after every test and remains unchanged. This mitigation does **not** provide true preemption, hard latency bounds under oversubscribed short arrivals, or evidence for all workload classes; reversible checkpoint/restore remains follow-on engineering if needed.
