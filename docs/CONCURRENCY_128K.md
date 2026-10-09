# RTX 5080 128K Concurrency Qualification

## October 9, 2026 — current-Q4 concurrency closeout

**Decision: keep C1 for interactive OpenClaw. C2 delivered 1.511x better corpus productivity but is NOT interactive-production-qualified because of >600-second TTFT outliers.** Scheduler/admission is tracked as P0 [#58](https://github.com/toddballinger/ninfer-5080/issues/58); read-only postmortem tooling merged in [PR #59](https://github.com/toddballinger/ninfer-5080/pull/59).

| Metric | C1 | C2 | C3 |
|---|---:|---:|---:|
| 75-job corpus completed | 75/75 | 75/75 | 66/75 (failed) |
| Corpus makespan | 4700.981s | 3112.180s | 4328.394s before abort |
| Queue timeout count | 0 | 0 | 1 (900s) |
| TTFT P50 | 0.25s | 0.29s | 0.30s |
| TTFT P95 | 0.37s | 2.25s | 647.54s |
| TTFT max | 0.38s | 664.72s | 813.70s |
| Completed requests with TTFT >600s | 0 | 2 | 4 |

C3 TTFT excludes 9 requests not completed; statistics are not fully matched. C2 occupancy included 1317 one-second samples with running=1 and waiting=1; C3 included 1788 with running=1 and waiting=2. Actual admission wait and deferral reasons are not yet explicitly instrumented; TTFT includes queue, prefill and initial decode. Long requests can have 65536 output-token allowances against **131072 shared** Q4 KV; future-token reservation and KV-aware eligibility are leading hypotheses, NOT confirmed causes.

Configuration: RTX 5080 16GB, same Qwen3.8-27B artifact, max context 131072, shared Q4 group64 KV capacity 131072, MTP3 draft3, Vision2048, host embeddings, CUDA Graph, rolling-tool, 75-job fixed-shuffle stochastic corpus seed 20260811. C1 pending timeout=180s; C2 and C3 timeout=900s.

Short decode pilots: C2 98.14 aggregate tok/s, minimum observed free GPU memory 624MiB. C3 87.17 aggregate tok/s, minimum free 466MiB, planned slack 228MiB. C4 failed before serving any request: engine runtime reservation required 3,692,193,024 bytes versus 3,688,397,824 available, short by 3,795,200 bytes (3.62MiB). C4 Graph-off and cache-budget proposals remain unqualified research.

Artifacts on Brain:
- C1: /home/openclaw/issue32-c1-campaigns/c1-20261008T150834Z-818276
- C2: /home/openclaw/issue32-c2-corpus/corpus-c2-20261008T232609Z-952752
- C3: /home/openclaw/issue32-c3-corpus/corpus-c3-20261009T005806Z-985819
- C4: /home/openclaw/issue32-c4-pilots/c4pilot-20261009T003705Z-981076

**Current-Q4 phase concluded, not deployed.** Issue [#32](https://github.com/toddballinger/ninfer-5080/issues/32) remains open for further KV codec/memory study (rk4v4-e8, KVarN, rk2v4-e8). Issue [#58](https://github.com/toddballinger/ninfer-5080/issues/58) owns scheduler/resource eligibility telemetry, bounded fairness, and interactive C2 qualification. Preserve production max-concurrency=1 until the P0 gate passes.


## Status

This document defines the current highest-priority bounded performance gate for the RTX 5080 production path.

The validated v1.5 baseline remains:

```text
max_context=131072
kv_capacity=131072
kv_dtype=q4-group64
max_concurrency=1
mtp_draft_tokens=3
vision_max_tokens=2048
cuda_graph=on
embedding_host=on
```

A recent production observation on the 16 GB RTX 5080 showed approximately:

```text
memory.used = 15148 MiB
memory.total = 16303 MiB
reported free ≈ 1155 MiB
```

That observation does **not** change the validated release profile by itself. It creates a high-value qualification question: whether the existing Q4 serving path can use available headroom for useful multi-request execution before the project spends the memory on a larger single-request context or changes KV representation.

Tracking:
- control queue: issue #17;
- memory / concurrency operating-point work: issue #32.

## Product objective

For OpenClaw, the first target is **true C2 local-worker execution** on one resident Qwen3.8-27B model.

The desired behavior is not two independent model instances. It is two active requests sharing:
- one resident weight set;
- one Engine;
- shared KV capacity;
- one shared runtime/workspace plan where the architecture permits;
- compact batched decode at round boundaries.

The product question is therefore:

> Can the current 128K-class Q4/MTP-3/Vision/CUDA-Graph profile admit and execute two useful local-worker requests concurrently with materially better aggregate task throughput and bounded per-request latency?

C4 is a secondary target for bursts of short tool/subagent work after C2 is qualified.

## Why this is ahead of narrower kernel work

A successful C2 operating point can reduce queue blocking between independent OpenClaw local-worker sessions and improve aggregate completed work across many small and medium requests.

This is potentially higher product leverage than a narrow single-request kernel gain because it changes the number of agent tasks that can make progress at once.

Do **not** assume C2 means exactly 2x throughput. The two requests share GPU compute, memory bandwidth and execution resources. Qualification must measure:
- aggregate decode throughput;
- per-request throughput;
- TTFT;
- admission wait;
- task completion wall time;
- fairness;
- VRAM and workspace headroom;
- MTP acceptance / accepted tokens per round;
- CUDA Graph behavior;
- cancellation and resource reclamation.

## First bounded gate: current Q4 only

Do not start by implementing a new KV codec.

First establish whether current Q4 can expose a better operating point by separating:
- `max_context`: per-sequence logical ceiling;
- `kv_capacity`: shared physical Main KV capacity;
- `max_concurrency`: admitted active-request count.

Initial sequence:

1. preserve `max_context=131072`;
2. keep Q4-group64, MTP-3, Vision-2048, host-mapped embeddings and CUDA Graph;
3. increase shared KV capacity only as far as the real RTX 5080 memory envelope safely permits;
4. test C1, then C2;
5. test C4 only for mixed shorter-request workloads after C2;
6. if C2 fails, identify the exact blocker before changing representation.

Potential blocker classes:
- insufficient Main or backend KV capacity;
- per-lane fixed state;
- duplicated request-transient memory;
- CUDA Graph family allowance;
- workspace scaling;
- scheduler/admission constraints;
- request/token/hidden-state row-association correctness under interleaved decode;
- allocator/page rounding;
- another measured runtime reservation.

If the blocker is scheduler/runtime architecture, compare it against the current upstream serving model before authorizing a fork-specific scheduler rewrite. Recent upstream work adds preemptive scheduling, continuation/checkpoint ownership, incremental execution permits, snapshot/replay recovery, mixed-arrival diagnostics and runtime metrics; these are comparison evidence, not automatic port requirements.

## Required request mixes

At minimum test:

```text
64K + 64K
96K + 32K
112K + 16K
~118K foreground + short background request
OpenClaw main/local-worker/compaction-style burst
two independent small local-worker tasks
```

The purpose is not to prove that two 128K-max requests can both simultaneously occupy 128K. Shared capacity is intentionally not divided equally by slot. The goal is to determine whether realistic simultaneous working sets fit and execute efficiently.

## Success criteria

C2 is production-interesting when it:

- preserves a 128K-class per-request logical ceiling;
- materially reduces queue blocking for independent local-worker work;
- improves aggregate completed-work throughput versus C1;
- keeps per-request latency degradation acceptable;
- preserves correctness, MTP semantics, Vision, prefix reuse and cancellation behavior;
- retains a practical physical VRAM safety margin.

If current Q4 qualifies, production work should prefer that result before a compressed-KV implementation.

If current Q4 does not qualify because memory is the limiting factor, issue #32 proceeds in this order:

1. `rk4v4-e8`;
2. `rk2v4-e8` only if the safer rotated-E8 point is insufficient and quality/performance gates justify the extra compression.

If the blocker is scheduler/runtime architecture rather than KV capacity, split that implementation work explicitly and keep the memory evidence in issue #32.

## Relationship to the v1.5 release

v1.5 remains the immutable production reference and C1 remains its documented validated serving profile.

This document describes the **next qualification target**, not a claim that C2 is already production-qualified.
