# RTX 5080 128K Concurrency Qualification

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
- allocator/page rounding;
- another measured runtime reservation.

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
