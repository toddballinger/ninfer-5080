# Issue #43 — CUDA module-loading static investigation handover

**Pinned baseline:** `2c3510cd74dea0983e146dc2c3cf650d58ddd4fc`  
**Scope:** Documentation-only. No CUDA execution, measurements, new instrumentation, kernel changes, service changes or production configuration changes.

## Source observations (bounded, not exhaustive)

The following entries come from exact source lines in the pinned checkout. A symbol match establishes only that the source references an API or option; it does **not** establish whether that path is compiled into the production profile, executed during startup, captured by a CUDA graph, or visited after the service reports READY.

- **CUDA attribute calls:** `src/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu:57` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **CUDA attribute calls:** `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu:37` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **Graph API references:** `src/core/decode_graph.cpp:35` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **Device memory API references:** `src/core/arena.cu:42` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **Device memory API references:** `src/targets/qwen3_6/impl/runtime/program_impl.h:2481` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **Vision option mentions:** `apps/cli/main.cpp:253` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.
- **Vision option mentions:** `apps/cli/options.cpp:169` contains a matching source reference. This is a candidate inspection point, not proof it executes in the production configuration.

Some searched patterns may have no selected source match. This must not be interpreted as proving the functionality is absent.

## Conclusions expressly **not** established

- The active `CUDA_MODULE_LOADING` value, its initialization timing, and whether the current deployment uses CUDA lazy or eager loading.
- Which CUDA functions are resident before `/v1/models` first reports ready.
- Whether graph capture touches every production-reachable long-prefill, Vision, MTP3, constrained-decision or recovery kernel.
- Whether CUDA code/module VRAM grows after READY, or whether startup headroom is sufficient for all permitted workloads.
- Whether a source reference to a function attribute operation implies a deferred CUDA module load; it does not, by itself.
- Any cold-start VRAM delta or the applicability of the external Strata memory figure to NInfer.

These unknowns cannot be resolved by source inspection alone. No claims about the outcome of Issue #43's runtime acceptance gate are made here.

## Proposed next gate — separately authorised nonproduction work

1. Capture the effective CUDA module-loading setting and CUDA driver/toolkit versions in an isolated cold-start environment.
2. Record device-memory usage after context initialization, model loading, planner reservation, graph capture, READY and first legal requests, with repeated cold-process samples.
3. Exercise a documented feature-touch matrix for long-context/tail, Vision, MTP3 boundaries, constrained output and relevant recovery paths; record first-touch deltas without assuming the static list is complete.
4. Compare LAZY and EAGER under otherwise identical profiles, recording headroom and time-to-ready effects.
5. Determine whether startup-resident modules plus a reserve, selective warm-touch or an explicit loading policy is the smallest robust remedy **only if** a material runtime gap is demonstrated.
6. Require a separate production-restore and safety gate before any service-level change.

**Acceptance boundary:** This document provides candidate inspection locations and a measurement plan; it neither verifies nor disproves the post-READY memory-residency invariant.

## Evidence

See the companion `route-references.txt`, mechanically selected and line-numbered from the pinned checkout. The evidence is supplied to the independent reviewer separately rather than duplicated in this document.
