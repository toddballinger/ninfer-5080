# Upstream Sync Status

This fork selectively integrates changes from [Neroued/ninfer](https://github.com/Neroued/ninfer) rather than attempting to remain commit-for-commit identical with upstream.

The fork contains RTX 5080-specific work, true-128K memory-fit changes, mixed-Q4/Q5 tuning, MTP/DFlash2 integration, and Vision changes that intentionally diverge from upstream. For that reason, GitHub's "ahead/behind" count is not a reliable statement of which upstream work has or has not been considered.

This file is the authoritative restart point for future upstream reviews.

## Upstream review checkpoint — assessed at fork `f6088f85`

- Upstream repository: `Neroued/ninfer`
- Upstream branch: `master`
- Upstream commits assessed through: `9e163eee4b8acec21ab0ac765107b6a3f287b217`
- Checkpoint commit: `9e163eee` — `perf(ops): route the q4/q5 a16 input projections by column band`
- Assessment date: **2026-09-20**
- Fork branch assessed against: `main`
- Fork head at assessment: `f6088f856627045f280e5be8a76fba068b6979e4`
- Fork checkpoint description: RTX 5080 128K Vision v1.3 validation

**Discovery of new upstream work should start with commits after `9e163eee`.** The separately tracked candidate queue below may still contain commits at or before that checkpoint that were assessed but deliberately left for later integration.

"Assessed through" means the upstream history reachable from that commit has been considered for relevance to this fork. It does **not** mean every upstream commit was merged. Some changes were already present under different SHAs, some were semantically backported, some were retuned for the RTX 5080, and some were intentionally deferred or skipped.

## Integration state after that review

The review checkpoint above is intentionally tied to the fork state that existed when the assessment was performed. Subsequent integration is recorded separately:

| Milestone | Commit | Meaning |
|---|---|---|
| `9e163eee` semantic port | `4b62aca386a0a214049201ebeb2a422b0cb609ce` | RTX 5080-specific Q4/Q5 attention and GDN input-projection routing port. |
| PR #3 merge | `33546d7d5be6d82eaac5e4a87a3f7e578f8a1a13` | Merged the `9e163eee` semantic port into `main`. |
| PR #5 merge | `b44b1958c301ec6bf4d18973a97d7b42fa6733aa` | Reconciled the validated v1.3 rolling-tool prefix-checkpoint feature into the PR #3 lineage. |
| PR #4 docs merge | `c8439fbcb89a4daf74cf2692a9425930998c763f` | Documentation-only descendant of `b44b1958`; no runtime source changed. |
| `a9a0d10a` semantic port | `00e8e47fa6001067257f7ae6594c2deeabaed590` | RTX 5080-qualified Q5 A16 LinearAdd T=1 Split2 and >512 narrow-tail routing while preserving the fork's 4096-row GEMV and C64 crossover policy. |
| PR #7 merge | `a074864142e6c3dee7bdb5e3b9fb8932e0fc0ac8` | Merged the qualified `a9a0d10a` semantic port and its commit-scoped validation record into `main`. |
| PR #8 docs merge | `a76ad8fef77d8ad1bc5ceba34e932431ce7df9a5` | Documentation-only follow-up recording the PR #7 merge in the upstream ledger; no runtime source changed. |

Runtime tree `b44b1958` was qualified with the exact 118,001-token workload at 131,072 context / 131,072 Q4 KV and produced 1378.85 tok/s prefill, 71.44 tok/s decode, 44.74% MTP acceptance and 2.31 tok/round. This qualification is attached to that exact runtime tree rather than described as a floating "current" result.

The subsequent `a9a0d10a` semantic port at `00e8e47fa6001067257f7ae6594c2deeabaed590` was independently qualified with the same exact workload and produced 1376.30 tok/s prefill, 71.53 tok/s decode, 44.74% MTP acceptance and 2.31 tok/round. Relative to `b44b1958`, that is -0.185% prefill and +0.126% decode and is classified as end-to-end equivalent within noise.

## Status vocabulary

| Status | Meaning |
|---|---|
| `MERGED` | Upstream change, or its semantic equivalent, is present in the fork. |
| `MERGED_AND_RETUNED` | Upstream mechanism is present, with RTX 5080-specific routing/tuning. |
| `ALREADY_PRESENT` | Equivalent functionality was already in the fork under a different history. |
| `PORT_AND_RETUNE` | Relevant upstream work identified for a future RTX 5080 integration pass. |
| `RECONCILE` | Relevant, but overlaps fork-specific code and must be compared rather than cherry-picked blindly. |
| `DEFER` | Potentially useful but not currently important to the validated Qwen3.8-27B RTX 5080 path. |
| `SKIP` | Not relevant to the current fork target. |
| `REJECTED_AFTER_TESTING` | Evaluated experimentally and deliberately not adopted. |

## Upstream integration acceptance policy

Upstream performance work is not merged merely because an isolated kernel or microbenchmark is faster. A change is eligible for integration when it has a demonstrated or prerequisite relationship to the validated RTX 5080 / Qwen3.8-27B production path and does at least one of the following:

- materially improves end-to-end prefill throughput or latency;
- materially improves decode throughput or latency;
- materially improves MTP proposal, verification, acceptance, or speculative-decoding performance;
- produces a substantial GPU-memory reduction that creates useful capacity on the 16 GB RTX 5080, such as higher-quality weights, larger KV/context, Vision/MTP capacity, or another production capability;
- fixes correctness, stability, serving, compatibility, or lifecycle behavior required by the validated production path.

Microbenchmark-only gains on an operator, quantization, geometry, or token interval that the production model does not exercise are normally `DEFER` or `SKIP`, unless they are a prerequisite for a later production-relevant change. Upstream crossover points measured on other GPUs are treated as hypotheses and must not be assumed to be optimal on the RTX 5080.

**Production relevance first, benchmark optimization second.** Before performing an exhaustive RTX 5080 retune, first establish that the affected operator / shape / quantization / token range is actually reached by the production workload. The goal is not to merge every upstream optimization or reach "0 commits behind"; the goal is to improve the practical Qwen3.8-27B configuration within the 16 GB RTX 5080 envelope.

## Upstream-derived changes already integrated

The fork history contains a number of upstream fixes and optimisations. Several were applied as semantic backports, so their fork SHAs intentionally differ from upstream.

| Upstream commit | Fork commit | Status | Notes |
|---|---|---|---|
| `b88c0f6f` | `3230f587` | `MERGED` | Completes host uploads before returning. |
| `641ef3e7` | `37800a4d` | `MERGED` | Skips NFC normalisation for text that is already pure ASCII. |
| `9f0575bb` | `77683b0d` | `MERGED` | Restores required BF16 definitions for the Q4 top-k kernel build. |
| `6d1da9ce` | `52776b66` + `0c3caf9a` | `MERGED_AND_RETUNED` | Backports the Q5 LinearAdd aggregate-cliff fix and then retunes the affected routing for RTX 5080. |
| `9e163eee` | `4b62aca3` / merge `33546d7d` | `MERGED_AND_RETUNED` | Ports Q4/Q5 A16 attention/GDN input-projection column-band routing while preserving fork-specific Q4/Q4, Q4 value_z, A8, 4096-geometry and workspace-aware behavior. |
| `a9a0d10a` | `00e8e47f` / merge `a0748641` | `MERGED_AND_RETUNED` | Ports Q5 A16 LinearAdd T=1 Split2 and >512 narrow-tail routing for the 5120-row shapes. Preserves 4096-row T=1 residual GEMV, RTX 5080 C64 crossover bands, residual-GEMV infrastructure and shared Q5 rowsplit behavior. RTX 5080 A/B showed 25-31% T=1 wins, 34-35% wins at T=513, 24% wins at T=1025, and no material regression at T=896. |
| `4cece118` | `4f453463` | `MERGED` | Semantic backport fixing malformed generated UTF-8. |
| `1d13c213` | `e5ce9863` | `MERGED` | Semantic backport binding the configured CUDA device across the engine lifecycle. |
| — | `dd2cb034` | `MERGED` | Follow-up compile repair required by the CUDA-device lifecycle backport. |

These changes were consolidated into the validated upstream integration history, including:

- `357a683b` — **Merge validated RTX 5080 upstream v1.1 integration checkpoint**
- `a2c2afb7` — **merge: validated RTX 5080 128K Vision v1.2**
- `f6088f85` — **docs: publish RTX 5080 128K Vision v1.3 validation**

The v1.1 integration checkpoint was validated on the RTX 5080 16 GB with Qwen3.8-27B, 131072 context, 131072 Q4 KV capacity, prefill chunk 896, MTP-3, and Vision. Its merge record explicitly includes the host-upload fix, ASCII NFC fast path, Q4 top-k BF16 build fix, and RTX 5080-qualified Q5 LinearAdd routing.

## Upstream commits assessed but not yet integrated

The following upstream work was identified during the 2026-09-20 review but was not integrated as part of that checkpoint. These entries form a candidate queue and should **not** be rediscovered from scratch.

| Upstream commit | Status | Current assessment |
|---|---|---|
| `bb844c43` | `DEFER` | **Validated cold path.** RTX 5080 semantic candidate `f69e95dd` was independently retuned and passed operator A/B plus the exact 118,001-token / true-128K qualification at 1374.84 tok/s prefill, 71.28 tok/s decode, 44.74% MTP acceptance and 2.31 tok/round. The production Q4 geometry histogram contains no generic `N=4096 K=5120` calls, so no current end-to-end benefit is demonstrated. Keep the candidate work for future prerequisite/relevance reassessment; do not merge solely for microbenchmark gains. |
| `beedffa0` | `DEFER` | **Validated cold generic path.** Retunes generic Q4 `7168x5120` Linear. Qwen3.8 contains Q4 `7168x5120` attention parents, but production consumes them through the specialised `attn_input_proj` path. The exact 118,001-token production histogram contains no generic `N=7168 K=5120` call. |
| `d3c125ed` | `DEFER` | **Validated cold generic path.** Retunes generic Q4 `34816x5120` Linear. The current RTX 5080 Qwen3.8 profile uses Q3 for the main MLP `gate_up` family, and the exact production histogram contains no generic Q4 `N=34816 K=5120` call. |
| `5b4303c0` | `DEFER` | **Not applicable to current production quantization.** Retunes Q4 `34816x5120` LinearSwiGLU, while the current Qwen3.8 RTX 5080 profile deliberately uses Q3 for MLP `gate_up`. Runtime evidence shows the hot production SwiGLU path is Q3, so the Q4 fused route is not the active path. Reassess only if the production artifact returns to a Q4 gate/up profile or a later change depends on it. |
| `028eb61e` | `DEFER` | Q8 predicated GEMM cache-policy tuning. Potentially relevant to MTP/Q8 paths but lower priority than mixed-Q4/Q5 execution. |
| `b39de4d5` | `DEFER` | Q5 pure-Linear route tuning with strong microbenchmark gains, but upstream notes no confirmed model-level caller for the relevant current paths. |
| `dc58675f` | `SKIP` | Sparse-MoE Q5 routed-down tuning for Qwen3.6-35B-A3B, not the dense Qwen3.8-27B target of this fork. |
| `1d8587bc`, `05507ab0`, `5f5fccab` | `DEFER` | NVFP4/W4A4 improvements. Useful upstream work, but not part of the current validated mixed-Q4 true-128K artifact. |
| `b9219f3f` → `98dada0e` → `8eaed538` | `DEFER` | Custom Jinja/chat-template stack and literal-content fix. Desirable for serving compatibility, but large enough to treat as a separate frontend integration milestone. |

## Current queue precedence — 2026-10-06

The historical upstream checkpoint and kernel observations below remain valid provenance, but they no longer define the immediate execution order.

The current bounded P1 gate is issue #32: qualify the **existing Q4-group64 C2 operating point** at a 131072 per-request logical ceiling before starting another narrow kernel implementation or a rotated-E8 codec port. Preserve MTP-3, Vision-2048 and CUDA Graph, and classify any failure as KV-capacity, per-lane Graph/workspace/state, admission/accounting, or scheduler/runtime architecture.

If current Q4 C2 is blocked by scheduler/runtime architecture, compare the blocker against the newer upstream serving work already folded into issue #17 before authorizing a fork-specific serving subsystem. If it is blocked by memory/KV capacity, issue #32 may proceed to rk4v4-e8 and then rk2v4-e8 only if justified. If C2 qualifies, return to the kernel queue headed by #18 -> #48 -> #30.

Accordingly, any older statement below naming the Q3/A8 retune as the unconditional “next performance priority” should be read as a **kernel-lane priority after the C2 gate**, not as the repository-wide immediate next action.

## Hot-path follow-up after the 118,001-token production trace

The exact true-128K qualification also identifies the operators worth prioritising for future performance work:

- Q3 LinearSwiGLU `34816x5120`: the production trace recorded 8,448 calls in the `T>=257` bucket and 960 calls at `T=4`. In the current Qwen3.8 RTX 5080 profile, `gate_up` is Q3 and the prefill policy is `AllowA8`; therefore bulk prefill reaches the fork-specific Q3/A8 fused SwiGLU path. This is a genuine production-hot path and should be treated as a fork-specific RTX 5080 tuning target rather than inferred from upstream Q4/NVFP4 work.
- Q4 Linear `5120x6144`: the trace recorded 8,384 calls at `T=896`, 64 at the final partial chunk, and 960 at `T=4`. Upstream `fb0035bc` adds a tuned shape implementation, but its `T>192` route remains `r64_c128`, the same bulk mechanism used by the fork, while the fork already owns an exact Qwen3.8 `T=4` path. It therefore does not currently demonstrate a production-hot route improvement at the observed widths; keep as `DEFER` unless RTX 5080 A/B evidence shows a gain at the actual partial/full chunk widths.
- Q4 Linear `5120x17408`: the trace recorded the same hot `T=896` / partial-chunk pattern and `T=4` calls. Current upstream `master` has no registered generic Q4 `5120x17408` shape, so there is no upstream route to port. Any improvement here is fork-specific work.
- Upstream `61535c6c` tunes Q4 `6144x5120`, which is the inverse geometry of the production-hot `5120x6144` projection and does not appear in the exact production histogram. It is not a current production target.

**Next performance priority:** profile and retune the fork-specific Q3/A8 LinearSwiGLU bulk-prefill path at the actual production widths, especially `T=896` and the final partial chunk, before spending effort on cold upstream Q4 shapes. Any candidate must still pass the exact 118,001-token true-128K qualification and preserve the 16 GB memory envelope.

Historical evidence makes this priority especially strong. The E101 Q3/A8 redesign was previously the largest single prefill gain in the fork: a controlled long-context comparison recorded approximately 1421.1 tok/s with E101 enabled versus 761.7 tok/s on the prior A16 large-prefill path (+86.6%), with a later five-run mean of 1391.9 tok/s and 0.18% CV. The current Q3 implementation still uses the folded INT8 schedule `Q3Int8SwiGluSchedule<64,256,16,128,3,1>`; its source comment records that this initial schedule was inherited from the RTX 4090 route and that Blackwell-specific retuning was intended to follow correctness/performance gating. That makes a focused RTX 5080 schedule retune at the current `T=896` production chunk width a higher-value experiment than importing additional cold Q4 shape work.


## Why GitHub can still report many commits "behind"

The fork and upstream histories diverged and some upstream commits were reapplied or adapted as new fork commits. Git therefore sees different commit identities even when equivalent code is already present.

For example:

- upstream `b88c0f6f` corresponds to fork `3230f587`
- upstream `9f0575bb` corresponds to fork `77683b0d`
- upstream `1d13c213` corresponds to fork `e5ce9863`
- upstream `4cece118` corresponds to fork `4f453463`

Accordingly, a large "behind" count must **not** be interpreted as "all of these commits still need merging."

## Procedure for the next upstream review

1. Read the checkpoint at the top of this file.
2. Fetch upstream `Neroued/ninfer:master`.
3. Review only commits after the recorded upstream checkpoint SHA.
4. Classify each relevant change using the status vocabulary above.
5. Integrate candidate work on a dedicated branch, not directly on `main`.
6. Prove production-path relevance before doing exhaustive kernel retuning; use runtime geometry/operator evidence where available.
7. Benchmark hardware-sensitive routing on the RTX 5080 rather than assuming upstream GPU crossover values apply.
8. Re-run the true-128K text and Vision validation before merging into `main`.
9. Update this file with:
   - the new upstream checkpoint SHA,
   - assessment date,
   - fork head used for the review,
   - decisions made,
   - fork SHAs for any integrated/backported work.
10. After the integration is validated, optionally add a Git tag such as `upstream-reviewed-through-<short-sha>` to make the review point machine-readable as well as documented.

## Principle

The goal of this fork is **not** to reach "0 commits behind upstream."

The goal is to preserve the validated RTX 5080 / Qwen3.8-27B / true-128K configuration while selectively incorporating upstream fixes and performance mechanisms that improve it without breaking its memory envelope, correctness, MTP behaviour, or Vision support.
