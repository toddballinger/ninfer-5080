# Issue #58: reversible C2 lane yielding — ownership audit and implementation gates

Status: **design/provenance only**, not an implemented suspension mechanism. Production remains C1. Evidence: Brain read-only inventory 2026-10-09 12:03 UTC, source e01f75e (PR #61 implementation), production health 200 before and after. The inventory did not identify a general suspend/resume API.

## Verified code boundaries

- `src/runtime/engine/concurrent_executor.h`: owns `slots_[lane]`, deadlines, output-stream publication and queue admission. `run_decode_round` calls `Program::decode_batch`, then `resolve_pending_batch`. `find_admission_lane` only considers empty slots. `abort_lane` is destructive.
- `src/targets/qwen3_6/impl/runtime/program.h`: `SequenceState` holds optional `SequenceKVBundle` (text and optional backend paged KV), `tail_hidden`, rewrite checkpoint hidden, sequence ledger, prefix identity, rope delta, text/backend KV valid frontiers, MTP draft tokens/count and rewrite checkpoint. `RequestControl` holds lifecycle, pending candidate, sampler settings, timings/statistics, and possibly an incomplete prefill session.
- `src/targets/qwen3_6/impl/runtime/program_impl.h`: `ProgramImplCore::decode_batch` dispatches ordinary, MTP and DFlash. `resolve_pending_batch` resolves accepted tokens and clears/commits pending state. `abort_lane` calls `clear_lane` (not suspension).
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h`: transient MTP/DFlash round workspaces and ingress/egress. DFlash comment states draft work is produced/verified within one completed round; this does not imply persistent state can be discarded.
- `src/targets/qwen3_6/impl/runtime/linear_state_slots.h`: linear/recurrent state slots currently indexed directly by lane, so transferring ownership or restoring a lane also requires accounting for these buffers.
- `src/targets/qwen3_6/impl/runtime/dflash_context.h`: rewrite-checkpoint helpers appear specialized to masked draft/prefix flows; **not** a general request suspension API.

## Proposed safe implementation stages

1. **Fence (implemented as a read-only predicate, build PASS):** target-private `ProgramImplCore::at_resolved_yield_boundary(lane)` returns false unless `RequestControl.lifecycle == Active`, `pending.kind == None`, no active prefill, no unpublished device round or in-flight state mutation. The scheduler must evaluate it only after successful resolve/publication and before a new decode. Add unit tests for rejection during pending and prefill.
2. **Transfer object:** define a move-only `SuspendedSequence` ownership token spanning sequence state, request control, paged-KV allocation, linear/recurrent lane state and any retained/vision/prefix references. Do not merely copy `SequenceState` (its tensors are device views and its allocation owns global capacity).
3. **Free or preserve capacity:** a suspended sequence must either keep its charged memory as a documented resident reservation or export its per-page KV and state to separately accounted host buffers, releasing page tables and slot resources atomically. Holding both decode slots' KV in place while claiming reusable slots cannot magically free VRAM/KV reservations. Fail closed if there is insufficient host memory or scratch space.
4. **Resume:** reconstruct backend allocation, recurrent state and `SequenceState` on a permitted slot; verify ledger/frontier/MT P alignment and sampling determinism; reconnect original output sink/deadline/cancellation without duplicate stream publication.
5. **Fairness:** yield only at a committed boundary if a feasible queued short request has been blocked by lane capacity, not by KV constraints. Keep an age/quantum mechanism so suspended longs resume and cannot starve.
6. **Acceptance:** deterministic ordinary decode, MTP3 and DFlash resume exactness, vision/prefix and tool-call histories, cancellation/disconnect/timeout during suspension, memory leak and host-transfer pressure; then Brain GPU pair+short replay with original C1 watchdog. The 24.073s false-hint short delay is the regression baseline.

## Non-solutions

- Calling `abort_lane` then prompting again: truncates/corrupts output and loses continuity.
- Forcibly reducing `max_tokens`: changes user-visible semantics.
- Blindly swapping scheduler `slots_` pointers: target KV, recurrent state, and runtime lane IDs still refer to the old lane.
- Counting a successful compile or short TTFT with opt-in lane isolation as proof of genuine preemption.

**Merge gate:** do not merge active scheduling/suspension behavior without an end-to-end exact-resume test. Contract docs, diagnostics and fail-closed scaffolding may merge separately.


## Verified incremental build — 2026-10-09 12:09 UTC

- User's Brain compilation of branch `issue58-reversible-yield-foundation`, source commit `90b82cc24c6bc8bc455aef8814476dd9e0edc72b` succeeded: `BUILD_RESULT=PASS`, `BUILD_EXIT_CODE=0`.
- Binary SHA-256: `00ea2e0e80372a6d88950d38d822f55d507b7cfb967157583430eb6b57aeb02f`.
- Implemented only a **read-only target-private predicate** `ProgramImplCore::at_resolved_yield_boundary(uint32_t lane) const noexcept`. Returns true only for active lifecycle, no pending candidate, no prefill, present KV, non-retained sequence, contiguous ledger/frontier, and valid text/MTP KV frontiers. It does **not** save, free, move, restore or resume a request. No scheduler call-site or active yielding proved.
- Compilation is **not an executable correctness or resume-exactness test**. No new GPU test was reported for this foundation code. Running production C1 has not been changed.
