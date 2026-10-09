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


## 2026-10-09 follow-up: fail-closed boundary tightening (commit 6056dff)

The target-private read-only fence now rejects lane/sequence identity mismatch, zero/empty ledger, mismatched resident prefix identity length, non-contiguous ledger/execution frontier, text KV frontier mismatch, and MTP without backend KV or exact MTP frontier. The frontier calculation avoids unsigned wraparound. DFlash is explicitly ineligible until its context/rewrite transfer contract is established. No active scheduling, state transfer or production changes.

**Still not sufficient for suspension:** even a true fence result does not establish that device kernels are quiescent, that CUDA graph scratch is safe, that linear-attention state is movable, that KV page table and allocator charges are independently transferrable, or that scheduler-owned streaming/cancellation can be atomically rebound. Callers must not interpret it as permission to evict a lane.

**Next validation:** incremental compile of the single changed runtime header using existing PR #63 build directory and ccache; no GPU inference, no service stop. Runtime tests must later cover pending, prefill, empty, frontier mismatch, prefix mismatch, absent MTP backend allocation, lane mismatch and DFlash rejection. Do not claim these tests have run. Follow with a separately approved, isolated checkpoint/restore correctness gate before attempting live preemption.


## 2026-10-09 host-testable fence extraction

PR #63 commits `d50ae01`, `b383efa`, `1e5c929` introduced `src/runtime/engine/issue58_yield_boundary_facts.h` and `tools/tests/issue58_yield_boundary_test.cpp`. The existing Qwen runtime predicate now converts real target state to plain scalar facts and delegates its decisions to the *same* constexpr function exercised by the standalone host test. The test covers active/empty, pending, prefill, retained, missing KV, bounds and lane mismatch, prefix and ledger mismatch, execution wraparound, text/MTP KV mismatch, missing backend allocation, unsupported DFlash, and valid ordinary/MTP cases. It uses `static_assert` plus runtime `assert`.

**Unverified at commit time:** host C++ test and changed target compile. No suspension ownership token, GPU page export, recurrent-state relocation, resumption or scheduler preemption is implemented. Keep the PR draft and production C1 unchanged. The next Brain check must first compile/run this small host-only test (no CUDA), then invoke the existing incremental `ninfer-serve` build only once to verify target integration.


## Move-only reservation scaffold (2026-10-09)

Commits `b5c8e70` and `bff0d09` add `src/runtime/engine/issue58_suspended_ownership.h` and `tools/tests/issue58_suspended_ownership_test.cpp`. The explicit `SuspendedOwnershipCharge` records request ID, original physical lane, text/backend KV, recurrent, hidden and host offload byte counters. The token is noncopyable and movable, and has no operation to release/resume a lane. Its test verifies move/copy type properties and retained counter values. **This is purely a host-side accounting design scaffold, not ownership of actual device memory.** No production code constructs it. No runtime behavior changed.

Do not allow scheduler lane reuse on the strength of this token: it has no allocator handle, CUDA event/stream fence, recurrent state storage, checkpointed hidden tensors, prefix reference owner or restoration transaction. The next implementation step must bind these real owners to the token and prove transfers preserve accounting under cancellation/failure; no live preemption is authorised.


## Additional fail-closed transaction protocol (2026-10-09)

Commits `6948d21` and `57172ae` add `src/runtime/engine/issue58_transfer_gate.h` and a host-only test. The noncopyable state machine imposes ordered stages `Active -> Quiesced -> BackedUp -> Released -> Restored`, failing terminally on invalid order or missing assertions for resolved boundary, CUDA quiescence, both Q4 text and MTP backend KV backups, recurrent-state backup, hidden-state backup, allocator release and restore. Cancellation enters terminal Failed state. This models **proof obligations**, not actual transfer.

**Critical distinction:** the `TransferProof` booleans are inputs supplied by hypothetical future ownership code, *not independently attested evidence*. No caller currently emits them; there are no CUDA fences, page migration, GDN replay-state backup, real reservation accounting, cancellation unwinding, streaming reconstruction or scheduler integration. An external caller must not signal release/restore until independently verified, and failure after partial physical transfer will need a separate rollback/recovery owner. This proof gate does not itself guarantee exception-safe rollback.

Next source implementation must first locate concrete KV allocation types and GDN state views, define who retains/relinquishes page-table entries and physical backing, and test exact snapshot/restore in isolation. Only after that should the transaction gate become an enforcement check on a real transfer path.


## Resident-charge overflow guard (2026-10-09)

Commits `f6ee549` and `9f5cb8a` extend the existing host-only `SuspendedOwnershipCharge` with `checked_charge_totals`: separately reports summed resident-device charges (text KV + backend KV + recurrent + hidden) and host offload bytes; rejects integer overflow and zero request identity. Host test now exercises exact totals and fail-closed invalid inputs. No claim that these counters reflect live allocator occupancy; all caller-provided charges must later be populated from authoritative page pool and state-store handles. This is not actual GPU state migration. Never subtract these totals from GPU usage until physical pages are provably released.


## Concrete allocator and GDN ownership findings (source audit, 2026-10-09)

Source: `src/core/paged_kv_cache.h` and `src/core/linear_attention_state.h`, plus Qwen3.6 runtime implementation.

**Paged KV:** `PagedKVPool` owns the fixed GPU plane tensors, shared block tables, free-page IDs, in-use row map and entitlement/mapped counters. `PagedKVAllocation` is already **noncopyable, movable RAII**; it holds a pool pointer, private page IDs, entitlement and bound row. It exposes `page_ids()`, `page_entitlement()`, `mapped_page_count()`, `bound_row()`, `release()`, `unbind_row()`, and `publish_mapping()`. Simply moving an allocation or putting it into a `SuspendedOwnershipToken` **does not release any physical page group or row**. Nor does merely unbinding a row release KV page capacity. Destruction/release frees pages and entitlement; this is destructive unless the physical plane data has first been safely snapshotted and synchronized. Page layout may be `PageMajor` or `HeadMajor`, with K/V/scales stored in multiple planes; per-page byte copying needs layout-aware offsets rather than assuming contiguous bytes or a single plane. Backup must cover both text and MTP backend pools, plus page-ID to logical-page order.

**GDN recurrent:** `LinearAttentionStatePool` is a fixed-capacity GPU backing view, not a per-request allocation. It exposes `slot_bytes()`, `copy_slot_to_host(src, host, stream)` and `copy_slot_from_host(host, dst, stream)`, already used by the target for decision-frontier snapshot/restore. `LinearStateSlots::current_state_slot` maps directly to physical lane. A suspended request must own a separately allocated, size-checked host image and synchronize copy completion before reuse. Existing shared checkpoint buffers cannot serve as independent concurrent suspension owners.

**Required lossless transaction (NOT implemented):** resolve an MTP round -> synchronize all relevant streams/events -> snapshot per-plane text/backend KV pages and logical mapping + recurrent slot + tail/rewrite hidden + request state -> validate independently owned backups and their sizes -> atomically retire page allocations and relinquish lane -> serve short request -> reserve and remap sufficient pages on restore -> rehydrate every KV plane and GDN slot + hidden + sampler/prefix/MTP state -> rebind scheduler request and continue output without duplication. Fail closed and keep original active if backup fails before release. Once page release has occurred, rollback/resume must be guaranteed a capacity reservation or retry queue; the proof-state machine alone cannot guarantee this.

**Do not implement live release yet:** the paged pool does not expose a per-plane page-copy/export API. First add a target-private export/import API with audited layout addressing and exact GPU roundtrip tests, then independently verify page counters before/after. Existing `copy_slot_to_host` supplies GDN mechanics but not a whole-request snapshot. No service changes, no C2 rollout.
