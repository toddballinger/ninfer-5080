# Issue #58 — reversible yielding: foundation scope and ownership handoff

PR #63 is **foundation-only**. It establishes fail-closed resolved-boundary classification, host-only suspension charge arithmetic, and an ordered proof-gate protocol. None of these components can release GPU memory or make a physical lane available.

## Files retained in PR #63

- `src/runtime/engine/issue58_yield_boundary_facts.h`: pure, testable boundary predicate. The target runtime's read-only predicate checks real lane state and rejects unresolved/unmatched frontiers, absent KV, and unsupported DFlash.
- `src/runtime/engine/issue58_suspended_ownership.h`: move-only *metadata*, not an ownership handle for actual GPU pages. `checked_charge_totals` rejects overflow and missing request identity, without sampling live allocator usage.
- `src/runtime/engine/issue58_transfer_gate.h`: host-only proof-stage ordering: Active → Quiesced → BackedUp → Released → Restored. Caller assertions are **not** attested CUDA/allocator proofs. No automatic rollback exists.
- Host-only tests for these three components.

## Experimental transfer work moved to separate branch

All experimental KV page geometry, CUDA page-image copying, move-only host image storage, allocator capture-plan adapter and corresponding tests are preserved on branch **`issue58-kv-transfer-followon`** at original ancestor `e6223a9aa02ede4886c847b5caefdc7c8d032ae8`. They are **not present in PR #63**. This includes the `PagedKVPool::plane_order()` read-only accessor.

See that branch's version of this document for allocator/GDN source analysis, design constraints and deferred GPU roundtrip. The stand-alone CUDA GPU test only compiled; it was not run because available VRAM was 786 MiB versus 1,500 MiB preflight guard.

## Merge acceptance

1. Verify the resulting PR diff has no experimental KV transfer code, CUDA runtime execution, memory freeing, or live scheduler activation.
2. Compile/check retained host tests and target-local boundary predicate against the final PR revision.
3. Confirm scope and documentation match actual delivered code.
4. Keep production C1 unchanged and preserve Issue #58 as **OPEN** for the functional follow-on implementation.

A successful foundation merge is **not** evidence that real MTP3 state can yield, release a lane and restore exactly.
