# Issue #58 — reversible yielding: foundation scope and ownership handoff

PR #63 was **MERGED** as foundation-only (squash `960ebe2ad2e2545a3d996a82740847c1df1cc1cc`). It establishes fail-closed resolved-boundary classification, host-only suspension charge arithmetic, and an ordered proof-gate protocol. None of these components can release GPU memory or make a physical lane available.

## Files retained in PR #63

- `src/runtime/engine/issue58_yield_boundary_facts.h`: pure, testable boundary predicate. The target runtime's read-only predicate checks real lane state and rejects unresolved/unmatched frontiers, absent KV, and unsupported DFlash.
- `src/runtime/engine/issue58_suspended_ownership.h`: move-only *metadata*, not an ownership handle for actual GPU pages. `checked_charge_totals` rejects overflow and missing request identity, without sampling live allocator usage.
- `src/runtime/engine/issue58_transfer_gate.h`: host-only proof-stage ordering: Active → Quiesced → BackedUp → Released → Restored. Caller assertions are **not** attested CUDA/allocator proofs. No automatic rollback exists.
- Host-only tests for these three components.

## Experimental transfer work moved to separate branch

All experimental KV page geometry, CUDA page-image copying, move-only host image storage, allocator capture-plan adapter and corresponding tests are preserved on branch **`issue58-kv-transfer-followon`** at original ancestor `e6223a9aa02ede4886c847b5caefdc7c8d032ae8`. They are **not present in PR #63**. This includes the `PagedKVPool::plane_order()` read-only accessor.

See that branch's version of this document for allocator/GDN source analysis, design constraints and deferred GPU roundtrip. The stand-alone CUDA GPU test only compiled; it was not run because available VRAM was 786 MiB versus 1,500 MiB preflight guard.

## Merge evidence and follow-on acceptance

1. Verify the resulting PR diff has no experimental KV transfer code, CUDA runtime execution, memory freeing, or live scheduler activation.
2. Compile/check retained host tests and target-local boundary predicate against the final PR revision.
3. Confirm scope and documentation match actual delivered code.
4. Keep production C1 unchanged and preserve Issue #58 as **OPEN** for the functional follow-on implementation.

A successful foundation merge is **not** evidence that real MTP3 state can yield, release a lane and restore exactly.

## 2026-10-09 final checkpoint

Final PR #63 host tests: 3/3 PASS on `8cf81fb4`; C1 health HTTP200 before/after; read-only target predicate and host-only safeguards merged. `SuspensionTransaction` is noncopyable **and nonmovable**, while moved-from `SuspendedOwnershipToken` invalidates its charge. Full final executable link and GPU checkpoint/restore tests were deferred and must never be represented as PASS.

The separate `issue58-kv-transfer-followon` experiment at `2efaa3bbe15173529df91df5223e7dde593ae179` contains `KvPinnedStaging` and its compile-only API test. Latest `ccache c++ -fsyntax-only` validation PASS as normal Brain user at 14:20:50 UTC (C1 HTTP200/200). **No runtime CUDA allocation/copy or stream lifetime test was run**. The helper currently relies on caller-proven stream completion; do not use it for active checkpointing until lifetime-safe transaction design and tests exist. Branch predates merge: port changes onto fresh `main` based branch rather than merging old branch as-is. Issue #58 remains OPEN.
