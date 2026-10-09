# Issue #58 — experimental KV transfer follow-on (checkpoint 2026-10-09 14:20:50 UTC)

Read [the authoritative continuation handover on main](https://github.com/toddballinger/ninfer-5080/blob/main/docs/ISSUE58_CONTINUATION_HANDOVER.md) first.

## Verified
- PR #63 merged into `main`: squash `960ebe2ad2e2545a3d996a82740847c1df1cc1cc`. Foundation-only; 3/3 final host tests PASS; no physical yielding.
- Experimental source on this branch: `2efaa3bbe15173529df91df5223e7dde593ae179` (the test commit before this documentation addition). New `src/runtime/engine/issue58_kv_pinned_staging.h` owns `cudaMallocHost` buffer, noncopyable and non-move-assignable, move-constructible; its cleanup **does not track in-flight CUDA operations**.
- Latest Brain test UTC 2026-10-09 14:20:50 as `toddballinger`: `ccache version 4.12.3`, `ccache c++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only` against CUDA headers with extracted four files: `PINNED_STAGING_COMPILE=PASS`, exit=0, `PRODUCTION_BEFORE=200`, `PRODUCTION_AFTER=200`. No CUDA device code, GPU execution or production deployment.
- Prior failure `sudo: interactive authentication is required` was solved by compiling as ordinary user in disposable source directory. Prior syntax failure was literal `\\n` characters in C++ header, fixed at `2efaa3b`.

## Pending engineering work
1. Reconcile **selected** experimental transfer files onto a **fresh branch from main**. This branch forked from earlier foundation work and includes commits now superseded by PR #63; **do not merge this historical branch wholesale**.
2. Make pinned staging lifetime actually safe: tie copy completion to a stream/event ownership object, prevent reset/free/move before completion, reliably surface allocation/transfer/free failure. Current lifetime contract is caller-only and not adequate for active requests.
3. Validate page geometry and allocator mapping stability; physically copy into pinned memory, synchronize and commit durable independent bytes. Test deterministic source/destination remap and negative cases.
4. Implement real atomic reservation/release/reacquisition and rollback with complete ordinary + MTP/GDN/recurrent/hidden/pending/frontier state preservation, without relying on caller booleans as proof.
5. Compile host tests first, use ccache for all C++/CUDA builds. Run a tiny GPU roundtrip only with adequate VRAM headroom and explicit approval if interrupting C1 would be needed. Later supervised C2 qualification must demonstrate short-turn TTFT and exact long generation resume at true 128K.

## Hard safety rules
Production Brain C1 on port 8080 must remain untouched; no implicit stop/start or deployment, no background unattended GPU work. Do not claim functional reversible yielding or production C2 support. Issue #58 remains OPEN. User prefers one bounded Bash block with OSC52 clipboard output, and SSH session must remain open.

## Tomorrow's first action
Inspect `main` HEAD and this branch; create a fresh follow-on branch from `main`, port only independent experimental KV transport artifacts with their tests, then design completion-owning pinned staging prior to live GPU tests.
