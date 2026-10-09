# Issue #58 — continuation handover (updated 2026-10-09 14:20 UTC)

> **Read first in a fresh chat.** This is an operational record of verified work, PRs, tests, environment and next actions. It does not claim production C2 is ready.

## Latest authoritative checkpoint — 2026-10-09 14:20:50 UTC (2026-10-10 00:50 Adelaide)

**Use this section first; later historical sections may describe older PR states.**

- **PR #63 MERGED** into `main` by squash commit `960ebe2ad2e2545a3d996a82740847c1df1cc1cc`; foundation source HEAD before squash `8cf81fb43eecc1fddd4406e42309455e991bbc37`. Final host gate **PASS 3/3** (`issue58_yield_boundary_test`, `issue58_suspended_ownership_test`, `issue58_transfer_gate_test`); C1 HTTP200 before/after. Added move-clearing charge token and made `SuspensionTransaction` strictly nonmovable. Targeted C++ objects compiled with ccache earlier. **The full target build/link at final revision, CUDA/GPU tests and physical yielding remain NOT RUN / NOT IMPLEMENTED.**
- **Issue #58 OPEN**: working C2 yield/checkpoint/resume, allocator release and starvation prevention remain unsolved; PR #62 historical draft regression remains separately tracked. Production `ninfer-local-model.service` **C1** remains untouched, HTTP200 as last checked.
- **Experimental follow-on branch** `issue58-kv-transfer-followon` pinned staging HEAD `2efaa3bbe15173529df91df5223e7dde593ae179`. It predates the PR #63 squash; **do not merge it wholesale**. Port desired transfer files onto a new branch from current `main` or carefully reconcile history first.
- **Latest pinned-staging validation: PASS** 2026-10-09 14:20:50 UTC. Ran as `toddballinger`, `ccache version 4.12.3`, extracted source at exact SHA `2efaa3b`, `ccache c++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only` against the CUDA headers: `PINNED_STAGING_COMPILE=PASS`, `VALIDATION_EXIT_CODE=0`, C1 HTTP200/200. **No GPU execution, CUDA rebuild or production deployment.** This validates syntax and type contract, **not** pinned allocation lifetime or stream synchronization.
- The earlier `sudo -n -u openclaw` script failed with `sudo: interactive authentication is required`. **Do not depend on noninteractive sudo** for ordinary compile gates. Successful workaround uses temporary git repository/source extraction as `toddballinger`, no sudo, no checkout of protected `/home/openclaw` trees. Previous syntax error from literal `\\n` in `issue58_kv_pinned_staging.h` was fixed in commit `2efaa3b`.
- **Next implementation gate:** an independently owned, actually stream-safe pinned staging/copy transaction that tracks in-flight copies and refuses early reset/free/reuse; small host/compile-only tests before any standalone GPU roundtrip. Existing `KvPinnedStaging` relies on caller-managed stream completion; its destructor `cudaFreeHost` assumes completed CUDA work. Treat as *experimental*, not a proven lifecycle solution. Follow with allocator page capture stability, release/reacquire transaction, complete MTP/recurrent/hidden state, exact restoration and C2 fairness/TTFT qualification.
- **No overnight/background GPU work authorised**; stop at documentation checkpoint. For every Brain script use ccache for **C++ and CUDA** launchers, prefer bounded existing builds, preserve C1 and copy concise report to clipboard via OSC52 without ending SSH session.

## Objective and invariants

Fix **C2 admission starvation** for NInfer Qwen3.8-27B MTP3 on a **single RTX 5080 16GB**, at **131072 context**, ultimately enabling two useful concurrent OpenClaw local-worker / other OpenAI-compatible requests **without cancelling, truncating, restarting or corrupting** active generations.

**Do not break production:** Brain's current `ninfer-local-model.service` runs healthy **C1** on port 8080, `--max-concurrency 1`. The service's original executable and unit must stay untouched until separately approved C2 rollout. All GPU experiments used a separate staged binary on port 18080, stopped C1 under supervision, armed an independent 360-second root watchdog and restored C1 at end. Verify HTTP200 on `http://127.0.0.1:8080/health`. Never run an unapproved test that takes production offline.

**For user scripts:** copy-paste Bash for Brain as `toddballinger`; script must automatically copy report via OSC52 for pasting back, and must *not* append `exit` to terminate SSH sessions. Use `ccache` and incremental build; do not needlessly recompile all CUDA again.

## GitHub development record

- Repo: `https://github.com/toddballinger/ninfer-5080`
- Issue `#58`: P0, **OPEN**, broader starvation, reversible preemption and C2 production qualification. Keep open until genuine reversible yield and rollout evidence.
- PR `#59`: forensic trace tools, merged earlier.
- PR `#60`: **MERGED** to `main` as squash `052658e72d118a8799b67ffdd922b98f2dd20ae0`. Adds `NINFER_SHORT_LANE_RESERVE=1` with C2; conservatively blocks a second long request, allows short backfill, trace `NINFER_ADMISSION_TRACE=1`; default OFF. Long threshold 8192; `NINFER_LONG_OUTPUT_THRESHOLD` bounded override. Not real preemption.
- PR `#61`: **MERGED** to `main` as squash `7f4e3fc14dba0b4026de74870db3a7a9cd67e246`. Adds optional top-level Chat Completions `ninfer_short_operation` boolean and `NINFER_TRUST_SHORT_OPERATION_HINT=1` trust opt-in. Requests without hint retain existing classification; default OFF. Hint changes **only admission class**, not output length, KV resource reservation or model behavior. Other protocols (Anthropic Messages and Responses) do **not** support hint yet; no harness emits it by default. No caller integration, no rollout.
- PR `#62`: draft `https://github.com/toddballinger/ninfer-5080/pull/62`, branch `issue58-c2-qualification`. Supervised regression reproduces client-neutral conservative fallback and **unsafe falsely hinted long**. Do not treat passed test as production clearance.
- PR `#63`: foundation-only `https://github.com/toddballinger/ninfer-5080/pull/63`, branch `issue58-reversible-yield-foundation`. Retains a **read-only** target-private `ProgramImplCore::at_resolved_yield_boundary(uint32_t lane) const noexcept`, host-only checked charge metadata and caller-asserted proof-stage ordering. Final narrowed host tests (3/3) and relevant ccache C++ integration objects PASS at `0fa1c2bb04be488ee3012ac9193496bcf2ffdb72`, but **full target link, CUDA and GPU validation have not been run**. Does **not** suspend, free, move, restore or resume live requests. Experimental physical KV transfer/backup/geometry and their tests are retained on separate `issue58-kv-transfer-followon` branch at starting SHA `e6223a9aa02ede4886c847b5caefdc7c8d032ae8` and absent from PR #63. Keep Issue #58 OPEN.

## Reproduced problem and GPU evidence

**Original regression:** two substantial generations monopolize C2 slots, delayed 256-token request timed out after ~100s. Trace `active=2`, `used_lanes=2/capacity_lanes=2`, while ~508 KV pages unreserved. Bottleneck **nonpreemptive lane occupancy**, not GPU KV page capacity.

**PR #60 lane reservation results:**
- C2 fairness 2×1536 long (test threshold1024) + short256: short TTFT0.234s, completion4.679s; long A completed18.248s, long B completed34.507s. All successful; production C1 restored.
- Mixed arrivals 2 longs + 8 short (2sec apart), 10/10 HTTP200; shorts max TTFT17.76s, deferred long B still completed at49.251s. C1 restored.
- All-long throughput cost: pair finish29.381s with policy OFF vs33.466s ON (~13.9% more wall clock).
- These runs show mitigation, not general preemption or strict real-time fairness.

**Classification need (PR #61):** historical JSONL 15,241 rows, 7,680 *occurrences* of `requested_output_tokens=16384`. Matched 3,838 request/completion records: 73.3% generated ≤512 tokens, 89.0% ≤1024, 95.7% ≤2048. Do not claim strictly OpenClaw-only from that log. Actual response length unknown before scheduling. OpenClaw `maxTokens=16384` frequently labels small requests long if using max-token classification; never simply reclassify all as short.

**PR #61 GPU validation:**
- C++ CUDA+ccache build PASS, `/home/openclaw/ninfer-issue58-classification-build/apps/ninfer-serve` SHA256 `0eaec0438ae5fb1620c34c152df951b17e07385c124d1c194dbf5ca2e3f4edb0`.
- API type tests string/number/array hint → HTTP400. Plumbing contract checks9/9 PASS.
- Supervised C2 2×1536 budget long + hinted16384: long A TTFT0.204s; hinted TTFT0.200s, completed31.284s stop; long B TTFT28.303s, completed46.820s; `CLASSIFICATION_HINT_REGRESSION=PASS`, `PRODUCTION_RESTORED=PASS`, `TEST_EXIT_CODE=0`. Logs `/tmp/issue58-c2-CYj73Q`. Hint improves admission TTFT but **does not guarantee quick completion**.

**PR #62 GPU qualification (2026-10-09 11:57 UTC):**
- Case A unhinted 16K and short256 alongside long1536: short TTFT0.216s, unhinted16K deferred TTFT14.550s, all HTTP200, `UNHINTED_CLIENT_REGRESSION=PASS`.
- Case B incorrectly hint a **long1536** as short: that hinted long TTFT0.226s; subsequent short256 **TTFT24.073s**, despite KV only52/2048 pages used; `FALSE_HINT_RISK_REPRODUCED=PASS`. Thus trusted hints **do not protect against misclassification**; real reversible preemption remains required.
- `C2_QUALIFICATION_TEST=PASS`, `TEST_EXIT_CODE=0`, `PRODUCTION_RESTORED=PASS`, `QUALIFICATION_RC=0`; logs `/tmp/issue58-c2-S1t0Jf`. This is a *successful reproduction of known unsolved failure*, not release readiness.

**PR #63 foundation build (user verified 2026-10-09 12:09 UTC):**
- Source `90b82cc24c6bc8bc455aef8814476dd9e0edc72b`, incremental CMake `ninfer-serve` target compiled and linked successfully: `BUILD_RESULT=PASS`, `BUILD_EXIT_CODE=0`.
- Binary SHA256 `00ea2e0e80372a6d88950d38d822f55d507b7cfb967157583430eb6b57aeb02f`.
- Predicate implementation in `src/targets/qwen3_6/impl/runtime/program_impl.h`, declaration in `src/targets/qwen3_6/impl/runtime/program.h`. Returns true only for active lifecycle, pending.kind None, no prefill, existing non-retained KV, contiguous ledger/execution frontiers and valid text/MTP KV. **Read-only. Does not suspend, save, offload, free, resume, reassign lanes or affect scheduler.** No GPU test or actual checkpoint correctness test has been performed on #63. No need for another massive rebuild solely to observe same predicate.

## Verified backend ownership and safety boundaries

- Scheduler `src/runtime/engine/concurrent_executor.h`: owns request queue, `slots_[lane]`, deadlines and streaming state. Calls `Program::decode_batch`, then `resolve_pending_batch`. Admission requires free slots. Never use `abort_lane` as yield.
- Target `src/targets/qwen3_6/impl/runtime/program.h`: `SequenceState` owns optional `SequenceKVBundle` (paged text and optional backend KV), hidden/tail/rewrite tensors, per-sequence ledger, prefix identity, rope delta, frontiers and MTP draft bookkeeping. `RequestControl` owns lifecycle/pending, sampler settings/timings/spec stats and optional prefill session.
- Target `src/targets/qwen3_6/impl/runtime/program_impl.h`: `ProgramImplCore::resolve_pending_batch` completes pending ordinary or speculative generation; `abort_lane` calls `clear_lane` and destroys state.
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h`: transient MTP/DFlash round buffers; cannot assume persistent lane state is preserved by discarding them.
- `src/targets/qwen3_6/impl/runtime/linear_state_slots.h`: recurrent/linear-attention state slot maps directly to lane index; must preserve/relocate recurrent state on yield.
- Specialized DFlash `rewrite_checkpoint_lane_bytes` is **not** a general suspend/resume facility.
- See `docs/ISSUE58_LANE_YIELD_CONTRACT.md` and `docs/ISSUE58_REVERSIBLE_YIELD_OWNERSHIP.md` for target contract/risks.

## Brain environment / production service

- Brain user `toddballinger`; source checkout under `/home/openclaw/` is **only traversable via `sudo -u openclaw -H`**. Main repo `/home/openclaw/ninfer-5080-issue32-concurrency`. Development worktree for PR #63 already used by prior incremental build; read current actual worktree path from user script (do not assume old SHA equals branch head).
- Earlier PR #61 worktree `/home/openclaw/ninfer-issue58-classification` and build `/home/openclaw/ninfer-issue58-classification-build`; test binary staged `/home/toddballinger/issue58-test/ninfer-classification-serve`.
- Production unit **systemd user** `ninfer-local-model.service`, service wrapper `/usr/local/sbin/openclaw-ninfer-user-service`; permitted stop/start via `sudo -n "$wrapper" stop|start` only for explicitly approved supervised GPU experiments. Production executable `/models/ninfer-builds/build-production-current-main-d5ee1bf/apps/ninfer-serve`; model `/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer`.
- Production C1 args: `--host 0.0.0.0 --port 8080 --model-id local-model --max-context 131072 --kv-capacity 131072 --prefill-chunk 1792 --kv-dtype q4 --spec mtp --draft-tokens 3 --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 180000 --vision --vision-max-tokens 2048 --default-thinking-budget 2048 --prefix-checkpoint-policy rolling-tool --embedding-host` (plus request JSONL path).
- Read-only health `curl -fsS http://127.0.0.1:8080/health` should return `{"status":"ok"}`.
- For every Brain command/script print concise report plus `OSC52` escape to copy text (base64 report to `\\033]52;c;...\\a`). Avoid a lot of repeated expensive CUDA rebuilds; `ccache` currently low hit rate (~0.46%).

## Next action for new chat (do as much GitHub work first)

1. Fetch `main`, Issue #58, and current PR #62/#63; reconcile HEAD/status; do not work in stale worktree. Review the **exact** predicate source and its declaration in PR #63.
2. Complete safe **fail-closed suspension boundary** tests first. The predicate is read-only; build PASS but there are zero registered CTest tests in the earlier build. Add actual unit tests where target state can be constructed/mocked; assert false for pending, prefill, empty, invalid frontiers.
3. Design/implement move-only suspension ownership with correct KV and linear state transfer and memory charges. Real suspension requires a request-safe boundary after `resolve_pending_batch` and before next decode, preserving output, deadline, cancellation, deterministic sampling, MTP alignment and prefix identity.
4. Do not call `abort_lane` or truncate maxTokens to release slots; do not claim working preemption until exact uninterrupted vs resume output matches, GPU MTP3 and pressure/cancellation gates pass.
5. After concrete code changes, give user one bounded Brain `ccache` build/test script, automatic OSC52 clipboard; only later ask for supervised GPU interruption when justified. Keep production C1 intact.
6. Update Issue #58, PR #63 and docs with evidence and pending gaps. PR #62 may merge as a **known-failure regression test** but never call it a resolved scheduler bug.

## Current hard truth

**The scheduler preemption feature is NOT IMPLEMENTED.** Current improvement is optional C2 long-lane isolation and an optional trusted-caller short-operation hint; both can be defeated by falsely hinted long work. The last verified work is a compiled read-only safe-boundary predicate. Further engineering is needed before C2 can safely become the default.
