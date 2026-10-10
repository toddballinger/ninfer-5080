# Issue #49 — Q3 SwiGLU T32 fallback route relevance, first bounded milestone

**Status: host-validated source-only route model; independent review complete, merge approval pending. Not a runtime measurement and not a CUDA optimization.**

Main base: `f510cf2f88b160c6b83fe294805fff6b9bc95e96` (after #69 merge). All route evidence below is taken from `src/ops/linear_swiglu/q3/q3_linear_swiglu.cu` at this base, not from a guessed generic GEMM policy.

## Source findings

- `q3_linear_swiglu_dispatch` selects specialized kernels for exactly T=1,2,3,4, before the large-T case.
- For `LinearPolicy::AllowA8` and T>=257 it chooses `q3_linear_swiglu_int8_launch` with 4096-token bounded activation tiles; T<=256 remains in A16.
- For remaining T>=5, loop launches `q3_linear_swiglu_t32_mma_launch` for every full 32-token tile, then 16/8/4/1..3 A16 tail routes. **Therefore the spill-heavy T32 kernel is reachable for T=32..256 under AllowA8, but not T=5..31.** T=32 is within the histogram's `T=17..32` bin even though T=17..31 do not launch T32.
- Under `LinearPolicy::A16Only`, the A16 tiled route can execute at T>=257 as well. Thus claiming universal T32 unreachability above 256 is incorrect: it depends on policy. The production relevance question must verify **the actual qualified configuration uses AllowA8**.
- The same source already contains atomic dispatch-call counters and a `Q3 SwiGLU call histogram` printed at process exit. Its bins are T=1,2,3,4,5..8,9..16,17..32,33..64,65..128,129..256,257+. These are **operator calls, not kernel launches or milliseconds**. Also the 17..32 bin is ambiguous for T32 reachability.

Issue #37's compiler census reported the T32 outlier (40 registers, 120 B stack, 712 B spill loads, 584 B spill stores) from CUDA 13.4 / sm_120a on an earlier v1.5 source. These are *compiler resource measurements*, not observed throughput or TTFT impact. This PR does not rerun compilation or assert the resource figures remain identical on current source.

## Offline host-only route model

`tools/analysis/issue49_q3_t32_routes.py` accepts **operator token-count call records**, one JSON object per line: `{"tokens":32,"calls":5}`. It models dispatch to T32/16/8/4/1..3 or INT8 under a **user-explicit policy**, producing weighted operator-call bands, estimated launch counts and calls with T32.

Example:
```bash
printf '%s\n' '{"tokens":31,"calls":10}' '{"tokens":32,"calls":5}' '{"tokens":256,"calls":2}' '{"tokens":257,"calls":8}' | python3 tools/analysis/issue49_q3_t32_routes.py -
```

**The input is not automatically captured from production.** Synthetic input demonstrates correct parsing and route math but cannot prove frequency. To assess actual relevance, supply legitimately collected non-sensitive call-count records from a separate authorised offline benchmark run. Do not fabricate workload data or equate request-token size with operator T without evidence.

Existing exit-time histogram is a useful *coarse* first read. It can bound low/high possible T32 call frequency, but a per-T=32 breakdown is needed for exact count. This analysis model does not modify or enable the server's existing counters.

## Validation / independent gate

```bash
python3 -B -m unittest -v tests/host/test_issue49_q3_t32_routes.py
python3 -m py_compile tools/analysis/issue49_q3_t32_routes.py tests/host/test_issue49_q3_t32_routes.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected **6 tests**. One-shot host validation should record Python version, HEAD, each command RC, changed-file names and output; test direct CLI JSONL piping as shown above. A failing assertion or unexpected dispatch-source change is a blocker, not permission to alter production code.

**Explicitly excluded:** CUDA compilation/patches, GPU profiling, synthetic performance claims, real traffic acquisition, restarting active C1 service, modifying runtime instrumentation or claims of measured TTFT. Real performance decision requires representative *measured* operator frequency, actual kernel latency at T32/crossover, percentage TTFT impact, compiler resource evidence for current code, and operator acceptance of any future benchmark. Do not close #49 based solely on this route model.

## ChatGPT → OpenClaw handover

Complete source and tests on branch `automation/chatgpt/ISSUE49_Q3_T32_ROUTE_AUDIT`; PR draft contains this plan. MAIN should fetch expected PR head, verify clean worktree and base, run only the above host checks, capture exact evidence, and update draft PR. Perform at most two deterministic local fixes within these three files; otherwise return a single blocker report. No local-model worker required for simple Python tests; do not loop broken tool-call invocations. No merge, deployment, issue closure or intermediary Telegram updates. Return `ISSUE49_VALIDATED_DRAFT` or `ISSUE49_BLOCKED`. ChatGPT independently reviews the final diff, then asks human approval before merge.

## Host validation and independent review (2026-10-10)

OpenClaw MAIN verified code HEAD `153358806470b2a3c219f695346a3518180b3436` using Python 3.14.4: 6 unit tests RC0; py_compile RC0; sample JSONL CLI RC0 (25 synthetic operator calls, 7 calls with T32, 21 modelled T32 launches); git diff/scope checks RC0. Three intended files only. Evidence: https://github.com/toddballinger/ninfer-5080/pull/70#issuecomment-6096975860.

Independent ChatGPT review inspected the entire three-file diff and its source route mapping. The dispatch model distinguishes T32-reaching operator calls from modelled kernel launches, explicitly gates the 257+ crossover on AllowA8, and does not infer runtime frequency or TTFT from a synthetic histogram. No production/CUDA edits. **Suitable for merge as a CPU audit utility only**, subject to human approval. The validation evidence attaches to the earlier code HEAD; this appended documentation changes no executable code.

The next gate for Issue #49 is a separately authorised, representative measurement of actual operator shapes and latency on the qualified profile, not an automatic kernel retune. Leave Issue #49 OPEN.
