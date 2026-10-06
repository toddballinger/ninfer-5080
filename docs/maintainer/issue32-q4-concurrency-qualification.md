# Issue #32 — RTX 5080 Q4 C2/C4 concurrency qualification

## Purpose

Qualify real concurrent serving for the current RTX 5080 16 GB production profile while preserving the **131072 per-request logical context ceiling**, Q4-group64 KV, MTP-3, CUDA Graph, and the existing Qwen3.8-27B artifact/runtime behavior.

This branch starts with **measurement and correctness qualification**, not a speculative scheduler rewrite or KV-codec port. Current main already contains a bounded concurrent executor, shared KV-capacity accounting, admission protection/backfill logic, request queueing, batched decode support, and a concurrency benchmark harness. The first question is therefore whether the current Q4 path already provides useful C2 on the target machine.

Only if C2 is blocked should implementation expand, and the blocker must be classified as one of:

1. shared Main-KV capacity;
2. backend/MTP KV capacity;
3. per-lane workspace or CUDA-Graph residency;
4. admission/accounting policy;
5. scheduler/runtime architecture;
6. correctness defect under interleaving.

## Web-side changes on this branch

Base:

```text
BASE_MAIN=361535a798a43beef4aaad858183e105fb6f34a8
BRANCH=feature/issue32-q4-c2-concurrency
```

The first web-side commit changes `tools/bench/run_serve_concurrency.py` so the campaign can qualify the actual Q4 production path instead of silently forcing INT8 KV. It also records request-latency percentiles and key memory-planning fields in the point/summary artifacts.

```text
WEB_COMMIT_1=470800c6da9aa41f30d52874c96280fe60fa4a72
```

No GPU qualification is claimed by the web-side work.

## Required local handoff invariant

Before any local build/test work, the local checkout must be made identical to the remote branch and the exact HEAD recorded:

```bash
git fetch origin
git switch feature/issue32-q4-c2-concurrency
git reset --hard origin/feature/issue32-q4-c2-concurrency
git status --short
git rev-parse HEAD
git rev-parse origin/feature/issue32-q4-c2-concurrency
```

Expected before local mutation:

```text
LOCAL_HEAD == REMOTE_HEAD
WORKTREE_CLEAN=YES
```

Do not use `sudo git`.

## Local Phase A — host-only validation

No GPU/service interruption is required for these checks.

1. Syntax-check the benchmark harness:

```bash
python3 -m py_compile tools/bench/run_serve_concurrency.py
```

2. Build/execute the existing host-side serve-option tests and any benchmark/parser tests already covered by the normal test target.

3. Confirm `--kv-dtype q4` is emitted by the concurrency harness and that the expected server-start identity is `q4-group64`.

4. Run a benchmark `--dry-run` with the exact intended Issue #32 arguments before touching the production service.

Host-only acceptance:

```text
PYTHON_SYNTAX=PASS
HOST_BUILD=PASS
SERVE_OPTION_TESTS=PASS
DRY_RUN_Q4_IDENTITY=PASS
NO_GPU_USED=YES
```

If this phase exposes a benchmark-only defect, fix it on this branch, commit it locally, push it, and prove local/remote HEAD equality again before GPU work.

## Local Phase B — current-Q4 C1/C2 feasibility gate

Use the canonical Brain RTX 5080 service/artifact identity. Resolve the exact artifact from the production service rather than guessing a path.

Keep:

- `--max-context 131072`
- `--kv-dtype q4`
- `--prefill-chunk 1792`
- MTP-3
- CUDA Graph enabled
- Qwen3.8-27B production artifact
- shared KV capacity resolved with `--kv-capacity auto` first
- C1 and C2 only for the first gate

Suggested campaign shape:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_8_27b="$ARTIFACT" \
  --mode mtp3 \
  --sampling stochastic \
  --suite decode-saturation \
  --suite corpus-makespan \
  --concurrency 1 \
  --concurrency 2 \
  --max-context 131072 \
  --kv-capacity auto \
  --kv-dtype q4 \
  --prefill-chunk 1792 \
  --output "$OUT"
```

The OpenClaw/local-worker mixed-arrival case remains a required follow-up even if the fixed corpus passes. At minimum qualify:

- one long foreground request plus one short request;
- 96K + 32K;
- 112K + 16K;
- approximately 118K foreground plus short tool/worker traffic.

Do not advance to C4 merely because C2 allocates. C2 must first demonstrate useful completed work per wall-clock second and bounded foreground regression.

## Required C1/C2 evidence

Capture and retain:

- exact commit/artifact/GPU/driver/CUDA identity;
- server command and request-log JSONL;
- resolved shared KV capacity;
- KV payload bytes;
- runtime reservation;
- planned slack;
- CUDA Graph allowance and observed bytes;
- aggregate decode tok/s;
- per-request latency mean/p50/p95/max;
- admission wait/queueing;
- average decode batch;
- MTP drafted/accepted tokens and acceptance ratio;
- cancellation and lane reclamation;
- post-run service health.

For MTP correctness, compare matched C1/C2 acceptance and explicitly rule out request/lane/state-row misassociation before treating lower acceptance as ordinary contention.

## Decision tree after C2

### C2 passes

If C2 preserves the 128K-class foreground profile with useful aggregate throughput, keep current Q4 and proceed to:

1. mixed-arrival OpenClaw qualification;
2. cancellation/reclamation stress;
3. repeated-run stability;
4. C4 for shorter mixed requests only after C2 acceptance.

### C2 fails from memory

Record which reservation grows per lane. Do **not** immediately port rk2v4-e8.

Issue #32 candidate order remains:

```text
current Q4-group64
  -> rk4v4-e8 vs KVarN K4V2-G128
  -> fully qualify stronger candidate
  -> rk2v4-e8 only if materially more compression is still required
```

### C2 fails from scheduler/runtime architecture

Do not hide the defect inside a KV-codec experiment. First compare the concrete blocker with current upstream scheduling/context-ownership work. Split a scheduler-specific issue only after a reproducible runtime blocker is established.

## Astra-oracle review gate

After local host/GPU evidence is complete, dispatch an independent read-only Astra review with:

```text
TASK_MODE=REVIEW
ISSUE=32
BRANCH=feature/issue32-q4-c2-concurrency
BASE=361535a798a43beef4aaad858183e105fb6f34a8
HEAD=<exact local/remote aligned SHA>
QUESTION=Does the evidence prove useful and correct current-Q4 C2 at the 131072 per-request ceiling, and is any proposed implementation the smallest justified next step?
```

Astra must review:

- branch diff and commit provenance;
- host-test evidence;
- C1/C2 GPU evidence;
- memory-residency accounting;
- MTP lane/state-order correctness;
- mixed long/short fairness;
- cancellation/reclamation;
- whether C4 is justified;
- whether any code change is actually required.

No PR should be created until Astra returns PASS, or a bounded REVISE packet is resolved and re-reviewed.

## Commit / remote alignment rule

Every local change made during handoff must be committed and pushed before review evidence is treated as final. Immediately before Astra and immediately before PR creation:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/feature/issue32-q4-c2-concurrency
```

Required:

```text
WORKTREE_CLEAN=YES
LOCAL_HEAD == REMOTE_HEAD
```

The PR should remain uncreated until those conditions and the Astra gate both pass.
