# Issue #56 — independent probabilistic MTP oracle (one-shot A3)

**Status:** Host reported 8 tests passing at earlier HEAD `b68f29891c4cd42f2b42fa134d4bd210e9d151ac`; subsequent independent review identified and corrected a near-zero residual cutoff. Updated 9-test HEAD requires revalidation. No model/CUDA/runtime modifications. Parent #56 remains open.

## Verified base and current source

Base main: `c718510dbc4182a2f831138d64bc114a61e1ccfd` (after #68 merge).

The current tree already has **a sparse-proposal rejection-sampling interface**: `include/ninfer/ops/speculative_round.h` documents seven draft positions, 16 sparse candidates per row, positive-temperature accept `min(1,p(d)/q(d))`, and residual `max(0,p-q)`. `src/ops/kernel/speculative_round.cuh` includes `speculative_sparse_probability`, `speculative_pick_sparse_residual`, and the original one-hot/greedy-draft route. These are not GPU-validated by this PR; do not imply #56 requires implementing an entirely missing mechanism.

The first gate is an **independent semantics oracle**, without NInfer imports, linking, model artifact, or CUDA. Code: `tests/host/test_issue56_speculative_oracle.py`.

## Mathematical correctness contract

For target distribution `p`, proposal distribution `q`, drafted symbol `d` with `q[d]>0`:

- accept with `a(d)=min(1,p[d]/q[d])`;
- if rejected, correct with `r(x)=max(p[x]-q[x],0) / sum_y max(p[y]-q[y],0)`;
- if all draft tokens are accepted, draw the bonus from the next target distribution;
- for EOS/EOG, stop at terminal emitted token, never emit later speculative drafts/bonus.

For a single verification position, analytic marginal identity is
`Pr(output=x) = q[x] min(1,p[x]/q[x]) + (sum_d max(q[d]-p[d],0)) * r[x] = p[x]`.
The one-hot, zero-q, near-zero-q and equal-distribution cases are included. In the exactly equal p=q case, residual mass is zero and there can be no rejection.

## Tests and independent fixtures

Nine Python `unittest` cases cover two-token exact example p=(.75,.25), q=(.25,.75), acceptance (1, 1/3), correction (1,0), 500 deterministic randomized marginal identities (including q zeros), near-zero/zero-q, equal distributions, greedy ties, EOS/EOG accepted-prefix stopping, seeded harness replay, correction/bonus shape and invalid-distribution inputs.

**Important boundaries:** Python's Random is a local deterministic test harness, **not** the NInfer GPU counter-based RNG. The `greedy_step` probe is mathematical boundary behavior, not a full CUDA temp-zero API equivalence check. The oracle's terminal handling is an external reference contract and not evidence that the deployed implementation already enforces it. In sampled float implementations, exact equality may vary with precision and finite numerics. This milestone establishes an independently calculated expected result, not GPU distribution parity.

## Local preflight and OpenClaw handover

ChatGPT ran an equivalent source copy locally on Python 3, with 8 tests PASS and `py_compile` PASS, before publishing the GitHub source. OpenClaw must execute the **committed branch file** to confirm its exact content and environment.

Branch: `automation/chatgpt/ISSUE56_CPU_SPECULATIVE_ORACLE`; head SHA: read current PR HEAD at validation time. Expected scope is precisely this document and `tests/host/test_issue56_speculative_oracle.py`.

One-shot validation commands from repo root:

```bash
python3 -B -m unittest -v tests/host/test_issue56_speculative_oracle.py
python3 -m py_compile tests/host/test_issue56_speculative_oracle.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Record Python version, exact branch/head/base, command RCs, case count, stdout/stderr and scope. If a deterministic fixture bug is identified, allow up to two tightly scoped fixes followed by **all** checks. Do not transform tests to match production CUDA output, and do not delegate mere terminal validation to the malfunctioning LOCAL-WORKER. If blocked, one consolidated evidence report, no iterative Telegram relay.

Result: `ISSUE56_VALIDATED_DRAFT` with evidence published on PR, otherwise `ISSUE56_BLOCKED` identifying smallest correction. Leave PR draft for independent ChatGPT review, no auto-merge and no Issue #56 closure.

**Explicit exclusions:** GPU/CUDA compile, real model or 128K benchmarks, MTP throughput sweep, changing production verification, OpenClaw service restart, model conversion, production deployment, or expanding this milestone into a new speculative algorithm.

## Independent review and bounded correction (2026-10-10)

OpenClaw reported 8 tests RC0, py_compile RC0, diff check RC0 on Python 3.14.4 for head `b68f29891c4cd42f2b42fa134d4bd210e9d151ac`; evidence: https://github.com/toddballinger/ninfer-5080/pull/69#issuecomment-6096813098.

Independent ChatGPT review found one numerical edge-case risk: the CPU reference returned *no residual* whenever the positive residual mass was `<=1e-14`, although near-identical non-equal p/q can produce a tiny legitimate rejection and therefore requires a residual correction distribution. Replaced the arbitrary cutoff with an exact-zero check and added one explicit near-equal regression fixture. These are **test-only modifications**; production CUDA remains untouched. The exact updated branch requires rerun: nine tests expected; do not reuse previous 8-test PASS evidence to approve current HEAD. No merge/issue closure until independent review of the final verified head.
