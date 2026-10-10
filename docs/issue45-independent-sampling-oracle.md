# Issue #45 — independent CPU numerical sampling oracle (first one-shot)

Status: **CPU-only fixture validation passed on OpenClaw host; independently reviewed, merge decision pending**. This is an audit-only PR, not a CUDA sampler fix and not grounds to close issue #45.

## Frozen base and source map

Main base: `318bc8a1378d3d02302a2708831235db13520b75`.

- `include/ninfer/ops/sampling.h` gives the public contract: greedy argmax ignores penalties, positive temperature applies presence/frequency once, top-k cap 20, min-p threshold relative to highest weight, top-p cumulative threshold relative to **pre-truncation** candidate total, seeded counter RNG.
- `src/ops/kernel/sampling_device.cuh` implements adjusted logits, candidate ordering and filtering (CUDA device implementation, **not invoked by this PR**).
- `src/ops/kernel/sampling.cuh` dispatches row/partial/final sampling and zero-temperature greedy path.
- `src/runtime/contract/sampling.cpp` resolves defaults and overrides and validates input parameter ranges.
- `tests/ops/test_sampling.cpp` already contains an FP64 CPU *oracle inside a GPU-backed test*. It is useful but **does not satisfy the independent test-design requirement by itself**.

## Independently specified numerical fixtures

`tests/host/test_issue45_sampling_oracle.py` uses Python standard-library `math`, `random`, `unittest` only, with no NInfer imports and no CUDA. Its analytical expectations are written from exact integer-weight ratios rather than snapshots of production output:

| Probe | Logit/exact weight input | Expected result |
|---|---|---|
| baseline | 6:3:1 | 0.6, 0.3, 0.1 |
| single presence penalty | count on weight 6; presence=ln2 | 3:3:1 ⇒ 3/7,3/7,1/7 |
| double-penalty discriminator | incorrect twice-applied ln2 | 1.5:3:1 ⇒ 3/11,6/11,2/11 (MUST differ) |
| frequency penalty | 8:4:2; count=2 on 8; freq=ln2 | 2:4:2 ⇒ .25,.5,.25 |
| min-p/top-p denominator-order | 6:3:1, min_p=.25, top_p=.65 | retain 6+3 ⇒ 2/3,1/3; denominator incorrectly recomputed after min-p retains only 6 |
| greedy | tied top logits | smallest token ID; ignores penalties |
| temperature two | 4:1 scaled by sqrt | 2/3,1/3 |
| padded rows | [0,0,1000] token_domain=2 | .5,.5 for first two, padded 1000 excluded |

Oracle's `seeded_draw` checks only Python harness reproducibility; **it intentionally does not claim GPU counter RNG parity**. Real seeded GPU comparison and OpenClaw structured output loop remain separate qualifications.

## Execution and acceptance

```bash
python3 -B -m unittest -v tests/host/test_issue45_sampling_oracle.py
python3 -m py_compile tests/host/test_issue45_sampling_oracle.py
git diff --check
```

OpenClaw validation on 2026-10-10 at source HEAD `0cc5c3361deae16db2d5f4c7a89c7bf612a7ac24` (Python 3.14.4): `python3 -B -m unittest -v tests/host/test_issue45_sampling_oracle.py` RC0, 8 tests; `python3 -m py_compile tests/host/test_issue45_sampling_oracle.py` RC0; `git diff --check` RC0; only this document and the standalone test changed. Evidence: PR #68 comment 6096693593. These are **host-reported**, not GPU/runtime observations.

## Findings and limitations

Source-level audit: the public contract and existing GPU-backed oracle describe a top-20 capped adjusted-logit sampling policy, not general unlimited top-k. The independent fixtures are designed to detect two specific failure modes: duplicate penalties and a different top-p denominator after min-p. They are *not* an empirical GPU comparison.

This first milestone ends with independently specified standalone fixtures and source mapping; it does **not** prove actual CUDA behavior, the no-thinking preset client mapping, statistical GPU sample distribution, or production performance. Those require separately authorised follow-up rather than enlarging this one-shot.

OpenClaw LOCAL-WORKER's unexecuted `<tool_call>` output blocked implementation. ChatGPT authored this small fixture directly through GitHub instead. Do not count that formatting failure as an engineering strike. Keep Issue #45 open after this first milestone; runtime sampler parity and OpenClaw structured-output qualifications remain unverified.

## Independent review (2026-10-10)

Reviewed PR #68's complete two-file diff and host evidence. The eight numerical examples use independently specified integer-weight probability ratios, rather than GPU-generated expected results. In particular, the single-penalty and min-p/top-p denominator-order fixtures distinguish two known bug classes. The test implementation is a CPU *contract oracle*, not a test of NInfer's current CUDA implementation. Python's `random.Random` test is explicitly only a self-consistency check, not NInfer counter RNG equivalence. The code scope is isolated, changes no runtime behavior, and is **suitable for merge as an audit/test milestone, subject to operator approval**. This is not proof that GPU sampling matches the contract; leave #45 open for GPU equivalence and client sampling mapping, and do not represent the issue as fully complete.
