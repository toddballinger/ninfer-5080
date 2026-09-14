# Qwen3.8-27B RTX 5080 128K reconstructed verified baseline

## Purpose

This document records the verified engineering state reconstructed after a period
of rapid experimentation in which source changes after the original frozen
reference were not committed individually.

The source state described here was validated as one integrated working-tree
state. The historical subdivisions below come from benchmark batches and source
audit evidence, not from individual Git commits.

Do not rewrite this history into artificial historical commits.

From the commit containing this document onward, use:

1. one logical source change per commit;
2. benchmark batch IDs in commit messages where practical;
3. permanent in-repository tooling for artifact creation;
4. SHA-256 provenance for model artifacts;
5. annotated tags for important verified milestones.

## Git ancestry

Original frozen reference:

- tag: `qwen38-mtp3-q4kv-chunk896-reference`
- commit: `a5935d129917b3135f910013b9f451aa6e28c470`
- subject: `Document Qwen3.8 RTX5080 MTP3 chunk896 reference`

There were no Git commits between that reference and this reconstructed
checkpoint. The complete source work existed as a dirty working-tree diff and
was validated through benchmark batches.

Batch 112 captured the complete pre-checkpoint binary patch:

- `/tmp/112-NINFER-20260913/full-working-tree.patch`
- SHA-256:
  `9e7685048b5ec384f50418cc55bbaa60e4ae3f16bcc25a8de103bae12f0d9696`

The `/tmp` path is historical evidence from that session and is not expected to
be permanent storage.

## Production runtime configuration

Validated production target:

- model family: Qwen3.8-27B
- GPU: NVIDIA GeForce RTX 5080 16 GB
- context: 131072
- KV cache: Q4 group64
- prefill chunk: 896
- speculative backend: MTP
- draft tokens: 3
- CUDA graph: disabled
- greedy decode for validation
- Vision support retained

## Production artifact

Current production-quality candidate:

`/models/ninfer-custom/qwen3_8_27b_5080_v8_q5mtp_24vz_7gv_q4.ninfer`

SHA-256:

`cd1839528cace785482878ea91191ae6c7cf7c40fa35e411e1a784283b4e1cff`

### Q4 value_z layers

24 layers:

`57,58,61,56,60,53,50,48,45,42,46,49,37,54,9,0,30,16,34,41,17,20,18,40`

### Q4 gate_value layers

7 layers:

`59,35,63,51,55,39,19`

### Total weight-memory saving

Approximately 210.625 MiB relative to the original Q5-weight artifact.

## Important comparison artifacts

Original Q5 artifact:

`/models/ninfer-custom/qwen3_8_27b_5080_v8_q5mtp.ninfer`

SHA-256:

`ace0edca91394e18a1ae3615b71731a0f90eb1c73a877043043bd90e35d76a67`

48-value_z Q4 artifact:

`/models/ninfer-custom/qwen3_8_27b_5080_v8_q5mtp_48q4-valuez.ninfer`

SHA-256:

`88fb735fd0e00a9dbcd3e8b30dae4d326fb4e3099a853ace4cd36c53056c7cc4`

One-value_z Q4 artifact:

`/models/ninfer-custom/qwen3_8_27b_5080_v8_q5mtp_1q4-valuez.ninfer`

SHA-256:

`34aa3111ec2f7ac2484c02c7eea87811f10795cf8887d0c859614530fda4b0a5`

Historical artifact-generation commands for these variants were not preserved
in Git. They must not be guessed. Permanent artifact tooling must be added to
the repository before creating future variants.

## Reconstructed source-change groups

### Q4 value_z / GDN support

Validated changes include:

- Q4/Q5-aware GDN input-projection validation;
- Q4 value_z execution path;
- Q4 value_z GEMV/SIMT routing;
- snapshot/record support;
- binder/load support for Q4 or Q5 value_z;
- qtype-aware workspace/capacity handling;
- associated tests.

Relevant benchmark sequence included Batches 001-032 and subsequent composition
testing.

### Q4 gate_value / attention support

Validated changes include:

- loader/binder acceptance of Q4 or Q5 gate_value;
- Q4 gate_value small-T execution;
- safe fallback for T >= 17;
- rejection of the invalid Q4/Q4 grouped-MMA formulation for larger T;
- associated attention tests.

The safe T >= 17 route is currently the primary suspect for the measured
prefill-throughput regression.

### CUDA first-use prewarm

Validated explicit prewarm coverage includes:

- Q4 small-T MMA;
- Q5 rowsplit MMA;
- Q4 rowsplit MMA;
- Q4 GQA prefill;
- INT8 projection;
- previously existing scalar/speculative paths.

These changes eliminated the chain of first-use allocations that previously
caused late startup OOM failures at exact 128K.

### Combined runtime backing allocation

Persistent and workspace arenas were changed from independent CUDA allocations
to slices of one runtime backing allocation.

This solved the practical CUDA allocation-granularity failure where separate
allocations failed despite nominally sufficient total free VRAM.

### prefill_hidden workspace alias

The full `[hidden, prefill_chunk]` normalized target hidden buffer was moved out
of permanent storage into the existing WorkspaceArena.

Only one final hidden column remains persistent.

Measured result:

- old full persistent buffer: 8.750000 MiB
- persistent tail: approximately 0.009766 MiB
- recovered VRAM: approximately 8.7403 MiB
- workspace peak increase: 0 MiB
- workspace remains approximately 150.999 MiB

Batch 101 validated 12/12 exact-128K cold starts after this change.

## Quality validation

Fixed-state composition result for the production artifact:

- JS divergence: approximately 0.00340549
- top-1 differences: 0
- classification: low perturbation

Stronger 20-test quality suite, Batch 095:

- production candidate: 14/20
- original Q5 artifact: 14/20
- candidate/base disagreements: 0

Category results were identical between candidate and Q5 baseline.

This supports retaining the 24VZ + 7GV artifact as the production-quality
candidate.

## Exact-128K startup validation

Before the prefill_hidden alias, exact 131072 operated with only about 2.65 MiB
nominal runtime-allocation slack and exhibited occasional cudaMalloc failure.

After the alias optimization, Batch 101 measured:

- runtime reservation: approximately 2613.1711 MiB
- runtime-reservation saving: approximately 8.7403 MiB
- free before runtime allocation: 2624.50-2624.5625 MiB
- planned slack: 11.33-11.39 MiB
- free after startup: 10.50-10.56 MiB
- exact-128K cold starts: 12/12 successful

## Fixed long benchmark

Beginning with Batch 104, use this deterministic long benchmark corpus for
matched performance comparisons:

- prompt tokens: 118001
- original prompt SHA-256:
  `078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd`

Historical session paths:

- `/tmp/103-NINFER-20260913/long-prompt.txt`
- `/tmp/104-NINFER-20260913/messages.json`

Future benchmark tooling should recreate and store this corpus permanently
inside the repository rather than depend on `/tmp`.

Batch 104, production candidate:

- prefill: 1244.99 tok/s
- decode: 117.02 tok/s
- planned slack: 11.39 MiB
- MTP acceptance: 89.86%

Decode throughput is prompt/MTP-acceptance dependent and must not be compared
across different prompt corpora as a pure kernel-speed metric.

## Matched prefill A/B

Batch 105 used the identical 118001-token corpus and A-B-B-A ordering.

Production 24VZ + 7GV:

- runs: 1240.59, 1236.02 tok/s
- mean: 1238.305 tok/s

48VZ / no-Q4-gate_value artifact:

- runs: 1387.15, 1383.89 tok/s
- mean: 1385.520 tok/s

Difference:

- +147.215 tok/s for 48VZ artifact
- +11.888%

This strongly reproduces the production prefill regression.

However, Batch 105 is not yet a causal gate_value isolation because the 48VZ
artifact also contains 24 additional Q4 value_z tensors.

Next required experiment:

- construct an exact 24VZ-only artifact;
- compare 24VZ-only versus 24VZ+7GV;
- use identical current binary and fixed 118001-token corpus.

If 24VZ-only returns near the 48VZ prefill rate, the seven Q4 gate_value layers
and their T >= 17 execution path are isolated as the regression source.

## Artifact provenance gap

The historical ad-hoc artifact rewriter used to create selective Q4 variants
was not preserved in Git and could not be recovered from Bash history.

Do not reproduce future variants with undocumented one-off `/tmp` scripts.

The next artifact tool should:

- live in the repository;
- take source artifact plus explicit tensor/layer selection;
- write a manifest beside the output;
- record source SHA-256;
- record output SHA-256;
- record tensor names and source/destination formats;
- refuse ambiguous tensor matches;
- support dry-run inspection;
- be covered by tests where practical.

## Rule from this checkpoint forward

Every production-relevant experiment should have a traceable tuple:

`Git commit -> artifact manifest/SHA -> benchmark batch -> conclusion`

Do not allow verified production changes to accumulate uncommitted again.
