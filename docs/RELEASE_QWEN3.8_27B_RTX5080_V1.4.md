# Qwen3.8-27B RTX 5080 128K + Vision v1.4

Published release: https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.4

```text
tag: qwen3.8-27b-rtx5080-128k-vision-v1.4
target commit: d5ee1bf130a45ce56f645dd44a6f1fa6f30c6a77
source tree: 35ef1538def9d4bd9dc0294c9364897cf1f4bce5
published: 2026-09-26T17:40:44Z
```

## Release summary

v1.4 is the production-qualified RTX 5080 release that converts the previous near-capacity
128K + Vision configuration into a substantially healthier memory envelope while retaining the
same canonical Qwen3.8-27B artifact.

Major changes since v1.3:

- host-mapped token embeddings via `--embedding-host`;
- CUDA Graph decode enabled in the recommended production profile;
- Q4/Q5 input-projection routing improvements;
- Q5 LinearAdd Split2 and narrow-tail routing improvements;
- cumulative `rolling-tool` prefix checkpoint support;
- constrained semantic decision execution through V2-D1;
- CLI `--prompt-file` support;
- deterministic realistic `workflow-118k-v1` benchmark fixture;
- CLI/serve Vision workspace-planning parity;
- expanded release provenance and qualification records.

## Artifact identities

```text
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
NINFER_SHA256=38affd44afede11682500cba846d8a8b5c93cfe70259c73a72c1d5e3cef163bf
NINFER_SERVE_SHA256=b936e179a06ad6b78b4fa4b3ae683efea928abf1888a6e2c3813fdeea9294a44
```

Canonical model path used for qualification:

```text
/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer
```

## Validated production profile

| Item | v1.4 |
|---|---|
| Model | Qwen3.8-27B |
| GPU | NVIDIA GeForce RTX 5080 16 GB |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| CUDA Graph | enabled |
| Host-mapped embeddings | enabled |
| Max concurrency | 1 |
| Vision | enabled |
| Vision token profile | 2048 |
| Default thinking budget | 2048 |
| Prefix checkpoint policy | `rolling-tool` |
| Max pending requests | 16 |
| Pending timeout | 180,000 ms |

Representative serving arguments:

```text
--host 0.0.0.0
--port 8080
--model-id local-model
--max-context 131072
--kv-capacity 131072
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--max-concurrency 1
--max-pending-requests 16
--pending-timeout-ms 180000
--embedding-host
--vision
--vision-max-tokens 2048
--default-thinking-budget 2048
--prefix-checkpoint-policy rolling-tool
```

CUDA Graph is enabled by default; the recommended profile intentionally omits
`--no-cuda-graph`.

## Canonical long-context benchmark

The forward-looking benchmark is:

```text
fixture: bench/fixtures/workflow-118k-v1/qwen38_118001_workflow_candidate.txt
SHA256: cb7c131bd20d78bd69396c019f988c81fad1941a853a8de7da7586e1aeb99718
prepared prompt tokens: 118001
```

The fixture contains mixed prose, source code, shell transcripts, JSON/configuration, tool-call
history, runtime logs, benchmark results and engineering discussion.

### CUDA Graph ON

Three independent exact long-decode measurements:

| Metric | Result |
|---|---:|
| Prefill median | **1,361.76 tok/s** |
| Prefill range | 1,351.79–1,362.69 tok/s |
| Decode median | **96.97 tok/s** |
| Decode mean | **97.01 tok/s** |
| Decode range | 96.97–97.08 tok/s |
| Decode standard deviation | 0.06 tok/s |
| MTP acceptance | **66.98%** |
| MTP accepted length | **3.01 tok/round** |
| Workspace peak | 116.00 MiB |
| Planned slack | **715.54 MiB** |

Measured Graph-ON decode sample:

```text
3 x 2048 = 6144 decoded tokens
```

Headline result:

> **~1,362 tok/s prefill and ~97 tok/s sustained decode at 118K prompt context on a single RTX 5080 16 GB.**

### CUDA Graph A/B

| Metric | Graph ON | Graph OFF | Difference |
|---|---:|---:|---:|
| Prefill median | 1,361.76 | 1,364.12 tok/s | -0.17% |
| Decode median | **96.97** | 95.09 tok/s | **+1.98%** |
| MTP acceptance | 66.98% | 66.98% | unchanged |
| MTP accepted length | 3.01 | 3.01 tok/round | unchanged |
| Workspace peak | 116.00 | 116.00 MiB | unchanged |
| Planned slack | 715.54 | 804.08 MiB | -88.54 MiB |

The complete A/B measured **12,288 decoded tokens**.

## Benchmark methodology

The normal model naturally reaches a model-default stop after roughly 45 generated tokens on this
deterministic fixture. Sustained throughput measurement therefore uses the same benchmark policy as
`ninfer_bench`:

```cpp
request.stop.include_model_defaults = false;
```

Only model-default stop termination is suppressed. The canonical prompt, model artifact, sampling,
runtime kernels and production inference paths are otherwise unchanged. Each measured run verifies
exactly 2,048 decoded tokens.

Benchmark-only CLI identity:

```text
SHA256=49ac07fc308ef83937771bb3cc11f685b17fa951539bb8a404e3c3204bd36bee
```

This benchmark-only CLI is not the production server binary.

## Host-mapped embeddings and VRAM recovery

PR #11 adds:

```text
--embedding-host
```

The token embedding table remains GPU-accessible through pinned mapped host memory/UVA instead of
occupying persistent device memory.

Measured v1.4 effect:

| Metric | v1.3 | v1.4 |
|---|---:|---:|
| Process VRAM | 15,824 MiB | **15,028 MiB** |
| Persistent VRAM recovered | — | **~796 MiB** |
| Host-resident embeddings | 0 | **795.70 MiB** |
| Free after startup, Graph OFF | ~8.56 MiB | **802.56 MiB** |
| Planned slack, Graph OFF | ~10.08 MiB | **804.08 MiB** |
| Free after startup, Graph ON | — | **794.56 MiB** |
| Planned slack, Graph ON | — | **715.54 MiB** |

CUDA Graph production fit:

```text
API_HEALTH=PASS
FREE_AFTER_STARTUP=794.56 MiB
PLANNED_SLACK=715.54 MiB
GRAPH_CURRENT=2.00 MiB
GRAPH_ALLOWANCE=82.00 MiB
VISION_WORKSPACE=132.3142 MiB
```

## Changes since v1.3

- **PR #3** — Q4/Q5 input-projection column routing.
- **PR #4** — production qualification record for PR #3.
- **PR #5** — reconcile validated rolling-tool prefix checkpoints into main.
- **PR #6** — attach validation records to exact commits.
- **PR #7** — Q5 LinearAdd Split2 and narrow-tail routing; representative affected operator points improved by roughly 25–34%.
- **PR #8** — record PR #7 upstream integration merge.
- **PR #9** — production-first upstream relevance gate.
- **PR #10** — official Hugging Face release provenance.
- **PR #11** — host-mapped embeddings.
- **PR #12** — CLI `--prompt-file` and large-prompt qualification support.
- **PR #13** — constrained semantic decision execution.
- **PR #14** — realistic 118K workflow fixture.
- **PR #15** — CLI/serve Vision workspace-planning parity.

## Validation status

| Validation | Result |
|---|---|
| Source tree matches merged `main` | PASS |
| Canonical model hash | PASS |
| Production server hash | PASS |
| 131,072 context | PASS |
| 131,072 Q4 KV | PASS |
| MTP-3 | PASS |
| Host-mapped embeddings | PASS |
| Vision 2048 | PASS |
| CUDA Graph startup | PASS |
| API health | PASS |
| Canonical 118,001-token prefill | PASS |
| Exact 2,048-token sustained decode | PASS |
| 3-run Graph ON benchmark | PASS |
| 3-run Graph OFF control | PASS |
| Production service restoration | PASS |

**V1.4 PRODUCTION RELEASE VALIDATED.**
