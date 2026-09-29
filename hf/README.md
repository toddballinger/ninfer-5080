---
license: apache-2.0
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - cuda
  - rtx-5080
  - long-context
  - speculative-decoding
  - multimodal
  - vision
---

# Qwen3.8-27B for NInfer — RTX 5080 16 GB, true 128K + Vision

Project-maintained NInfer artifact for **Qwen3.8-27B** on a single
**NVIDIA RTX 5080 16 GB**.

> **Current production runtime: v1.5**

The model artifact itself remains byte-identical across the later runtime
releases; v1.5 is the current production-qualified NInfer runtime/profile for
this artifact.

## At a glance

| Field | Current validated value |
|---|---|
| Artifact | `qwen3_8_27b.ninfer` |
| Artifact size | `16,461,267,456 bytes` |
| SHA-256 | `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` |
| Runtime release | **v1.5** |
| Primary hardware | RTX 5080 16 GB |
| Context | 131,072 tokens |
| KV capacity | 131,072 tokens |
| KV dtype | Q4 group64 |
| Prefill chunk | 1792 |
| Speculation | MTP-3 |
| CUDA Graph | enabled |
| Host-mapped embeddings | enabled |
| Vision profile | 2048 tokens |
| Main text-model quantization | mixed Q3/Q4/Q5, ~3.953 BPW |
| Canonical v1.5 prefill | **1,374.383 tok/s** |
| Canonical v1.5 sustained decode | **112.215 tok/s** |

Canonical project source and validation records:

https://github.com/toddballinger/ninfer-5080

## Download

Using the Hugging Face CLI:

```bash
hf download ninfer-5080/Qwen3.8-27B-RTX5080 \
  qwen3_8_27b.ninfer \
  qwen3_8_27b.ninfer.conversion.json \
  --local-dir .
```

Verify the artifact:

```bash
sha256sum qwen3_8_27b.ninfer
```

Expected:

```text
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21  qwen3_8_27b.ninfer
```

A matching SHA-256 identifies the exact validated project artifact regardless
of filename or download machine.

## Artifact identity vs runtime version

The model artifact and NInfer runtime are versioned independently.

The canonical artifact identity is:

```text
SIZE=16461267456
SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The same artifact has remained valid through later runtime optimization work.
A newer NInfer runtime does **not** imply that a new `.ninfer` file is
required.

## Current production release — v1.5

| Item | Value |
|---|---|
| Release tag | `qwen3.8-27b-rtx5080-128k-vision-v1.5` |
| Release commit | `e4edd6d5c5f9f7996de0f3d9f6c311e883452580` |
| Validated source tree | `e4353f061bf0e378c83472cb2bcaf99e65681f4e` |
| `ninfer-serve` SHA-256 | `928e5615ef453786f47f79b6af2152d2f8f8d61307656f23c47fa45b5ed41167` |
| CUDA | 13.4.92 |
| NVIDIA driver | 615.71.09 |

v1.5 standardizes release performance reporting on `ninfer_bench` and is the
current RTX 5080 production-qualified profile.

### Canonical v1.5 benchmark

`pp118001+tg2048` on a single RTX 5080 16 GB:

| Metric | Result |
|---|---:|
| Prefill | **1,374.383 tok/s** |
| Sustained decode | **112.215 tok/s** |
| Context / KV capacity | **131,072 / 131,072** |
| KV dtype | **Q4 group64** |
| Prefill chunk | **1792** |
| Speculation | **MTP-3** |
| CUDA Graph | **enabled** |
| Host-mapped embeddings | **enabled** |
| Vision profile | **2048 tokens** |

Benchmark contract:

```text
fixture=bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids
corpus_sha256=5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7
prompt_tokens=118001
test=pp118001+tg2048
warmup=1
measured_repetitions=2
prefill=1374.383 tok/s
sustained_decode=112.215 tok/s
```

Historical v1.3/v1.4 measurements remain in the repository validation records.

## Recommended v1.5 serving profile

```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id local-model \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 1792 \
  --kv-dtype q4 \
  --spec mtp \
  --draft-tokens 3 \
  --max-concurrency 1 \
  --max-pending-requests 16 \
  --pending-timeout-ms 180000 \
  --embedding-host \
  --vision \
  --vision-max-tokens 2048 \
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool
```

CUDA Graph is enabled by default.

### Validated startup envelope

```text
free after startup  885.94 MiB
planned slack       806.92 MiB
vision workspace    132.3142 MiB
```

## True 128K qualification

The production profile uses both a 131,072-token context and a 131,072-token
KV capacity. Qualification uses an actual **118,001-token prompt**, not merely
a configured maximum context value.

This is the workload used for the canonical v1.5 benchmark above.

## Multimodal validation

The HostMapped Vision path has been validated for:

- deterministic image understanding
- deterministic video understanding
- multi-image conversation history
- cached historical-media accounting
- coexistence with the full 131,072-token text context/KV allocation

Details:

https://github.com/toddballinger/ninfer-5080/blob/main/docs/VISION_128K.md

## Quantization profile

Main text-core distribution:

| Format | Share |
|---|---:|
| Q3G64_F16S | 42.42% |
| Q4G64_F16S | 45.92% |
| Q5G64_F16S | 11.57% |
| BF16 / FP32 | ~0.10% |

Effective main-model quantization: **~3.953 BPW**.

## Model sources

**Qwen3.8-27B**

- repo: `Qwen/Qwen3.8-27B`
- revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

**DFlash2**

- repo: `z-lab/Qwen3.8-27B-DFlash2`
- revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`

Both upstream Hugging Face repositories currently declare Apache-2.0
licensing.

## Reproducibility

The canonical groupwise artifact is built by a **CPU-only GitHub Actions
workflow on an explicitly provisioned runner — no local GPU is required**.

The integrated workflow:

- uses true row-sliced Safetensors reads;
- streams artifact payload assembly;
- pins conversion-critical source revisions and dependencies;
- performs CPU/RAM/disk runner-capacity checks before model downloads;
- structurally inspects the generated artifact;
- requires the exact canonical byte count and SHA-256 before publication;
- publishes `qwen3_8_27b.ninfer.conversion.json` alongside the artifact.

The production workflow has reproduced the exact canonical identity:

```text
artifact_bytes=16461267456
artifact_sha256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
canonical_groupwise_identity=PASS
```

Standard `ubuntu-latest` is not treated as sufficient for the complete 27B
conversion. The NVFP4 workflow profile is separate and does not inherit the
groupwise artifact's expected size or SHA-256.

## Documentation

- [Project overview](https://github.com/toddballinger/ninfer-5080)
- [v1.5 release record](https://github.com/toddballinger/ninfer-5080/blob/main/docs/RELEASE_QWEN3.8_27B_RTX5080_V1.5.md)
- [Validated manifest](https://github.com/toddballinger/ninfer-5080/blob/main/docs/VALIDATED_MANIFEST.md)
- [Benchmarks](https://github.com/toddballinger/ninfer-5080/blob/main/docs/BENCHMARKS.md)
- [Vision / true 128K](https://github.com/toddballinger/ninfer-5080/blob/main/docs/VISION_128K.md)
- [Memory profile](https://github.com/toddballinger/ninfer-5080/blob/main/docs/MEMORY_PROFILE.md)
- [Reproducibility](https://github.com/toddballinger/ninfer-5080/blob/main/docs/REPRODUCIBILITY.md)

## Credits

This artifact and its RTX 5080 production integration are maintained by
**Todd Ballinger / ninfer-5080**.

**starskyzheng** made a significant contribution through PR #1, including the
original low-memory Qwen3.8-27B conversion path and initial GitHub Actions
automation/reproducibility work.

The integrated implementation was subsequently extended and hardened in
`ninfer-5080`, including true row-sliced Safetensors reads, streamed payload
assembly, regression coverage, pinned conversion dependencies, exact artifact
size/SHA validation, provenance-report publishing, runner-capacity safeguards,
production-runtime integration and final validation.

This project also builds on:

- NInfer upstream — Neroued and contributors
- Qwen3.8-27B — Qwen team
- DFlash2 — z-lab and project contributors

Please preserve applicable upstream copyright, attribution and license notices.
