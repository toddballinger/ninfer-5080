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

Project-maintained NInfer model artifact for **Qwen3.8-27B** on a single
**NVIDIA RTX 5080 16 GB**, validated with:

- 131,072-token context
- 131,072-token KV capacity
- Q4 group64 KV
- MTP-3 speculative decoding
- Vision input
- mixed Q3/Q4/Q5 model quantization
- approximately 3.953 effective BPW for the main text model

Canonical source and validation records:

https://github.com/toddballinger/ninfer-5080

## Official artifact

| Field | Value |
|---|---|
| File | `qwen3_8_27b.ninfer` |
| Size | `16,461,267,456 bytes` |
| SHA-256 | `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` |
| Format | NInfer native `.ninfer` |
| Target | Qwen3.8-27B |
| Primary hardware profile | RTX 5080 16 GB |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Speculation | MTP-3 |
| Vision | validated |

This file is intended for **NInfer**. It is not a Transformers checkpoint,
Safetensors distribution, or GGUF file.

## Download

Using the Hugging Face CLI:

```bash
hf download ninfer-5080/Qwen3.8-27B-RTX5080 \
  qwen3_8_27b.ninfer \
  --local-dir .

Verify the artifact:

sha256sum qwen3_8_27b.ninfer

Expected:

c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21  qwen3_8_27b.ninfer

A matching SHA-256 identifies the exact validated project artifact regardless
of the filename or the machine from which it was downloaded.

Model artifact vs runtime version

The model artifact and the NInfer runtime are versioned independently.

The artifact currently published here has remained byte-identical across
multiple later runtime optimizations. A newer NInfer runtime therefore does
not imply that a new .ninfer model file is required.

The canonical artifact identity is:

bytes:
16461267456

SHA256:
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
Validated v1.5 production runtime

Validated source commit:

e4353f061bf0e378c83472cb2bcaf99e65681f4e

Validated ninfer-serve SHA-256:

928e5615ef453786f47f79b6af2152d2f8f8d61307656f23c47fa45b5ed41167

v1.5 is the current RTX 5080 production release. It qualifies CUDA 13.4.92 /
NVIDIA 615.71.09 and standardizes whole-model release performance reporting
on `ninfer_bench`.

Canonical `pp118001+tg2048` result:

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

The model artifact is unchanged.

Recommended serving profile

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

Validated v1.5 startup envelope:

```text
free after startup  885.94 MiB
planned slack       806.92 MiB
vision workspace    132.3142 MiB
```

True-128K validation

The production profile uses both a 131,072-token context and 131,072-token
KV capacity.

Canonical v1.5 whole-model benchmark:

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

Historical v1.3/v1.4 results remain in the repository validation records.

Multimodal validation

The final HostMapped Vision path has been validated for:

deterministic image understanding
deterministic video understanding
multi-image conversation history
cached historical-media accounting
coexistence with the full 131,072 text context/KV allocation

A synthetic red/blue image was correctly identified by side, and a deterministic
red → green → blue video was returned in the correct chronological order.

Details:

https://github.com/toddballinger/ninfer-5080/blob/main/docs/VISION_128K.md

Model sources

Target model:

Qwen/Qwen3.8-27B
revision:
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

DFlash2 source:

z-lab/Qwen3.8-27B-DFlash2
revision:
50307d4c4cde6860d4eee73e2547cd786fe8e8a4

Both upstream Hugging Face repositories currently declare Apache-2.0 licensing.

Quantization profile

Main text-core distribution:

Format	Share
Q3G64_F16S	42.42%
Q4G64_F16S	45.92%
Q5G64_F16S	11.57%
BF16 / FP32	~0.10%

Effective main-model quantization:

~3.953 BPW
Reproducibility

This artifact is **built automatically by a CPU-only GitHub Actions workflow
on an explicitly provisioned runner — no local GPU is required**.

The workflow uses the low-memory Qwen3.8-27B converter, pinned source revisions
and conversion-critical dependencies, and checks runner capacity before model
downloads begin.

The integrated low-memory path uses true row-sliced Safetensors reads and
streamed artifact payload assembly.

Before the canonical groupwise artifact can be published it must reproduce:

```text
SIZE=16461267456
SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The workflow also publishes the associated `.conversion.json` provenance
record.

Standard `ubuntu-latest` is not treated as sufficient for the complete 27B
conversion. The NVFP4 workflow profile is separate and does not inherit the
groupwise artifact's expected size or SHA-256.

Documentation
Project overview:
https://github.com/toddballinger/ninfer-5080
Validated manifest:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/VALIDATED_MANIFEST.md
v1.3 release:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md
Vision:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/VISION_128K.md
Reproducibility:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/REPRODUCIBILITY.md
Benchmarks:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/BENCHMARKS.md
Memory profile:
https://github.com/toddballinger/ninfer-5080/blob/main/docs/MEMORY_PROFILE.md
Credits

This artifact and its RTX 5080 production integration are maintained by
**Todd Ballinger / ninfer-5080**.

**starskyzheng** made a significant contribution through PR #1, including the
original low-memory Qwen3.8-27B conversion path and the initial GitHub Actions
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
