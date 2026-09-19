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
  - multimodal
  - conversational
  - cuda
  - rtx-5080
  - vision
---

# Qwen3.8-27B for NInfer — RTX 5080 profile (cloud-built)

This repository hosts
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the native
[NInfer](https://github.com/toddballinger/ninfer-5080) `.ninfer` artifact format,
**built automatically on a CPU-only GitHub Actions runner** (no local GPU required).

The artifact is byte-identical (SHA-256 match) to the validated RTX 5080
benchmark profile — true 131,072-token context/KV, Q4 KV, MTP-3 speculative
decoding, and Vision on a single 16 GB GPU.

The `.ninfer` file is intended **only for NInfer**. It is not a Transformers
checkpoint, a Safetensors distribution, or a GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| Size | 16,461,267,456 bytes (15.33 GiB) |
| SHA-256 | `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `groupwise-int-5080` |
| NInfer target key | `qwen3_8_27b` |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21' \
  'qwen3_8_27b.ninfer' | sha256sum --check
```

## Quantization profile

The text core is mixed Q3/Q4/Q5 groupwise, approximately **3.95 effective BPW**:

| Format | Share |
|---|---:|
| Q3G64_F16S | 42.42% |
| Q4G64_F16S | 45.92% |
| Q5G64_F16S | 11.57% |
| BF16 / FP32 | ~0.10% |

## Requirements

- [NInfer](https://github.com/toddballinger/ninfer-5080) (RTX 5080 fork),
  built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5080 16 GB (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/toddballinger/ninfer-5080#build) for
source-build dependencies.

## Recommended serving command

```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id qwen3.8-27b \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 896 \
  --kv-dtype q4 \
  --spec mtp \
  --draft-tokens 3 \
  --no-cuda-graph \
  --max-concurrency 1 \
  --vision \
  --vision-max-tokens 1792
```

For image / video input, structured chat history, and HTTP serving, see the
[NInfer documentation](https://github.com/toddballinger/ninfer-5080/tree/main/docs).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- a genuine 131,072-token context with 131,072-token Q4 KV capacity;
- MTP speculative decoding with a 3-token draft window;
- BF16 and INT8 group-64 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- the NInfer CLI;
- OpenAI Chat Completions and Anthropic Messages serving.

## Validated performance (5080 profile)

Measured on a clean RTX 5080 with CUDA 13.x, true 128K Vision profile, MTP-3:

| Metric | Value |
|---|---:|
| Long-context prefill (118,001 tokens) | ~1377 tok/s |
| Decode | ~71.5 tok/s |
| MTP acceptance | 44.74% |
| Vision decode (synthetic image, 512×256) | 118.3 tok/s |
| Vision prefill | 685.8 tok/s |
| Video decode (red→green→blue MP4) | 97.1 tok/s |

Because this artifact is SHA-identical to the reference profile, these numbers
apply to it directly. Full methodology and records:
[`docs/VALIDATED_MANIFEST.md`](https://github.com/toddballinger/ninfer-5080/blob/main/docs/VALIDATED_MANIFEST.md)
and
[`docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`](https://github.com/toddballinger/ninfer-5080/blob/main/docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md).

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| DFlash2 repository | `z-lab/Qwen3.8-27B-DFlash2` |
| DFlash2 revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Conversion recipe | `qwen3_8_27b-v2` (groupwise-int) |
| Build | GitHub Actions, CPU-only `ubuntu-latest`, low-memory chunked converter |
| Converter repository | `https://github.com/starskyzheng/ninfer-5080` |

The conversion is fully reproducible from the
[`build-qwen38-27b.yml`](https://github.com/starskyzheng/ninfer-5080/blob/main/.github/workflows/build-qwen38-27b.yml)
workflow, which runs entirely on CPU (`--device cpu`) with a row-chunked
low-memory converter so it fits inside a 15 GB standard GitHub runner. The
chunked path is verified SHA-identical to the canonical full conversion.

## Limits

- The artifact is accepted only by NInfer (RTX 5080 fork) and the matching
  registered target.
- NInfer executes on one RTX 5080 and one CUDA device.
- It does not provide multi-GPU execution, CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also
licensed under Apache-2.0. Users remain responsible for complying with the
license and applicable laws.
