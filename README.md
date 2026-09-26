# NInfer RTX 5080 — Qwen3.8-27B at true 128K + Vision on 16 GB

This fork documents and maintains a validated **Qwen3.8-27B** configuration for a single **NVIDIA RTX 5080 16 GB** with a genuine **131,072-token context and KV capacity**, Q4 KV, MTP-3 speculative decoding, and Vision support.

The project separates three things deliberately:

- **model artifact** — the converted `.ninfer` file, identified by an immutable SHA-256;
- **runtime release** — the exact NInfer source/binary that was validated against that artifact;
- **ongoing optimization work** — later source changes that may improve the runtime without changing the model artifact.

That distinction is important: a newer runtime does not imply that a different model file is required.

## Official model artifact

The canonical public artifact is now hosted by the project on Hugging Face:

**[ninfer-5080/Qwen3.8-27B-RTX5080](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080)**

| Field | Value |
|---|---|
| File | `qwen3_8_27b.ninfer` |
| Size | `16,461,267,456` bytes |
| SHA-256 | `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` |
| Target | Qwen3.8-27B / RTX 5080 16 GB |
| Effective main-model quantization | ~3.953 BPW |
| Context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Speculation | MTP-3 |
| Vision | validated |

Verify after download:

```bash
sha256sum qwen3_8_27b.ninfer
```

Expected:

```text
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The same artifact SHA has been retained across the original text-only release, Vision enablement, the v1.2/v1.3 production runtime work, the v1.4 production release, and subsequent qualified runtime optimizations.

## Validated runtime releases

### v1.4 production runtime

The current production release is **v1.4**, published from:

```text
source commit:
d5ee1bf130a45ce56f645dd44a6f1fa6f30c6a77

source tree:
35ef1538def9d4bd9dc0294c9364897cf1f4bce5

ninfer SHA-256:
38affd44afede11682500cba846d8a8b5c93cfe70259c73a72c1d5e3cef163bf

ninfer-serve SHA-256:
b936e179a06ad6b78b4fa4b3ae683efea928abf1888a6e2c3813fdeea9294a44
```

v1.4 preserves the full **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** profile and adds host-mapped token embeddings, enough recovered VRAM to run CUDA Graph decode as the recommended profile, constrained semantic decisions, a realistic mixed-workflow 118K benchmark, and CLI/serve Vision-planning parity.

The validated RTX 5080 production profile keeps **795.70 MiB** of token embeddings host-resident and starts with **715.54 MiB planned slack** with CUDA Graph enabled.

Canonical v1.4 benchmark on the 118,001-token mixed workflow:

| Metric | Result |
|---|---:|
| Prefill median | **1,361.76 tok/s** |
| Sustained decode median | **96.97 tok/s** |
| Sustained decode mean | **97.01 tok/s** |
| MTP acceptance | **66.98%** |
| MTP accepted length | **3.01 tok/round** |

The decode result is based on 3 × 2,048 exact decoded tokens with model-default EOS suppressed only for benchmark measurement, matching the repository benchmark policy.

See [the full v1.4 release record](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.4.md).

Historical release records remain preserved in [v1.3](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md) and [v1.2](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md).

## Recommended serving profile

The v1.4 RTX 5080 production profile uses full 128K context/KV, Vision 2048, host-mapped embeddings, MTP-3 and CUDA Graph decode:

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
  --embedding-host \
  --max-concurrency 1 \
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool \
  --vision \
  --vision-max-tokens 2048
```

CUDA Graph is enabled by default; the recommended v1.4 profile intentionally omits `--no-cuda-graph`.

Validated startup envelope:

```text
embedding host-resident  795.70 MiB
vision_encode             132.3142 MiB
free after startup        794.56 MiB
planned slack             715.54 MiB
graph observed/allowance    2.00 / 82.00 MiB
```

This replaces the pre-v1.4 recommendation to reduce Vision to 1792 for headroom. The 1792 and earlier 2048 measurements remain documented as historical validation points in [VISION_128K.md](docs/VISION_128K.md).

## Long-context validation

The canonical forward-looking benchmark is now the deterministic **workflow-118k-v1** mixed engineering-agent fixture:

```text
bench/fixtures/workflow-118k-v1/qwen38_118001_workflow_candidate.txt
SHA256=cb7c131bd20d78bd69396c019f988c81fad1941a853a8de7da7586e1aeb99718
prepared tokens=118001
```

The fixture mixes prose, source code, shell transcripts, configuration, JSON, tool-call history, runtime logs and engineering discussion. The older repetitive 118K corpus is retained only as historical evidence.

v1.4 canonical Graph-ON result:

| Metric | Result |
|---|---:|
| Prompt tokens | 118,001 |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| CUDA Graph | enabled |
| Host-mapped embeddings | enabled |
| Vision profile | 2048 |
| Prefill median | **1,361.76 tok/s** |
| Decode median | **96.97 tok/s** |
| Decode range | **96.97–97.08 tok/s** |
| MTP acceptance | **66.98%** |
| Acceptance length | **3.01 tok/round** |
| Planned slack | **715.54 MiB** |

A result is described here as **true 128K** only when both configured context and allocated KV capacity are actually `131072`.

## Multimodal validation

The final HostMapped Vision path has been validated end-to-end for:

- deterministic image understanding;
- deterministic video understanding;
- multi-image OpenWebUI history;
- cached historical media accounting;
- full 131,072 text context/KV coexistence.

A synthetic 512×256 red/blue image was correctly identified as red on the left and blue on the right. A six-second red → green → blue MP4 was correctly returned in chronological order.

See [VISION_128K.md](docs/VISION_128K.md) and [VALIDATED_MANIFEST.md](docs/VALIDATED_MANIFEST.md) for exact measurements.

## Model sources and quantization

Pinned model sources:

```text
Qwen/Qwen3.8-27B
revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
revision: 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

Main text-core quantization:

| Format | Share |
|---|---:|
| Q3G64_F16S | 42.42% |
| Q4G64_F16S | 45.92% |
| Q5G64_F16S | 11.57% |
| BF16 / FP32 | ~0.10% |

The effective main-model quantization is approximately **3.953 BPW**.

## Reproducibility and CI

The official Hugging Face artifact above is currently the **validated reference artifact**.

A CPU-only GitHub Actions conversion/publishing workflow is being developed separately. Its acceptance criterion is intentionally strict: before an automated build can replace or republish the canonical artifact, it should reproduce the exact expected byte size and SHA-256.

That means build automation can evolve without weakening the artifact identity contract.

## Documentation

Start with:

- [v1.4 release](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.4.md) — current production release record\n- [v1.3 release](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md) — historical production release record
- [Validated manifest](docs/VALIDATED_MANIFEST.md) — exact source, binary and model identities
- [Vision 128K](docs/VISION_128K.md) — Vision profiles, memory envelope and validation
- [Reproducibility](docs/REPRODUCIBILITY.md) — build and runtime reproduction
- [Benchmarks](docs/BENCHMARKS.md) — long-context and multimodal results
- [Memory profile](docs/MEMORY_PROFILE.md) — how the 16 GB fit is achieved
- [Upstream sync status](docs/UPSTREAM_SYNC_STATUS.md) — selective semantic-port ledger and review policy
- [Constrained decisions](docs/CONSTRAINED_DECISIONS_USAGE.md) — current C++ finite-decision API, multi-token tries, dependencies and backend limits
- [Technical deep dive](docs/TECHNICAL_DEEP_DIVE.md) — architecture and optimization details
- [Publishing and sharing](docs/PUBLISHING_AND_SHARING.md) — public-release guidance

The full documentation index is in [docs/README.md](docs/README.md).

## Upstream synchronization

This fork selectively incorporates upstream NInfer changes rather than tracking `Neroued/ninfer:master` commit-for-commit.

Upstream changes are evaluated for **production-path relevance first**. A microbenchmark improvement on a kernel or shape that the validated Qwen3.8-27B RTX 5080 workload does not exercise is normally deferred rather than merged only to reduce a GitHub “behind” count.

See [UPSTREAM_SYNC_STATUS.md](docs/UPSTREAM_SYNC_STATUS.md) for the current commit-scoped ledger and acceptance policy.

## Attribution and licensing

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches or converted artifacts, and verify the applicable upstream/model licenses for the material being redistributed.
