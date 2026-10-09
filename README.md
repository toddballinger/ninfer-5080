
> **Issue #58 status (2026-10-09 14:20 UTC):** [authoritative continuation handover](docs/ISSUE58_CONTINUATION_HANDOVER.md). PR #63 foundation **merged** (squash `960ebe2a`); experimental pinned CUDA staging **syntax test PASS** on follow-on branch `2efaa3b` using ccache, but no GPU transfer/resume validation. Production C1 remains healthy and unchanged. Real C2 reversible yielding is **not implemented**; Issue #58 remains open.

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

## Validated runtime release

### v1.5 — CUDA 13.4 / Blackwell production runtime

v1.5 is the current RTX 5080 production release. It qualifies **CUDA 13.4.92 / NVIDIA 615.71.09**, retunes the Q3/A8 large-prefill SwiGLU path for Blackwell/CC 12.0, and standardizes whole-model performance reporting on `ninfer_bench`.

Canonical 118K result on the committed immutable benchmark corpus:

| Metric | v1.5 |
|---|---:|
| Prefill | **1,374.383 tok/s** |
| Sustained decode | **112.215 tok/s** |
| Improvement | **+4.46% prefill / +2.32% decode** |
| Context / KV capacity | **131,072 / 131,072** |
| KV dtype | **Q4 group64** |
| Prefill chunk | **1792** |
| Speculation | **MTP-3** |
| CUDA Graph | **enabled** |
| Vision | **2048 tokens** |

The model artifact is unchanged. Full benchmark methodology, configuration and qualification evidence are documented in [Benchmarks](docs/BENCHMARKS.md) and the [v1.5 release record](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.5.md).

Historical release records: [v1.4](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.4.md), [v1.3](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md), [v1.2](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md).

## Recommended serving profile

> **2026-10-09 concurrency decision:** C2 was 1.511x faster on the 75-job corpus, but suffered two >600-second TTFTs (maximum 664.72s); it is **not production-qualified** for OpenClaw. C3 queue timeout and C4 startup memory failure are documented in [128K concurrency closeout](docs/CONCURRENCY_128K.md). Retain max-concurrency 1 until [P0 scheduler issue #58](https://github.com/toddballinger/ninfer-5080/issues/58) passes bounded-latency qualification. PR #59 adds diagnostics, not the fix.



The v1.5 RTX 5080 production profile uses full 128K context/KV, Vision 2048, host-mapped embeddings, MTP-3 and CUDA Graph decode:

```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id qwen3.8-27b \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 1792 \
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

CUDA Graph is enabled by default; the recommended v1.5 profile intentionally omits `--no-cuda-graph`.

Validated v1.5 production startup envelope:

```text
vision_encode             132.3142 MiB
free after startup        885.94 MiB
planned slack             806.92 MiB
graph observed/allowance    2.00 / 82.00 MiB
```

Detailed memory and Vision qualification is documented in [VISION_128K.md](docs/VISION_128K.md) and [MEMORY_PROFILE.md](docs/MEMORY_PROFILE.md).

### Next production qualification: C2 concurrency at 128K-class context

The v1.5 command above remains the validated C1 release profile. The current highest-priority bounded performance gate is now to determine whether the existing Q4-group64 serving path can preserve a **131,072-token per-request ceiling** while using shared KV/admission capacity for useful **C2** execution on the RTX 5080.

This is aimed directly at OpenClaw-style parallel local-worker activity: two independent requests should be able to make progress together when their combined active KV/resource reservations fit, rather than one waiting behind the other. C4 is a secondary target for bursts of shorter requests.

Do not infer 2x throughput from C2. Qualification must compare aggregate throughput, per-request latency, TTFT/admission wait, MTP acceptance, CUDA Graph/workspace behavior and VRAM headroom. The first gate uses the current Q4 profile; compressed KV is a follow-on only if current-Q4 memory is the limiting factor.

See [RTX 5080 128K concurrency qualification](docs/CONCURRENCY_128K.md).

## Long-context benchmark

Whole-model performance is reported with `ninfer_bench` against the committed immutable 118,001-token ID corpus:

```text
bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids
SHA256=5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7
test=pp118001+tg2048
warmup=1
measured repetitions=2
```

Current v1.5 result:

- **1,374.383 tok/s prefill**
- **112.215 tok/s sustained decode**
- **+4.46% prefill / +2.32% decode** versus the previous production configuration under the same benchmark contract

See [BENCHMARKS.md](docs/BENCHMARKS.md) for the full methodology, run data, memory envelope and engineering evidence.

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

Exact local CPU conversion:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b.ninfer \
  --device cpu
```

`--dflash2-model` is required because the complete registered artifact includes
DFlash2 tensor objects even when the serving profile normally selects MTP. The
converter writes a matching `.conversion.json` provenance record. See
[Reproducibility](docs/REPRODUCIBILITY.md) for the full conversion and verification
contract.

Main text-core quantization:

| Format | Share |
|---|---:|
| Q3G64_F16S | 42.42% |
| Q4G64_F16S | 45.92% |
| Q5G64_F16S | 11.57% |
| BF16 / FP32 | ~0.10% |

The effective main-model quantization is approximately **3.953 BPW**.

## Reproducibility and CI

The official Hugging Face artifact above remains the **validated reference artifact**.

The repository now includes an **automated CPU-only GitHub Actions conversion and publishing workflow**. No local GPU is required to build the artifact.

The full conversion runs on an explicitly provisioned runner with sufficient CPU, RAM and disk capacity; standard `ubuntu-latest` is not treated as sufficient for this large conversion.

The integrated low-memory path uses row-sliced Safetensors reads and streamed artifact payload assembly. Conversion-critical dependencies and source revisions are pinned, and the canonical groupwise artifact must match both the exact expected byte count and SHA-256 before publication.

The associated `.conversion.json` provenance record is published alongside the artifact.

The NVFP4 workflow profile remains a separate artifact profile and does not share the canonical groupwise artifact's expected size or SHA-256.

## Documentation

Start with:

- [v1.5 release](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.5.md) — current production release
- [Benchmarks](docs/BENCHMARKS.md) — canonical benchmark methodology and detailed results
- [Validated manifest](docs/VALIDATED_MANIFEST.md) — exact source, binary and model identities
- [Vision 128K](docs/VISION_128K.md) — Vision profiles, memory envelope and validation
- [Reproducibility](docs/REPRODUCIBILITY.md) — build and runtime reproduction
- [Memory profile](docs/MEMORY_PROFILE.md) — how the 16 GB fit is achieved
- [128K concurrency qualification](docs/CONCURRENCY_128K.md) — current-Q4 C2/C4 operating-point gate for OpenClaw
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
