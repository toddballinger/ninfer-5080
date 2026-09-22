# NInfer documentation

Start with the [project README](../README.md) for the validated RTX 5080 profile, official artifact, serving settings and release status.

## RTX 5080 true-128K project guides

| Document | Purpose |
|---|---|
| [Qwen3.8-27B RTX 5080 v1.3 release](RELEASE_QWEN3.8_27B_RTX5080_V1.3.md) | validated v1.3 production runtime, thinking-budget and rolling-tool checkpoint behavior |
| [Validated manifest](VALIDATED_MANIFEST.md) | immutable model/binary identities and commit-scoped qualification records |
| [True 128K + Vision on RTX 5080](VISION_128K.md) | 1792/2048 Vision profiles, memory envelope and multimodal validation |
| [Reproducibility](REPRODUCIBILITY.md) | exact model revisions, build path, hashes and runtime settings |
| [Benchmarks](BENCHMARKS.md) | long-context, multimodal and runtime comparison results |
| [Memory profile](MEMORY_PROFILE.md) | mixed quantization, workspace recovery and HostMapped Vision fit |
| [Upstream sync status](UPSTREAM_SYNC_STATUS.md) | selective semantic-port ledger, production-relevance policy and pending upstream work |
| [Technical deep dive](TECHNICAL_DEEP_DIVE.md) | architecture and optimization details |
| [Failures and lessons](FAILURES_AND_LESSONS.md) | dead ends, regressions and recovery work |
| [History](HISTORY.md) | chronological engineering journey |
| [Publishing and sharing](PUBLISHING_AND_SHARING.md) | official distribution locations and public-release guidance |

## Official RTX 5080 artifact

The project-maintained public artifact is:

**[ninfer-5080/Qwen3.8-27B-RTX5080](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080)**

```text
file:   qwen3_8_27b.ninfer
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

This is the canonical validated artifact for the RTX 5080 true-128K profile. Runtime releases may advance independently while continuing to use the same model artifact.

## General user guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tools and multimodal input |
| [Performance](performance.md) | general performance results and reproduction commands |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode and long-context inputs |

The executable `--help` output remains the exact source for command-line option spelling and defaults.

## Other upstream/community model artifacts

These are separate from the project-maintained RTX 5080 artifact above:

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model, artifact and maintenance contracts. These files are not additional user workflows or installed API documentation.

Runtime and Op references:

- [Small-scale concurrent inference architecture](maintainer/concurrent-inference-architecture.md)
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)
- [Embedding CPU offload](maintainer/embed-cpu-offload.md)

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.6-27B model semantics](maintainer/qwen3.6-27b-model.md)
- [Qwen3.6-27B artifact contracts, including NVFP4](maintainer/qwen3.6-27b-artifact.md)
- [Qwen3.8-27B DFlash2 mathematics and Engine state contract](maintainer/qwen3.8-27b-dflash2.md)
- [Qwen3.8-27B artifact contracts, including the NVFP4 target](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)

Pending implementation work:

- [Softmax Attention organization and migration](maintainer/softmax-attention.md) describes the single target state for an unfinished source/public-contract cutover; it is not the current implementation map.
