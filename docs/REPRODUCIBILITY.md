# Reproducibility Guide

This guide describes the validated RTX 5080 true-128K path with Vision enabled.

## Validated hardware/software

- NVIDIA GeForce RTX 5080 16 GB
- Linux
- GCC 15.2.0
- CUDA 13.4.92
- NVIDIA driver 615.71.09
- validated v1.5 production source: `e4353f061bf0e378c83472cb2bcaf99e65681f4e`

The original text-only release remains frozen at `473dade56031852a7d96edef049d859da96a6df9` / tag `qwen3.8-27b-rtx5080-128k-v1`.

## Pin model revisions

```text
Qwen/Qwen3.8-27B
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

## Convert the canonical artifact locally

The Qwen3.8 converter requires both the base checkpoint and the pinned DFlash2
checkpoint because the complete `.ninfer` artifact includes registered DFlash2
tensor objects as well as the base model. The argument is therefore required even
when the runtime will normally use MTP rather than DFlash2.

A CPU conversion is:

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b.ninfer \
  --device cpu
```

The converter performs the registered source/resource preflight and writes the
provenance record alongside the artifact as
`out/qwen3_8_27b.ninfer.conversion.json`.

For the canonical project artifact, verify the exact byte count and SHA-256 shown
below before treating the result as equivalent to the published reference artifact.
An alternative fine-tune may be structurally convertible and even produce the same
container byte count without being the canonical artifact; artifact identity is
hash-based, not size-based.

## Build

Use a Release build and record compiler/CUDA/driver details if benchmark parity matters:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

ninja -C build -j16 ninfer ninfer-serve
```

## Model artifact

Validated artifact:

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The text core is a mixed Q3/Q4/Q5 groupwise profile at approximately 3.953 effective BPW. The full `.ninfer` container size is not itself a GGUF-comparable BPW numerator.

## Recommended v1.5 true-128K Vision server

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

Validated startup envelope:

```text
free after startup  885.94 MiB
planned slack       806.92 MiB
vision workspace    132.3142 MiB
```

## Canonical v1.5 whole-model benchmark

Release performance is measured with `ninfer_bench`:

```text
fixture=bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids
corpus_sha256=5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7
test=pp118001+tg2048
warmup=1
measured_repetitions=2
prefill=1374.383 tok/s
sustained_decode=112.215 tok/s
```

## 118K regression qualification helper

For the RTX 5080 16 GB regression-acceptance path, run:

```bash
./tools/qualify_qwen38_118k.sh /path/to/qwen3_8_27b.ninfer
```

The helper now enables `--embedding-host` by default, matching the current 16 GB
RTX 5080 memory profile. Additional CLI arguments may still be appended after the
artifact path.

This helper is a long-context regression/stability qualification using the historical
118,001-token prompt, `prefill-chunk=896`, a short 32-token generation, and CUDA
Graph disabled. It is **not** the current v1.5 whole-model performance benchmark.
Release performance should continue to be compared with the committed
`ninfer_bench pp118001+tg2048` contract documented above and in
[BENCHMARKS.md](BENCHMARKS.md).

## Automated CPU-only artifact conversion

The repository includes a GitHub Actions workflow that performs the
Qwen3.8-27B conversion entirely on CPU. **No local GPU is required.**

The workflow is manually triggered and runs on an explicitly provisioned
runner with sufficient CPU, RAM and disk capacity. Standard `ubuntu-latest`
is not treated as sufficient for the complete 27B conversion.

The integrated low-memory path uses:

- pinned source revisions and conversion-critical dependencies;
- true row-sliced Safetensors reads rather than repeated full-tensor loads;
- streamed artifact payload assembly;
- exact canonical groupwise artifact byte-size and SHA-256 gates;
- `.conversion.json` provenance publication.

Canonical groupwise artifact:

```text
bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The NVFP4 workflow profile is a separate artifact profile and does not inherit
this groupwise artifact identity.

## Clean-GPU prerequisite

A representative successful launch started from:

```text
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB
```

Check before launch:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

The current v1.5 production profile is qualified at Vision-2048 with host-mapped embeddings. A clean GPU remains important because unrelated allocations directly reduce the available margin.

A community WSL2/RTX 5080 qualification in [Issue #39](https://github.com/toddballinger/ninfer-5080/issues/39) reproduced the 118,001-token regression pass with `--embedding-host` and measured only about 2-2.5 MiB additional memory moving from `--vision-max-tokens 1792` to `2048` in that setup. Treat this as an external WSL2 datapoint rather than the canonical native-Linux release measurement.

## True 128K definition

For this project, true 128K means both values are exactly 131072:

```text
--max-context 131072
--kv-capacity 131072
```

Do not describe a reduced KV allocation as the same result.

## Long-context regression acceptance

The Vision source was re-tested with the exact historical corpus:

```text
prompt SHA256: 078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
prompt tokens: 118001
max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4
speculation:   MTP-3
max-new:       32
```

Observed result:

```text
return code:           0
prefill:               1375.16 tok/s
decode:                  71.52 tok/s
MTP acceptance:          44.74%
MTP acceptance length:    2.31 tok/round
workspace peak:          116.00 MiB
free after startup:       44.56 MiB
planned slack:            46.39 MiB
```

Historical text release on the same acceptance workload:

```text
prefill:               1377.81 tok/s
decode:                  71.51 tok/s
MTP acceptance:          44.74%
MTP acceptance length:    2.31 tok/round
```

No meaningful text-path regression was observed.

## Image acceptance at 1792

A deterministic 512×256 PNG was generated with a red left half and blue right half. The server correctly returned that the left half was red and the right half blue.

Repeated hardware validation reported approximately:

```text
prompt:          211 tokens
prefill:         682.8-685.8 tok/s
decode:          118.1-118.3 tok/s
ttft:            719-725 ms
MTP:             3.10 tok/round (70.0%)
```

This validates the image acquisition, preprocessing, HostMapped Vision encode and generation path while retaining full `131072 / 131072` text context/KV.

## Video acceptance at 1792

Install `ffmpeg` on the validation host and generate a deterministic six-second MP4 containing three two-second solid-color scenes in chronological order: red, green, blue.

Send the video through the OpenAI-compatible chat endpoint with thinking disabled and constrain the response to the three color names. The validated response was exactly:

```text
red, green, blue
```

Observed request metrics:

```text
finish reason:   stop_token
prompt:          572 tokens
generated:       6 tokens
prefill:         1721.9 tok/s
decode:          97.1 tok/s
ttft:            1111 ms
wall:            1.16 s
MTP:             4.00 tok/round (100.0%)
```

This is an end-to-end functional validation of video acquisition, preprocessing, Vision encode and generation on the final true-128K HostMapped configuration. It is not a broad video-quality benchmark.

## Multi-image history acceptance

Validated on the final HostMapped Vision path:

- ordinary image input through the OpenAI-compatible server;
- photos and small-text/receipt input;
- OpenWebUI multi-image history;
- cached historical media plus a newly uploaded image;
- deterministic video input;
- full 131072 text context/KV retained.

Observed cache patterns included `media_cache=1/1/0` and `media_cache=2/1/0`, proving old cached images were no longer charged repeatedly against the fresh preprocessing cap.

## Validation hashes

```text
ninfer SHA256:
5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140

ninfer-serve SHA256:
61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```

Rebuilt binaries can differ byte-for-byte if compiler/toolkit inputs change, so always record the full build environment alongside hashes.

---

## Qwen3.8-27B RTX 5080 v1.2 final validation

Validated code head: `dd2cb0341c321f8a808a6ed75f0a53225983f718`

Final exact 118,001-token acceptance: **1380.61 tok/s prefill**, **71.57 tok/s decode**, **44.74% MTP acceptance**, **2.31 tok/round**, with full **131,072 context / 131,072 Q4 KV**.

Vision 2048 acceptance also passed deterministic image, video, cached-history (`1/1/0` and `2/1/0`) and strict no-OOM validation. Startup remained intentionally tight at **8.56 MiB free / 10.08 MiB planned slack**.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md`.

---

## Qwen3.8-27B RTX 5080 v1.3 final validation

Validated runtime code head: `a7c6bd78d55da1ab23b6d91fdcd1731b6dc69e4f`

v1.3 preserves the validated **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** RTX 5080 profile while adding corrected Q4 strided-output handling and a server default thinking budget.

Combined v1.3 live validation passed with a three-request MTP sanity average of **84.7 tok/s decode**, **40.47% MTP acceptance** and **2.213 tok/round**. Client `reasoning_budget=64` and server-default `reasoning_budget=2048` resolution both passed.

The exact v1.2 118,001-token long-context and deterministic Vision/OOM validation remains preserved in the v1.2 release record.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`.
