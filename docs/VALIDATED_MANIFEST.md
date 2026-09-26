# Validated Release Manifest

This file records immutable and subsequently validated milestones by exact commit, binary and artifact identity. Validation claims are attached to the source tree that was actually tested rather than to a floating branch label.


## Current production release — v1.4

```text
release tag: qwen3.8-27b-rtx5080-128k-vision-v1.4
release commit: d5ee1bf130a45ce56f645dd44a6f1fa6f30c6a77
source tree: 35ef1538def9d4bd9dc0294c9364897cf1f4bce5
model SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
ninfer SHA256: 38affd44afede11682500cba846d8a8b5c93cfe70259c73a72c1d5e3cef163bf
ninfer-serve SHA256: b936e179a06ad6b78b4fa4b3ae683efea928abf1888a6e2c3813fdeea9294a44
```

Validated production runtime:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4-group64
SPECULATION=mtp
DRAFT_TOKENS=3
EMBEDDING_HOST=on
CUDA_GRAPHS=on
VISION=on
VISION_MAX_TOKENS=2048
MAX_CONCURRENCY=1
DEFAULT_THINKING_BUDGET=2048
PREFIX_CHECKPOINT_POLICY=rolling-tool
```

Canonical v1.4 mixed-workflow qualification:

```text
FIXTURE=bench/fixtures/workflow-118k-v1/qwen38_118001_workflow_candidate.txt
FIXTURE_SHA256=cb7c131bd20d78bd69396c019f988c81fad1941a853a8de7da7586e1aeb99718
PROMPT_TOKENS=118001
GRAPH_ON_PREFILL_MEDIAN_TOK_S=1361.76
GRAPH_ON_DECODE_MEDIAN_TOK_S=96.97
GRAPH_ON_DECODE_MEAN_TOK_S=97.01
GRAPH_ON_DECODE_RANGE_TOK_S=96.97-97.08
GRAPH_ON_MTP_ACCEPTANCE_RATE=66.98%
GRAPH_ON_MTP_ACCEPTANCE_LENGTH=3.01
GRAPH_ON_TOTAL_DECODED_TOKENS=6144
GRAPH_OFF_DECODE_MEDIAN_TOK_S=95.09
GRAPH_ON_VS_OFF_DECODE_DELTA=+1.98%
TOTAL_GRAPH_AB_DECODED_TOKENS=12288
RESULT=PASS
```

Sustained decode used the unchanged canonical prompt and a benchmark-only request policy with
`request.stop.include_model_defaults = false`, matching `ninfer_bench`, so each measured run
reached exactly 2,048 decoded tokens instead of terminating at the model-default stop.

Validated v1.4 memory envelope:

```text
EMBEDDING_HOST_RESIDENT=795.70_MiB
PROCESS_VRAM=15028_MiB
FREE_AFTER_STARTUP_GRAPH_ON=794.56_MiB
PLANNED_SLACK_GRAPH_ON=715.54_MiB
GRAPH_OBSERVED=2.00_MiB
GRAPH_ALLOWANCE=82.00_MiB
VISION_ENCODE_WORKSPACE=132.3142_MiB
```

The older 118,001-token repetitive fixture remains a historical regression record only. The
canonical forward-looking benchmark is `workflow-118k-v1`.

## Vision source milestone — `7c10db07`

```text
repository: toddballinger/ninfer-5080
validated Vision commit: 7c10db07ac8c5803f921b83603b707750652873e
Vision branch: qwen3.8-27b-rtx5080-128k-vision
Vision release tag: qwen3.8-27b-rtx5080-128k-vision-v1
merged to main via PR #1
```

The Vision source adds independent Vision budgeting, earlier GDN temporary release, HostMapped Vision weights and cache-aware historical-media accounting.

## Feature-complete mainline runtime milestone — `b44b1958`

The runtime tree qualified after both the PR #3 `9e163eee` semantic port and PR #5 rolling-tool reconciliation is:

```text
runtime commit: b44b1958c301ec6bf4d18973a97d7b42fa6733aa
9e163eee semantic port: 4b62aca386a0a214049201ebeb2a422b0cb609ce
PR #3 merge: 33546d7d5be6d82eaac5e4a87a3f7e578f8a1a13
PR #5 merge / qualified runtime: b44b1958c301ec6bf4d18973a97d7b42fa6733aa
```

Exact post-reconciliation 118,001-token qualification:

```text
PROMPT_SHA256=078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
PROMPT_TOKENS=118001
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4-group64
SPECULATION=mtp
DRAFT_TOKENS=3
MAX_NEW=32
THINKING=off
GREEDY=on
CUDA_GRAPHS=off
PREFILL_TOK_S=1378.85
DECODE_TOK_S=71.44
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
GPU_WORKSPACE_PEAK=116.00_MiB
FREE_AFTER_STARTUP=44.56_MiB
PLANNED_SLACK=46.39_MiB
RESULT=PASS_EQUIVALENT_WITHIN_NOISE
```

PR #4 later advanced GitHub `main` to `c8439fbcb89a4daf74cf2692a9425930998c763f` with a documentation-only change to `docs/BENCHMARKS.md`. No runtime source changed, so the qualification above remains attached to `b44b1958`.

## Q5 A16 LinearAdd semantic-port milestone — `00e8e47f`

Upstream source mechanism:

```text
UPSTREAM_COMMIT=a9a0d10a933713d3066110a3caa4663c482319da
FORK_SEMANTIC_PORT=00e8e47fa6001067257f7ae6594c2deeabaed590
BASE_MAIN=2a40218c25941e0dbd27c6b10c7d125f1c122fc7
```

Preserved fork-specific behavior:

```text
4096_ROW_T1_RESIDUAL_GEMV=preserved
RTX5080_C64_CROSSOVER_BANDS=preserved
RESIDUAL_GEMV_INFRASTRUCTURE=preserved
SHARED_Q5_ROWSPLIT_GEMV=preserved
```

Operator A/B highlights:

```text
5120x6144_T1_SPEEDUP=1.342x
5120x17408_T1_SPEEDUP=1.453x
5120x6144_T513_SPEEDUP=1.520x
5120x17408_T513_SPEEDUP=1.543x
5120x6144_T621_SPEEDUP=1.138x
5120x17408_T621_SPEEDUP=1.084x
5120x6144_T1025_SPEEDUP=1.313x
5120x17408_T1025_SPEEDUP=1.322x
UNCHANGED_WIDTH_REGRESSION_GUARD=PASS
```

Exact 118,001-token qualification:

```text
PROMPT_SHA256=078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
PROMPT_TOKENS=118001
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4-group64
SPECULATION=mtp
DRAFT_TOKENS=3
MAX_NEW=32
THINKING=off
GREEDY=on
CUDA_GRAPHS=off
PREFILL_TOK_S=1376.30
DECODE_TOK_S=71.53
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
GPU_WORKSPACE_PEAK=116.00_MiB
FREE_AFTER_STARTUP=44.56_MiB
PLANNED_SLACK=46.39_MiB
PREFILL_DELTA_VS_B44B1958=-0.185%
DECODE_DELTA_VS_B44B1958=+0.126%
RESULT=PASS_EQUIVALENT_WITHIN_NOISE
```

## Original immutable text-only release

```text
validated commit: 473dade56031852a7d96edef049d859da96a6df9
release tag: qwen3.8-27b-rtx5080-128k-v1
```

The original tag remains unchanged as the historical baseline.

## Model artifact

```text
path used during validation:
/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer

bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Vision-source binaries

```text
ninfer:
5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140

ninfer-serve:
61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```

Historical text-release CLI binary:

```text
b38e987a16f7cdda3c5eee81b0ac821f9f2aad86f117b745401a9ef3b5c43012
```

## Model sources

```text
Qwen/Qwen3.8-27B
revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
revision: 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

## Quantization profile

```text
Q3G64_F16S_SHARE=42.42%
Q4G64_F16S_SHARE=45.92%
Q5G64_F16S_SHARE=11.57%
BF16_FP32_SHARE_APPROX=0.10%
Q4_VALUE_Z_COUNT=24
Q4_GATE_VALUE_COUNT=7
MAIN_TEXT_LOGICAL_PARAMS=26895998464
MAIN_TEXT_ENCODED_BYTES=13289938944
MAIN_TEXT_EFFECTIVE_BPW=3.953
QUANTIZED_MATRIX_PARAMS=26869760000
QUANTIZED_MATRIX_BYTES=13237452800
QUANTIZED_MATRIX_WEIGHTED_BPW=3.941
```

Public shorthand: **~3.95 BPW effective main-model quantization**.

## Common true-128K runtime

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
CUDA_GRAPHS=off
MAX_CONCURRENCY=1
```

## Recommended Vision profile

```text
VISION=on
VISION_MAX_TOKENS=1792
VISION_ENCODE_WORKSPACE=115.7751_MiB
FREE_AFTER_STARTUP=26.56_MiB
PLANNED_SLACK=28.88_MiB
```

Maximum validated Vision profile:

```text
VISION_MAX_TOKENS=2048
VISION_ENCODE_WORKSPACE=132.3142_MiB
FREE_AFTER_STARTUP=8.56_MiB
PLANNED_SLACK=10.08_MiB
```

## Vision-source long-context regression

Exact historical 118001-token corpus:

```text
PROMPT_SHA256=078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
PROMPT_TOKENS=118001
PREFILL_TOK_S=1375.16
DECODE_TOK_S=71.52
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
RETURN_CODE=0
```

Original text release on the same acceptance workload:

```text
PREFILL_TOK_S=1377.81
DECODE_TOK_S=71.51
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
```

No meaningful text-path regression was observed.

## Deterministic image validation at 1792

```text
IMAGE_INPUT=VALIDATED
IMAGE_TEST=512x256_RED_LEFT_BLUE_RIGHT
IMAGE_RESULT=LEFT_RED_RIGHT_BLUE
PROMPT_TOKENS=211
PREFILL_TOK_S=682.8
DECODE_TOK_S=118.1
TTFT_MS=725
MTP_TOK_PER_ROUND=3.10
MTP_ACCEPTANCE=70.0%
```

A previous run of the same deterministic image test measured 685.8 tok/s prefill and 719 ms TTFT; the repeated result confirms the functional path.

## Deterministic video validation at 1792

A six-second MP4 containing red, then green, then blue scenes was generated locally with ffmpeg and sent through the OpenAI-compatible server with thinking disabled.

```text
VIDEO_ON_FINAL_128K_HOSTMAPPED_PATH=VALIDATED
VIDEO_EXPECTED=red,green,blue
VIDEO_RESULT=red,green,blue
FINISH_REASON=stop_token
PROMPT_TOKENS=572
GENERATED_TOKENS=6
PREFILL_TOK_S=1721.9
DECODE_TOK_S=97.1
TTFT_MS=1111
WALL_S=1.16
MTP_TOK_PER_ROUND=4.00
MTP_ACCEPTANCE=100.0%
```

This is an end-to-end functional validation of video acquisition, preprocessing, Vision encode and generation. It is not a broad video-quality benchmark.

## Multi-image history validation

```text
OPENWEBUI_MULTI_IMAGE_HISTORY=VALIDATED
CACHED_MEDIA_FRESH_BUDGET_FIX=VALIDATED
OBSERVED_MEDIA_CACHE_PATTERNS=1/1/0,2/1/0
```

Cached historical media remains in the model prompt but is not charged repeatedly against the fresh preprocessing budget.

## Acceptance checks

```text
TRUE_131072_CONTEXT=PASS
TRUE_131072_KV=PASS
Q4_KV=PASS
MTP3=PASS
118001_TOKEN_REGRESSION=PASS
VISION_1792_STARTUP=PASS
VISION_2048_STARTUP=PASS
IMAGE_REQUEST=PASS
VIDEO_REQUEST=PASS
VIDEO_SEMANTIC_RESULT=PASS
MULTI_IMAGE_HISTORY=PASS
OOM_ERROR=NO
NONFINITE_WARNING=NO
```

---

## Qwen3.8-27B RTX 5080 v1.2 final validation

Validated code head: `dd2cb0341c321f8a808a6ed75f0a53225983f718`

Final exact 118,001-token acceptance: **1380.61 tok/s prefill**, **71.57 tok/s decode**, **44.74% MTP acceptance**, **2.31 tok/round**, with full **131,072 context / 131,072 Q4 KV**.

Vision 2048 acceptance also passed deterministic image, video, cached-history (`1/1/0` and `2/1/0`) and strict no-OOM validation. Startup remained intentionally tight at **8.56 MiB free / 10.08 MiB planned slack**.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md`.

---

## Qwen3.8-27B RTX 5080 v1.3 final validation

Validated production runtime code head: `ceb32f7d002edab224a83a2e2609f45fca4f8919`

```text
NINFER_SERVE_SHA256=3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
PREFIX_CHECKPOINT_POLICY=rolling-tool
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
VISION=on
VISION_MAX_TOKENS=2048
DEFAULT_THINKING_BUDGET=2048
```

v1.3 preserves the validated **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** RTX 5080 profile while adding corrected Q4 strided-output handling, server default thinking-budget support, and rolling tool checkpoints.

Production OpenClaw validation:

```text
RESTORE_CACHE_SEQUENCE=19023,21146,24664,26641
RESTORE_CACHE_ADVANCES=3
RESTORE_CACHE_PLATEAUS=0
RESTORE_CACHE_REGRESSIONS=0
PRODUCTION_ROLLING_CHECKPOINT=PASS
```

All five continuation requests had an uncached prompt suffix below 4,096 tokens.

The exact v1.2 118,001-token long-context and deterministic Vision/OOM validation remains preserved in the v1.2 release record.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`.

---

## Official Hugging Face publication — 2026-09-21

The project-maintained public artifact was published to:

```text
Hugging Face repository:
ninfer-5080/Qwen3.8-27B-RTX5080

artifact:
qwen3_8_27b.ninfer

bytes:
16461267456

SHA256:
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The publication was gated immediately before upload against the validated local artifact and the v1.3 production server binary.

```text
MODEL_VALIDATION=PASS
MODEL_SIZE=16461267456
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21

SERVER_VALIDATION=PASS
NINFER_SERVE_SHA256=3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
```

The uploaded model commit was:

```text
4120bd9a68cf2d2829c5f6b83ace8c9280b4b823
```

The Hugging Face repository also contains a model card and checksum record. A temporary write-access test file used before the release was removed after publication.

This publication establishes the project-owned Hugging Face repository as the canonical distribution location for the validated RTX 5080 artifact. The existing model SHA remains unchanged; later runtime work should not be described as requiring a new model artifact unless the artifact identity itself changes.

Automated CPU-only conversion/publication is being integrated separately. Any future workflow claiming byte-identical reproduction should enforce the exact expected byte size and SHA-256 before publishing.

