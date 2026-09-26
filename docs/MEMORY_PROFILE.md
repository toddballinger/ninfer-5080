# Memory Profile


## Current v1.4 production memory envelope

v1.4 changes the RTX 5080 memory picture materially. The token embedding table is now host-mapped
with `--embedding-host`, recovering about **796 MiB** of persistent VRAM. That headroom makes the
full Vision-2048 profile comfortable enough to enable CUDA Graph decode.

Validated production profile:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4-group64
PREFILL_CHUNK=896
MTP_DRAFT_TOKENS=3
VISION_MAX_TOKENS=2048
EMBEDDING_HOST=on
CUDA_GRAPHS=on
```

Measured startup envelope:

```text
embedding host-resident  795.70 MiB
process VRAM             15028 MiB
text_prefill             116.0127 MiB
mtp_prefill              116.0127 MiB
vision_encode            132.3142 MiB
free after startup       794.56 MiB
planned slack            715.54 MiB
graph observed             2.00 MiB
graph allowance           82.00 MiB
```

For comparison, the validated pre-v1.4 Vision-2048 profile had only **8.56 MiB free / 10.08 MiB
planned slack** with embeddings device-resident and CUDA Graph disabled. Those older measurements
below are retained as historical evidence, not the current recommendation.


The true-128K result is a near-capacity fit on a 16 GB RTX 5080. Vision is now part of the recommended configuration, so both text and Vision memory lifetimes matter.

## Text-model quantization

The text core is mixed Q3/Q4/Q5 groupwise:

| Format | Share | Encoded storage |
|---|---:|---:|
| Q3G64_F16S | 42.42% | 3.25 bpw |
| Q4G64_F16S | 45.92% | 4.25 bpw |
| Q5G64_F16S | 11.57% | 5.25 bpw |
| BF16 / FP32 | ~0.10% | small norms / misc. |

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953
quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

The exact-128K profile retains 24 GDN `value_z` tensors and 7 attention `gate_value` tensors in Q4, recovering about **210.625 MiB** versus the heavier comparison artifact.

## Artifact

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Text-only runtime on current Vision source

The current Vision source was regression-tested with the historical 118001-token workload and no Vision input:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
```

Measured values:

```text
GPU weights             ~12.64 GiB
KV cache payload          2.26 GiB
workspace peak          116.00 MiB
free after weights        2.56 GiB
free after startup       44.56 MiB
planned slack            46.39 MiB
```

The earlier text-only release had only about 11 MiB of planned slack. The GDN temporary lifetime reduction in the Vision source recovered roughly 35 MiB of peak workspace while preserving text performance.

## Why Vision originally did not fit

The historical Vision planner tied Vision workspace capacity to the large text-context capacity. That produced an impractically large Vision workspace at true 128K.

After decoupling the Vision token budget from text context, `--vision-max-tokens 1024` reduced Vision workspace to about 66.16 MiB. At that point the remaining shortfall was dominated by resident Vision weights, not Vision scratch.

Keeping the Vision weights GPU-resident still left the full-128K configuration short by roughly 240 MiB.

## HostMapped Vision weights

The validated solution keeps the Vision weights HostMapped instead of permanently resident in device memory. This recovers enough framebuffer to keep:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4
PREFILL_CHUNK=896
MTP_DRAFT_TOKENS=3
VISION=enabled
```

Image and video tests confirm the HostMapped path is usable. It can incur PCIe traffic versus fully GPU-resident Vision weights, but it makes true 128K + Vision possible on the 16 GB card.

## Vision token profiles

Measured Vision workspace:

```text
vision-max-tokens 1024 -> vision_encode  66.1580 MiB
vision-max-tokens 1792 -> vision_encode 115.7751 MiB
vision-max-tokens 2048 -> vision_encode 132.3142 MiB
```

At 1024, Vision workspace remains below the 116 MiB text-prefill peak. At 1792, Vision and text-prefill workspace are effectively the same size. At 2048, Vision becomes the workspace peak.

### Historical headroom profile: 1792

At `--vision-max-tokens 1792`:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      115.7751 MiB
free after weights   2.56 GiB
free after startup   26.56 MiB
planned slack        28.88 MiB
```

This profile retains substantially more startup margin than 2048 while keeping the full `131072 / 131072` context/KV allocation. Deterministic image and video requests both passed at this setting.

### Historical pre-v1.4 Vision-2048 profile

At `--vision-max-tokens 2048`:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after weights   2.56 GiB
free after startup    8.56 MiB
planned slack        10.08 MiB
```

This was a valid but extremely tight pre-v1.4 fit. v1.4 host-mapped embeddings replace this as the current production memory envelope.

## Cached historical media

OpenWebUI resends prior media as chat history. The frontend now distinguishes fresh preprocessing work from cached historical media: cache hits remain part of the prompt but are not charged repeatedly against the fresh media preprocessing cap.

This prevents a long multimodal chat from failing simply because previously processed images are resent. Historical Vision tokens still count toward the context window normally.

## Clean-start requirement

A representative successful launch began from:

```text
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB
```

Recommended preflight:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

At this utilization level, tens of MiB are material: they determine whether full `131072 / 131072` plus Vision starts or fails during runtime reservation.

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
