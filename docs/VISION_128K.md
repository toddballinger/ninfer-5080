# Qwen3.8-27B true 128K + Vision on RTX 5080 16 GB

Vision is now the recommended path in this repository. The original text-only release remains preserved as the historical baseline.

Validated Vision source commit before merge to `main`:

`7c10db07ac8c5803f921b83603b707750652873e`

## Recommended v1.5 serving command

The current RTX 5080 production recommendation is **Vision 2048 with host-mapped embeddings, CUDA Graph enabled, and a 1792-token prefill chunk**:

{f}bash
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
{f}

CUDA Graph is enabled by default; do not add `--no-cuda-graph` for the recommended v1.5 profile.

Validated v1.5 deployment envelope:

{f}text
vision_encode            132.3142 MiB
free after startup       885.94 MiB
planned slack            806.92 MiB
graph observed             2.00 MiB
graph allowance           82.00 MiB
{f}

v1.5 retains the HostMapped Vision path and host-resident token embeddings while moving the production prefill chunk from 896 to 1792 under CUDA 13.4.92 / NVIDIA 615.71.09.

### Historical pre-v1.4 profiles

Before `--embedding-host`, Vision 1792 was recommended for headroom:

```text
vision_encode             115.7751 MiB
free after startup         26.56 MiB
planned slack              28.88 MiB
```

The pre-v1.4 Vision-2048 profile also passed, but was extremely tight:

```text
vision_encode             132.3142 MiB
free after startup          8.56 MiB
planned slack              10.08 MiB
```

Those measurements are retained for historical comparison; they are not the current production
recommendation.

## Validation

The Vision source was re-tested with the exact historical 118,001-token corpus using full 131,072 context/KV, Q4 KV, chunk 896 and MTP-3:

| Metric | Original | Vision source |
|---|---:|---:|
| Prefill | 1377.81 tok/s | 1375.16 tok/s |
| Decode | 71.51 tok/s | 71.52 tok/s |
| MTP acceptance | 44.74% | 44.74% |
| Acceptance length | 2.31 | 2.31 |

No meaningful text-performance regression was observed.

Image understanding is empirically validated. OpenWebUI multi-image history is also validated: cached historical media no longer consumes the fresh preprocessing budget again, while genuinely new media still does.

Video input is now empirically validated on the final 128K HostMapped Vision configuration using a deterministic chronological-color MP4 test. This validates the end-to-end video acquisition, preprocessing, Vision encode and generation path on the RTX 5080 configuration; it is not a broad video-quality benchmark.

## v1.5 validation identity

{f}text
release tag:         qwen3.8-27b-rtx5080-128k-vision-v1.5
release commit:      e4edd6d5c5f9f7996de0f3d9f6c311e883452580
validated source:    e4353f061bf0e378c83472cb2bcaf99e65681f4e
model SHA256:        c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
ninfer-serve SHA256: 928e5615ef453786f47f79b6af2152d2f8f8d61307656f23c47fa45b5ed41167
{f}

## Historical Vision-source validation hashes

```text
model SHA256:        c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
ninfer SHA256:       5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140
ninfer-serve SHA256: 61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```
