# Benchmarks


## Canonical v1.4 RTX 5080 benchmark — `workflow-118k-v1`

The forward-looking long-context benchmark is the deterministic mixed engineering-agent workflow:

```text
fixture: bench/fixtures/workflow-118k-v1/qwen38_118001_workflow_candidate.txt
SHA256: cb7c131bd20d78bd69396c019f988c81fad1941a853a8de7da7586e1aeb99718
prepared prompt tokens: 118001
```

Configuration:

| Item | Value |
|---|---:|
| GPU | RTX 5080 16 GB |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| Host-mapped embeddings | enabled |
| Vision profile | 2048 |
| CUDA Graph | enabled |

Three exact Graph-ON long-decode runs:

| Metric | Result |
|---|---:|
| Prefill median | **1,361.76 tok/s** |
| Prefill range | 1,351.79–1,362.69 tok/s |
| Decode median | **96.97 tok/s** |
| Decode mean | **97.01 tok/s** |
| Decode range | **96.97–97.08 tok/s** |
| Decode standard deviation | **0.06 tok/s** |
| MTP acceptance | **66.98%** |
| MTP accepted length | **3.01 tok/round** |
| Workspace peak | **116.00 MiB** |
| Planned slack | **715.54 MiB** |

Each run measured exactly 2,048 decoded tokens, for **6,144 Graph-ON decoded tokens**.

### CUDA Graph A/B

| Metric | Graph ON | Graph OFF | Difference |
|---|---:|---:|---:|
| Prefill median | 1,361.76 | 1,364.12 tok/s | -0.17% |
| Decode median | **96.97** | 95.09 tok/s | **+1.98%** |
| MTP acceptance | 66.98% | 66.98% | unchanged |
| MTP accepted length | 3.01 | 3.01 tok/round | unchanged |
| Planned slack | 715.54 | 804.08 MiB | -88.54 MiB |

The complete A/B contains **12,288 decoded tokens**. CUDA Graph is therefore the recommended v1.4
production profile.

The normal model emits its default stop token after roughly 45 generated tokens on this deterministic
fixture. For sustained throughput measurement only, the benchmark driver sets
`request.stop.include_model_defaults = false`, the same policy used by `ninfer_bench`. The
prompt, model artifact, sampling and inference kernels are unchanged.

Older repeated-corpus results below remain historical comparison/regression data and are not the
canonical benchmark moving forward.


## Historical Vision-source regression — `7c10db07`

The Vision-enabled source was regression-tested with the exact historical 118,001-token workload while keeping the full 131,072-token maximum context and KV capacity allocated.

| Metric | Vision source |
|---|---:|
| Prompt tokens | 118001 |
| Max context | 131072 |
| KV capacity | 131072 |
| Prefill chunk | 896 |
| KV dtype | Q4 |
| Speculation | MTP-3 |
| Prefill | **1375.16 tok/s** |
| Decode | **71.52 tok/s** |
| MTP acceptance rate | **44.74%** |
| MTP acceptance length | **2.31 tok/round** |
| Workspace peak, text-only CLI | 116.00 MiB |
| Free after startup, text-only CLI | 44.56 MiB |
| Planned slack, text-only CLI | 46.39 MiB |

Validated Vision source commit before merge to `main`:

```text
7c10db07ac8c5803f921b83603b707750652873e
```

## Original text-only release comparison

| Result | Prefill | Decode | MTP acceptance | Acceptance length |
|---|---:|---:|---:|---:|
| Original validated text release | 1377.81 tok/s | 71.51 tok/s | 44.74% | 2.31 |
| Vision source, same 118001-token corpus | 1375.16 tok/s | 71.52 tok/s | 44.74% | 2.31 |

The Vision source is about 0.19% lower in prefill and effectively identical in decode/MTP behavior. This is within normal run-to-run variation; no meaningful text-path regression was observed.

## Historical Vision 1792 serving profile

The recommended Vision profile is empirically validated with the full `131072 / 131072` text context/KV allocation and:

```text
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--vision
--vision-max-tokens 1792
```

Measured startup envelope:

| Item | Value |
|---|---:|
| Text prefill workspace | 116.0127 MiB |
| MTP prefill workspace | 116.0127 MiB |
| Vision encode workspace | 115.7751 MiB |
| Free after startup | 26.56 MiB |
| Planned slack | 28.88 MiB |

A deterministic synthetic image request also passed. The input was a 512×256 image with a red left half and blue right half; the model correctly returned that the left half was red and the right half blue.

Request metrics:

```text
prompt=211
generated=62
prefill=685.8 tok/s
decode=118.3 tok/s
ttft=719 ms
MTP=3.10 tok/round (70.0%)
```

## Video validation

Video is empirically validated on the final 128K HostMapped Vision configuration. The test used a deterministic 6-second MP4 containing three solid-color scenes in chronological order: red, green and blue. With thinking disabled and a constrained answer format, the model returned `red, green, blue`.

The server retained the full `131072 / 131072` context/KV allocation. Request metrics were `prompt=572`, `generated=6`, `prefill=1721.9 tok/s`, `decode=97.1 tok/s`, `ttft=1111 ms`, `wall=1.16 s`, MTP `4.00 tok/round`, and `finish=stop_token`.

This is an end-to-end functional validation of video acquisition, preprocessing, Vision encode and generation, not a broad video-understanding quality benchmark.

## Vision 2048 serving profile

The maximum validated Vision profile kept the full `131072 / 131072` text context/KV allocation and used:

```text
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--vision
--vision-max-tokens 2048
```

Measured startup envelope:

| Item | Value |
|---|---:|
| Text prefill workspace | 116.0127 MiB |
| MTP prefill workspace | 116.0127 MiB |
| Vision encode workspace | 132.3142 MiB |
| Free after weights | 2.56 GiB |
| Free after startup | 8.56 MiB |
| Planned slack | 10.08 MiB |

A representative OpenWebUI image request at the 2048 profile reported:

```text
prompt=9134
prefill=2153.3 tok/s
decode=113.7 tok/s
ttft=8.35 s
MTP=3.01 tok/round (67.1%)
```

This includes the complete OpenWebUI prompt/tool payload and is not a standalone Vision-kernel microbenchmark.

## Multi-image history validation

After the cache-aware media-budget fix, OpenWebUI conversations containing old images plus a newly uploaded image completed instead of failing with `media_budget_exceeded`.

Observed request patterns included:

```text
media_cache=1/1/0
media_cache=2/1/0
```

Cached historical images remain present in the model prompt but do not consume the fresh preprocessing budget again.

## Historical comparison

| Result | Prefill | Decode |
|---|---:|---:|
| Older baseline | 1235.03 tok/s | — |
| Historical optimized | 1371.10 tok/s | — |
| Historical B133 | 1377.66 tok/s | 70.24 tok/s |
| Original final text release | 1377.81 tok/s | 71.51 tok/s |
| Vision source (`7c10db07`) | **1375.16 tok/s** | **71.52 tok/s** |

## Comparing results fairly

Public comparisons should report GPU, VRAM, quantization profile, actual prompt tokens, max context, allocated KV capacity, KV dtype, prefill chunk, speculation settings, Vision settings, prefill speed and decode speed. Short-prompt decode numbers are not directly comparable with the 118K active-prompt result.

---

## Qwen3.8-27B RTX 5080 v1.2 final validation

Validated code head: `dd2cb0341c321f8a808a6ed75f0a53225983f718`

Final exact 118,001-token acceptance: **1380.61 tok/s prefill**, **71.57 tok/s decode**, **44.74% MTP acceptance**, **2.31 tok/round**, with full **131,072 context / 131,072 Q4 KV**.

Vision 2048 acceptance also passed deterministic image, video, cached-history (`1/1/0` and `2/1/0`) and strict no-OOM validation. Startup remained intentionally tight at **8.56 MiB free / 10.08 MiB planned slack**.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md`.

---

## Qwen3.8-27B RTX 5080 v1.3 production validation — `ceb32f7d`

Validated production runtime code head: `ceb32f7d002edab224a83a2e2609f45fca4f8919`

v1.3 preserves the validated **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** RTX 5080 profile while adding corrected Q4 strided-output handling, server default thinking-budget support and configurable `stable-turn` / `rolling-tool` prefix checkpoints.

The validated OpenClaw production profile uses:

```text
--prefix-checkpoint-policy rolling-tool
```

The production rolling-tool smoke test advanced restore checkpoints `19023 -> 21146 -> 24664 -> 26641` with 3 advances, 0 plateaus and 0 regressions. The three-request MTP sanity sample averaged **84.7 tok/s decode**, **40.47% MTP acceptance** and **2.213 tok/round**. Client `reasoning_budget=64` and server-default `reasoning_budget=2048` resolution both passed.

The exact v1.2 118,001-token long-context and deterministic Vision/OOM validation remains preserved in the v1.2 release record.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`.


---

## PR #3 post-merge production qualification

Merge commit `33546d7d5be6d82eaac5e4a87a3f7e578f8a1a13` includes the semantic port of upstream `9e163eee4b8acec21ab0ac765107b6a3f287b217`, which retunes Q4/Q5 attention and GDN input-projection routing by column band while preserving the RTX 5080 fork's Q4/Q4, Q4 value_z, A8, 4096-geometry and workspace-aware behavior.

The operator-level A/B benchmark against pre-port baseline `bfe42032` showed substantial local gains:

- attention: **42 candidate wins / 1 minor warm-cache variance / 11 ties**
- GDN: **20 candidate wins / 0 losses / 16 ties**
- representative attention gains ranged from roughly **10–54%** in changed bands
- representative GDN gains ranged from roughly **12–53%** in changed bands

After merge, the exact historical 118,001-token production workload was rerun three times on RTX 5080 16 GB with:

```text
prompt tokens: 118001
max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4-group64
speculation:   MTP-3
max-new:       32
thinking:      disabled
sampling:      greedy
CUDA Graph:    disabled
```

Results:

| Run | Prefill | Decode | MTP acceptance | MTP length |
|---|---:|---:|---:|---:|
| 1 | 1377.46 tok/s | 71.50 tok/s | 44.74% | 2.31 tok/round |
| 2 | 1378.82 tok/s | 71.50 tok/s | 44.74% | 2.31 tok/round |
| 3 | 1378.51 tok/s | 71.40 tok/s | 44.74% | 2.31 tok/round |
| **Mean** | **1378.263 tok/s** | **71.467 tok/s** | **44.74%** | **2.31 tok/round** |

Against the strongest previously validated v1.2 reference of **1380.61 tok/s prefill / 71.57 tok/s decode**, the three-run mean is **-0.170% prefill and -0.144% decode**. This is within normal run-to-run variation and is treated as end-to-end performance equivalence rather than a regression.

The lack of a material whole-model throughput increase is expected. PR #3 optimizes specific Q4/Q5 attention and GDN input-projection kernels, while the 118K workload also spends substantial time in other transformer/runtime stages. The port does not change model weights, KV representation, persistent allocations, or the overall true-128K memory plan. The operator benchmark therefore captures the intended local speedups; the full-model benchmark serves primarily as a production-regression check.

Memory behavior remained unchanged at the validated geometry:

```text
GPU workspace peak:   116.00 MiB
free after startup:    44.56 MiB
planned slack:          46.39 MiB
```

Production qualification result: **PASS**.

---

## Feature-complete mainline runtime qualification — `b44b1958`

PR #5 reconciled the validated v1.3 rolling-tool checkpoint feature into the PR #3 / `9e163eee` lineage. The resulting runtime tree was:

```text
b44b1958c301ec6bf4d18973a97d7b42fa6733aa
```

That exact tree was rebuilt with ccache and rerun once against the historical 118,001-token acceptance workload:

| Metric | Result |
|---|---:|
| Prompt tokens | 118001 |
| Max context | 131072 |
| KV capacity | 131072 |
| Prefill chunk | 896 |
| KV dtype | q4-group64 |
| Speculation | MTP-3 |
| Max new | 32 |
| Thinking | disabled |
| Sampling | greedy |
| CUDA Graph | disabled |
| Prefill | **1378.85 tok/s** |
| Decode | **71.44 tok/s** |
| MTP acceptance | **44.74%** |
| MTP length | **2.31 tok/round** |
| GPU workspace peak | **116.00 MiB** |
| Free after startup | **44.56 MiB** |
| Planned slack | **46.39 MiB** |

Against the strongest v1.2 reference of **1380.61 tok/s prefill / 71.57 tok/s decode**, the `b44b1958` run is **-0.127% prefill and -0.182% decode**. MTP acceptance and acceptance length are unchanged. The result is classified as **equivalent within noise**.

The benchmark used model artifact SHA256 `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` and prompt SHA256 `078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd`.

GitHub `main` later advanced to `c8439fbcb89a4daf74cf2692a9425930998c763f` through PR #4. The only change between `b44b1958` and `c8439fb` is documentation in this file, so the runtime qualification remains attached to `b44b1958` rather than being relabelled as a benchmark of the documentation-only descendant.

---

## Q5 A16 LinearAdd `a9a0d10a` semantic port — `00e8e47f`

Upstream `a9a0d10a933713d3066110a3caa4663c482319da` was integrated as an
RTX 5080-specific semantic port at:

```text
00e8e47fa6001067257f7ae6594c2deeabaed590
```

The port deliberately preserves the fork's existing 4096-row T=1
residual-GEMV path, RTX 5080 C64 crossover bands, residual-GEMV
infrastructure and shared Q5 rowsplit implementation. The new T=1 Split2
and >512 narrow-tail mechanism applies to the two 5120-row Q5 A16
LinearAdd shapes.

### RTX 5080 operator A/B

| Shape | T | Baseline | Candidate | Improvement |
|---|---:|---:|---:|---:|
| 5120x6144 | 1 | 19.080 us | 14.216 us | **25.49%** |
| 5120x17408 | 1 | 53.622 us | 36.914 us | **31.16%** |
| 5120x6144 | 513 | 519.193 us | 341.638 us | **34.20%** |
| 5120x17408 | 513 | 1465.106 us | 949.533 us | **35.19%** |
| 5120x6144 | 621 | 528.892 us | 464.936 us | **12.09%** |
| 5120x17408 | 621 | 1489.453 us | 1373.820 us | **7.76%** |
| 5120x6144 | 1025 | 872.084 us | 664.394 us | **23.82%** |
| 5120x17408 | 1025 | 2453.607 us | 1855.929 us | **24.36%** |

The unchanged/fallback widths remained flat. In particular, T=896 measured
+0.18% for 5120x6144 and effectively 0.00% for 5120x17408, both within
measurement noise.

Regression-guard widths:

```text
512\n705\n768\n896\n1024
```

### Exact 118,001-token whole-model qualification

```text
prompt tokens: 118001
max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4-group64
speculation:   MTP-3
max-new:       32
thinking:      disabled
sampling:      greedy
CUDA Graph:    disabled
```

| Metric | `b44b1958` reference | `00e8e47f` |
|---|---:|---:|
| Prefill | 1378.85 tok/s | **1376.30 tok/s** |
| Decode | 71.44 tok/s | **71.53 tok/s** |
| MTP acceptance | 44.74% | **44.74%** |
| MTP length | 2.31 | **2.31** |
| GPU workspace peak | 116.00 MiB | **116.00 MiB** |
| Free after startup | 44.56 MiB | **44.56 MiB** |
| Planned slack | 46.39 MiB | **46.39 MiB** |

The deltas are **-0.185% prefill** and **+0.126% decode**. The full-model
result therefore remains equivalent within normal run-to-run noise while
the targeted operator cliffs are materially improved.

Validation identities:

```text
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
PROMPT_SHA256=078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
```

Qualification result: **PASS**.
Canonical 118,001-token qualification fixture

The repository includes the exact synthetic fixture used for the historical
118,001-token RTX 5080 long-context qualification:

bench/fixtures/qwen38_118001_prompt.txt

Identity:

bytes: 491625
SHA256: 078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
prepared prompt tokens: 118001

The fixture consists of deterministic repeated Greek-alphabet text and contains
no private user data, credentials, or external corpus material.

Run the canonical qualification with:

tools/qualify_qwen38_118k.sh /path/to/qwen3_8_27b.ninfer

The harness verifies both the fixture SHA-256 and the canonical validated model
artifact SHA-256 before execution.

The qualification configuration is:

max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4-group64
speculation:   MTP-3
max new:       32
thinking:      disabled
sampling:      greedy
CUDA Graph:    disabled

Additional NInfer CLI options can be appended after the model path to compare a
runtime feature against the same immutable fixture and configuration.
