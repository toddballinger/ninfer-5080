# NInfer RTX 5070 Ti — Qwen3.8-27B at true 128K + Vision on 16 GB

> 本仓库是 [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080) 的分支，
> 在其验证成果之上增加了 **RTX 5070 Ti 的原生 Windows (MSVC) 移植与实测**。
>
> **特别感谢 [@toddballinger](https://github.com/toddballinger)**（ninfer-5080 作者，验证了
> 128K / Q4-KV / MTP-3 / Vision 运行时）**和 [Neroued](https://github.com/Neroued)**（NInfer
> 原作者）—— 本移植建立在两位的成果之上。

**中文导读**

- **这是什么**：Qwen3.8-27B 在单张 16 GB 显卡上跑真 128K 上下文 + Vision 的推理运行时；本分支补上了 5070 Ti 的 Windows 原生构建。
- **为什么 5070 Ti 能直接用**：5070 Ti 与 5080 同为 GB203 / compute capability 12.0（`sm_120a`）、同为 16 GB 显存，因此模型 artifact（配方 `groupwise-int-5080`）与 131072-token KV 分配完全通用，无需重新量化。
- **Windows 构建与运行**：见 [docs/WINDOWS_5070TI.md](docs/WINDOWS_5070TI.md)。
- **本机实测（5070 Ti）**：prefill ~1301 tok/s、decode ~65 tok/s（8K 上下文 + MTP-3）。
- **破限（无审查）模型**：本人用 NInfer 转换工具，把社区 abliterated 权重 `vkshdev/Qwen-3.8-28B-uncensored` 自己转成了 `.ninfer` 格式（非社区现成 `.ninfer`，同一 NInfer 运行时直接加载），见 [YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer](https://huggingface.co/YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer)。

## 快速上手（Windows · RTX 5070 Ti）

### 一、安装

1. **环境**：Visual Studio 2022（勾选「使用 C++ 的桌面开发」工作负载）+ CUDA Toolkit 13.4。
2. **依赖**：FFmpeg + libcurl（推荐 vcpkg：`vcpkg install ffmpeg:x64-windows curl:x64-windows`）。
3. **模型**：下载 `qwen3_8_27b.ninfer`（16,461,267,456 字节，SHA-256 `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc6f6ed1ec6b050e21`）到本地。
4. **构建**：完整的环境准备、vcpkg 依赖、构建命令都在 [docs/WINDOWS_5070TI.md](docs/WINDOWS_5070TI.md)。

### 二、启动与关闭（开关）

服务跑在 `127.0.0.1:8100`。

**启动**（在仓库根目录执行，`<模型路径>` 换成实际的 `.ninfer` 文件）：

- **带图片识别（推荐）**——加 `--vision`；视觉编码器要额外约 2GB 显存，上下文需降到 16384：

```
build-windows\apps\ninfer-serve.exe <模型路径> --host 127.0.0.1 --port 8100 --max-context 16384 --kv-dtype q4 --spec mtp --draft-tokens 3 --embedding-host --max-concurrency 1 --model-id m --vision
```

- **纯文本（更长上下文）**——不带 `--vision`，上下文可用到 65536。

**打开页面**：用浏览器打开仓库里的 `chat.html`。发图会自动转成 JPEG 再上传（服务端 FFmpeg 未编译 PNG 解码器，只支持 JPEG/BMP 等内置格式）。

**关闭**：结束 `ninfer-serve` 进程（任务管理器里找 ninfer-serve，或 PowerShell 执行 `Stop-Process -Name ninfer-serve`）。

> 本机另附 `start_chat.ps1`（一键启动，默认带 vision + 起完自动打开页面）/ `stop_chat.ps1`（一键关闭），Windows 下请双击 `start_chat.bat` / `stop_chat.bat`（双击 `.ps1` 会用记事本打开而不是运行）。脚本内含本机绝对路径，换机器需改路径。

> 关于显存：开视觉要额外预留约 2GB（视觉编码器固定分配），16GB 卡下 65536 上下文装不下 vision，需降到 16384；想同时跑满 `--max-context 131072` 得把桌面挪到核显。

### 三、聊天页面设置

页面上有四个可以调的地方：

- **系统提示**（第二行输入框）：默认是一段「开放创作 + 理性讨论」的引导，可改成你想要的任何设定。
- **思考**（勾选框）：勾上模型先想一遍再说（更稳）；取消更直接、更快。
- **温度**：越高越有张力、越少套话，越低越收敛。
- **max tokens**：单次回复上限。

---

This fork documents and maintains a validated **Qwen3.8-27B** configuration for a single **NVIDIA RTX 5070 Ti 16 GB** with a genuine **131,072-token context and KV capacity**, Q4 KV, MTP-3 speculative decoding, and Vision support.

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
| Target | Qwen3.8-27B / RTX 5070 Ti 16 GB |
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

The same artifact SHA has been retained across the original text-only release, Vision enablement, the v1.2/v1.3 production runtime work, and subsequent qualified runtime optimizations.

## Uncensored (abliterated) model

In addition to the official aligned artifact, a **self-converted abliterated (uncensored)** build of Qwen3.8-27B is available in the same `.ninfer` format — produced by running the NInfer converter over the community abliterated weights (not a pre-existing community `.ninfer` release):

**[YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer](https://huggingface.co/YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer)**

| Field | Value |
|---|---|
| File | `qwen3_8_27b_uncensored.ninfer` |
| Size | ~15.33 GB |
| Recipe | `groupwise-int-5080` (Q3/Q4/Q5 mixed) |
| Base | Qwen3.8-27B, refusal-direction removed (ZeroFuse abliteration) |
| Source weights | `vkshdev/Qwen-3.8-28B-uncensored` (converted locally with the NInfer toolchain) |
| Vision | supported (`--vision`; lower `--max-context` to 8192 on 16 GB) |

It loads with the same NInfer runtime — no additional engine work is required. Quality impact vs the official build is small (~4% on competition-level problems, zero difference on everyday tasks); see the model card for the full measured comparison.

## Validated runtime releases

### v1.3 production runtime

The current production release record is **v1.3**, validated at:

```text
source commit:
ceb32f7d002edab224a83a2e2609f45fca4f8919

ninfer-serve SHA-256:
3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
```

v1.3 preserves the full 128K/Q4-KV/MTP-3/Vision profile and adds corrected Q4 strided-output handling, a server-wide default thinking budget, and rolling tool checkpoints for agent/tool-loop workloads.

See [the full v1.3 release record](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md).

### Later qualified runtime work

Runtime development continues after v1.3. For example, the Q5 A16 LinearAdd semantic port at `00e8e47f` was re-qualified with the exact 118,001-token workload while continuing to use the same canonical model SHA.

The repository therefore treats **runtime source identity** and **model artifact identity** separately. See [VALIDATED_MANIFEST.md](docs/VALIDATED_MANIFEST.md) for the exact commit-scoped qualification ledger.

## Recommended serving profiles

### Recommended headroom profile — Vision 1792

For the most comfortable 16 GB memory margin, use:

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
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool \
  --vision \
  --vision-max-tokens 1792
```

Measured startup envelope:

```text
vision_encode       115.7751 MiB
free after startup   26.56 MiB
planned slack        28.88 MiB
```

### Maximum validated profile — Vision 2048

Vision 2048 is also validated and is the profile used by the v1.3 production OpenClaw deployment, but the remaining GPU margin is much tighter:

```text
vision_encode      132.3142 MiB
free after startup   8.56 MiB
planned slack       10.08 MiB
```

Use a clean GPU for both profiles, especially 2048. See [VISION_128K.md](docs/VISION_128K.md) for the complete memory and multimodal validation record.

## Long-context validation

The project uses an exact historical **118,001-token** workload to prevent “128K” claims from being based only on a configured maximum.

A feature-complete mainline runtime at `b44b1958` produced:

| Metric | Result |
|---|---:|
| Prompt tokens | 118,001 |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| Prefill | **1378.85 tok/s** |
| Decode | **71.44 tok/s** |
| MTP acceptance | **44.74%** |
| Acceptance length | **2.31 tok/round** |

The later Q5 A16 LinearAdd semantic port at `00e8e47f` was independently qualified on the same workload at **1376.30 tok/s prefill** and **71.53 tok/s decode**, with the same model SHA and the same 44.74% MTP acceptance.

A result is described here as **true 128K** only when both the configured context and allocated KV capacity are actually `131072`.

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

- [v1.3 release](docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md) — current production release record
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

Upstream changes are evaluated for **production-path relevance first**. A microbenchmark improvement on a kernel or shape that the validated Qwen3.8-27B RTX 5070 Ti workload does not exercise is normally deferred rather than merged only to reduce a GitHub “behind” count.

See [UPSTREAM_SYNC_STATUS.md](docs/UPSTREAM_SYNC_STATUS.md) for the current commit-scoped ledger and acceptance policy.

## Attribution and licensing

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches or converted artifacts, and verify the applicable upstream/model licenses for the material being redistributed.
