# E101 — Q3 A8 INT8 Tensor-Core Prefill Validation

Commit: ce1f6fea8f0102b2c96706d10bddc651ba850b8c
Branch/tag: e101-q3-a8-int8-prefill

## Hardware

- GPU: NVIDIA GeForce RTX 5080 16 GB
- Driver: 595.91.07
- CUDA: 13.3
- Power limit: 330 W

## Model

- Qwen3.8-27B custom NInfer
- Effective model BPW: ~4.09
- Context: 98,304
- KV: Q4
- Prefill chunk: 1024
- MTP: 3 for OpenClaw V2
- Q3 T=1 decode path unchanged

## Controlled 90K A/B

Identical binary/model/configuration.
Only NINFER_Q3_E101 changed.

A16:
- 90,067 prompt tokens
- 761.7 tok/s
- TTFT 118.389 s

E101 first run:
- 90,067 prompt tokens
- 1421.1 tok/s
- TTFT 63.537 s

Gain:
- 1.866x
- +86.6%

## Five-run E101 repeatability

Prefill:
- 1389.4 tok/s
- 1389.5 tok/s
- 1394.6 tok/s
- 1394.5 tok/s
- 1391.5 tok/s

Mean: 1391.9 tok/s
Median: 1391.5 tok/s
Std dev: 2.56 tok/s
CV: 0.18%
Mean TTFT: 64.853 s

Mean gain vs same-binary A16:
- 1.827x
- +82.7%

## OpenClaw V2

NInfer / llama.cpp:

- 650:   1909.77 / 738.19 tok/s
- 2.5K:  2043.82 / 1388.36
- 10K:   2022.55 / 1672.97
- 32K:   1709.42 / 1575.15
- 64K:   1506.72 / 1401.02
- 90K:   1370.07 / 1265.78
- decode: 127.03 / 121.61 tok/s

90K NInfer advantage: ~8.2%

## OpenClaw V2 intelligence

NInfer:
- coding: 100.0%
- orchestration: 81.2%
- reasoning: 100.0%
- long_context: 66.7%
- instruction: 100.0%
- aggregate: 89.38%

No measurable regression versus the prior NInfer V2 intelligence result.

Do not interpret the lower llama aggregate in this run as an
E101 intelligence advantage; llama output varied between runs.

## Status

E101 VALIDATED.

The Q3 A8 INT8 Tensor-Core prefill route is suitable for promotion,
subject to retaining a documented A16 fallback and numerical-path notes.
