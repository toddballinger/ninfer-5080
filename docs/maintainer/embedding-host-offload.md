# Embedding host residency

`--embedding-host` keeps the `token_embedding` table in pinned host RAM instead of device memory. The
table is the model's largest single weight — the `[vocab_size, hidden_size]` `token_embedding` — so
binding it to host RAM frees that VRAM. The gather is the same CUDA `embedding` Op on every route;
the only difference is that the table's planes point at page-locked host memory, so the kernel reads
the needed rows over PCIe (UVA) instead of from device memory.

This reference owns the residency choice, the host-mapped materialization and its ownership, and the
memory model of the host route. The embedding mathematics and the per-layer composition stay in the
target model references; the `embedding` Op contract stays in [Op development](op-development.md);
operator-facing flags and the memory tradeoff stay in the [serving guide](../serving.md#embedding-host-residency).

## Residency

The `token_embedding` table is bound with exactly one of two placements, fixed at load:

- **Device placement** (default): the table is uploaded to device memory and the `embedding` Op
  reads it from the device.
- **Host-mapped placement** (`--embedding-host`): the table is retained in page-locked host RAM and the
  `embedding` Op reads the needed rows over PCIe (UVA).

The gather is the same `embedding` Op in both cases — the Op is agnostic to where the table lives,
because its kernels are plain pointer dereferences over the table planes. The flag is independent of
`--vision` and can be combined with it, and with every execution route, including the speculative
backends (`--spec mtp|dflash|dflash2`). The speculative backends reuse the same target
`token_embedding` table, and their target-verify and draft-stem gathers are interior device ops
inside the captured decode graph; because the gather is a device Op (not a host gather), it runs
inside the captured body unchanged. The same artifact serves both placements; host residency is a
binding choice at load, not an artifact property.

## Host-mapped materialization and ownership

The host route reuses the same `HostMapped` materialization the fork already uses for the Vision
tower weights and the DFlash2 candidate-selector codebooks:

- **Artifact** owns the retained host bytes and their page-locking:
  [`MappedHostBuffer`](../../src/core/arena.cu) is allocated with
  `cudaHostAlloc(cudaHostAllocMapped)` and mapped with `cudaHostGetDevicePointer`, so the host
  buffer is page-locked and directly addressable by CUDA kernels through the mapped device pointer.
  [`artifact::materialize`](../../src/artifact/materializer.cpp) copies the artifact payload into the
  mapped buffer and records `object.device = mapped->device_data()`, so
  `MaterializedArtifact::device_data` returns the mapped pointer for a host-mapped object.
- **Target** owns the residency choice: each target's `bind_artifact`
  (e.g. [`qwen3_6_27b/impl/load/bindings.cpp`](../../src/targets/qwen3_6_27b/impl/load/bindings.cpp))
  binds `text/token_embedding` with `TensorPlacement::HostMapped` when
  `StartupFeatures::embedding_host` is set, and `TensorPlacement::Device` otherwise. The flag flows from
  `EngineOptions::embedding_host` through
  [`startup_features`](../../src/targets/qwen3_6/export/ninfer/targets/qwen3_6/startup_features.h)
  into every target's binding.
- **Ops** own the gather: the `embedding` Op
  ([`include/ninfer/ops/embedding.h`](../../include/ninfer/ops/embedding.h)) reads the table planes
  (device or pinned host) and is unchanged by the residency.
- **Target** owns the gather sites: the prefill, ordinary-decode, and speculative gathers in the
  family runtime all call the `embedding` Op with the same `token_embedding` `Weight` view; there is
  no host-route branch.

## CUDA-Graph

No handoff is needed. The gather is a device Op, so it runs inside the captured decode graph exactly
as on the device route — including the speculative target-verify and draft-stem gathers, whose input
IDs are computed on-device by a preceding device op inside the same graph. The table is read over
PCIe on every gather; because a round gathers only a handful of rows (tens of rows, < 1 MB), the
per-round PCIe traffic is negligible. Prefill gathers the whole chunk (up to the prompt length),
which is a one-time cost per request.

## Memory model

- **Table weights** are bound with host-mapped placement at load, page-locked, and never uploaded;
  their resident bytes are reported as `server_start.load.embed_host_weight_bytes`, computed from
  the table's stored payload size in the host-mapped materialization.
- **No staging, no worker pool**: unlike a host-gather design, there is no pinned staging buffer and
  no host worker pool — the GPU reads the table directly over PCIe.
- **Device preflight and KV sizing** see the table as absent from the device arena: the
  host-mapped object is excluded from `device_capacity_bytes`, so the freed VRAM is available to the
  KV capacity resolution.

The numerical boundaries of the gather are unchanged by the residency: represented inputs (token
IDs), outputs (the gathered `[D,T]` BF16 rows), and the BF16 storage rounding match the device
route.

## Numerical contract

The host residency is a reachable production route of the `embedding` Op: the Op is qualified in the
device suite against the independent FP64 oracles, per the [Op development](op-development.md)
contract. The codec is exact — each value is the decoded code (or dense value) scaled by the exact
stored scale — so the only rounding is the final BF16 storage, which the quantized-embedding
criterion covers. Reading the table over PCIe does not change the values, only where they are
fetched from.

## Implementation pointers

| Concern | Source |
| --- | --- |
| Engine option | [`include/ninfer/types.h`](../../include/ninfer/types.h) (`EngineOptions::embedding_host`) |
| Feature flow | [`startup_features.h`](../../src/targets/qwen3_6/export/ninfer/targets/qwen3_6/startup_features.h) (`StartupFeatures::embedding_host`) |
| Residency binding (27B) | [`qwen3_6_27b/impl/load/bindings.cpp`](../../src/targets/qwen3_6_27b/impl/load/bindings.cpp) (`bind_artifact`) |
| Residency binding (9B) | [`qwen3_5_9b/impl/load/bindings.cpp`](../../src/targets/qwen3_5_9b/impl/load/bindings.cpp) (`bind_artifact`) |
| Residency binding (35B-A3B) | [`qwen3_6_35b_a3b/impl/load/bindings.cpp`](../../src/targets/qwen3_6_35b_a3b/impl/load/bindings.cpp) (`bind_artifact`) |
| Page-locked mapped host allocation | [`src/core/arena.cu`](../../src/core/arena.cu) (`MappedHostBuffer`) |
| Host-mapped materialization | [`src/artifact/materializer.cpp`](../../src/artifact/materializer.cpp) (`materialize`) |
| Memory reporting | [`src/targets/registry.cpp`](../../src/targets/registry.cpp) (`construct_registered`) |
