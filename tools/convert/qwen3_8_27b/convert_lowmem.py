"""Low-memory Qwen3.8-27B converter (wrapper, byte-identical output).

Same output as `tools.convert.qwen3_8_27b.convert` (SHA-identical artifact),
but processes the two huge pure-source matrices (token_embedding and
output_head, both 248320x5120 = 1.27B elements) in row chunks. The full
tensor never materializes, so peak memory drops from ~26 GB to ~6 GB and the
conversion fits on a 15 GB GitHub-hosted runner.

Chunked quantization is byte-identical to whole-matrix quantization because
row-split-k128-v1 quantization is per-row independent (each row computes its
own group scales/codes); the packed base/high/scale planes concatenate
exactly. Verified: chunked SHA == whole-matrix SHA.

Usage:
    python -m tools.convert.qwen3_8_27b.convert_lowmem \
        --model <dir> --dflash2-model <dir> --out out/qwen3_8_27b.ninfer \
        [--device cpu] [--chunk-rows 16384] [--chunk-threshold-elems 300000000]
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import (
    RowPlanes,
    assemble_row_planes,
    row_split_geometry,
    split_row_planes,
)
from tools.convert.common.quantize import pick_device, quantize_and_encode
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.recipe import SourceTensor
from tools.convert.qwen3_6_27b import draft_head, recipe as base_recipe

from . import dflash2_recipe, inventory
from .convert import _repo_root, build_conversion_report, encode_tensor_payload
from .dflash2_inventory import DFLASH2_TENSOR_SPECS

# Fallback to direct conversion main pieces we don't need to reimplement.
# preflight_conversion is imported from the canonical module so the object
# plan / directory / report stay identical.
from .convert import preflight_conversion


def _is_chunkable(spec, threshold_elems: int) -> bool:
    """True if this tensor should be row-chunked (pure source, huge, rank 2)."""
    if len(spec.shape) != 2:
        return False
    if spec.shape[0] * spec.shape[1] < threshold_elems:
        return False
    expr = base_recipe.RECIPES_BY_NAME[spec.name].expression
    return isinstance(expr, SourceTensor)


def _encode_chunked_source(
    reader: ShardReader,
    spec,
    src_name: str,
    chunk_rows: int,
    device: str | torch.device = "cpu",
) -> bytes:
    """Row-chunked quantize+encode for a pure source 2D tensor.

    Reads row slices directly from the shard (no full materialize), quantizes
    each chunk independently, and stitches the packed planes back into one
    payload using the whole-matrix geometry.
    """
    n, k = spec.shape
    geom = row_split_geometry(spec.format, (n, k))
    base_acc, high_acc, scale_acc = bytearray(), bytearray(), bytearray()

    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        src = reader.get(src_name)
        chunk = src[lo:hi].contiguous()
        del src
        payload = quantize_and_encode(chunk, spec.format, device=device)
        cgeom = row_split_geometry(spec.format, (hi - lo, k))
        planes = split_row_planes(payload, cgeom)
        base_acc += bytes(planes.base)
        high_acc += bytes(planes.high)
        scale_acc += bytes(planes.scale)
        del chunk, payload, planes
        gc.collect()

    stitched = assemble_row_planes(
        RowPlanes(bytes(base_acc), bytes(high_acc), bytes(scale_acc), n),
        spec.format,
        k,
    )
    return stitched if isinstance(stitched, bytes) else bytes(stitched)


def convert(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    chunk_rows: int = 16384,
    chunk_threshold_elems: int = 300_000_000,
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, dflash2_model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.base_source.source_tensor_count} base and "
        f"{preflight.dflash2_source.source_tensor_count} DFlash2 source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(inventory.OBJECT_SPECS)
    index = 0

    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer object plan differs from completed preflight")

        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)

        with ShardReader(model) as base_reader:
            for spec in inventory.BASE_TENSOR_SPECS:
                index += 1
                chunked = _is_chunkable(spec, chunk_threshold_elems)
                if chunked:
                    src_name = base_recipe.RECIPES_BY_NAME[
                        spec.name
                    ].expression.name
                    payload = _encode_chunked_source(
                        base_reader, spec, src_name, chunk_rows, resolved_device
                    )
                else:
                    tensor = materialize_tensor(spec, base_reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                gc.collect()
                print(
                    f"[{index}/{total}] {spec.name}"
                    + (" [chunked]" if chunked else ""),
                    flush=True,
                )

        with ShardReader.from_file(
            preflight.dflash2_model_dir / "model.safetensors"
        ) as dflash2_reader:
            for spec in DFLASH2_TENSOR_SPECS:
                index += 1
                tensor = dflash2_recipe.materialize_tensor(
                    spec.name, dflash2_reader
                )
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "dflash2_model": str(dflash2_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        model_dir=model,
        dflash2_model_dir=preflight.dflash2_model_dir,
        out_path=output,
        arguments=arguments,
        base_config_summary=preflight.base_config_summary,
        dflash2_config_summary=preflight.dflash2_config_summary,
        base_source_preflight=preflight.base_source,
        dflash2_source_preflight=preflight.dflash2_source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    import json

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def materialize_tensor(spec, reader, draft):
    """Delegate to the canonical materializer (non-chunked path)."""
    from tools.convert.qwen3_8_27b.convert import materialize_tensor as _mt

    return _mt(spec, reader, draft)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=16384)
    parser.add_argument(
        "--chunk-threshold-elems", type=int, default=300_000_000
    )
    args = parser.parse_args(argv)
    convert(
        args.model,
        args.dflash2_model,
        args.out,
        device=args.device,
        chunk_rows=args.chunk_rows,
        chunk_threshold_elems=args.chunk_threshold_elems,
    )


if __name__ == "__main__":
    main()
