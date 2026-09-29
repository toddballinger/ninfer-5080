from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.common.quantize import quantize_and_encode
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common.recipe import SourceTensor
from tools.convert.qwen3_6_27b import recipe as base_recipe
from tools.convert.qwen3_8_27b import inventory
from tools.convert.qwen3_8_27b.convert_lowmem import _encode_chunked_source


def _source_backed_2d_spec():
    for spec in inventory.BASE_TENSOR_SPECS:
        if len(spec.shape) != 2:
            continue

        recipe = base_recipe.RECIPES_BY_NAME.get(spec.name)
        if recipe is None:
            continue

        if isinstance(recipe.expression, SourceTensor):
            return spec, recipe.expression.name

    raise AssertionError("no source-backed rank-2 tensor found")


def test_get_rows_reads_requested_range(tmp_path):
    path = tmp_path / "model.safetensors"
    source = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    save_file({"weight": source}, path)

    with ShardReader.from_file(path) as reader:
        actual = reader.get_rows("weight", 2, 5)

    assert torch.equal(actual, source[2:5])


def test_get_rows_bounds(tmp_path):
    path = tmp_path / "model.safetensors"
    source = torch.zeros((4, 8), dtype=torch.float32)
    save_file({"weight": source}, path)

    with ShardReader.from_file(path) as reader:
        with pytest.raises(ValueError):
            reader.get_rows("weight", -1, 2)

        with pytest.raises(ValueError):
            reader.get_rows("weight", 3, 2)

        with pytest.raises(IndexError):
            reader.get_rows("weight", 0, 5)


def test_chunked_encoding_matches_canonical_without_full_get(tmp_path):
    production_spec, source_name = _source_backed_2d_spec()

    rows = 6
    cols = production_spec.shape[1]

    spec = SimpleNamespace(
        shape=(rows, cols),
        format=production_spec.format,
    )

    generator = torch.Generator().manual_seed(123456)

    source = torch.randn(
        (rows, cols),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    path = tmp_path / "model.safetensors"
    save_file({source_name: source}, path)

    canonical = quantize_and_encode(
        source,
        spec.format,
        device="cpu",
    )

    with ShardReader.from_file(path) as reader:

        def forbidden_get(name: str):
            raise AssertionError(
                f"full ShardReader.get() unexpectedly called for {name}"
            )

        reader.get = forbidden_get  # type: ignore[method-assign]

        chunked = _encode_chunked_source(
            reader,
            spec,
            source_name,
            chunk_rows=2,
            device="cpu",
        )

    streamed = b"".join(bytes(chunk) for chunk in chunked)

    assert streamed == canonical
    assert sum(len(chunk) for chunk in chunked) == len(canonical)
