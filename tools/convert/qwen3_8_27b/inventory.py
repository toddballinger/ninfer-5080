"""Persistent-object contract for the complete Qwen3.8-27B artifact.

The graph and all non-vocabulary storage roles are identical to the registered
Qwen3.6-27B groupwise artifact.  The embedding and full output head use the W8
format already supported by the 27B runtime.
"""

from __future__ import annotations

from tools.convert.qwen3_6_27b import inventory as qwen3_6_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_8_27b"

BF16 = qwen3_6_inventory.BF16
FP32 = qwen3_6_inventory.FP32
I32 = qwen3_6_inventory.I32
Q3 = qwen3_6_inventory.Q3
Q4 = qwen3_6_inventory.Q4
Q5 = qwen3_6_inventory.Q5
Q6 = qwen3_6_inventory.Q6
W8 = qwen3_6_inventory.W8

FORMAT_NAMES = qwen3_6_inventory.FORMAT_NAMES
LAYOUT_NAMES = qwen3_6_inventory.LAYOUT_NAMES
ResourceSpec = qwen3_6_inventory.ResourceSpec
StoredObjectSpec = qwen3_6_inventory.StoredObjectSpec
TensorSpec = qwen3_6_inventory.TensorSpec

FULL_ATTENTION_LAYERS = qwen3_6_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = qwen3_6_inventory.GDN_LAYERS
RESOURCE_SPECS = qwen3_6_inventory.RESOURCE_SPECS


def _w8_vocabulary_endpoint(spec: TensorSpec) -> TensorSpec:
    # RTX 5080 16 GB profile v3.
    if spec.name == "text/token_embedding":
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q5)

    if spec.name == "text/output_head":
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)

    # Keep fused GDN value/z on Q5.
    # Quantize the GDN output projection to Q4 instead; runtime uses
    # ordinary Q4 linear followed by residual_add.
    if spec.name.startswith("text/layers/") and spec.name.endswith("/gdn/output"):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)

    # Full-attention output projection. Qwen3.8 uses the ordinary Q4
    # linear + residual_add path; Qwen3.6 remains on fused Q5 linear_add.
    if spec.name.startswith("text/layers/") and spec.name.endswith("/attention/output"):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)

    # Largest tensor family in the model. Q3 is introduced narrowly here
    # so the rest of the existing Q4/Q5 kernel contracts remain unchanged.
    if spec.name.startswith("text/layers/") and spec.name.endswith("/mlp/gate_up"):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q3)

    if spec.name.startswith("text/layers/") and spec.name.endswith("/mlp/down"):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)

    return spec


TEXT_CORE_TENSOR_SPECS = tuple(
    _w8_vocabulary_endpoint(spec)
    for spec in qwen3_6_inventory.TEXT_CORE_TENSOR_SPECS
)
DRAFT_HEAD_TENSOR_SPECS = qwen3_6_inventory.DRAFT_HEAD_TENSOR_SPECS

def _q4_mtp(spec: TensorSpec) -> TensorSpec:
    if spec.name.startswith("mtp/") and spec.format == W8:
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)
    return spec

MTP_TENSOR_SPECS = tuple(
    _q4_mtp(spec)
    for spec in qwen3_6_inventory.MTP_TENSOR_SPECS
)
VISION_TENSOR_SPECS = qwen3_6_inventory.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = qwen3_6_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = qwen3_6_inventory.ALIAS_SPECS
