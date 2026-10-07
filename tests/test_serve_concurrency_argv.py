"""Focused regression coverage for the Issue #32 production-equivalent C1 harness argv.

These tests prove that the benchmark harness, in its default (production-equivalent) profile,
generates a serve-launch argv that reproduces the production serving unit's memory-relevant
semantics — the fix that makes the C1 GPU baseline a valid production-capacity baseline.
They exercise the argv construction (``server_command``) and the generated-vs-production
difference classification without launching a GPU or a server. No request-payload, thinking,
workload, sampling, or corpus semantics are touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.bench import run_serve_concurrency as bench
from tools.bench.run_serve_corpus import (
    CampaignError,
    TARGET_MODEL_IDS,
)

P = bench.Point


def _production_c1_args() -> argparse.Namespace:
    """A Namespace with exactly the production-equivalent C1 values (the Issue #32 baseline)."""
    return argparse.Namespace(
        port=8080,
        device=0,
        max_context=131072,
        kv_capacity="131072",
        kv_dtype="q4",
        prefill_chunk=1792,
        max_pending_requests=16,
        pending_timeout_ms=180000,
        vision_max_tokens=2048,
        no_vision=False,
        no_embedding_host=False,
        default_thinking_budget=2048,
        decode_tokens=8192,
        dry_run=False,
    )


def _c1_mtp3_point() -> bench.Point:
    return P(
        target="qwen3_8_27b",
        model_id=TARGET_MODEL_IDS["qwen3_8_27b"],
        artifact=Path("/models/ninfer-custom/placeholder.ninfer"),
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="stochastic",
        suite="decode-saturation",
        concurrency=1,
    )


def _c1_command() -> list[str]:
    # Direct coverage: the generated argv is the harness's real server_command on a C1 mtp3
    # point with the production-equivalent argument profile.
    return bench.server_command(
        Path("ninfer-serve"), _c1_mtp3_point(), Path("/tmp/ninfer-serve-request-log.jsonl"),
        _production_c1_args(),
    )


def _argv_contains(argv: list[str], flag: str, value: str | None = None) -> bool:
    if flag not in argv:
        return False
    if value is None:
        return True
    index = argv.index(flag)
    return index + 1 < len(argv) and argv[index + 1] == value


def test_c1_argv_emits_embedding_host() -> None:
    assert _argv_contains(_c1_command(), "--embedding-host")


def test_c1_argv_uses_full_proposal_head() -> None:
    # Production does not pass --lm-head-draft; the full LM-head path is the production default.
    assert "--lm-head-draft" not in _c1_command()


def test_c1_argv_enables_vision_2048() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--vision")
    assert _argv_contains(argv, "--vision-max-tokens", "2048")


def test_c1_argv_uses_q4_kv_131072() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--kv-dtype", "q4")
    assert _argv_contains(argv, "--kv-capacity", "131072")


def test_c1_argv_uses_mtp3_spec() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--spec", "mtp")
    assert _argv_contains(argv, "--draft-tokens", "3")


def test_c1_argv_keeps_cuda_graph_enabled() -> None:
    # CUDA Graph stays on (no --no-cuda-graph) to match production memory semantics.
    assert "--no-cuda-graph" not in _c1_command()


def test_c1_argv_uses_production_queue_pending_settings() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--max-pending-requests", "16")
    assert _argv_contains(argv, "--pending-timeout-ms", "180000")


def test_c1_argv_uses_max_context_and_prefill() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--max-context", "131072")
    assert _argv_contains(argv, "--prefill-chunk", "1792")


def test_c1_argv_uses_thinking_budget_and_rolling_tool() -> None:
    argv = _c1_command()
    assert _argv_contains(argv, "--default-thinking-budget", "2048")
    assert _argv_contains(argv, "--prefix-checkpoint-policy", "rolling-tool")


def test_c1_argv_omits_device_when_default() -> None:
    # Production omits --device (defaults to 0); the harness must not add it at index 0 so the
    # generated argv stays equivalent to the production C1 reference.
    assert "--device" not in _c1_command()


def test_c1_argv_keeps_prefix_reuse_on() -> None:
    # Production runs with default prefix-reuse (no --no-prefix-reuse).
    assert "--no-prefix-reuse" not in _c1_command()


def test_production_reference_profile_is_self_consistent() -> None:
    # The embedded production C1 reference encodes the memory-relevant flags the audit flagged.
    reference_flags = {flag: value for flag, value in bench.PRODUCTION_C1_FLAGS}
    assert reference_flags.get("--embedding-host") is None
    assert "--lm-head-draft" not in reference_flags
    assert "--no-cuda-graph" not in reference_flags
    assert reference_flags["--kv-dtype"] == "q4"
    assert reference_flags["--kv-capacity"] == "131072"
    assert reference_flags["--max-context"] == "131072"
    assert reference_flags["--max-pending-requests"] == "16"
    assert reference_flags["--pending-timeout-ms"] == "180000"
    assert reference_flags["--prefill-chunk"] == "1792"
    assert reference_flags["--spec"] == "mtp"
    assert reference_flags["--draft-tokens"] == "3"
    assert reference_flags["--vision-max-tokens"] == "2048"
    assert reference_flags["--default-thinking-budget"] == "2048"
    assert reference_flags["--prefix-checkpoint-policy"] == "rolling-tool"


def test_c1_reference_has_zero_material_difference_to_production() -> None:
    argv = _c1_command()
    rows = bench.production_difference_report(bench.PRODUCTION_C1_FLAGS, argv)
    material = [row for row in rows if row["classification"] == "material-unresolved"]
    assert not material, f"unresolved material mismatch(es): {material!r}"
    for row in rows:
        assert row["classification"] in ("benign", "instrumentation-only"), row


def test_drifted_profile_flags_material_difference() -> None:
    # A benchmark that drifts off the production profile (device embeddings / no vision) must be
    # classified as a material, unresolved difference — the audit's NOT_EQUIVALENT finding.
    base = _production_c1_args().__dict__ | {"no_vision": True, "no_embedding_host": True}
    drifted_argv = bench.server_command(
        Path("ninfer-serve"), _c1_mtp3_point(), Path("/tmp/x.jsonl"), argparse.Namespace(**base)
    )
    rows = bench.production_difference_report(bench.PRODUCTION_C1_FLAGS, drifted_argv)
    classifications = {row["flag"]: row["classification"] for row in rows}
    # Missing --vision and --embedding-host (production has them, the drifted argv does not)
    # must be classified as material-unresolved, never benign or instrumentation-only.
    assert classifications["--vision"] == "material-unresolved"
    assert classifications["--vision-max-tokens"] == "material-unresolved"
    assert classifications["--embedding-host"] == "material-unresolved"


def test_validate_args_accepts_production_c1_profile() -> None:
    args = _production_c1_args()
    args.concurrency = [1]
    args.suite = ["decode-saturation"]
    bench.validate_args(args)  # must not raise


def test_validate_args_rejects_invalid_queue_settings() -> None:
    args = _production_c1_args()
    args.concurrency = [1]
    args.suite = ["decode-saturation"]
    args.max_pending_requests = 0
    with pytest.raises(CampaignError):
        bench.validate_args(args)