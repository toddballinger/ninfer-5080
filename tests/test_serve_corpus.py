from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.bench.run_serve_concurrency import (
    PENDING_TIMEOUT_MS,
    STATS_INTERVAL_MS,
    Point,
    validate_server_start,
)
from tools.bench.run_serve_corpus import (
    CampaignError,
    require_server_log_identity,
    summary_row,
)


def test_request_log_v9_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 9,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=8)
    with pytest.raises(CampaignError):
        require_server_log_identity(stale, "server_start")


def test_summary_retains_one_canonical_weights_id() -> None:
    records = [{"weights_id": "nvfp4", "metrics": {}}]
    row = summary_row(
        "context_profile",
        "qwen3_6_27b",
        "fixture",
        "fixture",
        "mtp0",
        "greedy",
        records,
    )
    assert row["weights_id"] == "nvfp4"

    with pytest.raises(CampaignError):
        summary_row(
            "context_profile",
            "qwen3_6_27b",
            "fixture",
            "fixture",
            "mtp0",
            "greedy",
            [*records, {"weights_id": "groupwise-int", "metrics": {}}],
        )


def _concurrency_validate_args() -> argparse.Namespace:
    return argparse.Namespace(
        device=0,
        max_context=131072,
        kv_capacity="auto",
        prefill_chunk=1792,
        kv_dtype="q4",
    )


def _concurrency_validate_event(engine: dict) -> dict:
    return {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 9,
        "event": "server_start",
        "server_instance_id": "instance-1",
        "engine": engine,
        "sampling_defaults": {"greedy": False},
        "artifact": {"target": "qwen3_8_27b", "weights_id": "q4-group64"},
        "server": {"public_model_id": "qwen3.8-27b"},
    }


_MTP3_POINT = Point(
    target="qwen3_8_27b",
    model_id="qwen3.8-27b",
    artifact=Path("placeholder.gguf"),
    speculative_mode="mtp3",
    speculative_backend="mtp",
    draft_tokens=3,
    sampling_mode="stochastic",
    suite="decode-saturation",
    concurrency=2,
)


def _q4_engine_extras() -> dict:
    engine = {
        "device": 0,
        "max_context": 131072,
        "kv_capacity_mode": "auto",
        "max_concurrency": 2,
        "max_pending_requests": 1,
        "pending_timeout_ms": PENDING_TIMEOUT_MS,
        "prefill_chunk": 1792,
        "log_stats_interval_ms": STATS_INTERVAL_MS,
        "kv_cache": "q4-group64",
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": "mtp",
        "speculative_draft_window": 3,
        "proposal_head": "optimized",
    }
    engine["extra_metadata_key"] = 0
    engine["kv_capacity"] = 262144
    return engine


def test_concurrency_server_start_allows_extra_engine_keys() -> None:
    engine = _q4_engine_extras()
    validate_server_start(
        _concurrency_validate_event(engine),
        _MTP3_POINT,
        _concurrency_validate_args(),
    )


def test_concurrency_server_start_rejects_missing_required_field() -> None:
    engine = _q4_engine_extras()
    del engine["prefill_chunk"]
    with pytest.raises(CampaignError):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_wrong_required_field() -> None:
    engine = _q4_engine_extras()
    engine["kv_cache"] = "int8-group64"
    with pytest.raises(CampaignError):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )
    engine = _q4_engine_extras()
    engine["max_context"] = 262144
    with pytest.raises(CampaignError):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )
