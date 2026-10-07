from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.bench.run_serve_concurrency import (
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
        kv_capacity="131072",
        prefill_chunk=1792,
        kv_dtype="q4",
        max_pending_requests=16,
        pending_timeout_ms=180000,
        no_vision=False,
        vision_max_tokens=2048,
        no_embedding_host=False,
        default_thinking_budget=2048,
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
        "server": {"public_model_id": "qwen3.8-27b", "default_thinking_budget": 2048, "prefix_checkpoint_policy": "rolling-tool"},
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
    concurrency=1,
)


def _q4_engine_extras() -> dict:
    engine = {
        "device": 0,
        "max_context": 131072,
        "kv_capacity_mode": "explicit",
        "max_concurrency": 1,
        "max_pending_requests": 16,
        "pending_timeout_ms": 180000,
        "prefill_chunk": 1792,
        "log_stats_interval_ms": STATS_INTERVAL_MS,
        "kv_cache": "q4-group64",
        "vision": True,
        "cuda_graph": True,
        "prefix_reuse": True,
        "speculative_backend": "mtp",
        "speculative_draft_window": 3,
        "proposal_head": "full",
    }
    engine["extra_metadata_key"] = 0
    engine["kv_capacity"] = 131072
    engine["vision_max_tokens"] = 2048
    engine["embedding_host"] = True
    return engine


def test_concurrency_server_start_rejects_wrong_numeric_kv_capacity() -> None:
    # kv_capacity_mode="explicit" alone did not discriminate the numeric capacity; the gate now
    # enforces engine.kv_capacity against the requested value (Issue #32 commit-gate finding).
    engine = _q4_engine_extras()
    engine["kv_capacity"] = 16384
    with pytest.raises(CampaignError, match="kv_capacity"):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_missing_kv_capacity() -> None:
    engine = _q4_engine_extras()
    del engine["kv_capacity"]
    with pytest.raises(CampaignError, match="kv_capacity"):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_auto_kv_capacity() -> None:
    # An explicit 131072 request must not accept a serve that left capacity at auto.
    engine = _q4_engine_extras()
    engine["kv_capacity"] = "auto"
    with pytest.raises(CampaignError, match="kv_capacity"):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_embedding_host_mismatch() -> None:
    engine = _q4_engine_extras()
    engine["embedding_host"] = False
    with pytest.raises(CampaignError, match="embedding_host"):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_vision_max_tokens_mismatch() -> None:
    engine = _q4_engine_extras()
    engine["vision_max_tokens"] = 1024
    with pytest.raises(CampaignError, match="vision_max_tokens"):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_prefix_checkpoint_policy_mismatch() -> None:
    event = _concurrency_validate_event(_q4_engine_extras())
    event["server"]["prefix_checkpoint_policy"] = "session"
    with pytest.raises(CampaignError, match="prefix_checkpoint_policy"):
        validate_server_start(
            event,
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_accepts_matching_profile() -> None:
    # All material startup fields in agreement: a matching serve must be accepted.
    engine = _q4_engine_extras()
    event = _concurrency_validate_event(engine)
    event["server"]["prefix_checkpoint_policy"] = "rolling-tool"
    validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())


def test_concurrency_server_start_vision_max_tokens_enforced_when_vision_on() -> None:
    # Drifting only the Vision budget (serve reports 1024, the production 2048 is requested)
    # must be rejected while Vision itself remains on.
    engine = _q4_engine_extras()
    engine["vision_max_tokens"] = 1024
    with pytest.raises(CampaignError, match="vision_max_tokens"):
        validate_server_start(_concurrency_validate_event(engine), _MTP3_POINT, _concurrency_validate_args())


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
    engine["max_context"] = 131073
    with pytest.raises(CampaignError):
        validate_server_start(
            _concurrency_validate_event(engine),
            _MTP3_POINT,
            _concurrency_validate_args(),
        )
