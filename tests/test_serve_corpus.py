from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.bench import run_serve_corpus as corpus
from tools.bench import run_serve_concurrency as concurrency

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


def test_request_log_v11_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 11,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=9)
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
        "schema_version": 11,
        "event": "server_start",
        "server_instance_id": "instance-1",
        "engine": engine,
        "memory": {name: 0 for name in corpus.MEMORY_METRICS},
        "argv": ["ninfer-serve", "artifact.ninfer", "--embedding-host",
                 "--vision", "--vision-max-tokens", "2048",
                 "--prefix-checkpoint-policy", "rolling-tool"],
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
    event = _concurrency_validate_event(engine)
    event["argv"].remove("--embedding-host")
    with pytest.raises(CampaignError, match="embedding_host"):
        validate_server_start(
            event,
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_vision_max_tokens_mismatch() -> None:
    engine = _q4_engine_extras()
    event = _concurrency_validate_event(engine)
    event["argv"][event["argv"].index("2048")] = "1024"
    with pytest.raises(CampaignError, match="vision_max_tokens"):
        validate_server_start(
            event,
            _MTP3_POINT,
            _concurrency_validate_args(),
        )


def test_concurrency_server_start_rejects_prefix_checkpoint_policy_mismatch() -> None:
    event = _concurrency_validate_event(_q4_engine_extras())
    event["argv"][-1] = "session"
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
    event = _concurrency_validate_event(engine)
    event["argv"][event["argv"].index("2048")] = "1024"
    with pytest.raises(CampaignError, match="vision_max_tokens"):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())


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


@pytest.mark.parametrize("field", ["device", "max_context", "kv_capacity", "max_concurrency",
                                  "max_pending_requests", "pending_timeout_ms", "prefill_chunk",
                                  "log_stats_interval_ms", "speculative_draft_window"])
@pytest.mark.parametrize("bad", [None, True, "1", 1.5, -1, float("nan"), float("inf")])
def test_startup_geometry_rejects_invalid_numbers(field, bad):
    event = _concurrency_validate_event(_q4_engine_extras())
    event["engine"][field] = bad
    with pytest.raises(CampaignError, match=field):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())


@pytest.mark.parametrize("field", corpus.MEMORY_METRICS)
@pytest.mark.parametrize("bad", [None, True, "0", 0.5, -1, float("nan"), float("inf")])
def test_startup_memory_rejects_invalid_numbers(field, bad):
    event = _concurrency_validate_event(_q4_engine_extras())
    event["memory"][field] = bad
    with pytest.raises(CampaignError, match=field):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())
    with pytest.raises(CampaignError, match=field):
        concurrency.summary_row(event)


@pytest.mark.parametrize("bad", [None, {}, [], "memory"])
def test_missing_memory_never_becomes_zero_summary(bad):
    event = _concurrency_validate_event(_q4_engine_extras())
    event["memory"] = bad
    with pytest.raises(CampaignError, match="memory"):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())
    with pytest.raises(CampaignError, match="memory"):
        concurrency.summary_row(event)


def _request_done():
    return {
        "artifact_type": "ninfer_serve_request_log", "schema_version": 11,
        "event": "request_done",
        "request": {"model": _MTP3_POINT.model_id, "requested_output_tokens": 4,
                    "enable_thinking": False, "sampling": {"seed": 1}},
        "result": {"prompt_tokens": 8, "completion_tokens": 4, "computed_prefill_tokens": 8},
        "timings_seconds": {name: 0.25 for name in ("prepare", "ttft", "vision", "prefill", "decode", "total")},
        "speculative": {"backend": "mtp", "draft_window": 3, "rounds": 1,
                        "drafted_tokens": 3, "accepted_tokens": 2, "fallback_steps": 0},
    }


def _build_record(event):
    fixture = corpus.Fixture("fixture", [], False, 4, "context_profile")
    spec = corpus.RunSpec(_MTP3_POINT.target, _MTP3_POINT.model_id, Path("artifact.ninfer"),
                          "mtp3", "mtp", 3, "stochastic", fixture, 1)
    return corpus.build_result_record(spec, "weights", {},
                                     {"usage": {"prompt_tokens": 8, "completion_tokens": 4}}, event)


def test_v11_request_metrics_valid_in_both_harnesses():
    event = _request_done()
    assert _build_record(event)["metrics"]["accepted_tokens"] == 2
    assert concurrency.sum_request_done([event], "mtp", 3)["completion_tokens"] == 4


@pytest.mark.parametrize("section,field", [("result", name) for name in
    ("prompt_tokens", "completion_tokens", "computed_prefill_tokens")] +
    [("speculative", name) for name in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps")])
@pytest.mark.parametrize("bad", [None, True, "2", 2.5, -1, float("nan"), float("inf")])
def test_request_tokens_are_strict(section, field, bad):
    event = _request_done()
    event[section][field] = bad
    for parse in (_build_record, lambda e: concurrency.sum_request_done([e], "mtp", 3)):
        with pytest.raises(CampaignError):
            parse(event)


@pytest.mark.parametrize("field", ["prepare", "ttft", "vision", "prefill", "decode", "total"])
@pytest.mark.parametrize("bad", [None, True, "0.5", -0.5, float("nan"), float("inf")])
def test_request_timings_are_strict(field, bad):
    event = _request_done()
    event["timings_seconds"][field] = bad
    for parse in (_build_record, lambda e: concurrency.sum_request_done([e], "mtp", 3)):
        with pytest.raises(CampaignError):
            parse(event)


@pytest.mark.parametrize("bad", [None, True, "3", 3.0, -1, 2, 4])
def test_request_mtp_window_must_match(bad):
    event = _request_done()
    event["speculative"]["draft_window"] = bad
    for parse in (_build_record, lambda e: concurrency.sum_request_done([e], "mtp", 3)):
        with pytest.raises(CampaignError, match="draft_window"):
            parse(event)


@pytest.mark.parametrize("section,field", [("engine", name) for name in
    ("device", "max_context", "kv_capacity", "max_concurrency", "max_pending_requests",
     "pending_timeout_ms", "prefill_chunk", "log_stats_interval_ms", "speculative_draft_window")]
    + [("memory", name) for name in corpus.MEMORY_METRICS])
def test_startup_missing_consumed_field_rejects(section, field):
    event = _concurrency_validate_event(_q4_engine_extras())
    del event[section][field]
    with pytest.raises(CampaignError, match=field):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())


@pytest.mark.parametrize("section", ["result", "timings_seconds", "speculative"])
def test_request_missing_metrics_rejects(section):
    event = _request_done()
    del event[section]
    for parse in (_build_record, lambda e: concurrency.sum_request_done([e], "mtp", 3)):
        with pytest.raises(CampaignError):
            parse(event)


@pytest.mark.parametrize("flag", ["--embedding-host", "--vision-max-tokens", "--prefix-checkpoint-policy"])
def test_v11_missing_semantic_argv_rejects(flag):
    event = _concurrency_validate_event(_q4_engine_extras())
    event["argv"].remove(flag)
    with pytest.raises(CampaignError):
        validate_server_start(event, _MTP3_POINT, _concurrency_validate_args())


def test_summary_preserves_real_memory_and_valid_zero():
    report = _concurrency_validate_event(_q4_engine_extras())
    report.update(suite="corpus-makespan", target="target", weights_id="weights",
                  speculative_mode="mtp3", sampling_mode="stochastic", concurrency=1,
                  request_count=1, totals={"prompt_tokens": 8, "computed_prefill_tokens": 8, "decode_tokens": 3},
                  decode_batch={"average_size": 1},
                  metrics={"request_latency_seconds": {"mean": 1, "p50": 1, "p95": 1, "max": 1},
                           "makespan_seconds": 1, "requests_per_second": 1,
                           "computed_prefill_tokens_per_second": 8, "decode_tokens_per_second": 3})
    report["memory"]["kv_payload_bytes"] = 2424377344
    row = concurrency.summary_row(report)
    assert row["kv_payload_mib"] == 2424377344 / 1024**2
    assert row["planned_slack_mib"] == 0


@pytest.mark.parametrize("bad", [None, True, "4", 4.5, -1])
def test_http_token_counts_reject_coercion(bad):
    response = {"usage": {"prompt_tokens": 8, "completion_tokens": bad},
                "choices": [{"finish_reason": "length"}]}
    with pytest.raises(CampaignError):
        concurrency.parse_client_response(None, response, 0, 1)


def _throughput():
    return {"artifact_type": "ninfer_serve_request_log", "schema_version": 11,
            "event": "throughput", "interval_seconds": 1,
            "tokens": {"computed_prefill": 0, "committed_decode": 4},
            "decode_batch": {"rounds": 4, "row_rounds": 4},
            "scheduler": {"running": 1, "prefilling": 0, "decode_ready": 1}}


@pytest.mark.parametrize("bad", [None, True, "4", 4.5, -1])
def test_throughput_tokens_reject_coercion(bad):
    event = _throughput()
    event["tokens"]["committed_decode"] = bad
    with pytest.raises(CampaignError):
        concurrency.sum_throughput([event])


@pytest.mark.parametrize("bad", [None, True, "1", -1, float("nan"), float("inf")])
def test_throughput_timing_rejects_coercion(bad):
    event = _throughput()
    event["interval_seconds"] = bad
    with pytest.raises(CampaignError):
        concurrency.sum_throughput([event])


def test_valid_steady_throughput():
    assert concurrency.steady_metrics([_throughput()], 1)["decode_tokens_per_second"] == 4
