#!/usr/bin/env python3
"""Run the fixed serving corpus performance evaluation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import http.client
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "examples/cli/manifest.json"

TARGET_MODEL_IDS = {
    "qwen3_6_35b_a3b": "qwen3.6-35b-a3b",
    "qwen3_6_27b": "qwen3.6-27b",
    "qwen3_8_27b": "qwen3.8-27b",
}
TARGET_ORDER = tuple(TARGET_MODEL_IDS)
SPECULATIVE_MODES = {
    "mtp0": ("none", 0),
    "mtp3": ("mtp", 3),
    "dflash7": ("dflash", 7),
}
DEFAULT_MODES = ("mtp0", "mtp3")
SAMPLING_MODES = ("stochastic", "greedy")

SEEDS = (
    7632647173703958409,
    7968175640111700217,
    912910298659544128,
    9060622443728853932,
    4939353812939007330,
)

NIAH_FIXTURES = (
    "long_niah_8k",
    "long_niah_64k",
    "long_niah_128k",
    "long_niah_256k",
)

LONG_DECODE_FIXTURES = (
    "long_decode_aime26_01",
    "long_decode_aime26_15",
    "long_decode_aime26_30",
)

SCENARIO_FIXTURES = {
    "code": (
        "scenario_code_cuda",
        "scenario_code_python",
        "scenario_code_typescript",
    ),
    "story": (
        "scenario_story_zh_scifi",
        "scenario_story_en_mystery",
        "scenario_story_zh_dialogue",
    ),
    "translation": (
        "scenario_translation_zh_en",
        "scenario_translation_en_zh",
        "scenario_translation_markdown",
    ),
    "structured": (
        "scenario_structured_jsonl",
        "scenario_structured_csv",
        "scenario_structured_sql",
    ),
}

WARMUP_FIXTURE = "text_smoke_zh"
RUN_ARTIFACT_TYPE = "ninfer_serve_corpus_result"
RUN_SCHEMA_VERSION = 5
SERVER_LOG_ARTIFACT_TYPE = "ninfer_serve_request_log"
SERVER_LOG_SCHEMA_VERSION = 11
STARTUP_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
LOG_EVENT_TIMEOUT_SECONDS = 10.0


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    messages: list[dict[str, Any]]
    thinking: bool
    max_new: int
    suite: str
    category: str | None = None


@dataclasses.dataclass(frozen=True)
class RunSpec:
    target: str
    model_id: str
    artifact: Path
    speculative_mode: str
    speculative_backend: str
    draft_tokens: int
    sampling_mode: str
    fixture: Fixture
    seed: int

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.target,
            self.speculative_mode,
            self.sampling_mode,
            self.fixture.name,
            self.seed,
        )


class CampaignError(RuntimeError):
    pass


class ServerLogTail:
    def __init__(
        self, path: Path, process: subprocess.Popen[bytes], initial_offset: int
    ) -> None:
        self.path = path
        self.process = process
        self.offset = initial_offset
        self.buffer = b""
        self.pending: list[dict[str, Any]] = []

    def _check_process(self) -> None:
        returncode = self.process.poll()
        if returncode is not None:
            raise CampaignError(f"ninfer-serve exited unexpectedly with status {returncode}")

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return
        self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        for raw_line in lines:
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignError(f"invalid serving JSONL record in {self.path}: {exc}") from exc
            if not isinstance(event, dict):
                raise CampaignError(f"serving JSONL record is not an object in {self.path}")
            self.pending.append(event)

    def wait_for(
        self,
        predicate: Any,
        description: str,
        timeout: float = LOG_EVENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            self._read_new()
            for index, event in enumerate(self.pending):
                if predicate(event):
                    return self.pending.pop(index)
            self._check_process()
            if time.monotonic() >= deadline:
                raise CampaignError(f"timed out waiting for {description} in {self.path}")
            time.sleep(0.05)


class RunningServer:
    def __init__(
        self,
        command: Sequence[str],
        host: str,
        port: int,
        log_path: Path,
        *,
        capture_paths: tuple[Path, Path] | None = None,
        capture_drain_timeout: float = 1.0,
    ) -> None:
        self.command = list(command)
        self.host = host
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.tail: ServerLogTail | None = None
        if not math.isfinite(capture_drain_timeout) or capture_drain_timeout < 0:
            raise ValueError("capture_drain_timeout must be finite and nonnegative")
        self.capture_paths = capture_paths
        self.capture_drain_timeout = capture_drain_timeout
        self.captures: list[StreamCapture] = []
        self.capture_status: list[dict[str, object]] = []

    def __enter__(self) -> "RunningServer":
        initial_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        if self.capture_paths is None:
            self.process = subprocess.Popen(self.command, cwd=REPO_ROOT)
        else:
            from tools.bench.stream_capture import StreamCapture

            self.process = subprocess.Popen(self.command, cwd=REPO_ROOT,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for source, path, console in zip(
                    (self.process.stdout, self.process.stderr), self.capture_paths,
                    (sys.stdout, sys.stderr),
                ):
                    assert source is not None
                    try:
                        console_fd = console.fileno()
                    except (AttributeError, OSError, ValueError):
                        console_fd = None
                    self.captures.append(StreamCapture(source, path, console_fd))
            except BaseException as primary:
                try:
                    self.stop()
                except BaseException as cleanup:
                    primary.add_note(f"server cleanup failed: {cleanup!r}")
                finally:
                    self.capture_status.append({
                        "capture_complete": False, "capture_failed": True,
                        "capture_stop_reason": "setup_error", "archived_bytes": 0,
                        "capture_error": repr(primary),
                    })
                    for source in (self.process.stdout, self.process.stderr):
                        if source is not None and not source.closed:
                            try:
                                source.close()
                            except BaseException as cleanup:
                                primary.add_note(f"pipe cleanup failed: {cleanup!r}")
                raise
        self.tail = ServerLogTail(self.log_path, self.process, initial_offset)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.stop()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"server cleanup failed: {cleanup!r}")

    def stop(self) -> None:
        primary: BaseException | None = None
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        except BaseException as error:
            primary = error
        finally:
            self.capture_status = []
            for capture in self.captures:
                try:
                    status = capture.finish(self.capture_drain_timeout)
                    if status["capture_failed"]:
                        raise CampaignError(f"stream archival failed: {status['capture_error']}")
                except BaseException as cleanup:
                    if primary is None:
                        primary = cleanup
                    else:
                        primary.add_note(f"capture cleanup failed: {cleanup!r}")
                finally:
                    self.capture_status.append(capture.status())
        if primary is not None:
            raise primary

    def wait_until_ready(self) -> dict[str, Any]:
        if self.process is None or self.tail is None:
            raise CampaignError("server process was not started")
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            returncode = self.process.poll()
            if returncode is not None:
                raise CampaignError(f"ninfer-serve exited during startup with status {returncode}")
            connection = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
            try:
                connection.request("GET", "/health", headers={"Connection": "close"})
                response = connection.getresponse()
                body = response.read()
                if response.status == 200:
                    try:
                        health = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        health = None
                    if health == {"status": "ok"}:
                        break
            except OSError:
                pass
            finally:
                connection.close()
            if time.monotonic() >= deadline:
                raise CampaignError(
                    f"timed out waiting for ninfer-serve at http://{self.host}:{self.port}"
                )
            time.sleep(0.2)

        return self.tail.wait_for(
            lambda event: event.get("event") == "server_start",
            "server_start event",
        )

    def wait_for_request_done(self, server_instance_id: str) -> dict[str, Any]:
        if self.tail is None:
            raise CampaignError("server log tail is unavailable")

        def matches(event: dict[str, Any]) -> bool:
            if event.get("server_instance_id") != server_instance_id:
                return False
            if event.get("event") == "request_error":
                message = event.get("error", {}).get("message", "unknown generation error")
                raise CampaignError(f"serving request failed: {message}")
            return event.get("event") == "request_done"

        return self.tail.wait_for(matches, "request_done event")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve",
        type=Path,
        default=REPO_ROOT / "build/apps/ninfer-serve",
        help="ninfer-serve executable",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="TARGET=PATH",
        help="artifact for a registered target; repeat to benchmark multiple targets",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=tuple(SPECULATIVE_MODES),
        help="benchmark only this mode; repeat to select multiple (default: mtp0 and mtp3)",
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_MODES,
        default="stochastic",
        help="sampling profile for all requests (default: stochastic)",
    )
    parser.add_argument("--output", type=Path, required=True, help="campaign output directory")
    parser.add_argument("--port", type=int, default=8080, help="loopback serving port")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser.parse_args(argv)


def parse_artifacts(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: dict[str, Path] = {}
    for value in values:
        target, separator, raw_path = value.partition("=")
        if not separator or not target or not raw_path:
            raise CampaignError(f"invalid --artifact value {value!r}; expected TARGET=PATH")
        if target not in TARGET_MODEL_IDS:
            expected = ", ".join(TARGET_MODEL_IDS)
            raise CampaignError(f"unsupported artifact target {target!r}; expected {expected}")
        if target in parsed:
            raise CampaignError(f"duplicate artifact target: {target}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise CampaignError(f"artifact not found: {path}")
        parsed[target] = path
    return [(target, parsed[target]) for target in TARGET_ORDER if target in parsed]


def fixture_metadata(name: str) -> tuple[str, str | None]:
    if name in NIAH_FIXTURES:
        return "long_niah", None
    if name in LONG_DECODE_FIXTURES:
        return "long_decode", "reasoning"
    for category, names in SCENARIO_FIXTURES.items():
        if name in names:
            return "scenario", category
    if name == WARMUP_FIXTURE:
        return "warmup", None
    raise CampaignError(f"fixture has no campaign assignment: {name}")


def load_fixtures() -> dict[str, Fixture]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"failed to read {MANIFEST_PATH}: {exc}") from exc
    cases = {case["name"]: case for case in manifest["cases"]}
    selected_names = (
        *NIAH_FIXTURES,
        *LONG_DECODE_FIXTURES,
        *(name for names in SCENARIO_FIXTURES.values() for name in names),
        WARMUP_FIXTURE,
    )
    fixtures: dict[str, Fixture] = {}
    for name in selected_names:
        try:
            case = cases[name]
            messages_path = MANIFEST_PATH.parent / case["messages"]
            messages = json.loads(messages_path.read_text(encoding="utf-8"))
            suite, category = fixture_metadata(name)
            fixtures[name] = Fixture(
                name=name,
                messages=messages,
                thinking=bool(case["thinking"]),
                max_new=int(case["max_new"]),
                suite=suite,
                category=category,
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CampaignError(f"failed to load fixture {name!r} from the examples manifest: {exc}") from exc
    return fixtures


def block_fixture_names(speculative_backend: str) -> tuple[str, ...]:
    scenarios = tuple(name for names in SCENARIO_FIXTURES.values() for name in names)
    if speculative_backend == "none":
        return NIAH_FIXTURES
    if speculative_backend in {"mtp", "dflash"}:
        return (*LONG_DECODE_FIXTURES, *scenarios)
    raise CampaignError(f"unsupported speculative backend: {speculative_backend}")


def build_specs(
    artifacts: Sequence[tuple[str, Path]],
    fixtures: dict[str, Fixture],
    mode_names: Sequence[str],
    sampling_mode: str,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for target, artifact in artifacts:
        for mode_name in mode_names:
            backend, draft_tokens = SPECULATIVE_MODES[mode_name]
            if backend == "dflash" and target != "qwen3_6_35b_a3b":
                raise CampaignError("DFlash corpus measurements require the 35B-A3B target")
            for fixture_name in block_fixture_names(backend):
                for seed in SEEDS:
                    specs.append(
                        RunSpec(
                            target=target,
                            model_id=TARGET_MODEL_IDS[target],
                            artifact=artifact,
                            speculative_mode=mode_name,
                            speculative_backend=backend,
                            draft_tokens=draft_tokens,
                            sampling_mode=sampling_mode,
                            fixture=fixtures[fixture_name],
                            seed=seed,
                        )
                    )
    return specs


def request_payload(model_id: str, fixture: Fixture, seed: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": fixture.messages,
        "max_completion_tokens": fixture.max_new,
        "seed": seed,
        "stream": False,
        "enable_thinking": fixture.thinking,
    }


def send_json(connection: http.client.HTTPConnection, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Connection": "keep-alive",
            },
        )
    except (OSError, http.client.HTTPException) as exc:
        raise CampaignError(f"HTTP request failed: {exc}") from exc


def receive_json(connection: http.client.HTTPConnection) -> dict[str, Any]:
    try:
        response = connection.getresponse()
        response_body = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise CampaignError(f"HTTP request failed: {exc}") from exc
    if response.status != 200:
        detail = response_body.decode("utf-8", errors="replace")
        raise CampaignError(f"HTTP {response.status} {response.reason}: {detail}")
    try:
        parsed = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"serving response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CampaignError("serving response is not a JSON object")
    return parsed


def post_json(connection: http.client.HTTPConnection, payload: dict[str, Any]) -> dict[str, Any]:
    send_json(connection, payload)
    return receive_json(connection)


def require_integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{name} must be an integer >= {minimum}: {value!r}")
    return value


def require_seconds(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise CampaignError(f"{name} must be finite nonnegative seconds: {value!r}")
    return float(value)


def require_fields(actual: Any, expected: dict[str, Any], name: str) -> None:
    if not isinstance(actual, dict):
        raise CampaignError(f"{name} must be an object")
    for key, value in expected.items():
        found = actual.get(key)
        if type(found) is not type(value) or found != value:
            raise CampaignError(f"{name}.{key} mismatch: {found!r}; expected {value!r}")


MEMORY_METRICS = (
    "kv_payload_bytes", "runtime_reservation_bytes", "planned_slack_bytes",
    "cuda_graph_allowance_bytes", "cuda_graph_observed_bytes",
)


def validate_startup_metrics(event: dict[str, Any]) -> None:
    engine = event.get("engine")
    if not isinstance(engine, dict):
        raise CampaignError("server_start.engine must be an object")
    for name, minimum in (("device", 0), ("max_context", 1), ("kv_capacity", 1),
                          ("prefill_chunk", 1), ("speculative_draft_window", 0)):
        require_integer(engine.get(name), f"engine.{name}", minimum)
    for name, minimum in (("max_concurrency", 1), ("max_pending_requests", 0),
                          ("pending_timeout_ms", 0), ("log_stats_interval_ms", 0)):
        require_integer(engine.get(name), f"engine.{name}", minimum)
    memory = event.get("memory")
    if not isinstance(memory, dict):
        raise CampaignError("server_start.memory must be an object")
    for name in MEMORY_METRICS:
        require_integer(memory.get(name), f"memory.{name}")


def validate_request_metrics(event: dict[str, Any], backend: str, window: int) -> None:
    require_server_log_identity(event, "request_done")
    try:
        for name in ("prompt_tokens", "completion_tokens", "computed_prefill_tokens"):
            require_integer(event["result"][name], f"result.{name}")
        for name in ("prepare", "ttft", "vision", "prefill", "decode", "total"):
            require_seconds(event["timings_seconds"][name], f"timings_seconds.{name}")
        speculative = event["speculative"]
        require_fields(speculative, {"backend": backend, "draft_window": window}, "speculative")
        for name in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps"):
            require_integer(speculative[name], f"speculative.{name}")
        if speculative["accepted_tokens"] > speculative["drafted_tokens"]:
            raise CampaignError("speculative.accepted_tokens exceeds drafted_tokens")
    except (KeyError, TypeError) as exc:
        raise CampaignError(f"request_done is missing required metrics: {exc}") from exc


def require_server_log_identity(event: dict[str, Any], event_name: str) -> None:
    identity = (
        event.get("artifact_type"),
        event.get("schema_version"),
        event.get("event"),
    )
    expected = (SERVER_LOG_ARTIFACT_TYPE, SERVER_LOG_SCHEMA_VERSION, event_name)
    if type(event.get("schema_version")) is not int or identity != expected:
        raise CampaignError(f"unexpected serving log identity {identity!r}; expected {expected!r}")


def validate_server_start(event: dict[str, Any], spec: RunSpec, device: int) -> tuple[str, str]:
    require_server_log_identity(event, "server_start")
    validate_startup_metrics(event)
    engine = event["engine"]
    expected = {
        "device": device,
        "max_context": 262144,
        "kv_capacity": 262144,
        "prefill_chunk": 1024,
        "kv_cache": "int8-group64",
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": spec.speculative_backend,
        "speculative_draft_window": spec.draft_tokens,
        "proposal_head": "optimized" if spec.draft_tokens else "full",
    }
    require_fields(engine, expected, "engine")
    require_fields(event.get("sampling_defaults"),
                   {"greedy": spec.sampling_mode == "greedy"}, "sampling_defaults")
    if event.get("artifact", {}).get("target") != spec.target:
        raise CampaignError(
            "loaded artifact target mismatch: "
            f"{event.get('artifact', {}).get('target')!r} != {spec.target!r}"
        )
    weights_id = event.get("artifact", {}).get("weights_id")
    if not isinstance(weights_id, str) or not weights_id:
        raise CampaignError("server_start has no canonical artifact weights_id")
    if event.get("server", {}).get("public_model_id") != spec.model_id:
        raise CampaignError("server_start public model id does not match the campaign target")
    server_instance_id = event.get("server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise CampaignError("server_start has no server_instance_id")
    return server_instance_id, weights_id


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def build_result_record(
    spec: RunSpec,
    weights_id: str,
    payload: dict[str, Any],
    response: dict[str, Any],
    server_event: dict[str, Any],
) -> dict[str, Any]:
    require_server_log_identity(server_event, "request_done")
    validate_request_metrics(server_event, spec.speculative_backend, spec.draft_tokens)
    request = server_event.get("request", {})
    result = server_event.get("result", {})
    timings = server_event.get("timings_seconds", {})
    speculative = server_event.get("speculative", {})

    expected_request = {
        "model": spec.model_id,
        "requested_output_tokens": spec.fixture.max_new,
        "enable_thinking": spec.fixture.thinking,
        "seed": spec.seed,
    }
    actual_request = {
        "model": request.get("model"),
        "requested_output_tokens": request.get("requested_output_tokens"),
        "enable_thinking": request.get("enable_thinking"),
        "seed": request.get("sampling", {}).get("seed"),
    }
    require_fields(actual_request, expected_request, "request_done.request")

    try:
        prompt_tokens = require_integer(result["prompt_tokens"], "result.prompt_tokens")
        completion_tokens = require_integer(result["completion_tokens"], "result.completion_tokens")
        prepare_seconds = require_seconds(timings["prepare"], "timings_seconds.prepare")
        vision_seconds = require_seconds(timings["vision"], "timings_seconds.vision")
        prefill_seconds = require_seconds(timings["prefill"], "timings_seconds.prefill")
        decode_seconds = require_seconds(timings["decode"], "timings_seconds.decode")
        total_seconds = require_seconds(timings["total"], "timings_seconds.total")
        backend = str(speculative["backend"])
        speculative_rounds = require_integer(speculative["rounds"], "speculative.rounds")
        drafted_tokens = require_integer(speculative["drafted_tokens"], "speculative.drafted_tokens")
        accepted_tokens = require_integer(speculative["accepted_tokens"], "speculative.accepted_tokens")
        fallback_steps = require_integer(speculative["fallback_steps"], "speculative.fallback_steps")
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"request_done is missing required metrics: {exc}") from exc

    if backend != spec.speculative_backend:
        raise CampaignError(
            f"request_done speculative backend {backend!r} != "
            f"{spec.speculative_backend!r}"
        )

    usage = response.get("usage", {})
    require_integer(usage.get("prompt_tokens"), "usage.prompt_tokens")
    require_integer(usage.get("completion_tokens"), "usage.completion_tokens")
    if (
        usage.get("prompt_tokens") != prompt_tokens
        or usage.get("completion_tokens") != completion_tokens
    ):
        raise CampaignError("HTTP response usage does not match request_done token counts")

    decode_tokens = max(completion_tokens - 1, 0)
    metrics = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "finish_reason": result.get("finish_reason"),
        "prepare_seconds": prepare_seconds,
        "vision_seconds": vision_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
        "prefill_tok_s": safe_ratio(float(prompt_tokens), prefill_seconds),
        "server_ttft_ms": 1000.0 * (prepare_seconds + vision_seconds + prefill_seconds),
        "decode_tok_s": safe_ratio(float(decode_tokens), decode_seconds),
        "speculative_rounds": speculative_rounds,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "speculative_acceptance": safe_ratio(float(accepted_tokens), float(drafted_tokens)),
        "speculative_tokens_per_round": (
            1.0 + accepted_tokens / speculative_rounds if speculative_rounds > 0 else None
        ),
        "fallback_steps": fallback_steps,
    }
    return {
        "artifact_type": RUN_ARTIFACT_TYPE,
        "schema_version": RUN_SCHEMA_VERSION,
        "target": spec.target,
        "weights_id": weights_id,
        "model": spec.model_id,
        "artifact_path": str(spec.artifact),
        "fixture": spec.fixture.name,
        "suite": spec.fixture.suite,
        "category": spec.fixture.category,
        "seed": spec.seed,
        "speculative_mode": spec.speculative_mode,
        "speculative_backend": spec.speculative_backend,
        "draft_tokens": spec.draft_tokens,
        "sampling_mode": spec.sampling_mode,
        "request": payload,
        "response": response,
        "server_event": server_event,
        "metrics": metrics,
    }


def record_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    try:
        return (
            str(record["target"]),
            str(record["speculative_mode"]),
            str(record["sampling_mode"]),
            str(record["fixture"]),
            int(record["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"invalid corpus result record key: {exc}") from exc


def load_existing_records(
    path: Path,
    expected_specs: dict[tuple[str, str, str, str, int], RunSpec],
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = (record.get("artifact_type"), record.get("schema_version"))
                expected_identity = (RUN_ARTIFACT_TYPE, RUN_SCHEMA_VERSION)
                if identity != expected_identity:
                    raise CampaignError(
                        f"{path}:{line_number}: unexpected record identity {identity!r}"
                    )
                key = record_key(record)
                if key not in expected_specs:
                    raise CampaignError(f"{path}:{line_number}: result is outside this campaign")
                if key in records:
                    raise CampaignError(f"{path}:{line_number}: duplicate result for {key!r}")
                spec = expected_specs[key]
                if Path(record.get("artifact_path", "")).resolve() != spec.artifact:
                    raise CampaignError(
                        f"{path}:{line_number}: artifact path differs from the current command"
                    )
                records[key] = record
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"failed to read existing results from {path}: {exc}") from exc
    return records


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def server_command(
    serve: Path,
    spec: RunSpec,
    server_log: Path,
    port: int,
    device: int,
) -> list[str]:
    command = [
        str(serve),
        str(spec.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        spec.model_id,
        "--max-context",
        "262144",
        "--prefill-chunk",
        "1024",
        "--log-stats-interval-ms",
        "0",
        "--device",
        str(device),
        "--request-log-jsonl",
        str(server_log),
        "--kv-dtype",
        "int8",
        "--no-prefix-reuse",
    ]
    if spec.speculative_backend != "none":
        command.extend(
            [
                "--spec",
                spec.speculative_backend,
                "--draft-tokens",
                str(spec.draft_tokens),
                "--lm-head-draft",
            ]
        )
    if spec.sampling_mode == "greedy":
        command.append("--greedy")
    else:
        # Published stochastic measurements use this explicit profile; they must not drift when
        # product defaults follow a newly registered model recommendation.
        command.extend(
            [
                "--temperature",
                "0.6",
                "--top-p",
                "0.95",
                "--top-k",
                "20",
                "--min-p",
                "0",
                "--presence-penalty",
                "1.0",
                "--frequency-penalty",
                "0",
            ]
        )
    return command


def run_block(
    serve: Path,
    block_specs: Sequence[RunSpec],
    fixtures: dict[str, Fixture],
    output_dir: Path,
    port: int,
    device: int,
    run_handle: Any,
    records: dict[tuple[str, str, str, str, int], dict[str, Any]],
    completed_before_block: int,
    total: int,
) -> None:
    first = block_specs[0]
    server_log = (
        output_dir
        / "server"
        / f"{first.target}_{first.speculative_mode}_{first.sampling_mode}.jsonl"
    )
    command = server_command(serve, first, server_log, port, device)
    print(
        f"start {first.target}/{first.speculative_mode}: "
        f"{len(block_specs)} missing request(s)",
        flush=True,
    )
    with RunningServer(command, "127.0.0.1", port, server_log) as server:
        server_start = server.wait_until_ready()
        server_instance_id, weights_id = validate_server_start(server_start, first, device)

        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS
        )
        last_request_id: int | None = None
        try:
            warmup = fixtures[WARMUP_FIXTURE]
            post_json(connection, request_payload(first.model_id, warmup, SEEDS[0]))
            warmup_done = server.wait_for_request_done(server_instance_id)
            require_server_log_identity(warmup_done, "request_done")
            last_request_id = int(warmup_done.get("request", {}).get("request_id"))

            for block_index, spec in enumerate(block_specs, start=1):
                payload = request_payload(spec.model_id, spec.fixture, spec.seed)
                response = post_json(connection, payload)
                request_done = server.wait_for_request_done(server_instance_id)
                request_id = int(request_done.get("request", {}).get("request_id"))
                if request_id != last_request_id + 1:
                    raise CampaignError(
                        f"non-sequential serving request id {request_id}; expected {last_request_id + 1}"
                    )
                last_request_id = request_id
                record = build_result_record(spec, weights_id, payload, response, request_done)
                append_record(run_handle, record)
                records[spec.key] = record
                completed = completed_before_block + block_index
                print(
                    f"[{completed}/{total}] {spec.target}/{spec.speculative_mode} "
                    f"{spec.fixture.name} seed={spec.seed}",
                    flush=True,
                )
        finally:
            connection.close()


def metric_values(records: Iterable[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record["metrics"].get(name)
        if value is not None:
            values.append(float(value))
    return values


def sample_stats(records: Sequence[dict[str, Any]], name: str) -> tuple[int, float | None, float | None]:
    values = metric_values(records, name)
    if not values:
        return 0, None, None
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return len(values), mean, stddev


def select_records(
    records: dict[tuple[str, str, str, str, int], dict[str, Any]],
    target: str,
    speculative_mode: str,
    sampling_mode: str,
    fixtures: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        records[(target, speculative_mode, sampling_mode, fixture, seed)]
        for fixture in fixtures
        for seed in SEEDS
    ]


SUMMARY_FIELDS = (
    "section",
    "target",
    "weights_id",
    "group",
    "fixture",
    "speculative_mode",
    "sampling_mode",
    "samples",
    "prompt_tokens_mean",
    "prompt_tokens_stddev",
    "prefill_tok_s_mean",
    "prefill_tok_s_stddev",
    "server_ttft_ms_mean",
    "server_ttft_ms_stddev",
    "completion_tokens_mean",
    "completion_tokens_stddev",
    "decode_tok_s_mean",
    "decode_tok_s_stddev",
    "speculative_acceptance_mean",
    "speculative_acceptance_stddev",
    "speculative_tokens_per_round_mean",
    "speculative_tokens_per_round_stddev",
)


def set_stats(
    row: dict[str, Any],
    prefix: str,
    records: Sequence[dict[str, Any]],
    metric: str,
) -> None:
    _, mean, stddev = sample_stats(records, metric)
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_stddev"] = stddev


def summary_row(
    section: str,
    target: str,
    group: str,
    fixture: str,
    speculative_mode: str,
    sampling_mode: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    weights_ids = {str(record.get("weights_id", "")) for record in records}
    if len(weights_ids) != 1 or not next(iter(weights_ids)):
        raise CampaignError("summary group does not have one canonical weights_id")
    row: dict[str, Any] = {
        "section": section,
        "target": target,
        "weights_id": next(iter(weights_ids)),
        "group": group,
        "fixture": fixture,
        "speculative_mode": speculative_mode,
        "sampling_mode": sampling_mode,
        "samples": len(records),
    }
    set_stats(row, "prompt_tokens", records, "prompt_tokens")
    set_stats(row, "prefill_tok_s", records, "prefill_tok_s")
    set_stats(row, "server_ttft_ms", records, "server_ttft_ms")
    set_stats(row, "completion_tokens", records, "completion_tokens")
    set_stats(row, "decode_tok_s", records, "decode_tok_s")
    set_stats(row, "speculative_acceptance", records, "speculative_acceptance")
    set_stats(
        row,
        "speculative_tokens_per_round",
        records,
        "speculative_tokens_per_round",
    )
    return row


def build_summary_rows(
    records: dict[tuple[str, str, str, str, int], dict[str, Any]],
    target_order: Sequence[str],
    mode_names: Sequence[str],
    sampling_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_order:
        for mode_name in mode_names:
            backend, _ = SPECULATIVE_MODES[mode_name]
            if backend == "none":
                for fixture in NIAH_FIXTURES:
                    rows.append(
                        summary_row(
                            "context_profile",
                            target,
                            fixture,
                            fixture,
                            mode_name,
                            sampling_mode,
                            select_records(
                                records, target, mode_name, sampling_mode, (fixture,)
                            ),
                        )
                    )
                continue

            for fixture in LONG_DECODE_FIXTURES:
                rows.append(
                    summary_row(
                        "long_decode",
                        target,
                        "reasoning",
                        fixture,
                        mode_name,
                        sampling_mode,
                        select_records(
                            records, target, mode_name, sampling_mode, (fixture,)
                        ),
                    )
                )

            for category, fixture_names in SCENARIO_FIXTURES.items():
                for fixture in fixture_names:
                    rows.append(
                        summary_row(
                            "scenario_fixture",
                            target,
                            category,
                            fixture,
                            mode_name,
                            sampling_mode,
                            select_records(
                                records, target, mode_name, sampling_mode, (fixture,)
                            ),
                        )
                    )

                rows.append(
                    summary_row(
                        "scenario_category",
                        target,
                        category,
                        "",
                        mode_name,
                        sampling_mode,
                        select_records(
                            records, target, mode_name, sampling_mode, fixture_names
                        ),
                    )
                )
    return rows


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise CampaignError("summary contains a non-finite value")
    return value


def format_mean_stddev(row: dict[str, Any], prefix: str, digits: int = 1) -> str:
    mean = row.get(f"{prefix}_mean")
    stddev = row.get(f"{prefix}_stddev")
    if mean is None or stddev is None:
        return "—"
    return f"{float(mean):.{digits}f} ± {float(stddev):.{digits}f}"


def format_percent_mean_stddev(row: dict[str, Any], prefix: str) -> str:
    mean = row.get(f"{prefix}_mean")
    stddev = row.get(f"{prefix}_stddev")
    if mean is None or stddev is None:
        return "—"
    return f"{100.0 * float(mean):.1f}% ± {100.0 * float(stddev):.1f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def mode_display_name(mode_name: str) -> str:
    if mode_name == "mtp0":
        return "MTP0"
    if mode_name == "mtp3":
        return "MTP3"
    if mode_name == "dflash7":
        return "DFlash block=8 (k=7)"
    raise CampaignError(f"unsupported summary mode: {mode_name}")


def write_summaries(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row.get(field)) for field in SUMMARY_FIELDS} for row in rows
        )

    sections: list[str] = []
    mode_names = list(dict.fromkeys(str(row["speculative_mode"]) for row in rows))
    for mode_name in mode_names:
        label = mode_display_name(mode_name)
        mode_rows = [row for row in rows if row["speculative_mode"] == mode_name]
        context_rows = [
            row for row in mode_rows if row["section"] == "context_profile"
        ]
        long_decode_rows = [
            row for row in mode_rows if row["section"] == "long_decode"
        ]
        category_rows = [
            row for row in mode_rows if row["section"] == "scenario_category"
        ]

        if context_rows:
            table = markdown_table(
                (
                    "Target",
                    "Weights",
                    "Fixture",
                    "n",
                    "Prompt tokens",
                    "Prefill tok/s",
                    "Server TTFT ms",
                    "Decode tok/s",
                ),
                [
                    (
                        row["target"],
                        row["weights_id"],
                        row["fixture"],
                        str(row["samples"]),
                        format_mean_stddev(row, "prompt_tokens"),
                        format_mean_stddev(row, "prefill_tok_s"),
                        format_mean_stddev(row, "server_ttft_ms"),
                        format_mean_stddev(row, "decode_tok_s"),
                    )
                    for row in context_rows
                ],
            )
            sections.append(f"## {label} context-length profile\n\n{table}")

        if long_decode_rows:
            table = markdown_table(
                (
                    "Target",
                    "Weights",
                    "Fixture",
                    "n",
                    "Completion tokens",
                    "Decode tok/s",
                    "Spec acceptance",
                    "Spec tokens/round",
                ),
                [
                    (
                        row["target"],
                        row["weights_id"],
                        row["fixture"],
                        str(row["samples"]),
                        format_mean_stddev(row, "completion_tokens"),
                        format_mean_stddev(row, "decode_tok_s"),
                        format_percent_mean_stddev(row, "speculative_acceptance"),
                        format_mean_stddev(
                            row, "speculative_tokens_per_round", digits=2
                        ),
                    )
                    for row in long_decode_rows
                ],
            )
            sections.append(f"## {label} long-decode reasoning\n\n{table}")

        if category_rows:
            table = markdown_table(
                (
                    "Target",
                    "Weights",
                    "Category",
                    "n",
                    "Decode tok/s",
                    "Spec acceptance",
                    "Spec tokens/round",
                ),
                [
                    (
                        row["target"],
                        row["weights_id"],
                        row["group"],
                        str(row["samples"]),
                        format_mean_stddev(row, "decode_tok_s"),
                        format_percent_mean_stddev(row, "speculative_acceptance"),
                        format_mean_stddev(
                            row, "speculative_tokens_per_round", digits=2
                        ),
                    )
                    for row in category_rows
                ],
            )
            sections.append(f"## {label} cross-scenario decode\n\n{table}")
    markdown = (
        "# Serving corpus performance summary\n\n"
        "All values are arithmetic mean ± sample standard deviation. "
        f"Sampling: {rows[0]['sampling_mode'] if rows else 'n/a'}.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.port < 1 or args.port > 65535:
        raise CampaignError("--port must be in [1, 65535]")
    if args.device < 0:
        raise CampaignError("--device must be nonnegative")

    serve = args.serve.expanduser().resolve()
    if not serve.is_file():
        raise CampaignError(f"ninfer-serve executable not found: {serve}")
    if not os.access(serve, os.X_OK):
        raise CampaignError(f"ninfer-serve is not executable: {serve}")

    artifacts = parse_artifacts(args.artifact)
    mode_names = args.mode or list(DEFAULT_MODES)
    if len(mode_names) != len(set(mode_names)):
        raise CampaignError("duplicate --mode value")
    fixtures = load_fixtures()
    specs = build_specs(artifacts, fixtures, mode_names, args.sampling)
    expected_specs = {spec.key: spec for spec in specs}
    total = len(expected_specs)

    output_dir = args.output.expanduser().resolve()
    (output_dir / "server").mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.jsonl"
    records = load_existing_records(run_path, expected_specs)
    print(f"resume state: {len(records)}/{total} formal request(s) complete", flush=True)

    with run_path.open("a", encoding="utf-8") as run_handle:
        for target, _ in artifacts:
            for mode_name in mode_names:
                block_specs = [
                    spec
                    for spec in specs
                    if spec.target == target
                    and spec.speculative_mode == mode_name
                    and spec.key not in records
                ]
                if not block_specs:
                    print(f"skip {target}/{mode_name}: block complete", flush=True)
                    continue
                run_block(
                    serve,
                    block_specs,
                    fixtures,
                    output_dir,
                    args.port,
                    args.device,
                    run_handle,
                    records,
                    len(records),
                    total,
                )

    missing = set(expected_specs) - set(records)
    if missing:
        raise CampaignError(f"campaign ended with {len(missing)} missing formal request(s)")
    target_order = [target for target, _ in artifacts]
    summary_rows = build_summary_rows(records, target_order, mode_names, args.sampling)
    write_summaries(summary_rows, output_dir)
    print(
        f"completed {total} formal requests; summary: {output_dir / 'summary.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted; completed results remain in run.jsonl", file=sys.stderr)
        raise SystemExit(130) from None
