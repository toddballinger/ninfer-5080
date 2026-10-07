#!/usr/bin/env python3
"""Measure saturated decode scaling and fixed-corpus serving makespan."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import http.client
import json
import math
import os
import queue
import random
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus  # noqa: E402


SUITES = ("decode-saturation", "corpus-makespan")
STATS_INTERVAL_MS = 1000
PENDING_TIMEOUT_MS = 24 * 60 * 60 * 1000
SATURATION_FIXTURE = "long_decode_aime26_15"
SATURATION_SEEDS = (
    7632647173703958409,
    7968175640111700217,
    912910298659544128,
    9060622443728853932,
    4939353812939007330,
    3141592653589793238,
    2718281828459045235,
    1618033988749894848,
)
CORPUS_ORDER_SEED = 20260811
POINT_ARTIFACT_TYPE = "ninfer_serve_concurrency_bench_point"
SUMMARY_ARTIFACT_TYPE = "ninfer_serve_concurrency_bench_summary"
SCHEMA_VERSION = 3
KV_DTYPES = ("bf16", "int8", "q4")
KV_CACHE_LOG_NAMES = {
    "bf16": "bf16",
    "int8": "int8-group64",
    "q4": "q4-group64",
}

# Exact production C1 reference (active drop-in of ninfer-local-model.service). Memory-relevant
# flags only; the harness must match these to reproduce the production memory profile.
# Values that are host/environment-specific (bind host, model id, log path, binary/artifact
# install location) are identity/benign and do not affect memory-relevant semantics.
PRODUCTION_C1_FLAGS: list[tuple[str, str | None]] = [
    ("--host", "0.0.0.0"),
    ("--port", "8080"),
    ("--model-id", "local-model"),
    ("--max-context", "131072"),
    ("--kv-capacity", "131072"),
    ("--prefill-chunk", "1792"),
    ("--kv-dtype", "q4"),
    ("--spec", "mtp"),
    ("--draft-tokens", "3"),
    ("--max-concurrency", "1"),
    ("--max-pending-requests", "16"),
    ("--pending-timeout-ms", "180000"),
    ("--vision", None),
    ("--vision-max-tokens", "2048"),
    ("--default-thinking-budget", "2048"),
    ("--prefix-checkpoint-policy", "rolling-tool"),
    ("--request-log-jsonl", "/tmp/ninfer-serve-request-log.jsonl"),
    ("--embedding-host", None),
]

# Flags whose argv differences are pure identity (bind address, public label, log file path) and
# do not affect memory-relevant semantics.
BENIGN_IDENTITY_FLAGS = frozenset(("--host", "--model-id", "--port", "--request-log-jsonl"))
# Flags whose argv differences exist only to collect the benchmark's metrics; not production state.
INSTRUMENTATION_FLAGS = frozenset(("--log-stats-interval-ms",))
# Sampling-override flags are pinned benchmark methodology, not production state.
SAMPLING_OVERRIDE_FLAGS = frozenset(
    (
        "--temperature",
        "--top-p",
        "--top-k",
        "--min-p",
        "--presence-penalty",
        "--frequency-penalty",
    )
)


def _flag_sequence(argv: Sequence[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """Split an argv into positional tokens and a flag/value map (last value wins)."""
    positionals: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--"):
                flags[token] = str(argv[i + 1])
                i += 2
            else:
                flags[token] = "on"
                i += 1
        else:
            positionals.append(token)
            i += 1
    return tuple(positionals), flags


def _classify_difference(flag: str) -> str:
    if flag in INSTRUMENTATION_FLAGS:
        return "instrumentation-only"
    if flag in BENIGN_IDENTITY_FLAGS:
        return "benign"
    if flag in SAMPLING_OVERRIDE_FLAGS:
        return "instrumentation-only"
    return "material-unresolved"


def production_difference_report(
    production: Sequence[tuple[str, str | None]],
    generated: Sequence[str],
) -> list[dict[str, Any]]:
    """Classify every remaining difference between the generated argv and the production
    C1 reference: equivalent, benign (identity), or instrumentation-only, else material."""
    gen_positionals, gen_flags = _flag_sequence(generated)
    # Production reference: binary + artifact are positional; the rest are flag pairs.
    prod_positionals = ("ninfer-serve", "artifact.ninfer")
    prod_flags = {flag: (value if value is not None else "on") for flag, value in production}
    all_flags = sorted(set(gen_flags) | set(prod_flags))
    rows: list[dict[str, Any]] = []
    for flag in all_flags:
        in_gen = flag in gen_flags
        in_prod = flag in prod_flags
        if in_gen and in_prod and gen_flags[flag] == prod_flags[flag]:
            continue  # equivalent
        production_value = prod_flags.get(flag) if in_prod else None
        generated_value = gen_flags.get(flag) if in_gen else None
        if in_gen == in_prod and generated_value == production_value:
            continue  # equivalent (e.g. both absent)
        if in_gen and not in_prod:
            classification = _classify_difference(flag)
            detail = f"generated-only {flag}={generated_value}"
        elif in_prod and not in_gen:
            classification = _classify_difference(flag)
            detail = f"production-only {flag}={production_value} (missing in generated)"
        else:
            classification = _classify_difference(flag)
            detail = f"{flag}: production={production_value} generated={generated_value}"
        rows.append({"flag": flag, "classification": classification, "detail": detail})
    # Positional binary/artifact install locations differ between harness and production but are
    # identity-only (same weights id / serve build); classify as benign and include in the report.
    for index in (0, 1):
        gen_value = gen_positionals[index] if index < len(gen_positionals) else None
        prod_value = prod_positionals[index] if index < len(prod_positionals) else None
        if gen_value != prod_value:
            rows.append(
                {
                    "flag": f"positional[{index}]",
                    "classification": "benign",
                    "detail": f"binary/artifact install path: production={prod_value} generated={gen_value}",
                }
            )
    return rows


def _reference_argv() -> list[str]:
    """Token list for the embedded production C1 reference (flag + value pairs)."""
    tokens: list[str] = []
    for flag, value in PRODUCTION_C1_FLAGS:
        tokens.append(flag)
        if value is not None:
            tokens.append(value)
    return tokens


def _equivalence_checks(argv: Sequence[str]) -> list[tuple[str, bool]]:
    """Acceptance-criteria checks on the generated C1 argv, mirroring the audit findings the
    harness must reproduce against production. Values are derived from the real generated argv
    so the checks reflect the actual serve-launch profile, not a fixed reference."""
    argv_set = set(argv)
    kv_dtype = _flag_value(argv, "--kv-dtype")
    kv_capacity = _flag_value(argv, "--kv-capacity")
    max_context = _flag_value(argv, "--max-context")
    spec = _flag_value(argv, "--spec")
    vision_max = _flag_value(argv, "--vision-max-tokens")
    return [
        ("EMBEDDING_HOST_MATCH", "--embedding-host" in argv_set),
        ("PROPOSAL_HEAD_MATCH", "--lm-head-draft" not in argv_set),
        ("VISION_2048_MATCH", "--vision" in argv_set and vision_max == "2048"),
        ("KV_PROFILE_MATCH", kv_dtype == "q4" and kv_capacity in ("131072", "auto")),
        ("MTP_PROFILE_MATCH", spec == "mtp"),
        ("CUDA_GRAPH_MATCH", "--no-cuda-graph" not in argv_set),
    ]


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    """Return the value following a flag in an argv (last occurrence wins), or None."""
    value: str | None = None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            value = str(argv[index + 1])
    return value


@dataclasses.dataclass(frozen=True)
class Point:
    target: str
    model_id: str
    artifact: Path
    speculative_mode: str
    speculative_backend: str
    draft_tokens: int
    sampling_mode: str
    suite: str
    concurrency: int

    @property
    def key(self) -> str:
        return (
            f"{self.target}_{self.speculative_mode}_{self.sampling_mode}_"
            f"{self.suite.replace('-', '_')}_c{self.concurrency}"
        )


@dataclasses.dataclass(frozen=True)
class Job:
    index: int
    case_index: int
    fixture: corpus.Fixture
    seed: int
    max_tokens: int


@dataclasses.dataclass(frozen=True)
class ClientResult:
    job: Job
    started_at: float
    finished_at: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at


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
        choices=tuple(corpus.SPECULATIVE_MODES),
        help="speculative mode; repeat to select multiple (default: mtp0 and mtp3)",
    )
    parser.add_argument(
        "--sampling",
        choices=corpus.SAMPLING_MODES,
        default="stochastic",
        help="sampling profile for all requests (default: stochastic)",
    )
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        choices=SUITES,
        help="measurement suite; repeat to run both",
    )
    parser.add_argument(
        "--concurrency",
        action="append",
        required=True,
        type=int,
        metavar="N",
        help="startup max concurrency and client worker count; repeat to sweep",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=8192,
        help="per-request output budget for decode-saturation (default: 8192)",
    )
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument(
        "--kv-capacity",
        default="131072",
        metavar="N|auto",
        help="shared Main KV capacity passed to ninfer-serve (default: 131072; Issue #32 production baseline)",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=KV_DTYPES,
        default="q4",
        help="KV cache storage used by each benchmark point (default: q4; Issue #32 production baseline)",
    )
    parser.add_argument("--prefill-chunk", type=int, default=1792)
    parser.add_argument(
        "--max-pending-requests", type=int, default=16, help="admitted-request queue depth (default: 16; production baseline)"
    )
    parser.add_argument(
        "--pending-timeout-ms", type=int, default=180000, help="pending-request timeout (default: 180000; production baseline)"
    )
    parser.add_argument(
        "--no-embedding-host",
        action="store_true",
        help="do not pin the embedding table in host memory (production uses --embedding-host)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="do not enable media/Vision (production enables Vision with --vision-max-tokens)",
    )
    parser.add_argument(
        "--vision-max-tokens",
        type=int,
        default=2048,
        help="merged Vision token budget passed to --vision (default: 2048; production baseline)",
    )
    parser.add_argument(
        "--default-thinking-budget",
        type=int,
        default=2048,
        help="process default reasoning budget (default: 2048; production baseline)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="benchmark output directory"
    )
    parser.add_argument("--port", type=int, default=8080, help="loopback serving port")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--dry-run", action="store_true", help="print point commands and request counts only"
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.port < 1 or args.port > 65535:
        raise corpus.CampaignError("--port must be in [1, 65535]")
    if args.device < 0:
        raise corpus.CampaignError("--device must be nonnegative")
    if args.max_context <= 0:
        raise corpus.CampaignError("--max-context must be positive")
    if args.decode_tokens <= 0:
        raise corpus.CampaignError("--decode-tokens must be positive")
    if args.prefill_chunk <= 0 or args.prefill_chunk % 128 != 0:
        raise corpus.CampaignError("--prefill-chunk must be a positive multiple of 128")
    if args.kv_capacity != "auto":
        if not args.kv_capacity.isdigit() or int(args.kv_capacity) <= 0:
            raise corpus.CampaignError("--kv-capacity must be a positive integer or auto")
        if int(args.kv_capacity) < args.max_context:
            raise corpus.CampaignError("--kv-capacity must be at least --max-context")
    if args.max_pending_requests <= 0:
        raise corpus.CampaignError("--max-pending-requests must be positive")
    if args.pending_timeout_ms <= 0:
        raise corpus.CampaignError("--pending-timeout-ms must be positive")
    if args.vision_max_tokens < 0:
        raise corpus.CampaignError("--vision-max-tokens must be nonnegative")
    if args.default_thinking_budget <= 0:
        raise corpus.CampaignError("--default-thinking-budget must be positive")
    if len(args.concurrency) != len(set(args.concurrency)):
        raise corpus.CampaignError("duplicate --concurrency value")
    for concurrency in args.concurrency:
        if concurrency < 1 or concurrency > 8:
            raise corpus.CampaignError("--concurrency must be in [1, 8]")
    if len(args.suite) != len(set(args.suite)):
        raise corpus.CampaignError("duplicate --suite value")


def build_points(
    artifacts: Sequence[tuple[str, Path]], args: argparse.Namespace
) -> list[Point]:
    mode_names = args.mode or list(corpus.DEFAULT_MODES)
    if len(mode_names) != len(set(mode_names)):
        raise corpus.CampaignError("duplicate --mode value")

    points: list[Point] = []
    for target, artifact in artifacts:
        for mode_name in mode_names:
            backend, draft_tokens = corpus.SPECULATIVE_MODES[mode_name]
            if backend == "dflash" and target != "qwen3_6_35b_a3b":
                raise corpus.CampaignError("DFlash measurements require the 35B-A3B target")
            for suite in args.suite:
                for concurrency in args.concurrency:
                    points.append(
                        Point(
                            target=target,
                            model_id=corpus.TARGET_MODEL_IDS[target],
                            artifact=artifact,
                            speculative_mode=mode_name,
                            speculative_backend=backend,
                            draft_tokens=draft_tokens,
                            sampling_mode=args.sampling,
                            suite=suite,
                            concurrency=concurrency,
                        )
                    )
    return points


def build_jobs(
    point: Point, fixtures: dict[str, corpus.Fixture], decode_tokens: int
) -> list[Job]:
    if point.suite == "decode-saturation":
        fixture = fixtures[SATURATION_FIXTURE]
        return [
            Job(
                index=index,
                case_index=index,
                fixture=fixture,
                seed=SATURATION_SEEDS[index],
                max_tokens=decode_tokens,
            )
            for index in range(point.concurrency)
        ]

    names = corpus.block_fixture_names(point.speculative_backend)
    cases = [
        (fixture_index * len(corpus.SEEDS) + seed_index, name, seed)
        for fixture_index, name in enumerate(names)
        for seed_index, seed in enumerate(corpus.SEEDS)
    ]
    random.Random(CORPUS_ORDER_SEED).shuffle(cases)
    return [
        Job(
            index=index,
            case_index=case_index,
            fixture=fixtures[name],
            seed=seed,
            max_tokens=fixtures[name].max_new,
        )
        for index, (case_index, name, seed) in enumerate(cases)
    ]


def workload_order(point: Point) -> dict[str, Any]:
    if point.suite == "corpus-makespan":
        return {
            "kind": "fixed_shuffle",
            "seed": CORPUS_ORDER_SEED,
            "dispatch": "ordered_http_send",
        }
    return {
        "kind": "fixed_wave",
        "dispatch": "concurrent_http_send",
    }


def workload_order_label(point: Point) -> str:
    order = workload_order(point)
    if "seed" in order:
        return f"{order['kind']} seed={order['seed']}"
    return str(order["kind"])


def request_payload(point: Point, job: Job) -> dict[str, Any]:
    payload = corpus.request_payload(point.model_id, job.fixture, job.seed)
    payload["max_completion_tokens"] = job.max_tokens
    return payload


def server_command(
    serve: Path,
    point: Point,
    server_log: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(serve),
        str(point.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-id",
        point.model_id,
        "--max-context",
        str(args.max_context),
        "--kv-capacity",
        args.kv_capacity,
        "--max-concurrency",
        str(point.concurrency),
        "--max-pending-requests",
        str(args.max_pending_requests),
        "--pending-timeout-ms",
        str(args.pending_timeout_ms),
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--log-stats-interval-ms",
        str(STATS_INTERVAL_MS),
        "--request-log-jsonl",
        str(server_log),
        "--kv-dtype",
        args.kv_dtype,
    ]
    if args.device != 0:
        # Production omits --device (defaults to 0); only pass it when a non-default index is
        # selected, so the generated argv stays equivalent to the production C1 reference.
        command.extend(["--device", str(args.device)])
    if point.speculative_backend != "none":
        command.extend(
            [
                "--spec",
                point.speculative_backend,
                "--draft-tokens",
                str(point.draft_tokens),
            ]
        )
    # Production-equivalence: match the production serving unit's memory-relevant options.
    # The production unit does not pass --no-cuda-graph (CUDA Graph stays on) and runs with
    # default prefix-reuse (no --no-prefix-reuse) plus the rolling-tool checkpoint policy.
    if args.vision_max_tokens >= 0 and not args.no_vision:
        command.extend(["--vision", "--vision-max-tokens", str(args.vision_max_tokens)])
    if not args.no_embedding_host:
        command.append("--embedding-host")
    command.extend(["--default-thinking-budget", str(args.default_thinking_budget)])
    command.extend(["--prefix-checkpoint-policy", "rolling-tool"])
    if point.sampling_mode == "greedy":
        command.append("--greedy")
    else:
        # Preserve the sampling profile attached to the published benchmark methodology.
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


def validate_server_start(
    event: dict[str, Any], point: Point, args: argparse.Namespace
) -> tuple[str, str]:
    corpus.require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    expected = {
        "device": args.device,
        "max_context": args.max_context,
        "kv_capacity_mode": "auto" if args.kv_capacity == "auto" else "explicit",
        "max_concurrency": point.concurrency,
        "max_pending_requests": args.max_pending_requests,
        "pending_timeout_ms": args.pending_timeout_ms,
        "prefill_chunk": args.prefill_chunk,
        "log_stats_interval_ms": STATS_INTERVAL_MS,
        "kv_cache": KV_CACHE_LOG_NAMES[args.kv_dtype],
        "vision": not args.no_vision,
        "vision_max_tokens": args.vision_max_tokens,
        "cuda_graph": True,
        "prefix_reuse": True,
        "speculative_backend": point.speculative_backend,
        "speculative_draft_window": point.draft_tokens,
        # Production-equivalence: production does not pass --lm-head-draft, so the full
        # LM-head proposal path is used for speculative points.
        "proposal_head": "full",
    }
    actual = {name: engine[name] for name in expected if name in engine}
    if actual != expected or any(name not in engine for name in expected):
        raise corpus.CampaignError(f"server_start Engine configuration mismatch: {actual!r}")
    # Numeric engine.kv_capacity enforcement: kv_capacity_mode alone cannot discriminate a
    # material capacity (e.g. 16384 vs 131072 both report "explicit"); for an explicit numeric
    # request the serve must report that exact capacity (the Issue #32 commit-gate finding).
    if args.kv_capacity != "auto":
        kv = engine.get("kv_capacity")
        if isinstance(kv, int) and kv != int(args.kv_capacity):
            raise corpus.CampaignError(
                f"server_start numeric kv_capacity mismatch: engine={kv!r} requested={args.kv_capacity}"
            )
        if not isinstance(kv, int):
            raise corpus.CampaignError(
                f"server_start kv_capacity must be a positive integer when an explicit capacity is requested: engine={kv!r} requested={args.kv_capacity}"
            )
    if event.get("sampling_defaults", {}).get("greedy") != (
        point.sampling_mode == "greedy"
    ):
        raise corpus.CampaignError("server_start sampling mode does not match the point")
    if event.get("server", {}).get("default_thinking_budget") != args.default_thinking_budget:
        raise corpus.CampaignError(
            "server_start default_thinking_budget does not match the point"
        )
    # Embedding host: production pins the embedding table in host memory; the serve must report it.
    if engine.get("embedding_host") != (not args.no_embedding_host):
        raise corpus.CampaignError(
            f"server_start embedding_host does not match the request: engine={engine.get('embedding_host')!r} requested={not args.no_embedding_host!r}"
        )
    # Prefix checkpoint policy: production runs rolling-tool; the serve must report the same.
    if event.get("server", {}).get("prefix_checkpoint_policy") != "rolling-tool":
        raise corpus.CampaignError(
            f"server_start prefix_checkpoint_policy does not match production: {event.get('server', {}).get('prefix_checkpoint_policy')!r}; expected rolling-tool"
        )
    if event.get("artifact", {}).get("target") != point.target:
        raise corpus.CampaignError("loaded artifact target does not match the point")
    if event.get("server", {}).get("public_model_id") != point.model_id:
        raise corpus.CampaignError("server public model id does not match the point")

    server_instance_id = event.get("server_instance_id")
    weights_id = event.get("artifact", {}).get("weights_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise corpus.CampaignError("server_start has no server_instance_id")
    if not isinstance(weights_id, str) or not weights_id:
        raise corpus.CampaignError("server_start has no canonical weights_id")
    return server_instance_id, weights_id


def parse_client_response(
    job: Job, response: dict[str, Any], started_at: float, finished_at: float
) -> ClientResult:
    try:
        usage = response["usage"]
        choices = response["choices"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        finish_reason = str(choices[0]["finish_reason"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise corpus.CampaignError(f"invalid Chat Completions response: {exc}") from exc
    return ClientResult(
        job=job,
        started_at=started_at,
        finished_at=finished_at,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )


def run_clients(
    point: Point, jobs: Sequence[Job], port: int
) -> tuple[list[ClientResult], float, float]:
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)

    ready: queue.Queue[Exception | None] = queue.Queue()
    start_event = threading.Event()
    failed = threading.Event()
    result_lock = threading.Lock()
    ordered_dispatch = point.suite == "corpus-makespan"
    dispatch_condition = threading.Condition()
    next_dispatch_index = 0
    results: list[ClientResult] = []
    errors: list[Exception] = []

    def record_failure(error: Exception) -> None:
        with result_lock:
            errors.append(error)
        failed.set()
        with dispatch_condition:
            dispatch_condition.notify_all()

    def worker() -> None:
        nonlocal next_dispatch_index
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=corpus.REQUEST_TIMEOUT_SECONDS
        )
        try:
            connection.connect()
        except Exception as exc:
            ready.put(exc)
            connection.close()
            return
        ready.put(None)
        start_event.wait()
        try:
            while not failed.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    payload = request_payload(point, job)
                    if ordered_dispatch:
                        with dispatch_condition:
                            dispatch_condition.wait_for(
                                lambda: failed.is_set() or job.index == next_dispatch_index
                            )
                            if failed.is_set():
                                return
                            started_at = time.monotonic()
                            corpus.send_json(connection, payload)
                            next_dispatch_index += 1
                            dispatch_condition.notify_all()
                        response = corpus.receive_json(connection)
                    else:
                        started_at = time.monotonic()
                        response = corpus.post_json(connection, payload)
                    finished_at = time.monotonic()
                    result = parse_client_response(job, response, started_at, finished_at)
                except Exception as exc:
                    record_failure(exc)
                    return
                with result_lock:
                    results.append(result)
        finally:
            connection.close()

    threads = [
        threading.Thread(target=worker, name=f"bench-client-{index}")
        for index in range(point.concurrency)
    ]
    for thread in threads:
        thread.start()

    readiness_errors = [error for error in (ready.get() for _ in threads) if error is not None]
    if readiness_errors:
        failed.set()
        start_event.set()
        for thread in threads:
            thread.join()
        raise corpus.CampaignError(f"failed to connect benchmark clients: {readiness_errors[0]}")

    campaign_start = time.monotonic()
    start_event.set()
    for thread in threads:
        thread.join()

    if errors:
        error = errors[0]
        if isinstance(error, corpus.CampaignError):
            raise error
        raise corpus.CampaignError(f"concurrent request failed: {error}") from error
    if len(results) != len(jobs):
        raise corpus.CampaignError(
            f"completed {len(results)} client request(s), expected {len(jobs)}"
        )
    campaign_end = max(result.finished_at for result in results)
    return sorted(results, key=lambda result: result.job.index), campaign_start, campaign_end


def load_server_events(path: Path, server_instance_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("server_instance_id") != server_instance_id:
                    continue
                event_name = event.get("event")
                if not isinstance(event_name, str):
                    raise corpus.CampaignError(f"{path}:{line_number}: event name is missing")
                corpus.require_server_log_identity(event, event_name)
                events.append(event)
    except (OSError, json.JSONDecodeError) as exc:
        raise corpus.CampaignError(f"failed to read serving events from {path}: {exc}") from exc
    return events


def sum_request_done(events: Sequence[dict[str, Any]]) -> dict[str, int | float | None]:
    prompt_tokens = 0
    completion_tokens = 0
    computed_prefill_tokens = 0
    speculative_rounds = 0
    drafted_tokens = 0
    accepted_tokens = 0
    fallback_steps = 0
    decode_tokens = 0
    for event in events:
        try:
            result = event["result"]
            speculative = event["speculative"]
            prompt_tokens += int(result["prompt_tokens"])
            request_completion_tokens = int(result["completion_tokens"])
            completion_tokens += request_completion_tokens
            decode_tokens += max(request_completion_tokens - 1, 0)
            computed_prefill_tokens += int(result["computed_prefill_tokens"])
            speculative_rounds += int(speculative["rounds"])
            drafted_tokens += int(speculative["drafted_tokens"])
            accepted_tokens += int(speculative["accepted_tokens"])
            fallback_steps += int(speculative["fallback_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise corpus.CampaignError(f"request_done event is missing metrics: {exc}") from exc
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "computed_prefill_tokens": computed_prefill_tokens,
        "speculative_rounds": speculative_rounds,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "speculative_acceptance": (
            accepted_tokens / drafted_tokens if drafted_tokens > 0 else None
        ),
        "fallback_steps": fallback_steps,
    }


def sum_throughput(events: Sequence[dict[str, Any]]) -> dict[str, int | float | None]:
    computed_prefill_tokens = 0
    committed_decode_tokens = 0
    decode_rounds = 0
    decode_row_rounds = 0
    for event in events:
        try:
            computed_prefill_tokens += int(event["tokens"]["computed_prefill"])
            committed_decode_tokens += int(event["tokens"]["committed_decode"])
            decode_rounds += int(event["decode_batch"]["rounds"])
            decode_row_rounds += int(event["decode_batch"]["row_rounds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise corpus.CampaignError(f"throughput event is missing counters: {exc}") from exc
    return {
        "computed_prefill_tokens": computed_prefill_tokens,
        "committed_decode_tokens": committed_decode_tokens,
        "decode_rounds": decode_rounds,
        "decode_row_rounds": decode_row_rounds,
        "average_decode_batch": (
            decode_row_rounds / decode_rounds if decode_rounds > 0 else None
        ),
    }


def is_steady_interval(event: dict[str, Any], concurrency: int) -> bool:
    try:
        tokens = event["tokens"]
        batch = event["decode_batch"]
        scheduler = event["scheduler"]
        rounds = int(batch["rounds"])
        return (
            int(tokens["computed_prefill"]) == 0
            and rounds > 0
            and int(batch["row_rounds"]) == concurrency * rounds
            and int(scheduler["running"]) == concurrency
            and int(scheduler["prefilling"]) == 0
            and int(scheduler["decode_ready"]) == concurrency
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise corpus.CampaignError(
            f"throughput event is missing steady-state fields: {exc}"
        ) from exc


def steady_metrics(
    events: Sequence[dict[str, Any]], concurrency: int
) -> dict[str, int | float]:
    selected = [event for event in events if is_steady_interval(event, concurrency)]
    if not selected:
        raise corpus.CampaignError(
            f"concurrency {concurrency} produced no complete full-batch decode interval; "
            "increase --decode-tokens"
        )
    duration = sum(float(event["interval_seconds"]) for event in selected)
    tokens = sum(int(event["tokens"]["committed_decode"]) for event in selected)
    rounds = sum(int(event["decode_batch"]["rounds"]) for event in selected)
    row_rounds = sum(int(event["decode_batch"]["row_rounds"]) for event in selected)
    if duration <= 0.0 or rounds <= 0:
        raise corpus.CampaignError("steady decode interval has no measurable duration or rounds")
    return {
        "intervals": len(selected),
        "seconds": duration,
        "committed_decode_tokens": tokens,
        "decode_rounds": rounds,
        "decode_row_rounds": row_rounds,
        "average_decode_batch": row_rounds / rounds,
        "decode_tokens_per_second": tokens / duration,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise corpus.CampaignError("cannot compute a percentile from an empty sample")
    if quantile < 0.0 or quantile > 1.0:
        raise corpus.CampaignError("percentile quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def client_latency_metrics(results: Sequence[ClientResult]) -> dict[str, float]:
    elapsed = [result.elapsed_seconds for result in results]
    if not elapsed:
        raise corpus.CampaignError("concurrency point completed no client requests")
    return {
        "mean": sum(elapsed) / len(elapsed),
        "p50": percentile(elapsed, 0.50),
        "p95": percentile(elapsed, 0.95),
        "max": max(elapsed),
    }


def client_records(
    results: Sequence[ClientResult], campaign_start: float
) -> list[dict[str, Any]]:
    return [
        {
            "index": result.job.index,
            "case_index": result.job.case_index,
            "fixture": result.job.fixture.name,
            "seed": result.job.seed,
            "requested_output_tokens": result.job.max_tokens,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
            "start_offset_seconds": result.started_at - campaign_start,
            "finish_offset_seconds": result.finished_at - campaign_start,
            "elapsed_seconds": result.elapsed_seconds,
        }
        for result in results
    ]


def analyze_point(
    point: Point,
    command: Sequence[str],
    server_log: Path,
    server_start: dict[str, Any],
    weights_id: str,
    events: Sequence[dict[str, Any]],
    results: Sequence[ClientResult],
    campaign_start: float,
    campaign_end: float,
) -> dict[str, Any]:
    errors = [event for event in events if event.get("event") == "request_error"]
    if errors:
        message = errors[0].get("error", {}).get("message", "unknown request error")
        raise corpus.CampaignError(f"serving request failed: {message}")
    request_done = [event for event in events if event.get("event") == "request_done"]
    throughput = [event for event in events if event.get("event") == "throughput"]
    if len(request_done) != len(results):
        raise corpus.CampaignError(
            f"server recorded {len(request_done)} completed request(s), expected {len(results)}"
        )

    done_totals = sum_request_done(request_done)
    runtime_totals = sum_throughput(throughput)
    client_prompt = sum(result.prompt_tokens for result in results)
    client_completion = sum(result.completion_tokens for result in results)
    if client_prompt != done_totals["prompt_tokens"] or client_completion != done_totals[
        "completion_tokens"
    ]:
        raise corpus.CampaignError("client usage and request_done token totals differ")
    if runtime_totals["computed_prefill_tokens"] != done_totals["computed_prefill_tokens"]:
        raise corpus.CampaignError("throughput and request_done prefill token totals differ")
    if runtime_totals["committed_decode_tokens"] != done_totals["decode_tokens"]:
        raise corpus.CampaignError("throughput and request_done decode token totals differ")

    makespan = campaign_end - campaign_start
    if makespan <= 0.0:
        raise corpus.CampaignError("campaign makespan is not positive")
    metrics: dict[str, Any]
    latency = client_latency_metrics(results)
    if point.suite == "decode-saturation":
        if any(
            event.get("result", {}).get("finish_reason") != "output_limit"
            for event in request_done
        ):
            raise corpus.CampaignError(
                "decode-saturation request stopped before its output limit"
            )
        metrics = {
            "wave_makespan_seconds": makespan,
            "request_latency_seconds": latency,
            "steady": steady_metrics(throughput, point.concurrency),
        }
    else:
        metrics = {
            "makespan_seconds": makespan,
            "request_latency_seconds": latency,
            "requests_per_second": len(results) / makespan,
            "computed_prefill_tokens_per_second": (
                int(done_totals["computed_prefill_tokens"]) / makespan
            ),
            "decode_tokens_per_second": int(done_totals["decode_tokens"]) / makespan,
        }

    return {
        "artifact_type": POINT_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "target": point.target,
        "weights_id": weights_id,
        "model": point.model_id,
        "artifact_path": str(point.artifact),
        "speculative_mode": point.speculative_mode,
        "speculative_backend": point.speculative_backend,
        "draft_tokens": point.draft_tokens,
        "sampling_mode": point.sampling_mode,
        "suite": point.suite,
        "workload_order": workload_order(point),
        "concurrency": point.concurrency,
        "request_count": len(results),
        "command": list(command),
        "server_log": str(server_log),
        "engine": server_start.get("engine", {}),
        "memory": server_start.get("memory", {}),
        "environment": server_start.get("environment", {}),
        "totals": done_totals,
        "decode_batch": {
            "rounds": runtime_totals["decode_rounds"],
            "row_rounds": runtime_totals["decode_row_rounds"],
            "average_size": runtime_totals["average_decode_batch"],
        },
        "metrics": metrics,
        "requests": client_records(results, campaign_start),
    }


def run_point(
    serve: Path,
    point: Point,
    fixtures: dict[str, corpus.Fixture],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    jobs = build_jobs(point, fixtures, args.decode_tokens)
    server_log = output_dir / "server" / f"{point.key}.jsonl"
    command = server_command(serve, point, server_log, args)
    print(
        f"start {point.target}/{point.speculative_mode}/{point.suite} "
        f"C={point.concurrency} requests={len(jobs)} order={workload_order_label(point)}",
        flush=True,
    )

    with corpus.RunningServer(command, "127.0.0.1", args.port, server_log) as server:
        server_start = server.wait_until_ready()
        server_instance_id, weights_id = validate_server_start(server_start, point, args)
        results, campaign_start, campaign_end = run_clients(point, jobs, args.port)

    events = load_server_events(server_log, server_instance_id)
    report = analyze_point(
        point,
        command,
        server_log,
        server_start,
        weights_id,
        events,
        results,
        campaign_start,
        campaign_end,
    )
    point_path = output_dir / "points" / f"{point.key}.json"
    point_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if point.suite == "decode-saturation":
        result_text = (
            f"steady={report['metrics']['steady']['decode_tokens_per_second']:.1f}tok/s "
            f"batch={report['metrics']['steady']['average_decode_batch']:.2f}"
        )
    else:
        result_text = (
            f"makespan={report['metrics']['makespan_seconds']:.2f}s "
            f"batch={report['decode_batch']['average_size']:.2f}"
        )
    print(f"done {point.key}: {result_text} ({point_path})", flush=True)
    return report


def add_speedups(reports: Sequence[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for report in reports:
        key = (
            str(report["target"]),
            str(report["weights_id"]),
            str(report["speculative_mode"]),
            str(report["sampling_mode"]),
            str(report["suite"]),
        )
        if int(report["concurrency"]) == 1:
            baselines[key] = report

    for report in reports:
        key = (
            str(report["target"]),
            str(report["weights_id"]),
            str(report["speculative_mode"]),
            str(report["sampling_mode"]),
            str(report["suite"]),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        if report["suite"] == "decode-saturation":
            current = float(report["metrics"]["steady"]["decode_tokens_per_second"])
            reference = float(baseline["metrics"]["steady"]["decode_tokens_per_second"])
            report["speedup_vs_c1"] = current / reference
        else:
            current = float(report["metrics"]["makespan_seconds"])
            reference = float(baseline["metrics"]["makespan_seconds"])
            report["speedup_vs_c1"] = reference / current


SUMMARY_FIELDS = (
    "suite",
    "target",
    "weights_id",
    "speculative_mode",
    "sampling_mode",
    "corpus_order_seed",
    "concurrency",
    "request_count",
    "prompt_tokens",
    "computed_prefill_tokens",
    "decode_tokens",
    "average_decode_batch",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "latency_max_seconds",
    "resolved_kv_capacity",
    "kv_payload_mib",
    "runtime_reservation_mib",
    "planned_slack_mib",
    "cuda_graph_allowance_mib",
    "cuda_graph_observed_mib",
    "steady_seconds",
    "steady_decode_tokens_per_second",
    "makespan_seconds",
    "requests_per_second",
    "workload_prefill_tokens_per_second",
    "workload_decode_tokens_per_second",
    "speedup_vs_c1",
)


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "suite": report["suite"],
        "target": report["target"],
        "weights_id": report["weights_id"],
        "speculative_mode": report["speculative_mode"],
        "sampling_mode": report["sampling_mode"],
        "corpus_order_seed": report.get("workload_order", {}).get("seed"),
        "concurrency": report["concurrency"],
        "request_count": report["request_count"],
        "prompt_tokens": report["totals"]["prompt_tokens"],
        "computed_prefill_tokens": report["totals"]["computed_prefill_tokens"],
        "decode_tokens": report["totals"]["decode_tokens"],
        "average_decode_batch": report["decode_batch"]["average_size"],
        "latency_mean_seconds": report["metrics"]["request_latency_seconds"]["mean"],
        "latency_p50_seconds": report["metrics"]["request_latency_seconds"]["p50"],
        "latency_p95_seconds": report["metrics"]["request_latency_seconds"]["p95"],
        "latency_max_seconds": report["metrics"]["request_latency_seconds"]["max"],
        "resolved_kv_capacity": report.get("engine", {}).get("kv_capacity"),
        "kv_payload_mib": (
            float(report.get("memory", {}).get("kv_payload_bytes", 0)) / (1024.0 * 1024.0)
        ),
        "runtime_reservation_mib": (
            float(report.get("memory", {}).get("runtime_reservation_bytes", 0)) / (1024.0 * 1024.0)
        ),
        "planned_slack_mib": (
            float(report.get("memory", {}).get("planned_slack_bytes", 0)) / (1024.0 * 1024.0)
        ),
        "cuda_graph_allowance_mib": (
            float(report.get("memory", {}).get("cuda_graph_allowance_bytes", 0)) / (1024.0 * 1024.0)
        ),
        "cuda_graph_observed_mib": (
            float(report.get("memory", {}).get("cuda_graph_observed_bytes", 0)) / (1024.0 * 1024.0)
        ),
        "steady_seconds": None,
        "steady_decode_tokens_per_second": None,
        "makespan_seconds": None,
        "requests_per_second": None,
        "workload_prefill_tokens_per_second": None,
        "workload_decode_tokens_per_second": None,
        "speedup_vs_c1": report.get("speedup_vs_c1"),
    }
    if report["suite"] == "decode-saturation":
        row["average_decode_batch"] = report["metrics"]["steady"]["average_decode_batch"]
        row["steady_seconds"] = report["metrics"]["steady"]["seconds"]
        row["steady_decode_tokens_per_second"] = report["metrics"]["steady"][
            "decode_tokens_per_second"
        ]
        row["makespan_seconds"] = report["metrics"]["wave_makespan_seconds"]
    else:
        row["makespan_seconds"] = report["metrics"]["makespan_seconds"]
        row["requests_per_second"] = report["metrics"]["requests_per_second"]
        row["workload_prefill_tokens_per_second"] = report["metrics"][
            "computed_prefill_tokens_per_second"
        ]
        row["workload_decode_tokens_per_second"] = report["metrics"][
            "decode_tokens_per_second"
        ]
    return row


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise corpus.CampaignError("summary contains a non-finite value")
    return value


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def write_summaries(reports: Sequence[dict[str, Any]], output_dir: Path) -> None:
    rows = [summary_row(report) for report in reports]
    summary = {
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "points": list(reports),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row.get(field)) for field in SUMMARY_FIELDS} for row in rows
        )

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["target"]),
            str(row["weights_id"]),
            str(row["speculative_mode"]),
            str(row["suite"]),
        )
        groups.setdefault(key, []).append(row)

    sections: list[str] = []
    for (target, weights_id, mode, suite), group in groups.items():
        group.sort(key=lambda row: int(row["concurrency"]))
        title = f"## {target} / {weights_id} / {mode} / {suite}"
        if suite == "decode-saturation":
            table = markdown_table(
                (
                    "C",
                    "Requests",
                    "Steady s",
                    "Avg batch",
                    "Decode tok/s",
                    "P50 s",
                    "P95 s",
                    "KV MiB",
                    "Slack MiB",
                    "Speedup",
                ),
                [
                    (
                        str(row["concurrency"]),
                        str(row["request_count"]),
                        format_number(row["steady_seconds"]),
                        format_number(row["average_decode_batch"]),
                        format_number(row["steady_decode_tokens_per_second"], 1),
                        format_number(row["latency_p50_seconds"]),
                        format_number(row["latency_p95_seconds"]),
                        format_number(row["kv_payload_mib"], 1),
                        format_number(row["planned_slack_mib"], 1),
                        format_number(row["speedup_vs_c1"]),
                    )
                    for row in group
                ],
            )
        else:
            table = markdown_table(
                (
                    "C",
                    "Requests",
                    "Makespan s",
                    "Requests/s",
                    "Prefill tok/s",
                    "Decode tok/s",
                    "Avg batch",
                    "P50 s",
                    "P95 s",
                    "KV MiB",
                    "Slack MiB",
                    "Speedup",
                ),
                [
                    (
                        str(row["concurrency"]),
                        str(row["request_count"]),
                        format_number(row["makespan_seconds"]),
                        format_number(row["requests_per_second"], 4),
                        format_number(row["workload_prefill_tokens_per_second"], 1),
                        format_number(row["workload_decode_tokens_per_second"], 1),
                        format_number(row["average_decode_batch"]),
                        format_number(row["latency_p50_seconds"]),
                        format_number(row["latency_p95_seconds"]),
                        format_number(row["kv_payload_mib"], 1),
                        format_number(row["planned_slack_mib"], 1),
                        format_number(row["speedup_vs_c1"]),
                    )
                    for row in group
                ],
            )
        sections.append(f"{title}\n\n{table}")

    corpus_order_note = ""
    if any(report["suite"] == "corpus-makespan" for report in reports):
        corpus_order_note = (
            f" Corpus-makespan uses fixed shuffle seed {CORPUS_ORDER_SEED} and ordered HTTP sends."
        )
    markdown = (
        "# Concurrent serving benchmark\n\n"
        "Saturated decode rates use only complete intervals whose decode batch equals the "
        "configured concurrency. Corpus makespan spans simultaneous client release through the "
        f"last complete HTTP response.{corpus_order_note}\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    artifacts = corpus.parse_artifacts(args.artifact)
    fixtures = corpus.load_fixtures()
    points = build_points(artifacts, args)
    output_dir = args.output.expanduser().resolve()

    serve = args.serve.expanduser().resolve()
    if not args.dry_run:
        if not serve.is_file():
            raise corpus.CampaignError(f"ninfer-serve executable not found: {serve}")
        if not os.access(serve, os.X_OK):
            raise corpus.CampaignError(f"ninfer-serve is not executable: {serve}")
        (output_dir / "server").mkdir(parents=True, exist_ok=True)
        (output_dir / "points").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for point in points:
            log_path = output_dir / "server" / f"{point.key}.jsonl"
            jobs = build_jobs(point, fixtures, args.decode_tokens)
            print(
                f"# {point.key}: {len(jobs)} request(s), "
                f"order={workload_order_label(point)}"
            )
            print(shlex.join(server_command(serve, point, log_path, args)))

        # Production-equivalence dry-run: emit the exact production-C1 reference argv, the
        # generated (production-equivalent) C1 mtp3 argv (the first max-concurrency 1 point's
        # server_command), and classify every remaining difference between them. No synthetic
        # builder or reference-only path is introduced; the generated argv is the harness's
        # real per-point command, so the comparison exercises the actual serve-launch profile.
        reference = _reference_argv()
        print("\n# PRODUCTION-EQUIVALENCE C1 (production reference):")
        print(f"{serve} <artifact> {shlex.join(reference)}")
        c1_point = next((p for p in points if p.concurrency == 1), None)
        if c1_point is not None:
            c1_log = output_dir / "server" / f"{c1_point.key}.jsonl"
            generated_c1 = server_command(serve, c1_point, c1_log, args)
        else:
            generated_c1 = []
        if generated_c1:
            print("\n# PRODUCTION-EQUIVALENCE C1 (generated, production-equivalent profile):")
            print(shlex.join(generated_c1))
            rows = production_difference_report(PRODUCTION_C1_FLAGS, generated_c1)
            print("\n# GENERATED-VS-PRODUCTION DIFFERENCES (C1 reference):")
            if not rows:
                print("  (none) — generated argv is equivalent to the production C1 reference")
            for row in rows:
                print(f"  {row['classification']:<18} {row['flag']:<28} {row['detail']}")
            material = [r for r in rows if r["classification"] == "material-unresolved"]
            print("\n# ACCEPTANCE (production-equivalence dry-run):")
            for label, ok in _equivalence_checks(generated_c1):
                print(f"  {label}={'YES' if ok else 'NO'}")
            if not material:
                print("  PRODUCTION_EQUIVALENCE_DRY_RUN=PASS")
            else:
                print(f"  PRODUCTION_EQUIVALENCE_DRY_RUN=FAIL ({len(material)} material mismatch(es))")
            print(f"  REMAINING_MATERIAL_MISMATCHES={'NONE' if not material else str(len(material))}")
        return 0

    reports: list[dict[str, Any]] = []
    for point in points:
        reports.append(run_point(serve, point, fixtures, output_dir, args))

    add_speedups(reports)
    write_summaries(reports, output_dir)
    print(f"summary: {output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
