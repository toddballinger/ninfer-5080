#!/usr/bin/env python3
"""Issue #58: offline request event timeline and admission-evidence audit.

Read existing NInfer server JSONL. Does NOT infer queue/admission from request_start.
Never connects to a server, runs a model, or touches production.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

EVENTS = {"request_start", "request_done", "request_error", "request_rejected"}
TERMINAL = EVENTS - {"request_start"}


def inspect_events(events):
    grouped = defaultdict(list)
    unknown = 0
    for source, lineno, event in events:
        if not isinstance(event, dict):
            raise ValueError(f"{source}:{lineno}: event must be JSON object")
        kind = event.get("event")
        if kind not in EVENTS:
            continue
        request = event.get("request")
        rid = request.get("request_id") if isinstance(request, dict) else event.get("request_id")
        instance = event.get("server_instance_id")
        timestamp = event.get("timestamp_unix_ms")
        if rid is None or not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            unknown += 1
            continue
        # If instance is absent, isolate by file rather than silently combining servers.
        namespace = instance if isinstance(instance, str) and instance else source
        grouped[(namespace, str(rid))].append((timestamp, kind, f"{source}:{lineno}", event))
    records = []
    anomalies = []
    for (namespace, rid), values in sorted(grouped.items()):
        values.sort(key=lambda x: (x[0], x[2]))
        starts = [v for v in values if v[1] == "request_start"]
        finishes = [v for v in values if v[1] in TERMINAL]
        flags = []
        if len(starts) != 1:
            flags.append("missing_start" if not starts else "duplicate_start")
        if len(finishes) != 1:
            flags.append("missing_terminal" if not finishes else "multiple_terminal")
        if starts and finishes and finishes[0][0] < starts[0][0]:
            flags.append("terminal_before_start")
        if flags:
            anomalies.append({"namespace": namespace, "request_id": rid, "flags": flags})
        elapsed = (finishes[0][0] - starts[0][0]) if len(starts) == len(finishes) == 1 else None
        records.append({
            "namespace": namespace, "request_id": rid,
            "start_unix_ms": starts[0][0] if len(starts) == 1 else None,
            "terminal_unix_ms": finishes[0][0] if len(finishes) == 1 else None,
            "start_to_terminal_ms": elapsed,
            "terminal_kind": finishes[0][1] if len(finishes) == 1 else None,
            "events": [{"timestamp_unix_ms": v[0], "kind": v[1], "source": v[2]}
                       for v in values],
            "anomalies": flags,
            "admission_unix_ms": None,
            "queue_wait_ms": None,
            "admission_deferral_reason": None,
        })
    return {
        "schema": "issue58-offline-event-timeline-v1",
        "records": records, "anomalies": anomalies,
        "unusable_events": unknown,
        "request_count": len(records),
        "admission_directly_observed": False,
        "limitations": [
            "request_start is a logged event, NOT proof of actual admission or arrival",
            "start-to-terminal includes unknown queue, prefill, decode and response overhead",
            "JSONL has no independently established queue-enter/admission timestamps",
            "No causal inference about KV pressure, lane ownership or scheduler starvation",
            "Missing or split log files can produce incomplete timelines",
        ],
    }


def read_jsonl(paths):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield str(path), lineno, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: malformed JSONL") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="saved request log JSONL files; offline only")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_events(read_jsonl(args.jsonl)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
