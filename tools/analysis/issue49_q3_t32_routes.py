#!/usr/bin/env python3
"""Issue #49: source-derived Q3 SwiGLU route model and offline histogram.

Pure Python, CPU-only. No CUDA, runtime hooks, production service access or
claims about measured latency. Input is operator call counts, NOT request counts.
"""
import argparse
import json
import sys
from collections import Counter


def route(tokens, policy="AllowA8"):
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("token count must be a positive integer")
    if policy not in ("AllowA8", "A16Only"):
        raise ValueError("unsupported policy")
    if tokens == 1:
        return {"gemv": 1}
    if tokens == 2:
        return {"small2_mma": 1}
    if tokens == 3:
        return {"small3_pair": 1}
    if tokens == 4:
        return {"small4_mma": 1}
    if policy == "AllowA8" and tokens >= 257:
        return {"int8": (tokens + 4095) // 4096}
    remaining = tokens
    counts = Counter()
    for width, name in ((32, "t32_mma"), (16, "small16_pair"),
                        (8, "small8_pair"), (4, "small4_pair")):
        q, remaining = divmod(remaining, width)
        if q:
            counts[name] += q
    if remaining:
        counts[{1: "gemv", 2: "small2_pair", 3: "small3_pair"}[remaining]] += 1
    return dict(counts)


def aggregate(records, policy="AllowA8"):
    calls, kernels = 0, Counter()
    bands = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each JSONL record must be an object")
        n, count = record["tokens"], record.get("calls", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("calls must be a nonnegative integer")
        mapping = route(n, policy)
        calls += count
        kernels.update({k: v * count for k, v in mapping.items()})
        band = ("1..31" if n <= 31 else "32..256" if n <= 256 else "257+")
        bands[band] += count
    return {
        "policy": policy,
        "operator_calls": calls,
        "bands": dict(sorted(bands.items())),
        "kernel_launch_estimates": dict(sorted(kernels.items())),
        "t32_launches": kernels["t32_mma"],
        "calls_with_t32": None,  # filled by main, independently of launch count
        "observed_ttft_ms": None,
        "measured_kernel_latency_us": None,
        "warning": "Route model only: no latency/TTFT or actual runtime frequency measured.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help='JSONL records e.g. {"tokens":32,"calls":5}; "-" for stdin')
    parser.add_argument("--policy", choices=("AllowA8", "A16Only"), default="AllowA8")
    args = parser.parse_args(argv)
    stream = sys.stdin if args.jsonl == "-" else open(args.jsonl, encoding="utf-8")
    try:
        records = [json.loads(line) for line in stream if line.strip()]
    finally:
        if stream is not sys.stdin:
            stream.close()
    result = aggregate(records, args.policy)
    result["calls_with_t32"] = sum(
        row.get("calls", 1) for row in records
        if route(row["tokens"], args.policy).get("t32_mma", 0) > 0
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
