#!/usr/bin/env python3
"""Issue #26: deterministic host-only prefix-reuse opportunity campaign.

This is an exact identity *opportunity* model, NOT a NInfer cache simulator.
No tokenizer, model, CUDA or production interaction. No synthetic TTFT claims.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    label: str
    length: int
    vision_digest: str = ""
    position_delta: int = 0

    def __post_init__(self):
        if self.length < 1:
            raise ValueError("span length must be positive")


def prompt(*spans):
    """Emit exact per-token identities including synthetic Vision and position."""
    output = []
    for span in spans:
        # Hash-based pseudo-token identity; deliberately NOT actual tokenization.
        base = hashlib.sha256(span.label.encode("utf-8")).hexdigest()
        for i in range(span.length):
            token = (base, i, span.vision_digest, span.position_delta + len(output))
            output.append(token)
    return tuple(output)


def lcp(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def campaign():
    """Named synthetic scenarios; scenario isolation prevents accidental reuse."""
    sys = Span("system", 1024)
    history = Span("history", 65536)
    tool1 = Span("tool-output-1", 96)
    tool2 = Span("tool-output-2", 128)
    reasoning = Span("assistant-reasoning", 192)
    a = prompt(sys, history)
    b = prompt(sys, history, tool1)
    c = prompt(sys, history, tool1, tool2)
    rewritten = prompt(sys, history, Span("rewritten-assistant", 192), tool2)
    compact = prompt(sys, Span("compacted-summary", 2048), tool2)
    other = prompt(sys, Span("other-conversation", 8192))
    long = prompt(sys, Span("118k-history", 117760))
    return {
        "tool_loop": (a, b, c),
        "retry_aborted": (a, b, b),
        "compaction": (a, b, compact),
        "rewrite_suffix": (a, prompt(sys, history, reasoning), rewritten),
        "new_user_turn": (a, b, prompt(sys, history, tool1, Span("user-turn", 64))),
        "alternating_conversations": (a, other, b, other, c),
        "subagent_return": (a, other, c),
        "long_118k_tiny_suffix": (long, prompt(sys, Span("118k-history", 117760), Span("tiny-tool", 32))),
        "vision_identity_changed": (
            prompt(sys, Span("vision-patches", 256, "image-A")),
            prompt(sys, Span("vision-patches", 256, "image-B")),
        ),
        "vision_identity_same": (
            prompt(sys, Span("vision-patches", 256, "image-A")),
            prompt(sys, Span("vision-patches", 256, "image-A"), Span("tool-after-image", 32)),
        ),
        "position_identity_changed": (
            prompt(sys, Span("positioned", 128, position_delta=0)),
            prompt(sys, Span("positioned", 128, position_delta=1)),
        ),
    }


def evaluate(prompts, max_checkpoints=6):
    if max_checkpoints < 0:
        raise ValueError("max_checkpoints must be nonnegative")
    resident = None
    # A bounded exact-snapshot comparison arm, not an implementation of NInfer
    # state retention, and not a claim that real checkpoint restoration is possible.
    checkpoints = []
    events = []
    for index, incoming in enumerate(prompts):
        reuse_resident = lcp(resident, incoming) if resident is not None else 0
        options = [reuse_resident] + [lcp(snap, incoming) for snap in checkpoints]
        reuse_snapshot = max(options)
        events.append({
            "step": index,
            "prompt_tokens": len(incoming),
            "resident_lcp_upper_bound": reuse_resident,
            "bounded_snapshot_lcp_upper_bound": reuse_snapshot,
            "resident_unmatched_tokens": len(incoming) - reuse_resident,
            "snapshot_unmatched_tokens": len(incoming) - reuse_snapshot,
            "additional_reusable_tokens_upper_bound": reuse_snapshot - reuse_resident,
        })
        # Persist every prior prompt, latest-first, with deterministic eviction.
        # No inference about KV/GDN validity or memory feasibility.
        if max_checkpoints:
            checkpoints.insert(0, incoming)
            del checkpoints[max_checkpoints:]
        resident = incoming
    return events


def summary(max_checkpoints=6):
    output = {}
    for name, prompts in campaign().items():
        events = evaluate(prompts, max_checkpoints)
        output[name] = {
            "events": events,
            "total_prompt_tokens": sum(x["prompt_tokens"] for x in events),
            "resident_unmatched_tokens": sum(x["resident_unmatched_tokens"] for x in events),
            "snapshot_unmatched_tokens": sum(x["snapshot_unmatched_tokens"] for x in events),
            "additional_reusable_tokens_upper_bound": sum(
                x["additional_reusable_tokens_upper_bound"] for x in events),
        }
    return {"schema": "issue26-prefix-opportunity-v1",
            "synthetic": True,
            "measured_runtime": False,
            "checkpoint_limit": max_checkpoints,
            "scenario_results": output,
            "limitations": [
                "No runtime prefix diagnostics or actual cache-hit counts",
                "No KV/GDN/MTP checkpoint restoration or capacity modeling",
                "No TTFT, prefill latency, cache bandwidth, or GPU measurement",
                "Synthetic token identities are not NInfer tokenizer outputs",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(summary(args.checkpoints), indent=2))


if __name__ == "__main__":
    main()
