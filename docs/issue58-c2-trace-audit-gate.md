# Issue #58 — C2 evidence audit and supervised candidate run gate

## Scope
This PR prepares an **offline, CPU-only** audit of diagnostic stderr records added by merged PRs #73–#76 and numerical KV-denial trace added by #75. It **does not start a C2 server or run workloads**. Production C1 port 8080 stays unchanged. This work does not establish scheduler causal root cause or C2 release readiness.

Use `tools/bench/issue58_c2_trace_audit.py` on saved **raw** stderr captures. Report retains source/line identifiers for selected-admission timing, counts exact lane/policy/KV records and reports p50/p95/max queue residence among **admitted requests only**. It does not pretend the sampled trace records share a reliable request key: KV-denial records lack request IDs; lane observations and KV causes are different probes, potentially throttled independently. Never infer causal relationships from proximity alone. Rejected, timed-out, cancelled, and never-admitted requests are excluded from queue-time distribution (survivor bias). Existing `issue58_admission_forensics.py` separately reports completed TTFT only; do not merge its samples without verified identity and capture scope.

## Host-only validation
Verify exact PR HEAD, clean worktree, Python version, and three-file scope:
```bash
python3 -m unittest tests/test_issue58_c2_trace_audit.py -v
python3 -m py_compile tools/bench/issue58_c2_trace_audit.py tests/test_issue58_c2_trace_audit.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```
Report all RCs. Review parsers against actual source formats in `concurrent_executor.h` and `program_impl.h`. For a saved capture:
```bash
python3 tools/bench/issue58_c2_trace_audit.py /path/to/saved/raw.stderr.log > /tmp/issue58-trace-audit.json
```
Do not invent a capture path. No service access is required to validate the parser.

## Separate, explicitly authorised candidate C2 run (NOT authorised by this PR)
Before ANY candidate GPU run, require owner approval of exact separate staging binary/model build, 18080-only port (never production 8080), isolation and rollback strategy, root watchdog and C1 health checks before/after, CPU/GPU resource budgets, timeout ceilings, 128K context expectation, and specific short/long interleaved workload. Historical supervised tests stopped C1 and restored it; **do not repeat such interruption without separate approval**. No auto-run, no use of `ninfer-local-model.service` control commands here.

Candidate goals (only after approval): capture raw stderr for the same staged server instance with `NINFER_ADMISSION_TRACE=1`, full server JSONL with known instance boundary, client request schedule and exact per-request arrival/TTFT/failure outcomes. Compare C1 and C2 only with paired workloads and count all failures. Report the old 600+ second TTFT tail and false short-hint starvation separately. Diagnostic traces are sampled, do not assert a KV shortage from `not_admittable` alone.

## Next
Open a separate supervised experiment protocol PR if needed; only after one run establishes specific, reproducible denial behavior should scheduler modifications be proposed. Keep Issue #58 OPEN.
