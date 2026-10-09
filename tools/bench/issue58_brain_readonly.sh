#!/usr/bin/env bash
# Issue #58: read-only forensic analysis; OSC52 copies the report to clipboard.
# Run as openclaw; no service/gateway/GPU modifications.
set -euo pipefail
REPO="${NINFER_REPO:-/home/openclaw/ninfer-5080-issue32-concurrency}"
SCRIPT="$REPO/tools/bench/issue58_admission_forensics.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: missing $SCRIPT (checkout the issue58-admission-forensics branch or use --script location)" >&2
  exit 2
fi
C1="/home/openclaw/issue32-c1-campaigns/c1-20261008T150834Z-818276"
C2="/home/openclaw/issue32-c2-corpus/corpus-c2-20261008T232609Z-952752"
C3="/home/openclaw/issue32-c3-corpus/corpus-c3-20261009T005806Z-985819"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
python3 "$SCRIPT" "$C1" "$C2" "$C3" > "$tmp"
cat "$tmp"
# OSC52, with chunking avoided: full 3-run JSON may exceed terminal clipboard caps.
# Copy a compact verified report to clipboard. Full details remain above.
python3 - "$tmp" <<'PY'
import base64, json, sys
d=json.load(open(sys.argv[1]))
out=["#58 ADMISSION SUMMARY"]
for name, v in d.items():
    t=v["ttft_seconds"]
    out.append(f"{name}: completed={v['request_done_events']} errors={v['request_error_events']} timeouts={v['queue_timeout_events']} TTFT_p50={t['p50']}s p95={t['p95']}s max={t['max']}s")
    out.append(f"  states_seconds={v['running_waiting_seconds']}")
    out.append(f"  long_waits={v['ttft_over_seconds']}")
out.append("TTFT is not queue-only admission wait; pending exact resource-deferral instrumentation.")
txt="\n".join(out)
sys.stdout.write("\033]52;c;"+base64.b64encode(txt.encode()).decode()+"\a")
print("\nOSC52_CLIPBOARD=COMPACT_SUMMARY")
PY
