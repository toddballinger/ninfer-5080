#!/usr/bin/env bash
# Read-only narrow, line-numbered source extracts for an exact suspend/resume design.
# No GPU use, build, OpenClaw calls, or production service changes.
set -uo pipefail
LOG="$(mktemp /tmp/issue58-target-state.XXXXXX.log)"
RC=0
{
 echo "=== ISSUE58 TARGET STATE OWNERSHIP ==="
 echo "UTC=$(date -u -Is)"
 sudo -n -u openclaw -H python3 - <<'PY'
from pathlib import Path
root=Path("/home/openclaw/ninfer-issue58-classification")
cases={
 "src/targets/qwen3_6/impl/runtime/program.h":["struct SequenceKVBundle","struct SequenceState","struct RequestControl"],
 "src/targets/qwen3_6/impl/runtime/program_impl.h":["void ProgramImplCore::abort_lane","void ProgramImplCore::resolve_pending_batch","void ProgramImplCore::resolve_non_speculative_pending"],
 "src/targets/qwen3_6/impl/runtime/linear_state_slots.h":["struct LinearStateSlots"],
}
for name,anchors in cases.items():
 p=root/name
 print(f"FILE={name} PRESENT={p.is_file()}")
 if not p.is_file():continue
 lines=p.read_text(errors="replace").splitlines()
 for anchor in anchors:
  indices=[i for i,v in enumerate(lines) if anchor in v]
  print(f"ANCHOR={anchor!r} MATCHES={len(indices)}")
  for i in indices[:1]:
   start=max(0,i-3); end=min(len(lines),i+32)
   for j in range(start,end): print(f"{j+1:05d} {lines[j]}")
PY
 echo "PRODUCTION_HTTP=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
} >"$LOG" 2>&1 || RC=$?
cat "$LOG"
echo "STATE_INVENTORY_RC=$RC"
python3 - "$LOG" "$RC" <<'PY'
import base64,pathlib,sys
data=pathlib.Path(sys.argv[1]).read_text(errors="replace")
data+="\nSTATE_INVENTORY_RC="+sys.argv[2]
print("OSC52_CLIPBOARD=ISSUE58_TARGET_STATE")
print("\033]52;c;"+base64.b64encode(data.encode()).decode()+"\a",end="")
PY