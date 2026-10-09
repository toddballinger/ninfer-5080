#!/usr/bin/env bash
# Issue 58 build-only validation. NO GPU serving, service stops, or workloads.
# Logs stay on Brain and a compact report is copied via OSC52 even if build fails.
set -uo pipefail
SOURCE="${NINFER_SOURCE:-/home/openclaw/ninfer-5080-issue32-concurrency}"
WORKTREE="${NINFER_ISSUE58_WORKTREE:-/home/openclaw/ninfer-issue58-trace}"
BUILD="${NINFER_ISSUE58_BUILD:-/home/openclaw/ninfer-issue58-trace-build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="/tmp/issue58-trace-build-${STAMP}-$$.log"
RC=0
{
  echo "=== ISSUE58 ADMISSION TRACE BUILD ==="
  date -u
  echo "SOURCE=$SOURCE"
  git -C "$SOURCE" fetch origin issue58-source-admission-triage || exit 10
  if [[ ! -e "$WORKTREE/.git" ]]; then
     if [[ -e "$WORKTREE" ]]; then
        echo "ERROR: existing non-worktree path $WORKTREE; refusing to overwrite" ; exit 11
     fi
     git -C "$SOURCE" worktree add --detach "$WORKTREE" origin/issue58-source-admission-triage || exit 12
  fi
  echo "WORKTREE=$(git -C "$WORKTREE" rev-parse --show-toplevel)"
  echo "HEAD=$(git -C "$WORKTREE" rev-parse HEAD)"
  echo "=== DIFF VALIDATION ==="
  git -C "$WORKTREE" diff --check HEAD~1 HEAD || exit 13
  echo "=== CMAKE CONFIGURE ==="
  cmake -S "$WORKTREE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release || exit 14
  echo "=== NINFER SERVE COMPILE (NO EXECUTION) ==="
  cmake --build "$BUILD" --target ninfer-serve -j 4 || exit 15
  echo "BUILD_RESULT=PASS"
} >"$REPORT" 2>&1 || RC=$?
cat "$REPORT"
python3 - "$REPORT" "$RC" <<'PY'
import base64,sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text(errors="replace").splitlines()
rc=sys.argv[2]
tail="\n".join(lines[-50:])
report=f"ISSUE58 TRACE BUILD RC={rc}\nREPORT={sys.argv[1]}\n"+tail
print("\nOSC52_CLIPBOARD=BUILD_SUMMARY")
sys.stdout.write("\033]52;c;"+base64.b64encode(report.encode()).decode()+"\a")
PY
echo "REPORT_PATH=$REPORT"
echo "VALIDATION_RC=$RC"
# No 'exit' at the end of the interactive parent session; this script process returns.
