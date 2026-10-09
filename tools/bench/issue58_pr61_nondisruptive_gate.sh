#!/usr/bin/env bash
# No GPU/service interruption. Confirm exact built binary and run available CPU test suite.
set -uo pipefail
BUILD=/home/openclaw/ninfer-issue58-classification-build
SOURCE=/home/openclaw/ninfer-issue58-classification
REPORT="$(mktemp /tmp/issue58-cpu-gate.XXXXXX.log)"
RC=0
{
  echo "=== ISSUE58 PR61 NON-DISRUPTIVE RELEASE GATES ==="
  echo "UTC=$(date -u -Is)"
  echo "PRODUCTION_HTTP=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
  sudo -n -u openclaw -H bash -s -- "$SOURCE" "$BUILD" <<'INTERNAL'
set -uo pipefail
SOURCE="$1"; BUILD="$2"
echo "SOURCE_SHA=$(git -C "$SOURCE" rev-parse HEAD)"
echo "SOURCE_STATUS_BEGIN"
git -C "$SOURCE" status --short
echo "SOURCE_STATUS_END"
echo "BINARY_SHA256=$(sha256sum "$BUILD/apps/ninfer-serve" | awk '{print $1}')"
echo "TEST_LIST_BEGIN"
ctest --test-dir "$BUILD" -N 2>&1 | tail -40
echo "TEST_LIST_END"
INTERNAL
  echo "PRODUCTION_HTTP_FINAL=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
} >"$REPORT" 2>&1 || RC=$?
cat "$REPORT"
echo "DIAGNOSTIC_RC=$RC"
python3 - "$REPORT" "$RC" <<'PY'
import base64,pathlib,sys
data=pathlib.Path(sys.argv[1]).read_text(errors="replace")
data+="\nDIAGNOSTIC_RC="+sys.argv[2]+"\n"
print("OSC52_CLIPBOARD=ISSUE58_CPU_GATE")
print("\033]52;c;"+base64.b64encode(data.encode()).decode()+"\a",end="")
PY