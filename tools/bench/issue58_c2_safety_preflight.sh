#!/usr/bin/env bash
# Issue58 non-disruptive preflight: NEVER stops/starts a service or starts a model.
set -uo pipefail
REPORT="$(mktemp /tmp/issue58-preflight.XXXXXX.log)"
rc=0
{
  echo "=== ISSUE58 C2 SAFETY PREFLIGHT ==="; date -u -Is
  echo "USER=$(id -un) UID=$(id -u)"
  if [[ "$(id -un)" != "toddballinger" ]]; then echo "FAIL: must run as toddballinger"; rc=20; fi
  WRAPPER=/usr/local/sbin/openclaw-ninfer-user-service
  BIN=/home/toddballinger/issue58-test/ninfer-serve
  MODEL=/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer
  for x in "$WRAPPER" "$BIN" "$MODEL"; do
     if [[ -e "$x" ]]; then ls -l "$x"; else echo "FAIL: missing $x"; rc=21; fi
  done
  if sudo -n true 2>/dev/null; then echo "SUDO_NONINTERACTIVE=PASS"; else echo "SUDO_NONINTERACTIVE=FAIL"; rc=22; fi
  if sudo -n "$WRAPPER" is-active 2>&1; then echo "WRAPPER_ACTIVE=PASS"; else echo "WRAPPER_ACTIVE=FAIL"; rc=23; fi
  PROD_PID="$(systemctl --user show ninfer-local-model.service -p MainPID --value 2>/dev/null || true)"
  echo "PROD_PID=$PROD_PID"
  if [[ -z "$PROD_PID" || "$PROD_PID" == 0 || ! -r "/proc/$PROD_PID/cgroup" ]]; then echo "FAIL: missing production PID"; rc=24
  else
    ps -p "$PROD_PID" -o user=,args=
    cat "/proc/$PROD_PID/cgroup"
    if ! grep -q '/user-1000.slice/user@1000.service/app.slice/ninfer-local-model.service' "/proc/$PROD_PID/cgroup"; then echo "FAIL: unexpected service cgroup"; rc=25; fi
  fi
  CODE="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health || true)"
  echo "PRODUCTION_HEALTH_HTTP=$CODE"
  [[ "$CODE" == 200 ]] || rc=26
  if ss -lnt | grep -qE '[:.]18080[[:space:]]'; then echo "FAIL: test port 18080 in use"; rc=27; else echo "TEST_PORT_18080=FREE"; fi
  echo "=== WATCHDOG PERMISSION PROBE (NO SERVICE ACTIONS) ==="
  STAMP="/tmp/issue58-watchdog-permission.$$.txt"
  if sudo -n /bin/bash -c 'printf "ROOT_WATCHDOG_PERMISSION_OK\n" > "$1"; chmod 644 "$1"' _ "$STAMP" 2>&1; then
     cat "$STAMP"; sudo -n rm -f "$STAMP"
  else echo "FAIL: cannot create independent root watchdog marker"; rc=28; fi
  echo "=== SCRIPT STATIC SAFETY GATES ==="
  SCRIPT=/tmp/issue58_supervised_c2_probe.sh
  if [[ -r "$SCRIPT" ]] && bash -n "$SCRIPT"; then echo "SCRIPT_SYNTAX=PASS"; else echo "SCRIPT_SYNTAX=FAIL"; rc=29; fi
  echo "=== DRY RUN RESULT ==="
  if [[ "$rc" -eq 0 ]]; then echo "PREFLIGHT=PASS"; else echo "PREFLIGHT=FAIL RC=$rc"; fi
} > "$REPORT" 2>&1
cat "$REPORT"
python3 - "$REPORT" <<'PY'
import base64, pathlib, sys
body=pathlib.Path(sys.argv[1]).read_text(errors="replace")
print("REPORT_PATH="+sys.argv[1])
print("OSC52_CLIPBOARD=REPORT")
print("\033]52;c;"+base64.b64encode(body.encode()).decode()+"\a",end="")
PY
# Return outcome without closing the caller's interactive SSH session.
exit "$rc"
