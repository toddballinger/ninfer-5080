#!/usr/bin/env bash
# Issue58 read-only request-contract discovery; run as toddballinger.
set -uo pipefail
REPORT="$(mktemp /tmp/issue58-discovery.XXXXXX.log)"
{
  echo "=== ISSUE58 REQUEST CONTRACT DISCOVERY V2 ==="
  date -u -Is
  echo "=== OPENCLAW ACCOUNT ==="
  sudo -n -u openclaw -H bash -c '
    echo "PATH=$PATH"
    command -v openclaw || true
    for f in /home/openclaw/.npm-global/bin/openclaw /home/openclaw/.local/bin/openclaw /usr/local/bin/openclaw /usr/bin/openclaw; do
      [[ -e "$f" ]] && ls -l "$f"
    done
    for root in /home/openclaw/.npm-global/lib/node_modules/openclaw /home/openclaw/.local/share/pnpm/global/5/node_modules/openclaw /usr/local/lib/node_modules/openclaw /home/openclaw/.openclaw; do
      if [[ -d "$root" ]]; then
        echo "OPENCLAW_ROOT=$root"
        if [[ "$root" == */.openclaw ]]; then
          find "$root" -maxdepth 2 -type f -name "*config*" | head -5
        else
          grep -RIlm1 --include="*.js" --include="*.mjs" --include="*.ts" "max_completion_tokens\|max_tokens" "$root" 2>/dev/null | head -20
        fi
      fi
    done
  ' 2>&1
  echo "=== NINFER SERVE SOURCE ==="
  sudo -n -u openclaw -H bash -c '
    for root in /home/openclaw/ninfer-issue58-trace/src/serve /home/openclaw/ninfer-5080-issue32-concurrency/src/serve; do
      if [[ -d "$root" ]]; then
        echo "NINFER_ROOT=$root"
        grep -RIlm1 --include="*.cpp" --include="*.h" --include="*.hpp" "max_completion_tokens\|max_tokens\|chat/completions" "$root" 2>/dev/null | head -25
      fi
    done
  ' 2>&1
  echo "=== PRODUCTION HEALTH ==="
  curl -sS --max-time 5 -w '\nHTTP=%{http_code}\n' http://127.0.0.1:8080/health
} > "$REPORT" 2>&1
cat "$REPORT"
python3 - "$REPORT" <<'PY'
import base64,pathlib,sys
data=pathlib.Path(sys.argv[1]).read_bytes()
print("OSC52_CLIPBOARD=ISSUE58_DISCOVERY_V2")
print("\033]52;c;"+base64.b64encode(data).decode()+"\a",end="")
PY