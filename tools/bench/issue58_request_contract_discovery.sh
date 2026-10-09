#!/usr/bin/env bash
# Read-only: discover OpenClaw -> OpenAI-compatible request construction and NInfer request parser.
set -uo pipefail
export GIT_PAGER=cat PAGER=cat
REPORT="$(mktemp /tmp/issue58-request-contract.XXXXXX.log)"
{
  echo "=== ISSUE58 REQUEST-CONTRACT DISCOVERY ==="
  date -u -Is
  echo "=== OPENCLAW INSTALL ==="
  command -v openclaw || true
  openclaw --version 2>&1 | head -4 || true
  echo "=== CALLER PACKAGE ROOTS ==="
  for root in /home/openclaw/.npm-global/lib/node_modules/openclaw /usr/local/lib/node_modules/openclaw /home/openclaw/.local/share/pnpm/global/5/node_modules/openclaw /home/toddballinger/.npm-global/lib/node_modules/openclaw; do
    if [[ -d "$root" ]]; then
      echo "ROOT=$root"
      rg -l -m1 --glob '*.{js,mjs,cjs,ts}' 'max_completion_tokens|max_tokens|chat/completions' "$root" 2>/dev/null | head -12
    fi
  done
  echo "=== NINFER ROUTE/PARSER FILES ==="
  for root in /home/openclaw/ninfer-issue58-trace/src/serve /home/openclaw/ninfer-5080-issue32-concurrency/src/serve; do
    if [[ -d "$root" ]]; then
      echo "ROOT=$root"
      rg -l -m1 --glob '*.{cpp,h,hpp}' 'max_completion_tokens|max_tokens|chat/completions' "$root" 2>/dev/null | head -20
    fi
  done
  echo "=== PRODUCTION (UNCHANGED) ==="
  curl -sS --max-time 4 -w '\nHTTP=%{http_code}\n' http://127.0.0.1:8080/health
} > "$REPORT" 2>&1
cat "$REPORT"
python3 - "$REPORT" <<'PY'
import base64,pathlib,sys
v=pathlib.Path(sys.argv[1]).read_bytes()
print("OSC52_CLIPBOARD=ISSUE58_REQUEST_CONTRACT")
print("\033]52;c;"+base64.b64encode(v).decode()+"\a",end="")
PY
