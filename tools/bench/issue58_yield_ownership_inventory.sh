#!/usr/bin/env bash
# Read-only reversible-yield ownership inventory. Does not stop C1, use GPU, or rebuild.
set -uo pipefail
LOG="$(mktemp /tmp/issue58-yield-ownership.XXXXXX.log)"
RC=0
{
echo "=== ISSUE58 REVERSIBLE-YIELD OWNERSHIP INVENTORY ==="
echo "UTC=$(date -u -Is)"
echo "PRODUCTION_HTTP=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
sudo -n -u openclaw -H bash <<'SCAN'
set -uo pipefail
ROOT=/home/openclaw/ninfer-issue58-classification
[[ -d "$ROOT/src" ]] || ROOT=/home/openclaw/ninfer-5080-issue32-concurrency
echo "SOURCE_ROOT=$ROOT"
echo "SOURCE_SHA=$(git -C "$ROOT" rev-parse HEAD)"
echo "=== PROGRAM API IMPLEMENTATIONS ==="
grep -RInE --include='*.h' --include='*.hpp' --include='*.cpp'   'void abort_lane\(|resolve_pending_batch\(|decode_batch\(' "$ROOT/include" "$ROOT/src/targets" 2>/dev/null | head -90 || true
echo "=== KV / MTP STATE OWNERSHIP ==="
grep -RInE --include='*.h' --include='*.hpp'   'class .*State|struct .*State|class .*Kv|struct .*Kv|draft.*state|rng.*state|prefix.*owner'   "$ROOT/src/targets/qwen3_6" 2>/dev/null | head -95 || true
echo "=== EXISTING CHECKPOINT / SNAPSHOT / EVICTION APIS ==="
grep -RInE --include='*.h' --include='*.hpp' --include='*.cpp'   'suspend_lane|resume_lane|checkpoint_lane|restore_lane|evict_lane|snapshot_lane'   "$ROOT/src/targets" "$ROOT/include" 2>/dev/null | head -55 || true
SCAN
echo "PRODUCTION_HTTP_FINAL=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
} > "$LOG" 2>&1 || RC=$?
cat "$LOG"
echo "INVENTORY_RC=$RC"
python3 - "$LOG" "$RC" <<'PY'
import base64,pathlib,sys
txt=pathlib.Path(sys.argv[1]).read_text(errors="replace")
txt+="\nINVENTORY_RC="+sys.argv[2]
print("OSC52_CLIPBOARD=ISSUE58_YIELD_OWNERSHIP")
print("\033]52;c;"+base64.b64encode(txt.encode()).decode()+"\a",end="")
PY
