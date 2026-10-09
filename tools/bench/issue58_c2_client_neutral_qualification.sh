#!/usr/bin/env bash
# Issue #58 — supervised, bounded C2 admission diagnostic.
# This intentionally interrupts local NInfer C1 service for up to 6 minutes.
# Never deploys/reconfigures the production unit. Requires sudo -n access.
# Independent root watchdog restores C1 even if the SSH session disappears.
set -Eeuo pipefail
export GIT_PAGER=cat PAGER=cat GIT_TERMINAL_PROMPT=0
UNIT_WRAPPER=/usr/local/sbin/openclaw-ninfer-user-service
BIN=/home/toddballinger/issue58-test/ninfer-classification-serve
MODEL=/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer
PORT=18080
ROOT="$(mktemp -d /tmp/issue58-c2-XXXXXX)"
LOG="$ROOT/report.txt"
SERVERLOG="$ROOT/server.log"
READY="$ROOT/armed"
DISARM="$ROOT/disarmed"
TESTPIDFILE="$ROOT/test.pid"
RESTORED=0

copy_report() {
  echo "REPORT_DIR=$ROOT" | tee -a "$LOG"
  python3 - "$LOG" <<'PY'
import base64, pathlib, sys
p=pathlib.Path(sys.argv[1])
data=p.read_text(errors="replace")
lines=data.splitlines()
# Keep within terminal OSC52 clipboard limits, prioritizing diagnosis and tail.
summary="\n".join(lines[-110:])
print("OSC52_CLIPBOARD=ISSUE58_C2_DIAG")
print("\033]52;c;"+base64.b64encode(summary.encode()).decode()+"\a",end="")
PY
}
restore() {
  local code=$?
  trap - EXIT INT TERM HUP
  echo "=== RESTORE PRODUCTION C1 ===" | tee -a "$LOG"
  if [[ -s "$TESTPIDFILE" ]]; then
    local tp; tp="$(cat "$TESTPIDFILE")"
    kill -TERM "$tp" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "$tp" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$tp" 2>/dev/null || true
  fi
  if sudo -n "$UNIT_WRAPPER" start >>"$LOG" 2>&1; then
    for _ in $(seq 1 45); do
      if curl -fsS --max-time 2 http://127.0.0.1:8080/health >>"$LOG" 2>&1; then
        echo "PRODUCTION_RESTORED=PASS" | tee -a "$LOG"
        RESTORED=1
        touch "$DISARM"
        break
      fi
      sleep 2
    done
  fi
  if [[ "$RESTORED" -ne 1 ]]; then
    echo "PRODUCTION_RESTORED=FAIL - CHECK WRAPPER MANUALLY" | tee -a "$LOG"
  fi
  echo "=== TRACE EVENTS ===" | tee -a "$LOG"
  grep -E '\[ADMISSION-TRACE\]|request_queue_timeout|waiting for admission|error' "$SERVERLOG" | tail -60 | tee -a "$LOG" || true
  echo "TEST_EXIT_CODE=$code" | tee -a "$LOG"
  copy_report
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

{
echo "=== ISSUE58 C2 SUPERVISED DIAG ==="
date -u -Is
echo "REPORT_DIR=$ROOT"
echo "=== PREFLIGHT ==="
[[ "$(id -un)" == "toddballinger" ]] || { echo "STOP: run as toddballinger"; exit 20; }
[[ -x "$BIN" && -r "$MODEL" && -x "$UNIT_WRAPPER" ]] || { echo "STOP: binary/model/wrapper missing"; exit 21; }
sudo -n true || { echo "STOP: sudo noninteractive unavailable"; exit 22; }
command -v timeout >/dev/null || { echo "STOP: timeout missing"; exit 23; }
curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null || { echo "STOP: production unhealthy"; exit 24; }
if pgrep -f '^/home/toddballinger/issue58-test/ninfer-classification-serve ' >/dev/null; then echo "STOP: instrumented server already running"; exit 25; fi
[[ "$(systemctl --user show ninfer-local-model.service -p ActiveState --value)" == active ]] || { echo "STOP: production unit not active"; exit 26; }
[[ "$(systemctl --user show ninfer-local-model.service -p MainPID --value)" != 0 ]] || { echo "STOP: no production PID"; exit 27; }
ss -lnt | grep -q ":$PORT " && { echo "STOP: test port busy"; exit 28; }
echo "PREFLIGHT=PASS"
echo "=== ARM INDEPENDENT WATCHDOG ==="
# sudo creates a *root-owned* independent timer. It only acts if not disarmed.
sudo -n bash -c 'nohup bash -c '"'"'
  root="$1"; wrapper="$2"
  sleep 360
  [[ -e "$root/armed" && ! -e "$root/disarmed" ]] || exit 0
  echo "WATCHDOG_FIRED=$(date -u -Is)" >>"$root/watchdog.log"
  if [[ -s "$root/test.pid" ]]; then
      pid="$(cat "$root/test.pid")"
      if [[ -r "/proc/$pid/cmdline" ]] && tr "\0" " " < "/proc/$pid/cmdline" | grep -q "^/home/toddballinger/issue58-test/ninfer-classification-serve "; then
          kill -TERM "$pid" 2>/dev/null || true
      fi
  fi
  sleep 5
  "$wrapper" start >>"$root/watchdog.log" 2>&1 || true
  echo "WATCHDOG_RESTORE_ATTEMPTED=$(date -u -Is)" >>"$root/watchdog.log"
'"'"' _ "$1" "$2" >/dev/null 2>&1 </dev/null &' _ "$ROOT" "$UNIT_WRAPPER"
touch "$READY"
echo "=== STOP ORIGINAL C1 (WRAPPER) ==="
sudo -n "$UNIT_WRAPPER" stop
echo "=== WAIT FOR GPU RELEASE ==="
for _ in $(seq 1 40); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
  sleep 1
done
echo "=== START C2 TRACED TEST ON PORT $PORT ==="
NINFER_ADMISSION_TRACE=1 NINFER_SHORT_LANE_RESERVE=1 NINFER_TRUST_SHORT_OPERATION_HINT=1 NINFER_ADMISSION_CLASS_TRACE=1 NINFER_LONG_OUTPUT_THRESHOLD=1024 "$BIN" "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --model-id local-model \
  --max-context 131072 --kv-capacity 131072 --prefill-chunk 1792 \
  --kv-dtype q4 --spec mtp --draft-tokens 3 \
  --max-concurrency 2 --max-pending-requests 16 --pending-timeout-ms 180000 \
  --vision --vision-max-tokens 2048 --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool --embedding-host \
  >"$SERVERLOG" 2>&1 &
TESTPID=$!
echo "$TESTPID" > "$TESTPIDFILE"
echo "TEST_PID=$TESTPID"
echo "=== WAIT TEST HEALTH (MAX 90s) ==="
ready=0
for _ in $(seq 1 45); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$TESTPID" 2>/dev/null || break
  sleep 2
done
[[ "$ready" -eq 1 ]] || { echo "TEST_STARTUP_FAILED"; tail -50 "$SERVERLOG"; exit 30; }
echo "TEST_READY=PASS"
echo "=== HTTP REQUEST-SHAPE COMPATIBILITY GATE ==="
python3 - "$PORT" <<'PY'
import json,sys,urllib.request,urllib.error
url=f"http://127.0.0.1:{int(sys.argv[1])}/v1/chat/completions"
base={"model":"local-model","messages":[{"role":"user","content":"reply OK"}],"max_tokens":32}
for name,field in (("hint_invalid_string","yes"),("hint_invalid_number",1),("hint_invalid_array",[])):
  body={**base,"ninfer_short_operation":field}
  req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
  try:
    with urllib.request.urlopen(req,timeout=6) as response:
      code=response.status
  except urllib.error.HTTPError as exc:
    code=exc.code
  print(f"{name}_HTTP={code}",flush=True)
  assert code==400,f"{name}: unexpected HTTP {code}"
print("API_TYPE_REGRESSION=PASS",flush=True)
PY
echo "=== CLIENT-NEUTRAL C2 ADMISSION QUALIFICATION (MAX 150s) ==="
timeout --signal=TERM --kill-after=5s 150s python3 - "$PORT" <<'PY'
import concurrent.futures,json,sys,time,urllib.request,urllib.error
url=f"http://127.0.0.1:{int(sys.argv[1])}/v1/chat/completions"
def run(name,budget,hint,delay):
    time.sleep(delay)
    t=time.monotonic()
    long_prompt="Write a detailed numbered guide to concurrency and queue scheduling with many examples; continue until the output limit."
    prompt=long_prompt if name.startswith("long") or name.startswith("misclassified") else "Say OK and stop."
    payload={"model":"local-model","messages":[{"role":"user","content":prompt}],"max_tokens":budget,"stream":True,"temperature":0.1}
    if hint is not None:payload["ninfer_short_operation"]=hint
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer local-model"})
    result={"name":name,"budget":budget,"hint":hint}
    try:
        with urllib.request.urlopen(req,timeout=130) as resp:
            result["http"]=resp.status
            first=None;finish=None
            for line in resp:
                if not line.startswith(b"data:"):continue
                data=line[5:].strip()
                if data==b"[DONE]":break
                try: packet=json.loads(data)
                except (ValueError,UnicodeDecodeError):continue
                choice=(packet.get("choices") or [{}])[0]
                delta=choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):first=round(time.monotonic()-t,3)
                if choice.get("finish_reason"):finish=choice["finish_reason"]
            result.update(ttft=first,finish=finish,elapsed=round(time.monotonic()-t,3))
    except Exception as exc:result["error"]=str(exc)
    print(json.dumps(result),flush=True)
    return result
def batch(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        futs=[pool.submit(run,*x) for x in items]
        results={r["name"]:r for r in (f.result() for f in futs)}
    for name,r in results.items():
        assert r.get("http")==200 and r.get("ttft") is not None and r.get("finish") in ("stop","length"),f"request failed: {r}"
    return results
print("=== CASE A: absent/false 16K hint stays conservative; short backfills ===",flush=True)
a=batch([("long_a",1536,None,0),("unhinted_16384",16384,None,.1),("short_256",256,False,2)])
assert a["short_256"]["ttft"]<5,f"short blocked: {a}"
assert a["unhinted_16384"]["ttft"]>a["long_a"]["ttft"]+3,f"unhinted 16K misclassified: {a}"
print("UNHINTED_CLIENT_REGRESSION=PASS",flush=True)
print("=== CASE B: an explicitly false long hint demonstrates residual starvation risk ===",flush=True)
b=batch([("long_b",1536,None,0),("misclassified_long",1536,True,.1),("short_after_bad_hint",256,False,2)])
# A falsely hinted long occupies second C2 lane; do not misrepresent this as fair.
assert b["misclassified_long"]["ttft"]<5,f"hint not honored: {b}"
assert b["short_after_bad_hint"]["ttft"]>3,f"unexpected preemption in nonpreemptive scheduler: {b}"
print("FALSE_HINT_RISK_REPRODUCED=PASS",flush=True)
print("C2_QUALIFICATION_TEST=PASS",flush=True)
PY
echo "=== C2 DIAGNOSTIC COMPLETE ==="
} 2>&1 | tee -a "$LOG"
