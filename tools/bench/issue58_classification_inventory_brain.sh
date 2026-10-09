#!/usr/bin/env bash
# Issue #58: read-only privacy-preserving request classification inventory.
# No service operations, no prompts or completions printed. OSC52 copy.
set -uo pipefail
LOG=/tmp/ninfer-thinking-budget-final-20260916-150239.jsonl
REPORT="$(mktemp /tmp/issue58-classification.XXXXXX.log)"
{
  echo "=== ISSUE58 OPENCLAW CLASSIFICATION INVENTORY ==="
  date -u -Is
  echo "PRODUCTION_HTTP=$(curl -sS --max-time 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
  python3 - "$LOG" <<'PY'
import collections,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
print("LOG_PRESENT="+str(p.is_file()))
if not p.is_file(): raise SystemExit(0)
print("LOG_BYTES="+str(p.stat().st_size))
keys=collections.Counter()
requested=collections.Counter()
effective=collections.Counter()
rows=0
def walk(v, depth=0):
    if depth>6:return
    if isinstance(v,dict):
        for k,x in v.items():
            key=str(k).lower()
            keys[key]+=1
            if isinstance(x,int) and not isinstance(x,bool) and 0<=x<=262144:
                if key in ("max_tokens","max_completion_tokens","max_output_tokens","requested_max_tokens","requested_output_tokens"):
                    requested[(key,x)]+=1
                if key in ("effective_output_tokens","effective_max_output_tokens"):
                    effective[(key,x)]+=1
            if isinstance(x,(dict,list)):walk(x,depth+1)
    elif isinstance(v,list):
        for item in v[:12]:walk(item,depth+1)
with p.open(errors="replace") as fh:
    for line in fh:
        rows+=1
        try: walk(json.loads(line))
        except (ValueError,UnicodeError):pass
print("ROWS="+str(rows))
print("REQUESTED_OUTPUT_BUDGET_DISTRIBUTION:")
for (key,num),count in sorted(requested.items(),key=lambda x:(x[0][0],x[0][1])):
    print(f"  {key}={num} count={count}")
print("EFFECTIVE_OUTPUT_BUDGET_DISTRIBUTION:")
for (key,num),count in sorted(effective.items(),key=lambda x:(x[0][0],x[0][1])):
    print(f"  {key}={num} count={count}")
print("BUDGET_FIELDS_FOUND="+str(bool(requested or effective)))
print("AVAILABLE_BUDGET_KEY_NAMES="+",".join(sorted(k for k in keys if any(t in k for t in ("token","budget","output"))))[:1200])
PY
} > "$REPORT" 2>&1
cat "$REPORT"
python3 - "$REPORT" <<'PY'
import base64,pathlib,sys
s=pathlib.Path(sys.argv[1]).read_bytes()
print("OSC52_CLIPBOARD=ISSUE58_CLASSIFICATION")
print("\033]52;c;"+base64.b64encode(s).decode()+"\a",end="")
PY
