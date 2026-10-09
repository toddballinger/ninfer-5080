#!/usr/bin/env bash
set -uo pipefail
LOG=/tmp/ninfer-thinking-budget-final-20260916-150239.jsonl
REPORT="$(mktemp /tmp/issue58-budget-real.XXXXXX.log)"
{
  echo "=== ISSUE58 REAL COMPLETION VS REQUESTED BUDGET ==="
  date -u -Is
  python3 - "$LOG" <<'PY'
import json,sys,pathlib,collections
p=pathlib.Path(sys.argv[1])
if not p.exists(): print("LOG_MISSING"); raise SystemExit(2)
def unique_ints(obj,field):
    result=[]
    def go(x,depth=0):
        if depth>5:return
        if isinstance(x,dict):
            if isinstance(x.get(field),int) and not isinstance(x[field],bool): result.append(x[field])
            for v in x.values():
                if isinstance(v,(dict,list)):go(v,depth+1)
        elif isinstance(x,list):
            for v in x[:8]:go(v,depth+1)
    go(obj)
    return list(dict.fromkeys(result))
seen=collections.Counter(); request=collections.Counter(); pairs=collections.Counter(); buckets=collections.Counter(); sample=collections.Counter()
parsed=0
with p.open(errors="replace") as fh:
    for line in fh:
        try:o=json.loads(line)
        except (ValueError,UnicodeError):continue
        parsed+=1
        for k in ("requested_output_tokens","completion_tokens","reasoning_tokens","requested_output_tokens_source"):
            if isinstance(o,dict) and k in o:sample[k]+=1
        asked=unique_ints(o,"requested_output_tokens")
        completed=unique_ints(o,"completion_tokens")
        for n in asked:request[n]+=1
        if len(asked)==1 and len(completed)==1:
            n,m=asked[0],completed[0]
            pairs[(n,m)]+=1
            if n==16384:
                boundaries=[128,256,512,1024,2048,4096,8192,16384]
                label=next((f"<= {b}" for b in boundaries if m<=b),">16384")
                buckets[label]+=1
                seen["paired_16384"]+=1
        if len(asked)==1:seen["single_requested"]+=1
        if len(completed)==1:seen["single_completion"]+=1
print("PARSED_ROWS="+str(parsed))
print("UNIQUE_REQUEST_16384_ROWS="+str(request[16384]))
print("PAIRED_REQUEST_16384_COMPLETION="+str(seen["paired_16384"]))
print("SINGLE_COMPLETION_ROWS="+str(seen["single_completion"]))
print("REQUEST_16384_ACTUAL_COMPLETION_BUCKETS:")
for k in ["<= 128","<= 256","<= 512","<= 1024","<= 2048","<= 4096","<= 8192","<= 16384",">16384"]:
    print(f"  {k}: {buckets[k]}")
print("NOTE: this report does not distinguish OpenClaw from other clients unless log identifies origin; actual completion cannot be known before scheduling")
PY
} > "$REPORT" 2>&1
cat "$REPORT"
python3 - "$REPORT" <<'PY'
import base64,pathlib,sys
body=pathlib.Path(sys.argv[1]).read_bytes()
print("OSC52_CLIPBOARD=ISSUE58_BUDGET_REALITY")
print("\033]52;c;"+base64.b64encode(body).decode()+"\a",end="")
PY