#!/usr/bin/env python3
"""Read-only Issue #58 admission/TTFT analysis of NInfer serving captures.

Accepts a directory containing server/*.jsonl and forensics/*.stderr.log.
No GPU access, requests, service operations, or modifications.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

DONE = re.compile(r"\[req (\d+)\] done .*?\bttft=([\d.]+)ms")
STATE = re.compile(r"\bthroughput interval=([\d.]+)s .*?\brunning=(\d+).*?\bwaiting=(\d+)")
EVENT = re.compile(r"^\[([^]]+)\]")
def quantile(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    x = (len(v)-1)*p
    a = math.floor(x)
    b = math.ceil(x)
    return v[a]+(v[b]-v[a])*(x-a)

def analyse(root: Path) -> dict:
    paths = sorted((root/"server").glob("*.jsonl"))
    logs = sorted((root/"forensics").glob("*.stderr.log"))
    if not logs and (root/"supervision"/"runner_stderr.log").exists():
        logs = [root/"supervision"/"runner_stderr.log"]
    if not paths or not logs:
        raise ValueError(f"missing server JSONL or stderr capture in {root}")
    events=[]
    for path in paths:
        for i,line in enumerate(path.read_text(errors="replace").splitlines(),1):
            if not line.strip():
                continue
            try:
                e=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: invalid JSONL") from exc
            if isinstance(e,dict):
                events.append(e)
    starts={}
    completed={}
    errors=[]
    for e in events:
        kind=e.get("event")
        req=e.get("request") or {}
        rid=req.get("request_id",e.get("request_id"))
        if kind=="request_start" and rid is not None:
            starts[rid]=e
        elif kind=="request_done" and rid is not None:
            completed[rid]=e
        elif kind=="request_error":
            errors.append(e)
    ttft={}
    states=Counter()
    occupied_s=Counter()
    log_examples={}
    for path in logs:
        for line in path.read_text(errors="replace").splitlines():
            m=DONE.search(line)
            if m:
                rid=int(m.group(1))
                ttft[rid]=float(m.group(2))/1000
                log_examples[rid]=line.strip()
            m=STATE.search(line)
            if m:
                seconds=float(m.group(1))
                running=int(m.group(2))
                waiting=int(m.group(3))
                states[(running,waiting)]+=1
                occupied_s[(running,waiting)]+=seconds
    # Derive TTFT only for completed server requests: avoid accidentally
    # treating log fragments/failed requests as a valid sample.
    values=[v for k,v in ttft.items() if k in completed]
    slow=sorted(((v,k) for k,v in ttft.items() if k in completed),reverse=True)[:10]
    timeout=[e for e in errors if "waiting for admission" in json.dumps(e)
             or "request_queue_timeout" in json.dumps(e)]
    summary={
        "root":str(root),
        "request_start_events":len(starts),
        "request_done_events":len(completed),
        "request_error_events":len(errors),
        "queue_timeout_events":len(timeout),
        "ttft_samples":len(values),
        "ttft_seconds":{
            "p50":quantile(values,.5),"p90":quantile(values,.9),
            "p95":quantile(values,.95),"p99":quantile(values,.99),
            "max":max(values) if values else None},
        "ttft_over_seconds":{str(t):sum(v>t for v in values) for t in (1,5,30,60,180,300,600,900)},
        "running_waiting_seconds":{
            f"{r}:{w}":round(secs,3) for (r,w),secs in sorted(occupied_s.items())
        },
        "running_waiting_samples":{
            f"{r}:{w}":n for (r,w),n in sorted(states.items())
        },
        "slowest_ttft":[{"request_id":k,"ttft_seconds":v,"log":log_examples[k]} for v,k in slow],
        "timeout_requests":[{"request_id":(e.get("request") or {}).get("request_id"),
                             "error":e.get("error")} for e in timeout],
        "limitations":[
            "TTFT includes admission, prefill and initial decode; it is NOT pure queue time.",
            "Only successful completions enter TTFT percentiles; failures are separately counted.",
            "One-second throughput states show running/waiting, not why admission was deferred.",
            "Different request completion sets are not a paired performance comparison.",
        ],
    }
    if len(ttft)!=len(completed):
        summary["warning"]=f"stderr TTFT entries={len(ttft)}, JSONL done={len(completed)}; report may be incomplete"
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path,help="saved campaign root directories")
    p.add_argument("--output",type=Path,help="optional JSON output; never overwrites unless --force")
    p.add_argument("--force",action="store_true")
    a=p.parse_args()
    results={r.name:analyse(r) for r in a.runs}
    raw=json.dumps(results,indent=2,sort_keys=True)
    print(raw)
    if a.output:
        if a.output.exists() and not a.force:
            p.error(f"output already exists: {a.output} (use --force to replace)")
        a.output.write_text(raw+"\n")

if __name__=="__main__":
    main()
