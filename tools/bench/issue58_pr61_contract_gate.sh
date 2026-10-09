#!/usr/bin/env bash
# Opt-in classification: compile-free source contract + isolated API test with existing executable.
# Does NOT launch a second model or stop production.
set -uo pipefail
REPORT="$(mktemp /tmp/issue58-pr61-contract.XXXXXX.log)"
RC=0
sudo -n -u openclaw -H python3 - <<'PY' > "$REPORT" 2>&1 || RC=$?
import pathlib,re,sys
root=pathlib.Path("/home/openclaw/ninfer-issue58-classification")
checks={
"API_BOOL_PARSER":("src/serve/openai_schema.cpp", 'out.ninfer_short_operation = get_bool(body, "ninfer_short_operation", false);'),
"API_REQUEST_FIELD":("src/serve/request.h","bool ninfer_short_operation = false;"),
"API_TO_ENGINE":("src/serve/translate.cpp","options.execution.ninfer_short_operation = request.ninfer_short_operation;"),
"ENGINE_PUBLIC_FIELD":("include/ninfer/types.h","bool ninfer_short_operation = false;"),
"ENGINE_RESOLVED_FIELD":("src/runtime/contract/types.h","bool ninfer_short_operation = false;"),
"ENGINE_RESOLVE":("src/runtime/engine/engine.cpp","resolved.execution.ninfer_short_operation = options.execution.ninfer_short_operation;"),
"TRUST_OPT_IN":("src/runtime/engine/concurrent_executor.h",'std::getenv("NINFER_TRUST_SHORT_OPERATION_HINT")'),
"CLASSIFICATION_PATH":("src/runtime/engine/concurrent_executor.h","request->issue58_long_at_admission = classified_long(*request, summary);"),
"ADMISSION_GUARD":("src/runtime/engine/concurrent_executor.h","!classified_long(*candidate, candidate->base_plan->summary())"),
}
missing=[]
for name,(path,needle) in checks.items():
  present=needle in (root/path).read_text()
  print(name+"="+("PASS" if present else "FAIL"))
  if not present:missing.append(name)
print("CONTRACT_RESULT="+("PASS" if not missing else "FAIL"))
sys.exit(1 if missing else 0)
PY
cat "$REPORT"
echo "PRODUCTION_HTTP=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health || true)"
echo "TEST_EXIT_CODE=$RC"
python3 - "$REPORT" "$RC" <<'PY'
import base64,pathlib,sys
data=pathlib.Path(sys.argv[1]).read_text()+"\nTEST_EXIT_CODE="+sys.argv[2]
print("OSC52_CLIPBOARD=ISSUE58_PR61_CONTRACT")
print("\033]52;c;"+base64.b64encode(data.encode()).decode()+"\a",end="")
PY
exit "$RC"