#!/usr/bin/env bash
# Issue 58 build-only validation. NO GPU serving, service stops, or workloads.
# Logs stay on Brain and a compact report is copied via OSC52 even if build fails.
set -uo pipefail
export GIT_PAGER=cat PAGER=cat GIT_TERMINAL_PROMPT=0
SOURCE="${NINFER_SOURCE:-/home/openclaw/ninfer-5080-issue32-concurrency}"
WORKTREE="${NINFER_ISSUE58_WORKTREE:-/home/openclaw/ninfer-issue58-trace}"
BUILD="${NINFER_ISSUE58_BUILD:-/home/openclaw/ninfer-issue58-trace-build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="/tmp/issue58-trace-build-${STAMP}-$$.log"
RC=0
(
  echo "=== ISSUE58 ADMISSION TRACE BUILD ==="
  date -u
  echo "SOURCE=$SOURCE"
  git -C "$SOURCE" fetch origin "refs/heads/issue58-source-admission-triage:refs/remotes/origin/issue58-source-admission-triage" || exit 10
  if [[ ! -e "$WORKTREE/.git" ]]; then
     if [[ -e "$WORKTREE" ]]; then
        echo "ERROR: existing non-worktree path $WORKTREE; refusing to overwrite" ; exit 11
     fi
     git -C "$SOURCE" worktree add --detach "$WORKTREE" origin/issue58-source-admission-triage || exit 12
  fi
  if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
    echo "WORKTREE_DIRTY_REFUSING_TO_OVERWRITE"
    exit 18
  fi
  git -C "$WORKTREE" switch --detach origin/issue58-source-admission-triage || exit 19
  echo "WORKTREE=$(git -C "$WORKTREE" rev-parse --show-toplevel)"
  echo "HEAD=$(git -C "$WORKTREE" rev-parse HEAD)"
  echo "=== DIFF VALIDATION ==="
  git -C "$WORKTREE" diff --check HEAD~1 HEAD || exit 13
  echo "=== CUDA COMPILER DISCOVERY ==="
  NVCC=""
  for candidate in "${CUDACXX:-}" "${CUDA_HOME:-}/bin/nvcc" "${CUDA_PATH:-}/bin/nvcc" /usr/local/cuda/bin/nvcc /usr/local/cuda-13.4/bin/nvcc /usr/local/cuda-13.3/bin/nvcc /usr/local/cuda-13.2/bin/nvcc; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then NVCC="$candidate"; break; fi
  done
  if [[ -z "$NVCC" ]]; then NVCC="$(command -v nvcc || true)"; fi
  if [[ -z "$NVCC" ]]; then
    echo "CUDA_COMPILER_NOT_FOUND: nvcc is not accessible; no install or production changes attempted"
    ls -ld /usr/local/cuda* /opt/cuda* /usr/bin/nvcc 2>/dev/null || true
    find /home/openclaw -maxdepth 5 -name CMakeCache.txt -print 2>/dev/null | head -12
    exit 16
  fi
  export CUDACXX="$NVCC"
  echo "CUDACXX=$CUDACXX"
  "$CUDACXX" --version | tail -n 4
  echo "=== CCACHE DISCOVERY AND STATS ==="
  CCACHE_BIN="$(command -v ccache || true)"
  if [[ -z "$CCACHE_BIN" ]]; then
    echo "CCACHE_REQUIRED_BUT_NOT_FOUND: install ccache or expose its existing path; refusing uncached build"
    exit 17
  fi
  export CCACHE_DIR="${CCACHE_DIR:-/home/openclaw/.cache/ccache}"
  mkdir -p "$CCACHE_DIR"
  echo "CCACHE_BIN=$CCACHE_BIN"
  echo "CCACHE_DIR=$CCACHE_DIR"
  "$CCACHE_BIN" --version | head -n 2
  "$CCACHE_BIN" -s || true
  echo "=== CMAKE CONFIGURE (CXX + CUDA CCACHE) ==="
  cmake -S "$WORKTREE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_C_COMPILER_LAUNCHER="$CCACHE_BIN" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$CCACHE_BIN" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER="$CCACHE_BIN" || exit 14
  echo "=== CCACHE LAUNCHER VERIFICATION ==="
  grep -E '^CMAKE_(C|CXX|CUDA)_COMPILER_LAUNCHER:STRING=' "$BUILD/CMakeCache.txt" || true
  echo "=== NINFER SERVE COMPILE (NO EXECUTION) ==="
  cmake --build "$BUILD" --target ninfer-serve -j 4 || exit 15
  echo "=== CCACHE STATS AFTER BUILD ==="
  "$CCACHE_BIN" -s || true
  echo "BUILD_RESULT=PASS"
) >"$REPORT" 2>&1 || RC=$?
cat "$REPORT"
python3 - "$REPORT" "$RC" <<'PY'
import base64,sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text(errors="replace").splitlines()
rc=sys.argv[2]
tail="\n".join(lines[-50:])
report=f"ISSUE58 TRACE BUILD RC={rc}\nREPORT={sys.argv[1]}\n"+tail
print("\nOSC52_CLIPBOARD=BUILD_SUMMARY")
sys.stdout.write("\033]52;c;"+base64.b64encode(report.encode()).decode()+"\a")
PY
echo "REPORT_PATH=$REPORT"
echo "VALIDATION_RC=$RC"
# No 'exit' at the end of the interactive parent session; this script process returns.
