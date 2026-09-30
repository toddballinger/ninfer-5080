#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${NINFER_BIN:-$ROOT/build/apps/ninfer}"
FIXTURE="$ROOT/bench/fixtures/qwen38_118001_prompt.txt"

EXPECTED_PROMPT_SHA="078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd"
EXPECTED_MODEL_SHA="c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21"

main() {
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 <model.ninfer> [additional ninfer arguments...]" >&2
        return 2
    fi

    local MODEL="$1"
    shift
    local EXTRA_ARGS=("$@")

    if [ ! -x "$BIN" ]; then
        echo "ERROR: NInfer CLI not found: $BIN" >&2
        return 3
    fi

    if [ ! -f "$MODEL" ]; then
        echo "ERROR: model not found: $MODEL" >&2
        return 4
    fi

    local PROMPT_SHA MODEL_SHA
    PROMPT_SHA="$(sha256sum "$FIXTURE" | awk '{print $1}')"
    MODEL_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"

    if [ "$PROMPT_SHA" != "$EXPECTED_PROMPT_SHA" ]; then
        echo "ERROR: fixture SHA256 mismatch" >&2
        echo "EXPECTED=$EXPECTED_PROMPT_SHA" >&2
        echo "ACTUAL=$PROMPT_SHA" >&2
        return 5
    fi

    if [ "$MODEL_SHA" != "$EXPECTED_MODEL_SHA" ]; then
        echo "ERROR: canonical model SHA256 mismatch" >&2
        echo "EXPECTED=$EXPECTED_MODEL_SHA" >&2
        echo "ACTUAL=$MODEL_SHA" >&2
        return 6
    fi

    local OUT LOG RC
    OUT="${NINFER_118K_OUT:-/tmp/ninfer-qwen38-118k-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$OUT"
    LOG="$OUT/run.log"

    "$BIN" \
        "$MODEL" \
        --prompt-file "$FIXTURE" \
        --max-context 131072 \
        --kv-capacity 131072 \
        --prefill-chunk 896 \
        --kv-dtype q4 \
        --spec mtp \
        --draft-tokens 3 \
        --embedding-host \
        --max-new 32 \
        --no-thinking \
        --greedy \
        --no-cuda-graph \
        "${EXTRA_ARGS[@]}" \
        >"$LOG" 2>&1

    RC=$?

    local PROMPT_TOKENS PREFILL DECODE MTP_RATE MTP_LENGTH FREE_START SLACK

    PROMPT_TOKENS="$(
        awk '$1=="summary" && $2=="prompt" && $3=="tokens" {print $4}' \
            "$LOG" | tail -1
    )"

    PREFILL="$(
        awk '$1=="summary" && $2=="prefill" && $3=="speed" {print $4}' \
            "$LOG" | tail -1
    )"

    DECODE="$(
        awk '$1=="summary" && $2=="decode" && $3=="speed" {print $4}' \
            "$LOG" | tail -1
    )"

    MTP_RATE="$(
        awk '$1=="summary" && $2=="mtp" && $3=="acceptance" && $4=="rate" {print $5}' \
            "$LOG" | tail -1
    )"

    MTP_LENGTH="$(
        awk '$1=="summary" && $2=="mtp" && $3=="acceptance" && $4=="length" {print $5}' \
            "$LOG" | tail -1
    )"

    FREE_START="$(
        awk '$1=="summary" && $2=="free" && $3=="after" && $4=="startup" {print $5,$6}' \
            "$LOG" | tail -1
    )"

    SLACK="$(
        awk '$1=="summary" && $2=="planned" && $3=="slack" {print $4,$5}' \
            "$LOG" | tail -1
    )"

    echo "MODEL_SHA256=$MODEL_SHA"
    echo "PROMPT_SHA256=$PROMPT_SHA"
    echo "PROMPT_TOKENS=${PROMPT_TOKENS:-NA}"
    echo "MAX_CONTEXT=131072"
    echo "KV_CAPACITY=131072"
    echo "PREFILL_CHUNK=896"
    echo "KV_DTYPE=q4-group64"
    echo "SPECULATION=MTP-3"
    echo "EMBEDDING_HOST=enabled"
    echo "MAX_NEW=32"
    echo "THINKING=disabled"
    echo "SAMPLING=greedy"
    echo "CUDA_GRAPH=disabled"
    echo "PREFILL_TPS=${PREFILL:-NA}"
    echo "DECODE_TPS=${DECODE:-NA}"
    echo "MTP_ACCEPTANCE=${MTP_RATE:-NA}"
    echo "MTP_LENGTH=${MTP_LENGTH:-NA}"
    echo "FREE_AFTER_STARTUP=${FREE_START:-NA}"
    echo "PLANNED_SLACK=${SLACK:-NA}"
    echo "RESULT_DIR=$OUT"

    if [ "$RC" -ne 0 ]; then
        echo "QUALIFICATION=FAIL_RC_$RC"
        return "$RC"
    fi

    if [ "$PROMPT_TOKENS" != "118001" ]; then
        echo "QUALIFICATION=FAIL_PROMPT_TOKENS"
        return 7
    fi

    echo "QUALIFICATION=PASS"
    return 0
}

main "$@"
