#!/usr/bin/env bash
# tests/test_smoke.sh — end-to-end smoke for yent.aml.
#
# 1. Compile yent.aml through the system amlc.
# 2. If a Q8_0 weights file exists locally, run a 20-token generation
#    and assert the binary exits 0 and prints some non-empty output
#    after the "--- generation ---" marker.
# 3. Otherwise: skip generation, just assert compilation.
#
# Skipping generation is fine for CI without weights — the compile path
# alone exercises amlc + libnotorch + libaml + Accelerate linkage and
# all yent.aml syntax.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== test_smoke =="

# 1. Compile
if ! command -v amlc >/dev/null 2>&1; then
    echo "  SKIP: amlc not installed (run: cd ariannamethod.ai && sudo make install)"
    exit 0
fi

amlc yent.aml -o yent_smoke >/tmp/yent_smoke_compile.log 2>&1
if [ ! -x yent_smoke ]; then
    echo "  FAIL: amlc did not produce yent_smoke binary"
    cat /tmp/yent_smoke_compile.log
    exit 1
fi
echo "  PASS [compile]: yent_smoke ($(wc -c < yent_smoke) bytes)"

# 2. Generation (only if weights are present)
GGUF="weights/yent_v4/yent_v4_sft_q8_0.gguf"
if [ ! -f "$GGUF" ]; then
    echo "  SKIP [generate]: $GGUF not found"
    rm -f yent_smoke yent_smoke.c
    echo
    echo "smoke OK (compile only)"
    exit 0
fi

# Capture both stderr (header) and stdout (generated tokens)
OUT=$(./yent_smoke -w "$GGUF" -p "Q: ping
A:" -n 20 -t 0.7 --top-p 0.9 2>&1)
RC=$?
if [ $RC -ne 0 ]; then
    echo "  FAIL: yent exited $RC"
    echo "$OUT"
    exit 1
fi

# Find content after "--- generation ---" marker
GEN=$(printf '%s\n' "$OUT" | awk '/--- generation ---/{flag=1;next} flag')
GEN_TRIM=$(printf '%s' "$GEN" | tr -d '[:space:]')
if [ -z "$GEN_TRIM" ]; then
    echo "  FAIL: no generation produced after marker"
    echo "$OUT"
    exit 1
fi
echo "  PASS [generate]: $(printf '%s' "$GEN" | wc -c) bytes generated"

# Cleanup
rm -f yent_smoke yent_smoke.c
echo
echo "smoke OK"
