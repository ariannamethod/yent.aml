#!/usr/bin/env bash
# tests/test_smoke.sh — end-to-end smoke for jannus-r.
#
# Runs (in order):
#   1. cc test_calendar.c → run               (no deps)
#   2. cc test_spa.c → run                    (no deps)
#   3. amlc jannus-r.aml → compile only       (skip if amlc missing)
#   4. ./jannus-r if Q8 weights are present   (optional)
#
# Each step prints PASS/FAIL/SKIP and exits non-zero on FAIL.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== test_smoke =="

# 1. Calendar
cc -O2 -Wall -Wextra tests/test_calendar.c -o tests/test_calendar -lm
./tests/test_calendar
echo "  PASS [calendar suite]"

# 2. SPA
cc -O2 -Wall -Wextra tests/test_spa.c -o tests/test_spa -lm
./tests/test_spa
echo "  PASS [spa suite]"

# 3. amlc compile
if ! command -v amlc >/dev/null 2>&1; then
    echo "  SKIP [amlc compile] — amlc not installed"
else
    amlc jannus-r.aml -o jannus-r_smoke >/tmp/jannus-r_smoke_compile.log 2>&1
    if [ ! -x jannus-r_smoke ]; then
        echo "  FAIL [amlc compile]"; cat /tmp/jannus-r_smoke_compile.log; exit 1
    fi
    echo "  PASS [amlc compile] jannus-r_smoke ($(wc -c < jannus-r_smoke) bytes)"
fi

# 4. Generation if weights present
GGUF="../weights/yent_v4/yent_v4_sft_q8_0.gguf"
if [ -x jannus-r_smoke ] && [ -f "$GGUF" ]; then
    OUT=$(./jannus-r_smoke -w "$GGUF" -p "ping" 2>&1)
    if echo "$OUT" | grep -q "ORIGIN"; then
        echo "  PASS [generation] chain rendered"
    else
        echo "  FAIL [generation] no ORIGIN in output"; echo "$OUT"; exit 1
    fi
else
    echo "  SKIP [generation] — no weights"
fi

# Cleanup
rm -f tests/test_calendar tests/test_spa jannus-r_smoke jannus-r_smoke.c
echo
echo "smoke OK"
