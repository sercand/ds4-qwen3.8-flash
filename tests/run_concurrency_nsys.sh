#!/bin/bash
# nsys profile of a concurrent ds4-server decode.
#   ./tests/run_concurrency_nsys.sh <tag> [ENV=VAL ...]
# nsys must receive the interrupt itself: signalling only ds4-server tears the
# child down before the collector flushes, and the report comes out empty.
set -u
cd "${DS4_DIR:-/home/otsimo/work/ds4}"

OUT=${OUT:-/tmp/claude-1000/-home-otsimo-work-ds4/881de0e6-394d-4620-b8b9-58662f31f373/scratchpad}
mkdir -p "$OUT"
tag=${1:?tag}; shift

PORT=${PORT:-8902}
M=${MODEL:-/home/otsimo/work/qwen-3.8-flash/exl3-gguf/Qwen3.8-Flash-Next-EXL3-4.05bpw.gguf}
MT=${MTP:-/home/otsimo/work/qwen-3.8-flash/exl3-gguf/mtp-Qwen3.8-Flash-Next-EXL3.gguf}
CTXS=${CTXS:-4}
CLIENTS=${CLIENTS:-4}
SLOG=$OUT/server_nsys_$tag.log

env "$@" nsys profile --output "$OUT/nsys_$tag" --force-overwrite true \
      --trace=cuda --sample=none --cpuctxsw=none \
    ./ds4-server --host 127.0.0.1 --port $PORT --cuda -m "$M" --ctx "${CTX:-8192}" \
    --mtp-model "$MT" --mtp-draft "${MTP_DRAFT:-4}" \
    --exec-contexts $CTXS --cache-log-every 1000 > "$SLOG" 2>&1 < /dev/null &
nsys_pid=$!

for i in $(seq 1 300); do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
sleep 2

python3 tests/bench_concurrency.py --url http://127.0.0.1:$PORT \
    --sweep "$CLIENTS" --max-tokens "${MAX_TOKENS:-96}" 2>&1 | tail -20

kill -INT $nsys_pid 2>/dev/null
for i in $(seq 1 300); do kill -0 $nsys_pid 2>/dev/null || break; sleep 1; done
ls -la "$OUT"/nsys_$tag.nsys-rep 2>/dev/null
