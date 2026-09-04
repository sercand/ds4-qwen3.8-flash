#!/bin/bash
# Start ds4-server on the EXL3 checkpoint and run tests/bench_concurrency.py.
#   ./tests/run_concurrency_bench.sh <tag> [ENV=VAL ...]
# CTXS= execution contexts, SWEEP= client counts, MAX_TOKENS= per request.
# Writes bench_<tag>.log and server_bench_<tag>.log under /tmp scratch.
set -u
cd "${DS4_DIR:-/home/otsimo/work/ds4}"

OUT=${OUT:-/tmp/claude-1000/-home-otsimo-work-ds4/881de0e6-394d-4620-b8b9-58662f31f373/scratchpad}
mkdir -p "$OUT"
tag=${1:?tag}; shift

PORT=${PORT:-8901}
M=${MODEL:-/home/otsimo/work/qwen-3.8-flash/exl3-gguf/Qwen3.8-Flash-Next-EXL3-4.05bpw.gguf}
MT=${MTP:-/home/otsimo/work/qwen-3.8-flash/exl3-gguf/mtp-Qwen3.8-Flash-Next-EXL3.gguf}
CTXS=${CTXS:-2}
CTX=${CTX:-8192}
SWEEP=${SWEEP:-1,2,4,8}
MAX_TOKENS=${MAX_TOKENS:-192}
SLOG=$OUT/server_bench_$tag.log
BLOG=$OUT/bench_$tag.log

nvidia-smi --query-gpu=clocks.sm --format=csv,noheader
setsid env "$@" /home/otsimo/work/qwen-3.8-flash/gpu_lock.sh \
    ./ds4-server --host 127.0.0.1 --port $PORT --cuda -m "$M" --ctx "$CTX" \
    --mtp-model "$MT" --mtp-draft "${MTP_DRAFT:-4}" \
    --exec-contexts $CTXS --cache-log-every 1000 \
    ${EXTRA_ARGS:-} > "$SLOG" 2>&1 < /dev/null &

for i in $(seq 1 300); do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
sleep 2

python3 tests/bench_concurrency.py --url http://127.0.0.1:$PORT \
    --sweep "$SWEEP" --max-tokens "$MAX_TOKENS" 2>&1 | tee "$BLOG"
rc=${PIPESTATUS[0]}

pkill -INT -f "ds4-server --host 127.0.0.1 --port $PORT"
for i in $(seq 1 90); do
    pgrep -f "ds4-server --host 127.0.0.1 --port $PORT" >/dev/null || break
    sleep 1
done
pkill -9 -f "ds4-server --host 127.0.0.1 --port $PORT" 2>/dev/null
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader
exit $rc
