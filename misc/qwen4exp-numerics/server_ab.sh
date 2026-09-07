#!/bin/bash
# One arm of the live A/B: start ds4-server with the given env, run ab.py at the
# given concurrency, print the server's batch-log summary, stop the server.
#   server_ab.sh <label> <concurrency> [ENV=VAL ...]
set -u
LABEL=$1; CONC=$2; shift 2
cd "$(dirname "$0")/../.."
S=${SCRATCH:-/tmp/claude-1000/-home-otsimo-work-ds4/6fe3641f-0005-48f4-b0af-2ca14c6d141a/scratchpad}
PORT=${PORT:-8898}
MODEL=/home/otsimo/work/qwen-3.8-flash/exl3-gguf/Qwen3.8-Flash-Next-EXL3-4.05bpw.gguf
MTP=/home/otsimo/work/qwen-3.8-flash/exl3-gguf/mtp-Qwen3.8-Flash-Next-EXL3.gguf
VOCAB=/home/otsimo/work/qwen-3.8-flash/draft_vocab_counts.txt
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ] || pgrep -x ds4-server >/dev/null; then echo "GPU BUSY ($apps) -- not starting"; exit 1; fi
LOG=$S/srv_$LABEL.log
env DS4_LOCK_FILE=$S/ds4.lock DS4_SERVER_BATCH_LOG=1 "$@" ./ds4-server --host 127.0.0.1 --port $PORT --cuda -m $MODEL --ctx 16384 \
    --mtp-model $MTP --mtp-draft ${MTP_DRAFT:-4} --mtp-vocab $VOCAB --exec-contexts ${EXEC:-3} > $LOG 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "listening on" $LOG && break; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 $LOG; exit 1; }; sleep 1; done
grep -q "listening on" $LOG || { echo "server did not come up"; tail -5 $LOG; exit 1; }
echo "===== $LABEL: conc=$CONC env: $*"
python3 misc/qwen4exp-numerics/ab.py $PORT 1 60 >/dev/null 2>&1   # warm the caches with one short request
python3 misc/qwen4exp-numerics/ab.py $PORT $CONC 300
echo "-- server batch log: $(grep -c 'decode batch count=' $LOG) batch ticks; by kind:"
grep -o 'decode batch count=[0-9]* spec=[01]' $LOG | sort | uniq -c | sort -rn | head -4
grep 'spec stats' $LOG | head -3 | cut -c1-140
