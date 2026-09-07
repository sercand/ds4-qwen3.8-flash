#!/bin/sh
# Builds the qwen4exp numerics/perf harnesses against the repo's CUDA objects
# (run `make ds4-server` first).  Binaries land in $OUT (default: this dir).
set -e
cd "$(dirname "$0")/../.."
OUT=${OUT:-misc/qwen4exp-numerics}
OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_ple_stream.o ds4_cuda.o ds4_layer_pack.o cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o cuda/exl3/ds4_exl3.o"
for b in moepath_cmp moepath_trace q4espec_bench; do
    cc -O2 -std=c99 -D_GNU_SOURCE -I. -c -o "$OUT/$b.o" "misc/qwen4exp-numerics/$b.c"
    /usr/local/cuda/bin/nvcc -O3 -Xcompiler -pthread -o "$OUT/$b" "$OUT/$b.o" $OBJS -lm \
        -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64 -lcudart -lcublas -luring
done
echo "built: $OUT/{moepath_cmp,moepath_trace,q4espec_bench}"
