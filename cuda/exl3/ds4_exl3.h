/* ds4 entry points for the vendored exllamav3 EXL3 GEMM kernels (see VENDOR.md).
 *
 * Every activation is fp32 on both sides, as everywhere in ds4's qwen4exp
 * graph; the kernels convert to fp16 inside their input Hadamard phase.  A
 * trellis tensor is addressed by three device pointers -- the tiles, the
 * fp16 input scale vector suh[k] and the fp16 output vector svh[n] -- which
 * the GGUF stores back to back in one tensor payload (ds4.c, exl3_scale_bytes).
 *
 * The kernels are cooperative launches (grid-wide barriers between the input
 * transform and the matmul, and between 16-row passes), so a grid is at most
 * the device's co-resident block count and needs the per-stream lock buffer
 * this file owns.  Both entries return 1 on success, 0 on failure. */
#ifndef DS4_EXL3_H
#define DS4_EXL3_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[m][n] = Had128(x[m][k] * suh) @ B, then Had128 * svh on every 128-wide
 * output block.  Any m; rows are processed 16 at a time. */
int ds4_exl3_gemm(const float *x, const void *tiles, const void *suh, const void *svh,
                  float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                  cudaStream_t stream);

/* Expert fan-out over a stacked tensor ([E] tiles, [E] suh, [E] svh): slot j
 * multiplies its input by expert ids[j] into y[j][m][n].  The input is x
 * broadcast to every slot when x_per_slot is 0, else x[j][m][k]. */
int ds4_exl3_mgemm(const float *x, int x_per_slot,
                   const void *tiles, const void *suh, const void *svh,
                   const int32_t *ids, uint32_t n_slots,
                   float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                   cudaStream_t stream);

/* Size the transformed-activation scratch for the largest launch (slots * m *
 * k halfs) ahead of CUDA graph capture, inside which it cannot grow. */
int ds4_exl3_reserve(cudaStream_t stream, uint64_t had_halfs);

#ifdef __cplusplus
}
#endif

#endif
