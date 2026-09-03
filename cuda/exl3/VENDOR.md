# Vendored exllamav3 EXL3 kernels

The `.cuh` files here are copied from turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3), the reference
implementation of the EXL3 trellis format, with the torch host wrappers left
behind. ds4 drives them from `ds4_exl3.cu`, a plain C ABI (`ds4_exl3.h`)
called by `ds4_qwen4exp_gpu.cuh`.

## Upstream pin

| Field   | Value |
|---------|-------|
| Source  | https://github.com/turboderp-org/exllamav3 |
| Commit  | `499890c75d20d8e7c9d061f37189ae611a5c9f0b` (v1.4.6, 2026-09-02) |
| License | MIT, Copyright (c) 2025 Turboderp (`LICENSE` in this directory) |

## File inventory

| File | Origin (`exllamav3/exllamav3_ext/`) | Status |
|------|------|--------|
| `codebook.cuh` | `quant/codebook.cuh` | verbatim (mul1 decode: `x * 0x83DCD12D`, byte sum, fp16 affine) |
| `exl3_dq.cuh` | `quant/exl3_dq.cuh` | verbatim (bit-window extraction for K = 1..8) |
| `hadamard_inner.cuh` | `quant/hadamard_inner.cuh` | include path only |
| `exl3_gemm_inner.cuh` | `quant/exl3_gemm_inner.cuh` | include path only |
| `exl3_gemm_kernel.cuh` | `quant/exl3_gemm_kernel.cuh` | **modified**, see below |
| `exl3_kernel_map.cuh` | `quant/exl3_kernel_map.cuh` | **modified**: argument lists; host declarations dropped |
| `exl3_moe_common.cuh` | `quant/exl3_moe_common.cuh` | verbatim (fused MoE prefill kernel, not wired yet) |
| `exl3_moe_kernel.cuh` | `quant/exl3_moe_kernel.cuh` | include paths only (not wired yet) |
| `exl3_devctx.cuh` | `quant/exl3_devctx.cuh` | lock-buffer layout constants only |
| `ptx.cuh` | `ptx.cuh` | verbatim (mma.m16n8k16, cp.async, barriers) |
| `util.cuh` | `util.cuh` | cuBLAS error helpers removed |
| `compat.cuh` | `compat.cuh` | verbatim |

## ds4 modifications to the kernels

- Activations are fp32.  ds4 keeps every qwen4exp activation in fp32, so the
  GEMM's `A` and the mgemm's `A` are `const float*` and the input Hadamard
  phase (`had_fh_r_128_inner`) converts to the fp16 `A_had` the MMA loop
  reads.  Outputs are fp32 (`c_fp32 = true` is the only instantiation).
- `exl3_mgemm_kernel` addresses the experts of a projection as one stacked
  tensor: matrix `q` is `B_base + q * B_stride`, `suh_base + q * k`,
  `svh_base + q * n`, selected per slot by `int32` ids (ds4's router output).
  The pointer-table arguments, the expert-range filtering, the weighted
  reduction into slot 0 and the per-matrix width lists are removed; ds4 does
  the routing-weight combine in its own kernel.
- Instances: K = 4, 5, 6 only, codebook mul1 only (`ds4_exl3.cu`).  The
  QTIP-style GEMV and int8 GEMV fast paths are not vendored: upstream keeps
  them off on Blackwell and they exclude K = 6.

Why vendored rather than depended on: like `cuda/mmq`, ds4 is a flat C/CUDA
tree with no build dependencies, and exllamav3 is a torch extension.
