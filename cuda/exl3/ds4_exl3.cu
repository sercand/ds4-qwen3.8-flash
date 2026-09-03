/* Host side of the vendored EXL3 kernels: kernel instances for the bit widths
 * this model uses, the tile-shape heuristic, the cooperative launch geometry,
 * and the per-stream lock/scratch buffers.  Torch-free port of the launch
 * logic in exllamav3's quant/exl3_gemm.cu and exl3_kernel_map.cu. */
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <stdio.h>
#include <string.h>

#include "ds4_exl3.h"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"

/* Only the mul1 codebook (cb 2) and fp32 outputs are instantiated: the
 * qwen4exp checkpoint is mul1 throughout, K = 4 for routed experts and the
 * indexer, 5 for the MTP input projections, 6 for every other dense tensor. */
#define DS4_EXL3_MIN_BITS 4
#define DS4_EXL3_MAX_BITS 6

static fp_exl3_gemm_kernel g_gemm[DS4_EXL3_MAX_BITS - DS4_EXL3_MIN_BITS + 1][EXL3_GEMM_NUM_SHAPES + 1] = {
    { EXL3_GEMM_KERNEL_INSTANCES(4, true, 2) },
    { EXL3_GEMM_KERNEL_INSTANCES(5, true, 2) },
    { EXL3_GEMM_KERNEL_INSTANCES(6, true, 2) },
};
static fp_exl3_mgemm_kernel g_mgemm[DS4_EXL3_MAX_BITS - DS4_EXL3_MIN_BITS + 1][EXL3_GEMM_NUM_SHAPES + 1] = {
    { EXL3_MGEMM_KERNEL_INSTANCES(4, true, 2) },
    { EXL3_MGEMM_KERNEL_INSTANCES(5, true, 2) },
    { EXL3_MGEMM_KERNEL_INSTANCES(6, true, 2) },
};
static const int g_tile_k[] = { EXL3_GEMM_TILESIZE_K };
static const int g_tile_n[] = { EXL3_GEMM_TILESIZE_N };
static const int g_block_dim[] = { EXL3_GEMM_BLOCKDIM };

/* Per-stream state.  The lock buffer is shared by every launch on a stream
 * (the kernels leave it zeroed), so two streams must not share one; the
 * scratch holds the fp16 Hadamard-transformed activations. */
#define DS4_EXL3_MAX_STREAMS 8
struct exl3_stream_ctx {
    cudaStream_t stream;
    int         *locks;
    half        *had;
    uint64_t     had_halfs;
};
static exl3_stream_ctx g_ctx[DS4_EXL3_MAX_STREAMS];
static int g_n_ctx = 0;
static int g_num_sms = 0;
static int g_max_blocks = 0;   /* co-resident blocks: one per SM at 90 KB of shared memory */

static bool exl3_fail(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return false;
    fprintf(stderr, "ds4: exl3 %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();
    return true;
}

static bool exl3_device_init(void) {
    if (g_num_sms) return true;
    int dev = 0;
    if (exl3_fail(cudaGetDevice(&dev), "device query")) return false;
    if (exl3_fail(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev), "SM count")) return false;
    /* Every kernel instance needs the 90 KB dynamic shared memory opt-in
     * before its first launch; a capture would reject the attribute call. */
    int occ = 0;
    for (int b = 0; b <= DS4_EXL3_MAX_BITS - DS4_EXL3_MIN_BITS; b++) {
        for (int s = 1; s <= EXL3_GEMM_NUM_SHAPES; s++) {
            if (exl3_fail(cudaFuncSetAttribute((const void *)g_gemm[b][s],
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX),
                          "gemm shared memory opt-in")) return false;
            if (exl3_fail(cudaFuncSetAttribute((const void *)g_mgemm[b][s],
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX),
                          "mgemm shared memory opt-in")) return false;
            int o = 0;
            if (exl3_fail(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                  &o, g_gemm[b][s], g_block_dim[s], SMEM_MAX), "occupancy")) return false;
            occ = (occ == 0 || o < occ) ? o : occ;
        }
    }
    if (occ < 1) occ = 1;
    g_max_blocks = g_num_sms * occ;
    return true;
}

static exl3_stream_ctx *exl3_ctx(cudaStream_t stream, uint64_t had_halfs) {
    if (!exl3_device_init()) return NULL;
    exl3_stream_ctx *c = NULL;
    for (int i = 0; i < g_n_ctx; i++) if (g_ctx[i].stream == stream) c = &g_ctx[i];
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &cap);
    const bool capturing = cap != cudaStreamCaptureStatusNone;
    if (!c) {
        if (g_n_ctx == DS4_EXL3_MAX_STREAMS || capturing) {
            fprintf(stderr, "ds4: exl3: no lock buffer for this stream (%s)\n",
                    capturing ? "cannot allocate during graph capture" : "too many streams");
            return NULL;
        }
        c = &g_ctx[g_n_ctx];
        memset(c, 0, sizeof(*c));
        c->stream = stream;
        const size_t lock_bytes = (MAX_TILES_C + 2 * MAX_BARRIERS + MOE_SCHED_INTS) * sizeof(int);
        if (exl3_fail(cudaMalloc((void **)&c->locks, lock_bytes), "lock buffer") ||
            exl3_fail(cudaMemset(c->locks, 0, lock_bytes), "lock buffer clear")) return NULL;
        g_n_ctx++;
    }
    if (c->had_halfs < had_halfs) {
        if (capturing) {
            fprintf(stderr, "ds4: exl3: activation scratch too small during graph capture "
                            "(%llu < %llu halfs); reserve it first\n",
                    (unsigned long long)c->had_halfs, (unsigned long long)had_halfs);
            return NULL;
        }
        /* Growth waits for the stream: a running launch may still read it. */
        if (exl3_fail(cudaStreamSynchronize(stream), "scratch growth sync")) return NULL;
        if (c->had) (void)cudaFree(c->had);
        c->had = NULL;
        c->had_halfs = 0;
        uint64_t want = had_halfs < (8u << 20) ? (8u << 20) : had_halfs;   /* 16 MiB floor */
        if (exl3_fail(cudaMalloc((void **)&c->had, want * sizeof(half)), "activation scratch")) return NULL;
        c->had_halfs = want;
    }
    return c;
}

extern "C" int ds4_exl3_reserve(cudaStream_t stream, uint64_t had_halfs) {
    return exl3_ctx(stream, had_halfs) != NULL;
}

extern "C" int ds4_exl3_prepare_stream(cudaStream_t stream) {
    if (g_n_ctx == 0) return 1;
    uint64_t halfs = 0;
    for (int i = 0; i < g_n_ctx; i++) if (g_ctx[i].had_halfs > halfs) halfs = g_ctx[i].had_halfs;
    return exl3_ctx(stream, halfs) != NULL;
}

/* exllamav3's shape heuristic for Blackwell (select_gemm_shape, CC_BLACKWELL
 * branch), with the shape's tile constraints checked. */
static int exl3_select_shape(int size_m, int size_k, int size_n, int K, bool multi,
                             int bszm_in, int bszm_out) {
    (void)size_m;
    const bool mod_256 = (size_n % 256 == 0);
    const bool mod_512 = (size_n % 512 == 0);
    const int k_eff = size_k * bszm_in;
    const int n_eff = size_n * bszm_out;
    int shape;
    if ((K == 4 || K == 2) && !multi && k_eff <= 2048) shape = 1;
    else if (K >= 7) {
        if (mod_256 && n_eff <= 8192) shape = k_eff > 32768 ? 3 : 2;
        else if (mod_512 && n_eff > 32768) shape = 4;
        else shape = 2;
    }
    else if (mod_256 && n_eff <= 4096) shape = (k_eff > 8192 && K >= 3) ? 3 : 2;
    else if (mod_512 && n_eff > 16384) shape = 4;
    else if (mod_256) shape = 3;
    else shape = 2;
    for (int s = shape; s >= 1; s--) {
        if (size_k % g_tile_k[s] == 0 && size_n % g_tile_n[s] == 0) return s;
    }
    for (int s = 1; s <= EXL3_GEMM_NUM_SHAPES; s++) {
        if (size_k % g_tile_k[s] == 0 && size_n % g_tile_n[s] == 0) return s;
    }
    return 0;
}

static bool exl3_check_shape(uint32_t k, uint32_t n, uint32_t bits, const char *what) {
    if (bits < DS4_EXL3_MIN_BITS || bits > DS4_EXL3_MAX_BITS || k == 0 || n == 0 ||
        (k % 16u) != 0u || (n % 128u) != 0u) {
        fprintf(stderr, "ds4: exl3 %s: unsupported shape k=%u n=%u K=%u\n", what, k, n, bits);
        return false;
    }
    return true;
}

extern "C" int ds4_exl3_gemm(const float *x, const void *tiles, const void *suh, const void *svh,
                             float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                             cudaStream_t stream) {
    if (!x || !tiles || !suh || !svh || !y || m == 0) return 0;
    if (!exl3_check_shape(k, n, bits, "gemm")) return 0;
    exl3_stream_ctx *c = exl3_ctx(stream, (uint64_t)m * k);
    if (!c) return 0;

    const int shape = exl3_select_shape((int)m, (int)k, (int)n, (int)bits, false, 1, 1);
    if (!shape) return 0;
    fp_exl3_gemm_kernel kernel = g_gemm[bits - DS4_EXL3_MIN_BITS][shape];
    const int max_slices = (int)(k / g_tile_k[shape]) * (int)(n / g_tile_n[shape]);
    int num_blocks = max_slices < g_max_blocks ? max_slices : g_max_blocks;
    if (num_blocks < 1) num_blocks = 1;

    const half *suh_h = (const half *)suh;
    const half *svh_h = (const half *)svh;
    const uint16_t *B = (const uint16_t *)tiles;
    half *had = c->had;
    int *locks = c->locks;
    int size_m = (int)m, size_k = (int)k, size_n = (int)n;
    void *args[] = { (void *)&x, (void *)&B, (void *)&y, (void *)&size_m, (void *)&size_k,
                     (void *)&size_n, (void *)&locks, (void *)&suh_h, (void *)&had, (void *)&svh_h };
    return !exl3_fail(cudaLaunchCooperativeKernel((const void *)kernel, dim3(num_blocks), dim3(g_block_dim[shape]),
                                                  args, SMEM_MAX, stream), "gemm launch");
}

extern "C" int ds4_exl3_mgemm(const float *x, int x_per_slot,
                              const void *tiles, const void *suh, const void *svh,
                              const int32_t *ids, uint32_t n_slots,
                              float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                              cudaStream_t stream) {
    if (!x || !tiles || !suh || !svh || !ids || !y || m == 0 || n_slots == 0) return 0;
    if (!exl3_check_shape(k, n, bits, "mgemm")) return 0;
    /* One transformed slab per slot: an undersized scratch is silent
     * out-of-bounds corruption (exllamav3's comment: "found the hard way"). */
    exl3_stream_ctx *c = exl3_ctx(stream, (uint64_t)n_slots * m * k);
    if (!c) return 0;

    const int bszm_in = x_per_slot ? (int)n_slots : 1;
    const int bszm_out = (int)n_slots;
    const int shape = exl3_select_shape((int)m, (int)k, (int)n, (int)bits, true, bszm_in, bszm_out);
    if (!shape) return 0;
    fp_exl3_mgemm_kernel kernel = g_mgemm[bits - DS4_EXL3_MIN_BITS][shape];

    /* Geometry as exl3_mgemm_gr: blocks per slot from the tile count, then as
     * many slot groups side by side as fit the co-resident block budget. */
    const int tiles_n = (int)(k / g_tile_k[shape]) * (int)(n / g_tile_n[shape]);
    int per_group = tiles_n / (g_max_blocks > 128 ? 20 : 24);
    if (per_group < 1) per_group = 1;
    if (per_group > g_max_blocks) per_group = g_max_blocks;
    if (per_group * (int)n_slots > g_max_blocks) {
        per_group = g_max_blocks / (int)n_slots;
        if (per_group < 1) per_group = 1;
    }
    if (tiles_n / per_group > 48 && per_group * 2 <= g_max_blocks) per_group *= 2;
    int concurrency = g_max_blocks / per_group;
    if (concurrency > (int)n_slots) concurrency = (int)n_slots;
    if (concurrency > MAX_BARRIERS) concurrency = MAX_BARRIERS;
    if (concurrency < 1) concurrency = 1;

    const size_t stride = (size_t)k * n * bits / 8u / sizeof(uint16_t);
    const uint16_t *B = (const uint16_t *)tiles;
    const half *suh_h = (const half *)suh;
    const half *svh_h = (const half *)svh;
    half *had = c->had;
    int *locks = c->locks;
    int size_m = (int)m, size_k = (int)k, size_n = (int)n;
    void *args[] = { (void *)&x, (void *)&B, (void *)&stride, (void *)&y, (void *)&size_m,
                     (void *)&size_k, (void *)&size_n, (void *)&locks, (void *)&suh_h, (void *)&had,
                     (void *)&svh_h, (void *)&ids, (void *)&bszm_in, (void *)&bszm_out };
    return !exl3_fail(cudaLaunchCooperativeKernel((const void *)kernel, dim3(per_group, 1, concurrency),
                                                  dim3(g_block_dim[shape]), args, SMEM_MAX, stream),
                      "mgemm launch");
}
