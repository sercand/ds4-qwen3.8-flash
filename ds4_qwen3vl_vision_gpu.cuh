/* Qwen3-VL vision tower (Qwen3.8-Flash-Next), CUDA.
 *
 * Shape, and how it differs from the GLM-5.3 tower next door -- the differences
 * are why this is a separate file rather than a few extra parameters:
 *   - LayerNorm with weight AND bias, not RMSNorm
 *   - a plain fc1 -> GELU -> fc2 MLP, not SwiGLU
 *   - a learned 48x48 position embedding, bilinearly resampled to the grid and
 *     added before block 0; GLM has none
 *   - head_dim 72, not 64
 *   - no q/k norm
 *   - full bidirectional attention, no windowing
 *   - tanh-approximate GELU in the blocks but erf GELU in the merger; the
 *     reference really does use two different ones
 * Deepstack is absent from this checkpoint (deepstack_visual_indexes is empty),
 * so there is one merger and no per-layer injection into the LLM residual.
 */

#ifndef DS4_QWEN3VL_VISION_TYPES_DEFINED
#define DS4_QWEN3VL_VISION_TYPES_DEFINED

#define DS4_QWEN3VL_VISION_LAYERS    27u
#define DS4_QWEN3VL_VISION_DIM       1152u
#define DS4_QWEN3VL_VISION_HEADS     16u
#define DS4_QWEN3VL_VISION_HEAD_DIM  72u    /* 1152 / 16 */
#define DS4_QWEN3VL_VISION_ROT       36u    /* head_dim * partial_rotary 0.5 */
#define DS4_QWEN3VL_VISION_FF        4304u
#define DS4_QWEN3VL_VISION_OUT       2560u  /* == the LLM's hidden size */
#define DS4_QWEN3VL_VISION_PATCH_DIM 1536u  /* 3 * 2 * 16 * 16 */
#define DS4_QWEN3VL_VISION_MERGED    4608u  /* 1152 * spatial_merge^2 */
#define DS4_QWEN3VL_VISION_POS_SIDE  48u    /* sqrt(num_position_embeddings) */
#define DS4_QWEN3VL_VISION_EPS       1e-6f
#define DS4_QWEN3VL_VISION_ROPE_BASE 10000.0f

typedef struct {
    uint64_t norm1_weight;
    uint64_t norm1_bias;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t proj_weight;
    uint64_t proj_bias;
    uint64_t norm2_weight;
    uint64_t norm2_bias;
    uint64_t fc1_weight;
    uint64_t fc1_bias;
    uint64_t fc2_weight;
    uint64_t fc2_bias;
} ds4_qwen3vl_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t pos_embed;
    uint64_t merger_norm_weight;
    uint64_t merger_norm_bias;
    uint64_t merger_fc1_weight;
    uint64_t merger_fc1_bias;
    uint64_t merger_fc2_weight;
    uint64_t merger_fc2_bias;
    ds4_qwen3vl_vision_layer_weights layer[DS4_QWEN3VL_VISION_LAYERS];
} ds4_qwen3vl_vision_weights;
#endif

#ifndef DS4_QWEN3VL_VISION_STREAM
#define DS4_QWEN3VL_VISION_STREAM 0
#endif

__device__ __forceinline__ static float qwen3vl_bf16(const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

/* Patch n of the flattened sequence sits at grid (row, col).
 *
 * The sequence is in merge-block order: blocks row-major over the
 * (grid_h/2, grid_w/2) block grid, and within a block its 2x2 patches
 * row-major.  ds4_image_preprocess_qwen3vl emits patches in exactly this
 * order, the ViT's 2D RoPE indexes positions with it, and it is what makes the
 * merger's "four consecutive rows are one 2x2 block" reshape correct.  All
 * three have to agree, so the mapping lives in one place. */
__device__ __forceinline__ static void qwen3vl_patch_rc(
        uint32_t n, uint32_t grid_w, uint32_t *row, uint32_t *col) {
    const uint32_t bw = grid_w / 2u;
    const uint32_t block = n / 4u, local = n & 3u;
    *row = (block / bw) * 2u + (local >> 1);
    *col = (block % bw) * 2u + (local & 1u);
}

/* LayerNorm (mean and variance, not RMS) with weight and bias. */
__global__ static void qwen3vl_layernorm_kernel(
        float *out, const float *in, const uint16_t *weight,
        const uint16_t *bias, uint32_t dim) {
    const uint32_t r = blockIdx.x;
    const float *src = in + (size_t)r * dim;
    float *dst = out + (size_t)r * dim;
    __shared__ float s_mean, s_rstd;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) sum += src[i];
    sum = q4e_block_sum(sum);
    if (threadIdx.x == 0) s_mean = sum / (float)dim;
    __syncthreads();
    const float mean = s_mean;
    float var = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = src[i] - mean;
        var += d * d;
    }
    var = q4e_block_sum(var);
    if (threadIdx.x == 0) s_rstd = rsqrtf(var / (float)dim + DS4_QWEN3VL_VISION_EPS);
    __syncthreads();
    const float rstd = s_rstd;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (src[i] - mean) * rstd * qwen3vl_bf16(weight + i) +
                 qwen3vl_bf16(bias + i);
    }
}

/* out[i] += bias[i % dim], and optionally a residual. */
__global__ static void qwen3vl_bias_kernel(
        float *x, const uint16_t *bias, const float *residual,
        uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    if (residual) v += residual[i];
    x[i] = v;
}

/* gelu_pytorch_tanh, the blocks' activation. */
__global__ static void qwen3vl_gelu_tanh_kernel(
        float *x, const uint16_t *bias, uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    const float c = 0.7978845608028654f;   /* sqrt(2/pi) */
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

/* Exact erf GELU: nn.GELU() in the merger, which is NOT the tanh
 * approximation the blocks use.  erff is exact and the merger runs once. */
__global__ static void qwen3vl_gelu_erf_kernel(
        float *x, const uint16_t *bias, uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

/* Add the learned position embedding, bilinearly resampled from its fixed
 * 48x48 grid onto this image's grid.
 *
 * The reference interpolates with linspace(0, 47, n) -- align_corners style --
 * and a single point when the extent is 1, then applies the same merge-block
 * reorder as the patch sequence, which is why the (row, col) for output row n
 * comes from qwen3vl_patch_rc. */
__global__ static void qwen3vl_pos_embed_kernel(
        float *x, const uint16_t *table, uint32_t rows,
        uint32_t grid_h, uint32_t grid_w) {
    const uint32_t n = blockIdx.x;
    if (n >= rows) return;
    uint32_t row, col;
    qwen3vl_patch_rc(n, grid_w, &row, &col);

    const uint32_t side = DS4_QWEN3VL_VISION_POS_SIDE;
    const float hs = (grid_h > 1u) ? (float)(side - 1u) / (float)(grid_h - 1u) : 0.0f;
    const float ws = (grid_w > 1u) ? (float)(side - 1u) / (float)(grid_w - 1u) : 0.0f;
    const float hy = (float)row * hs, wx = (float)col * ws;
    const uint32_t h0 = (uint32_t)hy, w0 = (uint32_t)wx;
    const uint32_t h1 = (h0 + 1u < side) ? h0 + 1u : side - 1u;
    const uint32_t w1 = (w0 + 1u < side) ? w0 + 1u : side - 1u;
    const float dh = hy - (float)h0, dw = wx - (float)w0;
    const float c11 = dh * dw;
    const float c10 = dh - c11;          /* dh * (1 - dw) */
    const float c01 = dw - c11;          /* (1 - dh) * dw */
    const float c00 = 1.0f - dh - c01;   /* (1 - dh) * (1 - dw) */

    const uint32_t dim = DS4_QWEN3VL_VISION_DIM;
    const uint16_t *e00 = table + (size_t)(h0 * side + w0) * dim;
    const uint16_t *e01 = table + (size_t)(h0 * side + w1) * dim;
    const uint16_t *e10 = table + (size_t)(h1 * side + w0) * dim;
    const uint16_t *e11 = table + (size_t)(h1 * side + w1) * dim;
    float *dst = x + (size_t)n * dim;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] += c00 * qwen3vl_bf16(e00 + i) + c01 * qwen3vl_bf16(e01 + i) +
                  c10 * qwen3vl_bf16(e10 + i) + c11 * qwen3vl_bf16(e11 + i);
    }
}

/* Split the fused [q | k | v] projection into three planes and rotate q and k.
 *
 * The ViT's RoPE is 2D: the head's 72 dims split in half NeoX-style, pairing
 * channel i with i+36; the first 18 pairs rotate by the patch's ROW index and
 * the last 18 by its COLUMN index, both at base 1e4.  So the whole head
 * rotates -- "partial_rotary_factor 0.5" in the reference describes how the 36
 * cos/sin values are built before being used on both axes, not a partial
 * rotation. */
__global__ static void qwen3vl_qkv_rope_kernel(
        float *q, float *k, float *v, const float *qkv, const uint16_t *bias,
        uint32_t rows, uint32_t grid_w) {
    const uint32_t n = blockIdx.y;
    const uint32_t h = blockIdx.x;
    if (n >= rows) return;
    const uint32_t dim = DS4_QWEN3VL_VISION_DIM;
    const uint32_t hd = DS4_QWEN3VL_VISION_HEAD_DIM;
    const uint32_t half = hd / 2u;                 /* 36 */
    uint32_t row, col;
    qwen3vl_patch_rc(n, grid_w, &row, &col);

    const float *src = qkv + (size_t)n * 3u * dim;
    const uint32_t off = h * hd;
    float *qd = q + (size_t)n * dim + off;
    float *kd = k + (size_t)n * dim + off;
    float *vd = v + (size_t)n * dim + off;

    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
        qd[i] = src[off + i] + qwen3vl_bf16(bias + off + i);
        kd[i] = src[dim + off + i] + qwen3vl_bf16(bias + dim + off + i);
        vd[i] = src[2u * dim + off + i] + qwen3vl_bf16(bias + 2u * dim + off + i);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        /* Pair i uses frequency index i % 18; the row axis owns the first 18
         * pairs and the column axis the second 18. */
        const uint32_t fi = (i < half / 2u) ? i : (i - half / 2u);
        const uint32_t pos = (i < half / 2u) ? row : col;
        const float inv = powf(DS4_QWEN3VL_VISION_ROPE_BASE,
                               -(float)(2u * fi) / (float)DS4_QWEN3VL_VISION_ROT);
        float s, c;
        sincosf((float)pos * inv, &s, &c);
        const float q0 = qd[i], q1 = qd[i + half];
        qd[i] = q0 * c - q1 * s;
        qd[i + half] = q1 * c + q0 * s;
        const float k0 = kd[i], k1 = kd[i + half];
        kd[i] = k0 * c - k1 * s;
        kd[i + half] = k1 * c + k0 * s;
    }
}

/* Full bidirectional attention over the whole grid, online softmax.
 *
 * One block per (row, head); warps own whole key positions and reduce their
 * dot product with shuffles.  head_dim is 72, which is not a multiple of the
 * warp width, so a lane strides by 32 and the low 8 lanes carry a third
 * element -- correct for any head_dim, just slightly ragged for this one.
 * Attention never crosses an image, and grid_t is 1 for a still image, so the
 * key range is simply every row. */
__global__ static void qwen3vl_attention_kernel(
        float *out, const float *q, const float *k, const float *v,
        uint32_t rows) {
    const uint32_t n = blockIdx.y;
    const uint32_t h = blockIdx.x;
    if (n >= rows) return;
    const uint32_t dim = DS4_QWEN3VL_VISION_DIM;
    const uint32_t hd = DS4_QWEN3VL_VISION_HEAD_DIM;
    const uint32_t off = h * hd;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t warps = blockDim.x >> 5;
    const float scale = rsqrtf((float)hd);

    extern __shared__ float qwen3vl_attn_smem[];
    float *s_q = qwen3vl_attn_smem;                  /* hd */
    float *s_acc = s_q + hd;                         /* warps * hd */
    float *s_max = s_acc + (size_t)warps * hd;       /* warps */
    float *s_den = s_max + warps;                    /* warps */

    const float *qd = q + (size_t)n * dim + off;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) s_q[i] = qd[i];
    __syncthreads();

    float run_max = -INFINITY, run_den = 0.0f;
    float acc[3] = {0.0f, 0.0f, 0.0f};   /* ceil(72/32) slots per lane */
    for (uint32_t p = warp; p < rows; p += warps) {
        const float *kd = k + (size_t)p * dim + off;
        float dot = 0.0f;
        for (uint32_t d = lane; d < hd; d += 32u) dot += s_q[d] * kd[d];
        for (uint32_t m = 16u; m; m >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, m);
        const float score = dot * scale;
        const float new_max = fmaxf(run_max, score);
        const float rescale = __expf(run_max - new_max);
        const float weight = __expf(score - new_max);
        run_den = run_den * rescale + weight;
        run_max = new_max;
        const float *vd = v + (size_t)p * dim + off;
        uint32_t slot = 0;
        for (uint32_t d = lane; d < hd; d += 32u, slot++) {
            acc[slot] = acc[slot] * rescale + weight * vd[d];
        }
    }
    /* Publish each warp's partial softmax, then combine them. */
    {
        uint32_t slot = 0;
        for (uint32_t d = lane; d < hd; d += 32u, slot++) {
            s_acc[(size_t)warp * hd + d] = acc[slot];
        }
        if (lane == 0) { s_max[warp] = run_max; s_den[warp] = run_den; }
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        float m = -INFINITY;
        for (uint32_t w = 0; w < warps; w++) m = fmaxf(m, s_max[w]);
        float den = 0.0f, num = 0.0f;
        for (uint32_t w = 0; w < warps; w++) {
            const float r = __expf(s_max[w] - m);
            den += s_den[w] * r;
            num += s_acc[(size_t)w * hd + threadIdx.x] * r;
        }
        out[(size_t)n * dim + off + threadIdx.x] = den > 0.0f ? num / den : 0.0f;
    }
}

static const uint16_t *qwen3vl_weight(
        const void *model_map, uint64_t model_size, uint64_t offset,
        uint64_t elements, const char *label) {
    if (!model_map || elements > UINT64_MAX / sizeof(uint16_t) ||
        offset > model_size) return NULL;
    const uint64_t bytes = elements * sizeof(uint16_t);
    if (bytes > model_size - offset) return NULL;
#if defined(__HIP_PLATFORM_AMD__)
    return (const uint16_t *)cuda_model_range_ptr(model_map, offset, bytes, label);
#else
    return (const uint16_t *)cuda_resolve_weight_ptr(model_map, offset, bytes, 0, label);
#endif
}

/* Run the tower for one image.
 *
 * `patches` is [rows][1536] as ds4_image_preprocess_qwen3vl produced it, and
 * `out` receives [rows/4][2560] -- one vector per <|image_pad|> token, in the
 * same order the placeholders appear in the prompt.
 *
 * The Conv3d of the reference is a linear map here: its kernel equals its
 * stride and covers the whole patch, so it is exactly Linear(1536 -> 1152) on
 * the already-im2col'd rows.  ds4 has no conv2d anywhere and needs none. */
extern "C" int ds4_gpu_qwen3vl_vision_encode(
        float                            *out,
        const float                      *patches,
        uint32_t                          grid_h,
        uint32_t                          grid_w,
        const void                       *model_map,
        uint64_t                          model_size,
        const ds4_qwen3vl_vision_weights *weights) {
    if (!out || !patches || !model_map || !weights || grid_h == 0u || grid_w == 0u ||
        (grid_h & 1u) != 0u || (grid_w & 1u) != 0u ||
        grid_h > UINT32_MAX / grid_w) return 0;

    const uint32_t dim = DS4_QWEN3VL_VISION_DIM;
    const uint32_t rows = grid_h * grid_w;
    const uint32_t merged_rows = rows / 4u;
    const uint64_t row_dim = (uint64_t)rows * dim;
    const uint64_t row_ff = (uint64_t)rows * DS4_QWEN3VL_VISION_FF;
    const uint64_t merged_in = (uint64_t)merged_rows * DS4_QWEN3VL_VISION_MERGED;
    const uint64_t merged_out = (uint64_t)merged_rows * DS4_QWEN3VL_VISION_OUT;
    if (row_ff > SIZE_MAX / sizeof(float) || merged_in > SIZE_MAX / sizeof(float)) return 0;

    ds4_gpu_tensor *patch = NULL, *a = NULL, *b = NULL, *qkv = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *attn = NULL, *ff = NULL;
    ds4_gpu_tensor *mfc1 = NULL, *mout = NULL;
    const uint16_t *w16 = NULL;
    int ok = 0;

#define QV_ALLOC(name_, count_) do { \
        name_ = ds4_gpu_tensor_alloc((count_) * sizeof(float)); \
        if (!(name_)) goto cleanup; \
    } while (0)
    QV_ALLOC(patch, (uint64_t)rows * DS4_QWEN3VL_VISION_PATCH_DIM);
    QV_ALLOC(a, row_dim);
    QV_ALLOC(b, row_dim);
    QV_ALLOC(qkv, row_dim * 3u);
    QV_ALLOC(q, row_dim);
    QV_ALLOC(k, row_dim);
    QV_ALLOC(v, row_dim);
    QV_ALLOC(attn, row_dim);
    QV_ALLOC(ff, row_ff);
    QV_ALLOC(mfc1, merged_in);
    QV_ALLOC(mout, merged_out);
#undef QV_ALLOC

    if (!ds4_gpu_tensor_write(patch, 0, patches,
            (uint64_t)rows * DS4_QWEN3VL_VISION_PATCH_DIM * sizeof(float)) ||
        !ds4_gpu_begin_commands()) goto cleanup;

    /* patch embedding */
    ok = ds4_gpu_glm53_matmul_bf16(a, model_map, model_size, weights->patch_weight,
                                   DS4_QWEN3VL_VISION_PATCH_DIM, dim, patch, rows);
    if (ok) {
        w16 = qwen3vl_weight(model_map, model_size, weights->patch_bias, dim,
                             "qwen3vl patch bias");
        if (!w16) ok = 0;
    }
    if (ok) {
        qwen3vl_bias_kernel<<<(unsigned)((row_dim + 255u) / 256u), 256u, 0,
                              DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)a->ptr, w16, NULL, row_dim, dim);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl patch bias");
    }
    /* learned position embedding, resampled onto this grid */
    if (ok) {
        w16 = qwen3vl_weight(model_map, model_size, weights->pos_embed,
                             (uint64_t)DS4_QWEN3VL_VISION_POS_SIDE *
                             DS4_QWEN3VL_VISION_POS_SIDE * dim, "qwen3vl pos embed");
        if (!w16) ok = 0;
    }
    if (ok) {
        qwen3vl_pos_embed_kernel<<<rows, 256u, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)a->ptr, w16, rows, grid_h, grid_w);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl pos embed");
    }

    for (uint32_t il = 0; ok && il < DS4_QWEN3VL_VISION_LAYERS; il++) {
        const ds4_qwen3vl_vision_layer_weights *lw = &weights->layer[il];
        const uint16_t *nw = qwen3vl_weight(model_map, model_size, lw->norm1_weight,
                                            dim, "qwen3vl norm1 w");
        const uint16_t *nb = qwen3vl_weight(model_map, model_size, lw->norm1_bias,
                                            dim, "qwen3vl norm1 b");
        if (!nw || !nb) { ok = 0; break; }
        /* --- attention: pre-LN -> fused QKV -> 2D RoPE -> attn -> proj -> +res */
        qwen3vl_layernorm_kernel<<<rows, 256u, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)b->ptr, (const float *)a->ptr, nw, nb, dim);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl norm1") &&
             ds4_gpu_glm53_matmul_bf16(qkv, model_map, model_size, lw->qkv_weight,
                                       dim, 3u * dim, b, rows);
        if (!ok) break;
        w16 = qwen3vl_weight(model_map, model_size, lw->qkv_bias, 3u * dim,
                             "qwen3vl qkv bias");
        if (!w16) { ok = 0; break; }
        {
            const dim3 grid(DS4_QWEN3VL_VISION_HEADS, rows, 1);
            qwen3vl_qkv_rope_kernel<<<grid, 128u, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
                    (const float *)qkv->ptr, w16, rows, grid_w);
            ok = cuda_ok(cudaGetLastError(), "qwen3vl qkv rope");
        }
        if (!ok) break;
        {
            const uint32_t threads = 128u, warps = threads / 32u;
            const size_t smem = (DS4_QWEN3VL_VISION_HEAD_DIM +
                                 (size_t)warps * DS4_QWEN3VL_VISION_HEAD_DIM +
                                 2u * warps) * sizeof(float);
            const dim3 grid(DS4_QWEN3VL_VISION_HEADS, rows, 1);
            qwen3vl_attention_kernel<<<grid, threads, smem,
                                       DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)attn->ptr, (const float *)q->ptr,
                    (const float *)k->ptr, (const float *)v->ptr, rows);
            ok = cuda_ok(cudaGetLastError(), "qwen3vl attention");
        }
        if (!ok) break;
        ok = ds4_gpu_glm53_matmul_bf16(b, model_map, model_size, lw->proj_weight,
                                       dim, dim, attn, rows);
        if (!ok) break;
        w16 = qwen3vl_weight(model_map, model_size, lw->proj_bias, dim,
                             "qwen3vl proj bias");
        if (!w16) { ok = 0; break; }
        qwen3vl_bias_kernel<<<(unsigned)((row_dim + 255u) / 256u), 256u, 0,
                              DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)b->ptr, w16, (const float *)a->ptr, row_dim, dim);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl attn residual");
        if (!ok) break;
        /* b now holds the post-attention residual; make it the running state. */
        { ds4_gpu_tensor *sw = a; a = b; b = sw; }

        /* --- MLP: pre-LN -> fc1 -> tanh GELU -> fc2 -> +res */
        nw = qwen3vl_weight(model_map, model_size, lw->norm2_weight, dim, "qwen3vl norm2 w");
        nb = qwen3vl_weight(model_map, model_size, lw->norm2_bias, dim, "qwen3vl norm2 b");
        if (!nw || !nb) { ok = 0; break; }
        qwen3vl_layernorm_kernel<<<rows, 256u, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)b->ptr, (const float *)a->ptr, nw, nb, dim);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl norm2") &&
             ds4_gpu_glm53_matmul_bf16(ff, model_map, model_size, lw->fc1_weight,
                                       dim, DS4_QWEN3VL_VISION_FF, b, rows);
        if (!ok) break;
        w16 = qwen3vl_weight(model_map, model_size, lw->fc1_bias,
                             DS4_QWEN3VL_VISION_FF, "qwen3vl fc1 bias");
        if (!w16) { ok = 0; break; }
        qwen3vl_gelu_tanh_kernel<<<(unsigned)((row_ff + 255u) / 256u), 256u, 0,
                                   DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)ff->ptr, w16, row_ff, DS4_QWEN3VL_VISION_FF);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl gelu") &&
             ds4_gpu_glm53_matmul_bf16(b, model_map, model_size, lw->fc2_weight,
                                       DS4_QWEN3VL_VISION_FF, dim, ff, rows);
        if (!ok) break;
        w16 = qwen3vl_weight(model_map, model_size, lw->fc2_bias, dim, "qwen3vl fc2 bias");
        if (!w16) { ok = 0; break; }
        qwen3vl_bias_kernel<<<(unsigned)((row_dim + 255u) / 256u), 256u, 0,
                              DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)b->ptr, w16, (const float *)a->ptr, row_dim, dim);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl mlp residual");
        if (!ok) break;
        { ds4_gpu_tensor *sw = a; a = b; b = sw; }
    }

    /* --- merger: LN over 1152, then four consecutive rows ARE one 4608 vector.
     * No shuffle kernel: the patch order already put each 2x2 block's four
     * patches on four consecutive rows, so the reshape is free. */
    if (ok) {
        const uint16_t *nw = qwen3vl_weight(model_map, model_size,
                weights->merger_norm_weight, dim, "qwen3vl merger norm w");
        const uint16_t *nb = qwen3vl_weight(model_map, model_size,
                weights->merger_norm_bias, dim, "qwen3vl merger norm b");
        if (!nw || !nb) ok = 0;
        if (ok) {
            qwen3vl_layernorm_kernel<<<rows, 256u, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)b->ptr, (const float *)a->ptr, nw, nb, dim);
            ok = cuda_ok(cudaGetLastError(), "qwen3vl merger norm");
        }
    }
    if (ok) {
        ok = ds4_gpu_glm53_matmul_bf16(mfc1, model_map, model_size,
                                       weights->merger_fc1_weight,
                                       DS4_QWEN3VL_VISION_MERGED,
                                       DS4_QWEN3VL_VISION_MERGED, b, merged_rows);
    }
    if (ok) {
        w16 = qwen3vl_weight(model_map, model_size, weights->merger_fc1_bias,
                             DS4_QWEN3VL_VISION_MERGED, "qwen3vl merger fc1 bias");
        if (!w16) ok = 0;
    }
    if (ok) {
        qwen3vl_gelu_erf_kernel<<<(unsigned)((merged_in + 255u) / 256u), 256u, 0,
                                  DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)mfc1->ptr, w16, merged_in, DS4_QWEN3VL_VISION_MERGED);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl merger gelu") &&
             ds4_gpu_glm53_matmul_bf16(mout, model_map, model_size,
                                       weights->merger_fc2_weight,
                                       DS4_QWEN3VL_VISION_MERGED,
                                       DS4_QWEN3VL_VISION_OUT, mfc1, merged_rows);
    }
    if (ok) {
        w16 = qwen3vl_weight(model_map, model_size, weights->merger_fc2_bias,
                             DS4_QWEN3VL_VISION_OUT, "qwen3vl merger fc2 bias");
        if (!w16) ok = 0;
    }
    if (ok) {
        qwen3vl_bias_kernel<<<(unsigned)((merged_out + 255u) / 256u), 256u, 0,
                              DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)mout->ptr, w16, NULL, merged_out, DS4_QWEN3VL_VISION_OUT);
        ok = cuda_ok(cudaGetLastError(), "qwen3vl merger bias");
    }
    if (ok) ok = ds4_gpu_end_commands();
    if (ok) ok = ds4_gpu_tensor_read(mout, 0, out, merged_out * sizeof(float));

cleanup:
    ds4_gpu_tensor_free(patch); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(qkv); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(attn); ds4_gpu_tensor_free(ff);
    ds4_gpu_tensor_free(mfc1); ds4_gpu_tensor_free(mout);
    return ok;
}
