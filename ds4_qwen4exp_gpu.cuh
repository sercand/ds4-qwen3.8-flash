/* qwen4exp (Qwen3.8-Flash-Next) GPU operations.
 *
 * Included by ds4_cuda.cu so these kernels can use its device helpers and the
 * ds4_gpu_tensor layout, while keeping the architecture's code in one place.
 *
 * Two conventions run through everything here:
 *
 *  - The residual is DS4_N_HC parallel streams of n_embd, stored HC-outer /
 *    embd-inner, so one token occupies hc * n_embd contiguous floats.  There
 *    are no pre-attention or pre-FFN norms; the hyper-connection mix replaces
 *    them, and the final mix replaces the output norm.
 *  - Every norm weight in the file already carries the Gemma +1, and ssm_a
 *    already carries -exp(A_log).  The converter applied both, so the kernels
 *    below apply neither.  Re-applying either is silent: the model stays
 *    fluent and goes subtly wrong.
 */
#include "ds4_q4e_page.h"
#include "ds4_ple_stream.h"
#include "cuda/exl3/ds4_exl3.h"

/* Block-wide reductions over at most 32 warps.  ds4_cuda.cu already has the
 * warp-level pair; these lift them to a whole block, which the grouped norms
 * and the router need. */
__device__ static float q4e_block_sum(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    v = warp_sum_f32(v);
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    v = (threadIdx.x < n_warp) ? s[threadIdx.x] : 0.0f;
    if (warp == 0u) v = warp_sum_f32(v);
    __shared__ float total;
    if (threadIdx.x == 0u) total = v;
    __syncthreads();
    return total;
}

__device__ static float q4e_block_max(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    v = warp_max_f32(v);
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    v = (threadIdx.x < n_warp) ? s[threadIdx.x] : -INFINITY;
    if (warp == 0u) v = warp_max_f32(v);
    __shared__ float total;
    if (threadIdx.x == 0u) total = v;
    __syncthreads();
    return total;
}

/* ---------------------------------------------------------------------------
 * Hyper-connections.
 * ------------------------------------------------------------------------ */

/* Values one thread of q4e_hc_combine_norm_kernel carries: ceil(n_embd /
 * threads), 10 at the shipped 2560 with 256 threads. */
#define Q4E_HC_FUSE_MAX 16u

/* res[t][h][e] = embed[t][e] for every stream h.  The residual starts as the
 * embedding tiled across the streams, not zero padded. */
__global__ static void q4e_hc_init_kernel(
        float *res, const float *embed, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t e = (uint32_t)(idx % n_embd);
    const uint64_t t = idx / ((uint64_t)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e];
}

/* res = embed broadcast over the streams, plus a per-stream term.  This is the
 * MTP draft input: fc_embedding(emb) added to every stream of fc_hidden(res). */
__global__ static void q4e_hc_init_add_kernel(
        float *res, const float *embed, const float *h,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t e = (uint32_t)(idx % n_embd);
    const uint64_t t = idx / ((uint64_t)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e] + h[idx];
}

/* Grouped RMSNorm: the mean square is taken over one n_embd stream, but the
 * affine vector spans all hc * n_embd.  Normalizing over the full width
 * instead is a common and silent mistake. */
__global__ static void q4e_hc_norm_kernel(
        float *out, const float *res, const float *w,
        uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint32_t group = blockIdx.x;            /* one (token, stream) pair */
    const uint32_t h = group % n_hc;
    const float *src = res + (uint64_t)group * n_embd;
    float *dst = out + (uint64_t)group * n_embd;
    const float *wh = w + (uint64_t)h * n_embd;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float v = src[i];
        sum += v * v;
    }
    sum = q4e_block_sum(sum);
    const float scale = rsqrtf(sum / (float)n_embd + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = src[i] * scale * wh[i];
    }
}

/* y = silu(x / n_hc).  The division sits inside the activation, before the up
 * projection, and dropping it changes the gate materially. */
__global__ static void q4e_scale_silu_kernel(float *x, float inv_hc, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * inv_hc;
    x[i] = v / (1.0f + __expf(-v));
}

/* Collapse the gated streams into the block input: mean over hc, not sum. */
__global__ static void q4e_hc_collapse_kernel(
        float *out, const float *xn, const float *up,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t t = blockIdx.x;
    const float inv = 1.0f / (float)n_hc;
    const float *xn_t = xn + (uint64_t)t * n_hc * n_embd;
    const float *up_t = up + (uint64_t)t * n_hc * n_embd;
    float *out_t = out + (uint64_t)t * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            const uint32_t o = h * n_embd + e;
            const float g = 1.0f / (1.0f + __expf(-up_t[o]));
            acc += xn_t[o] * g;
        }
        out_t[e] = acc * inv;
    }
}

/* The residual update that closes a block, fused with the group-RMSNorm that
 * opens the next one.
 *
 * Both traverse the whole hc * n_embd residual, which at a prefill chunk is
 * the widest stream in the model (2048 tokens x 40 KiB), and the update's
 * result is exactly the norm's input.  Run apart they cost four passes over
 * it -- update reads and writes it, the norm reads it and writes the normed
 * copy -- and the block output once per stream; run together, three, and the
 * block output once.  Measured at 26k: 1.08 + 0.78 ms -> 1.17 ms per layer
 * and per direction.
 *
 * One block per token, not per (token, stream): that is what lets the block
 * output be read once for all n_hc streams.  The per-stream reduction, the
 * element-to-thread mapping and every expression are the ones the two
 * separate kernels used, so the result is bit-identical to them.
 *
 * The streams are walked grid-strided, so the launcher can spread them over
 * gridDim.y blocks instead: at one token, four blocks that each read the
 * block output are worth more than one block that reads it once and then
 * reduces four times in sequence (decode measured 22.6 against 23.9 tok/s
 * before this was a grid stride).
 *
 * `w` is the next mix's norm weight, or NULL where no mix follows in this
 * pass (the last layer) or something else reads the residual in between (the
 * PLE block); then this is the residual update alone and the mix norms its
 * own input. */
__global__ static void q4e_hc_combine_norm_kernel(
        float *res, float *xn, const float *block_out, const float *inject,
        const float *w, uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint32_t t = blockIdx.x;
    const float *src = block_out + (uint64_t)t * n_embd;
    const uint32_t h0 = blockIdx.y;
    const uint32_t hstep = gridDim.y;

    /* The block output, held for every stream, and the updated residual, held
     * for the norm's second pass.  Both loops are unrolled over a fixed bound
     * with a guard rather than run to a runtime trip count: a dynamic index
     * into a local array is a local-memory array, and the spill traffic cost
     * more than the pass it saved (1.51 ms against 1.17 ms per call at 26k).
     * The launcher refuses shapes above the bound. */
    float sv[Q4E_HC_FUSE_MAX];
#pragma unroll
    for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
        const uint32_t i = threadIdx.x + n * blockDim.x;
        sv[n] = (i < n_embd) ? src[i] : 0.0f;
    }

    for (uint32_t h = h0; h < n_hc; h += hstep) {
        const float wgt =
            2.0f / (1.0f + __expf(-inject[(uint64_t)t * n_hc + h] / (float)n_hc));
        float *dst = res + ((uint64_t)t * n_hc + h) * n_embd;
        float vv[Q4E_HC_FUSE_MAX];
        float sum = 0.0f;
#pragma unroll
        for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint32_t i = threadIdx.x + n * blockDim.x;
            if (i < n_embd) {
                float v = dst[i];
                v += sv[n] * wgt;
                dst[i] = v;
                vv[n] = v;
                sum += v * v;
            }
        }
        if (!w) continue;
        sum = q4e_block_sum(sum);
        const float scale = rsqrtf(sum / (float)n_embd + eps);
        const float *wh = w + (uint64_t)h * n_embd;
        float *out = xn + ((uint64_t)t * n_hc + h) * n_embd;
#pragma unroll
        for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint32_t i = threadIdx.x + n * blockDim.x;
            if (i < n_embd) out[i] = vv[n] * scale * wh[i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Mixture of experts.
 * ------------------------------------------------------------------------ */

/* Block-wide argmax, ties resolved toward the lower index.  The tie rule is
 * load bearing: it is what a stable descending argsort does, and the router
 * has to agree with the reference on which expert a tie picks. */
__device__ static void q4e_block_argmax(float v, int32_t i, float *out_v, int32_t *out_i) {
    __shared__ float s_v[32];
    __shared__ int32_t s_i[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t d = 16u; d > 0u; d >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, d);
        const int32_t oi = __shfl_down_sync(0xffffffffu, i, d);
        if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) { v = ov; i = oi; }
    }
    if (lane == 0u) { s_v[warp] = v; s_i[warp] = i; }
    __syncthreads();
    const uint32_t n_warp = (blockDim.x + 31u) >> 5u;
    if (warp == 0u) {
        v = (lane < n_warp) ? s_v[lane] : -INFINITY;
        i = (lane < n_warp) ? s_i[lane] : -1;
        for (uint32_t d = 16u; d > 0u; d >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, d);
            const int32_t oi = __shfl_down_sync(0xffffffffu, i, d);
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) { v = ov; i = oi; }
        }
        if (lane == 0u) { s_v[0] = v; s_i[0] = i; }
    }
    __syncthreads();
    *out_v = s_v[0];
    *out_i = s_i[0];
}

/* Softmax over all experts, then the top n_used by probability, then
 * renormalize over the chosen ones.
 *
 * The selection is n_used rounds of block argmax over a shared copy of the
 * logits, each round masking off what it took.  Doing it serially in one
 * thread instead -- the obvious way for k = 10 -- spills the running top-k
 * arrays to local memory and leaves 255 threads idle, which measured at 72 us
 * per layer, over 3 ms of every decode step. */
__global__ static void q4e_moe_route_kernel(
        int32_t *ids, float *weights, const float *logits,
        uint32_t n_expert, uint32_t n_used) {
    const uint32_t t = blockIdx.x;
    const float *row = logits + (uint64_t)t * n_expert;
    extern __shared__ float q4e_route_smem[];      /* n_expert */

    float m = -INFINITY;
    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const float v = row[e];
        q4e_route_smem[e] = v;
        m = fmaxf(m, v);
    }
    const float s_max = q4e_block_max(m);

    float sum = 0.0f;
    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
        sum += __expf(q4e_route_smem[e] - s_max);
    }
    const float inv_sum = 1.0f / q4e_block_sum(sum);

    /* 16 covers DS4_MAX_EXPERT_USED (10) with room to spare; the entry point
     * below rejects anything wider. */
    __shared__ float chosen_w[16];
    __shared__ int32_t chosen_i[16];
    for (uint32_t s = 0; s < n_used; s++) {
        float best = -INFINITY;
        int32_t best_i = -1;
        /* e only grows, so keeping the first of equal values is the lower id. */
        for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
            const float v = q4e_route_smem[e];
            if (v > best) { best = v; best_i = (int32_t)e; }
        }
        float top = 0.0f;
        int32_t top_i = -1;
        q4e_block_argmax(best, best_i, &top, &top_i);
        if (threadIdx.x == 0) {
            chosen_w[s] = top;
            chosen_i[s] = top_i;
            if (top_i >= 0) q4e_route_smem[top_i] = -INFINITY;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (uint32_t s = 0; s < n_used; s++) {
            chosen_w[s] = __expf(chosen_w[s] - s_max) * inv_sum;
            wsum += chosen_w[s];
        }
        /* The reference clamps the divisor before dividing. */
        wsum = fmaxf(wsum, 1e-20f);
        for (uint32_t s = 0; s < n_used; s++) {
            ids[(uint64_t)t * n_used + s] = chosen_i[s];
            weights[(uint64_t)t * n_used + s] = chosen_w[s] / wsum;
        }
    }
}

/* out = silu(gate) * up over the routed intermediate. */
__global__ static void q4e_swiglu_kernel(float *out, const float *gate, const float *up, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

/* Weighted sum of the n_used expert outputs for each token.  The MoE matmul
 * writes column-major over (token, slot), so slot s of token t is column
 * t * n_used + s. */
__global__ static void q4e_moe_combine_kernel(
        float *out, const float *down, const float *weights,
        uint32_t n_embd, uint32_t n_used) {
    const uint32_t t = blockIdx.y;
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd;
         e += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_used; s++) {
            const uint64_t col = (uint64_t)t * n_used + s;
            acc += down[col * n_embd + e] * weights[col];
        }
        out[(uint64_t)t * n_embd + e] = acc;
    }
}

/* moe_out += shared_out * sigmoid(shared_logit) -- the shared expert carries a
 * single scalar gate per token, not a per-channel one. */
__global__ static void q4e_shared_add_kernel(
        float *out, const float *shared, const float *logit, uint32_t n_embd) {
    const uint32_t t = blockIdx.y;
    const float g = 1.0f / (1.0f + __expf(-logit[t]));
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd;
         e += gridDim.x * blockDim.x) {
        out[(uint64_t)t * n_embd + e] += shared[(uint64_t)t * n_embd + e] * g;
    }
}

/* ---------------------------------------------------------------------------
 * Gated DeltaNet (the 36 linear-attention layers).
 * ------------------------------------------------------------------------ */

/* Causal depthwise conv over the q|k|v channel block, then silu.
 *
 * The kernel window reaches DS4_N_GDN_CONV - 1 tokens back, so the tail of the
 * previous chunk is carried in a per-sequence state of that many columns.  The
 * state is updated to the new tail here, which is what makes a chunked prefill
 * produce the same result as a single-shot one.
 *
 * `ckpt`, when given, receives the window as it stands after each token t <
 * min(ckpt_cap, n_tok - 1), laid out like `state` at ckpt[t].  Speculative
 * verify runs several tokens in one call and, when the target rejects a
 * suffix, restores the checkpoint of the last accepted one. */
__global__ static void q4e_gdn_conv_kernel(
        float *out, float *state, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = kernel - 1u;

    /* Gather this channel's history followed by the new tokens. */
    float win[8];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];

    const float *wc = w + (uint64_t)c * kernel;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float cur = x[(uint64_t)t * channels + c];
        float acc = 0.0f;
        for (uint32_t k = 0; k < hist; k++) acc += wc[k] * win[k];
        acc += wc[hist] * cur;
        out[(uint64_t)t * channels + c] = acc / (1.0f + __expf(-acc));
        for (uint32_t k = 0; k + 1u < hist; k++) win[k] = win[k + 1];
        if (hist > 0) win[hist - 1] = cur;
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + (uint64_t)t * hist * channels;
            for (uint32_t i = 0; i < hist; i++) ck[(uint64_t)i * channels + c] = win[i];
        }
    }

    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}

/* L2-normalize each 128-wide q and k head in place.  This is an L2 norm, not
 * an RMS norm: there is no weight and no epsilon-scaled mean. */
__global__ static void q4e_gdn_l2norm_kernel(
        float *qk, uint32_t head_dim, uint32_t n_heads, uint32_t stride, uint32_t offset) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    float *v = qk + (uint64_t)t * stride + offset + (uint64_t)h * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += v[i] * v[i];
    sum = q4e_block_sum(sum);
    const float inv = rsqrtf(sum + 1e-12f);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) v[i] *= inv;
    (void)n_heads;
}

/* decay = softplus(alpha + dt_bias) * ssm_a and beta = sigmoid(beta_proj).
 * ssm_a already holds -exp(A_log), so decay comes out negative and the
 * recurrence below exponentiates it directly. */
__global__ static void q4e_gdn_gates_kernel(
        float *decay, float *beta,
        const float *alpha_proj, const float *beta_proj,
        const float *dt_bias, const float *a, uint32_t n_head, uint32_t n_tok) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_head * n_tok) return;
    const uint32_t h = (uint32_t)(i % n_head);
    const float x = alpha_proj[i] + dt_bias[h];
    /* softplus, guarded the way the reference is so large x does not overflow */
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    decay[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + __expf(-beta_proj[i]));
}

/* The gated delta rule, one block per value head.
 *
 * The 128x128 state lives in shared memory for the whole chunk, so a prefill
 * reads and writes it once instead of once per token.  It is stored
 * transposed, m[j][i] = S[i][j], so the three inner products below all walk a
 * contiguous row.
 *
 * Value head h reads query/key head h % n_head_k -- modulo, not divide.  That
 * detail is easy to get backwards and produces plausible garbage.
 *
 * Per token:   S *= exp(decay)
 *              delta[j] = (v[j] - <S[:,j], k>) * beta
 *              S[:,j]  += delta[j] * k
 *              out[j]   = <S[:,j], q> / sqrt(head_dim)
 */
#define Q4E_GDN_SPLIT 4u   /* lanes per state column */
__global__ static void __launch_bounds__(512) q4e_gdn_recurrent_kernel(
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        uint32_t head_dim, uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    extern __shared__ float q4e_gdn_smem[];    /* head_dim * (head_dim + 1) */
    float *s_state = q4e_gdn_smem;
    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    /* Q4E_GDN_SPLIT adjacent lanes share one column j of S (row j of the
     * transposed state); each owns a quarter of the 128-long inner products
     * and the quarters meet in a shuffle.  One thread per column left the SM
     * with four warps and a 128-deep dependent FMA chain per token: 10.8 ms
     * for a 2048-token chunk.  The split reassociates the sums (a change of
     * the same kind as the chunk width, see trap 4). */
    const uint32_t j = threadIdx.x / Q4E_GDN_SPLIT;
    const uint32_t sub = threadIdx.x % Q4E_GDN_SPLIT;
    const uint32_t seg = head_dim / Q4E_GDN_SPLIT;          /* 32 */
    const float scale = rsqrtf((float)head_dim);
    const uint32_t row_stride = head_dim + 1u;

    float *m = state + (uint64_t)h * head_dim * head_dim;
    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        s_state[r * row_stride + c] = m[idx];
    }
    __syncthreads();

    __shared__ float s_k[128];
    __shared__ float s_q[128];
    __shared__ float s_kq;

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *base = qkv + (uint64_t)t * stride;
        const float *q_d = base + (uint64_t)hk * head_dim;
        const float *k_d = base + k_offset + (uint64_t)hk * head_dim;
        const float *v_d = base + v_offset + (uint64_t)h * head_dim;
        float kq_part = 0.0f;
        if (sub == 0u && j < head_dim) {
            const float kv = k_d[j];
            const float qv = q_d[j];
            s_k[j] = kv;
            s_q[j] = qv;
            kq_part = kv * qv;
        }
        kq_part = q4e_block_sum(kq_part);
        if (threadIdx.x == 0u) s_kq = kq_part;
        __syncthreads();

        const uint64_t gi = (uint64_t)t * n_head_v + h;
        const float dec = __expf(decay[gi]);
        const float bt = beta[gi];

        /* This lane's quarter of the row, walked from a rotated start so the
         * four lanes of a column, and the eight columns of a warp, hit
         * different banks. */
        float *row = s_state + (uint64_t)j * row_stride;
        const uint32_t i0 = sub * seg;
        float sk = 0.0f, sq = 0.0f;
        for (uint32_t n = 0; n < seg; n++) {
            const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
            const float mv = row[i];
            sk = fmaf(mv, s_k[i], sk);
            sq = fmaf(mv, s_q[i], sq);
        }
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        const float delta = (v_d[j] - dec * sk) * bt;

        if (sub == 0u) {
            attn_out[((uint64_t)t * n_head_v + h) * head_dim + j] =
                (dec * sq + delta * s_kq) * scale;
        }
        for (uint32_t n = 0; n < seg; n++) {
            const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
            row[i] = fmaf(row[i], dec, delta * s_k[i]);
        }
        __syncthreads();
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + ((uint64_t)t * n_head_v + h) * head_dim * head_dim;
            for (uint32_t r = sub; r < head_dim; r += Q4E_GDN_SPLIT) ck[r * head_dim + j] = s_state[r * row_stride + j];
        }
    }

    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        m[idx] = s_state[r * row_stride + c];
    }
}

/* Chunked-parallel gated delta rule, for prefill.
 *
 * The recurrence above is exact but sequential: one block per value head
 * walks the tokens one at a time, and every token pays a dependent pass over
 * the whole 128x128 state.  At a 2048-token chunk that is 322 ms of a 26k
 * step.  The chunked-parallel (FLA / FlashQLA) form rewrites a run of C
 * tokens as matmuls against the state at the run's start, so the state is
 * touched twice per run instead of C times, and the only sequential work left
 * is the run-to-run carry.
 *
 * With g_t = log decay_t, G_t = sum_{s<=t} g_s inside the run and S_0 the
 * state at its start, unrolling S_t = e^{g_t}(I - b_t k_t k_t^T) S_{t-1} +
 * b_t k_t v_t^T gives
 *
 *     S_t   = e^{G_t} S_0 + sum_{u<=t} e^{G_t - G_u} k_u d_u^T
 *     d_t   = b_t (v_t - e^{G_t} S_0^T k_t) - sum_{u<t} A[t][u] d_u
 *     A[t][u] = b_t e^{G_t - G_u} (k_u . k_t)                       (u < t)
 *     o_t   = (e^{G_t} S_0^T q_t + sum_{u<=t} P[t][u] d_u) / sqrt(D)
 *     P[t][u] = e^{G_t - G_u} (q_t . k_u)                           (u <= t)
 *
 * Every exponent is a difference G_t - G_u with u <= t, i.e. a decay in
 * [0, 1]: no 1/e^G rescaling and nothing to overflow, which is the reason for
 * this arrangement rather than the textbook "divide the keys by the cumulative
 * decay" one.
 *
 * C = 64 because the shared-memory working set is what bounds it: keys,
 * deltas, the C x C coefficient matrix and a staged q tile come to 97 KiB at
 * 64, against the 99 KiB a block may opt into here, and the per-token
 * arithmetic grows as 3D + C (65 k MAC per token per head at C = 64 against
 * the sequential form's 49 k), so a wider run buys fewer carries at a rising
 * cost and does not fit beside the state anyway.
 *
 * The state stays in registers: thread (j, sub) holds the 16 keys
 * [sub*16, sub*16+16) of value column j, so the block is 128 columns x
 * Q4E_GDN_CSPLIT lanes = 1024 threads and the eight lanes of a column reduce
 * through shuffles.  Each lane reads its slice as float4 in a rotated order
 * (the sequential kernel's trick) so the eight lanes of a column hit eight
 * different shared-memory bank groups.  Phases, per run: keys in,
 * coefficients, right-hand sides, the C-step forward substitution, outputs (q
 * staged in tiles so the key rows can stay resident), and the carry into the
 * next run.  Only the staging phases and the two gram matrices need block
 * barriers -- a column's deltas are written and read inside one warp, so the
 * substitution's C dependent steps use __syncwarp().
 *
 * What it costs and why, measured on a 2048-token chunk with a standalone
 * microbenchmark against the sequential kernel (same inputs, agreement 4e-5%
 * relative L2): 8.7 -> 5.3 ms per layer, 1.65x.  The remainder is
 * shared-memory traffic, not arithmetic: every token's keys and queries are
 * re-read by all 32 warps (each owns one value column and 16 keys of it), so
 * a run moves ~19 MB through shared memory for ~4 M multiply-adds.  Cutting
 * that further means giving each thread several value columns so one shared
 * read feeds several products; the two arrangements tried (four columns per
 * thread with 32-lane reductions, and the wider gram tiles) both lost more to
 * the longer reductions than they saved, so this is where it stands.
 *
 * Numerics differ from the sequential kernel -- these are different sums, not
 * a reassociation of the same ones -- so this path is gated on the oracle,
 * not on the prefill fingerprint.  It also produces no per-token state, so it
 * cannot fill the speculative-rollback checkpoints; the launcher keeps runs
 * that need them on the sequential kernel. */
#define Q4E_GDN_CHUNK 64u    /* tokens per run */
#define Q4E_GDN_QTILE 32u    /* q rows staged at a time in the output phase */
#define Q4E_GDN_CSPLIT 8u    /* lanes per value column: 1024 threads, 32 warps */

__global__ static void __launch_bounds__(128u * Q4E_GDN_CSPLIT) q4e_gdn_chunk_kernel(
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride, uint32_t n_tok) {
    const uint32_t D = 128u;                  /* head_dim, checked by the launcher */
    const uint32_t C = Q4E_GDN_CHUNK;
    const uint32_t DS = D + 1u;               /* delta row stride: +1 so the
                                               * per-column reads below spread
                                               * over all 32 banks */
    extern __shared__ float q4e_gdn_chunk_smem[];
    float *s_K = q4e_gdn_chunk_smem;          /* [C][D]  keys */
    float *s_D = s_K + (uint64_t)C * D;       /* [C][DS] rhs, then deltas */
    float *s_A = s_D + (uint64_t)C * DS;      /* [C][C]  coefficients, then P */
    float *s_Q = s_A + (uint64_t)C * C;       /* [QTILE][D] staged queries */
    float *s_G = s_Q + (uint64_t)Q4E_GDN_QTILE * D;   /* [C] cumulative log decay */
    float *s_b = s_G + C;                     /* [C] beta */

    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    const uint32_t NV4 = D / (4u * Q4E_GDN_CSPLIT);   /* float4 slots per lane */
    const uint32_t j = threadIdx.x / Q4E_GDN_CSPLIT;
    const uint32_t sub = threadIdx.x % Q4E_GDN_CSPLIT;
    const uint32_t i0 = sub * (D / Q4E_GDN_CSPLIT);
    const uint32_t rot = (sub * NV4) / Q4E_GDN_CSPLIT;  /* float4 rotation, see above */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warp = blockDim.x >> 5u;
    const float scale = rsqrtf((float)D);

    float4 *m4 = (float4 *)(state + ((uint64_t)h * D + j) * D + i0);
    float4 sreg[NV4];
#pragma unroll
    for (uint32_t n4 = 0; n4 < NV4; n4++) sreg[n4] = m4[(n4 + rot) & (NV4 - 1u)];

    for (uint32_t t0 = 0; t0 < n_tok; t0 += C) {
        const uint32_t n = (n_tok - t0 < C) ? (n_tok - t0) : C;

        /* Gates in parallel, then the inclusive prefix sum of the log decays
         * by one thread over shared memory: 64 dependent adds are cheaper
         * than a scan's barriers, but 128 strided global loads on that same
         * serial path are not. */
        for (uint32_t t = threadIdx.x; t < n; t += blockDim.x) {
            const uint64_t gi = (uint64_t)(t0 + t) * n_head_v + h;
            s_G[t] = decay[gi];
            s_b[t] = beta[gi];
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < n; t++) { acc += s_G[t]; s_G[t] = acc; }
        }
        for (uint32_t idx = threadIdx.x; idx < n * D; idx += blockDim.x) {
            const uint32_t t = idx / D, i = idx - t * D;
            s_K[idx] = qkv[(uint64_t)(t0 + t) * stride + k_offset + hk * D + i];
        }
        __syncthreads();

        /* A[t][u] = b_t e^{G_t - G_u} (k_u . k_t), strictly below the
         * diagonal.  One warp per entry: its 32 lanes take four keys each so
         * both rows are read as contiguous float4. */
        for (uint32_t rr = warp; rr < C; rr += n_warp) {
            /* One warp per row, its own key row hoisted into registers: each
             * column then costs one shared read instead of two.  Rows are
             * handed out in low/high pairs (rr and C-1-rr) because row t has
             * t columns, so a warp that takes both gets a constant C-1.
             *
             * The bound is C and not the run length: the pairing sends the
             * second half of the rows to rr >= n_warp, so stopping at a short
             * run's n left rows [32, 96-n) of a partial run unwritten -- the
             * substitution then read stale shared memory for every run whose
             * length fell in [33, 63].  Rows past n are skipped below, one
             * predicate instead of a truncated loop. */
            const uint32_t t = ((rr / n_warp) % 2u == 0u) ? rr : (C - 1u - (rr - n_warp));
            if (t >= n) continue;
            const float4 kt = ((const float4 *)(s_K + t * D))[lane];
            const float bt = s_b[t], gt = s_G[t];
            for (uint32_t u = 0; u < t; u++) {
                const float4 ku = ((const float4 *)(s_K + u * D))[lane];
                float dot = kt.x * ku.x + kt.y * ku.y + kt.z * ku.z + kt.w * ku.w;
                dot = warp_sum_f32(dot);
                if (lane == 0u) s_A[t * C + u] = bt * __expf(gt - s_G[u]) * dot;
            }
        }
        __syncthreads();

        /* rhs_t = b_t (v_t - e^{G_t} S^T k_t). */
        for (uint32_t t = 0; t < n; t++) {
            const float4 *k4 = (const float4 *)(s_K + t * D + i0);
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
            for (uint32_t n4 = 0; n4 < NV4; n4++) {
                const float4 kv = k4[(n4 + rot) & (NV4 - 1u)];
                p0 = fmaf(sreg[n4].x, kv.x, p0);
                p1 = fmaf(sreg[n4].y, kv.y, p1);
                p2 = fmaf(sreg[n4].z, kv.z, p2);
                p3 = fmaf(sreg[n4].w, kv.w, p3);
            }
            float p = (p0 + p1) + (p2 + p3);
#pragma unroll
            for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (sub == 0u) {
                const float v = qkv[(uint64_t)(t0 + t) * stride + v_offset + h * D + j];
                s_D[t * DS + j] = s_b[t] * (v - p * __expf(s_G[t]));
            }
        }
        __syncwarp();

        /* Forward substitution: d_t = rhs_t - sum_{u<t} A[t][u] d_u.  C
         * dependent steps, but each is one rank-1 update over the value
         * dimension instead of a pass over the state -- that trade is what
         * the chunk buys. */
        for (uint32_t t = 1u; t < n; t++) {
            float acc = 0.0f;
            for (uint32_t u = sub; u < t; u += Q4E_GDN_CSPLIT) {
                acc = fmaf(s_A[t * C + u], s_D[u * DS + j], acc);
            }
#pragma unroll
            for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (sub == 0u) s_D[t * DS + j] -= acc;
            __syncwarp();
        }

        /* Outputs.  The queries are staged a tile at a time so the keys stay
         * resident: P needs both, and both at full width do not fit. */
        for (uint32_t tb = 0; tb < n; tb += Q4E_GDN_QTILE) {
            const uint32_t nt = (n - tb < Q4E_GDN_QTILE) ? (n - tb) : Q4E_GDN_QTILE;
            for (uint32_t idx = threadIdx.x; idx < nt * D; idx += blockDim.x) {
                const uint32_t tt = idx / D, i = idx - tt * D;
                s_Q[idx] = qkv[(uint64_t)(t0 + tb + tt) * stride + hk * D + i];
            }
            __syncthreads();

            for (uint32_t pair = warp; pair < nt * C; pair += n_warp) {
                const uint32_t tt = pair / C, u = pair - tt * C;
                const uint32_t t = tb + tt;
                float v = 0.0f;
                if (u <= t) {
                    const float4 a = ((const float4 *)(s_Q + tt * D))[lane];
                    const float4 b = ((const float4 *)(s_K + u * D))[lane];
                    float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    dot = warp_sum_f32(dot);
                    v = __expf(s_G[t] - s_G[u]) * dot;
                }
                if (lane == 0u) s_A[pair] = v;
            }
            __syncthreads();

            for (uint32_t tt = 0; tt < nt; tt++) {
                const uint32_t t = tb + tt;
                const float4 *q4 = (const float4 *)(s_Q + tt * D + i0);
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
                for (uint32_t n4 = 0; n4 < NV4; n4++) {
                    const float4 qv = q4[(n4 + rot) & (NV4 - 1u)];
                    p0 = fmaf(sreg[n4].x, qv.x, p0);
                    p1 = fmaf(sreg[n4].y, qv.y, p1);
                    p2 = fmaf(sreg[n4].z, qv.z, p2);
                    p3 = fmaf(sreg[n4].w, qv.w, p3);
                }
                const float p = (p0 + p1) + (p2 + p3);
                float acc = 0.0f;
                for (uint32_t u = sub; u <= t; u += Q4E_GDN_CSPLIT) {
                    acc = fmaf(s_A[tt * C + u], s_D[u * DS + j], acc);
                }
                acc = fmaf(p, __expf(s_G[t]), acc);
#pragma unroll
                for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (sub == 0u) {
                    attn_out[((uint64_t)(t0 + t) * n_head_v + h) * D + j] = acc * scale;
                }
            }
            __syncthreads();
        }

        /* Carry: S <- e^{G_last} S + sum_t e^{G_last - G_t} k_t d_t^T. */
        const float g_last = s_G[n - 1u];
        const float dec = __expf(g_last);
#pragma unroll
        for (uint32_t n4 = 0; n4 < NV4; n4++) {
            sreg[n4].x *= dec; sreg[n4].y *= dec;
            sreg[n4].z *= dec; sreg[n4].w *= dec;
        }
        for (uint32_t t = 0; t < n; t++) {
            const float d = s_D[t * DS + j] * __expf(g_last - s_G[t]);
            const float4 *k4 = (const float4 *)(s_K + t * D + i0);
#pragma unroll
            for (uint32_t n4 = 0; n4 < NV4; n4++) {
                const float4 kv = k4[(n4 + rot) & (NV4 - 1u)];
                sreg[n4].x = fmaf(kv.x, d, sreg[n4].x);
                sreg[n4].y = fmaf(kv.y, d, sreg[n4].y);
                sreg[n4].z = fmaf(kv.z, d, sreg[n4].z);
                sreg[n4].w = fmaf(kv.w, d, sreg[n4].w);
            }
        }
        __syncthreads();   /* s_K / s_D are rewritten by the next run */
    }

#pragma unroll
    for (uint32_t n4 = 0; n4 < NV4; n4++) m4[(n4 + rot) & (NV4 - 1u)] = sreg[n4];
}

/* Per-head RMSNorm of the delta-rule output, then the sigmoid output gate.
 * ssm_norm is the one norm in the model the converter does NOT offset by one,
 * so it is applied raw here just like every other weight. */
__global__ static void q4e_gdn_out_gate_kernel(
        float *out, const float *attn, const float *z, const float *w,
        uint32_t head_dim, uint32_t n_head, float eps) {
    const uint32_t group = blockIdx.x;         /* (token, head) */
    const float *src = attn + (uint64_t)group * head_dim;
    const float *zr = z + (uint64_t)group * head_dim;
    float *dst = out + (uint64_t)group * head_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += src[i] * src[i];
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float g = 1.0f / (1.0f + __expf(-zr[i]));
        dst[i] = src[i] * scale * w[i] * g;
    }
    (void)n_head;
}

/* ---------------------------------------------------------------------------
 * PLE n-gram block (layer DS4_PLE_LAYER only).
 * ------------------------------------------------------------------------ */

/* Decode the streamed IQ4_NL rows into the flat per-token embedding.  Row r of
 * the gather belongs to token r / n_heads and occupies head slot r % n_heads,
 * so the heads land head-slowest inside the 2560-wide result, matching the
 * reference's reshape of a [head_dim, n_heads * n_tok] gather. */
__global__ static void q4e_ple_dequant_kernel(
        float *out, const uint8_t *rows, uint32_t head_dim, uint32_t n_heads,
        uint32_t row_bytes, uint32_t n_rows) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    const uint8_t *src = rows + (uint64_t)r * row_bytes;
    const uint32_t t = r / n_heads;
    const uint32_t h = r % n_heads;
    float *dst = out + (uint64_t)t * n_heads * head_dim + (uint64_t)h * head_dim;

    const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                              1,   13,  25,  38,  53,  69,  89, 113 };
    for (uint32_t b = threadIdx.x; b < head_dim / 32u; b += blockDim.x) {
        const uint8_t *blk = src + (uint64_t)b * 18u;
        const float d = __half2float(*(const __half *)blk);
        for (uint32_t j = 0; j < 16u; j++) {
            const uint8_t q = blk[2 + j];
            dst[b * 32u + j]       = d * (float)kv[q & 0x0Fu];
            dst[b * 32u + j + 16u] = d * (float)kv[q >> 4];
        }
    }
}

/* The EXL3 checkpoint's row codec, one 160-thread block per row: bit m of
 * weight i sits at ring position ((i - m/6) mod 160) * 6 + m%6 of a 6-bit
 * tail-biting trellis; the 16-bit state decodes through the mul1 codebook,
 * then the row's fp16 scale and the hash head's bias.  Port of exllamav3's
 * ngram_dequant_kernel, rounding where it rounds so rows match its table. */
__global__ static void q4e_ple_dequant_exl3_kernel(
        float *out, const uint8_t *rows, const __half *bias,
        uint32_t head_dim, uint32_t n_heads, uint32_t row_bytes, uint32_t n_rows) {
    const uint32_t r = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (r >= n_rows) return;
    const uint32_t words = row_bytes / 2u;
    __shared__ uint16_t sw[64];
    if (i < words) sw[i] = ((const uint16_t *)(rows + (uint64_t)r * row_bytes))[i];
    __syncthreads();
    if (i >= head_dim) return;

    const float scale = __half2float(__ushort_as_half(sw[0]));
    uint32_t state = 0;
#pragma unroll 4
    for (uint32_t m = 0; m < 16u; m++) {
        int pos = (int)i - (int)(m / 6u);
        if (pos < 0) pos += (int)head_dim;
        const uint32_t sb = (uint32_t)pos * 6u + m % 6u;
        state |= ((uint32_t)(sw[1 + (sb >> 4)] >> (sb & 15u)) & 1u) << m;
    }
    const uint32_t prod = state * 0x83DCD12Du;
    const float h = 1024.0f + (float)((prod & 0xFFu) + ((prod >> 8) & 0xFFu) +
                                      ((prod >> 16) & 0xFFu) + ((prod >> 24) & 0xFFu));
    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float k_bias = __half2float(__ushort_as_half(0xc931));
    const float cb = __half2float(__float2half_rn(h * k_inv + k_bias));

    const uint32_t t = r / n_heads;
    const uint32_t hd = r % n_heads;
    const float b = __half2float(bias[(uint64_t)hd * head_dim + i]);
    out[((uint64_t)t * n_heads + hd) * head_dim + i] = __half2float(__float2half_rn(fmaf(cb, scale, b)));
}

/* gate[t][h] = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))) with
 * s = <key[t][h], query[t][h]> / sqrt(n_embd), then broadcast the value across
 * the streams.  The signed square root before the sigmoid is easy to miss and
 * changes the gate's shape completely. */
__global__ static void q4e_ple_gated_value_kernel(
        float *gv, const float *key, const float *query, const float *value,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint64_t base = ((uint64_t)t * n_hc + h) * n_embd;

    float dot = 0.0f;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        dot += key[base + e] * query[base + e];
    }
    dot = q4e_block_sum(dot) * rsqrtf((float)n_embd);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float g = 1.0f / (1.0f + __expf(-copysignf(mag, dot)));

    const float *v = value + (uint64_t)t * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) gv[base + e] = v[e] * g;
}

/* Depthwise causal conv dilated by the n-gram size: taps sit at t, t-3, t-6,
 * t-9 for the shipped kernel of 4.  The history is (kernel - 1) * dilation
 * columns, carried per sequence so a chunked prefill matches a single-shot
 * one, and silu is applied to the result. */
__global__ static void q4e_ple_conv_kernel(
        float *out, float *state, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t dilation, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = (kernel - 1u) * dilation;

    float win[16];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];

    const float *wc = w + (uint64_t)c * kernel;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float cur = x[(uint64_t)t * channels + c];
        float acc = wc[kernel - 1u] * cur;
        for (uint32_t k = 0; k + 1u < kernel; k++) {
            /* tap k reads (kernel - 1 - k) * dilation positions back */
            const uint32_t back = (kernel - 1u - k) * dilation;
            acc += wc[k] * win[hist - back];
        }
        out[(uint64_t)t * channels + c] = acc / (1.0f + __expf(-acc));
        for (uint32_t i = 0; i + 1u < hist; i++) win[i] = win[i + 1];
        if (hist > 0) win[hist - 1] = cur;
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + (uint64_t)t * hist * channels;
            for (uint32_t i = 0; i < hist; i++) ck[(uint64_t)i * channels + c] = win[i];
        }
    }

    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}

__global__ static void q4e_add2_kernel(float *res, const float *a, const float *b, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] += a[i] + b[i];
}

/* ---------------------------------------------------------------------------
 * C entry points.  Weights are addressed by model offset and resolved through
 * the same weight cache every other qwen4exp-adjacent kernel uses.
 * ------------------------------------------------------------------------ */

static const float *q4e_weight(const void *model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes,
                               const ds4_gpu_tensor *anchor, const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) return NULL;
    return (const float *)cuda_resolve_weight_ptr(
            model_map, offset, bytes, ds4_tensor_device_idx(anchor), label);
}

/* Q8_0 mat-vec for a narrow output.
 *
 * The stock decode kernel gives one warp per output row, so a 320-row
 * projection fills 40 thread blocks and leaves most of the memory system
 * idle.  Both hyper-connection down projections have exactly that shape and
 * run 97 times per token, which made them the single largest phase of a
 * decode step.  Here one block owns one row and its threads split that row's
 * q8 blocks, so the grid is out_dim blocks wide and each row is read as one
 * contiguous stream.  Restricted to in_dim % 32 == 0, which every qwen4exp
 * projection satisfies. */
__global__ static void q4e_matvec_q8_0_narrow_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks) {
    const unsigned char *wr = w + (uint64_t)blockIdx.x * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = 0;
#pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            dot = __dp4a(load_i8x4_i32_unaligned(qs + i),
                         load_i8x4_i32_aligned(xqb + i), dot);
        }
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = q4e_block_sum(acc);
    if (threadIdx.x == 0u) out[blockIdx.x] = acc;
}

/* Q8_0 matmul for a handful of activation rows (2..16) -- the speculative
 * verify batch, 1 + K tokens.  The single-token kernels above and in ds4_cuda.cu read
 * a weight once per row they serve; the generic multi-row dispatch falls off
 * that path and was 60% of a verify step.  Both kernels below load each
 * weight block once and dot it with every row's quantized activation, so the
 * traffic is the single-token traffic plus a few KiB of activations.  Which
 * one runs is the same shape split as the single-token case: a block per
 * output row for the narrow projections, a warp per row otherwise. */
template <int NT>
__device__ __forceinline__ static void q4e_q8_0_rows_block(
        const unsigned char *wb, const int8_t *xq, const float *xscale,
        uint64_t b, uint64_t blocks, float *acc) {
    const float scale = __half2float(*(const __half *)wb);
    const int8_t *qs = (const int8_t *)(wb + 2);
    int32_t wq[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) wq[i] = load_i8x4_i32_unaligned(qs + 4u * i);
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int8_t *xqb = xq + ((uint64_t)t * blocks + b) * 32;
        int dot = 0;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            dot = __dp4a(wq[i], load_i8x4_i32_aligned(xqb + 4u * i), dot);
        }
        acc[t] += scale * xscale[(uint64_t)t * blocks + b] * (float)dot;
    }
}

template <int NT>
__global__ static void q4e_matmul_q8_0_rows_narrow_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks, uint32_t out_dim) {
    const unsigned char *wr = w + (uint64_t)blockIdx.x * blocks * 34;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        q4e_q8_0_rows_block<NT>(wr + b * 34, xq, xscale, b, blocks, acc);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = q4e_block_sum(acc[t]);
        if (threadIdx.x == 0u) out[(uint64_t)t * out_dim + blockIdx.x] = v;
    }
}

template <int NT>
__global__ static void q4e_matmul_q8_0_rows_warp_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * blocks * 34;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        q4e_q8_0_rows_block<NT>(wr + b * 34, xq, xscale, b, blocks, acc);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = v;
    }
}

template <int NT>
static int q4e_matmul_q8_0_rows_launch(
        float *out, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t blocks, uint32_t out_dim) {
    if (out_dim <= 1024u && blocks >= 32u) {
        const unsigned threads = blocks <= 128u ? 128u : 256u;
        q4e_matmul_q8_0_rows_narrow_kernel<NT><<<out_dim, threads, 0, cuda_decode_stream()>>>(
                out, w, xq, xscale, blocks, out_dim);
    } else {
        q4e_matmul_q8_0_rows_warp_kernel<NT><<<(out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
                out, w, xq, xscale, blocks, out_dim);
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp q8_0 rows matmul") ? 1 : -1;
}

/* Returns 0 when the shape or row count is not one this path takes. */
extern "C" int ds4_gpu_q4e_matmul_q8_0_rows(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !model_map || n_tok < 2u || n_tok > 16u) return 0;
    if ((in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float)) return 0;

    const int tier = ds4_tensor_device_idx(out);
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, tier, "q4e_rows_q8_0");
    if (!w) return 0;

    const uint64_t xq_bytes = (uint64_t)n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + (uint64_t)n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(tier, tmp_bytes, "q4e rows prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);

    dim3 qgrid((unsigned)blocks, n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp rows quantize")) return 0;

    float *o = (float *)out->ptr;
    switch (n_tok) {
    case 2u: return q4e_matmul_q8_0_rows_launch<2>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 3u: return q4e_matmul_q8_0_rows_launch<3>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 4u: return q4e_matmul_q8_0_rows_launch<4>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 5u: return q4e_matmul_q8_0_rows_launch<5>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 6u: return q4e_matmul_q8_0_rows_launch<6>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 7u: return q4e_matmul_q8_0_rows_launch<7>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 8u: return q4e_matmul_q8_0_rows_launch<8>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 9u: return q4e_matmul_q8_0_rows_launch<9>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 10u: return q4e_matmul_q8_0_rows_launch<10>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 11u: return q4e_matmul_q8_0_rows_launch<11>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 12u: return q4e_matmul_q8_0_rows_launch<12>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 13u: return q4e_matmul_q8_0_rows_launch<13>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 14u: return q4e_matmul_q8_0_rows_launch<14>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    case 15u: return q4e_matmul_q8_0_rows_launch<15>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    default: return q4e_matmul_q8_0_rows_launch<16>(o, w, xq, xscale, blocks, (uint32_t)out_dim);
    }
}

/* BF16 matmul for 2..16 activation rows: a warp per output row, lane-strided
 * bf16x2 weight loads dotted with every row's f32 activation.  The indexer
 * projections (2560 -> 128 and 2560 -> 512) are BF16 and run on every
 * forward; at a few rows the cuBLAS GemmEx path (with its f32 -> bf16
 * conversion) cost more than the arithmetic. */
template <int NT>
__global__ static void q4e_matmul_bf16_rows_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const __nv_bfloat162 *wr = (const __nv_bfloat162 *)(w + (uint64_t)row * in_dim);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint32_t i = lane; i < in_dim / 2u; i += 32u) {
        const float2 wv = __bfloat1622float2(wr[i]);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float2 xv = *(const float2 *)(x + (uint64_t)t * in_dim + 2u * i);
            acc[t] = fmaf(wv.x, xv.x, acc[t]);
            acc[t] = fmaf(wv.y, xv.y, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = v;
    }
}

template <int NT>
static int q4e_matmul_bf16_rows_launch(float *out, const __nv_bfloat16 *w, const float *x,
                                       uint32_t in_dim, uint32_t out_dim) {
    q4e_matmul_bf16_rows_kernel<NT><<<(out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
            out, w, x, in_dim, out_dim);
    return cuda_ok(cudaGetLastError(), "qwen4exp bf16 rows matmul") ? 1 : -1;
}

/* Returns 0 when the shape or row count is not one this path takes. */
extern "C" int ds4_gpu_q4e_matmul_bf16_rows(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !model_map || n_tok < 2u || n_tok > 16u) return 0;
    if ((in_dim & 1u) != 0u || in_dim == 0u || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * 2u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float)) return 0;
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, ds4_tensor_device_idx(out), "q4e_rows_bf16");
    if (!w) return 0;
    float *o = (float *)out->ptr;
    const float *xf = (const float *)x->ptr;
    switch (n_tok) {
    case 2u: return q4e_matmul_bf16_rows_launch<2>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 3u: return q4e_matmul_bf16_rows_launch<3>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 4u: return q4e_matmul_bf16_rows_launch<4>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 5u: return q4e_matmul_bf16_rows_launch<5>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 6u: return q4e_matmul_bf16_rows_launch<6>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 7u: return q4e_matmul_bf16_rows_launch<7>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 8u: return q4e_matmul_bf16_rows_launch<8>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 9u: return q4e_matmul_bf16_rows_launch<9>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 10u: return q4e_matmul_bf16_rows_launch<10>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 11u: return q4e_matmul_bf16_rows_launch<11>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 12u: return q4e_matmul_bf16_rows_launch<12>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 13u: return q4e_matmul_bf16_rows_launch<13>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 14u: return q4e_matmul_bf16_rows_launch<14>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 15u: return q4e_matmul_bf16_rows_launch<15>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    default: return q4e_matmul_bf16_rows_launch<16>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    }
}

/* fp16 weights: the EXL3 repack keeps the checkpoint's unquantized tensors
 * (hyper-connection mix, router, PLE projections, GDN alpha/beta, injection
 * logits) as fp16.  Up to 16 rows: a warp per output row reading half2 pairs
 * once and dotting them with every row's f32 activation, the bf16 kernel
 * above with the other 16-bit type.  Above 16 rows (prefill) cuBLAS. */
template <int NT>
__global__ static void q4e_matmul_f16_rows_kernel(
        float *out, const __half *w, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const __half2 *wr = (const __half2 *)(w + (uint64_t)row * in_dim);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint32_t i = lane; i < in_dim / 2u; i += 32u) {
        const float2 wv = __half22float2(wr[i]);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float2 xv = *(const float2 *)(x + (uint64_t)t * in_dim + 2u * i);
            acc[t] = fmaf(wv.x, xv.x, acc[t]);
            acc[t] = fmaf(wv.y, xv.y, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = v;
    }
}

template <int NT>
static int q4e_matmul_f16_rows_launch(float *out, const __half *w, const float *x,
                                      uint32_t in_dim, uint32_t out_dim) {
    q4e_matmul_f16_rows_kernel<NT><<<(out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
            out, w, x, in_dim, out_dim);
    return cuda_ok(cudaGetLastError(), "qwen4exp f16 rows matmul");
}

__global__ static void q4e_f32_to_f16_kernel(__half *out, const float *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

extern "C" int ds4_gpu_q4e_matmul_f16(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !model_map || n_tok == 0u) return 0;
    if ((in_dim & 1u) != 0u || in_dim == 0u || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * 2u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float)) return 0;
    const int tier = ds4_tensor_device_idx(out);
    const __half *w = (const __half *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, tier, "q4e_f16");
    if (!w) return 0;
    float *o = (float *)out->ptr;
    const float *xf = (const float *)x->ptr;
    switch (n_tok) {
    case 1u: return q4e_matmul_f16_rows_launch<1>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 2u: return q4e_matmul_f16_rows_launch<2>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 3u: return q4e_matmul_f16_rows_launch<3>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 4u: return q4e_matmul_f16_rows_launch<4>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 5u: return q4e_matmul_f16_rows_launch<5>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 6u: return q4e_matmul_f16_rows_launch<6>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 7u: return q4e_matmul_f16_rows_launch<7>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 8u: return q4e_matmul_f16_rows_launch<8>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 9u: return q4e_matmul_f16_rows_launch<9>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 10u: return q4e_matmul_f16_rows_launch<10>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 11u: return q4e_matmul_f16_rows_launch<11>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 12u: return q4e_matmul_f16_rows_launch<12>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 13u: return q4e_matmul_f16_rows_launch<13>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 14u: return q4e_matmul_f16_rows_launch<14>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 15u: return q4e_matmul_f16_rows_launch<15>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    case 16u: return q4e_matmul_f16_rows_launch<16>(o, w, xf, (uint32_t)in_dim, (uint32_t)out_dim);
    default: break;
    }
    if (!g_cublas_ready) return 0;
    const uint64_t n_in = (uint64_t)n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc_on(tier, n_in * sizeof(__half), "q4e f16 activations");
    if (!xh) return 0;
    q4e_f32_to_f16_kernel<<<(unsigned)((n_in + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(xh, xf, n_in);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp f16 activation conversion")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
            cuda_cublas_for_tier(tier), CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
            w, CUDA_R_16F, (int)in_dim, xh, CUDA_R_16F, (int)in_dim, &beta,
            o, CUDA_R_32F, (int)out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "qwen4exp f16 matmul");
}

/* EXL3 trellis weights.  The payload is the tiles followed by suh[k] and
 * svh[n] (per expert for stacked tensors); the kernels apply the scales and
 * the 128-point Hadamards themselves, so this only resolves pointers. */
static bool q4e_exl3_ptrs(const void *model_map, uint64_t model_size, uint64_t offset,
                          uint64_t bytes, uint32_t bits, uint64_t in_dim, uint64_t out_dim,
                          uint64_t n_expert, const ds4_gpu_tensor *anchor, const char *label,
                          const uint8_t **tiles, const uint8_t **suh, const uint8_t **svh) {
    const uint64_t tile_bytes = in_dim * out_dim * bits / 8u;
    if (bytes != n_expert * (tile_bytes + 2u * (in_dim + out_dim))) {
        fprintf(stderr, "ds4: %s: EXL3 payload is %llu bytes, expected %llu\n", label,
                (unsigned long long)bytes,
                (unsigned long long)(n_expert * (tile_bytes + 2u * (in_dim + out_dim))));
        return false;
    }
    const uint8_t *w = (const uint8_t *)q4e_weight(model_map, model_size, offset, bytes, anchor, label);
    if (!w) return false;
    *tiles = w;
    *suh = w + n_expert * tile_bytes;
    *svh = *suh + n_expert * in_dim * 2u;
    return true;
}

extern "C" int ds4_gpu_q4e_matmul_exl3(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t bytes, uint32_t bits, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !model_map || n_tok == 0u) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float)) return 0;
    const uint8_t *tiles, *suh, *svh;
    if (!q4e_exl3_ptrs(model_map, model_size, weight_offset, bytes, bits, in_dim, out_dim, 1u,
                       out, "qwen4exp exl3 dense", &tiles, &suh, &svh)) return 0;
    return ds4_exl3_gemm((const float *)x->ptr, tiles, suh, svh, (float *)out->ptr,
                         n_tok, (uint32_t)in_dim, (uint32_t)out_dim, bits, cuda_decode_stream());
}

/* Rows of x replicated once per routing slot: mgemm reads one input slab per
 * slot, and slot t * n_used + s is token t. */
__global__ static void q4e_moe_rows_per_slot_kernel(
        float *out, const float *x, uint32_t in_dim, uint32_t n_used, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t slot = i / in_dim;
    out[i] = x[(slot / n_used) * in_dim + (i - slot * in_dim)];
}

/* Routed EXL3 experts as an mgemm fan-out: one slot per (token, expert) with
 * the same input row, outputs [n_tok * n_used][out_dim] -- the layout the
 * swiglu and down stages already expect.  `x_rows` is the per-slot input
 * when the caller has one (down), else the per-token input to replicate. */
static int q4e_moe_exl3(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, bool x_per_slot,
                        const ds4_gpu_tensor *ids, const void *model_map, uint64_t model_size,
                        uint64_t weight_offset, uint64_t bytes, uint32_t bits,
                        uint32_t out_dim, uint32_t in_dim, uint32_t n_tok,
                        uint32_t n_expert, uint32_t n_used, const char *label) {
    const uint8_t *tiles, *suh, *svh;
    if (!q4e_exl3_ptrs(model_map, model_size, weight_offset, bytes, bits, in_dim, out_dim,
                       n_expert, out, label, &tiles, &suh, &svh)) return 0;
    const uint32_t n_slots = n_tok * n_used;
    const float *xin = (const float *)x->ptr;
    int per_slot = x_per_slot ? 1 : 0;
    if (!x_per_slot && n_tok > 1u) {
        const uint64_t n = (uint64_t)n_slots * in_dim;
        float *rep = (float *)cuda_tmp_alloc_on(ds4_tensor_device_idx(out), n * sizeof(float),
                                                "qwen4exp exl3 moe rows");
        if (!rep) return 0;
        q4e_moe_rows_per_slot_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
                rep, xin, in_dim, n_used, n);
        if (!cuda_ok(cudaGetLastError(), "qwen4exp exl3 moe rows")) return 0;
        xin = rep;
        per_slot = 1;
    }
    return ds4_exl3_mgemm(xin, per_slot, tiles, suh, svh, (const int32_t *)ids->ptr, n_slots,
                          (float *)out->ptr, 1u, in_dim, out_dim, bits, cuda_decode_stream());
}

/* F32 mat-vec with the contraction split across blocks.
 *
 * The generic F32 matmul gives one block per output row, which is fine for the
 * 512-wide router but leaves the hyper-connection injection logits -- 10240
 * inputs into 4 outputs, 97 times a token -- running on four blocks.  Here the
 * K range is cut into n_split pieces when the output is narrow, and a second
 * pass adds the pieces up.  Partials are split-major, so at n_split == 1 the
 * first kernel already writes the final layout and the second is skipped. */
__global__ static void q4e_matvec_f32_kernel(
        float *out, const float *w, const float *x,
        uint32_t in_dim, uint32_t chunk) {
    const uint32_t row = blockIdx.x;
    const uint32_t sp = blockIdx.y;
    const uint32_t i0 = sp * chunk;
    const uint32_t i1 = min(i0 + chunk, in_dim);
    const float *wr = w + (uint64_t)row * in_dim;
    float acc = 0.0f;
    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) acc += wr[i] * x[i];
    acc = q4e_block_sum(acc);
    if (threadIdx.x == 0u) out[(uint64_t)sp * gridDim.x + row] = acc;
}

__global__ static void q4e_matvec_f32_combine_kernel(
        float *out, const float *partial, uint32_t out_dim, uint32_t n_split) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_split; s++) acc += partial[(uint64_t)s * out_dim + row];
    out[row] = acc;
}

extern "C" int ds4_gpu_q4e_matvec_f32(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u) return 0;
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < in_dim * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) return 0;

    const int tier = ds4_tensor_device_idx(out);
    const float *w = q4e_weight(model_map, model_size, weight_offset, weight_bytes,
                                out, "q4e_f32_matvec");
    if (!w) return 0;

    /* Aim for a few hundred blocks without cutting a block's share of the
     * contraction below what its 256 threads can keep busy. */
    uint32_t n_split = 1u;
    while (out_dim * (uint64_t)(n_split * 2u) <= 512u &&
           in_dim / (n_split * 2u) >= 512u) {
        n_split *= 2u;
    }
    const uint32_t chunk = (uint32_t)((in_dim + n_split - 1u) / n_split);

    float *dst = (float *)out->ptr;
    float *partial = dst;
    if (n_split > 1u) {
        /* The split loop keeps out_dim * n_split at or below 512, so one small
         * dedicated allocation covers every shape.  It must not come from the
         * shared cuda_tmp scratch: that same buffer holds the activation
         * quantization for the surrounding matmuls, and a grow request from
         * any of them would free it out from under a captured graph. */
        static float *g_q4e_f32_partial = NULL;
        if (!g_q4e_f32_partial) {
            if (cudaMalloc((void **)&g_q4e_f32_partial, 512u * sizeof(float)) != cudaSuccess) {
                (void)cudaGetLastError();
                g_q4e_f32_partial = NULL;
                return 0;
            }
        }
        if ((uint64_t)out_dim * n_split > 512u) return 0;
        partial = g_q4e_f32_partial;
    }
    (void)tier;
    const dim3 grid((unsigned)out_dim, n_split, 1);
    q4e_matvec_f32_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            partial, w, (const float *)x->ptr, (uint32_t)in_dim, chunk);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp f32 matvec")) return -1;
    if (n_split > 1u) {
        q4e_matvec_f32_combine_kernel<<<(unsigned)((out_dim + 127u) / 128u), 128, 0,
                                        cuda_decode_stream()>>>(
                dst, partial, (uint32_t)out_dim, n_split);
        if (!cuda_ok(cudaGetLastError(), "qwen4exp f32 matvec combine")) return -1;
    }
    return 1;
}

/* Returns 0 when the shape is not one this kernel wants, so the caller falls
 * back to the generic dispatch. */
extern "C" int ds4_gpu_q4e_matvec_q8_0_narrow(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x) {
    if (!out || !x || !model_map) return 0;
    if ((in_dim & 31u) != 0u || in_dim < 1024u || out_dim == 0u) return 0;
    /* Above this width the stock warp-per-row grid is already wide enough. */
    if (out_dim > 1024u) return 0;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < in_dim * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) return 0;

    const int tier = ds4_tensor_device_idx(out);
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, tier, "q4e_narrow_q8_0");
    if (!w) return 0;

    const uint64_t scale_offset = (blocks * 32u + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(tier, tmp_bytes, "q4e narrow prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);

    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp narrow quantize")) return 0;

    const unsigned threads = blocks <= 128u ? 128u : 256u;
    q4e_matvec_q8_0_narrow_kernel<<<(unsigned)out_dim, threads, 0,
                                    cuda_decode_stream()>>>(
            (float *)out->ptr, w, xq, xscale, blocks);
    return cuda_ok(cudaGetLastError(), "qwen4exp narrow q8_0 matvec") ? 1 : -1;
}

extern "C" int ds4_gpu_q4e_hc_init(
        ds4_gpu_tensor *res, const ds4_gpu_tensor *embed,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!res || !embed || !n_embd || !n_hc || !n_tok) return 0;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (res->bytes < total * sizeof(float)) return 0;
    q4e_hc_init_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)res->ptr, (const float *)embed->ptr, n_embd, n_hc, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen4exp hc init");
}

extern "C" int ds4_gpu_q4e_hc_init_add(
        ds4_gpu_tensor *res, const ds4_gpu_tensor *embed, const ds4_gpu_tensor *h,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!res || !embed || !h || !n_embd || !n_hc || !n_tok) return 0;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (res->bytes < total * sizeof(float) || h->bytes < total * sizeof(float)) return 0;
    q4e_hc_init_add_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)res->ptr, (const float *)embed->ptr, (const float *)h->ptr,
            n_embd, n_hc, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen4exp hc init add");
}

extern "C" int ds4_gpu_q4e_hc_norm(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *res,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!out || !res || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)n_embd * n_hc * sizeof(float), out,
                                "qwen4exp hc norm");
    if (!w) return 0;
    q4e_hc_norm_kernel<<<(unsigned)(n_tok * n_hc), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)res->ptr, w, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp hc norm");
}

extern "C" int ds4_gpu_q4e_scale_silu(ds4_gpu_tensor *x, float inv_scale, uint64_t n) {
    if (!x || !n) return 0;
    q4e_scale_silu_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, inv_scale, n);
    return cuda_ok(cudaGetLastError(), "qwen4exp scale silu");
}

extern "C" int ds4_gpu_q4e_hc_collapse(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *xn, const ds4_gpu_tensor *up,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out || !xn || !up || !n_tok) return 0;
    q4e_hc_collapse_kernel<<<(unsigned)n_tok, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)xn->ptr, (const float *)up->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "qwen4exp hc collapse");
}

/* res += block_out (x) 2*sigmoid(inject / n_hc) per stream, and, when a norm
 * follows in this pass, the next mix's normed input in the same pass.
 * `fuse_norm` = 0 leaves xn alone (see q4e_hc_combine_norm_kernel).  The
 * injection logits come from the normed input of the *paired* mix, which the
 * caller must keep alive across the block. */
extern "C" int ds4_gpu_q4e_hc_combine(
        ds4_gpu_tensor *res, ds4_gpu_tensor *xn, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *inject, const void *model_map, uint64_t model_size,
        uint64_t next_norm_offset, int fuse_norm,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!res || !block_out || !inject || !n_tok) return 0;
    const unsigned threads = 256u;
    if (n_embd > threads * Q4E_HC_FUSE_MAX) return 0;
    const float *w = NULL;
    if (fuse_norm) {
        if (!xn) return 0;
        w = q4e_weight(model_map, model_size, next_norm_offset,
                       (uint64_t)n_embd * n_hc * sizeof(float), res,
                       "qwen4exp hc combine norm");
        if (!w) return 0;
    }
    /* One block per token once there are enough tokens to fill the machine
     * that way (the block output is then read once, not once per stream);
     * below that, one block per (token, stream) so a decode step's four
     * groups run at once instead of in sequence. */
    const dim3 grid(n_tok, n_tok >= 256u ? 1u : n_hc, 1);
    q4e_hc_combine_norm_kernel<<<grid, threads, 0, cuda_decode_stream()>>>(
            (float *)res->ptr, xn ? (float *)xn->ptr : NULL,
            (const float *)block_out->ptr, (const float *)inject->ptr,
            w, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp hc combine");
}

extern "C" int ds4_gpu_q4e_moe_route(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        uint32_t n_expert, uint32_t n_used, uint32_t n_tok) {
    if (!ids || !weights || !logits || !n_tok || n_used > 16u) return 0;
    /* The selection rounds read the logits out of shared memory. */
    const size_t smem = (size_t)n_expert * sizeof(float);
    q4e_moe_route_kernel<<<(unsigned)n_tok, 256, smem, cuda_decode_stream()>>>(
            (int32_t *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr,
            n_expert, n_used);
    return cuda_ok(cudaGetLastError(), "qwen4exp moe route");
}

extern "C" int ds4_gpu_q4e_swiglu(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint64_t n) {
    if (!out || !gate || !up || !n) return 0;
    q4e_swiglu_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n);
    return cuda_ok(cudaGetLastError(), "qwen4exp swiglu");
}

extern "C" int ds4_gpu_q4e_moe_combine(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *down, const ds4_gpu_tensor *weights,
        uint32_t n_embd, uint32_t n_used, uint32_t n_tok) {
    if (!out || !down || !weights || !n_tok) return 0;
    const dim3 grid((n_embd + 255u) / 256u, n_tok, 1);
    q4e_moe_combine_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)down->ptr, (const float *)weights->ptr,
            n_embd, n_used);
    return cuda_ok(cudaGetLastError(), "qwen4exp moe combine");
}

extern "C" int ds4_gpu_q4e_shared_add(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *shared, const ds4_gpu_tensor *logit,
        uint32_t n_embd, uint32_t n_tok) {
    if (!out || !shared || !logit || !n_tok) return 0;
    const dim3 grid((n_embd + 255u) / 256u, n_tok, 1);
    q4e_shared_add_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)shared->ptr,
            (const float *)logit->ptr, n_embd);
    return cuda_ok(cudaGetLastError(), "qwen4exp shared expert");
}

extern "C" int ds4_gpu_q4e_gdn_conv(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t channels, uint32_t kernel, uint32_t n_tok,
        ds4_gpu_tensor *ckpt, uint32_t ckpt_cap) {
    if (!out || !state || !x || !n_tok || kernel < 1u || kernel > 8u) return 0;
    if (ckpt && ckpt->bytes < (uint64_t)ckpt_cap * (kernel - 1u) * channels * sizeof(float)) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)channels * kernel * sizeof(float), out,
                                "qwen4exp gdn conv");
    if (!w) return 0;
    q4e_gdn_conv_kernel<<<(unsigned)((channels + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (float *)state->ptr, (const float *)x->ptr,
            w, channels, kernel, n_tok,
            ckpt ? (float *)ckpt->ptr : NULL, ckpt ? ckpt_cap : 0u);
    return cuda_ok(cudaGetLastError(), "qwen4exp gdn conv");
}

extern "C" int ds4_gpu_q4e_gdn_l2norm(
        ds4_gpu_tensor *qkv, uint32_t head_dim, uint32_t n_heads,
        uint32_t stride, uint32_t offset, uint32_t n_tok) {
    if (!qkv || !n_tok) return 0;
    const dim3 grid(n_heads, n_tok, 1);
    q4e_gdn_l2norm_kernel<<<grid, 128, 0, cuda_decode_stream()>>>(
            (float *)qkv->ptr, head_dim, n_heads, stride, offset);
    return cuda_ok(cudaGetLastError(), "qwen4exp gdn l2 norm");
}

extern "C" int ds4_gpu_q4e_gdn_gates(
        ds4_gpu_tensor *decay, ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *alpha_proj, const ds4_gpu_tensor *beta_proj,
        const void *model_map, uint64_t model_size,
        uint64_t dt_bias_offset, uint64_t a_offset,
        uint32_t n_head, uint32_t n_tok) {
    if (!decay || !beta || !alpha_proj || !beta_proj || !n_tok) return 0;
    const uint64_t vec_bytes = (uint64_t)n_head * sizeof(float);
    const float *dt = q4e_weight(model_map, model_size, dt_bias_offset, vec_bytes,
                                 decay, "qwen4exp gdn dt bias");
    const float *a = q4e_weight(model_map, model_size, a_offset, vec_bytes,
                                decay, "qwen4exp gdn a");
    if (!dt || !a) return 0;
    const uint64_t total = (uint64_t)n_head * n_tok;
    q4e_gdn_gates_kernel<<<(unsigned)((total + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)decay->ptr, (float *)beta->ptr,
            (const float *)alpha_proj->ptr, (const float *)beta_proj->ptr,
            dt, a, n_head, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen4exp gdn gates");
}

/* Shared-memory working set of q4e_gdn_chunk_kernel, and the one-time opt-in
 * for it.  A device that cannot give a block this much keeps prefill on the
 * sequential kernel, which is correct at any shape, only slower. */
static size_t q4e_gdn_chunk_smem_bytes(void) {
    const size_t C = Q4E_GDN_CHUNK, D = 128u;
    return (C * D + C * (D + 1u) + C * C + Q4E_GDN_QTILE * D + 2u * C) * sizeof(float);
}

/* Rows from which prefill takes the chunked kernel (DS4_QWEN4EXP_GDN_CHUNK_MIN,
 * default two runs).  0 keeps every shape on the sequential kernel, which is
 * how the two are A/B-ed against each other and against the oracle. */
static uint32_t q4e_gdn_chunk_min_tok(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_QWEN4EXP_GDN_CHUNK_MIN");
        cached = (env && env[0]) ? atoi(env) : (int)(2u * Q4E_GDN_CHUNK);
        if (cached < 0) cached = 0;
    }
    return (uint32_t)cached;
}

static int q4e_gdn_chunk_ready(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    const size_t need = q4e_gdn_chunk_smem_bytes();
    int optin = 0, dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               dev) != cudaSuccess ||
        (size_t)optin < need ||
        cudaFuncSetAttribute(q4e_gdn_chunk_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)need) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: qwen4exp chunked GDN needs %zu B of shared memory per "
                        "block, device offers %d; prefill stays sequential\n",
                need, optin);
        cached = 0;
        return 0;
    }
    cached = 1;
    return 1;
}

extern "C" int ds4_gpu_q4e_gdn_recurrent(
        ds4_gpu_tensor *attn_out, ds4_gpu_tensor *state, const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *decay, const ds4_gpu_tensor *beta,
        uint32_t head_dim, uint32_t n_head_k, uint32_t n_head_v,
        uint32_t stride, uint32_t n_tok,
        ds4_gpu_tensor *ckpt, uint32_t ckpt_cap) {
    if (!attn_out || !state || !qkv || !decay || !beta || !n_tok) return 0;
    if (head_dim != 128u) return 0;   /* s_k / s_q are sized for the shipped head */
    if (ckpt && ckpt->bytes < (uint64_t)ckpt_cap * n_head_v * head_dim * head_dim * sizeof(float)) return 0;

    /* A prefill chunk goes to the chunked-parallel kernel.  Two runs is the
     * floor: below that there is nothing to amortize and the sequential
     * kernel's single pass is cheaper.  It also produces no per-token state,
     * which is why the threshold sits far above any speculative verify (1 + K
     * rows, K <= 8) -- those are the only calls whose checkpoints are ever
     * read back, immediately, by q4e_spec_rollback -- and why decode, which
     * is one row, never reaches it.  The ckpt_cap term keeps that true even
     * if the threshold knob is set below a verify batch's row count. */
    const uint32_t chunk_min = q4e_gdn_chunk_min_tok();
    if (chunk_min && n_tok >= chunk_min && n_tok > 1u + ckpt_cap &&
        q4e_gdn_chunk_ready()) {
        const uint32_t k_offset = n_head_k * head_dim;
        const uint32_t v_offset = 2u * n_head_k * head_dim;
        q4e_gdn_chunk_kernel<<<(unsigned)n_head_v, (unsigned)head_dim * Q4E_GDN_CSPLIT,
                               q4e_gdn_chunk_smem_bytes(), cuda_decode_stream()>>>(
                (float *)attn_out->ptr, (float *)state->ptr, (const float *)qkv->ptr,
                (const float *)decay->ptr, (const float *)beta->ptr,
                n_head_k, n_head_v, k_offset, v_offset, stride, n_tok);
        return cuda_ok(cudaGetLastError(), "qwen4exp gdn chunked recurrence");
    }
    /* One float of row padding, to keep the state's shared reads off a single
     * bank -- see the kernel. */
    const size_t shared = (size_t)head_dim * (head_dim + 1u) * sizeof(float);
    static bool opted_in = false;
    if (!opted_in) {
        /* 64 KiB exceeds the default per-block limit and needs the opt-in. */
        if (cudaFuncSetAttribute(q4e_gdn_recurrent_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shared) != cudaSuccess) {
            return 0;
        }
        opted_in = true;
    }
    const uint32_t k_offset = n_head_k * head_dim;
    const uint32_t v_offset = 2u * n_head_k * head_dim;
    q4e_gdn_recurrent_kernel<<<(unsigned)n_head_v, (unsigned)head_dim * Q4E_GDN_SPLIT, shared,
                               cuda_decode_stream()>>>(
            (float *)attn_out->ptr, (float *)state->ptr, (const float *)qkv->ptr,
            (const float *)decay->ptr, (const float *)beta->ptr,
            head_dim, n_head_k, n_head_v, k_offset, v_offset, stride, n_tok,
            ckpt ? (float *)ckpt->ptr : NULL, ckpt ? ckpt_cap : 0u);
    return cuda_ok(cudaGetLastError(), "qwen4exp gdn recurrence");
}

extern "C" int ds4_gpu_q4e_gdn_out_gate(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *attn, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t head_dim, uint32_t n_head, uint32_t n_tok, float eps) {
    if (!out || !attn || !z || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)head_dim * sizeof(float), out,
                                "qwen4exp gdn norm");
    if (!w) return 0;
    q4e_gdn_out_gate_kernel<<<(unsigned)(n_tok * n_head), 128, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)attn->ptr, (const float *)z->ptr,
            w, head_dim, n_head, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp gdn out gate");
}

extern "C" int ds4_gpu_q4e_ple_dequant(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *rows,
        uint32_t head_dim, uint32_t n_heads, uint32_t row_bytes, uint32_t row_type,
        const void *model_map, uint64_t model_size, uint64_t bias_offset, uint32_t n_tok) {
    if (!out || !rows || !n_tok) return 0;
    const uint32_t n_rows = n_tok * n_heads;
    if (row_type == DS4_PLE_ROW_EXL3_K6) {
        if (head_dim != 160u || row_bytes != DS4_PLE_EXL3_ROW_BYTES) return 0;
        const __half *bias = (const __half *)q4e_weight(model_map, model_size, bias_offset,
                                                        (uint64_t)n_heads * head_dim * 2u, out,
                                                        "qwen4exp ple bias");
        if (!bias) return 0;
        q4e_ple_dequant_exl3_kernel<<<(unsigned)n_rows, 160, 0, cuda_decode_stream()>>>(
                (float *)out->ptr, (const uint8_t *)rows->ptr, bias,
                head_dim, n_heads, row_bytes, n_rows);
        return cuda_ok(cudaGetLastError(), "qwen4exp ple exl3 dequant");
    }
    q4e_ple_dequant_kernel<<<(unsigned)n_rows, 32, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const uint8_t *)rows->ptr,
            head_dim, n_heads, row_bytes, n_rows);
    return cuda_ok(cudaGetLastError(), "qwen4exp ple dequant");
}

extern "C" int ds4_gpu_q4e_ple_gated_value(
        ds4_gpu_tensor *gv, const ds4_gpu_tensor *key, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *value, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!gv || !key || !query || !value || !n_tok) return 0;
    const dim3 grid(n_tok, n_hc, 1);
    q4e_ple_gated_value_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)gv->ptr, (const float *)key->ptr, (const float *)query->ptr,
            (const float *)value->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "qwen4exp ple gate");
}

extern "C" int ds4_gpu_q4e_ple_conv(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t channels, uint32_t kernel, uint32_t dilation, uint32_t n_tok,
        ds4_gpu_tensor *ckpt, uint32_t ckpt_cap) {
    if (!out || !state || !x || !n_tok || (kernel - 1u) * dilation > 16u) return 0;
    if (ckpt && ckpt->bytes < (uint64_t)ckpt_cap * (kernel - 1u) * dilation * channels * sizeof(float)) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)channels * kernel * sizeof(float), out,
                                "qwen4exp ple conv");
    if (!w) return 0;
    q4e_ple_conv_kernel<<<(unsigned)((channels + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (float *)state->ptr, (const float *)x->ptr,
            w, channels, kernel, dilation, n_tok,
            ckpt ? (float *)ckpt->ptr : NULL, ckpt ? ckpt_cap : 0u);
    return cuda_ok(cudaGetLastError(), "qwen4exp ple conv");
}

extern "C" int ds4_gpu_q4e_add2(
        ds4_gpu_tensor *res, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n) {
    if (!res || !a || !b || !n) return 0;
    q4e_add2_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)res->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "qwen4exp add2");
}

/* ---------------------------------------------------------------------------
 * QSA attention layers (every DS4_N_FULL_ATTN_INTERVAL-th layer).
 *
 * The query projection emits a query and an output gate interleaved per head:
 * head h owns [q(head_dim) | gate(head_dim)] inside a 2 * head_dim * n_head
 * row.  Splitting the flat row down the middle instead is wrong and is the
 * single easiest mistake to make here.
 *
 * RoPE covers only the first n_rot of head_dim in NeoX pairing (i with
 * i + n_rot/2); the rest passes through.  The checkpoint's interleaved mRoPE
 * degenerates to this for text, where all three position axes carry the same
 * value.
 * ------------------------------------------------------------------------ */

__device__ static void q4e_rope_neox(
        float *v, uint32_t n_rot, float base, uint32_t pos, uint32_t lane, uint32_t nlanes) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t i = lane; i < half; i += nlanes) {
        /* Accurate powf/sincosf, not the fast-math intrinsics: the angle feeds
         * every attention layer, and __powf's error at base 1e7 is a
         * systematic rotation offset rather than noise that averages out.
         * RoPE is far off the critical path, so the precision is free. */
        const float theta = (float)pos * powf(base, -(float)(2u * i) / (float)n_rot);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = v[i];
        const float x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/* Per-head RMSNorm then partial RoPE, reading the query out of its strided
 * slot and writing it contiguous. */
__global__ static void q4e_qsa_q_norm_rope_kernel(
        float *q_out, float *gate_out, const float *qkv, const float *w,
        uint32_t head_dim, uint32_t n_head, uint32_t n_rot, float rope_base,
        const int32_t *pos, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint64_t src = (uint64_t)t * n_head * 2u * head_dim + (uint64_t)h * 2u * head_dim;
    const uint64_t dst = ((uint64_t)t * n_head + h) * head_dim;

    extern __shared__ float q4e_q_smem[];
    float *s_q = q4e_q_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = qkv[src + i];
        s_q[i] = v;
        sum += v * v;
        gate_out[dst + i] = qkv[src + head_dim + i];
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_q[i] *= scale * w[i];
    __syncthreads();
    q4e_rope_neox(s_q, n_rot, rope_base, (uint32_t)pos[t], threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) q_out[dst + i] = s_q[i];
}

/* Physical cache row of logical position p, through this sequence's page
 * table (ds4_q4e_page.h carries the geometry).  Every KV-touching kernel
 * below goes through this; a tile that is page-aligned and no wider than a
 * page translates its first position and walks the rest contiguously, which
 * is what keeps the translation to one lookup per tile. */
__device__ __forceinline__ static uint32_t q4e_kv_row(const int32_t *pages, uint32_t p) {
    return (uint32_t)pages[p >> DS4_Q4E_PAGE_SHIFT] * DS4_Q4E_PAGE_TOKENS +
           (p & DS4_Q4E_PAGE_MASK);
}

/* Key norm + RoPE straight into the f16 KV cache, and the value beside it.
 * The cache is f16 because it is read in full on every decode step; f32 would
 * double that traffic for no accuracy that survives the softmax. */
__global__ static void q4e_qsa_store_kv_kernel(
        __half *k_cache, __half *v_cache, const float *k, const float *v,
        const float *kw, uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot,
        float rope_base, const int32_t *pos, const int32_t *pages,
        uint32_t pool_slots, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint64_t src = ((uint64_t)t * n_head_kv + h) * head_dim;

    extern __shared__ float q4e_k_smem[];
    float *s_k = q4e_k_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_k[i] *= scale * kw[i];
    __syncthreads();
    q4e_rope_neox(s_k, n_rot, rope_base, (uint32_t)pos[t], threadIdx.x, blockDim.x);
    __syncthreads();

    /* The only write into the cache, and it is always at the frontier: the
     * page holding pos[t] is the last one the sequence owns. */
    const uint64_t slot = ((uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * n_head_kv + h) * head_dim;
    if (slot + head_dim > (uint64_t)pool_slots * n_head_kv * head_dim) return;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_cache[slot + i] = __float2half(s_k[i]);
        v_cache[slot + i] = __float2half(v[src + i]);
    }
}

/* Causal GQA over the resident cache with online softmax.  One block per
 * (query head, token); the 24 query heads share 2 key/value heads.
 *
 * Positions are spread over the block's warps, not over its threads: a warp
 * owns whole positions and reduces its dot product with shuffles, so nothing
 * in the position loop needs __syncthreads.  Walking positions one at a time
 * with a block-wide reduction each -- the obvious shape -- cost 1.35 us per
 * position and made attention 9% of a decode step at a 400-token context,
 * for 830 KiB of cache traffic.
 *
 * Each lane keeps head_dim / 32 accumulator slots in registers; the warps'
 * partial (max, denominator, accumulator) triples are merged at the end. */
#define Q4E_ATTN_LANE_SLOTS 8u          /* head_dim <= 32 * this */
#define Q4E_ATTN_WARPS      8u

__global__ static void q4e_qsa_attention_kernel(
        float *out, const __half *k_cache, const __half *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t last = (uint32_t)pos[t];
    const float scale = rsqrtf((float)head_dim);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = (head_dim + 31u) / 32u;
    const uint64_t qbase = ((uint64_t)t * n_head + h) * head_dim;

    float qv[Q4E_ATTN_LANE_SLOTS];
    float acc[Q4E_ATTN_LANE_SLOTS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    /* A warp walks positions Q4E_ATTN_WARPS apart, so this one pays a page
     * lookup per position; it is the A/B fallback path, not the hot one. */
    for (uint32_t p = warp; p <= last; p += Q4E_ATTN_WARPS) {
        const uint64_t kb = ((uint64_t)q4e_kv_row(pages, p) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            if (i < head_dim) dot += qv[j] * __half2float(k_cache[kb + i]);
        }
        /* Butterfly, so every lane ends up with the score. */
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    /* Merge the per-warp partials.  s_acc is warp-major so the final pass
     * reads one contiguous row per warp. */
    extern __shared__ float q4e_attn_smem[];
    float *s_acc = q4e_attn_smem;                       /* warps * head_dim */
    __shared__ float s_m[Q4E_ATTN_WARPS];
    __shared__ float s_l[Q4E_ATTN_WARPS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        if (i < head_dim) s_acc[(uint64_t)warp * head_dim + i] = acc[j];
    }
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    float gm = -INFINITY;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float den = 0.0f;
    for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) den += s_l[w] * __expf(s_m[w] - gm);
    const float inv = 1.0f / den;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float num = 0.0f;
        for (uint32_t w = 0; w < Q4E_ATTN_WARPS; w++) {
            num += s_acc[(uint64_t)w * head_dim + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/* Tiled causal GQA, for prefill.
 *
 * The decode kernel above gives one block per (head, query) and streams the
 * whole key cache past it, so a chunk of T queries reads the cache T times.
 * At 1024-token chunks that was 40% of prefill.  Here a block owns Q4E_ATTN_QT
 * queries of one head and stages each key tile in shared memory, so the cache
 * is read once per query tile instead of once per query.
 *
 * The score for key `l` lives entirely in lane `l`, which is what removes the
 * per-key warp reduction the decode kernel pays: one max-reduction per tile of
 * 32 keys rather than one per key.  s_k is stored dim-major so those reads hit
 * 32 different banks; s_v stays key-major because the accumulation walks dims. */
#define Q4E_ATTN_QT 16u          /* queries per block */
#define Q4E_ATTN_KT 32u          /* keys per tile: one per lane */
#define Q4E_ATTN_TILE_WARPS 8u

/* Key tiles start at multiples of Q4E_ATTN_KT and a page is a whole number of
 * them, so a tile never straddles a page and one translation covers it. */
static_assert(DS4_Q4E_PAGE_TOKENS % Q4E_ATTN_KT == 0u,
              "a key tile must not straddle a KV page");

__global__ static void q4e_qsa_attention_tiled_kernel(
        float *out, const __half *k_cache, const __half *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t0 = blockIdx.y * Q4E_ATTN_QT;
    if (t0 >= n_tok) return;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = head_dim / 32u;          /* dims per lane */
    const float scale = rsqrtf((float)head_dim);

    extern __shared__ char q4e_tile_smem[];
    __half *s_k = (__half *)q4e_tile_smem;                        /* [dim][KT] */
    __half *s_v = s_k + (size_t)head_dim * Q4E_ATTN_KT;           /* [KT][dim] */
    float  *s_q = (float *)(s_v + (size_t)head_dim * Q4E_ATTN_KT);/* [QT][dim] */
    float  *s_w = s_q + (size_t)Q4E_ATTN_QT * head_dim;           /* [warps][KT] */

    /* This block's queries, and the furthest key any of them may attend to. */
    const uint32_t n_q = (n_tok - t0 < Q4E_ATTN_QT) ? (n_tok - t0) : Q4E_ATTN_QT;
    uint32_t last = 0;
    for (uint32_t i = 0; i < n_q; i++) {
        const uint32_t p = (uint32_t)pos[t0 + i];
        if (p > last) last = p;
    }
    for (uint32_t idx = threadIdx.x; idx < n_q * head_dim; idx += blockDim.x) {
        const uint32_t qi = idx / head_dim, d = idx % head_dim;
        s_q[qi * head_dim + d] = q[((uint64_t)(t0 + qi) * n_head + h) * head_dim + d] * scale;
    }

    /* Two queries per warp; each lane keeps head_dim / 32 accumulator slots
     * for each of them. */
    float acc[2][Q4E_ATTN_LANE_SLOTS];
    float m[2], l[2];
    for (int s = 0; s < 2; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (uint32_t j = 0; j < slots; j++) acc[s][j] = 0.0f;
    }
    __syncthreads();

    for (uint32_t p0 = 0; p0 <= last; p0 += Q4E_ATTN_KT) {
        const uint32_t n_k = (last + 1u - p0 < Q4E_ATTN_KT) ? (last + 1u - p0) : Q4E_ATTN_KT;
        const uint32_t krow = q4e_kv_row(pages, p0);     /* one lookup per tile */
        for (uint32_t idx = threadIdx.x; idx < n_k * head_dim; idx += blockDim.x) {
            const uint32_t k = idx / head_dim, d = idx % head_dim;
            const uint64_t src = ((uint64_t)(krow + k) * n_head_kv + hkv) * head_dim + d;
            s_k[(size_t)d * Q4E_ATTN_KT + k] = k_cache[src];
            s_v[(size_t)k * head_dim + d] = v_cache[src];
        }
        __syncthreads();

        for (int s = 0; s < 2; s++) {
            const uint32_t qi = warp * 2u + (uint32_t)s;
            if (qi >= n_q) break;
            const uint32_t qpos = (uint32_t)pos[t0 + qi];
            const float *qrow = s_q + (size_t)qi * head_dim;

            /* Lane l scores key p0 + l against this query, alone. */
            float dot = 0.0f;
            if (lane < n_k && p0 + lane <= qpos) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot = fmaf(qrow[d], __half2float(s_k[(size_t)d * Q4E_ATTN_KT + lane]), dot);
                }
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
            }
            if (tile_max == -INFINITY) continue;          /* fully masked tile */
            const float m_new = fmaxf(m[s], tile_max);
            const float corr = __expf(m[s] - m_new);
            const float w = __expf(dot - m_new);          /* 0 where masked */
            s_w[(size_t)warp * Q4E_ATTN_KT + lane] = w;

            float wsum = w;
            for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            __syncwarp();
            for (uint32_t j = 0; j < slots; j++) {
                const uint32_t d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (uint32_t k = 0; k < n_k; k++) {
                    a = fmaf(s_w[(size_t)warp * Q4E_ATTN_KT + k],
                             __half2float(s_v[(size_t)k * head_dim + d]), a);
                }
                acc[s][j] = a;
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const float inv = 1.0f / l[s];
        const uint64_t base = ((uint64_t)(t0 + qi) * n_head + h) * head_dim;
        for (uint32_t j = 0; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

/* Split-key attention for decode and the speculative verify batch.
 *
 * The per-query kernel above gives one block per (head, query) and walks the
 * whole cache from that block, so at 26k context a 4-row verify pass ran 96
 * blocks that each streamed 53 MB serially: 230 ms of a 390 ms step, latency
 * bound at a few GB/s.  Here the key range is cut into Q4E_ATTN_SPLITS pieces
 * and a block owns one piece for one key/value head and up to Q4E_ATTN_QG of
 * the queries that share it -- every query head in the group, every row of the
 * batch -- so each K/V tile is staged once for 16 queries and the grid covers
 * the GPU.  Blocks leave (max, denominator, accumulator) partials that
 * q4e_qsa_attention_combine_kernel merges; with one split the result is
 * written directly.
 *
 * The split count is fixed rather than derived from the context length so the
 * launch geometry is stable under graph capture: the tiles per split come from
 * pos[] on the device, and splits past the end write an empty partial.
 *
 * Inner loops follow the tiled prefill kernel -- key l lives in lane l for the
 * scores, lane l owns dims 8l..8l+7 for the accumulation -- staged in 16-byte
 * f16 chunks so each shared load carries 8 dims and neither loop reduces.
 * Specialised to head_dim 256 (every qwen4exp attention layer). */
#define Q4E_ATTN_QG      16u          /* queries per block: 2 per warp */
#define Q4E_ATTN_SPLITS  48u          /* key splits per (kv head, query group) */
#define Q4E_ATTN_SPLIT_HD 256u
#define Q4E_ATTN_PART_STRIDE (Q4E_ATTN_SPLIT_HD + 8u)   /* m, l, pad, acc[256] */

__global__ static void __launch_bounds__(256, 2)
q4e_qsa_attention_split_kernel(
        float *out, float *part, const __half *k_cache, const __half *v_cache,
        const float *q, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint32_t KT = Q4E_ATTN_KT;
    constexpr uint32_t C8 = HD / 8u;                    /* 16-byte chunks per row */
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t hkv = blockIdx.z;
    const uint32_t split = blockIdx.y;
    const uint32_t G = n_head / n_head_kv;              /* query heads per kv head */
    const uint32_t NQ = G * n_tok;
    const uint32_t q0 = blockIdx.x * Q4E_ATTN_QG;
    if (q0 >= NQ) return;
    const uint32_t n_q = (NQ - q0 < Q4E_ATTN_QG) ? (NQ - q0) : Q4E_ATTN_QG;
    const float scale = rsqrtf((float)HD);

    /* Everything staged is f16: a 16-byte shared load then carries 8 dims,
     * which halves the shared-memory wavefronts of an f32 layout, and the
     * 41 KB footprint lets two blocks share an SM so one block's cache loads
     * overlap the other's arithmetic.  Accumulation stays f32. */
    extern __shared__ __align__(16) uint4 q4e_split_smem[];
    uint4 *s_k8 = q4e_split_smem;                          /* [C8][KT]: chunk-major */
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;                  /* [KT][C8]: key-major */
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;                  /* [QG][C8] */
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_ATTN_QG * C8); /* [warps][KT] */

    /* This block's key range. */
    const uint32_t last = (uint32_t)pos[n_tok - 1u];
    const uint32_t n_tiles = (last + KT) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t k_begin = split * tiles_per_split * KT;
    uint32_t k_end = k_begin + tiles_per_split * KT;
    if (k_end > last + 1u) k_end = last + 1u;

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (k_begin < k_end) {
        /* Stage the scaled queries as f16. */
        for (uint32_t idx = tid; idx < n_q * C8; idx += blockDim.x) {
            const uint32_t qi = idx / C8, c = idx % C8;
            const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + h) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)qi * C8 + c] = pk.u;
        }
        __syncthreads();

        for (uint32_t p0 = k_begin; p0 < k_end; p0 += KT) {
            const uint32_t n_k = (k_end - p0 < KT) ? (k_end - p0) : KT;
            /* k_begin and the stride are multiples of KT, so this tile lies in
             * one page: one page-table lookup covers all 32 keys. */
            const uint32_t krow = q4e_kv_row(pages, p0);
            /* K: lane = key, warp = chunk (4 passes cover 32 chunks); stored
             * chunk-major so a lane's score loop reads its own column. */
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = *(const uint4 *)(k_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c));
                }
            }
            /* V: lane = chunk, warp = key (4 passes cover 32 keys); key-major
             * so the accumulation reads a lane's own 8 dims of each key. */
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = *(const uint4 *)(v_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c));
                } else {
                    /* The accumulation runs the whole tile with zero weights on
                     * the tail, and 0 * NaN from stale memory is NaN. */
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            /* Scores: key p0 + lane against both of this warp's queries in one
             * pass over the staged K, four partial sums each so the FMA chain
             * is 64 deep rather than 256. */
            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= n_q) ? 0u : ((n_q - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }

#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const uint32_t qi = qi0 + (uint32_t)s;
                const uint32_t qpos = (uint32_t)pos[(q0 + qi) / G];
                const float d = (lane < n_k && p0 + lane <= qpos) ? dot[s] : -INFINITY;
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) {
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                }
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);          /* 0 where masked */
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();

                /* Lane owns dims 8*lane .. 8*lane+7: one 16-byte load per key,
                 * lanes contiguous, so no bank conflicts.  Masked keys carry a
                 * zero weight, so the loop always runs the full tile. */
#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t k = 0; k < KT; k++) {
                    const float wk = wrow[k];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)k * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]);
                    acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]);
                    acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]);
                    acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]);
                    acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }

    /* Epilogue: the final row when there is one split, a partial otherwise. */
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
        const uint64_t qg = (uint64_t)t * n_head + h;
        if (n_splits == 1u) {
            const float inv = 1.0f / l[s];
            float4 *dst = (float4 *)(out + qg * HD + 8u * lane);
            dst[0] = make_float4(acc[s][0] * inv, acc[s][1] * inv, acc[s][2] * inv, acc[s][3] * inv);
            dst[1] = make_float4(acc[s][4] * inv, acc[s][5] * inv, acc[s][6] * inv, acc[s][7] * inv);
        } else {
            float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
            if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
            float4 *dst = (float4 *)(prow + 8u + 8u * lane);
            dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
            dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
        }
    }
}

/* One block per (row, head); thread d merges dim d across the splits. */
__global__ static void q4e_qsa_attention_combine_kernel(
        float *out, const float *part, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    const uint64_t qg = blockIdx.x;
    const float *base = part + qg * n_splits * Q4E_ATTN_PART_STRIDE;
    const uint32_t d = threadIdx.x;
    float gm = -INFINITY;
    for (uint32_t sp = 0; sp < n_splits; sp++) gm = fmaxf(gm, base[(size_t)sp * Q4E_ATTN_PART_STRIDE]);
    float den = 0.0f, num = 0.0f;
    for (uint32_t sp = 0; sp < n_splits; sp++) {
        const float *prow = base + (size_t)sp * Q4E_ATTN_PART_STRIDE;
        const float ms = prow[0];
        if (ms == -INFINITY) continue;
        const float wgt = __expf(ms - gm);
        den = fmaf(prow[1], wgt, den);
        num = fmaf(prow[8u + d], wgt, num);
    }
    out[qg * HD + d] = num / den;
}

/* Greedy choice per logits row, on the device.  Reading a 9-row verify batch
 * (8.9 MB) back to pageable host memory cost 12-83 ms per step at 26k
 * context; the step only needs the argmax of every row and the logits of the
 * one row it commits.  Ties go to the lower index, as the host sampler's. */
__global__ static void q4e_argmax_rows_kernel(int32_t *out, const float *logits, uint32_t n_vocab) {
    const float *row = logits + (uint64_t)blockIdx.x * n_vocab;
    float best = -INFINITY;
    int32_t best_i = -1;
    for (uint32_t i = threadIdx.x; i < n_vocab; i += blockDim.x) {
        const float v = row[i];
        if (v > best) { best = v; best_i = (int32_t)i; }
    }
    float ov;
    int32_t oi;
    q4e_block_argmax(best, best_i, &ov, &oi);
    if (threadIdx.x == 0u) out[blockIdx.x] = oi;
}

extern "C" int ds4_gpu_q4e_argmax_rows(ds4_gpu_tensor *out_idx, const ds4_gpu_tensor *logits,
                                       uint32_t n_vocab, uint32_t n_rows) {
    if (!out_idx || !logits || !n_vocab || !n_rows) return 0;
    if (out_idx->bytes < (uint64_t)n_rows * sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_rows * n_vocab * sizeof(float)) return 0;
    q4e_argmax_rows_kernel<<<n_rows, 1024, 0, cuda_decode_stream()>>>(
            (int32_t *)out_idx->ptr, (const float *)logits->ptr, n_vocab);
    return cuda_ok(cudaGetLastError(), "qwen4exp argmax rows");
}

/* Pinned host memory for the per-step logits landing buffer: a pageable
 * 1 MB device-to-host copy ran at ~5 GB/s here, a pinned one is 10x that. */
extern "C" void *ds4_gpu_q4e_host_alloc(uint64_t bytes) {
    void *p = NULL;
    if (cudaMallocHost(&p, (size_t)bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    return p;
}

extern "C" void ds4_gpu_q4e_host_free(void *p) {
    if (p) (void)cudaFreeHost(p);
}

/* ---------------------------------------------------------------------------
 * QSA indexer (Qwen Sparse Attention).  Per the Qwen3.8-Flash-Next report
 * §2.1.2: an MQA indexer with H=4 query heads of 128 dims and one key head;
 * keys are mean-pooled over blocks of r=4 tokens, RMS-normed and partially
 * rotated (64 dims, block start position); queries are normed and rotated at
 * their own position; a block's score is sum_h relu(<q_h, k_b>) over blocks
 * fully observed by the query; the top ceil(2048/4)=512 blocks plus the tail
 * tokens of the incomplete block are what the core attention attends to.
 * Below 512 complete blocks the selection is every visible token, so the
 * dense kernels above are used there (and stay bit-identical).
 * ------------------------------------------------------------------------ */
#define Q4E_IDX_DIM   128u
#define Q4E_IDX_R     4u
#define Q4E_IDX_TAIL  (Q4E_IDX_R - 1u)

/* Raw indexer keys into the f16 cache at the rows' paged positions. */
__global__ static void q4e_idx_store_k_kernel(__half *cache, const float *k, const int32_t *pos,
                                              const int32_t *pages, uint32_t n_tok) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    const uint64_t dst = (uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * Q4E_IDX_DIM;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        cache[dst + i] = __float2half(k[(uint64_t)t * Q4E_IDX_DIM + i]);
    }
}

extern "C" int ds4_gpu_q4e_idx_store_k(ds4_gpu_tensor *cache, const ds4_gpu_tensor *k,
                                       const ds4_gpu_tensor *pos, const ds4_gpu_tensor *pages,
                                       uint32_t n_tok) {
    if (!cache || !k || !pos || !pages || !n_tok) return 0;
    q4e_idx_store_k_kernel<<<n_tok, 128, 0, cuda_decode_stream()>>>(
            (__half *)cache->ptr, (const float *)k->ptr, (const int32_t *)pos->ptr,
            (const int32_t *)pages->ptr, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx store k");
}

/* One block per key block b in [b0, b1): mean of its r raw keys, RMSNorm with
 * the (Gemma +1 baked) k-norm weight, RoPE at position r*b, stored f16.
 *
 * A page holds a whole number of key blocks, so block b's r raw keys are
 * contiguous inside one page and its pooled row sits in that page's pooled
 * slice: both addresses come from the one translation of position r*b. */
static_assert(DS4_Q4E_PAGE_TOKENS % Q4E_IDX_R == 0u,
              "a pooled key block must not straddle a KV page");
__global__ static void q4e_idx_pool_kernel(__half *pooled, const __half *cache, const float *w,
                                           const int32_t *pos, const int32_t *pages, uint32_t n_tok,
                                           uint32_t n_rot, float rope_base, float eps) {
    /* Blocks completed by this forward: [pos0/r, (pos_last+1)/r).  Derived on
     * the device so a captured graph stays right as the context grows. */
    const uint32_t b0 = (uint32_t)pos[0] / Q4E_IDX_R;
    const uint32_t b1 = ((uint32_t)pos[n_tok - 1u] + 1u) / Q4E_IDX_R;
    const uint32_t b = b0 + blockIdx.x;
    if (b >= b1) return;
    const uint32_t krow = q4e_kv_row(pages, b * Q4E_IDX_R);
    const uint32_t brow = krow / Q4E_IDX_R;          /* pooled row of block b */
    __shared__ float s_k[Q4E_IDX_DIM];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < Q4E_IDX_R; j++) {
            acc += __half2float(cache[((uint64_t)krow + j) * Q4E_IDX_DIM + i]);
        }
        acc *= 1.0f / (float)Q4E_IDX_R;
        s_k[i] = acc;
        sum += acc * acc;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)Q4E_IDX_DIM + eps);
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) s_k[i] *= scale * w[i];
    __syncthreads();
    q4e_rope_neox(s_k, n_rot, rope_base, b * Q4E_IDX_R, threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        pooled[(uint64_t)brow * Q4E_IDX_DIM + i] = __float2half(s_k[i]);
    }
}

extern "C" int ds4_gpu_q4e_idx_pool(ds4_gpu_tensor *pooled, const ds4_gpu_tensor *cache,
                                    const void *model_map, uint64_t model_size, uint64_t norm_offset,
                                    const ds4_gpu_tensor *pos, const ds4_gpu_tensor *pages,
                                    uint32_t n_tok, uint32_t n_rot, float rope_base, float eps) {
    if (!pooled || !cache || !pos || !pages || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, norm_offset,
                                (uint64_t)Q4E_IDX_DIM * sizeof(float), pooled, "qwen4exp idx k norm");
    if (!w) return 0;
    /* At most n_tok/r + 1 blocks complete in one forward. */
    q4e_idx_pool_kernel<<<n_tok / Q4E_IDX_R + 1u, 128, 0, cuda_decode_stream()>>>(
            (__half *)pooled->ptr, (const __half *)cache->ptr, w, (const int32_t *)pos->ptr,
            (const int32_t *)pages->ptr, n_tok, n_rot, rope_base, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx pool");
}

/* Per (row, head): RMSNorm + RoPE of the indexer query, stored f16. */
__global__ static void q4e_idx_q_kernel(__half *qn, const float *q, const float *w,
                                        const int32_t *pos, uint32_t n_head,
                                        uint32_t n_rot, float rope_base, float eps) {
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    const uint64_t base = ((uint64_t)t * n_head + h) * Q4E_IDX_DIM;
    __shared__ float s_q[Q4E_IDX_DIM];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        const float v = q[base + i];
        s_q[i] = v;
        sum += v * v;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)Q4E_IDX_DIM + eps);
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) s_q[i] *= scale * w[i];
    __syncthreads();
    q4e_rope_neox(s_q, n_rot, rope_base, (uint32_t)pos[t], threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) qn[base + i] = __float2half(s_q[i]);
}

extern "C" int ds4_gpu_q4e_idx_q(ds4_gpu_tensor *qn, const ds4_gpu_tensor *q,
                                 const void *model_map, uint64_t model_size, uint64_t norm_offset,
                                 const ds4_gpu_tensor *pos, uint32_t n_head, uint32_t n_tok,
                                 uint32_t n_rot, float rope_base, float eps) {
    if (!qn || !q || !pos || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, norm_offset,
                                (uint64_t)Q4E_IDX_DIM * sizeof(float), qn, "qwen4exp idx q norm");
    if (!w) return 0;
    const dim3 grid(n_head, n_tok, 1);
    q4e_idx_q_kernel<<<grid, 128, 0, cuda_decode_stream()>>>(
            (__half *)qn->ptr, (const float *)q->ptr, w, (const int32_t *)pos->ptr,
            n_head, n_rot, rope_base, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx q");
}

/* Block scores: score[t][b] = sum_h relu(<q_th, k_b>) for blocks fully seen
 * by row t (r*b + r - 1 <= pos_t), else -inf.  A thread block handles 16
 * rows x 64 key blocks with both operands staged as f16. */
#define Q4E_IDX_QT 16u
#define Q4E_IDX_BT 64u
/* A key-block tile covers Q4E_IDX_BT * Q4E_IDX_R positions (today exactly one
 * page) and tiles start at multiples of that, so a tile's pooled rows sit in
 * one page, contiguous behind one page-table lookup. */
static_assert(DS4_Q4E_PAGE_TOKENS % (Q4E_IDX_BT * Q4E_IDX_R) == 0u,
              "a key-block tile must not straddle a KV page");
__global__ static void __launch_bounds__(256) q4e_idx_score_kernel(
        float *score, const __half *qn, const __half *pooled, const int32_t *pos,
        const int32_t *pages, uint32_t n_head, uint32_t n_tok,
        uint32_t n_blocks_stride, const int32_t *pos_last) {
    /* Complete blocks so far, from the last row's position (device-derived
     * so captured graphs track the growing context); the row stride of
     * `score` is the fixed n_blocks_stride. */
    const uint32_t n_blocks = ((uint32_t)pos_last[0] + 1u) / Q4E_IDX_R;
    const uint32_t t0 = blockIdx.y * Q4E_IDX_QT;
    const uint32_t b0 = blockIdx.x * Q4E_IDX_BT;
    if (b0 >= n_blocks) return;
    extern __shared__ __align__(16) __half q4e_idx_smem[];
    __half *s_q = q4e_idx_smem;                                   /* [QT][n_head][128] */
    __half *s_k = s_q + (size_t)Q4E_IDX_QT * n_head * Q4E_IDX_DIM; /* [BT][128] */
    const uint32_t n_q = (n_tok - t0 < Q4E_IDX_QT) ? (n_tok - t0) : Q4E_IDX_QT;
    const uint32_t n_b = (n_blocks - b0 < Q4E_IDX_BT) ? (n_blocks - b0) : Q4E_IDX_BT;
    for (uint32_t idx = threadIdx.x; idx < n_q * n_head * Q4E_IDX_DIM / 8u; idx += blockDim.x) {
        ((uint4 *)s_q)[idx] = ((const uint4 *)(qn + (uint64_t)t0 * n_head * Q4E_IDX_DIM))[idx];
    }
    const uint32_t brow = q4e_kv_row(pages, b0 * Q4E_IDX_R) / Q4E_IDX_R;
    for (uint32_t idx = threadIdx.x; idx < n_b * Q4E_IDX_DIM / 8u; idx += blockDim.x) {
        ((uint4 *)s_k)[idx] = ((const uint4 *)(pooled + (uint64_t)brow * Q4E_IDX_DIM))[idx];
    }
    __syncthreads();
    /* 1024 (row, block) pairs over 256 threads: thread owns 4 blocks of one row
     * group so its q reads are shared. */
    for (uint32_t pair = threadIdx.x; pair < Q4E_IDX_QT * Q4E_IDX_BT; pair += blockDim.x) {
        const uint32_t qi = pair / Q4E_IDX_BT, bi = pair % Q4E_IDX_BT;
        if (qi >= n_q) break;
        const uint32_t t = t0 + qi, b = b0 + bi;
        float total = -INFINITY;
        if (bi < n_b && b * Q4E_IDX_R + Q4E_IDX_R - 1u <= (uint32_t)pos[t]) {
            total = 0.0f;
            const __half2 *k2 = (const __half2 *)(s_k + (size_t)bi * Q4E_IDX_DIM);
            for (uint32_t h = 0; h < n_head; h++) {
                const __half2 *q2 = (const __half2 *)(s_q + ((size_t)qi * n_head + h) * Q4E_IDX_DIM);
                float d = 0.0f;
#pragma unroll 8
                for (uint32_t i = 0; i < Q4E_IDX_DIM / 2u; i++) {
                    const float2 a = __half22float2(q2[i]), c = __half22float2(k2[i]);
                    d = fmaf(a.x, c.x, d);
                    d = fmaf(a.y, c.y, d);
                }
                total += fmaxf(d, 0.0f);
            }
        }
        score[(uint64_t)t * n_blocks_stride + b] = total;
    }
}

extern "C" int ds4_gpu_q4e_idx_score(ds4_gpu_tensor *score, const ds4_gpu_tensor *qn,
                                     const ds4_gpu_tensor *pooled, const ds4_gpu_tensor *pos,
                                     const ds4_gpu_tensor *pos_last, const ds4_gpu_tensor *pages,
                                     uint32_t n_head, uint32_t n_tok, uint32_t max_blocks) {
    if (!score || !qn || !pooled || !pos || !pos_last || !pages || !n_tok || !max_blocks) return 0;
    const uint32_t n_blocks = max_blocks;
    const size_t smem = ((size_t)Q4E_IDX_QT * n_head + Q4E_IDX_BT) * Q4E_IDX_DIM * sizeof(__half);
    static bool opted = false;
    if (!opted) {
        (void)cudaFuncSetAttribute(q4e_idx_score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        (void)cudaGetLastError();
        opted = true;
    }
    const dim3 grid((n_blocks + Q4E_IDX_BT - 1u) / Q4E_IDX_BT, (n_tok + Q4E_IDX_QT - 1u) / Q4E_IDX_QT, 1);
    q4e_idx_score_kernel<<<grid, 256, smem, cuda_decode_stream()>>>(
            (float *)score->ptr, (const __half *)qn->ptr, (const __half *)pooled->ptr,
            (const int32_t *)pos->ptr, (const int32_t *)pages->ptr, n_head, n_tok, max_blocks,
            (const int32_t *)pos_last->ptr);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx score");
}

/* Top-k blocks per row by radix select on the float bit pattern (4 passes of
 * 8 bits), then compaction.  -inf (invisible) blocks are never selected;
 * ties at the threshold are broken by index order of arrival, which only
 * changes which equally scored block enters.  Output: sel[t][0..cnt) block
 * indices, unordered, cnt <= k. */
__global__ static void __launch_bounds__(1024) q4e_idx_topk_kernel(
        int32_t *sel, int32_t *cnt, const float *score, uint32_t n_blocks_stride,
        const int32_t *pos_last, uint32_t k) {
    const uint32_t t = blockIdx.x;
    const uint32_t n_blocks = ((uint32_t)pos_last[0] + 1u) / Q4E_IDX_R;
    const float *row = score + (uint64_t)t * n_blocks_stride;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_remaining, s_take;
    __shared__ int32_t s_out;
    __shared__ uint32_t s_gt;
    const uint32_t tid = threadIdx.x;

    /* Monotonic key: larger float -> larger key. */
    auto key_of = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };
    const uint32_t neg_inf_key = key_of(-INFINITY);

    if (tid == 0u) { s_prefix = 0u; s_remaining = k; }
    __syncthreads();
    uint32_t prefix = 0u, remaining = k;
    for (int pass = 3; pass >= 0; pass--) {
        const uint32_t shift = 8u * (uint32_t)pass;
        const uint32_t mask_hi = pass == 3 ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (uint32_t i = tid; i < 256u; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_blocks; i += blockDim.x) {
            const uint32_t u = key_of(row[i]);
            if (u == neg_inf_key) continue;
            if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t acc = 0u;
            int chosen = -1;
            for (int bin = 255; bin >= 0; bin--) {
                if (acc + hist[bin] >= remaining) { chosen = bin; break; }
                acc += hist[bin];
            }
            if (chosen < 0) {
                /* Fewer visible blocks than k: take everything visible. */
                s_remaining = remaining - acc; s_prefix = prefix; s_take = 0xFFFFFFFFu;
            } else {
                s_remaining = remaining - acc;      /* still needed inside the bin */
                s_prefix = prefix | ((uint32_t)chosen << shift);
                s_take = 0u;
            }
        }
        __syncthreads();
        if (s_take == 0xFFFFFFFFu) { prefix = 0u; remaining = 0u; break; }
        prefix = s_prefix;
        remaining = s_remaining;
        __syncthreads();
    }
    /* prefix is now the full threshold key T (when a bin was found in every
     * pass); everything > T is in, and `remaining` of the == T ties. */
    const bool take_all = (s_take == 0xFFFFFFFFu);
    if (tid == 0u) { s_out = 0; s_gt = remaining; }
    __syncthreads();
    int32_t *out = sel + (uint64_t)t * k;
    for (uint32_t i = tid; i < n_blocks; i += blockDim.x) {
        const uint32_t u = key_of(row[i]);
        if (u == neg_inf_key) continue;
        bool take = take_all || u > prefix;
        if (!take && u == prefix) {
            /* One of the tied slots, if any are left. */
            const uint32_t slot = atomicSub(&s_gt, 1u);
            take = (slot != 0u && slot <= k);
            if (!take) atomicAdd(&s_gt, 1u);
        }
        if (take) {
            const int32_t o = atomicAdd(&s_out, 1);
            if ((uint32_t)o < k) out[o] = (int32_t)i;
        }
    }
    __syncthreads();
    if (tid == 0u) cnt[t] = s_out < (int32_t)k ? s_out : (int32_t)k;
}

extern "C" int ds4_gpu_q4e_idx_topk(ds4_gpu_tensor *sel, ds4_gpu_tensor *cnt, const ds4_gpu_tensor *score,
                                    const ds4_gpu_tensor *pos_last,
                                    uint32_t n_tok, uint32_t max_blocks, uint32_t k) {
    if (!sel || !cnt || !score || !pos_last || !n_tok || !max_blocks || !k) return 0;
    q4e_idx_topk_kernel<<<n_tok, 1024, 0, cuda_decode_stream()>>>(
            (int32_t *)sel->ptr, (int32_t *)cnt->ptr, (const float *)score->ptr, max_blocks,
            (const int32_t *)pos_last->ptr, k);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx topk");
}

/* Selected blocks -> token indices, plus the tail of the incomplete block. */
__global__ static void q4e_idx_expand_kernel(int32_t *tokens, int32_t *n_sel, const int32_t *sel,
                                             const int32_t *cnt, const int32_t *pos,
                                             uint32_t k, uint32_t width) {
    const uint32_t t = blockIdx.x;
    const int32_t c = cnt[t];
    int32_t *out = tokens + (uint64_t)t * width;
    for (uint32_t i = threadIdx.x; i < (uint32_t)c * Q4E_IDX_R; i += blockDim.x) {
        out[i] = sel[(uint64_t)t * k + i / Q4E_IDX_R] * (int32_t)Q4E_IDX_R + (int32_t)(i % Q4E_IDX_R);
    }
    const uint32_t p = (uint32_t)pos[t];
    const uint32_t tail0 = ((p + 1u) / Q4E_IDX_R) * Q4E_IDX_R;   /* first token of the incomplete block */
    const uint32_t n_tail = p + 1u - tail0;                        /* 0..r-1 */
    if (threadIdx.x < n_tail) out[(uint32_t)c * Q4E_IDX_R + threadIdx.x] = (int32_t)(tail0 + threadIdx.x);
    if (threadIdx.x == 0u) n_sel[t] = c * (int32_t)Q4E_IDX_R + (int32_t)n_tail;
}

extern "C" int ds4_gpu_q4e_idx_expand(ds4_gpu_tensor *tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel,
                                      const ds4_gpu_tensor *cnt, const ds4_gpu_tensor *pos,
                                      uint32_t n_tok, uint32_t k, uint32_t width) {
    if (!tokens || !n_sel || !sel || !cnt || !pos || !n_tok) return 0;
    q4e_idx_expand_kernel<<<n_tok, 256, 0, cuda_decode_stream()>>>(
            (int32_t *)tokens->ptr, (int32_t *)n_sel->ptr, (const int32_t *)sel->ptr,
            (const int32_t *)cnt->ptr, (const int32_t *)pos->ptr, k, width);
    return cuda_ok(cudaGetLastError(), "qwen4exp idx expand");
}

/* Sparse attention over a per-row token list: the split kernel's structure
 * with gathered K/V rows.  A block owns one row t, one kv head and all of
 * that head's query heads (12 -> 6 warps x 2), and one of Q4E_ATTN_GSPLITS
 * slices of the row's token list; partials merge in the combine kernel. */
#define Q4E_ATTN_GSPLITS 8u
__global__ static void __launch_bounds__(256, 2)
q4e_qsa_attention_gather_kernel(
        float *part, const __half *k_cache, const __half *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pages, uint32_t width,
        uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint32_t KT = Q4E_ATTN_KT;
    constexpr uint32_t C8 = HD / 8u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);
    const int32_t *list = tokens + (uint64_t)t * width;
    const uint32_t n_list = (uint32_t)n_sel[t];
    const uint32_t n_tiles = (n_list + KT - 1u) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t l_begin = split * tiles_per_split * KT;
    uint32_t l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) l_end = n_list;

    extern __shared__ __align__(16) uint4 q4e_gather_smem[];
    uint4 *s_k8 = q4e_gather_smem;
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_ATTN_QG * C8);
    __shared__ int32_t s_idx[KT];

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (l_begin < l_end) {
        for (uint32_t idx = tid; idx < G * C8; idx += blockDim.x) {
            const uint32_t gh = idx / C8, c = idx % C8;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + hkv * G + gh) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)gh * C8 + c] = pk.u;
        }
        __syncthreads();
        for (uint32_t p0 = l_begin; p0 < l_end; p0 += KT) {
            const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
            /* The token list is arbitrary positions, so translation is per key
             * -- but it happens once here, in the same staging step that
             * already reads the list, and the tile then indexes cache rows. */
            if (tid < KT) {
                s_idx[tid] = tid < n_k ? (int32_t)q4e_kv_row(pages, (uint32_t)list[p0 + tid]) : 0;
            }
            __syncthreads();
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = *(const uint4 *)(k_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c));
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = *(const uint4 *)(v_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c));
                } else {
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= G) ? 0u : ((G - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }
#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const float d = dot[s];
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t kk = 0; kk < KT; kk++) {
                    const float wk = wrow[kk];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)kk * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]); acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]); acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]); acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]); acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t gh = warp * 2u + (uint32_t)s;
        if (gh >= G) break;
        const uint64_t qg = (uint64_t)t * n_head + hkv * G + gh;
        float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
        if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
        float4 *dst = (float4 *)(prow + 8u + 8u * lane);
        dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
        dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
    }
}

extern "C" int ds4_gpu_q4e_qsa_attention_sparse(
        ds4_gpu_tensor *out, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *tokens, const ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *pages,
        uint32_t width,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv, uint32_t n_tok) {
    if (!out || !part || !k_cache || !v_cache || !q || !tokens || !n_sel || !pages || !n_tok) return 0;
    if (head_dim != Q4E_ATTN_SPLIT_HD || n_head_kv == 0u || (n_head % n_head_kv) != 0u ||
        n_head / n_head_kv > Q4E_ATTN_QG) return 0;
    if (part->bytes < (uint64_t)n_tok * n_head * Q4E_ATTN_GSPLITS * Q4E_ATTN_PART_STRIDE * sizeof(float)) return 0;
    const size_t smem = (size_t)Q4E_ATTN_SPLIT_HD * Q4E_ATTN_KT * sizeof(__half) * 2u +
                        (size_t)Q4E_ATTN_QG * Q4E_ATTN_SPLIT_HD * sizeof(__half) +
                        (size_t)Q4E_ATTN_TILE_WARPS * Q4E_ATTN_KT * sizeof(float);
    static bool opted = false;
    if (!opted) {
        (void)cudaFuncSetAttribute(q4e_qsa_attention_gather_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        (void)cudaGetLastError();
        opted = true;
    }
    const dim3 grid(Q4E_ATTN_GSPLITS, n_head_kv, n_tok);
    q4e_qsa_attention_gather_kernel<<<grid, 256, smem, cuda_decode_stream()>>>(
            (float *)part->ptr, (const __half *)k_cache->ptr, (const __half *)v_cache->ptr,
            (const float *)q->ptr, (const int32_t *)tokens->ptr, (const int32_t *)n_sel->ptr,
            (const int32_t *)pages->ptr, width, n_head, n_head_kv, Q4E_ATTN_GSPLITS);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp qsa attention gather")) return 0;
    q4e_qsa_attention_combine_kernel<<<n_head * n_tok, Q4E_ATTN_SPLIT_HD, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)part->ptr, Q4E_ATTN_GSPLITS);
    return cuda_ok(cudaGetLastError(), "qwen4exp qsa attention sparse combine");
}

/* out = attn * sigmoid(gate), then the caller projects with attn_output. */
__global__ static void q4e_qsa_gate_kernel(float *x, const float *gate, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

extern "C" int ds4_gpu_q4e_qsa_q_norm_rope(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, const ds4_gpu_tensor *qkv,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const ds4_gpu_tensor *pos,
        uint32_t head_dim, uint32_t n_head, uint32_t n_rot, float rope_base,
        uint32_t n_tok, float eps) {
    if (!q_out || !gate_out || !qkv || !pos || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)head_dim * sizeof(float), q_out,
                                "qwen4exp q norm");
    if (!w) return 0;
    const dim3 grid(n_head, n_tok, 1);
    q4e_qsa_q_norm_rope_kernel<<<grid, 128, head_dim * sizeof(float),
                                 cuda_decode_stream()>>>(
            (float *)q_out->ptr, (float *)gate_out->ptr, (const float *)qkv->ptr, w,
            head_dim, n_head, n_rot, rope_base, (const int32_t *)pos->ptr, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp qsa q");
}

extern "C" int ds4_gpu_q4e_qsa_store_kv(
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const ds4_gpu_tensor *pos, const ds4_gpu_tensor *pages,
        uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot, float rope_base,
        uint32_t pool_slots, uint32_t n_tok, float eps) {
    if (!k_cache || !v_cache || !k || !v || !pos || !pages || !n_tok) return 0;
    const float *w = q4e_weight(model_map, model_size, weight_offset,
                                (uint64_t)head_dim * sizeof(float), k_cache,
                                "qwen4exp k norm");
    if (!w) return 0;
    const dim3 grid(n_head_kv, n_tok, 1);
    q4e_qsa_store_kv_kernel<<<grid, 128, head_dim * sizeof(float),
                              cuda_decode_stream()>>>(
            (__half *)k_cache->ptr, (__half *)v_cache->ptr,
            (const float *)k->ptr, (const float *)v->ptr, w,
            head_dim, n_head_kv, n_rot, rope_base,
            (const int32_t *)pos->ptr, (const int32_t *)pages->ptr, pool_slots, eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp qsa kv store");
}

extern "C" int ds4_gpu_q4e_qsa_attention(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *pos, const ds4_gpu_tensor *pages,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv, uint32_t n_tok) {
    if (!out || !k_cache || !v_cache || !q || !pos || !pages || !n_tok) return 0;
    if (head_dim > 32u * Q4E_ATTN_LANE_SLOTS) {
        fprintf(stderr, "ds4: qwen4exp attention head_dim %u exceeds %u\n",
                head_dim, 32u * Q4E_ATTN_LANE_SLOTS);
        return 0;
    }
    /* Prefill takes the tiled kernel: it needs head_dim to divide into whole
     * lane slots and enough queries to make staging a tile worth it.  Verified
     * against the 89-token llama.cpp dump -- worst full-length L2 error 1.2%,
     * identical to the per-query kernel.  DS4_QWEN4EXP_NO_TILED_ATTN=1 falls
     * back to that kernel for A/B. */
    static const int tiled_off = getenv("DS4_QWEN4EXP_NO_TILED_ATTN") != NULL;
    if (!tiled_off && n_tok >= Q4E_ATTN_QT && (head_dim % 32u) == 0u) {
        const size_t smem = 2u * (size_t)head_dim * Q4E_ATTN_KT * sizeof(__half) +
                            (size_t)Q4E_ATTN_QT * head_dim * sizeof(float) +
                            (size_t)Q4E_ATTN_TILE_WARPS * Q4E_ATTN_KT * sizeof(float);
        static bool opted_in = false;
        if (!opted_in) {
            if (cudaFuncSetAttribute(q4e_qsa_attention_tiled_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem) != cudaSuccess) {
                (void)cudaGetLastError();
            }
            opted_in = true;
        }
        const dim3 tgrid(n_head, (n_tok + Q4E_ATTN_QT - 1u) / Q4E_ATTN_QT, 1);
        q4e_qsa_attention_tiled_kernel<<<tgrid, 32u * Q4E_ATTN_TILE_WARPS, smem,
                                         cuda_decode_stream()>>>(
                (float *)out->ptr, (const __half *)k_cache->ptr,
                (const __half *)v_cache->ptr, (const float *)q->ptr,
                head_dim, n_head, n_head_kv, (const int32_t *)pos->ptr,
                (const int32_t *)pages->ptr, n_tok);
        if (cuda_ok(cudaGetLastError(), "qwen4exp qsa attention tiled")) return 1;
        /* Fall through to the per-query kernel if the launch was rejected. */
    }

    /* Decode and the verify batch: the split-key kernel, unless
     * DS4_QWEN4EXP_NO_SPLIT_ATTN=1 asks for the per-query kernel (A/B). */
    static const int split_off = getenv("DS4_QWEN4EXP_NO_SPLIT_ATTN") != NULL;
    if (!split_off && head_dim == Q4E_ATTN_SPLIT_HD && n_head_kv != 0u &&
        (n_head % n_head_kv) == 0u && n_tok < Q4E_ATTN_QT) {
        const size_t smem = (size_t)Q4E_ATTN_SPLIT_HD * Q4E_ATTN_KT * sizeof(__half) * 2u +
                            (size_t)Q4E_ATTN_QG * Q4E_ATTN_SPLIT_HD * sizeof(__half) +
                            (size_t)Q4E_ATTN_TILE_WARPS * Q4E_ATTN_KT * sizeof(float);
        static bool split_opted_in = false;
        static float *part = NULL;
        static uint64_t part_cap = 0;
        /* Sized for the largest batch this path takes, so the buffer never
         * moves once a graph has captured its address. */
        const uint64_t part_need = (uint64_t)n_head * (Q4E_ATTN_QT - 1u) * Q4E_ATTN_SPLITS *
                                   Q4E_ATTN_PART_STRIDE * sizeof(float);
        if (!split_opted_in) {
            if (cudaFuncSetAttribute(q4e_qsa_attention_split_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem) != cudaSuccess) {
                (void)cudaGetLastError();
            }
            split_opted_in = true;
        }
        if (part_cap < part_need) {
            if (part) (void)cudaFree(part);
            part = NULL;
            if (cudaMalloc(&part, (size_t)part_need) != cudaSuccess) {
                (void)cudaGetLastError();
                part = NULL;
                part_cap = 0;
            } else {
                part_cap = part_need;
            }
        }
        if (part) {
            const uint32_t G = n_head / n_head_kv;
            const uint32_t n_groups = (G * n_tok + Q4E_ATTN_QG - 1u) / Q4E_ATTN_QG;
            const dim3 sgrid(n_groups, Q4E_ATTN_SPLITS, n_head_kv);
            q4e_qsa_attention_split_kernel<<<sgrid, 256, smem, cuda_decode_stream()>>>(
                    (float *)out->ptr, part, (const __half *)k_cache->ptr,
                    (const __half *)v_cache->ptr, (const float *)q->ptr,
                    n_head, n_head_kv, (const int32_t *)pos->ptr,
                    (const int32_t *)pages->ptr, n_tok, Q4E_ATTN_SPLITS);
            if (!cuda_ok(cudaGetLastError(), "qwen4exp qsa attention split")) return 0;
            q4e_qsa_attention_combine_kernel<<<n_head * n_tok, Q4E_ATTN_SPLIT_HD, 0,
                                               cuda_decode_stream()>>>(
                    (float *)out->ptr, part, Q4E_ATTN_SPLITS);
            return cuda_ok(cudaGetLastError(), "qwen4exp qsa attention combine");
        }
    }

    const dim3 grid(n_head, n_tok, 1);
    q4e_qsa_attention_kernel<<<grid, 32u * Q4E_ATTN_WARPS,
                               Q4E_ATTN_WARPS * head_dim * sizeof(float),
                               cuda_decode_stream()>>>(
            (float *)out->ptr, (const __half *)k_cache->ptr,
            (const __half *)v_cache->ptr, (const float *)q->ptr,
            head_dim, n_head, n_head_kv, (const int32_t *)pos->ptr,
            (const int32_t *)pages->ptr, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen4exp qsa attention");
}

extern "C" int ds4_gpu_q4e_qsa_gate(ds4_gpu_tensor *x, const ds4_gpu_tensor *gate, uint64_t n) {
    if (!x || !gate || !n) return 0;
    q4e_qsa_gate_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, (const float *)gate->ptr, n);
    return cuda_ok(cudaGetLastError(), "qwen4exp qsa gate");
}

/* Below this chunk size the routed matmuls stay on the per-token mmvq path.
 * The batched entries win by a wide margin on real prefill chunks, but they
 * quantize activations differently, so a handful of tokens is not worth the
 * change in rounding -- and it keeps short prompts on exactly the kernels the
 * llama.cpp trace comparison validates. */
#define Q4E_MOE_BATCH_MIN_TOK q4e_moe_batch_min_tok()
/* Mirrors the public `ds4_q4e_moe_map` in ds4_gpu.h.  This translation unit
 * does not include that header (project convention, see the note near the top
 * of ds4_cuda.cu), so the handle is redeclared and size-checked here. */
struct ds4_q4e_moe_map {
    const void *ids_src1;
    const void *ids_dst;
    const void *expert_bounds;
    uint32_t n_rows;
    uint32_t n_expert;
};
static_assert(sizeof(ds4_q4e_moe_map) == 3u * sizeof(void *) + 2u * sizeof(uint32_t),
              "ds4_q4e_moe_map must match the ds4_gpu.h decl");

/* The routed map the host passes from the gate/up GEMM to the down GEMM is
 * mmq's map behind an opaque handle, so ds4.c never sees mmq's header.  Both
 * hold the same three device pointers; the conversion lives here only. */
static void q4e_moe_map_export(ds4_q4e_moe_map *h, const ds4_mmq_moe_map *m) {
    if (!h) return;
    memset(h, 0, sizeof(*h));
    if (!m || !m->ids_src1) return;
    h->ids_src1 = m->ids_src1;
    h->ids_dst = m->ids_dst;
    h->expert_bounds = m->expert_bounds;
    h->n_rows = (uint32_t)m->n_rows;
    h->n_expert = (uint32_t)m->n_expert;
}

static void q4e_moe_map_import(ds4_mmq_moe_map *m, const ds4_q4e_moe_map *h) {
    memset(m, 0, sizeof(*m));
    if (!h || !h->ids_src1) return;
    m->ids_src1 = (const int32_t *)h->ids_src1;
    m->ids_dst = (const int32_t *)h->ids_dst;
    m->expert_bounds = (const int32_t *)h->expert_bounds;
    m->n_rows = (int)h->n_rows;
    m->n_expert = (int)h->n_expert;
}

static uint32_t q4e_moe_batch_min_tok(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_QWEN4EXP_MOE_BATCH_MIN");
        cached = (env && env[0]) ? atoi(env) : 32;
        if (cached < 2) cached = 2;
    }
    return (uint32_t)cached;
}

/* One dequantized weight, for the register-resident batched kernel below.
 * Lane `lane` of block `blk` is weight `lane` of that block. */
template <int TYPE>
__device__ __forceinline__ static float q4e_down_weight(const uint8_t *blk, uint32_t lane) {
    if (TYPE == 7) {
        /* Q5_1: scale and minimum, four bits in qs plus a fifth in qh.  Lane l
         * wants weight l, which lives in nibble (l >> 4) of qs[l & 15] with its
         * top bit at position l of qh. */
        const float d = __half2float(*(const __half *)blk);
        const float mn = __half2float(*(const __half *)(blk + 2));
        uint32_t qh;
        memcpy(&qh, blk + 4, sizeof(qh));
        const uint8_t packed = blk[8u + (lane & 15u)];
        const uint32_t nib = (lane < 16u) ? (packed & 0x0Fu) : (packed >> 4);
        return fmaf(d, (float)(nib | (((qh >> lane) & 1u) << 4)), mn);
    }
    const float d = __half2float(*(const __half *)blk);
    return d * (float)((const int8_t *)(blk + 2))[lane];
}

/* Expert-grouped routed down projection, for prefill.
 *
 * The per-(token, slot) kernel below reads a whole expert row for every column
 * that selected it.  At one token that is exactly the traffic the model needs;
 * at a 512-token chunk it is up to 512 times too much, and it is why prefill
 * measured 120 tok/s.  Here the columns are sorted by expert first, so a warp
 * loads its output row's weights into registers once and then sweeps every
 * column routed to that expert.
 *
 * BLOCKS is a template parameter so the weight array stays in registers; a
 * runtime bound would spill it to local memory and undo the point. */
template <int TYPE, int BLOCKS>
__global__ static void q4e_moe_down_grouped_kernel(
        float *dst, const uint8_t *W, const float *x,
        const int32_t *sorted_cols, const int32_t *bounds, const int32_t *active,
        uint32_t n_expert, uint32_t out_dim, uint32_t in_dim, uint32_t expert_bytes) {
    /* blockIdx.y indexes the experts that received a column, not all 512: a
     * verify batch touches ~40 of them, and 300k empty blocks per layer were
     * half the kernel's time. */
    if ((int32_t)blockIdx.y >= active[n_expert]) return;
    const uint32_t expert = (uint32_t)active[blockIdx.y];
    const int32_t c0 = bounds[expert];
    const int32_t c1 = bounds[expert + 1u];
    if (c1 <= c0) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const uint32_t block_bytes = (TYPE == 7) ? 24u : 34u;
    const uint8_t *wrow = W + (uint64_t)expert * expert_bytes +
                          (uint64_t)row * BLOCKS * block_bytes;

    float wv[BLOCKS];
#pragma unroll
    for (int b = 0; b < BLOCKS; b++) {
        wv[b] = q4e_down_weight<TYPE>(wrow + (uint64_t)b * block_bytes, lane);
    }

    for (int32_t c = c0; c < c1; c++) {
        const uint32_t col = (uint32_t)sorted_cols[c];
        const float *xrow = x + (uint64_t)col * in_dim;
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < BLOCKS; b++) acc = fmaf(wv[b], xrow[b * 32u + lane], acc);
        acc = warp_sum_f32(acc);
        if (lane == 0u) dst[(uint64_t)col * out_dim + row] = acc;
    }
}

/* Counting sort of the (token, slot) columns by expert.  Three tiny kernels:
 * histogram, exclusive scan over the expert axis, scatter.  The scatter's
 * atomicAdd leaves the order inside one expert unspecified, which does not
 * affect the result: every column writes its own output row. */
__global__ static void q4e_moe_sort_count_kernel(
        int32_t *counts, const int32_t *ids, uint32_t n_rows, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    const int32_t e = ids[i];
    if (e >= 0 && (uint32_t)e < n_expert) atomicAdd(&counts[e], 1);
}

__global__ static void q4e_moe_sort_scan_kernel(
        int32_t *bounds, int32_t *active, const int32_t *counts, uint32_t n_expert) {
    /* One block, serial over the experts: 512 entries once per layer is far
     * below the point where a parallel scan would pay for itself.  Also lists
     * the experts with at least one column; the count goes in active[n_expert]. */
    if (threadIdx.x != 0u) return;
    int32_t total = 0;
    int32_t n_active = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        bounds[e] = total;
        total += counts[e];
        if (counts[e] > 0) active[n_active++] = (int32_t)e;
    }
    bounds[n_expert] = total;
    active[n_expert] = n_active;
}

__global__ static void q4e_moe_sort_scatter_kernel(
        int32_t *sorted, int32_t *cursor, const int32_t *bounds,
        const int32_t *ids, uint32_t n_rows, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    const int32_t e = ids[i];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    const int32_t slot = atomicAdd(&cursor[e], 1);
    sorted[bounds[e] + slot] = (int32_t)i;
}

/* Sort the routed columns by expert into a scratch buffer that lives for the
 * process.  Returns 0 when the scratch cannot be sized, which leaves the
 * caller on its per-column path.  The buffers are reused across layers: the
 * sort is redone every call because the routing changes per layer. */
static int q4e_moe_sort_columns(const int32_t *ids, uint32_t n_rows,
                                uint32_t n_expert,
                                const int32_t **sorted_out,
                                const int32_t **bounds_out,
                                const int32_t **active_out) {
    static int32_t *scratch = NULL;
    static uint32_t scratch_rows = 0;
    static uint32_t scratch_experts = 0;
    const uint32_t need = n_rows + 3u * n_expert + 2u;
    if (!scratch || n_rows > scratch_rows || n_expert > scratch_experts) {
        if (scratch) (void)cudaFree(scratch);
        scratch = NULL;
        if (cudaMalloc((void **)&scratch, (size_t)need * sizeof(int32_t)) != cudaSuccess) {
            (void)cudaGetLastError();
            scratch = NULL;
            return 0;
        }
        scratch_rows = n_rows;
        scratch_experts = n_expert;
    }
    int32_t *sorted = scratch;                 /* n_rows */
    int32_t *counts = sorted + n_rows;         /* n_expert, reused as cursor */
    int32_t *bounds = counts + n_expert;       /* n_expert + 1 */
    int32_t *active = bounds + n_expert + 1u;  /* n_expert + 1 */

    cudaStream_t stream = cuda_decode_stream();
    (void)cudaMemsetAsync(counts, 0, (size_t)n_expert * sizeof(int32_t), stream);
    const unsigned grid = (n_rows + 255u) / 256u;
    q4e_moe_sort_count_kernel<<<grid, 256, 0, stream>>>(counts, ids, n_rows, n_expert);
    q4e_moe_sort_scan_kernel<<<1, 32, 0, stream>>>(bounds, active, counts, n_expert);
    (void)cudaMemsetAsync(counts, 0, (size_t)n_expert * sizeof(int32_t), stream);
    q4e_moe_sort_scatter_kernel<<<grid, 256, 0, stream>>>(sorted, counts, bounds,
                                                          ids, n_rows, n_expert);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp moe column sort")) return 0;
    *sorted_out = sorted;
    *bounds_out = bounds;
    *active_out = active;
    return 1;
}

/* Routed down projection.
 *
 * The down experts have K = moe_ff (640): a whole number of 32-weight legacy
 * blocks but not of a 256-weight super-block, and each (token, slot) pair
 * needs its own input row rather than one row broadcast across a token's
 * experts.  The vendored mmvq path does not handle that shape correctly for
 * Q5_1, so this walks the blocks directly.
 *
 * One warp owns one output row of one (token, slot), and lane l takes weight
 * l of every block, so the warp consumes each block in one pass and its loads
 * coalesce.  Splitting the 20 blocks across threads instead leaves 20 of 128
 * threads doing all the work, which measured at 69 GB/s -- a third of what
 * this machine can read.  dst is column major over (token, slot), matching
 * what the swiglu stage produced. */
template <int TYPE>
__global__ static void q4e_moe_down_kernel(
        float *dst, const uint8_t *W, const float *x, const int32_t *ids,
        uint32_t out_dim, uint32_t in_dim, uint32_t n_used, uint32_t expert_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const uint32_t col = blockIdx.y;                 /* token * n_used + slot */
    if (row >= out_dim) return;
    const int32_t expert = ids[col];
    const uint32_t blocks = in_dim / 32u;
    const uint32_t block_bytes = (TYPE == 7) ? 24u : 34u;

    const uint8_t *wrow = W + (uint64_t)expert * expert_bytes +
                          (uint64_t)row * blocks * block_bytes;
    const float *xrow = x + (uint64_t)col * in_dim;

    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = wrow + (uint64_t)b * block_bytes;
        const float xv = xrow[b * 32u + lane];
        if (TYPE == 7) {
            /* Q5_1: scale and minimum, four bits in qs plus a fifth in qh.
             * Lane l wants weight l, which lives in nibble (l >> 4) of
             * qs[l & 15] with its top bit at position l of qh. */
            const float d = __half2float(*(const __half *)blk);
            const float mn = __half2float(*(const __half *)(blk + 2));
            uint32_t qh;
            memcpy(&qh, blk + 4, sizeof(qh));
            const uint8_t packed = blk[8u + (lane & 15u)];
            const uint32_t nib = (lane < 16u) ? (packed & 0x0Fu) : (packed >> 4);
            const uint32_t q = nib | (((qh >> lane) & 1u) << 4);
            acc = fmaf(fmaf(d, (float)q, mn), xv, acc);
        } else {
            const float d = __half2float(*(const __half *)blk);
            const int8_t q = ((const int8_t *)(blk + 2))[lane];
            acc = fmaf(d * (float)q, xv, acc);
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) dst[(uint64_t)col * out_dim + row] = acc;
}

extern "C" int ds4_gpu_q4e_moe_down(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type,
        uint32_t out_dim, uint32_t in_dim,
        uint32_t n_tok, uint32_t n_expert, uint32_t n_used,
        const ds4_q4e_moe_map *map) {
    if (!out || !x || !ids || !n_tok || (in_dim % 32u) != 0u) return 0;
    if (weight_type >= 68u && weight_type <= 70u) {   /* DS4_TENSOR_EXL3_K4..K6 */
        const uint32_t bits = weight_type - 68u + 4u;
        const uint64_t bytes = (uint64_t)n_expert *
            ((uint64_t)in_dim * out_dim * bits / 8u + 2u * ((uint64_t)in_dim + out_dim));
        return q4e_moe_exl3(out, x, true, ids, model_map, model_size, weight_offset, bytes, bits,
                            out_dim, in_dim, n_tok, n_expert, n_used, "qwen4exp exl3 moe down");
    }
    const uint32_t block_bytes = (weight_type == 7u) ? 24u : (weight_type == 8u) ? 34u : 0u;
    if (block_bytes == 0u) {
        fprintf(stderr, "ds4: qwen4exp routed down type %u is not supported\n", weight_type);
        return 0;
    }
    const uint64_t expert_bytes = (uint64_t)out_dim * (in_dim / 32u) * block_bytes;
    const uint64_t bytes = (uint64_t)n_expert * expert_bytes;
    const void *w = q4e_weight(model_map, model_size, weight_offset, bytes, out,
                               "qwen4exp moe down");
    if (!w) return 0;

    /* Four warps per block, one output row each. */
    const unsigned threads = 128u;
    const unsigned rows_per_block = threads / 32u;

    /* A real chunk goes to the vendored expert-grouped GEMM.  It tiles both
     * the output rows and the assigned columns; the grouped kernel below
     * keeps a weight row in registers and sweeps that expert's columns, which
     * reads the activation once per output row -- about 67 GiB a layer at a
     * 1024-token chunk against 0.6 GiB of weights, and it was a third of
     * prefill.  Down needs one input row per (token, slot) rather than one
     * per token, which the MoE contract expresses as n_tokens = tokens *
     * slots with a single expert each, making the flat ids array the per-row
     * expert map it already is. */
    const uint32_t n_rows = n_tok * n_used;
    if (n_tok >= Q4E_MOE_BATCH_MIN_TOK) {
        /* The gate/up GEMM already sorted these very assignments by expert;
         * flattened, its dst column *is* this GEMM's row index, so the map
         * carries over unchanged (ds4_mmq_moe_map_flatten). */
        ds4_mmq_moe_map pair, flat;
        q4e_moe_map_import(&pair, map);
        ds4_mmq_moe_map_flatten(&flat, &pair);
        const ds4_mmq_moe_map *shared = flat.ids_src1 ? &flat : NULL;
        /* Flattening to one expert per row makes n_rows the gathered total, but
         * the rows still come from n_tok tokens that each picked an expert at
         * most once, so n_tok -- not n_rows -- bounds any expert's bucket. */
        const int brc = (weight_type == 7u)
            ? ds4_mmq_q5_1_moe(w, (const float *)x->ptr, (const int32_t *)ids->ptr,
                               (float *)out->ptr, (int)out_dim, (int)in_dim,
                               (int)n_rows, (int)n_expert, 1, cuda_decode_stream(),
                               (int)n_tok, shared)
            : ds4_mmq_q8_0_moe(w, (const float *)x->ptr, (const int32_t *)ids->ptr,
                               (float *)out->ptr, (int)out_dim, (int)in_dim,
                               (int)n_rows, (int)n_expert, 1, cuda_decode_stream(),
                               (int)n_tok, shared);
        if (brc == 0) return cuda_ok(cudaGetLastError(), "qwen4exp moe down batched");
        fprintf(stderr, "ds4: qwen4exp batched routed down returned %d; "
                        "falling back to the grouped kernel\n", brc);
    }

    /* Below that, group the columns by expert so each expert slab is still
     * read once.  640 is the only K this model uses, and the
     * register-resident weight array needs it at compile time. */
    if (n_tok > 1u && in_dim == 640u) {
        const int32_t *sorted = NULL;
        const int32_t *bounds = NULL;
        const int32_t *active = NULL;
        if (q4e_moe_sort_columns((const int32_t *)ids->ptr, n_rows, n_expert,
                                 &sorted, &bounds, &active)) {
            /* At most one expert per column can be active, so n_rows bounds
             * the active list and the grid stays fixed per row count. */
            const uint32_t max_active = n_rows < n_expert ? n_rows : n_expert;
            const dim3 ggrid((out_dim + rows_per_block - 1u) / rows_per_block,
                             max_active, 1);
            if (weight_type == 7u) {
                q4e_moe_down_grouped_kernel<7, 20><<<ggrid, threads, 0,
                                                     cuda_decode_stream()>>>(
                        (float *)out->ptr, (const uint8_t *)w, (const float *)x->ptr,
                        sorted, bounds, active, n_expert, out_dim, in_dim,
                        (uint32_t)expert_bytes);
            } else {
                q4e_moe_down_grouped_kernel<8, 20><<<ggrid, threads, 0,
                                                     cuda_decode_stream()>>>(
                        (float *)out->ptr, (const uint8_t *)w, (const float *)x->ptr,
                        sorted, bounds, active, n_expert, out_dim, in_dim,
                        (uint32_t)expert_bytes);
            }
            return cuda_ok(cudaGetLastError(), "qwen4exp moe down grouped");
        }
    }

    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    n_tok * n_used, 1);
    if (weight_type == 7u) {
        q4e_moe_down_kernel<7><<<grid, threads, 0, cuda_decode_stream()>>>(
                (float *)out->ptr, (const uint8_t *)w, (const float *)x->ptr,
                (const int32_t *)ids->ptr, out_dim, in_dim, n_used,
                (uint32_t)expert_bytes);
    } else {
        q4e_moe_down_kernel<8><<<grid, threads, 0, cuda_decode_stream()>>>(
                (float *)out->ptr, (const uint8_t *)w, (const float *)x->ptr,
                (const int32_t *)ids->ptr, out_dim, in_dim, n_used,
                (uint32_t)expert_bytes);
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp moe down");
}

/* Routed gate and up together.
 *
 * Both read the same activation, so running them as two calls quantizes that
 * activation to Q8_1 twice and pays the mmvq setup twice -- which at one token
 * is most of the cost, not the arithmetic.  The paired entries share the
 * quantization, and at a single token the fused one also folds the SwiGLU in,
 * turning three launches per layer into one.
 *
 * Sets *fused_silu when the result is already silu(gate) * up. */
extern "C" int ds4_gpu_q4e_moe_gate_up(
        ds4_gpu_tensor *mid, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint32_t weight_type,
        uint32_t out_dim, uint32_t in_dim,
        uint32_t n_tok, uint32_t n_expert, uint32_t n_used,
        int *fused_silu, ds4_q4e_moe_map *map_out) {
    if (!mid || !gate || !up || !x || !ids || !n_tok) return 0;
    *fused_silu = 0;
    if (map_out) memset(map_out, 0, sizeof(*map_out));

    if (weight_type >= 68u && weight_type <= 70u) {   /* DS4_TENSOR_EXL3_K4..K6 */
        const uint32_t bits = weight_type - 68u + 4u;
        const uint64_t bytes = (uint64_t)n_expert *
            ((uint64_t)in_dim * out_dim * bits / 8u + 2u * ((uint64_t)in_dim + out_dim));
        return q4e_moe_exl3(gate, x, false, ids, model_map, model_size, gate_offset, bytes, bits,
                            out_dim, in_dim, n_tok, n_expert, n_used, "qwen4exp exl3 moe gate") &&
               q4e_moe_exl3(up, x, false, ids, model_map, model_size, up_offset, bytes, bits,
                            out_dim, in_dim, n_tok, n_expert, n_used, "qwen4exp exl3 moe up");
    }

    uint64_t block_elems = 0, block_bytes = 0;
    switch (weight_type) {
    case 12u: block_elems = 256u; block_bytes = 144u; break;   /* Q4_K */
    case 13u: block_elems = 256u; block_bytes = 176u; break;   /* Q5_K */
    case 8u:  block_elems = 32u;  block_bytes = 34u;  break;   /* Q8_0: the all-Q8 MTP sidecar */
    default:
        fprintf(stderr, "ds4: qwen4exp routed gate/up type %u is not supported\n", weight_type);
        return 0;
    }
    if (in_dim % block_elems != 0u) return 0;
    const uint64_t bytes = (uint64_t)n_expert * out_dim * (in_dim / block_elems) * block_bytes;
    const void *wg = q4e_weight(model_map, model_size, gate_offset, bytes, mid, "qwen4exp moe gate");
    const void *wu = q4e_weight(model_map, model_size, up_offset, bytes, mid, "qwen4exp moe up");
    if (!wg || !wu) return 0;

    /* Prefill takes the expert-grouped path.  The mmvq entries below read a
     * whole expert slab per (token, slot), which at one token is exactly the
     * traffic the model requires but at a 512-token chunk is 512 times too
     * much -- the single reason prefill measured 120 tok/s.  The batched
     * entries sort rows by expert first, so each expert's weights are read
     * once per chunk however many tokens picked it. */
    if (n_tok >= Q4E_MOE_BATCH_MIN_TOK) {
        /* One routed map for the whole block: this GEMM and the down GEMM
         * after it sort the same assignments by expert, and building it twice
         * cost a whole-ids scan per expert per layer.  A failed build leaves
         * the map empty, which sends each GEMM back to building its own. */
        ds4_mmq_moe_map map;
        (void)ds4_mmq_moe_map_build(&map, (const int32_t *)ids->ptr, (int)n_tok,
                                    (int)n_expert, (int)n_used, cuda_decode_stream());
        const ds4_mmq_moe_map *shared = map.ids_src1 ? &map : NULL;
        int brc;
        if (weight_type == 12u) {
            /* The router picks 10 distinct experts per token, so no expert can
             * hold more than n_tok rows -- a tenth of the gathered-row bound
             * mmq would otherwise size its grid from. */
            brc = ds4_mmq_q4_K_moe_pair(wg, wu, (const float *)x->ptr,
                                        (const int32_t *)ids->ptr,
                                        (float *)gate->ptr, (float *)up->ptr,
                                        (int)out_dim, (int)in_dim, (int)n_tok,
                                        (int)n_expert, (int)n_used,
                                        cuda_decode_stream(), /*max_rows_per_expert=*/(int)n_tok,
                                        shared);
        } else if (weight_type == 8u) {
            brc = ds4_mmq_q8_0_moe(wg, (const float *)x->ptr,
                                   (const int32_t *)ids->ptr, (float *)gate->ptr,
                                   (int)out_dim, (int)in_dim, (int)n_tok,
                                   (int)n_expert, (int)n_used, cuda_decode_stream(),
                                   /*max_rows_per_expert=*/(int)n_tok, shared);
            if (brc == 0) {
                brc = ds4_mmq_q8_0_moe(wu, (const float *)x->ptr,
                                       (const int32_t *)ids->ptr, (float *)up->ptr,
                                       (int)out_dim, (int)in_dim, (int)n_tok,
                                       (int)n_expert, (int)n_used, cuda_decode_stream(),
                                       /*max_rows_per_expert=*/(int)n_tok, shared);
            }
        } else {
            brc = ds4_mmq_q5_K_moe(wg, (const float *)x->ptr,
                                   (const int32_t *)ids->ptr, (float *)gate->ptr,
                                   (int)out_dim, (int)in_dim, (int)n_tok,
                                   (int)n_expert, (int)n_used, cuda_decode_stream(),
                                   /*max_rows_per_expert=*/(int)n_tok, shared);
            if (brc == 0) {
                brc = ds4_mmq_q5_K_moe(wu, (const float *)x->ptr,
                                       (const int32_t *)ids->ptr, (float *)up->ptr,
                                       (int)out_dim, (int)in_dim, (int)n_tok,
                                       (int)n_expert, (int)n_used, cuda_decode_stream(),
                                       /*max_rows_per_expert=*/(int)n_tok, shared);
            }
        }
        if (brc == 0) {
            q4e_moe_map_export(map_out, shared);
            return cuda_ok(cudaGetLastError(), "qwen4exp moe gate/up batched");
        }
        fprintf(stderr, "ds4: qwen4exp batched routed gate/up returned %d; "
                        "falling back to the per-token path\n", brc);
    }

    /* The fully fused pair entry (ds4_mmq_q4_K_moe_pair_vec, which also applies
     * the SwiGLU) produces wrong values for this model's shapes -- it is tuned
     * for the DeepSeek routed layout -- so only the shared-quantization variant
     * is used here. */
    int rc;
    if (weight_type == 12u) {
        rc = ds4_mmq_q4_K_moe_pair_raw_vec(wg, wu, (const float *)x->ptr,
                                           (const int32_t *)ids->ptr,
                                           (float *)gate->ptr, (float *)up->ptr,
                                           (int)out_dim, (int)in_dim, (int)n_tok,
                                           (int)n_expert, (int)n_used, cuda_decode_stream());
    } else if (weight_type == 8u) {
        rc = ds4_mmq_q8_0_moe_vec(wg, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                  (float *)gate->ptr, (int)out_dim, (int)in_dim,
                                  (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        if (rc == 0) {
            rc = ds4_mmq_q8_0_moe_vec(wu, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                      (float *)up->ptr, (int)out_dim, (int)in_dim,
                                      (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        }
    } else {
        /* Q5_K has no paired entry; one layer of the shipped mix uses it. */
        rc = ds4_mmq_q5_K_moe_vec(wg, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                  (float *)gate->ptr, (int)out_dim, (int)in_dim,
                                  (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        if (rc == 0) {
            rc = ds4_mmq_q5_K_moe_vec(wu, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                      (float *)up->ptr, (int)out_dim, (int)in_dim,
                                      (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        }
    }
    if (rc != 0) {
        fprintf(stderr, "ds4: qwen4exp routed gate/up matmul failed (%d)\n", rc);
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp moe gate/up");
}

/* Routed expert matmul, dispatched on the tensor's quant type.
 *
 * Gate and up have K = n_embd and go through whichever path their type
 * supports; down has K = moe_ff (640), a whole number of 32-weight blocks but
 * not of a 256-weight super-block, which is why it uses the vec path and why
 * that path's K guard follows the block size.
 *
 * out is column major over (token, slot): column t * n_used + s. */
extern "C" int ds4_gpu_q4e_moe_matmul(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type,
        uint32_t out_dim, uint32_t in_dim,
        uint32_t n_tok, uint32_t n_expert, uint32_t n_used) {
    if (!out || !x || !ids || !n_tok) return 0;

    uint64_t block_elems = 0, block_bytes = 0;
    const char *label = NULL;
    switch (weight_type) {
    case 8u:  block_elems = 32u;  block_bytes = 34u;  label = "qwen4exp moe Q8_0"; break;
    case 12u: block_elems = 256u; block_bytes = 144u; label = "qwen4exp moe Q4_K"; break;
    case 13u: block_elems = 256u; block_bytes = 176u; label = "qwen4exp moe Q5_K"; break;
    default:
        fprintf(stderr, "ds4: qwen4exp routed expert type %u is not supported\n", weight_type);
        return 0;
    }
    if (in_dim % block_elems != 0u) return 0;
    const uint64_t bytes = (uint64_t)n_expert * out_dim * (in_dim / block_elems) * block_bytes;
    const void *w = q4e_weight(model_map, model_size, weight_offset, bytes, out, label);
    if (!w) return 0;

    int rc = -1;
    switch (weight_type) {
    case 8u:
        rc = ds4_mmq_q8_0_moe_vec(w, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                  (float *)out->ptr, (int)out_dim, (int)in_dim,
                                  (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        break;
    case 12u:
        rc = ds4_mmq_q4_K_moe_vec(w, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                  (float *)out->ptr, (int)out_dim, (int)in_dim,
                                  (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        break;
    case 13u:
        rc = ds4_mmq_q5_K_moe_vec(w, (const float *)x->ptr, (const int32_t *)ids->ptr,
                                  (float *)out->ptr, (int)out_dim, (int)in_dim,
                                  (int)n_tok, (int)n_expert, (int)n_used, cuda_decode_stream());
        break;
    default: break;
    }
    if (rc != 0) {
        fprintf(stderr, "ds4: qwen4exp routed expert matmul failed (%d)\n", rc);
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp moe matmul");
}
