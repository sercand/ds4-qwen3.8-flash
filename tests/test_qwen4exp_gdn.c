/* qwen4exp gated-delta-net prefill kernels against an exact reference.
 *
 * The two kernels behind ds4_gpu_q4e_gdn_recurrent -- the sequential
 * recurrence used by decode and the chunked-parallel one used by prefill --
 * are checked against the recurrence evaluated in double precision on the
 * host, at run lengths chosen to cover every partial-run shape the chunked
 * kernel can see.  That matters because the chunked kernel processes 64-token
 * runs and its intra-run work is laid out per run: a bug in the row
 * assignment showed up only for runs whose length fell in [33, 63], which no
 * end-to-end prompt in the gate happens to produce.  Both the per-token
 * outputs and the state carried out of the call are compared, the latter at
 * every run boundary.
 *
 * No model file and no NVML: synthetic inputs at the shipped shape, fixed
 * seed.  Skips cleanly when no CUDA device is visible.  Runs each kernel in
 * its own child process because the dispatch threshold is read from the
 * environment once per process (DS4_QWEN4EXP_GDN_CHUNK_MIN).
 *
 *   tests/test_qwen4exp_gdn            both kernels, the default shape list
 *   DS4_TEST_GDN_MAX=512 tests/...     stop the sweep early (quicker)
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* The shipped qwen4exp geometry (ds4.c: n_gdn_key_head 16, value 48, dim 128). */
#define HEAD_DIM 128u
#define N_HEAD_K 16u
#define N_HEAD_V 48u
#define STRIDE   (2u * N_HEAD_K * HEAD_DIM + N_HEAD_V * HEAD_DIM)
#define CHUNK    64u                     /* Q4E_GDN_CHUNK, for the shape list */
#define MAX_TOK  2100u
/* Heads carried through the double-precision reference: hk = h % n_head_k, so
 * these three cover the first, a middle and the last query/key head. */
static const uint32_t CHECK_HEADS[] = { 0u, 17u, 47u };
#define N_CHECK   (sizeof(CHECK_HEADS) / sizeof(CHECK_HEADS[0]))
/* Both kernels land at 2-4e-05% relative L2 against the double reference;
 * anything structurally wrong (a stale coefficient row, a missed token) is
 * orders of magnitude above this. */
#define TOL_PCT   5.0e-3

static uint64_t rng_state = 0x243f6a8885a308d3ull;
static double rnd(void) {           /* deterministic, no libc rand() */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ull) >> 11) / (double)(1ull << 53);
}
static double gauss(void) {
    double u = rnd(), v = rnd();
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

/* Reference: S[i][j]; per token S *= exp(g); d = beta (v - S^T k); S += k d^T;
 * out = S^T q / sqrt(D).  State snapshots are taken at the requested lengths. */
static void ref_head(uint32_t h, const float *qkv, const float *dec, const float *bet,
                     const float *st0, uint32_t n_tok, double *out,
                     const uint32_t *snap_at, uint32_t n_snap, double *snaps) {
    const uint32_t hk = h % N_HEAD_K, D = HEAD_DIM;
    double *S = (double *)malloc((size_t)D * D * sizeof(double));
    double *d = (double *)malloc(D * sizeof(double));
    for (uint32_t i = 0; i < D; i++)
        for (uint32_t j = 0; j < D; j++)
            S[(size_t)i * D + j] = st0[((size_t)h * D + j) * D + i];
    uint32_t s = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *q = qkv + (size_t)t * STRIDE + hk * D;
        const float *k = qkv + (size_t)t * STRIDE + N_HEAD_K * D + hk * D;
        const float *v = qkv + (size_t)t * STRIDE + 2u * N_HEAD_K * D + h * D;
        const double g = exp((double)dec[(size_t)t * N_HEAD_V + h]);
        const double b = (double)bet[(size_t)t * N_HEAD_V + h];
        for (size_t x = 0; x < (size_t)D * D; x++) S[x] *= g;
        for (uint32_t j = 0; j < D; j++) {
            double sk = 0.0;
            for (uint32_t i = 0; i < D; i++) sk += S[(size_t)i * D + j] * (double)k[i];
            d[j] = b * ((double)v[j] - sk);
        }
        for (uint32_t i = 0; i < D; i++)
            for (uint32_t j = 0; j < D; j++)
                S[(size_t)i * D + j] += (double)k[i] * d[j];
        for (uint32_t j = 0; j < D; j++) {
            double o = 0.0;
            for (uint32_t i = 0; i < D; i++) o += S[(size_t)i * D + j] * (double)q[i];
            out[(size_t)t * D + j] = o / sqrt((double)D);
        }
        while (s < n_snap && snap_at[s] == t + 1u) {
            /* snapshot in the kernel's layout: m[j][i] = S[i][j] */
            double *dst = snaps + (size_t)s * D * D;
            for (uint32_t j = 0; j < D; j++)
                for (uint32_t i = 0; i < D; i++) dst[(size_t)j * D + i] = S[(size_t)i * D + j];
            s++;
        }
    }
    free(S); free(d);
}

static double rel_l2(const double *ref, const float *got, size_t n, size_t stride_got) {
    double n2 = 0.0, dd = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double r = ref[i], g = (double)got[i * (stride_got ? stride_got : 1)];
        n2 += r * r; dd += (r - g) * (r - g);
    }
    return n2 > 0.0 ? 100.0 * sqrt(dd / n2) : 0.0;
}

static int run_mode(const char *mode, const char *chunk_min) {
    setenv("DS4_QWEN4EXP_GDN_CHUNK_MIN", chunk_min, 1);
    if (!ds4_gpu_init()) { fprintf(stderr, "FAIL: ds4_gpu_init\n"); return 1; }

    uint32_t max_tok = MAX_TOK;
    const char *env_max = getenv("DS4_TEST_GDN_MAX");
    if (env_max && env_max[0]) {
        const int v = atoi(env_max);
        if (v >= (int)CHUNK) max_tok = (uint32_t)v;
    }

    /* Lengths: every partial-run remainder from 33 to 63 on top of two full
     * runs (the window the row-assignment bug lived in), the boundaries around
     * it, and a few long shapes. */
    uint32_t lens[128]; uint32_t n_len = 0;
    const uint32_t fixed[] = { CHUNK, CHUNK + 1u, 2u * CHUNK - 1u, 2u * CHUNK,
                               2u * CHUNK + 1u, 3u * CHUNK + 32u, 999u, 1024u,
                               2048u, 2100u };
    for (uint32_t m = 33u; m <= 63u; m++) lens[n_len++] = 2u * CHUNK + m;
    for (uint32_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++) lens[n_len++] = fixed[i];
    uint32_t keep = 0;
    for (uint32_t i = 0; i < n_len; i++) if (lens[i] <= max_tok) lens[keep++] = lens[i];
    n_len = keep;

    const size_t qkv_n = (size_t)max_tok * STRIDE;
    const size_t gate_n = (size_t)max_tok * N_HEAD_V;
    const size_t st_n = (size_t)N_HEAD_V * HEAD_DIM * HEAD_DIM;
    const size_t out_n = (size_t)max_tok * N_HEAD_V * HEAD_DIM;
    float *qkv = (float *)malloc(qkv_n * sizeof(float));
    float *dec = (float *)malloc(gate_n * sizeof(float));
    float *bet = (float *)malloc(gate_n * sizeof(float));
    float *st0 = (float *)malloc(st_n * sizeof(float));
    float *got = (float *)malloc(out_n * sizeof(float));
    float *got_st = (float *)malloc(st_n * sizeof(float));
    if (!qkv || !dec || !bet || !st0 || !got || !got_st) { fprintf(stderr, "FAIL: host alloc\n"); return 1; }

    rng_state = 0x243f6a8885a308d3ull;
    for (size_t i = 0; i < qkv_n; i++) qkv[i] = (float)(gauss() * 0.1);
    /* q and k arrive L2-normalized per head from the caller's l2norm kernel. */
    for (uint32_t t = 0; t < max_tok; t++)
        for (uint32_t part = 0; part < 2u; part++)
            for (uint32_t h = 0; h < N_HEAD_K; h++) {
                float *v = qkv + (size_t)t * STRIDE + part * N_HEAD_K * HEAD_DIM + h * HEAD_DIM;
                double s2 = 0.0;
                for (uint32_t i = 0; i < HEAD_DIM; i++) s2 += (double)v[i] * v[i];
                const float inv = (float)(1.0 / sqrt(s2 + 1e-12));
                for (uint32_t i = 0; i < HEAD_DIM; i++) v[i] *= inv;
            }
    for (size_t i = 0; i < gate_n; i++) dec[i] = (float)(-0.05 - 0.6 * rnd());  /* log decay < 0 */
    for (size_t i = 0; i < gate_n; i++) bet[i] = (float)(0.2 + 0.7 * rnd());
    for (size_t i = 0; i < st_n; i++) st0[i] = (float)(gauss() * 0.05);

    /* One reference pass per checked head, snapshotting the state at every
     * tested length (which includes every run boundary in range). */
    double *ref_out = (double *)malloc((size_t)max_tok * HEAD_DIM * sizeof(double));
    double *snaps = (double *)malloc((size_t)n_len * HEAD_DIM * HEAD_DIM * sizeof(double));
    double **ref_by_head = (double **)malloc(N_CHECK * sizeof(double *));
    double **snap_by_head = (double **)malloc(N_CHECK * sizeof(double *));
    if (!ref_out || !snaps || !ref_by_head || !snap_by_head) { fprintf(stderr, "FAIL: ref alloc\n"); return 1; }
    uint32_t sorted[128];
    memcpy(sorted, lens, n_len * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_len; i++)
        for (uint32_t j = i + 1u; j < n_len; j++)
            if (sorted[j] < sorted[i]) { const uint32_t x = sorted[i]; sorted[i] = sorted[j]; sorted[j] = x; }
    for (uint32_t c = 0; c < N_CHECK; c++) {
        ref_by_head[c] = (double *)malloc((size_t)max_tok * HEAD_DIM * sizeof(double));
        snap_by_head[c] = (double *)malloc((size_t)n_len * HEAD_DIM * HEAD_DIM * sizeof(double));
        if (!ref_by_head[c] || !snap_by_head[c]) { fprintf(stderr, "FAIL: ref alloc\n"); return 1; }
        ref_head(CHECK_HEADS[c], qkv, dec, bet, st0, max_tok, ref_by_head[c],
                 sorted, n_len, snap_by_head[c]);
    }
    free(ref_out); free(snaps);

    ds4_gpu_tensor *t_qkv = ds4_gpu_tensor_alloc(qkv_n * sizeof(float));
    ds4_gpu_tensor *t_dec = ds4_gpu_tensor_alloc(gate_n * sizeof(float));
    ds4_gpu_tensor *t_bet = ds4_gpu_tensor_alloc(gate_n * sizeof(float));
    ds4_gpu_tensor *t_st = ds4_gpu_tensor_alloc(st_n * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    if (!t_qkv || !t_dec || !t_bet || !t_st || !t_out) { fprintf(stderr, "FAIL: device alloc\n"); return 1; }
    if (!ds4_gpu_tensor_write(t_qkv, 0, qkv, qkv_n * sizeof(float)) ||
        !ds4_gpu_tensor_write(t_dec, 0, dec, gate_n * sizeof(float)) ||
        !ds4_gpu_tensor_write(t_bet, 0, bet, gate_n * sizeof(float))) {
        fprintf(stderr, "FAIL: tensor_write\n"); return 1;
    }

    int bad = 0;
    double worst_out = 0.0, worst_st = 0.0;
    uint32_t worst_len = 0;
    for (uint32_t i = 0; i < n_len; i++) {
        const uint32_t L = lens[i];
        uint32_t si = 0;
        while (si < n_len && sorted[si] != L) si++;
        if (!ds4_gpu_tensor_write(t_st, 0, st0, st_n * sizeof(float))) { fprintf(stderr, "FAIL: state write\n"); return 1; }
        if (!ds4_gpu_q4e_gdn_recurrent(t_out, t_st, t_qkv, t_dec, t_bet,
                                       HEAD_DIM, N_HEAD_K, N_HEAD_V, STRIDE, L,
                                       NULL, 0u)) {
            fprintf(stderr, "FAIL: launch at n_tok=%u\n", L); return 1;
        }
        if (!ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(t_out, 0, got, (size_t)L * N_HEAD_V * HEAD_DIM * sizeof(float)) ||
            !ds4_gpu_tensor_read(t_st, 0, got_st, st_n * sizeof(float))) {
            fprintf(stderr, "FAIL: readback at n_tok=%u\n", L); return 1;
        }
        for (uint32_t c = 0; c < N_CHECK; c++) {
            const uint32_t h = CHECK_HEADS[c];
            /* outputs: ref is [t][j] dense, got is [(t*n_head_v + h)*D + j] */
            double n2 = 0.0, dd = 0.0;
            for (uint32_t t = 0; t < L; t++)
                for (uint32_t j = 0; j < HEAD_DIM; j++) {
                    const double r = ref_by_head[c][(size_t)t * HEAD_DIM + j];
                    const double g = (double)got[((size_t)t * N_HEAD_V + h) * HEAD_DIM + j];
                    n2 += r * r; dd += (r - g) * (r - g);
                }
            const double e_out = n2 > 0.0 ? 100.0 * sqrt(dd / n2) : 0.0;
            const double e_st = rel_l2(snap_by_head[c] + (size_t)si * HEAD_DIM * HEAD_DIM,
                                       got_st + (size_t)h * HEAD_DIM * HEAD_DIM,
                                       (size_t)HEAD_DIM * HEAD_DIM, 1);
            if (e_out > worst_out) { worst_out = e_out; worst_len = L; }
            if (e_st > worst_st) worst_st = e_st;
            if (e_out > TOL_PCT || e_st > TOL_PCT || !(e_out == e_out) || !(e_st == e_st)) {
                fprintf(stderr, "FAIL: %s n_tok=%u (run tail %u) head %u: out %.4g%% state %.4g%%\n",
                        mode, L, L % CHUNK, h, e_out, e_st);
                bad++;
            }
        }
    }
    fprintf(stderr, "  %s: %u lengths x %zu heads vs double reference, worst out %.3g%% "
                    "(n_tok=%u) worst state %.3g%% -- %s\n",
            mode, n_len, N_CHECK, worst_out, worst_len, worst_st, bad ? "FAIL" : "ok");

    ds4_gpu_tensor_free(t_qkv); ds4_gpu_tensor_free(t_dec); ds4_gpu_tensor_free(t_bet);
    ds4_gpu_tensor_free(t_st); ds4_gpu_tensor_free(t_out);
    for (uint32_t c = 0; c < N_CHECK; c++) { free(ref_by_head[c]); free(snap_by_head[c]); }
    free(ref_by_head); free(snap_by_head);
    free(qkv); free(dec); free(bet); free(st0); free(got); free(got_st);
    ds4_gpu_cleanup();
    return bad ? 1 : 0;
}

int main(void) {
    /* One child per kernel: the dispatch threshold is cached on first use, so
     * a single process can only exercise one of the two.  Forking before any
     * CUDA call keeps each child's context its own. */
    struct { const char *name, *min; } modes[] = {
        { "chunked  ", "64" },   /* every length below takes the chunked kernel */
        { "sequential", "0" },   /* threshold off: the recurrence for all */
    };
    int rc = 0;
    for (unsigned m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        const pid_t pid = fork();
        if (pid < 0) { fprintf(stderr, "FAIL: fork\n"); return 1; }
        if (pid == 0) _exit(run_mode(modes[m].name, modes[m].min));
        int st = 0;
        if (waitpid(pid, &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            fprintf(stderr, "FAIL: %s mode exited %d\n", modes[m].name,
                    WIFEXITED(st) ? WEXITSTATUS(st) : -1);
            rc = 1;
        }
    }
    fprintf(stderr, "test_qwen4exp_gdn %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
