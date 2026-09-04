/* Decode-rate sweep against context length.
 *
 * The model is designed to decode at a rate that does not depend on the
 * context (36 of 48 layers are GDN with a fixed recurrent state, and the 12
 * full-attention layers read a fixed ~2048-token QSA budget), and exllamav3
 * measures flat.  ds4 does not, so this reports ms/step at several contexts
 * from one process, plain and speculative, with the first step -- the graph
 * capture after a prefill -- separated out instead of averaged in.
 *
 * Env: DS4_QWEN4EXP_MODEL (or DS4_QWEN4EXP_EXL3_MODEL) -- skips without one,
 *      DS4_QWEN4EXP_MTP        the MTP sidecar (adds the speculative rows),
 *      DS4_QWEN4EXP_MTP_DRAFT  drafts per step (engine default otherwise),
 *      DS4_BENCH_CTXS          comma list of prompt lengths,
 *      DS4_BENCH_STEPS         decode steps per point (default 40),
 *      DS4_QWEN4EXP_EXL3_IDS   token-id file (comma separated).
 *
 * The id pool is repeated to reach the longer contexts: the kernels do not
 * care about token values, but speculative acceptance does, so read ms/step
 * and treat tokens/step as an upper bound. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4.h"

#define IDS_DEFAULT "/home/otsimo/work/qwen-3.8-flash/ids26k.txt"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Median and mean of the steady-state steps (everything after the first). */
static void summarize(const double *ms, int n, double *median, double *mean) {
    *median = *mean = 0.0;
    if (n <= 0) return;
    double *s = malloc((size_t)n * sizeof(double));
    memcpy(s, ms, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), cmp_double);
    *median = s[n / 2];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += ms[i];
    *mean = sum / n;
    free(s);
}

static void sm_clock(void) {
    FILE *p = popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null", "r");
    if (!p) return;
    char line[64] = {0};
    if (fgets(line, sizeof(line), p)) {
        line[strcspn(line, "\n")] = 0;
        printf("  SM clock: %s\n", line);
    }
    pclose(p);
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    if (!model || !model[0]) model = getenv("DS4_QWEN4EXP_EXL3_MODEL");
    printf("qwen4exp decode-vs-context sweep\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL or DS4_QWEN4EXP_EXL3_MODEL\n");
        return 0;
    }

    /* Token pool. */
    const char *ids_path = getenv("DS4_QWEN4EXP_EXL3_IDS");
    if (!ids_path || !ids_path[0]) ids_path = IDS_DEFAULT;
    int *pool = NULL;
    int pool_len = 0, pool_cap = 0;
    FILE *f = fopen(ids_path, "r");
    if (!f) {
        printf("  FAIL: cannot read %s\n", ids_path);
        return 1;
    }
    for (;;) {
        int v = 0;
        if (fscanf(f, "%d", &v) != 1) break;
        if (pool_len == pool_cap) {
            pool_cap = pool_cap ? pool_cap * 2 : 4096;
            pool = realloc(pool, (size_t)pool_cap * sizeof(int));
        }
        pool[pool_len++] = v;
        int c = fgetc(f);
        if (c != ',' && c != '\n' && c != ' ') ungetc(c, f);
    }
    fclose(f);
    if (pool_len < 64) {
        printf("  FAIL: %s holds %d ids\n", ids_path, pool_len);
        return 1;
    }

    /* Context points. */
    int ctxs[16];
    int n_ctxs = 0;
    const char *ctxs_env = getenv("DS4_BENCH_CTXS");
    if (!ctxs_env || !ctxs_env[0]) ctxs_env = "2048,8192,16384,32768,65536";
    for (const char *p = ctxs_env; *p && n_ctxs < 16; ) {
        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) ctxs[n_ctxs++] = (int)v;
        p = (*end == ',') ? end + 1 : end;
    }
    const char *steps_env = getenv("DS4_BENCH_STEPS");
    const int steps = steps_env && steps_env[0] ? atoi(steps_env) : 40;
    int max_ctx = 0;
    for (int i = 0; i < n_ctxs; i++) if (ctxs[i] > max_ctx) max_ctx = ctxs[i];
    const uint32_t ctx_size = (uint32_t)(max_ctx + steps * 16 + 256);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = ctx_size;
    opt.n_threads = 8;
    opt.power_percent = 100;
    const char *mtp = getenv("DS4_QWEN4EXP_MTP");
    if (mtp && mtp[0]) opt.mtp_path = mtp;

    sm_clock();
    ds4_engine *engine = NULL;
    const double t_open = now();
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        printf("  FAIL: cannot open %s\n", model);
        return 1;
    }
    printf("  model open in %.1fs, ctx %u, pool %d ids\n", now() - t_open, ctx_size, pool_len);
    const int with_mtp = ds4_engine_has_mtp(engine);
    const int mtp_k = with_mtp ? ds4_engine_mtp_draft_tokens(engine) : 0;
    const int eos = ds4_token_eos(engine);

    printf("\n  %8s %10s %9s %9s %9s | %9s %9s %8s\n",
           "context", "prefill", "first ms", "plain ms", "tok/s",
           "spec ms", "tok/step", "tok/s");

    char err[256];
    double *ms = malloc((size_t)steps * sizeof(double));
    int *prompt_ids = malloc((size_t)(max_ctx + 1) * sizeof(int));

    for (int ci = 0; ci < n_ctxs; ci++) {
        const int n = ctxs[ci];
        for (int i = 0; i < n; i++) prompt_ids[i] = pool[i % pool_len];
        ds4_tokens prompt = { prompt_ids, n, n };

        /* Plain decode. */
        ds4_session *s = NULL;
        if (ds4_session_create(&s, engine, (int)ctx_size) != 0) {
            printf("  FAIL: session create at %d\n", n);
            break;
        }
        const double t0 = now();
        if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
            printf("  FAIL: sync at %d: %s\n", n, err);
            ds4_session_free(s);
            break;
        }
        const double prefill = now() - t0;
        int token = ds4_session_argmax(s);
        int done = 0;
        for (int i = 0; i < steps && token >= 0; i++) {
            const double a = now();
            if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
                printf("  FAIL: decode at %d step %d: %s\n", n, i, err);
                break;
            }
            token = ds4_session_argmax(s);
            ms[done++] = (now() - a) * 1e3;
        }
        ds4_session_free(s);
        double p_med = 0.0, p_mean = 0.0;
        summarize(ms + 1, done - 1, &p_med, &p_mean);
        const double first = done > 0 ? ms[0] : 0.0;

        /* Speculative decode. */
        double s_med = 0.0, s_mean = 0.0, per_step = 0.0;
        if (with_mtp) {
            ds4_session *sp = NULL;
            int sdone = 0;
            long committed = 0;
            if (ds4_session_create(&sp, engine, (int)ctx_size) == 0 &&
                ds4_session_sync(sp, &prompt, err, sizeof(err)) == 0) {
                for (int i = 0; i < steps; i++) {
                    int toks[24];
                    const int first_tok = ds4_session_argmax(sp);
                    const double a = now();
                    const int got = ds4_session_eval_speculative_argmax(
                            sp, first_tok, mtp_k + 1, eos, toks,
                            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
                    if (got <= 0) {
                        printf("  FAIL: spec at %d step %d: %s\n", n, i, err);
                        break;
                    }
                    ms[sdone++] = (now() - a) * 1e3;
                    committed += got;
                }
            }
            if (sp) ds4_session_free(sp);
            if (sdone > 1) {
                summarize(ms + 1, sdone - 1, &s_med, &s_mean);
                per_step = (double)committed / sdone;
            }
        }

        printf("  %8d %7.0f t/s %9.1f %9.2f %9.1f | %9.2f %9.2f %8.1f\n",
               n, prefill > 0 ? n / prefill : 0.0, first, p_med,
               p_med > 0 ? 1e3 / p_med : 0.0,
               s_med, per_step, s_med > 0 ? per_step * 1e3 / s_med : 0.0);
        fflush(stdout);
    }

    free(ms);
    free(prompt_ids);
    free(pool);
    ds4_engine_close(engine);
    return 0;
}
