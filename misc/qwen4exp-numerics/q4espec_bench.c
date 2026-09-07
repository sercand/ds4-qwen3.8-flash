/* Decode-step timing at concurrency B for the qwen4exp paths the server can
 * take: speculative one-session-at-a-time, batched speculative
 * (DS4_QWEN4EXP_BATCH_SPEC=1), plain batched (MTP off), plain sequential.
 *
 *   q4espec_bench <mode> <B> [steps]     mode: specseq | specbat | plainbat | plainseq
 *
 * Prints ms/step, committed tokens per step and aggregate tok/s.  Env:
 * DS4_QWEN4EXP_MODEL, DS4_QWEN4EXP_MTP, DS4_QWEN4EXP_MTP_VOCAB (serve_ds4.sh's
 * --mtp-vocab), Q4E_BENCH_K (draft depth, default 4), Q4E_BENCH_CTX (2048). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4.h"

static const int PROMPTS[][6] = {
    { 760, 6511, 314, 9338, 369, -1 }, { 760, 3575, 315, 279, 9260, -1 },
    { 3923, 374, 279, 1888, 1648, -1 }, { 27, 91, 8043, 91, 29, 22691 },
    { 40, 1390, 311, 3371, 499, -1 },   { 785, 5896, 315, 279, 1879, -1 },
    { 8420, 374, 264, 2875, 3364, -1 }, { 32, 502, 1616, 311, 1744, -1 },
};
enum { MAX_SEQ = 8 };
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
static int amax(const float *l, int n) { int b = 0; for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i; return b; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s specseq|specbat|plainbat|plainseq B [steps]\n", argv[0]); return 2; }
    const char *mode = argv[1];
    const int B = atoi(argv[2]);
    const int steps = argc > 3 ? atoi(argv[3]) : 30, warm = 3;
    if (B < 1 || B > MAX_SEQ) { fprintf(stderr, "B must be 1..%d\n", MAX_SEQ); return 2; }
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    if (!model || !model[0]) { fprintf(stderr, "set DS4_QWEN4EXP_MODEL\n"); return 2; }
    const char *kenv = getenv("Q4E_BENCH_K"), *cenv = getenv("Q4E_BENCH_CTX");
    const int K = kenv && kenv[0] ? atoi(kenv) : 4, ctx = cenv && cenv[0] ? atoi(cenv) : 2048;
    const int spec = strncmp(mode, "spec", 4) == 0, batched = strstr(mode, "bat") != NULL;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model; opt.backend = DS4_BACKEND_CUDA; opt.context_size = (uint32_t)ctx;
    opt.n_threads = 8; opt.power_percent = 100; opt.exec_contexts = MAX_SEQ;
    opt.kv_pool_tokens = (uint32_t)(ctx * MAX_SEQ * 2);
    const char *mtp = getenv("DS4_QWEN4EXP_MTP"), *vocab = getenv("DS4_QWEN4EXP_MTP_VOCAB");
    if (spec && mtp && mtp[0]) { opt.mtp_path = mtp; opt.mtp_draft_tokens = K; }
    if (spec && vocab && vocab[0]) opt.mtp_vocab_path = vocab;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) { fprintf(stderr, "open failed\n"); return 1; }
    const int nv = ds4_engine_vocab_size(engine);
    float *lg = malloc((size_t)nv * sizeof(float));
    char err[512] = {0};
    ds4_session *ss[MAX_SEQ] = {0};
    ds4_decode_item it[MAX_SEQ];
    for (int i = 0; i < B; i++) {
        if (ds4_session_create(&ss[i], engine, (uint32_t)ctx) != 0) { fprintf(stderr, "session %d\n", i); return 1; }
        ds4_tokens p = {0};
        for (int j = 0; j < 6 && PROMPTS[i][j] >= 0; j++) ds4_tokens_push(&p, PROMPTS[i][j]);
        if (ds4_session_sync(ss[i], &p, err, sizeof(err)) != 0) { fprintf(stderr, "sync %d: %s\n", i, err); return 1; }
        ds4_tokens_free(&p);
        ds4_session_copy_logits(ss[i], lg, nv);
        it[i].session = ss[i]; it[i].token = amax(lg, nv);
    }
    double t_total = 0.0; long tok_total = 0; int timed = 0;
    for (int s = 0; s < warm + steps; s++) {
        const double t0 = now_s();
        long committed = 0;
        if (spec && batched) {
            int acc[MAX_SEQ][DS4_QWEN4EXP_SPEC_MAX_DRAFT + 1], com[MAX_SEQ];
            if (ds4_sessions_eval_speculative_batch(it, B, -1, acc, com, err, sizeof(err)) != 0) { fprintf(stderr, "specbat step %d: %s\n", s, err); return 1; }
            for (int i = 0; i < B; i++) committed += com[i];
        } else if (spec) {
            for (int i = 0; i < B; i++) {
                int acc[DS4_QWEN4EXP_SPEC_MAX_DRAFT + 1];
                const int n = ds4_session_eval_speculative_argmax(ss[i], it[i].token, DS4_QWEN4EXP_SPEC_MAX_DRAFT + 1, -1,
                                                                  acc, DS4_QWEN4EXP_SPEC_MAX_DRAFT + 1, err, sizeof(err));
                if (n <= 0) { fprintf(stderr, "specseq step %d seq %d: %s\n", s, i, err); return 1; }
                committed += n;
            }
        } else if (batched) {
            if (ds4_sessions_eval_batch(it, B, err, sizeof(err)) != 0) { fprintf(stderr, "plainbat step %d: %s\n", s, err); return 1; }
            committed = B;
        } else {
            for (int i = 0; i < B; i++)
                if (ds4_session_eval(ss[i], it[i].token, err, sizeof(err)) != 0) { fprintf(stderr, "plainseq step %d seq %d: %s\n", s, i, err); return 1; }
            committed = B;
        }
        for (int i = 0; i < B; i++) { ds4_session_copy_logits(ss[i], lg, nv); it[i].token = amax(lg, nv); }
        const double dt = now_s() - t0;
        if (s >= warm) { t_total += dt; tok_total += committed; timed++; }
    }
    printf("%-8s B=%d K=%d steps=%d: %.2f ms/step  %.2f tok/step  aggregate %.1f tok/s  per-stream %.1f tok/s\n",
           mode, B, spec ? K : 0, timed, 1e3 * t_total / timed, (double)tok_total / timed,
           (double)tok_total / t_total, (double)tok_total / t_total / B);
    for (int i = 0; i < B; i++) ds4_session_free(ss[i]);
    free(lg); ds4_engine_close(engine);
    return 0;
}
