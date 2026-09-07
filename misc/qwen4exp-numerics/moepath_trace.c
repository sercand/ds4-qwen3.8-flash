/* Per-tensor trace of the same token computed two ways: as the last row of an
 * N-row prefill (dump dir B) and as a single-row decode step after a fully
 * sequential prefix (dump dir A).  Run with DS4_QWEN4EXP_TRACE=1 and
 * DS4_QWEN4EXP_ROUTE_DUMP=<base>/route in the environment; the trace dump dir
 * is set here per step.  Env: DS4_QWEN4EXP_MODEL, Q4E_TRACE_N (16), Q4E_TRACE_DIR. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "ds4.h"

static const char *TEXT =
    "The lighthouse keeper had not spoken to another person in eleven days when the "
    "bottle washed up against the rocks below the north stair. It was an ordinary "
    "green glass bottle, the kind that once held cheap wine, and its cork had been "
    "sealed over with candle wax. Inside, rolled tight, was a page torn from a ledger. "
    "He carried it up the hundred and twelve steps to the lamp room before he opened "
    "it, because the wind at the water's edge would have taken the paper the moment "
    "the wax gave way. The handwriting was small and even, the kind taught in schools "
    "before typewriters, and it began with a date from forty years earlier.";

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    const char *base = getenv("Q4E_TRACE_DIR");
    if (!model || !base) { fprintf(stderr, "set DS4_QWEN4EXP_MODEL and Q4E_TRACE_DIR\n"); return 2; }
    const char *nenv = getenv("Q4E_TRACE_N");
    const uint32_t N = nenv && nenv[0] ? (uint32_t)atoi(nenv) : 16u;
    char dirB[512];
    snprintf(dirB, sizeof(dirB), "%s/B", base);
    mkdir(base, 0755); mkdir(dirB, 0755);

    ds4_engine_options opt; memset(&opt, 0, sizeof(opt));
    opt.model_path = model; opt.backend = DS4_BACKEND_CUDA; opt.context_size = 1024u;
    opt.n_threads = 8; opt.power_percent = 100; opt.exec_contexts = 2; opt.kv_pool_tokens = 4096u;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "open failed\n"); return 1; }
    ds4_tokens t = {0};
    ds4_tokenize_text(e, TEXT, &t);
    if ((uint32_t)t.len < N) { fprintf(stderr, "only %d tokens\n", (int)t.len); return 1; }
    char err[512] = {0};

    /* A: sequential, every step dumped into its own directory A<pos>. */
    {
        ds4_session *s = NULL;
        if (ds4_session_create(&s, e, 1024u) != 0) return 1;
        char d[600];
        for (uint32_t i = 0; i < N; i++) {
            snprintf(d, sizeof(d), "%s/A%u", base, i); mkdir(d, 0755);
            setenv("DS4_QWEN4EXP_TRACE_DUMP", d, 1);
            int rc;
            if (i == 0) {
                ds4_tokens p = {0}; ds4_tokens_push(&p, t.v[0]);
                rc = ds4_session_sync(s, &p, err, sizeof(err));
                ds4_tokens_free(&p);
            } else rc = ds4_session_eval(s, t.v[i], err, sizeof(err));
            unsetenv("DS4_QWEN4EXP_TRACE_DUMP");
            if (rc != 0) { fprintf(stderr, "step %u: %s\n", i, err); return 1; }
        }
        fprintf(stderr, "@@ A-done\n");
        ds4_session_free(s);
    }
    /* B: one N-row prefill. */
    {
        ds4_session *s = NULL;
        if (ds4_session_create(&s, e, 1024u) != 0) return 1;
        if (ds4_session_seed_prefill_for_test(s, t.v, N) != 0) return 1;
        uint32_t pos0 = 0, nt = N;
        int32_t *amx = malloc((size_t)N * sizeof(int32_t));
        fprintf(stderr, "@@ B-step-begin\n");
        setenv("DS4_QWEN4EXP_TRACE_DUMP", dirB, 1);
        const int rc = ds4_sessions_forward_segmented(&s, &pos0, &nt, 1u, amx, err, sizeof(err));
        unsetenv("DS4_QWEN4EXP_TRACE_DUMP");
        fprintf(stderr, "@@ B-step-end\n");
        if (rc != 0) { fprintf(stderr, "forward: %s\n", err); return 1; }
        free(amx);
        ds4_session_free(s);
    }
    printf("done N=%u\n", N);
    ds4_tokens_free(&t);
    ds4_engine_close(e);
    return 0;
}
