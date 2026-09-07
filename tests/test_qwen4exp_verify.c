/* Segmented batched forward gate.
 *
 * ds4_sessions_forward_segmented runs several per-sequence runs in one pass,
 * batching the dense/MoE weight reads while the recurrent kernels run per run.
 * It is the primitive under batched speculative verify and chunked prefill.
 *
 * This gate prefills several distinct prompts two ways and compares the
 * next-token prediction: once through the trusted single-sequence path
 * (ds4_session_sync), and once as one batched segmented forward over all of
 * them.  A batched pass crosses the MoE grouping threshold and re-rounds, so a
 * near-tied argmax may flip; those are reported, not failed (the same rule the
 * batched-decode gate uses).  A real bug flips a confident token.
 *
 * Env: DS4_QWEN4EXP_MODEL (skips without it), DS4_QWEN4EXP_MTP (optional). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

#define NSEQ 4
#define NEAR_TIE 0.25

/* Distinct prompts of varying length; small valid token ids. */
static const int32_t PROMPTS[NSEQ][20] = {
    { 12, 913, 44, 7, 220, 51, 990, 3, 1284, 66, 17, 802, 5, 91, -1 },
    { 100, 2, 55, 6021, 8, 314, 27, 9, 1180, 40, 3, 771, 15, 62, 208, 4, -1 },
    { 7, 1442, 33, 900, 12, 5, 61, 2231, 88, 14, -1 },
    { 501, 9, 26, 3, 1990, 71, 8, 143, 6, 22, 880, 4, 55, 190, 7, 61, 3, 12, -1 },
};

static int argmax_margin(const float *l, int n, double *margin) {
    int best = 0, second = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[best]) { second = best; best = i; }
    for (int i = 0; i < n; i++) if (i != best && l[i] > l[second]) second = i;
    *margin = (double)l[best] - (double)l[second];
    return best;
}

static uint32_t plen(int s) {
    uint32_t n = 0;
    while (n < 20 && PROMPTS[s][n] >= 0) n++;
    return n;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp segmented forward gate:\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL\n");
        return 0;
    }
    const int ctx = 512;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = (uint32_t)ctx;
    opt.n_threads = 8;
    opt.power_percent = 100;
    opt.exec_contexts = NSEQ;
    opt.kv_pool_tokens = (uint32_t)(ctx * NSEQ * 2);
    const char *mtp = getenv("DS4_QWEN4EXP_MTP");
    if (mtp && mtp[0]) opt.mtp_path = mtp;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        printf("  FAIL: cannot open the model\n");
        return 1;
    }
    const int n_vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)n_vocab * sizeof(float));
    char err[512] = {0};
    int fail = 0, near = 0;

    ds4_session *sess[NSEQ] = {0};
    int ref_tok[NSEQ];
    double ref_m[NSEQ];

    /* Reference: sync each prompt on its own, record the next-token argmax. */
    for (int i = 0; i < NSEQ; i++) {
        if (ds4_session_create(&sess[i], engine, (uint32_t)ctx) != 0) {
            printf("  FAIL: create %d\n", i); fail = 1; goto done;
        }
        ds4_tokens prompt = {0};
        for (uint32_t j = 0; j < plen(i); j++) ds4_tokens_push(&prompt, PROMPTS[i][j]);
        int rc = ds4_session_sync(sess[i], &prompt, err, sizeof(err));
        ds4_tokens_free(&prompt);
        if (rc != 0) { printf("  FAIL: sync %d: %s\n", i, err); fail = 1; goto done; }
        if (ds4_session_copy_logits(sess[i], logits, n_vocab) != n_vocab) {
            printf("  FAIL: ref logits %d\n", i); fail = 1; goto done;
        }
        ref_tok[i] = argmax_margin(logits, n_vocab, &ref_m[i]);
    }

    /* Batched: seed all prompts and run them as one segmented forward. */
    {
        ds4_session *ss[NSEQ];
        uint32_t pos0[NSEQ], ntok[NSEQ];
        uint32_t total = 0;
        for (int i = 0; i < NSEQ; i++) {
            if (ds4_session_seed_prefill_for_test(sess[i], PROMPTS[i], plen(i)) != 0) {
                printf("  FAIL: seed %d\n", i); fail = 1; goto done;
            }
            ss[i] = sess[i];
            pos0[i] = 0;
            ntok[i] = plen(i);
            total += ntok[i];
        }
        int32_t *argmax = malloc((size_t)total * sizeof(int32_t));
        if (ds4_sessions_forward_segmented(ss, pos0, ntok, NSEQ, argmax, err, sizeof(err)) != 0) {
            printf("  FAIL: segmented forward: %s\n", err);
            free(argmax); fail = 1; goto done;
        }
        free(argmax);
    }

    for (int i = 0; i < NSEQ; i++) {
        if (ds4_session_copy_logits(sess[i], logits, n_vocab) != n_vocab) {
            printf("  FAIL: batched logits %d\n", i); fail = 1; goto done;
        }
        double bat_m;
        int bat_tok = argmax_margin(logits, n_vocab, &bat_m);
        if (bat_tok == ref_tok[i]) {
            printf("  seq %d (len %u): next token %d matches (margin ref %.3f / batch %.3f)\n",
                   i, plen(i), ref_tok[i], ref_m[i], bat_m);
            continue;
        }
        double m = ref_m[i] < bat_m ? ref_m[i] : bat_m;
        if (m < NEAR_TIE) {
            near++;
            printf("  near-tie seq %d: sync %d vs batched %d, margin %.4f\n",
                   i, ref_tok[i], bat_tok, m);
        } else {
            fail = 1;
            printf("  FAIL: seq %d diverged: sync %d, batched %d (margin %.4f)\n",
                   i, ref_tok[i], bat_tok, m);
        }
    }

    printf("segmented forward gate: %s (%d near-tie%s)\n",
           fail ? "FAILED" : "ok", near, near == 1 ? "" : "s");

done:
    for (int i = 0; i < NSEQ; i++) if (sess[i]) ds4_session_free(sess[i]);
    free(logits);
    ds4_engine_close(engine);
    return fail;
}
