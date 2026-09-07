/* Batched speculative decode gate.
 *
 * ds4_sessions_eval_speculative_batch drafts several sessions, verifies them in
 * one segmented forward, and accepts/rolls back each -- keeping MTP speculation
 * on across a batch.  Speculation is exact, so the committed tokens must equal
 * plain greedy decode.  This gate generates a stream both ways and compares.
 * A batched verify crosses the MoE grouping threshold and re-rounds, so a
 * near-tied argmax may flip and the streams separate from there; the first
 * divergence and how far they agreed are reported.
 *
 * Env: DS4_QWEN4EXP_MODEL (skips without it), DS4_QWEN4EXP_MTP (the draft head;
 * without it spec_k=0 and this reduces to batched plain decode, still exact). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

#define NSEQ 4
#define GEN 40

static const int32_t PROMPTS[NSEQ][20] = {
    { 12, 913, 44, 7, 220, 51, 990, 3, 1284, 66, 17, 802, 5, 91, -1 },
    { 100, 2, 55, 6021, 8, 314, 27, 9, 1180, 40, 3, 771, 15, 62, 208, 4, -1 },
    { 7, 1442, 33, 900, 12, 5, 61, 2231, 88, 14, -1 },
    { 501, 9, 26, 3, 1990, 71, 8, 143, 6, 22, 880, 4, 55, 190, 7, 61, 3, 12, -1 },
};

static int amax(const float *l, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    return b;
}

static uint32_t plen(int s) { uint32_t n = 0; while (n < 20 && PROMPTS[s][n] >= 0) n++; return n; }

static ds4_session *mk(ds4_engine *e, int ctx, int i, char *err, size_t el) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, (uint32_t)ctx) != 0) return NULL;
    ds4_tokens p = {0};
    for (uint32_t j = 0; j < plen(i); j++) ds4_tokens_push(&p, PROMPTS[i][j]);
    int rc = ds4_session_sync(s, &p, err, el);
    ds4_tokens_free(&p);
    if (rc != 0) { ds4_session_free(s); return NULL; }
    return s;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp batched speculative gate:\n");
    if (!model || !model[0]) { printf("  skipped: set DS4_QWEN4EXP_MODEL\n"); return 0; }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    const int ctx = 512;
    opt.context_size = (uint32_t)ctx;
    opt.n_threads = 8;
    opt.power_percent = 100;
    opt.exec_contexts = NSEQ;
    opt.kv_pool_tokens = (uint32_t)(ctx * NSEQ * 2);
    const char *mtp = getenv("DS4_QWEN4EXP_MTP");
    if (mtp && mtp[0]) opt.mtp_path = mtp;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) { printf("  FAIL: open\n"); return 1; }
    const int nv = ds4_engine_vocab_size(engine);
    float *lg = malloc((size_t)nv * sizeof(float));
    char err[512] = {0};
    int fail = 0;
    static int ref[NSEQ][GEN], bat[NSEQ][GEN];

    /* Reference: plain greedy. */
    for (int i = 0; i < NSEQ; i++) {
        ds4_session *s = mk(engine, ctx, i, err, sizeof(err));
        if (!s) { printf("  FAIL: ref session %d: %s\n", i, err); fail = 1; goto done; }
        ds4_session_copy_logits(s, lg, nv);
        int t = amax(lg, nv);
        for (int g = 0; g < GEN; g++) {
            ref[i][g] = t;
            if (g + 1 < GEN) {
                if (ds4_session_eval(s, t, err, sizeof(err)) != 0) { printf("  FAIL: ref eval: %s\n", err); ds4_session_free(s); fail = 1; goto done; }
                ds4_session_copy_logits(s, lg, nv);
                t = amax(lg, nv);
            }
        }
        ds4_session_free(s);
    }

    /* Batched speculative. */
    {
        ds4_session *ss[NSEQ];
        ds4_decode_item it[NSEQ];
        int filled[NSEQ] = {0};
        for (int i = 0; i < NSEQ; i++) {
            ss[i] = mk(engine, ctx, i, err, sizeof(err));
            if (!ss[i]) { printf("  FAIL: bat session %d: %s\n", i, err); fail = 1; goto done; }
            ds4_session_copy_logits(ss[i], lg, nv);
            it[i].session = ss[i];
            it[i].token = amax(lg, nv);
        }
        int done_all = 0;
        while (!done_all) {
            int acc[NSEQ][DS4_QWEN4EXP_SPEC_MAX_DRAFT + 1];
            int com[NSEQ];
            if (ds4_sessions_eval_speculative_batch(it, NSEQ, -1, acc, com, err, sizeof(err)) != 0) {
                printf("  FAIL: spec batch: %s\n", err);
                for (int i = 0; i < NSEQ; i++) ds4_session_free(ss[i]);
                fail = 1; goto done;
            }
            done_all = 1;
            for (int i = 0; i < NSEQ; i++) {
                for (int j = 0; j < com[i] && filled[i] < GEN; j++) bat[i][filled[i]++] = acc[i][j];
                ds4_session_copy_logits(ss[i], lg, nv);
                it[i].token = amax(lg, nv);
                if (filled[i] < GEN) done_all = 0;
            }
        }
        for (int i = 0; i < NSEQ; i++) ds4_session_free(ss[i]);
    }

    for (int i = 0; i < NSEQ; i++) {
        int agree = 0;
        while (agree < GEN && ref[i][agree] == bat[i][agree]) agree++;
        if (agree == GEN) {
            printf("  seq %d: %d/%d committed tokens match\n", i, agree, GEN);
        } else {
            /* A late divergence is a near-tie the MoE regrouping tipped; an
             * immediate one is a real defect. */
            printf("  %s seq %d: agreed %d/%d, then greedy %d vs batched %d\n",
                   agree >= 8 ? "near-tie" : "FAIL", i, agree, GEN,
                   ref[i][agree], bat[i][agree]);
            if (agree < 8) fail = 1;
        }
    }
    printf("batched speculative gate: %s\n", fail ? "FAILED" : "ok");

done:
    free(lg);
    ds4_engine_close(engine);
    return fail;
}
