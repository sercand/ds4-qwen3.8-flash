/* Batched decode must be the same decode.
 *
 * ds4_sessions_eval_batch advances several independent sessions through one
 * pass over the weights.  The only thing that may differ from advancing them
 * one at a time is arithmetic: a batched pass runs the row-parallel matmuls at
 * B rows instead of 1, which reassociates their sums the same way a wider
 * prefill chunk does.  Everything a sequence *is* -- its recurrent state, its
 * conv windows, its KV pages -- has to come out identical.
 *
 * So the gate runs the same prompts twice, greedily, and compares the tokens:
 * once with every session stepped on its own, once with all of them stepped
 * together.  Greedy decoding turns any real state mix-up into a divergent
 * token within a few steps, while leaving the harmless last-bit differences
 * invisible unless they land on a near-tie -- which is reported rather than
 * failed, with the two logits that were close, exactly as the exl3 gate does.
 *
 * Batch widths are swept so the partial-tile cases are covered rather than
 * just the one that happens to divide evenly.
 *
 * Set DS4_QWEN4EXP_MODEL to the first GGUF shard to run; skips without it.
 * DS4_QWEN4EXP_MTP names the MTP sidecar (optional; the batched path never
 * speculates, but the sequential reference then exercises the same session
 * setup a server would have).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ds4.h"

/* Distinct openings so no two sessions share a prefix: a shared prefix would
 * let a page-table mix-up go unnoticed, because the wrong pages would hold
 * the right keys. */
static const int PROMPTS[][6] = {
    { 760, 6511, 314, 9338, 369, -1 },      /* "The capital of France is" */
    { 760, 3575, 315, 279, 9260, -1 },
    { 3923, 374, 279, 1888, 1648, -1 },
    { 27, 91, 8043, 91, 29, 22691 },
    { 40, 1390, 311, 3371, 499, -1 },
    { 785, 5896, 315, 279, 1879, -1 },
    { 8420, 374, 264, 2875, 3364, -1 },
    { 32, 502, 1616, 311, 1744, -1 },
};
enum { MAX_SEQ = 8, STEPS = 24, NEAR_TIE_MAX = 8 };

static int near_ties;

/* Greedy pick plus how far ahead of the runner-up it was. */
static int argmax_margin(const float *l, int n, double *margin) {
    int best = 0, second = -1;
    for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
    for (int i = 0; i < n; i++) {
        if (i == best) continue;
        if (second < 0 || l[i] > l[second]) second = i;
    }
    *margin = second < 0 ? 0.0 : (double)l[best] - (double)l[second];
    return best;
}

static void push_prompt(ds4_tokens *t, int which) {
    for (int i = 0; i < 6; i++) {
        const int id = PROMPTS[which][i];
        if (id < 0) break;
        ds4_tokens_push(t, id);
    }
}

/* Generate STEPS greedy tokens for `n` sessions.  `batched` steps them all
 * through one call; otherwise each is stepped on its own.  Records the token
 * each session produced at each step, and the winning margin with it. */
static int run(ds4_engine *engine, int n, int ctx, int batched,
               int out[MAX_SEQ][STEPS], double margin[MAX_SEQ][STEPS]) {
    const int n_vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)n_vocab * sizeof(float));
    ds4_session *sess[MAX_SEQ] = {0};
    ds4_decode_item items[MAX_SEQ];
    char err[512] = {0};
    int rc = 1;

    for (int i = 0; i < n; i++) {
        if (ds4_session_create(&sess[i], engine, (uint32_t)ctx) != 0) {
            printf("  FAIL: cannot create session %d\n", i);
            goto done;
        }
        ds4_tokens prompt = {0};
        push_prompt(&prompt, i);
        if (ds4_session_sync(sess[i], &prompt, err, sizeof(err)) != 0) {
            printf("  FAIL: sync %d: %s\n", i, err);
            ds4_tokens_free(&prompt);
            goto done;
        }
        ds4_tokens_free(&prompt);
        if (ds4_session_copy_logits(sess[i], logits, n_vocab) != n_vocab) {
            printf("  FAIL: cannot read prompt logits %d\n", i);
            goto done;
        }
        items[i].session = sess[i];
        items[i].token = argmax_margin(logits, n_vocab, &margin[i][0]);
        out[i][0] = items[i].token;
    }

    for (int step = 1; step < STEPS; step++) {
        if (batched) {
            if (ds4_sessions_eval_batch(items, n, err, sizeof(err)) != 0) {
                printf("  FAIL: batched step %d: %s\n", step, err);
                goto done;
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (ds4_session_eval(sess[i], items[i].token, err, sizeof(err)) != 0) {
                    printf("  FAIL: step %d session %d: %s\n", step, i, err);
                    goto done;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (ds4_session_copy_logits(sess[i], logits, n_vocab) != n_vocab) {
                printf("  FAIL: cannot read logits, step %d session %d\n", step, i);
                goto done;
            }
            items[i].token = argmax_margin(logits, n_vocab, &margin[i][step]);
            out[i][step] = items[i].token;
        }
    }
    rc = 0;

done:
    for (int i = 0; i < n; i++) if (sess[i]) ds4_session_free(sess[i]);
    free(logits);
    return rc;
}

/* One batch width: the same prompts sequentially, then batched. */
static int compare_width(ds4_engine *engine, int n, int ctx) {
    static int seq_tok[MAX_SEQ][STEPS], bat_tok[MAX_SEQ][STEPS];
    static double seq_m[MAX_SEQ][STEPS], bat_m[MAX_SEQ][STEPS];

    if (run(engine, n, ctx, 0, seq_tok, seq_m) != 0) return 1;
    if (run(engine, n, ctx, 1, bat_tok, bat_m) != 0) return 1;

    int diverged = 0;
    for (int i = 0; i < n; i++) {
        for (int s = 0; s < STEPS; s++) {
            if (seq_tok[i][s] == bat_tok[i][s]) continue;
            /* A margin this thin means the two orderings disagreed about a
             * tie, not about the state.  Report it and keep going: the
             * sequences separate from here, so the rest of this row is no
             * longer comparable. */
            const double margin = seq_m[i][s] < bat_m[i][s] ? seq_m[i][s] : bat_m[i][s];
            if (margin < 0.25 && near_ties < NEAR_TIE_MAX) {
                near_ties++;
                printf("    near-tie B=%d seq %d step %d: %d vs %d, margin %.4f\n",
                       n, i, s, seq_tok[i][s], bat_tok[i][s], margin);
                break;
            }
            printf("  FAIL: B=%d seq %d diverged at step %d: sequential %d, "
                   "batched %d (margin %.4f)\n",
                   n, i, s, seq_tok[i][s], bat_tok[i][s], margin);
            diverged = 1;
            break;
        }
    }
    if (!diverged) printf("  B=%d: %d sequences x %d tokens match\n", n, n, STEPS);
    return diverged;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp batched decode tests:\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL to the first GGUF shard\n");
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
    opt.exec_contexts = MAX_SEQ;
    /* The derived pool assumes one context's worth of positions; here every
     * width runs MAX_SEQ sessions at once and each needs its own pages. */
    opt.kv_pool_tokens = (uint32_t)(ctx * MAX_SEQ * 2);
    const char *mtp = getenv("DS4_QWEN4EXP_MTP");
    if (mtp && mtp[0]) opt.mtp_path = mtp;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        printf("  FAIL: cannot open the model\n");
        return 1;
    }

    /* 2 and 3 cover the odd widths, 4 an even one, 8 the context cap. */
    static const int widths[] = { 2, 3, 4, 8 };
    int fail = 0;
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        if (compare_width(engine, widths[i], ctx) != 0) fail = 1;
    }

    ds4_engine_close(engine);
    if (near_ties) {
        printf("  %d near-tie(s) reported, not failed\n", near_ties);
    }
    printf("%s\n", fail ? "batched decode tests: FAILED" : "batched decode tests: ok");
    return fail;
}
