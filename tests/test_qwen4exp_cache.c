/* qwen4exp prefix cache: a conversation whose client re-renders the history
 * must still record every turn.
 *
 * Turn 1 prefills prompt A and generates G.  Turn 2's client replays the
 * history without the reasoning, so its prompt B shares A plus the first few
 * generated tokens and then diverges -- the shape the server logs as
 * "responses replay ... missing reasoning state".  Turn 3 extends B with what
 * turn 2 generated.  Before the commit descended to the true divergence, turn
 * 2's spans were refused (a sibling already carried their first token) and
 * turn 3 matched only turn 2's prompt-time match, re-prefilling the whole
 * turn; the server printed "prefix cache could not record" every turn.
 *
 * Env: DS4_QWEN4EXP_EXL3_MODEL (or DS4_QWEN4EXP_MODEL) -- skips without one;
 *      DS4_QWEN4EXP_EXL3_IDS (token pool, default ids26k.txt). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

#define IDS_DEFAULT "/home/otsimo/work/qwen-3.8-flash/ids26k.txt"

static int load_ids(const char *path, int **out) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int cap = 4096, n = 0, v;
    int *ids = malloc((size_t)cap * sizeof(int));
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; ids = realloc(ids, (size_t)cap * sizeof(int)); }
        ids[n++] = v;
        int c = fgetc(f);
        if (c != ',' && c != '\n' && c != ' ') ungetc(c, f);
    }
    fclose(f);
    *out = ids;
    return n;
}

static int generate(ds4_session *s, int *out, int n, char *err, size_t errlen) {
    for (int i = 0; i < n; i++) {
        out[i] = ds4_session_argmax(s);
        if (out[i] < 0) return 0;
        if (ds4_session_eval(s, out[i], err, errlen) != 0) return 0;
    }
    return 1;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_EXL3_MODEL");
    if (!model || !model[0]) model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp prefix cache tests:\n");
    if (!model || !model[0]) { printf("  skipped: set DS4_QWEN4EXP_EXL3_MODEL\n"); return 0; }
    const char *ids_path = getenv("DS4_QWEN4EXP_EXL3_IDS");
    if (!ids_path || !ids_path[0]) ids_path = IDS_DEFAULT;
    int *pool = NULL;
    const int n_pool = load_ids(ids_path, &pool);
    if (n_pool < 12000) { printf("  FAIL: %s holds %d ids\n", ids_path, n_pool); return 1; }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = 16384;
    opt.n_threads = 8;
    opt.power_percent = 100;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) { printf("  FAIL: cannot open the model\n"); return 1; }

    char err[512] = {0};
    int fail = 0;
    enum { A_LEN = 3000, G_LEN = 40, SHARED = 6, TAIL = 300, G2_LEN = 20 };
    int g1[G_LEN], g2[G_LEN];
    ds4_tokens a = {0}, b = {0}, c = {0};
    for (int i = 0; i < A_LEN; i++) ds4_tokens_push(&a, pool[i]);

    /* Turn 1. */
    ds4_session *s1 = NULL;
    if (ds4_session_create(&s1, engine, 16384) != 0 || ds4_session_sync(s1, &a, err, sizeof(err)) != 0 ||
        !generate(s1, g1, G_LEN, err, sizeof(err))) {
        printf("  FAIL: turn 1: %s\n", err); return 1;
    }
    ds4_session_cache_commit(s1);
    ds4_session_free(s1);

    /* Turn 2: the re-rendered history shares A and the first SHARED generated
     * tokens, then carries different text. */
    for (int i = 0; i < A_LEN; i++) ds4_tokens_push(&b, pool[i]);
    for (int i = 0; i < SHARED; i++) ds4_tokens_push(&b, g1[i]);
    for (int i = 0; i < TAIL; i++) ds4_tokens_push(&b, pool[10000 + i]);
    ds4_session *s2 = NULL;
    if (ds4_session_create(&s2, engine, 16384) != 0) { printf("  FAIL: session 2\n"); return 1; }
    ds4_session_reuse r2;
    ds4_session_reuse_report(s2, &b, &r2);
    printf("  turn 2 plan: source=%s reused=%d matched=%d prefilled=%d\n",
           ds4_reuse_source_name(r2.source), r2.reused_tokens, r2.matched_tokens, r2.prefilled_tokens);
    if (r2.matched_tokens < A_LEN) { printf("  FAIL: turn 2 should match at least turn 1's prompt\n"); fail = 1; }
    if (ds4_session_sync(s2, &b, err, sizeof(err)) != 0 || !generate(s2, g2, G2_LEN, err, sizeof(err))) {
        printf("  FAIL: turn 2: %s\n", err); return 1;
    }
    ds4_session_cache_commit(s2);
    ds4_session_free(s2);

    /* Turn 3 extends turn 2 exactly: the tree must hold all of it. */
    for (int i = 0; i < b.len; i++) ds4_tokens_push(&c, b.v[i]);
    for (int i = 0; i < G2_LEN; i++) ds4_tokens_push(&c, g2[i]);
    ds4_session *s3 = NULL;
    if (ds4_session_create(&s3, engine, 16384) != 0) { printf("  FAIL: session 3\n"); return 1; }
    ds4_session_reuse r3;
    ds4_session_reuse_report(s3, &c, &r3);
    printf("  turn 3 plan: source=%s reused=%d matched=%d prefilled=%d (prompt %d)\n",
           ds4_reuse_source_name(r3.source), r3.reused_tokens, r3.matched_tokens, r3.prefilled_tokens, c.len);
    if (r3.matched_tokens < c.len) {
        printf("  FAIL: turn 2's spans were not recorded (matched %d of %d)\n", r3.matched_tokens, c.len);
        fail = 1;
    }
    if (r3.prefilled_tokens > 64) {
        printf("  FAIL: turn 3 re-prefills %d tokens\n", r3.prefilled_tokens);
        fail = 1;
    }
    if (ds4_session_sync(s3, &c, err, sizeof(err)) != 0) { printf("  FAIL: turn 3 sync: %s\n", err); fail = 1; }
    ds4_session_free(s3);

    ds4_tokens_free(&a); ds4_tokens_free(&b); ds4_tokens_free(&c);

    /* A history rewrite (a compaction, or a new branch): the conversation
     * A -> B -> C -> D exists in the tree with depth below the branch point,
     * then the client sends A -> B -> E -> F -> G.  C and E are both user
     * turns, so they share the turn header tokens and diverge after them; F
     * and G extend E.  Seen on the server as every turn of the new branch
     * reusing only A -> B ("matched" four tokens past the checkpoint) and
     * "could not record" on each of them. */
    {
        enum { HDR = 4, TURN = 2000, GEN = 24, TURNS_OLD = 2, TURNS_NEW = 3 };
        int hdr[HDR];
        for (int i = 0; i < HDR; i++) hdr[i] = pool[20000 + i];
        ds4_tokens path = {0};
        for (int i = 0; i < A_LEN; i++) ds4_tokens_push(&path, pool[i]);      /* A B */
        int fork_len = 0;
        for (int branch = 0; branch < 2; branch++) {
            if (branch == 1) { path.len = fork_len; }
            const int turns = branch == 0 ? TURNS_OLD : TURNS_NEW;
            for (int t = 0; t < turns; t++) {
                for (int i = 0; i < HDR; i++) ds4_tokens_push(&path, hdr[i]);
                const int off = 30000 + (branch * 8 + t) * TURN;
                for (int i = 0; i < TURN; i++) ds4_tokens_push(&path, pool[off + i]);
                ds4_session *s = NULL;
                ds4_session_reuse r;
                if (ds4_session_create(&s, engine, 16384) != 0) { printf("  FAIL: session\n"); return 1; }
                ds4_session_reuse_report(s, &path, &r);
                printf("  branch %d turn %d: prompt %d source=%s matched=%d prefilled=%d\n", branch, t, path.len,
                       ds4_reuse_source_name(r.source), r.matched_tokens, r.prefilled_tokens);
                /* Everything but this turn's own text must already be in the tree. */
                const int expect = path.len - TURN - HDR + (branch == 1 && t == 0 ? HDR : 0);
                if (t + branch > 0 && r.matched_tokens < expect) {
                    printf("  FAIL: branch %d turn %d matched %d, expected at least %d\n", branch, t,
                           r.matched_tokens, expect);
                    fail = 1;
                }
                int gen[GEN];
                if (ds4_session_sync(s, &path, err, sizeof(err)) != 0 || !generate(s, gen, GEN, err, sizeof(err))) {
                    printf("  FAIL: branch %d turn %d: %s\n", branch, t, err); return 1;
                }
                ds4_session_cache_commit(s);
                ds4_session_free(s);
                if (branch == 0 && t == 0) fork_len = A_LEN;   /* the rewrite keeps A B only */
                for (int i = 0; i < GEN; i++) ds4_tokens_push(&path, gen[i]);
            }
        }
        ds4_tokens_free(&path);
    }
    free(pool);
    ds4_engine_close(engine);
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
