/* Same N prompt rows three ways, in one process, compared on the last row's
 * logits: P = per-slot routed MoE prefill, G = expert-grouped routed MoE
 * prefill, seq = one token synced then single-row decode steps (the
 * llama.cpp-validated path; the reference).  Each is also run twice to show
 * run-to-run noise.  Env: DS4_QWEN4EXP_MODEL; Q4E_CMP_ROWS="8,15,16,17,32".
 * Build like tests/test_qwen4exp_batch (link against the CUDA CORE_OBJS). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4.h"

void ds4_gpu_q4e_set_moe_batch_min(uint32_t n);   /* test hook, ds4_qwen4exp_gpu.cuh */

static const char *TEXT =
    "The lighthouse keeper had not spoken to another person in eleven days when the "
    "bottle washed up against the rocks below the north stair. It was an ordinary "
    "green glass bottle, the kind that once held cheap wine, and its cork had been "
    "sealed over with candle wax. Inside, rolled tight, was a page torn from a ledger. "
    "He carried it up the hundred and twelve steps to the lamp room before he opened "
    "it, because the wind at the water's edge would have taken the paper the moment "
    "the wax gave way. The handwriting was small and even, the kind taught in schools "
    "before typewriters, and it began with a date from forty years earlier. An "
    "internal combustion engine converts the chemical energy in fuel into mechanical "
    "work through a repeating cycle of intake, compression, combustion and exhaust. "
    "During the intake stroke the piston moves down and draws a mixture of air and "
    "fuel into the cylinder through the open intake valve. The primes below one "
    "hundred are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, "
    "67, 71, 73, 79, 83, 89 and 97; there are twenty-five of them. Photosynthesis in "
    "green plants takes place in the chloroplasts, where light energy captured by "
    "chlorophyll drives the synthesis of sugars from carbon dioxide and water, "
    "releasing oxygen as a by-product. The Treaty of Westphalia in 1648 ended the "
    "Thirty Years' War and is often cited as the origin of the modern state system. "
    "In the second movement the cellos carry the theme while the violins answer in "
    "fragments, and the tempo marking asks for a walking pace that most conductors "
    "take a little slower than written. She closed the ledger, set the lamp, and "
    "watched the light sweep out across a sea that gave nothing back.";

static double rel_l2(const float *a, const float *ref, int n, double *max_abs) {
    double num = 0, den = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    *max_abs = mx;
    return den > 0 ? sqrt(num / den) : 0.0;
}
static int top2(const float *l, int n, double *margin) {
    int b = 0, s = -1;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    for (int i = 0; i < n; i++) if (i != b && (s < 0 || l[i] > l[s])) s = i;
    *margin = (double)l[b] - (double)l[s];
    return b;
}

static int run_rows(ds4_engine *e, const int32_t *toks, uint32_t n, uint32_t thresh, float *out, int nv) {
    ds4_session *s = NULL;
    char err[512] = {0};
    if (ds4_session_create(&s, e, 1024u) != 0) return 1;
    if (ds4_session_seed_prefill_for_test(s, toks, n) != 0) { ds4_session_free(s); return 1; }
    ds4_gpu_q4e_set_moe_batch_min(thresh);
    uint32_t pos0 = 0, nt = n;
    int32_t *amx = malloc((size_t)n * sizeof(int32_t));
    const int rc = ds4_sessions_forward_segmented(&s, &pos0, &nt, 1u, amx, err, sizeof(err));
    free(amx);
    ds4_gpu_q4e_set_moe_batch_min(0);
    if (rc != 0) { fprintf(stderr, "forward n=%u: %s\n", n, err); ds4_session_free(s); return 1; }
    if (ds4_session_copy_logits(s, out, nv) != nv) { ds4_session_free(s); return 1; }
    ds4_session_free(s);
    return 0;
}
static int run_seq(ds4_engine *e, const int32_t *toks, uint32_t n, float *out, int nv) {
    ds4_session *s = NULL;
    char err[512] = {0};
    if (ds4_session_create(&s, e, 1024u) != 0) return 1;
    ds4_tokens p = {0};
    ds4_tokens_push(&p, toks[0]);
    int rc = ds4_session_sync(s, &p, err, sizeof(err));
    ds4_tokens_free(&p);
    for (uint32_t i = 1; rc == 0 && i < n; i++) rc = ds4_session_eval(s, toks[i], err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "seq n=%u: %s\n", n, err); ds4_session_free(s); return 1; }
    if (ds4_session_copy_logits(s, out, nv) != nv) { ds4_session_free(s); return 1; }
    ds4_session_free(s);
    return 0;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    if (!model || !model[0]) { fprintf(stderr, "set DS4_QWEN4EXP_MODEL\n"); return 2; }
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model; opt.backend = DS4_BACKEND_CUDA; opt.context_size = 1024u;
    opt.n_threads = 8; opt.power_percent = 100; opt.exec_contexts = 2; opt.kv_pool_tokens = 4096u;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "open failed\n"); return 1; }
    const int nv = ds4_engine_vocab_size(e);
    ds4_tokens t = {0};
    ds4_tokenize_text(e, TEXT, &t);
    printf("text tokens: %d\n", (int)t.len);

    const char *rows_env = getenv("Q4E_CMP_ROWS");
    char rows[256];
    snprintf(rows, sizeof(rows), "%s", rows_env && rows_env[0] ? rows_env : "8,15,16,17,24,32,48");
    float *P = malloc((size_t)nv * 4), *P2 = malloc((size_t)nv * 4), *G = malloc((size_t)nv * 4),
          *G2 = malloc((size_t)nv * 4), *R = malloc((size_t)nv * 4);
    printf("%5s  %-22s %-22s %-22s   %s\n", "rows", "P vs seq", "G vs seq", "G vs P", "argmax(margin) seq / P / G");
    for (char *tok = strtok(rows, ","); tok; tok = strtok(NULL, ",")) {
        const uint32_t n = (uint32_t)atoi(tok);
        if (n == 0 || n > (uint32_t)t.len) { printf("%5u  (skipped: only %d tokens)\n", n, (int)t.len); continue; }
        if (run_rows(e, t.v, n, 1u << 20, P, nv) || run_rows(e, t.v, n, 1u << 20, P2, nv) ||
            run_rows(e, t.v, n, 2u, G, nv) || run_rows(e, t.v, n, 2u, G2, nv) || run_seq(e, t.v, n, R, nv)) {
            printf("%5u  FAILED\n", n); continue;
        }
        double mx, l2pp = rel_l2(P2, P, nv, &mx), l2gg = rel_l2(G2, G, nv, &mx);
        if (l2pp > 0 || l2gg > 0) printf("  (nondeterministic: P %.1e G %.1e)\n", l2pp, l2gg);
        double mxp, mxg, mxgp, mp, mg, mr;
        const double l2p = rel_l2(P, R, nv, &mxp), l2g = rel_l2(G, R, nv, &mxg), l2gp = rel_l2(G, P, nv, &mxgp);
        const int ap = top2(P, nv, &mp), ag = top2(G, nv, &mg), ar = top2(R, nv, &mr);
        printf("%5u  %.2e max %.4f  %.2e max %.4f  %.2e max %.4f   %d(%.3f) / %d(%.3f) / %d(%.3f)%s\n",
               n, l2p, mxp, l2g, mxg, l2gp, mxgp, ar, mr, ap, mp, ag, mg,
               (ap == ar && ag == ar) ? "" : "  <-- FLIP");
    }
    ds4_tokens_free(&t);
    free(P); free(P2); free(G); free(G2); free(R);
    ds4_engine_close(e);
    return 0;
}
