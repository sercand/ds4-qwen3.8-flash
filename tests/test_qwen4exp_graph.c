/* Phase-4 gate: run the qwen4exp graph on the oracle's prompt and compare.
 *
 * Token ids are supplied directly so this does not depend on the tokenizer
 * being wired up.  With DS4_QWEN4EXP_TRACE=1 the engine prints one line per
 * named intermediate, which diffs against misc/qwen4exp-oracle/golden_sums.txt.
 *
 * Set DS4_QWEN4EXP_MODEL to the first GGUF shard to run; skips without it.
 * With DS4_QWEN4EXP_MTP naming the MTP sidecar, the greedy continuation is
 * generated a second time through the speculative path and must reproduce the
 * plain one token for token. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ds4.h"

/* "The capital of France is" */
static const int TOKENS[] = { 760, 6511, 314, 9338, 369 };
enum { N_TOKENS = 5 };

/* llama.cpp picks token 11751 (" Paris") for this prompt.
 *
 * The gate is the chosen token, not the logit values.  ds4 and llama.cpp
 * quantize activations with different kernels, so ~400 quantized matmuls of
 * independent rounding leave the logits a couple of units apart on a +-20
 * scale while the distribution is the same.  Requiring bit-similar logits
 * would be testing that two implementations round identically, which they do
 * not and need not. */
static const int GOLD_ARGMAX = 11751;
static const float GOLD_LOGIT_SUM = -366740.937500f;
/* L2 norm of llama.cpp's logits for the same prompt (llama-eval-callback
 * prints l1/l2 per tensor).  The sum above cancels over 248k near-zero-mean
 * logits, so a path change that leaves the distribution equally good can
 * swing it from 0.3% to 8% -- the f32 cuBLAS fallback and the Q8-activation
 * rows kernel sit at +1.7% and -1.8% of this norm respectively while their
 * sums read 0.3% and 8.4% off.  Read the norm drift, not the sum drift. */
static const double GOLD_LOGIT_L2 = 1236.895196;

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp graph tests:\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL to the first GGUF shard\n");
        return 0;
    }

    /* The correctness gate needs only a handful of tokens, but the prefill
     * benchmark below wants a long prompt, so the context is sized to fit
     * whatever DS4_QWEN4EXP_PREFILL_TOKENS asks for. */
    int prefill_tokens = 0;
    const char *prefill_env = getenv("DS4_QWEN4EXP_PREFILL_TOKENS");
    if (prefill_env && prefill_env[0]) {
        const int v = atoi(prefill_env);
        if (v > 0) prefill_tokens = v;
    }
    int decode_tokens = 64;
    const char *want_env = getenv("DS4_QWEN4EXP_DECODE_TOKENS");
    if (want_env && want_env[0]) {
        const int v = atoi(want_env);
        if (v > 0) decode_tokens = v;
    }
    int ctx = prefill_tokens + decode_tokens + 64;
    if (ctx < 512) ctx = 512;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = (uint32_t)ctx;
    opt.n_threads = 8;
    opt.power_percent = 100;
    const char *mtp = getenv("DS4_QWEN4EXP_MTP");
    if (mtp && mtp[0]) opt.mtp_path = mtp;
    /* DS4_QWEN4EXP_MTP_DRAFT overrides the drafts per step (default 3). */
    const char *mtp_k = getenv("DS4_QWEN4EXP_MTP_DRAFT");
    if (mtp_k && mtp_k[0]) opt.mtp_draft_tokens = atoi(mtp_k);

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        printf("  FAIL: cannot open the model\n");
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, (uint32_t)ctx) != 0) {
        printf("  FAIL: cannot create a session\n");
        ds4_engine_close(engine);
        return 1;
    }

    /* The golden dump covers the five-token prompt above, which is short
     * enough that a kernel wrong only past a handful of positions still
     * passes.  DS4_QWEN4EXP_PROMPT_IDS overrides it with a comma-separated
     * list so the same comparison can be run against a longer
     * llama-eval-callback dump. */
    ds4_tokens prompt = {0};
    const char *ids_env = getenv("DS4_QWEN4EXP_PROMPT_IDS");
    if (ids_env && ids_env[0]) {
        const char *p = ids_env;
        while (*p) {
            char *end = NULL;
            const long v = strtol(p, &end, 10);
            if (end == p) break;
            ds4_tokens_push(&prompt, (int)v);
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
        printf("  prompt overridden: %d tokens\n", prompt.len);
    } else {
        for (int i = 0; i < N_TOKENS; i++) ds4_tokens_push(&prompt, TOKENS[i]);
    }

    char err[512] = {0};
    int fail = 0;
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        printf("  FAIL: sync: %s\n", err);
        fail = 1;
    } else {
        const int n_vocab = ds4_engine_vocab_size(engine);
        float *logits = malloc((size_t)n_vocab * sizeof(float));
        if (ds4_session_copy_logits(session, logits, n_vocab) != n_vocab) {
            printf("  FAIL: cannot read logits\n");
            fail = 1;
        } else {
            double sum = 0.0, sq = 0.0;
            int argmax = 0;
            for (int i = 0; i < n_vocab; i++) {
                sum += logits[i];
                sq += (double)logits[i] * (double)logits[i];
                if (logits[i] > logits[argmax]) argmax = i;
            }
            printf("  logits: sum=%.3f (llama.cpp %.3f) l2=%.3f (llama.cpp %.3f)\n",
                   sum, (double)GOLD_LOGIT_SUM, sqrt(sq), GOLD_LOGIT_L2);
            size_t len = 0;
            char *text = ds4_token_text(engine, argmax, &len);
            printf("  argmax: %d '%.*s'\n", argmax, (int)len, text ? text : "");

            /* GOLD_ARGMAX belongs to the fixed prompt; an overridden one has
             * no expectation to check against here -- diff it against a
             * regenerated dump instead (see misc/qwen4exp-oracle/README.md). */
            if (ids_env && ids_env[0]) {
                printf("  argmax check skipped: prompt overridden\n");
            } else if (argmax != GOLD_ARGMAX) {
                printf("  FAIL: argmax %d, llama.cpp picks %d\n", argmax, GOLD_ARGMAX);
                fail = 1;
            } else {
                printf("  argmax matches llama.cpp: PASS\n");
            }
            /* Informational: how far the two quantized paths have drifted. */
            printf("  logit-sum drift vs llama.cpp: %.1f%%\n",
                   fabs(sum - (double)GOLD_LOGIT_SUM) / fabs((double)GOLD_LOGIT_SUM) * 100.0);
            if (!(ids_env && ids_env[0])) {
                const double l2_drift = fabs(sqrt(sq) - GOLD_LOGIT_L2) / GOLD_LOGIT_L2 * 100.0;
                printf("  logit-L2 drift vs llama.cpp: %.1f%% (healthy < 3%%)\n", l2_drift);
                if (l2_drift > 5.0) {
                    printf("  FAIL: logit L2 norm drifted %.1f%%\n", l2_drift);
                    fail = 1;
                }
            }
        }
        free(logits);
    }

    /* Decode is a different path from prefill -- one token at a time, with the
     * KV cache and both recurrent states advancing in place -- so generate a
     * continuation rather than trusting the single prefill row.
     *
     * Read the continuation as a smoke test, not a gate.  "The capital of
     * France is" carries no chat framing, so after " Paris." the model is
     * genuinely undecided about what to continue with, and any change to a
     * prefill kernel's rounding lands it somewhere else -- a list of European
     * capitals, a list of dates, a news feed.  All are the model's own output;
     * none indicates a bug on their own.  The gates that mean something are
     * the argmax above and an elementwise diff of DS4_QWEN4EXP_TRACE=1 against
     * misc/qwen4exp-oracle/golden_sums.txt. */
    int *reference = NULL;
    int reference_len = 0;
    /* DS4_QWEN4EXP_SKIP_GREEDY=1 leaves out the plain generation so a profile
     * of the speculative session is not mixed with it; the stream comparison
     * is then skipped too. */
    const char *skip_greedy = getenv("DS4_QWEN4EXP_SKIP_GREEDY");
    if (!fail && skip_greedy && skip_greedy[0] && skip_greedy[0] != '0') {
        reference = calloc((size_t)decode_tokens + 1, sizeof(int));
        reference_len = decode_tokens;
        for (int i = 0; i < decode_tokens; i++) reference[i] = -1;
        printf("  greedy continuation skipped\n");
    } else if (!fail) {
        printf("  greedy continuation:");
        fflush(stdout);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int produced = 0;
        reference = calloc((size_t)decode_tokens + 1, sizeof(int));
        /* 64 tokens is enough to show the continuation is sane; a profiler or
         * benchmark run wants a longer steady state, so the count is
         * overridable -- and when it is, an end-of-text token no longer cuts
         * the run short, because the point is then the token rate. */
        const int want = decode_tokens;
        const int stop_at_eos = !(want_env && want_env[0]);
        int token = ds4_session_argmax(session);
        for (int i = 0; i < want && token >= 0; i++, produced++) {
            size_t len = 0;
            char *text = ds4_token_text(engine, token, &len);
            printf("%.*s", (int)len, text ? text : "");
            fflush(stdout);
            reference[reference_len++] = token;
            if (stop_at_eos && ds4_token_is_stop(engine, token)) break;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                printf("\n  FAIL: decode step %d: %s\n", i, err);
                fail = 1;
                break;
            }
            token = ds4_session_argmax(session);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double secs = (double)(t1.tv_sec - t0.tv_sec) +
                            (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("\n  decode: %d tokens in %.2fs = %.1f tok/s (llama.cpp: 24.1)\n",
               produced, secs, produced / secs);
    }

    /* Speculative decode.  The MTP head drafts K tokens per step and the
     * target verifies them in one pass; with greedy acceptance the committed
     * stream must be the plain greedy stream.  A different row count through
     * the dense matmuls can reassociate a sum and flip a near-tied argmax, so
     * a divergence is reported rather than failed -- but it should be rare,
     * and the first tokens must agree. */
    if (!fail && ds4_engine_has_mtp(engine) && reference_len > 0) {
        ds4_session *spec = NULL;
        if (ds4_session_create(&spec, engine, (uint32_t)ctx) != 0) {
            printf("  FAIL: cannot create the speculative session\n");
            fail = 1;
        } else if (ds4_session_sync(spec, &prompt, err, sizeof(err)) != 0) {
            printf("  FAIL: speculative sync: %s\n", err);
            fail = 1;
        } else {
            printf("  speculative continuation (K=%d):", ds4_engine_mtp_draft_tokens(engine));
            fflush(stdout);
            const int eos = ds4_token_eos(engine);
            int *got = calloc((size_t)reference_len + 32, sizeof(int));
            int got_len = 0;
            int steps = 0;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            while (got_len < reference_len) {
                int toks[17];
                const int first = ds4_session_argmax(spec);
                const int n = ds4_session_eval_speculative_argmax(
                        spec, first, reference_len - got_len, eos,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                if (n <= 0) {
                    printf("\n  FAIL: speculative step %d: %s\n", steps, err);
                    fail = 1;
                    break;
                }
                steps++;
                for (int i = 0; i < n && got_len < reference_len + 16; i++) {
                    size_t len = 0;
                    char *text = ds4_token_text(engine, toks[i], &len);
                    printf("%.*s", (int)len, text ? text : "");
                    got[got_len++] = toks[i];
                }
                fflush(stdout);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            const double secs = (double)(t1.tv_sec - t0.tv_sec) +
                                (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
            int agree = 0;
            while (agree < reference_len && agree < got_len && got[agree] == reference[agree]) agree++;
            printf("\n  speculative: %d tokens in %d steps, %.2fs = %.1f tok/s "
                   "(%.2f tokens/step)\n",
                   got_len, steps, secs, got_len / secs,
                   steps ? (double)got_len / steps : 0.0);
            if (reference[0] < 0) {
                printf("  speculative stream comparison skipped\n");
            } else if (agree == reference_len) {
                printf("  speculative stream matches greedy: PASS\n");
            } else {
                printf("  speculative stream diverges from greedy at token %d of %d "
                       "(greedy %d, speculative %d)\n",
                       agree, reference_len,
                       agree < reference_len ? reference[agree] : -1,
                       agree < got_len ? got[agree] : -1);
                if (agree < 8) {
                    printf("  FAIL: divergence too early to be rounding\n");
                    fail = 1;
                }
            }
            free(got);
            ds4_session_free(spec);
        }
    }
    free(reference);

    /* Prefill benchmark.  A fresh session, because syncing a longer prompt to
     * the one above would reuse its prefix and time only the tail.  The token
     * ids repeat the fixed prompt so the PLE gather sees realistic n-gram
     * repetition rather than a pathological all-hit or all-miss stream. */
    if (!fail && prefill_tokens > 0) {
        ds4_session *bench = NULL;
        if (ds4_session_create(&bench, engine, (uint32_t)ctx) != 0) {
            printf("  FAIL: cannot create the prefill session\n");
            fail = 1;
        } else {
            /* A varied token stream, not the fixed prompt repeated.  Repeats
             * put attention on hundreds of tied keys, where the softmax is so
             * ill-conditioned that two correct kernels disagree by percent on
             * the logits -- useless as a fingerprint, and unrepresentative of
             * expert routing too.  A fixed LCG over the common part of the
             * vocabulary keeps the run reproducible. */
            ds4_tokens long_prompt = {0};
            uint32_t lcg = 12345u;
            for (int i = 0; i < prefill_tokens; i++) {
                lcg = lcg * 1103515245u + 12345u;
                ds4_tokens_push(&long_prompt, (int)((lcg >> 16) % 20000u));
            }
            struct timespec p0, p1;
            clock_gettime(CLOCK_MONOTONIC, &p0);
            const int rc = ds4_session_sync(bench, &long_prompt, err, sizeof(err));
            clock_gettime(CLOCK_MONOTONIC, &p1);
            if (rc != 0) {
                printf("  FAIL: prefill sync: %s\n", err);
                fail = 1;
            } else {
                const double psecs = (double)(p1.tv_sec - p0.tv_sec) +
                                     (double)(p1.tv_nsec - p0.tv_nsec) * 1e-9;
                printf("  prefill: %d tokens in %.2fs = %.0f tok/s (target 800)\n",
                       prefill_tokens, psecs, prefill_tokens / psecs);
                /* A fingerprint of the long-prefill result.  The tiled
                 * attention kernel only engages above a chunk of 16 tokens, so
                 * this is the only place it is exercised; running twice with
                 * DS4_QWEN4EXP_NO_TILED_ATTN=1 must print the same numbers. */
                const int nv = ds4_engine_vocab_size(engine);
                float *lg = malloc((size_t)nv * sizeof(float));
                if (lg && ds4_session_copy_logits(bench, lg, nv) == nv) {
                    double s = 0.0;
                    int am = 0;
                    for (int i = 0; i < nv; i++) {
                        s += lg[i];
                        if (lg[i] > lg[am]) am = i;
                    }
                    printf("  prefill fingerprint: argmax=%d sum=%.3f\n", am, s);
                }
                free(lg);
            }
            ds4_tokens_free(&long_prompt);
            ds4_session_free(bench);
        }
    }

    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
