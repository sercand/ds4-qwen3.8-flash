/* qwen4exp pre-tokenizer gate.
 *
 * The GGUF declares tokenizer.ggml.pre = "qwen35"; ds4 implements that split
 * in bpe_tokenize_text_llama3_like(..., max_digit_run = 1).  These expectations
 * come from llama.cpp's own tokenizer on this exact GGUF:
 *
 *     build/bin/llama-tokenize -m <shard1> -p "<text>" --ids
 *
 * A wrong splitter still produces tokens, so nothing crashes -- the model just
 * sees a stream it was never trained on and degenerates a few tokens in.  That
 * is what this test exists to catch.
 *
 * Set DS4_QWEN4EXP_MODEL to the first GGUF shard to run; skips without it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

typedef struct {
    const char *text;
    int         n;
    int         ids[32];
} tok_case;

/* Digits one at a time and the contraction rule are where "qwen35" differs
 * from the DeepSeek splitter qwen4exp used to fall through to. */
static const tok_case CASES[] = {
    { "please explain linux cgroups to me", 7,
      { 29038, 10033, 35047, 272, 16261, 310, 728 } },
};
enum { N_CASES = (int)(sizeof(CASES) / sizeof(CASES[0])) };

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp tokenizer tests:\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL to the first GGUF shard\n");
        return 0;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = 512;
    opt.n_threads = 8;
    opt.power_percent = 100;
    /* Not inspect_only: the qwen4exp engine-open branch returns before
     * vocab_load in that mode, so the tokenizer would have no vocabulary.
     * That costs this test a full weight load. */

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        printf("  FAIL: cannot open the model\n");
        return 1;
    }

    int fail = 0;
    for (int c = 0; c < N_CASES; c++) {
        ds4_tokens got = {0};
        ds4_tokenize_text(engine, CASES[c].text, &got);
        int ok = got.len == CASES[c].n;
        for (int i = 0; ok && i < got.len; i++) {
            if (got.v[i] != CASES[c].ids[i]) ok = 0;
        }
        printf("  \"%s\"\n    ds4:       ", CASES[c].text);
        for (int i = 0; i < got.len; i++) printf("%d ", got.v[i]);
        printf("\n    llama.cpp: ");
        for (int i = 0; i < CASES[c].n; i++) printf("%d ", CASES[c].ids[i]);
        printf("\n    %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail = 1;
        ds4_tokens_free(&got);
    }

    ds4_engine_close(engine);
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
