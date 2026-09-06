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
    /* The rest come from the checkpoint's own tokenizer.json through the
     * `tokenizers` library (misc/qwen4exp-oracle/tok_ref.py): the Unicode
     * classes \p{L}, \p{M}, \p{N} and \s of the qwen35 regex, which an
     * ASCII-shaped approximation gets wrong on marks, non-decimal numbers
     * (superscripts, fractions, roman numerals), emoji and exotic spaces. */
    { "na\u00efve caf\u00e9 \u2014 2\u00bd cups of cr\u00e8me br\u00fbl\u00e9e", 13,
      { 3267, 36328, 571, 50203, 1892, 220, 17, 25229, 24569, 314, 189628, 209239, 241308 } },
    { "\u0130stanbul'da \u0131\u015f\u0131k \u00e7ok g\u00fczel, de\u011fil mi?", 12,
      { 46186, 43610, 4035, 64, 235017, 39541, 55105, 184185, 11, 177682, 9217, 30 } },
    { "\u6771\u4eac\u30bf\u30ef\u30fc\u306f\u9ad8\u3044\u3067\u3059\u3002\u4eca\u65e5\u306f\u6674\u308c\u3002", 10,
      { 115197, 246217, 14876, 155380, 36298, 1710, 191506, 100253, 30967, 1710 } },
    { "hello \U0001f44b\U0001f3fd world \U0001f1f9\U0001f1f7 ok", 14,
      { 14556, 59720, 233, 9008, 237, 121, 1814, 10838, 229, 117, 9008, 229, 115, 5226 } },
    { "e\u0301 combining mark and x\u00b2 + y\u00b3 = z\u00b9\u2070", 14,
      { 933, 33041, 1804, 321, 830, 28495, 478, 374, 41776, 283, 1110, 57487, 50382, 108 } },
    { "Arabic digits \u0663\u0664\u0665 and fullwidth \uff11\uff12\uff13\uff14 digits", 19,
      { 6737, 65982, 17955, 220, 149, 96, 149, 97, 149, 98, 321, 2400, 2998, 220, 19496, 24128, 32405, 44085, 17955 } },
    { "tabs\tand  spaces\n\nnewlines\r\n mixed\u3000ideographic", 11,
      { 29975, 50711, 220, 12258, 271, 902, 7718, 317, 9238, 21742, 90762 } },
    { "def f(x):\n    return x**2  # comment\n", 14,
      { 727, 281, 2007, 1590, 198, 262, 460, 830, 332, 17, 220, 653, 3847, 198 } },
    { "\u2167 roman, \u00bd \u00be fractions, \u2460 circled", 15,
      { 68086, 100, 46182, 11, 220, 25229, 220, 65121, 62700, 11, 220, 46701, 254, 4086, 806 } },
    { "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0438 \u03b5\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac", 6,
      { 152528, 149800, 185386, 7347, 210061, 159778 } },
    { "\u092e\u0930\u093e\u0920\u0940 \u0939\u093f\u0928\u094d\u0926\u0940 \u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0ba4\u0bae\u0bbf\u0bb4\u0bcd", 11,
      { 84237, 171597, 164567, 190488, 150127, 177453, 174056, 161539, 153106, 238318, 44808 } },
    { "I've you're they'll don't IT'S", 10,
      { 40, 2908, 488, 2224, 781, 3172, 1459, 914, 8435, 12887 } },
    { "a\u200db zero-width joiner and\u00a0nbsp here", 12,
      { 64, 373, 235, 65, 6942, 9069, 4973, 261, 321, 3966, 5496, 1532 } },
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
        printf("\n    reference: ");
        for (int i = 0; i < CASES[c].n; i++) printf("%d ", CASES[c].ids[i]);
        printf("\n    %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail = 1;
        ds4_tokens_free(&got);
    }

    ds4_engine_close(engine);
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
