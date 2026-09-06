/* NFC normalization (ds4_nfc.inc) against Python's unicodedata.normalize on
 * the shapes that matter for the tokenizer: combining marks composing onto
 * Latin letters, reordering of marks by combining class, composition
 * exclusions (U+0958 stays decomposed, U+212B becomes U+00C5), Hangul jamo
 * to syllables, and text that is already NFC coming back untouched.  The full
 * 3000-string random comparison lives in the session notes; this pins the
 * cases a table regeneration could break.  No model, no GPU. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_nfc.inc"

static const struct { const char *in, *nfc; } CASES[] = {
    { "plain ascii text", "plain ascii text" },
    { "é", "é" },                                  /* e + acute -> é */
    { "ệ", "ệ" },                            /* ordering then composition */
    { "ệ", "ệ" },                            /* same marks, other order */
    { "ḍ̇", "ḍ̇" },                      /* second mark has no composite */
    { "Å", "Å" },                                    /* Angstrom sign -> A ring (singleton) */
    { "Ω", "Ω" },                                    /* Ohm -> Omega */
    { "각", "각" },                        /* jamo L V T -> syllable */
    { "각", "각" },                              /* LV + T */
    { "क़", "क़" },                              /* composition exclusion stays apart */
    { "ぢ", "ぢ" },                              /* hiragana voiced mark */
    { "café naïve", "café naïve" },         /* already NFC */
    { "ṩ", "ṩ" },                                    /* s with dot below and above, stays */
    { "x́́", "x́́" },                      /* no composite for x */
};

int main(void) {
    int fail = 0;
    const int n = (int)(sizeof(CASES) / sizeof(CASES[0]));
    for (int i = 0; i < n; i++) {
        uint64_t olen = 0;
        char *o = nfc_normalize(CASES[i].in, strlen(CASES[i].in), &olen);
        const char *got = o ? o : CASES[i].in;
        const size_t glen = o ? (size_t)olen : strlen(CASES[i].in);
        const bool ok = glen == strlen(CASES[i].nfc) && memcmp(got, CASES[i].nfc, glen) == 0;
        /* Already-normalized input must come back as NULL (no copy). */
        const bool identity_ok = strcmp(CASES[i].in, CASES[i].nfc) != 0 || o == NULL;
        if (!ok || !identity_ok) {
            printf("  FAIL case %d: got", i);
            for (size_t k = 0; k < glen; k++) printf(" %02x", (unsigned char)got[k]);
            printf(" (%s)\n", identity_ok ? "value" : "copied an NFC input");
            fail = 1;
        }
        free(o);
    }
    printf("test_nfc: %d cases %s\n", n, fail ? "FAILED" : "PASS");
    return fail;
}
