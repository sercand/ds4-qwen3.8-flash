/* NFC normalization for the tokenizer input (UAX #15 canonical decomposition,
 * ordering and composition), on the tables of ds4_unicode_nfc.inc.  The Qwen3.8
 * tokenizer.json runs NFC before its pre-tokenizer regex; without it a
 * decomposed "e" U+0301 tokenizes as two pieces where the reference gives the
 * precomposed letter.  Text that is already NFC -- nearly all of it -- comes
 * back unchanged, and pure ASCII is recognized without decoding.
 *
 * Included by ds4.c; self-contained so tests/test_nfc.c can build it alone. */
#include "ds4_unicode_nfc.inc"

static uint8_t nfc_class(uint32_t cp) {
    size_t lo = 0, hi = sizeof(nfc_cccs) / sizeof(nfc_cccs[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < nfc_cccs[mid].lo) hi = mid;
        else if (cp > nfc_cccs[mid].hi) lo = mid + 1;
        else return nfc_cccs[mid].ccc;
    }
    return 0;
}

static const nfc_decomp *nfc_find_decomp(uint32_t cp) {
    size_t lo = 0, hi = sizeof(nfc_decomps) / sizeof(nfc_decomps[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < nfc_decomps[mid].cp) hi = mid;
        else if (cp > nfc_decomps[mid].cp) lo = mid + 1;
        else return &nfc_decomps[mid];
    }
    return NULL;
}

/* Primary composite of (a, b), or 0. */
static uint32_t nfc_compose_pair(uint32_t a, uint32_t b) {
    /* Hangul: L + V -> LV, LV + T -> LVT. */
    if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176) {
        return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    }
    if (a >= 0xAC00 && a <= 0xD7A3 && (a - 0xAC00) % 28 == 0 && b > 0x11A7 && b < 0x11C3) {
        return a + (b - 0x11A7);
    }
    size_t lo = 0, hi = sizeof(nfc_pairs) / sizeof(nfc_pairs[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const nfc_pair *p = &nfc_pairs[mid];
        if (a < p->first || (a == p->first && b < p->second)) hi = mid;
        else if (a > p->first || b > p->second) lo = mid + 1;
        else return p->composed;
    }
    return 0;
}

static uint32_t nfc_utf8_decode(const uint8_t *s, uint64_t len, uint64_t *pos) {
    const uint8_t c0 = s[*pos];
    int n = c0 < 0x80 ? 1 : (c0 >> 5) == 0x6 ? 2 : (c0 >> 4) == 0xe ? 3 : (c0 >> 3) == 0x1e ? 4 : 1;
    if (*pos + (uint64_t)n > len) n = 1;
    uint32_t cp = c0;
    if (n == 2) cp = ((uint32_t)(c0 & 0x1f) << 6) | (s[*pos + 1] & 0x3f);
    else if (n == 3) cp = ((uint32_t)(c0 & 0x0f) << 12) | ((uint32_t)(s[*pos + 1] & 0x3f) << 6) | (s[*pos + 2] & 0x3f);
    else if (n == 4) cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[*pos + 1] & 0x3f) << 12) |
                          ((uint32_t)(s[*pos + 2] & 0x3f) << 6) | (s[*pos + 3] & 0x3f);
    *pos += (uint64_t)n;
    return cp;
}

static uint64_t nfc_utf8_encode(uint32_t cp, uint8_t *out) {
    if (cp < 0x80) { out[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) { out[0] = (uint8_t)(0xc0 | (cp >> 6)); out[1] = (uint8_t)(0x80 | (cp & 0x3f)); return 2; }
    if (cp < 0x10000) {
        out[0] = (uint8_t)(0xe0 | (cp >> 12)); out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (uint8_t)(0x80 | (cp & 0x3f)); return 3;
    }
    out[0] = (uint8_t)(0xf0 | (cp >> 18)); out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3f)); out[3] = (uint8_t)(0x80 | (cp & 0x3f)); return 4;
}

/* NFC of text[0, len).  Returns a malloc'd buffer of *out_len bytes when the
 * normalized form differs from the input, NULL when the input is already NFC
 * (the common case, and always for ASCII). */
static char *nfc_normalize(const char *text, uint64_t len, uint64_t *out_len) {
    const uint8_t *s = (const uint8_t *)text;
    bool ascii = true;
    for (uint64_t i = 0; i < len; i++) if (s[i] >= 0x80) { ascii = false; break; }
    if (ascii) return NULL;

    /* Decompose.  Every decomposition is at most NFC_DECOMP_MAX code points
     * and Hangul at most 3, and a code point is at least one byte of input. */
    uint32_t *buf = malloc((size_t)(len * NFC_DECOMP_MAX + 1) * sizeof(uint32_t));
    uint8_t *cls = malloc((size_t)(len * NFC_DECOMP_MAX + 1));
    if (!buf || !cls) { free(buf); free(cls); return NULL; }
    uint64_t n = 0;
    for (uint64_t pos = 0; pos < len;) {
        const uint32_t cp = nfc_utf8_decode(s, len, &pos);
        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            const uint32_t si = cp - 0xAC00;
            buf[n++] = 0x1100 + si / 588;
            buf[n++] = 0x1161 + (si % 588) / 28;
            if (si % 28) buf[n++] = 0x11A7 + si % 28;
            continue;
        }
        const nfc_decomp *d = nfc_find_decomp(cp);
        if (d) { for (uint8_t k = 0; k < d->n; k++) buf[n++] = d->seq[k]; }
        else buf[n++] = cp;
    }
    for (uint64_t i = 0; i < n; i++) cls[i] = nfc_class(buf[i]);

    /* Canonical ordering: stable sort each run of non-starters by class. */
    for (uint64_t i = 0; i < n;) {
        if (cls[i] == 0) { i++; continue; }
        uint64_t j = i;
        while (j < n && cls[j] != 0) j++;
        for (uint64_t a = i + 1; a < j; a++) {
            const uint32_t v = buf[a]; const uint8_t c = cls[a];
            uint64_t b = a;
            while (b > i && cls[b - 1] > c) { buf[b] = buf[b - 1]; cls[b] = cls[b - 1]; b--; }
            buf[b] = v; cls[b] = c;
        }
        i = j;
    }

    /* Canonical composition (the standard's sample algorithm). */
    if (n > 1) {
        uint64_t starter_pos = 0, comp_pos = 1;
        uint32_t starter = buf[0];
        int last_class = cls[0] == 0 ? 0 : 256;
        for (uint64_t i = 1; i < n; i++) {
            const uint32_t ch = buf[i];
            const int ch_class = cls[i];
            const uint32_t composite = nfc_compose_pair(starter, ch);
            if (composite && (last_class < ch_class || last_class == 0)) {
                buf[starter_pos] = composite;
                starter = composite;
            } else {
                if (ch_class == 0) { starter_pos = comp_pos; starter = ch; }
                last_class = ch_class;
                buf[comp_pos] = ch;
                cls[comp_pos] = (uint8_t)ch_class;
                comp_pos++;
            }
        }
        n = comp_pos;
    }
    free(cls);

    uint8_t *out = malloc((size_t)n * 4 + 1);
    if (!out) { free(buf); return NULL; }
    uint64_t o = 0;
    for (uint64_t i = 0; i < n; i++) o += nfc_utf8_encode(buf[i], out + o);
    free(buf);
    out[o] = 0;
    if (o == len && memcmp(out, text, (size_t)len) == 0) { free(out); return NULL; }
    *out_len = o;
    return (char *)out;
}
