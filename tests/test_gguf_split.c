/* model_open() maps a GGUF split as one contiguous range so the rest of the
 * engine never learns the model spans files.  This test builds a small GGUF,
 * builds the same tensors as a two-way split, and requires both to resolve to
 * byte-identical tensor data.  It is the only coverage of the single-file path
 * on machines with no full model checkout, so keep it cheap and self-contained. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ds4.h"

#define GGUF_MAGIC 0x46554747u
#define VAL_UINT16 2u
#define VAL_UINT32 4u
#define VAL_INT32  5u
#define VAL_STRING 8u

typedef struct { uint8_t *p; size_t n, cap; } buf;

static void put(buf *b, const void *src, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2 + 4096;
        b->p = realloc(b->p, b->cap);
        if (!b->p) { fprintf(stderr, "oom\n"); exit(1); }
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void u16v(buf *b, uint16_t v) { put(b, &v, 2); }
static void u32v(buf *b, uint32_t v) { put(b, &v, 4); }
static void u64v(buf *b, uint64_t v) { put(b, &v, 8); }
static void strv(buf *b, const char *s) { uint64_t n = strlen(s); u64v(b, n); put(b, s, n); }
static void kv_u32(buf *b, const char *k, uint32_t v) { strv(b, k); u32v(b, VAL_UINT32); u32v(b, v); }
static void kv_u16(buf *b, const char *k, uint16_t v) { strv(b, k); u32v(b, VAL_UINT16); u16v(b, v); }
static void kv_i32(buf *b, const char *k, int32_t v)  { strv(b, k); u32v(b, VAL_INT32);  put(b, &v, 4); }
static void kv_str(buf *b, const char *k, const char *v) { strv(b, k); u32v(b, VAL_STRING); strv(b, v); }

/* Deterministic filler so the two layouts hold the same bytes. */
static void fill(uint8_t *dst, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) dst[i] = (uint8_t)(seed * 2654435761u + i * 40503u + (i >> 3));
}

#define ALIGN 32u
static size_t align_up_sz(size_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }

typedef struct { const char *name; uint64_t d0, d1; } tensor_spec;

static const tensor_spec SPECS[] = {
    { "token_embd.weight", 64, 8 },
    { "blk.0.attn_q.weight", 32, 16 },
    { "blk.0.ffn_down.weight", 16, 48 },
    { "output.weight", 64, 8 },
};
static const size_t N_SPECS = sizeof(SPECS) / sizeof(SPECS[0]);

/* F32 tensors keep the size arithmetic trivial: bytes = d0*d1*4. */
static uint64_t spec_bytes(const tensor_spec *s) { return s->d0 * s->d1 * 4u; }

static void write_gguf(const char *path, size_t first, size_t count,
                       int split_no, int split_count) {
    buf hdr = {0};
    u32v(&hdr, GGUF_MAGIC);
    u32v(&hdr, 3);
    u64v(&hdr, count);

    buf meta = {0};
    uint64_t n_kv = 0;
    if (split_count > 1) {
        kv_u16(&meta, "split.no", (uint16_t)split_no);
        kv_u16(&meta, "split.count", (uint16_t)split_count);
        kv_i32(&meta, "split.tensors.count", (int32_t)N_SPECS);
        n_kv = 3;
    }
    if (split_no == 0) {
        kv_str(&meta, "general.architecture", "ds4-split-test");
        kv_u32(&meta, "general.alignment", ALIGN);
        n_kv += 2;
    }
    u64v(&hdr, n_kv);
    put(&hdr, meta.p, meta.n);
    free(meta.p);

    buf dir = {0};
    uint64_t rel = 0;
    for (size_t i = 0; i < count; i++) {
        const tensor_spec *s = &SPECS[first + i];
        strv(&dir, s->name);
        u32v(&dir, 2);
        u64v(&dir, s->d0);
        u64v(&dir, s->d1);
        u32v(&dir, 0 /* F32 */);
        u64v(&dir, rel);
        rel = align_up_sz(rel + spec_bytes(s));
    }

    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(hdr.p, 1, hdr.n, f);
    fwrite(dir.p, 1, dir.n, f);
    const size_t head = align_up_sz(hdr.n + dir.n);
    for (size_t pad = hdr.n + dir.n; pad < head; pad++) fputc(0, f);
    free(hdr.p); free(dir.p);

    uint64_t at = 0;
    for (size_t i = 0; i < count; i++) {
        const tensor_spec *s = &SPECS[first + i];
        const uint64_t n = spec_bytes(s);
        uint8_t *data = malloc((size_t)n);
        /* Seed by global index so a tensor holds the same bytes in both layouts. */
        fill(data, (size_t)n, (uint32_t)(first + i) + 1u);
        fwrite(data, 1, (size_t)n, f);
        free(data);
        const uint64_t next = align_up_sz((size_t)(at + n));
        for (uint64_t pad = at + n; pad < next; pad++) fputc(0, f);
        at = next;
    }
    fclose(f);
}

int main(void) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = "/tmp";
    char single[1024], shard1[1024], shard2[1024];
    snprintf(single, sizeof(single), "%s/ds4_split_single-00001-of-00001.gguf", dir);
    snprintf(shard1, sizeof(shard1), "%s/ds4_split_pair-00001-of-00002.gguf", dir);
    snprintf(shard2, sizeof(shard2), "%s/ds4_split_pair-00002-of-00002.gguf", dir);

    write_gguf(single, 0, N_SPECS, 0, 1);
    write_gguf(shard1, 0, 2, 0, 2);
    write_gguf(shard2, 2, N_SPECS - 2, 1, 2);

    uint64_t d1 = 0, t1 = 0, b1 = 0, d2 = 0, t2 = 0, b2 = 0;
    ds4_test_model_tensor_digest(single, &d1, &t1, &b1);
    ds4_test_model_tensor_digest(shard1, &d2, &t2, &b2);

    remove(single); remove(shard1); remove(shard2);

    int fail = 0;
    printf("gguf split tests:\n");
    printf("  single file: %llu tensors, %llu bytes\n",
           (unsigned long long)t1, (unsigned long long)b1);
    printf("  two-way split: %llu tensors, %llu bytes\n",
           (unsigned long long)t2, (unsigned long long)b2);
    if (t1 != N_SPECS) { printf("  FAIL: single file tensor count\n"); fail = 1; }
    if (t1 != t2)      { printf("  FAIL: tensor counts differ\n"); fail = 1; }
    if (b1 != b2)      { printf("  FAIL: tensor byte totals differ\n"); fail = 1; }
    if (d1 != d2) {
        printf("  FAIL: digests differ (%llx vs %llx)\n",
               (unsigned long long)d1, (unsigned long long)d2);
        fail = 1;
    }
    if (!fail) printf("  split and single file resolve identical tensor data: PASS\n");
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
