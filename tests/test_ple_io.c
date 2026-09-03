/* PLE reader I/O backends.
 *
 * The row gather has two implementations of the same contract: a pool of
 * threads doing blocking preads, and an io_uring submission queue.  They must
 * be indistinguishable to a caller -- same bytes, same cache behaviour, same
 * handling of the awkward shapes the engine actually produces (a batch with
 * the same row twice, a batch wider than the queue, the final row sitting in
 * the file's last partial block).
 *
 * The table here is fabricated, so this needs no model and no GPU: every row
 * is filled from its own index, which makes a misdirected read obvious instead
 * of plausible.  See tests/test_qwen4exp_ple.c for the codec gate against
 * llama.cpp's real ple_embd. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4_ple_stream.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                              \
            printf("\n");                                                     \
            g_fail++;                                                         \
        }                                                                     \
    } while (0)

enum {
    TABLE_ROWS  = 20000,   /* 2.4 MB of rows, spanning ~600 4 KiB blocks */
    FILE_OFFSET = 1234,    /* deliberately not block aligned */
};

/* open() validates that EXL3 rows come with a bias table.  Nothing in the read
 * path touches it -- fetch returns raw quant bytes -- so zeros will do. */
static uint16_t g_head_bias[16 * 160];

static ds4_ple_params make_params(void) {
    ds4_ple_params p;
    memset(&p, 0, sizeof(p));
    p.row_type        = DS4_PLE_ROW_EXL3_K6;   /* 122-byte rows */
    p.ngram_size      = 3;
    p.heads_per_ngram = 8;
    p.n_heads         = 16;
    p.head_dim        = 160;
    p.head_bias       = g_head_bias;
    return p;
}

/* Row `i` byte `j` is a function of both, so a read that lands on the wrong
 * row or the wrong offset inside a row cannot look correct. */
static uint8_t row_byte(uint64_t row, uint32_t j) {
    return (uint8_t)(row * 31u + j * 7u + 11u);
}

/* Writes the fabricated table and returns an fd, or -1.  O_DIRECT reads see
 * only what has reached the device, so this fsyncs. */
static int make_table(const ds4_ple_params *p, char *path, size_t pathlen) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    snprintf(path, pathlen, "%s/ds4_ple_io_test.XXXXXX", dir);
    const int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return -1; }

    const uint32_t rb = ds4_ple_row_bytes(p);
    uint8_t *pad = calloc(1, FILE_OFFSET);
    if (!pad || write(fd, pad, FILE_OFFSET) != (ssize_t)FILE_OFFSET) { free(pad); close(fd); return -1; }
    free(pad);

    uint8_t *buf = malloc((size_t)rb * 512);
    if (!buf) { close(fd); return -1; }
    for (uint64_t base = 0; base < TABLE_ROWS; base += 512) {
        const uint64_t n = (TABLE_ROWS - base) < 512 ? (TABLE_ROWS - base) : 512;
        for (uint64_t r = 0; r < n; r++)
            for (uint32_t j = 0; j < rb; j++) buf[r * rb + j] = row_byte(base + r, j);
        if (write(fd, buf, (size_t)n * rb) != (ssize_t)(n * rb)) { free(buf); close(fd); return -1; }
    }
    free(buf);
    if (fsync(fd) != 0) { perror("fsync"); close(fd); return -1; }
    return fd;
}

static int verify(const ds4_ple_params *p, const uint64_t *rows, uint32_t n,
                  const uint8_t *out, const char *what) {
    const uint32_t rb = ds4_ple_row_bytes(p);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < rb; j++) {
            const uint8_t want = row_byte(rows[i], j);
            if (out[(size_t)i * rb + j] != want) {
                printf("  FAIL %s: entry %u (row %llu) byte %u: got %u, want %u\n",
                       what, i, (unsigned long long)rows[i], j,
                       out[(size_t)i * rb + j], want);
                g_fail++;
                return 1;
            }
        }
    }
    return 0;
}

/* One batch shape through one backend, checked byte for byte. */
static void run_shape(const char *backend, int fd, const ds4_ple_params *p,
                      const uint64_t *rows, uint32_t n, uint64_t cache_bytes,
                      int fetch_twice, const char *what)
{
    setenv("DS4_PLE_STREAM_IO", backend, 1);
    ds4_ple_stream *s = NULL;
    char err[256] = {0};
    if (ds4_ple_stream_open(&s, p, fd, FILE_OFFSET, TABLE_ROWS, cache_bytes,
                            err, sizeof(err)) != 0) {
        printf("  FAIL %s/%s: open: %s\n", backend, what, err);
        g_fail++;
        return;
    }

    const uint32_t rb = ds4_ple_row_bytes(p);
    uint8_t *out = calloc((size_t)n, rb);
    for (int pass = 0; pass < (fetch_twice ? 2 : 1); pass++) {
        memset(out, 0, (size_t)n * rb);
        if (ds4_ple_stream_fetch(s, rows, n, out, err, sizeof(err)) != 0) {
            printf("  FAIL %s/%s: fetch pass %d: %s\n", backend, what, pass, err);
            g_fail++;
            break;
        }
        verify(p, rows, n, out, what);
    }

    /* A silent fallback would make every check above pass on the pool, so the
     * backend actually used has to be asserted, not assumed. */
    CHECK(strcmp(ds4_ple_stream_backend(s), backend) == 0,
          "%s/%s: asked for the %s backend, got %s",
          backend, what, backend, ds4_ple_stream_backend(s));

    if (fetch_twice && cache_bytes) {
        ds4_ple_stats st;
        ds4_ple_stream_get_stats(s, &st);
        CHECK(st.hits >= n, "%s/%s: second pass should hit the cache, %llu hits of %llu lookups",
              backend, what, (unsigned long long)st.hits, (unsigned long long)st.lookups);
    }
    free(out);
    ds4_ple_stream_close(s);
}

#if DS4_HAVE_LIBURING
/* Same rows through both backends must produce identical bytes. */
static void run_equivalence(int fd, const ds4_ple_params *p,
                            const uint64_t *rows, uint32_t n, const char *what)
{
    const uint32_t rb = ds4_ple_row_bytes(p);
    uint8_t *got[2] = {0};
    const char *backend[2] = { "pool", "uring" };
    for (int b = 0; b < 2; b++) {
        setenv("DS4_PLE_STREAM_IO", backend[b], 1);
        ds4_ple_stream *s = NULL;
        char err[256] = {0};
        if (ds4_ple_stream_open(&s, p, fd, FILE_OFFSET, TABLE_ROWS, 1u << 20,
                                err, sizeof(err)) != 0) {
            printf("  FAIL equivalence/%s: open %s: %s\n", what, backend[b], err);
            g_fail++;
            return;
        }
        got[b] = calloc((size_t)n, rb);
        if (ds4_ple_stream_fetch(s, rows, n, got[b], err, sizeof(err)) != 0) {
            printf("  FAIL equivalence/%s: fetch %s: %s\n", what, backend[b], err);
            g_fail++;
            ds4_ple_stream_close(s);
            return;
        }
        ds4_ple_stream_close(s);
    }
    CHECK(memcmp(got[0], got[1], (size_t)n * rb) == 0,
          "equivalence/%s: the two backends disagree", what);
    free(got[0]);
    free(got[1]);
}
#endif

int main(void) {
    const ds4_ple_params p = make_params();
    char path[512];
    const int fd = make_table(&p, path, sizeof(path));
    if (fd < 0) { printf("cannot build the test table\n"); return 1; }
    unlink(path);   /* the fd keeps it alive */

    /* A decode step's rows, an MTP verify's rows, a prefill chunk's rows. */
    uint64_t one[1]   = { 7777 };
    uint64_t nine[9];
    uint64_t dup[12];
    uint64_t wide[2048];
    uint64_t tail[3];

    for (uint32_t i = 0; i < 9; i++) nine[i] = (uint64_t)i * 1237u % TABLE_ROWS;
    /* The same row twice in one batch: the second entry must defer to the
     * first's in-flight slot, not read a half-written cell. */
    for (uint32_t i = 0; i < 12; i++) dup[i] = (uint64_t)(i % 3) * 4099u % TABLE_ROWS;
    for (uint32_t i = 0; i < 2048; i++) wide[i] = ((uint64_t)i * 7919u + 13u) % TABLE_ROWS;
    /* The last row of the table ends inside the file's final partial block. */
    tail[0] = TABLE_ROWS - 1; tail[1] = TABLE_ROWS - 2; tail[2] = 0;

    struct { const uint64_t *rows; uint32_t n; int twice; const char *what; } shapes[] = {
        { one,  1,    0, "single row" },
        { nine, 9,    1, "decode step, cached twice" },
        { dup,  12,   0, "repeated rows in one batch" },
        { wide, 2048, 0, "batch wider than the queue" },
        { tail, 3,    0, "last row of the table" },
    };

    /* Without liburing the ring backend is compiled out, so only the pool
     * half of this can run.  It still has to run: the pool is what such a
     * build uses. */
#if DS4_HAVE_LIBURING
    const int n_backends = 2;
#else
    const int n_backends = 1;
    printf("built without liburing: the io_uring backend is compiled out\n");
#endif

    for (int b = 0; b < n_backends; b++) {
        const char *backend = b ? "uring" : "pool";
        printf("%s backend:\n", backend);
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
            run_shape(backend, fd, &p, shapes[i].rows, shapes[i].n, 1u << 20,
                      shapes[i].twice, shapes[i].what);
            printf("  %s\n", shapes[i].what);
        }
        /* No cache: every entry is a fresh read, including the duplicates. */
        run_shape(backend, fd, &p, dup, 12, 0, 0, "repeated rows, cache off");
        printf("  repeated rows, cache off\n");

        /* A queue far narrower than the batch: every slot is reused many
         * times, and the ring's submission queue fills before the batch does. */
        for (int qd = 1; qd <= 3; qd += 2) {
            char qds[16];
            snprintf(qds, sizeof(qds), "%d", qd);
            setenv("DS4_PLE_STREAM_QD", qds, 1);
            run_shape(backend, fd, &p, wide, 2048, 1u << 20, 0, "narrow queue");
            unsetenv("DS4_PLE_STREAM_QD");
            printf("  narrow queue, depth %d\n", qd);
        }
    }

#if !DS4_HAVE_LIBURING
    /* Asking for a backend this build does not have must be refused, not
     * quietly served by the pool: a silent downgrade is how a CPU regression
     * goes unnoticed. */
    setenv("DS4_PLE_STREAM_IO", "uring", 1);
    ds4_ple_stream *refused = NULL;
    char rerr[256] = {0};
    CHECK(ds4_ple_stream_open(&refused, &p, fd, FILE_OFFSET, TABLE_ROWS,
                              1u << 20, rerr, sizeof(rerr)) != 0,
          "DS4_PLE_STREAM_IO=uring should fail on a build without liburing");
    CHECK(strstr(rerr, "liburing") != NULL,
          "the refusal should name liburing, said: %s", rerr);
    if (refused) ds4_ple_stream_close(refused);
    unsetenv("DS4_PLE_STREAM_IO");
    printf("uring requested on a pool-only build: refused\n");
#endif

#if DS4_HAVE_LIBURING
    printf("cross-backend equivalence:\n");
    run_equivalence(fd, &p, nine, 9, "decode step");
    run_equivalence(fd, &p, wide, 2048, "prefill chunk");
    run_equivalence(fd, &p, tail, 3, "table tail");
    printf("  identical bytes\n");
#endif

    close(fd);
    if (g_fail) { printf("\n%d check(s) failed\n", g_fail); return 1; }
    printf("\nall PLE I/O backend checks passed\n");
    return 0;
}
