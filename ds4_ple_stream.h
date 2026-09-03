#ifndef DS4_PLE_STREAM_H
#define DS4_PLE_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* qwen4exp PLE n-gram hash embeddings.
 *
 * The table is 320,001,536 rows of 160 IQ4_NL weights -- 26.8 GiB, far too much
 * to keep resident next to the 77 GiB of model weights.  It is also the one
 * tensor in the model that is *not* streamed in bulk: each token needs
 * n_heads (16) rows chosen by a hash of its trailing n-gram, so the access
 * pattern is a scattered gather of 90-byte rows over the whole table, and a
 * decode step touches ~1.4 KB of it.
 *
 * That is why this is its own reader rather than an extension of the routed
 * expert cache: that cache is keyed [layer][expert] over multi-megabyte slices,
 * which is the wrong index and the wrong granularity here.
 *
 * The rows for a step are known before the step's first layer runs, so callers
 * should start the gather early and only wait just before the PLE layer.
 *
 * There are two I/O backends behind the same fetch call.  io_uring is the
 * default where liburing was available at build time; otherwise, or with
 * DS4_PLE_STREAM_IO=pool, a pool of threads doing blocking preads.  Both reach
 * the device's random-read ceiling, but the ring reaches it from one thread at
 * roughly a quarter of the CPU, which is what matters on a machine that runs
 * other work beside the model.  Knobs: DS4_PLE_STREAM_IO (uring|pool, where
 * "uring" refuses to fall back silently), DS4_PLE_STREAM_QD (reads in flight,
 * default 256) and DS4_PLE_STREAM_WORKERS (pool threads, default 64). */

enum {
    DS4_PLE_MAX_NGRAM = 4,
    DS4_PLE_MAX_HEADS = 32,
    /* IQ4_NL packs 32 weights into an 18-byte block. */
    DS4_PLE_BLOCK_ELEMS = 32,
    DS4_PLE_BLOCK_BYTES = 18,
    /* The EXL3 checkpoint's row codec: an fp16 row scale, then the 160
     * weights as a 6-bit tail-biting trellis ring (exllamav3's
     * exl3_ngram_trellis v1, mul1 codebook), 122 bytes a row, plus a
     * per-head bias vector added after decoding. */
    DS4_PLE_EXL3_BITS = 6,
    DS4_PLE_EXL3_ROW_BYTES = 2 + 160 * DS4_PLE_EXL3_BITS / 8,
};

enum {
    DS4_PLE_ROW_IQ4_NL = 0,
    DS4_PLE_ROW_EXL3_K6 = 1,
};

typedef struct {
    uint32_t row_type;         /* DS4_PLE_ROW_*: how a row is stored */
    const uint16_t *head_bias; /* EXL3 rows: fp16 [n_heads][head_dim], NULL otherwise */
    uint32_t ngram_size;       /* trailing tokens the hash reads, 3 */
    uint32_t heads_per_ngram;  /* rows per n-gram order, 8 */
    uint32_t n_heads;          /* (ngram_size - 1) * heads_per_ngram, 16 */
    uint32_t head_dim;         /* weights per row, 160 */
    int      eos_token;        /* PLE's own EOS, 248044, not the chat EOS */
    /* Straight from GGUF metadata.  The reference derives them from splitmix64
     * and a prime search; reproducing that bit-exactly would be risk with no
     * upside, so they are always loaded. */
    uint64_t multipliers[DS4_PLE_MAX_NGRAM];
    uint64_t head_offsets[DS4_PLE_MAX_HEADS];
    uint64_t head_vocab[DS4_PLE_MAX_HEADS];
} ds4_ple_params;

typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t reads;        /* pread calls after coalescing */
    uint64_t read_bytes;   /* bytes actually asked of the device */
    uint64_t evictions;
    double   read_seconds;
} ds4_ple_stats;

typedef struct ds4_ple_stream ds4_ple_stream;

/* Bytes one row occupies in the file. */
uint32_t ds4_ple_row_bytes(const ds4_ple_params *p);

/* Row indices for positions [pos0, pos0 + n) of `tokens`, which must hold the
 * whole prefix up to pos0 + n.  Writes n * p->n_heads entries, token-major.
 * Pure and allocation-free so it can be unit tested against the reference. */
void ds4_ple_row_ids(const ds4_ple_params *p, const int *tokens,
                     uint32_t pos0, uint32_t n, uint64_t *out_rows);

/* Decode one row into head_dim floats.  `head` is the hash head the row
 * belongs to (row index % n_heads); the EXL3 codec adds that head's bias. */
void ds4_ple_dequant_row(const ds4_ple_params *p, const uint8_t *src, uint32_t head, float *dst);

/* `fd` stays owned by the caller and must outlive the stream.  A cache_bytes of
 * 0 disables caching and reads every row straight through. */
int ds4_ple_stream_open(ds4_ple_stream **out,
                        const ds4_ple_params *params,
                        int fd, uint64_t file_offset, uint64_t n_rows,
                        uint64_t cache_bytes,
                        char *err, size_t errlen);
void ds4_ple_stream_close(ds4_ple_stream *s);

/* Copy the raw quant bytes of `n` rows into out[n * row_bytes].  Thread-safe;
 * fetches are serialized. */
int ds4_ple_stream_fetch(ds4_ple_stream *s, const uint64_t *rows, uint32_t n,
                         uint8_t *out, char *err, size_t errlen);

/* Warm the cache with `rows` on a background thread, so a later fetch of them
 * hits.  Returns without waiting; a prefetch still running is finished first.
 * A no-op without a cache.  `rows` is copied. */
int ds4_ple_stream_prefetch(ds4_ple_stream *s, const uint64_t *rows, uint32_t n);

/* Which I/O backend the last fetch actually used: "uring" or "pool".  The
 * choice can fall back at runtime, so callers and tests must ask rather than
 * assume the one they requested. */
const char *ds4_ple_stream_backend(const ds4_ple_stream *s);

void ds4_ple_stream_get_stats(const ds4_ple_stream *s, ds4_ple_stats *out);
uint64_t ds4_ple_stream_cache_rows(const ds4_ple_stream *s);

#endif
