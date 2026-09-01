#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_ple_stream.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

/* ---------------------------------------------------------------------------
 * Hash and decode.  Both are exact ports of the reference and are verified
 * against llama.cpp's ple_embd tensor by tests/test_qwen4exp_ple.c.
 * ------------------------------------------------------------------------ */

uint32_t ds4_ple_row_bytes(const ds4_ple_params *p) {
    return (p->head_dim / DS4_PLE_BLOCK_ELEMS) * DS4_PLE_BLOCK_BYTES;
}

void ds4_ple_row_ids(const ds4_ple_params *p, const int *tokens,
                     uint32_t pos0, uint32_t n, uint64_t *out_rows) {
    const uint64_t eos = (uint64_t)p->eos_token;

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t pos = pos0 + i;

        /* The window is truncated by an EOS anywhere in it, transitively: once
         * cut, every older slot reads as EOS too.  A token that is itself EOS
         * does not cut its own context, which is why the scan starts at s = 1.
         * Missing predecessors before the sequence start read as EOS as well. */
        uint64_t ctx[DS4_PLE_MAX_NGRAM];
        ctx[0] = (uint64_t)tokens[pos];
        bool cut = false;
        for (uint32_t s = 1; s < p->ngram_size; s++) {
            const bool have = !cut && pos >= s;
            const uint64_t t = have ? (uint64_t)tokens[pos - s] : eos;
            cut = cut || !have || t == eos;
            ctx[s] = cut ? eos : t;
        }

        /* One hash per n-gram order; the heads of an order share it and differ
         * only by their prime modulus and row offset.  Everything is unsigned
         * 64-bit and wraps, which is what makes the modulo non-negative without
         * the reference's explicit remainder fixup. */
        uint64_t *dst = out_rows + (uint64_t)i * p->n_heads;
        for (uint32_t order = 2; order <= p->ngram_size; order++) {
            uint64_t mixed = ctx[0] * p->multipliers[0];
            for (uint32_t j = 1; j < order; j++) {
                mixed ^= ctx[j] * p->multipliers[j];
            }
            const uint32_t base = (order - 2) * p->heads_per_ngram;
            for (uint32_t g = 0; g < p->heads_per_ngram; g++) {
                const uint32_t h = base + g;
                dst[h] = mixed % p->head_vocab[h] + p->head_offsets[h];
            }
        }
    }
}

static const int8_t DS4_PLE_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static float ple_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Subnormal: renormalize into a float32 exponent. */
            uint32_t e = 0, m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3FFu;
            bits = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

void ds4_ple_dequant_row(const ds4_ple_params *p, const uint8_t *src, float *dst) {
    const uint32_t blocks = p->head_dim / DS4_PLE_BLOCK_ELEMS;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = src + (size_t)b * DS4_PLE_BLOCK_BYTES;
        uint16_t raw;
        memcpy(&raw, blk, sizeof(raw));
        const float d = ple_half_to_float(raw);
        float *out = dst + (size_t)b * DS4_PLE_BLOCK_ELEMS;
        for (uint32_t j = 0; j < DS4_PLE_BLOCK_ELEMS / 2; j++) {
            const uint8_t q = blk[2 + j];
            out[j]      = d * (float)DS4_PLE_IQ4NL[q & 0x0Fu];
            out[j + 16] = d * (float)DS4_PLE_IQ4NL[q >> 4];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Row cache.
 *
 * Rows are Zipf distributed over natural text, so a cache well short of the
 * full table still absorbs most lookups.  The index is open addressed and the
 * replacement policy is CLOCK: a second-chance sweep costs one bit per slot and
 * avoids the O(all slots) victim scans the expert cache pays for.
 * ------------------------------------------------------------------------ */

#define PLE_EMPTY UINT32_MAX

/* One miss is a 90-byte random read, so the table is latency bound, not
 * bandwidth bound: a serial loop over a prefill chunk's ~16k misses runs the
 * NVMe at queue depth 1 and costs ~35 us each.  Workers exist purely to keep
 * several reads in flight; each carries its own O_DIRECT bounce buffer and its
 * own stats, which are merged into the shared counters after the join. */
typedef struct ple_worker {
    struct ds4_ple_stream *s;
    pthread_t th;
    uint32_t  index;      /* stride offset into the miss list */
    uint64_t  epoch;      /* last batch this worker ran */
    uint8_t  *bounce;
    size_t    bounce_cap;
    uint64_t  reads;
    uint64_t  read_bytes;
    double    read_seconds;
    char      err[256];
    int       rc;
} ple_worker;

struct ds4_ple_stream {
    ds4_ple_params params;
    int      fd;
    uint64_t file_offset;
    uint64_t n_rows;
    uint32_t row_bytes;

    /* Slot storage. */
    uint8_t  *arena;
    uint64_t *slot_row;   /* row id resident in each slot */
    uint8_t  *slot_ref;   /* CLOCK second-chance bit */
    uint32_t  n_slots;
    uint32_t  next_free;  /* slots handed out before the first eviction */
    uint32_t  hand;

    /* Open-addressed row id -> slot index, power-of-two sized. */
    uint32_t *index;
    uint32_t  index_mask;

    /* Slots holding a read that this batch has issued but not yet completed.
     * They are already in the index, so a repeated row inside one batch finds
     * them, and must defer its copy instead of reading the stale cell. */
    uint8_t  *slot_pending;

    /* Miss batch scratch, sized for the largest fetch seen.  `pend_slot` is
     * per fetch entry and holds PLE_EMPTY once the entry is already satisfied. */
    uint64_t  *miss_row;
    uint8_t  **miss_dst;
    uint32_t  *pend_slot;
    uint32_t   miss_cap;

    /* Miss read pool.  Idle workers block on cv_work; a batch bumps `epoch`
     * and broadcasts, and the submitting thread runs stride 0 itself so a
     * small batch never pays a wakeup. */
    ple_worker      *workers;
    uint32_t         n_workers;   /* including the submitting thread */
    pthread_mutex_t  mu;
    pthread_cond_t   cv_work;
    pthread_cond_t   cv_done;
    uint64_t         epoch;
    uint32_t         batch_n;
    uint32_t         batch_left;
    bool             pool_ready;
    bool             shutdown;

    /* Direct reads keep the 26.8 GiB table out of the page cache, which would
     * otherwise compete with the resident weights for the same memory. */
    bool     direct;
    uint64_t direct_align;

    ds4_ple_stats stats;
};

static void ple_pool_stop(ds4_ple_stream *s);

static double ple_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void ple_fail(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static uint64_t ple_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static uint32_t ple_index_find(const ds4_ple_stream *s, uint64_t row) {
    uint32_t i = (uint32_t)(ple_mix64(row) & s->index_mask);
    for (;;) {
        const uint32_t slot = s->index[i];
        if (slot == PLE_EMPTY) return PLE_EMPTY;
        if (s->slot_row[slot] == row) return slot;
        i = (i + 1u) & s->index_mask;
    }
}

static void ple_index_insert(ds4_ple_stream *s, uint64_t row, uint32_t slot) {
    uint32_t i = (uint32_t)(ple_mix64(row) & s->index_mask);
    while (s->index[i] != PLE_EMPTY) i = (i + 1u) & s->index_mask;
    s->index[i] = slot;
}

/* Backward-shift deletion: linear probing cannot tombstone without eventually
 * filling the table with tombstones, and this cache evicts continuously. */
static void ple_index_remove(ds4_ple_stream *s, uint64_t row) {
    uint32_t i = (uint32_t)(ple_mix64(row) & s->index_mask);
    for (;;) {
        const uint32_t slot = s->index[i];
        if (slot == PLE_EMPTY) return;
        if (s->slot_row[slot] == row) break;
        i = (i + 1u) & s->index_mask;
    }
    s->index[i] = PLE_EMPTY;
    uint32_t j = (i + 1u) & s->index_mask;
    while (s->index[j] != PLE_EMPTY) {
        const uint32_t moved = s->index[j];
        s->index[j] = PLE_EMPTY;
        ple_index_insert(s, s->slot_row[moved], moved);
        j = (j + 1u) & s->index_mask;
    }
}

static uint32_t ple_claim_slot(ds4_ple_stream *s) {
    if (s->next_free < s->n_slots) return s->next_free++;
    for (;;) {
        const uint32_t slot = s->hand;
        s->hand = (s->hand + 1u) % s->n_slots;
        /* A slot claimed earlier in this same batch has a read in flight into
         * it; handing it out twice would make both entries wrong.  A batch is
         * at most a few thousand rows against millions of slots, so the hand
         * only ever steps over a handful of these. */
        if (s->slot_pending && s->slot_pending[slot]) continue;
        if (s->slot_ref[slot]) {
            s->slot_ref[slot] = 0;
            continue;
        }
        ple_index_remove(s, s->slot_row[slot]);
        s->stats.evictions++;
        return slot;
    }
}

/* ------------------------------------------------------------------------- */

int ds4_ple_stream_open(ds4_ple_stream **out,
                        const ds4_ple_params *params,
                        int fd, uint64_t file_offset, uint64_t n_rows,
                        uint64_t cache_bytes,
                        char *err, size_t errlen) {
    if (!out || !params) return 1;
    *out = NULL;

    if (params->ngram_size < 2 || params->ngram_size > DS4_PLE_MAX_NGRAM) {
        ple_fail(err, errlen, "PLE n-gram size %u is out of range", params->ngram_size);
        return 1;
    }
    if (params->n_heads == 0 || params->n_heads > DS4_PLE_MAX_HEADS) {
        ple_fail(err, errlen, "PLE head count %u is out of range", params->n_heads);
        return 1;
    }
    if (params->head_dim == 0 || (params->head_dim % DS4_PLE_BLOCK_ELEMS) != 0) {
        ple_fail(err, errlen, "PLE head dim %u is not a multiple of %d",
                 params->head_dim, DS4_PLE_BLOCK_ELEMS);
        return 1;
    }

    ds4_ple_stream *s = calloc(1, sizeof(*s));
    if (!s) { ple_fail(err, errlen, "out of memory"); return 1; }
    s->params = *params;
    s->fd = fd;
    s->file_offset = file_offset;
    s->n_rows = n_rows;
    s->row_bytes = ds4_ple_row_bytes(params);

    /* Per-slot overhead: the row id, the CLOCK bit, and the index entry, whose
     * table is kept at most half full so probe chains stay short. */
    const uint64_t per_slot = (uint64_t)s->row_bytes + sizeof(uint64_t) + 1u + 2u * sizeof(uint32_t);
    uint64_t slots = cache_bytes / per_slot;
    if (slots > n_rows) slots = n_rows;
    if (slots > UINT32_MAX / 4u) slots = UINT32_MAX / 4u;

    if (slots > 0) {
        s->n_slots = (uint32_t)slots;
        uint32_t cap = 1u;
        while (cap < s->n_slots * 2u) cap <<= 1;
        s->index_mask = cap - 1u;

        s->arena        = malloc((size_t)s->n_slots * s->row_bytes);
        s->slot_row     = malloc((size_t)s->n_slots * sizeof(*s->slot_row));
        s->slot_ref     = calloc(s->n_slots, 1);
        s->slot_pending = calloc(s->n_slots, 1);
        s->index        = malloc((size_t)cap * sizeof(*s->index));
        if (!s->arena || !s->slot_row || !s->slot_ref || !s->slot_pending || !s->index) {
            ple_fail(err, errlen, "cannot allocate a %llu MiB PLE row cache",
                     (unsigned long long)(cache_bytes >> 20));
            ds4_ple_stream_close(s);
            return 1;
        }
        memset(s->index, 0xFF, (size_t)cap * sizeof(*s->index));
    }

    /* O_DIRECT needs the offset, length and buffer aligned to the logical block
     * size; rows are 90 bytes at arbitrary offsets, so reads go through a
     * bounce buffer and the payload is sliced back out. */
#ifdef O_DIRECT
    char proc[64];
    snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
    const int dfd = open(proc, O_RDONLY | O_DIRECT);
    if (dfd >= 0) {
        s->fd = dfd;
        s->direct = true;
        s->direct_align = 4096;
    }
#endif

    *out = s;
    return 0;
}

void ds4_ple_stream_close(ds4_ple_stream *s) {
    if (!s) return;
    ple_pool_stop(s);
    if (s->direct && s->fd >= 0) close(s->fd);
    free(s->arena);
    free(s->slot_row);
    free(s->slot_ref);
    free(s->slot_pending);
    free(s->index);
    free(s->miss_row);
    free(s->miss_dst);
    free(s->pend_slot);
    free(s);
}

/* Reads one row through `w`'s private bounce buffer and charges the time to
 * `w`, so several of these can run concurrently on the pool. */
static int ple_read_row(ds4_ple_stream *s, ple_worker *w, uint64_t row,
                        uint8_t *dst, char *err, size_t errlen) {
    const uint64_t off = s->file_offset + row * s->row_bytes;
    const double t0 = ple_now();

    uint64_t start = off;
    size_t len = s->row_bytes;
    size_t skew = 0;
    if (s->direct) {
        const uint64_t a = s->direct_align;
        start = off / a * a;
        skew = (size_t)(off - start);
        len = (size_t)(((skew + s->row_bytes) + a - 1) / a * a);
        if (w->bounce_cap < len) {
            free(w->bounce);
            w->bounce = NULL;
            if (posix_memalign((void **)&w->bounce, (size_t)a, len) != 0) {
                w->bounce = NULL;
                w->bounce_cap = 0;
                ple_fail(err, errlen, "cannot allocate a PLE bounce buffer");
                return 1;
            }
            w->bounce_cap = len;
        }
    }

    uint8_t *into = s->direct ? w->bounce : dst;
    size_t done = 0;
    while (done < len) {
        const ssize_t got = pread(s->fd, into + done, len - done, (off_t)(start + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            ple_fail(err, errlen, "PLE row read failed: %s", strerror(errno));
            return 1;
        }
        if (got == 0) {
            /* The final row can sit inside the last partial block of the file. */
            if (!s->direct || done + skew >= s->row_bytes) break;
            ple_fail(err, errlen, "PLE row read hit end of file");
            return 1;
        }
        done += (size_t)got;
    }
    if (s->direct) memcpy(dst, w->bounce + skew, s->row_bytes);

    w->reads++;
    w->read_bytes += len;
    w->read_seconds += ple_now() - t0;
    return 0;
}

/* Runs this worker's stride of the current batch. */
static void ple_run_stride(ds4_ple_stream *s, ple_worker *w) {
    w->rc = 0;
    w->err[0] = '\0';
    for (uint32_t i = w->index; i < s->batch_n; i += s->n_workers) {
        if (ple_read_row(s, w, s->miss_row[i], s->miss_dst[i],
                         w->err, sizeof(w->err)) != 0) {
            w->rc = 1;
            return;
        }
    }
}

static void *ple_worker_main(void *arg) {
    ple_worker *w = (ple_worker *)arg;
    ds4_ple_stream *s = w->s;

    pthread_mutex_lock(&s->mu);
    for (;;) {
        while (!s->shutdown && s->epoch == w->epoch) {
            pthread_cond_wait(&s->cv_work, &s->mu);
        }
        if (s->shutdown) break;
        w->epoch = s->epoch;
        pthread_mutex_unlock(&s->mu);

        ple_run_stride(s, w);

        pthread_mutex_lock(&s->mu);
        if (--s->batch_left == 0) pthread_cond_signal(&s->cv_done);
    }
    pthread_mutex_unlock(&s->mu);
    return NULL;
}

/* Below this many misses the wakeup round trip costs more than the reads it
 * would overlap, so the submitting thread just does them all. */
#define PLE_POOL_MIN_MISSES 8

static uint32_t ple_worker_count(void) {
    const char *env = getenv("DS4_PLE_STREAM_WORKERS");
    if (env && *env) {
        const long v = strtol(env, NULL, 10);
        if (v >= 1 && v <= 256) return (uint32_t)v;
    }
    return 16;
}

/* Lazily started so a run that never misses pays nothing.  Failing to start a
 * thread is not fatal: the pool simply ends up narrower, down to the
 * submitting thread alone. */
static void ple_pool_start(ds4_ple_stream *s) {
    if (s->pool_ready) return;
    s->pool_ready = true;

    const uint32_t want = ple_worker_count();
    s->workers = calloc(want, sizeof(*s->workers));
    if (!s->workers) { s->n_workers = 0; return; }

    /* Slot 0 is the submitting thread's record: it owns a bounce buffer and
     * stats like the others, but no pthread. */
    for (uint32_t i = 0; i < want; i++) {
        s->workers[i].s = s;
        s->workers[i].index = i;
    }
    s->n_workers = 1;

    if (pthread_mutex_init(&s->mu, NULL) != 0) return;
    if (pthread_cond_init(&s->cv_work, NULL) != 0) return;
    if (pthread_cond_init(&s->cv_done, NULL) != 0) return;

    for (uint32_t i = 1; i < want; i++) {
        if (pthread_create(&s->workers[i].th, NULL, ple_worker_main, &s->workers[i]) != 0) break;
        s->n_workers = i + 1;
    }
}

static void ple_pool_stop(ds4_ple_stream *s) {
    if (!s->pool_ready || !s->workers) return;
    if (s->n_workers > 1) {
        pthread_mutex_lock(&s->mu);
        s->shutdown = true;
        pthread_cond_broadcast(&s->cv_work);
        pthread_mutex_unlock(&s->mu);
        for (uint32_t i = 1; i < s->n_workers; i++) pthread_join(s->workers[i].th, NULL);
        pthread_cond_destroy(&s->cv_done);
        pthread_cond_destroy(&s->cv_work);
        pthread_mutex_destroy(&s->mu);
    }
    for (uint32_t i = 0; i < s->n_workers; i++) free(s->workers[i].bounce);
    free(s->workers);
    s->workers = NULL;
    s->n_workers = 0;
}

/* Issues `s->batch_n` reads across the pool and waits for all of them. */
static int ple_run_batch(ds4_ple_stream *s, char *err, size_t errlen) {
    const uint32_t n = s->batch_n;
    if (n == 0) return 0;

    ple_pool_start(s);
    if (s->n_workers == 0) {
        ple_fail(err, errlen, "cannot allocate the PLE read pool");
        return 1;
    }

    const bool parallel = s->n_workers > 1 && n >= PLE_POOL_MIN_MISSES;
    const uint32_t saved = s->n_workers;
    if (!parallel) s->n_workers = 1;   /* stride 1: worker 0 takes everything */

    if (parallel) {
        pthread_mutex_lock(&s->mu);
        s->batch_left = s->n_workers - 1;   /* worker 0 is this thread */
        s->epoch++;
        pthread_cond_broadcast(&s->cv_work);
        pthread_mutex_unlock(&s->mu);
    }

    ple_run_stride(s, &s->workers[0]);

    if (parallel) {
        pthread_mutex_lock(&s->mu);
        while (s->batch_left != 0) pthread_cond_wait(&s->cv_done, &s->mu);
        pthread_mutex_unlock(&s->mu);
    }
    s->n_workers = saved;

    /* Merge the per-worker stats and surface the first error. */
    int rc = 0;
    for (uint32_t i = 0; i < s->n_workers; i++) {
        ple_worker *w = &s->workers[i];
        s->stats.reads        += w->reads;
        s->stats.read_bytes   += w->read_bytes;
        s->stats.read_seconds += w->read_seconds;
        w->reads = w->read_bytes = 0;
        w->read_seconds = 0.0;
        if (w->rc && !rc) { rc = 1; ple_fail(err, errlen, "%s", w->err); }
        w->rc = 0;
    }
    return rc;
}

int ds4_ple_stream_fetch(ds4_ple_stream *s, const uint64_t *rows, uint32_t n,
                         uint8_t *out, char *err, size_t errlen) {
    if (!s || (!rows && n)) return 1;
    if (n == 0) return 0;

    if (s->miss_cap < n) {
        uint64_t  *mr = realloc(s->miss_row,  (size_t)n * sizeof(*mr));
        uint8_t  **md = realloc(s->miss_dst,  (size_t)n * sizeof(*md));
        uint32_t  *ps = realloc(s->pend_slot, (size_t)n * sizeof(*ps));
        if (mr) s->miss_row  = mr;
        if (md) s->miss_dst  = md;
        if (ps) s->pend_slot = ps;
        if (!mr || !md || !ps) {
            ple_fail(err, errlen, "cannot size the PLE miss batch to %u rows", n);
            return 1;
        }
        s->miss_cap = n;
    }

    /* Pass 1: serve every hit, and claim a slot and queue a read for every
     * miss.  The slot goes into the index straight away so the rest of the
     * batch coalesces onto it rather than queueing the same row twice. */
    uint32_t n_miss = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t row = rows[i];
        if (row >= s->n_rows) {
            ple_fail(err, errlen, "PLE row %llu is outside the table",
                     (unsigned long long)row);
            s->batch_n = 0;
            goto unwind;
        }
        uint8_t *dst = out + (size_t)i * s->row_bytes;
        s->stats.lookups++;
        s->pend_slot[i] = PLE_EMPTY;

        if (s->n_slots == 0) {
            s->stats.misses++;
            s->miss_row[n_miss] = row;
            s->miss_dst[n_miss] = dst;      /* uncached: read straight out */
            n_miss++;
            continue;
        }

        const uint32_t slot = ple_index_find(s, row);
        if (slot != PLE_EMPTY) {
            s->slot_ref[slot] = 1;
            if (s->slot_pending[slot]) {
                s->pend_slot[i] = slot;     /* queued earlier in this batch */
            } else {
                memcpy(dst, s->arena + (size_t)slot * s->row_bytes, s->row_bytes);
            }
            s->stats.hits++;
            continue;
        }

        s->stats.misses++;
        const uint32_t victim = ple_claim_slot(s);
        s->slot_row[victim] = row;
        s->slot_ref[victim] = 1;
        s->slot_pending[victim] = 1;
        ple_index_insert(s, row, victim);
        s->miss_row[n_miss] = row;
        s->miss_dst[n_miss] = s->arena + (size_t)victim * s->row_bytes;
        n_miss++;
        s->pend_slot[i] = victim;
    }

    s->batch_n = n_miss;
    if (ple_run_batch(s, err, errlen) != 0) goto unwind;

    /* Pass 3: the reads have landed, so the deferred copies can be served. */
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = s->pend_slot[i];
        if (slot == PLE_EMPTY) continue;
        memcpy(out + (size_t)i * s->row_bytes,
               s->arena + (size_t)slot * s->row_bytes, s->row_bytes);
    }
    for (uint32_t j = 0; j < n_miss; j++) {
        if (s->n_slots) s->slot_pending[(size_t)(s->miss_dst[j] - s->arena) / s->row_bytes] = 0;
    }
    return 0;

unwind:
    /* A failed batch leaves claimed slots holding nothing.  Drop them so a
     * later fetch re-reads the row instead of trusting the empty cell. */
    for (uint32_t j = 0; j < n_miss && s->n_slots; j++) {
        const uint32_t slot = (uint32_t)((size_t)(s->miss_dst[j] - s->arena) / s->row_bytes);
        s->slot_pending[slot] = 0;
        ple_index_remove(s, s->slot_row[slot]);
    }
    return 1;
}

void ds4_ple_stream_get_stats(const ds4_ple_stream *s, ds4_ple_stats *out) {
    if (!s || !out) return;
    *out = s->stats;
}

uint64_t ds4_ple_stream_cache_rows(const ds4_ple_stream *s) {
    return s ? s->n_slots : 0;
}
