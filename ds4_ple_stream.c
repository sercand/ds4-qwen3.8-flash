#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_ple_stream.h"

#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#if DS4_HAVE_LIBURING
#include <liburing.h>
#endif

/* ---------------------------------------------------------------------------
 * Hash and decode.  Both are exact ports of the reference and are verified
 * against llama.cpp's ple_embd tensor by tests/test_qwen4exp_ple.c.
 * ------------------------------------------------------------------------ */

uint32_t ds4_ple_row_bytes(const ds4_ple_params *p) {
    if (p->row_type == DS4_PLE_ROW_EXL3_K6) {
        return 2u + p->head_dim * DS4_PLE_EXL3_BITS / 8u;
    }
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

static uint16_t ple_float_to_half(float f) {
    /* Round-to-nearest-even f32 -> f16, subnormals included: the decoded
     * rows hold values below 2^-14, and __float2half_rn keeps them. */
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t absb = bits & 0x7FFFFFFFu;
    if (absb >= 0x7F800000u) return (uint16_t)(sign | 0x7C00u | ((absb & 0x7FFFFFu) ? 0x200u : 0u));
    if (absb >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);
    if (absb < 0x33000000u) return (uint16_t)sign;
    const uint32_t exp = absb >> 23;
    const uint32_t mant = absb & 0x7FFFFFu;
    if (absb < 0x38800000u) {
        /* Subnormal: the significand with its hidden bit, scaled to 2^-24 units. */
        const uint32_t m = mant | 0x800000u;
        const uint32_t shift = 126u - exp;
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (hm & 1u))) hm++;
        return (uint16_t)(sign | hm);
    }
    uint32_t h = sign | ((exp - 112u) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

/* The EXL3 row: bit m of weight i sits at ring position ((i - m/K) mod 160)*K
 * + m%K, a 6-bit tail-biting trellis; the 16-bit state decodes through the
 * mul1 codebook (x * 0x83DCD12D, byte sum + 1024 read as fp16, affine), then
 * the row scale and the head bias.  Port of exllamav3's ngram_dequant_kernel,
 * with the same fp16 rounding of the codebook value. */
static void ple_dequant_row_exl3(const ds4_ple_params *p, const uint8_t *src,
                                 uint32_t head, float *dst) {
    const uint32_t K = DS4_PLE_EXL3_BITS;
    const uint32_t n = p->head_dim;
    uint16_t words[1 + 160 * DS4_PLE_EXL3_BITS / 16];
    memcpy(words, src, 2u + n * K / 8u);
    const float scale = ple_half_to_float(words[0]);
    const float k_inv = ple_half_to_float(0x1eee);
    const float k_bias = ple_half_to_float(0xc931);
    const uint16_t *bias = p->head_bias ? p->head_bias + (size_t)head * n : NULL;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t state = 0;
        for (uint32_t m = 0; m < 16; m++) {
            int32_t pos = (int32_t)i - (int32_t)(m / K);
            if (pos < 0) pos += (int32_t)n;
            const uint32_t sb = (uint32_t)pos * K + m % K;
            state |= (uint32_t)((words[1 + (sb >> 4)] >> (sb & 15)) & 1u) << m;
        }
        const uint32_t prod = state * 0x83DCD12Du;
        const float h = 1024.0f + (float)((prod & 0xFFu) + ((prod >> 8) & 0xFFu) +
                                          ((prod >> 16) & 0xFFu) + ((prod >> 24) & 0xFFu));
        const float cb = ple_half_to_float(ple_float_to_half(h * k_inv + k_bias));
        const float b = bias ? ple_half_to_float(bias[i]) : 0.0f;
        /* One fused multiply-add, as nvcc contracts the kernel's expression. */
        dst[i] = ple_half_to_float(ple_float_to_half(fmaf(cb, scale, b)));
    }
}

void ds4_ple_dequant_row(const ds4_ple_params *p, const uint8_t *src, uint32_t head, float *dst) {
    if (p->row_type == DS4_PLE_ROW_EXL3_K6) {
        ple_dequant_row_exl3(p, src, head, dst);
        return;
    }
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
    uint8_t  *bounce;
    size_t    bounce_cap;
    uint64_t  reads;
    uint64_t  read_bytes;
    double    read_seconds;
    char      err[256];
    int       rc;
} ple_worker;

/* One outstanding ring read.  A read can come back short of the block it asked
 * for, so a slot tracks how much of its row has actually landed. */
struct ple_ring_slot {
    uint32_t miss;    /* which entry of the miss list this serves */
    uint64_t start;   /* aligned file offset being read */
    size_t   skew;    /* where the row begins inside the block */
    size_t   len;     /* bytes the read asked for */
    size_t   done;    /* bytes received so far */
};

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

    /* Miss read pool.  A batch posts the semaphore once per worker it wants
     * -- as many as there are misses, so a decode step's handful of rows
     * wakes a handful of threads, not all of them -- and every reader,
     * the submitting thread included, takes miss indices from `next` until
     * the list is drained.  Cold rows cost one device round trip each, so
     * what matters is how many are in flight at once. */
    ple_worker      *workers;
    uint32_t         n_workers;   /* including the submitting thread */
    pthread_mutex_t  mu;
    pthread_cond_t   cv_done;
    sem_t            sem_work;
    atomic_uint      next;
    uint32_t         batch_n;
    uint32_t         batch_left;
    bool             pool_ready;
    bool             shutdown;

    /* Fetches are serialized: the miss batch above is per stream, and the
     * prefetch thread fetches on behalf of a later chunk while the caller may
     * fetch (or another context may decode) at the same time. */
    pthread_mutex_t  fetch_mu;

    /* Prefetch: rows a coming prefill chunk will ask for are fetched into the
     * cache by a thread of their own while the current chunk runs on the GPU,
     * so the real fetch then hits.  The output of that fetch is discarded. */
    pthread_t        pf_th;
    pthread_mutex_t  pf_mu;
    pthread_cond_t   pf_cv;
    uint64_t        *pf_rows;
    uint8_t         *pf_out;
    uint32_t         pf_n, pf_cap;
    bool             pf_ready, pf_busy;

    /* io_uring backend.  The pool above reaches the device's IOPS ceiling only
     * by having a thread blocked per outstanding read: 64 readers cost ~2.6
     * cores of kernel time for a prefill chunk's gather and still sit below
     * the ceiling, and 192 cost 3.5 cores to reach it.  A ring reaches the
     * same ceiling from one thread at ~1 core, because registered buffers skip
     * the per-read page pinning and one enter call submits the whole batch.
     * That CPU is what this is for -- the gather is already off the critical
     * path -- so it matters on a box that also runs CPU-only work.
     *
     * One ring per stream is safe: every fetch is serialized by fetch_mu.  The
     * pool stays as the fallback for when the ring cannot be set up. */
    bool     ring_ok;       /* whether the ring came up */
    uint32_t ring_qd;       /* reads in flight */
    size_t   ring_stride;   /* bytes per slot's bounce buffer */
#if DS4_HAVE_LIBURING
    struct io_uring ring;
    uint8_t        *ring_bounce;   /* ring_qd * ring_stride, registered */
    struct iovec   *ring_iov;
    struct ple_ring_slot *ring_slot;
    uint32_t       *ring_free;     /* stack of idle slot indices */
    uint32_t       *ring_pend;     /* stack of slots waiting for an sqe */
#endif

    /* Direct reads keep the 26.8 GiB table out of the page cache, which would
     * otherwise compete with the resident weights for the same memory. */
    bool     direct;
    uint64_t direct_align;

    ds4_ple_stats stats;
};

static void ple_pool_stop(ds4_ple_stream *s);
static uint32_t ple_worker_count(void);
#if DS4_HAVE_LIBURING
static int  ple_ring_open(ds4_ple_stream *s, char *err, size_t errlen);
static void ple_ring_close(ds4_ple_stream *s);
#endif

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
    if (params->row_type == DS4_PLE_ROW_EXL3_K6 &&
        (params->head_dim != 160 || !params->head_bias)) {
        ple_fail(err, errlen, "PLE EXL3 rows need head_dim 160 and a head bias table");
        return 1;
    }

    ds4_ple_stream *s = calloc(1, sizeof(*s));
    if (!s) { ple_fail(err, errlen, "out of memory"); return 1; }
    if (pthread_mutex_init(&s->fetch_mu, NULL) != 0) {
        ple_fail(err, errlen, "cannot init the PLE fetch lock");
        free(s);
        return 1;
    }
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

    /* Backend choice.  The ring is the default wherever liburing is present,
     * because it costs a quarter of the CPU for the same reads; DS4_PLE_STREAM_IO
     * takes "pool" to force the thread pool back, or "uring" to make a ring that
     * will not start an error instead of a silent downgrade. */
    const char *io = getenv("DS4_PLE_STREAM_IO");
    const bool insist = io && strcmp(io, "uring") == 0;
    const bool want_uring = !(io && strcmp(io, "pool") == 0);
#if DS4_HAVE_LIBURING
    if (want_uring) {
        char rerr[192] = {0};
        if (ple_ring_open(s, rerr, sizeof(rerr)) != 0) {
            if (insist) {
                ple_fail(err, errlen, "PLE io_uring backend unavailable: %s", rerr);
                ds4_ple_stream_close(s);
                return 1;
            }
            fprintf(stderr, "ds4: PLE io_uring backend unavailable (%s); "
                            "falling back to %u reader threads\n",
                    rerr, ple_worker_count());
        }
    }
#else
    (void)want_uring;
    if (insist) {
        ple_fail(err, errlen,
                 "DS4_PLE_STREAM_IO=uring, but this build has no liburing "
                 "(install liburing-dev and rebuild)");
        ds4_ple_stream_close(s);
        return 1;
    }
#endif

    *out = s;
    return 0;
}

void ds4_ple_stream_close(ds4_ple_stream *s) {
    if (!s) return;
#if DS4_HAVE_LIBURING
    ple_ring_close(s);
#endif
    if (s->pf_ready) {
        pthread_mutex_lock(&s->pf_mu);
        s->shutdown = true;
        pthread_cond_broadcast(&s->pf_cv);
        pthread_mutex_unlock(&s->pf_mu);
        pthread_join(s->pf_th, NULL);
        pthread_cond_destroy(&s->pf_cv);
        pthread_mutex_destroy(&s->pf_mu);
        free(s->pf_rows);
        free(s->pf_out);
    }
    ple_pool_stop(s);
    pthread_mutex_destroy(&s->fetch_mu);
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
/* Where a row's read lands: the aligned block to ask the device for, the row's
 * offset inside it, and how many bytes that is.  Both backends go through this
 * so their idea of the file cannot drift apart. */
static void ple_row_geometry(const ds4_ple_stream *s, uint64_t row,
                             uint64_t *start, size_t *skew, size_t *len) {
    const uint64_t off = s->file_offset + row * s->row_bytes;
    if (!s->direct) {
        *start = off;
        *skew  = 0;
        *len   = s->row_bytes;
        return;
    }
    const uint64_t a = s->direct_align;
    *start = off / a * a;
    *skew  = (size_t)(off - *start);
    *len   = (size_t)(((*skew + s->row_bytes) + a - 1) / a * a);
}

#if DS4_HAVE_LIBURING
/* The widest block a single row can need: it may straddle two of them. */
static size_t ple_bounce_stride(const ds4_ple_stream *s) {
    if (!s->direct) return s->row_bytes;
    const uint64_t a = s->direct_align;
    return (size_t)(((a - 1) + s->row_bytes + a - 1) / a * a);
}
#endif

static int ple_read_row(ds4_ple_stream *s, ple_worker *w, uint64_t row,
                        uint8_t *dst, char *err, size_t errlen) {
    const double t0 = ple_now();

    uint64_t start;
    size_t len, skew;
    ple_row_geometry(s, row, &start, &skew, &len);
    if (s->direct) {
        const uint64_t a = s->direct_align;
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
/* Every reader takes the next miss until the list is drained: a slow read
 * holds up one row, not a fixed stride of them. */
static void ple_run_shared(ds4_ple_stream *s, ple_worker *w) {
    w->rc = 0;
    w->err[0] = '\0';
    for (;;) {
        const uint32_t i = atomic_fetch_add(&s->next, 1u);
        if (i >= s->batch_n) return;
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
    for (;;) {
        while (sem_wait(&s->sem_work) != 0) { /* EINTR */ }
        if (s->shutdown) break;
        ple_run_shared(s, w);
        pthread_mutex_lock(&s->mu);
        if (--s->batch_left == 0) pthread_cond_signal(&s->cv_done);
        pthread_mutex_unlock(&s->mu);
    }
    return NULL;
}

/* Below this many misses a wakeup costs more than the read it would overlap,
 * so the submitting thread does them alone. */
#define PLE_POOL_MIN_MISSES 4

static uint32_t ple_worker_count(void) {
    const char *env = getenv("DS4_PLE_STREAM_WORKERS");
    if (env && *env) {
        const long v = strtol(env, NULL, 10);
        if (v >= 1 && v <= 256) return (uint32_t)v;
    }
    /* A 2048-token prefill chunk misses ~20k rows of 4 KiB direct reads.  64
     * in flight is where this stops paying: it takes 92 ms against 223 ms at
     * 16, and the 54 ms that 256 threads reach costs 3.5 cores of kernel time
     * to get.  Past 64 the ring is the way to buy depth -- see ple_ring_batch
     * -- so this stays where the CPU cost is still defensible for a fallback. */
    return 64;
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
    for (uint32_t i = 0; i < want; i++) s->workers[i].s = s;
    s->n_workers = 1;

    if (pthread_mutex_init(&s->mu, NULL) != 0) return;
    if (pthread_cond_init(&s->cv_done, NULL) != 0) return;
    if (sem_init(&s->sem_work, 0, 0) != 0) return;

    for (uint32_t i = 1; i < want; i++) {
        if (pthread_create(&s->workers[i].th, NULL, ple_worker_main, &s->workers[i]) != 0) break;
        s->n_workers = i + 1;
    }
}

static void ple_pool_stop(ds4_ple_stream *s) {
    if (!s->pool_ready || !s->workers) return;
    if (s->n_workers > 1) {
        s->shutdown = true;
        for (uint32_t i = 1; i < s->n_workers; i++) sem_post(&s->sem_work);
        for (uint32_t i = 1; i < s->n_workers; i++) pthread_join(s->workers[i].th, NULL);
        pthread_cond_destroy(&s->cv_done);
        sem_destroy(&s->sem_work);
        pthread_mutex_destroy(&s->mu);
    }
    for (uint32_t i = 0; i < s->n_workers; i++) free(s->workers[i].bounce);
    free(s->workers);
    s->workers = NULL;
    s->n_workers = 0;
}

/* ---------------------------------------------------------------------------
 * io_uring backend.
 *
 * Same contract as the pool -- fill miss_dst[i] with row miss_row[i] -- from a
 * single thread.  The reads are what they always were; what changes is the cost
 * of issuing them.  A registered file and registered bounce buffers mean the
 * kernel neither looks up the descriptor nor pins pages per read, and one
 * enter call carries a whole batch, so the same queue depth costs about a
 * quarter of the CPU that a thread-per-read pool needs for it.
 * ------------------------------------------------------------------------ */

#if DS4_HAVE_LIBURING

static uint32_t ple_ring_depth(void) {
    const char *env = getenv("DS4_PLE_STREAM_QD");
    if (env && *env) {
        const long v = strtol(env, NULL, 10);
        if (v >= 1 && v <= 4096) return (uint32_t)v;
    }
    /* Measured on GB10: the device's random-read ceiling is ~640k IOPS and it
     * needs somewhere above 200 reads in flight to get there.  256 reaches it
     * and costs 2 MiB of bounce buffers at a 4 KiB alignment. */
    return 256;
}

static void ple_ring_close(ds4_ple_stream *s) {
    if (s->ring_ok) io_uring_queue_exit(&s->ring);
    free(s->ring_bounce);
    free(s->ring_iov);
    free(s->ring_slot);
    free(s->ring_free);
    free(s->ring_pend);
    s->ring_bounce = NULL;
    s->ring_iov    = NULL;
    s->ring_slot   = NULL;
    s->ring_free   = NULL;
    s->ring_pend   = NULL;
    s->ring_ok     = false;
}

/* Brings up the ring with its registered fd and buffers.  Every failure path
 * leaves the stream usable on the pool, so the caller can either warn or
 * insist depending on how the backend was chosen. */
static int ple_ring_open(ds4_ple_stream *s, char *err, size_t errlen) {
    const uint32_t qd = ple_ring_depth();
    const size_t stride = ple_bounce_stride(s);
    /* O_DIRECT wants the buffer aligned too; 4 KiB covers every device this
     * runs on and statx reports far less is actually required. */
    const size_t buf_align = s->direct && s->direct_align > 4096
                               ? (size_t)s->direct_align : 4096u;

    int rc = io_uring_queue_init(qd, &s->ring, 0);
    if (rc < 0) {
        ple_fail(err, errlen, "io_uring_queue_init: %s", strerror(-rc));
        return 1;
    }
    s->ring_ok = true;   /* from here on ple_ring_close owns the ring */

    if (posix_memalign((void **)&s->ring_bounce, buf_align, (size_t)qd * stride) != 0) {
        s->ring_bounce = NULL;
        ple_fail(err, errlen, "cannot allocate %llu KiB of PLE ring buffers",
                 (unsigned long long)(((uint64_t)qd * stride) >> 10));
        ple_ring_close(s);
        return 1;
    }
    s->ring_iov  = calloc(qd, sizeof(*s->ring_iov));
    s->ring_slot = calloc(qd, sizeof(*s->ring_slot));
    s->ring_free = malloc((size_t)qd * sizeof(*s->ring_free));
    s->ring_pend = malloc((size_t)qd * sizeof(*s->ring_pend));
    if (!s->ring_iov || !s->ring_slot || !s->ring_free || !s->ring_pend) {
        ple_fail(err, errlen, "cannot allocate the PLE ring slot tables");
        ple_ring_close(s);
        return 1;
    }
    for (uint32_t i = 0; i < qd; i++) {
        s->ring_iov[i].iov_base = s->ring_bounce + (size_t)i * stride;
        s->ring_iov[i].iov_len  = stride;
    }
    if ((rc = io_uring_register_buffers(&s->ring, s->ring_iov, qd)) < 0) {
        ple_fail(err, errlen, "io_uring_register_buffers: %s", strerror(-rc));
        ple_ring_close(s);
        return 1;
    }
    if ((rc = io_uring_register_files(&s->ring, &s->fd, 1)) < 0) {
        ple_fail(err, errlen, "io_uring_register_files: %s", strerror(-rc));
        ple_ring_close(s);
        return 1;
    }
    s->ring_qd     = qd;
    s->ring_stride = stride;
    return 0;
}

/* Runs the whole miss list through the ring, ring_qd reads in flight, copying
 * each row out of its bounce block as that block lands. */
static int ple_ring_batch(ds4_ple_stream *s, char *err, size_t errlen) {
    const uint32_t n  = s->batch_n;
    const uint32_t qd = s->ring_qd;
    const double t0 = ple_now();
    uint64_t bytes = 0, reads = 0;
    uint32_t issued = 0, completed = 0, inflight = 0;
    uint32_t nfree = qd, npend = 0;
    int rc = 0;

    /* Slots start idle; `pend` holds slots waiting for an sqe, which is either
     * a fresh miss or the remainder of a read that came back short. */
    for (uint32_t i = 0; i < qd; i++) s->ring_free[i] = i;

    /* An error stops the batch here rather than reading the rest of a list
     * whose result is already being thrown away; the drain below then waits
     * for whatever is still outstanding. */
    while (completed < n && !rc) {
        /* Hand idle slots the next misses. */
        while (issued < n && nfree > 0) {
            const uint32_t si = s->ring_free[--nfree];
            struct ple_ring_slot *sl = &s->ring_slot[si];
            sl->miss = issued++;
            sl->done = 0;
            ple_row_geometry(s, s->miss_row[sl->miss], &sl->start, &sl->skew, &sl->len);
            s->ring_pend[npend++] = si;
        }

        /* Submit as many of them as the queue will take. */
        uint32_t submitted = 0;
        while (npend > 0) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&s->ring);
            if (!sqe) break;                    /* queue full; drain first */
            const uint32_t si = s->ring_pend[--npend];
            struct ple_ring_slot *sl = &s->ring_slot[si];
            uint8_t *slot_buf = s->ring_bounce + (size_t)si * s->ring_stride;
            io_uring_prep_read_fixed(sqe, 0, slot_buf + sl->done,
                                     (unsigned)(sl->len - sl->done),
                                     sl->start + sl->done, (int)si);
            sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, si);
            inflight++;
            submitted++;
        }
        if (submitted == 0 && inflight == 0) {
            ple_fail(err, errlen, "PLE ring made no progress: %u of %u rows", completed, n);
            rc = 1;
            break;
        }

        int got = io_uring_submit_and_wait(&s->ring, inflight ? 1u : 0u);
        if (got < 0) {
            if (got == -EINTR || got == -EAGAIN) continue;
            ple_fail(err, errlen, "io_uring_submit_and_wait: %s", strerror(-got));
            rc = 1;
            break;
        }

        /* Reap everything that has landed, then advance the queue once. */
        unsigned head, seen = 0;
        struct io_uring_cqe *cqe;
        io_uring_for_each_cqe(&s->ring, head, cqe) {
            seen++;
            const uint32_t si = (uint32_t)io_uring_cqe_get_data64(cqe);
            struct ple_ring_slot *sl = &s->ring_slot[si];
            const int res = cqe->res;
            inflight--;
            reads++;
            if (res < 0) {
                if (!rc) {
                    ple_fail(err, errlen, "PLE row read failed: %s", strerror(-res));
                    rc = 1;
                }
                s->ring_free[nfree++] = si;
                continue;
            }
            bytes += (uint64_t)res;
            sl->done += (size_t)res;
            /* The block asked for is always at least the row, so covering the
             * row is the only completion test needed. */
            const size_t need = sl->skew + s->row_bytes;
            if (sl->done >= need) {
                /* A read that stops short of the block but past the row is the
                 * file's last partial block, not an error. */
                memcpy(s->miss_dst[sl->miss],
                       s->ring_bounce + (size_t)si * s->ring_stride + sl->skew,
                       s->row_bytes);
                completed++;
                s->ring_free[nfree++] = si;
            } else if (res == 0) {
                if (!rc) {
                    ple_fail(err, errlen, "PLE row read hit end of file");
                    rc = 1;
                }
                s->ring_free[nfree++] = si;
            } else {
                s->ring_pend[npend++] = si;   /* short read: ask for the rest */
            }
        }
        if (seen) io_uring_cq_advance(&s->ring, seen);
    }

    /* An error can leave the loop with reads still outstanding, and those write
     * into the bounce buffers and, through miss_dst, into cache slots that the
     * unwind is about to hand back.  So wait for them.  If even that fails the
     * ring's state is no longer known, and dropping it puts the next fetch on
     * the pool rather than on a queue we cannot account for. */
    while (inflight > 0) {
        struct io_uring_cqe *cqe = NULL;
        const int w = io_uring_wait_cqe(&s->ring, &cqe);
        if (w < 0) {
            if (w == -EINTR) continue;
            ple_ring_close(s);
            fprintf(stderr, "ds4: PLE ring left %u read(s) unaccounted (%s); "
                            "falling back to the reader pool\n",
                    inflight, strerror(-w));
            return 1;
        }
        io_uring_cqe_seen(&s->ring, cqe);
        inflight--;
        reads++;
    }

    /* Unlike the pool, which sums each reader's own latency, this is the
     * batch's wall time: with one thread issuing them the two are the same
     * measurement only when the queue depth is one. */
    s->stats.reads      += reads;
    s->stats.read_bytes += bytes;
    s->stats.read_seconds += ple_now() - t0;
    return rc;
}

#endif /* DS4_HAVE_LIBURING */

/* Issues `s->batch_n` reads across the pool and waits for all of them. */
static int ple_run_batch(ds4_ple_stream *s, char *err, size_t errlen) {
    const uint32_t n = s->batch_n;
    if (n == 0) return 0;

#if DS4_HAVE_LIBURING
    /* When the ring came up the pool is never started, so its threads never
     * exist rather than sitting idle. */
    if (s->ring_ok) return ple_ring_batch(s, err, errlen);
#endif

    ple_pool_start(s);
    if (s->n_workers == 0) {
        ple_fail(err, errlen, "cannot allocate the PLE read pool");
        return 1;
    }

    atomic_store(&s->next, 0u);
    uint32_t wake = 0;   /* readers besides this thread */
    if (s->n_workers > 1 && n >= PLE_POOL_MIN_MISSES) {
        wake = (n < s->n_workers ? n : s->n_workers) - 1u;
    }
    if (wake) {
        pthread_mutex_lock(&s->mu);
        s->batch_left = wake;
        pthread_mutex_unlock(&s->mu);
        for (uint32_t i = 0; i < wake; i++) sem_post(&s->sem_work);
    }

    ple_run_shared(s, &s->workers[0]);

    if (wake) {
        pthread_mutex_lock(&s->mu);
        while (s->batch_left != 0) pthread_cond_wait(&s->cv_done, &s->mu);
        pthread_mutex_unlock(&s->mu);
    }

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

static int ple_fetch_locked(ds4_ple_stream *s, const uint64_t *rows, uint32_t n,
                            uint8_t *out, char *err, size_t errlen);

int ds4_ple_stream_fetch(ds4_ple_stream *s, const uint64_t *rows, uint32_t n,
                         uint8_t *out, char *err, size_t errlen) {
    if (!s || (!rows && n)) return 1;
    if (n == 0) return 0;
    pthread_mutex_lock(&s->fetch_mu);
    const int rc = ple_fetch_locked(s, rows, n, out, err, errlen);
    pthread_mutex_unlock(&s->fetch_mu);
    return rc;
}

static void *ple_prefetch_main(void *arg) {
    ds4_ple_stream *s = arg;
    pthread_mutex_lock(&s->pf_mu);
    for (;;) {
        while (!s->shutdown && !s->pf_busy) pthread_cond_wait(&s->pf_cv, &s->pf_mu);
        if (s->shutdown) break;
        pthread_mutex_unlock(&s->pf_mu);
        char err[256];
        (void)ds4_ple_stream_fetch(s, s->pf_rows, s->pf_n, s->pf_out, err, sizeof(err));
        pthread_mutex_lock(&s->pf_mu);
        s->pf_busy = false;
        pthread_cond_broadcast(&s->pf_cv);
    }
    pthread_mutex_unlock(&s->pf_mu);
    return NULL;
}

int ds4_ple_stream_prefetch(ds4_ple_stream *s, const uint64_t *rows, uint32_t n) {
    if (!s || !rows || n == 0 || s->n_slots == 0) return 0;   /* nothing to warm without a cache */
    if (!s->pf_ready) {
        if (pthread_mutex_init(&s->pf_mu, NULL) != 0) return 1;
        if (pthread_cond_init(&s->pf_cv, NULL) != 0) return 1;
        if (pthread_create(&s->pf_th, NULL, ple_prefetch_main, s) != 0) return 1;
        s->pf_ready = true;
    }
    pthread_mutex_lock(&s->pf_mu);
    while (s->pf_busy) pthread_cond_wait(&s->pf_cv, &s->pf_mu);
    if (s->pf_cap < n) {
        uint64_t *r = realloc(s->pf_rows, (size_t)n * sizeof(*r));
        uint8_t  *o = realloc(s->pf_out, (size_t)n * s->row_bytes);
        if (r) s->pf_rows = r;
        if (o) s->pf_out = o;
        if (!r || !o) { pthread_mutex_unlock(&s->pf_mu); return 1; }
        s->pf_cap = n;
    }
    memcpy(s->pf_rows, rows, (size_t)n * sizeof(*rows));
    s->pf_n = n;
    s->pf_busy = true;
    pthread_cond_signal(&s->pf_cv);
    pthread_mutex_unlock(&s->pf_mu);
    return 0;
}

static int ple_fetch_locked(ds4_ple_stream *s, const uint64_t *rows, uint32_t n,
                            uint8_t *out, char *err, size_t errlen) {

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

const char *ds4_ple_stream_backend(const ds4_ple_stream *s) {
    if (!s) return "none";
    return s->ring_ok ? "uring" : "pool";
}

void ds4_ple_stream_get_stats(const ds4_ple_stream *s, ds4_ple_stats *out) {
    if (!s || !out) return;
    *out = s->stats;
}

uint64_t ds4_ple_stream_cache_rows(const ds4_ple_stream *s) {
    return s ? s->n_slots : 0;
}
