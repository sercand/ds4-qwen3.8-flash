/* Unit test for the per-device selective model cache
 * (mgpu-selective-model-cache).
 *
 * Exercises:
 *   - ds4_gpu_device_cache_tensors with disjoint ranges on device 0
 *   - ds4_gpu_lookup_cache at range bases and at interior offsets
 *     (proves the subrange pointer offset arithmetic is right)
 *   - device-id resolution
 *   - on multi-GPU boxes: caching on device 1 and active-device
 *     preference in lookup */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);   \
            return 1;                                                   \
        }                                                               \
    } while (0)

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_model_cache: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices\n");
        return 0;
    }

    CHECK(ds4_gpu_init(), "ds4_gpu_init");

    /* Build a synthetic 1-MiB "model" in host memory. */
    const size_t total = 1024 * 1024;
    void *host = NULL;
    CHECK(cudaMallocHost(&host, total) == cudaSuccess, "cudaMallocHost");
    unsigned char *bytes = (unsigned char *)host;
    for (size_t i = 0; i < total; i++) bytes[i] = (unsigned char)(i & 0xff);
    CHECK(ds4_gpu_set_model_map(host, total), "set_model_map");

    /* Three disjoint ranges on device 0. */
    ds4_tensor_range ranges[3];
    ranges[0].source_offset = 0;          ranges[0].bytes = 256 * 1024; ranges[0].target_device = 0;
    ranges[1].source_offset = 384 * 1024; ranges[1].bytes = 128 * 1024; ranges[1].target_device = 0;
    ranges[2].source_offset = 768 * 1024; ranges[2].bytes = 256 * 1024; ranges[2].target_device = 0;

    CHECK(ds4_gpu_device_cache_tensors(0, ranges, 3) == 0,
          "device_cache_tensors dev 0 (3 ranges)");

    /* Base lookups + interior offset arithmetic. */
    int dev = -1; void *base0 = NULL, *interior0 = NULL;
    CHECK(ds4_gpu_lookup_cache(0, 1024, &dev, &base0) == 1, "lookup range 0 base");
    CHECK(dev == 0, "range 0 device");
    CHECK(base0 != NULL, "range 0 ptr");
    /* An interior offset must return base0 + delta. */
    CHECK(ds4_gpu_lookup_cache(100, 1024, &dev, &interior0) == 1, "lookup range 0 interior");
    CHECK(dev == 0, "range 0 interior device");
    CHECK(interior0 == (char *)base0 + 100, "interior offset arithmetic");

    void *base1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024, 1024, &dev, &base1) == 1, "lookup range 1 base");
    CHECK(dev == 0, "range 1 device");
    /* Interior of range 1 at +200 bytes should be base1 + 200. */
    void *interior1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024 + 200, 1024, &dev, &interior1) == 1, "lookup range 1 interior");
    CHECK(interior1 == (char *)base1 + 200, "range 1 interior offset");

    void *base2 = NULL;
    CHECK(ds4_gpu_lookup_cache(900 * 1024, 1024, &dev, &base2) == 1, "lookup range 2");
    CHECK(dev == 0 && base2 != NULL, "range 2 device+ptr");

    /* Convenience wrapper. */
    CHECK(ds4_gpu_lookup_cache_device(0, 1024) == 0, "lookup_device range 0");

    /* Lookup must be overflow-safe: a query with bytes=UINT64_MAX must
     * not wrap around into a false hit. */
    int dev_overflow = -1; void *ptr_overflow = NULL;
    int hit = ds4_gpu_lookup_cache(100, UINT64_MAX, &dev_overflow, &ptr_overflow);
    /* Either miss (preferred), or hit but the path must NOT have wrapped.
     * Accept miss only — a wrap-induced hit would be a bug. */
    CHECK(hit == 0, "lookup with bytes=UINT64_MAX does not wrap into a false hit");

    /* Bounds-check: ranges that overflow the model must be rejected
     * before any allocation. */
    ds4_tensor_range bad_overflow = { 0, total + 1, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_overflow, 1) != 0,
          "overflow range rejected");
    ds4_tensor_range bad_offset = { total + 1, 16, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_offset, 1) != 0,
          "out-of-range offset rejected");
    ds4_tensor_range bad_wrap = { total - 4, UINT64_MAX, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_wrap, 1) != 0,
          "wrap-around range rejected");

    /* Gap not covered by selective ranges. The legacy chunked path may
     * happen to cover it (it caches the whole model span); accept either
     * outcome, but if it returns 1 the device must be 0. */
    int dev_gap = -1; void *ptr_gap = NULL;
    int gap_hit = ds4_gpu_lookup_cache(300 * 1024, 1024, &dev_gap, &ptr_gap);
    if (gap_hit) {
        CHECK(dev_gap == 0, "gap fallback device");
    }

    if (dev_count >= 2) {
        /* Cache a different range on device 1. */
        ds4_tensor_range r2;
        r2.source_offset = 256 * 1024;
        r2.bytes         = 128 * 1024;
        r2.target_device = 1;
        CHECK(ds4_gpu_device_cache_tensors(1, &r2, 1) == 0, "cache dev 1");

        /* With cudaGetDevice() == 1, the lookup should resolve to dev 1
         * for this range (the only selective entry covering it). */
        (void)cudaSetDevice(1);
        int dd = -1; void *pp = NULL;
        CHECK(ds4_gpu_lookup_cache(256 * 1024 + 10, 1024, &dd, &pp) == 1,
              "lookup dev 1");
        CHECK(dd == 1, "lookup resolves to dev 1");
        CHECK(pp != NULL, "lookup ptr non-null");
        (void)cudaSetDevice(0);
    }

    /* Weight-arena packing plan: a planned arena must hold the spans it was
     * planned for and nothing more.  This is the regression guard for the
     * 17.6 GiB of stranded chunk tails that fixed-size chunks used to cost
     * on qwen4exp (203 spans, 78.66 GiB of weights, 96.25 GiB of arenas). */
    {
        /* Cap one chunk so the unplanned-span check below cannot ask for a
         * default 1792 MiB chunk on a busy GPU. */
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1);

        char path[] = "/tmp/ds4_arena_planXXXXXX";
        const int fd = mkstemp(path);
        CHECK(fd >= 0, "mkstemp arena plan model");
        (void)unlink(path);
        const size_t file_bytes = 16u * 1024u * 1024u;
        void *filler = calloc(1, file_bytes);
        CHECK(filler != NULL, "calloc filler");
        CHECK(write(fd, filler, file_bytes) == (ssize_t)file_bytes, "write arena plan model");
        free(filler);
        void *fmap = mmap(NULL, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        CHECK(fmap != MAP_FAILED, "mmap arena plan model");
        CHECK(ds4_gpu_set_model_fd_for_map(fd, fmap), "set_model_fd_for_map");

        /* The last span's payload is deliberately not a multiple of 256, so
         * the slot's round-up is exercised rather than being a no-op that
         * would hide a mistake in the alignment arithmetic. */
        const uint64_t offs[5]  = { 0, 1u << 20, 5u << 20, 7u << 20, 9u << 20 };
        const uint64_t sizes[5] = { 1u << 20, 3u << 20, 2u << 20, 700u * 1024u,
                                    (1u << 20) + 100u };
        const uint32_t n_planned = 5;
        uint64_t want = 0, unrounded = 0;
        for (uint32_t i = 0; i < n_planned; i++) {
            want += (sizes[i] + 1024u + 255u) & ~(uint64_t)255u;
            unrounded += sizes[i] + 1024u;
        }
        CHECK(want > unrounded, "a span whose payload+pad needs rounding is in the set");

        const uint64_t before = ds4_gpu_model_weight_arena_bytes();
        const uint64_t planned = ds4_gpu_plan_model_weight_arena(sizes, n_planned);
        CHECK(planned == want, "plan covers exactly the span slots, rounding included");
        for (uint32_t i = 0; i < n_planned; i++) {
            CHECK(ds4_gpu_cache_model_range(fmap, file_bytes, offs[i], sizes[i], "plan-span") == 1,
                  "cache planned span");
        }
        CHECK(ds4_gpu_model_weight_arena_bytes() - before == planned,
              "planned arena bytes equal the span slots (no stranded tail)");

        /* A span the plan never saw still caches, on the fixed-chunk rule. */
        CHECK(ds4_gpu_cache_model_range(fmap, file_bytes, 8u << 20, 512u * 1024u,
                                        "unplanned-span") == 1,
              "cache unplanned span");
        CHECK(ds4_gpu_model_weight_arena_bytes() - before > planned,
              "unplanned span opens a chunk of its own");

        (void)ds4_gpu_set_model_fd_for_map(-1, NULL);
        (void)munmap(fmap, file_bytes);
        (void)close(fd);
        unsetenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    }

    ds4_gpu_cleanup();
    (void)cudaFreeHost(host);
    fprintf(stderr, "test_gpu_model_cache PASS (devs=%d)\n", dev_count);
    return 0;
}
