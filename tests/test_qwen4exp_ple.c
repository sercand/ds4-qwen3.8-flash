/* Phase-2 gate for qwen4exp PLE streaming.
 *
 * The n-gram hash, the shard byte offsets and the IQ4_NL decode all have to
 * agree with the reference exactly -- a single wrong row silently corrupts the
 * embedding with plausible-looking noise.  The expected values below come from
 * llama.cpp's `ple_embd` tensor on the same GGUF for the same tokens; see
 * misc/qwen4exp-oracle/.
 *
 * Set DS4_QWEN4EXP_MODEL to the first shard to run; without it the test skips,
 * since the model is 104 GiB. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ds4.h"

/* "The capital of France is" under the model's own tokenizer. */
static const int TOKENS[] = { 760, 6511, 314, 9338, 369 };
enum { N_TOKENS = 5 };

/* First three and last three values of each token's 2560-wide PLE embedding,
 * and the sum over all of them, as printed by llama-eval-callback. */
static const float GOLD[N_TOKENS][6] = {
    { -0.0117f,  0.0069f, -0.0117f,  0.0078f,  0.0027f, -0.0134f },
    { -0.0181f,  0.0098f,  0.0127f, -0.0011f,  0.0028f,  0.0077f },
    { -0.0041f,  0.0071f,  0.0002f,  0.0113f,  0.0185f, -0.0080f },
    {  0.0158f, -0.0048f, -0.0072f,  0.0087f,  0.0111f, -0.0119f },
    {  0.0064f, -0.0002f,  0.0120f, -0.0096f, -0.0043f,  0.0174f },
};
static const float GOLD_SUM = 1.408114f;

/* The EXL3 repack decodes a different quantization of the table, so its gate
 * is exllamav3's own decode of the same rows (misc/qwen4exp-oracle/exl3_ple_ref.py,
 * a (n_tokens, n_embd) f32 .npy named by DS4_QWEN4EXP_PLE_REF): every value must
 * match to fp16 resolution, since both sides round the row to fp16. */
static float *load_npy_f32(const char *path, long want) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    const unsigned hlen = pre[8] | (pre[9] << 8);
    if (fseek(f, (long)hlen, SEEK_CUR) != 0) { fclose(f); return NULL; }
    float *v = malloc((size_t)want * sizeof(float));
    if (!v || fread(v, sizeof(float), (size_t)want, f) != (size_t)want) { free(v); fclose(f); return NULL; }
    fclose(f);
    return v;
}

static int check_ref(const float *emb, uint32_t n_embd, const float *ref, const char *label) {
    /* The reference is exllamav3's torch decode (two rounded fp32 ops); its
     * CUDA kernel, like ours, fuses the multiply-add, so a value may land one
     * fp16 ulp away from the torch number.  Anything beyond one ulp is a real
     * codec mismatch. */
    int bad = 0, ulp = 0;
    double max_err = 0.0, sum = 0.0;
    for (uint32_t i = 0; i < (uint32_t)N_TOKENS * n_embd; i++) {
        const double e = fabs((double)emb[i] - (double)ref[i]);
        int ex = 0;
        (void)frexp(fabs((double)ref[i]), &ex);
        const double one_ulp = ldexp(1.0, ex - 11) + 1e-7;
        if (e > one_ulp) bad++;
        else if (e > 0.0) ulp++;
        if (e > max_err) max_err = e;
        sum += emb[i];
    }
    if (bad) {
        printf("  FAIL %s: %d of %u values differ from exllamav3 by more than an fp16 ulp "
               "(max |diff| %.3g)\n", label, bad, (unsigned)N_TOKENS * n_embd, max_err);
        return 1;
    }
    printf("  %-22s sum=%.6f matches exllamav3 (%d values one ulp off, max |diff| %.3g): PASS\n",
           label, sum, ulp, max_err);
    return 0;
}

static int check(const float *emb, uint32_t n_embd, const char *label) {
    int fail = 0;
    double sum = 0.0;
    for (uint32_t t = 0; t < N_TOKENS; t++) {
        const float *row = emb + (size_t)t * n_embd;
        for (uint32_t i = 0; i < n_embd; i++) sum += row[i];
        const float got[6] = { row[0], row[1], row[2],
                               row[n_embd - 3], row[n_embd - 2], row[n_embd - 1] };
        for (int k = 0; k < 6; k++) {
            if (fabsf(got[k] - GOLD[t][k]) > 6e-5f) {
                printf("  FAIL %s: token %u value %d: got %.6f expected %.4f\n",
                       label, t, k, (double)got[k], (double)GOLD[t][k]);
                fail = 1;
            }
        }
    }
    if (fabs(sum - (double)GOLD_SUM) > 3e-4) {
        printf("  FAIL %s: sum %.6f expected %.6f\n", label, sum, (double)GOLD_SUM);
        fail = 1;
    }
    if (!fail) printf("  %-22s sum=%.6f matches llama.cpp: PASS\n", label, sum);
    return fail;
}

int main(void) {
    const char *model = getenv("DS4_QWEN4EXP_MODEL");
    printf("qwen4exp PLE streaming tests:\n");
    if (!model || !model[0]) {
        printf("  skipped: set DS4_QWEN4EXP_MODEL to the first GGUF shard\n");
        return 0;
    }

    uint32_t n_embd = 0;
    /* 2560 is the model's embedding width; allocate for it up front and check. */
    float *emb = malloc((size_t)N_TOKENS * 4096 * sizeof(float));
    if (!emb) { printf("  FAIL: out of memory\n"); return 1; }

    int fail = 0;
    char err[512] = {0};
    ds4_ple_stats stats;
    const char *ref_path = getenv("DS4_QWEN4EXP_PLE_REF");
    float *ref = NULL;
    if (ref_path && ref_path[0]) {
        ref = load_npy_f32(ref_path, (long)N_TOKENS * 2560);
        if (!ref) { printf("  FAIL: cannot read %s\n", ref_path); free(emb); return 1; }
    }

    /* A cache large enough to hold every row this test touches, and one far too
     * small for them: both must produce identical embeddings. */
    const uint64_t sizes[] = { 64u << 20, 0u };
    const char *labels[] = { "64 MiB cache", "no cache" };

    for (int c = 0; c < 2; c++) {
        memset(emb, 0, (size_t)N_TOKENS * 4096 * sizeof(float));
        memset(&stats, 0, sizeof(stats));
        if (ds4_test_qwen4exp_ple_embed(model, TOKENS, N_TOKENS, sizes[c],
                                        emb, &n_embd, &stats, err, sizeof(err)) != 0) {
            printf("  FAIL: %s\n", err);
            free(emb);
            return 1;
        }
        if (n_embd != 2560) {
            printf("  FAIL: embedding width %u, expected 2560\n", n_embd);
            fail = 1;
            break;
        }
        fail |= ref ? check_ref(emb, n_embd, ref, labels[c]) : check(emb, n_embd, labels[c]);
        printf("      lookups=%llu hits=%llu misses=%llu reads=%llu read=%.1f KiB in %.2f ms\n",
               (unsigned long long)stats.lookups,
               (unsigned long long)stats.hits,
               (unsigned long long)stats.misses,
               (unsigned long long)stats.reads,
               (double)stats.read_bytes / 1024.0,
               stats.read_seconds * 1e3);
    }

    free(emb);
    free(ref);
    printf("\n%s\n", fail ? "FAILED" : "all tests passed");
    return fail;
}
