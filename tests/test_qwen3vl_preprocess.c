/* Qwen3-VL preprocessing against the HF reference.
 *
 * Two modes:
 *   grid W H [W H ...]        print "W H grid_w grid_h tokens" per size
 *   patches W H OUT.f32       write the flattened patch tensor for a synthetic image
 *
 * The synthetic image is a deterministic gradient so the Python side can build the
 * identical input without shipping a fixture.  See tests/run_qwen3vl_preprocess.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_image.h"

/* Must match make_image() in tests/run_qwen3vl_preprocess.py exactly. */
static void fill_synthetic(uint8_t *rgb, uint32_t width, uint32_t height) {
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *p = rgb + ((size_t)y * width + x) * 3;
            p[0] = (uint8_t)((x * 7u + y * 13u) & 0xFFu);
            p[1] = (uint8_t)((x * 3u + y * 29u + 17u) & 0xFFu);
            p[2] = (uint8_t)((x * 11u ^ (y * 5u)) & 0xFFu);
        }
    }
}

static int run(uint32_t width, uint32_t height, ds4_image_patches *out) {
    ds4_image image;
    memset(&image, 0, sizeof(image));
    image.width = width;
    image.height = height;
    image.rgb = malloc((size_t)width * height * 3);
    if (!image.rgb) return 0;
    fill_synthetic(image.rgb, width, height);

    char error[256];
    /* 64 = min_pixels 65536 / 1024, 16384 = max_pixels 16777216 / 1024: the
     * checkpoint's own preprocessor_config budget, so the reference agrees. */
    int ok = ds4_image_preprocess_qwen3vl(out, &image, 64, 16384, error, sizeof(error));
    if (!ok) fprintf(stderr, "preprocess failed for %ux%u: %s\n", width, height, error);
    free(image.rgb);
    return ok;
}

int main(int argc, char **argv) {
    /* raw IN OUT: IN is "<u32 width><u32 height><rgb...>", so the Python driver can
     * feed arbitrary pixels (smooth, photographic, pathological) without a decoder. */
    if (argc >= 4 && strcmp(argv[1], "raw") == 0) {
        FILE *fp = fopen(argv[2], "rb");
        if (!fp) { perror("fopen"); return 1; }
        ds4_image image;
        memset(&image, 0, sizeof(image));
        if (fread(&image.width, 4, 1, fp) != 1 || fread(&image.height, 4, 1, fp) != 1) {
            fprintf(stderr, "short header\n"); fclose(fp); return 1;
        }
        size_t bytes = (size_t)image.width * image.height * 3;
        image.rgb = malloc(bytes);
        if (!image.rgb || fread(image.rgb, 1, bytes, fp) != bytes) {
            fprintf(stderr, "short pixels\n"); fclose(fp); free(image.rgb); return 1;
        }
        fclose(fp);
        ds4_image_patches patches;
        char error[256];
        int ok = ds4_image_preprocess_qwen3vl(&patches, &image, 64, 16384,
                                              error, sizeof(error));
        free(image.rgb);
        if (!ok) { fprintf(stderr, "%s\n", error); return 1; }
        FILE *out_fp = fopen(argv[3], "wb");
        if (!out_fp) { perror("fopen"); ds4_image_patches_free(&patches); return 1; }
        size_t values = (size_t)patches.patch_count * 3 * 2 * 16 * 16;
        size_t wrote = fwrite(patches.patches, sizeof(float), values, out_fp);
        fclose(out_fp);
        printf("%u %u %u %u %u\n", image.width, image.height,
               patches.grid_width, patches.grid_height, patches.image_token_count);
        ds4_image_patches_free(&patches);
        return wrote == values ? 0 : 1;
    }

    if (argc >= 4 && strcmp(argv[1], "patches") == 0) {
        uint32_t width = (uint32_t)strtoul(argv[2], NULL, 10);
        uint32_t height = (uint32_t)strtoul(argv[3], NULL, 10);
        ds4_image_patches patches;
        if (!run(width, height, &patches)) return 1;
        FILE *fp = fopen(argv[4], "wb");
        if (!fp) { perror("fopen"); ds4_image_patches_free(&patches); return 1; }
        size_t values = (size_t)patches.patch_count * 3 * 2 * 16 * 16;
        if (fwrite(patches.patches, sizeof(float), values, fp) != values) {
            fprintf(stderr, "short write\n");
            fclose(fp);
            ds4_image_patches_free(&patches);
            return 1;
        }
        fclose(fp);
        printf("%u %u %u %u %u\n", width, height,
               patches.grid_width, patches.grid_height, patches.image_token_count);
        ds4_image_patches_free(&patches);
        return 0;
    }

    if (argc >= 4 && strcmp(argv[1], "grid") == 0) {
        for (int i = 2; i + 1 < argc; i += 2) {
            uint32_t width = (uint32_t)strtoul(argv[i], NULL, 10);
            uint32_t height = (uint32_t)strtoul(argv[i + 1], NULL, 10);
            ds4_image_patches patches;
            if (!run(width, height, &patches)) {
                printf("%u %u FAIL\n", width, height);
                continue;
            }
            printf("%u %u %u %u %u\n", width, height,
                   patches.grid_width, patches.grid_height, patches.image_token_count);
            ds4_image_patches_free(&patches);
        }
        return 0;
    }

    fprintf(stderr,
            "usage: %s grid W H [W H ...]\n"
            "       %s patches W H OUT.f32\n"
            "       %s raw IN.raw OUT.f32\n",
            argv[0], argv[0], argv[0]);
    return 2;
}
