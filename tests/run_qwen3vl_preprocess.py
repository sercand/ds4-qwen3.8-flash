#!/usr/bin/env python3
"""Check ds4's Qwen3-VL preprocessing against the reference.

Stage 1 (no torch needed): sweep image sizes through the reference `smart_resize` and
compare the resulting (grid_w, grid_h, tokens) with ds4's.  This is where the rounding
rules bite -- Python's round() is half-to-even, and the over-budget branch rescales
from the ORIGINAL pixel count, not the aligned one.

Stage 2 (needs transformers+torch): compare the actual patch tensor for a few sizes.

usage: run_qwen3vl_preprocess.py BINARY [--patches]
"""
import math
import subprocess
import sys

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 65536, 16777216


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Verbatim from transformers/models/qwen2_vl/image_processing_qwen2_vl.py."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def sizes():
    out = []
    # Exact multiples, and every offset around them: the half-to-even cases live at
    # +/- factor/2, which is exactly where lround() and round() disagree.
    for base in (256, 512, 1024, 2048):
        for delta in (-17, -16, -15, -1, 0, 1, 15, 16, 17):
            out.append((base + delta, base + delta))
    # Non-square, including one axis needing a different rounding direction.
    out += [(640, 480), (480, 640), (1920, 1080), (1080, 1920), (33, 4000),
            (4000, 33), (720, 1280), (1281, 721)]
    # Below min_pixels (the ceil branch) and above max_pixels (the floor branch).
    out += [(8, 8), (16, 16), (64, 64), (100, 100), (255, 255), (1, 100), (100, 1)]
    out += [(8000, 8000), (5000, 4000), (16000, 200), (4096, 4096), (4097, 4097)]
    # Exact .5 multiples of the factor: 16 = factor/2 -> round-half-to-even territory.
    out += [(48, 48), (80, 80), (112, 112), (144, 144), (272, 272), (304, 304)]
    return out


def main():
    binary = sys.argv[1]
    want_patches = "--patches" in sys.argv

    cases = sizes()
    args = [binary, "grid"]
    for w, h in cases:
        args += [str(w), str(h)]
    lines = subprocess.run(args, capture_output=True, text=True, check=True).stdout.split()

    got = {}
    it = iter(lines)
    for token in it:
        w = int(token)
        h = int(next(it))
        gw = next(it)
        if gw == "FAIL":
            got[(w, h)] = None
            continue
        got[(w, h)] = (int(gw), int(next(it)), int(next(it)))

    failures = 0
    checked = 0
    for w, h in cases:
        try:
            hb, wb = smart_resize(h, w)
            expect = (wb // PATCH, hb // PATCH, (hb // PATCH) * (wb // PATCH) // (MERGE * MERGE))
        except ValueError:
            expect = None
        actual = got.get((w, h))
        checked += 1
        if actual != expect:
            failures += 1
            print(f"MISMATCH {w}x{h}: ds4={actual} reference={expect}")
    print(f"grid sweep: {checked - failures}/{checked} sizes match the reference")
    if failures:
        return 1

    if want_patches:
        return check_patches(binary)
    return 0


def check_patches(binary):
    """Compare the patch tensor with the HF reference.

    A note on tolerance, because it is not arbitrary.  The reference implementations
    disagree with EACH OTHER on high-frequency content: PIL (the slow processor)
    resamples uint8 in two separable passes with the intermediate quantized back to
    uint8, while torchvision on float tensors (the fast processor) does not.  Measured
    on a deliberately pathological sawtooth with hard 255->0 wraps, torchvision-float
    differs from PIL by up to 20.7 levels -- and torchvision-uint8 differs from PIL by
    only 2.  ds4 computes the filter in double and rounds once, so it lands on the
    float side.

    On smooth and photographic content -- the actual use case -- all of them agree to
    within one uint8 step.  So the strict gate runs on smooth images, and the sawtooth
    is carried as an informational case with a bound covering the spread between the
    references rather than pretending one of them is canonical.
    """
    import tempfile, os
    import numpy as np
    try:
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    except ImportError as exc:
        print(f"patch check skipped: {exc}")
        return 0

    proc = Qwen2VLImageProcessor(
        patch_size=PATCH, merge_size=MERGE, temporal_patch_size=2,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, do_convert_rgb=True)

    # One uint8 step survives the (x/255 - 0.5)/0.5 normalize as 2/255.
    LSB = 2.0 / 255.0
    SMOOTH_TOLERANCE = LSB * 1.5
    SAWTOOTH_BOUND = 0.25          # the measured PIL-vs-torchvision-float spread

    def make(kind, w, h):
        x = np.arange(w)[None, :]
        y = np.arange(h)[:, None]
        img = np.empty((h, w, 3), np.uint8)
        if kind == "smooth":
            img[..., 0] = (127.5 + 120 * np.sin(2 * np.pi * x / w)).astype(np.uint8)
            img[..., 1] = (127.5 + 120 * np.cos(2 * np.pi * y / h)).astype(np.uint8)
            img[..., 2] = (x / max(w - 1, 1) * 255).astype(np.uint8)
        else:
            img[..., 0] = (x * 7 + y * 13) & 0xFF
            img[..., 1] = (x * 3 + y * 29 + 17) & 0xFF
            img[..., 2] = (x * 11 ^ (y * 5)) & 0xFF
        return img

    failures = 0
    for kind, tolerance, fatal in (("smooth", SMOOTH_TOLERANCE, True),
                                   ("sawtooth", SAWTOOTH_BOUND, False)):
        for w, h in [(100, 100), (200, 150), (640, 480), (1024, 768), (1920, 1080)]:
            img = make(kind, w, h)
            ref = proc(images=Image.fromarray(img), return_tensors="np")
            reference = ref["pixel_values"]
            _gt, gh, gw = ref["image_grid_thw"][0]

            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fp:
                raw = fp.name
                fp.write(int(w).to_bytes(4, "little"))
                fp.write(int(h).to_bytes(4, "little"))
                fp.write(img.tobytes())
            with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as fp:
                dump = fp.name
            try:
                out = subprocess.run([binary, "raw", raw, dump],
                                     capture_output=True, text=True, check=True).stdout.split()
                mine = np.fromfile(dump, dtype=np.float32).reshape(-1, 3 * 2 * PATCH * PATCH)
            finally:
                os.unlink(raw)
                os.unlink(dump)

            if (int(out[2]), int(out[3])) != (int(gw), int(gh)):
                print(f"MISMATCH {kind} {w}x{h}: grid "
                      f"ds4=({out[2]},{out[3]}) ref=({gw},{gh})")
                failures += 1
                continue
            if mine.shape != reference.shape:
                print(f"MISMATCH {kind} {w}x{h}: shape {mine.shape} vs {reference.shape}")
                failures += 1
                continue
            delta = np.abs(mine - reference)
            worst, average = float(delta.max()), float(delta.mean())
            status = "ok" if worst <= tolerance else "OVER"
            note = "" if fatal else "  (informational: references disagree here)"
            print(f"{kind:9s} {w}x{h}: max={worst:.6f} ({worst / LSB:.2f} LSB) "
                  f"mean={average:.9f} {status}{note}")
            if worst > tolerance and fatal:
                failures += 1

    print(f"patch check: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
