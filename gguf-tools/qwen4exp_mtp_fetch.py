#!/usr/bin/env python3
"""Fetch the MTP weights for Qwen3.8-Flash-Next.

The GGUF that unsloth publishes drops the multi-token-prediction head, so the
draft model has to come from the base repo.  Downloading the base repo means
217 GB of BF16; the MTP head is 31 tensors and 5.2 GB of that, scattered over
28 of the 131 shards.  Every shard is a safetensors file, whose header is a
JSON map of tensor -> byte range, so each tensor can be pulled with one ranged
GET and nothing else is transferred.

Output is a directory of raw little-endian tensor payloads plus manifest.json,
which is what qwen4exp_mtp_quantize.py consumes.

  python3 gguf-tools/qwen4exp_mtp_fetch.py --out /path/to/mtp-raw
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next"
BASE = "https://huggingface.co/{repo}/resolve/main/{name}"
PREFIX = "mtp"

# BF16 is the only dtype the MTP tensors use, but the header is authoritative.
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1}


def url_for(name):
    return BASE.format(repo=REPO, name=name)


def http_get(url, rng=None, tries=5, token=None):
    """GET with optional Range, retrying on the transient 5xx HF likes to emit."""
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url)
        if rng is not None:
            req.add_header("Range", "bytes=%d-%d" % rng)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            # A 416 or 404 will never succeed; anything else might.
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404, 416):
                raise
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise last


def read_header(name, token):
    """The first 8 bytes of a safetensors file are the header length."""
    n = struct.unpack("<Q", http_get(url_for(name), (0, 7), token=token))[0]
    return json.loads(http_get(url_for(name), (8, 8 + n - 1), token=token)), 8 + n


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return "%.2f %s" % (n, unit)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the raw payloads")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token, if the repo ever becomes gated")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve ranges and print the plan without downloading")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sys.stderr.write("reading the shard index\n")
    index = json.loads(http_get(url_for("model.safetensors.index.json"),
                                token=args.token).decode())
    weight_map = index["weight_map"]
    wanted = {k: v for k, v in weight_map.items() if k.split(".")[0] == PREFIX}
    if not wanted:
        sys.stderr.write("no %s.* tensors in the index\n" % PREFIX)
        return 1

    by_shard = {}
    for tensor, shard in wanted.items():
        by_shard.setdefault(shard, []).append(tensor)

    sys.stderr.write("%d tensors across %d shards\n" % (len(wanted), len(by_shard)))

    manifest = {"repo": REPO, "tensors": {}}
    total = 0
    done = 0

    for shard in sorted(by_shard):
        header, data_start = read_header(shard, args.token)
        for tensor in sorted(by_shard[shard]):
            meta = header[tensor]
            begin, end = meta["data_offsets"]
            nbytes = end - begin
            total += nbytes
            out_path = os.path.join(args.out, tensor + ".bin")
            manifest["tensors"][tensor] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "bytes": nbytes,
                "file": os.path.basename(out_path),
                "shard": shard,
            }

            expect = 1
            for d in meta["shape"]:
                expect *= d
            expect *= DTYPE_BYTES.get(meta["dtype"], 0)
            if expect and expect != nbytes:
                sys.stderr.write("  %s: header says %d bytes, shape implies %d\n"
                                 % (tensor, nbytes, expect))
                return 1

            if args.dry_run:
                sys.stderr.write("  %-58s %-5s %-22s %s\n"
                                 % (tensor, meta["dtype"], meta["shape"], human(nbytes)))
                continue

            # Resume: a payload of exactly the right size is already ours.
            if os.path.exists(out_path) and os.path.getsize(out_path) == nbytes:
                done += nbytes
                sys.stderr.write("  have %-52s %s\n" % (tensor, human(nbytes)))
                continue

            t0 = time.time()
            # Chunked so a dropped connection costs one chunk, not 3.4 GB.
            chunk = 256 << 20
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                off = 0
                while off < nbytes:
                    take = min(chunk, nbytes - off)
                    lo = data_start + begin + off
                    f.write(http_get(url_for(shard), (lo, lo + take - 1), token=args.token))
                    off += take
            os.rename(tmp, out_path)
            done += nbytes
            dt = time.time() - t0
            sys.stderr.write("  got  %-52s %s in %.1fs (%s/s)\n"
                             % (tensor, human(nbytes), dt, human(nbytes / max(dt, 1e-3))))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    sys.stderr.write("%s across %d tensors -> %s\n"
                     % (human(total), len(wanted), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
