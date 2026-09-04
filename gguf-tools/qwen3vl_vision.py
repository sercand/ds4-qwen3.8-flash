#!/usr/bin/env python3
"""Create or validate the standalone Qwen3.8-Flash-Next vision encoder GGUF.

The serving GGUF (an exllamav3 repack) is text-only: it carries no `model.visual.*`
tensors at all.  This tool builds the sidecar that `ds4 --vision` loads, straight from
the official BF16 checkpoint.  Every visual tensor lives in shard 1 of 131, so only
that one shard has to be on disk:

    hf download Qwen/Qwen3.8-Flash-Next model-00001-of-00131.safetensors
    hf download Qwen/Qwen3.8-Flash-Next config.json preprocessor_config.json \
        model.safetensors.index.json

Do not build this from a quantized community repack.  Those store the tower as
MXFP8 plus 4-bit NVFP4 MLP down-projections, which both costs accuracy and makes a
per-layer comparison against the HF reference ambiguous -- a deviation could be the
quantizer or a porting bug, and you cannot tell which.

Payload bytes are copied verbatim, so the output is bit-identical to the source
tensors and `--validate --verify-payload` can prove it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import threading

from glm53_manifest import load_index, load_safetensors_header
from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_ARRAY,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
    GGUF_VERSION,
    QTYPE_BF16,
    QTYPE_NAMES,
    TensorPlan,
    align,
    fail,
    kv_f32,
    kv_string,
    kv_u32,
    pack_string,
    qtype_nbytes,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
    tensor_header,
)

DEFAULT_SOURCE_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SOURCE_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"
ARCHITECTURE = "qwen38f-vision"
PREFIX = "model.visual."
EXPECTED_TENSORS = 333

# Qwen3-VL's vision blocks use LayerNorm, and `vision_config` does not carry an eps:
# the reference reads `getattr(vision_config, "rms_norm_eps", 1e-6)`, so 1e-6 it is.
# Pinned here rather than read, so a future config that *does* carry one cannot change
# the encoder's numerics behind our back without this line being revisited.
LAYER_NORM_EPS = 1e-6

# vision_config fields that the ds4 encoder hardcodes in its kernels.  A checkpoint
# that disagrees would silently produce garbage, so refuse it here instead.
# `model_type` is spelled differently depending on which transformers version wrote
# the config -- the official repo says "qwen4_exp", community repacks converted with a
# newer transformers say "qwen3_5_vision".  Same tower either way; every geometric
# field below is identical, and those are what the kernels actually depend on.
ALLOWED_VISION_MODEL_TYPES = ("qwen4_exp", "qwen3_5_vision")

REQUIRED_VISION = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 2560,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
}


def kv_f32_array(key, values):
    return (
        pack_string(key)
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_FLOAT32, len(values))
        + struct.pack(f"<{len(values)}f", *values)
    )


class VisionSourceDB:
    """`SourceDB` restricted to `model.visual.*`.

    The stock `SourceDB` insists every shard named in the index is present.  This
    checkpoint has 131 of them and ~99 GB of language-model weights we have no use
    for, so this walks only the shards that actually hold visual tensors.
    """

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        _document, weight_map = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
        self.weight_map = {n: s for n, s in weight_map.items() if n.startswith(PREFIX)}
        if not self.weight_map:
            fail(f"{hf_dir}: index lists no {PREFIX}* tensors; is this the multimodal checkpoint?")
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()

        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path} (only shards holding {PREFIX}* are needed)")
            for name, info in load_safetensors_header(path).items():
                if name not in self.weight_map:
                    continue
                if self.weight_map[name] != shard:
                    fail(f"index assigns {name} to {self.weight_map[name]!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)

        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            fail(f"source tensor not found: {name}")

    def _fd(self, shard):
        with self._fd_lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(os.path.join(self.hf_dir, shard), os.O_RDONLY)
                self._fds[shard] = fd
            return fd

    def iter_read(self, name, chunk_size=16 << 20):
        info = self.info(name)
        fd = self._fd(info["shard"])
        offset = info["offset"]
        remaining = info["nbytes"]
        while remaining:
            length = min(remaining, chunk_size)
            data = os.pread(fd, length, offset)
            if len(data) != length:
                fail(f"short payload read for {name}")
            yield data
            offset += length
            remaining -= length


def load_source_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    with open(os.path.join(hf_dir, "preprocessor_config.json"), encoding="utf-8") as fp:
        processor = json.load(fp)
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        fail("config.json has no vision_config; this is the text-only checkpoint")

    if vision.get("model_type") not in ALLOWED_VISION_MODEL_TYPES:
        fail(f"unexpected vision_config.model_type: {vision.get('model_type')!r} "
             f"(expected one of {ALLOWED_VISION_MODEL_TYPES})")
    for key, expected in REQUIRED_VISION.items():
        if vision.get(key) != expected:
            fail(f"unexpected vision_config.{key}: {vision.get(key)!r} (expected {expected!r})")

    # DeepStack multi-level fusion is a real architectural feature of stock Qwen3-VL,
    # and ds4's encoder does not implement it.  Flash-Next ships with it disabled; if a
    # future checkpoint turns it on, the tower would need extra mergers and per-layer
    # injection into the LLM residual, so fail loudly rather than silently dropping it.
    if vision.get("deepstack_visual_indexes"):
        fail(f"deepstack_visual_indexes is non-empty ({vision['deepstack_visual_indexes']}); "
             "the ds4 encoder implements the single-merger tower only")

    if processor.get("patch_size") != vision["patch_size"]:
        fail("processor and model patch sizes differ")
    if processor.get("merge_size") != vision["spatial_merge_size"]:
        fail("processor and model merge sizes differ")
    if processor.get("temporal_patch_size") != vision["temporal_patch_size"]:
        fail("processor and model temporal patch sizes differ")
    for key in ("image_mean", "image_std"):
        values = processor.get(key)
        if not isinstance(values, list) or len(values) != vision["in_channels"]:
            fail(f"preprocessor_config.{key} must have {vision['in_channels']} entries")
    size = processor.get("size")
    if not isinstance(size, dict) or "shortest_edge" not in size or "longest_edge" not in size:
        fail("preprocessor_config.size must carry shortest_edge and longest_edge")

    for key in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not isinstance(config.get(key), int):
            fail(f"config.json is missing {key}")
    return config, vision, processor


def build_vision_plan(db):
    plan = []
    offset = 0
    for name in sorted(db.tensors):
        info = db.info(name)
        if info["dtype"] != "BF16":
            fail(f"{name}: expected BF16 source, got {info['dtype']}; "
                 "build from the official checkpoint, not a quantized repack")
        if ".blocks." in name:
            role = "block"
        elif ".patch_embed." in name:
            role = "patch_embedding"
        elif ".merger." in name:
            role = "merger"
        elif ".pos_embed." in name:
            role = "position_embedding"
        else:
            fail(f"{name}: unclassified vision tensor")
        item = TensorPlan(
            name=name,
            shape=tuple(reversed(info["shape"])),
            qtype=QTYPE_BF16,
            role=role,
            source=name,
            raw_copy=True,
        )
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        if item.nbytes != info["nbytes"]:
            fail(f"{name}: planned {item.nbytes} bytes, source has {info['nbytes']}")
        item.offset = offset
        offset += align(item.nbytes)
        plan.append(item)

    if len(plan) != EXPECTED_TENSORS:
        fail(f"expected {EXPECTED_TENSORS} vision tensors, found {len(plan)}")
    return plan


def vision_metadata(hf_dir, source_revision):
    config, vision, processor = load_source_config(hf_dir)
    prefix = ARCHITECTURE
    merge = vision["spatial_merge_size"]
    patch = vision["patch_size"]
    # One LLM token covers merge^2 patches of patch^2 pixels each.
    pixels_per_token = patch * patch * merge * merge
    min_pixels = int(processor["size"]["shortest_edge"])
    max_pixels = int(processor["size"]["longest_edge"])
    records = [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "Qwen3.8-Flash-Next Vision Encoder"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", source_revision),
        kv_u32(f"{prefix}.block_count", vision["depth"]),
        kv_u32(f"{prefix}.embedding_length", vision["hidden_size"]),
        kv_u32(f"{prefix}.feed_forward_length", vision["intermediate_size"]),
        kv_u32(f"{prefix}.attention.head_count", vision["num_heads"]),
        kv_u32(f"{prefix}.projection_length", vision["out_hidden_size"]),
        kv_u32(f"{prefix}.patch_size", patch),
        kv_u32(f"{prefix}.temporal_patch_size", vision["temporal_patch_size"]),
        kv_u32(f"{prefix}.spatial_merge_size", merge),
        kv_u32(f"{prefix}.channel_count", vision["in_channels"]),
        kv_u32(f"{prefix}.position_embedding_count", vision["num_position_embeddings"]),
        kv_f32(f"{prefix}.attention.layer_norm_epsilon", LAYER_NORM_EPS),
        kv_u32(f"{prefix}.image_token_id", config["image_token_id"]),
        kv_u32(f"{prefix}.image_start_token_id", config["vision_start_token_id"]),
        kv_u32(f"{prefix}.image_end_token_id", config["vision_end_token_id"]),
        kv_u32(f"{prefix}.image.min_pixels", min_pixels),
        kv_u32(f"{prefix}.image.max_pixels", max_pixels),
        kv_u32(f"{prefix}.image.min_tokens", min_pixels // pixels_per_token),
        kv_u32(f"{prefix}.image.max_tokens", max_pixels // pixels_per_token),
        kv_f32_array(f"{prefix}.image.mean", processor["image_mean"]),
        kv_f32_array(f"{prefix}.image.std", processor["image_std"]),
    ]
    return records


def layout(plan, metadata):
    header_bytes = 4 + 4 + 8 + 8
    header_bytes += sum(len(record) for record in metadata)
    header_bytes += sum(len(tensor_header(item)) for item in plan)
    data_offset = align(header_bytes)
    data_bytes = sum(align(item.nbytes) for item in plan)
    return data_offset, data_bytes


def print_summary(plan, metadata):
    data_offset, data_bytes = layout(plan, metadata)
    print(f"tensors: {len(plan)}")
    print(f"metadata_records: {len(metadata)}")
    print(f"metadata_bytes: {data_offset}")
    print(f"tensor_bytes: {sum(item.nbytes for item in plan)}")
    print(f"file_bytes: {data_offset + data_bytes}")
    roles = {}
    for item in plan:
        roles[item.role] = roles.get(item.role, 0) + item.nbytes
    for role, size in sorted(roles.items()):
        print(f"role_bytes: {role} {size}")
    return data_offset, data_bytes


def create_gguf(path, plan, metadata, db, overwrite):
    data_offset, data_bytes = print_summary(plan, metadata)
    required = data_offset + data_bytes + (1 << 30)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free < required:
        fail(f"insufficient free space: need output plus reserve {required}, have {free}")
    if os.path.exists(path) and not overwrite:
        fail(f"output exists: {path}; use --overwrite")

    partial = path + ".partial"
    if os.path.exists(partial):
        if not overwrite:
            fail(f"partial output exists: {partial}; use --overwrite")
        os.unlink(partial)

    with open(partial, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(plan), len(metadata)))
        for record in metadata:
            fp.write(record)
        for item in plan:
            fp.write(tensor_header(item))
        if fp.tell() > data_offset:
            fail("GGUF header exceeds its planned data offset")
        fp.write(bytes(data_offset - fp.tell()))

        for index, item in enumerate(plan, 1):
            if fp.tell() != data_offset + item.offset:
                fail(f"{item.name}: output offset mismatch")
            written = 0
            for chunk in db.iter_read(item.source):
                fp.write(chunk)
                written += len(chunk)
            if written != item.nbytes:
                fail(f"{item.name}: copied {written} bytes, expected {item.nbytes}")
            fp.write(bytes(align(item.nbytes) - item.nbytes))
            if index % 50 == 0 or index == len(plan):
                print(f"copied tensors: {index}/{len(plan)}", file=sys.stderr, flush=True)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(partial, path)


def read_metadata_value(fp, value_type):
    if value_type == GGUF_STRING:
        return read_gguf_string(fp, "GGUF metadata string")
    if value_type == GGUF_UINT32:
        return read_u32(fp, "GGUF metadata uint32")
    if value_type == GGUF_UINT64:
        return read_u64(fp, "GGUF metadata uint64")
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
    if value_type == GGUF_ARRAY:
        element_type = read_u32(fp, "GGUF array element type")
        count = read_u64(fp, "GGUF array count")
        if element_type == GGUF_FLOAT32 and count <= 1024:
            return list(struct.unpack(f"<{count}f", read_exact(fp, count * 4, "GGUF float array")))
        fail(f"unsupported validation array type/count: {element_type}/{count}")
    skip_gguf_value(fp, value_type)
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gguf(path, plan, expected_metadata, db, verify_payload, expected_sha256):
    expected_values = {}
    for record in expected_metadata:
        reader = io.BytesIO(record)
        key = read_gguf_string(reader, "expected metadata key")
        value_type = read_u32(reader, "expected metadata type")
        expected_values[key] = read_metadata_value(reader, value_type)

    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version != GGUF_VERSION:
            fail(f"expected GGUF v{GGUF_VERSION}, got v{version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        if tensor_count != len(plan):
            fail(f"tensor count {tensor_count} != expected {len(plan)}")
        if metadata_count != len(expected_metadata):
            fail(f"metadata count {metadata_count} != expected {len(expected_metadata)}")

        actual_values = {}
        for _ in range(metadata_count):
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            actual_values[key] = read_metadata_value(fp, value_type)
        if actual_values != expected_values:
            fail("GGUF metadata differs from the pinned conversion metadata")

        for index, item in enumerate(plan):
            name = read_gguf_string(fp, f"tensor {index} name")
            rank = read_u32(fp, f"tensor {index} rank")
            shape = tuple(read_u64(fp, f"tensor {index} dimension") for _ in range(rank))
            qtype = read_u32(fp, f"tensor {index} type")
            offset = read_u64(fp, f"tensor {index} offset")
            if (name, shape, qtype, offset) != (item.name, item.shape, item.qtype, item.offset):
                fail(f"tensor {index} header differs from the conversion plan: {name}")

        data_offset, data_bytes = layout(plan, expected_metadata)
        if fp.tell() > data_offset:
            fail("parsed GGUF header exceeds the planned data offset")
        actual_size = os.fstat(fp.fileno()).st_size
        if actual_size != data_offset + data_bytes:
            fail(f"file size {actual_size} != expected {data_offset + data_bytes}")

        verified = 0
        if verify_payload:
            for index, item in enumerate(plan, 1):
                fp.seek(data_offset + item.offset)
                for source in db.iter_read(item.source):
                    if read_exact(fp, len(source), item.name) != source:
                        fail(f"{item.name}: payload differs from the official source")
                    verified += len(source)
                padding = read_exact(fp, align(item.nbytes) - item.nbytes, f"{item.name} padding")
                if any(padding):
                    fail(f"{item.name}: nonzero alignment padding")
                if index % 50 == 0 or index == len(plan):
                    print(f"verified payloads: {index}/{len(plan)}", file=sys.stderr, flush=True)

    print(f"validated {path}: {len(plan)} {QTYPE_NAMES[QTYPE_BF16]} tensors, {actual_size} bytes")
    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        fail(f"SHA-256 {digest} != expected {expected_sha256.lower()}")
    print(f"SHA-256: {digest}")
    if verify_payload:
        print(f"source payloads matched byte for byte: {verified} bytes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", required=True, help="official Qwen3.8-Flash-Next snapshot")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", help="create this vision GGUF")
    output.add_argument("--validate", metavar="GGUF", help="validate an existing vision GGUF")
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument("--expected-sha256", help="expected hash when validating")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.out and not args.validate:
        parser.error("one of --out, --validate, or --dry-run is required")
    if args.verify_payload and not args.validate:
        parser.error("--verify-payload requires --validate")
    if args.expected_sha256 and not args.validate:
        parser.error("--expected-sha256 requires --validate")
    return args


def main():
    args = parse_args()
    db = VisionSourceDB(args.hf)
    plan = build_vision_plan(db)
    metadata = vision_metadata(args.hf, args.source_revision)
    if args.dry_run:
        print_summary(plan, metadata)
    elif args.out:
        create_gguf(args.out, plan, metadata, db, args.overwrite)
        print(f"qwen3vl-vision: wrote {args.out}", file=sys.stderr)
    else:
        validate_gguf(args.validate, plan, metadata, db,
                      args.verify_payload, args.expected_sha256)


if __name__ == "__main__":
    main()
