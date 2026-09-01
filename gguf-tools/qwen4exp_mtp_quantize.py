#!/usr/bin/env python3
"""Quantize the Qwen3.8-Flash-Next MTP head into a ds4 sidecar GGUF.

Input is the raw BF16 payload directory that qwen4exp_mtp_fetch.py produces.
Output is a single-file GGUF holding one decoder layer named `blk.0.*` -- the
same names weights_bind_qwen4exp_layer already binds, so ds4 reuses the layer
binder verbatim -- plus the MTP-only projections under `mtp.*`.

Quant roles mirror the main model so the draft and the target agree on
precision: routed gate/up Q4_K, routed down Q5_1, every other 2-D weight Q8_0,
norms F32.

Every norm in the head is a GemmaRMSNorm (vLLM applies `x * (1 + w)`), and the
target GGUF stores those weights with the +1 already baked in -- ds4's kernels
multiply by the stored vector as-is.  The raw HF payload does not, so the +1 is
added here.  Verified against the target: its blk.3.attn_q_norm averages 1.28
where the raw MTP q_norm averages 2.68 (-> 3.68), and pre_fc_norm_embedding is
-0.76 raw, which applied without the +1 would be a negative scale.

  python3 gguf-tools/qwen4exp_mtp_quantize.py \
      --raw /path/to/mtp-raw --out qwen38-flash-mtp.gguf
"""

import argparse
import ctypes
import json
import os
import struct
import sys

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_UINT32, GGUF_FLOAT32, GGUF_STRING, GGUF_ARRAY = 4, 6, 8, 9

QTYPE_F32, QTYPE_Q5_1, QTYPE_Q8_0, QTYPE_Q4_K = 0, 7, 8, 12
QTYPE_NAMES = {QTYPE_F32: "F32", QTYPE_Q5_1: "Q5_1",
               QTYPE_Q8_0: "Q8_0", QTYPE_Q4_K: "Q4_K"}
QTYPE_LAYOUT = {QTYPE_F32: (1, 4), QTYPE_Q5_1: (32, 24),
                QTYPE_Q8_0: (32, 34), QTYPE_Q4_K: (256, 144)}

# Model geometry, cross-checked against the fetched tensor shapes.
N_EMBD, N_HC, N_HC_LOWRANK = 2560, 4, 320
N_EXPERT, N_FF_EXP = 512, 640
N_HEAD, N_HEAD_KV, HEAD_DIM = 24, 2, 256
IDX_HEADS, IDX_KV_HEADS, IDX_HEAD_DIM = 4, 1, 128


def fail(msg):
    sys.stderr.write("error: %s\n" % msg)
    raise SystemExit(1)


def align(v, a=GGUF_ALIGNMENT):
    return (v + a - 1) // a * a


def pack_string(v):
    data = v.encode("utf-8") if isinstance(v, str) else v
    return struct.pack("<Q", len(data)) + data


def kv_string(key, value):
    return pack_string(key) + struct.pack("<I", GGUF_STRING) + pack_string(value)


def kv_u32(key, value):
    return pack_string(key) + struct.pack("<II", GGUF_UINT32, value)


def kv_f32(key, value):
    return pack_string(key) + struct.pack("<If", GGUF_FLOAT32, value)


class Quantizer:
    """Thin ctypes shim over gguf-tools/quants.c, so the sidecar is quantized
    by exactly the code ds4 dequantizes with."""

    def __init__(self, lib_path):
        import numpy as np
        self.np = np
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        for t in (QTYPE_Q4_K, QTYPE_Q5_1, QTYPE_Q8_0):
            self.lib.ds4q_quantize_init(t)

    def quantize(self, array, qtype):
        """array is [.., ncols] float32, C-contiguous; ncols is the row length."""
        np = self.np
        if qtype == QTYPE_F32:
            return np.ascontiguousarray(array, dtype=np.float32).tobytes()
        a = np.ascontiguousarray(array, dtype=np.float32)
        ncols = a.shape[-1]
        nrows = a.size // ncols
        block, block_bytes = QTYPE_LAYOUT[qtype]
        if ncols % block:
            fail("row length %d is not a multiple of %d for %s"
                 % (ncols, block, QTYPE_NAMES[qtype]))
        out = bytearray(nrows * (ncols // block) * block_bytes)
        buf = (ctypes.c_char * len(out)).from_buffer(out)
        src = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n = self.lib.ds4q_quantize_chunk(qtype, src, ctypes.cast(buf, ctypes.c_void_p),
                                         0, nrows, ncols, None)
        if n != len(out):
            fail("quantize returned %d bytes, expected %d" % (n, len(out)))
        del buf
        return bytes(out)


class Raw:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.tensors = self.manifest["tensors"]

    def f32(self, name):
        import numpy as np
        meta = self.tensors.get(name)
        if meta is None:
            fail("missing tensor %s in the raw payload" % name)
        if meta["dtype"] != "BF16":
            fail("%s is %s, expected BF16" % (name, meta["dtype"]))
        with open(os.path.join(self.root, meta["file"]), "rb") as f:
            raw = f.read()
        if len(raw) != meta["bytes"]:
            fail("%s is %d bytes, manifest says %d" % (name, len(raw), meta["bytes"]))
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(meta["shape"])


def build_plan(raw, routed_qtype=None):
    """(gguf_name, float32 array, qtype).  GGUF dim order is reversed from the
    HF [out, in] convention: ne0 is the row length, i.e. the input dim."""
    P = "mtp.layers.0."
    plan = []

    def add(name, arr, qtype):
        plan.append((name, arr, qtype))

    def gemma(arr):
        """GemmaRMSNorm affine: the kernels apply the stored vector directly,
        so bake the +1 exactly like the target GGUF does."""
        return arr.astype("float32") + 1.0

    # --- hyper-connection pair (attn and ffn) -----------------------------
    for hf, gg in (("attn_hyper_connection", "hc_attn"),
                   ("mlp_hyper_connection", "hc_ffn")):
        add("blk.0.%s_norm.weight" % gg, gemma(raw.f32(P + hf + ".hc_norm.weight")), QTYPE_F32)
        add("blk.0.%s_down.weight" % gg,
            raw.f32(P + hf + ".input_mix_weight_down.weight"), QTYPE_Q8_0)
        add("blk.0.%s_up.weight" % gg,
            raw.f32(P + hf + ".input_mix_weight_up.weight"), QTYPE_Q8_0)
        add("blk.0.%s_inject.weight" % gg,
            raw.f32(P + hf + ".block_inject_weight.weight"), QTYPE_Q8_0)

    # --- full attention ----------------------------------------------------
    A = P + "self_attn."
    add("blk.0.attn_q.weight", raw.f32(A + "q_proj.weight"), QTYPE_Q8_0)
    add("blk.0.attn_k.weight", raw.f32(A + "k_proj.weight"), QTYPE_Q8_0)
    add("blk.0.attn_v.weight", raw.f32(A + "v_proj.weight"), QTYPE_Q8_0)
    add("blk.0.attn_q_norm.weight", gemma(raw.f32(A + "q_norm.weight")), QTYPE_F32)
    add("blk.0.attn_k_norm.weight", gemma(raw.f32(A + "k_norm.weight")), QTYPE_F32)
    add("blk.0.attn_output.weight", raw.f32(A + "o_proj.weight"), QTYPE_Q8_0)

    # The indexer ships one fused projection feeding q and k; ds4 (like
    # llama.cpp) binds them separately.  q takes 4 heads x 128, k takes 1 x 128.
    qk = raw.f32(A + "indexer.index_qk_proj.weight")
    q_rows = IDX_HEADS * IDX_HEAD_DIM
    k_rows = IDX_KV_HEADS * IDX_HEAD_DIM
    if qk.shape[0] != q_rows + k_rows:
        fail("indexer qk proj has %d rows, expected %d" % (qk.shape[0], q_rows + k_rows))
    add("blk.0.indexer.q_proj.weight", qk[:q_rows], QTYPE_Q8_0)
    add("blk.0.indexer.k_proj.weight", qk[q_rows:], QTYPE_Q8_0)
    add("blk.0.indexer.q_norm.weight", gemma(raw.f32(A + "indexer.q_layernorm.weight")), QTYPE_F32)
    add("blk.0.indexer.k_norm.weight", gemma(raw.f32(A + "indexer.k_layernorm.weight")), QTYPE_F32)

    # --- MoE ---------------------------------------------------------------
    M = P + "mlp."
    add("blk.0.ffn_gate_inp.weight", raw.f32(M + "gate.weight"), QTYPE_Q8_0)
    add("blk.0.ffn_gate_inp_shexp.weight",
        raw.f32(M + "shared_expert_gate.weight"), QTYPE_Q8_0)

    # gate_up_proj is [n_expert, 2*ff, n_embd] with gate first, matching
    # vLLM's packed_modules_mapping {"gate_up_proj": ["gate_proj", "up_proj"]}.
    gu = raw.f32(M + "experts.gate_up_proj")
    if gu.shape != (N_EXPERT, 2 * N_FF_EXP, N_EMBD):
        fail("gate_up_proj shape %s unexpected" % (gu.shape,))
    # Routed gate/up copy the target's Q4_K by default; --routed-q8 builds
    # the all-Q8_0 variant for the acceptance A/B (about 3.3 GiB).
    rq = QTYPE_Q4_K if routed_qtype is None else routed_qtype
    add("blk.0.ffn_gate_exps.weight", gu[:, :N_FF_EXP, :], rq)
    add("blk.0.ffn_up_exps.weight", gu[:, N_FF_EXP:, :], rq)

    dn = raw.f32(M + "experts.down_proj")
    if dn.shape != (N_EXPERT, N_EMBD, N_FF_EXP):
        fail("down_proj shape %s unexpected" % (dn.shape,))
    # The target model uses Q5_1 here, but quants.c cannot produce Q5_1 (it
    # quantizes to Q8_0/Q8_K/Q2_K/Q4_K/IQ2_XXS only).  Q8_0 costs 262 MiB more
    # and is strictly more accurate, which only helps draft acceptance; ds4's
    # routed down path already dispatches on the type.
    add("blk.0.ffn_down_exps.weight", dn, QTYPE_Q8_0)

    add("blk.0.ffn_gate_shexp.weight", raw.f32(M + "shared_expert.gate_proj.weight"), QTYPE_Q8_0)
    add("blk.0.ffn_up_shexp.weight", raw.f32(M + "shared_expert.up_proj.weight"), QTYPE_Q8_0)
    add("blk.0.ffn_down_shexp.weight", raw.f32(M + "shared_expert.down_proj.weight"), QTYPE_Q8_0)

    # --- MTP-only projections ---------------------------------------------
    add("mtp.pre_fc_norm_embedding.weight", gemma(raw.f32("mtp.pre_fc_norm_embedding.weight")), QTYPE_F32)
    add("mtp.pre_fc_norm_hidden.weight", gemma(raw.f32("mtp.pre_fc_norm_hidden.weight")), QTYPE_F32)
    add("mtp.fc_embedding.weight", raw.f32("mtp.fc_embedding.weight"), QTYPE_Q8_0)
    add("mtp.fc_hidden.weight", raw.f32("mtp.fc_hidden.weight"), QTYPE_Q8_0)

    # The final mixer collapses the four streams; it has no inject, exactly
    # like the target model's output_mix (vLLM maps its block_inject_weight to
    # None).  ds4's q4e_hc_mix takes NULL there.
    H = "mtp.hyper_connection_mixer."
    add("mtp.hc_norm.weight", gemma(raw.f32(H + "hc_norm.weight")), QTYPE_F32)
    add("mtp.hc_down.weight", raw.f32(H + "input_mix_weight_down.weight"), QTYPE_Q8_0)
    add("mtp.hc_up.weight", raw.f32(H + "input_mix_weight_up.weight"), QTYPE_Q8_0)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(__file__),
                                                  "libds4quants.so"))
    ap.add_argument("--routed-q8", action="store_true",
                    help="quantize the routed gate/up experts as Q8_0 instead of Q4_K")
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        fail("%s not found; run `make -C gguf-tools libds4quants.so`" % args.lib)

    raw = Raw(args.raw)
    quant = Quantizer(args.lib)
    plan = build_plan(raw, QTYPE_Q8_0 if args.routed_q8 else None)

    kv = b"".join([
        kv_string("general.architecture", "qwen4exp-mtp"),
        kv_string("general.name", "Qwen3.8 Flash Next MTP"),
        kv_string("general.source.repo", raw.manifest.get("repo", "")),
        kv_u32("qwen4exp-mtp.block_count", 1),
        kv_u32("qwen4exp-mtp.embedding_length", N_EMBD),
        kv_u32("qwen4exp-mtp.expert_count", N_EXPERT),
        kv_u32("qwen4exp-mtp.expert_used_count", 10),
        kv_u32("qwen4exp-mtp.expert_feed_forward_length", N_FF_EXP),
        kv_u32("qwen4exp-mtp.attention.head_count", N_HEAD),
        kv_u32("qwen4exp-mtp.attention.head_count_kv", N_HEAD_KV),
        kv_u32("qwen4exp-mtp.attention.key_length", HEAD_DIM),
        kv_u32("qwen4exp-mtp.hyper_connection.count", N_HC),
        kv_u32("qwen4exp-mtp.hyper_connection.lowrank", N_HC_LOWRANK),
        kv_f32("qwen4exp-mtp.attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32("general.quantization_version", 2),
    ])
    n_kv = 15

    # Two passes: the header needs every tensor's offset, so quantize first.
    sys.stderr.write("quantizing %d tensors\n" % len(plan))
    blobs = []
    infos = b""
    offset = 0
    for name, arr, qtype in plan:
        payload = quant.quantize(arr, qtype)
        # GGUF dims are reversed relative to the HF [.., out, in] order.
        dims = list(reversed(list(arr.shape)))
        infos += pack_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", qtype, offset)
        blobs.append(payload)
        offset = align(offset + len(payload))
        sys.stderr.write("  %-38s %-22s %-5s %8.2f MiB\n"
                         % (name, dims, QTYPE_NAMES[qtype], len(payload) / 1048576.0))

    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(plan), n_kv) + kv + infos
    data_start = align(len(header))

    with open(args.out, "wb") as f:
        f.write(header)
        f.write(b"\0" * (data_start - len(header)))
        for payload in blobs:
            f.write(payload)
            pad = align(len(payload)) - len(payload)
            if pad:
                f.write(b"\0" * pad)

    total = os.path.getsize(args.out)
    sys.stderr.write("wrote %s (%.2f GiB, %d tensors)\n"
                     % (args.out, total / (1 << 30), len(plan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
