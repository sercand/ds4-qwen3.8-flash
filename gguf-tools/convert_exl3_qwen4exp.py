#!/usr/bin/env python3
"""Repack an EXL3 Qwen3.8-Flash-Next checkpoint (exllamav3 safetensors) as GGUF for ds4.

This is a container conversion, not a requantization: every trellis tile,
input/output scale vector (suh/svh), fp16 and bf16 weight is copied byte for
byte.  What changes is only what the GGUF convention already asks of any
qwen4exp file, exactly as llama.cpp's converter does for the bf16 source:

  * norms that the graph applies as (1 + w) are stored with the +1 folded in
    (hyper-connection, q/k, indexer, PLE, MTP pre-fc norms), as f32;
  * A_log is stored as ssm_a = -exp(A_log), dt_bias / conv1d / small norms as f32;
  * gated-DeltaNet value heads are permuted from HF's grouped order to ggml's
    tiled order.  For trellis tensors that is a permutation of whole 128-wide
    Hadamard blocks (8 tile columns), so the reconstructed weights are the
    original rows in the new order, bit for bit;
  * the indexer's fused index_qk_proj is split into q_proj / k_proj at a
    128-column boundary (same argument);
  * the 512 expert tensors of a projection are stacked into one tensor.

Custom GGUF tensor types (ds4 only; ids well above ggml's):

  68/69/70  exl3_k4/k5/k6  trellis, one 16x16 tile (256 weights) per 32*K bytes.
            Payload = [E][k/16][n/16][16K] u16 tiles, then [E][k] f16 suh, then
            [E][n] f16 svh.  dims are GGUF-style [k, n(, E)].
  72        exl3_ngram6    n-gram table rows: f16 scale word + 160 x 6-bit
            tail-biting ring = 122 bytes per 160-weight row.  The per-head bias
            goes in per_layer_token_embd.bias (f16 [160, 16]).

Usage:
  python convert_exl3_qwen4exp.py <exl3 dir> <out.gguf> [--mtp-out mtp.gguf]
        [--no-ngram] [--layers N] [--check-gguf <UD gguf shard 1>]

Runs on CPU only (numpy + torch for bf16 reads); streams tensor by tensor.
"""
import argparse, json, os, struct, sys, time
import numpy as np
import torch
from safetensors import safe_open

# ---------------------------------------------------------------- GGUF writer

GGUF_MAGIC = b"GGUF"
ALIGN = 32
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
# ggml tensor types
G_F32, G_F16, G_BF16 = 0, 1, 30
# ds4 exl3 types
G_EXL3_K4, G_EXL3_K5, G_EXL3_K6, G_EXL3_NGRAM6 = 68, 69, 70, 72
EXL3_TYPE = {4: G_EXL3_K4, 5: G_EXL3_K5, 6: G_EXL3_K6}


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_bytes(key, vtype, value):
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == T_STR:
        out += _str(value)
    elif vtype == T_ARR:
        etype, items = value
        out += struct.pack("<IQ", etype, len(items))
        if etype == T_STR:
            out += b"".join(_str(s) for s in items)
        else:
            fmt = {T_U32: "<I", T_I32: "<i", T_U64: "<Q", T_I64: "<q", T_F32: "<f"}[etype]
            out += b"".join(struct.pack(fmt, v) for v in items)
    else:
        fmt = {T_U32: "<I", T_I32: "<i", T_U64: "<Q", T_I64: "<q", T_F32: "<f", T_BOOL: "<B",
               T_U8: "<B"}[vtype]
        out += struct.pack(fmt, value)
    return out


class TensorInfo:
    __slots__ = ("name", "dims", "gtype", "nbytes", "offset", "producer")

    def __init__(self, name, dims, gtype, nbytes, producer):
        self.name, self.dims, self.gtype, self.nbytes, self.producer = name, dims, gtype, nbytes, producer
        self.offset = 0


class GGUFWriter:
    """Two-pass writer: register every tensor with its byte size and a producer
    callable, then write().  Producers yield bytes-like chunks; data is streamed
    so the 100 GB output never sits in memory."""

    def __init__(self, path):
        self.path = path
        self.kv = []
        self.tensors = []

    def add_kv(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def add_tensor(self, name, dims, gtype, nbytes, producer):
        assert nbytes > 0, name
        self.tensors.append(TensorInfo(name, list(dims), gtype, nbytes, producer))

    def write(self, verbose=True):
        # layout
        off = 0
        for t in self.tensors:
            t.offset = off
            off += (t.nbytes + ALIGN - 1) // ALIGN * ALIGN
        header = bytearray()
        header += GGUF_MAGIC + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        for k, vt, v in self.kv:
            header += kv_bytes(k, vt, v)
        for t in self.tensors:
            header += _str(t.name) + struct.pack("<I", len(t.dims))
            header += b"".join(struct.pack("<Q", d) for d in t.dims)
            header += struct.pack("<IQ", t.gtype, t.offset)
        data_pos = (len(header) + ALIGN - 1) // ALIGN * ALIGN
        header += b"\0" * (data_pos - len(header))

        t0 = time.time()
        done = 0
        total = off
        with open(self.path, "wb") as f:
            f.write(header)
            fd = f.fileno()
            for i, t in enumerate(self.tensors):
                pos = data_pos + t.offset
                assert f.tell() <= pos
                f.write(b"\0" * (pos - f.tell()))
                n = 0
                for chunk in t.producer():
                    mv = memoryview(chunk).cast("B")
                    f.write(mv)
                    n += len(mv)
                assert n == t.nbytes, f"{t.name}: produced {n} bytes, declared {t.nbytes}"
                done += t.nbytes
                if verbose and (i % 25 == 0 or t.nbytes > 1 << 30):
                    el = time.time() - t0
                    print(f"  [{i + 1}/{len(self.tensors)}] {t.name} {t.nbytes / 1e6:.1f} MB  "
                          f"{done / 1e9:.1f}/{total / 1e9:.1f} GB  {done / 1e9 / max(el, 1e-9):.2f} GB/s",
                          flush=True)
                if done % (8 << 30) < t.nbytes:
                    f.flush()
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
            f.write(b"\0" * (data_pos + off - f.tell()))
        return data_pos + off


# ---------------------------------------------------------- safetensors source

class Source:
    """All shards of the checkpoint behind one name -> (shard, dtype, shape) map."""

    def __init__(self, d):
        self.dir = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        self.shard_of = idx["weight_map"]
        self.files = {}
        self.meta = {}
        for shard in sorted(set(self.shard_of.values())):
            path = os.path.join(d, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    self.meta[k] = (shard, v["dtype"], tuple(v["shape"]))
        self.ngram_path = os.path.join(d, "ngram_embedding.safetensors")

    def handle(self, shard):
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
        return self.files[shard]

    def info(self, name):
        return self.meta[name]

    def has(self, name):
        return name in self.meta

    def get(self, name):
        shard, _, _ = self.meta[name]
        return self.handle(shard).get_tensor(name)

    def drop_cache(self, shard):
        """Release page cache for a shard we are done with."""
        try:
            fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def as_u16(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def as_f16(t):
    return t.contiguous().to(torch.float16).numpy()


def to_f32(t):
    return t.contiguous().to(torch.float32).numpy()


# ---------------------------------------------------------- transforms

def v_head_perm(num_k, num_v, blocks_per_head=1):
    """Grouped (HF) -> tiled (ggml) order of the V heads, at `blocks_per_head`
    granularity: new position r*num_k + kh takes old head kh*per_k + r."""
    per_k = num_v // num_k
    perm = np.empty(num_v, dtype=np.int64)
    for kh in range(num_k):
        for r in range(per_k):
            perm[r * num_k + kh] = kh * per_k + r
    if blocks_per_head == 1:
        return perm
    return (perm[:, None] * blocks_per_head + np.arange(blocks_per_head)[None, :]).reshape(-1)


class Exl3Tensor:
    """One exl3 linear: trellis (k/16, n/16, 16K) u16, suh (k) f16, svh (n) f16."""

    def __init__(self, src, prefix):
        self.src, self.prefix = src, prefix
        shard, dt, shape = src.info(prefix + ".trellis")
        assert dt == "I16" and len(shape) == 3, (prefix, dt, shape)
        self.k, self.n, self.K = shape[0] * 16, shape[1] * 16, shape[2] // 16
        assert shape[2] == 16 * self.K and self.K in EXL3_TYPE, (prefix, shape)
        assert src.info(prefix + ".suh")[2] == (self.k,) and src.info(prefix + ".svh")[2] == (self.n,)
        assert self.k % 16 == 0 and self.n % 128 == 0, (prefix, self.k, self.n)

    @property
    def tile_bytes(self):
        return self.k * self.n * self.K // 8

    def load(self):
        return (as_u16(self.src.get(self.prefix + ".trellis")),
                as_f16(self.src.get(self.prefix + ".suh")),
                as_f16(self.src.get(self.prefix + ".svh")))


def exl3_payload_bytes(k, n, K, E=1):
    return E * (k * n * K // 8 + 2 * k + 2 * n)


def emit_exl3(w, name, tensors, n_perm128=None, k_perm128=None, n_slice=None, out_name=None):
    """Register one dense exl3 tensor.  n_perm128 / k_perm128 permute whole
    128-wide blocks along n / k (values are block indices); n_slice = (lo, hi)
    keeps output columns lo..hi (multiples of 128)."""
    k, n, K = w.k, w.n, w.K
    lo, hi = (0, n) if n_slice is None else n_slice
    assert lo % 128 == 0 and hi % 128 == 0 and 0 <= lo < hi <= n
    n_out = hi - lo

    def produce():
        tr, suh, svh = w.load()
        tr = tr.reshape(k // 16, n // 16, 16 * K)
        if n_slice is not None:
            tr = tr[:, lo // 16: hi // 16]
            svh = svh[lo:hi]
        if n_perm128 is not None:
            tr = tr.reshape(k // 16, n_out // 128, 8, 16 * K)[:, n_perm128].reshape(k // 16, n_out // 16, 16 * K)
            svh = svh.reshape(n_out // 128, 128)[n_perm128].reshape(-1)
        if k_perm128 is not None:
            tr = tr.reshape(k // 128, 8, n_out // 16, 16 * K)[k_perm128].reshape(k // 16, n_out // 16, 16 * K)
            suh = suh.reshape(k // 128, 128)[k_perm128].reshape(-1)
        yield np.ascontiguousarray(tr)
        yield np.ascontiguousarray(suh)
        yield np.ascontiguousarray(svh)

    tensors.add_tensor(out_name or name, [k, n_out], EXL3_TYPE[K], exl3_payload_bytes(k, n_out, K), produce)


def emit_expert_stack(src, layer_prefix, proj, name, n_expert, tensors):
    """Stack experts.{0..E-1}.<proj> into one [k, n, E] exl3 tensor."""
    first = Exl3Tensor(src, f"{layer_prefix}.experts.0.{proj}")
    k, n, K = first.k, first.n, first.K

    def produce():
        tiles = np.empty((n_expert, k // 16, n // 16, 16 * K), dtype=np.uint16)
        suh = np.empty((n_expert, k), dtype=np.float16)
        svh = np.empty((n_expert, n), dtype=np.float16)
        for e in range(n_expert):
            w = Exl3Tensor(src, f"{layer_prefix}.experts.{e}.{proj}")
            assert (w.k, w.n, w.K) == (k, n, K), (layer_prefix, proj, e)
            tr, su, sv = w.load()
            tiles[e] = tr.reshape(k // 16, n // 16, 16 * K)
            suh[e] = su
            svh[e] = sv
        yield tiles
        yield suh
        yield svh

    tensors.add_tensor(name, [k, n, n_expert], EXL3_TYPE[K], exl3_payload_bytes(k, n, K, n_expert), produce)


def emit_raw(src, hf_name, gguf_name, tensors, dims=None, perm_rows=None):
    """Copy an fp16/bf16 tensor verbatim (optionally permuting rows)."""
    shard, dt, shape = src.info(hf_name)
    gtype = {"F16": G_F16, "BF16": G_BF16}[dt]
    if dims is None:
        dims = list(reversed(shape))          # GGUF ne order: innermost first
    nbytes = int(np.prod(shape)) * 2

    def produce():
        t = src.get(hf_name)
        if perm_rows is not None:
            t = t[torch.as_tensor(perm_rows)]
        yield t.contiguous().view(torch.int16).numpy()

    tensors.add_tensor(gguf_name, dims, gtype, nbytes, produce)


def emit_f32(src, hf_name, gguf_name, tensors, fn=None, dims=None):
    """Store as f32 after an exact convention transform (fn on a float32 array)."""
    shard, dt, shape = src.info(hf_name)
    n = int(np.prod(shape))
    if dims is None:
        dims = list(reversed([d for d in shape if d != 1]))   # squeeze the conv1d's middle 1

    def produce():
        a = to_f32(src.get(hf_name)).reshape(-1)
        if fn is not None:
            a = fn(a)
        yield np.ascontiguousarray(a.astype(np.float32))

    tensors.add_tensor(gguf_name, dims, G_F32, n * 4, produce)


# ---------------------------------------------------------- the model

def load_cfg(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    return cfg, cfg["text_config"]


def add_model_kv(w, cfg, tc, d, ngram):
    ple_layers = [i - 1 for i in tc["ple_layer_ids"]]
    full = [t == "full_attention" for t in tc["layer_types"]]
    w.add_kv("general.architecture", T_STR, "qwen4exp")
    w.add_kv("general.type", T_STR, "model")
    w.add_kv("general.name", T_STR, "Qwen3.8 Flash Next EXL3 4.05bpw (exllamav3 container repack)")
    w.add_kv("general.file_type", T_U32, 1)
    w.add_kv("qwen4exp.block_count", T_U32, tc["num_hidden_layers"])
    w.add_kv("qwen4exp.context_length", T_U32, tc["max_position_embeddings"])
    w.add_kv("qwen4exp.embedding_length", T_U32, tc["hidden_size"])
    w.add_kv("qwen4exp.attention.head_count", T_U32, tc["num_attention_heads"])
    w.add_kv("qwen4exp.attention.head_count_kv", T_U32, tc["num_key_value_heads"])
    w.add_kv("qwen4exp.rope.dimension_sections", T_ARR, (T_I32, list(tc["rope_parameters"]["mrope_section"]) + [0]))
    w.add_kv("qwen4exp.rope.freq_base", T_F32, float(tc["rope_parameters"]["rope_theta"]))
    w.add_kv("qwen4exp.attention.layer_norm_rms_epsilon", T_F32, float(tc["rms_norm_eps"]))
    w.add_kv("qwen4exp.expert_count", T_U32, tc["num_experts"])
    w.add_kv("qwen4exp.expert_used_count", T_U32, tc["num_experts_per_tok"])
    w.add_kv("qwen4exp.attention.key_length", T_U32, tc["head_dim"])
    w.add_kv("qwen4exp.attention.value_length", T_U32, tc["head_dim"])
    w.add_kv("qwen4exp.expert_feed_forward_length", T_U32, tc["moe_intermediate_size"])
    w.add_kv("qwen4exp.expert_shared_feed_forward_length", T_U32, tc["shared_expert_intermediate_size"])
    w.add_kv("qwen4exp.ssm.conv_kernel", T_U32, tc["linear_conv_kernel_dim"])
    w.add_kv("qwen4exp.ssm.state_size", T_U32, tc["linear_key_head_dim"])
    w.add_kv("qwen4exp.ssm.group_count", T_U32, tc["linear_num_key_heads"])
    w.add_kv("qwen4exp.ssm.time_step_rank", T_U32, tc["linear_num_value_heads"])
    w.add_kv("qwen4exp.ssm.inner_size", T_U32, tc["linear_num_value_heads"] * tc["linear_value_head_dim"])
    w.add_kv("qwen4exp.full_attention_interval", T_U32, tc["full_attention_interval"])
    w.add_kv("qwen4exp.rope.dimension_count", T_U32, int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"]))
    w.add_kv("qwen4exp.hyper_connection.count", T_U32, tc["hc_count"])
    w.add_kv("qwen4exp.hyper_connection.low_rank", T_U32, tc["hc_lowrank"])
    w.add_kv("qwen4exp.attention.indexer.head_count", T_U32, tc["indexer_n_heads"])
    w.add_kv("qwen4exp.attention.indexer.key_length", T_U32, tc["indexer_head_dim"])
    w.add_kv("qwen4exp.attention.indexer.top_k", T_U32, tc["indexer_budget"])
    w.add_kv("qwen4exp.attention.compress_ratios", T_ARR,
             (T_U32, [tc["indexer_compress_ratio"] if f else 0 for f in full]))
    w.add_kv("qwen4exp.ple.layers", T_ARR, (T_U32, ple_layers))
    w.add_kv("qwen4exp.ple.ngram_size", T_U32, tc["ngram_size"])
    w.add_kv("qwen4exp.ple.heads_per_ngram", T_U32, tc["heads_per_ngram"])
    w.add_kv("qwen4exp.ple.conv_kernel", T_U32, tc["ple_conv_kernel_size"])
    w.add_kv("qwen4exp.ple.eos_token_id", T_U32, tc["eos_token_id"])
    w.add_kv("qwen4exp.ple.image_token_id", T_U32, cfg["image_token_id"])
    w.add_kv("qwen4exp.embedding_length_per_layer_input", T_U32, ngram["row_dim"])
    w.add_kv("qwen4exp.ple.layer_multipliers", T_ARR, (T_U64, [int(x) for x in ngram["multipliers"]]))
    w.add_kv("qwen4exp.ple.head_offsets", T_ARR, (T_U64, [int(x) for x in ngram["offsets"]]))
    w.add_kv("qwen4exp.ple.head_vocab_sizes", T_ARR, (T_U64, [int(x) for x in ngram["vocab"]]))
    add_tokenizer_kv(w, d, tc)


def add_tokenizer_kv(w, d, tc):
    tok = json.load(open(os.path.join(d, "tokenizer.json")))
    vocab = tok["model"]["vocab"]
    n_vocab = tc["vocab_size"]
    tokens = [None] * n_vocab
    types = [1] * n_vocab                     # NORMAL
    for s, i in vocab.items():
        tokens[i] = s
    for a in tok["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a["special"] else 4   # CONTROL / USER_DEFINED
    for i in range(n_vocab):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = 5                       # UNUSED
    merges = tok["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [" ".join(m) for m in merges]
    gen = json.load(open(os.path.join(d, "generation_config.json")))
    eos = gen["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    w.add_kv("tokenizer.ggml.model", T_STR, "gpt2")
    w.add_kv("tokenizer.ggml.pre", T_STR, "qwen35")
    w.add_kv("tokenizer.ggml.tokens", T_ARR, (T_STR, tokens))
    w.add_kv("tokenizer.ggml.token_type", T_ARR, (T_I32, types))
    w.add_kv("tokenizer.ggml.merges", T_ARR, (T_STR, merges))
    w.add_kv("tokenizer.ggml.eos_token_id", T_U32, eos)
    w.add_kv("tokenizer.ggml.padding_token_id", T_U32, gen["pad_token_id"])
    w.add_kv("tokenizer.ggml.bos_token_id", T_U32, gen["bos_token_id"])
    w.add_kv("tokenizer.ggml.add_bos_token", T_BOOL, 0)
    tmpl = os.path.join(d, "chat_template.jinja")
    if os.path.exists(tmpl):
        w.add_kv("tokenizer.chat_template", T_STR, open(tmpl).read())


def read_ngram_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        meta = h["__metadata__"]
        assert meta["format"] == "exl3_ngram_trellis" and meta["version"] == "1" and meta["codebook"] == "mul1"
        K, row_dim = int(meta["K"]), int(meta["row_dim"])
        assert K == 6 and row_dim == 160, meta
        pfx = [k for k in h if k.endswith(".trellis")][0][: -len("trellis")]

        def small(name, dt):
            a, b = h[pfx + name]["data_offsets"]
            f.seek(base + a)
            return np.frombuffer(f.read(b - a), dtype=dt)
        tr = h[pfx + "trellis"]
        rows, words = tr["shape"]
        assert words == 1 + row_dim * K // 16 == 61 and tr["dtype"] == "I16"
        return {
            "K": K, "row_dim": row_dim, "rows": rows, "row_bytes": words * 2,
            "data_off": base + tr["data_offsets"][0],
            "data_bytes": tr["data_offsets"][1] - tr["data_offsets"][0],
            "multipliers": small("layer_multipliers", "<i8"),
            "offsets": small("head_offsets", "<i8"),
            "vocab": small("head_vocab_sizes", "<i8"),
            "head_bias": small("head_bias", "<f2").reshape(h[pfx + "head_bias"]["shape"]),
        }


def add_block(src, w, pfx, blk, il, full_attn, tc, with_ple):
    """One decoder block under HF prefix `pfx` as GGUF blk.<blk>.*."""
    H, D = tc["hc_count"], tc["hidden_size"]
    n_k, n_v, hd = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_value_head_dim"]
    plus1 = lambda a: a + 1.0

    for site, g in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        emit_f32(src, f"{pfx}.{site}.hc_norm.weight", f"blk.{blk}.{g}_norm.weight", w, plus1)
        emit_raw(src, f"{pfx}.{site}.input_mix_weight_down.weight", f"blk.{blk}.{g}_down.weight", w)
        emit_raw(src, f"{pfx}.{site}.input_mix_weight_up.weight", f"blk.{blk}.{g}_up.weight", w)
        emit_raw(src, f"{pfx}.{site}.block_inject_weight.weight", f"blk.{blk}.{g}_inject.weight", w)

    if full_attn:
        a = f"{pfx}.self_attn"
        for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            emit_exl3(Exl3Tensor(src, f"{a}.{hf}"), f"blk.{blk}.{gg}.weight", w)
        emit_f32(src, f"{a}.q_norm.weight", f"blk.{blk}.attn_q_norm.weight", w, plus1)
        emit_f32(src, f"{a}.k_norm.weight", f"blk.{blk}.attn_k_norm.weight", w, plus1)
        idx = Exl3Tensor(src, f"{a}.indexer.index_qk_proj")
        n_q = tc["indexer_n_heads"] * tc["indexer_head_dim"]
        emit_exl3(idx, None, w, n_slice=(0, n_q), out_name=f"blk.{blk}.indexer.q_proj.weight")
        emit_exl3(idx, None, w, n_slice=(n_q, idx.n), out_name=f"blk.{blk}.indexer.k_proj.weight")
        emit_f32(src, f"{a}.indexer.q_layernorm.weight", f"blk.{blk}.indexer.q_norm.weight", w, plus1)
        emit_f32(src, f"{a}.indexer.k_layernorm.weight", f"blk.{blk}.indexer.k_norm.weight", w, plus1)
    else:
        la = f"{pfx}.linear_attn"
        perm_heads = v_head_perm(n_k, n_v)                       # 48 head ids
        qk_blocks = 2 * n_k * tc["linear_key_head_dim"] // 128   # q|k blocks stay in place
        perm_qkv = np.concatenate([np.arange(qk_blocks), qk_blocks + perm_heads])
        emit_exl3(Exl3Tensor(src, f"{la}.in_proj_qkv"), f"blk.{blk}.attn_qkv.weight", w, n_perm128=perm_qkv)
        emit_exl3(Exl3Tensor(src, f"{la}.in_proj_z"), f"blk.{blk}.attn_gate.weight", w, n_perm128=perm_heads)
        emit_exl3(Exl3Tensor(src, f"{la}.out_proj"), f"blk.{blk}.ssm_out.weight", w, k_perm128=perm_heads)
        emit_raw(src, f"{la}.in_proj_a.weight", f"blk.{blk}.ssm_alpha.weight", w, perm_rows=perm_heads)
        emit_raw(src, f"{la}.in_proj_b.weight", f"blk.{blk}.ssm_beta.weight", w, perm_rows=perm_heads)
        emit_f32(src, f"{la}.A_log", f"blk.{blk}.ssm_a", w, lambda a: -np.exp(a)[perm_heads])
        emit_f32(src, f"{la}.dt_bias", f"blk.{blk}.ssm_dt.bias", w, lambda a: a[perm_heads])
        emit_f32(src, f"{la}.norm.weight", f"blk.{blk}.ssm_norm.weight", w)
        conv_ch = 2 * n_k * tc["linear_key_head_dim"] + n_v * hd
        ch_perm = np.concatenate([np.arange(qk_blocks * 128), qk_blocks * 128 + v_head_perm(n_k, n_v, hd)])
        emit_f32(src, f"{la}.conv1d.weight", f"blk.{blk}.ssm_conv1d.weight", w,
                 lambda a, c=conv_ch: a.reshape(c, -1)[ch_perm].reshape(-1),
                 dims=[tc["linear_conv_kernel_dim"], conv_ch])

    if with_ple:
        p = f"{pfx}.ple"
        emit_raw(src, f"{p}.key_proj.weight", f"blk.{blk}.ple_key.weight", w)
        emit_raw(src, f"{p}.value_proj.weight", f"blk.{blk}.ple_value.weight", w)
        for nm in ("key", "query", "conv"):
            emit_f32(src, f"{p}.norm_{nm}.weight", f"blk.{blk}.ple_norm_{nm}.weight", w, plus1)
        emit_f32(src, f"{p}.conv1d.weight", f"blk.{blk}.ple_conv1d.weight", w,
                 dims=[tc["ple_conv_kernel_size"], H * D])

    m = f"{pfx}.mlp"
    emit_raw(src, f"{m}.gate.weight", f"blk.{blk}.ffn_gate_inp.weight", w)
    emit_raw(src, f"{m}.shared_expert_gate.weight", f"blk.{blk}.ffn_gate_inp_shexp.weight", w, dims=[D])
    for hf, gg in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
        emit_expert_stack(src, m, hf, f"blk.{blk}.{gg}.weight", tc["num_experts"], w)
    for hf, gg in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")):
        emit_exl3(Exl3Tensor(src, f"{m}.shared_expert.{hf}"), f"blk.{blk}.{gg}.weight", w)


def build_model(src, out_path, cfg, tc, ngram, n_layers, with_ngram):
    w = GGUFWriter(out_path)
    add_model_kv(w, cfg, tc, src.dir, ngram)
    lm = "model.language_model"
    plus1 = lambda a: a + 1.0
    emit_raw(src, f"{lm}.embed_tokens.weight", "token_embd.weight", w)
    emit_exl3(Exl3Tensor(src, "lm_head"), "output.weight", w)
    emit_f32(src, f"{lm}.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight", w, plus1)
    emit_raw(src, f"{lm}.hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight", w)
    emit_raw(src, f"{lm}.hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight", w)
    ple_layers = {i - 1 for i in tc["ple_layer_ids"]}
    for il in range(n_layers):
        add_block(src, w, f"{lm}.layers.{il}", il, il, tc["layer_types"][il] == "full_attention", tc, il in ple_layers)
    if with_ngram:
        def produce_table():
            with open(ngram["path"], "rb") as f:
                f.seek(ngram["data_off"])
                left = ngram["data_bytes"]
                while left:
                    b = f.read(min(left, 64 << 20))
                    assert b
                    left -= len(b)
                    yield b
        w.add_tensor("per_layer_token_embd.weight", [ngram["row_dim"], ngram["rows"]], G_EXL3_NGRAM6,
                     ngram["data_bytes"], produce_table)
        hb = ngram["head_bias"]
        w.add_tensor("per_layer_token_embd.bias", [hb.shape[1], hb.shape[0]], G_F16, hb.nbytes,
                     lambda: [np.ascontiguousarray(hb)])
    return w


def build_mtp(src, out_path, cfg, tc):
    w = GGUFWriter(out_path)
    w.add_kv("general.architecture", T_STR, "qwen4exp-mtp")
    w.add_kv("general.type", T_STR, "model")
    w.add_kv("general.name", T_STR, "Qwen3.8 Flash Next EXL3 MTP head (exllamav3 container repack)")
    w.add_kv("qwen4exp-mtp.block_count", T_U32, tc["mtp"]["num_hidden_layers"])
    w.add_kv("qwen4exp-mtp.embedding_length", T_U32, tc["hidden_size"])
    plus1 = lambda a: a + 1.0
    emit_f32(src, "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding.weight", w, plus1)
    emit_f32(src, "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight", w, plus1)
    emit_exl3(Exl3Tensor(src, "mtp.fc_embedding"), "mtp.fc_embedding.weight", w)
    emit_exl3(Exl3Tensor(src, "mtp.fc_hidden"), "mtp.fc_hidden.weight", w)
    emit_f32(src, "mtp.hyper_connection_mixer.hc_norm.weight", "mtp.hc_norm.weight", w, plus1)
    emit_raw(src, "mtp.hyper_connection_mixer.input_mix_weight_down.weight", "mtp.hc_down.weight", w)
    emit_raw(src, "mtp.hyper_connection_mixer.input_mix_weight_up.weight", "mtp.hc_up.weight", w)
    for il in range(tc["mtp"]["num_hidden_layers"]):
        add_block(src, w, f"mtp.layers.{il}", il, il, True, tc, False)
    return w


def class_table(w):
    """Bytes per class, for the step-1 gate against the report's table."""
    cls = {}
    for t in w.tensors:
        n = t.name
        if t.gtype in (G_EXL3_K4, G_EXL3_K5, G_EXL3_K6):
            K = {G_EXL3_K4: 4, G_EXL3_K5: 5, G_EXL3_K6: 6}[t.gtype]
            key = f"exl3 K={K} {'routed' if '_exps.' in n else 'dense'}"
        elif t.gtype == G_EXL3_NGRAM6:
            key = "ngram rows"
        else:
            key = {G_F32: "f32", G_F16: "f16", G_BF16: "bf16"}[t.gtype]
        c = cls.setdefault(key, [0, 0])
        c[0] += 1
        c[1] += t.nbytes
    for k in sorted(cls):
        print(f"  {k:22s} {cls[k][0]:6d} tensors  {cls[k][1] / 1e9:9.3f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--mtp-out", default=None)
    ap.add_argument("--no-ngram", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="convert only the first N layers (smoke test)")
    ap.add_argument("--plan-only", action="store_true", help="print the class table and exit")
    args = ap.parse_args()

    cfg, tc = load_cfg(args.src)
    src = Source(args.src)
    ngram = read_ngram_header(src.ngram_path)
    ngram["path"] = src.ngram_path
    n_layers = args.layers or tc["num_hidden_layers"]
    print(f"config: {tc['num_hidden_layers']} layers, hidden {tc['hidden_size']}, experts {tc['num_experts']}, "
          f"vocab {tc['vocab_size']}; ngram rows {ngram['rows']} x {ngram['row_bytes']} B")

    w = build_model(src, args.out, cfg, tc, ngram, n_layers, not args.no_ngram)
    print(f"model: {len(w.tensors)} tensors, {sum(t.nbytes for t in w.tensors) / 1e9:.2f} GB")
    class_table(w)
    if args.mtp_out:
        wm = build_mtp(src, args.mtp_out, cfg, tc)
        print(f"mtp: {len(wm.tensors)} tensors, {sum(t.nbytes for t in wm.tensors) / 1e9:.3f} GB")
        class_table(wm)
    if args.plan_only:
        return
    t0 = time.time()
    n = w.write()
    print(f"wrote {args.out}: {n / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    if args.mtp_out:
        t0 = time.time()
        n = wm.write()
        print(f"wrote {args.mtp_out}: {n / 1e9:.3f} GB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
