#!/usr/bin/env python3
"""Byte-level round trip of convert_exl3_qwen4exp.py output against the exl3 safetensors.

    python check_exl3_gguf.py <exl3 dir> <model.gguf> [--mtp mtp.gguf] [--layers N]

Every check is exact (np.array_equal / bit compare): stacked experts, dense trellis
tensors, the value-head permutation of the GDN tensors, the indexer split, the
fp16/bf16 copies, the +1 norms, -exp(A_log), and the n-gram table's first and last
rows plus the head bias.  Prints one line per check and exits non-zero on the first
mismatch.
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_exl3_qwen4exp import (Source, read_ngram_header, v_head_perm, G_F32, G_F16, G_BF16,
                                   G_EXL3_K4, G_EXL3_K5, G_EXL3_K6, G_EXL3_NGRAM6)

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        u32 = lambda: struct.unpack("<I", f.read(4))[0]
        u64 = lambda: struct.unpack("<Q", f.read(8))[0]
        s = lambda: f.read(u64()).decode("utf-8", "replace")

        def skip(t):
            if t == 8:
                s()
            elif t == 9:
                et, n = u32(), u64()
                if et == 8:
                    for _ in range(n):
                        s()
                else:
                    f.read(SZ[et] * n)
            else:
                f.read(SZ[t])
        assert f.read(4) == b"GGUF" and u32() == 3
        nt, nkv = u64(), u64()
        for _ in range(nkv):
            s()
            skip(u32())
        self.tensors = {}
        for _ in range(nt):
            name = s()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            self.tensors[name] = (dims, u32(), u64())
        pos = f.tell()
        self.data_pos = (pos + 31) // 32 * 32

    def raw(self, name, nbytes=None, skip=0):
        dims, t, off = self.tensors[name]
        self.f.seek(self.data_pos + off + skip)
        return self.f.read(nbytes)

    def exl3(self, name):
        dims, t, off = self.tensors[name]
        K = {G_EXL3_K4: 4, G_EXL3_K5: 5, G_EXL3_K6: 6}[t]
        k, n = dims[0], dims[1]
        E = dims[2] if len(dims) == 3 else 1
        tiles = E * k * n * K // 8
        buf = self.raw(name, tiles + E * 2 * (k + n))
        tr = np.frombuffer(buf[:tiles], np.uint16).reshape(E, k // 16, n // 16, 16 * K)
        suh = np.frombuffer(buf[tiles: tiles + E * 2 * k], np.float16).reshape(E, k)
        svh = np.frombuffer(buf[tiles + E * 2 * k:], np.float16).reshape(E, n)
        return tr, suh, svh, K

    def f32(self, name):
        dims, t, off = self.tensors[name]
        assert t == G_F32, name
        return np.frombuffer(self.raw(name, int(np.prod(dims)) * 4), np.float32)

    def u16(self, name):
        dims, t, off = self.tensors[name]
        assert t in (G_F16, G_BF16), name
        return np.frombuffer(self.raw(name, int(np.prod(dims)) * 2), np.uint16)


def st_u16(src, name):
    return src.get(name).contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)


def st_f32(src, name):
    return src.get(name).to(torch.float32).numpy().reshape(-1)


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        sys.exit(1)


def check_exl3(g, src, gname, hf, perm_n=None, perm_k=None, n_slice=None):
    tr, suh, svh, K = g.exl3(gname)
    s_tr = st_u16(src, hf + ".trellis")
    s_suh = st_u16(src, hf + ".suh")
    s_svh = st_u16(src, hf + ".svh")
    k, n = s_suh.shape[0], s_svh.shape[0]
    s_tr = s_tr.reshape(k // 16, n // 16, -1)
    if n_slice:
        lo, hi = n_slice
        s_tr, s_svh, n = s_tr[:, lo // 16: hi // 16], s_svh[lo:hi], hi - lo
    if perm_n is not None:
        s_tr = s_tr.reshape(k // 16, n // 128, 8, -1)[:, perm_n].reshape(k // 16, n // 16, -1)
        s_svh = s_svh.reshape(n // 128, 128)[perm_n].reshape(-1)
    if perm_k is not None:
        s_tr = s_tr.reshape(k // 128, 8, n // 16, -1)[perm_k].reshape(k // 16, n // 16, -1)
        s_suh = s_suh.reshape(k // 128, 128)[perm_k].reshape(-1)
    ok = (np.array_equal(tr[0], s_tr) and np.array_equal(suh[0].view(np.uint16), s_suh)
          and np.array_equal(svh[0].view(np.uint16), s_svh) and K == s_tr.shape[-1] // 16)
    check(ok, f"{gname} == {hf} (K={K}{', perm' if perm_n is not None or perm_k is not None else ''}"
              f"{', slice' if n_slice else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("gguf")
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    src = Source(a.src)
    tc = json.load(open(os.path.join(a.src, "config.json")))["text_config"]
    g = GGUF(a.gguf)
    lm = "model.language_model"
    n_layers = a.layers or tc["num_hidden_layers"]
    n_k, n_v = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    perm = v_head_perm(n_k, n_v)
    qk_blocks = 2 * n_k * tc["linear_key_head_dim"] // 128
    perm_qkv = np.concatenate([np.arange(qk_blocks), qk_blocks + perm])

    check(np.array_equal(g.u16("token_embd.weight"), st_u16(src, f"{lm}.embed_tokens.weight")), "token_embd bf16 copy")
    check_exl3(g, src, "output.weight", "lm_head")
    check(np.array_equal(g.f32("output_hc_norm.weight"),
                         st_f32(src, f"{lm}.hyper_connection_mixer.hc_norm.weight") + 1.0), "output_hc_norm +1")
    check(np.array_equal(g.u16("output_hc_down.weight"),
                         st_u16(src, f"{lm}.hyper_connection_mixer.input_mix_weight_down.weight")), "output_hc_down f16")

    # a full sweep over every layer on the fp16/f32 side, exl3 spot checks per layer type
    for il in range(n_layers):
        p = f"{lm}.layers.{il}"
        full = tc["layer_types"][il] == "full_attention"
        for site, gn in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            assert np.array_equal(g.f32(f"blk.{il}.{gn}_norm.weight"), st_f32(src, f"{p}.{site}.hc_norm.weight") + 1.0)
            assert np.array_equal(g.u16(f"blk.{il}.{gn}_down.weight"), st_u16(src, f"{p}.{site}.input_mix_weight_down.weight"))
            assert np.array_equal(g.u16(f"blk.{il}.{gn}_up.weight"), st_u16(src, f"{p}.{site}.input_mix_weight_up.weight"))
            assert np.array_equal(g.u16(f"blk.{il}.{gn}_inject.weight"), st_u16(src, f"{p}.{site}.block_inject_weight.weight"))
        assert np.array_equal(g.u16(f"blk.{il}.ffn_gate_inp.weight"), st_u16(src, f"{p}.mlp.gate.weight"))
        assert np.array_equal(g.u16(f"blk.{il}.ffn_gate_inp_shexp.weight"), st_u16(src, f"{p}.mlp.shared_expert_gate.weight"))
        if full:
            assert np.array_equal(g.f32(f"blk.{il}.attn_q_norm.weight"), st_f32(src, f"{p}.self_attn.q_norm.weight") + 1.0)
            assert np.array_equal(g.f32(f"blk.{il}.indexer.k_norm.weight"), st_f32(src, f"{p}.self_attn.indexer.k_layernorm.weight") + 1.0)
        else:
            la = f"{p}.linear_attn"
            assert np.array_equal(g.f32(f"blk.{il}.ssm_a"), (-np.exp(st_f32(src, f"{la}.A_log")))[perm])
            assert np.array_equal(g.f32(f"blk.{il}.ssm_dt.bias"), st_f32(src, f"{la}.dt_bias")[perm])
            assert np.array_equal(g.f32(f"blk.{il}.ssm_norm.weight"), st_f32(src, f"{la}.norm.weight"))
            conv = st_f32(src, f"{la}.conv1d.weight").reshape(-1, tc["linear_conv_kernel_dim"])
            ch = np.concatenate([np.arange(qk_blocks * 128), qk_blocks * 128 + v_head_perm(n_k, n_v, tc["linear_value_head_dim"])])
            assert np.array_equal(g.f32(f"blk.{il}.ssm_conv1d.weight").reshape(-1, 4), conv[ch])
            assert np.array_equal(g.u16(f"blk.{il}.ssm_alpha.weight").reshape(n_v, -1), st_u16(src, f"{la}.in_proj_a.weight").reshape(n_v, -1)[perm])
    print(f"ok   fp16/f32/permuted small tensors of {n_layers} layers")

    # exl3 spot checks: layer 0 (GDN) and the first full-attention layer
    check_exl3(g, src, "blk.0.attn_qkv.weight", f"{lm}.layers.0.linear_attn.in_proj_qkv", perm_n=perm_qkv)
    check_exl3(g, src, "blk.0.attn_gate.weight", f"{lm}.layers.0.linear_attn.in_proj_z", perm_n=perm)
    check_exl3(g, src, "blk.0.ssm_out.weight", f"{lm}.layers.0.linear_attn.out_proj", perm_k=perm)
    check_exl3(g, src, "blk.0.ffn_up_shexp.weight", f"{lm}.layers.0.mlp.shared_expert.up_proj")
    fa = tc["layer_types"].index("full_attention")
    if fa < n_layers:
        n_q = tc["indexer_n_heads"] * tc["indexer_head_dim"]
        check_exl3(g, src, f"blk.{fa}.attn_q.weight", f"{lm}.layers.{fa}.self_attn.q_proj")
        check_exl3(g, src, f"blk.{fa}.attn_output.weight", f"{lm}.layers.{fa}.self_attn.o_proj")
        check_exl3(g, src, f"blk.{fa}.indexer.q_proj.weight", f"{lm}.layers.{fa}.self_attn.indexer.index_qk_proj", n_slice=(0, n_q))
        check_exl3(g, src, f"blk.{fa}.indexer.k_proj.weight", f"{lm}.layers.{fa}.self_attn.indexer.index_qk_proj", n_slice=(n_q, n_q + tc["indexer_head_dim"]))

    # stacked experts: every expert of layer 0's gate, and a few of the last layer's down
    for il, proj, gn, experts in ((0, "gate_proj", "ffn_gate_exps", range(512)),
                                  (n_layers - 1, "down_proj", "ffn_down_exps", (0, 1, 255, 511))):
        tr, suh, svh, K = g.exl3(f"blk.{il}.{gn}.weight")
        for e in experts:
            hf = f"{lm}.layers.{il}.mlp.experts.{e}.{proj}"
            k, n = tr.shape[1] * 16, tr.shape[2] * 16
            assert np.array_equal(tr[e], st_u16(src, hf + ".trellis").reshape(k // 16, n // 16, -1)), (il, proj, e)
            assert np.array_equal(suh[e].view(np.uint16), st_u16(src, hf + ".suh")), (il, proj, e)
            assert np.array_equal(svh[e].view(np.uint16), st_u16(src, hf + ".svh")), (il, proj, e)
        print(f"ok   blk.{il}.{gn} stacked experts ({len(list(experts))} checked, K={K})")

    if "per_layer_token_embd.weight" in g.tensors:
        ng = read_ngram_header(src.ngram_path)
        dims, t, off = g.tensors["per_layer_token_embd.weight"]
        check(t == G_EXL3_NGRAM6 and dims == [160, ng["rows"]], "ngram table type/dims")
        with open(src.ngram_path, "rb") as f:
            f.seek(ng["data_off"])
            first = f.read(122 * 1024)
            f.seek(ng["data_off"] + ng["data_bytes"] - 122 * 1024)
            last = f.read(122 * 1024)
        check(g.raw("per_layer_token_embd.weight", 122 * 1024) == first, "ngram first 1024 rows")
        check(g.raw("per_layer_token_embd.weight", 122 * 1024, skip=ng["data_bytes"] - 122 * 1024) == last, "ngram last 1024 rows")
        check(np.array_equal(g.u16("per_layer_token_embd.bias"), ng["head_bias"].view(np.uint16).reshape(-1)), "ngram head bias")

    if a.mtp:
        m = GGUF(a.mtp)
        check_exl3(m, src, "mtp.fc_hidden.weight", "mtp.fc_hidden")
        check_exl3(m, src, "mtp.fc_embedding.weight", "mtp.fc_embedding")
        check(np.array_equal(m.f32("mtp.pre_fc_norm_hidden.weight"), st_f32(src, "mtp.pre_fc_norm_hidden.weight") + 1.0), "mtp pre_fc_norm_hidden +1")
        check_exl3(m, src, "blk.0.attn_q.weight", "mtp.layers.0.self_attn.q_proj")
        tr, suh, svh, K = m.exl3("blk.0.ffn_up_exps.weight")
        e = 300
        hf = f"mtp.layers.0.mlp.experts.{e}.up_proj"
        check(np.array_equal(tr[e], st_u16(src, hf + ".trellis").reshape(160, 40, -1)) and
              np.array_equal(svh[e].view(np.uint16), st_u16(src, hf + ".svh")), "mtp expert 300 up")
    print("all checks passed")


if __name__ == "__main__":
    main()
