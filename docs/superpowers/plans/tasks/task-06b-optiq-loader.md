# Task 6b: OptiQ safetensors loader (secondary model source)

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 6b (read it before executing; the spec argues this plan)

**Depends on:** T3/T6 (both on disk: `task-03-gguf.md`, `task-06-engine-qwen35.md`) + T2/T4/T5 facts:
- T3: `TensorView{name,dtype,data,shape}`, `TensorDesc{name,dtype,shape,elements,abs_offset,bytes}`, `TensorShape{dims,ndim}` (GGUF order: `dims[0]` = fastest = in-dim), `GgufFile::open_shared`, `GgufError`, `bind_qwen35(&GgufFile,&ShapeProfile)->ModelBinding`, `DType::{F16,F32}`.
- T6: `Qwen35Model` + `Model::load(&mut self, backend: &dyn Backend, model_path: &Path, profile: &ShapeProfile) -> Result<()>` (T2 SSOT), `WeightRegister` injection, dtype→kernel dispatch table over {F32,F16,Q8_0,Q4_K,Q5_K,Q6_K}.
- T2: `HfConfig::from_file` already parses this repo's config shape (`model_type "qwen3_5"` → reads `text_config` only — task-02 plan lines 125–138, 184–188; verified same nesting as the OptiQ config.json probed below).
- T4/T5: F16/F32 MSL entries already exist — **this task adds no kernels** (rule 3): `kernel_mul_mv_f16_f32` (`metal/dense.metal:789`, T4 `mul_mv_dense.metal`), `kernel_mul_mv_ext_f16_f32_r1_{2..5}` (:1873–1876), `kernel_mul_mm_f16_f32` (:2456–2459, T5 keeps it), `kernel_get_rows_f16` (`metal/get_rows.metal:79–81`, T4 copies whole file).

**Produces** (public API; paths `shisu/crates/shisu-gguf/src/`):
```rust
// binding.rs (additive refactor — see DEVIATION 5)
pub trait TensorSource {
    fn architecture(&self) -> &str;                    // "qwen35"
    fn arch_u32(&self, suffix: &str) -> Option<u32>;   // synthesized from config.json
    fn find(&self, name: &str) -> Option<&TensorDesc>;
    fn get(&self, name: &str) -> Option<TensorView<'_>>;
}
impl TensorSource for GgufFile {}                      // forwards existing methods
pub fn bind_qwen35(src: &dyn TensorSource, profile: &ShapeProfile) -> Result<ModelBinding<'_>>;
pub fn bind_qwen35_desc(src: &dyn TensorSource, profile: &ShapeProfile) -> Result<DescBinding>;
// DescBinding = same role tree, fields TensorDesc (owned, no data borrow)

// modelfile.rs
pub enum ModelFile { Gguf(GgufFile), Safetensors(SafetensorsFile) }
impl ModelFile {
    pub fn open(path: impl AsRef<Path>) -> Result<ModelFile>; // dir → safetensors, file → gguf
    pub fn is_mmap_backed(&self) -> bool;              // true = register_no_copy legal
    pub fn materialize<'s>(&self, name: &str, scratch: &'s mut Vec<u8>) -> Result<&'s [u8]>;
}
impl TensorSource for ModelFile {}

// mlx_quant.rs — pure fns, no IO, no unsafe
pub fn dequant_affine(w: &[u8], scales: &[u8], biases: &[u8],
                      out_rows: usize, in_dim: usize, group_size: usize, bits: u32,
                      out: &mut Vec<u8>);              // u32-packed + bf16 s/b → f16 LE
pub fn bf16_to_f32(h: u16) -> f32;  pub fn f32_to_f16_bits(x: f32) -> u16;
```

## Global Constraints

Master plan Rules 1–7, one line each:
1. All new code under `shisu/`; pure Rust (this task touches no kernel files).
2. 1500-LoC cap — pre-split: name map + transforms in `st_names.rs`, loader in `safetensors.rs`, dispatch enum in `modelfile.rs`; the cap is far from binding (largest file ≈ 450).
3. No kernels, no arithmetic re-association: dequant math = mlx's exact formula (below), f32 intermediates, single f16 round at the end.
4. Metal phase: record numbers, no perf gate here (T20 owns gates).
5. Env knobs `SHISU_`-prefixed: new `SHISU_TEST_MODEL_OPTIQ` (path to the OptiQ dir, default `/Users/sercand/models/Qwen3.5-4B/mlx` — verified to hold `config.json` 55543 B, `model.safetensors` 3269669552 B, `model.safetensors.index.json` 100945 B); reuses `SHISU_TEST_MODEL` (T3). No downloads, no cache dir.
6. `warnings = deny` + clippy deny; errors = `GgufError` (new variants, no new error type, no anyhow); `tracing` for load progress.
7. No license headers.

Task-specific:
- **No new workspace deps.** Header parse = `serde_json` over a hand-read `u64` LE length + JSON (the `safetensors` crate eagerly loads tensors and hides raw `data_offsets` — streaming needs them); file IO = `memmap2` (T3 DEVIATION 3).
- `reader.rs`/`metadata.rs`/`quant.rs`/`quant_tables.rs` stay untouched; `binding.rs` gets the additive seam only; `gguf.rs` does not exist (DEVIATION 1).
- Run `cargo test -p shisu-gguf` / `-p shisu-engine` only; model+GPU tests `#[ignore]`-gated; T20 owns the final gate.

## Source References (verified)

All HF probes fetched **this session (2026-09-05)** via HF API + HTTP Range (no weight download beyond the header).

| Source | Verified facts | How used |
|---|---|---|
| `https://huggingface.co/api/models/mlx-community/Qwen3.5-4B-OptiQ-4bit` | Files: `model.safetensors` **3,269,669,552 B (single file, NOT sharded)**, `model.safetensors.index.json` 100,945, `config.json` 55,543, `optiq_metadata.json` 26,224, `kv_config.json` 528, `tokenizer.json` 19,989,343, `tokenizer_config.json` 1,135, `chat_template.jinja` 7,756, `generation_config.json` 148, `optiq/mtp.safetensors` 86,701,044, `optiq/optiq_vision.safetensors` 667,061,501. | Fetch script file list; vision + MTP files are **out of scope** (master plan Scope: vision out; MTP is T17's CUDA-phase territory) and must never be opened. |
| `…/raw/main/config.json` | `quantization_config` = `{group_size:64, bits:4, mode:"affine", "<tensor-FQN>":{bits:4|8, group_size:64}}` — 249 per-tensor entries keyed by full `language_model.model.…` names (incl. `embed_tokens`); duplicated as top-level `quantization` key. `text_config`: `model_type "qwen3_5_text"`, hidden 2560, 32 layers, 16Q/4KV, head_dim 256, inter 9216, vocab 248320, GDN 16 qk / 32 v heads @128, conv 4, `full_attention_interval` 4, `layer_types` = Attn iff `(il+1)%4==0`, rms_eps 1e-6, `tie_word_embeddings` true. | bits/group per tensor; ShapeProfile via T2 `HfConfig`; loader validates `text_config.model_type`. |
| `…/resolve/main/model.safetensors` header (Range 0–119,983: u64 LE `119976` + JSON) | `__metadata__ = {"format":"mlx"}`; **924 tensors**: dtypes `{BF16:651, U32:249, F32:24}`; quantized tensors are triples `X.weight` (U32 packed) + `X.scales` + `X.biases` (BF16, shape `[out, in/64]`); names prefixed `language_model.model.`; **no `lm_head`** (tied); attn layers {3,7,…,31}; data starts at byte 119,984. **Tensor-table digest (sha256 over sorted `name\|dtype\|d0,d1,…` rows, LF-joined, trailing LF): `dbb3e6b46cc1ff543940fb7ac70791d675e21cea6dcceab33a16d020633f1836`**. | Header parse contract; fixture digest; loader asserts. |
| header cross-check (computed this session) | bits derived from packed shape (`bits = packed_last·32 / (scales_last·group_size)`) match `quantization_config` for **all 249** tensors; histogram **{4-bit: 173, 8-bit: 76}** (8-bit incl. `embed_tokens`, `in_proj_qkv`, `q_proj`, `o_proj`, `down_proj` on most layers). `optiq_metadata.json` `per_layer` **disagrees on 1 tensor** (`layers.31.self_attn.v_proj`: says 4, header+config say 8). | DEVIATION 3: bits come from packed shape, validated against `quantization_config`; `optiq_metadata.json` is never read. |
| GGUF-name arithmetic (computed this session) | 924 safetensors tensors collapse to exactly **426 GGUF-named tensors = 249 F16 (dequantized) + 177 F32** (24 conv1d + 24 ssm_a + 24 dt_bias + 24 ssm_norm + 16 q/k_norm + 32 attn_norm + 32 post_norm + 1 output_norm) — identical count to the Q4_K_M GGUF's 426 tensors and its F32:177 (T3 Tests). | Completeness assert in the loader + layout test. |
| mlx `mlx/ops.cpp` @ main (fetched this session) | `affine_dequantize` :5223–5315: w **must be uint32** (:5464–5467); size check `w.shape(-1)*32/bits == scales.shape(-1)*group_size` (:5247–5249); unpack loop :5269–5280 = **LSB-first fields** (`(w << (32-(start+bits))) >> (32-bits)`, start = 0, bits, 2·bits… → field 0 = low bits); affine :5294–5299 = reshape `[…, groups, group_size]`, `w*scales + biases`, output dtype = scales dtype. | The dequant formula (verbatim below). |
| mlx `python/mlx/nn/layers/quantized.py` @ main | :264, :368 `in_dims = (in_dims * 32) // self.bits` — packed weights stored as **uint32 words**, `32/bits` weights per word along the last (in) axis; attrs are `weight`/`scales`/`biases` (no `_packed` suffix). | Packing layout. |
| `metal/dense.metal` (ds4, cwd) | :788–789 `kernel_mul_mv_{f32,f16}_f32` instantiations; :1873–1876 `kernel_mul_mv_ext_f16_f32_r1_{2..5}`; :2456–2459 `kernel_mul_mm_{f16,…}_f32`; `metal/get_rows.metal:79–81` `kernel_get_rows_f16`. | Proof the engine's F16/F32 dispatch rows have real MSL entries (T4/T5 copies) — T6b adds none. |
| `task-04-metal-backend.md` :55, :112 | `register_no_copy` = ds4 `ds4_metal.m:11492–11563` pattern: page-align down/up + `newBufferWithBytesNoCopy` — **only sound over mmap-backed pages**, never over heap `Vec` bytes. | `ModelFile::is_mmap_backed()` gate in the engine upload loop. |
| `task-06-engine-qwen35.md` :98–108 | `load` = `GgufFile::open_shared` + `bind_qwen35` + register every bound tensor; struct holds `binding`; dispatch table {F32,F16,Q8_0,Q4_K,Q5_K,Q6_K}. | The load path T6b rewrites (Step 6). |

⚠ **DEVIATION 1 (module layout):** master plan names `gguf.rs` + "rename crate module tree if needed". T3's actual modules are `reader.rs`/`metadata.rs`/`quant.rs`/`quant_tables.rs`/`binding.rs` (task-03 Produces); renaming to `gguf.rs` would churn every T6/T17 import for zero benefit. **Final layout (decided, no rename):** existing five files kept; new `safetensors.rs` (header + `SafetensorsFile`), `st_names.rs` (mlx→GGUF name map + transforms, pre-split for rule 2), `mlx_quant.rs` (pure dequant), `modelfile.rs` (`ModelFile` dispatch). Crate stays named `shisu-gguf` (SSOT in crate map + T6/T17 Consumes).

⚠ **DEVIATION 2 (repo shape vs master-plan assumptions):** the master plan says "safetensors dir" — true, but: single unsharded `model.safetensors`; tensor names carry a `language_model.` prefix that must be stripped; quantized weights are **U32-packed `weight` + BF16 `scales`/`biases`** (not the older uint8 `_weight_packed`/f16 `_scales` convention, and not f16 scales as the task brief assumed); config.json is a multimodal wrapper (LLM params under `text_config`); sibling `optiq/` files (vision, MTP) must be excluded — the loader follows `model.safetensors.index.json`'s weight_map and opens **only** shards referenced by `language_model.model.*` tensors (verified: 924 → `model.safetensors`, 297 → `optiq/optiq_vision.safetensors`).

⚠ **DEVIATION 3 (bits source of truth):** `optiq_metadata.json` `per_layer` is unreliable (1/249 mismatch vs header). Loader derives bits from packed shape, validates against `config.json.quantization_config` (0 mismatches), typed error on disagreement.

⚠ **DEVIATION 4 (6 GiB budget):** full-model f16 dequant = **4.206 B logical elements × 2 B = 7.83 GiB device weights** (computed from the header this session) — the ≤6 GiB gate (T6, GGUF path) is **unreachable** for OptiQ because no mlx-affine packed kernels exist (T5 covers GGUF quants only; repack to GGUF q4_1 blocks would need new kernels + a dtype the dispatch table lacks). Decision: dequantize-to-f16 stays; the 6 GiB gate remains **GGUF-only**; the OptiQ e2e gates host RSS < 12 GiB (streaming bound, see Step 6 math) and records the device footprint. Native packed-kernel support is a CUDA-phase candidate, out of scope here.

⚠ **DEVIATION 5 (binding seam):** T3 froze `bind_qwen35(&GgufFile, …)`. Format-agnostic binding needs a trait (master plan: "identical `binding.rs` layer-role mapper"). Additive refactor: `TensorSource` trait (Produces), role core generic over the weight handle `W` with `pub type ModelBinding<'f> = …<TensorView<'f>>` (T6's alias unchanged) and `DescBinding = …<TensorDesc>`; `bind_qwen35(&GgufFile,…)` call sites stay source-compatible (`&GgufFile` coerces to `&dyn TensorSource`).

## Plan

- [ ] **Step 1: Local-path resolver + layout fixture.** Create `shisu/scripts/test_model_optiq.sh` (pattern of T3's `test_model.sh`; NEVER downloads): `DIR="${SHISU_TEST_MODEL_OPTIQ:-/Users/sercand/models/Qwen3.5-4B/mlx}"`, exact-size check for `config.json` (55543), `model.safetensors` (3269669552), `model.safetensors.index.json` (100945); echo the dir. **Ignore** `optiq/*`, `kv_config.json`, `tokenizer*`. Regenerate `shisu/test-vectors/qwen35-optiq-layout.json` with an example bin `examples/dump_optiq_layout.rs` (argv = dir): `{repo, file, bytes, n_tensors: 924, dtype_counts: {BF16:651,U32:249,F32:24}, header_sha256: "dbb3e6b4…f1836" (row format above), quantization_config: <verbatim from config.json>}`.

- [ ] **Step 2: `mlx_quant.rs`.** Pure functions. Exact formula (mlx `ops.cpp:5269–5299`, cite in code comments):
  ```
  word  = LE u32 at w + row*(in_dim*bits/8) + (j*bits)/8
  q_j   = (word >> (bits * (j % (32/bits)))) & ((1<<bits)-1)      // LSB-first field
  out_ij = f32_to_f16_bits( f32(q_j) * bf16_to_f32(scale[row][j/group_size])
                          + bf16_to_f32(bias [row][j/group_size]) )  // f32 math, one f16 round
  ```
  `bits ∈ {4, 8}` only (reject others); `group_size` divides `in_dim`; row-major throughout; `out` = f16 LE, `[out_rows, in_dim]`. `bf16_to_f32(h) = f32::from_bits((h as u32) << 16)`; f32→f16 = round-to-nearest-even (RNE) — use a local bit-trick or `half`-free impl (no new deps).

- [ ] **Step 3: `safetensors.rs` + `st_names.rs`.** Header parse: read `u64` LE length, then exactly that many JSON bytes (Range-free: file is local; mmap the whole shard RO); validate `__metadata__.format == "mlx"`; per entry `{dtype ∈ {BF16,F32,U32,F16}, shape, data_offsets}` (offsets relative to `8+len`). `st_names.rs` name map (strip `language_model.` prefix; suffix triples collapse to one GGUF tensor; **all byte layouts already match GGUF order — reverse the safetensors shape, drop trailing size-1 dims, never transpose**):

  | mlx name (after prefix strip) | GGUF name | transform |
  |---|---|---|
  | `model.embed_tokens` | `token_embd.weight` | dequant → F16 |
  | `model.norm.weight` | `output_norm.weight` | BF16→F32 |
  | `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | BF16→F32 |
  | `model.layers.N.post_attention_layernorm.weight` | `blk.N.post_attention_norm.weight` | BF16→F32 |
  | `model.layers.N.mlp.{gate,up,down}_proj` | `blk.N.ffn_{gate,up,down}.weight` | dequant → F16 |
  | `model.layers.N.linear_attn.in_proj_qkv` | `blk.N.attn_qkv.weight` | dequant → F16 |
  | `model.layers.N.linear_attn.in_proj_z` | `blk.N.attn_gate.weight` | dequant → F16 |
  | `model.layers.N.linear_attn.in_proj_a` / `in_proj_b` | `blk.N.ssm_alpha.weight` / `ssm_beta.weight` | dequant → F16 |
  | `model.layers.N.linear_attn.out_proj` | `blk.N.ssm_out.weight` | dequant → F16 |
  | `model.layers.N.linear_attn.conv1d.weight` `[8192,4,1]` | `blk.N.ssm_conv1d.weight` `[4,8192]` | BF16→F32, squeeze last dim (byte order already matches) |
  | `model.layers.N.linear_attn.A_log` `[32]` F32 | `blk.N.ssm_a.weight` | **value transform `a = −expf(A_log)`** (GGUF/ds4 convention: `a` holds −exp(A_log), task-06 `gdn.rs` gates step) |
  | `model.layers.N.linear_attn.dt_bias` | `blk.N.ssm_dt.bias` | BF16→F32 |
  | `model.layers.N.linear_attn.norm.weight` | `blk.N.ssm_norm.weight` | BF16→F32 |
  | `model.layers.N.self_attn.{q,k,v}_proj` | `blk.N.attn_{q,k,v}.weight` | dequant → F16 |
  | `model.layers.N.self_attn.o_proj` | `blk.N.attn_output.weight` | dequant → F16 |
  | `model.layers.N.self_attn.{q,k}_norm.weight` | `blk.N.attn_{q,k}_norm.weight` | BF16→F32 |

  `a→alpha`, `b→beta` per Qwen3-Next convention (alpha feeds softplus decay, beta sigmoid — task-06 gates step); the e2e token-equality test is the swap detector. Unmapped `language_model.model.*` tensors → typed error; anything else (vision/MTP) → skipped by the index filter. Build `Vec<TensorDesc>` with **output** dtype (F16/F32), GGUF-order shape, `bytes` = dequantized size, `abs_offset` = file offset (informational). Assert 426 descs = 249 F16 + 177 F32. `SafetensorsFile: TensorSource` — `architecture()` = `"qwen35"` (after checking `text_config.model_type == "qwen3_5_text"`), `arch_u32("block_count")` = `num_hidden_layers`, `arch_u32("full_attention_interval")` = 4; `get(name)` dequantizes on demand into a stable-address slot (`Vec<OnceLock<Box<[u8]>>>`) — used by tests; the engine hot path uses `materialize` (never fills slots → no accumulation).

- [ ] **Step 4: `binding.rs` refactor (additive).** `TensorSource` trait per Produces; `impl TensorSource for GgufFile` (forwards `metadata().architecture()`, `arch_u32`, `find`, `get`); role core generic over `W` (fetch closure `Fn(&str) -> Option<W>`); `bind_qwen35` = fetch-via-`get`, `bind_qwen35_desc` = fetch-via-`find().cloned()`. Layout checks (dtype/ndim/dims) run identically for both — the safetensors descs carry output dtypes so T3's Step-5 checks (conv1d F32 `[4,8192]`, ssm_a/dt_bias F32 `[32]`, q_norm `[256]` F32, q `[2560,8192]` …) pass unchanged.

- [ ] **Step 5: `modelfile.rs`.** `ModelFile::open(path)`: `path.is_dir()` → `SafetensorsFile::open(dir)` (requires `config.json` + `model.safetensors`; if `model.safetensors.index.json` exists it is the location source-of-truth, shards opened lazily and only those referenced by LM tensors; missing files → typed error naming the path); else → `GgufFile::open_shared(path)` (GGUF magic check unchanged). `materialize(name, scratch)`: GGUF → return the mmap slice (lifetime coerces); safetensors → pread/mmap-read packed+scales+biases, `dequant_affine`/cast into `scratch` (reused, `clear()` not `shrink`), return slice; `MADV_DONTNEED` (`memmap2::advise_range`) on consumed source ranges. `is_mmap_backed()` = `matches!(self, Gguf(_))`.

- [ ] **Step 6: `shisu-engine/src/qwen35.rs` (modify).** `load`: `ModelFile::open_shared`-equivalent = `ModelFile::open(model_path)`; `bind_qwen35_desc(&file, profile)` replaces `bind_qwen35`; upload loop iterates the desc tree: `let bytes = file.materialize(name, &mut scratch)?;` then `if file.is_mmap_backed() { register(bytes) } else { alloc + copy_h2d }` (T4's no-copy wrap is mmap-pages-only — DEVIATION in task-04 :55). The model struct stores `DescBinding` (owned) + `ModelFile` (GGUF keeps the mmap alive for zero-copy buffers); **no code path may read `TensorView.data` after load** — grep `.data` in `qwen35.rs`/`layers.rs` at execution time and move any such read into the load loop (dispatch metadata reads `desc.dtype/shape` — same fields). Memory math (record in a code comment): device weights 7.83 GiB f16 + KV/GDN/scratch ≈ 0.5 GiB; host peak = largest single materialize (`token_embd` 635.7 M elems → 1.18 GiB scratch) + source window ≈ **1.3 GiB** — the host never holds all tensors.

## Tests

`cd shisu && cargo test -p shisu-gguf` / `-p shisu-engine`; model/GPU tests `#[ignore]` + SKIP-with-note when `SHISU_TEST_MODEL_OPTIQ` unset (macOS only; nothing here is Linux-gated).

- [ ] `shisu-gguf/tests/unit_mlx_quant.rs` (always runs, no model): (a) LSB-first unpack — word `0x89ABCDEF`, bits 4 → nibbles `[F,E,D,C,B,A,9,8]` in order; bits 8 → bytes `[EF,CD,AB,89]`. (b) affine exactness with scales=1/biases=0 bf16 → integer-valued f16 exact; scale=0 → output == bias. (c) `bf16_to_f32` round-trips vs hand-computed bit patterns; group boundary (j = group_size−1 vs j = group_size uses group 1). (d) shape guards: bits ∉ {4,8}, group_size ∤ in_dim → typed errors.
- [ ] `shisu-gguf/tests/optiq_layout.rs` (needs `SHISU_TEST_MODEL_OPTIQ`): header digest == fixture `header_sha256`; 924 tensors, dtype counts, `__metadata__.format=="mlx"`; GGUF-name stream = 426 descs = 249 F16 + 177 F32; `bind_qwen35_desc` → 32 layers, 24 GDN + 8 Attn iff `(il+1)%4==0`, `lm_head.is_none()`; spot descs: `blk.0.ssm_conv1d` F32 `[4,8192]`, `blk.0.ssm_a` F32 `[32]`, `blk.3.attn_q` F16 `[2560,8192]`, `token_embd` F16 `[2560,248320]`; `materialize("blk.0.ffn_gate.weight")` length == 9216·2560·2.
- [ ] `shisu-gguf/tests/dequant_oracle.rs` (ignored; requires `pip install mlx safetensors` — SKIP with note when `python3 -c "import mlx.core, safetensors"` fails): dump the oracle with exactly:
  ```sh
  python3 - "$SHISU_TEST_MODEL_OPTIQ/model.safetensors" /tmp/optiq-oracle-gate0.bin <<'PY'
  import sys, mlx.core as mx
  from safetensors import safe_open
  with safe_open(sys.argv[1], framework="mlx") as f:
      q = f.get_tensor("language_model.model.layers.0.mlp.gate_proj.weight")          # uint32
      s = f.get_tensor("language_model.model.layers.0.mlp.gate_proj.scales").astype(mx.float32)
      b = f.get_tensor("language_model.model.layers.0.mlp.gate_proj.biases").astype(mx.float32)
  mx.dequantize(q, s, b, group_size=64, bits=4)[0:64].tofile(sys.argv[2])              # f32, 64×2560
  PY
  ```
  (scales/biases cast to f32 **before** `dequantize` so the oracle output dtype is f32 — bf16 output would carry 2^-8 rounding and break the 1e-3 gate). Rust side: `materialize` the same tensor, compare rows 0–63: `|f16 − oracle| ≤ 1e-3 + 1e-3·|oracle|` (master plan tol 1e-3, mixed abs/rel form).
- [ ] `shisu-engine/tests/optiq_decode.rs` (ignored; macOS + `SHISU_TEST_MODEL_OPTIQ` + `SHISU_TEST_MODEL` + `metal_available()`): load the **dir** through `Model::load` (dispatch proof), greedy (temperature 0) 32 tokens on T6's golden prompt, assert equality with the first 32 ids of `shisu/test-vectors/qwen35-greedy-64.json`. **Divergence rule:** if the implementation-time run diverges earlier, the oracle test must be green first (loader exonerated); then record `divergence_at` in `qwen35-optiq-layout.json` and the test asserts equality up to `min(32, divergence_at)` with `divergence_at ≥ 8` — quantization-noise divergence between two different quants is legitimate, a loader bug is not; never silently widen. RSS: run T6's `shisu-bench` qwen35 bin as a child under `/usr/bin/time -l` with `SHISU_TEST_MODEL=$SHISU_TEST_MODEL_OPTIQ` (dir works via the same dispatch), snapshot + restore `shisu-bench/baseline/m1max-qwen35.json` around it, assert max RSS < 12 GiB (DEVIATION 4), record the number in the test output.
- [ ] Expected before implementation: compile failure (new modules absent); after: unit tests green always, model tests SKIP without env vars, green on this Mac with both models fetched.

## Acceptance

- [ ] `shisu/crates/shisu-gguf/src/{safetensors.rs,st_names.rs,mlx_quant.rs,modelfile.rs}` exist, each < 1500 LoC; `reader.rs`/`metadata.rs`/`quant.rs`/`quant_tables.rs` byte-unchanged (`git diff --stat`); `binding.rs` diff is additive-only (trait + generic core + desc entry).
- [ ] `ModelFile::open`, `TensorSource`, `bind_qwen35_desc`, `dequant_affine` public with the exact signatures above; `bind_qwen35(&GgufFile,…)` call sites in T6 code compile unchanged.
- [ ] Dequant formula cites `mlx/ops.cpp:5269–5299`; LSB-first unpack verified by unit test; bits derived from packed shape, `quantization_config` validated, `optiq_metadata.json` unread.
- [ ] `test-vectors/qwen35-optiq-layout.json` checked in and matching this plan's recorded facts (924 / {BF16:651,U32:249,F32:24} / digest `dbb3e6b4…f1836` / bits {4:173,8:76}).
- [ ] GGUF-name stream = 426 = 249 F16 + 177 F32 asserted; `bind_qwen35_desc` maps 24 GDN + 8 Attn; `ssm_a = −exp(A_log)` transform present and cited.
- [ ] `cargo test -p shisu-gguf` green (with + without `SHISU_TEST_MODEL_OPTIQ`); `-- --ignored` green with mlx installed; `shisu-engine` optiq decode green (32-token equality, RSS < 12 GiB recorded).
- [ ] No new workspace deps; no `safetensors` crate; no kernels added or edited; no `DS4_`/`ATLAS_` env reads; no license headers.

## Commit

```sh
git add shisu/crates/shisu-gguf shisu/crates/shisu-engine shisu/test-vectors/qwen35-optiq-layout.json shisu/scripts/test_model_optiq.sh
git commit -m "feat(gguf): OptiQ safetensors loader, mlx affine dequant, ModelFile format dispatch"
```
