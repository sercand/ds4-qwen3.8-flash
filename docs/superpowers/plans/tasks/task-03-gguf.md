# Task 3: shisu-gguf

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 3 (read it before executing; the spec argues this plan)

**Depends on:** T1 (workspace `shisu/crates/shisu-gguf` stub exists), T2 (consumed, exact names from `task-02-core.md`):
- `shisu_core::dtype::DType` (`#[repr(u32)]`, discriminants = ds4 GGUF ids, ds4.c:2306-2327; `from_gguf_id(u32)->Option<DType>`, `gguf_id`, `type_name`, `block_elems`, `block_bytes`, `is_quantized`). Re-exported at crate root.
- `shisu_core::shape::{ShapeProfile, LayerKind, Qwen35Shape, GqaShape, GdnShape}` — `ShapeProfile::{Qwen35(Qwen35Shape), Qwen4Exp(Qwen4ExpShape)}`, `layer_kind(&self, il: u32) -> Option<LayerKind>`, `n_layers()/hidden()/vocab()`; `LayerKind::{Gdn, Attn}`.
- `shisu_core::{CoreError, Result}` — this crate defines its own `GgufError` (thiserror) and aliases `pub type Result<T> = std::result::Result<T, GgufError>`.

**Produces** (public API T6/T6b/T17 build on; all paths `shisu/crates/shisu-gguf/src/`):
```rust
// reader.rs
pub const GGUF_MAGIC: u32 = 0x46554747;   // ds4.c:1290
pub const MAX_DIMS: usize = 8;            // DS4_MAX_DIMS, ds4.c:1291

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape { pub dims: [u64; MAX_DIMS], pub ndim: u8 } // GGUF order: dims[0] = fastest (in-dim)
impl TensorShape { pub fn elements(&self) -> u64; pub fn dim(&self, i: usize) -> u64; }

#[derive(Debug, Clone)]
pub struct TensorDesc { pub name: String, pub dtype: DType, pub shape: TensorShape,
                        pub elements: u64, pub abs_offset: u64, pub bytes: u64 } // bytes incl. exl3 scale vectors (ds4.c:2791)

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> { pub name: &'a str, pub dtype: DType, pub data: &'a [u8], pub shape: TensorShape }

pub struct GgufFile { /* memmap + Vec<TensorDesc> + GgufMetadata + version/alignment/tensor_data_pos */ }
impl GgufFile {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<GgufFile>;         // private RO mmap (ds4.c:2933 CPU path)
    pub fn open_shared(path: impl AsRef<std::path::Path>) -> Result<GgufFile>;  // shared RO mmap — T4 wraps slices as no-copy MTLBuffers (ds4.c:2921-2934)
    pub fn version(&self) -> u32;                 // 3 only (ds4.c:2831)
    pub fn alignment(&self) -> u64;               // default 32 (ds4.c:2727-2745)
    pub fn metadata(&self) -> &GgufMetadata;
    pub fn tensors(&self) -> &[TensorDesc];
    pub fn find(&self, name: &str) -> Option<&TensorDesc>;      // ds4.c:3328-3337
    pub fn get(&self, name: &str) -> Option<TensorView<'_>>;    // data = raw quant payload; dequant is kernel-side
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError { NotGguf, Version(u32), Truncated { pos: u64 }, NestingTooDeep,
    UnsupportedType { tensor: String, dtype_id: u32 }, DimLimit { tensor: String, ndim: u32 },
    OutOfBounds { tensor: String }, UnknownDType { id: u32 }, SplitUnsupported { count: u32 },
    MissingTensor(String), LayoutMismatch { tensor: String, want: String, got: String },
    TypeMismatch { tensor: String, want: &'static str, got: DType },
    #[error(transparent)] Io(#[from] std::io::Error) }

// metadata.rs
pub enum MetaValue { U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32),
                     Bool(bool), Str(String), U64(u64), I64(i64), F64(f64), Array(Vec<MetaValue>) }
pub struct GgufMetadata { /* kv: Vec<(String, MetaValue)>, arch: String (default "deepseek4" when key absent, ds4.c:3257-3260) */ }
impl GgufMetadata {
    pub fn architecture(&self) -> &str;
    pub fn get(&self, key: &str) -> Option<&MetaValue>;
    pub fn get_str(&self, key: &str) -> Option<&str>;
    pub fn get_u32(&self, key: &str) -> Option<u32>;    // width-compat: u16/i16/u32/i32 (ds4.c:2577-2606)
    pub fn get_u64(&self, key: &str) -> Option<u64>;    // u32-widening (ds4.c:2608-2622)
    pub fn get_f32(&self, key: &str) -> Option<f32>;    // f64/u32/i32 narrowing (ds4.c:2624-2650)
    pub fn get_bool(&self, key: &str) -> Option<bool>;
    pub fn get_array(&self, key: &str) -> Option<&[MetaValue]>;
    pub fn arch_u32(&self, suffix: &str) -> Option<u32>; // "<arch>.<suffix>" (ds4.c:3219-3235)
    pub fn arch_u64(&self, suffix: &str) -> Option<u64>;
    pub fn arch_f32(&self, suffix: &str) -> Option<f32>;
}

// quant.rs — byte-offset descriptors (offsets + sizes; accessors read LE from &[u8])
pub const QK_K: usize = 256;         // ds4.c:980
pub const QK_MXFP4: usize = 32;      // ds4.c:981
pub const QK_LEGACY: usize = 32;     // ds4.c:982
pub mod q8_0   { pub const SIZE: usize = 34;  pub const OFF_D: usize = 0;  pub const OFF_QS: usize = 2; }   // f16 d + [i8;32] (gguf_types[8], ds4.c:2270)
pub mod q4_k   { pub const SIZE: usize = 144; pub const OFF_D: usize = 0; pub const OFF_DMIN: usize = 2; pub const OFF_SCALES: usize = 4; pub const OFF_QS: usize = 16; }
pub mod q2_k   { pub const SIZE: usize = 84;  pub const OFF_SCALES: usize = 0; pub const OFF_QS: usize = 16; pub const OFF_D: usize = 80; pub const OFF_DMIN: usize = 82; }
pub mod q5_k   { pub const SIZE: usize = 176; pub const OFF_D: usize = 0; pub const OFF_DMIN: usize = 2; pub const OFF_SCALES: usize = 4; pub const OFF_QH: usize = 16; pub const OFF_QS: usize = 48; }
pub mod q6_k   { pub const SIZE: usize = 210; pub const OFF_QL: usize = 0; pub const OFF_QH: usize = 128; pub const OFF_SCALES: usize = 192; pub const OFF_D: usize = 208; }
pub mod q8_k   { pub const SIZE: usize = 292; pub const OFF_D: usize = 0; pub const OFF_QS: usize = 4; pub const OFF_BSUMS: usize = 260; }
pub mod iq2_xxs{ pub const SIZE: usize = 66;  pub const OFF_D: usize = 0; pub const OFF_QS: usize = 2; }
pub mod iq4_nl { pub const SIZE: usize = 18;  pub const OFF_D: usize = 0; pub const OFF_QS: usize = 2; }
pub mod q5_1   { pub const SIZE: usize = 24;  pub const OFF_D: usize = 0; pub const OFF_M: usize = 2; pub const OFF_QH: usize = 4; pub const OFF_QS: usize = 8; }
pub mod mxfp4  { pub const SIZE: usize = 17;  pub const OFF_E: usize = 0; pub const OFF_QS: usize = 1; }   // e8m0 scale + 16 packed nibbles
pub mod exl3   { // pseudo-types ds4.c:2294-2303; tile = 256 weights, 32*K bytes
    pub const TILE_ELEMS: usize = 256;
    pub fn k_bits(dtype: DType) -> Option<u32>;            // Exl3K4/5/6 -> 4/5/6 (ds4.c:2479-2481)
    pub fn tile_bytes(k_bits: u32) -> usize;               // 32 * K
    pub const NGRAM6_ELEMS: usize = 160; pub const NGRAM6_BYTES: usize = 122; // f16 scale + 160×6-bit ring
    pub fn scale_bytes(ndim: u8, d0: u64, d1: u64, d2: u64) -> u64; // experts*2*(d0+d1), ds4.c:2483-2487
}
pub fn block_bytes_for(dtype: DType, elements: u64) -> Option<u64>; // ceil(e/block_elems)*block_bytes + exl3 scale vecs (ds4.c:2460-2467, 2791)

// quant_tables.rs (tables kept separate from quant.rs for reviewability; 1500 cap)
pub static IQ4NL_VALUES: [i8; 16];        // ds4.c:1047-1049
pub static KMASK_IQ2XS: [u8; 8];          // ds4.c:1124-1126
pub static KSIGNS_IQ2XS: [u8; 128];       // ds4.c:1128-1137
pub static IQ2XXS_GRID: [u64; 256];       // ds4.c:1139-1204

// binding.rs — tensor-name → layer-role mapper driven by ShapeProfile
pub struct GdnWeights<'f>    { pub qkv: TensorView<'f>, pub gate: TensorView<'f>, pub conv1d: TensorView<'f>,
                               pub a: TensorView<'f>, pub alpha: TensorView<'f>, pub beta: TensorView<'f>,
                               pub dt_bias: TensorView<'f>, pub norm: TensorView<'f>, pub out: TensorView<'f> }
pub struct AttnWeights<'f>   { pub q: TensorView<'f>, pub k: TensorView<'f>, pub v: TensorView<'f>,
                               pub q_norm: TensorView<'f>, pub k_norm: TensorView<'f>, pub output: TensorView<'f> }
pub struct FfnWeights<'f>    { pub gate: TensorView<'f>, pub up: TensorView<'f>, pub down: TensorView<'f> }
pub struct BoundLayer<'f>    { pub kind: LayerKind, pub input_norm: TensorView<'f>, pub post_norm: TensorView<'f>,
                               pub ffn: FfnWeights<'f>, pub gdn: Option<GdnWeights<'f>>, pub attn: Option<AttnWeights<'f>> }
pub struct ModelBinding<'f>  { pub token_embd: TensorView<'f>, pub output_norm: TensorView<'f>,
                               pub lm_head: Option<TensorView<'f>>, // None = tied to token_embd (Q4_K_M has no output.weight)
                               pub layers: Vec<BoundLayer<'f>> }
pub fn bind_qwen35(file: &GgufFile, profile: &ShapeProfile) -> Result<ModelBinding<'_>>;
```

## Global Constraints

Master plan Rules 1–7, one line each:
1. All new code under `shisu/`; no C/C++/ObjC host code (this task is pure Rust).
2. Hard 1500-LoC-per-file cap (local gate `shisu/scripts/file_size_check.sh`) — IQ2/IQ4 lookup tables live in `quant_tables.rs` (the 256×u64 grid alone is ~70 lines; kept split for reviewability, not for the cap).
3. No kernels here: dequant **kernels** stay in kernel files (T4/T5); this crate is host-side parse + descriptors only. The CPU `ds4_vec_dot_*` reference oracles (ds4.c:3733-4965) are NOT ported in this task (T17 owns CPU oracles).
4. No perf surface; no timing claims.
5. Env knobs `SHISU_`-prefixed: this task introduces `SHISU_TEST_MODEL` only — the local GGUF path, default `/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf` (master plan rule 5; no downloads, no cache dir).
6. `warnings = deny` + clippy deny; `thiserror` errors (library crate — no anyhow); `parking_lot`/`tracing` where relevant (parse is single-threaded: `tracing` logs only).
7. No license headers.

Task-specific:
- Tensor bytes are NEVER copied or dequantized by this crate — `TensorView.data` is a borrow into the mmap (ds4.c:2235-2238 "leaves tensor bytes in place").
- GGUF dims keep llama.cpp order: `dims[0]` = fastest-varying = matmul in-dim (matches ds4 `t->dim[0]` usage, e.g. ds4.c:6151).
- Do not run formatters/linters/project-wide suites; `cargo test -p shisu-gguf` only (T20 owns the final gate).

## Source References (verified)

ds4 root = cwd. Every line opened while writing this plan.

| Source | Lines | What lives there / how used |
|---|---|---|
| `ds4.c` | 1290-1305 | `DS4_GGUF_MAGIC 0x46554747`, `DS4_MAX_DIMS 8`, `ds4_str`, `ds4_cursor` → `GgufError`-based `Cursor` struct. |
| `ds4.c` | 2181-2229 | cursor helpers (`cursor_has/read/skip/u32/u64/string`, `align_up`) — LE reads, bounds-checked, first-error-wins message. Ported as `Cursor` methods. |
| `ds4.c` | 2241-2255 | `GGUF_VALUE_*` (13 types) → `MetaType`. |
| `ds4.c` | 2257-2304 | `gguf_type_info` + `gguf_types[]` block table incl. q8_0(32,34) at :2270 and the **EXL3 pseudo-ids** `[68]=exl3_k4(256,128) [69]=k5(256,160) [70]=k6(256,192) [72]=exl3_ngram6(160,122)` at :2300-2303 → `quant.rs` consts (DType already carries ids via T2). |
| `ds4.c` | 2329-2371 | `ds4_kv`, `ds4_tensor`, `ds4_model` (version/n_kv/n_tensors/alignment/tensor_data_pos/shards) → `TensorDesc`/`GgufFile` fields. |
| `ds4.c` | 2387-2447 | `scalar_value_size`, `skip_value` (array nesting cap 8) → metadata skip. |
| `ds4.c` | 2449-2467 | `tensor_type/name/nbytes` (ceil-blocks math, overflow guard) → `block_bytes_for`. |
| `ds4.c` | 2469-2487 | exl3 helpers: `tensor_type_is_exl3`, `exl3_type_bits`, `exl3_scale_bytes` (payload = [E] tiles + [E] suh[k] + [E] svh[n] fp16) → `quant::exl3`. |
| `ds4.c` | 2509-2677 | `cursor_at`, `model_find_kv`, getters incl. `model_get_u32_compat` (u16/i16 split-key widths, :2575-2606), `model_get_u64_compat`, `model_get_f32_compat`, `model_get_bool`, `model_get_array` → `GgufMetadata` getters (compat widths copied verbatim). |
| `ds4.c` | 2719-2749 | `parse_metadata`: n_kv sanity vs file size, `general.alignment` capture (default 32), lazy value decode via `value_pos`. |
| `ds4.c` | 2757-2814 | `parse_tensor_directory`: name/ndim(1..=8)/dims/type/rel_offset; exl3 `bytes += exl3_scale_bytes` (:2791); `data_pos = align_up(pos, alignment)`; abs_offset rebase + per-shard bounds check. |
| `ds4.c` | 2821-2834 | `model_read_shard_header`: magic + **version == 3 only**. |
| `ds4.c` | 2840-2904 | `model_split_path` (`-NNNNN-of-NNNNN.gguf`, 20-char suffix), `model_map_split` (back-to-back mmap, `DS4_MAX_SHARDS 32`). See ⚠ DEVIATION 4. |
| `ds4.c` | 2908-3016 | `model_open`: open+fstat(≥32B), mmap `MAP_SHARED` for Metal / `MAP_PRIVATE` CPU (:2933-2934, Darwin VM rationale :2921-2932), split.count/split.no flow. |
| `ds4.c` | 3237-3337 | `model_summary` (metadata key names incl. `general.architecture` default "deepseek4" :3257-3260, `arch.<suffix>` pattern :3219-3235), `model_find_tensor` → `find()`. |
| `ds4.c` | 4966-5033 | "Fixed Weight Binding and Model Validation" header + `tensor_by_namef`/`required_tensorf` (`"blk.%u.…"` printf names) → binding name macros → `format!`. |
| `ds4.c` | 5045-5108 | `tensor_expect_layout` (type/ndim/dims strict), dense-quant type-set predicates → `expect_layout`/`expect_type_in` helpers (typed errors, not exit()). |
| `ds4.c` | 5927-5991 | qwen4exp GDN layout validation: `gdn_in = 2*gdn_k + gdn_v` (:5929), conv1d `[conv, gdn_in]` F32, `a`/`dt_bias` `[v_heads]` F32, `alpha`/`beta` `[hidden, v_heads]` f32-or-f16, `norm` `[head_dim]` (:5983-5991) — the dim formulas `bind_qwen35` reuses. |
| `ds4.c` | 7478-7528 | `weights_bind_qwen4exp_layer`: role branch `ds4_qwen4exp_layer_is_full_attn(il)` (:7488); attn names `blk.%u.attn_{q,k,v,q_norm,k_norm,output}.weight` (:7489-7494); GDN names `blk.%u.{attn_qkv,attn_gate,ssm_conv1d,ssm_a,ssm_alpha,ssm_beta,ssm_dt,ssm_norm,ssm_out}` (:7500-7508) — **identical suffixes to the real Qwen3.5 GGUF** (verified below). |
| `ds4.c` | 945-949 | `ds4_qwen4exp_layer_is_full_attn`: `(il + 1) % interval == 0` — the hybrid rule T2's `layer_kind` encodes; binding cross-checks GGUF `qwen35.full_attention_interval` against it. |
| `ds4.c` | 7713-7830 | `weights_bind_layer` (family dispatch) + `weights_bind` (token_embd/output bind :7800-7805, per-layer loop :7816-7818, validate last :7829) → `bind_qwen35` structure. |
| `ds4.c` | 965-1060 | "GGUF Quant Block Formats": `QK_K/QK_MXFP4/QK_LEGACY` (:980-982), `block_q2_K/q4_K/q5_K/q6_K/q8_K/iq2_xxs/mxfp4/q5_1/iq4_nl` layouts (:984-1044), `ds4_iq4nl_values` (:1047-1049), size asserts 84/144/176/210/292/66/17/24/18 (:1051-1060) → `quant.rs` offset consts + `const _: () = assert!(…)` on documented sizes. |
| `ds4.c` | 1124-1204 | `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid[256]` → `quant_tables.rs` (copied verbatim, hex literals). |
| `ds4.h` | whole file (768 ln) | ⚠ contains ONLY the engine/session boundary API — **no GGUF/tensor types**; all parser structs are ds4.c-internal statics. Nothing to port from ds4.h. |
| HF repo | api tree listing, fetched 2026-09-05 | `unsloth/Qwen3.5-4B-GGUF` contains `Qwen3.5-4B-Q4_K_M.gguf` = **2,740,937,888 bytes**; URL `https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf`. |
| HF model header | first 32 MB fetched + parsed 2026-09-05 | Ground truth for the manifest test (see Tests): GGUF v3, **426 tensors, 46 KV keys**, `general.architecture = qwen35`, `qwen35.block_count=32`, `embedding_length=2560`, `feed_forward_length=9216`, `attention.head_count=16`, `head_count_kv=4`, `key_length=value_length=256`, `rope.dimension_count=64`, `rope.dimension_sections=[11,11,10,0]`, `rope.freq_base=1e7`, `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.inner_size=4096`, `full_attention_interval=4`, `tokenizer.ggml.eos_token_id=248046`. |

⚠ **DEVIATION 1 (master-plan §460-1281):** verified — §460-955 is the *shape profiles* section (`ds4_shape`, `DS4_SHAPE_QWEN38F`; T2's territory), not quant descriptors. The quant block descriptors are §965-1060 + IQ2 tables §1124-1204. Also: **no `block_q8_0` C struct exists** — q8_0 geometry lives only in `gguf_types[8] = {"q8_0", 32, 34}` (ds4.c:2270); **no "q4e" tensor format exists** — `gguf_types` has no q4e entry; "q4e_*" is the qwen4exp family prefix (page geometry = `ds4_q4e_page.h` → T2's `q4e_page`); **no exl3 C structs** — geometry is the pseudo-type table rows + `exl3_scale_bytes`. This plan ports exactly what exists.

⚠ **DEVIATION 2 (master-plan §2231-4966):** GGUF parse is §2231-3016 ✓, but the tensor-name→layer-role mapping ("Fixed Weight Binding and Model Validation") starts at **§4966** and ends at `weights_bind` :7830; §3733-4965 is the CPU `ds4_vec_dot_*` oracle (kernel-side reference, out of scope here). Mapping is ported from §4966-7830.

⚠ **DEVIATION 3 (new workspace dep):** T1's `[workspace.dependencies]` has no mmap crate; ds4 uses raw `mmap` (ds4.c:2934). Add `memmap2 = "0.9"` to the workspace manifest (boring, MIT/Apache, no transitive weight). `open()` → `Mmap::map` (private); `open_shared()` → `MmapOptions::map_shared` for T4's no-copy MTLBuffer wrapping.

⚠ **DEVIATION 4 (split GGUF deferred):** `model_map_split` stitches shards into one back-to-back address range (ds4.c:2857-2904); memmap2 cannot alias multiple files as one range. The Metal test model is single-file, so: parse `split.count`; when > 1 return `GgufError::SplitUnsupported { count }` with a message naming shard 00001. Full split support lands with the CUDA phase (qwen4exp GGUFs ship split) — recorded here, not silently dropped.

## Plan

- [ ] **Step 1: Manifests.** Add `memmap2 = "0.9"` to `shisu/Cargo.toml` `[workspace.dependencies]` (comment: replaces ds4 raw mmap, ds4.c:2934). In `shisu/crates/shisu-gguf/Cargo.toml` add `[dependencies]`: `shisu-core = { workspace = true }`, `thiserror`, `tracing`, `memmap2` (all `workspace = true`).

- [ ] **Step 2: `reader.rs` — cursor + header + directory.** Port (adapt: `ds4_die`/`exit(1)` → typed `GgufError`, first-error cursor message → `Truncated { pos }`):
  - `Cursor` over `&[u8]`: `u32/u64/f32/string/skip` LE + bounds (ds4.c:2187-2224), `align_up` (:2226-2229).
  - `read_header`: magic/version(==3)/n_tensors/n_kv (ds4.c:2821-2834); file ≥ 32 B (ds4.c:2919).
  - `parse_tensor_directory` (ds4.c:2757-2814): count-vs-remaining sanity; ndim 1..=8; dims product with overflow guard; `DType::from_gguf_id` else `UnsupportedType`; `block_bytes_for` (warn-and-keep semantics become typed error — ds4.c:2785-2790 only warns because it sizes other formats; shisu rejects unknown ids at parse); exl3 scale-vector add; `abs_offset = align_up(dir_end, alignment) + rel_offset`; bounds check.
  - `GgufFile::open/open_shared` per DEVIATION 3/4; `find/get/tensors/metadata`.
  - `TensorShape::elements` = product of dims (overflow-checked).

- [ ] **Step 3: `metadata.rs`.** `MetaType`/`MetaValue` (ds4.c:2241-2255); `parse_metadata` (ds4.c:2719-2749) decoding every value eagerly into `Vec<(String, MetaValue)>` (⚠ adaptation: ds4 stores `value_pos` lazily; eager decode is simpler in Rust and the whole KV table is ≤ a few MB incl. tokenizer arrays — measure, and if the token array decode is hot, keep `Array` as a raw byte range instead; default eager). `skip_value` nesting cap 8 (ds4.c:2409-2413). Getters with the exact compat-width behavior of ds4.c:2533-2677. `architecture()` defaults `"deepseek4"` when the key is absent (ds4.c:3257-3260).

- [ ] **Step 4: `quant.rs` + `quant_tables.rs`.** Offset consts exactly as the Produces block (derived from the C field orders ds4.c:984-1044; q8_0 from gguf_types row). Add `const _: () = assert!(q4_k::OFF_QS + 128 == q4_k::SIZE);`-style cross-checks for every module. `block_bytes_for` = ds4.c:2460-2467 + exl3 tail (:2791, :2483-2487). Tables copied verbatim from ds4.c:1047-1049, 1124-1204. No dequant functions (kernel-side; rule 3).

- [ ] **Step 5: `binding.rs`.** `bind_qwen35(file, profile)`:
  - Assert `file.metadata().architecture() == "qwen35"` and `arch_u32("block_count") == profile.n_layers()`; cross-check `arch_u32("full_attention_interval")` against `profile.layer_kind` pattern (error on disagreement — ds4.c:940-949 validates the same way).
  - Per layer `il in 0..n_layers`: `kind = profile.layer_kind(il)`; bind `blk.{il}.attn_norm.weight` → `input_norm`, `blk.{il}.post_attention_norm.weight` → `post_norm`, `ffn_{gate,up,down}.weight` → `ffn`; then by kind:
    - `Gdn` → the 9 names of ds4.c:7500-7508 (`attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`).
    - `Attn` → the 6 names of ds4.c:7489-7494 (`attn_q`, `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`).
  - Layout checks (adapt `tensor_expect_layout` ds4.c:5045-5084 to typed errors; dim formulas from `Qwen35Shape`, pattern of ds4.c:5927-5991): `qkv.dims = [hidden, 2*qk_heads*head_dim + v_heads*head_dim]` (= 8192 for qwen35), `gate = [hidden, v_heads*head_dim]`, `conv1d = [conv_kernel, qkv_out]` F32, `a`/`dt_bias = [v_heads]` F32, `alpha`/`beta = [hidden, v_heads]` F32-or-F16-or-quant, `norm = [gdn.head_dim]`, `out = [v_heads*head_dim, hidden]`; attn: `q = [hidden, 2*q_heads*gqa.head_dim]` (gated: 8192 = 2×16×256), `k`/`v = [hidden, kv_heads*head_dim]`, `q_norm`/`k_norm = [gqa.head_dim]`, `output = [q_heads*head_dim, hidden]`; ffn: `gate`/`up = [hidden, intermediate]`, `down = [intermediate, hidden]`. Weight matrices accept any of {F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K} (the Q4_K_M file mixes all of these; predicate pattern ds4.c:5086-5108); norm/vector tensors must be F32 (or F16 for alpha/beta).
  - Globals: `token_embd.weight` required, `output_norm.weight` required, `output.weight` optional → `lm_head: None` when absent (tied).

- [ ] **Step 6: `scripts/test_model.sh`** (create `shisu/scripts/`) — resolves the local model; NEVER downloads:
  ```sh
  #!/usr/bin/env bash
  set -euo pipefail
  FILE="${SHISU_TEST_MODEL:-/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf}"
  SIZE=2740937888
  [ -f "$FILE" ] || { echo "missing model: $FILE" >&2; exit 1; }
  [ "$(wc -c <"$FILE")" -eq "$SIZE" ] || { echo "size mismatch: $FILE"; exit 1; }
  echo "$FILE"
  ```

- [ ] **Step 7: `examples/dump_manifest.rs`** — opens `argv[1]`, prints one line per tensor: `name \t ndim \t d0[,d1…] \t gguf_type_id \t rel_offset` (rel = `abs_offset - tensor_data_pos`). Generate the checked-in fixture once:
  ```sh
  cd shisu && SHISU_TEST_MODEL="$(scripts/test_model.sh)" \
    cargo run -p shisu-gguf --example dump_manifest -- "$SHISU_TEST_MODEL" \
    > test-vectors/qwen35-q4km-manifest.tsv
  wc -l test-vectors/qwen35-q4km-manifest.tsv   # expect 426
  ```

## Tests

All in `shisu/crates/shisu-gguf/tests/`. Model-dependent tests read `SHISU_TEST_MODEL` (default: the local GGUF path, master rule 5); when the file is absent they print `SKIP: model not found` and pass (nothing is ever downloaded; the dev box has the file). No Linux-only tests — everything here is host-side and runs identically on Linux.

- [ ] `unit_no_model.rs` (always runs): (a) `block_bytes_for` per format vs the ds4.c:1051-1060 assert sizes (e.g. Q4_K × 256·100 elems = 14400; Q8_0 × 320 = 340; MXFP4 × 32 = 17; IQ4_NL × 160 = 90; Exl3K6 tile 256 elems = 192 + `scale_bytes` for a [2560,640] 2-D tensor = 2·(2560+640) = 6400). (b) `GgufMetadata` compat getters against a hand-built KV table (u16→u32, f64→f32, i32→f32 paths of ds4.c:2577-2650). (c) truncated-file errors: 0-byte, 16-byte, magic-only, header-without-directory → `NotGguf`/`Truncated`. (d) `bind_qwen35` role-name table against `ShapeProfile::Qwen35` built from the T2 fixture `shisu/test-vectors/qwen35-config.json` (via `HfConfig::from_file`): `layer_kind` pattern is Attn iff `(il+1)%4==0` → 8 Attn + 24 GDN.

- [ ] `parse_qwen35.rs` (needs `SHISU_TEST_MODEL`): open Q4_K_M; assert `version()==3`, `alignment()==32`, `tensors().len()==426`, `metadata()` KV count 46 and `architecture()=="qwen35"`; per-dtype counts `{F32:177, Q8_0:48, Q4_K:131, Q5_K:48, Q6_K:22}`; `token_embd` is `Q6_K` shape `[2560, 248320]`; `find("output.weight").is_none()`; every `TensorView.data.len() == desc.bytes` and every tensor lies inside the mapping. Then byte-compare the full manifest against `test-vectors/qwen35-q4km-manifest.tsv` (same dump format as Step 7) — exact equality of all 426 lines.

- [ ] `bind_qwen35.rs` (needs `SHISU_TEST_MODEL`): `bind_qwen35(&file, &profile)` with the fixture-derived profile; assert `layers.len()==32`, exactly **24 `Gdn` + 8 `Attn`**, Attn iff `(il+1)%4==0`; every Gdn layer has `attn.is_none()` and all 9 GDN tensors with the dims of Step 5 (e.g. `blk.0`: qkv `[2560,8192]` Q5_K, conv1d `[4,8192]` F32, alpha `[2560,32]` Q8_0); every Attn layer has `gdn.is_none()` and all 6 tensors (e.g. `blk.3`: q `[2560,8192]` Q4_K, k `[2560,1024]`, q_norm `[256]` F32); `lm_head.is_none()`; `data` slices are non-empty and offset-consistent (`desc.abs_offset + bytes ≤ file len`).

- [ ] Run: `cd shisu && cargo test -p shisu-gguf` (unit tests) and
  `SHISU_TEST_MODEL="$(scripts/test_model.sh)" cargo test -p shisu-gguf` (full).
  Expected before implementation: compile failure (empty modules); after: all pass, model tests SKIP without the env var.

## Acceptance

- [ ] `shisu/crates/shisu-gguf/src/{reader.rs,metadata.rs,quant.rs,quant_tables.rs,binding.rs}` exist; each < 1500 LoC (`wc -l`).
- [ ] `GgufFile::open`, `TensorView{name,dtype,data,shape}`, `bind_qwen35` public with the exact signatures above (`cargo doc -p shisu-gguf` or file inspection).
- [ ] Block descriptor consts match ds4.c:1051-1060 sizes; compile-time asserts present.
- [ ] `scripts/test_model.sh` executable; resolves the default local path; 2,740,937,888-byte check; never downloads.
- [ ] `test-vectors/qwen35-q4km-manifest.tsv` checked in, 426 lines.
- [ ] `cargo test -p shisu-gguf` green on macOS with and without `SHISU_TEST_MODEL`.
- [ ] Hybrid mapping: 24 GDN + 8 attn FFN blocks asserted by `bind_qwen35.rs`.
- [ ] No dequant kernels, no anyhow, no `ATLAS_` strings, no license headers.

## Commit

```
git add shisu/Cargo.toml shisu/crates/shisu-gguf shisu/scripts/test_model.sh shisu/test-vectors/qwen35-q4km-manifest.tsv shisu/Cargo.lock
git commit -m "feat(gguf): GGUF reader, quant block descriptors, qwen35 tensor binding"
```
