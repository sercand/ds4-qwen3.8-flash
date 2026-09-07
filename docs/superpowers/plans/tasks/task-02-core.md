# Task 2: shisu-core

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 2
**Depends on:** Task 1 — workspace `shisu/` with `[workspace.dependencies]`
(serde 1 derive, serde_json 1 preserve_order, thiserror 2, parking_lot 0.12 —
versions verified at `atlas/Cargo.toml:60,66,69,80`), stub crate
`shisu/crates/shisu-core/`, `[workspace.lints]` `warnings = "deny"` + clippy
denies, rust-toolchain 1.93.1, edition 2024. ASSUMPTION: task-01 plan file did
not exist when this was written; if T1's scaffold differs, adapt paths, never
names.

**Produces** (canonical module paths in comments; ALL re-exported at crate
root, so `shisu_core::DType` etc. work — already agreed with Task 3):

```rust
// backend.rs — object-safe; CUDA (T16) and Metal (T4) implement it.
pub struct DeviceBuffer { pub id: u64, pub bytes: u64 }            // owned alloc
pub struct DeviceView { pub id: u64, pub offset: u64, pub len: u64 } // non-owning range
pub struct GraphId(pub u64);
pub struct KernelRef { pub module: &'static str, pub name: &'static str }
pub struct Extent3 { pub x: u32, pub y: u32, pub z: u32 }
pub enum KernelArg<'a> { U32(u32), I32(i32), U64(u64), F32(f32), Ptr(DeviceView), Bytes(&'a [u8]) }
pub struct LaunchArgs<'a> { pub grid: Extent3, pub block: Extent3, pub smem_bytes: u32, pub args: &'a [KernelArg<'a>] }
pub trait Backend: Send + Sync {
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer>;
    fn free(&self, buf: DeviceBuffer) -> Result<()>;
    fn copy_h2d(&self, dst: DeviceView, src: &[u8]) -> Result<()>;
    fn copy_d2h(&self, src: DeviceView, dst: &mut [u8]) -> Result<()>;
    fn copy_d2d(&self, dst: DeviceView, src: DeviceView) -> Result<()>;
    fn run(&self, kernel: KernelRef, args: &LaunchArgs<'_>) -> Result<()>;
    fn supports_graphs(&self) -> bool;
    fn graph_capture(&self, body: &mut dyn FnMut() -> Result<()>) -> Result<GraphId>;
    fn graph_replay(&self, graph: GraphId) -> Result<()>;
    fn graph_release(&self, graph: GraphId) -> Result<()>;
    fn sync(&self) -> Result<()>;
    fn name(&self) -> &'static str;
}
// model.rs — object-safe; Qwen35Model (T6) and Qwen4ExpModel (T17) implement it.
pub struct SessionId(pub u64);
pub struct SampleRequest { pub temperature: f32, pub top_k: i32, pub top_p: f32, pub min_p: f32, pub rng: u64 }
pub trait Model: Send {
    fn load(&mut self, backend: &dyn Backend, model_path: &std::path::Path, profile: &ShapeProfile) -> Result<()>;
    fn open_session(&mut self) -> Result<SessionId>;
    fn close_session(&mut self, session: SessionId) -> Result<()>;
    fn prefill_chunk(&mut self, session: SessionId, tokens: &[u32], is_last_chunk: bool) -> Result<Option<DeviceView>>;
    fn decode_batch(&mut self, sessions: &[SessionId], tokens: &[u32]) -> Result<DeviceView>;
    fn sample(&mut self, logits: &DeviceView, row: u32, req: &mut SampleRequest) -> Result<u32>;
    fn teardown(&mut self) -> Result<()>;
}
// shape.rs
pub enum LayerKind { Gdn, Attn }
pub struct GqaShape { pub q_heads: u32, pub kv_heads: u32, pub head_dim: u32, pub n_rot: u32 }
pub struct GdnShape { pub v_heads: u32, pub qk_heads: u32, pub head_dim: u32, pub conv_kernel: u32 }
pub struct Qwen35Shape { pub n_layers: u32, pub hidden: u32, pub vocab: u32, pub gqa: GqaShape, pub gdn: GdnShape, pub layers: Vec<LayerKind>, pub intermediate_size: u32, pub attn_output_gate: bool, pub rms_eps: f32, pub rope_theta: f32, pub eos_token_id: u32 }
pub struct Qwen4ExpShape { pub n_layers: u32, pub hidden: u32, pub vocab: u32, pub gqa: GqaShape, pub mrope_sec: [u32; 3], pub gdn: GdnShape, pub n_expert: u32, pub n_expert_used: u32, pub n_expert_shared: u32, pub n_ff_exp: u32, pub n_indexer_heads: u32, pub n_indexer_head_dim: u32, pub n_indexer_top_k: u32, pub n_indexer_compress: u32, pub n_hc: u32, pub n_hc_lowrank: u32, pub full_attn_interval: u32, pub ple_layer: u32, pub n_ple_ngram: u32, pub n_ple_heads_per_ngram: u32, pub n_ple_head_dim: u32, pub n_ple_conv: u32, pub ple_eos_token: u32, pub rms_eps: f32, pub rope_theta: f32, pub rope_orig_ctx: u64 }
pub enum ShapeProfile { Qwen35(Qwen35Shape), Qwen4Exp(Qwen4ExpShape) }
impl ShapeProfile {
    pub fn qwen4exp() -> ShapeProfile;                       // fixed DS4_SHAPE_QWEN38F table
    pub fn n_layers(&self) -> u32;  pub fn hidden(&self) -> u32;  pub fn vocab(&self) -> u32;
    pub fn layer_kind(&self, il: u32) -> Option<LayerKind>;  // None past n_layers
}
// config.rs
impl HfConfig {
    pub fn from_file(path: &std::path::Path) -> Result<HfConfig>;
    pub fn from_json_str(text: &str) -> Result<HfConfig>;
    pub fn shape_profile(&self) -> Result<ShapeProfile>;     // unknown arch -> CoreError::UnknownArch
}
// dtype.rs — #[repr(u32)], discriminants = ds4 GGUF ids (ds4.c:2306-2327)
pub enum DType { F32=0, F16=1, Q4_0=2, Q5_0=6, Q5_1=7, Q8_0=8, Q2_K=10, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, IQ2_XXS=16, IQ4_NL=20, I32=26, BF16=30, MXFP4=39, Exl3K4=68, Exl3K5=69, Exl3K6=70, Exl3Ngram6=72 }
impl DType { pub fn from_gguf_id(id: u32) -> Option<DType>; pub fn gguf_id(self) -> u32; pub fn type_name(self) -> &'static str; pub fn block_elems(self) -> u32; pub fn block_bytes(self) -> u32; pub fn is_quantized(self) -> bool; }
// q4e_page.rs — consts below (verbatim) + host mirror of q4e_kv_row
pub fn kv_row(pages: &[u32], p: u32) -> u32;
// error.rs — thiserror
pub enum CoreError { UnknownArch(String), ShapeMismatch { field: &'static str, expected: u64, got: u64 }, ConfigMissing(String), BufferBounds { id: u64, offset: u64, len: u64, bytes: u64 }, SessionNotFound(u64), NotLoaded, Unsupported(&'static str), Backend { backend: &'static str, message: String }, Io(#[from] std::io::Error), Json(#[from] serde_json::Error) }
pub type Result<T> = std::result::Result<T, CoreError>;
```

## Global Constraints

- Rule 1: all new code under `shisu/`; no C/C++/ObjC host code (none here).
- Rule 2: 1500-LoC file cap (local gate `shisu/scripts/file_size_check.sh`) — every T2 file lands < 300 LoC; no splits needed.
- Rule 3: kernels byte-for-byte (no kernels in this task).
- Rule 4: no perf regression; CuMetal numbers never gate (no perf surface here).
- Rule 5: env knobs are `SHISU_*` only — shisu-core reads none.
- Rule 6: workspace lints `warnings = "deny"` + clippy denies; `thiserror`
  library errors (anyhow is binary-boundary only); `parking_lot` for locks.
- Rule 7: no license headers.
- Extras: pure Rust — NO cudarc/objc2/Metal/CUDA deps; must compile and test
  clean on macOS AND Linux; no `unsafe`; `Backend`/`Model` must stay
  object-safe (`Box<dyn Model>` is owned by the T10 scheduler thread);
  `teardown` is explicit, never `Drop` (atlas rationale,
  `spark-model/src/traits/model.rs:98-115`: "Drop can express neither the
  ordering nor the failure").

## Source References (verified)

| Source | Lines | What / how used |
|---|---|---|
| `ds4_q4e_page.h` | 1–47 (whole) | page geometry macros + rationale; consts copied verbatim (Step 4) |
| `ds4_qwen4exp_gpu.cuh` | 2647–2650 | `q4e_kv_row(pages,p)` = `pages[p>>SHIFT]*TOKENS + (p&MASK)` — mirrored by `q4e_page::kv_row` |
| `ds4.c` | 460–592 | "Model Shape Profiles" section; `ds4_shape` struct incl. GDN fields (561–564), `n_full_attn_interval`/`n_indexer_compress` (568–569) — field vocabulary |
| `ds4.c` | 754–802 | `DS4_SHAPE_QWEN38F` fixed table → `Qwen4ExpShape::qwen4exp()`; layer rule "(i+1) % 4 == 0 → QSA attention" (754–756) |
| `ds4.c` | 804–840 | global is `g_ds4_shape`. ⚠ DEVIATION: master plan calls it "g_shape"; real symbol `g_ds4_shape` |
| `ds4.c` | 2263–2304 | `gguf_types[]` block table (elems,bytes): f32(1,4) f16(1,2) q4_0(32,18) q5_0(32,22) q5_1(32,24) q8_0(32,34) q2_k(256,84) q4_k(256,144) q5_k(256,176) q6_k(256,210) q8_k(256,292) iq2_xxs(256,66) iq4_nl(32,18) mxfp4(32,**17**) exl3_k4(256,128) k5(256,160) k6(256,192) exl3_ngram6(160,122) → `DType::block_elems/block_bytes` |
| `ds4.c` | 2306–2327 | `DS4_TENSOR_*` ids → `DType` discriminants. No q4e tensor type exists — "q4e_*" is the qwen4exp family prefix (kv_row, page cache), not a GGUF dtype |
| `ds4_gpu.h` | 47–61 | flat tensor API mirrored by `Backend`: `_alloc(bytes)`:47, `_free`:50, `_write(t,off,data,bytes)`=copy_h2d:54, `_read`=copy_d2h:55, `_copy(dst,doff,src,soff,bytes)`=copy_d2d:56–58 |
| `ds4_gpu.h` | 111 | `ds4_gpu_synchronize(void)` → `Backend::sync` |
| `ds4_gpu.h` | 3132–3158 | decode-island graph capture (`ds4_gpu_decode_graph_begin/_end/_abort`, "CUDA backend; Metal … stay eager") → `graph_capture` + `supports_graphs` semantics |
| `ds4_metal.m` | 2577–2578 (pattern, ~200×) | `dispatchThreadgroups:threadsPerThreadgroup:` — Metal grid/block have the same 3-D semantics as CUDA `gridDim`/`blockDim`, so one `LaunchArgs` serves both |
| `ds4.h` | 596–598 | `ds4_sample_logits(logits, n_vocab, temperature, top_k, top_p, min_p, uint64_t *rng)` → `SampleRequest` field set (rng advanced in place) |
| `atlas …/traits/model.rs` | 98–115 | explicit-`teardown` Model trait pattern (narrowed to `Send`: Metal engine internals are not `Sync`) |
| HF config.json | https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json | fetched 2026-09-05 while authoring; content confirmed below |

⚠ **DEVIATION (source-corrected):** master plan Task 2 Interfaces says qwen35
GQA `head_dim:128`. The actual `Qwen/Qwen3.5-4B` config.json says
`"head_dim": 256` (with `partial_rotary_factor: 0.25` → `n_rot = 64`); the 128
belongs to the GDN heads (`linear_key_head_dim`/`linear_value_head_dim` = 128,
which matches). Validation assert below uses **256**. All other master-plan
Qwen35 constants are confirmed by the fetched config: `num_hidden_layers=32`,
`hidden_size=2560`, 16Q/4KV, GDN 32V/16QK head_dim 128 conv 4,
`vocab_size=248320`, `layer_types` = 8×(3×`linear_attention`+1×
`full_attention`), `full_attention_interval=4`, `intermediate_size=9216`,
`attn_output_gate=true`, `rms_norm_eps=1e-6`, `rope_theta=1e7`,
`eos_token_id=248044`. The config is a multimodal wrapper
(`Qwen3_5ForConditionalGeneration`, `model_type: "qwen3_5"`); the parser reads
`text_config` only — vision is out of scope (master plan Scope).

## Plan

- [ ] **Step 1: Fetch the fixture.**
  ```sh
  mkdir -p shisu/test-vectors
  curl -fsSL https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json \
      -o shisu/test-vectors/qwen35-config.json
  python3 -c "import json; c=json.load(open('shisu/test-vectors/qwen35-config.json'))['text_config']; \
      print(c['model_type'], c['num_hidden_layers'], c['head_dim'])"
  ```
  Expected: `qwen3_5_text 32 256`. If the fetch fails, STOP and report —
  NEVER hand-write a substitute fixture (a fabricated config would silently
  encode wrong constants).

- [ ] **Step 2: Deps.** In `shisu/crates/shisu-core/Cargo.toml` add
  `[dependencies] serde/serde_json/thiserror = { workspace = true }` and
  `[dev-dependencies] parking_lot = { workspace = true }` (fake-backend test).

- [ ] **Step 3: `error.rs`.** `CoreError` exactly as in Produces, with
  `#[error(...)]` display strings; `pub type Result<T>`.

- [ ] **Step 4: `q4e_page.rs`.** Consts verbatim from `ds4_q4e_page.h`
  (C names kept so the CUDA-phase `.cuh` mirror maps 1:1):
  ```rust
  pub const DS4_Q4E_PAGE_TOKENS: u32 = 256;   // ds4_q4e_page.h:33
  pub const DS4_Q4E_PAGE_SHIFT: u32 = 8;      // :34
  pub const DS4_Q4E_PAGE_MASK: u32 = DS4_Q4E_PAGE_TOKENS - 1; // :35
  pub const DS4_Q4E_MROPE_BACK: u32 = 4;      // :45
  pub const BYTES_PER_POSITION: u64 = 30 * 1024;              // :6-7 "30 KiB per position"
  pub const BYTES_PER_PAGE: u64 = BYTES_PER_POSITION * DS4_Q4E_PAGE_TOKENS as u64; // 7.5 MiB, :7
  ```
  Plus `kv_row(pages: &[u32], p: u32) -> u32` mirroring
  `q4e_kv_row` (ds4_qwen4exp_gpu.cuh:2647–2650). Doc comments carry the
  header's 4/32/64 divisibility rationale (:9–14) and the mrope-back
  rationale (:37–44).

- [ ] **Step 5: `dtype.rs`.** Enum + methods per Produces. `block_elems`/
  `block_bytes` transcribed from the `gguf_types[]` table row in Source
  References — note `MXFP4` is 32 elems / **17** bytes (ds4.c:2293), NOT 18
  like `Q4_0`/`IQ4_NL`; `is_quantized = block_elems() > 1`.

- [ ] **Step 6: `shape.rs`.** Structs per Produces. `Qwen4ExpShape::qwen4exp()`
  copies `DS4_SHAPE_QWEN38F` (ds4.c:759–802) verbatim:
  n_layers 48, hidden 2560, vocab 248320, gqa{24,2,256,64},
  mrope_sec [11,11,10], gdn{48,16,128,4}, n_expert 512, n_expert_used 10,
  n_expert_shared 1, n_ff_exp 640, n_indexer_heads 4, n_indexer_head_dim 128,
  n_indexer_top_k 2048, n_indexer_compress 4, n_hc 4, n_hc_lowrank 320,
  full_attn_interval 4, ple_layer 1, n_ple_ngram 3, n_ple_heads_per_ngram 8,
  n_ple_head_dim 160, n_ple_conv 4, ple_eos_token 248044, rms_eps 1e-6,
  rope_theta 1e7, rope_orig_ctx 262144.
  `Qwen4ExpShape::layer_kind(il)` = `Attn` iff `(il+1) % full_attn_interval == 0`
  (ds4.c:754–756 rule); `Qwen35Shape::layer_kind` indexes `layers`.
  Also `pub(crate) fn expect(field: &'static str, got: u32, want: u32) -> Result<()>`
  → `CoreError::ShapeMismatch`, used by config.rs.

- [ ] **Step 7: `config.rs`.** Two-stage parse so unknown archs give the typed
  error without arch-specific fields: `HfConfig` deserializes only
  `model_type: String`, `architectures: Vec<String>` (default), and raw
  `text_config: Option<serde_json::Value>`; `shape_profile()` matches
  `model_type`:
  - `"qwen3_5" | "qwen3_5_text"` → deserialize `text_config` into a private
    `Qwen35TextConfig` (fields: num_hidden_layers, hidden_size, vocab_size,
    num_attention_heads, num_key_value_heads, head_dim,
    linear_num_value_heads, linear_num_key_heads, linear_key_head_dim,
    linear_value_head_dim, linear_conv_kernel_dim, layer_types: Vec<String>,
    intermediate_size, attn_output_gate, full_attention_interval,
    rms_norm_eps, eos_token_id, rope_parameters{rope_theta,
    partial_rotary_factor}) → validate EVERY field against the Qwen35
    constants (DEVIATION-corrected head_dim 256) via `expect()` → build
    `Qwen35Shape`. `n_rot = (head_dim as f64 * partial_rotary_factor).round()`
    (= 64). `layer_types`: `linear_attention`→Gdn, `full_attention`→Attn,
    anything else → `ConfigMissing`; cross-check the list against the
    `(il+1) % 4` interval rule so a hand-edited fixture can't silently
    disagree.
  - `"qwen4exp"` → `ShapeProfile::qwen4exp()` (no public HF config; the ds4
    table is its SSOT).
  - anything else → `Err(CoreError::UnknownArch(model_type))`.

- [ ] **Step 8: `backend.rs`.** Types per Produces. Semantics to document on
  each method (mirrors ds4_gpu.h): `alloc` ids are backend-local, id 0 = null;
  `free` only accepts handles from `alloc` (views are a distinct type, so a
  double-free is a backend-tracked error → `CoreError::Backend`);
  `DeviceBuffer::view(offset, len)` bounds-checks against `bytes` →
  `CoreError::BufferBounds`; `copy_d2d` requires `dst.len == src.len`;
  `run` args are in kernel declaration order (CUDA `kernelParams`; Metal
  binding index = ordinal); `graph_capture` is CUDA-only — non-graph backends
  return `Err(CoreError::Unsupported(_))` and `supports_graphs() == false`
  (ds4_gpu.h:3132–3158 eager fallback); `sync` = device-wide idle.
  Every method takes `&self` — object safety is a hard constraint (no
  `self: &Arc<Self>` receivers).

- [ ] **Step 9: `model.rs`.** Types per Produces. Semantics: `load` is
  idempotent-error (`NotLoaded` guards every other method);
  `prefill_chunk` returns `Some(last-token logits view)` only when
  `is_last_chunk`, else `None` (state advanced); `decode_batch` requires
  `sessions.len() == tokens.len()`, returns `[n, vocab]` f32 logits, row i ↔
  sessions[i] (ds4's batched decode-island semantics, ds4_gpu.h:3132+);
  `sample` reads back row `row` via the backend, samples host-side (the
  bit-exact sampler port lands in T6/T17), and advances `req.rng` in place;
  `teardown` frees device memory in reverse construction order and is
  explicit, not Drop.

- [ ] **Step 10: `lib.rs`.** Replace T1 stub: `pub mod` for all seven modules
  + `pub use` re-exports of every Produces item at the crate root.

## Tests

All pure host code — run identically on macOS and Linux; no GPU gating; no
numerical tolerances (no math in this task). Run: `cargo test -p shisu-core`
and `cargo check -p shisu-core` (workspace `warnings = "deny"` applies).

- `tests/qwen35_config.rs`:
  - fixture at `concat!(env!("CARGO_MANIFEST_DIR"), "/../../test-vectors/qwen35-config.json")`
    parses → `ShapeProfile::Qwen35` with n_layers 32, hidden 2560, vocab
    248320, gqa{16,4,256,64}, gdn{32,16,128,4}, intermediate_size 9216,
    attn_output_gate true, and `layers` == 8×(Gdn,Gdn,Gdn,Attn) (assert per
    index against `(il+1) % 4`).
  - tampered fixture (load fixture, `serde_json::Value` mutate
    `text_config.num_hidden_layers = 31`, re-serialize) →
    `CoreError::ShapeMismatch { field: "num_hidden_layers", expected: 32, got: 31 }`.
  - `{"model_type":"llama","architectures":["LlamaForCausalLM"]}` →
    `CoreError::UnknownArch("llama")` (typed, not a serde error).
  - `ShapeProfile::qwen4exp()` spot-checks: n_layers 48, gdn.v_heads 48,
    n_expert 512, `layer_kind(3) == Attn`, `layer_kind(47) == Attn`,
    `layer_kind(48) == None`.
- `tests/q4e_page.rs`: `TOKENS == 1 << SHIFT`; `TOKENS % 4 == 0 && % 32 == 0
  && % 64 == 0` (header rationale); `BYTES_PER_PAGE == 7_864_320`;
  `kv_row(&[5,0,7], 300) == 7*256 + 44`.
- `tests/dtype.rs`: `from_gguf_id` round-trips every variant; unknown id 99 →
  `None`; pins q8_0 (32,34), q4_k (256,144), mxfp4 (32,17), iq2_xxs (256,66),
  exl3_ngram6 (160,122).
- `tests/traits_contract.rs`: in-memory `FakeBackend` (parking_lot Mutex +
  HashMap<u64,Vec<u8>>) and `FakeModel` implementing both traits — compile-
  time proof of object safety (`&dyn Backend`, `Box<dyn Model>`); h2d/d2h
  round-trip through a bounded `view`; out-of-range view → `BufferBounds`;
  double `free` → `Backend` error; `graph_capture` → `Unsupported`; Model
  lifecycle: pre-load call → `NotLoaded`, then load→open_session→prefill
  (mid-chunk `None`, last-chunk `Some`)→decode_batch→sample (rng advanced)→
  teardown. This fake is the pattern T16/T17 reuse for host-side unit tests.

## Acceptance

- [ ] `shisu/test-vectors/qwen35-config.json` exists, fetched (not hand-written).
- [ ] `shisu-core/src/{shape,config,dtype,error,backend,model,q4e_page}.rs` exist; each < 1500 LoC.
- [ ] `q4e_page.rs` consts byte-equal in value to `ds4_q4e_page.h` (file inspection: 256/8/255/4, 30 KiB, 7.5 MiB).
- [ ] `cargo test -p shisu-core` green on macOS; same test list compiles on Linux (local gate `scripts/check.sh`, T20).
- [ ] `Backend::run(&self, KernelRef, &LaunchArgs)` and `Model` five-method + session methods match the SSOT names exactly; traits object-safe (contract test compiles).
- [ ] Unknown arch yields `CoreError::UnknownArch`, not a serde error.
- [ ] No cudarc/objc2/Metal deps in `shisu-core/Cargo.toml`.

## Commit

```sh
git add shisu/crates/shisu-core shisu/test-vectors/qwen35-config.json
git commit -m "feat: shisu-core traits, shape profiles, q4e page geometry"
```
