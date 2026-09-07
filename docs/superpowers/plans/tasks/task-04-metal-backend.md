# Task 4: shisu-metal backend

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 4 (read it before executing; the spec argues this plan)

**Depends on:** T1, T2 (consumed, exact names from `task-02-core.md`), T3 (name-level only):
- `shisu_core::{Backend, DeviceBuffer{id,bytes}, DeviceView{id,offset,len}, KernelRef{module,name}, KernelArg, LaunchArgs{grid,block,smem_bytes,args}, Extent3, GraphId, CoreError, Result}` — trait signatures verbatim from T2's Produces block; `CoreError::Unsupported(&'static str)` / `CoreError::Backend{backend,message}` / `CoreError::BufferBounds{..}` are the only error paths this crate raises.
- T1: crate stub `shisu/crates/shisu-metal/` with `default = ["metal"]`, `metal = []` (this task binds the optional objc2 deps), workspace deps `objc2 0.6`, `objc2-metal 0.3`, `objc2-foundation 0.3`, `parking_lot`, `tracing`.
- T3: `GgufFile::open_shared` → shared RO mmap; `TensorView.data` is a borrow into it. This task's `register_no_copy` wraps such slices as zero-copy `MTLBuffer`s (no compile-time dep on shisu-gguf).

**Produces** (public API T5/T6 build on; paths `shisu/crates/shisu-metal/src/`):
```rust
// lib.rs — everything below is #[cfg(feature = "metal")]
pub struct MetalBackend { /* device, queue, library, buffer table, pipeline cache */ }
impl MetalBackend {
    pub fn new() -> Result<MetalBackend>;   // MTLCreateSystemDefaultDevice + queue + runtime MSL compile
    /// Wrap a slice backed by a shared file mapping (T3 open_shared) as a
    /// zero-copy buffer. Caller must keep the mapping alive longer than the id.
    pub fn register_no_copy(&self, bytes: &[u8]) -> Result<DeviceBuffer>;
}
pub fn metal_available() -> bool;           // probe; tests skip when false
impl shisu_core::Backend for MetalBackend { /* all 12 T2 methods, verbatim signatures */ }
```
- `name() -> "metal"`, `supports_graphs() -> false`, `graph_capture/graph_replay/graph_release` → `Err(CoreError::Unsupported("metal: …"))` (master plan SSOT: "Metal uses retained command-encoders"; eager per T2/ds4_gpu.h:3132-3158).
- **Specialization convention** (T2's `LaunchArgs` has no function-constant slot): `KernelRef.name` may carry a suffix `#<idx>:<t><val>[,<idx>:<t><val>…]`, `t ∈ {s = short/i16, b = bool}`, parsed in `pipeline.rs`; e.g. `kernel_mul_mv_ext_q4_K_f32_r1_1#600:s2,601:s16`. Pipeline cache key = the full name string → master-plan "keyed by (kernel, specialization)". `KernelRef.module` is metadata (file stem), not a lookup key — one flat runtime library.
- **Launch geometry** (T5/T6 grid formulas depend on this): `LaunchArgs.grid` IS the threadgroup count dispatched (CUDA `gridDim` semantics — copied kernels read `tgpig` as `blockIdx`; T2 SSOT note at `ds4_metal.m:2577`), `LaunchArgs.block` = threads per threadgroup (`blockDim`). `run()` passes both straight to `dispatchThreadgroups:threadsPerThreadgroup:` — no ceil/division anywhere.
- T5 appends its `metal/qwen35/*.metal` entries to the `SOURCES` list in `compile.rs` (append-only edit) and launches through the same `Backend::run`.

## Global Constraints

Master plan Rules 1–7, one line each:
1. All new code under `shisu/`; the only non-Rust files are the copied `.metal` kernels (rule-1 allowed kinds).
2. 1500-LoC cap applies to `.metal` files too — dense.metal (2459 ln) is pre-split by family (Step 3 table); every Rust file lands < 400 LoC.
3. Kernels copied byte-for-byte; only whole dsv4/glm kernels are deleted, never edited; arithmetic-order changes are bugs.
4. No perf gate in this task (Metal baselines are recorded in T6/T20); no timing claims.
5. Env knobs `SHISU_`-prefixed: this task adds `SHISU_METAL_SOURCE_DIR` (diagnostic source override, ds4's per-file `DS4_METAL_*_SOURCE` pattern collapsed to one dir, rule 5).
6. `warnings = deny` + clippy deny; errors are `shisu_core::CoreError` (no new error type, no anyhow); `parking_lot` locks; `tracing` for compile/pipeline logs.
7. No license headers.

Task-specific:
- Run cargo from `shisu/`. Do not run formatters/linters/project-wide suites; `cargo test -p shisu-metal` only (T20 owns the final gate).
- GPU tests are `#[ignore]`-gated (gate-safe); local proof is `-- --ignored`.
- No `unsafe` outside `run.rs`/`device.rs` objc2 call sites and the memcpy/no-copy wrap.

## Source References (verified)

ds4 root = cwd; every line opened while writing this plan. `objc2-metal` rows = `~/.cargo/registry/src/*/objc2-metal-0.3.2/`.

| Source | Lines | What lives there / how used |
|---|---|---|
| `ds4_metal.m` | 4267–4324 | base prelude `ds4_gpu_source`: `metal_stdlib` include + `#ifdef DS4_METAL_HAS_TENSOR` MPP includes, `MAX/MIN/SWAP/QK8_0=32/QK_K=256/N_SIMDWIDTH=32/N_R0_Q8_0=2/N_SG_Q8_0=4/FC_MUL_MV=600/FC_MUL_MM=700/FC_BIN=1300/FOR_UNROLL/M_PI_F`, `kernel_touch_u8_stride`, `enum ds4_sort_order`, `block_q8_0`/`block_q8_K` → becomes `metal/generic/00_common.metal` (drop the tensor-API ifdefs — out of scope). |
| `ds4_metal.m` | 4326–4396 | `ds4_gpu_full_source()`: `required_sources` array :4334–4357 (`@{env-override-name, path}` pairs, 22 files); concat loop :4359–4394 — env path, then `path`, then `./path`; missing → `fprintf` + nil; append `"\n// appended <path>\n<content>\n"`. Pattern for `compile.rs`. |
| `ds4_metal.m` | 6381–6485 | init: `MTLCreateSystemDefaultDevice()` :6381, `newCommandQueue` :6389, `MTLCompileOptions` :6430, `preprocessorMacros` :6476, `newLibraryWithSource:options:error:` :6477; compile failure → `fprintf(stderr, … localizedDescription)` :6479–6480. |
| `ds4_metal.m` | 2211–2234 / 2141–2175 / 3012–3069 | pipeline cache `g_pipeline_cache` (dict, key string): plain `ds4_gpu_get_pipeline` (key = name); `ds4_gpu_get_mul_mm_pipeline` (key `%s_bci=%d_bco=%d`, `MTLFunctionConstantValues` bool @700/701); `ds4_gpu_get_mul_mv_ext_pipeline` (key `%s_nsg=%d_nxpsg=%d`, short @600/601) → `newFunctionWithName:constantValues:error:` + `newComputePipelineStateWithFunction:error:`. |
| `ds4_metal.m` | 8850–8893 | `ds4_gpu_tensor_write/read` = bounds-check + `memcpy` into `[buffer contents]` (unified memory, no blit) → `copy_h2d/copy_d2h`; `ds4_gpu_tensor_copy` = `MTLBlitCommandEncoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` :8884–8891 → `copy_d2d`. |
| `ds4_metal.m` | 11492–11563 | `ds4_gpu_wrap_model_exact_range_impl`: page-align offset down (`page_offset`, `leading`), round len up to page, clamp to mapping, `maxBufferLength` check :11523, `newBufferWithBytesNoCopy:…deallocator:nil` :11543, options `MTLResourceStorageModeShared` (`ds4_gpu_model_resource_options` :1237–1243), cache key `map:size:page_offset:view_bytes` → `register_no_copy`. |
| `ds4_metal.m` | 1268–1270, 1994–1995 | alloc: `newBufferWithLength:options:MTLResourceStorageModeShared` → `Backend::alloc`. |
| `ds4_metal.m` | 2577–2578 | `dispatchThreadgroups:MTLSizeMake(ceil(n/256),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)` — grid/block are CUDA `gridDim`/`blockDim` semantics (T2 SSOT note): `LaunchArgs.grid` IS the dispatched threadgroup count, `block` = threads per threadgroup; `run()` passes `grid` straight to `dispatchThreadgroups` (copied kernels read `tgpig` as blockIdx). |
| `ds4_metal.m` | 10160–10163 | `ds4_gpu_synchronize` → `sync()` (commit + `waitUntilCompleted`). |
| `ds4_metal.m` | 18346–18545 | q4_K gemv host path: ext-family name table `kernel_mul_mv_ext_q4_K_f32_r1_{1..5}` :18346–18370; dispatch :18517–18544 — `nsg=2`, `nxpsg=ds4_gpu_mv_ext_nxpsg`, `r1ptg=1` for n_tok=1, args via `setBytes atIndex:0`, buffers w/x/out @1/2/3, grid `((out_dim+r0ptg-1)/r0ptg, (n_tok+r1ptg-1)/r1ptg, 1)`, block `(32, nsg, 1)`, `r0ptg=(32/nxpsg)*nsg`. NOTE :18468–18474: the faster "classic" `kernel_mul_mv_q4_K_dense_f32` lives in `metal/moe.metal:3372` (MoE file, out of scope) — see ⚠ DEVIATION 2. |
| `ds4_metal.m` | 5029–5074 | `ds4_gpu_make_mv_ext_args` (field values for a plain [out,in]×[in,n] gemv), `ds4_gpu_mv_ext_nxpsg` (16 if in%256==0 && n_tok<3, else 8/4), `ds4_gpu_mv_ext_r1ptg` — port to the test helper. |
| `ds4_metal.m` | 10699–10739 | get_rows q4_K/q4_0 dispatch: `ds4_gpu_get_rows_q8_0_args`, block 256 (q4_K), grid `(ceil(n_embd/256), n_tokens, 1)`, bindings args/w/tokens/out @0..3 (tokens may be `setBytes` for a single id). |
| `ds4_metal.m` | 22319–22378 | softmax dispatch: args fill (scale=1, max_bias=0), `_4` when width%4==0, threads pow2 ≥32 ≤ width/4, grid `(rows, planes, 1)`, shmem 32×f32, bindings args/src/src/src/dst @0..4. |
| `ds4_metal.m` | 21174–21189 | rms_norm dispatch: grid `(rows, planes, 1)`, threads ≤ `maxTotalThreadsPerThreadgroup`, shmem 32×f32, bindings args/src/src/src/src @0..4 (src aliased 4× for the F=1 path). |
| `ds4_metal.m` | 31923–31945 | get_rows f32 dispatch: `ds4_gpu_get_rows_args`, grid `(ne10*ceil(ne00t/block.x), ne11, ne12)`, block (1,1,1) in ds4 (kernel processes one element per thread — `break` at get_rows.metal:72). |
| `metal/{norm,softmax,unary,bin,glu,concat,cpy,get_rows,set_rows,sum_rows,repeat,argsort,dsv4_rope}.metal` | wc: 385/241/312/217/63/62/247/186/55/102/52/275/883 | each file is self-contained (defines its own `ds4_metal_args_*` structs at top); entry points + strip list in Step 3. |
| `metal/dense.metal` | 2459 (wc) | q4_K/q8_0 claims verified: `ds4_dense_block_q4_K` **:1494**, `dequantize_dense_q4_K` **:1523**, `dequantize_dense_q4_K_t4` **:1587**, `kernel_mul_mv_ext_q4_K_f32_r1_{1..5}` instantiations **:1889–1893**, `kernel_mul_mv_q8_0_f32` **:187–188**, `kernel_mul_mv_q8_0_f32_pair` **:205–206**, `kernel_mul_mv_ext_q8_0_f32_r1_{2..5}` :1878–1881, ext impl :1596–1694, disp :1698–1710, `ds4_metal_args_mul_mv` :6–50, `ds4_metal_args_mul_mv_ext` :52–71, FC decls :3–4. MPP/nax block is `#ifdef DS4_METAL_HAS_TENSOR` :1903–2036 (strip). |
| `metal/get_rows.metal` | 3–37, 79–81, 150 | `ds4_metal_args_get_rows`, `ds4_metal_args_get_rows_q8_0` (also used by the q4_K kernel), `ds4_get_rows_block_q4_K` :32, host names `kernel_get_rows_{f32,f16,i32}` :79–81, `kernel_get_rows_q4_K_f32` :150. Only f32/f16/i32/q8_0/q4_0/q4_K exist — **no q6_k path** (see risk note). |
| `metal/moe.metal` | 3372 | `kernel_mul_mv_q4_K_dense_f32` (classic fast gemv) — MoE file, NOT copied; ext family covers the tests. |
| `objc2-metal-0.3.2/Cargo.toml` | `[features]` | `MTLFunctionConstantValues` is its own feature (NOT in T1's list); `MTLCompileOptions` ships under `MTLLibrary` (already listed). See ⚠ DEVIATION 3. |

⚠ **DEVIATION 1 (master-plan Files list):** the 13 listed files do not contain the q4_k/q8_0 gemv kernels the task's own checkbox and test list require — they live in `metal/dense.metal` (not in the Files list). This plan extracts the q8_0/q4_K/f32-f16 matvec families from dense.metal into `metal/generic/mul_mv_q8_0.metal` + `mul_mv_ext.metal` + `mul_mv_dense.metal` (Step 3). The generic prefill GEMM `kernel_mul_mm*` (:2038–2459) is NOT copied — prefill GEMM is T5's `gemm_q4k.metal`; T5/T6 may copy `kernel_mul_mm` later if needed.

⚠ **DEVIATION 2 (master-plan checkbox wording):** "…plus `kernel_mul_mv_q8_0_f32` / `kernel_mul_mv_q8_0_f32_pair`" verified in dense.metal :187/:205 ✓, but the *fast* q4_K decode gemv `kernel_mul_mv_q4_K_dense_f32` cited by the host path (ds4_metal.m:18474) is in `metal/moe.metal:3372`, not dense.metal. MoE is out of scope (master plan Scope); the ext family (`kernel_mul_mv_ext_q4_K_f32_r1_*`) is the copied q4_K gemv.

⚠ **DEVIATION 3 (T1 manifest extension):** add `"MTLFunctionConstantValues"` to the workspace `objc2-metal` feature list (verified feature in objc2-metal 0.3.2; needed for `newFunctionWithName:constantValues:error:`). `dispatch2` from T1's list is NOT used by this crate — only `newLibraryWithData` needs `DispatchData`; we use `newLibraryWithSource`.

⚠ **DEVIATION 4 (source loading):** ds4 reads `.metal` files from CWD at runtime with per-file env overrides (:4359–4367). shisu uses `include_str!` into a static `SOURCES` list (deterministic, CWD-independent, gate-safe) + one `SHISU_METAL_SOURCE_DIR` override that re-reads a file by name from disk — same diagnostic power

**Risk note (for T6, not fixed here):** the Q4_K_M test GGUF has `token_embd` = **Q6_K** (T3 plan, Tests) and 48 Q5_K tensors; ds4 Metal has no q5_k/q6_k get_rows or gemv (verified: get_rows.metal carries f32/f16/i32/q8_0/q4_0/q4_K only). The engine must repack those tables host-side or T5 adds kernels — recorded here so it is not discovered silently in T6.

## Plan

- [ ] **Step 1: Manifest.** `shisu/Cargo.toml`: add `"MTLFunctionConstantValues"` to the `objc2-metal` features (DEVIATION 3). `shisu/crates/shisu-metal/Cargo.toml`: `[dependencies]` = `shisu-core`, `tracing`, `parking_lot` + optional `objc2`, `objc2-metal`, `objc2-foundation` (all `workspace = true`); no `thiserror`/`anyhow` — errors are `shisu_core::CoreError` (rule 6); `[features] default = ["metal"]`, `metal = ["dep:objc2", "dep:objc2-metal", "dep:objc2-foundation", "shisu-core/metal"]`; `[dev-dependencies] memmap2` (no-copy test only).

- [ ] **Step 2: `metal/generic/00_common.metal`.** Transcribe the prelude from `ds4_metal.m:4267–4324` as real MSL (not a C string): includes (`metal_stdlib` only — drop the `DS4_METAL_HAS_TENSOR` branches), all `#define`s verbatim, `kernel_touch_u8_stride`, `enum ds4_sort_order`, `struct block_q8_0`, `struct block_q8_K`.

- [ ] **Step 3: Copy + adapt the kernel set** into `shisu/crates/shisu-metal/metal/generic/`. "verbatim" = byte-for-byte minus the listed deletions (whole kernels + their now-unused structs). Each result < 1500 LoC:

| dst | src (ln) | keep | strip (dsv4/glm specializations) |
|---|---|---|---|
| `norm.metal` | norm.metal (385) | :1–183 — `ds4_metal_args_norm`, `kernel_rms_norm_f32_4`/`_mul_f32_4` (host_name :84–85), `kernel_add_rms_norm_mul_f32_4` :87, `kernel_rms_norm_scale_f32_4` :137 | `kernel_dsv4_qkv_rms_norm_f32_4` :185, `kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32` :254 (+ their arg structs) |
| `softmax.metal` | softmax.metal (241) | whole file — `kernel_soft_max_f32`/`_4` :240–241 | none |
| `unary.metal` | unary.metal (312) | everything except next cell — `kernel_unary_f32_f32{,_4}`, `kernel_unary_f16_f16` :310–312 (FC_BIN=1300 op select) | `kernel_dsv4_softplus_sqrt_f32_4` :290–307 |
| `bin.metal` | bin.metal (217) | whole file — `kernel_bin_fuse_f32_f32_f32` :196 (FC_BIN op select), `kernel_add2_f32` :198, `kernel_add3_f32` :208 | none |
| `glu.metal` | glu.metal (63) | whole file — `kernel_swiglu_f32` :16, `kernel_swiglu_flat_f32` :42 | none |
| `concat.metal` | concat.metal (62) | whole file — `kernel_concat` :34 | none |
| `cpy.metal` | cpy.metal (247) | :1–149 — `kernel_cpy_{f32_f32,f32_f16,f16_f32,f16_f16}` :55–58, `kernel_cpy_contig_f32_f16_4` :65, `kernel_cpy_contig_f16_f32_4` :89, `kernel_cpy_contig_f16_f16_bits_4` :115 | `kernel_dsv4_flash_kv_stage_f16` :151–247 |
| `get_rows.metal` | get_rows.metal (186) | whole file — `kernel_get_rows_{f32,f16,i32}` :79–81, `kernel_get_rows_q8_0_f32` :83, `kernel_get_rows_q4_0_f32` :115, `kernel_get_rows_q4_K_f32` :150 | none |
| `set_rows.metal` | set_rows.metal (55) | whole file — `kernel_set_rows_f32_i32` :55 | none |
| `sum_rows.metal` | sum_rows.metal (102) | whole file — `kernel_sum_rows_f32_f32` :102 | none |
| `repeat.metal` | repeat.metal (52) | whole file — `kernel_repeat_f32` :52 | none |
| `argsort.metal` | argsort.metal (275) | whole file — `kernel_argsort_f32_i32_desc` :117, `kernel_argsort_merge_f32_i32_desc` :275 | none |
| `dsv4_rope.metal` | dsv4_rope.metal (883) | structs :1–101 + `kernel_dsv4_rope_tail_f32` :102, `…_inplace_pair` :208, `…_inplace_pair_affine` :476 — ds4-style partial rope: copies the nope PREFIX, rotates the TAIL with YaRN corr_dims (:99–136). Kept per master-plan file list only; **NOT qwen35 rotate_half** (first `n_rot` dims, pairs `(i, i+half)`, `ds4_qwen4exp_gpu.cuh:2101–2115`) — T5's `kernel_q35_{q,k}_norm_rope_*` replace it; T6 must not launch these for qwen35 (task-05 DEVIATION 3) | `…_inplace_pair_shared4` :277, `kernel_dsv4_head_rms_norm_rope_tail_f32` :349, `kernel_dsv4_kv_rope_fp8_store_f32` :508, `kernel_flash_attn_ext_vec_reduce_rope` :594, `kernel_dsv4_comp_row_finalize_f32` :647, `kernel_dsv4_flash_attn_vec_packed32_reduce_rope_f16_dk512_dv512` :832 |
| `mul_mv_q8_0.metal` (new) | dense.metal | :1–71 (FC decls + `ds4_metal_args_mul_mv` + `ds4_metal_args_mul_mv_ext`), :73–198 (`helper_mv_reduce_and_write`, `kernel_mul_mv_q8_0_f32_impl`, `kernel_mul_mv_q8_0_f32`), :205–451 (`kernel_mul_mv_q8_0_f32_pair`), dequantize_q8_0 :1476, _pairs :1548, _t4 :1564 | everything else (≈480 LoC result) |
| `mul_mv_ext.metal` (new) | dense.metal | `ds4_dense_block_q4_K` :1494, `dequantize_dense_q4_K` :1523, `dequantize_dense_q4_K_t4` :1587, `kernel_mul_mv_ext_q4_f32_impl` :1596–1694, `_disp` :1698–1710, instantiations :1878–1881 (q8_0 r1_2..5) + :1889–1893 (q4_K r1_1..5) | f32/f16/q4_0 ext instantiations, `kernel_mul_mv_ext_q8_0_pair_swiglu_*` :1719–1898, MPP/nax :1903–2036, `kernel_mul_mm*` :2038–2459 (T5 territory), dsv4 fused matvecs :454–1440 |
| `mul_mv_dense.metal` (new) | dense.metal | `kernel_mul_mv_t_t` :772–789 (f32/f16), `kernel_mul_mv_t_t_4` :890–907, `kernel_mul_mv_t_t_short` :1442–1463, `dequantize_f32/f16` :1465–1473 | compressor pair/quad + dsv4 fusions :909–1440 |

- [ ] **Step 4: `compile.rs`.** `const SOURCES: &[(&str name, &str src)]` = `include_str!("../metal/generic/…")` in fixed order: `00_common` first, then the Step-3 files (order matters only for `#define` visibility; the prelude covers it). `pub(crate) fn full_source() -> String`: join with `"\n// appended <name>\n"` (ds4 :4393 marker kept for compile-error line triage); when `SHISU_METAL_SOURCE_DIR` is set, re-read `<dir>/<name>` and substitute (missing file → `CoreError::Backend` naming it, ds4 :4387–4392 semantics). `pub(crate) fn compile_library(device) -> Result<Retained<MTLLibrary>>`: `MTLCompileOptions::new()`, `newLibraryWithSource_options_error`; on `Err` → `tracing::error!` the full `localizedDescription` + source length, return `CoreError::Backend{backend:"metal", message}` (ds4 :6479–6480 path). No preprocessorMacros (drift flags are dsv4-only).

- [ ] **Step 5: `pipeline.rs`.** `parse_kernel_name(name) -> Result<(&str entry, Vec<FnConst>)>` implementing the `#idx:t{val}` suffix (unit-tested, non-ignored). `PipelineCache = HashMap<String, Retained<MTLComputePipelineState>>` (key = full `KernelRef.name`): empty constants → `library.newFunctionWithName`; else `MTLFunctionConstantValues::new()` + `setConstantValue_atIndex_type` (`s` → `MTLDataTypeShort` i16, `b` → `MTLDataTypeBool`) → `newFunctionWithName_constantValues_error` → `device.newComputePipelineStateWithFunction_error` (ds4 :2211–2234/:3012–3069 pattern; no fast-lookup memo — that's a ds4 perf hack, out of scope).

- [ ] **Step 6: `device.rs`.** `MetalBackend { device, queue, library, inner: Mutex<BackendState> }`; `BackendState { next_id: u64 (start 1; 0 = null per T2), buffers: HashMap<u64, BufferEntry> }`, `BufferEntry { buffer: Retained<MTLBuffer>, base: u64, bytes: u64 }` (`base` = leading page slack for no-copy wraps; owned allocs use 0). `new()`: `metal_available()` check, `MTLCreateSystemDefaultDevice`, `newCommandQueue`, `compile_library`. `alloc`: `newBufferWithLength_options(MTLResourceStorageModeShared)` (ds4 :1268). `free`: remove entry, missing id → `CoreError::Backend` (double-free per T2). `register_no_copy`: ds4 :11492–11563 pattern — page-align down, round up, clamp, `maxBufferLength` check, `newBufferWithBytesNoCopy_length_options_deallocator(ptr-page_offset, view_bytes, Shared, None)`, entry `base = offset & (page-1)`, `bytes = len`; doc-comment states the mapping-lifetime contract (T3 `open_shared` must outlive the id). `copy_h2d`/`copy_d2h`: bounds-check `view.offset+view.len ≤ entry.bytes` → `BufferBounds`, then `memcpy` via `buffer.contents() + base + offset` (ds4 :8850–8868). `copy_d2d`: equal `len` (T2), blit encoder `copyFromBuffer…size` + commit (ds4 :8884–8891). `sync`: wait on retained committed command buffers, drain list. `metal_available()`: probe `MTLCreateSystemDefaultDevice()` non-null.
  Thread-safety: `MetalDevice`/`MTLCommandQueue`/`MTLBuffer`/`MTLComputePipelineState` are `Send + Sync` in objc2 (documented thread-safe on the Metal side), so `MetalBackend` is `Send + Sync` as T2's `Backend: Send + Sync` requires; only the `HashMap`s sit behind `parking_lot::Mutex`. Command buffers are committed per op and retained in a `Vec` drained by `sync()` (eager model; T6 may batch later).

- [ ] **Step 7: `run.rs`.** `Backend::run`: resolve pipeline (Step 5); launch geometry is CUDA-semantics passthrough (T2 SSOT, ds4 :2577): `dispatchThreadgroups_threadsPerThreadgroup(grid → MTLSize, block → MTLSize)` — `grid` is the threadgroup count, `block` the threads per threadgroup, NO ceil conversion; validate `block.x*block.y*block.z ≤ pipeline.maxTotalThreadsPerThreadgroup` → `CoreError::Backend`; new command buffer + compute encoder; bind `args` in order: ordinal i → binding index i (T2 SSOT) — `Ptr(view)` → buffer lookup + `setBuffer_offset_atIndex(entry.buffer, entry.base + view.offset, i)`, scalars/`Bytes` → `setBytes_atIndex`; `smem_bytes > 0` → `setThreadgroupMemoryLength_atIndex(0)`; `endEncoding`, commit, retain CB for `sync()`. `graph_*`/`supports_graphs` stubs per Produces.

- [ ] **Step 8: `lib.rs`.** Keep T1's deny attrs + doc comment; `#[cfg(feature = "metal")] pub mod` the four modules (private is fine — re-export `MetalBackend`, `metal_available` at crate root); without the feature the crate is an empty stub.

## Tests

macOS only; GPU tests `#[ignore = "requires Metal GPU"]` so the default gate (`cargo test --workspace`, no `--ignored`) never needs a GPU. Run: `cd shisu && cargo test -p shisu-metal` then `cargo test -p shisu-metal -- --ignored`. Tolerances (master plan Task 4): **1e-3 quantized, 1e-5 f32**. CPU oracles = tiny hand-written Rust f32 loops in `tests/oracle.rs` (no deps, no ds4 code): rms_norm (sum-of-squares → scale), row softmax (max-subtract/exp/sum), gather, q4_K dequant (6-bit scale/min unpack per `metal/get_rows.metal:39–43` semantics) + dot product.

- [ ] `tests/name_parse.rs` (NOT ignored): the `#idx:t{val}` suffix parser — no suffix, one `s`, mixed `s`/`b`, malformed (`#600:x2`, `#abc`, trailing `#`) → typed error.
- [ ] `tests/metal_kernels.rs` (all ignored; skip with a printed note when `!metal_available()`):
  1. **compile clean**: `MetalBackend::new()` is `Ok`; pipeline lookup succeeds for every host-visible kernel name in the Step 3 files (enumerate them in the test) + the parameterized `kernel_mul_mv_ext_q4_K_f32_r1_1#600:s2,601:s16`.
  2. **contract**: `name()=="metal"`, `supports_graphs()==false`, `graph_capture` → `Unsupported`; alloc→h2d→d2h round-trip equality; `copy_d2d` equality; out-of-range view → `BufferBounds`; double `free` → `Backend`.
  3. **no-copy wrap** (dev-dep `memmap2`): write a temp file, `MmapOptions` shared map, `register_no_copy` on a mid-mapping slice (deliberately page-misaligned), d2h the full slice back → byte equality; drop the buffer id before dropping the map.
  4. **norm**: `kernel_rms_norm_f32_4`, rows=3 × n=128 f32 (ne00_t=32), grid `(3,1,1)`, block `(32,1,1)`, smem 128 B, args `ds4_metal_args_norm` (fill per `metal/norm.metal:40–43` semantics; weight variant via `kernel_rms_norm_mul_f32_4`) vs oracle, tol 1e-5.
  5. **softmax**: `kernel_soft_max_f32_4`, rows=2 × width=64, grid `(2,1,1)`, block `(32,1,1)`, smem 128 B, bindings args/src/src/src/dst @0..4 (mask/sinks alias src, ds4 :22369–22374), tol 1e-5.
  6. **get_rows**: `kernel_get_rows_f32` (table 8×128 f32, ids [3], grid `(3*ceil(128/1),1,1)`, block `(1,1,1)`) tol 1e-5; `kernel_get_rows_q4_K_f32` (table 8 rows × 256-elt q4_K blocks — synthetic blocks per test 7 — `ds4_metal_args_get_rows_q8_0`, grid `(ceil(256/256),3,1)`, block `(256,1,1)`, bindings args/w/tokens/out @0..3 per ds4 :10728–10737) tol 1e-3.
  7. **q4_k gemv**: `kernel_mul_mv_ext_q4_K_f32_r1_1#600:s2,601:s16`, in_dim=256, out_dim=32, n_tok=1 → nxpsg=16, r0ptg=4, grid `(8,1,1)`, block `(32,2,1)`, `smem_bytes=0` (the ext path sets no threadgroup memory, ds4 :18517–18544); args per `ds4_gpu_make_mv_ext_args` (ds4_metal.m:5029–5055) called as `(in_dim, out_dim, n_tok, row_bytes, row_bytes)` → `nb00 = nb01 = 144` (q4_K row bytes); synthetic blocks: `d=1.0f`, `dmin=0`, `scales[12] = {01,01,01,01,00,00,00,00,01,01,01,01}` (verified against `ds4_get_rows_q4_K_scale_min`, get_rows.metal:39–43: every group scale unpacks to 1, every min to 0), `qs` nibbles cycling 0..15 → dequantized row = the exact nibble values; oracle = plain dot with those ints, tol 1e-3.
- [ ] Expected before implementation: `cargo test -p shisu-metal` fails to compile (stub crate); after: unit tests pass, GPU tests pass under `--ignored` on this M1 Max.

## Acceptance

- [ ] `shisu/crates/shisu-metal/src/{device.rs,pipeline.rs,compile.rs,run.rs}` + `metal/generic/*.metal` exist; every `.metal` and `.rs` file < 1500 LoC (`wc -l`).
- [ ] `Backend` impl signatures match T2 verbatim (file inspection vs `task-02-core.md` Produces); `graph_capture` → `Unsupported`, `supports_graphs() == false`.
- [ ] `MetalBackend::register_no_copy(&self, bytes: &[u8]) -> Result<DeviceBuffer>` public (T6 weight binding).
- [ ] Kernel copies are deletions-only: `diff` of each copied file vs its ds4 original shows only removed blocks (rule 3); no `DS4_METAL_HAS_TENSOR`/MPP code, no dsv4/glm kernel names remain except the kept rope-tail family.
- [ ] dense.metal claims re-verified in the copied files: `ds4_dense_block_q4_K`, `dequantize_dense_q4_K{,_t4}`, `kernel_mul_mv_ext_q4_K_f32_r1_{1..5}`, `kernel_mul_mv_q8_0_f32{,_pair}` present.
- [ ] `cargo test -p shisu-metal` green (unit) and `cargo test -p shisu-metal -- --ignored` green on macOS; GPU tests are `#[ignore]`d so the default gate never runs them.
- [ ] No `ATLAS_` strings, no license headers, no new crates.io deps beyond `memmap2` (dev) + workspace objc2 stack; `dispatch2` not referenced.

## Commit

```sh
git add shisu/Cargo.toml shisu/Cargo.lock shisu/crates/shisu-metal
git commit -m "feat(metal): Metal backend with runtime-compiled generic kernel set"
```
