# Task 16: shisu-cuda backend

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 16 (261–267) +
§ CuMetal dev lane (39–74, hard-limits table 57–70) + crate map/SSOT (76–126; `Backend` trait at 119) +
Roadmap Wave 3a/3b (309–317). Read all four before executing.
**Depends on:**
- T2 (`task-02-core.md` 16–37): the 12-method `Backend` trait verbatim — `run(&self, KernelRef, &LaunchArgs)`,
  `graph_capture/graph_replay/graph_release`, `GraphId`, `DeviceBuffer{id,bytes}`, `DeviceView{id,offset,len}`,
  `KernelArg{U32,I32,U64,F32,Ptr,Bytes}`, `Extent3`, `Model`; errors are `CoreError::Backend{backend:"cuda",…}` /
  `Unsupported` / `BufferBounds` — no new error type escapes the crate.
- T4 (`task-04-metal-backend.md` 24–26): the sibling `Backend` impl — same vocabulary: `KernelRef.name` may carry
  the `#<idx>:<t><val>` suffix (T4 parses it as a Metal function constant; CUDA resolves-or-strips it, §launch.rs);
  `grid` = threadgroup-count passthrough, no ceil/division in the backend (T4:25).
- T14 (`task-14-kernel-build-crate.md` 16–21): `shisu_kernels::ptx(module) -> &'static [u8]` (unknown ⇒ `&[]` —
  fail-closed here), `ptx_modules()`, `KERNEL_SET_HASH`. Registry keys = 17 modules: `exl3`, the 7 mmq stems
  (`ds4_ggml_stubs ds4_mmq ds4_mmq_d2r quantize mmid mmvq ds4_repack`), `q4e_hc … q4e_misc`.
- T15 (`task-15-q4e-extraction.md` 18–27, 54–55): `kernels/cuda/q4e/KERNEL_NAMES.md` = entry-name lookup SSOT
  (plain + mangled template names, appendix A = 50 entries) and the grid/block/smem formula rows; the 6
  `cudaFuncSetAttribute` sites (1903, 1953, 2934, 3249, 3338, 3372) whose opt-in T16 replays generically.
- T1: workspace root (`shisu/Cargo.toml`, `[workspace.lints]`); `shisu-cuda` is a CUDA-phase crate (master 132) —
  **this task creates it**.

**Produces** (public API T17 consumes; paths `shisu/crates/shisu-cuda/`):

```rust
// src/lib.rs — the whole crate compiles on macOS; every driver call is a typed error there (§sys).
pub struct CudaBackend { /* primary ctx, streams, module registry, buffer table, blas, decode-graph cache */ }
impl CudaBackend {
    pub fn new(ordinal: u32, decode_graphs: bool) -> Result<CudaBackend>; // ctx+streams+load all 17 PTX modules
    pub fn set_tf32(&self, on: bool) -> Result<()>;                       // ds4 math-mode switch (ds4_cuda.cu:2973-2977)
    // classic cuBLAS tier (the ds4-parity GEMMs; stream-bound, capture-safe):
    pub fn gemm_ex(&self, ta: Op, tb: Op, m: u32, n: u32, k: u32,
                   a: DeviceView, lda: i64, b: DeviceView, ldb: i64,
                   c: DeviceView, ldc: i64, dt: GemmDType) -> Result<()>;
    pub fn gemm_strided_batched_ex(&self, /* same + stride_a, stride_b, stride_c, batch */) -> Result<()>;
    pub fn lt_gemm_act_weight_t(&self, /* Lt tier, opt-in, never default — DEVIATION 1 */) -> Result<()>;
    // qwen4exp decode-island graph cache (mirrors ds4_gpu.h:3149-3157 tri-state contract):
    pub fn decode_graphs_supported(&self) -> bool;
    pub fn decode_graph_begin(&self, key: &DecodeGraphKey) -> Result<DecodeGraphStep>; // Replay|Capture|Eager
    pub fn decode_graph_end(&self, key: &DecodeGraphKey) -> Result<()>;
    pub fn decode_graph_abort(&self, key: &DecodeGraphKey);
    pub fn decode_graphs_invalidate(&self);
    // exl3 cooperative-launch support (ds4_exl3.cu:72-157 host rules; geometry stays T17):
    pub fn max_co_resident_blocks(&self, kernel: KernelRef, block: Extent3, smem: u32) -> Result<u32>;
    pub fn reserve_stream_scratch(&self, bytes: u64) -> Result<()>;
}
pub enum DecodeGraphStep { Replay, Capture, Eager }
pub struct DecodeGraphKey { pub il: u32, pub island: u32, pub variant: u32,
                            pub cur_hc: DeviceView, pub after_attn_hc: DeviceView,
                            pub after_ffn_hc: DeviceView, pub attn_norm: DeviceView }
impl shisu_core::Backend for CudaBackend { /* all 12 T2 methods, verbatim signatures */ }
```
- `name() -> "cuda"`, `supports_graphs() -> true`.
- `KernelRef.module` IS the registry key here (T4:24 says it is metadata on Metal — one flat library; this is the
  one deliberate asymmetry between the two impls). `run` resolves `ptxmod::entry(module, name)`; the `#idx:t<val>`
  suffix never reaches the driver (§launch.rs).
- grid/block/smem are verbatim passthrough to `cuLaunchKernel` (T4:25 discipline); the formulas live in
  KERNEL_NAMES.md rows and T17 computes them (T15:54).
- No weight-binding API: weights reach the device via `alloc`+`copy_h2d`; ds4's direct host-mapping path
  (ds4_gpu.h:122–129) is NOT ported — risk note for T17, same style as T4's q5_k note.

## Global Constraints

1–7 from master plan (30–37). Task-specific:
- Mac stays `SHISU_SKIP_BUILD=1` (master 70 NVRTC row + 72 contract): no driver on this box; every
  Linux-only claim is `[INFERENCE]` and lands via Wave 3b.
- CuMetal rows that bind design: cuBLASLt (62) — a Mac "works without cublasLt" result proves nothing,
  blas.rs is Linux-validated; CUDA graphs (63) — capture/replay proven only for bounded topologies,
  clone/update/child/multi-stream incomplete ⇒ graph.rs is authored here, validated on Linux;
  cooperative grids (61) — not a lane target, exl3 is Linux-only; device properties (69) — never branch
  kernel/algorithm selection on them.
- cudarc 0.19 is a Linux-only dependency (master 264); the measured CuMetal incompatibility (264: 482 versioned
  `_v2` symbols vs 115 unversioned exports) is why no Mac driver-API claim is made anywhere in this plan.
- `warnings = deny`, clippy deny, `thiserror` internal / `CoreError` at the boundary, `parking_lot`, `tracing`,
  1500-LoC cap, no license headers, no formatters/suites (T20 owns the gate).

## Source References (verified)

atlas root = `/Users/sercand/Developer/src/github.com/sercand/atlas`; ds4 root = this repo.

| Source | Lines | What / how used |
|---|---|---|
| atlas `crates/spark-runtime/build.rs` | 27–50, 52–57, 59–82 | `rerun-if-env-changed` set (32–35); the SKIP path still emits link hints then returns (36–47: `-lcuda -lcublasLt -lcudart` + search dirs); the early return (52–57: no `cuda` feature ⇒ `return` before any link line — why an Apple-Silicon metal build never requests `-lcuda`); real-path hints (59–82) — shape copied into `shisu-cuda/build.rs`. |
| atlas `crates/spark-runtime/src/lib.rs` | 8–14 | cfg-gated `cublaslt` module + `#[path]` stub swap — the module-swap precedent for `sys/`. |
| atlas `crates/atlas-core/src/registry.rs` | 196–252 | dual module load: cudarc safe load + raw `cuModuleLoadData`; PTX must be NUL-terminated, cubin sniffed by ELF magic (207–212) — `ptxmod` policy. |
| ″ | 254–315 | function-handle cache (`OnceLock`/map, `CString` names, error text via `cuGetErrorName/String`) — `ptxmod::entry` model. |
| ″ | 424–493 | `launch_on_stream`: `cuLaunchKernel` with a runtime-built `[*mut c_void]` argv array ("avoids cudarc layout issues", 228); `>48 KiB` ⇒ `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES=8)` replayed once per handle (446–463) — the smem policy §ptxmod copies. |
| ″ | 62–118 | `cuda_error_text` (name+string), teardown no-op tolerance (codes 4/201/709), `RawCudaFunc` Send/Sync rationale. |
| atlas `crates/atlas-core/src/cuda_host.rs` | 18–93 | process-scoped primary context + stream, `OnceLock` singleton, ordinal fixed at first call, rebind ⇒ error — `device.rs` init model. |
| atlas `crates/spark-runtime/src/cuda_backend/gpu_impl_graph.rs` | 28–94 | begin/end/instantiate/launch/destroy sequence, abort-without-probe, RELAXED-mode note — `graph.rs` sequence (mode differs, DEVIATION 5). |
| atlas `crates/spark-runtime/src/cublaslt.rs` | 19–155, 202–321 | Lt FFI decls (opaque handles, attr constants incl. `PREF_MAX_WORKSPACE_BYTES`), `Ctx{handle, 64 MiB ws}` `OnceLock`, prewarm, then the full desc/layout/preference/heuristic/matmul sequence in `bf16_gemm_act_weight_t` — the Lt tier source (ds4 has none, DEVIATION 1). |
| atlas `crates/atlas-kernels/src/lib.rs` | 32–45 | `ptx()` accessor + `KERNEL_SET_HASH` + staleness-hole closure (T14 copied it; T16 consumes it). |
| ds4 `ds4_gpu.h` | 44–61, 111 | the flat host API `Backend` mirrors: init/cleanup/alloc/managed/view/free/write/read/copy + `ds4_gpu_synchronize` → `sync`. |
| ds4 `ds4_gpu.h` | 3132–3158 | decode-graph key struct + `begin/end/abort/invalidate` tri-state contract (replayed=1/captured=0/eager=-1) — `DecodeGraphCache` semantics. |
| ds4 `ds4_gpu.h` | 3161–3178 | q4e op entry points = the launch surface T17 replaces with `Backend::run` (T16 supplies the mechanism, not these ops). |
| ds4 `ds4_cuda.cu` | 954–1050 | decode-graph design: warm/capture/replay/dead states, byte-identical replay claim, 48-byte key + `static_assert`, 64×2×32 table, invalidation (1035–1050). |
| ds4 `ds4_cuda.cu` | 1052–1186 | find/slot alloc + table-full warn-once; pre-capture obligations (cublas stream bind, exl3 `prepare_stream`, 1117–1125); `BeginCapture(Global)` (1126–1127); end = EndCapture→Instantiate→Destroy→first-launch; replay on the exec's stream. |
| ds4 `ds4_cuda.cu` | 1320–1334 | abort path (capture stream left clean). |
| ds4 `ds4_cuda.cu` | 2927–2983 | init: setDevice, work stream, boundary event, `cublasCreate` + TF32-vs-default math mode (2973–2977). |
| ds4 `ds4_cuda.cu` | 15095–15098, 16331–16334, 27306–27309 | `cublasGemmEx` / `cublasGemmStridedBatchedEx` call shapes (OP_T/OP_N, f32 compute) — the blas tier's argv. |
| ds4 `ds4_cuda.cu` | 9422–9460, 22762–22805 | dsv4 hand-built static graphs (`cudaGraphAddKernelNode` + `ExecKernelNodeSetParams`) — evidence the exec-update API is dsv4-only, not ported (DEVIATION 6). |
| ds4 `ds4_qwen4exp_gpu.cuh` | 1876–1968 | chunk smem formula + one-shot `static bool opted` attribute gate; recurrent 64 KiB opt-in; launcher `<<<grid,block,smem>>>` — formulas move to KERNEL_NAMES.md (T15), the attribute pattern becomes the generic rule (DEVIATION 3). |
| ds4 `ds4_qwen4exp_gpu.cuh` | 2925–2934, 3240–3260, 3330–3345, 3365–3380 | the other 4 opt-in sites (idx_score, gather, tiled, split) — same one-shot pattern. |
| ds4 `cuda/exl3/ds4_exl3.cu` | 60–157 | device init: SM-count attr, per-kernel 90 KB opt-in + occupancy→`g_max_blocks` ("capture would reject the attribute call", 78); stream ctx: lock buffer + scratch growth, never allocate during capture, grown-from scratch never freed (136–138); `prepare_stream`. |
| ds4 `cuda/exl3/ds4_exl3.cu` | 197–230, 234–283, 375–387 | the 3 `cudaLaunchCooperativeKernel` sites (gemm 228, mgemm 280, moe 383) + `void *args[]` packing + geometry heuristics (mechanism → T16, heuristics → T17, DEVIATION 4). |
| ds4 `Makefile` | 111 | `CUDA_LDLIBS = -lcudart -lcublas` — no cublasLt (DEVIATION 1 evidence). |
| task plans | T2 16–37; T4 24–26; T14 16–21; T15 18–27, 54–55, 310–340 | trait surface; suffix grammar + grid semantics; `ptx()` contract + module names; KERNEL_NAMES.md + formulas + opt-in sites + entry inventory. |

⚠ **DEVIATION 1 (ds4 has no cublasLt):** `grep -c cublasLt` over `ds4_cuda.cu`/`ds4.c`/`ds4_gpu.h` = 0; the
Makefile links classic `-lcublas` (111). ds4's GEMM tier is `cublasCreate` + TF32 math mode + `GemmEx`/
`GemmStridedBatchedEx`. `blas.rs` ports THAT as the default (T18 golden parity needs ds4's reduction order —
mmq itself notes order drift vs the cublas+dequant pipeline, ds4_cuda.cu:1199–1201). The Lt tier
(handle/desc/heuristic/workspace) is copied from atlas `cublaslt.rs`, exposed as `lt_gemm_act_weight_t`,
opt-in, never default; its algo-heuristic cache is a shisu addition (atlas re-heuristics per call).

⚠ **DEVIATION 2 (ds4 never uses the driver API):** no `cuModuleLoadData`/`cuLaunchKernel`/`cuGraph*` exists in
ds4 — every kernel is a static runtime-API `<<<>>>` launch inside its TU. "Port host-side from ds4_cuda.cu"
holds only for the decode-graph bookkeeping + capture-stream rules; the module registry, argv packing, and
`cuGraph*` sequence follow atlas `registry.rs`/`gpu_impl_graph.rs` instead.

⚠ **DEVIATION 3 (smem opt-in becomes a generic rule):** ds4 hardcodes 6 one-shot `static bool opted` sites with
per-kernel formulas. T16 replaces them with one lazy backend rule: first `run` with `smem_bytes > 48 KiB` ⇒
`cuFuncSetAttribute(…, MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)` once per (module,entry), remembered in a set
(atlas registry.rs:446–463). Sizes come from T17's KERNEL_NAMES.md-driven `LaunchArgs.smem_bytes`, never a
backend table. The attribute MUST precede capture (ds4_exl3.cu:78) — guaranteed because the warm (eager) pass
always runs before the capture pass (ds4_cuda.cu:1119–1120).

⚠ **DEVIATION 4 (exl3 host launcher split):** `ds4_exl3.cu` is host C++ that rule 1 does not port. T16 ships the
mechanism only: coop dispatch (`cuLaunchCooperativeKernel`), occupancy query (`max_co_resident_blocks` = SM count
× `cuOccupancyMaxActiveBlocksPerMultiprocessor`, ds4_exl3.cu:72–106), and the stream-scratch rules
(`reserve_stream_scratch`: never allocate while capturing; grown-from scratch is never freed because captured
graphs still name the old pointer, 136–138). The shape/block-count heuristics (shape selection 159–196,
launch geometry 197–387) are T17's host rework.

⚠ **DEVIATION 5 (capture mode):** ds4 uses `cudaStreamCaptureModeGlobal` (1126–1127); atlas uses RELAXED
(gpu_impl_graph.rs:29–32, for NCCL). shisu has no NCCL ⇒ Global (fail-fast on illegal cross-stream work), single
capture stream.

⚠ **DEVIATION 6 (no graph exec-update):** ds4's `cudaGraphExecKernelNodeSetParams` users are the two dsv4 static
graphs (9451, 22792) — dsv4 is out of scope (master Scope). qwen4exp bakes pointers into the key instead, so
`GraphId` = opaque `CUgraphExec` handle; no node-update API. CuMetal row 63 (clone/update incomplete) makes this
the safe shape regardless.

## Plan

- [ ] **Step 0 — Preflight:** read T2/T4/T14/T15 Produces blocks; confirm `shisu/kernels/KERNEL.toml` exists
  (T14) and `KERNEL_NAMES.md` exists or is pending (T15 Step 16 fills the mangled column on Linux; plain names
  work before it).

- [ ] **Step 1 — Crate + workspace wiring:** `shisu/crates/shisu-cuda/Cargo.toml`: deps `shisu-core`,
  `shisu-kernels`, `thiserror`, `parking_lot`, `tracing`; `[target.'cfg(target_os = "linux")'.dependencies]
  cudarc = { workspace = true, features = ["driver"] }` (target-specific ⇒ cudarc is not even resolved on macOS;
  satisfies master 264's cfg rule). Edit `shisu/Cargo.toml`: members += `crates/shisu-cuda`,
  `[workspace.dependencies] shisu-cuda = { path = "crates/shisu-cuda" }` — additive hunks only (T14 Step 4
  discipline). Engine does NOT depend on this crate yet (T17 wires it, Linux-target-only).

- [ ] **Step 2 — `build.rs` (≤60 ln):** copy atlas spark-runtime/build.rs shape: `rerun-if-changed=build.rs` +
  `rerun-if-env-changed` for `SHISU_SKIP_BUILD`/`CUDA_HOME`; **macOS early return first** (atlas 52–57 pattern —
  `CARGO_CFG_TARGET_OS != "linux"` ⇒ `return` emitting nothing); Linux: `cargo:rustc-link-search`
  `$CUDA_HOME/lib64`, `/usr/local/cuda/lib64`, `/usr/lib/{x86_64,aarch64}-linux-gnu`, then
  `dylib=cublasLt` + `dylib=cublas`. NO `-lcuda`: atlas's own comment (build.rs:37–38) records that dlopen-mode
  cudarc does not emit it — atlas must for its raw `cu*` externs; shisu's raw externs are cublas/cublasLt only
  and libcuda is reached through cudarc's dlopen. NO `-lcudart` (no runtime-API symbols; d2d is
  `cuMemcpyDtoDAsync`).

- [ ] **Step 3 — `src/sys/mod.rs` (≤160 ln) — the cfg mechanism = module swap, not a trait:**
  `#[cfg(target_os = "linux")] #[path = "nvidia.rs"] mod imp;` / `#[cfg(not(target_os = "linux"))] mod stub;`
  (stub fns live inline in mod.rs), re-exported as `pub(crate) mod nvidia { pub use imp::*; }`. Both arms expose
  identical signatures; call sites in device/ptxmod/launch/graph/blas read `sys::nvidia::…` and are
  OS-agnostic. Rejected: a trait (vtable on the launch hot path + boxing) and const generics (cannot swap fn
  bodies per-OS without monomorphizing the whole orchestration). Module swap keeps atlas's own
  cublaslt/stub precedent (lib.rs:8–14) and master 264's "compiles against `sys::nvidia` stubs" wording.
  Error currency: `pub enum SysError { Cu(CuResult-ish code + name/string text), Host(&'static str) }`
  (`thiserror`); `cuGetErrorName/String` formatting copied from atlas registry.rs:65–88;
  `impl From<SysError> for CoreError` ⇒ `CoreError::Backend{backend:"cuda", message}` — nothing else escapes.
  macOS stub: every fn returns `Err(SysError::Host("cuda: no driver on this target"))`; the stub IS the
  type-check harness (master 264).

- [ ] **Step 4 — `src/sys/nvidia.rs` (≤480 ln, Linux-only):** thin `pub(crate) unsafe fn` wrappers over
  `cudarc::driver::sys` raw bindings (cudarc ships the full generated `cu*` surface; its safe
  launch-builder cannot take a runtime-built argv array — atlas's "cudarc layout issues", registry.rs:228).
  Surface: ctx (`cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`), mem
  (`cuMemAlloc_v2/cuMemFree_v2`, `cuMemcpy{HtoD,DtoH,DtoD}Async_v2`), module
  (`cuModuleLoadData/cuModuleUnload/cuModuleGetFunction`), func (`cuFuncSetAttribute`), launch
  (`cuLaunchKernel`, `cuLaunchCooperativeKernel`), occupancy (`cuOccupancyMaxActiveBlocksPerMultiprocessor`,
  `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` — diagnostics/coop sizing only, never kernel selection,
  row 69), stream (`cuStreamCreate/Destroy/Synchronize`, `cuStreamBeginCapture/EndCapture`), graph
  (`cuStreamBeginCapture_v2`-form, `cuGraphInstantiateWithFlags`, `cuGraphLaunch`, `cuGraphDestroy`,
  `cuGraphExecDestroy`), plus `cuGetErrorName/String`. Raw `extern "C"` only for cublas/cublasLt (atlas
  gpu_impl.rs:24–25 precedent): `cublasCreate_v2`, `cublasSetStream_v2`, `cublasSetMathMode`,
  `cublasGemmEx`, `cublasGemmStridedBatchedEx`, and the Lt set copied from atlas cublaslt.rs:19–120.

- [ ] **Step 5 — `src/device.rs` (≤320 ln):** `DeviceCtx::new(ordinal)` = primary-ctx retain + set-current,
  process-scoped singleton semantics copied from atlas cuda_host.rs:18–93 (ordinal fixed at first call; second
  call with a different ordinal ⇒ `Unsupported`); work stream = legacy default (stream 0 — eager decode parity
  with ds4's legacy-stream choice, atlas registry.rs:440–445 comment) + one dedicated capture stream. Buffer
  table `Mutex<HashMap<u64, (CUdeviceptr, u64)>>`, id 0 = null (T2); `alloc/free/copy_h2d/copy_d2h/copy_d2d`
  mirror ds4_gpu.h:44–61 (free of unknown id ⇒ `BufferBounds`). `max_co_resident_blocks` +
  `reserve_stream_scratch` per DEVIATION 4 (lock buffer sized like ds4_exl3.cu:124:
  `(MAX_TILES_C + 2*MAX_BARRIERS + MOE_SCHED_INTS)` ints, 16 MiB scratch floor, growth never frees).

- [ ] **Step 6 — `src/ptxmod.rs` (≤260 ln):** load loop over `shisu_kernels::ptx_modules()`:
  empty blob ⇒ fail-closed `Backend` error naming the module + `KERNEL_SET_HASH` (T14's `&[]` contract means
  SKIP_BUILD/unknown never reaches the driver); PTX passed as NUL-terminated C string, cubin sniffed by ELF
  magic (atlas registry.rs:207–212). Handle cache `Mutex<HashMap<(&str,&str), CUfunction>>` keyed
  (module, entry); `entry()` errors list the module's known entries on miss. `Drop` unloads modules, tolerating
  no-op codes 4/201/709 (atlas registry.rs:90–106). smem opt-in replay per DEVIATION 3 (set of handles already
  opted; `Mutex` guard checked in `launch`, not a syscall when unset).

- [ ] **Step 7 — `src/launch.rs` (≤280 ln):** `pack_args(&[KernelArg]) -> Argv` — one slot per arg in order:
  `Ptr` ⇒ u64 slot = buffer base + `DeviceView.offset` (bounds-checked ⇒ `BufferBounds`); `U32/I32/F32` ⇒ 4-byte
  slot; `U64` ⇒ 8-byte slot; `Bytes` ⇒ raw struct slot (exl3/mmq pass structs by value — the driver copies the
  declared size from the kernel signature, so slot size = `KernelArg::Bytes` len; each slot is its own cell, so
  alignment is per-slot and always ≥ 8). Returns `Vec<*mut c_void>` for `cuLaunchKernel.kernelParams`.
  Entry resolution: split `KernelRef.name` at `'#'`; no suffix ⇒ base name is the entry (plain `extern "C"`
  kernels; a stray Metal FC suffix is a no-op on CUDA). Suffix present ⇒ template family: parse `t<val>`
  (same grammar as T4's pipeline.rs parser; `s`/`b` letters accepted, value = the template parameter, e.g.
  NT), look `(base, value)` up in a `KERNEL_ENTRIES` static **generated by build.rs from the KERNEL_NAMES.md
  table** (include! + hash, T14 discipline; mangled column pending the Linux run ⇒ unresolved rows fail-closed
  with a named error, `[INFERENCE]`: mangled rows land in Wave 3b). Unknown value ⇒ error listing instantiated
  rows (T15 instantiates only the launcher's sets). Dispatch: `module == "exl3"` ⇒
  `cuLaunchCooperativeKernel` (the only coop TU — T14:64–65: exl3 coop, mmq production TUs non-coop; q4e plain), else
  `cuLaunchKernel`; grid/block/smem verbatim; smem-opt-in check first.

- [ ] **Step 8 — `src/blas.rs` (≤320 ln):** classic tier (default): handle created in `CudaBackend::new`
  (ds4_cuda.cu:2971), `set_tf32` ⇒ `CUBLAS_TF32_TENSOR_OP_MATH` vs `CUBLAS_DEFAULT_MATH` (2973–2977; ds4's
  default is TF32 — engine passes its quality-mode choice; no env knob, rule 5); `bind_stream` called by
  graph.rs before capture/abort/end (1121/1142/1328 — cublas must ride the capture stream or capture fails);
  `gemm_ex`/`gemm_strided_batched_ex` signatures = ds4 call shapes (15095, 16331, 27306). Lt tier (opt-in,
  DEVIATION 1): `Ctx{handle, 64 MiB workspace}` `OnceLock` + prewarm + desc/layout/preference/heuristic copied
  from atlas cublaslt.rs:19–155/202–321, plus a bounded `HashMap<(ta,tb,dtypes,m,n,k), algo>` heuristic cache
  (shisu addition; atlas re-heuristics per call). CuMetal row 62: nothing here is validated on the Mac.

- [ ] **Step 9 — `src/graph.rs` (≤380 ln):** (a) `Backend::graph_capture(body)`: bind blas stream →
  `cuStreamBeginCapture(Global, DEVIATION 5)` on the capture stream → `body()` (nested `run`s route to the
  capture stream via a `Mutex<Option<CUstream>>` current-capture cell) → EndCapture →
  `cuGraphInstantiateWithFlags(0)` → `cuGraphDestroy` (exec keeps it) ⇒ `GraphId`; `graph_replay` =
  `cuGraphLaunch(exec, work_stream)`; `graph_release` = `cuGraphExecDestroy`. (b) `DecodeGraphCache` — the
  qwen4exp layer-graph bookkeeping ported from ds4_cuda.cu:954–1334: table `64 × 2 × 32` (il × island ×
  variant; variant = n_tok ∈ {1..16, 64}, ds4 964), entry = `{key: DecodeGraphKey, exec, state:
  Empty|Warmed|Ready|Dead}`; `begin`: Ready+key-equal ⇒ Replay; Empty ⇒ mark Warmed + Eager (warm pass makes
  allocator deterministic and triggers any smem opt-in); Warmed ⇒ Capture; Dead ⇒ Eager; table full ⇒ Eager +
  warn-once (ds4 1068–1080). Key equality = `memcmp` over the whole 48-byte key (ds4 1059–1060) — the 4
  DeviceViews bake the buffers in, so a reallocated buffer changes the key ⇒ fresh entry. `end`: instantiate +
  first launch + Ready; any driver error ⇒ mark Dead, abort capture, Eager fallback forever (ds4 1144–1175,
  1320–1334).
  `decode_graphs_invalidate` destroys all execs (engine calls it on buffer reallocation / `Model::teardown`,
  ds4 1035–1050). `decode_graphs: bool` in `new` replaces ds4's `DS4_CUDA_DECODE_GRAPHS` env (rule 5).

- [ ] **Step 10 — `src/lib.rs` (≤120 ln):** `CudaBackend` struct + `new` (device → blas → ptxmod load-all →
  cache) + `impl Backend` (12 methods: `alloc/free/copy_*/sync` → device; `run` → launch+ptxmod;
  `graph_*` → graph; `name/supports_graphs` constants; `graph_capture` on a non-Linux build ⇒ `Unsupported`
  via stubs). Re-export `DecodeGraphKey/Step`, `GemmDType`, `Op`.

- [ ] **Step 11 — macOS verify:** `SHISU_SKIP_BUILD=1 cargo check -p shisu-cuda` (stub path compiles; no
  cudarc in the resolved graph — `cargo tree -p shisu-cuda | grep cudarc` empty on macOS);
  `cargo test -p shisu-cuda` (all macOS tests below run, GPU tests skipped); `grep -r ATLAS_ shisu/crates/shisu-cuda`
  empty.

- [ ] **Step 12 — Linux gate `[INFERENCE]` (not executable on this Mac — no CUDA driver, master 70/264):**
  `SHISU_SKIP_BUILD=1` (T14's gate) already stubs the compile lane on driverless hosts; run the
  `#[ignore]` tests in the Wave 3b Linux runbook. Nothing in this task's merge notes may claim a Linux result.

## Tests

macOS, no driver (unit tests run against the pure logic + the sys stubs; the fake `Backend` from T2's
`traits_contract.rs` pattern is the seam that proves the trait-level contract — the same `LaunchArgs` a fake
records is the `LaunchArgs` `pack_args` consumes, so the two sides cannot drift; master 266: these are "the
parts that can rot silently"):
- `tests/pack_args.rs` — argv packing invariants: one slot per arg in order; `Ptr` slot = base+offset
  (DeviceView arithmetic, out-of-range ⇒ `BufferBounds`); scalar slot widths 4/4/8/4; `Bytes` slot is the raw
  struct bytes verbatim; empty args ⇒ empty argv; slots 8-aligned.
- `tests/name_resolve.rs` — `#idx:t<val>`: no suffix ⇒ identity; `#0:s4` on a template family ⇒ the table's
  mangled row for value 4; unknown value ⇒ error listing instantiated values; malformed suffix (`#x`, `#0:q1`)
  ⇒ error; same case table as T4's parser tests (shared grammar, two parsers — pinned by identical cases).
- `tests/registry.rs` — module registry: empty PTX blob ⇒ fail-closed error naming the module (no driver
  call); the 17 expected module names; handle-cache second lookup hits the cache (stub counts calls); error
  text carries `KERNEL_SET_HASH`.
- `tests/decode_graph.rs` — state machine: first sighting ⇒ Eager+Warmed; second ⇒ Capture; capture error ⇒
  Dead then Eager forever; table full ⇒ Eager + exactly one warn; key inequality on any of the 4 DeviceViews ⇒
  distinct entries; `invalidate` clears Ready entries; nested begin ⇒ Eager (no recursive capture).
- `tests/sys_stub.rs` — every `sys::nvidia` entry point on macOS returns
  `Err(…)` mapping to `CoreError::Backend{backend:"cuda"}` — the type-checked stub surface (master 264).
Linux, `#[ignore]` + `#[cfg(target_os = "linux")]` `[INFERENCE]` (Wave 3b box):
- load-all: every `ptx()` blob non-empty and `cuModuleLoadData` OK; `cuModuleGetFunction` succeeds for every
  KERNEL_NAMES.md entry (mangled rows included once T15 Step 16 lands).
- trivial launch: `q4e_add2_kernel` on a small buffer, d2h compare.
- smem opt-in: recurrent shape (64 KiB) — attribute set once, second launch skips the syscall, launch succeeds.
- graph round-trip: capture two launches → replay → output equals eager; `invalidate` then re-capture works.
- coop: `max_co_resident_blocks` ≥ 1 on an exl3 entry; `reserve_stream_scratch` grows without freeing the
  prior buffer.

## Acceptance

- [ ] `shisu/crates/shisu-cuda/src/{lib,device,ptxmod,blas,graph,launch}.rs` + `src/sys/{mod,nvidia}.rs` +
  `build.rs` exist; every file < 1500 LoC (local gate `scripts/file_size_check.sh`).
- [ ] `SHISU_SKIP_BUILD=1 cargo check -p shisu-cuda` green on macOS via the stub path; cudarc absent from the
  macOS resolve graph (target-specific dep); `cargo test -p shisu-cuda` green on macOS.
- [ ] `impl Backend` matches T2's trait verbatim; `run`/`KernelRef`/`LaunchArgs` vocabulary identical to T4
  (module-as-key asymmetry documented in Produces).
- [ ] `graph_capture` is the only capture path; decode-graph cache mirrors ds4's tri-state + 48-byte-key
  semantics; no `cudaGraphExecKernelNodeSetParams` equivalent (DEVIATION 6).
- [ ] blas default = classic cuBLAS (ds4 parity); Lt tier opt-in; no blas behavior claimed from the Mac
  (CuMetal row 62).
- [ ] Linux-only claims appear solely as `[INFERENCE]` + the Wave 3b `#[ignore]` tests; nothing asserts a
  Linux result that was not executed.
- [ ] No `ATLAS_` string under `shisu/`; no new env knobs (decode-graphs and TF32 are API switches, rule 5);
  `shisu/Cargo.toml` change additive-only; no formatters/suites run (T20 owns the gate).

## Commit

```sh
git add shisu/Cargo.toml shisu/crates/shisu-cuda
git commit -m "feat(cuda): shisu-cuda backend - driver-API module registry, argv launches, cuBLAS tier, decode-graph cache (Task 16)"
```
