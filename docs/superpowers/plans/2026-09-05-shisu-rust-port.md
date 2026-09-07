# shisu — ds4 → Rust Port Implementation Plan (v2: Metal-first)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Max 2 concurrent subagents; serialize same-crate edits through one owner.

**Goal:** Port the qwen3.8-flash-next (qwen4exp) inference path of ds4 to a lean Rust workspace named `shisu`, with qwen3.5-4b running first-class on macOS Metal (dev machine: M1 Max), OpenAI/Anthropic APIs over a shared IR, and fast disk IO for KV checkpoints and (CUDA phase) PLE n-gram streaming.

**Architecture:** CUDA kernels are copied verbatim from ds4 and compiled by a build crate to PTX, loaded at runtime via the CUDA driver API (atlas `atlas-kernels` pattern — zero Rust overhead in the hot path). Metal kernels are copied/adapted MSL compiled at runtime (`newLibraryWithSource`, ds4_metal.m pattern). Rust replaces only host orchestration: GGUF loading, graph capture, scheduling, sampling dispatch, HTTP. The API layer is atlas spark-server's "narrow waist": each wire surface (OpenAI chat/responses/completions, Anthropic messages) converts to one neutral IR; internals never read wire types.

**Build order is Metal-first**: the dev machine is macOS, so the full stack (backend → engine → server → APIs → kvstore) is built and proven on qwen3.5-4b/Metal. The CUDA/qwen4exp phase's *shipping* target stays Linux+NVIDIA. What changed: this Mac now has **CuMetal** (`cumetalc`) installed, so the CUDA phase no longer waits idle for the Linux box — kernel TUs get compiled and numerically oracle'd here (see [CuMetal dev lane](#cumetal-dev-lane-cumetalc--verified-2026-09-05)), while driver-API/cublasLt/graph/perf work still executes only on Linux.

**Tech Stack:** Rust edition 2024 (pin 1.93.1), cudarc 0.19 (driver+nvrtc only, CUDA phase, **Linux-only** — see CuMetal limits), axum 0.8 + hyper 1 + tokio, serde/serde_json(preserve_order), thiserror+anyhow, clap 4, tracing, objc2 (Metal), io-uring (Linux-only, PLE), safetensors, minijinja + tokenizers, nvcc (PTX, CUDA phase), CuMetal 0.4.0 `cumetalc` (macOS dev lane: CUDA→Metal compile + numerical oracle; never a runtime dependency).

**Sources:**
- ds4 (C): `/Users/sercand/Developer/src/github.com/sercand/ds4-qwen3.8-flash` — kernels + algorithms, copied.
- atlas (Rust): `/Users/sercand/Developer/src/github.com/sercand/atlas` — build scripts + server/IR patterns, copied/adapted.
- CuMetal: https://github.com/Lulzx/cuda-metal (Apache-2.0) — installed here via Homebrew: `cumetal 0.4.0`, prefix `/opt/homebrew/opt/cumetal`, `cumetalc`/`cumetal`/`cumetal-ptx2llvm` on PATH, CUDA-capable clang `/opt/homebrew/opt/llvm/bin/clang++`, `cumetal doctor` clean.
- Models (local paths — no downloads): GGUF `/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf` (primary Metal test model + CUDA-lane smoke model on this Mac; `SHISU_TEST_MODEL` default), MLX/OptiQ dir `/Users/sercand/models/Qwen3.5-4B/mlx` (secondary format = mlx-community/Qwen3.5-4B-OptiQ-4bit; `SHISU_TEST_MODEL_OPTIQ` default), qwen4exp GGUF (CUDA phase, Linux).

---

## Scope (locked decisions)

**In:**
- **qwen3.5-4b on macOS Metal** — dense hybrid: 32 layers, 8×(3×Gated DeltaNet→FFN→1×Gated Attention→FFN), hidden 2560, GQA 16Q/4KV @ head_dim 128, GDN 32V/16QK @ head_dim 128, vocab 248,320. NOT MoE. Load `unsloth/Qwen3.5-4B-GGUF` (GGUF q4_k/q8_0 — ds4 Metal already has q8_0/q4_k dequant kernels) as primary; `mlx-community/Qwen3.5-4B-OptiQ-4bit` (safetensors + mlx group-quant 4/8-bit mixed) as secondary loader task.
- **qwen4exp (qwen3.8-flash-next) exl3/q4 on Linux CUDA** — kernels copied from ds4; PLE n-gram SSD streaming; MTP spec decode. Deferred *execution* phase, same plan; kernel authoring/verification starts now on this Mac via CuMetal.
- OpenAI chat completions + responses + completions, Anthropic messages; SSE streaming; **one KV-cache system for every model on macOS**: radix tree of token spans + refcounted paged KV + GDN/SSM checkpoints + Marconi utility eviction (ds4.c 55072-56100), with the disk checkpoint store as its spill tier.

**Out (do not port):** DeepSeek-V4, GLM-5.2/5.3, qwen3.6-35b-a3b (too big for dev machine — superseded by qwen3.5-4b), ROCm, multi-GPU/TP/distributed, vision, `ds4_agent.c` CLI, `ds4_web.c`, `dir-steering`, `gguf-tools`. CPU reference ports only as tiny per-op test oracles. **Also out:** shipping CuMetal — no `libcumetal` linkage, no `.metallib` produced by `cumetalc` in any release artifact, no CuMetal requirement in any user-facing build path. **Also out:** GitHub Actions / CI workflow setup — no `.github/workflows/*` files; every gate runs as a local script or `cargo test`.

**Rules (global constraints):**
1. All new code under `shisu/`. No C/C++/ObjC host code — only `.cu`/`.cuh`/`.metal` kernel files (copied from ds4). Single exception: `tests/cumetal/*.cu` oracle harnesses (Task 3b), which are test tools compiled by `cumetalc`, never linked into the product.
2. Hard 1500-LoC-per-file cap (local gate `scripts/file_size_check.sh`; GitHub Actions is out of scope). Kernel files split by family; no 33K-line TUs.
3. Kernels copied byte-for-byte where possible; only `#include` plumbing and launch wrappers change. Arithmetic-order changes are bugs (golden vectors must pass). The CuMetal compat shim (`kernels/cumetal_compat/`) may add *declarations* (missing half/bf16 intrinsics) but never redefines an existing kernel's arithmetic.
4. No performance regression: CUDA phase — `shisu-bench` decode/prefill tok/s ≥ ds4 baseline on same GPU+model. Metal phase — record baselines on M1 Max (no ds4 comparison exists for qwen3.5-4b). **CuMetal numbers never gate anything**: translated kernels run ~1.03–1.20× hand-written Metal on memory-bound cases upstream, `clock()` is a synthetic monotonic counter (not GPU cycles), stream priorities are 0, and the cuBLASLt path can silently fall back to CPU. CuMetal is a correctness/compile tool only.
5. Env knobs prefixed `SHISU_` (`SHISU_SKIP_BUILD=1` stubs kernel builds, `SHISU_CUDA_ARCH`, `SHISU_PLE_IO=uring|pool`, `SHISU_TEST_MODEL` = path to local GGUF, default `/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf`, `SHISU_TEST_MODEL_OPTIQ` = secondary loader dir, default `/Users/sercand/models/Qwen3.5-4B/mlx`, `SHISU_CUDA_IMPL=nvidia|cumetal`, `SHISU_CUDA_COMPILE=1` opts into a real kernel compile, `SHISU_CUMETAL_STRICT=1` turns CuMetal compile warnings into task failures, `SHISU_CUMETAL_PREFIX` = cumetal install prefix, default `/opt/homebrew/opt/cumetal`).
6. Workspace lints: `warnings=deny`, clippy all deny; `thiserror` library errors, `anyhow` binary boundaries; `parking_lot` locks; `tracing` logging.
7. License headers: none (user sets license later). CuMetal-derived compat headers keep their Apache-2.0 notice.

## CuMetal dev lane (`cumetalc`) — verified 2026-09-05

[CuMetal](https://github.com/Lulzx/cuda-metal) compiles CUDA C++/PTX into Metal Shading Language, so CUDA kernels run on this M1 Max without being rewritten. It is experimental and implements a *tested subset* of CUDA. In this plan it is a **development tool**: it retires CUDA-phase risk early and gives the Metal-first phase a second numerical reference. It is never a shipping dependency and never a performance reference.

**Two jobs it does here**
1. **Differential oracle for Task 5.** The qwen35 GDN kernels exist only as CUDA (`q4e_gdn_*` in `ds4_qwen4exp_gpu.cuh`; ds4 Metal has no delta-net kernel). Compiling the real CUDA source through `cumetalc` and running it on the same GPU gives an arithmetic-order-faithful second implementation to diff the hand-written MSL against — strictly stronger than a hand-written Rust oracle for catching re-association and indexing bugs (constraint 3).
2. **Early CUDA-phase execution.** Task 14/15 (build crate, q4e TU split) and the kernel half of Task 16/17 get compile-verified and numerically checked here, so the Linux box runs code that already compiles instead of discovering mechanical-split errors there.

**Verified on this machine** (M1 Max, macOS 26.5.1, `cumetal 0.4.0`, `cumetal doctor` all-green):
- Working invocation: `cumetalc --cuda-device --mode experimental x.cu -o x --ptx-strict -I <inc>`. A probe kernel using dynamic `extern __shared__`, `__syncthreads`, `__shfl_down_sync` (full-warp mask) and `atomicAdd(float*)` compiled and returned numerically correct results on the Apple GPU.
- The plain `cumetalc x.cu -o x` path fails with `'cuda_runtime.h' file not found` unless `-I /opt/homebrew/opt/cumetal/include` is passed. Always pass includes explicitly.
- Useful flags: `--mode xcrun|experimental`, `--cuda-device`, `--cuda-arch sm_XX`, `--cuda-clang`, `--cuda-inline-threshold`, `--emit llvm|cumetal-ir|metal-ir|msl|metallib|exe`, `--backend legacy|cumetal-ir`, `--fp64=fast48|wide48|ieee64|native|emulate|warn`, `--ptx-strict`, `--save-temps`, `--no-link`, `-I`, `-D`.
- `libcumetal.dylib` exports the CUDA runtime API (`cudaMalloc`/`cudaMemcpy`/`cudaLaunchKernel`/…) and 115 `cu*` driver entry points, plus `libcublasLt.dylib`/`libcublas.dylib`/`libcudnn.dylib`/… shims. No `nvrtc*` symbols in this install.
- **cudarc cannot bind it** (measured): cudarc 0.19.3 dlopens `libcuda.dylib` first and binds 482 driver symbols under versioned names (`cuCtxCreate_v2`, `cuDeviceGetUuid_v2`, `cuMemAlloc_v2`, …); CuMetal exports unversioned names only (`cuCtxCreate`, `cuMemAlloc`, `cuMemcpyHtoD`) and no `_v2` aliases. A `libcuda.dylib`→`libcumetal.dylib` symlink dlopens and `cuModuleLoadData`/`cuModuleGetFunction` on hand-written PTX succeed, but `cuLaunchKernel` returned `CUDA_ERROR_INVALID_VALUE` and the probe then crashed. Conclusion: the driver-API route is not a supported host path — do not spend time on it.
- **Homebrew bottle header gap:** the installed include dir omits 8 entries that the `v0.4.0` tag ships under `runtime/api/`: `cuda_bf16.h`, `cuda_gl_interop.h`, `math_constants.h`, `mma.h`, `sm_35_intrinsics.h`, `vector_functions.h`, `vector_types.h`, and the `cuda/` subdir. Fetching them into a local dir and adding `-I` clears the include stage (this is Task 3b's `FETCH.sh`).
- **Intrinsic gap:** ds4's `cuda/` tree uses 16 half/bf16 helpers CuMetal's clean-room headers do not declare (`__halves2half2`, `__low2half`, `__high2half`, `__lows2half2`, `__highs2half2`, `__floats2half2_rn`, `__float2half2_rn`, `__float2half_rn`, `__float22half2_rn`, `__int2half_rn`, `__ushort_as_half`, `__half_as_ushort`, `__half_raw`, `__halves2bfloat162`, `__bfloat1622float2`, `__float2half`). All are bit-level packing/conversion helpers → one force-included shim header. This is the whole measured porting cost of the lane, and it is mechanical.
- Upstream precedent that matters: llama.cpp's **unmodified** GGML CUDA backend builds against CuMetal and decoded coherently (SmolLM2-135M, M4 Pro), with FlashAttention advertised as unsupported so llama.cpp falls back to ordinary attention. A GGUF inference path over CUDA kernels on Apple Silicon is a proven shape.

**Hard limits → what stays Linux-only** (from upstream `docs/known-gaps/*`; each row is a rule for whoever writes the task):

| Area | CuMetal limit | Consequence for shisu |
|---|---|---|
| Cooperative grids | resident grid ≤ 1 block per GPU core, oversubscription rejected | exl3 (`cuda/exl3/ds4_exl3.cu` uses cooperative groups) is **not** a CuMetal target; Linux-only |
| cuBLASLt | CPU fallback is FP32/FP64 column-major exact-shape only; FP16/TF32 Lt compute rejected | a Mac "works without cublasLt" result proves nothing; `blas.rs` is validated on Linux |
| CUDA graphs | capture/replay proven for bounded topologies; clone/update, child graphs, arbitrary multi-stream incomplete | decode-graph capture (T16) authored here, validated on Linux |
| Shared memory | beyond static arrays + one runtime-sized `extern __shared__` is open; predicated barriers fail | q4e kernels with compound shared layouts may fail to compile — that is a diagnostic, not a licence to rewrite arithmetic |
| Atomics | form-specific; threadgroup float atomics become CAS loops; float min/max/CAS refused | reduction-order regressions are plausible; every oracle must be numeric, never "it ran" |
| FP64 | emulated (`fast48` ≈ 48-bit significand, binary32 exponent range) | no FP64 parity claim from the Mac |
| Timing | `clock()` is a synthetic monotonic counter; stream priorities 0; NVML telemetry unsupported | constraint 4 — CuMetal never satisfies a perf gate |
| Memory | raw `malloc` pointers are not kernel-bindable; only tracked allocations | oracle harnesses must allocate via `cudaMalloc` |
| Device properties | reports compute capability 8.0, zero PCI ids, synthetic values | never branch kernel selection on device properties |
| NVRTC | not in this install (`nvrtcGetPTX` fails upstream anyway) | `SHISU_SKIP_BUILD=1` covers the Mac; runtime compile stays Linux |

**Contract:** `SHISU_CUDA_IMPL=nvidia|cumetal` (Linux default `nvidia`, macOS dev default `cumetal`, `SHISU_SKIP_BUILD=1` wins over both); `SHISU_CUDA_COMPILE=1` required for any real compile; `SHISU_CUMETAL_PREFIX` (default `/opt/homebrew/opt/cumetal`); `SHISU_CUMETAL_STRICT=1` for task gates. Compat assets live in `kernels/cumetal_compat/` and `tests/cumetal/`; `STATUS.md` is the single place recording which TUs compile and which construct blocks the rest.

**Model for the lane:** `SHISU_TEST_MODEL=/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf` (local file, verified on disk, 2740937888 bytes — no download). qwen4exp GGUF is not needed to exercise kernels on the Mac — real qwen3.5-4b weight slices give real shapes for the oracle.

## Crate map (workspace under `shisu/`)

```
shisu/
  Cargo.toml                    # workspace; [workspace.dependencies] copied from atlas
  rust-toolchain.toml           # 1.93.1, rustfmt+clippy
  deny.toml                     # copied from atlas
  scripts/file_size_check.sh    # 1500-LoC-per-file gate (local; GitHub Actions out of scope)
  scripts/check.sh              # local gate: cargo check/test/clippy + cargo deny + file-size cap
  crates/
    shisu-core/                 # shapes/config (HF config.json + ds4 g_shape profiles), dtypes, error,
                                # Backend + Model traits, q4e page geometry consts (from ds4_q4e_page.h)
    shisu-gguf/                 # GGUF parser, quant block formats (q8_0/q4_k/q4e/iq2/mxfp4/exl3 trellis),
                                # tensor binding. Port ds4.c §460-1281, §2231-4966
    shisu-metal/                # Metal Backend (objc2): device init, pipeline cache, runtime MSL
                                # compile (newLibraryWithSource, ds4_metal.m pattern)
      metal/
        generic/                # adapted ds4 metal/: norm, rope(dsv4_rope), softmax, unary, bin, glu,
                                # concat, cpy, get_rows, set_rows, sum_rows, repeat, argsort, flash_attn
        qwen35/                 # new MSL: gemv_q4k/gemv_q8, gemm_q4k (prefill), gdn_conv, gdn_recurrent,
                                # gdn_chunk (prefill), gated_attn, rmsnorm_rope fused
    shisu-engine/               # backend-agnostic orchestration: qwen35 forward (GDN+GatedAttn+FFN),
                                # sampling (bit-exact port ds4.c §70205+), KV cache + GDN state mgmt,
                                # sessions; qwen4exp forward + MTP added in CUDA phase
    shisu-kvstore/              # THE KV-cache system (all models, macOS-first): radix tree of token spans +
                                # refcounted page pool + GDN/SSM checkpoints + Marconi eviction (ds4.c 55072-56100)
    shisu-ir/                   # neutral IR copied from atlas spark-server/src/ir/: ChatRequest, Message,
                                # Role, ContentPart, StreamDelta, DeltaStream, FinishReason, Usage
    shisu-server/               # `shisu` binary: clap CLI, axum router, openai/ + anthropic/ adapters
                                # (atlas copies), tokenizer (minijinja chat render), scheduler thread,
                                # response_store, auth, SSE
    shisu-kernels/              # CUDA phase: build.rs nvcc→PTX (Linux) | cumetalc→metallib (macOS dev) registry over kernels/
    kernels/cuda/{q4e,exl3,mmq}/# CUDA phase: q4e split from ds4_qwen4exp_gpu.cuh + vendored exl3/mmq
    kernels/cumetal_compat/     # CuMetal dev lane (Task 3b): include/ (headers the Homebrew bottle omits),
                                # cumetal_half_shim.h (missing half/bf16 intrinsics), STATUS.md (per-TU verdict),
                                # FETCH.sh (pin the v0.4.0 tag SHA the headers came from)
    tests/cumetal/oracle_gdn.cu # CuMetal differential-oracle harness (test tool, not shipped; rule 1 exception)
    shisu-cuda/                 # CUDA phase: Backend impl — driver init, PTX load, cublasLt, graph capture (Linux-only FFI)
    shisu-ple/                  # CUDA phase: PLE ngram SSD streaming (ds4_ple_stream.c port, io-uring+pool)
    shisu-bench/                # perf gates: decode/prefill tok/s, kvstore resume latency, baselines
  test-vectors/                 # copied from ds4 tests/test-vectors (golden logits/vectors, CUDA phase)

**Cross-task contracts (SSOT):**
- `shisu_core::Backend` trait — mirrors ds4_gpu.h flat API for the needed subset: `alloc/free/copy_h2d/d2h/copy_d2d`, `launch(module, kernel, grid, block, smem, args)` (CUDA) / `dispatch(kernel, args, threadgroup)` (Metal) unified as `Backend::run(&self, KernelRef, &LaunchArgs)`; `graph_capture` (CUDA only, Metal uses retained command-encoders). CUDA + Metal impls.
- `shisu_core::Model` trait — `load`, `prefill_chunk`, `decode_batch`, `sample`, `teardown` (explicit, not Drop; atlas pattern).
- `shisu_kernels::ptx(module: &str) -> &'static [u8]` — CUDA-phase PTX registry; `SHISU_SKIP_BUILD=1` emits empty stub. `shisu_kernels::metallib(module: &str) -> &'static [u8]` is the CuMetal-lane sibling (macOS dev only, empty on every shipping target); kernel-name → entry-name mapping is shared by both so a kernel is registered once.
- CuMetal lane contract: `kernels/cumetal_compat/STATUS.md` is the SSOT for which TUs compile under `cumetalc` and which construct blocks the rest; Tasks 5/14/15/17 read it instead of re-probing. The oracle harness exposes one entry point — `oracle_gdn <kernel> <in.bin> <out.bin> <shape…>` — so Rust tests can shell out to it without knowing CUDA.
- IR names verbatim from atlas: `ChatRequest`, `SamplingParams`, `ThinkingDirective`, `Message`, `ContentPart`, `Role`, `StreamDelta{Content,Reasoning,ToolCallStart,ToolCallArgs,Finish,Error}`, `DeltaStream`, `FinishReason`, `Usage`.
- q4e page geometry: `shisu_core::q4e_page` — 256-token pages, 30 KiB/position, consts verbatim from `ds4_q4e_page.h`; CUDA-phase build.rs emits the `.cuh` mirror from Rust consts.
- Shape profiles (`shisu_core::shape`): `ShapeProfile::Qwen35` (32 layers, hybrid pattern from config.json `layer_types` or the 8×(3 GDN+1 attn) rule) and `ShapeProfile::Qwen4Exp` (CUDA phase).

---

## Phase 0 — Bootstrap

### Task 1: Workspace scaffold
**Files:** create `shisu/Cargo.toml`, `shisu/rust-toolchain.toml`, `shisu/deny.toml`, empty crates `shisu-core shisu-gguf shisu-metal shisu-engine shisu-kvstore shisu-ir shisu-server shisu-bench` (lib.rs stubs + Cargo.toml; `shisu-kernels`, `shisu-cuda`, `shisu-ple` added in CUDA phase), `scripts/{file_size_check.sh,check.sh}`.
**Interfaces:** workspace deps copied from atlas root Cargo.toml (serde_json preserve_order, thiserror 2, anyhow, parking_lot, axum 0.8, hyper 1, tokio, clap 4, tracing, objc2 stack, minijinja, tokenizers 0.23, safetensors, sha1, hex; cudarc + io-uring declared but only used in CUDA phase).
- [ ] Copy atlas workspace manifest, rename env prefixes ATLAS_→SHISU_, drop atlas-only members; `[lints]` per global constraint 6. Default features: `metal` on macOS, `cuda` on Linux (per-target default-features=false pattern from atlas so nothing silently pulls cudarc on macOS).
- [ ] Write `scripts/file_size_check.sh` (1500-LoC gate over `crates/**/*.rs` + `metal/**`) and `scripts/check.sh` (`cargo check/test/clippy --workspace` + `cargo deny check` + file-size cap); no CI workflow files (GitHub Actions out of scope).
- [ ] `cargo check --workspace` passes on macOS.
- [ ] Commit `chore: shisu workspace scaffold`.

## Phase 1 — Core data layer (macOS-native)

### Task 2: shisu-core
**Files:** `shisu-core/src/{shape.rs,config.rs,dtype.rs,error.rs,backend.rs,model.rs,q4e_page.rs}`.
**Interfaces:** produces `Backend`/`Model` traits + `q4e_page` consts + `ShapeProfile::{Qwen35,Qwen4Exp}` (SSOT above). `Qwen35` profile fields: n_layers=32, hidden=2560, gqa {q_heads:16, kv_heads:4, head_dim:128}, gdn {v_heads:32, qk_heads:16, head_dim:128, conv_kernel:4}, vocab=248320, layer pattern 8×(3×GDN→FFN→1×Attn→FFN) — all read from HF config.json with these as validation asserts.
- [ ] `q4e_page.rs` consts copied from `ds4_q4e_page.h` (host-side only for now; CUDA phase adds the `.cuh` emit).
- [ ] Tests: config.json fixture (downloaded from Qwen/Qwen3.5-4B, checked into `test-vectors/`) parses → profile asserts hold; unknown arch → typed error.
- [ ] Commit.

### Task 3: shisu-gguf
**Files:** `shisu-gguf/src/{reader.rs,metadata.rs,quant.rs,binding.rs}`.
**Consumes:** `shisu_core::dtype`. **Produces:** `GgufFile::open(path)`, `TensorView{name,dtype,data,shape}`, q8_0/q4_k block descriptors from ds4.c §460-1281 (q4e/iq2/mxfp4/exl3 descriptors ported now, exercised in CUDA phase), tensor-name → layer-role mapping driven by `ShapeProfile`.
- [ ] Port GGUF parse + tensor mapping from ds4.c §2231-4966 (host-side only; quantized *kernels* stay in kernel files).
- [ ] Test: parse the local GGUF at `SHISU_TEST_MODEL` (default `/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf`; no download); tensor count/names/dtypes match recorded manifest; hybrid layer pattern maps correctly (24 GDN + 8 attn FFN blocks).
- [ ] Commit.

### Task 3b: CuMetal compat bundle + GDN oracle spike (risk retirement, runs in parallel with T3/T4)
**Files:** create `shisu/kernels/cumetal_compat/{FETCH.sh,include/,cumetal_half_shim.h,STATUS.md}`, `shisu/tests/cumetal/oracle_gdn.cu`, `shisu/scripts/cumetal_probe.sh`.
**Purpose:** decide, with evidence, how much of the CUDA phase can actually be executed on this Mac — before Phase 4 depends on it. Everything here is a dev/test asset: nothing is linked into the shipped binary, so rule 1's "no host C/C++" still holds for `crates/` (the harness is a `.cu` compiled by `cumetalc` into a standalone test executable).
- [ ] `FETCH.sh` pulls the 8 headers the Homebrew bottle omits but the `v0.4.0` tag ships — `cuda_bf16.h`, `cuda_gl_interop.h`, `math_constants.h`, `mma.h`, `sm_35_intrinsics.h`, `vector_functions.h`, `vector_types.h`, and the `cuda/` subdir — from `runtime/api/` at a pinned tag SHA into `include/`, with the Apache-2.0 provenance recorded in a header comment.
- [ ] `cumetal_half_shim.h` implements the half/bf16 helpers ds4 uses that CuMetal's clean-room headers do not declare — measured set: `__halves2half2`, `__low2half`, `__high2half`, `__lows2half2`, `__highs2half2`, `__floats2half2_rn`, `__float2half2_rn`, `__float2half_rn`, `__float22half2_rn`, `__int2half_rn`, `__ushort_as_half`, `__half_as_ushort`, `__half_raw`, `__halves2bfloat162`, `__bfloat1622float2`, `__float2half` — as inline device functions over the bit patterns, each checked against the CUDA definition on the host. Force-included with `-include` on the cumetal lane only.
- [ ] `oracle_gdn.cu`: extract `q4e_gdn_recurrent_kernel` (`ds4_qwen4exp_gpu.cuh:451`, `__launch_bounds__(512)`, dynamic shared memory, `cudaFuncSetAttribute` at :1953, launched at :1962) with the `ds4_gpu_tensor` accessors stubbed; compile `cumetalc --cuda-device --mode experimental oracle_gdn.cu -o oracle_gdn --ptx-strict -I …`; run on the M1 Max; compare against the naive Rust f32 oracle at tol 1e-5.
- [ ] `STATUS.md` records the verdict per construct actually hit: exact command, exact diagnostics, which of {dynamic `extern __shared__`, `__syncthreads`, `__shfl_down_sync`, `atomicAdd(float*)`, `__launch_bounds__`, large dynamic smem opt-in, bf16 conversions} compiled and which failed. Upstream's own limits say compound shared-memory layouts beyond one runtime-sized `extern __shared__` and non-uniform barrier CFGs fail explicitly — expect to find out which side of that line the real kernel is on.
- [ ] Gate (either outcome is a completed task; the deliverable is the verdict): if the kernel compiles and matches the oracle, Tasks 5 and 17 gain the CuMetal differential oracle; if it does not, record the blocking construct, and Phase 4 stays exactly as specified for Linux with CuMetal used only as a compile checker.
- [ ] Commit.

## Phase 2 — Metal backend + qwen3.5-4b engine (the dev-machine flagship)

### Task 4: shisu-metal backend
**Files:** `shisu-metal/src/{device.rs,pipeline.rs,compile.rs,run.rs}`, `metal/generic/*.metal` (copy+adapt ds4 `metal/{norm,softmax,unary,bin,glu,concat,cpy,get_rows,set_rows,sum_rows,repeat,argsort,dsv4_rope}.metal`), `Cargo.toml` (objc2, objc2-foundation, objc2-metal, dispatch2; feature `metal`).
**Consumes:** `shisu_core::Backend`. **Produces:** `MetalBackend: Backend` — device init, buffer alloc, pipeline cache keyed by (kernel, specialization), runtime compile: concatenate `metal/generic/*.metal` + `metal/qwen35/*.metal` (required_sources list, ds4_metal.m:4334 pattern) → `newLibraryWithSource`.
- [ ] Copy generic kernels; strip dsv4/glm specializations; keep q8_0/q4_k dequant paths — verified present in ds4 Metal: `metal/dense.metal:1494` `ds4_dense_block_q4_K`, `:1523`/`:1587` `dequantize_dense_q4_K{,_t4}`, `:1889+` `[[host_name("kernel_mul_mv_ext_q4_K_f32_r1_N")]]` instantiations, plus `kernel_mul_mv_q8_0_f32` / `kernel_mul_mv_q8_0_f32_pair`; `metal/get_rows.metal` and `metal/moe.metal` also carry q4_K paths. MXFP4 exists but is out of scope for qwen3.5-4b.
- [ ] Test (macOS): compile library clean; norm/softmax/get_rows/q4_k gemv vs CPU oracle f32 (tolerance 1e-3 quantized / 1e-5 f32).
- [ ] Commit.

### Task 5: qwen35 MSL kernels
**Files:** `metal/qwen35/{gemv_q4k.metal,gemm_q4k.metal,gdn_conv.metal,gdn_recurrent.metal,gdn_chunk.metal,gated_attn.metal,rmsnorm_rope.metal}`, `metal/qwen35/q35_common.h`.
**Interfaces:** kernel names + arg layouts documented in `metal/qwen35/KERNELS.md` (consumed by Task 6 dispatch); GDN semantics mirror ds4 `q4e_gdn_{conv,l2norm,gates,recurrent,chunk,out_gate}` CUDA kernels (read `ds4_qwen4exp_gpu.cuh` GDN section as the reference algorithm — same math, MSL translation).
- [ ] Reference sources, in priority order: (1) the CUDA `q4e_gdn_*` kernels for arithmetic order (constraint 3 — the MSL port must reproduce the CUDA reduction order, not a mathematically-equal re-association); (2) ds4 `metal/glm53_kda.metal`, which already implements a delta-rule linear-attention recurrence in MSL (`delta_v = (sv[v] - hk) * beta; h = fma(k4, float4(delta_v), h)`) — the closest existing MSL idiom in this codebase; (3) `metal/flash_attn.metal` for the gated-attention adaptation. ds4 Metal has **no** GDN/delta-net kernel today (verified: only `glm53_kda.metal` and unrelated `delta` locals in `dsv4_misc.metal`), so this is genuinely new code.
- [ ] GDN recurrent decode kernel (fixed-size state per layer: 32 V heads × 128×128 + conv state), GDN chunked prefill, gated attention decode+prefill (reuse/adapt `flash_attn.metal`), fused rmsnorm+rope.
- [ ] Test (macOS): each kernel vs naive Rust f32 oracle on small shapes (state update invariants: delta-rule correctness, conv window); prefill/decode consistency (chunked prefill of N tokens == N recurrent steps, tol 1e-3).
- [ ] Test (macOS, if Task 3b cleared its gate): **CuMetal differential oracle** — compile the CUDA `q4e_gdn_*` reference through `cumetalc` into the Task 3b harness, run it on the same inputs as the MSL kernel, and diff. This is a second implementation of the *same source arithmetic* on the same GPU, so it catches translation bugs (reduction order, conv window indexing, gate sign) that a hand-written Rust oracle can miss. Tolerance 1e-5 f32; disagreements are root-caused to a line, never widened.
- [ ] Commit.

### Task 6: shisu-engine qwen35 forward
**Files:** `shisu-engine/src/{qwen35.rs,gdn.rs,attn.rs,ffn.rs,sampling.rs,kv.rs,sessions.rs}` (siblings split at 1500 LoC).
**Consumes:** `Backend`, `shisu_gguf`, `ShapeProfile::Qwen35`. **Produces:** `Qwen35Model: Model`.
- [ ] Forward: embed → 32 hybrid layers (GDN state buffers vs attention KV) → final norm → lm_head; decode = batched, prefill = chunked.
- [ ] Sampling bit-exact port of ds4.c §70205+ (hand-rolled argmax/top-k/top-p/min-p/unrolled softmax — preserve reduction order verbatim; shared with CUDA phase).
- [ ] KV cache: the Task 7 radix-tree cache — paged KV for the 8 attention layers + GDN-state checkpoints at node boundaries + Marconi eviction — NOT contiguous buffers; macOS-first, the same system Task 17 uses for qwen4exp.
- [ ] Test (macOS, `SHISU_TEST_MODEL=/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf`): greedy 64-token decode coherent English; memory ≤ 6 GiB; decode tok/s recorded to `shisu-bench/baseline/m1max-qwen35.json`.
- [ ] Commit.

### Task 6b: OptiQ safetensors loader (secondary model source)
**Files:** create `shisu-gguf/src/safetensors.rs` (rename crate module tree if needed: `shisu-gguf` becomes the model-file loader: `gguf.rs` + `safetensors.rs` + `mlx_quant.rs`), modify `shisu-engine/src/qwen35.rs` (`Model::load` accepts GGUF path or safetensors dir).
**Consumes:** `ShapeProfile::Qwen35`, `Backend`. **Produces:** `ModelFile::open(path)` → same `TensorView` stream as GGUF, so the engine forward path is format-agnostic.
- [ ] Probe first: read `/Users/sercand/models/Qwen3.5-4B/mlx` (local OptiQ dir — `config.json`, `model.safetensors`, `optiq_metadata.json`); record `quantization_config` (mlx-affine group sizes, per-layer 4/8-bit map) into `test-vectors/qwen35-optiq-layout.json`.
- [ ] Implement mlx group-quant dequant (affine: scales + biases per group, 4-bit and 8-bit variants) → f16/f32 `TensorView`; GGUF and safetensors tensors feed the identical `binding.rs` layer-role mapper.
- [ ] Test: one layer's dequantized weight vs mlx oracle (dump via `mlx.core` python one-liner on `/Users/sercand/models/Qwen3.5-4B/mlx`, tol 1e-3); full-model greedy decode on the OptiQ dir produces same tokens as GGUF Q4_K_M within sampling tolerance.
- [ ] Commit.

## Phase 3 — KV cache + API layer (all testable on macOS)

### Task 7: shisu-kvstore — the radix-tree KV cache
**Files:** `shisu-kvstore/src/{tree.rs,pool.rs,page_table.rs,ckpt.rs,evict.rs,spill.rs}`.
**Produces:** the one KV-cache system for every model on every platform: `SpanTree` (token-span radix tree, ds4.c 55072-55600) + refcounted page pool + `PageTable` (logical→pool page) + GDN/SSM checkpoint store (ds4.c 55892-56100) + Marconi utility eviction (`utility = prefill_s_saved / MiB × exp2(-idle/600)`; leaves before interior nodes; ds4.c 55080-55120) + disk spill tier keeping the ds4_kvstore.c 48B-header file format. API: `lookup(tokens) -> (PageTable, Option<Ckpt>)`, `commit(frontier, tokens)`, `evict_to(budget)`, `spill(node)`/`restore(sha)`.
- [ ] Tests (macOS): page refcounts, span split/relink, checkpoint utility ordering, Marconi beats LRU on the SSM-entry workload, spill/restore round-trip byte-exact vs the ds4 header fixture.
- [ ] Commit.

### Task 8: Engine checkpoint integration
**Files:** modify `shisu-engine/src/{sessions.rs,kv.rs}`, create `shisu-engine/src/kv_ckpt.rs`.
- [ ] Resume = `SpanTree::lookup(rendered_tokens)` → page table + deepest GDN checkpoint → skip prefill to that position (engine-owned serialization; `kv_ckpt.rs` maps tree nodes ↔ the disk spill format).
- [ ] Test: resume-from-checkpoint continuation tokens == cold-run tokens; resume latency recorded in shisu-bench.
- [ ] Commit.

### Task 9: shisu-ir
**Files:** copy+trim atlas `crates/spark-server/src/ir/{mod,message,request,response,stream}.rs` into `shisu-ir/` (drop video; keep `ContentPart::Image` shape for future).
- [ ] Tests: copy applicable atlas `ir/tests.rs` subset.
- [ ] Commit.

### Task 10: shisu-server skeleton
**Files:** `shisu-server/src/{main.rs,cli.rs,state.rs,router.rs}`, `api/{misc_handlers.rs,inference_types.rs}`, `scheduler/{mod.rs,queue.rs}`, `tokenizer/{chat_render.rs,jinja_helpers.rs,message_preprocess.rs}` (atlas copies; minijinja + preserve_order + custom `tojson`).
**Consumes:** `shisu_ir`, `Model` trait. **Produces:** `shisu serve` (clap: `--model`, `--bind`, `--port`, `--auth-tokens`, `--kvstore-dir`, `--ctx`); axum router (atlas `serve_router.rs` pattern: hyper manual accept loop, header_read_timeout, CORS, catch-panic, graceful shutdown drain); routes `/v1/models`, `/health`, `/metrics`, `/cache`; scheduler thread owning `Box<dyn Model>` behind mpsc `InferenceRequest` channel (atlas boundary pattern, not trait-call-from-handler).
- [ ] Test: `tower::ServiceExt::oneshot` — /health, /v1/models, auth timing-safe reject; qwen35 model boots under scheduler thread.
- [ ] Commit.

### Task 11: OpenAI adapters
**Files:** `shisu-server/src/openai/{mod.rs,chat_request.rs,to_ir.rs,encode.rs,encode_stream.rs,responses.rs,responses_lowering.rs,completions.rs,stream_chunk.rs}`, `response_store.rs`.
- [ ] Responses API lowered to chat request (`lower_responses_to_chat`, `previous_response_id` via response_store) — atlas pattern.
- [ ] Tests: wire fixtures → IR → template string golden; SSE chunk framing golden (`delta_to_chunk_events` incl. `include_usage` terminal frame); live e2e vs running qwen35 server (blocking + stream, chat + responses + completions).
- [ ] Commit.

### Task 12: Anthropic adapter
**Files:** `shisu-server/src/anthropic/{types.rs,to_ir.rs,translator.rs,handlers.rs,handlers_stream.rs}` (atlas copies: `AnthropicTranslator` state machine, `anthropic_sse_from_deltas` mpsc(1024) + per-delta flush + KeepAlive).
- [ ] Tests: `/v1/messages` blocking + streaming golden SSE fixtures; live e2e vs qwen35 server.
- [ ] Commit.

### Task 13: Scheduler completion
**Files:** `shisu-server/src/scheduler/{lifecycle.rs,batching.rs,cancel.rs}`.
- [ ] Port ds4_server.c coordinator semantics: resident sessions, bounded prefill quanta behind model mutex, per-session KV ownership, cancellation keeps worker ownership; `DeltaStream` back to handlers.
- [ ] Test: two concurrent streaming requests interleave; mid-stream cancel frees session; radix-tree prefix hit skips prefill to the deepest cached span + checkpoint.
- [ ] Commit.

## Phase 4 — CUDA / qwen4exp (kernel work starts on this Mac via CuMetal; driver, perf and parity gates on Linux)

Split rule for this phase: anything that is *CUDA source* (kernel bodies, TU splits, per-op numerics) is authored and compile-verified here through `cumetalc`, and numerically oracle'd on the Apple GPU through the Task 3b harness. Anything that is *NVIDIA host machinery* (cudarc driver API, cublasLt, CUDA-graph capture, cooperative launch, io-uring) is written here but only executed on the Linux box, against a fake `Backend` in unit tests. `SHISU_CUDA_IMPL=nvidia` is the only shipping path; `SHISU_CUDA_IMPL=cumetal` is a dev/test lane.

### Task 14: Kernel build crate
**Files:** create `shisu-kernels/build.rs` + `build_target.rs` + `build_codegen.rs` + `build_cumetal.rs` (adapt atlas `crates/atlas-kernels/build*.rs`), `kernels/KERNEL.toml`, `kernels/cuda/exl3/` + `kernels/cuda/mmq/` (verbatim `cp -r` from ds4 `cuda/`), `kernels/cumetal_compat/` (from Task 3b), `shisu-kernels/src/lib.rs` (`pub fn ptx(module:&str)->&'static [u8]` + `pub fn metallib(module:&str)->&'static [u8]`).
- [ ] build.rs backend selection: `SHISU_SKIP_BUILD=1` → empty stub (default on both OSes); `SHISU_CUDA_IMPL=nvidia` (Linux default) → per-dir KERNEL.toml → nvcc `--ptx` per unique (source,arch,flags), dedup, emit `OUT_DIR/target_ptx.rs`; `SHISU_CUDA_IMPL=cumetal` (macOS dev) → same KERNEL.toml → `cumetalc --cuda-device --mode experimental --emit metallib --ptx-strict -I kernels/cumetal_compat/include -I kernels/cumetal_compat -include cumetal_half_shim.h` per TU, emit `OUT_DIR/target_metallib.rs`. Flags from ds4 Makefile `NVCC_BASE_FLAGS`: `-O3 -g -lineinfo --use_fast_math`; default `compute_75` PTX + host cubin; `SHISU_CUDA_ARCH=sm_120a,sm_121a` adds `-DSHISU_CUDA_HAVE_MXF4=1`.
- [ ] cumetal lane is **opt-in and never in CI**: it needs Xcode's Metal toolchain + Homebrew LLVM, and its output is a `.metallib`, not PTX. A cumetal compile failure is a diagnostic, not a build failure, unless `SHISU_CUMETAL_STRICT=1`.
- [ ] No CI workflow (GitHub Actions out of scope): the Linux nvcc gate runs manually on the Linux box via `SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels`.
- [ ] Verify: `SHISU_SKIP_BUILD=1 cargo build -p shisu-kernels` on macOS; `SHISU_CUDA_IMPL=cumetal SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels` compiles the TUs that Task 3b cleared; real nvcc build compiles exl3+mmq TUs on Linux.
- [ ] Commit.

### Task 15: q4e kernel extraction (CUDA, no Rust)
**Files:** create `kernels/cuda/q4e/shisu_dev_inc/{common.cuh,tensor.cuh}` (extract device helpers + `ds4_gpu_tensor` accessors from ds4_cuda.cu), split `ds4_qwen4exp_gpu.cuh` (4000 ln) into `hc.cu moe.cu gdn.cu ple.cu matvec.cu qsa.cu idx.cu misc.cu` + `q4e.cuh`; `KERNEL.toml` per family.
- [ ] Mechanical split: kernel bodies verbatim; each `.cu` includes `shisu_dev_inc/*` + generated `q4e_page.cuh`. No arithmetic edits.
- [ ] Verify (Linux, authoritative): `nvcc --ptx` each TU clean (strict warnings); `cuobjdump --dump-elf-symbols` set equality vs original TU.
- [ ] Verify (this Mac, cheap and early): `SHISU_CUMETAL_STRICT=1` compile of every split TU via `cumetalc --cuda-device --ptx-strict`. Catches missing includes, accidental arithmetic edits, and unsupported-construct fallout without a Linux box. Record per-TU status in `kernels/cumetal_compat/STATUS.md`; a TU blocked by a documented CuMetal gap (compound shared layouts, unsupported atomic form) is annotated, not fixed by rewriting arithmetic.
- [ ] Commit.

### Task 16: shisu-cuda backend
**Files:** `shisu-cuda/src/{device.rs,ptxmod.rs,blas.rs,graph.rs,launch.rs}`, `shisu-cuda/src/sys/{mod.rs,nvidia.rs}`.
**Consumes:** `shisu_kernels::ptx`, `shisu_core::Backend`. **Produces:** `CudaBackend: Backend` — cuModuleLoadData registry, kernel-handle cache, cublasLt tier (build.rs link hints copied from atlas `spark-runtime/build.rs` incl. macOS early-return), CUDA-graph capture + decode-graph cache (port host-side from ds4_cuda.cu, qwen4exp layer graph only), cooperative-launch support (exl3 needs it).
- [ ] **cudarc is Linux-only.** Verified on this Mac: `libcumetal.dylib` exports 115 `cu*` driver symbols with unversioned names (`cuCtxCreate`, `cuMemAlloc`, `cuMemcpyHtoD`) and no `nvrtc*`; cudarc 0.19.3 binds 482 symbols under versioned names (`cuCtxCreate_v2`, `cuDeviceGetUuid_v2`, …), so it cannot bind CuMetal. Keep cudarc behind `#[cfg(target_os = "linux")]`; the macOS build of `shisu-cuda` compiles the orchestration against `sys::nvidia` stubs so the code is type-checked everywhere but only callable on Linux.
- [ ] Test (Linux+GPU, `#[ignore]`d otherwise): load every PTX module, trivial launch, graph capture/launch round-trip.
- [ ] Test (macOS, no GPU driver needed): arg-packing, module-registry, and decode-graph bookkeeping unit tests against a fake `Backend`; these are the parts that can rot silently while waiting for Linux hardware.
- [ ] Commit.

### Task 17: shisu-engine qwen4exp forward
**Files:** `shisu-engine/src/qwen4exp/{mod.rs,hc.rs,qsa.rs,gdn.rs,moe.rs,mtp.rs,paged_kv.rs}`.
**Consumes:** `Backend`, `shisu_gguf` (q4e/exl3 descriptors from Task 3), `q4e_page`. **Produces:** `Qwen4ExpModel: Model`; reuses `sampling.rs` (Task 6) and `sessions.rs`.
- [ ] Port per-layer forward from ds4 CUDA-graph reference path (hc → QSA attn → GDN → MoE → PLE layer placeholder wired in Task 19), paged-KV pool + span-tree host bookkeeping (ds4.c §70205+ host part), MTP speculative decode (ds4.c §67082-70205).
- [ ] Test (macOS): CPU-oracle per-op tests (matvec/norm/gdn-recurrent, tol 1e-5) + CuMetal-oracle per-op tests for every op whose kernel cleared Task 3b's gate, driven with real qwen3.5-4b weight slices from `SHISU_TEST_MODEL` so shapes are not toy; forward-graph bookkeeping tests on a fake `Backend`.
- [ ] Test (Linux GPU, `#[ignore]`d): full-model smoke on `SHISU_TEST_MODEL` — 32-token greedy decode completes.
- [ ] Commit.

### Task 18: Golden-vector parity
**Files:** `shisu-bench/src/bin/parity.rs`, `test-vectors/` (copy ds4 `tests/test-vectors/flash*`), `shisu-engine/tests/parity.rs`.
- [ ] Replay official golden prompts (greedy, fixed seed); assert token-id sequence + top-logit parity vs ds4 vectors; mismatches traced to arithmetic root cause (constraint 3).
- [ ] Commit.

### Task 19: shisu-ple + integration
**Files:** `shisu-ple/src/{handle.rs,cache.rs,uring.rs,pool.rs,prefetch.rs}`; modify `shisu-engine/src/qwen4exp/{mod.rs,ple.rs}`.
**Produces:** `PleStream::open(shard_path, cache_bytes)`, `fetch(ngram_rows)`, `prefetch(...)`, `stats()` — mirrors `ds4_ple_stream.h` 1:1.
- [ ] Port ds4_ple_stream.c (1159 ln): 320,001,536-row × 160 IQ4_NL shard, 16 scattered ~90-byte rows/token; io-uring backend (Linux) + blocking-pread pool fallback; `SHISU_PLE_IO` override.
- [ ] Engine coupling: gather starts before layer 0, awaited just before the PLE layer (ds4.c pattern); `q4e_ple_dequant` finishes on GPU.
- [ ] Tests: synthetic shard fixture — fetch correct rows on both backends; cache budget respected; prefetch overlap via stats; p99 fetch latency recorded.
- [ ] Commit.

## Phase 5 — Verification

### Task 20: Perf gates + end-to-end
**Files:** `shisu-bench/src/{decode.rs,prefill.rs,kv.rs,ple.rs,baseline.rs}`, `shisu-bench/baseline/{m1max-qwen35.json,ds4-qwen4exp.json}`, `shisu-server/tests/e2e.rs`.
- [ ] Gates: macOS qwen35 decode/prefill/resume-latency vs recorded baselines (regression >5% fails `scripts/check.sh`); CUDA qwen4exp decode/prefill tok/s ≥ ds4 baseline + PLE p99 ≤ ds4 stats (Linux+NVIDIA only).
- [ ] CuMetal lane gate (macOS, cheap, runs whenever `SHISU_CUDA_COMPILE=1`): every TU listed `ok` in `kernels/cumetal_compat/STATUS.md` still compiles, and every oracle-backed kernel still matches its MSL/Rust twin at 1e-5. This is a **compile + numerical** gate only — no throughput number from CuMetal is recorded or compared (constraint 4).
- [ ] e2e: boot server, exercise all four endpoints (chat blocking+stream, responses, completions, messages), assert schema-valid JSON/SSE.
- [ ] `cargo clippy --workspace --all-targets` clean; `cargo deny check`; `scripts/file_size_check.sh` green.
- [ ] Commit `chore: perf baselines + e2e`.

---

## Roadmap / parallelism (≤2 subagents, macOS-first)

```
Wave 1: T1 ──► T2 ──┬── T3 ──┐
                    ├────────┴── T4 ── T5 ── T6          (Metal + engine proven on M1 Max)
                    └── T3b (CuMetal oracle spike; independent, retires CUDA-phase risk)
Wave 2: T6b ∥ T7 ∥ T9  →  T8  →  T10  →  T11 ∥ T12  →  T13    (API layer, live qwen35 server)
Wave 3a (this Mac, via CuMetal): T14 ──┬── T15 ──┐
                                      └─────────┴── T16* ── T17*   (kernels compiled + numerically
                                                                    oracle'd on Apple GPU; host logic
                                                                    against a fake Backend)
Wave 3b (Linux+GPU): T14(nvcc) ── T15(cuobjdump) ── T16(graphs/cublasLt) ── T17(smoke) ── T18 ── T19
Wave 4: T20
```

Dependency notes: T5 needs T4's compile path and (if T3b cleared the gate) the CuMetal oracle as a second reference; T6 needs T3+T4+T5; T6b needs T6 (secondary source; engine is GGUF-complete without it); T8 needs T6+T7; T11/T12 parallel after T10. CUDA phase (T14-T19) is independent of the API layer. T14/T15 and the *kernel* half of T16/T17 can start on this Mac as soon as T3b reports; the driver-API, cublasLt, CUDA-graph, cooperative-launch, golden-vector and perf work (T16 real backend, T18, T19, T20 CUDA gates) stays Linux-only.

Per-task subagent brief: include this plan path, ds4/atlas source paths, the task's Files/Interfaces blocks, global constraints 1-7, the CuMetal section's hard-limits table if the task touches CUDA, and `SHISU_TEST_MODEL` path. No task runs formatters/linters/full test-suites mid-batch; Task 20 owns the final gate.

---

## Plan verification log (2026-09-05, against this checkout)

Checked before executing anything. Everything below was read/measured, not assumed.

**Holds as written:** ds4 line refs — `ds4.c` 81 198 ln (§460-1281 GGUF, §2231-4966 tensor mapping, §67082-70205 MTP, §70205+ sampling all in range), `ds4_ple_stream.c` 1159 ln, `ds4_kvstore.c` 1411 ln, `ds4_q4e_page.h` 47 ln, `ds4_metal.m` 44 741 ln with the `required_sources` concat pattern at :4334, `ds4_gpu.h` 3281 ln. `metal/` contains every file Task 4 lists (norm, softmax, unary, bin, glu, concat, cpy, get_rows, set_rows, sum_rows, repeat, argsort, dsv4_rope, flash_attn, dense); `cuda/` contains exactly `exl3` + `mmq`; `tests/test-vectors/flash*` exists. atlas has `crates/spark-server/src/ir/{mod,message,request,response,stream,tests}.rs`, `crates/atlas-kernels/build{,_target,_codegen}.rs` (plus `_diagnose/_parse/_shadow` to mine for patterns), `crates/spark-runtime/build.rs`, `.github/workflows/{ci,file-size-cap,kernel-compile}.yml`, `scripts/ci_gpu_stubs.sh`, and `rust-toolchain.toml` pinned `1.93.1`. `unsloth/Qwen3.5-4B-GGUF` exists with `Qwen3.5-4B-Q4_K_M.gguf` (Task 3/6 path is correct).

**Corrected in this revision:** `ds4_qwen4exp_gpu.cuh` is 4000 ln, not 3925. nvcc flags are `-O3 -g -lineinfo --use_fast_math` (`NVCC_BASE_FLAGS`, Makefile:98) — the `-g` was missing. Task 4's q4_k claim now cites the actual `metal/dense.metal` symbols. Task 5 now names `metal/glm53_kda.metal` as the only existing MSL delta-rule reference in ds4 (there is no GDN kernel in `metal/`).

**New in this revision:** CuMetal dev lane (section above), Task 3b, the macOS halves of Tasks 14/15/16/17, the Wave 3a/3b split, and the measured cudarc/CuMetal incompatibility that keeps `shisu-cuda`'s FFI Linux-only.
