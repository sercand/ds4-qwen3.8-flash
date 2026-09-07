# Task 14: Kernel build crate

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 14 (lines 246–253) +
§ CuMetal dev lane (39–74) + Roadmap Wave 3a/3b (309–317). Read all three before executing.
**Depends on:** Task 1 (workspace root `shisu/Cargo.toml`, `[workspace.lints]`, local gate
`shisu/scripts/check.sh` exporting `SHISU_SKIP_BUILD=1`; T1 plan line 45 deliberately leaves `shisu-kernels` uncreated —
**this task creates it**). Task 3b (`shisu/kernels/cumetal_compat/{FETCH.sh,include/,cumetal_half_shim.h,STATUS.md}`
and its *verified* `cumetalc` invocation form + ⚠ entries, reused verbatim). No T2–T13 interfaces.
**Produces** (exact API T15/T16/T17 build on; paths relative to repo root):

- Crate `shisu/crates/shisu-kernels/` (`build.rs`, `build_target.rs`, `build_codegen.rs`,
  `build_cumetal.rs`, `src/lib.rs`, `Cargo.toml`) + workspace membership + `[workspace.dependencies]`
  entry `shisu-kernels = { path = "crates/shisu-kernels" }` (edit `shisu/Cargo.toml`, T1-owned file — additive only).
- `shisu/kernels/KERNEL.toml` + per-dir manifests — **the schema below is the SSOT**; T15's
  q4e-family manifests and the T16/T17 loaders consume these keys verbatim.
- `src/lib.rs` generated surface (master-plan SSOT line 121; names verbatim):
  - `pub fn ptx(module: &str) -> &'static [u8]` — unknown module ⇒ `&[]` + `debug_assert!` (fail-closed at load, no panic in release)
  - `pub fn metallib(module: &str) -> &'static [u8]` — same discipline; `&[]` on every shipping/skip build
  - `pub fn ptx_modules() -> Vec<(&'static str, &'static [u8])>` / `pub fn metallib_modules() -> …` (registry audits)
  - `pub const KERNEL_SET_HASH: &str` (set-content fingerprint; closes the stale-`include!` hole)
- No new scripts/workflows: GitHub Actions is out of scope (master Scope-Out). The Linux nvcc lane is a
  manual runbook (Step 10), not a CI job.

## Global Constraints

Master-plan Rules 1–7: (1) all new code under `shisu/`; only `.cu/.cuh/.h/.md` kernel files copied — no host C.
(2) 1500-LoC cap per file — budgets below. (3) copies are byte-for-byte; only `#include` plumbing and
launch wrappers may change (none change here — T14 compiles the TUs *as-is*). (4) CuMetal numbers
never gate anything — this task records compile verdicts only. (5) env knobs `SHISU_`-prefixed only.
(6) workspace lints (`warnings=deny`, clippy deny); thiserror/anyhow unused here (build-script panics +
`cargo:warning=` are the error currency; lib.rs has zero deps). (7) no license headers — except vendored
`cuda/exl3/LICENSE`, `cuda/mmq/VENDOR.md` ride along verbatim.

Task-specific (CuMetal hard-limits rows that bind this task, master plan 57–70):
**Cooperative grids** — `cuda/exl3/ds4_exl3.cu` launches cooperatively (verified below) ⇒ exl3 is
**not** a cumetal-lane target; Linux-only, excluded in its manifest. **NVRTC** — absent on the Mac ⇒
`SHISU_SKIP_BUILD=1` is the default on *both* OSes (the local gate exports it; a bare `cargo build`
without nvcc must stay green). Shared-mem/atomics/FP64/timing rows bind T15/T17 (the build crate only
invokes the compilers); the lane's opt-in/never-a-default-build-failure rule is master-plan line 249.

## Source References (verified)

atlas root = `/Users/sercand/Developer/src/github.com/sercand/atlas`; ds4 root = cwd. Every row opened this session.

| Source | Lines | What / how used |
|---|---|---|
| atlas `crates/atlas-kernels/build.rs` | 145–167 | `cargo:rerun-if-env-changed=` emissions for every selector env var (cached build would otherwise keep a stale registry); `ATLAS_SKIP_BUILD` gate. Copied with `SHISU_*` names. |
| ″ | 169–189 | Skip path: write empty-registry `target_ptx.rs` stub + `rustc-env` hash, return. Model for the stub (plus `target_metallib.rs`). |
| ″ | 255–267, 455–529 | Dedup split into `compile_jobs` keyed **unique (source, arch, sorted flags)** + `copy_jobs` (byte-copies); `thread::scope` parallel nvcc with worker count `min(parallelism, jobs)`; dedup-ratio `cargo:warning=` line. |
| ″ | 546–564, 639–649 | `cargo:rustc-env=…KERNEL_SET_HASH=` from FNV-1a/64 content hash (12 hex) of the generated source — the recompilation trigger cargo cannot see through a build-script-generated `include!`; `…_PTX_DIR=` style OUT_DIR export. |
| ″ | 1319–1333, 1383–1392 | `#[path = "build_*.rs"] mod …;` inclusion so one file serves build-script *and* integration tests (a build script's `#[cfg(test)]` modules never run under `cargo test`). |
| atlas `crates/atlas-kernels/build_target.rs` | 19–45, 70–148, 456–482 | `ComputeTarget` trait (output_extension/compile/uses_cuda_module_api), `NvccTarget::compile` argv `["--ptx", "-arch=<arch>", "-O3", …]` (line 96), factory. Mirrored as `CompileTarget`/`NvccTarget`. |
| atlas `crates/atlas-kernels/build_codegen.rs` | 24–45, 98–116 | `generate_target_ptx_rs` shape: `include_bytes!` consts typed `&[u8]`, `ptx_modules()`/`metallib_modules()` alias pair — one uniform byte representation for text PTX and binary metallib. |
| ″ | 350–365 | `find_cuda_dir()` — nvcc discovery (`CUDA_HOME`/PATH conventions); adapted. |
| atlas `crates/atlas-kernels/src/lib.rs` | 32–45 | `include!(concat!(env!(OUT_DIR), "/target_ptx.rs"))` + `KERNEL_SET_HASH` const + the documented staleness hole and its closure — copied discipline. |
| atlas `crates/atlas-kernels/Cargo.toml` | 11–24 | `[build-dependencies] toml = "0.8"`, `[dev-dependencies] toml = "0.8"` (same crate ⇒ `Cargo.lock` unchanged) — copied. |
| atlas `crates/atlas-kernels/tests/kernel_shadow_detector.rs` | 24–25 | `#[path = "../build_shadow.rs"] mod build_shadow;` — the integration-test-side inclusion pattern T14's tests copy for `build_cumetal.rs`. |
| atlas `kernels/gb10/qwen3.6-27b/nvfp4/KERNEL.toml` | 1–12 | Manifest precedent: `[build] extra_nvcc_flags = [...]` + `[modules] stem = "module"` overrides. shisu schema extends, never breaks these key names. |
| atlas `.github/workflows/kernel-compile.yml` | 4–21, 69, 93–110, 112–146, 148–161 | Reference for the Linux-box runbook (Step 10): run the **real** build script (never a shell nvcc loop), fail-closed "at least one real `target_ptx.rs`" existence check (`grep -c 'do not edit'`) + `-- --ignored` registry tests. The container/runner rows are moot (no GHA); the container image `nvidia/cuda:13.0.2-devel-ubuntu24.04` is the recommended Linux-box environment. |
| ds4 `Makefile` | 98–99 | `NVCC_BASE_FLAGS := -O3 -g -lineinfo --use_fast_math -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread`; `NVCCFLAGS ?= $(NVCC_BASE_FLAGS) $(NVCC_ARCH_FLAGS)`. Flag SSOT (master-plan verification-log line 329 confirms the `-g`). |
| ″ | 102–103, 105 | `MMQ_INCLUDES := -Icuda/mmq`; 7 production mmq TUs = `ds4_ggml_stubs ds4_mmq ds4_mmq_d2r quantize mmid mmvq ds4_repack`; `EXL3_OBJS := cuda/exl3/ds4_exl3.o` (one TU). = the `sources` lists. |
| ″ | 465–466, 471–490 | Per-TU recipes: exl3 `$(NVCC) $(NVCC_BASE_FLAGS) $(EXL3_ARCH_FLAGS) -std=c++17`; mmq `$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES)`. ⇒ `-std=c++17` everywhere; mmq needs the dir itself on the include path. |
| ″ | 60–63, 85–95 | Pinned `sm_120a/121a` builds add `-DDS4_CUDA_HAVE_MXF4=1` (⚠3); exl3 gets host-GPU cubin alone, `-arch=sm_80` fallback — "fp16 tensor-core code … **no Turing form at all**" ⇒ ⚠2 arch pin. |
| ds4 `cuda/exl3/` | whole dir, 17 files / 3,557 ln (`wc -l`) | Verbatim copy candidate incl. `LICENSE`, `VENDOR.md`. |
| ds4 `cuda/exl3/ds4_exl3.cu` | 6–7, 228, 280, 383 | `#include <cooperative_groups.h>`, `cg = cooperative_groups`, 3× `cudaLaunchCooperativeKernel` ⇒ cooperative TU ⇒ **excluded from the cumetal lane** (hard-limits row "Cooperative grids"; master plan line 61 + `cuda/exl3/VENDOR.md:61-62` "Launched cooperatively so the co-residency its group barriers assume is enforced"). |
| ds4 `cuda/mmq/` | 24 top-level files / 24,504 ln; `test/` 16 files, `vendors/` 2 files | Verbatim copy incl. non-built `test/`+`vendors/` (byte-for-byte traceability); nvcc path builds only the 7 `MMQ_OBJS` TUs. mmq production TUs use **no** cooperative launch (grep: only `test/proto_*.cu` + device-prop stub `ds4_ggml_stubs.cu:74`). |
| T3b plan `docs/superpowers/plans/tasks/task-03b-cumetal-compat.md` | 203–219 | Verified invocation: `cumetalc --cuda-device --mode experimental <src> -o <out> --ptx-strict -I kernels/cumetal_compat/include -I kernels/cumetal_compat --cuda-include cumetal_half_shim.h -I "$PREFIX/include"`; `-include` is **rejected** (⚠2); gate order SKIP ⇒ COMPILE ⇒ binary-present. Reuse verbatim. |
| master plan | 121, 244 (T3b plan), 248–251 | Accessor SSOT signatures; STATUS.md "(T14/15 append per-TU rows here)" reservation; Task 14 checkbox text this plan mirrors. |

⚠ **DEVIATION 1 (`-include`, master plan line 248):** `cumetalc` rejects `-include` (T3b ⚠ 2,
measured); the composed argv uses `--cuda-include cumetal_half_shim.h` instead.
⚠ **DEVIATION 2 (exl3 arch, master plan line 248 "default `compute_75` PTX"):** the vendored EXL3
kernels have no Turing form (Makefile:85–89 builds that TU at `-arch=sm_80`). The exl3 manifest pins
`arch = ["compute_80"]` (the PTX-emitting equivalent for driver JIT on sm_80+); mmq keeps the
`compute_75` default (Makefile:69–79 keeps a compute_75 cubin as the portable fallback).
⚠ **DEVIATION 3 (MXF4 define name):** ds4 uses `-DDS4_CUDA_HAVE_MXF4=1` (Makefile:61,63); master plan
line 248 renames it `SHISU_CUDA_HAVE_MXF4` — adopted. Safe here: the flag-bearing copies (`cuda/exl3`,
`cuda/mmq`) reference it **nowhere** (grep this session: sole user is `ds4_cuda.cu:460,3167,13192+`, host
code that is *not* copied). T15's device-side split TUs must use the `SHISU_` name when they branch on it.

## Plan

File budgets (Rule 2): `build.rs` ≤ 380, `build_target.rs` ≤ 160, `build_cumetal.rs` ≤ 330,
`build_codegen.rs` ≤ 160, `src/lib.rs` ≤ 90, `tests/build_logic.rs` ≤ 300.

- [ ] **Step 1 — Preflight (read-only).** `shisu/Cargo.toml`, `shisu/crates/shisu-core/` exist (T1 landed);
  `shisu/kernels/cumetal_compat/FETCH.sh` exists (T3b landed — if absent, STOP: Wave 3a is gated on it,
  master plan line 317); `shisu/crates/shisu-kernels` and `shisu/kernels/cuda` absent.

- [ ] **Step 2 — Verbatim kernel copies (Rule 3).**
  ```sh
  mkdir -p shisu/kernels/cuda
  cp -r cuda/exl3 shisu/kernels/cuda/exl3
  cp -r cuda/mmq  shisu/kernels/cuda/mmq
  ```
  Whole directories — `test/`, `vendors/`, `LICENSE`, both `VENDOR.md` ride along. Proof of fidelity
  (the *only* allowed later delta is the added `KERNEL.toml`s of Step 3):
  `diff -r cuda/exl3 shisu/kernels/cuda/exl3` and `diff -r cuda/mmq shisu/kernels/cuda/mmq` must print
  only `Only in shisu/…: KERNEL.toml` lines.

- [ ] **Step 3 — KERNEL.toml schema + three manifests.** Top-level `shisu/kernels/KERNEL.toml` holds the
  lane defaults; `shisu/kernels/cuda/<dir>/KERNEL.toml` deep-merges over it (dir wins per key).
  **Every key is listed; unknown keys are a build error** (typo-safety beats atlas's hand-rolled leniency).

  ```toml
  # shisu/kernels/KERNEL.toml — lane defaults (T15/T16/T17 consume verbatim)
  [build]
  std = "c++17"                        # every ds4 TU rule carries -std=c++17 (Makefile:466,472)
  base_flags = ["-O3", "-g", "-lineinfo", "--use_fast_math"]   # NVCC_BASE_FLAGS, Makefile:98 (host-only -Xcompiler tokens dropped)
  extra_nvcc_flags = []               # appended last; atlas-compat key name (atlas KERNEL.toml:2)
  defines = []                        # ["K=V", …] → -DK=V
  includes = []                       # dirs relative to kernels/ → -I
  arch = ["compute_75"]               # default PTX target (master plan 248); overridden per dir / $SHISU_CUDA_ARCH
  outputs = ["ptx"]                   # "ptx" | "cubin"; cubin additionally emits a SASS object per sm_* arch into OUT_DIR/ptx (T16 consumption; not required by T14 acceptance)

  [cumetal]
  enabled = true                      # lane still requires SHISU_CUDA_IMPL=cumetal && SHISU_CUDA_COMPILE=1
  flags = ["--cuda-device", "--mode", "experimental", "--ptx-strict"]   # verified form, T3b 209–212
  extra_flags = []
  include_dirs = ["cumetal_compat/include", "cumetal_compat"]   # relative kernels/
  shim = "cumetal_compat/cumetal_half_shim.h"     # force-included via --cuda-include (⚠1)
  ```

  Per-dir `[build]` adds: `sources = ["a.cu", …]` (TUs to compile; every other file is include-only),
  plus the `[cumetal]` keys `exclude = ["stem", …]` (TUs barred from the lane) and
  `sources = …` (override; `null` = same as `[build].sources`). `[modules]` maps file-stem → module
  name (atlas pattern, atlas KERNEL.toml:4–12); unlisted stems use the stem. **Module name is the
  single registry key for both lanes** — `ptx("x")`/`metallib("x")` are one registration (master plan 121).

  Worked example — `shisu/kernels/cuda/exl3/KERNEL.toml`:
  ```toml
  [build]
  sources = ["ds4_exl3.cu"]           # EXL3_OBJS, Makefile:105
  arch = ["compute_80"]               # ⚠2 — no Turing form (Makefile:85–89)
  [modules]
  ds4_exl3 = "exl3"
  [cumetal]
  exclude = ["ds4_exl3"]              # cooperative launch (ds4_exl3.cu:228,280,383) — hard-limits row 1
  ```
  `shisu/kernels/cuda/mmq/KERNEL.toml`:
  ```toml
  [build]
  sources = ["ds4_ggml_stubs.cu", "ds4_mmq.cu", "ds4_mmq_d2r.cu", "quantize.cu", "mmid.cu", "mmvq.cu", "ds4_repack.cu"]   # MMQ_OBJS, Makefile:103
  includes = ["cuda/mmq"]             # MMQ_INCLUDES, Makefile:102
  # arch/defines: lane defaults (compute_75)
  ```

- [ ] **Step 4 — Crate manifest + workspace wiring.** `shisu/crates/shisu-kernels/Cargo.toml`:
  package fields via `.workspace = true`, `publish = false`, no `[dependencies]` (lib is dep-free),
  `[build-dependencies] toml = "0.8"`, `[dev-dependencies] toml = "0.8"` (atlas Cargo.toml:11–24),
  `[lints] workspace = true`. Edit `shisu/Cargo.toml`: append `"crates/shisu-kernels"` to `members`
  and `shisu-kernels = { path = "crates/shisu-kernels" }` to `[workspace.dependencies]` — additive hunks only.

- [ ] **Step 5 — `build.rs` (entry, gates, orchestration).**
  1. Emit `cargo:rerun-if-env-changed=` for `SHISU_SKIP_BUILD`, `SHISU_CUDA_IMPL`, `SHISU_CUDA_ARCH`,
     `SHISU_CUDA_COMPILE`, `SHISU_CUMETAL_STRICT`, `SHISU_CUMETAL_PREFIX` (atlas build.rs:145–157 pattern).
  2. **Gate:** `SHISU_SKIP_BUILD=1|true` wins over everything (rule 5) ⇒ write the two skip stubs
     (`target_ptx.rs`, `target_metallib.rs`: empty registries — every accessor compiles, returns empty;
     atlas build.rs:169–189), emit `cargo:rustc-env=SHISU_KERNEL_SET_HASH=<fnv1a of stub text>` +
     `cargo:rustc-env=SHISU_KERNELS_OUT_DIR=<out>`, return. Else if `SHISU_CUDA_COMPILE != 1` ⇒ same stub
     path, `cargo:warning=` noting why (master plan 72: "required for any real compile").
  3. Impl select: `SHISU_CUDA_IMPL=nvidia|cumetal`; default `nvidia` on Linux, `cumetal` on macOS
     (master plan 72 contract). `nvidia` on macOS without a real nvcc ⇒ `cargo:warning=` + stub (never a
     hard fail on the Mac); `cumetal` on Linux ⇒ hard error (lane is macOS-only).
  4. `build_cumetal::plan()` parses manifests (Step 7) → jobs; dedup unique
     `(canonical source path, arch, sorted effective flags)` → one compile job, duplicates become
     byte-copy jobs (atlas build.rs:255–267). Parallel `std::thread::scope` compile, workers
     `min(available_parallelism, compile_jobs)`; emit the dedup-ratio `cargo:warning=` line.
  5. nvidia lane: each job → `build_target::NvccTarget::compile` → `OUT_DIR/ptx/<dir>/<stem>__<arch>.ptx`;
     any hard error panics (fail-closed; a shipping binary with missing PTX must not link silently).
     Then `build_codegen::generate(...)` → `OUT_DIR/target_ptx.rs` (+ hash/OUT_DIR rustc-env, Step 8).
  6. cumetal lane: jobs minus `[cumetal].exclude`; command per Step 7; **a compiler failure is
     `cargo:warning=<tu>: <first diagnostic line>` and the module is omitted**, unless
     `SHISU_CUMETAL_STRICT=1` ⇒ panic listing every failed TU (master plan 249). Emit
     `OUT_DIR/target_metallib.rs` from the successes. The lane never gates a default build (249) and its bytes are
     never shipped (Scope "Out").

- [ ] **Step 6 — `build_target.rs`.** `trait CompileTarget: Send + Sync { fn output_extension; fn compile(&self, source, arch, flags, out) -> Result<(), String>; }`
  + `struct NvccTarget { nvcc: PathBuf }` (find via `CUDA_HOME`/`PATH`, atlas build_codegen.rs:350–365
  adapted). argv (order fixed, pinned by a unit test): `["--ptx", "-arch=<arch>", "-O3", "-g",
  "-lineinfo", "--use_fast_math", "-std=c++17", <-I…>, <-D…>, <extra_nvcc_flags…>, <source>, "-o", <out>]`.
  `SHISU_CUDA_ARCH` (comma list) replaces the manifest arch list; entries matching
  `sm_120a|sm_121a|compute_120a|compute_121a` add `-DSHISU_CUDA_HAVE_MXF4=1` to *all* jobs' defines (⚠3).

- [ ] **Step 7 — `build_cumetal.rs` (pure, testable: no process spawn outside `run_lane`).**
  `parse_manifests(kernels_root) -> LanePlan` (toml crate; unknown-key error; deep merge dir-over-default);
  `job_key(source, arch, flags)`; `compose_nvcc_flags(plan, dir, arch)`; `compose_cumetal_argv(prefix, shim,
  include_dirs, flags, source, out)` returning exactly the T3b-verified command (⚠1):
  `cumetalc --cuda-device --mode experimental <src> -o <out> --ptx-strict --emit metallib
  -I <kernels>/cumetal_compat/include -I <kernels>/cumetal_compat --cuda-include cumetal_half_shim.h
  -I $PREFIX/include` (`--emit metallib`: master plan line 50 flag list; `SHISU_CUMETAL_PREFIX` default
  `/opt/homebrew/opt/cumetal`). `run_lane()` shells out via `std::process::Command`, collects
  `(tu, Result)` verdicts for the caller's warning/strict handling.

- [ ] **Step 8 — `build_codegen.rs`.** Mirror atlas build_codegen.rs:24–45: one header comment line
  `// Auto-generated by build.rs — do not edit.` (the runbook fail-closed grep matches this literal), then per
  module `const <UPPER_STEM>_PTX: &[u8] = include_bytes!(concat!(env!("SHISU_KERNELS_OUT_DIR"),
  "/ptx/<dir>/<stem>__<arch>.ptx"));` (first/default arch wins the `ptx()` lookup; other arches stay on
  disk under OUT_DIR) and `const …_METALLIB` likewise; statics arrays + `match` bodies for
  `ptx/ptx_modules/metallib/metallib_modules`; skip-path stubs are the same generators with empty tables.
  Hash the generated string (FNV-1a 64 → 12 hex, atlas build.rs:639–649) →
  `cargo:rustc-env=SHISU_KERNEL_SET_HASH=`; also emit `SHISU_KERNELS_OUT_DIR` (atlas lib.rs:34–45 closes the
  untracked-`include!` staleness hole exactly this way).

- [ ] **Step 9 — `src/lib.rs`.** `#![deny(warnings)] #![deny(clippy::all)]` (T1 pattern); crate docs;
  `include!(concat!(env!("SHISU_KERNELS_OUT_DIR"), "/target_ptx.rs"));` + `…target_metallib.rs`;
  `pub const KERNEL_SET_HASH: &str = env!("SHISU_KERNEL_SET_HASH");`. Accessor discipline (produces
  block): unknown module ⇒ `debug_assert!(false, "unknown kernel module: {module}")` then `&[]`; release
  returns empty and T16's `cuModuleLoadData` fails closed on 0 bytes. No panics on the hot path, zero deps.

- [ ] **Step 10 — Linux runbook (no files; GitHub Actions out of scope).** On the Linux box
  (`nvidia/cuda:13.0.2-devel-ubuntu24.04` = the recommended environment), run the **real** build
  `SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels` (SHISU_SKIP_BUILD unset), then the fail-closed
  checks (atlas kernel-compile.yml 148–161 mechanism): ≥1 generated `target_ptx.rs` whose first line is
  the do-not-edit literal AND `find target -name '*.ptx' | wc -l` ≥ 8 (1 exl3 + 7 mmq — the
  near-empty-resolution guard), then `cargo test -p shisu-kernels -- --ignored`.

- [ ] **Step 11 — macOS verification (this Mac).**
  ```sh
  cd shisu
  SHISU_SKIP_BUILD=1 cargo check -p shisu-kernels        # stub path type-checks
  SHISU_SKIP_BUILD=1 cargo test  -p shisu-kernels        # the gate-green default test
  cargo test   -p shisu-kernels                          # no COMPILE=1 → stubs; logic tests still run
  SHISU_CUDA_IMPL=cumetal SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels
  ```
  Then record the 7 mmq TUs + the exl3 exclusion in `shisu/kernels/cumetal_compat/STATUS.md` at the
  reserved row (T3b plan line 244): one row per TU, status `ok`/`BLOCKED`, evidence = exact
  diagnostic; the exl3 row cites this plan's cooperative-launch grep. A BLOCKED mmq TU is a recorded
  diagnostic (hard-limits shared-memory row) — **do not** edit kernel arithmetic to unblock it (Rule 3).
  Also run the fake-`cumetalc` lane test (Tests) so the macOS run is reproducible without Xcode luck.

- [ ] **Step 12 — Linux gate (authoritative, NOT runnable on this Mac).** The nvcc lane's real compile of
  exl3+mmq is proven by the Step 10 runbook: `SHISU_CUDA_COMPILE=1
  SHISU_CUDA_IMPL=nvidia cargo build -p shisu-kernels` on the Linux box (master plan 251). Mark the
  corresponding `#[test]`s `#[ignore]` + `#[cfg(target_os = "linux")]`; do not attempt nvcc here.

## Tests

`shisu/crates/shisu-kernels/tests/build_logic.rs` — `#[path = "../build_cumetal.rs"] mod
build_cumetal;` + `#[path = "../build_codegen.rs"] mod build_codegen;` (atlas
`tests/kernel_shadow_detector.rs:24–25` pattern; build-script `#[cfg(test)]` modules never run).
Pure logic + tempdirs only ⇒ run **on this M1 Max without nvcc/Xcode**:

1. **Manifest parse/merge:** tempdir `kernels/` fixture (top + 2 dirs); defaults flow, dir keys win,
   unknown key ⇒ error; `[modules]` override applied; `sources`/`exclude` semantics.
2. **Dedup keying:** two dirs sharing a source+arch+flags triple ⇒ 1 compile job + 1 copy job; changing one
   `-D` splits the key. Assert the job table, not file names.
3. **Flag composition:** pinned argv equality for one nvidia job (Step 6 order) and for the cumetal argv
   (Step 7 form, incl. `--cuda-include` not `-include`, ⚠1); `SHISU_CUDA_ARCH=sm_120a` adds
   `-DSHISU_CUDA_HAVE_MXF4=1` (⚠3).
4. **Skip-stub path:** call the gated build body (factored `fn run(out_dir, env)`) with `SHISU_SKIP_BUILD=1`
   + tempdir OUT_DIR ⇒ both generated files exist, contain the empty-registry bodies, and `ptx("anything")`
   ⇒ empty via the generated lookup fn string (evaluate the codegen'd text for the literal, or expose the
   table builder).
5. **Fake cumetalc e2e:** tempdir `bin/cumetalc` shell stub (parses `-o <out>`, writes deterministic bytes;
   exit code from an env knob). Non-strict failing TU ⇒ lane continues, module absent, warning collected;
   `SHISU_CUMETAL_STRICT=1` ⇒ `run_lane` verdicts fail the test loudly. `excluded` TU never invoked
   (assert on the stub's call log).
6. **Codegen shape:** generated source starts with the do-not-edit literal; every module key appears in the
   match; hash changes iff content changes.
7. **Linux-only (ignored; run in the Step 10 runbook):** `#[ignore] #[cfg(linux)]` — with a real nvcc on
   PATH and `SHISU_CUDA_COMPILE=1`: 8 `.ptx` outputs non-empty and each contains a `.version`
   directive (the loadable-PTX sniff atlas asserts in its registry tests), no module silently skipped.

Run: `SHISU_SKIP_BUILD=1 cargo test -p shisu-kernels` (gate default path) and
`cargo test -p shisu-kernels` (same suite — no gate depends on a compiler being present).
Tolerances: none — this task runs no kernels; numerics are T15/T17's oracles.

## Acceptance

- [ ] Files exist: `shisu/crates/shisu-kernels/{Cargo.toml,build.rs,build_target.rs,build_codegen.rs,build_cumetal.rs,src/lib.rs}`,
      `shisu/kernels/KERNEL.toml`, `shisu/kernels/cuda/{exl3,mmq}/KERNEL.toml`,
      each `.rs` < 1500 LoC.
- [ ] `diff -r` vs `cuda/{exl3,mmq}` prints only the `KERNEL.toml` "Only in" lines (Rule 3).
- [ ] `SHISU_SKIP_BUILD=1 cargo build -p shisu-kernels` and `… cargo test -p shisu-kernels` green on macOS (master plan 251 first clause; the default path).
- [ ] `SHISU_CUDA_IMPL=cumetal SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels` compiles the TUs T3b cleared and appends per-TU rows to `kernels/cumetal_compat/STATUS.md` (exl3 row = cooperative-launch exclusion, evidenced); failures are diagnostics unless `SHISU_CUMETAL_STRICT=1` (master plan 249/251).
- [ ] Accessor signatures exactly `pub fn ptx(module: &str) -> &'static [u8]` / `pub fn metallib(module: &str) -> &'static [u8]` (master plan 121); unknown ⇒ empty, debug-only assertion, no release panic.
- [ ] All env knobs `SHISU_`-prefixed (rule 5); no `ATLAS_` string anywhere under `shisu/`.
- [ ] Real-nvcc build of exl3+mmq TUs is proven only by the Step 10 runbook on the Linux box (master plan 251 second clause — [INFERENCE] here: not executable on this Mac; T16's environment).
- [ ] `shisu/Cargo.toml` change is additive (members + one path dep); nothing else touched; no formatters/suites run (Task 20 owns the gate).

## Commit

Single commit (master plan "Commit" checkbox):

```sh
git add shisu/Cargo.toml shisu/crates/shisu-kernels shisu/kernels
git commit -m "feat: kernel build crate — KERNEL.toml registry, nvcc PTX + cumetal dev lane (Task 14)"
```
