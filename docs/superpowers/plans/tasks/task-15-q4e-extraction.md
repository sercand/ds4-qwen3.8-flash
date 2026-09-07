# Task 15: q4e kernel extraction (CUDA, no Rust)

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 15 (254–259)
+ § CuMetal dev lane (39–74, hard-limits table 57–70) + Roadmap Wave 3a/3b (309–317).
**Depends on:**
- T14 (`task-14-kernel-build-crate.md`): KERNEL.toml schema **SSOT** (its 100–145 — keys
  `[build] sources/std/base_flags/extra_nvcc_flags/defines/includes/arch/outputs`, `[cumetal]
  enabled/flags/extra_flags/include_dirs/shim/exclude`, `[modules]` stem map), the verified lane argv
  (⚠1: `--cuda-include`, never `-include`), gates `SHISU_SKIP_BUILD` ⇒ `SHISU_CUDA_COMPILE` ⇒
  `SHISU_CUDA_IMPL`, and the ⚠3 `SHISU_CUDA_HAVE_MXF4` rename obligation (⚠6: vacuous here).
- T3b: `shisu/kernels/cumetal_compat/{include/,cumetal_half_shim.h,STATUS.md}`; STATUS row format +
  the reserved row `(T14/15 append per-TU rows here)` (T3b plan:244); measured rows: compound
  static+dynamic smem **CORRUPTS** (T3b:239), dynamic-smem opt-in > 32,768 B **BLOCKED** (T3b:238),
  `__launch_bounds__`/shfl/fma f32 ok (T3b:232–237), D ≤ 64 oracle caveat.
- T2: `shisu-core/src/q4e_page.rs` consts (task-02 plan 149–157; C names kept) → the `q4e_page.cuh`
  mirror. T5: MSL twin set (kernel_q35_gdn_*, q35_attn_*, q35_*_norm_rope) = kernels with a numeric
  oracle. T16 is the consumer (`cuModuleGetFunction` by name — Appendix A is its lookup SSOT).
**Produces** (paths under `shisu/`, repo root = cwd):
- `kernels/cuda/q4e/shisu_dev_inc/{common.cuh,tensor.cuh}`, `kernels/cuda/q4e/q4e.cuh`,
  `kernels/cuda/q4e/q4e_page.cuh` (generated mirror, Step 4), and 9 TUs —
  `{hc,moe,gdn,ple,matvec,qsa,qsa_sparse,idx,misc}.cu`, the 50 device kernels of
  `ds4_qwen4exp_gpu.cuh` byte-for-byte (⚠1: qsa split along the source's dense/sparse seam — a design choice under the 1500 cap, not cap-forced; Rule 2).
- `kernels/cuda/q4e/KERNEL.toml` (T14 schema; module keys `q4e_hc q4e_moe q4e_gdn q4e_ple
  q4e_matvec q4e_qsa q4e_qsa_sparse q4e_idx q4e_misc` — ⚠4).
- `kernels/cuda/q4e/KERNEL_NAMES.md` — entry-name inventory (Appendix A seeded; mangled
  template names captured by the Linux oracle run). T16 resolves every kernel from it.
- `shisu/scripts/q4e_ptx_oracle.sh` (Linux-authoritative split-fidelity gate).
- Per-TU rows appended to `kernels/cumetal_compat/STATUS.md` at T3b's reserved row.

## Global Constraints

Rules 1–7, one line each: (1) new files only under `shisu/`; `.cu/.cuh` kernel files are the allowed
kinds; no host C/C++ added — the `ds4_gpu_q4e_*` host entry points are NOT carried (owners T16/T17).
(2) 1500-LoC cap → qsa pre-split (⚠1); budgets: gdn ≈ 485, qsa ≈ 360, qsa_sparse ≈ 430, moe ≈ 300,
matvec ≈ 210, idx ≈ 385, hc ≈ 185, ple ≈ 140, misc ≈ 35. (3) kernel bodies **byte-for-byte**; the ONLY
deltas: `#include` plumbing, `extern "C"` linkage tokens, explicit template instantiation directives
(⚠2); arithmetic-order changes are bugs. (4) CuMetal numbers never gate; lane records compile verdicts
+ numerics only. (5) no new env knobs (T14's `SHISU_*` set suffices). (6) no Rust in this task; scripts
are shell. (7) no license headers. Binding CuMetal hard-limit rows (57–70): shared memory (compound
layouts), atomics (every numeric is a diff, never "it ran"), memory (oracle allocs via cudaMalloc),
device properties (never branch on them), timing (none recorded), NVRTC (Mac builds stay
`SHISU_SKIP_BUILD=1`); the lane is opt-in and never gates a default build (249).

## Source References (verified)

All rows opened this session; ds4 root = cwd.

| Source | Lines | What / how used |
|---|---|---|
| `ds4_qwen4exp_gpu.cuh` | 1–18 | banner: included by `ds4_cuda.cu` (:33408) "so these kernels can use its device helpers"; its `#include "ds4_q4e_page.h"` + `"ds4_ple_stream.h"` serve host launchers only — census: `DS4_PLE_*` occur only at 1992–1993 (host), `DS4_N_HC/DS4_PLE_LAYER/…` only in comments. |
| ″ | 23–53, 225–260, 2060–2120, 2155–2161 | cross-family `__device__` helpers `q4e_block_sum`, `q4e_block_max`, `q4e_block_argmax`, `q4e_rope_neox`, `q4e_rope_mrope`, `q4e_kv_row` → `q4e.cuh` (call-site census: block_sum ×15 across 6 families; block_argmax 296,2687; rope_mrope 2143,2187,2798,2842; kv_row ×8 qsa+idx). |
| ″ | 61,450,604–606,1281,1393,2214–2215,2304–2306,2444–2447,2727–2729,2868–2869,3072,3430 | `Q4E_*` file-scope defines. Device-side carried with the consuming family; **host-only, dropped**: 1281 `Q4E_F16_PARTIAL_FLOATS` (used 1292), 1393 `Q4E_EXL3_RECONSTRUCT_ROWS` (used 1416,1502), 3430 `Q4E_MOE_BATCH_MIN_TOK` (used 3728,3851). |
| ″ | family regions | kernel inventory = Step table + Appendix A (50 distinct `__global__` kernels; the 51st hit is the forward decl 1273–1274 of `q4e_matvec_f32_combine_kernel`, dropped with the host launchers). |
| ″ | 1078–1141, 1176–1228, 1283–1351, 3687–4000 | host launchers `q4e_*_launch<NT>`/`extern "C" ds4_gpu_q4e_*` — NOT extracted; sole evidence for template arg sets (q8_0 narrow+warp NT 2..16 @1125–1139; bf16 NT 2..16 @1200–1214; f16 NT 1..16 @1336–1351; down_grouped `<7,20>`/`<8,20>` @3768,3774; down `<7>`/`<8>` @3787,3792) and grid/block/smem formulas = **T16's LaunchArgs contract** (record in KERNEL_NAMES.md rows). |
| ″ | 1903,1953,2934,3249,3338,3372 | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize,…)` sites (chunk, recurrent, idx_score, gather, tiled, split) — host-side opt-ins; T16 replays them; lane smem rows cite them. |
| `ds4_cuda.cu` | 1–6, 33408 | include set to mirror: `<cuda_runtime.h> <cuda_fp16.h> <cuda_bf16.h>`; the inclusion point of the `.cuh`. |
| `ds4_cuda.cu` | 5374–5386, 5392–5402 | `warp_sum_f32`, `warp_max_f32`, `load_i8x4_i32_aligned`, `load_i8x4_i32_unaligned` → `shisu_dev_inc/common.cuh`. **Complete** set of device deps the `.cuh` borrows from its including TU (verified: intersection of `^__device__` defs in ds4_cuda.cu with identifiers used in the `.cuh` = exactly these 4; `dot4_f32`/`dot_i8x32_dp4a` unused there). |
| `ds4_cuda.cu` | 1000,1032 (via .cuh), 1160 | device intrinsics beyond the 4 helpers: `__dp4a` (matvec), `__bfloat1622float2` (matvec bf16 kernel — shim-covered name, T3b list), `__half2float`/`__float2half_rn` (declared by `cuda_fp16.h`), `uint4` + `__align__(16)` extern smem (qsa_sparse/idx), `sincosf/powf/log1pf/__expf/rsqrtf/fmaxf` math — lane-risk row candidates (Step 14). |
| `ds4_gpu.h` | 20–23 | `typedef struct ds4_gpu_tensor ds4_gpu_tensor;` → `tensor.cuh` (⚠3: zero device uses; all 155 refs are host-launcher code). |
| `ds4_q4e_page.h` | 33–45 | the 4 macros `DS4_Q4E_PAGE_TOKENS 256u / SHIFT 8u / MASK (TOKENS-1u) / MROPE_BACK 4u` → generated `q4e_page.cuh`. |
| `Makefile` | 98–99, 460–461 | `NVCC_BASE_FLAGS := -O3 -g -lineinfo --use_fast_math …`; monolith recipe `ds4_cuda.o: $(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu` = the baseline's build flags. |
| ″ | 60–63 | `-DDS4_CUDA_HAVE_MXF4=1` sites are host-side arch pins; `.cuh` contains **zero** preprocessor conditionals (verified) ⇒ T14 ⚠3 needs no device-side rename (⚠6). |
| atlas `crates/atlas-kernels/build_shadow.rs` + `tests/kernel_shadow_detector.rs` | 106–108; 116–117 | the `extern "C" __global__ void name(...)` registry convention T16's `cuModuleGetFunction` lookup presumes ⇒ basis of delta class D1 (⚠2). |
| T14 plan | 100–145, 168–200 | manifest keys verbatim; skip/compile/impl gate order; codegen + `OUT_DIR` (`$SHISU_KERNELS_OUT_DIR`) conventions. |
| T3b plan | 66–82, 221–247 | measured lane constructs; STATUS.md template + reserved row; compound-smem and smem-cap verdicts. |
| T2 plan | 149–157 | `q4e_page.rs` Rust const names/values mirroring the header 1:1. |

⚠ **DEVIATION 1 (9th TU — design split, not cap-forced):** the master Files list names 8 `.cu` files, but the
qsa family's 8 kernel spans alone are 730 lines (34+51+87+131+204+24+193+6) + defines/boilerplate
≈ 770 — under the 1500 cap one `qsa.cu` would fit; split anyway along the pipeline seam the source itself draws (dense vs sparse):
`qsa.cu` = q_norm_rope, store_kv, attention, attention_tiled, gate; `qsa_sparse.cu` =
attention_split, attention_gather, attention_combine (+ their `Q4E_ATTN_QG/SPLITS/SPLIT_HD/
PART_STRIDE/GSPLITS` defines). Family taxonomy stays 8; TU count 9.

⚠ **DEVIATION 2 (linkage delta class, Rule 3 "launch wrappers"):** add `extern "C"` to the 44
non-template `__global__` declarations and drop their `static` (bodies untouched, byte-equal);
atlas precedent above; without it every PTX entry name is Itanium-mangled and T16's by-name
lookup is brittle. The 6 template kernels keep `template <int …> __global__ static void`
verbatim + gain explicit instantiation directives (D2) for exactly the launcher-used argument
sets; `extern "C"` is illegal on templates, so their mangled entry names are recorded in
KERNEL_NAMES.md. Fidelity guard: the PTX body-hash gate (Step 12) proves no byte of any body moved.

⚠ **DEVIATION 3 (`tensor.cuh` is a compatibility header):** master:255 says "extract …
`ds4_gpu_tensor` accessors from ds4_cuda.cu"; verified there are **no device-side accessors** —
all 155 `ds4_gpu_tensor` and 9 `ds4_tensor_device_idx` uses sit in host-launcher ranges
(≥1078), every kernel signature is raw pointers (agrees with T3b ⚠3). The file is still created
(master Files SSOT) holding the opaque typedef from `ds4_gpu.h:20–23` + a doc comment; zero
device code, provably inert in every TU that includes it.

⚠ **DEVIATION 4 (one manifest, not eight):** T14's schema is *directory*-granular
(T14:100–127); `kernels/cuda/q4e/KERNEL.toml` carries all 9 sources and per-family granularity
lives in the `[modules]` map — master:255 "KERNEL.toml per family" is honoured one row per
family. (Eight directories would fall outside T14's `kernels/cuda/<dir>/KERNEL.toml` scan.)

⚠ **DEVIATION 5 (`q4e_page.cuh` ownership):** master:124 has build.rs emit the `.cuh` mirror;
T14's codegen (Step 8) emits only `target_ptx.rs`/`target_metallib.rs`. T15 checks in the
generated mirror (Step 4) so raw nvcc/cumetalc gates are self-contained; T14's codegen should
later regenerate it into `$SHISU_KERNELS_OUT_DIR` with `-I` precedence winning (flag for T14/T20;
until then the checked-in mirror is authoritative and its 4 values are pinned against
`shisu_core::q4e_page`, T2:152–157).

⚠ **DEVIATION 6 (T14 ⚠3 MXF4 rename — vacuous):** the `.cuh` has zero `#if/#ifdef` (verified),
so no device-side branch on `*_CUDA_HAVE_MXF4` exists to rename; the flag rides in
`defines = []` harmlessly when `$SHISU_CUDA_ARCH` names sm_120a/121a.

## Plan

**Step table — the split map** (`source` = `ds4_qwen4exp_gpu.cuh`; device spans only; host
launchers at 970–1077, 1176–1228, 1283–1443, 1460–1575, 1599–2059, 2691–2726, 3272–3480,
3591–3645, 3687–4000 stay in ds4):

| target `.cu` | kernel start lines | #K | dynamic smem (`extern __shared__`) | lane numeric class |
|---|---|---|---|---|
| `hc.cu` | 65,77,91,114,122,167 (+`Q4E_HC_FUSE_MAX` 61) | 6 | — (helper statics only) | ok |
| `moe.cu` | 261,321,331,348,1445,3510,3551,3559,3576,3646 (+`q4e_down_weight` 3481) | 10 | route 266+**statics 284–285 → compound** | route NUMERIC-INVALID; rest ok |
| `gdn.cu` | 373,405,421,451,608,813 (+defines 450,604–606) | 6 | recurrent 457+**statics 480–482 → compound**; chunk 618 pure | recurrent NUMERIC-INVALID (+66,048 B@D128 SMEM_BLOCKED); chunk ok (99,072 B@prod SMEM_BLOCKED, D≤64 runnable) |
| `ple.cu` | 839,867,905,928 | 4 | — (dequant_exl3 static 874) | ok |
| `matvec.cu` | 988,1039,1057,1149,1229,1308,1576,1590 (+`q4e_q8_0_rows_block` 1018) | 8 | — | ok |
| `qsa.cu` | 2121,2163,2217,2313,3266 (+defines 2214–2215,2304–2306) | 5 | q_norm 2131+helper statics → **compound**; store_kv 2173+helper → **compound**; attention 2267+**2269–2270** → **compound**; tiled 2326 pure | those 3 NUMERIC-INVALID; tiled ok (smem 2·hd·KT·2+QT·hd·4+8·KT·4) |
| `qsa_sparse.cu` | 2449,3073,2653 (+defines 2444–2447,3072) | 3 | split 2473 `__align__(16) uint4` pure; gather 3093+**3098 → compound** | gather NUMERIC-INVALID; split ok |
| `idx.cu` | 2732,2760,2826,2875,2951,3042 (+defines 2727–2729,2868–2869) | 6 | score 2886 pure; topk statics 2957–2960; pool 2774, q 2832 static | ok |
| `misc.cu` | 960 (add2), 2677 (argmax_rows) | 2 | — | ok |

- [ ] **Step 1 — Preflight (read-only).** `shisu/kernels/cumetal_compat/STATUS.md` +
  `FETCH.sh` exist (T3b landed; else STOP — lane gated on it). `shisu/kernels/cuda/q4e/`
  absent. Re-derive the census: `grep -c '__global__' ds4_qwen4exp_gpu.cuh` → 51 (50 defs +
  1 forward decl at 1273); `grep -c '#if' ds4_qwen4exp_gpu.cuh` → 0 (⚠6); the 4-helper
  intersection (Source References row) recomputes to {warp_sum_f32, warp_max_f32,
  load_i8x4_i32_aligned, load_i8x4_i32_unaligned}.
- [ ] **Step 2 — `shisu_dev_inc/common.cuh`.** Include guard; `#include <cuda_fp16.h>`,
  `<cuda_bf16.h>`, `<math.h>`, `<stdint.h>` (mirrors ds4_cuda.cu:1–3 + `INFINITY` from
  math.h — used at .cuh:47 etc.); then the four helpers **verbatim** from
  `ds4_cuda.cu:5374–5386` and `5392–5402`. Provenance comment: source file + line ranges +
  "byte-for-byte; do not edit arithmetic".
- [ ] **Step 3 — `shisu_dev_inc/tensor.cuh`.** Guard + `typedef struct ds4_gpu_tensor
  ds4_gpu_tensor;` (`ds4_gpu.h:20–23`, keep the `DS4_GPU_TENSOR_DEFINED` guard spelling so a
  later shared include cannot re-typedef) + doc comment recording the zero-device-use census
  (⚠3). No functions, no device code.
- [ ] **Step 4 — `q4e_page.cuh` (generated mirror, ⚠5).** Content = guard + exactly:
  ```c
  #define DS4_Q4E_PAGE_TOKENS 256u
  #define DS4_Q4E_PAGE_SHIFT   8u
  #define DS4_Q4E_PAGE_MASK   (DS4_Q4E_PAGE_TOKENS - 1u)
  #define DS4_Q4E_MROPE_BACK   4u
  ```
  Header comment: "GENERATED mirror of `shisu_core::q4e_page` (task-02:152–157) ← original
  `ds4_q4e_page.h:33–45`; regenerate via T14 codegen; values must stay byte-equal." Consumers:
  `q4e.cuh` (kv_row) and the `static_assert`s at 2310, 2758, 2873 (they travel inside their
  kernels — verbatim).
- [ ] **Step 5 — `q4e.cuh`.** Guard; includes `common.cuh`, `tensor.cuh`, `q4e_page.cuh`;
  then verbatim: `q4e_block_sum` 23–37, `q4e_block_max` 39–53, `q4e_block_argmax` 225–260,
  `q4e_rope_neox` 2060–2095, `q4e_rope_mrope` 2096–2120, `q4e_kv_row` 2155–2161 (with its
  `__device__ __forceinline__`). These are cross-family (census in Source References); keep
  file order as in the `.cuh`. Provenance comment as Step 2.
- [ ] **Step 6 — Extraction protocol (applies to Steps 7–10).** For each family: file header
  comment (`extracted verbatim from ds4_qwen4exp_gpu.cuh lines X–Y, …; deltas: D1 extern "C",
  D2 instantiations — nothing else`), then `#include "q4e.cuh"`; then the listed kernel spans
  in source order, each span from its `__global__` (or `template` line) through its closing
  `}` — copy mechanically (sed by line range), never retype. Allowed text outside spans:
  the kernel's own preceding block comment (it documents arithmetic intent — carry it).
  D1: `extern "C" __global__ void q4e_…` on the 44 plain kernels. NO other edit; a span that
  needs any other change means the split is wrong — stop and re-extract.
- [ ] **Step 7 — Simple families.** `hc.cu`, `ple.cu`, `idx.cu`, `misc.cu` per the table
  (no templates). `misc.cu` = add2 (elementwise plumbing) + argmax_rows (sampling helper);
  neither fits an architecture family; both stay one file (≈35 LoC).
- [ ] **Step 8 — `gdn.cu`.** Table spans + defines 450, 604–606. Do **not** fold the
  recurrent static smem (480–482) into the dynamic region the way T3b's *harness* did — that
  fold is an oracle-harness adaptation; here the kernel is the ds4 original and the compound
  layout is recorded as a STATUS annotation instead (Step 14). Host helpers
  `q4e_gdn_chunk_smem_bytes/min_tok/ready` (1876–1899) are launch policy: not extracted;
  their formulas go to KERNEL_NAMES.md rows for T16.
- [ ] **Step 9 — `matvec.cu` / `moe.cu` (templates).** Extract spans per table; append the D2
  instantiation directives (one line per concrete entry, exact signatures from the definitions):
  `q4e_matmul_q8_0_rows_narrow_kernel<2..16>`, `q4e_matmul_q8_0_rows_warp_kernel<2..16>`,
  `q4e_matmul_bf16_rows_kernel<2..16>`, `q4e_matmul_f16_rows_kernel<1..16>`,
  `q4e_moe_down_grouped_kernel<7,20>`, `<8,20>`, `q4e_moe_down_kernel<7>`, `<8>`
  (`template __global__ void q4e_matmul_bf16_rows_kernel<2>(float *, const __nv_bfloat16 *,
  const float *, uint32_t, uint32_t);` — the documented explicit-instantiation form; arg types
  verbatim from each definition). The forward decl 1273–1274 is dropped (its only consumer was
  the dropped host launcher at 1302). If the nvcc probe rejects the directive form, fall back to
  a never-called `__attribute__((unused)) static` host referrer that `<<<>>>`-launches each
  instantiation once (launch-wrapper class); record the choice in the file header.
- [ ] **Step 10 — `qsa.cu` / `qsa_sparse.cu`.** ⚠1 split; carries the `static_assert`s in
  their kernels (2310 tiled; the 2758/2873 asserts travel with idx; 2473's `__align__(16)`
  extern stays with split). Do not touch `__launch_bounds__` qualifiers.
- [ ] **Step 11 — `KERNEL.toml`** (keys verbatim from T14:100–127):
  ```toml
  # shisu/kernels/cuda/q4e/KERNEL.toml
  [build]
  sources = ["hc.cu","moe.cu","gdn.cu","ple.cu","matvec.cu","qsa.cu","qsa_sparse.cu","idx.cu","misc.cu"]
  includes = ["cuda/q4e"]              # q4e.cuh / shisu_dev_inc/* / q4e_page.cuh resolve here (-I kernels/cuda/q4e)
  defines = []                          # SHISU_CUDA_HAVE_MXF4 arrives via T14 build_target on sm_120a/121a (⚠6: unused here)
  arch = ["compute_75"]                 # lane default (master:248)
  outputs = ["ptx"]
  [cumetal]
  exclude = []                          # evidence rule below: exclude ONLY on a hard diagnostic recorded in STATUS.md
  [modules]
  hc = "q4e_hc"
  moe = "q4e_moe"
  gdn = "q4e_gdn"
  ple = "q4e_ple"
  matvec = "q4e_matvec"
  qsa = "q4e_qsa"
  qsa_sparse = "q4e_qsa_sparse"
  idx = "q4e_idx"
  misc = "q4e_misc"
  ```
  `[build].std/base_flags` and `[cumetal].enabled/flags/include_dirs/shim` inherit the T14
  top-level defaults (`-std=c++17`; `-O3 -g -lineinfo --use_fast_math` Makefile:98; T3b argv
  + `--cuda-include cumetal_half_shim.h`). **Exclude evidence rule:** all 9 start enabled —
  every construct the TUs hit is either already measured ok by T3b (dynamic extern smem,
  `__syncthreads` uniform, both shfl butterflies, `__launch_bounds__`, f32 fma, bf16 via shim)
  or an unmeasured-but-declared construct (`__dp4a`, `uint4` aligned extern, `sincosf/powf/
  log1pf`, 1024-thread bounds, `__forceinline__`); T3b measured zero compile-time blockers, so a
  preemptive exclude would be a guess. A hard rejection at Step 14 is the only warrant to move a
  stem to `exclude`, with the diagnostic quoted in the same STATUS.md row.
- [ ] **Step 12 — `scripts/q4e_ptx_oracle.sh` (Linux-authoritative; macOS-inert guard at
  top).** Exact argv (CUDA_HOME per T14's `find_cuda_dir`):
  ```sh
  FLAGS="-O3 -g -lineinfo --use_fast_math -std=c++17 -arch=compute_75"
  STRICT="-Xcompiler=-Wall -Xcompiler=-Werror"                     # gate: zero diagnostics
  INCS="-I shisu/kernels/cuda/q4e -I shisu/kernels/cuda/q4e/shisu_dev_inc"
  # (a) each TU compiles clean:      $NVCC $FLAGS $STRICT $INCS --ptx -o $OUT/<t>.ptx <t>.cu
  # (b) monolith baseline, same way: $NVCC $FLAGS --ptx -I. -o $OUT/ds4_cuda.ptx ds4_cuda.cu   # Makefile:460-461 flags
  # (c) ELF symbol sets:             $NVCC $FLAGS -arch=sm_75 -cubin -o $OUT/<t>.cubin <t>.cu ; cuobjdump --dump-elf-symbols …
  #     entry set = { q4e_* from the 9 cubins } ∪ mangled template entries
  # (d) oracle: compare (demangle via c++filt, strip '(', i.e. base+template-args) — split set
  #     MUST equal the monolith set filtered to the Appendix-A names; AND per-entry PTX body hash
  #     (sha256 of the .entry block between its first '{' and matching '}') MUST equal the
  #     monolith's for the same demangled name. Any diff names the kernel; fixing = re-extract
  #     from ds4, NEVER hand-patch arithmetic (Rule 3).
  # (e) emit $OUT/KERNEL_NAMES.md rows: file | entry (plain or mangled) | module | demangled
  ```
  `__global__ static` on the monolith side and `extern "C"` on the split side make raw names
  differ (⚠2); the comparison is defined on **demangled base names** exactly once per side, and
  template entries must match byte-equal mangled (identical signatures both sides).
- [ ] **Step 13 — Kernel-fidelity diff (macOS, no compiler needed — the Rule 3 proof).**
  For every Appendix-A entry: `diff <(sed -n 'A,Bp' ds4_qwen4exp_gpu.cuh) <(block-extract .cu)`
  after stripping only the D1/D2 delta tokens from the split side (`sed 's/extern "C" __global__/__global__/; s/template __global__/template … __global__/'`… normalize by removing the
  added tokens, never by editing either side); empty diff required for all 50 bodies + the
  6 device helpers; script it next to the oracle (`--mode=fidelity`). Count gate:
  `grep -c '__global__'` per file == table (#K column). This runs today on this Mac and is the
  gate that lets the Linux box re-run already-sourced code.
- [ ] **Step 14 — CuMetal lane (macOS, opt-in; Wave 3a).**
  `cd shisu && SHISU_CUDA_IMPL=cumetal SHISU_CUDA_COMPILE=1 cargo build -p shisu-kernels`
  (T14 lane ⇒ T14:argv; raw single-TU probe mirrors it: `cumetalc --cuda-device --mode
  experimental <t>.cu -o target/cumetal/<t> --ptx-strict -I kernels/cumetal_compat/include
  -I kernels/cumetal_compat --cuda-include cumetal_half_shim.h -I "$SHISU_CUMETAL_PREFIX"
  /include -I kernels/cuda/q4e`). `SHISU_CUMETAL_STRICT=1` for the task gate. Append one row
  per TU to STATUS.md's reserved row (T3b format): `| q4e/<t>.cu | ok|BLOCKED | first
  diagnostic verbatim |`; add construct rows for first-time hits (`__dp4a`, aligned `uint4`
  extern smem, `sincosf/powf/log1pf`, `__launch_bounds__(1024)`, `__forceinline__`); add the
  per-kernel annotations `NUMERIC-INVALID (compound static+dynamic smem — T3b:239)` to the 6
  kernels of the table (route, recurrent, q_norm_rope, store_kv, attention, gather) and
  `SMEM_BLOCKED` where production shapes exceed 32,768 B (recurrent@128, chunk). A BLOCKED row
  is a recorded diagnostic; **do not** rewrite arithmetic or fold smem to unblock (Rule 3).
- [ ] **Step 15 — Numeric-diff applicability (no new runs).** The 1e-5 diff is owed only by kernels
  that (i) have a T5 MSL twin, (ii) sit in a TU whose row is `ok`, and (iii) are not
  NUMERIC-INVALID: that is {gdn conv, l2norm, gates, out_gate} and nothing else runnable
  (recurrent/chunk stay SMEM-/compound-gated at production shapes, D ≤ 64 caveat T3b; qsa twins sit
  in compound TUs). Where `tests/cumetal/oracle_gdn` already covers a kernel (recurrent), the
  carry-over holds — harness body ≡ split body modulo T3b's storage fold, so its 3.8e-6 PASS stands;
  note it in STATUS.md. New harnesses (chunk, conv, attn…) are T5/T17 obligations; Linux-side
  numerics stay T17/T18 — never claimed from this Mac.
- [ ] **Step 16 — Wrap.** Populate KERNEL_NAMES.md from Step 12(e) *structure* now (plain
  names + module + grid/block/smem formulas lifted verbatim from the launcher lines per the
  Source-References row); the mangled column is filled by the Linux run and committed with it.

## Tests

- **Fidelity (macOS, no nvcc):** Step 13 diff recipe — all 56 blocks (50 kernels + 6 shared helpers)
  empty-diff; count gate per file; `grep -c '#if' == 0` preserved across the 9 TUs; `q4e_page.cuh`
  values `grep`-equal `ds4_q4e_page.h:33–45` and T2 consts.
- **Lane compile (macOS, opt-in):** Step 14 command exits 0 and STATUS.md gains 9 rows with evidence;
  under `SHISU_CUMETAL_STRICT=1` warnings fail the row. SKIP path (`SHISU_SKIP_BUILD=1`) exits 0 < 1 s.
- **Split-fidelity oracle (Linux, authoritative — [INFERENCE] until the box runs):** Step 12 (a)–(d)
  green: 9 PTXs warning-clean; entry-set equality vs the monolith (demangled base names; template
  mangled byte-equal); body-hash equality. Add to T14's Linux runbook (its Step 10 checks) as an extra
  gate, raising the "≥ 8 .ptx" floor to "≥ 17" (8 exl3+mmq + 9 q4e).
- **Numeric (macOS):** only Step 15's eligible set via `oracle_gdn` at 1e-5; disagreements root-cause to a line, never widen (constraint 3). No throughput recorded.
- No cargo tests in T15 (no Rust); no formatters/linters (Task 20 owns the gates).

## Acceptance

- [ ] `shisu/kernels/cuda/q4e/{shisu_dev_inc/{common,tensor}.cuh,q4e.cuh,q4e_page.cuh,hc,moe,gdn,ple,
      matvec,qsa,qsa_sparse,idx,misc}.cu,KERNEL.toml,KERNEL_NAMES.md}` + `shisu/scripts/q4e_ptx_oracle.sh`
      exist; every file < 1500 LoC.
- [ ] Fidelity diff (Step 13) empty for all 56 blocks; per-file `__global__` counts == table
      (6+10+6+4+8+5+3+6+2 = 50).
- [ ] `SHISU_SKIP_BUILD=1 cargo build -p shisu-kernels` still green (T14 untouched except the
      documented follow-up ⚠5, which is *not* done here).
- [ ] Lane run (Step 14) appended 9 STATUS.md rows with exact commands + first diagnostics;
      no kernel text changed to make a row pass; compound-smem/SMEM_BLOCKED annotations present
      (6 kernels named, per table).
- [ ] `KERNEL.toml` parses under T14's schema (keys ⊆ the T14 key set; unknown-key ⇒ build
      error); module keys `q4e_*` unique across the registry (collide-free with exl3/mmq).
- [ ] Linux gate claims appear only as the Linux box's documented runbook result (Wave 3b); nothing in this
      task's merge notes asserts a Linux result that was not executed.
- [ ] `diff -r cuda/exl3 cuda/mmq shisu/kernels/cuda/{exl3,mmq}` (T14's copies) unaffected;
      nothing outside `shisu/kernels/cuda/q4e/`, `shisu/scripts/q4e_ptx_oracle.sh`, and the
      STATUS.md append touched.

## Commit

```sh
git add shisu/kernels/cuda/q4e shisu/scripts/q4e_ptx_oracle.sh shisu/kernels/cumetal_compat/STATUS.md
git commit -m "feat(kernels): q4e device-code extraction — 9 family TUs, dev includes, page-cuh mirror, PTX split oracle (Task 15)"
```

## Appendix A — entry-name inventory (50 kernels; T16 lookup SSOT)

Plain entries are the literal names (extern "C"); ⊤ = template, instantiated per the listed
argument sets (mangled names captured by Step 12(e)). Format `file: kernel (start-line)`.

- **hc.cu:** q4e_hc_init_kernel(65) · q4e_hc_init_add_kernel(77) · q4e_hc_norm_kernel(91) ·
  q4e_scale_silu_kernel(114) · q4e_hc_collapse_kernel(122) · q4e_hc_combine_norm_kernel(167)
- **moe.cu:** q4e_moe_route_kernel(261) · q4e_swiglu_kernel(321) · q4e_moe_combine_kernel(331) ·
  q4e_shared_add_kernel(348) · q4e_moe_rows_per_slot_kernel(1445) ·
  q4e_moe_down_grouped_kernel⊤<7,20>,<8,20>(3510) · q4e_moe_sort_count_kernel(3551) ·
  q4e_moe_sort_scan_kernel(3559) · q4e_moe_sort_scatter_kernel(3576) · q4e_moe_down_kernel⊤<7>,<8>(3646)
- **gdn.cu:** q4e_gdn_conv_kernel(373) · q4e_gdn_l2norm_kernel(405) · q4e_gdn_gates_kernel(421) ·
  q4e_gdn_recurrent_kernel(451) · q4e_gdn_chunk_kernel(608) · q4e_gdn_out_gate_kernel(813)
- **ple.cu:** q4e_ple_dequant_kernel(839) · q4e_ple_dequant_exl3_kernel(867) ·
  q4e_ple_gated_value_kernel(905) · q4e_ple_conv_kernel(928)
- **matvec.cu:** q4e_matvec_q8_0_narrow_kernel(988) · q4e_matmul_q8_0_rows_narrow_kernel⊤2..16(1039) ·
  q4e_matmul_q8_0_rows_warp_kernel⊤2..16(1057) · q4e_matmul_bf16_rows_kernel⊤2..16(1149) ·
  q4e_matmul_f16_rows_kernel⊤1..16(1229) · q4e_f32_to_f16_kernel(1308) ·
  q4e_matvec_f32_kernel(1576) · q4e_matvec_f32_combine_kernel(1590)
- **qsa.cu:** q4e_qsa_q_norm_rope_kernel(2121) · q4e_qsa_store_kv_kernel(2163) ·
  q4e_qsa_attention_kernel(2217) · q4e_qsa_attention_tiled_kernel(2313) · q4e_qsa_gate_kernel(3266)
- **qsa_sparse.cu:** q4e_qsa_attention_split_kernel(2449) · q4e_qsa_attention_gather_kernel(3073) ·
  q4e_qsa_attention_combine_kernel(2653)
- **idx.cu:** q4e_idx_store_k_kernel(2732) · q4e_idx_pool_kernel(2760) · q4e_idx_q_kernel(2826) ·
  q4e_idx_score_kernel(2875) · q4e_idx_topk_kernel(2951) · q4e_idx_expand_kernel(3042)
- **misc.cu:** q4e_add2_kernel(960) · q4e_argmax_rows_kernel(2677)

Shared device helpers (in `q4e.cuh`, compiled into every TU): q4e_block_sum(23) ·
q4e_block_max(39) · q4e_block_argmax(225) · q4e_rope_neox(2060) · q4e_rope_mrope(2096) ·
q4e_kv_row(2155). Family-local helpers stay in their TU: q4e_q8_0_rows_block(1018, matvec) ·
q4e_down_weight(3481, moe).
