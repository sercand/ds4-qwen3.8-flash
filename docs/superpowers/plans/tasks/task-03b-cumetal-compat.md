# Task 3b: CuMetal compat bundle + GDN oracle spike

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 3b **and** the
whole § CuMetal dev lane (that section is this task's requirements document).
**Depends on:** Task 1 only, for layout conventions (`shisu/` root, `SHISU_*` env
names). No interface consumed from T2/T3; runs in parallel with T3/T4 (Wave 1:
`T2 ──► T3b`, independent branch). Needs no model file — synthetic shapes only.
**Produces** (later tasks read this block; paths relative to repo root):

- `shisu/kernels/cumetal_compat/include/` — 14 vendored header files (the 8
  missing entries) from CuMetal commit `cebf68a403601d59e8f67cdf5d85973c50fc0732`
  (annotated tag `v0.4.0`), each with an Apache-2.0 provenance header comment.
- `shisu/kernels/cumetal_compat/cumetal_half_shim.h` — force-include header
  defining exactly the 8 half/bf16 helpers CuMetal 0.4.0 lacks (names below).
- `shisu/kernels/cumetal_compat/STATUS.md` — **SSOT** verdict table: which
  constructs compile/run under `cumetalc` and which block the rest. T5/T14/T15/T17/T20 read it.
- `shisu/tests/cumetal/oracle_gdn.cu` → binary contract
  `oracle_gdn <kernel> <in.bin> <out.bin> <head_dim> <n_head_k> <n_head_v> <n_tok>`
  with `<kernel>` = `gdn_recurrent` (name reserved for future `gdn_chunk`).
- `shisu/tests/cumetal/gdn_ref.rs` → `rustc -O` standalone tool:
  `gdn_ref gen <in.bin> <shape…>` | `ref <in.bin> <ref.bin> <shape…>` | `cmp <a> <b> <tol>`.
- `shisu/tests/cumetal/shim_selftest.cu` → bit-exact device test of the 8 shim helpers.
- `shisu/scripts/cumetal_probe.sh` — single entry point (compile + run + diff);
  honors `SHISU_SKIP_BUILD` / `SHISU_CUDA_COMPILE` / `SHISU_CUMETAL_STRICT` / `SHISU_CUMETAL_PREFIX`.
- The **verdict**: whether T5/T17 gain the CuMetal differential oracle (gate below).

## Global Constraints

Master plan Rules 1–7, one line each:

1. All new code under `shisu/`; no C/C++/ObjC host code — **this task is the
   sanctioned exception**: `tests/cumetal/*.cu` are dev/test tools compiled by
   `cumetalc` into standalone executables, never under `crates/`, never linked
   into any product crate, never shipped (Scope "Out" forbids CuMetal in any
   release artifact). Nothing here adds a Rust dependency.
2. 1500-LoC cap (local gate `shisu/scripts/file_size_check.sh` scans `crates/**/*.rs` only, but keep every file small:
   `oracle_gdn.cu` ≈ 400 LoC budget — if over, split host file-I/O into
   `tests/cumetal/oracle_io.h`; `gdn_ref.rs` ≈ 220; `cumetal_probe.sh` ≈ 100).
3. Kernel copied byte-for-byte; arithmetic-order changes are bugs. The oracle
   copies the kernel body verbatim; the ONLY adaptations are storage-layout and
   plumbing (listed in Step 6), each recorded in the file header and STATUS.md.
4. CuMetal numbers never gate anything — no throughput/timing is recorded.
   `clock()` is synthetic; this task records compile status + numerics only.
5. Env knobs `SHISU_*` only: `SHISU_CUMETAL_PREFIX` (default
   `/opt/homebrew/opt/cumetal`), `SHISU_CUDA_COMPILE=1` (required for any real
   compile), `SHISU_CUMETAL_STRICT=1` (warnings ⇒ failure), `SHISU_SKIP_BUILD=1`
   wins over everything (probe script exits 0).
6. Workspace lints don't apply (no Rust crate code; `gdn_ref.rs` is compiled
   standalone by `rustc -O`, not cargo — keep it warning-free anyway).
7. No license headers on shisu files; **CuMetal-derived headers keep the
   Apache-2.0 provenance comment** (rule 7's explicit carve-out).

Task-specific extras (CuMetal hard-limits rows this task actually touches):
shared memory (compound layouts), atomics (every oracle numeric, never "it
ran"), memory (oracle allocates via `cudaMalloc` only — raw `malloc` pointers
are not kernel-bindable), device properties (never branch kernel selection on
them — the device reports synthetic cc 8.0), timing (none recorded).

## Source References (verified)

ds4 paths relative to repo root; cumetal paths to `/opt/homebrew/opt/cumetal`.
Every row was opened/measured while writing this plan (2026-09-05, M4, macOS
26.5.2 — master plan says 26.5.1; `cumetal doctor` all-green, 0.4.0).

| Source | Lines / probe | What / how used |
|---|---|---|
| `ds4_qwen4exp_gpu.cuh` | 451–542 | `q4e_gdn_recurrent_kernel` body — copied verbatim into the oracle. Constructs: `__launch_bounds__(512)` (:451), dynamic `extern __shared__ float q4e_gdn_smem[]` (:457, `head_dim*(head_dim+1)` floats), static `__shared__ s_k[128], s_q[128], s_kq` (:480–482), `__syncthreads` ×3 (:478,:499,:531, all block-uniform CFG), `__shfl_xor_sync(0xffffffff,…,1/2)` butterfly (:517–520), `__expf` (:502), `rsqrtf` (:470), `fmaf` (:514–515,:529), ckpt write-back guarded by uniform `if (ckpt && …)` (:532). **No atomics, no half/bf16, no cooperative groups** — pure f32. |
| `ds4_qwen4exp_gpu.cuh` | 450 | `#define Q4E_GDN_SPLIT 4u` — copied. |
| `ds4_qwen4exp_gpu.cuh` | 23–37 | `q4e_block_sum` (static `s[32]`, `total`) — copied (adapted to take the two arrays as pointers; arithmetic verbatim). |
| `ds4_cuda.cu` | 5374–5379 | `warp_sum_f32`: `__shfl_down_sync(0xffffffffu, v, offset)` for offset 16→1 — copied verbatim. |
| `ds4_qwen4exp_gpu.cuh` | 1917–1969 | wrapper `ds4_gpu_q4e_gdn_recurrent`: `shared = head_dim*(head_dim+1)*4` (:1949), `cudaFuncSetAttribute(…, MaxDynamicSharedMemorySize, shared)` (:1953–1955), launch `<<<n_head_v, head_dim*Q4E_GDN_SPLIT, shared, stream>>>` (:1962–1967), `k_offset = n_head_k*head_dim`, `v_offset = 2*n_head_k*head_dim` (:1960–1961). The oracle replaces this wrapper; kernel params are raw pointers (`->ptr` unwrapping lives only in the wrapper). |
| `ds4_gpu.h` | 22 | `ds4_gpu_tensor` is an opaque typedef used only by host wrappers — the extracted kernel needs no stubs (⚠ DEVIATION 3). |
| installed `include/` | probe `ls` | **all 8 claimed entries absent now**: `cuda_bf16.h`, `cuda_gl_interop.h`, `math_constants.h`, `mma.h`, `sm_35_intrinsics.h`, `vector_functions.h`, `vector_types.h`, `cuda/` subdir (dir has sm_20/30/60/70/80 but no sm_35, no `cuda/`). |
| `cumetalc` usage | probe | flags confirmed; **`-include` is REJECTED** (`unknown option: -include`); force-include flag is **`--cuda-include <path>`** (verified: a TU using a shim symbol with no `#include` compiled and linked). |
| `cuda_fp16.h` (installed) | 82–95, 121, 176, 200, 212–226 | declares `__float2half`(:82,:212), `__float2half_rn`(:83,:215), `__float2half2_rn`(:84,:218), `__low2half`(:91), `__high2half`(:92), `__half2half2`(:95), `__float22half2_rn`(:121), `__int2half_rn`(:176), `__floats2half2_rn`(:226); device `__half` is `typedef _Float16` (:200) → bit-casts MUST go through `memcpy`, not struct members. |
| upstream `runtime/api/` @ `cebf68a…` | GitHub API + raw fetch | all 8 entries exist; `cuda/` = `barrier`, `functional`, `iterator`, `pipeline`, `std/{functional,tuple,type_traits}` (14 files total, 36 KB); LICENSE is Apache-2.0; `cuda_bf16.h` declares `__nv_bfloat16`/`__nv_bfloat162`/`__float2bfloat16{,_rn}`/`__bfloat162float` but **not** `__halves2bfloat162`/`__bfloat1622float2` (shim must). |
| ds4 `cuda/` tree + `ds4_cuda.cu` + `.cuh` | grep counts | all 16 master-plan intrinsic names occur (e.g. `__lows2half2` in `cuda/mmq/mma.cuh:701`, `cuda/exl3/codebook.cuh:51`; `__halves2bfloat162` in `cuda/exl3/util.cuh:37`; `__ushort_as_half` in `cuda/exl3/util.cuh:126`). None occur in `q4e_gdn_recurrent_kernel` (f32-only). |
| device props (probe) | `cudaGetDeviceProperties` | `sharedMemPerBlock = sharedMemPerBlockOptin = 32768`, `maxThreadsPerBlock = 1024`, cc 8.0, name "Apple M4" (synthetic). |
| opt-in probe | `cudaFuncSetAttribute(…, 66048)` | → `cudaErrorInvalidValue`. `(…, 16640)` → OK. Production shape (head_dim 128 → 66048 B) is **runtime-blocked**; head_dim ≤ 64 fits. |
| compound-smem probe | static write after dynamic fill | static `__shared__` **aliases the `extern __shared__` base**: `dyn[0]` read back the value written to `stat[0]`. Silent corruption, no warning even with `--ptx-strict`. |
| folded-smem oracle probe | D=64, NK=2, NV=4, NT=3, random f32 | verbatim kernel body + all arrays folded into one dynamic region: compiles `--ptx-strict` clean, runs, matches exact-tree oracle: worst attn err 1.19e-6, worst state err 3.8e-6 → **PASS at 1e-5**. |
| shim probe | 8 helpers, bit patterns | compiled + ran on GPU, all bit-exact (`3c00 4400 4000 4800 1234 1 1`). |

⚠ **DEVIATION 1 (shim set is 8, not 16):** the master plan's measured set of 16
"not declared" is wrong for 8 of them — the installed `cuda_fp16.h` already
declares `__low2half`, `__high2half`, `__floats2half2_rn`, `__float2half2_rn`,
`__float2half_rn`, `__float22half2_rn`, `__int2half_rn`, `__float2half` (lines
cited above). The shim defines **exactly the 8 genuinely missing** helpers;
re-defining the present 8 breaks the build. `__half_raw` is a POD struct, not a
function.
⚠ **DEVIATION 2 (force-include flag):** master plan Task 14 writes
`-include cumetal_half_shim.h`; measured: `cumetalc` rejects `-include`. The
working flag is `--cuda-include cumetal_half_shim.h` (verified end-to-end).
⚠ **DEVIATION 3 (no tensor stubs needed):** Task 3b's checklist says "with the
`ds4_gpu_tensor` accessors stubbed"; verified: the kernel signature
(:452–456) takes raw `float*`/`uint32_t` — `ds4_gpu_tensor` appears only in the
host wrapper (:1917+), which the oracle replaces wholesale. Nothing to stub.
⚠ **DEVIATION 4 (oracle is exact-tree, not "naive"):** a naive left-to-right
f32 sum over a 128-term row can re-associate more than 1e-5 away from the
kernel's shuffle trees. The Rust oracle reproduces the exact reduction trees
(below); "naive" is read as "independent hand-written reference", tol 1e-5 unchanged.

## Plan

- [ ] **Step 1: Preflight re-measure** (cheap, read-only; if any result differs
  from the Source References, update STATUS.md accordingly — the verdict is
  whatever the machine says today):
  ```sh
  cumetal doctor
  ls /opt/homebrew/opt/cumetal/include | grep -c . # 8 entries must be absent
  mkdir -p shisu/target/cumetal-probe && cd shisu/target/cumetal-probe
  # tiny .cu printing cudaDeviceProp.sharedMemPerBlock{,Optin}; compile with
  #   cumetalc --cuda-device --mode experimental props.cu -o props --ptx-strict \
  #     -I /opt/homebrew/opt/cumetal/include
  ```
  Expect `32768 / 32768`. Also re-confirm `-include` is rejected.

- [ ] **Step 2: Tree.** `mkdir -p shisu/kernels/cumetal_compat/include/cuda/std shisu/tests/cumetal`.

- [ ] **Step 3: `kernels/cumetal_compat/FETCH.sh`.** Pins
  `SHA=cebf68a403601d59e8f67cdf5d85973c50fc0732` (dereferenced commit of
  annotated tag `v0.4.0`; tag object `01ad04e1548c9977d01a9587cdd5ff732d5ad94f`
  — record both in a comment). Loops the 14 paths
  (`cuda_bf16.h cuda_gl_interop.h math_constants.h mma.h sm_35_intrinsics.h
  vector_functions.h vector_types.h cuda/barrier cuda/functional cuda/iterator
  cuda/pipeline cuda/std/functional cuda/std/tuple cuda/std/type_traits`) from
  `https://raw.githubusercontent.com/Lulzx/cuda-metal/$SHA/runtime/api/<path>`
  into `include/<path>`, prepends a 4-line provenance comment (repo, tag+SHA,
  Apache-2.0, "do not edit — re-run FETCH.sh"), then verifies each file against
  a pinned `sha256` table (the 14 hashes were recorded 2026-09-05; paste them
  into the script — e.g. `cuda_bf16.h` =
  `713ac79fcb89a80b25203e7677c59514cbf24c52f1ac20c5c2c829c51586939b`).
  Idempotent; `curl -fsSL` failure ⇒ non-zero exit. Commit the fetched files
  (36 KB total) so the lane works offline.

- [ ] **Step 4: `kernels/cumetal_compat/cumetal_half_shim.h`.** Include-guard
  wrapped; `#include <cuda_fp16.h>` + `<cuda_bf16.h>` first (force-include runs
  before the TU's own includes; `cuda_bf16.h` resolves from `include/` via `-I`).
  Define exactly these 8 as `static __device__ __forceinline__` (one-line
  semantics; bodies are bit-level, verified pattern in the plan-author probe):
  - `__halves2half2(lo, hi) → half2` — pack two halves, `lo`→`.x`, `hi`→`.y`.
  - `__lows2half2(a, b) → half2` — `{a.x, b.x}` (CUDA sig: two `half2`s).
  - `__highs2half2(a, b) → half2` — `{a.y, b.y}`.
  - `__ushort_as_half(u16) → half` — bit-reinterpret; **`memcpy`**, device
    `__half` is `_Float16` (cuda_fp16.h:200), no member access.
  - `__half_as_ushort(half) → u16` — inverse, `memcpy`.
  - `struct __half_raw { unsigned short x; }` + `__host__ __device__` ctor from
    `__half` and `operator __half()` (both `memcpy`) — needed by
    `cuda/mmq/test/test_mmq_parity.cu`'s `__half_raw r = h;` idiom.
  - `__halves2bfloat162(lo, hi) → bf162` — pack two `__nv_bfloat16`.
  - `__bfloat1622float2(bf162) → float2` — `(float)` per lane (upstream
    `cuda_bf16.h` gives `__nv_bfloat16::operator float`).
  Add a comment block listing the 8 names the shim must NOT define (already in
  `cuda_fp16.h`, lines cited in Source References).

- [ ] **Step 5: `tests/cumetal/shim_selftest.cu`.** One kernel exercising all 8
  helpers on fixed bit patterns (`0x3C00`=1.0, `0x4400`=3.0, bf16 ctor 1.5 /
  −2.25), host checks the returned `u16`s against hand-computed patterns —
  exact equality, no tolerance. This is the achievable form of the master
  plan's "checked against the CUDA definition on the host": there is no CUDA
  on this Mac, so the check is device-run against integer bit patterns
  (⚠ DEVIATION noted in STATUS.md, one line).

- [ ] **Step 6: `tests/cumetal/oracle_gdn.cu`.** Extraction scope, copied
  verbatim: `Q4E_GDN_SPLIT` (:450), `warp_sum_f32` (ds4_cuda.cu:5374–5379),
  `q4e_block_sum` (:23–37), `q4e_gdn_recurrent_kernel` (:451–542). The only
  adaptations (record each in the file header + STATUS.md; rule 3 holds — no
  arithmetic operator changes):
  1. **Fold static smem into the single dynamic region** (measured: static
     `__shared__` aliases the `extern __shared__` base → silent corruption).
     Layout: `s_state[D*(D+1)] | s_k[128] | s_q[128] | s_kq[1] | bs_s[32] |
     bs_total[1]`; `q4e_block_sum` becomes `q4e_block_sum(float* s, float*
     total, float v)` — same tree, arrays passed in.
  2. Host `main` replaces the ds4 wrapper: argv contract
     `oracle_gdn gdn_recurrent <in.bin> <out.bin> <head_dim> <n_head_k> <n_head_v> <n_tok>`;
     derives `k_offset = n_head_k*head_dim`, `v_offset = 2*n_head_k*head_dim`,
     `stride = (2*n_head_k + n_head_v)*head_dim` (:1960–1961 + packed-qkv
     layout); `ckpt = NULL`, `ckpt_cap = 0`.
  3. Allocation is `cudaMalloc`/`cudaMemcpy` only (hard-limits "Memory" row).
  4. `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, smem)` before launch
     (pattern :1949–1967); on `cudaErrorInvalidValue` print
     `SMEM_BLOCKED <bytes> <limit>` and exit 6 — that is a recorded verdict,
     not a crash.
  `.bin` layout, f32 LE: `in.bin` = `qkv[n_tok×stride]`, `state[nv×D×D]`,
  `decay[n_tok×nv]`, `beta[n_tok×nv]`; `out.bin` = `attn_out[n_tok×nv×D]`,
  `state_out[nv×D×D]`. Exit 0 on success; `cudaGetLastError`/
  `cudaDeviceSynchronize` failures print `cudaGetErrorString` and exit non-zero.

- [ ] **Step 7: `tests/cumetal/gdn_ref.rs`.** Standalone (`rustc -O`, no
  deps). Modes `gen` (deterministic LCG in [−1,1] for qkv/state, decay in
  [−0.1,0.1], beta in [0,1]), `ref`, `cmp`. `ref` reproduces the kernel's
  **exact** arithmetic in `f32` with native `f32::mul_add`: per token —
  `kq_part` nonzero only at `sub==0 && j<D` (`k[j]*q[j]`); block-sum tree =
  per-warp `shfl_down` fold (offsets 16,8,4,2,1 over 32 lanes, 8 warps of a
  256-thread block at D=64) then lane-0 fold over `s[0..n_warp)`; per column
  quarter walks `i = sub*seg + ((n + 8*sub) & (seg-1))`, merge
  `sk = (q0+q1)+(q2+q3)` (xor1/xor2 butterfly); `delta = (v[j] − dec*sk)*bt`;
  `out = (dec*sq + delta*kq)*rsqrt(D)`; state update `row[i] = fma(row[i],
  dec, delta*k[i])` in the same walk order. `dec` uses IEEE `expf` — the
  kernel's `__expf`/`rsqrtf` fast-math difference is absorbed by tol 1e-5
  (measured residual 3.8e-6).

- [ ] **Step 8: `scripts/cumetal_probe.sh`** (run from `shisu/`). Gate order:
  `SHISU_SKIP_BUILD=1` ⇒ exit 0; `SHISU_CUDA_COMPILE != 1` ⇒ print skip reason,
  exit 0; `! command -v cumetalc` ⇒ exit 0 unless `SHISU_CUMETAL_STRICT=1`
  (then exit 1). Then, with `PREFIX=${SHISU_CUMETAL_PREFIX:-/opt/homebrew/opt/cumetal}`
  and `OUT=target/cumetal-probe`:
  ```sh
  cumetalc --cuda-device --mode experimental tests/cumetal/oracle_gdn.cu \
      -o "$OUT/oracle_gdn" --ptx-strict \
      -I kernels/cumetal_compat/include -I kernels/cumetal_compat \
      --cuda-include cumetal_half_shim.h -I "$PREFIX/include"
  ```
  (same invocation for `shim_selftest.cu`). Capture stderr; with
  `SHISU_CUMETAL_STRICT=1`, any line matching `warning` ⇒ exit 1.
  `rustc -O tests/cumetal/gdn_ref.rs -o "$OUT/gdn_ref"`. Shape matrix:
  `(64 2 4 3)` and `(64 16 32 1)` (real qwen3.5-4b GDN head counts, decode
  shape) must PASS `cmp` at 1e-5; `(128 16 16 1)` is expected to exit 6 with
  `SMEM_BLOCKED 67208 32768` — record, do not fail.

- [ ] **Step 9: `kernels/cumetal_compat/STATUS.md`.** Template (pre-filled with
  the 2026-09-05 plan-author measurements below; executor re-runs the probe and
  updates every row to today's result — a row may only say `ok` if the exact
  command in the row passed):

  ```markdown
  # CuMetal lane status (SSOT — Tasks 5/14/15/17/20 read this, do not re-probe)
  Toolchain: cumetal 0.4.0, macOS <ver>, prefix $SHISU_CUMETAL_PREFIX
  Verdict command: SHISU_CUDA_COMPILE=1 scripts/cumetal_probe.sh
  | Construct / TU | Status | Evidence (exact diagnostic or max_err) |
  |---|---|---|
  | dynamic extern __shared__ | ok | folded oracle PASS, max_err 3.8e-6 |
  | __syncthreads (uniform CFG) | ok | folded oracle PASS |
  | __shfl_down_sync (full mask) | ok | tagged-lane microtest exact |
  | __shfl_xor_sync (full mask) | ok | tagged-lane microtest exact |
  | atomicAdd(float*) | not-exercised | q4e_gdn_recurrent has no atomics; covered by master-plan probe 2026-09-05 |
  | __launch_bounds__(512) | ok | compiles + runs |
  | large dynamic smem opt-in | BLOCKED >32 KiB | cudaFuncSetAttribute(66048) → cudaErrorInvalidValue; sharedMemPerBlockOptin=32768 |
  | compound static+dynamic smem | CORRUPTS | static aliases extern base (dyn[0] reads stat[0] write); harness folds statics into dynamic region |
  | bf16 conversions | ok | shim_selftest bit-exact |
  | half/bf16 shim (8 defs) | ok | shim_selftest bit-exact |
  | oracle_gdn.cu @ D=64 | ok | max_err < 1e-5 vs gdn_ref |
  | oracle_gdn.cu @ D=128 | BLOCKED | SMEM_BLOCKED 67208 32768 (production GDN head_dim) |
  | (T14/15 append per-TU rows here) | | |
  ## Verdict
  <gate outcome + blocking construct, one paragraph>
  ```

- [ ] **Step 10: Run + record.** `SHISU_CUDA_COMPILE=1 scripts/cumetal_probe.sh`;
  fill the Verdict section; apply the gate below.

**Gate (either outcome completes the task — the verdict is the deliverable):**
kernel compiles + matches oracle at the runnable shapes ⇒ T5/T17 gain the
CuMetal differential oracle **at head_dim ≤ 64** (production head_dim 128 is
smem-blocked on this device, so shape-exact 128 verification stays
Linux/golden-vector — say so in STATUS.md). Any hard failure (won't compile,
numeric mismatch not root-caused to a line) ⇒ record the blocking construct;
Phase 4 stays exactly as specified for Linux and CuMetal degrades to a compile
checker. The measured evidence at plan-authoring time points at the first
branch with the two caveats (fold adaptation + D≤64).

## Tests

The probe **is** the test suite (no cargo tests exist yet for these paths):

- `scripts/cumetal_probe.sh` (macOS only; on Linux or without `cumetalc` it is
  a no-op unless STRICT) — exit 0 = lane green. Tolerances: 1e-5 absolute on
  `attn_out` and `state_out` (master plan); shim selftest is bit-exact.
- `SHISU_CUMETAL_STRICT=1` variant: any compiler warning line fails the run
  (this is the mode T15 uses for split TUs).
- `SHISU_SKIP_BUILD=1 scripts/cumetal_probe.sh` must exit 0 in <1 s (gate path).
- Numerics are the proof, never "it ran" (hard-limits Atomics row rationale):
  `gdn_ref cmp` reports max abs error and the failing index.

## Acceptance

- [ ] `shisu/kernels/cumetal_compat/{FETCH.sh,include/,cumetal_half_shim.h,STATUS.md}`,
      `shisu/tests/cumetal/{oracle_gdn.cu,gdn_ref.rs,shim_selftest.cu}`,
      `shisu/scripts/cumetal_probe.sh` exist; each < 1500 LoC.
- [ ] `FETCH.sh` pins `cebf68a403601d59e8f67cdf5d85973c50fc0732`, verifies 14
      sha256s, headers carry the Apache-2.0 provenance comment.
- [ ] Shim defines exactly the 8 missing helpers (grep: none of the 8
      already-declared names is defined).
- [ ] `SHISU_CUDA_COMPILE=1 scripts/cumetal_probe.sh` exits 0; both D=64 shapes
      PASS at 1e-5; D=128 recorded `SMEM_BLOCKED`.
- [ ] STATUS.md verdict table fully filled with today's evidence + the Verdict
      paragraph states the gate outcome and the head_dim ≤ 64 caveat.
- [ ] Nothing added under `shisu/crates/`; no Cargo.toml touched (rule 1).
- [ ] `SHISU_SKIP_BUILD=1` short-circuit verified.

## Commit

```sh
git add shisu/kernels/cumetal_compat shisu/tests/cumetal shisu/scripts/cumetal_probe.sh
git commit -m "test: CuMetal compat bundle + GDN oracle spike (Task 3b)"
```
