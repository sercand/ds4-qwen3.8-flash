# Task 6: shisu-engine qwen35 forward

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 6 (read it before executing; the spec argues this plan)

**Depends on:** T1 (crate stub), T2/T3/T4/T5/T7 (consumed, exact names from their plan files in `docs/superpowers/plans/tasks/`):
- T2: `shisu_core::{Model, Backend, DeviceBuffer, DeviceView, KernelRef{module,name}, KernelArg, LaunchArgs{grid,block,smem_bytes,args}, Extent3, SessionId, SampleRequest{temperature,top_k,top_p,min_p,rng}, ShapeProfile::Qwen35, CoreError, Result}` — trait signatures verbatim (Step 8 block).
- T3: `GgufFile::open_shared`, `bind_qwen35(&GgufFile,&ShapeProfile)->ModelBinding{token_embd,output_norm,lm_head:Option,layers:Vec<BoundLayer{kind,input_norm,post_norm,ffn,gdn:Option<GdnWeights>,attn:Option<AttnWeights>}}>`, `TensorView{name,dtype,data,shape}`, `DType`.
- T4: `MetalBackend::new/register_no_copy(&[u8])->Result<DeviceBuffer>/metal_available` (dev-dep only — the engine crate itself never depends on shisu-metal); `Backend::run` grid/block = CUDA semantics; `KernelRef.name` `#idx:t<val>` FC suffix.
- T5 (`task-05-qwen35-msl-kernels.md`, on disk): **all qwen35 kernel names + arg layouts + grid/block/smem formulas are SSOT'd in `shisu/crates/shisu-metal/metal/qwen35/KERNELS.md`** — this plan never invents a name; every dispatch cites the KERNELS.md entry. Names T6 dispatches (T5 Produces): `kernel_mul_mv_ext_q5_K_f32_r1_{1..5}`, `kernel_mul_mv_ext_q6_K_f32_r1_{1..5}` (ext-family args + `#600:s…,601:s…` suffix), `kernel_get_rows_q6_K_f32`, `kernel_mul_mm_{q4_K,q8_0,f16}_f32`, `kernel_q35_gdn_{conv,l2norm,gates,recurrent,chunk,out_gate}`, `kernel_q35_{q_norm_rope,k_norm_rope_kv,attn_decode,attn_prefill,attn_gate_mul}`. **No host-side repack in T6** (T5 adds the q5_K/q6_K kernels; rationale in its Step 3). If KERNELS.md at execution time lacks an entry the executor STOPs and reports — never invents a launch.
- T7 (`task-07-kvstore.md`, on disk): `shisu_kvstore::{KvCache, CacheGeometry, Backing, Frontier, PageTable, CkptRef, CommitOpts, Lookup, Resume, ReuseSource, CacheStats, SpillConfig}` + `KvCache::next_admission` — the engine consumes the ONE KV-cache system (master Task 6: "the Task 7 radix-tree cache … NOT contiguous buffers"). T6 derives qwen35's `CacheGeometry` (task-07 §Step 2: 8 MiB page / 51.2 MiB ckpt), implements `Backing` over the Metal pool buffers, and owns a `Frontier` per session.

**Produces** (public API T8/T10/T13/T17 build on; paths `shisu/crates/shisu-engine/src/`):
```rust
// qwen35.rs
pub struct Qwen35Model { /* Arc<dyn Backend>, binding, sessions, scratch, weight bufs */ }
impl Qwen35Model {
    pub fn new(backend: std::sync::Arc<dyn Backend>) -> Self;
    /// Zero-copy weight registration hook (T4 register_no_copy); default = alloc+copy_h2d.
    pub fn with_weight_register(self, f: WeightRegister) -> Self;
    pub fn with_ctx(self, cap: u32) -> Self;               // default 4096
}
pub type WeightRegister = std::sync::Arc<dyn Fn(&[u8]) -> shisu_core::Result<DeviceBuffer> + Send + Sync>;
impl shisu_core::Model for Qwen35Model { /* the 7 T2 methods, verbatim — see Step 8 */ }
// sampling.rs — pure host f32, backend-agnostic, reused verbatim by T17
pub fn sample_logits(logits: &[f32], req: &mut SampleRequest) -> u32;   // = ds4_sample_logits
pub fn rng_next(state: &mut u64) -> u64;  pub fn rng_f32(state: &mut u64) -> f32;
pub fn argmax(logits: &[f32]) -> usize;                                  // unrolled8 semantics
// sampling_probs.rs — spec-decode distribution API (T17 acceptance/rejection path)
pub fn build_probabilities(logits: &[f32], temperature: f32, top_k: i32, top_p: f32, min_p: f32, probs: &mut [f32]) -> bool;
pub fn sample_probabilities(probs: &[f32], rng: &mut u64) -> i32;
// sessions.rs / kv.rs / gdn.rs — pub(crate) engine internals (T8 resumes through the Task 7 cache; T13 drives chunk ends via `next_admission`)
```

## Global Constraints

Master plan Rules 1–7, one line each:
1. All new code under `shisu/`; pure Rust (no kernel files in this task — kernels are T5's).
2. 1500-LoC cap — pre-split: `layers.rs` (per-layer orchestration) and `sampling_probs.rs` join the master-plan 7 files; the plan's own parenthetical "(siblings split at 1500 LoC)" authorizes this. ⚠ DEVIATION (additive).
3. Arithmetic-order changes are bugs — this task's sampler and gate/norm formulas are bit-exact ports; no re-association, no FMA "optimization", f32 throughout.
4. Metal phase: record baselines, no ds4 comparison exists for qwen3.5-4b; T20 owns regression gates.
5. Env knobs `SHISU_`-prefixed: this task reads only `SHISU_TEST_MODEL` (T3's). ds4's `DS4_CPU_DISABLE_UNROLLED_ARGMAX` (ds4.c:40907) is NOT ported — unrolled8 is the default and only path.
6. `warnings = deny` + clippy deny; errors are `shisu_core::CoreError` (no new error type, no anyhow); `parking_lot`; `tracing` for load/teardown logs.
7. No license headers.

Task-specific:
- The engine crate depends ONLY on `shisu-core`, `shisu-gguf`, `tracing`, `parking_lot` — no objc2/Metal/cudarc; it compiles on Linux against a fake Backend (T16/T17 reuse).
- Run cargo from `shisu/`; `cargo test -p shisu-engine` only (T20 owns the final gate). GPU/model tests `#[ignore]`-gated.
- ⚠ DEVIATION (Backend trait is T2 SSOT and frozen): `register_no_copy` is NOT on `Backend`. The engine gets zero-copy weights via the `WeightRegister` closure injected at construction (T10 passes `|b| metal_backend.register_no_copy(b)`); `None` → `alloc`+`copy_h2d` fallback (correct everywhere, ~2× weight RSS — test/dev path only).
- ⚠ DEVIATION (`Model::load` takes `&dyn Backend` with no lifetime link): the persistent handle comes from `Qwen35Model::new(Arc<dyn Backend>)`; `load` asserts `std::ptr::eq(backend, &*self.backend)` else `CoreError::Backend`.

## Source References (verified)

ds4 root = cwd; `cuh` = `ds4_qwen4exp_gpu.cuh`. Every line opened while writing this plan.

| Source | Lines | What / how used |
|---|---|---|
| `ds4.c` | 40819–40920 | `argmax_f32_unrolled8_range` (8-lane, merge ties→lower id :40851–40866, scalar tail), `sample_argmax_unrolled8`, `sample_argmax` — greedy path. |
| `ds4.c` | 40942–40955 | PRNG: `sample_rng_next` = xorshift64* — zero state seeded `0x9e3779b97f4a7c15`; `x^=x>>12; x^=x<<25; x^=x>>27; *state=x; return x*0x2545f4914f6cdd1d`. `sample_rng_f32` = `((x>>40)&0xffffff) as f32 / 16777216.0f`. |
| `ds4.c` | 40957–41005 | `sample_candidate{id,logit,prob}`, comparator logit-desc/id-asc (:40963–40975, a TOTAL order → Rust `sort_unstable_by` is deterministic), min-heap sift up/down. |
| `ds4.c` | 41010–41163 | `sample_build_probabilities` (heap-cap 1024/512 fast path, `full_sum` in index order, qsort, min-p/top-p filter, `inv_sum` normalize) + `sample_probabilities` (sum+best pass, one rng draw, cumulative subtract) — → `sampling_probs.rs` (T17). |
| `ds4.c` | 41192–41278 | `sample_fast_top_p`: cap 512 heap, bail if `finite>512 && top_p≥0.999`, `heap_sum < top_p*sum`, min-p tail-uncertain bail (:41257–41262), rng walk on raw probs. |
| `ds4.c` | 41280–41441 | `sample_full_vocab`: pass-1 max/best/finite; fast path gate `top_p<1`; `top_p≥1` branch with log-space reject boundary (`logf`+8×`nextafterf`+`expf` probe :41327–41339) + `prob_scratch` walk; else min-p prefilter or full gather, qsort, filter, walk. |
| `ds4.c` | 41443–41513 | `sample_top_p_min_p` dispatcher (the exact stage order — Step 1) incl. insertion-sorted top-k array :41468–41483 and softmax-over-k :41486–41492. |
| `ds4.c` | 73705–73719 | public entries `ds4_sample_logits` (scratch wrapper) / `ds4_session_sample`. ⚠ DEVIATION: master plan cites "§70205+" for sampling — verified wrong: §70205 sits inside the MTP draft host region (T17's §67082–70205+ territory); the sampler is §40819–41513 + §73705–73719. |
| `ds4.c` | 381 | `DS4_NEG_INF = -1.0e30f` (not `-inf`) — argmax init + non-finite handling uses `isfinite` + this sentinel. |
| `ds4.c` | 69086–69115 | `q4e_gdn_layer` host order: qkv matmul → conv(+silu, stateful) → L2-norm q then k heads (v raw) → alpha/beta matmuls → gates → recurrent → z matmul → out_gate → out matmul. |
| `cuh` | 373–401 | `q4e_gdn_conv_kernel`: conv window state layout `state[i*channels+c]`, i∈[0,kernel−1) (row-major, channels fastest); per-token window shift; silu inline `acc/(1+exp(−acc))`. |
| `cuh` | 405–416 | `q4e_gdn_l2norm_kernel`: L2 (not RMS) per 128-wide head, eps `1e-12`, block-tree sum. |
| `cuh` | 418–433 | `q4e_gdn_gates_kernel`: `decay = softplus(alpha_proj+dt_bias[h]) * a[h]` (a holds −exp(A_log); guard `x>20 ? x : log1p(exp(x))`), `beta = sigmoid(beta_proj)`. |
| `cuh` | 435–542 | `q4e_gdn_recurrent_kernel`: state per V head = D×D stored **transposed** `m[j][i]=S[i][j]`; value head h reads qk head `h % n_head_k` (modulo — comment :442–443); per token `S*=exp(decay); delta=(v−<S,k>)·beta; S+=delta⊗k; out=<S,q>/sqrt(D)`. Chunked prefill kernel :544+ (numerics differ by design — oracle-gated, cuh:599–603). |
| `cuh` | 810–816 | `q4e_gdn_out_gate_kernel`: per-head RMSNorm with `ssm_norm` applied RAW (no +1 offset), then sigmoid output gate from `attn_gate` projection. |
| `ds4.c` | 69208–69295 | `q4e_qsa_layer` host order: q proj → q_norm_rope (splits gate) → k/v proj → store KV (k_norm+rope fused) → attention → `qsa_gate` → attn_output matmul. |
| `cuh` | 2048–2052 | gated-attention layout law: q-proj row = per-head `[q(head_dim) | gate(head_dim)]` **interleaved** in `2*head_dim*n_head`; "splitting the flat row down the middle … is the single easiest mistake to make here". |
| `cuh` | 2096–2117 | `q4e_rope_mrope` — the rope source of truth: rotates the **PREFIX**: pairs `(v[i], v[i+n_rot/2])` for `i < n_rot/2` (n_rot=64 → pairs (i, i+32)), dims ≥ n_rot untouched. |
| `cuh` | 2121–2141 | `q4e_qsa_q_norm_rope_kernel`: per-head RMSNorm(q) + partial rope; gate extracted as `qkv[src + head_dim + i]` (raw, never activated). |
| `metal/dsv4_rope.metal` | 99–136 | `kernel_dsv4_rope_tail_f32` copies the nope PREFIX and rotates the **TAIL** (dsv4 layout) — ⚠ NOT qwen35's rope; T4's plan note calling it "qwen35 n_rot=64" is wrong (T5 confirmed + source re-verified). The engine MUST NOT dispatch the kept tail-rope family. |
| `cuh` | 3265–3270 | `q4e_qsa_gate_kernel`: `out = attn * sigmoid(gate)` BEFORE the `attn_output` projection. |
| `cuh` | 2150–2158, 2163–2168 | `q4e_kv_row(pages,p)` — physical KV row of logical position p through the session's page table; the comment: "Every KV-touching kernel below goes through this; a tile that is page-aligned and no wider than a page translates its first position and walks the rest contiguously." `q4e_qsa_store_kv_kernel` takes `const int32_t *pages` (:2167). The engine's kernels keep this indirection (⚠ supersedes task-05 Steps 7/9 "contiguous KV (drop pages)" — the Task 7 paged cache is the only KV system). |
| T2 plan `task-02-core.md` | Produces + DEVIATION | `Qwen35Shape`: head_dim **256**, n_rot 64, gdn{32,16,128,4}, vocab 248320, intermediate 9216, rope_theta 1e7, rms_eps 1e-6, eos 248044 (config.json-verified). |
| T3 plan `task-03-gguf.md` | Produces + Tests | binding structs/names; Q4_K_M facts: 426 tensors, `token_embd` Q6_K `[2560,248320]`, `lm_head=None` (tied), 24 GDN + 8 Attn iff `(il+1)%4==0`, model file 2,740,937,888 B. |
| T4 plan `task-04-metal-backend.md` | Produces + risk note | `register_no_copy`, FC-suffix convention, ext-family dispatch math, the q5_k/q6_k gap this task consumes T5's decision for. |
| `cuh` | 1935–1937, 3332 | Path-selection gates the engine mirrors: GDN chunk iff `n_tok >= chunk_min(=2C=128) && n_tok > 1 + ckpt_cap(=0)`; attention tiled-prefill iff `n_tok >= Q4E_ATTN_QT(16) && head_dim%32==0`, else decode kernel (grid `(n_head, n_tok)` handles multi-row). ds4's `DS4_QWEN4EXP_GDN_CHUNK_MIN` / `DS4_QWEN4EXP_NO_TILED_ATTN` env knobs NOT ported (rule 5) — fixed constants. |
| T5 plan `task-05-qwen35-msl-kernels.md` + KERNELS.md | Produces (on disk) | Exact dispatch names (see Depends); q5_K/q6_K kernels, no repack; PREFIX rope + f16 KV store at slot=pos via `kernel_q35_{q,k}_norm_rope*`; gate slice = separate raw f32 buffer from q_norm_rope, `attn_gate_mul` standalone before o_proj. |

## Plan

- [ ] **Step 1: `sampling.rs` — bit-exact port (write FIRST, test without any model).** Port in this exact stage order (each maps to the Source-References ranges; keep the C control flow 1:1, f32 ops in the same order):
  1. `temperature <= 0` → `argmax` (unrolled8; ties → lower id; init `best=0, best_v=NEG_INF=-1.0e30f`); **no rng advance**.
  2. Clamps: `top_p<=0||top_p>1 → 1`; `min_p<0 → 0`; `top_k>1024 → 1024`; `top_k>n_vocab → n_vocab`.
  3. `top_k<=0` → full-vocab path: pass-1 (max/best/finite, `isfinite` skip); if `top_p<1` try fast path (heap cap 512, both bail conditions verbatim); if `top_p>=1` the log-space-reject branch (`f32::ln` + 8× `next_down` + `exp` probe, scratch `-1.0` sentinel, single rng walk); else candidate gather (min-p prefilter via scratch or full), `sort_unstable_by(cmp_desc)`, filter loop (`filtered_sum/base_sum >= top_p` break), one rng draw + cumulative subtract.
  4. `top_k>0`: insertion-sorted descending top-k (skip-if-`v<=vals[n-1]`, tie keeps earlier id), softmax over k with `max_logit=vals[0]`, min-p/top-p filter, one rng draw, walk.
  Exactly ONE `rng_f32` draw per stochastic sample; `req.rng` advanced in place (T2 semantics). `expf/logf/log1pf/nextafterf` → `f32::exp/ln/ln_1p/next_down` (same macOS libm as ds4's host build). Model `sample()` keeps a persistent `Vec<f32>` scratch (no per-call alloc; ds4's `xmalloc` at :73708 becomes the reused buffer).
- [ ] **Step 2: `sampling_probs.rs`.** Port `sample_build_probabilities` (:41010–41138, incl. heap fast path, `full_sum` index-order accumulation, `min_p_proves_tail_filtered`) and `sample_probabilities` (:41140–41163) verbatim — T17's spec-decode acceptance path consumes them; skip the `DS4_TEST_HOOKS` residual (T17 ports it with its MTP work).
- [ ] **Step 3: `sessions.rs`.** `pub(crate) struct Session { id: SessionId, n_written: u32, kv: kv::SessionKv, gdn_conv: Vec<DeviceBuffer> /*24*/, gdn_state: Vec<DeviceBuffer> /*24*/, logits_row: u32 }` + `SessionStore { by_id: HashMap<u64, Session>, next: u64 }`. `open_session`: allocate + zero-fill (copy_h2d of zeros) the live recurrent state only — GDN recurrent 32×128×128×4 B = 2 MiB/GDN-layer, conv 3×8192×4 B = 96 KiB/GDN-layer, one logits row 248320×4 B — plus an empty `kv::SessionKv` (Step 4): attention KV is owned by the shared Task 7 cache pool; a session holds a `Frontier` (page table + live tree node), never per-session KV buffers. `close_session` frees the GDN buffers and drops the Frontier (page refs released; the tree keeps what it wants). Position overflow (`n_written == ctx cap`) → `CoreError::Backend` naming session + cap.
- [ ] **Step 4: `kv.rs` — adapter over the Task 7 cache (the ONE KV-cache system; the old contiguous per-session buffers are dead design).**
  1. `CacheGeometry` at load — derivation contract + numbers live in task-07 §Step 2, never restate totals here: `page_tokens = 256` (`shisu_core::q4e_page`, task-02 §Step 4); `page_bytes` = 8 attn × 2(K,V) × 256 × (4 kv_heads × 256 head_dim) × 2 B f16 = **8 MiB**; `ckpt_bytes` = 24 GDN × (3×8192 conv + 32×128×128 state) × 4 B + 248320×4 logits = **53,684,224 B ≈ 51.2 MiB**; `prefill_s_per_token` = measured prefill s/token from `shisu-bench/baseline/m1max-qwen35.json` (Step 9 records it; task-07 §Step 2 — recorded, not guessed); `decay_s = 600.0`.
  2. `KvCache::open(&geo, Box::new(MetalBacking), pool_pages, ckpt_slots, spill)` — defaults 512 pool pages (4 GiB, lazily touched) + 16 ckpt slots, overridable via `Qwen35Model::with_kv_pool(pages, slots)`; `spill: Option<SpillConfig>` threaded from the caller (T10's `--kvstore-dir`). No new env knobs (rule 5).
  3. `pub(crate) struct MetalBacking` implements `shisu_kvstore::Backing` over 16 pool buffers (`[pool_pages·256, 4096]` f16 — one per attn layer × {K,V}): `copy_page` = 16 × `Backend::copy_d2d` of the page's 256-row run (ds4 `q4e_page_copy` shape, ds4.c 56232–56265); `page_to_host`/`page_from_host` = `copy_d2h`/`copy_h2d` of the same runs (spill/restore path only).
  4. `pub(crate) struct SessionKv { frontier: shisu_kvstore::Frontier, pages_dev: DeviceBuffer }` — the table's `pages` u32 array re-uploaded to `pages_dev` whenever `grow` adds pages (kernels read it like ds4's `const int32_t *pages`); drop = Frontier drop → unrefs (task-07 `PageTable` Drop rule).
  5. Kernel seam: KV store + attention kernels address rows via `kv_row(pages, pos)` (cuh:2150–2158, store_kv signature :2163–2168). ⚠ DEVIATION (supersedes task-05 Steps 7/9 "contiguous KV (drop pages)" note): paged addressing on every platform; if KERNELS.md at execution time records a pages-less KV signature the executor STOPs and reports — same rule as a missing kernel name.
  6. Commit + resume: on `prefill_chunk(is_last_chunk)` and each `decode_batch` step call `cache.commit(&mut frontier, tokens, q, CommitOpts { with_logits, chunk_only, hints, state, logits })` — `q` = absolute end position of the committed span (ds4's `q <= node->end` no-op rule, ds4.c:67704); `state`/`logits` blobs (`copy_d2h` snapshots) are passed only when `cache.next_admission(&frontier, tokens, hints) == Some(q)` (ds4's host pattern — the store re-derives admission inside `commit`; task-07 Deviation 3 makes each snapshot a ~51 MiB d2h, so never pay it speculatively). Resume (driven by T8): `cache.lookup(tokens)` → `Resume { frontier, ckpt, ckpt_pos, matched }` → `ckpt_state()` → `copy_h2d` into the session's GDN buffers + logits row, `n_written = ckpt_pos`, skip prefill of `matched` tokens.
- [ ] **Step 5: `gdn.rs`.** `pub(crate) fn forward_gdn(...)` driving, per ds4.c:69086–69115 order: qkv gemv (dtype-dispatched) → `kernel_q35_gdn_conv` (state `[kernel-1][8192]` layout per cuh:383/400; handles n_tok 1 AND multi-token — token loop inside) → `kernel_q35_gdn_l2norm` (q,k heads, eps 1e-12) → alpha/beta gemvs → `kernel_q35_gdn_gates` (softplus guard verbatim) → recurrence: `kernel_q35_gdn_chunk` iff `n_tok >= 128` (= 2C, fixed constant — ds4's `DS4_QWEN4EXP_GDN_CHUNK_MIN` knob NOT ported, rule 5; gate mirrored from cuh:1935–1937 with ckpt_cap=0), else `kernel_q35_gdn_recurrent` (state per V head D×D **transposed**, qk head = `h % 16`) → z gemv → `kernel_q35_gdn_out_gate` (raw `ssm_norm` RMSNorm + sigmoid(z)) → out gemv. State buffers are per-session (Step 3); batched decode runs the stateful kernels once per session row (ds4's `g->batch` row-loop pattern, ds4.c:69044–69054).
- [ ] **Step 6: `attn.rs`.** Per ds4.c:69208–69295: q gemv → `kernel_q35_q_norm_rope` (per-head RMSNorm(q_norm) + PREFIX rope — pairs `(i, i+32)` of the first `n_rot=64` of 256 dims, theta 1e7, cuh:2096–2117 — + extraction of the per-head `[q(256)|gate(256)]` INTERLEAVED halves from the `[tok][head][2*hd]` q-projection, cuh:2048–2052; gate_out = separate RAW f32 buffer, never activated) → k/v gemvs → `kernel_q35_k_norm_rope_kv` (k_norm + same prefix rope + f16 KV store at the page-table row `kv_row(pages, session.n_written + t)` — the kernel takes the session's `pages` buffer like ds4's store_kv (cuh:2163–2168); v copied raw f16) → causal GQA attention (16Q/4KV, head_dim 256, scale 1/√256, span `[0, n_written+n_tok)`, KV read through the same `kv_row` indirection, cuh:2150–2158): `kernel_q35_attn_prefill` iff `n_tok >= 16` (QT gate mirrored from cuh:3332), else `kernel_q35_attn_decode` (grid `(n_head, n_tok)` — handles multi-row) → `kernel_q35_attn_gate_mul` (`attn *= sigmoid(gate)` after attention, BEFORE o_proj, cuh:3266–3269) → attn_output gemv. ⚠ NEVER dispatch T4's kept `kernel_dsv4_rope_tail_*` family — dsv4 tail-layout rope (dsv4_rope.metal:99–136), wrong for qwen35. Names/args per KERNELS.md (pages-less KV signature → STOP and report, Step 4.5).
- [ ] **Step 7: `ffn.rs`.** rmsnorm(post_norm) → gate/up gemvs → swiglu → down gemv. One function, ~80 LoC.
- [ ] **Step 8: `qwen35.rs` + `layers.rs`.** Lifecycle: `load` = `GgufFile::open_shared` + `bind_qwen35` + validate `profile.layer_kind` pattern; register every bound tensor via `WeightRegister` (fallback alloc+copy_h2d); build the dtype→kernel dispatch table over {F32, F16, Q8_0, Q4_K, Q5_K, Q6_K} (names = constants citing KERNELS.md entries; any bound dtype outside the table → `CoreError::Unsupported` at load, never a silent fallback): multi-row projections use `kernel_mul_mm_{q4_K,q8_0,f16}_f32` where the dtype has an mm (n_tok ≥ 2), else row-loop the ext gemv `r1_5` variant — q5_K/q6_K have NO mm kernel. Scratch sized for `MAX_PREFILL_CHUNK = 512` rows (qkv 8192, inter 9216, hidden 2560, logits `[batch+1, vocab]`). `teardown` frees device buffers in reverse construction order (T2 semantics). Forward (layers.rs): embed = `kernel_get_rows_q6_K_f32` on `token_embd`; per layer `x += blk(rmsnorm(input_norm, x))` then `x += ffn(rmsnorm(post_norm, x))` with blk = gdn/attn by `LayerKind`; final `rmsnorm(output_norm)` → lm_head tied to `token_embd` → row-loop `kernel_mul_mv_ext_q6_K_f32_r1_5` (no q6_K mm exists — T5's mm set is q4_K/q8_0/f16). Model impl VERBATIM (T2 SSOT):
  ```rust
  fn load(&mut self, backend: &dyn Backend, model_path: &std::path::Path, profile: &ShapeProfile) -> Result<()>;
  fn open_session(&mut self) -> Result<SessionId>;
  fn close_session(&mut self, session: SessionId) -> Result<()>;
  fn prefill_chunk(&mut self, session: SessionId, tokens: &[u32], is_last_chunk: bool) -> Result<Option<DeviceView>>;
  fn decode_batch(&mut self, sessions: &[SessionId], tokens: &[u32]) -> Result<DeviceView>;
  fn sample(&mut self, logits: &DeviceView, row: u32, req: &mut SampleRequest) -> Result<u32>;
  fn teardown(&mut self) -> Result<()>;
  ```
  Semantics (T2): mid-chunk `prefill_chunk` advances state and returns `None`; `is_last_chunk` returns `Some(view)` of the last-token logits row. `decode_batch` requires `sessions.len()==tokens.len()`, advances each session one step, writes row i into the shared `[n, vocab]` logits buffer, returns its view. `sample` = `copy_d2h` of row `row` (993 KB) → `sampling::sample_logits` (advances `req.rng`).
- [ ] **Step 9: `shisu-bench` runner.** Create `shisu-bench/src/bin/qwen35.rs`: load `SHISU_TEST_MODEL`, prefill a fixed 64-token prompt, decode 128 tokens greedy, time both phases, write/merge `shisu-bench/baseline/m1max-qwen35.json` with fields: `machine` ("Apple M1 Max"), `os`, `date`, `git_rev`, `model` (file name), `backend` ("metal"), `ctx`, `prefill_tokens`, `prefill_tokens_per_sec`, `decode_tokens`, `decode_tokens_per_sec`, `peak_rss_bytes` (0 when not externally measured). T6 records; T20 gates regression.

## Tests

macOS; model+GPU tests `#[ignore]`. `SHISU_TEST_MODEL` defaults to `/Users/sercand/models/Qwen3.5-4B/unsloth/Qwen3.5-4B-Q4_K_M.gguf` (present locally — no download); skip with a printed note when that file is absent or `!metal_available()`. Run: `cd shisu && cargo test -p shisu-engine` then `… -- --ignored`.

- [ ] `tests/sampling.rs` (always runs — no model, no GPU; THE bit-exactness proof). Method: port-first, expected values hand-traced from the C source (ds4 is NOT compiled; the PRNG is exact integer arithmetic so vectors are hand-computable; softmax cases use logits chosen so `expf` is exact). Cases:
  1. PRNG stream: seeds 0 (→ seeded `0x9e3779b97f4a7c15`), 1, 42 → first 8 `rng_next` u64s + `rng_f32` values, hand-traced from :40942–40955; assert exact + state-after.
  2. argmax: tie between equal maxima → lower id; all-`NEG_INF` → 0; unrolled tail (n not multiple of 8).
  3. Uniform logits (all equal, temp 1, top_p 1): probs = 1/n exactly (`expf(0)=1`, exact f32 sum) → r = u·n lands in a hand-computable bucket → exact expected token for 3 seeds; rng advanced exactly once.
  4. top_k=2 + top_p=0.5 + min-p cutoff boundaries (walk the :41495–41512 filter by hand); temperature=0 → argmax and rng UNCHANGED; `top_k>1024` clamp; non-finite entries skipped.
  5. `build_probabilities`/`sample_probabilities` round-trip: sums to 1 within 1e-6, filtered-out ids exactly 0.
- [ ] `tests/sessions.rs` (always runs): FakeBackend (T2 contract-test pattern) — open/close frees buffers; `prefill_chunk` mid → `None`, last → `Some`; `decode_batch` row i ↔ sessions[i] mapping; position overflow → typed error; teardown reverse-order frees; `load` backend-identity mismatch → `Backend` error.
- [ ] `tests/qwen35_decode.rs` (ignored; macOS + `SHISU_TEST_MODEL`):
  1. **greedy 64-token golden**: fixed prompt "The capital of France is" encoded via dev-dep `tokenizers` + `tokenizer.json` copied at implementation time from `/Users/sercand/models/Qwen3.5-4B/mlx/tokenizer.json` into `shisu/test-vectors/` (local file, no download; skip with a note when absent); assert the exact 64-token id sequence equals `shisu/test-vectors/qwen35-greedy-64.json` (`{model_bytes: 2740937888, prompt, tokens[64]}`), recorded ONCE at implementation time with the decoded text human-checked for coherent English (greedy + bit-exact sampler ⇒ deterministic; later runs prove no regression). Also assert no EOS (248044) in the 64.
  2. **memory ≤ 6 GiB**: run the Step-9 bench binary as a child under `/usr/bin/time -l`, parse "maximum resident set size" (bytes), assert `< 6_442_450_944`. (External measurement — no libc dep added.)
  3. **tok/s**: same run prints decode tok/s; the executor records it into `shisu-bench/baseline/m1max-qwen35.json` (recorded, not asserted).
- [ ] Expected before implementation: `cargo test -p shisu-engine` fails to compile (stub crate); after: unit tests green, ignored tests green on this M1 Max with the model present.

## Acceptance

- [ ] `shisu-engine/src/{qwen35.rs,layers.rs,gdn.rs,attn.rs,ffn.rs,sampling.rs,sampling_probs.rs,kv.rs,sessions.rs}` exist, each < 1500 LoC (`wc -l`).
- [ ] `Qwen35Model` implements the T2 `Model` trait verbatim (file inspection vs `task-02-core.md` Produces); object-safe (`Box<dyn Model>` compiles in the sessions test).
- [ ] Sampler port cites the verified line ranges (§40819–41513, §73705–73719 — NOT §70205); stage order, clamps, single-draw-per-sample, tie→lower-id, NEG_INF sentinel, xorshift64* constants all match the C source; `sampling.rs` has zero unsafe and zero deps beyond std.
- [ ] GDN state conventions: conv window `[kernel-1][channels]`, recurrent per-V-head D×D transposed, qk head `h % 16`; attention gate split INTERLEAVED per head and `attn*sigmoid(gate)` before `attn_output` (cuh refs in code comments).
- [ ] Every kernel name in the dispatch table appears in `metal/qwen35/KERNELS.md`; no invented names; unsupported bound dtype fails at load.
- [ ] Rope = PREFIX pairs `(i, i+32)` via `kernel_q35_q_norm_rope`/`kernel_q35_k_norm_rope_kv`; grep finds no `dsv4_rope_tail` dispatch in `shisu-engine/`; KV pages are f16.
- [ ] KV = the Task 7 cache: no contiguous per-session KV allocation anywhere in `shisu-engine/`; `kv.rs` implements `shisu_kvstore::Backing` over the 16 pool buffers; KV-touching kernels take the session page table (`kv_row` seam, cuh:2150–2158); geometry numbers appear only in the documented derivation citing task-07 §Step 2.
- [ ] `cargo test -p shisu-engine` green; `-- --ignored` green on macOS with the default local `SHISU_TEST_MODEL` (greedy golden, RSS < 6 GiB, tok/s recorded).
- [ ] `shisu-bench/baseline/m1max-qwen35.json` exists with the schema fields above.
- [ ] No Metal/CUDA deps in `shisu-engine/Cargo.toml` (dev-deps excepted); no `ATLAS_`/`DS4_` env reads; no license headers.

## Commit

```sh
git add shisu/crates/shisu-engine shisu/crates/shisu-bench shisu/test-vectors/qwen35-greedy-64.json shisu/test-vectors/tokenizer.json shisu/Cargo.toml shisu/Cargo.lock
```
