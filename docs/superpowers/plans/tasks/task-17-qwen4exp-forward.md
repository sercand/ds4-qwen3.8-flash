# Task 17: shisu-engine qwen4exp forward

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 17 (269–275)
+ § CuMetal dev lane (39–74, hard-limits table 57–70) + crate map/SSOT (76–126) +
Roadmap Wave 3a/3b (302–319) + verification log (323–331). Read all five before executing.
**Depends on:**
- T2 (`task-02-core.md` 38–49, 55–61, 149–157): the `Model` trait verbatim — `load(&mut self,
  &dyn Backend, &Path, &ShapeProfile)`, `open_session`, `close_session`, `prefill_chunk(→ Option<DeviceView>)`,
  `decode_batch(→ DeviceView)`, `sample(&DeviceView, row, &mut SampleRequest)`, `teardown`;
  `ShapeProfile::Qwen4Exp(Qwen4ExpShape)` (`qwen4exp()` fixed table: 48 layers, hidden 2560, vocab 248320,
  gqa{24,2,256,64}, gdn{48,16,128,4}, 512/10/1 experts, n_hc, ple_layer, indexer params; `layer_kind(il)`
  = Attn iff `(il+1) % 4 == 0`); `q4e_page::{DS4_Q4E_PAGE_TOKENS=256, SHIFT=8, MASK=255, MROPE_BACK=4, kv_row}`.
- T3 (`task-03-gguf.md` 13–90, 141–167): `GgufFile::open_shared`, `TensorDesc{dtype,shape,abs_offset,bytes}`
  (bytes incl. exl3 scale vectors), `quant::exl3::{k_bits, tile_bytes, scale_bytes}`, `block_bytes_for`;
  DEVIATION 4 — split GGUF (`SplitUnsupported`) until the CUDA phase: qwen4exp GGUFs ship split, so full-model
  load on Linux needs that gap closed first (`[INFERENCE]` — no q4e GGUF on this Mac to test either way).
- T6 (`task-06-engine-qwen35.md` 11–31, 86–97): `sampling.rs` (`sample_logits`, `rng_next/f32`, `argmax`)
  and `sampling_probs.rs` (`build_probabilities`, `sample_probabilities`) — **reused verbatim, zero new
  sampler code**; `sessions.rs` `SessionStore` id/lifecycle machinery (q4e sessions carry their own state
  struct — T6's qwen35 KV/GDN buffers are not reused); `kv.rs`'s single-seam addressing pattern is what
  `paged_kv.rs` implements for real.
- T15 (`task-15-q4e-extraction.md` 110–122, 253–264, 310–340): `kernels/cuda/q4e/KERNEL_NAMES.md` = the
  50-entry name SSOT **and** the grid/block/smem formula rows lifted from ds4's host launchers — T17 computes
  launch geometry from those rows; the 6 NUMERIC-INVALID kernels + SMEM_BLOCKED rows bound the oracle set.
- T16 (`task-16-cuda-backend.md` 23–60, 128–133, 210–223): `CudaBackend::new(ordinal, decode_graphs)`,
  `set_tf32`, `gemm_ex`/`gemm_strided_batched_ex` (classic cuBLAS = ds4-parity tier), `decode_graphs_supported/
  begin/end/abort/invalidate` with `DecodeGraphKey{il, island, variant, cur_hc, after_attn_hc, after_ffn_hc,
  attn_norm}` + tri-state `DecodeGraphStep{Replay, Capture, Eager}`, `max_co_resident_blocks`,
  `reserve_stream_scratch`; `KernelRef{name}` may carry `#<idx>:t<val>` (template NT sets resolve through the
  KERNEL_NAMES.md table); `module == "exl3"` auto-dispatches cooperative launch; grid = block-count
  passthrough, no ceil/division in the backend.

**Produces** (paths `shisu/crates/shisu-engine/src/qwen4exp/`; engine stays pure Rust — it talks only to
`dyn Backend`, so every step below compiles and unit-tests on macOS against T2's `FakeBackend`):

```rust
// mod.rs
pub struct Qwen4ExpModel { /* Arc<dyn Backend>, Q4eApi, binding, graph buffers, pool/tree, sessions, mtp */ }
impl Qwen4ExpModel {
    pub fn new(backend: std::sync::Arc<dyn Backend>, q4e: std::sync::Arc<Q4eApi>) -> Self;
    pub fn with_weight_register(self, f: crate::qwen35::WeightRegister) -> Self; // T6 pattern
    pub fn with_ctx(self, cap: u32) -> Self;
    pub fn with_spec(self, k: u32) -> Self;              // 0 = plain decode (ds4 spec_k)
}
// mod.rs — engine-side mirror of T16's inherent surface (T2 traits frozen; T6 WeightRegister precedent:
// a closure bundle injected at construction; the T10 factory maps it to CudaBackend field-for-field,
// FakeBackend implements it in tests). Key/step are structural clones of DecodeGraphKey/DecodeGraphStep.
pub struct Q4eApi {
    pub graphs_supported: Box<dyn Fn() -> bool + Send + Sync>,
    pub graph_begin: Box<dyn Fn(&Q4eGraphKey) -> shisu_core::Result<Q4eGraphStep> + Send + Sync>,
    pub graph_end: Box<dyn Fn(&Q4eGraphKey) -> shisu_core::Result<()> + Send + Sync>,
    pub graph_abort: Box<dyn Fn(&Q4eGraphKey) + Send + Sync>,
    pub graphs_invalidate: Box<dyn Fn() + Send + Sync>,   // + gemm_ex / max_co_resident_blocks / reserve_stream_scratch
}
pub struct Q4eGraphKey { pub il: u32, pub island: u32, pub variant: u32,
    pub cur_hc: DeviceView, pub after_attn_hc: DeviceView,
    pub after_ffn_hc: DeviceView, pub attn_norm: DeviceView }   // attn_norm slot = kv_pages table
pub enum Q4eGraphStep { Replay, Capture, Eager }
impl shisu_core::Model for Qwen4ExpModel { /* the 7 T2 methods, verbatim — see Step 10 */ }
// paged_kv.rs (pub(crate))
pub(crate) struct PagePool { free: Vec<u32>, refs: Vec<u32>, .. }   // ds4.c:55129–55183
pub(crate) struct SpanTree { root: NodeId, .. }                     // ds4.c:55142–55161
pub(crate) fn page_bytes(idx: bool, mtp: bool) -> u64;              // derived, ds4.c:56012–56026
pub(crate) fn kv_reserve(&mut self, n_pos: u32) -> Result<()>;      // grow page table ⇒ graph invalidate
// mtp.rs (pub(crate))
pub(crate) fn spec_step(&mut self, sess: SessionId, first: u32, eos: u32,
                        accepted: &mut Vec<u32>, req: &mut SampleRequest) -> Result<usize>;
// ple.rs (pub(crate)) — the T19 seam
pub(crate) fn ple_gather_start(&mut self, history: &[i32], pos0: u32, n_tok: u32) -> PleGather;
pub(crate) fn ple_block(&mut self, handle: PleGather, il: u32, n_tok: u32) -> Result<()>;
// hc.rs / qsa.rs / gdn.rs / moe.rs / matmul.rs / exl3_host.rs — pub(crate) per-op drivers,
// each fn mirrors one ds4 host function (Source table); no public API escapes mod.rs.
```

## Global Constraints

1–7 from master plan (30–37). Task-specific:
1. 1500-LoC cap — far from binding here; the 7-file list is split to 10 on module-seam grounds (⚠ DEVIATION 4).
2. Arithmetic-order changes are bugs: every per-op driver reproduces ds4's launch ORDER and argument values
   exactly; geometry formulas are transcribed from KERNEL_NAMES.md rows (T15), never re-derived.
3. The sampler is NOT re-ported. `sample()` = `copy_d2h` row → `crate::sampling::sample_logits` (bit-exact
   port, T6). Spec verify uses the greedy argmax path exactly as ds4 does (⚠ DEVIATION 1).
4. Engine depends only on `shisu-core`/`shisu-gguf`/`tracing`/`parking_lot` — no cudarc/shisu-cuda import;
   the T16-specific calls (`gemm_ex`, decode-graph cache, coop support) arrive as the injected `Q4eApi`
   closure bundle (T6's `WeightRegister` precedent; T2 traits stay frozen). The T10 factory builds it from a
   `CudaBackend`; tests build it from T2's `FakeBackend` (records launches + drives the tri-state).
5. No new env knobs (rule 5): ds4's `DS4_QWEN4EXP_*` knobs (spec K, verify graphs, GDN chunk min, MoE batch
   min) become constructor/builder parameters; `SHISU_TEST_MODEL` is the only env read.
6. `warnings = deny`, clippy deny, `CoreError` only, no license headers, no formatters/suites (T20 gate).
7. CuMetal hard-limits (57–70): coop grids + cuBLAS + CUDA graphs + compound smem ⇒ every kernel-numeric
   claim from this Mac is CONDITIONAL (Tests); exl3, blas, decode-graph capture, and full-model smoke are
   Linux-only and appear here only as `[INFERENCE]` + `#[ignore]` tests.

## Source References (verified)

| Source | Lines | What / how used |
|---|---|---|
| `ds4.c` | 67082–67095 | q4e host region header: layer = hc-mix → token mixer (GDN 3/4, QSA 1/4) → combine → mix → MoE+shared → combine; PLE layer folds in before the first mix; no pre-norms — the mixes are the norms. |
| `ds4.c` | 68799–68889 | `q4e_matmul_at` dtype dispatch (f32 split-k matvec, f16 ≤16-row fused / cuBLAS above, q8_0 narrow, bf16, EXL3 → `ds4_gpu_q4e_matmul_exl3`, mmq tier) — `matmul.rs` reproduces the selection table; `LaunchArgs` geometry from KERNEL_NAMES.md. |
| `ds4.c` | 68912–69014 | `q4e_embed` (gather rows), `q4e_hc_mix` (norm? + down/silu/up + inject), `q4e_ple_block` — `hc.rs`/`ple.rs` order. |
| `ds4.c` | 69035–69115 | `q4e_gdn_conv_step` / `q4e_gdn_recurrent_step` / `q4e_gdn_layer` host order (qkv → conv → L2-norm q,k → alpha/beta → gates → recurrent → z → out_gate → out) — `gdn.rs`. |
| `ds4.c` | 69122–69345 | `q4e_qsa_store_step`, `q4e_idx_store_step`, `q4e_qsa_attention_step`, `q4e_qsa_layer` (q/k/v proj → norm+rope → store → idx pipeline → dense-or-sparse attn → gate) — `qsa.rs`. |
| `ds4.c` | 69348–69455 | `q4e_moe` (route → sort → gate/up → swiglu → down → combine) + `q4e_moe_shared` — `moe.rs`. |
| `ds4.c` | 69457–69585 | island encoders: island 0 = [PLE block] + hc_mix(attn) + QSA-or-GDN + hc_combine(next=ffn_norm); island 1 = hc_mix(ffn) + MoE + hc_combine(next=next layer's attn_norm); xn-chain breaks at first island / PLE island / last layer (69519–69527); `q4e_run_island` tri-state loop + key: `variant = n_tok + (qsa_sparse?64:0)`, `attn_norm = kv_pages` (per-context identity, 69556–69569). |
| `ds4.c` | 69591–69660 | `q4e_ple_gather` / `_gather_batch` / `_prefetch` — host gather at the PLE layer's head so disk reads hide behind already-enqueued layers (69761–69764) — the T19 seam shape. |
| `ds4.c` | 69664–69828 | `q4e_forward`: kv_reserve(pos0+n_tok) → ids/pos/mrope upload (MROPE_BACK prefix rows) → embed → hc_init → `qsa_sparse = idx_ready && (pos0+n_tok)/4 > TOP_K/4` (69752) → graphs_ok gate (69753–69757) → 48×2 islands → output mix collapse → vocab proj on last rows → argmax_rows for verify. |
| `ds4.c` | 69847–70001 | `q4e_forward_batch`: n sessions × 1 row each, per-row frontier positions, shared scratch — `decode_batch` semantics. |
| `ds4.c` | 70025–70220 | `Q4E_MTP_SLOT = N_LAYER-1`; `q4e_mtp_draft`: draft shares the target's page table, reserves rows above its own frontier; fc over `norm(embed)‖norm(hidden)` → hc_dim up-projection. |
| `ds4.c` | 70222–70307 | `q4e_mtp_draft_rows` / `pend_set` / `flush_pending` / `after_prefill_chunk`; `q4e_spec_rollback` = copy back rows [0,keep) of every GDN conv+state and the PLE conv window. |
| `ds4.c` | 70332–70532 | `q4e_spec_step`: K clamps → MTP flush + chain drafts (stop at eos) → n-gram proposal believed ONLY if longer than MTP-K AND first token agrees (70411) → verify `q4e_forward(1+K, 1+K)` → greedy argmax-prefix accept (70461–70466) → rollback → logits row `a` → pend rows staged for next step. |
| `ds4.c` | 54928–55036 | `Q4E_SPEC_MAX_DRAFT 8`, n-gram drafter: `Q4E_NGRAM_MAX_MATCH 6`, min-match knob, special-token floor, 256-entry fail ring keyed by (src, hash). |
| `ds4.c` | 55123–55183 | pool/tree structs: `q4e_page_pool{free_ids,refs}`, `q4e_tree_node{parent,child,next,start,end,tok,pages,pg_lo,pg_hi,ckpt,live,hit}`, `q4e_span_tree{pool,root,nodes,page_evictions}`; reverse-order page init as the non-contiguity test (55166–55173). |
| `ds4.c` | 55629–55638 | host shorthands: `Q4E_HC_DIM = EMBD*HC`, `Q_DIM`, `KV_DIM = KV_HEADS*HD`, `GDN_K/GDN_V/GDN_IN = 2K+V`, `PLE_HIST`, `IDX_QB 512`, `IDX_WIDTH 2051`. |
| `ds4.c` | 55917–55955 | ckpt budget (40 default/256 max), pool caps (resident 4, 600k tokens), admission stride 2048 — sizing policy constants for `paged_kv.rs`. |
| `ds4.c` | 56008–56043 | `q4e_page_bytes(idx,mtp)` DERIVED: 12 attn × 2 × 256 × 512 × 2 B = 6,291,456; +idx 12×(256+64)×128×2 = 983,040; +mtp 2×256×512×2 = 524,288; +mtp-idx 81,920 ⇒ **7,880,704 B/page** (header's "30 KiB/pos" = T2's nominal 7,864,320 — use the derived fn, not the const). `q4e_ckpt_bytes` = Σgdn(conv+state) + PLE window + 1 hc row + vocab logits. |
| `ds4.c` | 67846–67978 | payload: `q4e_payload_state_bytes` + `q4e_payload_state` serialize **GDN conv+state per layer, PLE conv window, pending draft row** (+ logits/pages/tokens) — answers T8's open question: qwen4exp resume is NOT KV-only. |
| `ds4.c` | 7551–7660 | MTP weight binding: `q4e_mtp_split_eh_proj` (eh `[2*embd, embd]` → fc split) + `Q4E_MTP_BIND` name table (enorm/hnorm/fc/pre_fc, shared-vs-sidecar head names). |
| `ds4.c` | 40819–41513 | sampler region boundaries re-verified: `argmax_f32_unrolled8_range` 40819, xorshift64* 40942–40955, `sample_build_probabilities` 41010–41138, `sample_fast_top_p` 41192–41278, dispatcher ends 41513. **T6's `sampling.rs`/`sampling_probs.rs` already port this — T17 reuses, adds nothing.** |
| `ds4_gpu.h` | 3132–3158 | decode-graph key + tri-state contract (replay=1/capture=0/eager=−1) — T16's `DecodeGraphStep` mirrors it; header comment's island wording is generic, the REAL semantics are ds4.c:69457–69534 (island 0 = attn island, 1 = ffn island). |
| `ds4_gpu.h` | 3161–3275 | the ~40 `ds4_gpu_q4e_*` entry points = the flat API `Backend::run` + `gemm_ex` replace; matvec f32 (3178), f16 ≤16 fused (3185), exl3 + pair (3189–3190), moe_exl3_fused "0 = take staged path" (3273), moe_matmul (3275). |
| `ds4_cuda.cu` | 953–1050 | decode-graph design: warm→capture→replay→dead, byte-identical replay claim, 48-byte key static_assert, 64×2×32 table, `invalidate` clears all (1035–1050). |
| `ds4_cuda.cu` | 1052–1186 | find = linear scan of 32 slots + memcmp; table-full ⇒ eager forever + warn-once; begin: warm −1 / capture 0 (pre-bind cublas stream + exl3 `prepare_stream` — T16 graph.rs does this) / replay 1; end: EndCapture→Instantiate→first-launch. |
| `ds4_qwen4exp_gpu.cuh` | 61, 450, 604–606, 1281, 1393, 3430 | `Q4E_HC_FUSE_MAX 16`, `Q4E_GDN_SPLIT 4`, chunk 64/qtile 32/csplit 8, f16 partial slab, `Q4E_EXL3_RECONSTRUCT_ROWS 144`, MoE batch-min knob — host-side thresholds T17 re-implements. |
| `ds4_qwen4exp_gpu.cuh` | 1395–1520 | `ds4_gpu_q4e_matmul_exl3` (+pair): payload = tiles+suh+svh; `n_tok > 144 && cublas && !capturing && dims%128==0` ⇒ reconstruct-to-f16 + `cublasGemmEx` (1416–1437), else coop `ds4_exl3_gemm`; `q4e_moe_exl3` = mgemm fan-out with per-slot row replication (1460–1493) — `exl3_host.rs`. |
| `ds4_qwen4exp_gpu.cuh` | 1935–1937, 2214–2215, 2304–2306, 2444–2447, 2727–2729, 2868–2869, 3072 | GDN chunk gate (`n_tok ≥ chunk_min && n_tok > 1+ckpt_cap`); attn lane/warp, QT/KT tiles, sparse split geometry (QG 16, GSPLITS 48→8, PART_STRIDE 264), idx DIM 128/R 4, score tiles 16×64 — the constants the geometry formulas cite. |
| `ds4_q4e_page.h` | 1–47 | page geometry + rationale (256 = lcm of pooled-block 4, key tile 32, score tile 64); `kv_row` translation; MROPE_BACK 4 rows in front of every mrope table upload. |
| `cuda/exl3/ds4_exl3.cu` | 148–387 | the host rules T16 DEVIATION 4 left to T17: `exl3_select_shape` (159–186: K∈{4,2}&&!multi&&k_eff≤2048→1; K≥7 mod256 n_eff≤8192→2/3; mod512 wide→4; fallback scan by tile divisibility, 0 = unsupported); `check_shape` (k%16, n%128); gemm blocks `clamp(4·tiles_n, 8, min(g_max_blocks, tiles_k·tiles_n))` (214–218); mgemm `dim3(per_group, 1, concurrency)` with per-group = max_blocks/concurrency capped by tiles (261–267); moe groups of SM-per-expert blocks + staging sized `groups·n_tok·row_bytes`, never allocate during capture (340–366). |
| `tests/test-vectors/` | listing | only glm/flash sampler vectors exist — NO qwen4exp golden material; T18 owns generating q4e parity vectors. |
| task plans | T2 38–61, 149–157; T3 13–90, 161–167; T6 11–31, 63, 86–97; T15 110–122, 253–264, 310–340; T16 23–60, 128–133, 210–223 | trait/shape/page consts; exl3 descriptors + split gap; sampler reuse surface + sessions pattern; KERNEL_NAMES rows + oracle eligibility; T16 surface + coop dispatch rule. |

⚠ **DEVIATION 1 (master-plan sampler cite is wrong — re-confirmed):** master 187/272 say sampling is
"§70205+". Verified: the sampler is §40819–41513 (boundaries opened above); §70205 sits inside the MTP
pend/rollback region. T6 already corrected this (task-06:63) and shipped the port; T17 adds ZERO sampler
code — `sample()` delegates to `sampling::sample_logits`, and spec verify uses `argmax` (ds4's verify is
greedy argmax-prefix, ds4.c:70461–70466; `sampling_probs.rs` stays available for stochastic paths only).

⚠ **DEVIATION 2 (master-plan MTP + paged-KV cites are wrong):** master 272 says "MTP speculative decode
(ds4.c §67082-70205)" and "paged-KV pool + span-tree host bookkeeping (ds4.c §70205+ host part)". Verified:
§67082–70205 is the ENTIRE q4e host forward region (region header 67085, matmul dispatch 68799, layer fns
68912–69455, islands 69457–69585, forward 69664, forward_batch 69847, MTP starts 70025); MTP/spec continues
to **§70532** (`#endif /* DS4_NO_GPU */` at 70533). The paged-KV pool + span tree are NOT at §70205+ — they
are §55123–55183 (pool/tree), §55629–55638 + §55917–55955 (geometry/caps), §56008–56043 (page/ckpt bytes),
§67846–67978 (payload); §70205+ is MTP's tail (pend/flush/rollback/spec_step) + `ds4_session_create`.

⚠ **DEVIATION 3 (ds4_cuda.cu is NOT the q4e forward driver):** `grep ds4_gpu_q4e_ ds4_cuda.cu` = 0 matches.
The forward driver whose ORDER T17 reproduces is `ds4.c` §68799–70532; the `ds4_gpu_q4e_*` host launchers
(the flat API `Backend::run` replaces, and the source of the launch-geometry rules) live in
`ds4_qwen4exp_gpu.cuh` (e.g. 1395–1441); `ds4_cuda.cu` carries only the decode-graph machinery (953–1186),
which T16 already ports. Order ⇒ ds4.c; geometry ⇒ KERNEL_NAMES.md; graph protocol ⇒ T16.

⚠ **DEVIATION 4 (7 files → 10, additive — design seams, not cap-forced):** three files join the master
list on design grounds, not the 1500 cap: `matmul.rs` (dtype dispatch §68799–68889 is shared by every op
file — a shared file beats duplication), `exl3_host.rs` (T16 DEVIATION 4 explicitly assigns shape selection
159–196 + geometry 197–387 to T17), `ple.rs` (the T19 wiring seam — without it the placeholder pollutes
mod.rs). The 7 named files stay; these 3 join.

⚠ **NOTE (T8 answer, from source):** the q4e checkpoint payload serializes the GDN conv window + recurrent
state per layer, the PLE conv window, and the pending MTP draft row (ds4.c:67869–67877, 67954–67978) —
qwen4exp resume is NOT KV-only; T8's plan must extend its payload to these buffers when it wires q4e.

## Plan

- [ ] **Step 0 — Preflight.** Read `kernels/cuda/q4e/KERNEL_NAMES.md` (T15) end-to-end: every kernel T17
  launches must exist there with a geometry row; a missing row = STOP (T15 gap, not T17's to invent).
  Confirm `sampling.rs`/`sampling_probs.rs`/`sessions.rs` landed (T6) and `CudaBackend` exposes the
  Produces surface (T16). No new kernel names, ever.
- [ ] **Step 1 — `paged_kv.rs` (≤460 ln).** `PagePool` (free list + refs, reverse-init option 55166–55173),
  `SpanTree` (node fields verbatim 55143–55154; insert/split on divergence, `live` refcounts, LRU `hit`
  decay evicting leaves), `page_bytes(idx,mtp)` derived per 56012–56026 (assert 7,880,704 with both flags),
  `kv_reserve(g, n_pos)` = `q4e_pages_for` + table grow. **Invalidate rule:** captured graphs bake the page
  table pointer (`key.attn_norm = kv_pages`, 69569) — any table realloc/grow calls
  `decode_graphs_invalidate()` before publishing the new pointer.
- [ ] **Step 2 — `matmul.rs` (≤260 ln).** Port `q4e_matmul_at` dispatch (68799–68889): per bound `DType`
  pick {f32 matvec + combine, f16 rows ≤16 via `#n:t<val>` templates else `gemm_ex`, q8_0 narrow rows,
  bf16 rows, exl3 → Step 3, mmq prefill tier}; buffer-size guards mirror 1400–1406 (typed error, not stderr).
- [ ] **Step 3 — `exl3_host.rs` (≤320 ln).** Rust port of the host rules: `select_shape` table + tile
  fallback scan (159–186), `check_shape` (k%16/n%128), gemm/mgemm/moe block math (214–218, 261–267,
  340–347), reconstruct-vs-trellis decision (cuh 1416: `n_tok > 144 && !capturing && dims%128==0` ⇒
  reconstruct kernel + `gemm_ex` f16), moe staging via `reserve_stream_scratch`, occupancy via
  `max_co_resident_blocks`; launches go through `Backend::run` with `module:"exl3"` (coop dispatch is
  T16's rule). All numerics `[INFERENCE]` until Linux.
- [ ] **Step 4 — `hc.rs` (≤220 ln).** `hc_init`/`init_add`, `hc_norm` (xn), `hc_mix` (norm?→down→scale_silu
  →up→inject add), `hc_combine` (fused residual update + next-island group-norm; `xn_ready`/`next_norm`
  chain per 69519–69527), `hc_collapse` (output mixer). Fuse cap `Q4E_HC_FUSE_MAX 16`.
- [ ] **Step 5 — `gdn.rs` (≤360 ln).** `q4e_gdn_layer` order (69086–69115): qkv matmul → conv (window
  `(CONV-1)×GDN_IN` f32, stateful shift) → l2norm q,k → alpha/beta matmuls → gates → recurrent (n_tok ≤
  chunk gate) or chunk kernel iff `n_tok ≥ chunk_min && n_tok > 1 + ckpt_cap` (cuh 1935–1937; chunk_min a
  builder param, ds4's knob) → out_gate → out matmul. State sizing: conv + `V_HEADS×D×D` f32 per GDN layer,
  FIXED per execution context (not per position) — matches the payload geometry (67871–67873).
- [ ] **Step 6 — `qsa.rs` (≤460 ln).** `q4e_qsa_layer` (69208–69296): q proj → q_norm_rope (mrope table with
  MROPE_BACK prefix rows) → k/v proj → store_kv (f16, rows via `q4e_page::kv_row`) → idx pipeline
  (store_k → pool r=4 → idx_q → score (16×64 tiles, `IDX_WIDTH 2051`) → topk → expand) → attention: dense
  warp/tiled (`n_tok ≥ Q4E_ATTN_QT 16` ⇒ tiled) or sparse split→gather→combine when `qsa_sparse`
  (69752 rule) → gate kernel. Sparse params: QG 16, GSPLITS 8, PART_STRIDE 264.
- [ ] **Step 7 — `moe.rs` (≤420 ln).** route (softmax top-10/512) → sort count/scan/scatter → gate+up via
  `exl3_host` mgemm-pair (or staged matmul per dtype; `moe_exl3_fused` "0 ⇒ staged" contract 3272–3273) →
  swiglu → down: grouped templates `#7:t<NT>`/`#8:t<NT>` (KERNEL_NAMES rows) or mgemm; combine + shared
  expert (`matmul_exl3_pair` + `shared_add`); batch-min knob (cuh 3430).
- [ ] **Step 8 — `ple.rs` (≤140 ln).** Placeholder that T19 fills without restructuring: `PleGather` handle
  (host row reads + H2D), `ple_gather_start` issued by mod.rs **before the layer loop** (ds4 issues it at
  the PLE layer's head, 69765–69769, to hide reads behind enqueued layers — the handle lets T19 start even
  earlier and await at the same site), `ple_block` = dequant → gated value → conv (window
  `PLE_HIST×HC_DIM` f32) into the residual. Weight names bound via T3 with ds4's `ple.*` table.
- [ ] **Step 9 — `mtp.rs` (≤460 ln).** Bind via T3 using the 7551–7660 name table (incl. `split_eh_proj`);
  pend rows (`pend_set`/`flush_pending`, hc_dim rows), `q4e_mtp_draft` (shares the target's page table,
  reserves rows above its frontier — 70043–70057), n-gram drafter (54928–55036: max-match 6, fail ring,
  special-token floor), `spec_step` per 70332–70532: K clamps (`Q4E_SPEC_MAX_DRAFT 8`, ctx, accepted_cap) →
  flush + chain (stop at eos) → ngram gate (longer AND first-token agreement) → verify forward `1+K` rows
  with `1+K` logit rows → greedy argmax-prefix accept → `spec_rollback` (copy rows [0,a) of conv/state/PLE
  window) → logits row `a` → pend staging. Verify rows ride the graph variants `n_tok = 2..9` only when
  `with_verify_graphs` is on (ds4 keeps it opt-in, 69742–69749).
- [ ] **Step 10 — `mod.rs` (≤480 ln).** `load`: `open_shared` + per-layer bind (T3 names; attn iff
  `layer_kind(il)==Attn`) + MTP bind + `WeightRegister`/`copy_h2d` + graph buffers (tokens/positions/mrope
  with MROPE_BACK, res/mixed/xn/blk_out/inject, logits, argmax dev+host) + per-context state (GDN conv/
  state, PLE window, pend). `open_session` = claim root span + zero state. `prefill_chunk`: chunk loop over
  `q4e_forward` body (embed → hc_init → 48×2 islands → collapse → logits iff last chunk). `decode_batch`:
  `forward_batch` semantics (row r = session r's frontier token). `sample`: d2h row → `sampling::sample_logits`.
  Island wrapper = `q4e_run_island` exactly: build `Q4eGraphKey{il, island, variant = n_tok +
  (qsa_sparse?64:0), cur_hc: res, after_attn_hc: blk_out, after_ffn_hc: mixed, attn_norm: kv_pages}` →
  `q4e.graph_begin` → `Replay` ⇒ done / `Capture` ⇒ encode, `graph_end`, on err `graph_abort` + eager / `Eager` ⇒
  encode; `graphs_ok` gate per 69753–69757. `teardown`: sessions → pool → buffers, reverse order.
- [ ] **Step 11 — Wire-up.** `qwen4exp/mod.rs` is the module root (`mod hc; mod qsa; …`); `shisu-engine/src/
  lib.rs` gains `pub mod qwen4exp;`; the T10 factory maps `ShapeProfile::Qwen4Exp` → `Qwen4ExpModel`.

## Tests

macOS; everything except the Linux smoke runs against T2's `FakeBackend` (extended to record launches —
kernel name, grid/block/smem, argv — and to serve the `Q4eApi` decode-graph tri-state).

- [ ] `tests/q4e_cpu_oracle.rs` (always; the master's "CPU-oracle per-op, tol 1e-5" item, made honest):
  the CUDA kernels cannot execute on this Mac, so the CPU oracle is a Rust f32 reference per op —
  matvec (f32 split-k + f16/q8_0 rows), hc norm/mix/combine, gdn recurrent step — transcribed from the
  kernel source in arithmetic order, checked against hand-computed cases and self-consistency invariants
  (1e-5). These references are the host-side oracle T18's Linux parity diffs the kernels against; they
  prove the FORMULAS, never claim kernel numerics.

- [ ] `tests/q4e_paged_kv.rs` (always): `page_bytes(true,true) == 7_880_704` and the no-idx/no-mtp variants
  (arithmetic from 56012–56026); `kv_row` spot cases incl. page-straddling positions; span-tree insert →
  divergence split → `live` refcounts → leaf eviction frees pages; `kv_reserve` growth fires the
  invalidate hook exactly once per realloc.
- [ ] `tests/q4e_graph_bookkeeping.rs` (always): launch trace shows 48 layers × 2 islands in ds4's order
  (PLE block inside island 0 of `ple_layer`); key fields per Step 10 (variant 1..16 and +64 sparse);
  `Capture`-then-fail path aborts and re-encodes eagerly without losing the island (69571–69583); xn-chain
  breaks at first island / PLE island / last layer (no norm launch there).
- [ ] `tests/q4e_spec.rs` (always): accept-prefix incl. eos stop; rollback issues exactly the [0,a) copies
  of every GDN conv/state + PLE window (count + offsets from the trace); ngram gate (rejected when not
  longer, rejected on first-token disagreement); K clamps at ctx edge and `accepted_cap`.
- [ ] `tests/q4e_exl3_host.rs` (always): `select_shape` table rows (K=4 !multi k_eff≤2048→1; K=7 mod256
  n_eff≤8192→2, k_eff>32768→3; mod512 wide→4; tile-infeasible fallback and 0); gemm block clamp;
  reconstruct gate flips at n_tok 144/145 and on `capturing`.
- [ ] `tests/q4e_cumetal_oracle.rs` (macOS, opt-in lane, **CONDITIONAL set only**): per-op 1e-5 diffs vs the
  T5 MSL twins / Rust reference, restricted to kernels whose T15 STATUS row is `ok` AND not
  NUMERIC-INVALID: hc (all 6), matvec (all 8), misc (2), idx (6), ple (4), moe except `q4e_moe_route_kernel`,
  gdn conv/l2norm/gates/out_gate, qsa tiled+gate, qsa_sparse split+combine. NEVER oracle'd here (compound
  static+dynamic smem corrupts on the lane — T3b:239): `q4e_moe_route_kernel`, `q4e_gdn_recurrent_kernel`,
  `q4e_qsa_q_norm_rope_kernel`, `q4e_qsa_store_kv_kernel`, `q4e_qsa_attention_kernel`,
  `q4e_qsa_attention_gather_kernel`; plus the D ≤ 64 gate — smem-sized kernels (tiled, chunk) run only at
  toy head_dim ≤ 64, production head_dim 256 numerics are owed by T18 on Linux. Real qwen3.5-4b weight
  slices from `SHISU_TEST_MODEL` give the shapes (master 74).
- [ ] `tests/q4e_model.rs` (Linux GPU, `#[ignore]`): 32-token greedy decode on `SHISU_TEST_MODEL`
  (q4e GGUF) completes; decode-graph replay counters > 0. `[INFERENCE]` — not runnable on this Mac: no
  driver, no q4e GGUF, split-GGUF load still open (T3 DEVIATION 4). No golden assertions — T18 owns parity
  vectors (none exist today; `tests/test-vectors/` holds only glm/flash sampler material).
- [ ] No sampler tests here: T6's `tests/sampling.rs` is the bit-exactness proof; re-running it is T20's gate.

## Acceptance

- [ ] `shisu-engine/src/qwen4exp/{mod,hc,qsa,gdn,moe,mtp,paged_kv,ple,matmul,exl3_host}.rs` exist, each
  < 1500 LoC; engine gains no CUDA/cudarc/shisu-cuda dependency (T16's surface arrives only as the
  injected `Q4eApi` closure bundle).
- [ ] `Qwen4ExpModel` implements T2's `Model` verbatim; `sample` delegates to T6 `sampling.rs` (grep: zero
  sampler math in `qwen4exp/`); spec verify is the greedy argmax prefix per ds4.c:70461–70466.
- [ ] Island semantics byte-faithful: 2 islands/layer with ds4's contents, key = `{il, island,
  variant = n_tok + 64·sparse, res, blk_out, mixed, kv_pages}`, tri-state + abort-retry per 69571–69583;
  page-table growth invalidates the graph cache.
- [ ] Every launched kernel name appears in `KERNEL_NAMES.md`; geometry comes from its rows (no invented
  formulas); exl3 host rules match ds4_exl3.cu:159–387 line-for-line in comments.
- [ ] macOS suite green without a GPU; CuMetal oracle set matches the CONDITIONAL list exactly (6
  NUMERIC-INVALID kernels excluded; D ≤ 64 for smem-sized kernels); Linux-only claims appear solely as
  `[INFERENCE]` + `#[ignore]` tests.
- [ ] No new env knobs; no `ATLAS_`/`DS4_` env reads; no formatters/suites run (T20 owns the gate).

## Commit

```sh
git add shisu/crates/shisu-engine
git commit -m "feat(engine): qwen4exp forward — islands, paged-KV span tree, MTP spec decode, exl3 host tier (Task 17)"
```
