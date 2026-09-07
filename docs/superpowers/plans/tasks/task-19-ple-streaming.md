# Task 19: shisu-ple + integration

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 19 (282–288)
+ § Rules (30–37; knob list 35) + § Crate map (76–126; `shisu-ple` row 114, created in the CUDA
phase per 132 and task-01:45) + § CuMetal lane (39–74) + Roadmap Wave 3b (302–319; T19 row 313) +
verification log (323–331: "`ds4_ple_stream.c` 1159 ln" 327). Read all five before executing.
**Depends on:**
- T2 (`task-02-core.md` 16–37, 69–76): `Backend::copy_h2d(DeviceView, &[u8])` = the H2D seam for
  raw rows (ds4's `ds4_gpu_tensor_write`); `DeviceView{id,offset,len}`; `CoreError`; `DType::Exl3Ngram6=72`
  with (160,122) block geometry (task-02:104) — the GGUF dtype that selects the EXL3 row codec;
  `ShapeProfile::qwen4exp()` PLE fields (n_hc, ple_layer 1, n_ple_ngram 3, heads_per_ngram 8,
  head_dim 160, ple_eos_token 248044).
- T1 (`task-01-workspace-scaffold.md` 45, 213–222): `shisu-ple` deliberately not created until this
  task; `io-uring = "0.7"` already in `[workspace.dependencies]` "behind a linux target table by
  shisu-ple (Task 19)"; `[workspace.lints]` deny set; rust 1.93.1 edition 2024.
- T16 (`task-16-cuda-backend.md` 166–177, 255–262): the cfg mechanism to copy — module swap
  (`#[cfg(target_os = "linux")] #[path=…] mod imp;` + stub arm), not a trait; Linux-only claims
  appear solely as `[INFERENCE]` + `#[cfg(target_os = "linux")] #[ignore]` tests; `SHISU_SKIP_BUILD=1`
  Mac contract.
- T17 (`task-17-qwen4exp-forward.md` 68–73, 198–202): the seam T19 fills — `ple.rs` today is a
  placeholder with `ple_gather_start(&mut self, history, pos0, n_tok) -> PleGather` (issued by
  mod.rs before the layer loop) and `ple_block(handle, il, n_tok) -> Result<()>` (awaits at the PLE
  layer, then dequant→gate→conv into the residual); `Qwen4ExpModel` owns the `ple_rows` device
  buffer (T ≤ 2048 rows × n_heads × row_bytes, ds4.c:68488) and the island loop that calls the two
  hooks; `Q4eApi`/`Backend::run` is the only launch path.
- T15 (`task-15-q4e-extraction.md` 117): `kernels/cuda/q4e/ple.cu` carries the 4 PLE kernels
  (dequant 839, dequant_exl3 867, gated_value 905, conv 928) under module `q4e_ple`; launch
  geometry comes from `KERNEL_NAMES.md` rows — T19 supplies the two dequant launches' args, never
  invents geometry. Kernel numerics belong to T15/T18, not here.
- T18 (`task-18-golden-parity.md` 220–239): parity vectors are T18's; T19 records p99 as an
  OBSERVATION only — T20 owns every perf gate (master 294).

**Produces** (paths `shisu/crates/shisu-ple/`; pure Rust + one Linux-only FFI dep — compiles and
unit-tests on macOS; the crate is created here, added to `shisu/Cargo.toml` members +
`[workspace.dependencies]` path entry, additive hunks only):

```rust
// lib.rs (≤80 ln) — modules codec/handle/cache/pool/prefetch + #[cfg(linux)] uring; re-exports.
// error.rs lives in lib.rs: #[derive(thiserror::Error)] pub enum PleError {
//   InvalidParams(String), Io(#[from] std::io::Error), RowOutOfRange{row,n_rows},
//   BackendRefused(&'static str), ShortRead{row,got,want}, … }  pub type Result<T>;

// codec.rs — the pure, allocation-free core (ds4_ple_stream.h:80–91 "unit tested against the reference")
pub fn row_bytes(p: &PleParams) -> u32;                    // 90 IQ4_NL / 122 EXL3 (c:28–33)
pub fn row_ids(p: &PleParams, tokens: &[i32], pos0: u32, n: u32, out: &mut [u64]); // c:35–73
pub fn dequant_row(p: &PleParams, src: &[u8], head: u32, dst: &mut [f32]);         // c:165–183

// handle.rs
pub enum PleRowType { Iq4Nl, Exl3K6 }                      // ds4_ple_stream.h:47–50
pub struct PleParams { pub row_type: PleRowType,
    pub head_bias: Option<std::sync::Arc<Vec<u16>>>,       // f16 [n_heads][head_dim], EXL3 only
    pub ngram_size: u32, pub heads_per_ngram: u32, pub n_heads: u32, pub head_dim: u32,
    pub eos_token: i32,
    pub multipliers: [u64; 4], pub head_offsets: [u64; 32], pub head_vocab: [u64; 32] } // GGUF metadata, NEVER re-derived (h:60–62)
pub enum PleIo { Auto, Uring, Pool }                       // Auto = ring where available (c:491–493)
pub struct PleConfig { pub io: PleIo, pub qd: u32 /*256*/, pub workers: u32 /*64*/ }
pub struct PleStats { pub lookups: u64, pub hits: u64, pub misses: u64, pub reads: u64,
    pub read_bytes: u64, pub evictions: u64, pub read_seconds: f64 } // field-for-field ds4_ple_stats (h:68–76)
pub enum PleBackend { Uring, Pool }                        // h:112–115: ASK, never assume
pub struct PleStream { /* fd, offset, n_rows, cache, pool, ring?, pf thread, stats */ }
impl PleStream {
    pub fn open(shard: &std::fs::File, file_offset: u64, n_rows: u64,
                params: PleParams, cache_bytes: u64, cfg: &PleConfig) -> Result<PleStream>;
    pub fn fetch(&self, rows: &[u64], out: &mut [u8]) -> Result<()>;  // serialized, coalesced (c:1003–1011)
    pub fn prefetch(&self, rows: &[u64]) -> Result<()>;  // fire-and-forget; finishes an in-flight
                                                         // prefetch first; no-op w/o cache (c:1030–1054)
    pub fn backend(&self) -> PleBackend;
    pub fn stats(&self) -> PleStats;
    pub fn cache_rows(&self) -> u64;
    pub fn close(self) -> Result<()>;                      // explicit (T2 teardown rule); Drop best-effort-joins pf
}

// engine side — T17's qwen4exp/ple.rs body, filled in (≤260 ln):
pub(crate) struct PleGather { ids: Vec<u64>, join: Option<std::thread::JoinHandle<shisu_core::Result<()>>>,
                              staging: std::sync::Arc<parking_lot::Mutex<Vec<u8>>> }
pub(crate) fn ple_gather_start(&mut self, history: &[i32], pos0: u32, n_tok: u32) -> PleGather;
pub(crate) fn ple_block(&mut self, h: PleGather, il: u32, n_tok: u32) -> Result<()>;
pub(crate) fn ple_gather_start_batch(&mut self, items: &[(&[i32], u32)]) -> PleGather; // decode_batch (ds4.c:69617–69645)
pub(crate) fn ple_prefetch_next(&mut self, history: &[i32], pos0: u32, n_tok: u32);     // prefill loop (ds4.c:72484–72488)
```

## Global Constraints

1–7 from master plan (30–37). Task-specific:
1. The pure codec core master's 5-file list omits (half↔float + read geometry) plus the Linux-only
   uring cfg gate ⇒ 7 files (⚠ DEVIATION 4), budgets in Step 0; every file is far under the 1500 cap.
2. `row_ids` and both dequant codecs are bit-exact ports: unsigned 64-bit wrapping hash, the
   EOS-cut-transitive window rule, the EXL3 mul1 codebook with its two fp16 roundings and the
   single fused `fmaf` (c:132–163 comments). Arithmetic-order changes are bugs (rule 3).
3. `SHISU_PLE_IO=uring|pool` (master 35) is the crate's ONLY env read; QD/workers/cache_bytes are
   `PleConfig`/`open` parameters, not env (⚠ DEVIATION 3, T16's knob→constructor precedent).
4. io_uring is Linux-only BY CONSTRUCTION (⚠ DEVIATION 5): `io-uring` is a
   `[target.'cfg(target_os = "linux")'.dependencies]` entry (atlas spark-storage/Cargo.toml:45–50
   verbatim pattern; T1 pre-declared it) and `mod uring` is `#[cfg(target_os = "linux")]`. The
   macOS build type-checks everything else; `PleIo::Uring` there ⇒ `PleError::BackendRefused`
   naming io_uring — the loud refusal, never a silent downgrade (c:510–516, test_ple_io.c:247–261).
5. Zero throughput GATING: p99 fetch latency is recorded (bounded histogram + the ds4 mean/max
   wait counters) and reported; the gate belongs to T20 (master 294). No parity vectors here (T18).
6. `warnings = deny`, clippy deny, `thiserror` at the lib boundary (no anyhow — no bin here),
   `parking_lot` (Mutex/Condvar, never std::sync), `tracing` (never println), no license headers,
   no formatters/full suites (T20 gate). No tokio: the engine scheduler is a plain thread (T10) and
   `Model` is a sync trait (T2) — prefetch is one OS thread + condvar, mirroring c:279–285.
7. CuMetal never gates (rule 4): the two dequant kernels are compiled-only on this Mac; their
   numerics are T15/T18's, and nothing here claims them.

## Source References (verified)

ds4 root = cwd; atlas root = `/Users/sercand/Developer/src/github.com/sercand/atlas`. Every range opened while writing this plan.

| Source | Lines | What / how used |
|---|---|---|
| `ds4_ple_stream.h` | 10–31 | design rationale: 320,001,536 rows × 160 IQ4_NL = 26.8 GiB (llama.cpp checkpoint geometry), scattered gather of 90-byte rows, ~1.4 KB touched per decode step, NOT the routed-expert cache, "start the gather early and only wait just before the PLE layer", knob semantics incl. `"uring"` refusing silent fallback. |
| ″ | 33–50 | MAX_NGRAM 4 / MAX_HEADS 32 / BLOCK 32×18 / EXL3_BITS 6 / EXL3_ROW_BYTES = 2+160·6/8 = 122; row types IQ4_NL=0, EXL3_K6=1. |
| ″ | 52–66 | `ds4_ple_params`: row_type, head_bias, ngram 3, heads_per_ngram 8, n_heads 16, head_dim 160, eos 248044, multipliers/head_offsets/head_vocab STRAIGHT FROM GGUF METADATA — "reproducing splitmix64 bit-exactly would be risk with no upside" (60–62). |
| ″ | 68–118 | `ds4_ple_stats` 7 fields; API: row_bytes/row_ids ("pure and allocation-free")/dequant_row (head = row_index % n_heads)/open (caller keeps fd; cache_bytes 0 = no cache)/fetch (thread-safe, serialized)/prefetch (returns without waiting, finishes in-flight first, copies rows)/backend ("ask rather than assume")/get_stats/cache_rows. |
| `ds4_ple_stream.c` | 28–73 | row_bytes per codec; row_ids: EOS-transitive window cut (42–45: EOS doesn't cut its own context, scan starts s=1; missing predecessors read as EOS), per-order hash `ctx[0]*m[0] ^ ctx[j]*m[j]`, `row = mixed % head_vocab[h] + head_offsets[h]`, unsigned wrap = free non-negative modulo. |
| ″ | 75–183 | IQ4NL LUT (75–77), half→f32 with subnormal renorm (79–102), f32→half RNE incl. subnormals (104–130), EXL3 trellis dequant: bit m of weight i at `((i−m/6) mod 160)·6 + m%6`, 16-bit state × 0x83DCD12D, byte-sum+1024 read as fp16, `fmaf(cb, scale, bias)` rounded once through half (132–163); IQ4_NL block decode (165–183). |
| ″ | 194–316 | cache rationale (Zipf rows, CLOCK second chance, 188–191); worker struct w/ private bounce + stats (201–211); ring slot w/ short-read tracking (215–221); stream struct: slot arena/index/pending (230–252), miss pool semantics (254–269), fetch_mu (271–274), prefetch thread fields (276–285), ring fields + the CPU-cost rationale (287–308), O_DIRECT probe (310–313). |
| ″ | 325–402 | mix64 (339–346); open-addressed index find/insert/backward-shift-remove (348–382 — "cannot tombstone without eventually filling the table"); claim_slot CLOCK hand skipping pending slots (384–402). |
| ″ | 406–551 | open: param validation 414–431 (EXL3 needs head_dim 160 + bias table); per_slot = row_bytes+8+1+2·4, index ≤ half full (446–457); O_DIRECT via /proc/self/fd (476–485); backend selection + LOUD uring refusal (487–517); close = ring→pf join→pool→fds→arena (523–551). |
| ″ | 558–718 | row_geometry shared by BOTH backends "so their idea of the file cannot drift apart" (555–571); bounce_stride = widest 2-block straddle (575–579); read_row: bounce pread loop, EOF-in-last-partial-block rule (582–627); drain-until-empty miss loop (632–644); worker_main (646–658); PLE_POOL_MIN_MISSES 4 (660–662); worker_count: env 665, default 64 with the measured 92/223/54 ms rationale (664–676); lazy pool_start, slot 0 = submitting thread (681–702); pool_stop (704–718). |
| ″ | 733–944 | ring_depth: env 734, default 256 — "~640k IOPS needs >200 in flight" (733–743); ring_open: registered buffers + fixed file, every failure leaves the stream pool-usable (763–811); ring_batch: free/pend slot stacks, read_fixed + short-read requeue, completion test `done >= skew+row_bytes`, error drain-then-drop-the-ring fallback (815–944). |
| ″ | 949–1159 | run_batch: ring if up, else lazy pool, wake min(n,workers)−1 readers (949–983); per-worker stats merge (985–997); fetch = fetch_mu + fetch_locked (1003–1011); prefetch_main/prefetch (1013–1054); fetch_locked 3 passes: hits + claim-and-queue with in-batch coalescing via pending slots (1073–1119), run, deferred copies + pending clear (1124–1133), unwind drops claimed slots on error (1136–1144); backend/get_stats/cache_rows (1147–1159). |
| `ds4.c` | 900–908, 951–955 | PLE shape macros; `ds4_qwen4exp_layer_has_ple(il)` = qwen4exp && ngram≠0 && il==ple_layer. |
| ″ | 2489–2507 | `qwen4exp_ple_params`: **row_type = table->type == DS4_TENSOR_EXL3_NGRAM6 ? EXL3_K6 : IQ4_NL** (2494–2495); head_bias = mapped `per_layer_token_embd.bias`; multipliers/offsets/vocab copied from `g_ds4_ple` (GGUF-loaded). |
| ″ | 5940–5952, 7807–7814 | "Two row codecs: llama.cpp's IQ4_NL rows, or the EXL3 checkpoint's 6-bit trellis rows, which also carry a per-head bias"; EXL3 ⇒ bias tensor required, f16 [16,160]; table rows ≥ Σ head_vocab; tensors bound whenever ple_layer ∈ slice (bytes never resident). |
| ″ | 6983–6996, 7067–7103 | GGUF metadata: `qwen4exp.ple.layers` (exactly one), ngram_size/heads_per_ngram/conv_kernel/eos cross-checks, `layer_multipliers`/`head_offsets`/`head_vocab_sizes` u64 arrays → `g_ds4_ple`; validation: offsets are a contiguous prefix sum of vocab sizes, vocab≠0, multipliers ODD (bijection on low bits). |
| ″ | 55781–55798, 65453, 68488, 68547–68569 | graph fields ple_rows/ple_stream/ple_params/ple_bias_offset/ple_row_ids/ple_pf_ids/ple_row_data; cache default 512 MiB; device buffer T×n_heads×row_bytes; open via `model_shard_of(abs_offset)` → shard_fd + file_offset, n_rows = table dim[1], ONE reader per engine, logs backend + MiB. |
| ″ | 67312–67342 | stats dump: waits count, mean/max wait, total s, lookups, hit %, reads, mean read ms → shisu `tracing::info!` at teardown + p99 histogram. |
| ″ | 68966–69014 | `q4e_ple_block`: dequant(ple_rows→ple_emb) → k/v matmuls → hc_norm k, norm res→q → gated_value → norm cv → dilated conv (window (kernel−1)·dilation, per-sequence in batch mode) → add2 into res; trace sites 68972/68986/69011. |
| ″ | 69587–69658 | `q4e_ple_gather`: row_ids(history,pos0,n) → fetch → timed wait counters → `ds4_gpu_tensor_write(ple_rows,…)` (69597–69610); `_gather_batch`: per-session row_ids from each session's own checkpoint history at pos len−1, ONE fetch of n×heads ("the reader coalesces and the cache is shared", 69613–69645); `q4e_ple_prefetch`: row_ids into pf buffer + stream prefetch (69647–69658). |
| ″ | 69759–69785 | the layer loop: gather issued at the head of the PLE layer, "once the layers before the PLE block are already enqueued, hides that behind their GPU time" — synchronous fetch, no async handle (see DEVIATION 2). |
| ″ | 72479–72488 | prefill loop: after each non-last chunk's forward, prefetch the NEXT chunk's rows, capped at tok_cap. |
| ″ | 80006–80047 | standalone path (test hook): find `per_layer_token_embd.weight` → shard_of → params → open(dim[1], cache_bytes) → row_ids → fetch → host dequant loop (`head = i % n_heads`) → stats. |
| `ds4_qwen4exp_gpu.cuh` | 839–899, 1986–2007 | the 2 dequant kernels (IQ4_NL: 32 threads/row; EXL3: 160 threads/row, 64-word smem staging) + launcher: grid = n_tok·n_heads blocks, bias uploaded via the weight path, EXL3 guards head_dim 160/row_bytes 122 — geometry transcribed to KERNEL_NAMES.md by T15; T19 launches through `Backend::run`. |
| `tests/test_ple_io.c` | 34–58, 63–147, 149–276 | fixture: TABLE_ROWS 20000, FILE_OFFSET 1234 (not block-aligned), `row_byte(row,j) = row·31+j·7+11`, mkstemp+unlink+fsync, EXL3_K6 params w/ zero bias ("fetch returns raw quant bytes"); shapes: single / decode-9 twice-cached / dup-12 / wide-2048 / table-tail; backend asserted not assumed (133–137); hits ≥ n on 2nd pass (139–144); cross-backend byte equality (149–181); QD sweep 1..3 (237–244); loud refusal names liburing (247–261). |
| `tests/test_qwen4exp_ple.c` | 1–36, 101–157 | the model-gated codec gate: GOLD vs llama.cpp `ple_embd` (IQ4_NL checkpoint) or exllamav3 `.npy` ref via env (EXL3 repack, ±1 fp16 ulp); 64 MiB cache and no-cache must give identical embeddings; needs the 104 GiB model → not runnable here. |
| `Makefile` | 20–34, 106–111, 318–319, 588–598 | liburing is a PERFORMANCE dep, not a hard one: pkg-config probe → `-DDS4_HAVE_LIBURING=1` + `-luring`, else ring compiled out and `=uring` refused; per-TU compile rule; both test targets. |
| atlas `crates/spark-storage` | Cargo.toml 45–50; src/backend/mod.rs 15–24; src/high_speed_swap.rs 10–17; src/backend/io_uring.rs 18–45 | the Linux-cfg trio: target-gated `io-uring = "0.7"` dep with the "macOS is cfg(unix) but lacks the syscall layer" comment; `#[cfg(target_os="linux")] pub mod io_uring;` module swap behind one trait; backend-alias so the orchestrator has ONE body; `IoUring::builder().setup_sqpoll(2000).build(qd)` + `types::Buf` registered buffers + `submit_and_wait(1)` + cqe drain — the Rust io-uring idiom `uring.rs` follows (ds4's C ring semantics stay the contract). |
| task plans | T1 45, 213–222; T2 16–37, 69–76; T15 117; T16 166–177, 255–262; T17 68–73, 198–202; T18 220–239 | crate reservation + io-uring workspace dep; traits/dtype; ple.cu module; cfg module-swap + Linux test pattern; the ple.rs seam + ple_rows buffer; parity/perf ownership boundary. |

⚠ **DEVIATION 1 (row type: the shipped table is checkpoint-dependent — BOTH codecs are ported):**
master 285 says "320,001,536-row × 160 IQ4_NL shard … ~90-byte rows". Verified: that is the llama.cpp
checkpoint's geometry only (header 10–15; 320,001,536 × 90 B = 26.8 GiB). ds4 selects the codec at load
from the GGUF tensor type — `per_layer_token_embd.weight` type `DS4_TENSOR_EXL3_NGRAM6` (=72) ⇒ EXL3_K6
122-byte rows + mandatory f16 `per_layer_token_embd.bias` (ds4.c:2494–2495, 5940–5952); else IQ4_NL
90-byte rows. The EXL3 repack is what ds4's own I/O test exercises (test_ple_io.c:46) and the one with
a dedicated exllamav3 gate (test_qwen4exp_ple.c:33–36). shisu ports both codecs and picks by
`DType::Exl3Ngram6` exactly as ds4 does — hardcoding either bakes in a checkpoint assumption. The
"~90-byte"/26.8 GiB figures are IQ4_NL-only; EXL3 is 122 B/row (39.0 GiB).

⚠ **DEVIATION 2 (master 286's "gather starts before layer 0" is not the ds4 mechanism):** verified
at ds4.c:69759–69769 — the gather is a SYNCHRONOUS `fetch` at the head of the PLE layer, issued after
layers 0..ple_layer−1 have been enqueued to the GPU (the host has not synced), so the reads hide behind
already-enqueued GPU work; there is no async gather handle anywhere in ds4. The only async path is the
prefetch thread warming the NEXT prefill chunk's rows (72484–72488 → 69652–69658). shisu reproduces
both mechanisms and keeps T17's handle shape: `ple_gather_start` computes row_ids inline and runs the
blocking fetch on one scoped thread (overlapping the host-side enqueue of the preceding islands — a
strict superset of ds4's overlap, which matters because ple_layer = 1 leaves ds4 only layer 0 to hide
behind); `ple_block` joins the thread, `copy_h2d`s the raw bytes into `ple_rows`, and launches the GPU
dequant — the same await site as ds4. No tokio: `Model` is a sync trait on a plain scheduler thread
(T2/T10); prefetch = one OS thread + parking_lot Condvar (c:279–285/1013–1054).

⚠ **DEVIATION 3 (ds4 reads three env knobs; shisu reads one):** ds4 reads `DS4_PLE_STREAM_IO`
(c:491), `DS4_PLE_STREAM_QD` (c:734, default 256), `DS4_PLE_STREAM_WORKERS` (c:665, default 64).
Master 35 names only `SHISU_PLE_IO`. Per rule 5 and T16's knob→constructor precedent: `SHISU_PLE_IO`
stays an env read (it must be settable per-process for the loud-refusal test semantics), QD and
worker count become `PleConfig` fields with ds4's defaults, cache_bytes is an `open` parameter
(default 512 MiB, ds4.c:65453). No `DS4_*`/`ATLAS_*` reads anywhere.

⚠ **DEVIATION 4 (5 files → 7, additive; master's list has no home for the pure core):** master's
`{handle,cache,uring,pool,prefetch}` omits the unit-testable heart — row_ids + both dequant codecs
+ half↔float (c:28–183, ~160 ln, zero I/O) — and the shared read geometry (row_geometry/bounce_stride,
c:555–579, used by BOTH backends "so their idea of the file cannot drift apart"). Add `codec.rs`
(pure) and keep the geometry in `handle.rs` beside the fd/offset/alignment it reads. Master's five
names stay. Budgets (Rust; every file sits far under the 1500 cap): lib.rs 80, codec.rs 210, handle.rs 260, cache.rs 330
(index/CLOCK + fetch_locked 3-pass), pool.rs 230, uring.rs 280 (Linux-only), prefetch.rs 110
≈ 1500 ln vs the C 1159 (thiserror + typed API + bounds checks account for the delta).

⚠ **DEVIATION 5 (macOS: io-uring does not exist — the cfg rule, mirroring T16 Step 3):**
`io-uring` is a Linux-target dependency (T1 already declared `io-uring = "0.7"` in
`[workspace.dependencies]` for exactly this, task-01:213–222; atlas spark-storage/Cargo.toml:45–50 is
the verbatim pattern) and `mod uring` is `#[cfg(target_os = "linux")]` (atlas backend/mod.rs:15–24). On
macOS `PleIo::Uring` ⇒ `PleError::BackendRefused("io_uring: no io-uring on this target")` — the same
loud-refusal contract as a liburing-less Linux build (c:510–516), so the refusal path is tested HERE
even though the ring is not. The O_DIRECT probe (c:476–485) is portable code that simply fails on
Darwin (no /proc) ⇒ buffered mode; `row_geometry` handles both shapes (c:558–571). macOS proves:
row_ids vs the C oracle, both codecs vs the C oracle, cache budget/CLOCK/evictions/coalescing/unwind,
pool-backend correctness on all five ds4 fixture shapes, prefetch overlap via stats, bounce/alignment
arithmetic, refusal semantics. Linux-only `[INFERENCE]`: ring open (registered buffers + fixed file),
ring_batch short-read continuation, QD sweep, p99 latency.

⚠ **DEVIATION 6 (cross-language oracle replaces the model-gated codec test):**
test_qwen4exp_ple.c's GOLD table needs the 104 GiB GGUF — unavailable here or on any gate host. The portable
proof: `ds4_ple_stream.c` compiles on macOS as-is (pthread/sem/CLOCK_MONOTONIC all Darwin;
`DS4_HAVE_LIBURING` undefined; the O_DIRECT probe inert), so `shisu/scripts/ple_codec_vectors.sh`
(shell per T15's "scripts are shell" precedent; the throwaway C main it writes goes to `$TMPDIR`,
never committed under `shisu/` — rule 1) compiles the real C file once, runs `row_ids` + both
`dequant_row` codecs over deterministic synthetic rows, and emits
`shisu/test-vectors/ple-codec-vectors.json` — committed, and the Rust codec tests diff against it
byte-for-byte. The llama.cpp/exllamav3 model gates stay T18/Linux territory.

## Plan

- [ ] **Step 0 — Preflight.** Read T17's `ple.rs` placeholder + `mod.rs` call sites; confirm
  `kernels/cuda/q4e/KERNEL_NAMES.md` has the two dequant rows (T15) — missing row = STOP (T15 gap).
  Confirm `io-uring = "0.7"` in `[workspace.dependencies]` (T1). File budgets per DEVIATION 4.
- [ ] **Step 1 — Crate + wiring.** `shisu/crates/shisu-ple/Cargo.toml`: deps `shisu-core`,
  `thiserror`, `parking_lot`, `tracing`; `[target.'cfg(target_os = "linux")'.dependencies]
  io-uring = { workspace = true }`. `shisu/Cargo.toml`: members += `crates/shisu-ple`, path dep
  entry (additive hunks). `shisu-engine/Cargo.toml`: + `shisu-ple` (pure Rust — the engine's
  no-CUDA rule from T17 constraint 4 is unaffected).
- [ ] **Step 2 — `codec.rs` (≤210 ln).** `row_bytes`; `row_ids` transcribing c:35–73 (EOS-transitive
  cut, per-order hash, `% head_vocab + head_offsets`, u64 wrapping); `half_to_float`/`float_to_half`
  (c:79–130, subnormals + RNE); IQ4_NL block decode (c:165–183); EXL3 trellis decode (c:137–163:
  ring bit map, mul1 codebook `state·0x83DCD12D`, byte-sum+1024 → fp16, `k_inv=0x1eee`/
  `k_bias=0xc931` halves, ONE `fmaf` then one half-round — use Rust `f32::mul_add`, which lowers to
  a single fma exactly as nvcc contracts). No unsafe, no allocation.
- [ ] **Step 3 — `handle.rs` (≤260 ln).** `PleParams` validation (c:414–431: ngram 2..4, heads
  1..32, head_dim %32, EXL3 ⇒ head_dim 160 + bias present) + the ds4.c:7086–7103 invariants
  (offsets = contiguous prefix sum of vocab, vocab≠0, multipliers odd); slot sizing
  `per_slot = row_bytes+8+1+2·4`, index pow2 ≥ 2·slots (c:446–457); O_DIRECT probe (try
  `OpenOptions::custom_flags(O_DIRECT)` on a `/proc/self/fd/<n>` reopen — Linux succeeds, Darwin
  errors ⇒ buffered; c:476–485); backend selection: `PleIo` × `SHISU_PLE_IO` override, ring-first
  with fallback-warn, `Uring` insistence ⇒ `BackendRefused` (c:487–517); `row_geometry`/
  `bounce_stride` (c:558–579); `close` order ring→pf→pool→fd→arena (c:523–551); stats/backend/
  cache_rows accessors.
- [ ] **Step 4 — `cache.rs` (≤330 ln).** Slot arena + `slot_row`/`slot_ref`/`slot_pending`;
  open-addressed index with mix64 and BACKWARD-SHIFT deletion (c:348–382); CLOCK `claim_slot`
  skipping pending slots (c:384–402); `fetch_locked` 3 passes verbatim (c:1056–1145): pass 1 serve
  hits / claim+index misses up front so in-batch repeats coalesce onto the pending slot / uncached
  mode reads straight to `out`; pass 2 = `run_batch` (Step 5/6 seam); pass 3 deferred copies +
  pending clear; unwind on error drops every claimed slot from the index (c:1136–1144). All state
  under one `parking_lot::Mutex` (fetch_mu equivalent).
- [ ] **Step 5 — `pool.rs` (≤230 ln).** Lazy-start worker pool (c:681–702): slot 0 = submitting
  thread; per-worker bounce buffer + stats merged after join (c:985–997); drain-until-empty miss
  loop (c:632–644); wake `min(n, workers)−1` only when `n ≥ 4` (c:660–662, 966–975); `read_row`
  with the EOF-in-last-partial-block rule (c:582–627). Threads: `std::thread` + parking_lot
  Condvar + a counter (sem→condvar translation; same wake-count semantics).
- [ ] **Step 6 — `uring.rs` (≤280 ln, `#[cfg(target_os = "linux")]`).** `ring_open`: qd slots,
  4 KiB-aligned bounce arena, `register_buffers` + `register_files`, every failure path leaves the
  stream pool-usable (c:763–811); `ring_batch`: free/pend slot stacks, `read_fixed` +
  `IOSQE_FIXED_FILE`, short-read requeue with `done` offset, completion test `done ≥ skew+row_bytes`,
  EOF rule, error ⇒ drain outstanding then drop the ring to the pool (c:815–944). Rust idiom from
  atlas `io_uring.rs` (builder, `types::Buf`, `submit_and_wait`), ds4 semantics as the contract.
  Non-Linux builds compile a `pub(crate) fn unavailable()` stub returning `BackendRefused`.
- [ ] **Step 7 — `prefetch.rs` (≤110 ln).** One lazily-spawned thread (c:1032–1037); `prefetch`
  waits for the previous batch to finish, copies rows, signals, returns (c:1038–1053); no-op
  without a cache (c:1031); the thread's fetch output is discarded (cache warming only); `close`
  sets shutdown + joins (c:528–538).
- [ ] **Step 8 — Engine: params + open (`ple.rs`).** In `Qwen4ExpModel::load`: read GGUF metadata
  arrays `qwen4exp.ple.layer_multipliers`/`head_offsets`/`head_vocab_sizes` + the ngram/heads/eos
  cross-checks via T3's metadata reader (ds4.c:7067–7103); row_type from the bound tensor's
  `DType::Exl3Ngram6` (2494–2495); head_bias = the f16 bias tensor bytes (resident, mapped —
  dequant_row needs it host-side only for the CPU oracle path; the GPU path gets it as a device
  view uploaded once at load, mirroring `q4e_weight` at cuh:1994); locate the table tensor
  (`per_layer_token_embd.weight`, ds4.c:7807–7814) → shard file + file_offset + n_rows = dim[1]
  (68547–68557) → `PleStream::open(…, cache_bytes default 512 MiB, …)`; `tracing::info!` backend +
  cache MiB (68568–68569).
- [ ] **Step 9 — Engine: gather/await/H2D (`ple.rs`).** `ple_gather_start`: `row_ids` into the
  handle, spawn the fetch thread writing into the shared staging buffer (DEVIATION 2);
  `ple_gather_start_batch`: per-session ids from each session's own history at pos len−1, ONE
  fetch (ds4.c:69617–69645); `ple_block`: join + time the wait (counters + bounded p99 histogram),
  `Backend::copy_h2d(ple_rows.view(0, n_rows·row_bytes), staging)`, then
  `Backend::run(KernelRef{module:"q4e_ple", name: dequant | dequant_exl3}, grid = n_tok·n_heads,
  block = 32 | 160, args per KERNEL_NAMES.md row incl. the bias DeviceView for EXL3)`; the rest of
  the block (k/v matmuls, gated_value, conv, add2) stays T17's. mod.rs wiring unchanged: start
  before the layer loop, block at `layer_kind`-independent `il == ple_layer`.
- [ ] **Step 10 — Engine: prefetch + stats surface.** `prefill_chunk`: after each non-last chunk's
  forward is enqueued, `ple_prefetch_next(history, pos+take, min(rest, tok_cap))` (ds4.c:72484–
  72488). Teardown/session-end: `stats()` + wait counters → `tracing::info!` in ds4's report shape
  (67315–67327: waits, mean/max wait, hit %, reads, mean read ms) + p99 recorded to the
  shisu-bench JSON artifact — OBSERVATION ONLY, no assertion (T20 gates).
- [ ] **Step 11 — Oracle vectors + macOS verify.** Run `scripts/ple_codec_vectors.sh` against
  `ds4_ple_stream.c` (this repo) → commit `test-vectors/ple-codec-vectors.json`; run the crate
  tests; `grep -r 'DS4_\|ATLAS_' shisu/crates/shisu-ple` empty.
- [ ] **Step 12 — Linux gate `[INFERENCE]`** (not executable here — no Linux box, no q4e GGUF):
  ring-vs-pool byte equivalence on the fixture, QD sweep, full-model PLE-layer smoke, p99 capture —
  the Wave 3b `#[ignore]` run.

## Tests

macOS, always (synthetic fixture mirrors test_ple_io.c: 20,000 rows × 122 B EXL3 rows at
FILE_OFFSET 1234, `row_byte(row,j) = row·31+j·7+11`, temp file + fsync, zero bias table — built at
test time, not committed):
- `tests/ple_codec.rs`: `row_ids` vs the C oracle vectors — incl. EOS-transitive cut, EOS token not
  cutting its own context, positions before sequence start, multi-token windows; IQ4_NL + EXL3
  `dequant_row` byte-exact vs the C oracle (the EXL3 set exercises both fp16 roundings + the fma);
  `row_bytes` 90/122; params validation table (even multiplier, non-contiguous offsets, EXL3
  without bias ⇒ typed `InvalidParams`).
- `tests/ple_cache.rs`: slot-count formula from cache_bytes; CLOCK: ref-bit second chance,
  evictions counter, backward-shift delete keeps probe chains findable; in-batch duplicate rows
  coalesce onto the pending slot (dup-12 shape, reads == distinct rows); cache_bytes 0 ⇒ every
  entry a fresh read; failed fetch unwinds claimed slots (fault-injection fd).
- `tests/ple_io_pool.rs`: the five ds4 shapes (single / decode-9 twice / dup-12 / wide-2048 /
  table-tail) byte-exact on the pool backend; second pass `hits ≥ n`; `backend() == Pool`;
  `SHISU_PLE_IO=uring` ⇒ `BackendRefused` naming io_uring (macOS arm of the refusal contract).
- `tests/ple_prefetch.rs`: prefetch then fetch ⇒ stats hits without new reads; prefetch with
  cache_bytes 0 is a no-op; a second prefetch blocks until the first drains (busy semantics).
- `tests/ple_engine_bookkeeping.rs` (T17's FakeBackend): launch trace shows the `ple_rows` H2D of
  `n_tok·n_heads·row_bytes` and the dequant launch (name per row_type, grid = n_tok·n_heads)
  strictly before the PLE island and after the preceding islands; `decode_batch` issues ONE fetch
  for n sessions; `prefill_chunk` issues exactly one prefetch per non-last chunk.
Linux `#[ignore]` + `#[cfg(target_os = "linux")]` `[INFERENCE]` (Wave 3b):
- ring backend: open with registered buffers/fd, byte-identical to pool on all five shapes, QD
  sweep 1/3/256, table-tail short-read continuation, `SHISU_PLE_IO=uring` succeeds and
  `backend() == Uring`.
- full-model: PLE-layer smoke on `SHISU_TEST_MODEL` (q4e GGUF) — embeddings match the T18 golden
  (T18 owns the assertion; T19 only runs the path); p99 fetch latency recorded to
  `shisu-bench/baseline/` — recorded, never asserted (T20 gates; rule 4).
No parity vectors authored here (T18); no throughput assertions anywhere.

## Acceptance

- [ ] `shisu/crates/shisu-ple/src/{lib,codec,handle,cache,pool,prefetch}.rs` + `src/uring.rs`
  (Linux-cfg) exist, each < 1500 LoC; crate + workspace wiring is additive; `shisu-engine` gains
  the `shisu-ple` dep and a filled-in `qwen4exp/ple.rs` (< 1500 LoC).
- [ ] API mirrors `ds4_ple_stream.h` 1:1 in meaning: open/fetch/prefetch/backend/stats/cache_rows +
  pure `row_bytes`/`row_ids`/`dequant_row`; `PleStats` field-for-field; caller keeps the fd
  (`&File` borrow); refusal is loud and typed on both no-ring paths (macOS, liburing-less Linux).
- [ ] Both row codecs ported and selected by GGUF `Exl3Ngram6` (DEVIATION 1); hash + codecs
  byte-exact vs the committed C-generated oracle (DEVIATION 6); params loaded from GGUF metadata,
  never re-derived (header 60–62).
- [ ] Engine coupling = ds4's mechanism + T17's handle: synchronous await at the PLE-layer head, fetch
  overlapped with preceding-island enqueue, next-chunk prefetch after each prefill chunk, batch path =
  one coalesced fetch per decode_batch (DEVIATION 2); raw rows reach `ple_rows` via `Backend::copy_h2d`;
  dequant finishes on GPU via `Backend::run` with KERNEL_NAMES.md geometry.
- [ ] `SHISU_PLE_IO` is the only env read; QD/workers/cache are constructor params (DEVIATION 3);
  no `DS4_`/`ATLAS_` strings under `shisu/`.
- [ ] macOS suite green without Linux, GPU, or the model; every Linux-only claim appears solely as
  `[INFERENCE]` + `#[ignore]` (DEVIATION 5); p99 recorded, never gated; no parity ownership (T18).
- [ ] No formatters/linters/full suites run (T20 owns the gate).

## Commit

```sh
git add shisu/Cargo.toml shisu/crates/shisu-ple shisu/crates/shisu-engine shisu/test-vectors/ple-codec-vectors.json \
        shisu/scripts/ple_codec_vectors.sh shisu/Cargo.lock
git commit -m "feat(ple): shisu-ple row streaming — dual-codec gather, CLOCK cache, io-uring+pool backends, engine PLE-layer wiring (Task 19)"
```
