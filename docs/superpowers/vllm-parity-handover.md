# Handover: ds4 qwen4exp decode does not scale with concurrency (vLLM does)

**For:** the next engineer/model picking up the vLLM-parity throughput work.
**Repo:** `/home/otsimo/work/ds4`, branch `qwen3.8-flash-next`. Model:
Qwen3.8-Flash-Next (qwen4exp family), EXL3 4.05 bpw GGUF.
**Companion docs:** `vllm-parity-architecture.md`, `vllm-parity-kernels.md` (same
folder) describe the batching plan; this doc is the current state + the real
problem + where to dig.

---

## 0. Status after the 2026-09-07 follow-up session (read this first)

Everything below this section is the original handover; where it conflicts
with this section, this section is right.

**Corrections to the original.** The server coordinator wiring was already
committed (4f4d9e2), not uncommitted. The path that has to match vLLM is the
*batched speculative* tick, not plain batched decode: vLLM's table is MTP-3
engine steps (B=1 61.5 ms / 3.0 tok, B=4 96.2 ms / 2.84 tok per stream).
Measured with `misc/qwen4exp-numerics/q4espec_bench`, the batched-spec tick at
B=3 was **150 ms = drafts 17 + verify 123 (15 rows) + accept 9**; the verify's
eager profile is MoE 63%, GDN 18%, HC mix 10%, attention 6%.

**What landed (uncommitted working tree):**
- Accept phase: `q4e_forward_segmented` projects every segment into its own
  rows of the logits buffer (`logit_rows` = (1+K)*8 with MTP) and syncs once;
  `q4e_spec_step_batch` reads the committed row instead of re-projecting per
  session. 9 -> ~4 ms.
- Expert-grouped MoE for same-sequence rows: `DS4_QWEN4EXP_MOE_BATCH_MIN`
  default 32 -> 4 (measured: 5 consecutive verify rows +11%, 15 rows +19%;
  2-3 unrelated rows a wash). Plain batched decode rows stay per-slot below 32
  (`unrelated_rows` in `q4e_moe`), so that path is byte-for-byte what it was.
- `ds4_gpu_q4e_matmul_f16`: narrow F16 tensors (the MoE router, ssm_alpha)
  stay on the fp32-activation kernel at any row count instead of fp16+cuBLAS
  above 16 rows.
- Server defaults: `DS4_QWEN4EXP_BATCH_SPEC` on when the engine has a draft
  head (`=0` opts out); batch threshold 2 in that mode (3 for plain);
  in-batch draft depth capped at 3 (`DS4_QWEN4EXP_BATCH_SPEC_K`, 0 = no cap).
- `DS4_QWEN4EXP_SPEC_LOG=2` prints `q4e spec batch timing` per tick.

**Measured now (GB10, 3 concurrent distinct 300-token greedy requests, ctx
16384, exec-contexts 3, --mtp-draft 4 --mtp-vocab; `misc/qwen4exp-numerics/
{ab.py,server_ab.sh}`):**

| arm | count-3 tick | steady-state aggregate | end-to-end aggregate |
|---|---|---|---|
| plain batched (old default) | 54.6 ms / 3 tok | 55 tok/s | 51.5 |
| batched spec K=4, min 3 | 122 ms / ~9.6 tok | ~78 | 56.1 |
| batched spec K=4, min 2 | 122 ms | ~78 | 63.6 |
| **batched spec K=3, min 2 (new default)** | **108 ms / ~8.6 tok** | **~79** | **65.7** |
| vLLM (README, MTP-3) | 95 ms | 91.9 | 91.9 |

Single stream is unchanged (the bench's toy prompt: 68 -> 61 ms per spec
step from the grouped verify rows). End-to-end trails steady state because
the fastest request finishes first; it was 51.5 before.

**Remaining gap to vLLM's 92 and the levers, in order:** (1) batch the MTP
draft across sessions -- 12 serialized 1-row passes with a sync each cost 17
ms of the 108; needs per-row MTP KV planes (they are per-context, not
pooled); (2) a register-resident single-launch segmented GDN kernel (2
blocks/SM instead of 1, bit-exact, ~-5 ms at B=3 and more at B=8); (3) the
15-row verify's MoE (~50 ms) is at the bandwidth floor for ~100 distinct
experts -- only fewer rows help, which is why K=3.

**Numerics -- what the gates actually measure.** Multi-row passes are not
batch-invariant: exl3's routed GEMM accumulates in an order that depends on
the launch geometry (row count), so a 16-row prefill's residual differs from
the sequential decode's by ~1e-5 at layer 0. That is harmless until a top-10
expert choice sits within that of its runner-up: then the token's residual
moves by percent (row 3 / layer 16 in the traced 16-row case: rank-10 expert
137 vs 412), and the GDN state carries it into every later row. Final logits
then differ 4-6e-2 rel-L2 (max ~1 logit) instead of 6e-4, and a greedy token
with a margin under ~1 can flip. This affects every prefill (they were always
>=32 rows) and any batched decode; vLLM behaves the same relative to its own
single-stream path. The grouped MoE kernel is *not* worse than the per-slot
one here (7e-4 vs 6e-4 spread across row counts). So `tests/test_qwen4exp_
batch`'s token-identity check fails or passes depending on which ties happen
to flip (it failed at B=4/B=8 with 0.36-0.72 margins after a rounding-level
change to the 5-token prompt prefill; the same position was a 0.001 tie one
build earlier), `test_qwen4exp_specbatch` and `test_qwen4exp_verify` pass.
Making batched == sequential would need batch-invariant kernels (fixed
K-split geometry in exl3 mgemm, no cuBLAS-fp16 HC mixes above 16 rows, split
attention at every width) -- a separate project with a prefill cost; the
per-tensor tools to measure it are `misc/qwen4exp-numerics/moepath_{cmp,
trace}.c` + `trace_{diff,rows}.py` (see that directory's build.sh).
The PLE reader was checked and is not involved (pool and io_uring backends
give byte-identical results).

---

## 1. The problem (measured, not theorised)

Aggregate decode throughput does **not** scale with concurrent streams. vLLM on
the same GPU (GB10) and same model does. Single-stream is a tie; scaling is the
whole gap.

| streams | ds4 aggregate | vLLM aggregate (Mia-AiLab NVFP4) |
|---|---|---|
| 1 | 46.3 tok/s | 46.3 tok/s |
| 2 | ~73 (est) | 73.0 |
| 3 | **~50** | **91.9** |
| 4 | — | 108.1 |

Measured on ds4 (2026-09-07, GPU free, ctx 16384, exec-contexts 3, MTP on,
3 concurrent × 300 greedy tokens):
- sequential (1 at a time): 42.6 tok/s
- concurrent, plain batched decode (MTP off at 3): **49.9 tok/s**
- concurrent, batched speculative (MTP on): 46.3 tok/s (byte-identical output to
  sequential — correctness confirmed — but slower than plain batched)

So at 3 streams ds4 aggregate is ~50 vs vLLM ~92: a **~1.8× gap**, entirely in
scaling. Per-stream at 3: ds4 ~16.6 tok/s, vLLM ~31.

## 2. What it is NOT (ruled out with evidence)

- **NOT weight size / bandwidth / NVFP4.** Both weight sets are ~equal on disk:
  ds4 EXL3 **98.13 GiB**, vLLM NVFP4 **98.61 GiB** (ds4 marginally smaller). The
  EXL3 file is 98 GiB but ds4 loads only **61.76 GiB resident** — the ~36 GiB
  difference is the **PLE n-gram table, which ds4 streams from disk** (see
  `ds4_ple_stream`, `ds4.c:55898`; startup log "%.2f GiB PLE table streams from
  disk", `ds4.c:66287`). So ds4 reads *fewer* resident weight bytes/token than a
  fully-resident model, which is exactly why it ties vLLM single-stream despite
  trellis decode. Bandwidth is not the gap, and NVFP4's only edge (FP4 MMA
  compute) does not touch the dominant phases below.
- **NOT the MoE grouping threshold.** Tested live: `DS4_QWEN4EXP_MOE_BATCH_MIN=2`
  (force grouped MoE at decode B) moved 3 streams 50 → 53 tok/s and cost
  single-stream 46 → 41. Not the lever; leave the threshold at 32.
- **NOT attention.** Only ~5-8% of a decode step (profile below).

## 3. Root cause: per-sequence phases that don't overlap across the batch

Decode-step phase profile (`DS4_QWEN4EXP_PROFILE=1`, batched decode, perturbed by
sync but ranking is real):

| phase | share | scales with B? | why |
|---|---|---|---|
| **gated deltanet** | **~24-32%** | yes, ~linear | per-sequence 128×128 recurrent state update; compute/latency-bound; kernel already fills all 48 SMs at B=1 (66 KB smem → **1 block/SM**), so B streams = B waves |
| **moe gate+up+down** | **~33-38%** | yes | at a 3-token batch, routed experts read weights per token; no cross-sequence amortization at small B |
| hc mix (dense HC GEMMs) | ~11-20% | **no — amortizes** | shared weights, one read for B rows (this is what the batching already won) |
| qsa attention | ~5-8% | yes (small) | per-sequence KV; one split-kernel launch per sequence |
| others (ple, embed, route, output) | rest | mixed | — |

Arithmetic: ~57% (GDN+MoE) scales ~3× at B=3, ~30% is fixed → B=3 forward ≈ 2.4×
B=1 → aggregate ≈ 46 × 3/2.4 ≈ 58 (matches measured ~50). For vLLM's 92, its B=3
forward is ~1.5× B=1, i.e. it amortizes ~75% of the work vs ds4's ~30%. **The gap
is that vLLM's DeltaNet, MoE and attention kernels overlap several sequences in
one pass; ds4's serialize per sequence.**

## 4. The three levers to close it (ranked)

### Lever A — DeltaNet occupancy (biggest, ~32%)
`q4e_gdn_recurrent_kernel` / `q4e_gdn_recurrent_batch_kernel`
(`ds4_qwen4exp_gpu.cuh` ~463 / the `_batch` variant added this session). Grid is
`(n_head_v=48, B)`; each block loads a 128×129 fp32 state into **66 KB smem** →
only 1 block/SM on GB10 (99 KB opt-in cap), so 48 heads already fill the 48 SMs at
B=1 and B streams serialize into B waves.
- **Fix idea 1:** cut the smem footprint (tile the 128×128 state, or drop the +1
  bank-padding column and re-derive the bank-conflict avoidance) so 2 blocks fit
  per SM → B=2 overlaps in one wave.
- **Fix idea 2:** a batched-chunked DeltaNet (FLA / flash-linear-attention style)
  that processes B sequences' single-token updates with far more independent work
  per SM to hide the dependent-FMA latency. The chunked prefill kernel
  `q4e_gdn_chunk_kernel` is the starting point; decode is its degenerate case.
- Validate with `tests/test_qwen4exp_batch` (batched == sequential, token-ident.).

### Lever B — MoE that amortizes shared experts across the batch (~35%)
At decode B the routed tokens (`B × n_used`, n_used=10) go through the per-slot
fan-out (`ds4_gpu_q4e_moe_gate_up`, per-`(token,expert)` weight read) because
`q4e_moe_batch_min_tok()`=32 (`ds4_qwen4exp_gpu.cuh` ~4479). Simply lowering it
did **not** help (§2) because at B=3 the ~30 activations hit ~30 distinct experts.
The win is a kernel that, when multiple tokens in the batch route to the **same**
expert (which happens — MoE routing is non-uniform, popular experts get several),
reads that expert's weight **once** and applies it to all its tokens. The
weight-stationary grouped kernel `ds4_exl3_moe_ws` / `exl3_moe_ws.cuh` already
groups by expert; the question is why it doesn't win at small B (staging overhead?
too few rows per expert to amortize the trellis decode?). Profile it in isolation
(scratchpad `q4ebench` harness) at B∈{3,4,8}.

### Lever C — batched PagedAttention (~5-8%, smallest but true parity)
`q4e_qsa_attention_step` runs one launch per sequence (the split kernel
`q4e_qsa_attention_split_kernel` is single-sequence: shared KV range + one page
table across its query group). A per-query-row page-table + position indirection
would let all B sequences' queries run in one launch (vLLM PagedAttention). A
per-query **non**-tiled batch kernel was written this session
(`q4e_qsa_attention_batch_kernel`, currently **unused** — it reduces in a
different FP order and flips near-ties vs the split kernel) — do **not** use it;
extend the *split+combine* kernels with per-row `pages`/`pos` instead. The packed
per-row page-table array already exists: `g->batch_ptrs`, kv block at
`(2*N_LAYER+1)*n` (see `ds4_gpu_q4e_qsa_store_kv_batch`, already batched).

NVFP4 weights are **not** a lever for these phases (GDN is fp32; MoE at small B is
bandwidth-bound on same-size weights). Skip that path.

## 5. What is already done and validated (build on this, don't redo)

Committed in **4f4d9e2 "Batched-decode"** (branch `qwen3.8-flash-next`). Server
coordinator wiring (`ds4_server.c`) is **uncommitted** working-tree changes.

- **Stage 0 (config):** `q4e_spec_log_level()` parses `DS4_QWEN4EXP_SPEC_LOG`
  (0/empty=off — the unit's `=0` used to stay on); `server_batch_decode_min()`
  reads `DS4_BATCH_DECODE_MIN` (default 3).
- **Stage 1 (batched stateful kernels):** `g->batch_ptrs` device pointer array
  (tok_cap-sized), packed once/step by `ds4_gpu_q4e_ptrs_pack`. Batched kernels:
  `q4e_gdn_conv_batch`, `q4e_ple_conv_batch`, `q4e_gdn_recurrent_batch`,
  `q4e_qsa_store_kv_batch`, `q4e_idx_store_k_batch` (all `ds4_qwen4exp_gpu.cuh`).
  Test `tests/test_qwen4exp_batch` (B=2,3,4,8 token-identical to sequential).
  **This is what amortizes the dense/HC GEMMs — the ~20% that already scales.**
- **Stage 2:** determined a no-op (see §2).
- **Stage 3:** iteration-level batching already exists in `decode_worker_main`;
  `DS4_BATCH_DECODE_MIN` gates it.
- **Stage 4 core (segmented forward = chunked-prefill primitive):**
  `ds4_sessions_forward_segmented` / `q4e_forward_segmented` (`ds4.c`): several
  per-sequence runs in one pass, dense/MoE batched, GDN/PLE/attention per-segment
  (reuse single-seq kernels on row-views, incl. verify checkpoints). `q4e_batch`
  extended with `n_seg/seg_row0/seg_len/seg_owner/segmented`. Test
  `tests/test_qwen4exp_verify` (4-prompt batched prefill == sync, 0 near-ties).
  **Server scheduler assembly of a mixed prefill+decode tick is NOT done** — this
  is the remaining Stage 4 (TTFT) work; mirror the DeepSeek mixed path
  `ds4_sessions_eval_batch_with_prefill` / `metal_graph_eval_mixed_prefill_decode`
  (`ds4.c:79876`).
- **Stage 5 (batched speculative verify):** `ds4_sessions_eval_speculative_batch`
  / `q4e_spec_step_batch` (draft per session via `q4e_spec_draft_only`, one
  segmented verify forward, per-session accept + `q4e_spec_rollback` +
  `q4e_mtp_pend_set`). Test `tests/test_qwen4exp_specbatch` (committed == greedy).
  **Server (uncommitted):** `decode_worker_main` runs it when `slot->decode_spec`;
  `server_eval_speculative_batch`; decode-gate branch, greedy-only, gated by
  `DS4_QWEN4EXP_BATCH_SPEC=1` (default off). Smoke-tested live: 3 concurrent
  greedy → `count=3 spec=1` engaged, coherent output, byte-identical to
  sequential. **Note it is ~8% slower than plain batched** because the verify
  forward does *more* of the non-amortizing GDN/MoE work — do not enable by
  default until Levers A/B land.

## 6. Reproduce the measurement

GPU discipline (mandatory): before any run confirm `systemctl is-active
qwen38-ds4` is inactive, `ps -eo comm | grep ds4-server` is empty, `free -g` shows
memory free. Never run two model processes. Use `DS4_LOCK_FILE=<scratch>/ds4.lock`
(the `/tmp/ds4.lock` default is root-owned). `ncu` has no permission here; `nsys`
drops kernels on long decode.

```
# build
make ds4-server tests/test_qwen4exp_batch tests/test_qwen4exp_verify \
     tests/test_qwen4exp_specbatch DS4_LOCK_FILE=<scratch>/ds4.lock

# engine gates (need the model + MTP sidecar)
export DS4_QWEN4EXP_MODEL=/home/otsimo/work/qwen-3.8-flash/exl3-gguf/Qwen3.8-Flash-Next-EXL3-4.05bpw.gguf
export DS4_QWEN4EXP_MTP=/home/otsimo/work/qwen-3.8-flash/exl3-gguf/mtp-Qwen3.8-Flash-Next-EXL3.gguf
./tests/test_qwen4exp_batch        # B=2,3,4,8 == sequential
./tests/test_qwen4exp_verify       # segmented prefill == sync
./tests/test_qwen4exp_specbatch    # batched spec == greedy

# decode phase profile
DS4_QWEN4EXP_PROFILE=1 ./tests/test_qwen4exp_batch   # prints per-phase ms/step

# throughput A/B: start server (port 8898), fire N concurrent DISTINCT greedy
# prompts (identical prompts serialize on the cache), read decode t/s from the
# server log (DS4_SERVER_BATCH_LOG=1 prints "decode batch count=N spec=0|1").
# Server model + flags:
#   --cuda -m <EXL3 gguf> --ctx 16384 --mtp-model <mtp> --mtp-draft 4
#   --exec-contexts 3   [DS4_QWEN4EXP_BATCH_SPEC=1 to keep MTP across the batch]
```

Env knobs: `DS4_BATCH_DECODE_MIN` (batch threshold), `DS4_QWEN4EXP_MOE_BATCH_MIN`
(MoE grouping threshold, default 32), `DS4_QWEN4EXP_BATCH_SPEC` (batched spec,
default off), `DS4_QWEN4EXP_PROFILE`, `DS4_SERVER_BATCH_LOG`, `DS4_QWEN4EXP_SPEC_LOG`.

## 7. Files

Engine: `ds4.c` (q4e_forward_batch/segmented, spec batch, q4e_batch struct ~55739,
q4e_graph_alloc/free, stateful step fns ~69185-69483). Kernels:
`ds4_qwen4exp_gpu.cuh` (GDN ~463/620, split attn ~3030, MoE dispatch ~4479, all
`_batch` launchers). `cuda/exl3/exl3_moe_ws.cuh` + `ds4_exl3.cu` (grouped MoE).
Prototypes: `ds4_gpu.h`, `ds4.h`. Server: `ds4_server.c` (decode_worker_main
~13675, server_eval_speculative_batch, decode gate ~14375, batch_spec init
~16345). Tests: `tests/test_qwen4exp_{batch,verify,specbatch}.c`. Per-kernel
harness: session scratchpad `q4ebench/`.

## 8. Bottom line

The batching machinery (segmented forward, batched recurrent/store kernels,
batched speculative verify) is correct and tested and is the right foundation.
It amortizes the dense/HC GEMMs — the only ~20% of decode that *can* amortize
without kernel work. Matching vLLM's 46→92 scaling needs the DeltaNet occupancy
rewrite (Lever A) and the expert-amortizing MoE (Lever B); attention batching
(Lever C) is parity polish. It is **not** the weight format and **not** bandwidth
— those are a tie, confirmed on disk (98.13 vs 98.61 GiB) and at single stream.
