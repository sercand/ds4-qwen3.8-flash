# vLLM Parity — CUDA Kernel Plan

*Device-side half of the effort. The host orchestration, scheduler, forward drivers and config are in the companion doc [`vllm-parity-architecture.md`](./vllm-parity-architecture.md). Read both: most stages span the two, and each doc owns its half.*

## Implementation status (2026-09-07)

- **Stage 1 landed and validated.** Batched (one launch, per-row state/page pointers packed once per step in `g->batch_ptrs`): GDN conv, PLE conv, GDN recurrent, QSA KV store, indexer key store. `tests/test_qwen4exp_batch` passes at B=2,3,4,8 (batched == sequential, token-identical).
- **Attention stays per-row (by design).** The split decode kernel `q4e_qsa_attention_split_kernel` shares one KV range and page table across its query group — it is single-sequence by construction, and a per-query batch kernel reduces in a different floating-point order and flips near-ties versus single-stream decode. KV reads do not amortize across sequences, so per-row split calls are the right choice. A `q4e_qsa_attention_batch` kernel/launcher exists but is unused.
- **`idx_pool` stays per-row** (it spans a per-sequence block range).
- **Stage 2 is a no-op.** Batched decode already batches the MoE through the per-slot path (`ds4_gpu_q4e_moe_gate_up` over `n_tok`); the grouped kernel re-quantizes activations differently, so lowering `DS4_QWEN4EXP_MOE_BATCH_MIN` would change decode rounding for no gain.
- **Segmented forward landed and validated** (the shared core of Stages 4 and 5). `q4e_forward_segmented` runs several per-sequence runs in one pass: dense/MoE batch over all rows, while GDN conv/recurrent, PLE conv and attention run per segment on row-views of the shared scratch (reusing the single-sequence kernels, including checkpoint writes for verify rollback). `q4e_batch` carries `n_seg/seg_row0/seg_len/seg_owner/segmented`. Tests: `tests/test_qwen4exp_verify` (batched prefill == sync) and `tests/test_qwen4exp_specbatch` (batched speculative verify == greedy). No new attention/GDN kernels were needed — the per-segment reuse of validated kernels is what kept the risk down.
- **Stage 5 complete** (batched speculative verify) on this segmented forward; **Stage 4's engine primitive** is the same segmented forward with a prefill run mixed in. The remaining Stage 4 work is host-side scheduler assembly (architecture doc), not a kernel.

## Context

**Why this work exists.** The production qwen4exp / Qwen3.8-Flash-Next server time-slices one weight-streaming pass across conversations, so per-stream decode falls from ~60 tok/s alone to ~17-25 tok/s at 2-3 concurrent streams and aggregate throughput is flat at ~51 tok/s. The fix is vLLM-style batching: run all sequences through one forward per step so a single weight read amortizes across the whole batch. See the architecture doc for the full diagnosis.

**What already exists at the kernel level** (verified by exploration, anchors below):
- **Batched decode already runs** through `q4e_forward_batch` (`ds4.c:70032`): B sequences stack on the M dimension, one weight pass.
- **The dense EXL3 GEMMs are fully arbitrary-M.** `ds4_exl3_gemm` (`cuda/exl3/ds4_exl3.cu:201`; header: "any m; rows are processed 16 at a time", `ds4_exl3.h:23`). The launch grid keys off `k`/`n` only; `exl3_select_shape` does `(void)size_m` (`ds4_exl3.cu:165`). The same kernel serves a 2048-row prefill chunk and a 1-row decode (`ds4_qwen4exp_gpu.cuh:1504`, reconstruct+cuBLAS above 144 rows, direct kernel below).
- **The weight-stationary MoE is a token-agnostic grouped GEMM.** `ds4_exl3_moe_ws` (`cuda/exl3/ds4_exl3.cu` ~`455`, kernels in `exl3_moe_ws.cuh`) recovers `token = slot_sorted[row] / n_used` and gathers `x + token*k` (`exl3_moe_ws.cuh:101`); nothing assumes the tokens share a sequence.
- **The KV cache is paged.** Fixed 256-token pages (`ds4_q4e_page.h`), `q4e_kv_row(pages, ...)` translation (`ds4_qwen4exp_gpu.cuh:2398`). The attention/store/indexer kernels already gather through a `pages` block table — but one table per launch.

So the dense and MoE math already batches. The kernel work is confined to the **stateful** kernels, which today run as a host loop of B single-sequence launches inside `q4e_forward_batch` (`ds4.c:69203-69483`).

**Scope (confirmed with the user):** full vLLM parity including chunked prefill; MTP preserved in the batch via batched speculative verify. This doc covers the device kernels those require.

## The six stateful kernels that do not yet batch

Inside `q4e_forward_batch` these run once per row against `g->batch->seq[r]`'s own state, not as one batched launch:

| Kernel | Location | Today | Needs |
|---|---|---|---|
| GDN conv step | `ds4.c:69222` → `ds4_qwen4exp_gpu.cuh:620`/`2160` | grid `n_head_v`, one seq | sequence dim + per-row conv-window pointer |
| GDN recurrent step | `ds4.c:69257` → `ds4_qwen4exp_gpu.cuh:463` | grid `n_head_v`, one seq | sequence dim + per-row 128×128 state pointer |
| PLE conv | `ds4.c:69185` | one seq | sequence dim + per-row window |
| QSA store | `ds4.c:69322` → `ds4_qwen4exp_gpu.cuh:2406` | one `pages` per launch | per-row block table + position |
| Indexer store/pool | `ds4.c:69354` → `ds4_qwen4exp_gpu.cuh:3268` | one `pages` per launch | per-row block table + position |
| QSA attention (dense + sparse) | `ds4.c:69381`, sparse loop `69444` → `ds4_qwen4exp_gpu.cuh:3618`/`3955` | one `pages`, grid.z = query | per-query-row block table + selected-block list |

## Stage 1 — Batch the six stateful kernels

The core throughput win: one launch serves B sequences with per-row state and page tables, so the recurrent and attention parts stop paying "B small launches" and match the already-batched dense/MoE parts.

- **GDN recurrent + conv, PLE conv** (`ds4_qwen4exp_gpu.cuh:463`, `620`, `2160`; PLE branch `ds4.c:69185`). Add a sequence dimension to the grid (e.g. `grid.y = row`) and pass a per-row array of device state pointers (`gdn_state[il]`, `gdn_conv[il]`, `ple_conv` for each `seq[r]`), built once per pass. Each block indexes `state[row]`. State is per-sequence contiguous and owned by the session graph, so no relayout is needed — only an indirection array.
- **QSA store, indexer store/pool, dense + sparse attention** (`ds4_qwen4exp_gpu.cuh:2406`, `3268`, `3618`, `3955`; sparse loop `ds4.c:69444`). Replace the single `pages`/`pos` argument with a **per-query-row indirection**: a `[row] -> block-table` map and a `[row] -> position` array. `blockIdx.z` already indexes the query token; extend it to also select that row's page table via `q4e_kv_row(pages_for_row, ...)`. The gather primitive is unchanged — its input widens from one table to a table-of-tables. The sparse path additionally carries a per-row selected-block list, which is already per-query in structure. The store kernel always writes at the frontier page (`ds4_qwen4exp_gpu.cuh:2435`), which stays correct per row.
- **Replace the host loops** in `q4e_forward_batch`'s island (`ds4.c:69203-69483`) with the single batched launches once each kernel accepts per-row inputs.
- **Watch the mrope window.** `q4e_forward_batch` already builds a per-row `mrope`/`mrope_win` (`ds4.c:70052`) because the pooled indexer key reaches back before a row's own token; keep that per-row window when batching the indexer pool kernel.

## Stage 2 — Batched MoE at decode B

At true decode batch sizes route the `B x n_used` routed tokens through the grouped WS MoE so each expert's weights are read once for the whole batch.

- **The WS kernel already batches** sorted tokens regardless of source sequence (`exl3_moe_ws.cuh:101`, `ds4_exl3.cu:419`, grid `n_expert * (inter/128)` independent of token count). The lever is `DS4_QWEN4EXP_MOE_BATCH_MIN` (default 32, `ds4_qwen4exp_gpu.cuh:4098`; gates at `1749`, `4358`, `4481`): below it, routed experts fall to the per-slot fan-out (`ds4_exl3_mgemm`, which re-reads an expert slab per `(token, slot)` — exact at 1 token, wasteful at many). Lower / auto-tune it so a batched decode of B sequences (`B x 10` routed tokens) takes the grouped path.
- **Verify the WS staging lifecycle outside prefill.** The WS/fused MoE are labelled "prefill only" (`ds4_exl3.h:56`) only because their staging is allocated outside graph capture. Batched decode already disables graph capture (`ds4.c:70128`), so this likely works as-is; confirm the staging buffers in `exl3_stream_ctx` (the `ws`/`ws_bytes` fields) are sized for the batched-decode row count and freed per step, not leaked.

## Stage 4 (kernel half) — Segmented / ragged GDN for chunked prefill

**This is the top risk and gets a measured spike before the architecture doc's Stage 4 scheduler lands.**

Chunked prefill packs a run of `t` sequential prefill rows for one sequence plus single decode rows for other sequences into one forward. The recurrent stack must then handle a **ragged** batch: some rows are a sequential chain for one sequence's GDN state (today the chunk kernel `q4e_gdn_chunk_kernel`, `ds4_qwen4exp_gpu.cuh:620`), others are independent single-row updates for other sequences (the recurrent kernel `q4e_gdn_recurrent_kernel`, `ds4_qwen4exp_gpu.cuh:463`).

- **Spike first:** prototype a segmented GDN kernel that takes a per-row `(sequence, is-chunk-run, run-offset)` descriptor and advances each sequence's 128×128 state correctly — a sequential scan within a run, a single step for lone rows — in one launch. Measure it against the serial chunk + recurrent kernels at representative shapes (one 512-row prefill run + a handful of decode rows) before committing.
- The conv windows and PLE conv extend the same way (a run advances the window by `t`, a lone row by 1).
- If the segmented kernel does not pay off, fall back: keep prefill and decode as separate passes (architecture Stage 3 only), and revisit chunked prefill later. The architecture doc gates its Stage 4 on this spike's result.

## Stage 5 (kernel half) — Per-row sequence map for batched verify

Batched speculative verify pushes `sum_r (1 + K_r)` rows through one forward, where different rows belong to different sequences. The stateful kernels from Stage 1 already take a per-row state/page indirection; verify reuses it with a `[row] -> sequence` map so each verified row applies to the right recurrent state and block table. No new kernel beyond Stage 1's indirection — the draft/verify/accept **loop** is host-side (architecture doc). The dense verify GEMM path is already a small-batch kernel: `q4e_matmul_at` routes `2 ≤ n_tok ≤ 16` to `ds4_gpu_q4e_matmul_q8_0_rows`/`_bf16_rows` ("read weights once, multiple rows", `ds4.c:68985`), and EXL3 decode graphs reserve activation scratch for `1 + spec_k` rows (`ds4_qwen4exp_gpu.cuh:1523`) — widen that reservation to the batched verify row count.

## Correctness gates (kernel)

- **Batched-vs-serial equivalence:** feed the same B sequences through the existing per-row host loop and through the new batched launch; compare per-row logits. This is the primary gate for every kernel in Stages 1, 4, 5.
- **Engine gate:** `tests/test_qwen4exp_exl3.c` (prompts toy/gate/long; argmax + logit-L2, tie margin 0.25 per the `qwen4exp-exl3-oracle-tolerance` note; seed the speculative stream with the oracle token to survive the 2473-position near-tie).
- **Partial-tile coverage:** test every `(n mod tile)` residue class — the lesson from the chunked-GDN bug (`ds4-kernel-gates-must-cover-partial-tiles`): route the oracle through the new kernel and test tail residues, not just round shapes.
- **Per-kernel harness:** the scratchpad `q4ebench` builds a device image from layer 0/3 tensors (`ds4_gpu_set_model_map` + `DS4_CUDA_DIRECT_MODEL=1` + `DS4_CUDA_NO_FD_CACHE=1`) and times each entry point at a fixed T — use it for clean per-kernel numbers on an idle GPU.

## Critical files (device side)

- `ds4_qwen4exp_gpu.cuh` — GDN/PLE kernels (`463`, `620`, `2160`), QSA store/attention (`2406`, `3618`, `3955`), indexer (`3268`), MoE dispatch + batch floor (`1749`, `4098-4106`, `4358`, `4481`), dense matmul wrapper (`1504`), decode-scratch reservation (`1523`).
- `cuda/exl3/ds4_exl3.cu` / `.h` — arbitrary-M GEMM (`201`), grouped MoE `ds4_exl3_moe_ws` (`455` region) and staging (`exl3_stream_ctx`).
- `cuda/exl3/exl3_moe_ws.cuh` — the WS MoE kernels (token recovery `101`, passes `257-323`).
- `ds4.c` — the stateful step loop / island the kernels are launched from (`69185-69483`), `q4e_forward_batch` (`70032`), `q4e_kv_row` usage, decode/verify drivers (`70690-70799`), the dense small-batch matmul routing (`68985`).
- `ds4_q4e_page.h` — page geometry (reused unchanged).

## GPU discipline (mandatory)

Before any full-model or on-GPU run: confirm `systemctl is-active qwen38-ds4` is inactive, `ps -eo comm | grep ds4-server` is empty, and `free -g` shows the memory free. Never run two model processes (a prior lapse caused a machine reboot). Use `DS4_LOCK_FILE=<scratch>/ds4.lock`. `ncu` has no permission on this box; `nsys` is fine on short runs but drops kernels on long decode runs.
