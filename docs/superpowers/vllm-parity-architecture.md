# vLLM Parity — Architecture & Scheduler Plan

*Host-side half of the effort. The CUDA device kernels are covered in the companion doc [`vllm-parity-kernels.md`](./vllm-parity-kernels.md). Read both: most stages span the two, and each doc owns its half.*

## Implementation status (2026-09-07)

- **Stage 0 landed.** `q4e_spec_log_level()` parses `DS4_QWEN4EXP_SPEC_LOG`; `DS4_BATCH_DECODE_MIN` env knob overrides the batch-decode threshold.
- **Stage 3 mechanism exists.** `decode_worker_main` coalesces every decode-ready sequence each tick (iteration-level batching); `DS4_BATCH_DECODE_MIN` sets when it engages. With Stage 5 it keeps MTP.
- **Stage 5 landed and validated end to end.** `ds4_sessions_eval_speculative_batch` (engine) drafts each session, verifies all in one segmented forward, accepts/rolls back each. The coordinator (`decode_worker_main`) runs it when `slot->decode_spec` is set; `server_eval_speculative_batch` enqueues; the decode gate routes greedy batched decode through it. Gated by `DS4_QWEN4EXP_BATCH_SPEC=1` (default off). Unit test `tests/test_qwen4exp_specbatch` (committed == greedy); **smoke-tested** — 3 concurrent distinct greedy requests engaged `count=3 spec=1` 60 times with coherent output and no errors.
- **Stage 4 engine primitive validated; server scheduler remaining.** The segmented forward (`ds4_sessions_forward_segmented`, kernel doc) already runs a prefill run mixed with decode rows; `tests/test_qwen4exp_verify` validates it. The remaining piece is assembling a mixed tick (prefill chunk + decode tokens) in the scheduler instead of separate prefill grants — a deep prompt-phase/coordinator change, mirror `ds4_sessions_eval_batch_with_prefill` / `metal_graph_eval_mixed_prefill_decode` (`ds4.c:79876`).
- See the companion kernel doc for the validated Stage 1 batching and the Stage 2 no-op reasoning.

## Context

**Why this work exists.** On the production server (`/home/otsimo/work/qwen-3.8-flash/serve_ds4.sh`, `--exec-contexts 3`, model qwen4exp / Qwen3.8-Flash-Next) a request whose prompt is 98% cached still takes ~27 s to first byte, and per-stream decode falls from ~60 tok/s alone to ~17-25 tok/s with 2-3 concurrent conversations. Aggregate throughput is flat at ~51 tok/s from two streams up.

Root cause: **ds4 time-slices one weight-streaming pass across conversations.** N streams read the weights N times, and a new turn's prefill competes with ongoing decode for the single model lock. vLLM instead runs all sequences through one batched forward per step and mixes prefill chunks with decode tokens (continuous batching + chunked prefill + PagedAttention).

**What already exists in ds4** (verified by exploration, anchors below):
- **PagedAttention is essentially built.** Fixed 256-token pages (`ds4_q4e_page.h`), a global refcounted pool (`q4e_page_pool`, `ds4.c:55190`), per-sequence block tables (`q4e_page_table`, `ds4.c:55203`), a prefix-sharing radix tree with copy-on-write split and LRU eviction (`q4e_tree_*`, `ds4.c:55209+`), shared across execution contexts.
- **A true batched decode already exists.** `q4e_forward_batch` (`ds4.c:70032`) stacks B sequences on the M dimension, runs one weight pass, and scatters per-row logits back to each session. Dispatched from `ds4_sessions_eval_batch_cuda` (`ds4.c:80148`).
- **The dense EXL3 GEMMs and the weight-stationary MoE are already arbitrary-M**, so passing B as the token count needs no kernel change (owned by the kernel doc).

**Intended outcome.** Turn the time-sliced multi-context executor into a single continuously-batched executor: one forward per scheduler tick serves every runnable sequence, prefill chunks ride with decode tokens in the same forward, MTP speculation is preserved across the batch, and one weight read amortizes across all sequences. Target: first byte under load in low single-digit seconds; aggregate tok/s that scales with concurrency until compute-bound.

**Scope (confirmed with the user):** full vLLM parity including chunked prefill; MTP preserved in the batch via batched speculative verify.

## Division of labor with the kernel doc

| Concern | This doc (architecture) | Kernel doc |
|---|---|---|
| Config, knobs, benchmark harness | ✅ | — |
| Forward drivers (`q4e_forward_batch`, new `q4e_forward_mixed`), scratch partitioning | ✅ | — |
| Continuous-batching scheduler, admission/eviction | ✅ | — |
| Chunked-prefill scheduling (prefill share of a tick) | ✅ | — |
| Speculative verify **orchestration** (draft/verify/accept loop) | ✅ | — |
| Per-row batching of the 6 stateful kernels | — | ✅ |
| Batched MoE at small B | — | ✅ |
| Segmented/ragged GDN kernel for chunked prefill | — | ✅ |
| Per-row sequence-map plumbing inside the stateful kernels | — | ✅ |

## Stage 0 — Config fixes and measurement baseline

No kernel change; establishes a correct yardstick for every later stage.

- **Fix the `DS4_QWEN4EXP_SPEC_LOG` trap.** The three call sites test `getenv(...) != NULL` (e.g. `ds4.c:68402`, `70636`, `72679`), so `Environment=DS4_QWEN4EXP_SPEC_LOG=0` in the systemd unit does **not** disable per-token spec logging — the server has emitted thousands of journal lines on the hot path. Parse the value in C (treat `0`/empty as off). `serve_ds4.sh` also defaults it to `1` when unset, so a C-side fix is the reliable one.
- **Promote two compile constants to env knobs and sweep them:** `DS4_BATCH_DECODE_MIN_GENERATIONS` (`ds4_server.c:12722`) and `DS4_QWEN4EXP_MOE_BATCH_MIN` (`ds4_qwen4exp_gpu.cuh:4098`, default 32). Current data shows batched decode at 3 streams does **not** beat per-context MTP (52 vs 51 tok/s) — that null result is exactly what Stages 1-2 fix, so record it as the baseline.
- **Deliverable:** a repeatable multi-stream benchmark — 2-3 concurrent large-context (60-100k) Responses conversations with tools and reasoning — reporting TTFT and per-stream/aggregate decode against the current binary.

## Stage 3 — Always-on continuous (iteration-level) batched decode

*(Depends on kernel Stages 1-2. Numbered 3 to match the companion doc's staging.)*

Make the batched executor the default decode path, admitting/evicting sequences per step instead of engaging only at 3 streams.

- **Lower the effective batch-decode threshold to 1.** `decode_worker_main` (`ds4_server.c:13658`) already assembles a cross-slot `items[]` and calls `ds4_sessions_eval_batch` (`ds4_server.c:13711`); the scheduler comments already note "nothing here assumes a step is one row" (`ds4_server.c:12634`). Have it coalesce **every** runnable sequence each tick.
- **Continuous admission/eviction.** A newly-arrived or newly-decodable sequence joins the next tick's batch without waiting for the current batch to drain; a finished sequence drops out. Rework the gate at `server_batch_decode_now` (`ds4_server.c:12724`) and the decode branch in `generate_job` (`ds4_server.c:14375`) so the executor is one loop over the live set rather than per-slot workers racing for grants.
- **Keep per-sequence state where it is.** Each `seq[r]` graph owns its recurrent state, block table, n-gram memory; the batch is just which rows are live this tick. No change to `q4e_scratch_bind` ownership (`ds4.c:68753`).
- **Concurrency caps.** `q4e_forward_batch` bounds B by `tok_cap` and `logit_rows` (`ds4.c:70041`); execution contexts cap at `DS4_EXEC_CONTEXTS_MAX = 8` (`ds4.h:578`, `Q4E_CTX_MAX` `ds4.c:56033`). Under one batched executor the context count matters less; the true limits become resident recurrent-state slots (113 MB each, `ds4.c:55983`) and the KV pool (`Q4E_POOL_RESIDENT`, `ds4.c:56007`). Re-derive these budgets so B can exceed 8 when memory allows.

## Stage 4 — Chunked prefill mixed with decode (the TTFT win; hardest)

Let a new request's prefill chunk ride in the same forward as ongoing decode tokens, so first byte no longer waits behind other streams' generation. The device-side segmented recurrent kernel is in the kernel doc; this doc owns the driver, the scratch, and the scheduler.

- **Mirror the DeepSeek reference for qwen4exp.** `metal_graph_eval_mixed_prefill_decode` (`ds4.c:79876`) already packs `prefill_rows` prompt tokens plus `decode_count` decode tokens into one buffer; `ds4_sessions_eval_batch_with_prefill` (`ds4.c:76497`, cuda variant `80382`) is the entry. Build the qwen4exp analogue — a new `q4e_forward_mixed` (or a widened `q4e_forward_batch`) that accepts a **ragged** batch: a run of `t` sequential prefill rows for sequence X (positions p..p+t) plus single decode rows for other sequences, with a per-row `[row] -> sequence` map handed to the batched stateful kernels.
- **Shared-scratch reuse, not replication.** The single engine-wide `tok_cap`-sized scratch (`ds4.c:68429-68446`, sized in `q4e_scratch_ensure` `ds4.c:68529`) is why prefill and decode cannot run concurrently today. A mixed pass of `n = prefill_chunk + B_decode` rows fits while `n <= tok_cap`; the routed intermediates are already `tok_cap x n_used` wide (`ds4.c:68545`). Reuse the one scratch for the combined `n` rows (as the DeepSeek path does) — do **not** add a second multi-GB allocation. Confirm the `q4e_scratch_bind` one-context invariant (`ds4.c:68753`, `70023`) still holds when a single pass carries rows owned by several sequences.
- **Scheduler.** Repurpose the prefill-grant machinery (`server_model_enter_prefill`, `server_prefill_before_decode_locked`, `server_prefill_chunk_rows`, `mixed_prefill_quantum`; `ds4_server.c:12642`, `12695-12960`, `16334`) so a tick carries `min(prefill_chunk, budget)` prefill rows for one admitted request alongside the decode batch, instead of a separate prefill quantum. `mixed_prefill_quantum` becomes the prefill share of a mixed tick.
- **Risk gate:** land this only after the kernel doc's segmented-GDN spike reports acceptable numbers. Until then the executor stays decode-only (Stage 3), which already improves TTFT by making decode contention cheaper.

## Stage 5 — Batched speculative verify orchestration

Preserve MTP across the batch (the user's choice). The per-row kernel plumbing is in the kernel doc; this doc owns the draft/verify/accept loop.

- **The primitive already exists per sequence.** Speculative verify pushes `1 + K` rows through one forward (`ds4_session_eval_speculative`, `q4e_forward(..., 1u + K, 1u + K)`, `ds4.c:70730`); drafting runs the MTP block with arbitrary n (`q4e_mtp_draft`, `ds4.c:70412`). A per-sequence verify is already a small batch.
- **Cross-sequence loop.** Each tick: draft K tokens for every runnable sequence, verify `sum_r (1 + K_r)` rows in one forward, then accept/rollback per sequence. Reuse the existing rollback checkpoints (`gdn_conv_ckpt`/`gdn_state_ckpt`/`ple_conv_ckpt`, `ds4.c:55893`) and the n-gram memory, indexed per row. Keep the draft-KV/indexer planes (`Q4E_MTP_SLOT`) per sequence.
- **Compose with Stage 4:** a mixed tick may then contain prefill rows plus decode-verify rows for several sequences. Reuse the near-tie handling from the EXL3 gate (seed the speculative stream with the oracle token, per the `qwen4exp-exl3-oracle-tolerance` note) so batched verify stays deterministic in tests.

## Critical files (host side)

- `ds4.c` — `q4e_forward_batch` (`70032`) and the new `q4e_forward_mixed`; scratch sizing (`68429-68631`, `68753`); page pool / tree / checkpoints (`55190-56600`, `67631-67789`); speculative verify driver (`70690-70799`); the DeepSeek mixed-forward reference (`79876`, `76497`, `80382`).
- `ds4_server.c` — the scheduler: `decode_worker_main` (`13658`), `dispatch_jobs_locked` (`15221`), prefill-grant arbitration (`12695-12960`), the batch-decode gate (`12722`, `14375`), startup mode wiring (`16306-16340`), and the `SPEC_LOG` fix.
- `serve_ds4.sh` / the `qwen38-ds4` systemd unit — `SPEC_LOG` export fix and any new knobs.

## Verification

- **GPU discipline (mandatory).** Before any full-model run confirm the service is inactive, no `ds4-server` process is running, and memory is free: `systemctl is-active qwen38-ds4`, `ps -eo comm | grep ds4-server`, `free -g`. Never run two model processes. Use `DS4_LOCK_FILE=<scratch>/ds4.lock` (the `/tmp/ds4.lock` default is root-owned).
- **End-to-end benchmark (the Stage 0 yardstick), rerun each stage:** 2-4 concurrent large-context Responses conversations with tools and reasoning; measure TTFT and per-stream + aggregate decode. Success = TTFT in low single-digit seconds under load and aggregate tok/s rising with concurrency (versus today's flat ~51).
- **Correctness** rides on the kernel doc's batched-vs-serial equivalence gate plus the engine-level `tests/test_qwen4exp_exl3` (prompts toy/gate/long; argmax + logit-L2, tie margin 0.25).
- **Deployment is the user's:** `sudo cp ds4-server /usr/local/bin/ds4-server` then restart `qwen38-ds4` (no sudo in this environment). Commits are the user's call.

## Sequencing and risk

Order: Stage 0 (baseline) → kernel Stages 1-2 → Stage 3 (always-on continuous batched decode) → Stage 4 (chunked prefill, gated on the kernel-doc segmented-GDN spike) → Stage 5 (batched speculative verify). Stages 1-3 deliver the aggregate-throughput win and de-risk the executor before the hardest work. NVFP4 weights (FP4 MMA) remain the orthogonal lever to raise the compute ceiling; out of scope here per the standing EXL3 quality decision.
