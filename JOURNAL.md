# JOURNAL — shisu task-plan authoring

Mission: write one BRIEF plan doc per task (22 tasks: T1,T2,T3,T3b,T4,T5,T6,T6b,T7–T20) of
`docs/superpowers/plans/2026-09-05-shisu-rust-port.md` into `docs/superpowers/plans/tasks/`.
Spawn ≤2 subagents in parallel to write plans; Main reads+verifies each. Goal done when all
22 plans written+verified. Do NOT implement anything.

## Current state (update after every subtask)
- GOAL COMPLETE 2026-09-07: all 22 plans written+verified. Final line counts: 01:483 (grandfathered),
  02:283, 03:241, 03b:296, 04:149, 05:116, 06:152, 06b:163, 07:188, 08:129, 09:234, 10:340, 11:338,
  12:161, 13:340, 14:289, 15:340, 16:315, 17:288, 18:267, 19:340, 20:335. All <=340 except T1.
- 2026-09-07 sweep (user directives): (a) GHA out of scope everywhere — CI/workflow wording -> local
  gate (scripts/check.sh / file_size_check.sh); atlas .yml rows kept ONLY as mechanism/port sources
  (T01 65-67, T20 122-123, T14 214); ci_gpu_stubs/kernel-compile.yml/security.yml files dropped.
  (b) NO model downloads: T03 fetch_test_model.sh -> test_model.sh (local default
  /Users/sercand/models/Qwen3.5-4B-Q4_K_M.gguf, 2,740,937,888 B verified); T06b -> test_model_optiq.sh
  (default /Users/sercand/models/Qwen3.5-4B/mlx; config.json 55543 / model.safetensors 3269669552 /
  index 100945 verified). (c) 500->1500 LoC cap refs purged (user manual + T06b L45).
- T20 (last subagent's file) fully re-verified by Main: 335 ln; all cross-plan refs checked vs current
  files — fixed stale task-06 ref (109,125 -> 116,132) x2; fixed ds4.c PLE-stats ref 67312-67342 ->
  67395-67420 x2 (67312 = phase-prof enum, not the dump). Re-proved from source: percentile grep = 0
  hits in all 4 perf files; CSV header @782; summarize=median-of-steady @41-53; stats dump mean+max
  @67395-67420; atlas ci.yml "GPU box" quote @775-777; argmax_excluding(eos) @901-903.
- T19 written+verified (340 ln). Deviations: (1) master's "160 IQ4_NL ~90B rows" is the llama.cpp
  checkpoint only; ds4 picks codec by GGUF tensor type (DS4_TENSOR_EXL3_NGRAM6=72 at ds4.c:2326 ->
  EXL3_K6 122B rows + mandatory f16 per_layer_token_embd.bias, ds4.c:2494/5943-5952) => port BOTH.
  (2) master 286 "gather starts before layer 0" WRONG: ds4.c:69759-69769 gathers synchronously AT
  the PLE layer inside the layer loop (hidden behind already-enqueued GPU work); only async path is
  the prefetch thread after each non-last prefill chunk (72484-72488). (3) ds4 reads 3 env knobs
  (IO/QD 256/WORKERS 64); shisu keeps only SHISU_PLE_IO, rest become PleConfig fields; cache default
  512 MiB (ds4.c:65453). (4) master's 5 files omit the pure core -> 7 files (adds codec.rs).
  (5) io-uring is cfg(target_os=linux) (atlas spark-storage/Cargo.toml:45-50); macOS Uring => typed
  refusal, same contract as liburing-less Linux. (6) test_qwen4exp_ple.c GOLD needs the 104 GiB GGUF
  (header 9-10) => shisu uses a C-compiled codec-vector oracle instead.
- T17 written+verified (287 ln). Deviations: sampler ds4.c:40819-41513 (T6 owns it, T17 reuses);
  MTP region ends 70532 not 70205; paged-KV/span-tree at ds4.c 55123-55183/55629-55638/55917-55955/
  56008-56043/67846-67978 NOT 70205+; ds4_cuda.cu has ZERO ds4_gpu_q4e_ (forward order is ds4.c
  68799-70532); 7 files -> 10 (matmul.rs, exl3_host.rs, ple.rs). T8 ANSWER: checkpoint payload DOES
  include GDN conv+state + PLE conv window + pending MTP draft row (ds4.c:67869-67877) => not KV-only.
- T18 CRITICAL finding (Main-verified): tests/test-vectors/flash* are DeepSeek-V4-Flash API vectors
  (README + manifest.json "model":"deepseek-v4-flash") and DeepSeek-V4 is OUT of scope => NO golden
  vectors exist for either ported model. Plan must define parity without them.
- T16 written+verified (315 ln). PlanT16 died at exit 1 mid-citation-fix; Main finished the fixes
  (rules ref 78-84 -> 30-37; CuMetal rows 65/66/67/68 -> 62/63/61/70; exl3 moe coop 385 -> 383).
- T16 key findings: ds4 has ZERO cublasLt (grep=0; Makefile:111 -lcudart -lcublas) => classic cuBLAS
  is the parity tier, Lt tier copied from atlas cublaslt.rs and opt-in. ds4 never uses driver API
  (all static <<<>>>); atlas registry.rs/gpu_impl_graph.rs are the module/launch/graph sources.
  sys/{mod,nvidia}.rs = #[path] module swap (atlas lib.rs:8-14 precedent), not a trait.
- D7: Planner subagents may die mid-verification; verify their file yourself before accepting.
- CRITICAL IR TRUTH (T9 verified; feed into T10/T11/T12/T13 dispatches): atlas IR =
  `crates/spark-server/src/ir/` — 6 files (message/request/response/stream/tests/mod), 808 LoC.
  CanonicalRequest/CanonicalEvent/Adapter trait DO NOT EXIST. Narrow waist = ChatRequest/
  ChatResponse/StreamDelta/DeltaStream. FinishReason wire: stop|length|tool_calls|content_filter
  + FINISH_REASON_TIMEOUT="timeout" (response.rs:105,118-143). Anthropic mapping lives in
  anthropic/helpers.rs:58-76 + translate.rs:32-35 (T12's, not vendored in T9).
- KEY cross-task facts (feed into later dispatches):
  * Sampler is ds4.c:40819-41513 (NOT §70205 — master plan wrong; xorshift64* :40942-40955).
    build_probabilities :41010-41138 -> shisu-core/sampling_probs.rs (T17); fast_top_p :41192-41278.
  * Engine NEVER depends on shisu-metal (dev-dep only); Model::load binds+registers, forward is pure
    Backend::run. grid = CUDA threadgroup-count passthrough (T4 fixed).
  * KvStore (T7): put needs text param; 43-char filenames; compat policy is ENGINE-side (T8 owns
    model_id/quant/ctx rejection); report() is new. T8 must answer: does ds4 payload include GDN
    recurrent state? (if not, KV-only resume of GDN models is wrong).
  * CuMetal oracle gate = PASS head_dim≤64 only; D=128 SMEM_BLOCKED. T14/T17 gates conditional.
  * KernelRef `#idx:t<val>` FC suffix -> compile.rs SOURCES (q5_k/q6_k = 48.0/48.1, no ds4 Metal
    kernel; T6 uses GGUF Q4_K_M).
- Verification method per plan: read full file; spot-check 2-4 cited source refs by opening
  them; check SSOT names; check ≤340 ln (T1 exempt); mark todo done.

## Next step
1. None — goal complete. All 22 plans written, verified, swept (GHA-out/no-downloads/1500-cap),
   T20 cross-refs fixed. If the user requests changes, edit in place; journal is the index.

## Decisions (append on every choice)
- D1: Plans live in `docs/superpowers/plans/tasks/task-NN[-b]-slug.md`; names: task-01-workspace-scaffold,
  02-core, 03-gguf, 03b-cumetal-compat, 04-metal-backend, 05-qwen35-msl-kernels, 06-engine-qwen35,
  06b-optiq-loader, 07-kvstore, 08-checkpoint-integration, 09-ir, 10-server-skeleton,
  11-openai-adapters, 12-anthropic-adapter, 13-scheduler-completion, 14-kernel-build-crate,
  15-q4e-extraction, 16-cuda-backend, 17-qwen4exp-forward, 18-golden-parity, 19-ple, 20-perf-e2e.
- D2: Batch order = dependency order; ≤2 concurrent (user rule); verify each plan before next batch slot reuse.
- D3: User format change mid-run: BRIEF docs ≤340 ln (no implementation manuals); guide rewritten;
  running agents steered via hub send; T1's 612-ln file grandfathered by user ("T1 is ok").
- D4: Planner prompts require: read guide + full master plan; verify every source ref; ⚠ DEVIATION
  notes for wrong master-plan refs; align with existing plans' Produces blocks; write only their file.
- D5: Verified deviations found by planners (accepted, evidence-backed): qwen35 GQA head_dim=256
  (n_rot 64, partial_rotary 0.25) not 128 — confirmed via live HF config.json fetch; g_ds4_shape not
  g_shape; quant descriptors at ds4.c:965-1060/1124-1204 not §460-1281; binding at §4966-7830 not
  §2231-4966; no block_q8_0 struct / no q4e dtype / no exl3 structs; memmap2 new dep; split-GGUF
  deferred to CUDA phase (SplitUnsupported).
- D6 (2026-09-07, user): GitHub Actions fully out of scope — every gate is a local script
  (`scripts/check.sh`, `scripts/file_size_check.sh`); atlas `.yml` files are cited ONLY as
  mechanism/port sources, never shipped; Linux nvcc proof = T14 Step 10 runbook.
- D7 (2026-09-07, user): NO model downloads in any plan. Test models are local: GGUF default
  `/Users/sercand/models/Qwen3.5-4B-Q4_K_M.gguf` (2,740,937,888 B), OptiQ dir default
  `/Users/sercand/models/Qwen3.5-4B/mlx` (config.json 55543 / model.safetensors 3269669552 /
  index 100945 — all verified on disk). `SHISU_MODEL_DIR`/`~/.cache/shisu` retired.
- D8 (2026-09-07, user): file cap is 1500 LoC everywhere (atlas's 500 was never shisu's);
  user hand-fixed most refs, Main finished (T06b L45).

## Dead ends (append on every failure)
- (none yet)
