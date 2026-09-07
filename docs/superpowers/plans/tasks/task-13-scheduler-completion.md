# Task 13: Scheduler completion

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 13 (:236-240)
**Depends on:**
- T10 (`task-10-server-skeleton.md`): the contract to **keep** — `EngineRequest::Generate(
  GenerateRequest{prompt_tokens: Arc<Vec<u32>>, sample: SampleRequest, max_tokens: u32, reply:
  oneshot::Sender<Result<DeltaStream>>})`, `EngineHandle::generate(...) -> Result<DeltaStream>`,
  `spawn_engine(Box<dyn Model>, ctx) -> (EngineHandle, JoinHandle<()>)` (T10:48-52); bounded-send
  policy (atlas `mod_helpers/send.rs:4-31,46-96`: try_send → 1 ms poll → 5000 ms → abandon;
  `Closed` = client hung up); JoinHandle retained, `teardown()` after drain (`serve_load.rs:926-932`).
  T10:83-84 defers batching/residency/cancellation to THIS task; its Step-5.4 note: "T13 replaces
  step 4's inline loop … the EngineRequest/DeltaStream contract is what it keeps".
- T9 (`task-09-ir.md`): `StreamDelta` verbatim (`Content{text,token_ids}`, `Reasoning`,
  `ToolCallStart`, `ToolCallArgs`, `Refusal`, `Finish{reason,usage}`); `Usage` fields T13 fills
  (`prompt_tokens`, `completion_tokens`, `cached_prompt_tokens`, `time_to_first_token_ms`,
  `response_tokens_per_second` — atlas `ir/response.rs:67-85`); `FINISH_REASON_TIMEOUT = "timeout"`
  carried as `FinishReason::Other`, lossless via `as_wire` (atlas `ir/response.rs:87-116`).
- T2 (`task-02-core.md`): `Model` methods — `open_session`, `close_session`,
  `prefill_chunk(&mut,SessionId,&[u32],is_last)`, `decode_batch(&mut,&[SessionId],&[u32]) ->
  Result<DeviceView>` (row i ↔ sessions[i]; single decode thread ⇒ `&mut self`, T2:13),
  `sample(&logits, row, &mut SampleRequest)` (rng advances in place — one `SampleRequest` per
  seq), `teardown` once by the loop owner. `SessionId(u64)` Copy; upcast verified (T8:83).
- T6: `Qwen35Model` is the production model; `eos_token_id` arrives from `ShapeProfile` at boot.
- T7 (`task-07-kvstore.md`): `/cache` disk section live since T7; `KvStore::report()` counters and
  the hit-half-life policy are T7's — T13 adds the engine section only (ds4 `:15508-15536`).
- T8 (`task-08-checkpoint-integration.md`): `Checkpointed: Model` supertrait (Decision 8,
  task-08:83) with `kv_enable`, `kv_resume(session, prompt_text) -> Result<Option<ResumeInfo>>`,
  `kv_save(session, tokens, prompt_text, logits_view, Reason, ext_flags) -> Result<bool>`,
  `kv_shutdown_sweep(text_of)`; T8:94 assigns T13 to wire these call sites ("T13 wires the call
  sites, not the decisions"); T8:91: `find_prefix` re-hashes candidate prefixes O(n) — T13
  memoizes the hit per request.
- T11 (`task-11-openai-adapters.md`): consumes `ToolCallStart/ToolCallArgs/Refusal` "as-is and
  never parses tool-call text" (task-11:18-20) → parser ownership decided below; its 503
  "Scheduler queue full" mapping (Step 7) is the shape the queue-full policy must produce.

**Produces** (T12/T20 + the already-written T11/T12 handlers consume only these):
- `scheduler/queue.rs` (extended, additive): unchanged `EngineRequest` (single `Generate` variant)
  + `generate()` signature; new `pub struct SchedulerConfig { max_num_seqs: usize,
  mixed_quantum: usize, solo_chunk: usize, request_timeout: Option<Duration>,
  queue_admission: Duration, eos_tokens: Vec<u32> }` (`Default` = env defaults, Step 1); new
  `pub fn spawn_engine_with_checkpoints(model: Box<dyn shisu_engine::kv_ckpt::Checkpointed>,
  ctx: u32, cfg: SchedulerConfig) -> (EngineHandle, JoinHandle<()>)`; `pub struct SchedulerStats`
  (atomics + `render(&self, &mut impl Write) -> fmt::Result<()>`).
- `scheduler/{mod.rs,batching.rs,lifecycle.rs,cancel.rs,tool_stream.rs}` internals as needed.
- `tool_stream.rs`: `pub struct ToolStream` — `new(tools: Option<Vec<ToolDefinition>>)`,
  `feed/finish(&mut self, …) -> Vec<ToolOut>`, `saw_call(&self) -> bool`; `pub enum ToolOut {
  Content(String), Start{id,index,name}, ArgsDelta{index,fragment}, Call(shisu_ir::ToolCall),
  Refusal(String) }` mapped 1:1 onto `StreamDelta`.
- `api/misc_handlers.rs`: `metrics()` renders the new counters; `cache_report()` gains the engine
  section. `lib.rs::run()`: one-line entry-point switch (recorded in the commit body).

## Global Constraints

Rules 1–7 (master plan), one line each:
1. All new code under `shisu/crates/shisu-server/src/scheduler/` (+ the two additive edits);
   pure Rust, no host C/ObjC, no kernels.
2. 1500-LoC cap — split: `mod.rs` (loop + stats), `queue.rs` (types/spawns/bounded sends),
   `batching.rs` (ready set + admission), `lifecycle.rs` (seq machine + finish), `cancel.rs`
   (cancel/deadline), `tool_stream.rs` (parser; split `tool_stream/{dsml,json}.rs` if tight).
3. Vendored-from-atlas pattern copies drop the SPDX line (rule 7); no `ATLAS_*`/`DS4_*` env
   strings anywhere (DS4 paths in doc comments only).
4. No perf gate here (T20 owns numbers); correctness = determinism + the fake-Model tests.
5. Env knobs, `SHISU_`-prefixed, read **once in `lib.rs::run()`** into `SchedulerConfig`
   (scheduler files read zero env): `SHISU_MAX_NUM_SEQS` (4), `SHISU_MIXED_PREFILL_QUANTUM`
   (128 = ds4 default, `ds4_server.c:15969`), `SHISU_SOLO_PREFILL_CHUNK` (2048 = ds4 solo width
   `:12820`), `SHISU_REQUEST_TIMEOUT_MS` (0 = off), `SHISU_QUEUE_ADMISSION_MS` (250 — max block
   inside `generate()` before 503). T10's `SHISU_STREAM_SEND_*` behaviour (5000 ms) is reused
   verbatim, no new knob.
6. `warnings=deny` + clippy deny; `parking_lot` for locks (`parking_lot::Mutex` +
   `parking_lot::Condvar` for the pending queue); `thiserror` for any new error; `tracing` logs
   mirror atlas's field style (`slot = …, emitted = …`).
7. No license headers on vendored copies.

Task-specific:
- **Preserved vs replaced.** PRESERVED from T10: `EngineRequest`/`GenerateRequest` shapes,
  `generate()` signature + reply-first-then-stream semantics, `spawn_engine` signature (T10's
  `tests/fake_model`/`engine_thread` stay green), bounded-send policy, receiver-thread-over-mpsc
  (`scheduler/mod.rs:342-368`), JoinHandle + deferred `teardown()`, single-thread `Box<dyn Model>`
  ownership (T10:81-82). REPLACED: the Step-5.4 inline per-request FIFO loop → the multi-sequence
  coordinator: condvar drain, bounded slot table (`max_num_seqs`, cf. `serve_load.rs:754-759`),
  batched `decode_batch` over the ready set, interleaved bounded prefill quanta, cancellation,
  deadlines, checkpoint hooks, usage accounting.
- The coordinator thread is the ONLY toucher of `EngineModel` (Plain/Checkointed wrapper) and
  the ONLY caller of `open_session`/`close_session` — cancellation "retains worker ownership"
  (ds4 `test_cancel_running_job_keeps_worker_ownership`, `ds4_server.c:20911-20924`): a
  client-side hangup marks the seq dead; only the coordinator frees it.
- Deterministic time: the loop uses an injected clock (`trait Clock { fn now(&self) -> Instant }`,
  `SystemClock` default, `FakeClock` in tests) — `tests/scheduler_*.rs` must be sleep-free.
- `cargo test -p shisu-server` from `shisu/`; T11/T12 handler files are FROZEN for this commit.

### Tool-call deltas — ownership decision (parser lives in T13)

`[WARN] DEVIATION:` the master-plan T13 `Files:` (:237) lists only `{lifecycle,batching,cancel}.rs`,
Task 11's `Files:` names no parser file, and T9's Step 3a says "the parser is T11's job". The
later, binding contract overrides all: T11's plan (:18-20) — **engine/scheduler EMITS** the
`ToolCallStart/ToolCallArgs/Refusal` deltas, "T11 consumes them as-is and never parses tool-call
text" — and its constraints restate "T13 owns … the tool-call parser that emits the deltas". Deltas
are minted where decoded tokens become `StreamDelta::Content` — in the coordinator, before the
bounded send — so the raw-text parser is T13's: `scheduler/tool_stream.rs`. ds4 lineage confirms
the streaming-side placement (parser beside the SSE writers: `openai_tool_stream_update`
`ds4_server.c:7711-7795`; states enum `:6887-6895`; registry `:7549-7573`; writers `:7002-7065`;
suppression `:7086-7102,7898-7910`), pinned by the never-leak self-test (`:17874-17875`, machine
test `:18001-18039`). Atlas twin: `StreamingToolDetector`/`DetectorOutput`
(`tool_parser/streaming.rs:11-17,28-93`), consumed per token at
`api/chat_stream/handle_token.rs:577-602` — engine-side here exactly because T11 forbids API-side
parsing. Semantics: with tools rendered, every sampled token's text flows through the seq's
`ToolStream`; outside an envelope → `Content`; inside → tag text suppressed, `Start`/`ArgsDelta`/
`Call`; malformed/unterminated at EOF → drop text + one `Refusal` + `FinishReason::ContentFilter`
(T12 maps to Anthropic `refusal`). Scope: DSML XML flavour (invoke/parameter elements of the
tools envelope, `:7549-7573`) + JSON-object flavour (`\boxed{…}`, `tool_call` / `tool_result`
markers; atlas
`tool_parser/hermes.rs`); values type-coerced per `ToolDefinition`s (atlas `type_coerce.rs`
pattern), unparseable args JSON ⇒ `Refusal`, never a fabricated `{}`.

## Source References (verified)

Atlas paths relative to `…/sercand/atlas/crates/spark-server/src/`; ds4 to the repo root. Every
row was opened this session.

| Source | Lines | What lives there / how used |
|---|---|---|
| `scheduler/mod.rs` | 1-13 | architecture comment: one decode thread, blocks on the condvar when idle (zero CPU), drains queue after each step — T13's loop skeleton rationale |
| `scheduler/mod.rs` | 338-369 | `Mutex<PendingQueue>` + `Condvar` receiver thread: `blocking_recv` → push → notify; `closed` flag + notify on channel close — copied structure (no LoRA/spec pools) |
| `scheduler/mod.rs` | 409,463-469 | tick order: drain → admit → prefill/decode → rotations only at QUIESCENT points (`active.is_empty() && prefilling.is_empty()`); shisu drops rotations entirely |
| `scheduler/mod.rs` | 597-624 | drain phase: `drain_pending_requests` → `start_new` → continue in-progress prefills (`continue_in_progress_prefills` + `did_mixed_step`) — the interleave T13 ports for prefill quanta |
| `scheduler/mod.rs` | 1082-1161 | shutdown tail: complete active seqs, `sync_all`→`teardown`, "Scheduler stopped" log — T13's tail (+ `kv_shutdown_sweep` before `teardown`, mirroring ds4 `:16540-16548`) |
| `scheduler/mod_helpers.rs` | 97-140 | `drain_pending_requests`: condvar wait while idle & !closed (no busy-poll), bounded 10 ms wait variant — port the blocking variant only (no parked/spilled pool in T6 layout) |
| `scheduler/mod_helpers.rs` | 239-284 | `enforce_request_deadlines`: zero-cost skip when nothing deadlined; deadline cut sets `guard_stop` → finish reports `"timeout"`, never `"length"` — ported verbatim in `cancel.rs` (GUARD_REQUEST_TIMEOUT) |
| `scheduler/mod_helpers.rs` | 305-351 | `retire_finished_sequences`: two-phase retire (drop finished via `finish_sequence`, keep survivors) — ported without SSM compaction (sessions are engine-owned) |
| `scheduler/mod_helpers/send.rs` | 4-31,46-96 | bounded `try_send`→poll→deadline→abandon; `Closed` = client hung up (info log), abandon = same retirement path — T10 already ships this; T13's `cancel.rs` reacts to its outcomes |
| `scheduler/lifecycle.rs` | 8-66 | the SINGLE guard→wire-reason mapping; server-side cuts are `"length"`-family, never `"stop"` |
| `scheduler/lifecycle.rs` | 68-129 | `derive_finish_reason` precedence: deadline→`"timeout"`; EOS→`"stop"`; tool-call close→`"tool_calls"`; budget→`"length"`; cancel/shutdown→`"stop"` — T13's finish table |
| `scheduler/lifecycle.rs` | 131-261 | `finish_sequence`/`send_error`: final-frame + free-resources single point — model for `lifecycle::finish` (usage fill, Finish delta best-effort, close session) |
| `main_modules/serve_load.rs` | 754-759, 921-932 | queue sized `max_num_seqs`; JoinHandle retained, join **before** teardown — both rules restated in `queue.rs` doc comments |
| `tool_parser/streaming.rs` | 11-17, 28-93 | `DetectorOutput` (Content/Start/ArgsDelta/Call) + `StreamingToolDetector::feed` incl. malformed-drop semantics (`:15-19` doc) — `ToolOut` mirrors it 1:1 |
| `api/chat_stream/handle_token.rs` | 577-602 | per-token consumer: detector feed → SSE events, `call.add_namespace`-style mapping per event kind — shows exactly what T11 expects to receive (and why it can't parse) |
| `ir/response.rs` | 67-85, 87-116 | `Usage` incl. ttft/tps/cached fields T13 fills; `FINISH_REASON_TIMEOUT` const carried as `Other`, `as_wire`-lossless |
| `ds4_server.c` | 6887-6895, 7549-7573, 7711-7795, 7002-7065 | DSML states enum, syntax registry, `openai_tool_stream_update` feed loop, writer emitters — `tool_stream.rs` scanner + emit paths |
| `ds4_server.c` | 7086-7102, 7898-7910 | SUPPRESS mode + per-token mode switch (TEXT/THINKING/SUPPRESS/TOOL) — the never-leak routing |
| `ds4_server.c` | 17874-17875, 18001-18039 | never-leak invariant (`every char accounted, raw tags never reach the client`) + full tool-machine self-test — becomes `tests/tool_stream.rs` |
| `ds4_server.c` | 12609-12626, 12818-12833 | batched step: every generating context gets one decode step before any prefill quantum; quantum bounded (default 128 `:15969`), solo prefill full width (`:12820` 2048) — `batching.rs` policy |
| `ds4_server.c` | 15583-15606, 20911-20924 | cancel path marks the job, worker (not the caller) frees it — `cancel.rs` ownership rule + test shape |
| `ds4_server.c` | 14119-14121, 13181, 14615, 16540-16548 | checkpoint reasons at the four lifecycle points (cold / continued / stream-end / shutdown sweep) — T13's `kv_save` call sites |
| `ds4_server.c` | 15508-15536 | `/cache` JSON: `{requests,hits,hit_rate,tokens_reused,tokens_prefilled,states{…},disk{…}}` — engine-section shape for `cache_report()` |
| `docs/…/task-10…md` / `task-11…md` | 48-52, 83-84 / 18-20 | binding T10 contract + "engine EMITS … T11 consumes as-is" parser-ownership evidence |

## Plan

- [ ] **1. `scheduler/queue.rs` (extend; keep every T10 line compiling).**
  Add `SchedulerConfig` (field list + defaults per Global Constraints 5) with `Default`;
  add `spawn_engine_with_checkpoints(Box<dyn Checkpointed>, ctx, cfg) -> (EngineHandle,
  JoinHandle<()>)`; keep `spawn_engine(model, ctx)` delegating with `SchedulerConfig::default()`
  and `EngineModel::Plain` (T10 tests keep passing untouched).
  In `generate()`: replace the bare `tx.send(r).await` with
  `tokio::time::timeout(cfg.queue_admission, tx.send(r)).await` — `Err(Elapsed)` or
  `Err(SendError)` ⇒ `CoreError::Backend { backend: "engine", message: "scheduler queue full".into() }`
  (this is THE queue-full trigger; T11 maps it to 503 — atlas's twin `chat_blocking.rs:174-178`
  maps close-to-503 only; shisu bounds the wait so capacity pressure yields 503, not latency). Channel
  capacity: the coordinator entry sizes `mpsc::channel(cfg.max_num_seqs.max(1))` (atlas
  `serve_load.rs:754`); `spawn_engine` keeps T10's capacity constant so its FIFO loop stays
  reachable for T10's own tests. `EngineRequest` gains NO variants — cancellation rides on
  `Closed` detection (§ 6), stats ride on shared atomics (§ 8); record this rationale in a doc
  comment ("may gain variants; none needed yet — contract frozen").
- [ ] **2. `scheduler/mod.rs` (replace the Step-5.4 inline loop; keep the file's T10 doc header).**
  Structure: `struct PendingQueue { reqs: VecDeque<EngineRequest>, closed: bool }` +
  `Arc<(Mutex<PendingQueue>, Condvar)>` (atlas `mod.rs:338-369`); the tokio-side receiver task
  T10 already has (Step 5.1) keeps pushing — T13 moves the *loop body* into
  `async fn run_coordinator(model: EngineModel, ctx: u32, cfg: SchedulerConfig,
  mut rx, stats: Arc<SchedulerStats>)` running on the engine thread (T10:81 rule unchanged).
  `enum EngineModel { Plain(Box<dyn Model>), Checkpointed(Box<dyn Checkpointed>) }` +
  `as_model_mut()` upcast (T8:83) + `fn ckpt(&mut) -> Option<&mut dyn Checkpointed>`.
  Tick order per tick (atlas `mod.rs:463-469` minus rotations): `drain` (mod_helpers.rs:97-140
  blocking-condvar pattern; `parking_lot::Condvar`) → `admit` (§ 4) → if prefills pending AND
  active nonempty: one decode step over the ready set first, then ONE quantum (`batching.rs`,
  ds4 `:12609-12626`) → else decode-only or solo-prefill (`solo_chunk` width, ds4 `:12820`) →
  `sample` per row (per-seq `SampleRequest`) → `tool_stream.feed` → bounded sends (§ 6) →
  `enforce_request_deadlines` (§ 6) → `retire_finished` (mod_helpers.rs:305-351 pattern) →
  idle: `cv.wait` while nothing active/pending/!closed (atlas `mod.rs:5-11` zero-CPU rule).
  Shutdown tail (atlas `mod.rs:1082-1161`): on `closed && empty && active empty`:
  `kv_shutdown_sweep(text_of)` (T8; `text_of` from the coordinator's own `SeqSlot.prompt_text`
  bookkeeping) → best-effort Finish/Error to every live sink → `model.teardown()` →
  `tracing::info!("Scheduler stopped")`. No LoRA/MTP/spec pools, no swap pool (atlas `preempt.rs`
  NOT ported — T6's contiguous buffers leave no block pool to spill; preemption = the timeout sweep;
  `[WARN] DEVIATION` recorded in the commit body).
  Deterministic time via `trait Clock` (field `clock: Arc<dyn Clock + Send + Sync>` defaulted to
  `SystemClock`); every `Instant::now()` in the loop goes through it.
- [ ] **3. `scheduler/tool_stream.rs` (new; parser, per § Tool-call deltas).**
  State machine mirroring ds4 `:6887-6895` + `:7898-7910`: `Text → (Text | BetweenInvokes |
  BetweenParams | ParamValue | Done | Error)`, plus a `mode: Text|Tool|Suppress`. Holdback:
  any suffix that could begin an opening/closing tag is held (ds4 `raw_partial_lit`
  discipline `:7061-7065`; a held tail is flushed as `Content` once it can no longer match);
  UTF-8 split across token boundaries buffers by bytes, not chars. `feed(text) -> Vec<ToolOut>`
  pumps until the input is consumed; `finish()` at end-of-stream: unterminated envelope → drop
  + one `Refusal("tool call could not be parsed")` + `saw_call=false`; a complete call →
  `Call(ToolCall)` (`ToolDefinition`-driven coercion); after ≥1 complete call `saw_call()` drives
  `FinishReason::ToolCalls`. Never emit envelope tag text as `Content` (never-leak, ds4
  `:17874-17875`). ~350 LoC cap; if over, split `tool_stream/{dsml,json}.rs`.
- [ ] **4. `scheduler/batching.rs` (new).** `SlotTable { slots: Vec<Option<Box<SeqSlot>>>,
  max: usize }` with `admit_ready(&mut, now) -> Vec<Box<SeqSlot>>` taking pending FIFO while a
  slot is free; `ready_set(&self) -> (Vec<SessionId>, Vec<u32>)` (active, non-finished,
  non-cancelled; next-token vector row-aligned per T2's `decode_batch` contract);
  `plan(&self, cfg, now) -> TickPlan { decode: (Vec<SessionId>, Vec<u32>), prefill_quantum:
  Option<usize> }` implementing the ds4 fairness rule: every decoding context gets exactly one
  decode step before any prefill quantum; quantum = `min(cfg.mixed_quantum, remaining)` while
  shared, `cfg.solo_chunk` when nothing decodes; over-admission is impossible by construction
  (`slots.len() <= max`). Residency policy: a slot is held from `open_session` to exactly one
  `close_session` by the coordinator; resident set ≤ `max_num_seqs` ⇒ no LRU eviction of *live*
  sessions (they retire via timeout/cancel/finish — the master-plan "resident sessions" row is
  the bounded slot table; the eviction counter counts shutdown-sweep reclaims only;
  `[WARN] DEVIATION` from ds4's disk-swap preemption, which needs the block pool T6 deferred).
- [ ] **5. `scheduler/lifecycle.rs` (new).** `pub(crate) struct SeqSlot { id, session, prompt,
  prompt_text, has_tools, tool: Option<ToolStream>, out: Vec<u32>, sample: SampleRequest,
  cursor: usize (prefill progress), finished, guard_stop: Option<&'static str>,
  request_start: <Clock::Instant>, first_token_at: Option<…>, last_logits: Option<DeviceView>,
  sink: mpsc::Sender<StreamDelta>, reply taken (sent immediately after admission — before the
  first prefill, so `generate()` unblocks at admission time, T10 semantics) }`.
  `admit`: `open_session` → if ckpt-enabled and `prompt_text` non-empty: `kv_resume`
  (T8; memoize the prefix-scan per distinct `prompt_text` Arc-ptr within one tick to honour
  T8:91) → on `ResumeInfo`: cursor = `tokens_reused`, first token sampled from
  `frontier_logits` (T8 Decision 3), `stats.checkpoint_hits_total += 1`,
  `cached_prompt_tokens = tokens_reused`; else cold: cursor 0.
  `advance_prefill(&mut model, &mut slot, quantum)`: chunk = `min(len-cursor, quantum)` →
  `prefill_chunk(session, &prompt[cursor..end], is_last)` → on `is_last` keep the returned
  `DeviceView` as `last_logits` and `sample(_, 0, &mut slot.sample)` → first token →
  stamp `first_token_at`, `stats.prefill_tokens_total += chunk`.
  `advance_decode` bookkeeping after the batched `decode_batch`: per row `sample(view, row,
  &mut slot.sample)` → stop checks in derive_finish precedence order (lifecycle.rs:84-95):
  `timeout_at` expired → guard → `Other("timeout")`; `tok ∈ eos_tokens` → `Stop`;
  `out.len() >= max_tokens` or `pos >= ctx` → `Length`; `tool.saw_call()` at stream-end →
  `ToolCalls`.
  `finish(&mut model, &mut slot, reason)`: `Usage { prompt_tokens: prompt.len(), completion_tokens:
  out.len(), cached_prompt_tokens, reasoning_tokens: 0, accepted_prediction_tokens: 0,
  time_to_first_token_ms, response_tokens_per_second }` (ttft = first_token_at − request_start; tps
  = completion ÷ (now − first_token), 0.0 with none emitted) → one `Finish` best-effort → drop sink →
  `kv_save(session, &tokens_all, prompt_text, last_logits, reason_for_save, 0)` —
  `Reason::Cold` after a full cold prefill (ds4 `:14119-14121`), `Reason::Continued` at normal
  completion (ds4 `:13181,14615`) — the engine's own gates (min_tokens, frontier) decide
  whether a file appears (T8:92; T13 never re-implements them) → `close_session` → stats bump, slot
  freed. Exactly-one closer: only this module (coordinator thread) calls `close_session` (T2's
  per-session KV ownership).
- [ ] **6. `scheduler/cancel.rs` (new).** `handle_send_outcome(&SchedulerStats, &mut SeqSlot,
  SendOutcome)` — `Sent` ⇒ continue; `DeadlineAbandoned` | `Closed` ⇒ `slot.cancelled = true`,
  `stats.cancellations_total += 1` (atlas policy: an abandoned send is exactly a hung-up
  client, `send.rs:46-96`); the seq is EXCLUDED from the next `ready_set()` but retired by the
  coordinator itself at the current step boundary — worker retains ownership (ds4
  `:20911-20924`). `enforce_request_deadlines(active, now, cfg.request_timeout)`: zero-cost
  skip when no seq carries a deadline (atlas `mod_helpers.rs:250-255`), else set
  `guard_stop = Some(GUARD_REQUEST_TIMEOUT)` + `finished = true` +
  `tracing::warn!` with the atlas field set (`slot, emitted_tokens, budget_s`) — the wire
  reason then derives `"timeout"` via `FinishReason::Other(FINISH_REASON_TIMEOUT.to_string())`
  (ir `:87-116`), never `"length"` (lifecycle.rs:84-86 precedence). No `EngineRequest::Cancel`
  variant: the receiver drop is the cancel signal; proactive cancel is out of scope (T11 trimmed).
- [ ] **7. Wire the coordinator together in `mod.rs` (`run_coordinator` body).** Compose §2-§6
  exactly once per file per tick; keep every `decode_batch` result's `DeviceView` alive for the
  whole sampling+emit segment (no copies per token); batched call only over the ready set —
  single-step sessions are impossible by construction (T2 pins row i ↔ sessions[i]).
  Interleave check (`debug_asserts!`): between two consecutive `decode_batch` calls touching seq S
  at most one `prefill_chunk` per quantum — the model-mutex fairness property (master plan "bounded
  prefill quanta behind the model mutex"; here the single thread IS the mutex).
- [ ] **8. Stats + `/metrics` + `/cache`.** `SchedulerStats` (queue.rs): atomics
  `requests_total, admitted_total, active_gauge, pending_gauge, prefill_tokens_total,
  decode_tokens_total, completions_total, cancellations_total, timeouts_total,
  checkpoint_hits_total, checkpoint_tokens_reused_total, session_opens_total,
  session_closes_total, shutdown_reclaims_total, ttft_ms_sum, tokens_per_second_ema` + `render`
  (Prometheus 0.0.4 text, `shisu_` prefix — these are the counters T10's `/metrics` doc
  promised "after T13"). `lib.rs::run()`: build `SchedulerConfig` from the five knobs +
  `ShapeProfile.eos_token_id`, call `spawn_engine_with_checkpoints` (one-line switch from
  `spawn_engine`; `AppState` gains `pub sched: Arc<SchedulerStats>` — additive, precedented by
  T11's `response_store`/`conv_chat` fields), keep the join-before-return rule.
  `api/misc_handlers.rs`: `metrics()` appends `sched.render(out)`; `cache_report()` extends the T10
  disk section with `states { resident, capacity, evictions }` + top-line
  `requests,hits,hit_rate,tokens_reused,tokens_prefilled` (ds4 `:15508-15536`; `pages{}` omitted —
  `[WARN] DEVIATION`, no page pool at T6's ctx-contiguous layout).
- [ ] **9. T11/T12 coordination check (no edit).** `GenerateRequest` gained NO fields, so handler
  construction sites cannot break (`grep -rn "GenerateRequest {" src` still compiles). A follow-up
  (commit-body note, not this task) may let T11's `tokenizer/render_prompt.rs` hand rendered ChatML
  text to the engine for checkpoint keying (`render_prompt_ex`, additive to a T11-owned file).

## Tests

Pattern: T10's `tests/fake_model.rs` (scripted `Model`, `Mutex` call log) + T2's `traits_contract`
fake; `FakeClock` injected via `SchedulerConfig` — the `scheduler_*` tests are sleep-free. Each
drives real `EngineHandle::generate()` and reads the `DeltaStream` via `futures::StreamExt::next`.

- [ ] `tests/scheduler_lifecycle.rs`: two concurrent `generate()`s with two scripted sequences
  → both streams make progress every tick (interleaving; master-plan test 1); decode_batch call
  log shows one batch width 2 while both active; `session_opens == session_closes == 2`;
  `teardown_calls == 1` after join; every stream ends with exactly one `Finish`.
- [ ] `tests/scheduler_batching.rs`: N=3 requests with differing `max_tokens`; assert every
  `decode_batch` call's `sessions.len() <= cfg.max_num_seqs`, prefill chunks `<=
  cfg.mixed_quantum` while shared and `== cfg.solo_chunk` solo, and the ds4 fairness rule from
  the call log: no two prefill quanta for the same seq between its decode steps
  (interleave property). EOS stops its seq (Finish `Stop`) while others continue.
- [ ] `tests/scheduler_cancel.rs`: (a) drop the `DeltaStream` mid-generation → bounded send
  reports `Closed` → seq retired, FakeModel observes exactly one `close_session` for it
  (worker ownership, ds4 `:20911-20924`), other seqs unaffected, `cancellations_total == 1`;
  (b) abandon (receiver never polls, clock advanced past the send deadline) behaves identically;
  (c) shutdown with live seqs → each receives a terminal frame, sweep ran once.
- [ ] `tests/scheduler_timeout.rs`: `SHISU_REQUEST_TIMEOUT_MS`-equivalent cfg + FakeClock past
  `request_start + timeout` → Finish `Other("timeout")` (`as_wire == "timeout"`, not
  `"length"` — atlas `lifecycle.rs:84-86`), `timeouts_total == 1`, session closed.
- [ ] `tests/scheduler_queue_full.rs`: capacity-`max_num_seqs` channel, coordinator blocked
  (gate FakeModel's decode on a `tokio::sync::Barrier`) → `generate()` returns a
  `CoreError::Backend{backend:"engine"}` containing "scheduler queue full" within
  `queue_admission + slack` (no panic, no unbounded wait).
- [ ] `tests/scheduler_checkpoint.rs`: fake `Checkpointed` (scripted `kv_resume` →
  `Some(ResumeInfo{tokens_reused: k, …})`) → `prefill_chunk` called only for the `len-k`
  suffix (master-plan "checkpoint hit skips prefill"), `cached_prompt_tokens == k`,
  `checkpoint_hits_total == 1`; second identical prompt in same tick hits the memo (one
  `kv_resume`, two sequences).
- [ ] `tests/tool_stream.rs`: feed scripted fragments token-by-token: (i) DSML envelope (tools
  element, invoke name attribute, two parameter children) → one `Start` + ordered `ArgsDelta`s +
  one `Call` with coerced values, ZERO `Content` containing tag text; (ii) tag split across three
  tokens mid-literal; (iii) a partial closing tag that never completes → held tail flushed as
  `Content`, no leak; (iv) JSON-object flavour → same event shape; (v) malformed args JSON → one
  `Refusal`, no `Start`; (vi) tag-like prose passes through; never-leak (ds4 `:17874-17875`).
- [ ] Unit tests in `src/scheduler/batching.rs` (pure `plan`): fairness between quanta (ds4
  `:12609-12626`); widths ≤ caps; solo ⇒ `solo_chunk` (`:12820`).
- [ ] Run: `cargo test -p shisu-server scheduler` · `cargo test -p shisu-server tool_stream` ·
  `cargo clippy -p shisu-server --all-targets`; `--features metal` boot smoke as T10's gated test.

## Acceptance

- [ ] Master-plan T13 checkboxes (:238-239) green — resident sessions (bounded slot table), bounded
  prefill quanta behind the single model thread, per-session KV ownership (coordinator = sole
  closer), cancellation keeps worker ownership, `DeltaStream` to handlers unchanged — witnessed by
  the interleave, mid-stream cancel frees the session, checkpoint hit skips prefill tests.
- [ ] `grep -n "teardown()\|close_session" shisu/crates/shisu-server/src/scheduler/*.rs` hits only
  coordinator paths (T10:81 rule).
- [ ] Every `src/scheduler/*.rs` ≤ 1500 LoC (`wc -l`); T10's `tests/engine_thread.rs` untouched and
  green; T11/T12 handler files zero-diff in this commit.
- [ ] `/metrics` renders the Step-8 counters; `/cache` carries the engine section (gated boot test).

## Commit

```
git add shisu/crates/shisu-server
git commit -m "feat(server): batched scheduler coordinator — resident slots, prefill quanta, cancellation, timeouts, checkpoint hooks, tool-call deltas"
```
(Master plan: bare "Commit." → Task 1 `type: summary` convention.)
