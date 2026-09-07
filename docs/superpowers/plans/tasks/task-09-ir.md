# Task 9: shisu-ir

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 9
**Depends on:**
- Task 1 — stub crate `shisu/crates/shisu-ir/` exists; workspace deps usable as `{ workspace = true }`: `serde 1 (derive)`, `serde_json 1 (preserve_order)`, `futures 0.3`, and path dep `shisu-core = { path = …, default-features = false }` (task-01 Step 5).
- Task 2 — `shisu_core::SampleRequest { temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: u64 }` (task-02 Produces, `model.rs`). The `From` impl below is the only cross-crate coupling this task creates.

**Produces** (crate root `shisu_ir::*`; names verbatim from the master-plan SSOT — `ChatRequest`, `SamplingParams`, `ThinkingDirective`, `Message`, `ContentPart`, `Role`, `StreamDelta{Content,Reasoning,ToolCallStart,ToolCallArgs,Finish,Error}`, `DeltaStream`, `FinishReason`, `Usage`):

```rust
// re-exports (atlas ir/mod.rs list, video entries dropped, see Step 4)
pub use message::{ContentPart, ImageData, ImageSource, Message, Reasoning, Role, ToolCall};
pub use request::{ChatRequest, EffortLevel, ReasoningEffort, ResponseFormat,
                  SamplingParams, ThinkingDirective, parse_wire_effort};
pub use response::{ChatResponse, Choice, ChoiceLogprobs, FINISH_REASON_TIMEOUT,
                   FinishReason, TokenLogprob, Usage};
pub use stream::{DeltaStream, StreamDelta};
pub use tools::{FunctionDefinition, RepetitionDetectionParams, ToolChoice,
                ToolChoiceFunction, ToolDefinition};
// convert.rs — the one added API (see Step 6)
impl From<&ChatRequest> for shisu_core::SampleRequest;
```

## Global Constraints

- Rule 1: all new code under `shisu/` (this task touches only `shisu/crates/shisu-ir/`).
- Rule 2: 1500-LoC cap — largest vendored file is `request.rs` at 244 ln; `message.rs` shrinks with the video trim; no pre-split needed.
- Rule 3: vendored Rust is copied verbatim except the per-file edit list below (the Rust analogue of "byte-for-byte; only plumbing changes").
- Rule 4: no perf surface. Rule 5: no env knobs read here.
- Rule 6: `[lints] workspace = true`; lib.rs keeps task-01's `#![deny(warnings)] #![deny(clippy::all)]` and adds `#![forbid(unsafe_code)]` (task-02 "no unsafe" convention; ⚠ no atlas precedent — grep finds zero `forbid(unsafe_code)` in atlas crates — but shisu-ir is pure data so it is free).
- Rule 7: strip the `// SPDX-License-Identifier: AGPL-3.0-only` first line from every vendored file.
- Task extras: NO axum/hyper/tokio/http deps, no `AppError`-style server error types, no engine types beyond the `SampleRequest` `From` impl, and **do NOT rename any stop_reason / wire string** (T10/T11/T12 emit them verbatim; see Stop-Reason SSOT).

## Source References (verified)

⚠ **DEVIATION (assignment reference fix):** the atlas IR lives at
`crates/spark-server/src/ir/` (master plan is right; `crates/atlas-spark-server/`
does not exist). It is **6 files, 808 LoC total** — not 10–11. `CanonicalRequest`,
`CanonicalEvent`, an `Adapter` trait, and a `canonical.rs` **do not exist anywhere
in atlas** (grep verified). The narrow waist is `ChatRequest` (request direction),
`ChatResponse`/`StreamDelta`/`DeltaStream` (response direction); "one IR, N
adapters" is a module-layout pattern (`api/chat`, `anthropic/` convert into the
IR), not a trait. All later-task references below use the real names.

All atlas paths relative to `/Users/sercand/Developer/src/github.com/sercand/atlas`.

| Source | Lines | What lives there / how used |
|---|---|---|
| `crates/spark-server/src/ir/mod.rs` | 1–26 | module list + flat re-exports + `#[cfg(test)] mod tests` → becomes `shisu-ir/src/lib.rs` |
| `crates/spark-server/src/ir/message.rs` | 1–233 | `Role`(+`as_wire`/`from_wire`/`From<&str>`), `Message`(+`synthetic_system`/`prepend_text`/`text`/`media_kinds`), `MediaKind`, `ContentPart`, `VideoSource`, `ImageSource`, `ImageData`(+`from_uri`), `ToolCall`, `Reasoning`. Only external dep: `serde_json::Value` (:226). Video items trimmed (Step 4) |
| `crates/spark-server/src/ir/request.rs` | 1–244 | `ChatRequest` (23 fields), `SamplingParams` (7 `Option` fields), `ResponseFormat`, `ThinkingDirective`, `EffortLevel`, `ReasoningEffort`(+`as_str`), `parse_wire_effort` (SSOT vocab), `ThinkingDirective::is_explicit`. The ONLY file with `crate::` coupling — 3 sites, Step 3 |
| `crates/spark-server/src/ir/response.rs` | 1–144 | `ChatResponse`, `Choice`, `ChoiceLogprobs`, `TokenLogprob`, `Usage`, `FINISH_REASON_TIMEOUT` (:105), `FinishReason`(+`From<&str>` :118–130, `as_wire` :132–144). Internal-only import `super::message::ToolCall` (:9) — verbatim |
| `crates/spark-server/src/ir/stream.rs` | 1–50 | `StreamDelta` 7 variants, `DeltaStream = Pin<Box<dyn futures::Stream<Item = StreamDelta> + Send>>` (:50). External dep: `futures` |
| `crates/spark-server/src/ir/tests.rs` | 1–111 | the 5 ir unit tests (see Tests); the ir files themselves have NO internal `#[cfg(test)]` modules (grep: only `mod.rs:25`) |
| `crates/spark-server/src/tool_parser.rs` | 94–132 | `ToolDefinition`, `FunctionDefinition`, `ToolChoice`(+`is_none`), `ToolChoiceFunction` — serde types, provider-neutral (request.rs:23–24 says so) → moved into shisu-ir `tools.rs` |
| `crates/spark-server/src/api/inference_types.rs` | 21–44 | `RepetitionDetectionParams { min_pattern_size, max_pattern_size, min_count }` (serde Deserialize, snake_case) → moved into `tools.rs` |
| `crates/spark-server/src/anthropic/helpers.rs` | 58–76 | `convert_stop_reason` — Anthropic-side mapping of the canonical strings; NOT vendored here (T12 owns it); cited so the SSOT table below is complete |
| `crates/spark-server/src/anthropic/translate.rs` | 32–35 | `FinishReason::Stop` + `matched_stop.is_some()` → `"stop_sequence"` (T12-side special case) |
| `crates/spark-server/Cargo.toml` | 47–48, 80 | `serde`/`serde_json` = workspace, `futures = "0.3"` — matches shisu workspace (`serde 1 derive`, `serde_json 1 preserve_order` at atlas root `Cargo.toml:60,66`; `futures = "0.3"` task-01 Step 5). `preserve_order` matters here: `ToolDefinition.function.parameters` key order must survive to T10's minijinja `<tools>` render (atlas root Cargo.toml:61–65 rationale) |
| `crates/spark-server/tests/{integration,closure_attestation}.rs` | headers | NOT ir tests (GPU model smoke / kernel-closure hash) — nothing to carry |

## Stop-Reason SSOT (verified from `ir/response.rs` this session — never rename)

```rust
pub const FINISH_REASON_TIMEOUT: &str = "timeout";            // response.rs:105
// FinishReason::as_wire()  (response.rs:135–143)
Stop => "stop"   Length => "length"   ToolCalls => "tool_calls"
ContentFilter => "content_filter"   Other(s) => s              // lossless
// From<&str>  (response.rs:118–130): the four above map back; anything
// else => Other(input) verbatim.
```

Adjacent wire vocabularies in the same crate, also SSOT, also verbatim:
`Role::as_wire` → `"system"|"user"|"assistant"|"tool"` (message.rs:26–34);
`ReasoningEffort::as_str` → `"low"|"medium"|"high"|"xhigh"` — `Max` renders
`"xhigh"`, NEVER `"max"` (request.rs:180–187); `parse_wire_effort` accepts
`none|minimal|low|medium|high|xhigh|max`, unknown → `None` (request.rs:210–235).
The Anthropic mapping (`"stop"→"end_turn"`, `"tool_calls"→"tool_use"`,
`"length"→"max_tokens"`, `"content_filter"→"refusal"`, `"timeout"→"max_tokens"`,
unknown→`"end_turn"`, plus `"stop_sequence"` when `matched_stop` is set) lives in
T12's copy of `anthropic/helpers.rs:58–76` + `translate.rs:32–35` — recorded here
so T10/T11/T12 can pin against it; it is NOT part of shisu-ir.

## Plan

- [ ] **Step 1: `shisu/crates/shisu-ir/Cargo.toml`** — keep task-01's manifest, add
  `[dependencies] serde = { workspace = true }`, `serde_json = { workspace = true }`,
  `futures = { workspace = true }`, `shisu-core = { workspace = true }` (path dep is
  `default-features = false` at the workspace entry — nothing in shisu-ir needs a
  backend feature). No dev-deps (futures 0.3 defaults ship `executor` + `iter`).

- [ ] **Step 2: Vendor.** `cp` atlas `crates/spark-server/src/ir/{message,request,response,stream,tests}.rs`
  into `shisu/crates/shisu-ir/src/`, then in ALL five files delete the SPDX line 1
  (rule 7). Everything else verbatim except the edits in Steps 3–5.

- [ ] **Step 3: De-couple `request.rs`** — the complete coupling list (grep of the
  whole atlas ir/ dir found exactly these three `crate::` sites; message/response/
  stream/tests have zero):

  | Site | Import change | Replacement |
  |---|---|---|
  | request.rs:25 `pub tools: Vec<crate::tool_parser::ToolDefinition>` | → `crate::tools::ToolDefinition` | type moved into new `src/tools.rs` (Step 3a) |
  | request.rs:26 `pub tool_choice: Option<crate::tool_parser::ToolChoice>` | → `crate::tools::ToolChoice` | same |
  | request.rs:53 `pub repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>` | → `crate::tools::RepetitionDetectionParams` | struct moved (with its doc comment) into `tools.rs` |

- [ ] **Step 3a: `src/tools.rs`** (new, ~70 ln) — copy verbatim (minus SPDX):
  `ToolDefinition`, `FunctionDefinition`, `ToolChoice`, `ToolChoiceFunction` from
  atlas `tool_parser.rs:94–132` (serde derives kept; `serde_json` already a dep),
  and `RepetitionDetectionParams` from `api/inference_types.rs:21–44`. Header
  comment records both source paths. Do NOT copy the rest of `tool_parser.rs`
  (parsers are T11's job) nor its wire `ToolCall` (message.rs already owns the
  IR `ToolCall`).

- [ ] **Step 4: Video trim (`message.rs`)** — master plan: "drop video; keep
  `ContentPart::Image` shape for future". Delete: `MediaKind` enum (:152–162),
  `VideoSource` (:173–185), `ContentPart::Video` variant (:170),
  `Message::media_kinds` (:130–149) and the `MediaKind` doc paragraph (:152–157).
  Keep `ContentPart::{Text, Image}`, `ImageSource`, `ImageData`(+`from_uri`)
  untouched. Adjust the `ContentPart` doc comment to note video was dropped
  (vision is out of scope, master plan Scope).

- [ ] **Step 5: `src/lib.rs`** — replace the task-01 stub. Body = atlas `mod.rs`
  content minus SPDX: `pub mod` for `message, request, response, stream, tools,
  convert` + the re-export block from Produces (atlas list minus
  `MediaKind`/`VideoSource`, plus `ImageSource, Reasoning, ToolCall` — atlas
  reaches those via `ir::message::…` (e.g. `api/chat/template.rs:241`); flat
  re-export is additive and gives T11/T12 `shisu_ir::ToolCall`) + `pub use
  tools::{…}`. Attributes: `#![deny(warnings)] #![deny(clippy::all)]
  #![forbid(unsafe_code)]` + the narrow-waist doc comment from atlas mod.rs:3–8
  (reworded: surfaces are `shisu-server`'s openai/anthropic adapters).

- [ ] **Step 6: `src/convert.rs`** (new, ~45 ln) — the IR→engine bridge:

  ```rust
  impl From<&ChatRequest> for shisu_core::SampleRequest {
      fn from(r: &ChatRequest) -> Self {
          shisu_core::SampleRequest {
              temperature: r.sampling.temperature.unwrap_or(1.0),
              top_k: r.sampling.top_k.map(|v| v as i32).unwrap_or(0),
              top_p: r.sampling.top_p.unwrap_or(1.0),
              min_p: 0.0,                    // NOT carried — see below
              rng: r.seed.unwrap_or(0),      // T10 reseeds when seed is None
          }
      }
  }
  ```

  Fields NOT carried, documented in the impl's doc comment:
  - `min_p`: `SamplingParams.min_p` exists but is NOT mapped — the impl fills
    `0.0` (no-op); T10's sampling setup (the `shisu-server` analogue of atlas
    `api/chat/sampling_setup.rs`) overwrites it from the resolved server preset,
    because atlas semantics are `None` = server default applies downstream
    (request.rs:84–86) and a `From` impl cannot see that preset.
  - `repetition_penalty` / `presence_penalty` / `frequency_penalty` /
    `top_n_sigma`: `SampleRequest` has no such fields (task-02 field list);
    they stay on `ChatRequest.sampling` for T10/T13 to read directly.
  - `rng`: mapped from `ChatRequest.seed`; when `seed` is `None` the value `0`
    is a placeholder T10 MUST replace with an entropy seed before sampling.
  - `⚠ DEVIATION:` the assignment named this `From<&CanonicalRequest>` — no such
    type exists in atlas (see Source References); `ChatRequest` is the canonical
    request envelope.

- [ ] **Step 7: `tests.rs`** — adapt the carried tests (see Tests) and append the
  new ones. Keep `#[cfg(test)] mod tests;` in `lib.rs` (atlas mod.rs:25–26 pattern).

## Tests

Pure host code; identical on macOS and Linux; no tolerances, no GPU, no env vars.
Run from `shisu/`: `cargo test -p shisu-ir` (workspace `warnings = "deny"` applies).

Carried from atlas `ir/tests.rs` (the complete atlas ir test inventory — 5 fns;
`spark-server/tests/` holds none for ir):
- `role_wire_roundtrip` — carry verbatim.
- `text_concatenates_text_parts_in_order_ignoring_images` — carry; drop the
  `media_kinds()` assertion (line 39) with the video trim.
- `media_kinds_follow_content_order_across_modalities` — DROP (pure video test).
- `assistant_tool_call_and_reasoning_are_first_class` — carry; drop the
  `media_kinds().is_empty()` assertion (line 94).
- `tool_message_carries_error_flag_and_call_id` — carry verbatim.

New (each pins a contract a later task consumes; no atlas ir test covers them):
- `finish_reason_wire_roundtrip` — `From<&str>`→`as_wire` round-trip for
  `"stop"|"length"|"tool_calls"|"content_filter"`, `FINISH_REASON_TIMEOUT`
  round-trips via `Other("timeout")` verbatim, and an arbitrary
  `Other("weird_engine_reason")` survives unchanged. This is the T10/T11/T12
  stop-string SSOT guard.
- `wire_effort_vocabulary` — `parse_wire_effort` over all seven accepted
  spellings (assert `"xhigh"` and `"max"` yield the identical tuple, and
  `ReasoningEffort::Max.as_str() == "xhigh"`), unknown → `None`.
- `sample_request_mapping` — full fixture `ChatRequest` (helper fn; the struct
  has no `Default`) with temperature 0.7 / top_k 40 / top_p 0.9 / min_p 0.1 /
  seed 42 → `SampleRequest { 0.7, 40, 0.9, min_p: 0.0, rng: 42 }` (min_p NOT
  carried is asserted, not accidental); silent-sampling request → neutral
  defaults `1.0 / 0 / 1.0 / 0.0 / 0`.
- `delta_stream_is_send_and_pollable` — build a `DeltaStream` via
  `Box::pin(futures::stream::iter([Content, ToolCallStart, ToolCallArgs,
  Finish]))`, drive it with `futures::executor::block_on` + `StreamExt::next`,
  assert the exact variant sequence. Pins the `Pin<Box<dyn Stream + Send>>`
  contract the T10 scheduler moves across threads.

⚠ **DEVIATION (assignment test list):** "adapter round-trip openai→canonical→
anthropic and back", "SSE framing", and "normalize/tool_args" are NOT T9 tests.
Wire types and SSE encoders live in `shisu-server` (T11 `delta_to_chunk_events`,
T12 `anthropic_sse_from_deltas`) — vendoring them here violates this task's
non-goals; and atlas ir has no `normalize`/`tool_args` helper (grep verified —
`ToolCall.arguments` is a plain `serde_json::Value`). The round-trip through the
narrow waist is exercised by T11/T12 adapter tests; T9 pins the neutral
vocabulary those tests compare against.

## Acceptance

Mirrors master plan Task 9 checkboxes ("copy+trim atlas ir/{mod,message,request,
response,stream}.rs into shisu-ir/ (drop video; keep ContentPart::Image)" +
"copy applicable atlas ir/tests.rs subset"):

- [ ] `shisu-ir/src/{lib,message,request,response,stream,tools,convert,tests}.rs`
  exist; every file < 1500 LoC (largest ≈ request.rs 244 ln).
- [ ] `grep -rn 'crate::tool_parser\|crate::api\|AppError\|axum\|hyper' shisu/crates/shisu-ir/src` → no output (all three coupling sites replaced; no HTTP).
- [ ] `grep -rn 'Video\|MediaKind' shisu/crates/shisu-ir/src` → no output; `ContentPart::Image` + `ImageSource` + `ImageData` still present.
- [ ] `grep -rn 'SPDX' shisu/crates/shisu-ir/src` → no output (rule 7).
- [ ] `lib.rs` has `#![forbid(unsafe_code)]` alongside the task-01 deny attrs.
- [ ] Stop-reason strings byte-equal to the SSOT table (`"stop"`, `"length"`,
  `"tool_calls"`, `"content_filter"`, `FINISH_REASON_TIMEOUT = "timeout"`) — file inspection.
- [ ] `impl From<&ChatRequest> for shisu_core::SampleRequest` exists in
  `convert.rs` with the not-carried-fields doc comment.
- [ ] `cargo test -p shisu-ir` green on macOS (4 carried/adapted + 4 new tests);
  no new crates.io deps beyond the workspace entries task-01 already declared.

## Commit

```sh
git add shisu/crates/shisu-ir
git commit -m "feat: shisu-ir neutral chat IR (atlas spark-server narrow waist)"
```
