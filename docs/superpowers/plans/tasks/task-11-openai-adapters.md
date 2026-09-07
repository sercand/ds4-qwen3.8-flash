# Task 11: OpenAI adapters

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 11
**Depends on:**
- T10 (`task-10-server-skeleton.md`): `AppState{engine:EngineHandle,model_id,ctx,started_at}`,
  `EngineHandle::generate(GenerateRequest) -> shisu_core::Result<DeltaStream>`,
  `AppError`/`ApiSurface::OpenAi`, the three `not_implemented(OpenAi, …)` router stubs T11
  replaces, `tokenizer::{chat_render,jinja_helpers,message_preprocess}` (T10 Step 11: "T11's
  golden fixtures exercise the render path"), `api/inference_types::{ModelInfo,
  ModelListResponse}` (T10 Step 7: "T11 may move these into `openai/` — it owns that rename").
- T9: `shisu_ir::{ChatRequest,SamplingParams,ThinkingDirective,Message,ContentPart,Role,
  ToolCall,StreamDelta,DeltaStream,FinishReason,Usage,ChatResponse,Choice,ToolDefinition,
  ToolChoice}` + `From<&ChatRequest> for SampleRequest` (convert.rs). `StreamDelta` = atlas's
  **7** variants incl. `Refusal{text}` (atlas `ir/stream.rs:11–45`; the master-plan SSOT line
  omits it; T9 carried the file verbatim, so it exists).
- T2/T6 only through the T10 boundary. **T12** consumes `ids::uuid_v4`,
  `tokenizer::render_prompt`, `api::blocking::collect_chat_response` (hub-coordinated).
  **T13 boundary:** the engine/scheduler *emits* `ToolCallStart/ToolCallArgs/Refusal` deltas
  (atlas counterparts `tool_parser/streaming*`, `api/chat_stream/handle_token.rs`); T11
  consumes them as-is and never parses tool-call text.

**Produces** (crate root `shisu_server::*`; T12/T13/T20 build on these exact names):

```rust
// ids.rs — vendored atlas ids.rs:1–58 (SPDX stripped; pub(crate) -> pub)
pub fn uuid_v4() -> String;                       // ids.rs:18–50 (/dev/urandom, ts fallback)
pub fn unix_timestamp() -> u64;                   // ids.rs:53–57  ("created" fields)
// openai/mod.rs — id minters, atlas openai/mod.rs:36–44 verbatim
pub fn new_completion_id() -> String;             // "cmpl-" + uuid    (text_completion)
pub fn new_chunk_id() -> String;                  // "chatcmpl-" + uuid (chat chunks)
// tokenizer/render_prompt.rs — the single render entry BOTH T11 and T12 handlers call
pub fn render_prompt(req: &shisu_ir::ChatRequest) -> Result<Vec<u32>, AppError>;
// api/blocking.rs — shared DeltaStream -> ChatResponse collector (T12 blocking reuses it)
pub async fn collect_chat_response(deltas: shisu_ir::DeltaStream, model: String,
    stop_sequences: &[String]) -> Result<shisu_ir::ChatResponse, AppError>;
pub(crate) fn strip_stop_sequences_matched(text: String, stops: &[String])
    -> (String, Option<String>);                  // atlas api/inference_impl.rs:450–480
pub(crate) fn tokenize_stop_sequences(stops: &[String]) -> Vec<u32>; // atlas :417–441 text-side
// openai/sse.rs — shared SSE encoder (atlas openai/encode_stream.rs:21–175)
pub(crate) fn delta_to_chunk_events(d: &StreamDelta, model: &str, id: &str,
    include_usage: bool) -> Vec<axum::response::sse::Event>;
fn delta_to_payloads(d: &StreamDelta, model: &str, id: &str, include_usage: bool)
    -> Vec<String>;                               // test seam (axum Event is write-only)
pub(crate) fn encode_sse_response(deltas: DeltaStream, model: String,
    include_usage: bool) -> axum::response::Response;   // role prologue + [DONE] + KeepAlive
pub(crate) async fn encode_completions_sse(deltas: DeltaStream, model: String,
    echo_prompt: Option<String>, include_usage: bool) -> axum::response::Response;
pub(crate) async fn encode_responses_sse(deltas: DeltaStream, resp: ResponsesStreamEnvelope)
    -> axum::response::Response;                  // named-event frames, sequence_number += 1
// openai/responses_lowering.rs — verbatim atlas :30–33
pub fn lower_responses_to_chat(r: ResponsesRequest,
    resolve_prior: impl FnOnce(&str) -> Option<Vec<IncomingMessage>>)
    -> Result<ChatCompletionRequest, LowerResponsesError>;   // ::BadRequest|::PriorNotFound
// response_store.rs — memory-only LRU+TTL (atlas response_store.rs + store_impl.rs trimmed)
pub enum StoredKind { Response, ChatCompletion }             // id_prefix(): "resp_"/"chatcmpl-"
pub struct StoredEntry { id, kind, model, created_at, messages: Vec<IncomingMessage>, body: Value }
pub struct GetResult  { model, created_at, messages, body }
impl ResponseStore { pub fn new(max_entries: usize, ttl: Duration) -> Arc<Self>;
    pub fn insert(&self, StoredEntry); pub fn get(&self, &str, StoredKind) -> Option<GetResult>;
    pub fn delete(&self, &str, StoredKind) -> bool; pub fn len(&self) -> usize;
    pub fn ttl(&self) -> Duration; }
// handlers (router.rs swaps the three T10 stubs for these; + 1 new GET row);
// state.rs gains ONE field: pub response_store: Arc<ResponseStore>
pub async fn openai::chat::chat_completions(State<Arc<AppState>>, Bytes) -> Response;
pub async fn openai::completions::completions(State<Arc<AppState>>, Bytes) -> Response;
pub async fn openai::responses::responses_endpoint(State<Arc<AppState>>, Bytes) -> Response;
pub async fn openai::responses::get_stored_response(State<Arc<AppState>>, Path<String>) -> Response;
```

## Global Constraints

Rules 1–7 (master plan), one line each:
1. All code under `shisu/crates/shisu-server/`; pure Rust, no host C/ObjC.
2. 1500-LoC cap — the Plan tree keeps atlas's split shape (design choice, not cap-forced); `sse.rs` ships as `sse.rs`+`sse_responses.rs` day one.
3. Vendored Rust verbatim minus SPDX (the Rust analogue of byte-for-byte); only the recorded
   trim lists change semantics, each noted in the file's header comment.
4. No perf gate; T20 owns baselines.
5. Env knobs `SHISU_`-prefixed only. New: `SHISU_STORE_MAX_ENTRIES` (10 000) /
   `SHISU_STORE_TTL_SECONDS` (86 400) — atlas's `ATLAS_STORE_*` (`store_impl.rs:19–27`) renamed;
   `SHISU_TEST_MODEL` gates the `#[ignore]` e2e only.
6. `warnings=deny` + clippy deny; `AppError` the only error type (thiserror); `anyhow` at the
   binary boundary; `parking_lot` for the store lock; `tracing` logs.
7. No SPDX line in any vendored file.

Task-specific:
- Consume `DeltaStream` **only** from T10 `EngineHandle::generate`; never touch `Box<dyn Model>`
  or the engine outside that channel contract.
- No tokenizer/scheduler logic here — T13 owns the coordinator, residency, and the tool-call
  parser that emits the deltas. T11 renders prompts (via T10's tokenizer modules) + assembles wire output.
- Stop reasons on the OpenAI surface are the **identity** `FinishReason::as_wire()`
  (`"stop"|"length"|"tool_calls"|"content_filter"`, `Other(s)` lossless incl. `"timeout"` — T9
  SSOT); the Anthropic `end_turn|tool_use|max_tokens|refusal|stop_sequence` mapping is T12's
  (`anthropic/helpers.rs:58–76`, `translate.rs:32–35`). T11 NEVER re-maps.
- Run cargo from `shisu/`; `cargo test -p shisu-server` only. T20 owns workspace gates.

⚠ **DEVIATION (assignment "413 context overflow"):** no 413 exists anywhere. ds4 answers
context overflow with **400** `{"error":{…,"code":"context_length_exceeded","n_prompt_tokens":N,
"n_ctx":M}}` (`ds4_server.c:6675–6703`; its own test pins `HTTP/1.1 400` at `:17095`; route gate
`:15727–15731`). atlas also 400 ("Prompt too long …", `api/chat/mod.rs:373–379`;
`api/completions.rs:117–125`). shisu = 400 with ds4's richer body via the atlas builder shape.
⚠ **DEVIATION (assignment "responses surface … if absent"):** atlas HAS a full Responses
surface (`openai/responses.rs`, `responses_lowering.rs`, `api/responses*.rs`, `response_store/`);
it is vendored/adapted, not re-invented. Router rows per atlas `serve_router.rs:46–58` minus
conversations/cancel/input_items (non-goals).
⚠ **DEVIATION (master-plan Files list):** atlas splits handlers into `api/` for its 500 cap
(`api/chat/mod.rs:3–4` comment); the master list names only wire/encoder files. Handlers go in
`openai/{chat,completions,responses}.rs` (assignment), shared response-direction helpers in
`api/blocking.rs` (new, sibling of `error.rs`); `encode_stream.rs` ships as `sse.rs` — one seam
per wire format (atlas `encode_stream.rs:3–7` rationale).

## Source References (verified)

Atlas paths relative to `/Users/sercand/Developer/src/github.com/sercand/atlas`; ds4 paths to
the repo root. Every row opened this session.

| Source | Lines | What lives there / how used |
|---|---|---|
| `crates/spark-server/src/openai/mod.rs` | 1–44 | module list + `new_completion_id`("cmpl-")/`new_chunk_id`("chatcmpl-") → `openai/mod.rs` verbatim |
| `crates/spark-server/src/openai/chat_request.rs` | 1–572 | `ChatCompletionRequest` wire, `StreamOptions{include_usage,:221–226}`, `JsonSchemaSpec`, `ThinkingConfig`/`ReasoningConfig`, `ChatTemplateKwargs`, `validate_reasoning_effort:336`, `client_thinking_directive:385` → **trimmed** (Step 2): drop lora/beams/MTP/dry/lz/tscg, dump/service_tier, web-search/annotations, prompt-logprobs, grammar; `image_url`/`video_url` ⇒ 400 (vision out of master Scope) |
| `crates/spark-server/src/openai/chat_message.rs` | 1–474 | `IncomingMessage`/`ParsedContent`, `synthetic_system/synthetic_user_text`, `from_responses_input_item:152` (Responses `input[]` → messages — the only reason media-shaped fields stay; drop MediaRef fetch branches) |
| `crates/spark-server/src/openai/chat_response.rs` | 1–224 | `ChatCompletionResponse`(+`new:161`,`with_tool_calls:193`), `ChatChoice`, wire `Usage`/`PromptTokensDetails`/`CompletionTokensDetails`, `ModelInfo::advertise:149` (absorbs T10's ModelInfo rename) |
| `crates/spark-server/src/openai/stream_chunk.rs` | 1–332 | `ChatCompletionChunk/ChunkChoice/ChunkDelta` + ctors `role:70 content:121 reasoning:96 tool_call_start:148 tool_call_args:189 done:222 usage_only:250 refusal:268 final_no_usage:295 with_token_ids:324`; `fp_atlas`→`fp_shisu`; absorbs wire `ToolCall/FunctionCall` (`tool_parser.rs:155–167`) + `ChunkToolCall/ChunkFunction` (`:175–~200`) — T9 left them out ("parsers are T11's job") |
| `crates/spark-server/src/openai/to_ir.rs` | 1–435 | `From<&IncomingMessage> for Message:15–70`, `From<ChatCompletionRequest> for ir::ChatRequest:71–149` (infallible), `resolve_top_logprobs:152` (drop); tests minus media rows |
| `crates/spark-server/src/openai/encode.rs` | 1–146 | `encode_chat_response:19–123`: ir::ChatResponse → wire JSON, `chatcmpl-`+ir.id (:84), usage details (:25–41), `store:true` insert (:100–114); drop annotations/dump/logprobs arms |
| `crates/spark-server/src/openai/encode_stream.rs` | 1–369 | **the SSE seam**: `delta_to_chunk_events:21–31`, `delta_to_payloads:36–112` (mapping table below), `chunk_json:114`, `wire_usage:122–140` (total=prompt+completion; audio 0), `encode_sse_response:145–175` (`Sse::new().keep_alive(KeepAlive::default())`, role prologue, `data:[DONE]` :160–161); tests :177–369 seed our fixtures |
| `crates/spark-server/src/openai/completions.rs` | 1–349 | `CompletionRequest:13–115` (`stream_options:94–96`), `PromptInput:150–165` (untagged Text/TokenIds), `CompletionResponse::{new:189,from_choices:204}` (`cmpl-`), `CompletionChunk` ctors `text:243 echo:261 finish_no_usage:284 usage_only:302 done:314`; drop `Tokenize*:335–349`, `CompletionLogprobs:117` |
| `crates/spark-server/src/openai/responses.rs` | 1–375 | `ResponsesRequest` (input/instructions/previous_response_id/store), output/item/content/summary types, `ResponsesStreamEvent` 17 variants :198–325 (vocab doc :179–195), `responses_event_name:343–374` |
| `crates/spark-server/src/openai/responses_lowering.rs` | 1–237 | `LowerResponsesError{BadRequest,PriorNotFound}`, `lower_responses_to_chat:30–237`: prior prepend :37–46, instructions→system :48–54, input match :55–71, tools split + hosted-tool errors :73–149, field map :150–236 |
| `crates/spark-server/src/api/completions.rs` | 1–503 | handler pattern: `resolve_prompts:43–84`, `validate_token_ids:86–94`, JSON/empty/prompt-too-long 400s :103–125, param 400 table :137–183, stream `n>1||prompts>1` 400 :254, `completions_stream:279–495` (echo decode :300–304, include_usage :290), `not_supported:501` |
| `crates/spark-server/src/api/chat/mod.rs` | 1–511 | handler pattern: self-parse `from_slice` 400 :108–116, effort 400 :121–123, prompt-too-long 400 :373–379, `ChatOutcome{Blocking(Box<ChatResponse>),Streaming(DeltaStream)}:49–58`, `chat_completions_inner:214`; DROP auto-swap :125–167, dump :183–193, vision, beams :295–300 |
| `crates/spark-server/src/api/chat_phases.rs` | 1–93 | the validation message table to reproduce verbatim: empty messages :24–28, >2048 :32–36, temperature 0..2 :42–46, top_p :60–64, max_tokens≥1, n 1..128 :72–76, tool_choice vocab + required-without-tools :80–92 |
| `crates/spark-server/src/api/chat_blocking{,_choice}.rs` | 1–559; 1–215 | blocking-assembly shape (`run_blocking_path:71`, usage sums :112–119, `build_choice_message`) — superseded by `collect_chat_response` over the T10 `DeltaStream` (n==1); the n>1 scheduler loop is NOT vendored |
| `crates/spark-server/src/api/inference_impl.rs` | 417–510 | `tokenize_stop_sequences:417`, `strip_stop_sequences:443`, `strip_stop_sequences_matched:450–480` (fills `matched_stop`), `extract_thinking:483` → `api/blocking.rs` |
| `crates/spark-server/src/api/compact.rs` | 174–240 | error builders mirrored via `AppError`: `openai_error_response:174`, `…_with_param:180` (param e.g. `messages[0].role`; code e.g. `context_length_exceeded` :178–179), `type_for_status:205–215`, `error_body:222–239` (`{error:{message,type,param,code,…}}`) |
| `crates/spark-server/src/api/responses.rs` | 1–165 | `responses_endpoint:48–165`: 400s :54–78, prior resolve closure over `store.get(id, StoredKind::Response).map(|e| e.messages)` :105–110, lowering 400 arms :112–122 (`param:"previous_response_id"`, `code:"response_not_found"`), store hand-off :158–161 |
| `crates/spark-server/src/api/responses_stream.rs` + `responses_translate.rs` | 498; 249 | `responses_endpoint_stream:47` deltas→named events (sequence_number per `responses.rs:193`); blocking translate arms :97–101/:180–184 → shisu `sse_responses.rs` + `responses.rs` handler halves |
| `crates/spark-server/src/response_store.rs` + `store_impl.rs` | 1–431; 1–198 | `StoredKind:39–49` (id prefixes), `StoredEntry:56`, `ResponseStore:405–426`, `GetResult:421–426`, `Inner{map,order VecDeque}:416–419`; ops `from_env:23` (env rename per Rule 5), `with_config:75`, `insert:117`, `get:139–165` (kind-mismatch ⇒ None, TTL sweep), `delete:167`, `len/ttl:179–189`. DROP `FilesystemBackend/DiskOp:111–403` |
| `crates/spark-server/src/api/stored.rs` | 40–100 | GET-by-id pattern: `Json(entry.body)`; miss ⇒ 404 "…may have expired, or was never stored (set store: true…)" :93–96 → `get_stored_response` |
| `crates/spark-server/src/ids.rs` | 1–58 | `getrandom:9–15`, `uuid_v4:18–50`, `unix_timestamp:53–57`; header :3–6 = prefixes belong to the wire-format owner → T11 owns the file, T12 applies `msg_` itself |
| `crates/spark-server/src/ir/{stream,response}.rs` | 11–50; 12–85 | consumed, not vendored: 7-delta `StreamDelta`, `DeltaStream:50`; `ChatResponse/Choice` (incl. `matched_stop:43`), `Usage{prompt_tokens,completion_tokens,cached_prompt_tokens,reasoning_tokens,accepted_prediction_tokens,time_to_first_token_ms,response_tokens_per_second}:67–85` = the usage-accounting input |
| ds4 `ds4_server.c` | 6668–6703, 15727–15731, 17085–17116 | `request_exceeds_context` (prompt ≥ ctx boundary, :6668–6673) + `http_error_context_length_exceeded` 400 body (`code`, `n_prompt_tokens`, `n_ctx`; anthropic twin shape :17116); route-level gate |
| ds4 `ds4_server.c` | 6745–6760, 6779–6856, 7002–7047 | chat SSE writers: frame `"object":"chat.completion.chunk"` :6746; `sse_usage_chunk` gated by `stream_include_usage` :6798, `"choices":[],"usage":` :6805, `append_openai_usage_json:6779–6794` (**adds cache_read/cache_write fields**); `sse_done` = usage chunk then `data: [DONE]\n\n` :6819–6823; finish writer :6825–6856; delta + tool-arg writers :7002–7047; self-tests :17455–17462, :17504–17540, :17636–17643 pin the vocabulary |
| ds4 `ds4_server.c` | 8114–8510, 9293–9295, 17855–17875 | Responses SSE (the richer ds4 set, **notes only — shisu follows atlas**): `response.created` :8114; `output_item.added` reasoning :8127 / message :8231 / function-call :8400; `item.done` :8214/:8301; `output_text.delta/.done` :8257/:8272; terminal `completed|failed|incomplete` :8501–8505; usage `{input_tokens,output_tokens,cache_read_input_tokens,cache_creation_input_tokens}` :9293–9295; `function_call_arguments.delta` :17855–17857; raw `<tool_call>` must never leak :17874 |
| ds4 `ds4_server.c` | 9467–9568, 15670–15717 | `content_block_start/delta/stop` = ds4's Anthropic-side set → T12's territory, attributed only; route table (T10-verified): the three OpenAI POSTs T11 must answer |

## Plan

Module tree (`shisu/crates/shisu-server/src/`, one file per line; `#` = atlas source; est. LoC):

```text
ids.rs                            # ids.rs:1–58                          (~60)
response_store.rs                 # response_store.rs + store_impl.rs    (~330, mem-only)
api/blocking.rs                   # inference_impl:417–510 + collector   (~200)
openai/mod.rs                     # openai/mod.rs                        (~60)
openai/chat_request.rs            # (trimmed per Source row)             (~420)
openai/chat_message.rs            # (trimmed)                            (~430)
openai/chat_response.rs                                                  (~230)
openai/stream_chunk.rs            # + wire ToolCall/ChunkToolCall        (~370)
openai/to_ir.rs                   # (trimmed)                            (~330)
openai/encode.rs                  # (trimmed)                            (~110)
openai/sse.rs                     # encode_stream.rs + completions half  (~460)
openai/sse_responses.rs           # responses_stream.rs state machine    (~300)
openai/chat.rs                    # api/chat/mod.rs (trimmed) + dispatch (~350)
openai/completions.rs             # wire types + api/completions.rs half (~470)
openai/responses.rs               # types + api/responses.rs handler     (~440)
openai/responses_lowering.rs      # verbatim (trimmed)                   (~230)
tokenizer/render_prompt.rs        # wires T10 tokenizer module           (~120)
```

- [ ] **Step 1: `ids.rs` + `openai/mod.rs`.** Vendor `ids.rs:1–58` minus SPDX, fns → `pub`;
  `openai/mod.rs` = atlas list + both id minters verbatim. No new external deps.
- [ ] **Step 2: wire types.** Vendor the five wire files with the Source-table trims.
  `stream_chunk.rs` absorbs `tool_parser.rs:155–167` (`ToolCall{call_type` serde-renamed
  `type}`) + `:175–~200` (`ChunkToolCall/ChunkFunction`); `fp_atlas`→`fp_shisu` in the nine
  ctors; `system_fingerprint` stays `Some(...)`.
- [ ] **Step 3: `to_ir.rs`.** Vendored minus `resolve_top_logprobs` + media test rows; the
  `From` stays infallible (validation is handler-side, atlas :72–74 comment); thinking
  directive/effort flow verbatim (`client_thinking_directive:385`, T9's `parse_wire_effort`).
- [ ] **Step 4: `api/blocking.rs`.** Vendor `strip_stop_sequences_matched`/`tokenize_stop_sequences`
  (text-side only; engine stop tokens are T13) from atlas :417–480. `collect_chat_response`
  drives the stream with `futures::StreamExt::next`: accumulate Content/Reasoning/Refusal;
  `ToolCallStart{index,id,name}` opens a slot, `ToolCallArgs{index,fragment}` appends
  (out-of-order index ⇒ `AppError::Internal`); `Finish` sets reason+usage; `Error` ⇒ Internal.
  Post-pass: strip stop sequences → fill `Choice.matched_stop` (earliest match wins; OpenAI
  spec: returned text must not contain the stop), matched ⇒ downgrade reason to `Stop`;
  `tool_calls` via `serde_json::from_str(&args)`, parse-failure keeps `Value::String(raw)`
  (lossless, atlas PCND rule). Mints id/created via `ids`, single `Choice{index:0}`; the
  caller fills `usage.prompt_tokens` (it knows the rendered length) — document it.
- [ ] **Step 5: `openai/sse.rs`.** Vendored seam verbatim (incl. its tests module → seeds the
  SSE fixtures). Mapping — one `data:` frame per row unless noted:

  | `StreamDelta` | chunk (`ChatCompletionChunk` ctor) |
  |---|---|
  | (prologue) | `role_chunk` — `delta.role:"assistant"` |
  | `Content{text,token_ids}` | `content_chunk` — `delta.content=text`; `with_token_ids` |
  | `Reasoning{text,token_ids}` | `reasoning_chunk` — `delta.reasoning_content` only, no `content` mirror (`stream_chunk.rs:46–51`) |
  | `ToolCallStart{index,id,name}` | `tool_call_start_chunk` — `role:"assistant"`, `delta.tool_calls:[{index,id,type:"function",function:{name,arguments:""}}]` |
  | `ToolCallArgs{index,fragment,token_ids}` | `tool_call_args_fragment` — `delta.tool_calls:[{index,function:{arguments:fragment}}]` |
  | `Refusal{text}` | `refusal_chunk` — `delta.refusal` (emitter arrives with T13; map ships now) |
  | `Finish`, `include_usage=false` | `done_chunk(reason.as_wire(), wire_usage)` |
  | `Finish`, `include_usage=true` | **two frames**: `usage_only_chunk` (`choices:[]`, usage) THEN `final_chunk_no_usage` (residual token_ids ride the final frame; `encode_stream.rs:89–99`) |
  | `Error{message}` | forwarded verbatim as `data:<message>` (`encode_stream.rs:110`) |
  | (terminator) | `data: [DONE]` (`encode_stream.rs:160–161`; ds4 parity `ds4_server.c:6822`) |

  Completions half: `encode_completions_sse` — `text_chunk` per Content, Reasoning dropped
  (legacy surface, no thinking: atlas `completions.rs:334`), optional `echo_chunk` first when
  `echo:true` (caller-decoded prompt, atlas `api/completions.rs:300–304`), same usage/finish/
  `[DONE]` framing, never `tool_calls` (atlas :327–340).
- [ ] **Step 6: `openai/sse_responses.rs`.** State machine emitting `ResponsesStreamEvent`s
  framed `event: <responses_event_name(e)>\ndata: {json}\n\n`, `sequence_number` +1 per event
  (`responses.rs:193`). Sequence: `Created` → Content opens `OutputItemAdded(message)`+
  `ContentPartAdded` → `OutputTextDelta`s; Reasoning opens a `reasoning` item + the
  `ReasoningSummary*` quartet; closes `OutputTextDone`+`OutputItemDone`; ToolCall →
  `OutputItemAdded(function_call)`+`FunctionCallArgumentsDelta`+`…Done`; terminal
  `Completed{ResponsesResponse}` with usage (`input_tokens`/`output_tokens` naming per the
  response types; ds4 parity `:9293`) or `Failed` (`:8501–8505` equivalence). Keep per-delta
  flush + `KeepAlive::default()` (T10 Step 7 pattern).
- [ ] **Step 7: `openai/chat.rs` handler.** Flow (atlas `api/chat/mod.rs:92–405` minus
  auto-swap/dump/vision/beams/rate-limit): parse `Bytes` (400 "Invalid request JSON") →
  `validate_reasoning_effort` (400) → validation table (chat_phases.rs:24–93 verbatim) →
  `From` → `render_prompt` (Step 9) → context guard `prompt.len() >= state.ctx` ⇒ 400
  `context_length_exceeded` body (⚠ row above; boundary `ds4_server.c:6668–6673`) →
  `SampleRequest::from(&req)` (T9 convert.rs; min_p/seed server-resolved engine-side per T9's
  not-carried doc) with `max_tokens` clamped to `ctx - prompt_len` → stream: `generate` →
  `encode_sse_response` (400 if `n>1`, atlas `chat_stream_dispatch.rs:68–71` parity); else
  `generate` → `collect_chat_response` → `encode_chat_response` (`store:true` inserts
  `StoredKind::ChatCompletion` under `chatcmpl-<uuid>`). Channel closed ⇒ 503 "Scheduler queue
  full" (atlas `chat_blocking.rs:174–178`).
- [ ] **Step 8: completions + responses handlers.** Completions per atlas
  `api/completions.rs:97–277`: `resolve_prompts:43–84`, token-id vocab check (400, :86–94),
  param 400 table :137–183 verbatim, `prompts>1||n>1` stream 400 :254. Responses per atlas
  `api/responses.rs:48–165`: resolve `previous_response_id` via the store closure (:105–110)
  → `lower_responses_to_chat` (:112, the two 400 arms :114–122) → run the Step 7 chat
  pipeline inner fn → encode `ResponsesResponse` (`resp_` id, translate pattern) or
  `encode_responses_sse`; on completion insert transcript+body as `StoredKind::Response`.
  `GET /v1/responses/{id}` → `Json(entry.body)` / 404 (`stored.rs:92–96`).
- [ ] **Step 9: `tokenizer/render_prompt.rs`.** The single entry: `message_preprocess` →
  `chat_render` (minijinja, `preserve_order`, custom `tojson` — T10 Step 11) → tokenizer
  `encode(add_special_tokens=false)`; errors ⇒ `AppError::BadRequest("Tokenization error: …")`
  (atlas `template.rs:102–107`). qwen35 template from the GGUF-adjacent chat_template (T10
  boot resolves it); first real exercise of the `<tools>` render path (T9 preserve_order).
- [ ] **Step 10: `response_store.rs`.** Trimmed vendor (Source row): `parking_lot::Mutex`,
  `VecDeque` LRU order, TTL-on-read sweep, kind-mismatch ⇒ None without evicting, insert
  evicts `pop_front` past `max_entries`; env fallbacks `SHISU_STORE_*` only when args are
  defaults + `with_config` test hook. No filesystem backend.
- [ ] **Step 11: wire into T10.** `lib.rs`: add `pub mod api, ids, openai, response_store;`.
  `state.rs`: +`pub response_store: Arc<ResponseStore>` (built in `run()`). `router.rs`: swap
  the three OpenAI stubs to `post(openai::chat::chat_completions)` /
  `post(openai::completions::completions)` / `post(openai::responses::responses_endpoint)`,
  add `get(openai::responses::get_stored_response)` on `/v1/responses/{id}` (auth-free path?
  No — middleware gates `/v1/` prefixes, T10 Step 9). Move `ModelInfo/ModelListResponse` →
  `openai/chat_response.rs` (T10-sanctioned; `misc_handlers` import re-points). `error.rs`:
  give `AppError::BadRequest` optional `param`/`code` (`bad_request(msg, param, code)`)
  preserving the existing OpenAI body + `type_for_status`.
- [ ] **Step 12: carried tests.** atlas `openai/tests/{chat_wire,responses,completions}.rs` +
  the `encode_stream.rs`/`to_ir.rs` test modules ride into the files they test (minus
  video/annotation/media rows); they seed, not replace, the Tests section.

## Tests

`cd shisu && cargo test -p shisu-server` (macOS, no GPU/model; fake engine = T10's re-declared
`FakeModel`). Bodies via `axum::body::to_bytes`; SSE asserted through `delta_to_payloads`
(raw `data:` strings — axum `Event` is write-only, `encode_stream.rs:33–35`).

- `tests/openai_wire.rs` — fixture JSON → wire → `ChatRequest` field landing; the validation
  table (empty messages, >2048, temperature>2, top_p≤0, `max_tokens:0`, `n:0/129`, bad
  `tool_choice`, required-without-tools) ⇒ 400 `.error{type:"invalid_request_error",param,
  code}` with the **verbatim messages** of `chat_phases.rs:24–93`.
- `tests/openai_blocking.rs` — golden blocking JSON (injected clock/id): key set
  `id,object:"chat.completion",created,model,system_fingerprint,choices[{index,message{role,
  content,reasoning_content?,tool_calls?},finish_reason,logprobs:null}],usage{…details blocks
  audio:0, rejected_prediction_tokens:0}` (pinned by `encode_stream.rs:122–140` +
  `chat_response.rs`); tool-call stream reassembles to valid `function.arguments` JSON;
  stop-sequence fixture ⇒ truncated content + `finish_reason:"stop"` + `Choice.matched_stop`
  (feeds T12's stop_sequence echo); `store:true` ⇒ retrievable, `store:false` ⇒ 404.
- `tests/openai_sse.rs` — exact `data:` sequences: (a) text: role → content× → done
  (`"length"`) → `[DONE]`; (b) tool-call: role → start(`arguments:""`,`type:"function"`) →
  args fragments → usage-only(`choices:[]`) → final(no usage) → `[DONE]` (include_usage=true
  two-frame framing, `encode_stream.rs:88–100`); (c) reasoning: `reasoning_content` only,
  never `content` (:46–50); (d) `Error{message}` verbatim frame; (e) completions echo-first +
  `object:"text_completion"`; (f) responses: `responses_event_name` sequence
  `created → output_item.added → output_text.delta… → output_item.done → completed` with
  monotonic `sequence_number` (`responses.rs:193`).
- `tests/openai_stop_reasons.rs` — mapping-table test: every `FinishReason` variant's final
  chunk == `as_wire()` — identity `Stop→"stop", Length→"length", ToolCalls→"tool_calls",
  ContentFilter→"content_filter", Other("timeout")→"timeout", Other(x)→x` — plus contrast
  that no `end_turn|max_tokens|refusal|stop_sequence` string can appear on this surface.
- `tests/openai_errors.rs` — 400 parse body; 400 `context_length_exceeded` (`code`,
  `n_prompt_tokens`, `n_ctx`, `param ∈ {messages,prompt}`); completions prompt-too-long /
  out-of-range ids / bad temperature ⇒ 400; unknown `previous_response_id` ⇒ 400
  `code:"response_not_found"` `param:"previous_response_id"`; GET miss ⇒ 404; engine channel
  closed ⇒ 503.
- `tests/responses_lowering.rs` — `input` string / item-array → messages; `instructions` →
  head system message; prior transcript prepended from the store; hosted built-ins ⇒ per-name
  BadRequest (`responses_lowering.rs:73–77`).
- `tests/render_prompt.rs` — wire fixture → IR → **template-string golden** (master-plan T11
  item; `<tools>` key order survives preserve_order) + token count == `usage.prompt_tokens`.
- `tests/e2e_openai.rs` — `#[ignore]`, macOS + `SHISU_TEST_MODEL`: boot qwen35 via T10 `run()`
  on loopback, live blocking+stream × chat/responses/completions; schema-valid JSON/SSE;
  `Σ token_ids == usage.completion_tokens` when opted in. Skip-with-note when unset (T10 rule).

## Acceptance

Mirrors the master-plan Task 11 checkboxes (responses lowered to chat with
`previous_response_id` via response_store; wire→IR→template goldens; SSE framing goldens
incl. the `include_usage` terminal frame; live e2e):

- [ ] All 17 files of the tree exist, each < 1500 LoC (`wc -l`); `grep -rn SPDX` empty; no
  `ATLAS_`/`DS4_` reads (store knobs are `SHISU_STORE_*`, Rule 5).
- [ ] The three POSTs answer real handlers (501 stubs gone), `GET /v1/responses/{id}` live;
  engine touched only via `EngineHandle::generate` (`dyn Model` grep still scheduler-only).
- [ ] Carried `encode_stream.rs` test module green ⇒ atlas byte-compatible framing; `[DONE]` +
  usage-only terminal frame pinned by `tests/openai_sse.rs`.
- [ ] `collect_chat_response` is the single collector for all three surfaces (T12 cites it);
  `Choice.matched_stop` populated on stop-termination.
- [ ] Identity stop-reason table green; `grep -rn 'end_turn\|stop_sequence' src/openai` empty.
- [ ] Context overflow = 400 with `code:"context_length_exceeded"` + `n_prompt_tokens` +
  `n_ctx`; `grep -rn 413 src` empty (⚠-DEVIATION honored).
- [ ] `lower_responses_to_chat` + `previous_response_id` round-trip through `ResponseStore`
  green (multi-turn resume asserts the prepend).
- [ ] `cargo test -p shisu-server` green; `-- --ignored` green under `SHISU_TEST_MODEL`;
  `cargo check -p shisu-server` (no features) clean.
- [ ] Non-goals hold: no tool-call text parsing in `openai/` (`grep <tool_call>` only in
  tests), no tokenizer/scheduler code, no logprobs emission, no disk response persistence,
  no `/v1/conversations` route.

## Commit

```sh
git add shisu/crates/shisu-server
git commit -m "feat(server): OpenAI chat/completions/responses adapters over the IR

Vendored-trimmed from atlas spark-server: wire types, the delta_to_chunk_events
SSE seam (usage-only terminal frame + [DONE]), lower_responses_to_chat, memory
response_store, ids. Shared collect_chat_response (stop-strip + matched_stop)
and tokenizer::render_prompt are the cross-surface entries T12 reuses. Context
overflow follows ds4's 400 context_length_exceeded body."
```
