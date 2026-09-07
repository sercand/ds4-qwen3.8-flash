# Task 10: shisu-server skeleton

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 10
**Depends on:**
- T1 (`task-01-workspace-scaffold.md`): stub crate `shisu/crates/shisu-server/` is a **lib stub
  whose `main.rs` explicitly lands in Task 10** (T1 Step 8 note). Workspace deps usable as
  `{ workspace = true }`: `axum 0.8 (json)`, `hyper 1`, `hyper-util 0.1`, `tower 0.5 (util)`,
  `tower-http 0.7 (catch-panic,cors)`, `tokio 1 (full,parking_lot)`, `tokio-stream 0.1`, `futures 0.3`,
  `http-body 1`, `clap 4 (derive)`, `tracing`, `tracing-subscriber 0.3 (env-filter)`, `serde`,
  `serde_json (preserve_order)`, `thiserror`, `anyhow`, `minijinja 2`, `tokenizers 0.23`,
  path deps `shisu-core/-gguf/-metal/-engine/-kvstore/-ir`.
- T2: `shisu_core::{Model, Backend, ShapeProfile, SampleRequest, SessionId, CoreError, Result}`;
  `Model: Send` + object-safe ⇒ `Box<dyn Model>` movable into the engine thread (T2 Extras); `teardown` explicit.
- T6: `shisu_engine::qwen35::Qwen35Model::new(Arc<dyn Backend>)` + `.with_ctx(u32)` + `impl shisu_core::Model`.
- T7: `shisu_kvstore::{KvStore, KvReport, KvEntry, key_kind}` — `KvStore::open(&Path, budget_mb)`,
  `report() -> KvReport { dir, budget_bytes, total_bytes, entries }` (T7 Deviation 5: built for this route).
- T9: `shisu_ir::{ChatRequest, StreamDelta, DeltaStream, FinishReason, Usage, …}`;
  `DeltaStream = Pin<Box<dyn Stream<Item = StreamDelta> + Send>>` — T9's
  `delta_stream_is_send_and_pollable` test exists precisely for this crossing.
- T4 (optional `metal` feature only): `shisu_metal::{MetalBackend::new, metal_available}`.

**Produces** (crate root `shisu_server::*`; T11/T12/T13 build on these exact names):

```rust
// cli.rs — clap (master-plan SSOT flags)
pub struct Cli { #[command(subcommand)] pub command: Command }
pub enum Command { Serve(ServeArgs) }
pub struct ServeArgs { model: PathBuf, bind: String /*env SHISU_BIND*/, port: u16 /*SHISU_PORT*/,
                       ctx: u32 /*SHISU_CTX*/, kvstore_dir: Option<PathBuf> /*SHISU_KVSTORE_DIR*/,
                       kvstore_budget_mb: u64 /*SHISU_KVSTORE_BUDGET_MB*/,
                       auth_tokens: Vec<String> /*SHISU_AUTH_TOKENS, comma-separated*/ }
impl ServeArgs { pub fn validate(&self) -> anyhow::Result<()>; }   // ctx>0, port!=0, dir exists-check deferred
// state.rs
pub struct AppState { pub engine: EngineHandle, pub model_id: String, pub ctx: u32,
                      pub kv: Option<Arc<KvStore>>, pub auth: Option<AuthConfig>,
                      pub started_at: std::time::Instant }
// router.rs
pub fn build_router(state: Arc<AppState>) -> axum::Router;
pub async fn serve(state: Arc<AppState>, bind: &str, port: u16) -> anyhow::Result<()>; // accept loop + drain
// error.rs
pub enum ApiSurface { OpenAi, Anthropic }
pub enum AppError { NotImplemented(&'static str), NoModel, Unauthorized(AuthReject),
                    BadRequest(String), Internal(String) }   // impl IntoResponse per surface
// scheduler/mod.rs + scheduler/queue.rs
pub struct EngineHandle { /* tx + model_id/ctx snapshot; Clone */ }
pub enum EngineRequest { Generate(GenerateRequest) }
pub struct GenerateRequest { pub prompt_tokens: Arc<Vec<u32>>, pub sample: shisu_core::SampleRequest,
                             pub max_tokens: u32, pub reply: tokio::sync::oneshot::Sender<
                                 shisu_core::Result<shisu_ir::DeltaStream>> }
impl EngineHandle { pub async fn generate(&self, r: GenerateRequest)
                       -> shisu_core::Result<shisu_ir::DeltaStream>; }
pub fn spawn_engine(model: Box<dyn shisu_core::Model>, ctx: u32) -> (EngineHandle, std::thread::JoinHandle<()>);
// api/misc_handlers.rs: health, metrics, list_models, cache_report, not_implemented(surface, what)
// api/inference_types.rs: ModelInfo, ModelListResponse, CacheReport (serde, wire-shaped)
// main.rs: #[tokio::main] fn main() -> anyhow::Result<()> { shisu_server::run(Cli::parse()).await }
```

## Global Constraints

Rules 1–7 (master plan), one line each:
1. All new code under `shisu/crates/shisu-server/`; pure Rust, no host C/ObjC.
2. 1500-LoC cap — pre-split: `router.rs` and `scheduler/{mod.rs,queue.rs}` separate; `api/` is a directory.
3. No kernels; vendored Rust (auth, tokenizer helpers) copied verbatim minus SPDX line.
4. No perf gate here; T20 owns baselines.
5. Env knobs `SHISU_`-prefixed only. This task reads: `SHISU_BIND`, `SHISU_PORT`, `SHISU_CTX`,
   `SHISU_KVSTORE_DIR`, `SHISU_KVSTORE_BUDGET_MB`, `SHISU_AUTH_TOKENS`, `SHISU_MAX_BODY_BYTES`,
   `SHISU_LOG`, `SHISU_TEST_MODEL`. No `ATLAS_*`/`DS4_*`.
6. `warnings=deny` + clippy deny; `thiserror` for `AppError`, `anyhow` only at the binary
   boundary (`main.rs`/`run`); `parking_lot` for any lock; `tracing` for logs.
7. No license headers — strip atlas's SPDX line from every vendored file.

Task-specific:
- ⚠ **DEVIATION (file names):** master-plan Task 10 `Files:` is SSOT → `router.rs` + `cli.rs`
  (assignment said `routes.rs`/`config.rs`/`error.rs`). `config.rs` folds into `cli.rs` (clap IS
  the config surface; atlas converted its `ATLAS_*` knobs into flags, `cli/serve_args.rs:170-172`);
  env is per-flag *fallback*, flag wins. `error.rs` is additive (Step 3).
- ⚠ **DEVIATION (`SHISU_HOST`):** flag is `--bind` (atlas `--bind`, serve_args.rs:1007-1013; ds4
  `--host`, ds4_server.c:16034) → env mirror **`SHISU_BIND`**.
- ⚠ **DEVIATION (routes):** master plan adds **`/metrics`** (Step 7) to the assignment list;
  `/v1/models/{id}` (ds4_server.c:15686-15694, atlas serve_router.rs:66) deferred to T11.
- The engine thread is the ONLY thing that touches `Box<dyn Model>`; handlers never call a trait
  method (master plan: "atlas boundary pattern, not trait-call-from-handler").
- No batching, no session residency, no cancellation policy here — T13 owns the scheduler
  completion; this task ships a single-worker FIFO loop.
- Run cargo from `shisu/`; `cargo test -p shisu-server` / `cargo check -p shisu-server
  --features metal` only. T20 owns workspace gates.

## Source References (verified)

Atlas paths relative to `…/sercand/atlas`; ds4 to the repo root. Every row opened while writing.

| Source | Lines | What lives there / how used |
|---|---|---|
| `crates/spark-server/Cargo.toml` | 53-62, 79-85, 91, 99 | exact dep versions this crate declares: `clap 4 (derive)`, `axum 0.8 (json)`, `hyper 1 (server,http1,http2,client)`, `hyper-util 0.1 (server-auto,client-legacy,http1,tokio)`, `tower 0.5 (util)`, `tokio 1 (full,parking_lot)`, `tokio-stream 0.1`, `futures 0.3`, `tower-http 0.7 (catch-panic,cors)`, `minijinja 2 (builtins,adjacent_loop_items,json,preserve_order)` — all already in the shisu workspace (T1 Step 5) |
| `crates/spark-server/src/main_modules/serve_router.rs` | 20-169 | the router pattern: `CorsLayer::new().allow_origin(Any).allow_methods([GET,POST,OPTIONS])`, `CatchPanicLayer::new()` (body NOT overridden — no backtrace leak), `Router::new().route(...)` with axum-0.8 `{id}` path syntax, `DefaultBodyLimit::max(env ATLAS_MAX_BODY_BYTES or 32 MiB)`, `middleware::from_fn*` layer order (fault → rate-limit → auth → observability → cors → catch-panic), `.with_state(host)`, `0.0.0.0` exposure warning + loopback hint |
| `crates/spark-server/src/main_modules/serve_router.rs` | 180-197, 203-225, 227-317 | **bind first, then announce**: `TcpListener::bind` → `ready_line()` (`0.0.0.0`/`::` as `127.0.0.1`, IPv6 bracketed); `serve_with_header_timeout`: manual accept loop over `hyper_util::server::conn::auto::Builder` + `TokioIo/TokioExecutor/TokioTimer`, `.http1().timer(..).header_read_timeout(30s)`, `into_make_service_with_connect_info::<SocketAddr>()`, `serve_connection_with_upgrades`, `tokio::select!` accept-vs-shutdown → `drain_in_flight(15s)`; accept errors log+continue (never kill the server) |
| `crates/spark-server/src/tui/shutdown.rs` | 6-17, 132-145, 147-160 | graceful-shutdown mechanics to copy in simplified form: `tokio::signal::ctrl_c()` + `tokio::signal::unix::signal(SignalKind::terminate())` → `Notify`; `wait()` registers interest before re-checking the flag (race-free) |
| `crates/spark-server/src/main_modules/app_state.rs` | 33-52 | `AppState` shape: plain struct of `Arc`s + `request_tx: mpsc::Sender<InferenceRequest>` (tokio mpsc, NOT std) — the handler→engine boundary field |
| `crates/spark-server/src/api/inference_types.rs` | 74-185, 281 | `InferenceRequest` enum: `Blocking { …, response_tx: tokio::sync::oneshot::Sender<Result<InferenceResponse>> }` (:185), `Streaming { …, token_tx: tokio::sync::mpsc::Sender<StreamEvent> }` (:281) — the reply-channel-in-the-request pattern |
| `crates/spark-server/src/scheduler/mod.rs` | 342-368 | **the thread crossing**: `std::thread::spawn` + `while let Some(req) = rx.blocking_recv()` (tokio `Receiver::blocking_recv` is legal on a non-runtime thread) feeding a `Mutex<PendingQueue>` + `Condvar`; receiver closed → `closed = true` + notify |
| `crates/spark-server/src/scheduler/mod_helpers/send.rs` | 4-31, 46-96, 98-121 | scheduler-thread → client-channel backpressure SSOT: `try_send` → 1 ms poll → bounded deadline (`ATLAS_STREAM_SEND_DEADLINE_MS`, default 5000) → abandon + `tracing::warn`; `Closed` = "client hung up" → `tracing::info`; `capture_runtime_handle()` (`Handle::current()` in async context, `OnceLock`) for terminal sends |
| `crates/spark-server/src/main_modules/serve_load.rs` | 754, 926-932 | `mpsc::channel::<InferenceRequest>(args.max_num_seqs)`; scheduler thread handle **retained** (join before teardown, else teardown races the loop) |
| `crates/spark-server/src/api/misc_handlers.rs` | 143-190, 196-215, 57-140 | `readiness(model, fault) -> (StatusCode, Value)` as a pure fn (`200 {"status":"ready","model":…}` / `503 {"status":"loading"}`); `health_live`; `/metrics` = hand-written `text/plain; version=0.0.4; charset=utf-8` |
| `crates/spark-server/src/api/models.rs` | 19-59 | `/v1/models` → `Json(ModelListResponse { object: "list", data })`; empty list (not 503) when no model |
| `crates/spark-server/src/api/compact.rs` | 174-215 | OpenAI error body + `type_for_status` map (`400 invalid_request_error`, `401 authentication_error`, `404 not_found_error`, `429 rate_limit_exceeded`, else `server_error`) |
| `crates/spark-server/src/api/completions.rs` | 501-503 | `not_supported(msg) = openai_error_response(StatusCode::NOT_IMPLEMENTED, msg)` — the 501 stub shape atlas uses for unimplemented surfaces (`api/stubs.rs:42-79`) |
| `crates/spark-server/src/openai/encode_stream.rs` | 142-174 | **SSE pattern (verified)**: axum `Sse::new(stream).keep_alive(KeepAlive::default()).into_response()` over a `futures` stream of `sse::Event` — NOT a hand-rolled `Response<Body>`; `anthropic/handlers_stream.rs:80-84` uses `Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))` |
| `crates/spark-server/src/main_modules/middleware.rs` | 48-118 | `require_auth_middleware`: gate `path.starts_with("/v1/")` (+`/tokenize`,`/detokenize`), `/health`+`/metrics` stay open; `Authorization: Bearer ` strip; 401 body `{error:{message,type:"invalid_request_error",param:null,code:"missing_api_key"|"invalid_api_key"}}` |
| `crates/spark-server/src/auth.rs` | 92-118 | `AuthConfig::validate` (linear scan OR-ing `ct_eq` so the scan itself is constant-time) + `ct_eq` (length-equal, XOR-accumulate, branch-free `{0,1}`) — vendored verbatim |
| `crates/spark-server/src/lib.rs` | 1-33 | atlas is `[[bin]] name="spark"` + a partial lib for `--lib` tests; shisu uses a full lib + thin bin (T1 already made shisu-server a lib stub) |
| `crates/spark-server/src/tokenizer/{chat_render.rs,jinja_helpers.rs,message_preprocess.rs}` | 139 / 482 / 282 ln | the three master-plan-named tokenizer copies (minijinja + `preserve_order` + custom `tojson`) |
| `ds4_server.c` | 15670-15717 | **ds4 route table (complete)**: `OPTIONS`→204; `GET /v1/models`; `GET /cache`; `GET /v1/models/<alias>`; `POST /v1/messages`, `/v1/chat/completions`, `/v1/responses`, `/v1/completions`; anything else → `404 "unknown endpoint"` |
| `ds4_server.c` | 15408-15465, 15508-15536 | `/v1/models` JSON (`{"object":"list","data":[{id,object,created,owned_by,name,context_length,top_provider{…},supported_parameters[…]}]}`); `/cache` JSON (`{requests,hits,hit_rate,tokens_reused,tokens_prefilled,states{…},pages{…},disk{hits,writes},path{…}}`) |
| `ds4_server.c` | 15949-15971, 16026-16050 | ds4 config defaults: `host "127.0.0.1"`, `port 8000`, `ctx_size 32768`; flags `-m/--model`, `-c/--ctx`, `--host`, `--port`, `--kv-disk-dir`, `--kv-disk-space-mb` |
| ds4 (grep) | — | **ds4 has NO `/health`, `/metrics`, `/props`, `/tokenizer`, `/tokenize` route** (grep of `hr.path` comparisons + `"/health"`/`"health"`: zero route hits; only the CORS `Access-Control-Allow-Methods` line mentions OPTIONS). `/health` + `/metrics` are master-plan additions. ds4's `/cache` is NOT a kvstore dump: it is `server_cache_report` (engine states/pages + `disk{hits,writes}` counters), with the kvstore reached through the `kv_cache_*` wrappers at `ds4_server.c:10998-11420` |

## Plan

- [ ] **Step 1: `shisu/crates/shisu-server/Cargo.toml`** — replace the T1 stub manifest.
  `[lints] workspace = true`; `publish = false`; add:
  ```toml
  [features]
  # Empty default per T1 DEVIATION 2 (feature resolution is target-independent); macOS builds pass
  # `--features metal`. Feature-off builds compile + type-check but `serve` refuses to boot a
  # model with a typed error — same rule Task 16 applies to shisu-cuda.
  default = []
  metal = ["dep:shisu-metal"]

  [dependencies]
  shisu-core = { workspace = true }
  shisu-gguf = { workspace = true }
  shisu-engine = { workspace = true }
  shisu-kvstore = { workspace = true }
  shisu-ir = { workspace = true }
  shisu-metal = { workspace = true, optional = true }
  axum/hyper/hyper-util/tower/tower-http/tokio/tokio-stream/futures/http-body/
  serde/serde_json/thiserror/anyhow/clap/tracing/tracing-subscriber/minijinja/tokenizers
    = { workspace = true }

  [dev-dependencies]
  tempfile = "3"        # added to [workspace.dependencies] by T7 Step 1 — reuse, do not re-pin

  [[bin]]
  name = "shisu"
  path = "src/main.rs"
  ```
  No `prometheus` dep (atlas has one for `/metrics`; shisu hand-rolls the text, Step 7).
  No `reqwest`/`rustls` — nothing here talks outbound.

- [ ] **Step 2: `src/cli.rs`** — clap 4 derive `shisu serve`: `--model --config --bind --port
  --auth-tokens --kvstore-dir --ctx --kvstore-budget-mb` (the last feeds `KvStore::open`; ds4
  `--kv-disk-space-mb`, ds4_server.c:16050). Defaults `127.0.0.1` / `8000` (ds4's default,
  ds4_server.c:15964 — NOT atlas's 8888) / ctx `4096` (T6) / budget `4096` (T7). Env fallback
  via clap `env` attr (`SHISU_BIND`/`SHISU_PORT`/`SHISU_CTX`/`SHISU_KVSTORE_DIR`/
  `SHISU_KVSTORE_BUDGET_MB`/`SHISU_AUTH_TOKENS`); flag wins. Needs `"env"` added to the
  workspace `clap` entry — one-word additive edit to a T1-owned line, call it out in the commit.
  `--auth-tokens`: comma list (`value_delimiter = ','`). `validate()` rejects `ctx == 0`,
  `port == 0`, `kvstore-budget-mb == 0` (T7 maps 0→4096 silently; be explicit). `model_id` =
  file stem of `--model`.

- [ ] **Step 3: `src/error.rs`** (new, ~110 LoC) — `ApiSurface { OpenAi, Anthropic }` +
  `AppError` (thiserror) + `impl IntoResponse`; surface carried per variant
  (`NotImplemented(ApiSurface, &'static str)`) so one type renders both shapes:
  - OpenAI `{"error":{"message","type","param":null,"code"}}` via atlas `type_for_status`
    (`api/compact.rs:205-215`); 501 → `type "server_error"`, `code "not_implemented"`.
  - Anthropic `{"type":"error","error":{"type":"invalid_request_error","message"}}`
    (T12 owns the real translator; this is T10's only Anthropic-surface shape).
  - `404`/`405` re-rendered through `AppError` via `Router::fallback` (no bare-text bodies).
  ⚠ DEVIATION (additive): master plan has no `error.rs` (atlas keeps this in `api/compact.rs`);
  a dedicated module is justified — both surfaces need it from `router.rs` pre-T11/T12, cap rule.

- [ ] **Step 4: `src/scheduler/queue.rs`** — `EngineRequest`, `GenerateRequest`, `EngineHandle`
  (Produces block). `tokio::sync::mpsc::channel::<EngineRequest>(QUEUE_CAPACITY = 256)` (atlas
  sizes by `max_num_seqs`, serve_load.rs:754; no batch size yet → fixed const is honest).
  `generate` = `tx.send().await` then `rx.await`; `SendError` (engine dead) / `RecvError`
  (dropped mid-request) map to `CoreError::Backend { backend: "engine", .. }`.

- [ ] **Step 5: `src/scheduler/mod.rs`** — `spawn_engine` + the loop. The boundary, exactly:
  1. `run` builds backend + model and calls `Model::load` on the calling thread BEFORE the
     listener binds → fail fast (non-zero exit, never accepted a connection). No readiness
     oneshot needed: nothing runs before the bind; `Model: Send` (T2) makes the move legal.
  2. `let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);` then
     `std::thread::spawn(move || engine_loop(model, rx, ctx))`; keep the `JoinHandle`.
  3. `engine_loop`: `while let Some(req) = rx.blocking_recv() { handle(model, req); }` — tokio's
     documented async-channel-from-OS-thread await; panics on a runtime worker, hence its own
     `std::thread`, never `tokio::spawn` (atlas does exactly this, scheduler/mod.rs:349-357).
  4. `Generate`: `let (dtx, drx) = mpsc::channel::<StreamDelta>(64);` reply FIRST with
     `Box::pin(ReceiverStream::new(drx))` (streaming may start before a token exists), then
     `open_session` → chunked `prefill_chunk` → `decode_batch` + `sample` (T2/T6 semantics),
     per-token `blocking_send(Content)`, then `Finish { reason, usage }`, `close_session`.
     Bound the sends: copy atlas's `try_send` → 1 ms poll → deadline (5000 ms,
     `SHISU_STREAM_SEND_TIMEOUT_MS`) → abandon + warn pattern (mod_helpers/send.rs:46-96);
     `Closed` ⇒ client hung up ⇒ stop generating. Mid-generation errors go as
     `StreamDelta::Error` (T9 variant) before the stream ends.
  5. All `EngineHandle` clones dropped ⇒ `blocking_recv()` returns `None` ⇒ `model.teardown()`
     (explicit, never `Drop` — T2/atlas rationale) + `tracing::info!`; `run` joins the
     `JoinHandle` after `serve()` returns so teardown finishes before exit.
  6. T13 replaces step 4's inline loop with the coordinator; the `EngineRequest`/`DeltaStream`
     contract above is what it keeps.

- [ ] **Step 6: `src/state.rs`** — `AppState` per Produces (plain struct, `Arc`-shared, `engine`
  `Clone`). Deliberately NOT atlas's `ModelHost` hot-swap indirection: one model per process ⇒
  `.with_state` bound once is correct (doc-comment this so nobody "fixes" it later).

- [ ] **Step 7: `src/api/{mod.rs,inference_types.rs,misc_handlers.rs}`**
  - `inference_types.rs`: `ModelInfo { id, object:"model", created, owned_by:"shisu", name,
    context_length, top_provider{context_length,max_completion_tokens,is_moderated:false},
    supported_parameters }` + `ModelListResponse { object:"list", data }` — key names from ds4's
    `/v1/models` body (ds4_server.c:15408-15465); `supported_parameters` lists only what will
    exist after T11/T12 (`tools, tool_choice, max_tokens, temperature, top_p, top_k, min_p,
    stop, seed, stream, reasoning_effort`). T11 may move these into `openai/` — it owns that rename.
  - `misc_handlers.rs`: `health` → pure fn `readiness(model_loaded: bool) -> (StatusCode, Value)`
    (atlas misc_handlers.rs:152-166 minus the GPU-fault arm, no shisu equivalent yet):
    `200 {"status":"ready","model":id}` / `503 {"status":"loading"}`. `metrics` → hand-written
    `text/plain; version=0.0.4; charset=utf-8` with ONLY existing counters: `shisu_up`,
    `shisu_uptime_seconds`, `shisu_model_loaded`, `shisu_kvstore_enabled`, `shisu_kvstore_entries`,
    `shisu_kvstore_bytes` (from `KvReport`). `list_models` → the single `model_id`, never 503
    (atlas models.rs:22-29). `cache_report` → `CacheReport { disk: { enabled, dir, budget_bytes,
    total_bytes, entries: [KvEntry + key_kind] } }`; `kv == None` ⇒ `enabled:false, dir:null`,
    zeros, `entries:[]`. ⚠ DEVIATION: ds4's `/cache` also carries engine `states`/`pages`/`path`
    (ds4_server.c:15508-15536); those counters do not exist until T13, so T10 ships the disk
    section only, keeping ds4's `disk` grouping. `not_implemented(surface, what)` →
    `AppError::NotImplemented` → 501.

- [ ] **Step 8: `src/router.rs`** — `build_router(Arc<AppState>) -> Router`, adapted from atlas
  `serve_router.rs:44-160` (axum 0.8 `{id}` syntax):
  ```text
  GET  /health | /metrics | /v1/models | /cache   -> api::handlers (Step 7)
  POST /v1/chat/completions | /v1/completions | /v1/responses -> 501 not_implemented(OpenAi) -> T11
  POST /v1/messages                                 -> 501 not_implemented(Anthropic) -> T12
  ```
  Layers (atlas order): `DefaultBodyLimit::max(env SHISU_MAX_BODY_BYTES or 32 MiB)`,
  `from_fn_with_state(require_auth_middleware)`, `CorsLayer` (Any origin, GET/POST/OPTIONS, Any
  headers — matches ds4's permissive CORS, ds4_server.c:6622), `CatchPanicLayer::new()` (default
  body, no backtrace leak), `.with_state(state)`, `.fallback` → 404 through `AppError`.
  `serve(state, bind, port)`: bind → log the ready line (atlas `ready_line` rendering:
  `0.0.0.0`/`::` shown as `127.0.0.1`, IPv6 bracketed) → `hyper_util::server::conn::auto::Builder`
  accept loop with `header_read_timeout(30s)` + `into_make_service_with_connect_info::<SocketAddr>()`
  + `serve_connection_with_upgrades`, `tokio::select!` vs `shutdown_wait()` (Step 10) → on signal:
  stop accepting, `timeout(15s)` drain of the in-flight connection set, return `Ok(())`.
  Accept errors: `tracing::warn!` + 10 ms sleep + continue (never kill the server).

- [ ] **Step 9: `src/auth.rs`** — vendor atlas `auth.rs` `AuthConfig` (+`from_inline`, `validate`,
  `ct_eq`, lines 92-118) minus SPDX; `require_auth_middleware` adapted from
  `main_modules/middleware.rs:61-118`. Gate: `path.starts_with("/v1/") || path == "/cache"`
  (`/cache` exposes the store dir; ds4 has no auth at all — strict improvement). `/health` +
  `/metrics` stay open (scrape targets, atlas's rule). 401 = atlas's shape with
  `code: "missing_api_key" | "invalid_api_key"`.
- [ ] **Step 10: `src/shutdown.rs`** — simplified atlas `tui/shutdown.rs`: `static NOTIFY:
  OnceLock<Notify>` + `AtomicBool REQUESTED`; `install()` spawns the `ctrl_c` + SIGTERM listener
  task; `wait()` = register-then-recheck (race-free order, shutdown.rs:132-145). No TUI, no startup-escape hatch.

- [ ] **Step 11: `src/tokenizer/{mod.rs,chat_render.rs,jinja_helpers.rs,message_preprocess.rs}`**
  — the master-plan Files list, vendored from atlas `tokenizer/` (139/482/282 ln) minus SPDX,
  minus ds4/vision branches; minijinja `json`+`preserve_order` features + the custom `tojson`
  filter verbatim — the `<tools>` key order is a correctness property (T9 Source References).
  T10 wires nothing: the module compiles and its own unit tests pass; T11's golden fixtures
  exercise the render path.

- [ ] **Step 12: `src/lib.rs` + `src/main.rs`** — `lib.rs`: T1's deny attrs + `pub mod api, auth,
  cli, error, router, scheduler, shutdown, state, tokenizer` + `pub async fn run(Cli) ->
  anyhow::Result<()>`: install tracing (`EnvFilter::builder().with_env_var("SHISU_LOG")`,
  default `"info"` — rule 5, NOT `RUST_LOG`); match `Command::Serve`; `validate()`; open
  `KvStore` if `--kvstore-dir` (error ⇒ fail fast); backend: `#[cfg(feature = "metal")]
  MetalBackend::new()`, feature-off ⇒ `anyhow!("shisu built without a backend; rebuild
  --features metal")`; `ShapeProfile` via `HfConfig::from_file(cli.config()?)?.shape_profile()?` —
  `--config` flag, default = `config.json` beside `--model`; missing file = hard boot error (this box's
  unsloth dir holds ONLY the .gguf — pass `--config /Users/sercand/models/Qwen3.5-4B/mlx/config.json`);
  `Qwen35Model::new(backend.clone()).with_ctx(ctx)` + `load(...)`, `spawn_engine`, `build_router`,
  `serve`. `main.rs` = `#[tokio::main] async fn main() -> anyhow::Result<()>` calling `run` (rule 6).

## Tests

`cd shisu && cargo test -p shisu-server` (macOS, no GPU, no model). Router tests drive
`build_router` via `tower::ServiceExt::oneshot` — no socket/port; bodies via
`axum::body::to_bytes`. Fake engine = local `tests/fake_model.rs` (`FakeBackend` =
`parking_lot::Mutex` + `HashMap<u64, Vec<u8>>`, `FakeModel` = scripted tokens) per T2's
`tests/traits_contract.rs` pattern — not importable across crates (T6/T16 reuse the shape).

- `tests/router.rs`: `/health` → 200 + `{"status":"ready","model":…}`; `/v1/models` → 200,
  `object == "list"`, one entry `id == model_id` + `context_length == ctx`; each of the four
  generation routes → **501** with the right per-surface shape (OpenAI
  `.error.code == "not_implemented"`; `/v1/messages` top-level `type == "error"`); `/cache` with
  an empty `TempDir` store → 200 + full key set (`disk.enabled true`, `dir`, `budget_bytes`,
  `total_bytes == 0`, `entries == []`), `kv: None` ⇒ `enabled:false`; unknown path → 404 JSON
  (not axum's empty body); `GET /v1/chat/completions` → 405; `OPTIONS /v1/models` → 204 + allow-methods header.
- `tests/auth.rs`: `auth: Some(["sk-test"])` — no header → 401 `missing_api_key`; wrong token →
  401 `invalid_api_key`; `Bearer sk-test` → 200; `/health`+`/metrics` open without header;
  `/cache` → 401. Plus a direct `ct_eq` unit test (equal/unequal/length-mismatch/empty) — the
  timing-safe reject is the master-plan acceptance item, so the constant-time compare is tested
  where it is vendored.
- `tests/engine_thread.rs`: `spawn_engine(Box::new(FakeModel::new(tokens)), ctx)` →
  `handle.generate(..)` returns a `DeltaStream`; drive with `futures::StreamExt::next`, assert
  the exact variant sequence (Content×N then `Finish{reason: Stop}`) — the proof the
  `oneshot<DeltaStream>` + `ReceiverStream` crossing works from a non-async thread; dropping
  every `EngineHandle` clone ends the loop, the `JoinHandle` returns with
  `FakeModel.teardown_calls == 1`; a delta receiver dropped mid-stream → bounded-send abandons, not wedged.
- `tests/cli.rs`: `ServeArgs::try_parse_from(["serve","--model","m.gguf"])` → defaults
  `127.0.0.1:8000`, ctx 4096; `SHISU_PORT=9999` → 9999; flag + env both set → flag wins;
  `--port abc` → clap exit 2; `validate()` rejects `--ctx 0`. Env cases in this file alone (own process), per T7's rule.
- `tests/boot.rs` (`#[ignore]`, macOS + `SHISU_TEST_MODEL` + `metal_available()`): the
  master-plan "qwen35 model boots under scheduler thread" case — real `MetalBackend` +
  `Qwen35Model::load` + `spawn_engine` + `build_router`, one `oneshot` `GET /health` → 200 ready, drop + join. Skips without the env var.

## Acceptance

Mirrors master plan Task 10 (clap `shisu serve`; axum router per atlas `serve_router.rs` with
manual hyper accept loop + header_read_timeout + CORS + catch-panic + graceful drain; misc
routes; scheduler thread owning `Box<dyn Model>` behind an mpsc channel; oneshot route tests;
auth timing-safe reject; qwen35 boots under the scheduler thread):

- [ ] Files exist, each < 1500 LoC (`wc -l`): `src/{main.rs,cli.rs,state.rs,router.rs,error.rs,
  auth.rs,shutdown.rs,lib.rs}`, `api/{mod.rs,misc_handlers.rs,inference_types.rs}`,
  `scheduler/{mod.rs,queue.rs}`, `tokenizer/{mod.rs,chat_render.rs,jinja_helpers.rs,message_preprocess.rs}`.
- [ ] `cargo check -p shisu-server` AND `cargo check -p shisu-server --features metal` clean;
  `cargo test -p shisu-server` green; `-- --ignored` green with `SHISU_TEST_MODEL`.
- [ ] T10 ships no SSE: `grep -c 'text/event-stream\|Sse::new'` over src → 0 (stubs are 501);
  the `Sse::new(..).keep_alive(..)` pattern is recorded in Step 8's source table for T11/T12.
- [ ] `grep -rn 'ATLAS_\|DS4_' shisu/crates/shisu-server/src` empty; `grep -rn SPDX` empty;
  every env read `SHISU_`-prefixed (rule 5).
- [ ] `dyn Model` appears only inside `scheduler/` (grep); `teardown()` called on loop exit,
  never in a `Drop` impl.
- [ ] The four generation routes answer 501 with typed per-surface JSON; each stub's doc comment
  names its replacement task (T11/T12).
- [ ] `/health`, `/metrics`, `/v1/models`, `/cache` return the Step 7 schemas (router tests, not eyeballs).
- [ ] `shisu serve --help` lists `--model --bind --port --ctx --kvstore-dir
  --kvstore-budget-mb --auth-tokens`; a busy port fails AFTER nothing was announced
  (bind-before-announce, atlas `serve_router.rs:171-197`).
- [ ] Ctrl-C/SIGTERM: stop accepting, drain, join the engine thread, exit 0 with the
  `Shutdown complete` log line (manual check; no streaming route exists yet).

## Commit

```sh
git add shisu/crates/shisu-server shisu/Cargo.toml shisu/Cargo.lock
git commit -m "feat(server): axum skeleton, engine-thread boundary, misc routes, 501 stubs

Clap shisu serve; atlas-shaped state/router/auth/metrics/cache layer; single-worker engine
thread owns Box<dyn Model> behind a tokio mpsc. Adds the `env` feature to the workspace clap entry.
```
