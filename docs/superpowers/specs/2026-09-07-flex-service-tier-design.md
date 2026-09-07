# Flex service tier (OpenAI `service_tier=flex`) for ds4-server -- design

## Context

ds4-server schedules requests FIFO with prefix affinity (`dispatch_jobs_locked`,
ds4_server.c:15325) and time-slices the model between execution contexts round-robin
(`server_model_enter_prefill` / `server_model_enter_decode`, :12750-12830). There is no
request priority anywhere: `job`, `request` and `server_slot` carry no tier, and a
`service_tier` body field is silently swallowed by each parser's `json_skip_value`
catch-all. HTTP headers are discarded after `Content-Length` is read (:15489).

Goal: a **flex** tier, selectable per request with OpenAI's `"service_tier":"flex"` body
field or an `X-Service-Tier: flex` header. Flex work gets zero share of the model while
any normal request is dispatched: flex is admitted only when the server is otherwise
idle, and a flex request that is already generating **pauses mid-stream** the moment a
normal request is dispatched (slot and session stay resident, the SSE stream idles with
keepalive comments) and resumes when normal traffic drains. User decision 2026-09-07:
pause, not "let it finish", not evict.

The vLLM-parity work (docs/superpowers/vllm-parity-architecture.md) is mid-flight on the
same executor. This design does not change the grant predicates or the coordinator's
harvest loop. It adds a **pause point between steps** on the flex worker thread plus a
few worker-published per-slot facts and two derived predicates that the future single
tick loop (Stage 3/4) can evaluate directly as "row eligible this tick".

Scope: flex parking applies to `--exec-contexts` (qwen4exp, `multi_ctx_mode`). In
`--batched-session` mode and single-slot mode flex affects admission/queue order only.

First implementation step: write this design to
`docs/superpowers/specs/2026-09-07-flex-service-tier-design.md` (repo convention: plans
and specs live under docs/superpowers/) and commit it before code.

## Design

### Request classification
- `request.flex` set from body `"service_tier":"flex"` (chat, responses, completions;
  Anthropic's `service_tier` means something else, so header only there) or from the
  header `X-Service-Tier: flex` (case-insensitive). Other tier strings are accepted as
  normal; a non-string is a 400 via the parsers' existing `goto bad`.
- Echo `"service_tier":"flex"` in the chat-completion and responses response objects
  when set.

### Per-slot facts (all model_mu, all published by the slot's worker or by dispatch)
- `slot->tier` (NORMAL/FLEX): set in `dispatch_jobs_locked` inside the existing
  model_mu section (:15366); cleared in `slot_worker_main` under `s->mu` + model_mu in the
  same critical section that clears `busy` and re-dispatches (:15440-15443), and in the
  cancel-detach branch of `server_cancel_job` next to `awaiting_first_grant` (:15727).
  **No counter**: `server_normal_active_locked(s)` scans slots for
  `tier == NORMAL || promoted`. (A counter drifts on cancel-before-pickup, which never
  reaches `generate_job`; this is the same failure class the comment at :15716 records.)
- `slot->promoted`: a flex slot that a normal request is bound to (see Promotion).
- `slot->generating`: set/cleared by `server_generation_enter/leave`, which gain a slot
  parameter (call sites :14412, :14707).
- `slot->parked`: the worker is at the pause point.
- Derived: `server_slot_runnable_locked(s, slot)` =
  `!(tier == FLEX && !promoted && server_normal_active_locked(s) > 0)`;
  `server_eligible_generations_locked(s)` = count of `generating && !parked`.

### Admission (`dispatch_jobs_locked`)
A flex job is placeable only if all hold, evaluated once per pass (model_mu may be taken
under `s->mu`; verified order is `s->mu -> tool_mu -> model_mu -> inference_mu`):
1. no non-flex job anywhere in the queue (admission-only rule; it must never enter the
   pause predicate or a bound normal waiting on a paused flex deadlocks),
2. `server_normal_active_locked(s) == 0`,
3. slots whose `work->req.flex` is set `< s->flex_cap`, default `max(1, slot_count-1)`
   so one slot always stays free for normal traffic; `--flex-contexts N` overrides
   (clamped to `[1, slot_count]`).
Unplaceable flex jobs are skipped and the FIFO scan continues, so a normal job behind a
flex job is still dispatched. `dequeue()` (single-slot mode) prefers the first non-flex
job. `server_cancel_job` calls `dispatch_jobs_locked` after detaching a *queued* job
too (today only the assigned-slot branch does), otherwise a flex skipped because of a
later-cancelled normal stays queued until an unrelated event.

### Promotion (conversation-affinity binding)
`job_busy_owner_locked` / `job_required_slot_locked` can bind a normal follow-up turn to
the slot running its flex predecessor; `job_slot_score` then returns INT_MIN everywhere
and the normal waits for a flex that is parked whenever any other normal runs:
starvation. Unbinding is wrong (the frontier is not in the shared tree until the flex
generation commits, :15221-15227). Instead, when a non-flex job's bound slot holds a
flex job, set `slot->promoted = true` under model_mu and broadcast `model_cv`: the flex
slot becomes effectively normal, resumes if parked, is not parked again, and counts as
normal-active so other flex slots park. Cleared with the other per-slot flags.

### Pause point
`static bool server_flex_pause_point(server *s, server_slot *slot, job *j)`, called on
the flex worker thread:
- (a) top of the decode loop in `generate_job_inner`, before the grant-path selection
  at :14464; false ends the request with the existing "client disconnected" finish;
- (b) in `server_prefill_yield_cb` (:12899) after `server_model_leave`, before
  `server_model_enter_prefill`; false returns -1 (engine interrupt path). At that point
  the worker holds no lock and the engine has fully handed off (ds4.c:73277-73285).
Body, under model_mu: `while (!runnable && !job_cancelled(j) && !g_stop_requested &&
!s->model_stopping) { parked = true; awaiting_first_grant = false;
timedwait(model_cv, 5 s); on timeout: unlock, server_stream_keepalive(slot->progress,
": keepalive"), relock; }` then `parked = false`, broadcast. It mutates no counters.
Clearing `awaiting_first_grant` matters: a fully cached flex prompt reaches (a) before
its first grant, and `server_startup_pending_locked` would otherwise make a normal
prefill defer to it with nobody broadcasting.
A normal dispatched while a flex step is already granted or pending costs at most that
one step (~70 ms decode, or the rest of one prefill chunk).

### Executor readers
- Replace `active_generations` reads at :12701 (`server_prefill_before_decode_locked`),
  :12745 (`server_batch_decode_now`), :12854 (`server_prefill_quantum`), :13749
  (coordinator coalesce) with `server_eligible_generations_locked`. Otherwise a parked
  flex skews the prefill/decode ratio, pushes a lone normal into batched (non-MTP) mode,
  and makes the coordinator wait its coalesce window for a row that never comes.
- `server_prefill_chunk_rows` (:12892): a `parked` peer is not contended, so a normal
  prefill runs full width while flex is parked.
- `server_startup_pending_locked` (:12686): skip `parked` slots.
- Log pause/resume (`ds4-server: flex slot %d parked` / `resumed after %.1f s`) under
  the existing `DS4_SERVER_BATCH_LOG` gate.

### Keepalive plumbing
Only the worker thread writes to the fd (the client thread only polls/drains after
`enqueue`), and every SSE emitter sends a whole event from one `buf`, so a comment line
at a pause point never lands mid-event. Two fixes needed:
- `send_all` (:5475) can block up to `DS4_SERVER_SEND_STALL_TIMEOUT_MS` (2 s), so the
  write happens with model_mu released.
- Headers may not have been sent yet: `server_progress_cb` (:13167) only sends
  `sse_headers` on `prefill_chunk`/`prefill_display`, and the qwen4exp multi-context
  prefill reports `"prefill"` (ds4.c:73344), so today no headers/keepalives go out during
  a qwen4exp prefill at all. Factor :13180-13200 into
  `server_stream_keepalive(server_prefill_progress *p, const char *comment)` (sends
  headers first when `!headers_sent`, no-op when `!p->stream`, marks
  `stream_failed` + `job_mark_cancelled` on failure), use it from both the callback and
  the pause point, and make the callback treat `"prefill"` as keepalive-eligible. Add a
  worker-owned `slot->progress` pointer set where `ds4_session_set_progress` is
  installed (:14115, :13496) and cleared where it is removed (:13500, :13519, :14199,
  :14233, :14250, :14262). Non-streaming requests get nothing; disconnect detection stays
  with the client thread, whose `server_cancel_job` broadcasts `model_cv` (:15745).

## Files
- `ds4_server.c`: `request` (:779) + `request_init` (:1011); parsers :3892-pattern at
  :3929/:5135/:5390 region; `http_request`/`read_http_request` (:15450-15535; factor a
  `header_value()` helper shared with `content_length`); `client_main` (:15807-15828)
  merges the header flag; echo in `responses_sse_created` (:8111),
  `responses_sse_completed` (:8492), `final_responses_response` (:9114),
  `final_response` (:9186), chat SSE chunk header; `server_slot`/`server` structs
  (:10230/:10273); executor :12658-12897; generation bookkeeping :13552-13566; progress
  cb :13167; decode loop :14464; `generate_job` :15213; dispatch/queue :15325-15405;
  `slot_worker_main` :15440; `server_cancel_job` :15693; startup flags ~:16230/:16410.
- `tests/test_flex_tier.py` (new).
- `docs/superpowers/specs/2026-09-07-flex-service-tier-design.md` (new).

## Tests
Unit tests in `ds4_server.c`, registered in `ds4_server_unit_tests_run` (:22370), run
with `make ds4_server_test && make test-server` (not part of `make test`):
- `test_flex_parse_body_and_header`: body tier for chat/responses/completions; header
  for all four APIs; `auto`/`default` -> normal; non-string -> 400.
- `test_flex_response_echo`: `final_response` / `final_responses_response` contain
  `"service_tier":"flex"` only when set (mirror :17670).
- `test_flex_dispatch_gating` (fake server + slots, pattern of :16810): flex skipped
  while a normal is queued or a slot is normal-active; normal behind flex placed; cap
  honoured; `dequeue` prefers normal.
- `test_flex_cancel_before_pickup_clears_tier`: cancel-detach leaves
  `server_normal_active_locked == 0` (extend :20977 family).
- `test_flex_queued_normal_cancel_redispatches`.
- `test_flex_bound_normal_promotes_flex_slot`.
- `test_flex_parked_excluded`: eligible generations, `server_prefill_chunk_rows`
  (extend :16769), `server_startup_pending_locked`, `server_batch_decode_now`.
- `test_flex_runnable_predicate`: promoted flex is runnable; flex with normal active is
  not; queued-normal alone does not park (invariant).

Integration (GPU + qwen4exp model, `DS4_LOCK_FILE` set; shape of
`tests/run_concurrency_bench.sh`), new `tests/test_flex_tier.py` against
`ds4-server --exec-contexts 2`:
1. Streaming flex request generating; submit a normal: flex emits only `: keepalive`
   lines from the normal's dispatch to its completion; normal TTFT and tok/s equal a
   solo run within noise; flex resumes and (greedy) matches a solo flex run's text.
2. Normal arrives during a flex *prefill*: flex parks at the next chunk boundary,
   headers/keepalives are sent, normal prefill runs full width.
3. Two flex requests: second queues (cap 1); a normal then goes straight to the free
   slot.
4. Normal follow-up turn of a flex conversation while another normal runs: promotion
   lets it complete.
5. `tests/bench_concurrency.py` with no flex traffic: aggregate tok/s and TTFT unchanged.

## Verification
1. `make ds4_server_test && make test-server` green (existing scheduler/cancel tests plus
   the above).
2. `make ds4-server` (name targets explicitly; `make all` prints help on CUDA), run
   `tests/test_flex_tier.py`.
3. Manual: `curl -N -H 'X-Service-Tier: flex' .../v1/chat/completions` while a normal
   request runs; watch keepalive lines and the parked/resumed log lines.

## Trade-offs to be aware of
- Production runs `EXEC_CONTEXTS=2`, so `flex_cap = 1`: one flex at a time, and while a
  flex is parked only one normal can run. Raising `--exec-contexts` restores normal
  concurrency; evicting parked flex sequences is the follow-up if that is not enough.
- A parked flex pins its KV pages (`q4e_kv_reserve` only reclaims tree checkpoints,
  ds4.c:56575). A long-context flex can make a normal prefill fail with "KV page pool
  exhausted". Documented for now; gating flex admission on pool headroom is a follow-up.
- A flex prefill running solo finishes its current full-width chunk (seconds) before
  parking.
- Non-streaming flex requests cannot be kept alive; clients need long timeouts, as
  OpenAI recommends for flex. No pause deadline or 429: flex waits until it runs or the
  client disconnects.
- `--batched-session` and single-slot modes: flex is queue ordering only, no parking.

## Implementation status (2026-09-07)

Landed on `qwen3.8-flash-next`:
- request parsing, body + header (83d83b7, e548562);
- response echo on every OpenAI chat/completions/responses object, streaming
  chunks included (inside 4f4d9e2, then b88c23c);
- per-slot facts `tier`/`promoted`/`generating`/`parked` and the predicates
  `server_normal_active_locked`, `server_slot_runnable_locked`,
  `server_eligible_generations_locked` replacing the four `active_generations`
  readers (2d23997);
- admission gating with `--flex-contexts`, promotion on affinity binding,
  queued-cancel re-dispatch, `dequeue` preferring normal, the shared
  `server_stream_keepalive` (headers first; fires on the qwen4exp `"prefill"`
  event), and `server_flex_pause_point` at the decode-loop top and between
  prefill chunks (89d9cc9), plus the final-review fixes (08cde46): keepalive kept
  through decode, "prefill" event keepalive-only, cap at slot_count-1,
  promotion recomputed per pass.

Unit tests: `make ds4_server_test && ./ds4_server_test`. Integration:
`tests/test_flex_tier.py` against `ds4-server --exec-contexts 2`. Run
2026-09-07 15:12-15:25 on the GB10 (EXL3 4.05bpw, MTP draft 4, ctx 8192,
thinking off, greedy):
- A (normal arrives while a flex is generating): normal TTFT 0.25 s, normal
  wall 0.54 s, flex tokens leaked during the normal: 0; server log shows
  `flex slot 0 parked` / `resumed after 0.5 s`.
- B (two flex, cap 1, then a normal): second flex not admitted; normal TTFT
  0.13 s with one flex running and one queued; the queued flex ran afterwards.
- C (normal arrives during a 5909-token flex prefill): normal TTFT 0.52 s, flex
  parked between chunks (`resumed after 0.4 s`) and completed (prefill 4.56 s
  total). Parks were shorter than the 5 s keepalive period, so no keepalive
  comment was observed end to end; the unit test covers that path.
- The solo-vs-parked text equality held on one run and differed at a near-tie
  on another (known cross-run nondeterminism of the MoE, not the scheduler);
  the script now asserts only the pre-park prefix and reports the rest.
- Not covered end to end: promotion (spec test 4) and the no-flex throughput
  baseline (spec test 5, `tests/bench_concurrency.py`).

Follow-ups not done: gating flex admission on KV pool headroom; evicting a
parked flex under slot pressure; a pause deadline / 429.

Fold-in for the vLLM-parity tick loop (Stage 3/4): the live set for a tick is
`{slot : generating && server_slot_runnable_locked(s, slot)}` and the prefill
share uses the same predicate; `server_flex_pause_point` then disappears and
`parked` becomes what the tick computes rather than what the worker publishes.
