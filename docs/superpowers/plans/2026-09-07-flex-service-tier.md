# Flex Service Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Requests marked `service_tier=flex` (body field or `X-Service-Tier: flex` header) only use the model when no normal request is dispatched, and a generating flex request pauses mid-stream while normal traffic runs.

**Architecture:** All changes live in `ds4_server.c` (one big C file, tests included under `DS4_SERVER_TEST`). The request parsers gain a `flex` flag; the dispatcher refuses flex jobs while normal work exists; the flex slot's worker thread calls a pause point between decode steps and between prefill chunks that waits on `model_cv` while any slot holds a normal job. The executor's grant functions are untouched; four readers of `active_generations` switch to a derived count that excludes parked flex slots. Only worker-published per-slot facts are stored (no counters), so cancel paths cannot drift.

**Tech Stack:** C11, pthreads, hand-written JSON parser (`json_string`, `json_bool`, …), `make ds4_server_test && ./ds4_server_test` (no GPU), Python 3 stdlib for the integration test.

**Spec:** `docs/superpowers/specs/2026-09-07-flex-service-tier-design.md`

## Global Constraints

- Lock order is `s->mu -> tool_mu -> model_mu -> inference_mu`. Never take `s->mu` or `tool_mu` while holding `model_mu`.
- Never write to a client fd while holding `model_mu` (`send_all` can block 2 s).
- Only the slot's worker thread writes to the client fd.
- "A normal is queued" is an admission rule only. It must never make a flex slot park (deadlock with the affinity binding).
- Flex parking applies only when `s->multi_ctx_mode`. In `--batched-session` or single-slot mode flex is queue ordering only.
- Build and test: `make ds4_server_test && ./ds4_server_test` (prints `ok` lines; any `assertion failed` is a failure). `make all` prints help on this CUDA box: always name targets.
- The working tree already has uncommitted vLLM-parity changes in `ds4_server.c`, `ds4.c`, `ds4.h`, `ds4_gpu.h`, `ds4_qwen4exp_gpu.cuh`. Commit only the hunks you add: use `git add -p ds4_server.c` and stage flex hunks only, or `git commit ds4_server.c` is NOT acceptable. When in doubt run `git diff --cached --stat` before committing.
- Line numbers below are from the tree at plan time; re-locate with the quoted anchor text (`grep -n`).

---

### Task 1: Request tier from body and header

**Files:**
- Modify: `ds4_server.c` `request` struct (~:779-839), `parse_chat_request` (~:3929 `"stream"` branch), `parse_responses_request` (~:5135), `parse_completion_request` (~:5390), `http_request` + `read_http_request` (~:15450-15535), `client_main` (~:15828 `req.raw_body = xstrndup`)
- Test: `ds4_server.c` new `test_flex_request_tier_parsing`, registered in `ds4_server_unit_tests_run` (~:22370)

**Interfaces:**
- Produces: `bool request.flex`; `bool http_request.flex_header`; `static bool parse_service_tier_value(const char **p, request *r)`; `static bool header_value(const char *h, size_t n, const char *name, char *out, size_t outlen)`.

- [ ] **Step 1: Write the failing test**

Add before `static void ds4_server_unit_tests_run(void)`:

```c
static void test_flex_request_tier_parsing(void) {
    request r;
    char err[128] = {0};

    /* Body: OpenAI-shaped parsers accept "service_tier". */
    TEST_ASSERT(parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"service_tier\":\"flex\"}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.flex);
    request_free(&r);

    TEST_ASSERT(parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"service_tier\":\"default\"}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!r.flex);
    request_free(&r);

    TEST_ASSERT(parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!r.flex);
    request_free(&r);

    /* A non-string tier is a 400, like every other mistyped field. */
    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"service_tier\":1}",
        128, 32768, &r, err, sizeof(err)));

    TEST_ASSERT(parse_completion_request(NULL,
        "{\"prompt\":\"hi\",\"service_tier\":\"flex\"}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.flex);
    request_free(&r);

    TEST_ASSERT(parse_responses_request(NULL, NULL,
        "{\"input\":\"hi\",\"service_tier\":\"flex\"}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.flex);
    request_free(&r);

    /* The value parser on its own. */
    request_init(&r, REQ_CHAT, 128);
    const char *p = "\"flex\"";
    TEST_ASSERT(parse_service_tier_value(&p, &r) && r.flex && *p == '\0');
    p = "\"FLEX\"";
    r.flex = false;
    TEST_ASSERT(parse_service_tier_value(&p, &r) && r.flex);
    p = "null";
    TEST_ASSERT(!parse_service_tier_value(&p, &r));
    request_free(&r);

    /* Header scan. */
    const char *hdr = "POST /v1/chat/completions HTTP/1.1\r\n"
                      "Content-Length: 5\r\n"
                      "x-service-tier:  Flex \r\n\r\n";
    char v[32];
    TEST_ASSERT(header_value(hdr, strlen(hdr), "X-Service-Tier", v, sizeof(v)));
    TEST_ASSERT(!strcmp(v, "Flex"));
    TEST_ASSERT(header_value(hdr, strlen(hdr), "Content-Length", v, sizeof(v)));
    TEST_ASSERT(!strcmp(v, "5"));
    TEST_ASSERT(!header_value(hdr, strlen(hdr), "Authorization", v, sizeof(v)));
    TEST_ASSERT(content_length(hdr, strlen(hdr)) == 5);
}
```

Register it: add `test_flex_request_tier_parsing();` as the first line inside `ds4_server_unit_tests_run`.

Check first whether `parse_responses_request` with `NULL` engine/server and a plain `"input"` string works in other tests: `grep -n 'parse_responses_request(NULL' ds4_server.c`. If no existing test calls it with NULL, drop the responses block from the test (the branch is covered by code review) rather than fighting the fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_server_test 2>&1 | tail -5`
Expected: compile error, `request` has no member named `flex` / `parse_service_tier_value` undeclared.

- [ ] **Step 3: Add the request field and the value parser**

In the `request` struct, after `bool stream_include_usage;` add:

```c
    /* OpenAI service_tier=flex (or header X-Service-Tier: flex): background
     * work that only runs while no normal request is dispatched, and parks
     * mid-stream when one is.  See docs/superpowers/specs/2026-09-07-flex-service-tier-design.md. */
    bool flex;
```

`request_init` does `memset(r, 0, ...)`, so no init line is needed. Right after `parse_ignore_eos_value` (~:1025) add:

```c
/* "service_tier": "flex" selects the background tier; every other string
 * ("auto", "default", "priority") is the normal tier.  A non-string is a
 * malformed request like any other mistyped field. */
static bool parse_service_tier_value(const char **p, request *r) {
    char *v = NULL;
    if (!p || !r || !json_string(p, &v)) return false;
    r->flex = v && strcasecmp(v, "flex") == 0;
    free(v);
    return true;
}
```

- [ ] **Step 4: Parse the body field in the three OpenAI-shaped parsers**

In `parse_chat_request`, `parse_responses_request` and `parse_completion_request`, directly after each `} else if (!strcmp(key, "stream")) { ... }` block, add:

```c
        } else if (!strcmp(key, "service_tier")) {
            if (!parse_service_tier_value(&p, r)) {
                free(key);
                goto bad;
            }
```

Do NOT add it to `parse_anthropic_request` (Anthropic's `service_tier` has different values and meaning).

- [ ] **Step 5: Header scan**

Replace `content_length` (~:15472) with a generic helper plus a thin wrapper:

```c
/* Value of header `name` (case-insensitive), trimmed, or false when absent.
 * `h`/`n` span the request head up to and including the blank line. */
static bool header_value(const char *h, size_t n, const char *name,
                         char *out, size_t outlen) {
    const size_t name_len = strlen(name);
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len > name_len && line[name_len] == ':' &&
            strncasecmp(line, name, name_len) == 0) {
            const char *v = line + name_len + 1;
            const char *vend = line + len;
            while (v < vend && isspace((unsigned char)*v)) v++;
            while (vend > v && isspace((unsigned char)vend[-1])) vend--;
            size_t vlen = (size_t)(vend - v);
            if (vlen >= outlen) vlen = outlen - 1;
            memcpy(out, v, vlen);
            out[vlen] = '\0';
            return true;
        }
        if (p < end) p++;
    }
    return false;
}

static long content_length(const char *h, size_t n) {
    char v[32];
    if (!header_value(h, n, "Content-Length", v, sizeof(v))) return 0;
    return strtol(v, NULL, 10);
}
```

Add `bool flex_header;` to `http_request` (after `size_t body_len;`). In `read_http_request`, right after `long clen = content_length(b.ptr, (size_t)hend);` add:

```c
    char tier[32];
    r->flex_header = header_value(b.ptr, (size_t)hend, "X-Service-Tier",
                                  tier, sizeof(tier)) &&
                     strcasecmp(tier, "flex") == 0;
```

(`http_request_free` already zeroes the struct.)

- [ ] **Step 6: Merge the header into the request**

In `client_main`, change

```c
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
```

to

```c
    if (ok) {
        req.raw_body = xstrndup(hr.body, hr.body_len);
        if (hr.flex_header) req.flex = true;
    }
```

- [ ] **Step 7: Run tests**

Run: `make ds4_server_test 2>&1 | grep -E "error|warning" ; ./ds4_server_test 2>&1 | tail -3`
Expected: no compiler errors; the binary exits 0 with no `assertion failed` lines.

- [ ] **Step 8: Commit**

```bash
git add -p ds4_server.c   # stage only the flex hunks (request.flex, parse_service_tier_value, service_tier branches, header_value, flex_header, client_main merge, the new test + its registration)
git diff --cached --stat
git commit -m "server: parse service_tier=flex from body and X-Service-Tier header

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Echo `service_tier` in OpenAI-shaped responses

**Files:**
- Modify: `ds4_server.c` `sse_chunk` (~:6742), `responses_sse_created` (~:8111), `responses_sse_completed` (~:8508), `responses_final_response` (~:9111, body header at ~:9128), `final_response` (~:9185)
- Test: `ds4_server.c` new `test_flex_response_echo`

**Interfaces:**
- Consumes: `request.flex` (Task 1).
- Produces: `static void append_service_tier_json(buf *b, const request *r)`.

- [ ] **Step 1: Write the failing test**

```c
static void test_flex_response_echo(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    int sv[2];

    r.flex = true;
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    TEST_ASSERT(final_response(sv[0], false, &r, "cmpl-flex", "OK", NULL, NULL, "stop", 3, 1));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"service_tier\":\"flex\"") != NULL);
    free(out); close(sv[0]); close(sv[1]);

    r.flex = false;
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    TEST_ASSERT(final_response(sv[0], false, &r, "cmpl-norm", "OK", NULL, NULL, "stop", 3, 1));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "service_tier") == NULL);
    free(out); close(sv[0]); close(sv[1]);

    r.flex = true;
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    TEST_ASSERT(sse_chunk(sv[0], &r, "cmpl-flex", "tok", NULL));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"service_tier\":\"flex\"") != NULL);
    free(out); close(sv[0]); close(sv[1]);

    r.api = API_RESPONSES;
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    TEST_ASSERT(responses_final_response(sv[0], false, &r, "resp_flex", "OK", NULL, NULL,
                                         "stop", 3, 1));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"service_tier\":\"flex\"") != NULL);
    free(out); close(sv[0]); close(sv[1]);

    request_free(&r);
}
```

Register after `test_flex_request_tier_parsing();`.

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_server_test && ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: compiles; at least 3 assertion failures (the `service_tier` substrings are missing).

- [ ] **Step 3: Implement**

Add near `append_openai_usage_json` (~:6779):

```c
/* OpenAI echoes the tier the request ran under.  Emitted only when the
 * request asked for flex, so normal responses are byte-identical to before. */
static void append_service_tier_json(buf *b, const request *r) {
    if (r && r->flex) buf_puts(b, ",\"service_tier\":\"flex\"");
}
```

Insert `append_service_tier_json(&b, r);` immediately after each `json_escape(&b, r->model);` in these five functions: `sse_chunk` (both the chat and text_completion branches), `final_response` (both branches), `responses_sse_created`, `responses_sse_completed`, `responses_final_response`. Each of those `json_escape(&b, r->model)` calls is followed by a `,"..."` continuation, so a leading comma in the helper keeps the JSON valid.

- [ ] **Step 4: Run tests**

Run: `make ds4_server_test && ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: `0`.

- [ ] **Step 5: Commit**

```bash
git add -p ds4_server.c
git commit -m "server: echo service_tier=flex in OpenAI responses

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Per-slot tier facts and the derived executor predicates

**Files:**
- Modify: `ds4_server.c` `struct server_slot` (~:10230), `struct server` (~:10273), executor helpers (~:12658-12897), `server_generation_enter/leave` (~:13552-13566) and their call sites (~:14412, ~:14707), `decode_worker_main` coalesce condition (~:13749)
- Test: `ds4_server.c` new `test_flex_predicates`, extend `test_multi_ctx_adaptive_chunk` (~:16769)

**Interfaces:**
- Produces:
  ```c
  typedef enum { SLOT_TIER_NONE = 0, SLOT_TIER_NORMAL, SLOT_TIER_FLEX } slot_tier;
  /* server_slot fields, all model_mu: */ slot_tier tier; bool promoted; bool generating; bool parked;
  /* server field: */ int flex_cap;
  static int  server_normal_active_locked(const server *s);
  static bool server_slot_runnable_locked(const server *s, const server_slot *slot);
  static int  server_eligible_generations_locked(const server *s);
  static void server_generation_enter(server *s, server_slot *slot);
  static void server_generation_leave(server *s, server_slot *slot);
  ```

- [ ] **Step 1: Write the failing tests**

```c
static void test_flex_predicates(void) {
    server s = {0};
    server_slot slots[3] = {0};
    s.slots = slots;
    s.slot_count = 3;
    s.multi_ctx_mode = true;
    s.mixed_prefill_quantum = 512;

    /* Nothing dispatched: no normal work, every slot runnable. */
    TEST_ASSERT(server_normal_active_locked(&s) == 0);
    slots[0].tier = SLOT_TIER_FLEX;
    TEST_ASSERT(server_slot_runnable_locked(&s, &slots[0]));

    /* A normal slot parks every flex slot that is not promoted. */
    slots[1].tier = SLOT_TIER_NORMAL;
    TEST_ASSERT(server_normal_active_locked(&s) == 1);
    TEST_ASSERT(!server_slot_runnable_locked(&s, &slots[0]));
    TEST_ASSERT(server_slot_runnable_locked(&s, &slots[1]));
    slots[0].promoted = true;
    TEST_ASSERT(server_slot_runnable_locked(&s, &slots[0]));
    TEST_ASSERT(server_normal_active_locked(&s) == 2);   /* promoted counts as normal */
    slots[0].promoted = false;

    /* Eligible generations exclude a parked slot. */
    slots[0].generating = true;
    slots[1].generating = true;
    TEST_ASSERT(server_eligible_generations_locked(&s) == 2);
    slots[0].parked = true;
    TEST_ASSERT(server_eligible_generations_locked(&s) == 1);

    /* The prefill-before-decode credit rule counts eligible generations only:
     * one normal step since the quantum is enough even though two generate. */
    s.decode_waiting = 1;
    s.decodes_since_prefill = 1;
    TEST_ASSERT(server_prefill_before_decode_locked(&s));
    s.decodes_since_prefill = 0;
    TEST_ASSERT(!server_prefill_before_decode_locked(&s));

    /* A parked peer neither narrows a prefill nor makes it defer. */
    job flex_in_flight = {0};
    slots[0].running = &flex_in_flight;   /* the flex has a request in flight */
    TEST_ASSERT(server_prefill_chunk_rows(&s, &slots[1]) == 0);
    slots[0].parked = false;
    TEST_ASSERT(server_prefill_chunk_rows(&s, &slots[1]) == 512);
    slots[0].running = NULL;
    slots[0].parked = true;
    slots[0].awaiting_first_grant = true;
    TEST_ASSERT(!server_startup_pending_locked(&s, 1));
    slots[0].parked = false;
    TEST_ASSERT(server_startup_pending_locked(&s, 1));
    slots[0].awaiting_first_grant = false;

    /* Enter/leave publish the per-slot fact. */
    s.multi_ctx_mode = true;
    pthread_mutex_init(&s.model_mu, NULL);
    pthread_cond_init(&s.model_cv, NULL);
    server_generation_enter(&s, &slots[2]);
    TEST_ASSERT(slots[2].generating && s.active_generations == 1);
    server_generation_leave(&s, &slots[2]);
    TEST_ASSERT(!slots[2].generating && s.active_generations == 0);
    pthread_cond_destroy(&s.model_cv);
    pthread_mutex_destroy(&s.model_mu);
}
```

Register after `test_flex_response_echo();`. Note `server_prefill_chunk_rows` locks `s->model_mu`, so initialise the mutex before those asserts: move the `pthread_mutex_init`/`cond_init` lines to the top of the test and the destroys to the end.

- [ ] **Step 2: Run to verify it fails**

Run: `make ds4_server_test 2>&1 | grep -m3 error`
Expected: `SLOT_TIER_FLEX` undeclared / no member `tier`.

- [ ] **Step 3: Add the fields**

Above `struct server_slot` add:

```c
/* The tier of the job a slot holds, from dispatch until the worker (or a
 * cancel before pickup) releases the slot.  NONE while idle.  Guarded by
 * model_mu, like the other executor facts on the slot. */
typedef enum { SLOT_TIER_NONE = 0, SLOT_TIER_NORMAL, SLOT_TIER_FLEX } slot_tier;
```

Inside `struct server_slot`, after `bool awaiting_first_grant;` add:

```c
    /* Flex tier (spec docs/superpowers/specs/2026-09-07-flex-service-tier-design.md).
     * All four are facts published by dispatch or by this slot's own worker;
     * nothing here is a counter, so a cancel path cannot leave drift behind. */
    slot_tier tier;
    bool promoted;    /* a normal request is bound to this flex slot: run as normal */
    bool generating;  /* inside generate_job_inner's decode loop */
    bool parked;      /* the worker is waiting at server_flex_pause_point */
```

In `struct server`, after `int mixed_prefill_quantum;` add:

```c
    /* Most slots flex jobs may hold at once (default slot_count-1, so one
     * slot always stays free for normal traffic; --flex-contexts). */
    int flex_cap;
```

- [ ] **Step 4: Add the predicates and switch the readers**

Right after `server_time_sliced` (~:12656) add:

```c
/* model_mu held: slots holding a normal request, or a flex one a normal
 * request is bound to (promoted).  A scan, not a counter -- see the
 * awaiting_first_grant comment in server_cancel_job for why a counter here
 * would drift on a cancel before pickup. */
static int server_normal_active_locked(const server *s) {
    int n = 0;
    for (int i = 0; s && i < s->slot_count; i++) {
        const server_slot *slot = &s->slots[i];
        if (slot->tier == SLOT_TIER_NORMAL || (slot->tier == SLOT_TIER_FLEX && slot->promoted)) n++;
    }
    return n;
}

/* model_mu held: may this slot take a step now?  Only an unpromoted flex
 * slot with normal work dispatched is held back.  The future single tick
 * loop evaluates exactly this per row. */
static bool server_slot_runnable_locked(const server *s, const server_slot *slot) {
    if (!s || !slot) return false;
    if (slot->tier != SLOT_TIER_FLEX || slot->promoted) return true;
    return server_normal_active_locked(s) == 0;
}

/* model_mu held: generating contexts that will actually ask for steps.  A
 * parked flex is resident but asks for nothing, and counting it would make
 * the credit rule wait for a step that never comes, push a lone normal into
 * the batched (non-MTP) path, and hold the coordinator's coalesce window. */
static int server_eligible_generations_locked(const server *s) {
    int n = 0;
    for (int i = 0; s && i < s->slot_count; i++) {
        if (s->slots[i].generating && !s->slots[i].parked) n++;
    }
    return n;
}
```

Then:
- `server_startup_pending_locked`: change the condition to `if (!slot->parked && slot->awaiting_first_grant && !slot->prefill_waiting && !slot->decode_waiting)`.
- `server_prefill_before_decode_locked`: `return s->decode_waiting == 0 || s->decodes_since_prefill >= server_eligible_generations_locked(s);`
- `server_batch_decode_now`: `const bool batch = server_eligible_generations_locked(s) >= server_batch_decode_min();`
- `server_prefill_quantum`: `bool generation_active = server_eligible_generations_locked(s) > 0;`
- `server_prefill_chunk_rows`: `if (other == slot || other->parked) continue;`
- `decode_worker_main` coalesce loop: replace `s->decode_pending < s->active_generations` with `s->decode_pending < server_eligible_generations_locked(s)`.
- `server_generation_enter/leave`: add a `server_slot *slot` parameter; set `slot->generating = true` / `false` inside the model_mu section (keep the `active_generations` counter as-is for the logs and existing tests). Update the two call sites: `server_generation_enter(s, slot);` at ~:14412 and `server_generation_leave(s, slot);` at ~:14707. Grep `server_generation_enter(` and `server_generation_leave(` to be sure there are no others (the tests at ~:16736 set `s.active_generations` directly; leave them).

`test_multi_ctx_executor_alternation` (~:16736) sets `s.active_generations` directly and expects the credit rule to use it. Update that test to set `slots[i].generating` on a fake 2-slot array instead (declare `server_slot slots[2] = {0}; s.slots = slots; s.slot_count = 2;` and flip `slots[0].generating`/`slots[1].generating` where it sets `active_generations = 1/2`). Keep the assertions.

- [ ] **Step 5: Run tests**

Run: `make ds4_server_test 2>&1 | grep -E "error|warning"; ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: no diagnostics, `0`.

- [ ] **Step 6: Commit**

```bash
git add -p ds4_server.c
git commit -m "server: per-slot tier facts and eligible-generation predicates for flex

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Admission, tier bookkeeping, promotion and `--flex-contexts`

**Files:**
- Modify: `ds4_server.c` `dispatch_jobs_locked` (~:15325), `dequeue` (~:15392), `slot_worker_main` (~:15440-15443), `server_cancel_job` (~:15697-15731), `server_config` (~:15935) + flag parsing (~:16156 `--mixed-prefill-quantum` neighbour), startup (`s.mixed_prefill_quantum = cfg.mixed_prefill_quantum;` ~:16440)
- Test: `ds4_server.c` new `test_flex_dispatch_gating`, `test_flex_cancel_paths`, `test_flex_dequeue_prefers_normal`, `test_flex_bound_normal_promotes_flex_slot`

**Interfaces:**
- Consumes: `slot_tier`, `slot->tier`, `slot->promoted`, `server_normal_active_locked`, `s->flex_cap` (Task 3); `request.flex` (Task 1).
- Produces: `static bool server_flex_placeable_locked(server *s)` (s->mu held; takes model_mu inside); `static void server_slot_release_tier(server *s, server_slot *slot)` (s->mu held; takes model_mu inside).

- [ ] **Step 1: Write the failing tests**

```c
/* Flex admission: only when no normal is queued, no slot holds a normal, and
 * fewer than flex_cap slots hold flex.  A normal behind a skipped flex still
 * goes, FIFO. */
static void test_flex_dispatch_gating(void) {
    server s;
    server_slot slots[2];
    test_cancel_server_init(&s);
    memset(slots, 0, sizeof(slots));
    s.slots = slots; s.slot_count = 2; s.multi_ctx_mode = true; s.flex_cap = 1;
    for (int i = 0; i < 2; i++) { slots[i].srv = &s; slots[i].id = i; }

    job flex1, flex2, normal1;
    test_cancel_job_init(&flex1); flex1.req.flex = true;
    test_cancel_job_init(&flex2); flex2.req.flex = true;
    test_cancel_job_init(&normal1);

    /* Idle server: one flex goes, the second waits on the cap. */
    s.head = &flex1; flex1.next = &flex2; s.tail = &flex2;
    dispatch_jobs_locked(&s);
    TEST_ASSERT(slots[0].assigned == &flex1 && slots[0].tier == SLOT_TIER_FLEX);
    TEST_ASSERT(s.head == &flex2 && slots[1].assigned == NULL);

    /* A normal arrives behind the queued flex: it is dispatched past it. */
    flex2.next = &normal1; s.tail = &normal1;
    dispatch_jobs_locked(&s);
    TEST_ASSERT(slots[1].assigned == &normal1 && slots[1].tier == SLOT_TIER_NORMAL);
    TEST_ASSERT(s.head == &flex2 && s.tail == &flex2 && flex2.next == NULL);

    /* Normal finishes; flex1 also finishes; flex2 must not go while a normal is queued. */
    job normal2; test_cancel_job_init(&normal2);
    slots[1].assigned = NULL; slots[1].busy = false; slots[1].work = NULL; slots[1].tier = SLOT_TIER_NONE;
    slots[0].assigned = NULL; slots[0].busy = false; slots[0].work = NULL; slots[0].tier = SLOT_TIER_NONE;
    flex2.next = &normal2; s.tail = &normal2;
    dispatch_jobs_locked(&s);
    TEST_ASSERT(slots[0].assigned == &normal2 || slots[1].assigned == &normal2);
    /* ...and not while that normal holds a slot... */
    TEST_ASSERT(s.head == &flex2);
    TEST_ASSERT(slots[0].assigned != &flex2 && slots[1].assigned != &flex2);
    /* ...but as soon as the normal slot is released it is admitted. */
    for (int i = 0; i < 2; i++) if (slots[i].assigned == &normal2) {
        slots[i].assigned = NULL; slots[i].busy = false; slots[i].work = NULL; slots[i].tier = SLOT_TIER_NONE;
    }
    dispatch_jobs_locked(&s);
    TEST_ASSERT(s.head == NULL);
    TEST_ASSERT(slots[0].assigned == &flex2 || slots[1].assigned == &flex2);

    test_cancel_job_destroy(&normal2); test_cancel_job_destroy(&normal1);
    test_cancel_job_destroy(&flex2); test_cancel_job_destroy(&flex1);
    test_cancel_server_destroy(&s);
}

/* Cancel before pickup releases the tier; cancel of a queued normal
 * re-dispatches so a flex it was blocking is admitted. */
static void test_flex_cancel_paths(void) {
    server s;
    server_slot slots[2];
    test_cancel_server_init(&s);
    memset(slots, 0, sizeof(slots));
    s.slots = slots; s.slot_count = 2; s.multi_ctx_mode = true; s.flex_cap = 1;
    for (int i = 0; i < 2; i++) { slots[i].srv = &s; slots[i].id = i; }

    job normal, flex;
    test_cancel_job_init(&normal);
    test_cancel_job_init(&flex); flex.req.flex = true;

    /* Dispatched normal, cancelled before its worker runs. */
    s.head = s.tail = &normal;
    dispatch_jobs_locked(&s);
    TEST_ASSERT(server_normal_active_locked(&s) == 1);
    server_cancel_job(&s, &normal);
    TEST_ASSERT(normal.done);
    TEST_ASSERT(server_normal_active_locked(&s) == 0);
    TEST_ASSERT(slots[0].tier == SLOT_TIER_NONE && slots[1].tier == SLOT_TIER_NONE);

    /* Queued flex blocked only by a queued normal that is then cancelled:
     * the cancel must re-run dispatch. */
    job normal_q; test_cancel_job_init(&normal_q);
    slots[0].busy = true; slots[0].work = &normal_q;  /* pretend slot 0 is occupied by something else */
    slots[0].tier = SLOT_TIER_NONE;                    /* ...that is neither tier (e.g. a rebuild) */
    s.head = &flex; flex.next = &normal_q; s.tail = &normal_q;
    /* Both fit only slot 1; the normal is ahead by rule (1) but slot 1 is one
     * slot: FIFO among placeable jobs picks the normal. */
    dispatch_jobs_locked(&s);
    TEST_ASSERT(slots[1].assigned == &normal_q);
    TEST_ASSERT(s.head == &flex);
    /* Release slot 0 so the flex has somewhere to go, then cancel the normal
     * while it is still only assigned. */
    slots[0].busy = false; slots[0].work = NULL;
    server_cancel_job(&s, &normal_q);
    TEST_ASSERT(normal_q.done);
    TEST_ASSERT(s.head == NULL);
    TEST_ASSERT(slots[0].assigned == &flex || slots[1].assigned == &flex);

    /* Queued normal (not yet assigned) cancelled: dispatch runs and admits the flex. */
    job flex2, normal_q2;
    test_cancel_job_init(&flex2); flex2.req.flex = true;
    test_cancel_job_init(&normal_q2);
    for (int i = 0; i < 2; i++) { slots[i].assigned = NULL; slots[i].busy = false; slots[i].work = NULL; slots[i].tier = SLOT_TIER_NONE; }
    slots[0].busy = true; slots[0].work = &flex;  slots[0].tier = SLOT_TIER_FLEX;   /* cap reached by a running flex */
    s.head = &flex2; flex2.next = &normal_q2; s.tail = &normal_q2;
    dispatch_jobs_locked(&s);
    TEST_ASSERT(slots[1].assigned == &normal_q2);
    slots[1].assigned = NULL; slots[1].busy = false; slots[1].work = NULL; slots[1].tier = SLOT_TIER_NONE;
    slots[0].busy = false; slots[0].work = NULL; slots[0].tier = SLOT_TIER_NONE;
    /* Put normal_q2 back in the queue unassigned, then cancel it from the queue. */
    s.head = &flex2; flex2.next = &normal_q2; s.tail = &normal_q2; normal_q2.done = false;
    server_cancel_job(&s, &normal_q2);
    TEST_ASSERT(s.head == NULL);
    TEST_ASSERT(slots[0].assigned == &flex2 || slots[1].assigned == &flex2);

    test_cancel_job_destroy(&normal_q2); test_cancel_job_destroy(&flex2);
    test_cancel_job_destroy(&normal_q); test_cancel_job_destroy(&flex);
    test_cancel_job_destroy(&normal);
    test_cancel_server_destroy(&s);
}

static void test_flex_dequeue_prefers_normal(void) {
    server s;
    test_cancel_server_init(&s);
    job flex, normal;
    test_cancel_job_init(&flex); flex.req.flex = true;
    test_cancel_job_init(&normal);
    s.head = &flex; flex.next = &normal; s.tail = &normal;
    TEST_ASSERT(dequeue(&s) == &normal);
    TEST_ASSERT(s.head == &flex && s.tail == &flex && flex.next == NULL);
    TEST_ASSERT(dequeue(&s) == &flex);
    TEST_ASSERT(s.head == NULL && s.tail == NULL);
    test_cancel_job_destroy(&normal); test_cancel_job_destroy(&flex);
    test_cancel_server_destroy(&s);
}

/* A normal follow-up turn bound (D2) to the slot generating its flex
 * predecessor promotes that slot instead of starving behind it. */
static void test_flex_bound_normal_promotes_flex_slot(void) {
    server s;
    server_slot slots[2];
    test_cancel_server_init(&s);
    memset(slots, 0, sizeof(slots));
    s.slots = slots; s.slot_count = 2; s.multi_ctx_mode = true; s.flex_cap = 1;
    for (int i = 0; i < 2; i++) { slots[i].srv = &s; slots[i].id = i; }

    enum { N = 64 };
    int live[N + 8], follow[N + 8];
    for (int i = 0; i < N + 8; i++) { live[i] = i; follow[i] = i; }
    job flex_running, followup;
    test_cancel_job_init(&flex_running); flex_running.req.flex = true;
    flex_running.req.prompt = (ds4_tokens){.v = live, .len = N, .cap = N};
    test_cancel_job_init(&followup);
    followup.req.prompt = (ds4_tokens){.v = follow, .len = N + 8, .cap = N + 8};

    slots[0].busy = true; slots[0].work = &flex_running; slots[0].tier = SLOT_TIER_FLEX;
    TEST_ASSERT(job_busy_owner_locked(&s, &followup) == 0);
    TEST_ASSERT(!slots[0].promoted);

    s.head = s.tail = &followup;
    dispatch_jobs_locked(&s);
    /* Still queued (its slot is busy), but the slot it waits for now runs as normal. */
    TEST_ASSERT(s.head == &followup);
    TEST_ASSERT(slots[0].promoted);
    TEST_ASSERT(server_normal_active_locked(&s) == 1);
    TEST_ASSERT(server_slot_runnable_locked(&s, &slots[0]));

    /* Releasing the slot clears promotion with the tier. */
    pthread_mutex_lock(&s.mu);
    server_slot_release_tier(&s, &slots[0]);
    pthread_mutex_unlock(&s.mu);
    TEST_ASSERT(slots[0].tier == SLOT_TIER_NONE && !slots[0].promoted && !slots[0].parked);

    test_cancel_job_destroy(&followup); test_cancel_job_destroy(&flex_running);
    test_cancel_server_destroy(&s);
}
```

Register all four after `test_flex_predicates();`.

- [ ] **Step 2: Run to verify they fail**

Run: `make ds4_server_test 2>&1 | grep -m3 error`
Expected: `server_slot_release_tier` undeclared (and, once declared, assertion failures on gating).

- [ ] **Step 3: Tier bookkeeping helpers**

Add above `dispatch_jobs_locked`:

```c
/* s->mu held.  Forget the tier the slot was holding, and with it promotion
 * and parking: the slot is idle again as far as the executor is concerned. */
static void server_slot_release_tier(server *s, server_slot *slot) {
    pthread_mutex_lock(&s->model_mu);
    slot->tier = SLOT_TIER_NONE;
    slot->promoted = false;
    slot->parked = false;
    pthread_cond_broadcast(&s->model_cv);   /* a parked flex re-checks runnable */
    pthread_mutex_unlock(&s->model_mu);
}

/* s->mu held.  May a flex job be placed this pass?  Three rules: no normal
 * job anywhere in the queue (admission-only -- it must never feed the pause
 * predicate, or a normal bound to a parked flex slot would deadlock it), no
 * slot holding normal work, and fewer than flex_cap slots holding flex. */
static bool server_flex_placeable_locked(server *s) {
    for (const job *j = s->head; j; j = j->next) {
        if (!j->req.flex) return false;
    }
    int flex_held = 0;
    for (int i = 0; i < s->slot_count; i++) {
        const job *work = s->slots[i].work;
        if (work && work->req.flex) flex_held++;
    }
    pthread_mutex_lock(&s->model_mu);
    const bool normal_active = server_normal_active_locked(s) > 0;
    pthread_mutex_unlock(&s->model_mu);
    if (normal_active) return false;
    const int cap = s->flex_cap > 0 ? s->flex_cap : (s->slot_count > 1 ? s->slot_count - 1 : 1);
    return flex_held < cap;
}
```

- [ ] **Step 4: Gate and record in `dispatch_jobs_locked`**

Inside the `for (;;)` loop, before `pthread_mutex_lock(&s->tool_mu);` add `const bool flex_ok = server_flex_placeable_locked(s);`. In the job scan, as the first statement of the `for (job *j = s->head; ...)` body, add `if (j->req.flex && !flex_ok) continue;`.

After `if (required < 0) required = job_busy_owner_locked(s, j);` add promotion:

```c
            /* A normal request bound to a slot that is running flex work
             * would wait behind a flex that parks whenever another normal
             * runs.  Promote the slot: it runs as normal until released. */
            if (required >= 0 && !j->req.flex) {
                server_slot *owner = &s->slots[required];
                if (owner->work && owner->work->req.flex && !owner->promoted) {
                    pthread_mutex_lock(&s->model_mu);
                    owner->promoted = true;
                    pthread_cond_broadcast(&s->model_cv);
                    pthread_mutex_unlock(&s->model_mu);
                }
            }
```

(`tool_mu` is held here; the order `s->mu -> tool_mu -> model_mu` is the verified one.)

In the assignment block, inside the existing `pthread_mutex_lock(&s->model_mu); ... pthread_mutex_unlock(&s->model_mu);` section, add `chosen_slot->tier = chosen->req.flex ? SLOT_TIER_FLEX : SLOT_TIER_NORMAL; chosen_slot->promoted = false; chosen_slot->parked = false;` before the broadcast.

- [ ] **Step 5: Release the tier where the slot is released**

`slot_worker_main`: between `slot->work = NULL;` and `dispatch_jobs_locked(s);` add `server_slot_release_tier(s, slot);`.

`server_cancel_job`, assigned-slot branch: replace the block

```c
            pthread_mutex_lock(&s->model_mu);
            slot->awaiting_first_grant = false;
            pthread_mutex_unlock(&s->model_mu);
```

with

```c
            pthread_mutex_lock(&s->model_mu);
            slot->awaiting_first_grant = false;
            pthread_mutex_unlock(&s->model_mu);
            server_slot_release_tier(s, slot);   /* same reason, same ordering */
```

Queued-job branch: after the `for (job *it = s->head; ...)` loop, change the following `if (!detached && server_time_sliced(s))` so that a detach from the queue also re-dispatches:

```c
    if (detached && server_time_sliced(s)) {
        /* The job that just left the queue may have been the normal that was
         * keeping a flex job out; give the queue another look. */
        dispatch_jobs_locked(s);
    }
    if (!detached && server_time_sliced(s)) {
```

- [ ] **Step 6: `dequeue` prefers normal**

Replace the body after the wait in `dequeue` with:

```c
    job *prev = NULL, *j = s->head;
    for (job *it = s->head, *p = NULL; it; p = it, it = it->next) {
        if (!it->req.flex) { j = it; prev = p; break; }
    }
    if (prev) prev->next = j->next; else s->head = j->next;
    if (s->tail == j) s->tail = prev;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;
    return j;
```

- [ ] **Step 7: `--flex-contexts` and the default cap**

`server_config`: add `int flex_contexts;` after `bool batched_decode_set;`. Flag parsing, next to `--mixed-prefill-quantum`:

```c
        } else if (!strcmp(arg, "--flex-contexts")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                server_log(DS4_LOG_DEFAULT, "ds4-server: --flex-contexts must be positive");
                exit(2);
            }
            c.flex_contexts = v;
```

Startup, after `s.mixed_prefill_quantum = cfg.mixed_prefill_quantum;`:

```c
    /* Flex tier: how many contexts background requests may hold at once.
     * One fewer than the contexts by default, so a normal request always
     * finds a free slot instead of waiting for a flex to finish. */
    s.flex_cap = slot_count > 1 ? slot_count - 1 : 1;
    if (cfg.flex_contexts > 0) {
        s.flex_cap = cfg.flex_contexts > slot_count ? slot_count : cfg.flex_contexts;
    }
```

Also add the flag to the usage text if `ds4_server.c` prints one: `grep -n '"--mixed-prefill-quantum' ds4_server.c ds4_help.h` and mirror the neighbour's format; if there is no usage entry for `--mixed-prefill-quantum`, skip this.

- [ ] **Step 8: Run tests**

Run: `make ds4_server_test 2>&1 | grep -E "error|warning"; ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: no diagnostics, `0`. If `test_flex_cancel_paths` fails on the "pretend occupied by something else" sub-case, simplify that sub-case rather than weakening the release logic: the properties that matter are (a) cancel-before-pickup leaves `server_normal_active_locked == 0`, and (b) cancelling a queued normal admits the waiting flex.

- [ ] **Step 9: Commit**

```bash
git add -p ds4_server.c
git commit -m "server: flex admission gating, promotion, --flex-contexts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Stream keepalive that can run before the SSE headers

**Files:**
- Modify: `ds4_server.c` `server_prefill_progress` (~:12340-12366), `server_progress_cb` (~:13167-13200), `struct server_slot` (add `server_prefill_progress *progress;`), the `ds4_session_set_progress` install/clear sites (~:13496, :13500, :13519, :14115, :14199, :14233, :14250, :14262)
- Test: `ds4_server.c` new `test_flex_stream_keepalive_sends_headers_first`

**Interfaces:**
- Produces: `static bool server_stream_keepalive(server_prefill_progress *p, const char *comment)` — no-op (returns true) when `!p || !p->stream || p->fd < 0 || p->stream_failed`; sends `sse_headers` first when `!p->headers_sent`; writes `comment` (must already end in `"\n\n"`); on failure sets `stream_failed`, marks the job cancelled, returns false. `slot->progress` is the worker-owned pointer to the live progress struct (NULL when none is installed).

- [ ] **Step 1: Write the failing test**

```c
static void test_flex_stream_keepalive_sends_headers_first(void) {
    job j;
    test_cancel_job_init(&j);
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    server_prefill_progress p = { .request_job = &j, .fd = sv[0], .stream = true };

    TEST_ASSERT(server_stream_keepalive(&p, ": keepalive\n\n"));
    TEST_ASSERT(p.headers_sent);
    TEST_ASSERT(server_stream_keepalive(&p, ": keepalive\n\n"));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 200") != NULL);
    TEST_ASSERT(strstr(out, "text/event-stream") != NULL);
    const char *first = strstr(out, ": keepalive\n\n");
    TEST_ASSERT(first != NULL && strstr(first + 1, ": keepalive\n\n") != NULL);
    free(out);
    close(sv[0]); close(sv[1]);

    /* Non-streaming: writes nothing, reports success. */
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    server_prefill_progress q = { .request_job = &j, .fd = sv[0], .stream = false };
    TEST_ASSERT(server_stream_keepalive(&q, ": keepalive\n\n"));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);
    TEST_ASSERT(out[0] == '\0');
    free(out);
    close(sv[0]); close(sv[1]);

    /* The multi-context engine reports "prefill": that must count. */
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    server_prefill_progress r = { .request_job = &j, .fd = sv[0], .stream = true,
                                  .prompt_tokens = 100 };
    server_progress_cb(&r, "prefill", 10, 100);
    TEST_ASSERT(r.headers_sent);
    close(sv[0]); close(sv[1]);
    test_cancel_job_destroy(&j);
}
```

Register after the Task 4 tests.

- [ ] **Step 2: Run to verify it fails**

Run: `make ds4_server_test 2>&1 | grep -m3 error`
Expected: `server_stream_keepalive` undeclared.

- [ ] **Step 3: Implement**

Add just above `server_progress_cb`:

```c
/* Keep a streaming client alive while the server is not producing tokens:
 * the SSE headers go out first if they have not yet, then `comment` (a `:`
 * line, ignored by SSE clients; must end in a blank line).  Shared by the
 * prefill progress callback and the flex pause point.  Returns false, marks
 * the stream failed and cancels the job when the socket is gone.  Never call
 * with model_mu held: send_all can block for DS4_SERVER_SEND_STALL_TIMEOUT_MS. */
static bool server_stream_keepalive(server_prefill_progress *p, const char *comment) {
    if (!p || !p->stream || p->fd < 0 || p->stream_failed) return true;
    if (p->request_job && job_cancelled(p->request_job)) return false;
    bool ok = true;
    if (!p->headers_sent) {
        p->headers_sent = true;
        ok = sse_headers(p->fd, p->enable_cors);
    }
    if (ok && comment) ok = send_all(p->fd, comment, strlen(comment));
    if (!ok) {
        p->stream_failed = true;
        if (p->request_job) job_mark_cancelled(p->request_job);
        return false;
    }
    p->last_keepalive = now_sec();
    return true;
}
```

Rewrite the head of `server_progress_cb` to use it and to accept the `"prefill"` event:

```c
static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = ud;
    if (!p || !event || job_cancelled(p->request_job)) return;
    const bool is_chunk = strcmp(event, "prefill_chunk") == 0 ||
                          strcmp(event, "prefill") == 0;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (!is_chunk && !is_display) return;

    double now = now_sec();
    /* Keep the HTTP/SSE connection alive while prefill runs: headers on the
     * first callback, then a comment line every few seconds. */
    if (!p->headers_sent) {
        if (!server_stream_keepalive(p, NULL)) return;
    } else if (now - p->last_keepalive >= 5.0) {
        if (!server_stream_keepalive(p, ": prefill\n\n")) return;
    }
    if (is_display) return;
```

and delete the old `if (p->stream && p->fd >= 0 && !p->stream_failed) { ... }` block it replaces. Check that the rest of the callback (the `"prefill"` event's `current/total` semantics from `ds4.c:73344`) still logs sensibly: run `grep -n '"prefill"' ds4.c | head` and read the call to confirm `current` is the position and `total` the prompt length, same as `prefill_chunk`. If the meaning differs, keep `"prefill"` for the keepalive branch only and `return` before the logging when `strcmp(event, "prefill") == 0`.

Add `server_prefill_progress *progress;` to `struct server_slot` (after `bool parked;`), with the comment `/* the worker's live progress struct while one is installed; the pause point keeps the stream alive through it */`. At every `ds4_session_set_progress(slot->session, server_progress_cb, &X);` add `slot->progress = &X;` on the next line, and at every `ds4_session_set_progress(slot->session, NULL, NULL);` add `slot->progress = NULL;`. There are two installs (`&rebuild_progress`, `&progress`) and six clears; grep to confirm none is missed: `grep -n "ds4_session_set_progress" ds4_server.c`.

The `progress` struct in `generate_job_inner` lives until the function returns, and `slot->progress` is only read by the same worker thread, so the pointer is valid whenever non-NULL. Also set `slot->progress = NULL;` at the top of `generate_job` (before `generate_job_inner`) as a belt-and-braces reset.

- [ ] **Step 4: Run tests**

Run: `make ds4_server_test 2>&1 | grep -E "error|warning"; ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: no diagnostics, `0`. `test_cancelled_progress_callback_is_inert` must still pass (its progress has `fd = 0`, `stream = false`).

- [ ] **Step 5: Commit**

```bash
git add -p ds4_server.c
git commit -m "server: shared SSE keepalive that sends headers first; keepalive on qwen4exp prefill events

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The pause point

**Files:**
- Modify: `ds4_server.c` executor section (add `server_flex_pause_point` after `server_prefill_chunk_rows`, ~:12897), `server_prefill_yield_cb` (~:12899), `generate_job_inner` decode loop (top of loop body ~:14415, before the `dsml_decode_state` line)
- Test: `ds4_server.c` new `test_flex_pause_point_parks_and_resumes`

**Interfaces:**
- Consumes: `server_slot_runnable_locked`, `slot->parked`, `slot->progress`, `server_stream_keepalive` (Tasks 3, 5).
- Produces: `static bool server_flex_pause_point(server *s, server_slot *slot, job *j)` — returns true when the slot may take its next step, false when the request was cancelled or the server is stopping. Only meaningful in `multi_ctx_mode`; a no-op (true) otherwise.

- [ ] **Step 1: Write the failing test**

```c
typedef struct {
    server *srv;
    server_slot *slot;
    job *j;
    bool result;
    bool returned;
} test_flex_pause_arg;

static void *test_flex_pause_main(void *ud) {
    test_flex_pause_arg *a = ud;
    a->result = server_flex_pause_point(a->srv, a->slot, a->j);
    a->returned = true;
    return NULL;
}

/* A flex slot parks while a normal slot holds work, clears its startup flag
 * so no prefill defers to it, and resumes when the normal slot is released.
 * A cancel while parked returns false. */
static void test_flex_pause_point_parks_and_resumes(void) {
    server s;
    server_slot slots[2];
    test_cancel_server_init(&s);
    memset(slots, 0, sizeof(slots));
    s.slots = slots; s.slot_count = 2; s.multi_ctx_mode = true;
    for (int i = 0; i < 2; i++) { slots[i].srv = &s; slots[i].id = i; }
    job flex; test_cancel_job_init(&flex); flex.req.flex = true;

    slots[0].tier = SLOT_TIER_FLEX;
    slots[0].awaiting_first_grant = true;
    /* Nothing normal: returns at once, nothing parked. */
    TEST_ASSERT(server_flex_pause_point(&s, &slots[0], &flex));
    TEST_ASSERT(!slots[0].parked);

    slots[1].tier = SLOT_TIER_NORMAL;
    test_flex_pause_arg a = { .srv = &s, .slot = &slots[0], .j = &flex };
    pthread_t th;
    TEST_ASSERT(pthread_create(&th, NULL, test_flex_pause_main, &a) == 0);
    for (int i = 0; i < 200 && !slots[0].parked; i++) usleep(1000);
    pthread_mutex_lock(&s.model_mu);
    TEST_ASSERT(slots[0].parked);
    TEST_ASSERT(!slots[0].awaiting_first_grant);
    TEST_ASSERT(!a.returned);
    /* Release the normal slot the way slot_worker_main does. */
    slots[1].tier = SLOT_TIER_NONE;
    pthread_cond_broadcast(&s.model_cv);
    pthread_mutex_unlock(&s.model_mu);
    pthread_join(th, NULL);
    TEST_ASSERT(a.returned && a.result);
    TEST_ASSERT(!slots[0].parked);

    /* Promotion also releases a parked flex. */
    slots[1].tier = SLOT_TIER_NORMAL;
    a = (test_flex_pause_arg){ .srv = &s, .slot = &slots[0], .j = &flex };
    TEST_ASSERT(pthread_create(&th, NULL, test_flex_pause_main, &a) == 0);
    for (int i = 0; i < 200 && !slots[0].parked; i++) usleep(1000);
    pthread_mutex_lock(&s.model_mu);
    slots[0].promoted = true;
    pthread_cond_broadcast(&s.model_cv);
    pthread_mutex_unlock(&s.model_mu);
    pthread_join(th, NULL);
    TEST_ASSERT(a.result);
    slots[0].promoted = false;

    /* Cancel while parked: false, not parked. */
    a = (test_flex_pause_arg){ .srv = &s, .slot = &slots[0], .j = &flex };
    TEST_ASSERT(pthread_create(&th, NULL, test_flex_pause_main, &a) == 0);
    for (int i = 0; i < 200 && !slots[0].parked; i++) usleep(1000);
    job_mark_cancelled(&flex);
    pthread_mutex_lock(&s.model_mu);
    pthread_cond_broadcast(&s.model_cv);
    pthread_mutex_unlock(&s.model_mu);
    pthread_join(th, NULL);
    TEST_ASSERT(!a.result);
    TEST_ASSERT(!slots[0].parked);

    test_cancel_job_destroy(&flex);
    test_cancel_server_destroy(&s);
}
```

Register after the Task 5 test. (`usleep` needs `<unistd.h>`, already included.)

- [ ] **Step 2: Run to verify it fails**

Run: `make ds4_server_test 2>&1 | grep -m3 error`
Expected: `server_flex_pause_point` undeclared.

- [ ] **Step 3: Implement the pause point**

After `server_prefill_chunk_rows` add:

```c
/* The flex tier's one scheduling hook.  A flex slot's worker calls this
 * between steps -- at the top of each decode iteration and between prefill
 * chunks -- and waits here while any slot holds normal work.  The grant
 * functions are untouched: a normal dispatched while this slot is already
 * inside a grant costs at most that one step.
 *
 * Nothing but `parked` is written, so cancel and error paths cannot drift a
 * counter.  awaiting_first_grant is cleared because a fully cached prompt
 * reaches this point before its first grant, and a normal prefill would
 * otherwise defer to a startup that is parked.  The keepalive write runs
 * with model_mu released (send_all can block for two seconds). */
#define DS4_FLEX_KEEPALIVE_SEC 5.0

static bool server_flex_pause_point(server *s, server_slot *slot, job *j) {
    if (!s || !slot || !s->multi_ctx_mode) return true;
    if (g_stop_requested || job_cancelled(j)) return false;
    const bool log = getenv("DS4_SERVER_BATCH_LOG") != NULL;
    double parked_at = 0.0;

    pthread_mutex_lock(&s->model_mu);
    while (!g_stop_requested && !s->model_stopping && !job_cancelled(j) &&
           !server_slot_runnable_locked(s, slot)) {
        if (!slot->parked) {
            slot->parked = true;
            slot->awaiting_first_grant = false;
            parked_at = now_sec();
            if (log) server_log(DS4_LOG_DEFAULT, "ds4-server: flex slot %d parked", slot->id);
            pthread_cond_broadcast(&s->model_cv);
        }
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        timespec_add_us(&deadline, (long)(DS4_FLEX_KEEPALIVE_SEC * 1e6));
        int rc = pthread_cond_timedwait(&s->model_cv, &s->model_mu, &deadline);
        if (rc == ETIMEDOUT && slot->progress) {
            pthread_mutex_unlock(&s->model_mu);
            (void)server_stream_keepalive(slot->progress, ": keepalive\n\n");
            pthread_mutex_lock(&s->model_mu);
        }
    }
    const bool was_parked = slot->parked;
    slot->parked = false;
    const bool ok = !g_stop_requested && !s->model_stopping && !job_cancelled(j);
    if (was_parked) pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
    if (was_parked && log) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: flex slot %d %s after %.1f s",
                   slot->id, ok ? "resumed" : "released", now_sec() - parked_at);
    }
    return ok;
}
```

`timespec_add_us` is defined at ~:13722, after this point in the file; move its definition (and nothing else) up to just above `server_flex_pause_point`, or add a forward declaration `static void timespec_add_us(struct timespec *ts, long us);` above the pause point.

- [ ] **Step 4: Call sites**

(a) `generate_job_inner` decode loop: as the first statement inside the `while (!g_stop_requested && !job_cancelled(j) && completion < max_tokens && ...)` body (before the `dsml_decode_state dsml_state = ...` line), add:

```c
        if (j->req.flex && !server_flex_pause_point(s, slot, j)) {
            finish = "error";
            snprintf(err, sizeof(err), "%s",
                     g_stop_requested ? "shutdown requested" : "client disconnected");
            break;
        }
```

Confirm `finish` and `err` are the variables the loop already uses for the same purpose at the `server_model_enter_decode` failure a few lines below (~:14468-14473).

(b) `server_prefill_yield_cb`: between `server_model_leave(t->srv); t->held = false;` and `if (!server_model_enter_prefill(t->srv, t->slot)) return -1;` add:

```c
    if (t->slot->tier == SLOT_TIER_FLEX &&
        !server_flex_pause_point(t->srv, t->slot, t->slot->work)) {
        return -1;
    }
```

`t->slot->work` is the running job (set at dispatch, cleared after the worker returns; guarded by `s->mu`, but on the worker's own slot it is stable for the life of the request, the same reasoning `job_busy_owner_locked` documents). Reading `t->slot->tier` without model_mu is fine for the same reason: it is written at dispatch before this worker starts and cleared by this worker after it finishes.

- [ ] **Step 5: Run tests**

Run: `make ds4_server_test 2>&1 | grep -E "error|warning"; ./ds4_server_test 2>&1 | grep -c "assertion failed"`
Expected: no diagnostics, `0`.

- [ ] **Step 6: Build the real server and smoke-run existing scheduler tests**

Run: `make ds4-server 2>&1 | grep -E "error|warning" ; ls -la ds4-server`
Expected: no diagnostics, binary rebuilt (fresh timestamp).

- [ ] **Step 7: Commit**

```bash
git add -p ds4_server.c
git commit -m "server: flex pause point between decode steps and prefill chunks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Integration test against a live server

**Files:**
- Create: `tests/test_flex_tier.py`
- Reference: `tests/bench_concurrency.py`, `tests/run_concurrency_bench.sh` (how the server is started with `--exec-contexts`)

**Interfaces:**
- Consumes: a running `ds4-server` at `--exec-contexts 2` on `http://127.0.0.1:PORT` with the qwen4exp GGUF (see `tests/run_concurrency_bench.sh` for the launch line; `DS4_LOCK_FILE` must be set on this box). `DS4_SERVER_BATCH_LOG=1` in the server's environment makes the parked/resumed lines visible.

- [ ] **Step 1: Write the test script**

```python
#!/usr/bin/env python3
"""Flex tier end-to-end check against a running ds4-server (--exec-contexts 2).

Scenario A: a streaming flex request is generating; a normal request arrives.
  The flex stream must go silent (only ': keepalive' comments) from the normal's
  first byte to its completion, then resume and finish.  Greedy, so the flex
  text must equal a solo flex run.
Scenario B: two flex requests -> the second is not admitted (flex_cap 1);
  a normal then starts within a bounded TTFT.
Usage: python3 tests/test_flex_tier.py --base http://127.0.0.1:8080 [--model NAME]
"""
import argparse, json, sys, threading, time, urllib.request

def sse_stream(base, payload, headers, events):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\n")
            now = time.time()
            if line.startswith(": "):
                events.append((now, "comment", line))
            elif line.startswith("data: ") and line != "data: [DONE]":
                obj = json.loads(line[6:])
                delta = obj["choices"][0]["delta"].get("content") if obj.get("choices") else None
                if delta:
                    events.append((now, "token", delta))
                if obj.get("service_tier"):
                    events.append((now, "tier", obj["service_tier"]))

def payload(model, text, max_tokens, flex_body=False, stream=True):
    p = {"model": model, "messages": [{"role": "user", "content": text}],
         "max_tokens": max_tokens, "temperature": 0.0, "stream": stream}
    if flex_body:
        p["service_tier"] = "flex"
    return p

FLEX_PROMPT = "Write a long, detailed essay about the history of the bicycle. Do not stop early."
NORMAL_PROMPT = "List ten prime numbers, one per line."

def scenario_a(base, model):
    solo = []
    sse_stream(base, payload(model, FLEX_PROMPT, 160, flex_body=True), {}, solo)
    solo_text = "".join(d for _, k, d in solo if k == "token")
    assert any(k == "tier" and d == "flex" for _, k, d in solo), "service_tier not echoed"

    flex_events, normal_events = [], []
    t = threading.Thread(target=sse_stream, args=(base, payload(model, FLEX_PROMPT, 160), {"X-Service-Tier": "flex"}, flex_events))
    t.start()
    while sum(1 for _, k, _ in flex_events if k == "token") < 8:
        time.sleep(0.05)
    t_normal0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 48), {}, normal_events)
    t_normal1 = time.time()
    t.join()

    normal_first = min(ts for ts, k, _ in normal_events if k == "token")
    leaked = [ts for ts, k, _ in flex_events if k == "token" and normal_first + 0.25 < ts < t_normal1 - 0.05]
    flex_text = "".join(d for _, k, d in flex_events if k == "token")
    print(f"A: normal TTFT {normal_first - t_normal0:.2f}s, normal wall {t_normal1 - t_normal0:.2f}s, "
          f"flex tokens leaked during normal: {len(leaked)}, flex tokens total {sum(1 for _, k, _ in flex_events if k == 'token')}")
    assert not leaked, f"flex emitted {len(leaked)} tokens while a normal request ran"
    assert flex_text == solo_text, "flex text diverged from the solo run"
    return True

def scenario_b(base, model):
    ev1, ev2, evn = [], [], []
    t1 = threading.Thread(target=sse_stream, args=(base, payload(model, FLEX_PROMPT, 120, flex_body=True), {}, ev1))
    t2 = threading.Thread(target=sse_stream, args=(base, payload(model, FLEX_PROMPT + " Second.", 120, flex_body=True), {}, ev2))
    t1.start()
    while not any(k == "token" for _, k, _ in ev1):
        time.sleep(0.05)
    t2.start()
    time.sleep(2.0)
    assert not any(k == "token" for _, k, _ in ev2), "second flex was admitted despite flex_cap 1"
    t0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 32), {}, evn)
    ttft = min(ts for ts, k, _ in evn if k == "token") - t0
    print(f"B: normal TTFT with one flex running and one queued: {ttft:.2f}s")
    assert ttft < 15.0, "normal request did not get the free slot promptly"
    t1.join(); t2.join()
    assert any(k == "token" for _, k, _ in ev2), "queued flex never ran"
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen4exp")
    a = ap.parse_args()
    ok = scenario_a(a.base, a.model) and scenario_b(a.base, a.model)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
```

Check the model id the server expects: `curl -s http://127.0.0.1:PORT/v1/models | head -c 400` and pass it with `--model`.

- [ ] **Step 2: Start the server and run**

Use the launch line from `tests/run_concurrency_bench.sh` with `CTXS=2`, adding `DS4_SERVER_BATCH_LOG=1` to the environment, e.g. (adapt paths from that script):

```bash
DS4_SERVER_BATCH_LOG=1 ./ds4-server --exec-contexts 2 <model and port args from run_concurrency_bench.sh> 2>&1 | tee /tmp/claude-1000/-home-otsimo-work-ds4/f0897f04-33ed-4944-aa09-85a220262506/scratchpad/flex-server.log &
python3 tests/test_flex_tier.py --base http://127.0.0.1:<port> --model <id>
grep -E "flex slot [0-9]+ (parked|resumed)" /tmp/claude-1000/-home-otsimo-work-ds4/f0897f04-33ed-4944-aa09-85a220262506/scratchpad/flex-server.log
```

Expected: `A: ... flex tokens leaked during normal: 0`, `B: normal TTFT ... < 15 s`, `PASS`, and at least one `parked` and one `resumed` log line. If the GPU is busy (`DS4_LOCK_FILE` held), record that the integration step was not run and say so in the handoff; do not claim it passed.

- [ ] **Step 3: Regression guard**

Run `tests/bench_concurrency.py` the way `tests/run_concurrency_bench.sh` does (no flex traffic) and compare aggregate tok/s and TTFT with the numbers in `docs/superpowers/vllm-parity-architecture.md`'s baseline table (or the last run logged there). Expected: within noise (a few percent).

- [ ] **Step 4: Commit**

```bash
git add tests/test_flex_tier.py
git commit -m "tests: flex tier end-to-end check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Document the feature

**Files:**
- Modify: `docs/superpowers/specs/2026-09-07-flex-service-tier-design.md` (append an "Implementation status" section), `docs/superpowers/vllm-parity-architecture.md` (one paragraph under Stage 3 noting the runnable predicate to fold into the tick loop)
- Modify: the server's operator documentation if one exists for flags (`grep -rn "exec-contexts" README* docs/ | grep -v superpowers`); add `--flex-contexts`, `X-Service-Tier`, and `service_tier` there. If none exists, skip.

- [ ] **Step 1: Write the status section**

Append to the spec:

```markdown
## Implementation status (2026-09-07)

Landed on `qwen3.8-flash-next`: request parsing (body + header), response echo,
per-slot tier facts (`tier`, `promoted`, `generating`, `parked`), the derived
predicates `server_normal_active_locked` / `server_slot_runnable_locked` /
`server_eligible_generations_locked`, admission gating with `--flex-contexts`,
promotion on affinity binding, the shared `server_stream_keepalive`, and
`server_flex_pause_point` at the decode-loop top and between prefill chunks.
Unit tests: `make ds4_server_test && ./ds4_server_test`. Integration:
`tests/test_flex_tier.py` (record the run's numbers here).

Follow-ups not done: gating flex admission on KV pool headroom; evicting a
parked flex under slot pressure; a pause deadline / 429.
```

- [ ] **Step 2: Note the fold-in for the tick loop**

In `docs/superpowers/vllm-parity-architecture.md`, under the Stage 3 "Continuous admission/eviction" bullet, add:

```markdown
Flex tier (spec `specs/2026-09-07-flex-service-tier-design.md`): when the executor
becomes one tick loop, the live set for a tick is `{slot : generating &&
server_slot_runnable_locked(s, slot)}` and the prefill share uses the same
predicate; `server_flex_pause_point` then disappears and `parked` is what the tick
computes rather than what the worker publishes.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-09-07-flex-service-tier-design.md docs/superpowers/vllm-parity-architecture.md
git commit -m "docs: flex tier status and tick-loop fold-in note

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(`docs/superpowers/vllm-parity-architecture.md` is currently untracked; adding it commits the whole file, which is the intent of the user's docs convention. If the user prefers it uncommitted, leave that file out and only commit the spec.)
