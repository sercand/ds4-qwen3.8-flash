# Task 8: Engine checkpoint integration

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 8 (read it before executing; the spec argues this plan)
**Depends on:** T6 (`Qwen35Model`, `SessionStore`, `Session { id, n_written, kv, gdn_conv, gdn_state, logits_row }`, `SessionKv{cap,n_layers,kv_heads,head_dim,k/v: Vec<DeviceBuffer>}` f16, `gdn_conv: Vec<DeviceBuffer>` ×24, `gdn_state: Vec<DeviceBuffer>` ×24, `logits_row: DeviceBuffer` 248320×f32 — all `pub(crate)`), T7 (`KvStore::{open,get,put,report,touch}`, `HeaderParts`, `KvHit{entry,text,payload}`, `KvEntry`, `Reason`, `KvReport{entries,dir}`), T2 (`Backend::alloc/copy_d2h/copy_h2d`, `DeviceView{backend,buf,offset}`, `Model`, `ShapeProfile`, `CoreError`, `Result`).
⚠ DEVIATION (engine crate deps): `shisu-engine/Cargo.toml` gains `shisu-kvstore` + `sha1` + `hex` (workspace deps from T1). T6's "no Metal/CUDA deps in shisu-engine" rule is about backends; the checkpoint layer is host-side and T7 predates T8, so this is the intended cutover.

**Produces** (consumed by T10's `--kvstore-dir` wiring, T13's scheduler lifecycle, T20's resume-latency gate):

```rust
// shisu-engine/src/kv_ckpt.rs
pub const CKPT_MIN_TOKENS_DEFAULT: u32 = 512;   // ds4_kvstore.c:33 KV_CACHE_DEFAULT_MIN_TOKENS
pub const QWEN35_MODEL_ID: u8 = 5;              // ds4.c:509-513: next id after ds4's 0..=4
pub const QWEN4EXP_MODEL_ID: u8 = 4;            // == DS4_VARIANT_QWEN38F
pub use shisu_kvstore::Reason;                  // {Cold, Continued, Evict, Shutdown, ..}
pub struct CkptOptions { pub min_tokens: u32, pub reject_different_quant: bool }
pub struct ResumeInfo { pub tokens_reused: u32, pub text_bytes: u32,
                        pub tokens: Vec<u32>, pub frontier_logits: DeviceView }
pub trait Checkpointed: Model {
    fn kv_enable(&mut self, store: KvStore, opts: CkptOptions) -> Result<()>;
    fn kv_checkpoint_enabled(&self) -> bool;
    fn kv_resume(&mut self, session: SessionId, prompt_text: &[u8]) -> Result<Option<ResumeInfo>>;
    fn kv_save(&mut self, session: SessionId, tokens: &[u32], prompt_text: &[u8],
               logits: &DeviceView, reason: Reason, ext_flags: u8) -> Result<bool>;
    fn kv_shutdown_sweep(&mut self, text_of: &mut dyn FnMut(SessionId) -> Option<(Vec<u32>, Vec<u8>)>) -> usize;
}
impl Checkpointed for Qwen35Model { /* in kv_ckpt.rs */ }
```

⚠ DEVIATION (`ResumeInfo` extra fields): the assignment/master plan name only `tokens_reused`/`text_bytes`. `tokens` is required because the engine has no tokenizer: ds4 hands the caller the exact stored token history and lets it re-tokenize only the suffix (`ds4_kvstore.c:1331-1339` builds `effective_prompt = exact_prefix ++ tokenize(suffix)`); shisu's caller (T10/T13, which owns the tokenizer) needs the same vector to build the effective prompt. `frontier_logits` is required because the payload carries the frontier logits row (see Decision 3) — without it an exact re-send has no logits to sample and re-running one decode step would corrupt the path (it would write a KV row at `n_written`).
⚠ DEVIATION (T7 additive): `shisu-kvstore` gains `KvStore::remove(&self, prefix_sha1: &str) -> Result<bool>`. T7 Deviation 8 assigns the model/quant/ctx compatibility policy to the engine (this task), and that policy is ds4's *unlink*-on-incompatible (`ds4_server.c:883-895`, `ds4_kvstore.c:1344-1351,1356-1369`); T7 exposes no delete. One method + one test, in T8's commit.

## Global Constraints

- Rule 1 (no host C/C++ in `crates/`): everything here is Rust; the only C is read as reference.
- Rule 2 (copy kernels verbatim): N/A — host orchestration only.
- Rule 3 (arithmetic order): the payload is a byte image of device buffers; no math is performed on restore.
- Rule 4 (no perf claims from CuMetal): resume latency is *recorded*, never gated here (T20 owns gates).
- Rule 5 (1500-LoC cap, local gate `scripts/file_size_check.sh`): pre-split `kv_ckpt.rs` (policy/orchestration) + `kv_payload.rs` (byte layout) — serialization + geometry check alone approach the cap.
- Rule 6 (`[lints]` workspace-wide): `thiserror` for any new error, `parking_lot` for locks, `tracing` for logs; no `unwrap()` on IO paths — a failed read/write is a cold start, never a panic (ds4 treats every store/load failure as "run cold", `ds4_kvstore.c:1356-1369`).
- Rule 7 (no `ATLAS_*`/`DS4_*` strings, no license headers) in shipped code; `DS4_*` is allowed only in doc comments citing sources.
- Extras: the engine crate reads **no env vars** (T6 rule: only `SHISU_TEST_MODEL`); `SHISU_KVSTORE_DIR` is T10's flag/env (Decision 6). Deterministic byte-exactness, not tolerance, is the unit-test contract (the payload is a memcpy, not a computation).

## Source References (verified)

| ds4 file:lines | what / how used |
|---|---|
| `ds4_kvstore.c:985-998` | `ds4_kvstore_store_live_prefix_text(kc, e, s, tokens, store_len, text, text_bytes, reason, ext_flags, &wrote)` — the save entry point shape T8's `kv_save` mirrors (caller passes tokens + text). |
| `ds4_kvstore.c:971-981` | store-side gates: `store_len < opt.min_tokens` → skip; `quant_bits ∉ {2,4}` → skip. |
| `ds4_kvstore.c:1001-1013` | `sha1(text)` → hex → `ds4_kvstore_put` (T7's `put`). |
| `ds4_kvstore.c:1264-1300` | `find_text_prefix`: hash every prefix `1..=len`, keep the entry with the **largest `text_bytes`**, tie-break **more tokens** (`:1259-1260`). T8 owns this scan (T7's `get`/`find_prefix` are exact-sha only). |
| `ds4_kvstore.c:1302-1390` | `try_load_text`: `get(sha)` → verify `hit.text` is a byte prefix of the request text (`:1315-1319`) → header sanity (`tokens != 0`, `quant_bits ∈ {2,4}`, `ctx_size <= engine ctx`, `:1321-1328`) → `effective_prompt = exact ++ tokenize(suffix)` (`:1331-1339`) → unlink on corrupt token history (`:1344-1351`) / unreadable payload (`:1356-1369`). |
| `ds4_kvstore.c:875-897` | `kv_cache_existing_compatible` (ds4_server.c) + `:1275-1276` put-side reject → the engine-side compat policy (T7 Deviation 8). |
| `ds4_server.c:11322-11568` | `kv_cache_store_slot` — the wrapper: min_tokens gate, frontier-logits gate (`:11476-11482`), staged-payload reuse (`:11460-11475`), reason/ext_flags selection (`:11385-11405`, `KV_EXT_TOOL_MAP`/`KV_EXT_THINKING_VISIBLE`). |
| `ds4_server.c:14119-14121,14177-14181,14202-14204` | `reason = KV_REASON_COLD` — after a cold prefill of the whole prompt. |
| `ds4_server.c:13181,13222,14615` | `ds4_kvstore_store_continued` — mid-stream progress + stream end. |
| `ds4_server.c:13899-13912` | `reason = KV_REASON_EVICT` — save the outgoing session immediately before a disk load repurposes its slot. |
| `ds4_server.c:16540-16548` | shutdown sweep: loop resident slots, skip `tokens < min_tokens`, store with `reason = KV_REASON_SHUTDOWN`. |
| `ds4_agent.c:1318,1363,1410,1529,1583,1610,1670,1734,1789,1846,1896,1957,2176,2186,2193` | `KV_REASON_AGENT_SYSTEM`/`AGENT_SESSION` — **not ported** (agent out of scope). |
| `ds4.c:67954-67978` | `q4e_payload_state` — **GDN conv window + full recurrent matrix per GDN layer**, plus PLE conv + pending draft row. Sizes at `:67869-67877`. |
| `ds4.c:57062-57063,58812-58823` | generic payload writes `layer_kda_conv_state` + `layer_kda_recurrent_state` for the GLM-5.3 delta-rule layers — second family, same conclusion. |
| `ds4.c:67879-67893` | q4e header u32 fields incl. `gdn_layers`, `gdn_conv_elems`, `gdn_state_elems` — geometry is pinned in the file. |
| `ds4.c:68061-68115` | `q4e_payload_check`: per-field geometry compare; `gdn_layers` mismatch → reject (`:68078`); `gdn_state_elems` mismatch → reject (`:68082-68084`); `pos < ctx_size` (`:68100-68103`). |
| `ds4.c:68035,68168-68169,68266` | payload carries the frontier logits row; loader sets `g->logits_pos = tokens` ("the file carried the path's final row") → **no decode step re-run on resume**. |
| `ds4.c:73585-73594` | `ds4_session_frontier_logits_current` — refuse to store when the frontier row isn't the frontier's own. |
| `ds4.c:509-513,66444-66447` | `model_id` = compile-time shape variant (`DS4_VARIANT_QWEN38F = 4`), *not* a file hash. |
| `ds4.c:58350-58359` | `ds4_engine_routed_quant_bits`: first FFN gate tensor → `Q4_K|MXFP4 → 4`, else `2`. |
| `ds4_kvstore.c:653-665` | `ds4_kvstore_render_tokens_text` — ds4 *detokenizes* the stored prefix to get the hashed text. shisu cannot (no tokenizer in the engine) → Decision 5. |
| `ds4_server.c:3407,5458` | `ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt)` — the hashed text is the **rendered ChatML string**, not the raw client messages (`:11712` hashes prefixes of that same `prompt_text`). |
| `ds4_server.c:16370-16374` | `if (cfg.kv_disk_dir) kv_cache_open(...)` — checkpointing is **off unless a dir is given**; no `DS4_KVSTORE_DIR` env exists anywhere in ds4 (verified by grep). |
| `ds4_kvstore.c:33`, `ds4_kvstore.h:26` | `KV_CACHE_DEFAULT_MIN_TOKENS 512`, `DS4_KVSTORE_DEFAULT_MB 4096`. |

## Decisions

1. **GDN state IS in the payload — no gating.** The assignment's hazard ("if ds4's payload is KV-only, resuming a hybrid model restores wrong conv/recurrent state") does not hold: ds4's q4e payload serializes the GDN conv window and the full recurrent matrix per GDN layer (`ds4.c:67954-67978`, sized `:67869-67877`), and the GLM-5.3 family does the same for its KDA delta-rule state (`ds4.c:57062-57063,58812-58823`). No ds4 family resumes from KV alone. qwen35's GDN state is fixed-size (24 × (96 KiB conv + 2 MiB state) ≈ 50.3 MiB), so it is always stored; the geometry header pins `gdn_layers`/element counts so a differently-shaped model can never load the file (`ds4.c:68078,68082-68084`).
2. **Payload is a shisu layout, not a ds4 byte-copy.** The 48-byte *file header* is ds4's verbatim (T7 owns it). The payload region is engine-owned (master plan: "engine-owned serialization"). Layout (all LE, documented at the top of `kv_payload.rs`):
   `u32[13]` header: `0 magic 0x50564B53 ("SKVP")`, `1 version = 1`, `2 pos`, `3 n_layers`, `4 n_attn_layers`, `5 n_gdn_layers`, `6 vocab`, `7 ctx_cap`, `8 kv_row_bytes` (= `kv_heads*head_dim*2`, f16), `9 gdn_conv_elems` (= `(conv_kernel-1)*gdn_in_channels`), `10 gdn_state_elems` (= `v_heads*head_dim*head_dim`), `11 weight_dtype` (`DType::gguf_id` of the layer-0 FFN gate), `12 kv_dtype` (f16 = 1) → `tokens[pos]: u32` → `logits[vocab]: f32` → per GDN layer `conv: f32[conv_elems]`, `state: f32[state_elems]` → per attn layer `K rows [0,pos): f16`, `V rows [0,pos): f16`.
   State before KV mirrors ds4 (`ds4.c:68039-68044`). Only rows `[0,pos)` are written: attention reads `[0, n_written)` only, so rows ≥ pos are never observed. Sizes at ctx 4096: 128 MiB KV + 50.3 MiB GDN + 0.95 MiB logits ≈ 179 MiB/checkpoint (T7's 4 GiB default budget ≈ 22 full-ctx entries).
3. **Frontier logits travel in the payload; no decode-step re-run.** See `ds4.c:68035,68266`. `kv_save` therefore *takes* the logits `DeviceView` the caller just got from `prefill_chunk(is_last=true)`/`decode_batch` — passing it is the structural port of ds4's `frontier_logits_current` gate (`ds4.c:73585-73594`, enforced at `ds4_server.c:11476-11482`): a caller mid-prefill has no view to hand over. `kv_resume` restores it into the session's `logits_row` and returns it in `ResumeInfo`.
4. **`model_id` / `quant_bits` are coarse pre-filters, not identity.** `model_id`: `Qwen35 → 5`, `Qwen4Exp → 4` (ds4's ids 0..=4 are shape variants, `ds4.c:509-513`; 5 keeps a shared cache dir from ever meaning the same thing as a ds4 file). `quant_bits`: ds4's rule verbatim on the layer-0 FFN gate tensor — `Q4_K | MXFP4 → 4`, everything else `→ 2` (`ds4.c:58350-58359`); the header validity gate `quant_bits ∈ {2,4}` (T7) is what makes a 0-valued (unquantized/unknown) build refuse to checkpoint. The *exact* weight dtype is pinned by payload field 11, which is what actually stops a q4_k ↔ q8_0 cross-load (ds4 relies on the same mechanism, `q4e_payload_check` `ds4.c:68061-68115`).
5. **Text boundary: the engine hashes the bytes it is handed.** ds4 derives the key by detokenizing the stored prefix (`ds4_kvstore.c:653-665`) of the *rendered ChatML prompt* (`ds4_server.c:3407,5458,11712`); the responses-surface `cache_text_override` variants (`ds4_server.c:11544-11578`) are server policy. shisu's engine has no tokenizer/detokenizer, so `kv_save` receives `prompt_text: &[u8]` and `kv_resume` receives the same bytes; T10/T13 own the derivation (rendered ChatML from `tokenizer/chat_render.rs`) and must hand the *same* derivation on both sides. A store whose text was derived differently simply never matches → cold start (safe, no false hit). The engine additionally verifies `hit.text` is a byte prefix of the request text before trusting a hit (`ds4_kvstore.c:1315-1319`).
6. **Env: T8 adds none.** ds4 has no `DS4_KVSTORE_DIR` (grep: zero hits); the server flag `--kv-disk-dir` defaults to unset = disabled (`ds4_server.c:16370-16374`). `SHISU_KVSTORE_DIR` is reserved for T10 as the env fallback of `--kvstore-dir`, default unset = checkpointing OFF; budget comes from T7's default (4096 MB) unless T10 passes one. `KvCheckpoint` exists only inside `SessionStore` as `Option<..>`, `None` = disabled.
7. **Store-length policy is NOT ported.** ds4's `--kv-cache-boundary-align/-boundary-trim/-cold-max-tokens/-continued-interval` decide *how long a prefix to store*; that is server/scheduler policy (T13). T8 ports the mechanism + the intrinsic `min_tokens` gate (`ds4_kvstore.c:971`) only.
8. **`Checkpointed: Model` supertrait.** T2's `Model` is frozen SSOT and T6's `Qwen35Model` is `Frozen`; the checkpoint entry points must be reachable through `Box<dyn Model>` (T10's scheduler holds exactly that) without touching either. A `pub trait Checkpointed: Model` with default-method-free methods, implemented in `kv_ckpt.rs`, is the only cutover that adds no `Model` method and no shim. T10 downcasts `Box<dyn Model>` → `&mut dyn Checkpointed` (trait upcasting `&mut dyn Checkpointed → &mut dyn Model` verified compiling on this toolchain, rustc 1.93.1).

## Plan

1. [ ] `shisu-engine/Cargo.toml`: add `shisu-kvstore`, `sha1`, `hex` from `[workspace.dependencies]`.
2. [ ] `shisu-kvstore/src/store.rs`: add `pub fn remove(&self, prefix_sha1: &str) -> Result<bool>` (unlink the sha-named file; `Ok(false)` when absent; never touches the in-memory index in a way that leaves a stale entry). One unit test: put → remove → `get` is `None`, remove again → `Ok(false)`.
3. [ ] `kv_payload.rs` (new): the 13-field header consts + `write_payload(...)` / `read_payload(bytes, &Expect) -> Result<PayloadParts, Reject>` where `Reject { AsConfigured, NeverReadable }` maps ds4's rc 1 / rc 2 (`ds4.c:68061-68115`): `AsConfigured` = version/geometry matches *this* build but a field disagrees with the running config (keep the file); `NeverReadable` = magic/version/shape the build can never read (unlink it, `ds4_kvstore.c:1356-1369`). Pure byte math, no device access — unit-testable alone.
4. [ ] `kv_ckpt.rs` (new): `KvCheckpoint` struct + `CkptOptions` + consts (Decision 4) + the prefix scan:
   `fn find_prefix(&self, prompt_text: &[u8]) -> Option<KvEntry>` over `store.report().entries`, filtering `text_bytes <= prompt_text.len()`, `tokens >= min_tokens`, `model_id == self.model_id`, `ctx_size <= self.ctx_size`, and (when `reject_different_quant`) `quant_bits == self.quant_bits`; then `sha1(prompt_text[..text_bytes]) == entry.sha`; keep the largest `text_bytes`, tie-break more tokens (`ds4_kvstore.c:1245-1260`). Note in a comment that this re-hashes candidate prefixes (O(n) sha1 over ≤ ctx-sized buffers) and that T13 may memoize per request.
5. [ ] `kv_ckpt.rs`: `kv_save` — gates in ds4's order: disabled → `Ok(false)`; `tokens.len() != session.n_written` → `Ok(false)` (ds4's live-frontier check, `ds4_kvstore.c:986-988`); `tokens.len() < min_tokens` → `Ok(false)`; `prompt_text.is_empty()` → `Ok(false)`. Then d2h the buffers (one `copy_d2h` per buffer, no intermediate copies beyond the payload Vec), serialize via `kv_payload`, `sha1(prompt_text)`, `store.put(sha, prompt_text, HeaderParts{ model_id, quant_bits, tokens: pos, ctx_size }, reason, ext_flags, payload)` (T7's signature incl. its `HeaderParts` fix). On `put` returning `false` (T7's compat short-circuit) → `remove(sha)` then re-`put` once (ds4's unlink-and-rewrite, `ds4_server.c:883-895`). Return `wrote_file`.
6. [ ] `kv_ckpt.rs`: `kv_resume` — `find_prefix` → `store.get(sha)` (this is what bumps `hits`, T7) → verify `hit.text` byte-prefix → parse payload → geometry check → on `NeverReadable` `store.remove(sha)` and return `Ok(None)` → restore: `copy_h2d` each GDN conv/state buffer, each attn K/V rows `[0,pos)`, the logits row; set `session.n_written = pos` → `Ok(Some(ResumeInfo{ tokens_reused: pos, text_bytes, tokens, frontier_logits }))`.
7. [ ] `sessions.rs`: add `pub(crate) ckpt: Option<KvCheckpoint>` to `SessionStore`, `pub(crate) fn resident_ids(&self) -> impl Iterator<Item = SessionId> + '_`, and the `CloseReason → Reason` mapping doc-comment table (cold / continued / evict / shutdown with the ds4 line refs from the Source table) so T13 wires the call sites, not the decisions.
8. [ ] `kv_ckpt.rs`: `impl Checkpointed for Qwen35Model` — `kv_enable` derives `model_id`/`quant_bits`/`ctx_size` from the loaded `ShapeProfile` + bound FFN gate dtype and stores the `KvCheckpoint` in `self.sessions.ckpt`; `kv_shutdown_sweep` iterates `resident_ids()`, asks the caller-supplied `text_of` closure for each session's `(tokens, text)`, and calls `kv_save(.., Reason::Shutdown, 0)` — mirrors `ds4_server.c:16540-16548` (skip `tokens < min_tokens`, count writes). If a `Qwen35Model` field needed here is private, add a `pub(crate)` accessor in `qwen35.rs` and note it in the commit body.
9. [ ] `kv.rs`: add only the accessors `kv_ckpt` needs that don't exist (`layers()` → `&[(DeviceBuffer, DeviceBuffer)]`, `row_bytes()`); do not restructure the page-ready layout.
10. [ ] `shisu-bench/src/bin/kv_resume.rs` (new, manual/`#[ignore]`-style binary): cold prefill vs resume of the same prompt, writes `shisu-bench/baseline/m1max-qwen35-resume.json` `{ machine, model, ctx, prompt_tokens, cold_prefill_ms, resume_ms, tokens_reused, continuation_tokens_equal }`. T20 owns the gate; this only records.
11. [ ] Grep gate: `grep -rn "SHISU_KVSTORE_DIR" shisu/crates/shisu-engine` empty (Decision 6); `grep -rn "DS4_" shisu/crates/shisu-engine/src/kv_ckpt.rs` hits only inside `///`/`//` comments.

## Tests

Unit tests live **inside the crate** (`src/kv_ckpt/tests.rs`, `mod` declared in `kv_ckpt.rs`) because `Session`/`SessionKv` are `pub(crate)` (T6) — an integration test in `tests/` cannot build one. Fixtures: T6's `FakeBackend` pattern (HashMap-backed `Backend`, `copy_d2h`/`copy_h2d` are memcpy) + a hand-built `Session` with 8 attn K/V buffers, 24 conv + 24 state buffers, one logits row, filled from a seeded LCG so byte-equality is meaningful. Store dir via `tempfile::TempDir` (dev-dep).

- [ ] `round_trip_cold`: save(`Reason::Cold`) → fresh session → `kv_resume` → `Some`; `tokens_reused == pos`, `text_bytes == prompt.len()`, `tokens ==` saved ids; every KV/GDN/logits buffer of the resumed session `copy_d2h`-compares **byte-equal** to the source; `store.report().entries[0].hits == 1` (bumped by `get`).
- [ ] `partial_prefix_and_longest_wins`: store prefix A (short) and prefix B = A + more (long); resume with B + " tail" → `tokens_reused == B.pos` (longest wins, `ds4_kvstore.c:1259-1260`); resume with text shorter than A → `None`.
- [ ] `reject_compat`: entries written with `model_id = 4`, `quant_bits = 2`, `ctx_size = 8192` → `kv_resume` → `None`; a payload whose `gdn_layers`/`gdn_state_elems` disagree with the build → `None` **and** the file is gone (`NeverReadable` unlink).
- [ ] `gdn_state_travels`: after resume, conv windows and recurrent matrices are byte-equal (this is the GDN regression test for Decision 1 — it fails loudly if anyone drops the state from the payload).
- [ ] `gates`: `tokens.len() != n_written`, `tokens.len() < min_tokens`, empty text → `Ok(false)` and no file created.
- [ ] `#[ignore]` e2e (macOS, `SHISU_TEST_MODEL=…/Qwen3.5-4B-Q4_K_M.gguf`): greedy 64-token continuation after a save → fresh `Qwen35Model` + fresh session + `kv_resume` of the same prompt → **continuation token ids identical to the cold run** (master-plan checkbox; byte-exact, no tolerance — the restored state is a memcpy and sampling is deterministic), plus `resume_ms < cold_prefill_ms` recorded (not asserted) into the resume baseline JSON.
- Run: `cargo test -p shisu-engine kv_ckpt` · `cargo test -p shisu-kvstore remove` · `cargo test -p shisu-engine -- --ignored kv_resume_e2e` · `cargo clippy -p shisu-engine -p shisu-kvstore --all-targets`.

## Acceptance

- [ ] Checkpoint = rendered-prompt-prefix sha1 → KV **+ GDN state** snapshot; payload layout (which buffers, rows `[0,pos)`, sizes, offsets) documented at the top of `kv_ckpt.rs`/`kv_payload.rs` with the ds4 line refs.
- [ ] Resume restores KV rows `[0,pos)`, GDN conv + recurrent state, the frontier logits row, and `n_written`; **no decode step is re-run** (evidence: `ds4.c:68266`).
- [ ] Engine-side compatibility policy: model/quant/ctx mismatch rejected at scan time; never-readable payload unlinked (T7 Deviation 8 closed).
- [ ] `kv_shutdown_sweep` exists and mirrors `ds4_server.c:16540-16548`; reason mapping documented for T13 (cold/continued/evict/shutdown; agent reasons explicitly not ported).
- [ ] Checkpointing default OFF (`ckpt: None` until `kv_enable`); engine reads no env.
- [ ] `cargo test -p shisu-engine` and `-p shisu-kvstore` green on macOS; ignored e2e green with `SHISU_TEST_MODEL`; `shisu-bench/baseline/m1max-qwen35-resume.json` produced once and committed.
- [ ] No file > 1500 LoC (`kv_ckpt.rs` + `kv_payload.rs` split holds); no `ATLAS_*`; `DS4_*` only in comments.

## Commit

```
git add shisu/crates/shisu-engine shisu/crates/shisu-kvstore/src/store.rs shisu/crates/shisu-kvstore/tests shisu/crates/shisu-bench shisu/Cargo.toml shisu/Cargo.lock shisu-bench/baseline/m1max-qwen35-resume.json
git commit -m "feat: engine KV checkpoint integration (ds4 kvstore orchestration port)"
```

(Master plan Task 8 says "Commit." with no message; message follows the Task 1 `type: summary` convention.)
