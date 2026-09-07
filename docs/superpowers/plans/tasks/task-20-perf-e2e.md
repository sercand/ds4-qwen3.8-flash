# Task 20: Perf gates + end-to-end

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 20 (292-298)
+ § Rules (30-37; Rule 4 at 34, Rule 5 knob list 35) + § CuMetal dev lane (39-74; hard-limits
table 57-70) + crate map/SSOT (76-126) + Roadmap Wave 4 (302-319; T20 row) + verification log
(323-331). Read all six before executing. This is the LAST task: the final quality gate for the
whole workspace — it verifies gates, it does not re-own any of them.

**Depends on:**
- T1 (`task-01-workspace-scaffold.md` 23-29, 38, 63-67): local gate `scripts/check.sh` (exports
  `SHISU_SKIP_BUILD=1`; runs check/test/clippy --workspace + `cargo deny check` + file-size cap)
  + `scripts/file_size_check.sh` (1500-LoC); `deny.toml` (incl. RUSTSEC-2024-0436 ignore);
  `rust-toolchain.toml` 1.93.1; `[workspace.lints]`. Consumed, not rebuilt.
- T6 (`task-06-engine-qwen35.md` 116, 132): `shisu-bench/src/bin/qwen35.rs` +
  `baseline/m1max-qwen35.json` v1 fields (`machine, os, date, git_rev, model, backend, ctx,
  prefill_tokens, prefill_tokens_per_sec, decode_tokens, decode_tokens_per_sec,
  peak_rss_bytes`); "T6 records; T20 gates". The greedy-64 lock stays T6/T18's.
- T8 (`task-08-checkpoint-integration.md` 97, 109, 119): `shisu-bench/src/bin/kv_resume.rs` +
  `baseline/m1max-qwen35-resume.json` `{machine, model, ctx, prompt_tokens, cold_prefill_ms,
  resume_ms, tokens_reused, continuation_tokens_equal}`; "T20 owns the gate; this only records".
- T10 (`task-10-server-skeleton.md` 112, 229-230, 283-286): route table, the four POST routes,
  error-shape assertions (OpenAI `.error.code`, Anthropic top-level `type == "error"`, `/cache`,
  405, OPTIONS 204 + allow-methods). T11 (190-203, 298-300): chat/responses/completions wire
  shapes, SSE `data:` framing + `[DONE]`, its own per-endpoint live e2e. T12 (78-91, 135):
  Anthropic event table (`message_start` … `message_stop`), its own live e2e. T13 (36-50): the
  full-stack `serve()` contract (coordinator + `spawn_engine_with_checkpoints`).
- T14 (`task-14-kernel-build-crate.md` 156-175, 210-231): `kernels/cumetal_compat/STATUS.md`
  rows, knobs `SHISU_SKIP_BUILD` / `SHISU_CUDA_COMPILE` / `SHISU_CUMETAL_STRICT`, and the Linux
  nvcc runbook (its Step 10; GitHub Actions out of scope). T15 (249-253): STATUS row
  format + the 6 NUMERIC-INVALID compound-smem q4e kernels. T3b/T5/T17: the oracle twins the
  lane gate re-runs (`oracle_gdn` harness, MSL twins, `tests/q4e_cpu_oracle.rs` at 1e-5).
- T18 (`task-18-golden-parity.md` 69-71, 259): parity is T18's; T20 owns the lane gate + the
  project-wide gates. T19 (`task-19-ple-streaming.md` 97-98, 331): PLE stats + bounded wait
  histogram recorded as OBSERVATION; "T20 gates it" — correctness stays T19's.
- atlas (T1's provenance; mechanism source only — shisu has no GHA): `.benchmarks/<gate>/
  <date>-<treehash>.json` record layout + `pr-benchmark-gate` (ci.yml 773-852: "benchmark records
  can only be produced on the GPU machine" 775-777; the gate verifies committed records, never
  re-measures) — adapted into `perf_gate check` + `scripts/check.sh`; cargo-deny command
  (security.yml 40-55: `check advisories licenses sources bans`) — already a check.sh step (T1).

**Produces** (all under `shisu/`; LoC budgets vs T1's 1500 cap):
- `crates/shisu-bench/src/measure.rs` (≤150, additive — see DEVIATION 3 note): the shared
  timing harness — outer repetitions, warmup, first-step drop, median, spread. One statistic
  implementation for every gate.
- `crates/shisu-bench/src/decode.rs` (≤220): decode tok/s — greedy steps with EOS excluded
  (ds4_bench.c:901-903 semantics), per-step ms, step 1 dropped, median-of-steps × 7 outer reps.
- `crates/shisu-bench/src/prefill.rs` (≤180): prefill tok/s at fixed token counts — 1 warmup +
  5 timed prefills, median (metal_prefill_variant_bench.c:360-456 repeat pattern).
- `crates/shisu-bench/src/kv.rs` (≤150): resume-latency gate — reads T8's
  `m1max-qwen35-resume.json`, gates `resume_ms`, asserts invariant `resume_ms <
  cold_prefill_ms`; measures nothing itself (T8's `kv_resume` bin is the recorder).
- `crates/shisu-bench/src/ple.rs` (≤200): PLE fetch-latency distribution — percentiles
  (p50/p99/p999) from T19's bounded wait histogram, ds4-comparable aggregates (mean wait ms,
  max wait ms, hit %, read mean ms), gate math per DEVIATION 1. Percentile math is pure and
  unit-tested on macOS with synthetic histograms.
- `crates/shisu-bench/src/baseline.rs` (≤280): baseline/record schemas, read/compare/report,
  tree-coverage rule, exit codes. `--check` is arithmetic on committed JSON only — never
  measures, never writes.
- `crates/shisu-bench/src/bin/perf_gate.rs` (≤280, new): the single CLI —
  `perf_gate record --target metal|cuda|resume [--model $SHISU_TEST_MODEL]` (pinned machines
  only) and `perf_gate check` (gate: no model, no GPU). `bin/qwen35.rs` (T6) is rewritten as a
  thin wrapper over the lib so exactly one measurement implementation exists (clean cutover;
  its CLI surface stays so T6's child-process memory test keeps working).
- `baseline/m1max-qwen35.json` re-recorded as **v2**: all v1 field names kept (values become
  the new protocol's medians, so v1 readers still work) + `schema_version: 2`, `protocol`
  (warmup/steps/reps/statistic), `metrics.decode_ms_step_median`, `metrics.prefill_ms_median`,
  `spread.{decode,prefill}`. `baseline/ds4-qwen4exp.json` (new, DEVIATION 4): ds4-side CUDA
  reference + provenance. `baseline/m1max-qwen35-resume.json` stays T8's file, consumed as-is.
- `records/` (new dir): measurement records `<target>-<yyyymmdd>-<rev12>.json` (atlas
  `.benchmarks/` pattern), committed alongside perf-touching changes.
- `crates/shisu-bench/tests/baseline_logic.rs` (≤250, always-on): schema round-trip, ratio
  gate, coverage rule, spread rejection, percentile correctness — no model, no GPU.
- `crates/shisu-kernels/tests/cumetal_lane_gate.rs` (≤200, `#[ignore]`, macOS) +
  `scripts/cumetal_lane_gate.sh`: the lane gate (Step 7).
- `crates/shisu-server/tests/e2e.rs` (≤460, `#[ignore]`): full-stack boot, four endpoints
  blocking+streaming, misc routes, over real loopback HTTP (Step 8).
- One additive hunk in T1's `scripts/check.sh` (Step 10): the `perf_gate check` step. `cargo
  deny check` is already there (T1). No other gate edits; no `.github/` files exist.

## Global Constraints

1. **Rule 4 is absolute:** no throughput/timing number produced by the CuMetal lane is recorded
   or compared anywhere in this task (master 34, 295; task-18:69-71). `clock()` is synthetic and
   cuBLASLt can silently fall back to CPU (master 57-70). The lane gate is compile + numerical.
2. **The gate never measures.** Measurement happens only on pinned machines (M1 Max dev box;
   Wave 3b Linux+NVIDIA box). `scripts/check.sh` verifies committed records (atlas ci.yml
   773-852 mechanism, adapted). This is what makes the 5% gate honest — DEVIATION 3.
3. **Baselines are recorded-then-compared, never auto-overwritten.** `check` never writes;
   `record` writes into `records/`; promoting a record into `baseline/` is a human copy whose
   commit message cites old→new numbers.
4. Constants appear once, each with a source: `GATE_RATIO = 1.05` (master 294), `TWIN_TOL = 1e-5`
   (master 295; T3b/T17 oracles), `SPREAD_MAX = 0.02` (this plan's noise policy, DEVIATION 3).
   No tolerance is ever edited to make a gate pass (Rule 3 culture, task-18:256-258).
5. **Zero new env knobs** (Rule 5): the CLI flags above carry everything; existing knobs
   (`SHISU_TEST_MODEL`, `SHISU_SKIP_BUILD`, `SHISU_CUDA_COMPILE`, `SHISU_CUMETAL_STRICT`,
   `SHISU_PLE_IO`) are consumed as owned. No `DS4_`/`ATLAS_` reads.
6. `warnings = deny` + clippy deny (T1 lints); `anyhow` allowed inside `shisu-bench` bins only
   (bin crate, atlas precedent); `tracing` for run lifecycle; no license headers; no file >1500.
7. **Not re-done here** (owned elsewhere, verify-only): T1's lint/deny/file-size/toolchain
   wiring; T6's greedy-64 lock; T14's build knobs; T15's PTX body-hash oracle; T18's parity
   (token/logit correctness — zero throughput numbers there by design); T19's PLE codec/cache
   correctness. T20 adds gates and e2e, nothing else.

## Source References (verified)

| Source | Range | What it pins |
|---|---|---|
| `ds4_bench.c` | 7-15 | Methodology: fixed token sequence walked to frontier points; only the newest prefill interval timed; **snapshot save/restore intentionally outside both timing windows** |
| `ds4_bench.c` | 219-225 | Defaults: `ctx_start 2048, ctx_max 32768, step_incr 2048, gen_tokens 128` |
| `ds4_bench.c` | 290-303, 372-397 | Options `--ctx-start/--ctx-max/--step-incr/--gen-tokens/--csv/--teacher-forced-decode` (mutually exclusive with `--dspark`) |
| `ds4_bench.c` | 741-748, 901-903 | Teacher-forced pulls prompt tokens; default decode = `ds4_session_argmax_excluding(eos)` |
| `ds4_bench.c` | 782, 983-1017 | CSV header `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes`; row written after restore (outside windows) |
| `tests/bench_qwen4exp_ctx.c` | 1-19, 41-53 | First step (graph capture) excluded; `summarize` = **median** + mean of steady-state steps |
| `tests/bench_qwen4exp_ctx.c` | 55-64, 102-144 | SM clock recorded per run via `nvidia-smi`; `DS4_BENCH_CTXS`/`DS4_BENCH_STEPS` env; per-context table columns |
| `tests/test_qwen4exp_graph.c` | 224, 277, 334 | decode / speculative / prefill tok/s print lines |
| `tests/test_qwen4exp_exl3.c` | 174, 216, 259 | prefill / greedy / spec tok/s print lines |
| `speed-bench/metal_decode_schedule_bench.c` | 457-521, 556 | `cfg.warmup` steps before counting; `seconds=%.6f tokens_per_second=%.4f` |
| `speed-bench/metal_prefill_variant_bench.c` | 360-456, 441 | `repeats` × interleaved slots, exact-match logits gate across variants, per-run + aggregate lines |
| `ds4_ple_stream.c` | 196-211, 937-944, 985-996 | Per-worker cumulative `read_seconds`/`err`, merged across readers; **no percentile anywhere** |
| `ds4.c` | 67395-67420 | PLE stats dump: waits, **mean** ms each, **max** ms, total s, hit %, reads mean ms — mean+max only |
| grep (this task) | — | `p99|percentile|quantile` over `ds4_ple_stream.c`, `ds4.c`, `ds4_bench.c`, `tests/bench_qwen4exp_ctx.c`: **zero hits** |
| atlas `.github/workflows/ci.yml` | 50-75, 193-194, 773-852 | Mechanism source for the local gate (no GHA in shisu): the fmt/clippy command sequence; `pr-benchmark-gate`: records committed from the GPU machine, the gate checks `record_still_stands` + tree coverage, never re-measures |
| atlas `.github/workflows/security.yml` | 40-55 | `check advisories licenses sources bans` — the cargo-deny command now run by `scripts/check.sh` (T1) |
| `task-01` | 23-29, 38, 63-67 | `scripts/check.sh` (`SHISU_SKIP_BUILD=1`), 1500-LoC `scripts/file_size_check.sh`, `deny.toml`, toolchain 1.93.1 |
| `task-06` / `task-08` | 116, 132 / 97, 109, 119 | v1 baseline fields; resume JSON fields; both "record only, T20 gates" |
| `task-10` / `task-11` / `task-12` / `task-13` | 112, 229-230, 283-286 / 190-203, 298-300 / 78-91, 135 / 36-50 | Routes + error shapes; OpenAI+SSE shapes; Anthropic event table; full-stack `serve()` |
| `task-14` / `task-15` / `task-18` / `task-19` | 156-175, 210-231 / 249-253 / 69-71 / 97-98, 331 | Lane knobs + STATUS.md; NUMERIC-INVALID rows; Rule 4 + lane-gate ownership; p99 recorded, T20 gates |

⚠ **DEVIATION 1 — "PLE p99 ≤ ds4 stats" is not computable as written.** ds4 exposes no
percentile: `ds4_ple_stream.c` carries only cumulative `read_seconds` merged across workers
(196-211, 985-996) and `ds4.c:67395-67420` prints mean + max waits only; grep for
`p99|percentile|quantile` across all four ds4 perf files: zero hits. The gate becomes: (a)
shisu's **own p99** (from T19's bounded wait histogram) is gated against a **shisu-recorded**
baseline (`ple_wait_p99_ms ≤ baseline × 1.05`), first recorded on the Wave 3b box; (b) the
**ds4 comparison** uses the metrics ds4 can actually emit — mean wait ms, max wait ms, hit %,
read mean ms — stored in the record's `ds4_reference` block and compared on mean only (max is
single-sample noise; recorded, never gated). Master's "≤ ds4 stats" is honored on the mean; the
p99 half gates against shisu's own history.

⚠ **DEVIATION 2 — there is no ds4 resume-latency number.** `ds4_bench.c:14-15` states snapshot
save/restore is *intentionally outside both timing windows* (save 843-868, restore 983-993), and
ds4's snapshot is an in-memory blob, not the disk checkpoint path shisu gates. The resume gate is
purely shisu-side: cold prefill vs resume-from-checkpoint on the same prompt (T8's
`kv_resume.rs` protocol), `resume_ms ≤ baseline resume_ms × 1.05` plus the sanity invariant
`resume_ms < cold_prefill_ms` (T8:109 records it un-asserted; T20 asserts it). Baseline file:
T8's `m1max-qwen35-resume.json`, recorded once by T8, re-recorded only by explicit human run.

⚠ **DEVIATION 3 — the 5% perf gate is a record-verification gate, not a runner measurement.**
A 5% threshold on one sample from a shared machine is not reproducible. Design (atlas
ci.yml 773-852, adapted): measurement runs **only** on pinned machines; the bench writes a
record JSON; the record is committed; the local gate (`scripts/check.sh`) runs `perf_gate check`, which
(a) requires a record whose `git_rev` is an ancestor of HEAD and whose diff to HEAD touches no
`shisu/crates/`, `shisu/kernels/`, `shisu/Cargo.*`, `shisu/rust-toolchain.toml` path
(`record_still_stands`), and (b) recomputes every gated ratio from the committed JSON. Exit 0
pass / 1 regression >5% / 2 stale-or-missing record ⇒ re-record required. The measurement
protocol (in `measure.rs`, mirroring ds4's own stabilizers): warmup 1 prefill + drop decode
step 1 (graph/state warmup — `bench_qwen4exp_ctx.c:41-53` drops exactly the first step);
statistic = **median** of per-step ms over 128 steps (`summarize` precedent) and of 5 timed
prefills (`metal_prefill_variant_bench.c` repeats pattern; `metal_decode_schedule_bench.c`
warmup gate); **7 outer repetitions**, reported value = median across reps; `spread` =
max/min − 1 across reps must be ≤ 2% or `record` **refuses to write** and exits nonzero. Noise
policy: a noisy pinned machine fails the spread check → quiesce and re-run; the threshold is
never loosened and best-of-N is never taken (best-of-N biases the gate toward fast noise and
hides regressions). Manual recheck path: re-run `record`, commit the record with its spread;
the commit message cites old→new numbers.

⚠ **DEVIATION 4 — `ds4-qwen4exp.json` provenance is named exactly; it cannot be produced on
this Mac** `[INFERENCE]` (no NVIDIA GPU, no q4e GGUF here). Recorded on the Wave 3b box from
the ds4 repo root, two commands, both results merged into one JSON:
`./ds4-bench --cuda -m $Q4E --prompt-file p.txt --csv ds4-q4e.csv` (defaults 2048→32768 step
2048, gen 128, greedy-with-EOS-excluded — the canonical frontier CSV, header at
`ds4_bench.c:782`), and `DS4_BENCH_CTXS=2048,8192,16384,32768 DS4_BENCH_STEPS=40
./bench_qwen4exp_ctx` (median ms/step per context — the stable statistic). Fields:
`{machine, gpu, driver, cuda_version, sm_clock_mhz_at_record, git_rev, model, model_bytes,
backend: "cuda", frontiers: [{ctx, prefill_tps, gen_tps, gen_steady_ms_median,
gen_first_ms, kvcache_bytes}], ple_reference: {mean_wait_ms, max_wait_ms, hit_pct,
read_mean_ms}}` (SM-clock provenance = `bench_qwen4exp_ctx.c:55-64` precedent;
`ple_reference` from the ds4 stats dump, DEVIATION 1(b)). The shisu CUDA gate compares
`decode_ms_step_median` and prefill tok/s per frontier against this file at the same ctx
points; `gen_first_ms` is recorded for context, not gated.

⚠ **DEVIATION 5 — e2e does not duplicate T11/T12's per-endpoint live tests.** T11 already runs
blocking+stream × chat/responses/completions; T12 runs × messages. `e2e.rs` is the distinct
claim: **one** boot of the **full T13 stack** (coordinator + `spawn_engine_with_checkpoints` +
`build_router`, not T10's naive loop) on `127.0.0.1:0` (bind port 0, read back
`local_addr()`), then all four endpoints in one session plus cross-endpoint interleaving over
real HTTP, plus the misc routes T10 only covered via oneshot. Schema validation is hand-rolled
key-set/type/vocabulary assertions over `serde_json::Value` parsed from raw socket bytes — no
JSON-schema crate, no new dep. **Not a tautology:** the asserted key sets, id prefixes, and
vocabulary strings are transcribed from ds4_server.c's actual writers and the published
OpenAI/Anthropic shapes cited in T11 (190-203) and T12 (78-91) — an external contract checked
against live bytes, never shisu's own structs re-serialized back at itself. `/v1/models` and
`/cache` DO belong in the set: one GET each (cheap router regression coverage over a real
socket).

⚠ **DEVIATION 6 — GitHub Actions is out of scope (master Scope-Out); the gate is local.**
T1's `scripts/check.sh` already runs `cargo deny check` (atlas security.yml 40-55 command
adapted: `check advisories licenses sources bans`). T20's only gate edit is one additive hunk
in `check.sh` (Step 10): `perf_gate check` (arithmetic only, no model, no GPU). The lane gate
is NOT part of the default gate: the lane is opt-in and never gates a default build (master
249; cumetalc is not on every machine) — it runs on the pinned Mac whenever
`SHISU_CUDA_COMPILE=1` via `scripts/cumetal_lane_gate.sh`.

## Plan

- [ ] **Step 0 — Preflight (read-only).** Confirm present: `shisu-bench/src/bin/qwen35.rs` +
  `baseline/m1max-qwen35.json` (T6), `bin/kv_resume.rs` + `baseline/m1max-qwen35-resume.json`
  (T8), `kernels/cumetal_compat/STATUS.md` + the T14 Linux runbook,
  `shisu-bench/src/lib.rs` + `vecfile.rs` (T18), T19's wait-counter/histogram surface on
  `PleStats`, `shisu-kernels` metallib module registry (T14). If any verdict in STATUS.md
  differs from what the machine says today, update STATUS.md first (T3b rule: the verdict is
  whatever the machine says). Record `git_rev` of the tree being gated.
- [ ] **Step 1 — `measure.rs` + `baseline.rs` + always-on tests.** `measure.rs`: `RepLoop`
  (outer reps, warmup, first-drop, median, spread) — the only place median/spread are
  computed. `baseline.rs`: v2 schema structs (v1 fields kept readable), record naming
  `<target>-<yyyymmdd>-<rev12>.json`, `record_still_stands` (ancestor check + path-scoped
  diff via `git diff --name-only <rev>..HEAD`), ratio gate with `GATE_RATIO`/`SPREAD_MAX`,
  exit codes 0/1/2. Tests in `tests/baseline_logic.rs` (no model/GPU): percentile correctness
  on synthetic histograms (odd/even counts, single sample, ties), ratio gate boundary at
  exactly 1.05 (pass) and 1.0501 (fail), coverage rule rejects a record whose rev is not an
  ancestor and accepts a docs-only diff, `check` never writes a file.
- [ ] **Step 2 — `decode.rs` + `prefill.rs`.** Decode: load `SHISU_TEST_MODEL`, fixed prompt
  (T6's 64-token prompt for the baseline ctx; T18's long-context prompt for the 4096 point),
  greedy steps with EOS excluded (ds4_bench.c:901-903), per-step `Instant` timing, protocol
  per DEVIATION 3. Prefill: 1 warmup + 5 timed prefills at 1024 tokens, median tok/s. Both
  return a `Metrics` struct; neither prints baseline paths (the bin decides).
- [ ] **Step 3 — `bin/perf_gate.rs` + `bin/qwen35.rs` cutover.** `perf_gate record --target
  metal|cuda|resume` (cuda arm returns a typed error on macOS, T16 stub rule; resume arm shells
  to nothing — it *reads* T8's JSON, measures nothing) writes a record; spread > 2% ⇒ refuse +
  exit nonzero. `perf_gate check` verifies every committed record against its baseline per
  DEVIATION 3. Rewrite `bin/qwen35.rs` as a thin wrapper (same CLI, same stdout tok/s line, so
  T6's `/usr/bin/time -l` child test survives) calling `decode.rs`/`prefill.rs`. Delete any
  duplicated timing code from the old bin.
- [ ] **Step 4 — Re-record the macOS baseline (pinned M1 Max).** Run `perf_gate record
  --target metal` twice; both spreads ≤ 2%; the two records agree within 1%. Promote the
  second into `baseline/m1max-qwen35.json` (v2) by hand; commit message cites old→new decode
  and prefill tok/s. `m1max-qwen35-resume.json` is NOT touched (T8's file, T8's numbers).
- [ ] **Step 5 — `kv.rs`.** Gate: `resume_ms ≤ baseline × 1.05` and `resume_ms <
  cold_prefill_ms`; `continuation_tokens_equal == true` must be present in the record (T8's
  byte-exactness, asserted here as a gate precondition, correctness owned by T8). No new
  measurement code.
- [ ] **Step 6 — `ple.rs`.** Percentiles from T19's bounded histogram (p50/p99/p999); record
  section `{ple_wait_mean_ms, ple_wait_p99_ms, ple_wait_max_ms, ple_hit_rate,
  ple_read_mean_ms, ds4_reference{...}}`; gate mean + p99 at 1.05 against the shisu baseline
  (first Wave 3b run records it), ds4 comparison on mean only (DEVIATION 1). Unit test:
  histogram → percentile edge cases (bucket-boundary rounding documented).
- [ ] **Step 7 — CuMetal lane gate (macOS, compile + numerical ONLY).**
  `scripts/cumetal_lane_gate.sh` = the last step of every lane run: (1)
  `SHISU_CUDA_IMPL=cumetal SHISU_CUDA_COMPILE=1 SHISU_CUMETAL_STRICT=1 cargo build -p
  shisu-kernels` in non-strict-permissive mode so the lane continues past failures; (2)
  `cumetal_lane_gate.rs` parses STATUS.md and asserts **set equality**: every row with verdict
  `ok` appears in `metallib_modules()`, and no row with verdict `SMEM_BLOCKED` /
  `BLOCKED` / `NUMERIC-INVALID` / excluded (cooperative grids, cuBLASLt — master 57-70; the 6
  compound-smem q4e kernels per task-15:249-253) is counted as a failure — they are expected
  absent, by construction, never by exception list; (3) re-run the existing oracle tests
  (`-- --ignored`: T3b `oracle_gdn` harness, T5 MSL-vs-CuMetal differentials, T17
  `q4e_cpu_oracle.rs` twins) at `TWIN_TOL = 1e-5`. The gate reads pass/fail and max-abs-diff
  only; **no CuMetal timing is read, recorded, or compared** (Rule 4). head_dim ≤ 64 PASS /
  D=128/256 SMEM_BLOCKED are lane artifacts, documented in STATUS.md, not failures.
- [ ] **Step 8 — `shisu-server/tests/e2e.rs`** (`#[ignore]`, macOS + `SHISU_TEST_MODEL` +
  `--features metal`). Boot the full T13 stack on `127.0.0.1:0`. Cases, each one blocking +
  one streaming: `/v1/chat/completions` (blocking: `id` prefix `chatcmpl-`, `object
  == "chat.completion"`, `choices[0].message.{role,content}` non-empty, `finish_reason` in the
  T11 vocabulary, `usage` three non-negative ints; stream: every frame `data: ` framed, chunk
  `object == "chat.completion.chunk"`, concatenated deltas == blocking content (greedy is
  deterministic — T6 lock), final `data: [DONE]`); `/v1/responses` (named events
  `response.created` … `response.completed`, monotonic `sequence_number`);
  `/v1/completions` (`object == "text_completion"`, `choices[0].text`, `usage`);
  `/v1/messages` (top-level `type == "message"`, content blocks, `stop_reason`; stream
  `message_start` → `content_block_delta`+ → `message_stop`, T12's table). Errors: one OpenAI
  4xx asserting `.error.code`, one Anthropic 4xx asserting top-level `type == "error"`
  (T10:283-286). Misc: `GET /v1/chat/completions` → 405 + `Allow`; `OPTIONS` → 204 +
  allow-methods; `GET /v1/models`; `GET /cache`. Cross-endpoint: one chat stream + one messages
  stream interleaved (proves the T13 coordinator interleaves over real HTTP, not just oneshot).
- [ ] **Step 9 — Linux+NVIDIA side** `[INFERENCE]` (Wave 3b box; exact commands so it can run
  unattended): record DEVIATION 4's `ds4-qwen4exp.json` (two ds4 commands), then
  `perf_gate record --target cuda --model $Q4E` (decode/prefill per frontier + `ple.rs`
  section), commit the record + baseline. `perf_gate check` gates them thereafter. Nothing in
  this step is executable on this Mac; the plan names commands, not results.
- [ ] **Step 10 — gate hunk (additive, T1's file).** Append to `scripts/check.sh`:
  `SHISU_SKIP_BUILD=1 cargo run -p shisu-bench --bin perf_gate -- check` — no model, no GPU,
  deterministic arithmetic (DEVIATION 6). The step must be green on docs-only diffs
  (stale-record rule must not fire when no measured input changed).
- [ ] **Step 11 — Final gate (the master 297 checklist, via `shisu/scripts/check.sh`):**
  `cargo clippy --workspace --all-targets --features metal` (check.sh runs the same; verify
  the metal feature is exercised — if T1's copy omitted it, add one step, additive);
  `cargo deny check` (check.sh step); file-size gate = `scripts/file_size_check.sh`
  (`find crates -name '*.rs' -print0 | xargs -0 wc -l` >1500 ⇒ fail);
  `cargo test --workspace --features metal` with
  `SHISU_SKIP_BUILD=1` (skips nvcc/runtime-compile in build.rs → stub registries, T14 — it
  does NOT skip clippy/deny/file-size, and the CuMetal lane is separately gated on
  `SHISU_CUDA_COMPILE=1`). Then the lane gate (Step 7) once on this Mac.

## Tests

- Always-on (no model, no GPU): `shisu-bench/tests/baseline_logic.rs` — schemas, ratio
  boundary, coverage, spread refusal, percentiles; e2e *helpers* (SSE frame parser, key-set
  assert fns) compile-checked but their cases `#[ignore]`.
- macOS `#[ignore]`: `perf_gate record --target metal` smoke (spread gate exercised);
  `e2e.rs` full suite; `cumetal_lane_gate.rs` (lane present).
- Linux `#[ignore]` `[INFERENCE]`: `perf_gate record --target cuda` + `check` against
  `ds4-qwen4exp.json`.
- Run: `cargo test -p shisu-bench` · `cargo test -p shisu-server -- --ignored e2e` ·
  `cargo run -p shisu-bench --bin perf_gate -- check`.

## Acceptance

- [ ] Every gate in master 294-297 has an owner, a command, a baseline file, and a provenance
  block (machine, gpu, driver, git_rev, model + bytes, run count, statistic); `check` is
  deterministic arithmetic and passes on a docs-only diff with a stale-free record.
- [ ] Zero throughput numbers recorded or compared from CuMetal anywhere (Rule 4); lane gate =
  STATUS `ok`-set equality + 1e-5 twin tests; SMEM_BLOCKED/NUMERIC-INVALID never fail the gate.
- [ ] p99 and resume gates match DEVIATIONS 1-2 (shisu-baseline p99 + ds4 mean comparison;
  resume vs T8's file); no ds4 number was invented for a metric ds4 cannot emit.
- [ ] Baselines changed only via record→spread-check→human-promote; `check` wrote nothing.
- [ ] e2e: four endpoints × blocking+stream, SSE `[DONE]` + Anthropic `message_stop`, 405/
  OPTIONS/`/v1/models`/`/cache`, interleaved streams; assertions traceable to ds4_server.c /
  published shapes, not to shisu structs.
- [ ] `scripts/check.sh` green end-to-end (clippy --features metal, deny, file-size,
  test --workspace --features metal with `SHISU_SKIP_BUILD=1`, `perf_gate check`);
  the `check.sh` diff is exactly the one appended step.
- [ ] No new env knobs; no `ATLAS_`/`DS4_` reads; no parity (T18), PLE-correctness (T19), or
  PTX-oracle (T15) re-runs owned here; no file >1500 LoC.

## Commit

```sh
git add shisu/crates/shisu-bench shisu/crates/shisu-server/tests/e2e.rs \
 shisu/crates/shisu-kernels/tests/cumetal_lane_gate.rs shisu/scripts/cumetal_lane_gate.sh \
 shisu/scripts/check.sh shisu/Cargo.lock
git commit -m "chore: perf baselines + e2e"
```

Commit message body cites: old→new macOS baseline tok/s (Step 4), record spreads, and the
Wave 3b recording commands still pending `[INFERENCE]`.
