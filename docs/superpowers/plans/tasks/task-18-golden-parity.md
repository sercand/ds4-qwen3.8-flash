# Task 18: Golden-vector parity

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 18 (277–280)
+ § Scope (21–37; Out list 28, Rule 3 at 33, Rule 4 at 34) + § CuMetal dev lane (39–74)
+ crate map/SSOT (76–126; `test-vectors/` row 116) + Roadmap Wave 3b (302–319; T18 row 313,
"golden-vector … stays Linux-only" 317) + verification log (323–331; "flash* exists" 327).
Read all six before executing.
**Depends on:**
- T2 (`task-02-core.md` 24–49, 259–266): `Model`/`Backend` traits verbatim + the `FakeBackend`
  contract-test pattern (the always-green host-side seam).
- T6 (`task-06-engine-qwen35.md` 109, 115–126, 137): `sampling.rs` bit-exactness proof is
  `shisu-engine/tests/sampling.rs` — **T18 references it, adds zero sampler code/tests**; the
  token lock `shisu/test-vectors/qwen35-greedy-64.json` (`{model_bytes, prompt, tokens[64]}`,
  task-06:123); `shisu-bench/src/bin/qwen35.rs` + `shisu-bench/baseline/m1max-qwen35.json`
  (the bench-crate layout T18 extends); the `tokenizers`-dev-dep tokenize pattern (task-06:123).
- T6b (`task-06b-optiq-loader.md` 145): `shisu-engine/tests/optiq_decode.rs` — GGUF-vs-OptiQ
  32-token equality is **owned there**; T18 references it, does not duplicate.
- T15 (`task-15-q4e-extraction.md` 257–264, 271–281): "Linux-side numerics stay T17/T18 — never
  claimed from this Mac"; the PTX body-hash split-fidelity oracle (`scripts/q4e_ptx_oracle.sh`)
  is step 3 of T18's bisect ladder; the 6 NUMERIC-INVALID kernels are a **CuMetal-lane-only**
  restriction (compound static+dynamic smem corrupts on the lane, T3b:239) — they run fine on
  real NVIDIA hardware and are in T18's Linux oracle set.
- T16 (`task-16-cuda-backend.md` 25–52, 284–291): `CudaBackend` surface + the Linux
  `#[ignore]` + `#[cfg(target_os = "linux")]` test pattern T18's Linux half rides on; classic
  cuBLAS `GemmEx` tier is the ds4-parity tier (T16 DEVIATION 1) — the reason a same-box
  ds4-vs-shisu diff can be tight.
- T17 (`task-17-qwen4exp-forward.md` 128, 229–234, 250–262): `tests/q4e_cpu_oracle.rs` = Rust
  f32 per-op references transcribed in ds4's arithmetic order — "**the host-side oracle T18's
  Linux parity diffs the kernels against**"; production head_dim 256 numerics "are owed by T18
  on Linux"; "No golden assertions — T18 owns parity vectors (none exist today)".
- T3 (`task-03-gguf.md` via task-17:13–16): split-GGUF load is open (`SplitUnsupported`) until
  the CUDA phase; q4e GGUFs ship split ⇒ any q4e full-model run on Linux is gated on that gap
  closing (`[INFERENCE]` — no q4e GGUF on this Mac).

**Produces** (paths under `shisu/`; repo root = cwd):
- `shisu/test-vectors/` — fixture home (master 116): `README.md` (grammars verbatim + capture
  + bisect procedure), `prompts/*.txt` (model-agnostic prompt corpus copied from ds4),
  `grammar-samples/{official.vec,local-golden.vec}` (1-case excerpts, parser conformance only),
  `qwen35-local-golden.skeleton.vec` + `qwen35-local-golden.vec` (captured on this Mac —
  **regression lock**), `q4e-ds4-golden.vec` (captured from **ds4 itself** on the Linux box —
  the real parity anchor, `[INFERENCE]`).
- `shisu-bench/src/vecfile.rs` + `shisu-bench/src/lib.rs` (new lib target exposing it) +
  `shisu-bench/src/bin/parity.rs` — fixture parser/writer SSOT + the parity CLI:
  ```
  cargo run -p shisu-bench --bin parity -- \
    --model "$SHISU_TEST_MODEL" --backend metal|cuda \
    --vectors shisu/test-vectors/<file>.vec [--case ID]… [--strict] \
    [--capture OUT.vec --label STR] [--json OUT.json]
  ```
  Replay mode asserts per case (grammar below); `--strict` tightens to the same-source budget
  (token sequence exact + top-20 max|Δlogit| ≤ 1e-3). Capture mode fills a case SKELETON (a
  `--vectors` file with `case` lines but no `top` lines) and writes the completed fixture to
  OUT.vec (`--label` = git rev + model file name + bytes; refuses to overwrite).
  Output: ds4-style per-case stderr lines + optional JSON summary artifact.
- `shisu-engine/tests/parity.rs` — macOS-runnable subset + Linux `#[ignore]` half (Tests).
- `shisu/scripts/ds4_dump_to_vec.py` — ds4 `--dump-logprobs` JSON → `ds4-local-golden-v1` vec.

## Global Constraints

1–7 from master plan (30–37). Task-specific:
1. Rule 3 is this task's core discipline: a mismatch is a bug to root-cause to a kernel/op and a
   ds4 line, **never** a reason to widen a tolerance. The bisect ladder (README + Step 6):
   full-model golden names (case, step) → dump that logits row → per-op kernel ladder vs T17's
   `q4e_cpu_oracle.rs` refs at 1e-5 (first op over 1e-5 names the kernel) → kernel matches its
   oracle but model diverges ⇒ diff launch geometry/args against `KERNEL_NAMES.md` rows and the
   ds4 launcher lines (T15) → kernel diverges ⇒ PTX body-hash vs the monolith
   (`q4e_ptx_oracle.sh`): equal hash = host bug, unequal = extraction bug, fix = re-extract from
   ds4, never hand-patch arithmetic.
2. Rule 4: T18 records **zero** throughput numbers; the CuMetal lane re-gate belongs to T20
   (master 295) — T18 never re-runs or re-gates it.
3. A locally-captured golden is a **regression lock, not a correctness proof** — the README
   states this in those words; nothing in T18 may claim otherwise (see DEVIATION 1/2).
4. Fixture grammars are kept **verbatim** from ds4 (both header strings, line shapes, and the
   assertion fields) so fixtures stay diffable against ds4's tooling; shisu adds only header
   comment lines (label, tokenizer/template sha, measured floor).
5. 1500-LoC cap: `vecfile.rs` ≤ 250, `parity.rs` bin ≤ 300, `tests/parity.rs` ≤ 460, the python
   converter ≤ 120.
6. No new env knobs (rule 5): `SHISU_TEST_MODEL` (+ T6b's `SHISU_TEST_MODEL_OPTIQ`) are the only
   env reads; ds4's `DS4_TEST_VECTOR_FILE`/`DS4_TEST_LOCAL_GOLDEN_FILE` become CLI flags.
7. `warnings = deny`, clippy deny, `thiserror` in the lib / `anyhow` at the bin boundary,
   `parking_lot`, `tracing`, no license headers, no formatters/full suites (T20 owns the gate).

## Source References (verified)

ds4 root = cwd. Every range opened while writing this plan.

| Source | Lines | What / how used |
|---|---|---|
| `tests/test-vectors/README.md` | 1–105 | Title "DeepSeek V4 Flash Test Vectors" (1–6: captured from `deepseek-v4-flash`, greedy, thinking off, `top_logprobs=20`); per-checkpoint dirs (8–18); file roles incl. local-golden's stated purpose "catches substantial backend drift" (20–29); fetcher CLI (31–42); runner flags + env overrides (44–61); glm dir (63–76); "intentionally trivial to parse from C … hex-encoded by bytes" (93–96); manual `--dump-logprobs` recipe (98–105). |
| `tests/test-vectors/flash-0731/manifest.json` | 1–51 | `"model": "deepseek-v4-flash"`, endpoint `api.deepseek.com/chat/completions`, top_logprobs 20, max_tokens 4, 5 cases (3 short + 2 long ≈18.5k chars). |
| `tests/test-vectors/flash-pre-0731/manifest.json` | 1–51 | Same model, `"checkpoint": "pre-0731"` — same out-of-scope problem. |
| `tests/test-vectors/glm-openrouter/manifest.json` | 1–15 | `"model": "z-ai/glm-5.2"` via OpenRouter, provider pinned `parasail/fp8`, no fallbacks — also out of scope (master 28). |
| `tests/test-vectors/flash-0731/official.vec` | 1–48 | Grammar `# ds4-official-logprob-vectors-v1`: `case <id> <ctx> <steps> <prompt-file>` / `step <i> <selected-hex> <n>` / `top <token-hex> <logprob>`; **tokens are hex UTF-8 bytes, not ids** (hosted API exposes no ids); this checkpoint's slice is top-1 with logprob 0. |
| `tests/test-vectors/flash-0731/local-golden.vec` | 1–70 | Grammar `# ds4-local-golden-v1`: `case <id> <mode> <ctx> <frontier> <prompt-file> <top-count>` / `top <rank> <token-id> <logit>`; 64-entry top-k at a 4096-token frontier — **this is the grammar shisu reuses** (token ids + logits, exactly what a local engine can emit). |
| `tests/test-vectors/flash-0731/official/short_italian_fact.official.json` | 1–43 | Audit record (`ds4-official-logprobs-v1`): request echo, `logits_available: false` — hosted vectors are a logprob slice, never full logits. |
| `tests/test-vectors/fetch_official_vectors.py` | 21–24, 101–126, 190–219, 222–284 | MODEL/ENDPOINT constants; POST = OpenAI chat/completions with `temperature:0, max_tokens:4, logprobs:true, top_logprobs:20, thinking disabled`; compact-fixture writer; CLI (`--checkpoint` required, `--out`, `--only`; `DEEPSEEK_API_KEY`). The whole requirement set is generic — but no qwen endpoint exists to point it at (DEVIATION 2). |
| `tests/ds4_test.c` | 5306–5379 | `test_logprob_vector_case`: chat-render prompt → prefill → per step: greedy `ds4_session_argmax`; selected token compared **by token bytes** (5336–5340); every official top token must appear in the local top-20 and `|Δlogprob| ≤ 4.0` (5342–5365); the LOCAL token is teacher-forced forward (5367–5374). |
| `tests/ds4_test.c` | 5381–5444 | One case permanently disabled with a written API-vs-official-graph rationale (5383–5391); harness pins `DS4_METAL_PREFILL_CHUNK=2048`, non-Metal4 route, canonical streaming prefill (5403–5412); default fixture + `DS4_TEST_VECTOR_FILE` override (5395–5398). |
| `tests/ds4_test.c` | 5519–5590 | local-golden parser (rank must be sequential 5556; ntop ≤ 128) + overlap / max-abs helpers. |
| `tests/ds4_test.c` | 5592–5671 | `test_local_golden_case_run`: modes `text`/`rendered`/`chat` (5599–5607); prefix truncated to `frontier`; full-vocab logits copied; asserts top1 id exact, top5 overlap ≥ 4/5, top20 ≥ 15/20, top64 ≥ 40/64, top20 max|Δ| ≤ 8.0 — "intentionally tolerant … catch substantial backend drift … not tiny floating-point differences" (5654–5663). |
| `tests/ds4_test.c` | 5673–5685, 6811–6813, 6853–6854 | `--local-golden-vectors` default + env override; CLI targets `--logprob-vectors` / `--local-golden-vectors` / `--metal-ssd-streaming-cache-pressure`; env docs. Runner binary = `ds4_test` (Makefile:677). |
| `tests/ds4_test.c` | 5990–6019, 6046–6062 | `test_load_mpp_cases`: the SAME official.vec drives route-vs-route "Tensor equivalence" (greedy fail, top1 mismatch, min top5/top-k overlap, worst rms/max_abs/top20_max_abs) — the ds4 precedent for internal-consistency parity on one model, which is what T18's macOS half generalizes. |
| `ds4_cli.c` | 722–735, 825–913 | `json_write_token` = `{"id","text","bytes"}`; `run_logprob_dump` (`--dump-logprobs`): greedy steps, top-k ≤ 128 entries with **both `logit` and `logprob`** — the capture source for `q4e-ds4-golden.vec` on Linux. |
| `ds4_cli.c` | 932–959 | `run_decode_consistency` (`--decode-consistency N`, ds4_help.c:290): live-decode logits vs fresh full prefill — the ds4 precedent for T18's chunked-prefill invariant. |
| `ds4.c` / `ds4_metal.m` / `ds4_cuda.cu` | grep counts | `qwen3.5`/`Qwen3_5`/`qwen35` matches: **2 / 0 / 0** — and both `ds4.c` hits are comments about the `tokenizer.ggml.pre = "qwen35"` string (39955, 40100), not a model path: **ds4 has no qwen3.5-4b forward at all**, so no ds4-generated golden for the Metal target is even possible (confirms master 34). `qwen4exp`/`QWEN38F` in `ds4.c`: **332 lines / 337 matches** — the q4e host forward (§67082–70532, T17) exists, so **ds4 itself is the q4e reference implementation on Linux**. |
| task plans | T2 259–266; T6 109, 115–126, 137; T6b 145; T15 257–264, 271–281; T16 284–291; T17 128, 229–234, 250–262 | FakeBackend pattern; greedy-64 lock + bench layout; OptiQ token-equality owner; "Linux-side numerics stay T17/T18"; Linux `#[ignore]` pattern; CPU oracle = T18's per-op reference; "T18 owns parity vectors (none exist today)". |
| atlas `crates/spark-runtime/tests/fast_weights_parity.rs` 1–14; `crates/atlas-rdma/tests/transcript_golden.rs` 1–14 | — | Two patterns copied: (a) two implementations of one contract must agree byte-for-byte on a synthetic fixture; (b) a frozen golden whose header states exactly what it does and does NOT prove. |

⚠ **DEVIATION 1 (no golden vectors exist for either port target — the master's "copy ds4
`tests/test-vectors/flash*`" is not executable as parity material):** verified above —
`flash-0731` and `flash-pre-0731` are `deepseek-v4-flash` (README 1–6; both manifests),
`glm-openrouter` is `z-ai/glm-5.2`; DeepSeek-V4 and GLM-5.2 are explicitly Out (master 28), and
ds4 contains no qwen3.5-4b forward (the only `qwen35` hits in ds4.c are tokenizer-`pre` comments),
so the master 279 checkbox "assert token-id
sequence + top-logit parity vs ds4 vectors" has no fixture and no ds4 path behind it for
qwen3.5-4b (master 34 says exactly this). T18 therefore defines parity concretely:
(a) the **grammars** are copied verbatim, not the fixtures — `ds4-local-golden-v1` (token ids +
logits) is the working grammar for both port targets; `ds4-official-logprob-vectors-v1` is kept
in the README + parser for any future hosted slice;
(b) **qwen3.5-4b/Metal** (no ds4 comparison exists): parity = engine-level self-consistency
invariants (chunked prefill == N recurrent steps — ds4's own `--decode-consistency` precedent;
run-to-run bit-exact determinism) + the captured `qwen35-local-golden.vec` regression lock,
referencing (not duplicating) T6's greedy-64 token lock, T6b's OptiQ equality, T8's
resume==cold;
(c) **qwen4exp/CUDA** (ds4 HAS the path): the only real ds4-vs-shisu diff that exists — capture
`q4e-ds4-golden.vec` from ds4 on the Linux box and replay it in shisu at the strict budget, plus
the per-op kernel ladder vs T17's CPU oracle at 1e-5 at production shapes (head_dim 256,
including the 6 CuMetal-NUMERIC-INVALID kernels — that label is a lane artifact only).

⚠ **DEVIATION 2 (no hosted-qwen fetcher step — there is nothing to fetch from):** the fetcher's
requirements are generic (OpenAI-compatible chat/completions + `logprobs`/`top_logprobs` +
`temperature:0`, script 101–126), but the repo contains no hosted qwen3.8-flash/qwen4exp
endpoint (the name is the internal next-gen model; the only fetchers are DeepSeek-official and
OpenRouter-GLM), and even a hosted Qwen3.5-4B slice would be a coarse drift check against a
foreign quant/serving stack (ds4 itself compares by token BYTES at ±4.0 nats, ds4_test.c:5359)
— never bit parity against the GGUF. So T18 ships no fetch step; goldens are locally captured,
and the README states in plain words: **a locally-captured golden locks regression; it proves
nothing about ds4.** The one genuine ds4 proof is (c) above.

⚠ **DEVIATION 3 (two tolerance profiles, both inherited, neither widened):** ds4's official
budget (±4.0 nats) and local-golden budget (top1 exact, ≥4/5, ≥15/20, ≥40/64, ≤8.0) are
calibrated for hosted-API and cross-backend drift. shisu keeps ds4's local-golden floors as the
outer cross-build lock, and adds `--strict` for same-source diffs: greedy token-id sequence
exact, top-20 overlap 20/20, top-20 max|Δlogit| ≤ 1e-3, per-op kernel-vs-oracle 1e-5 (T17/T15
budget). Rationale: byte-identical kernels + ds4 launch order + the same classic-cuBLAS GemmEx
tier (T16 DEVIATION 1) ⇒ same reduction order ⇒ same bits on the same GPU; anything above the
floor is a bug per Rule 3 and goes through the bisect ladder. Tolerances are constants with
source citations; a diff that changes one is a review stop.

⚠ **DEVIATION 4 (additive file layout):** the master Files list (bin + fixtures + engine test)
would duplicate the fixture parser across the bin and the test. `shisu-bench` gains a lib target
(`src/lib.rs` + `src/vecfile.rs` = parser/writer/report SSOT); `shisu-engine` adds a dev-dependency
on `shisu-bench` (dev-deps don't touch the shipping graph — T4's dev-dep precedent). The three
master-named files stay.

## Plan

- [ ] **Step 0 — Preflight.** Confirm on disk: `shisu-engine/tests/sampling.rs` (T6),
  `shisu/test-vectors/qwen35-greedy-64.json` (T6), `shisu-engine/tests/optiq_decode.rs` (T6b),
  `shisu-engine/tests/q4e_cpu_oracle.rs` (T17), `kernels/cuda/q4e/KERNEL_NAMES.md` +
  `scripts/q4e_ptx_oracle.sh` (T15). Missing dependency ⇒ STOP and report — T18 consumes these,
  it does not recreate them.
- [ ] **Step 1 — `shisu/test-vectors/README.md` + grammar samples + prompts.** Copy both ds4
  grammar headers verbatim with provenance (source file + line ranges from the table above);
  state the regression-lock sentence (DEVIATION 2) and the bisect ladder (Global Constraint 1)
  in the README. Copy `grammar-samples/` = one `case` excerpt of each ds4 grammar (from
  flash-0731) labeled "parser conformance only — model out of scope, never replayed". Copy the
  5 model-agnostic prompt texts from `flash-0731/prompts/` + `tests/long_context_story_prompt.txt`
  into `prompts/` (plain user prompts, no model material).
- [ ] **Step 2 — `shisu-bench/src/vecfile.rs` (≤250 ln) + `src/lib.rs`.** Line-oriented parser
  + writer for both grammars, mirroring ds4's reader semantics exactly: sequential-rank check
  (ds4_test.c:5556), `ntop ≤ 128` (5530), `ctx > frontier > 0` (5528–5529), hex-byte decode for
  the official grammar; typed errors naming file + line. Extra header comment fields (ignored
  by ds4's `#` skipping): `# label`, `# model_bytes`, `# tokenizer_sha256`, `# template_sha256`,
  `# captured_from ds4|shisu`. JSON summary writer for `--json`.
- [ ] **Step 3 — `shisu-bench/src/bin/parity.rs` (≤300 ln).** clap 4 CLI per Produces. Replay:
  tokenize per mode (`text` = raw tokenize; `chat` = the T6 `tokenizers` + recorded template
  render — the fixture's tokenizer/template shas must match the run, else typed error), truncate
  to `frontier`, `prefill_chunk` loop, `copy_d2h` logits row, top-k via T6's `argmax`-family
  ordering, assert the ds4 floor set (or `--strict` set, DEVIATION 3), print
  `shisu-parity: case <id> top1 ref=… cand=… top5_overlap=…/5 top20_overlap=…/20
  top64_overlap=…/64 top20_max_abs=…` (ds4_test.c:5647–5652 shape). Capture: same walk over a
  skeleton (case lines, no tops), writes the completed vec; refuses overwrite; `--label` lands
  in the header. `--backend cuda` constructs `CudaBackend` (Linux only; macOS ⇒ typed error,
  matching T16's stub rule).
  No throughput timing anywhere (Rule 4).
- [ ] **Step 4 — `shisu/scripts/ds4_dump_to_vec.py` (≤120 ln).** Convert ds4
  `--dump-logprobs` JSON (`{"id","text","bytes"}` + `logit` + `logprob` per top entry,
  ds4_cli.c:872–903) into `ds4-local-golden-v1`: step 0's top-k becomes the `case`/`top` block
  (mode `chat`, ctx + frontier = the dump's echoed `ctx`/`prompt_tokens`); `--label`,
  `--top-keep N` (default 64). Stdlib only, like ds4's fetcher.
- [ ] **Step 5 — capture `qwen35-local-golden.vec` (macOS, this box).**
  Commit `qwen35-local-golden.skeleton.vec` = one case line in ds4's shape —
  `case long_story_4096 text 5000 4096 prompts/long_context_story_prompt.txt 64` (ds4's own
  local-golden case, local-golden.vec:5) — then
  `SHISU_TEST_MODEL=…/Qwen3.5-4B-Q4_K_M.gguf cargo run -p shisu-bench --bin parity -- --model …
  --backend metal --vectors shisu/test-vectors/qwen35-local-golden.skeleton.vec --capture
  shisu/test-vectors/qwen35-local-golden.vec --label "$(git rev-parse HEAD)"`. Human-check the
  greedy continuation for coherent English before committing (T6's discipline); run twice,
  assert the two captures are byte-identical (determinism floor). This file is a regression
  lock (DEVIATION 2 wording in its header).
- [ ] **Step 6 — `shisu-engine/tests/parity.rs` (≤460 ln).** Three tiers (Tests section spells
  the assertions). macOS always: vecfile conformance vs `grammar-samples/` + malformed-line
  rejection table (this is the gate-runnable subset). macOS `#[ignore]`: qwen35 invariants +
  golden replay. Linux `#[ignore]` + `#[cfg(target_os = "linux")]`: kernel ladder vs T17 CPU
  oracle at 1e-5 production shapes; decode-graph replay == eager bit-identical (ds4's own
  byte-identical-replay claim, ds4_cuda.cu:954–1050 via T16); `q4e-ds4-golden.vec` replay at
  `--strict`.
- [ ] **Step 7 — Linux capture + first parity run `[INFERENCE]` (not executable on this Mac —
  no NVIDIA driver, no q4e GGUF, T3 `SplitUnsupported` must be closed first).** Wave 3b box:
  `./ds4 --cuda --nothink --temp 0 -n 8 --ctx 16384 --prompt-file <p> --dump-logprobs <p>.json
  --logprobs-top-k 64` per prompt → `ds4_dump_to_vec.py` → commit `q4e-ds4-golden.vec` with the
  box/GPU/driver in `--label` → `parity --backend cuda --vectors q4e-ds4-golden.vec --strict`.
  Any mismatch: bisect ladder (Global Constraint 1) before anything merges; the fix commit names
  kernel + ds4 line.
- [ ] **Step 8 — gate wiring.** The local gate `shisu/scripts/check.sh` (T1) already runs
  `cargo test -p shisu-engine` ⇒ the always-tier lands with zero extra wiring; the Linux GPU run
  (Wave 3b, manual runbook per T16 Step 12) adds `-- --ignored`. No CuMetal gate (T20's lane rule, master 295).

## Tests

- **Always (macOS + Linux via `scripts/check.sh`, no model/GPU):** `tests/parity.rs` always-tier — both grammar
  samples parse to the exact case/step/top structures (token-byte hex decode spot case: `416461`
  → "Ada"); writer→parser round-trip byte-equal; malformed lines (bad rank order, ntop > 128,
  `frontier ≥ ctx`, non-hex token) each produce the named typed error.
- **macOS `#[ignore]` (`SHISU_TEST_MODEL` + `metal_available()`):**
  1. determinism floor: two identical greedy runs produce bit-identical logits rows;
  2. chunked prefill == N recurrent steps: same greedy token sequence and top-1 logit
     |Δ| ≤ 1e-3 over a 64-token prompt (engine-level generalization of T5's kernel-level 1e-3
     test and ds4's `--decode-consistency`);
  3. golden replay: `qwen35-local-golden.vec` at the ds4 floor set + greedy-64 token equality
     (reads T6's fixture — the lock pair).
  Referenced, not duplicated: T6 `tests/sampling.rs` (sampler bit-exactness proof), T6b
  `optiq_decode.rs` (GGUF-vs-OptiQ tokens), T8 resume==cold.
- **Linux GPU `#[ignore]` `[INFERENCE]`:** per-op kernel ladder vs `q4e_cpu_oracle.rs` at 1e-5
  (production head_dim 256; all 50 kernels including the 6 lane-invalid ones); decode-graph
  replay logits bit-identical to eager; `q4e-ds4-golden.vec` at `--strict` (greedy token-id
  sequence exact + top-20 max|Δlogit| ≤ 1e-3).
- No sampler tests, no throughput assertions, no CuMetal runs anywhere in T18.

## Acceptance

- [ ] `shisu/test-vectors/{README.md,prompts/,grammar-samples/,qwen35-local-golden.skeleton.vec,
  qwen35-local-golden.vec}` +
  `shisu-bench/src/{lib.rs,vecfile.rs,bin/parity.rs}` + `shisu-engine/tests/parity.rs` +
  `shisu/scripts/ds4_dump_to_vec.py` exist, each code file < 1500 LoC; `q4e-ds4-golden.vec`
  exists iff the Linux run happened, else its absence + the `[INFERENCE]` note is the state.
- [ ] No ds4 flash*/glm fixture copied as parity material; grammar-samples carry the
  "parser conformance only" label; grammars byte-equal to ds4's headers.
- [ ] README contains, verbatim in sense: the regression-lock sentence, the bisect ladder, and
  "CuMetal numbers never gate" (Rule 4). Zero throughput numbers in any T18 artifact.
- [ ] `cargo test -p shisu-engine` green on macOS (always-tier); `-- --ignored` green with the
  qwen35 GGUF; Linux half compiles on macOS against T16's stubs and runs only on the Wave 3b box.
- [ ] Sampler proof untouched: `git diff` shows no change under `sampling*` and no new sampler
  test; OptiQ equality test untouched (T6b owns it).
- [ ] Tolerance constants appear exactly once each with a source citation; any closed parity
  mismatch is recorded in the commit message naming kernel/op + ds4 line; no tolerance value was
  changed to make a test pass (Rule 3).
- [ ] No new env knobs; no `ATLAS_`/`DS4_` env reads; no formatters/full suites run (T20 gate).

## Commit

```sh
git add shisu/test-vectors shisu/crates/shisu-bench shisu/crates/shisu-engine/tests/parity.rs \
        shisu/crates/shisu-engine/Cargo.toml shisu/scripts/ds4_dump_to_vec.py shisu/Cargo.lock
git commit -m "test(parity): golden-vector parity harness — .vec grammar reuse, qwen35 locks, q4e ds4-captured anchor (Task 18)"
```
