# Task 1: Workspace Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Depends on:** none — this is the first task (Wave 1: `T1 ──► T2 …`). Nothing is consumed from other tasks.

**Produces** (later tasks read this block; every path is relative to the ds4 repo root `/Users/sercand/Developer/src/github.com/sercand/ds4-qwen3.8-flash`):

- Workspace root `shisu/Cargo.toml` — virtual workspace, `resolver = "2"`, members exactly:
  `crates/shisu-core`, `crates/shisu-gguf`, `crates/shisu-metal`, `crates/shisu-engine`,
  `crates/shisu-kvstore`, `crates/shisu-ir`, `crates/shisu-server`, `crates/shisu-bench`.
- `[workspace.dependencies]` entries usable as `{ workspace = true }` by every later task:
  `serde`, `serde_json` (preserve_order), `thiserror`, `anyhow`, `parking_lot`, `tracing`,
  `tracing-subscriber`, `clap`, `axum`, `http-body`, `hyper`, `hyper-util`, `tower`,
  `tower-http`, `tokio`, `tokio-stream`, `futures`, `minijinja`, `tokenizers`, `safetensors`,
  `sha1`, `hex`, `objc2`, `objc2-metal`, `objc2-foundation`, `dispatch2`,
  `cudarc` (CUDA phase only), `io-uring` (CUDA phase only), and path entries
  `shisu-core` … `shisu-bench` (`shisu-core` with `default-features = false`).
- Features: `shisu-core` has `default = []`, `cuda = []`, `metal = []` (Task 2 fills the
  traits; Task 16 binds `cuda = ["dep:cudarc"]`); `shisu-metal` has `default = ["metal"]`,
  `metal = []` (Task 4 binds the optional objc2 deps).
- Lint policy: `[workspace.lints.rust] warnings = "deny"`, `[workspace.lints.clippy] all =
  { level = "deny", priority = -1 }` + the atlas allow-list; every crate carries `[lints] workspace = true`.
- Toolchain pin `shisu/rust-toolchain.toml` = 1.93.1 (rustfmt+clippy); `shisu/deny.toml`;
   `shisu/.gitignore` (`/target`).
- Local gate scripts `shisu/scripts/file_size_check.sh` (1500-LoC cap) + `shisu/scripts/check.sh`
   (`cargo check/test/clippy --workspace` + `cargo deny check` + file-size cap; exports
   `SHISU_SKIP_BUILD=1`). GitHub Actions is out of scope — no `.github/` files exist (master rule 2 + Scope-Out).

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 1 (read it before executing; the spec argues this plan)

## Global Constraints

Master plan Rules 1–7, one line each:

1. All new code under `shisu/`. No C/C++/ObjC host code — only `.cu`/`.cuh`/`.metal` kernel files (copied from ds4); sole exception `tests/cumetal/*.cu` (Task 3b).
2. Hard 1500-LoC-per-file cap, local gate `scripts/file_size_check.sh` (logic ported from atlas `file-size-cap.yml`; GitHub Actions out of scope).
3. Kernels copied byte-for-byte where possible; arithmetic-order changes are bugs. (No kernels in this task.)
4. No performance regression; CuMetal numbers never gate anything. (No perf surface in this task.)
5. Env knobs prefixed `SHISU_` — this task introduces `SHISU_SKIP_BUILD=1` in the local gate env only (`scripts/check.sh`). **No `ATLAS_*` string may appear anywhere under `shisu/`.**
6. Workspace lints: `warnings = "deny"`, clippy all deny; `thiserror` library errors, `anyhow` binary boundaries; `parking_lot` locks; `tracing` logging.
7. License headers: none (user sets license later) — do **not** copy atlas's `// SPDX-License-Identifier: AGPL-3.0-only` first lines.

Task-specific:
- Run all cargo commands from `shisu/` so `shisu/rust-toolchain.toml` applies.
- `shisu-kernels`, `shisu-cuda`, `shisu-ple` are **deliberately NOT created** — CUDA phase (Tasks 14/16/19). Likewise `kernels/`, `kernels/cumetal_compat/` (Task 3b), `test-vectors/` (Tasks 2/18). No `.github/` and no `scripts/ci_gpu_stubs.sh` — GitHub Actions is out of scope (master Scope-Out).
- Do not run formatters/linters/project-wide suites beyond this plan's own steps; Task 20 owns the final gate.

## Source References (verified)

All atlas paths relative to `/Users/sercand/Developer/src/github.com/sercand/atlas`. Every line below was opened and read while writing this plan.

| Source | Lines | What lives there / how it is used |
|---|---|---|
| `Cargo.toml` | 1–20 | `[workspace] resolver = "2"` + members list — structure copied, members replaced by the 8 shisu crates. |
| `Cargo.toml` | 27–28 | `[patch.crates-io] cudarc = { path = "vendor/cudarc" }` — **NOT copied** (atlas SCALE/AMD symbol-stub hack; shisu uses crates.io cudarc). |
| `Cargo.toml` | 30–41 | `[workspace.package]` (edition 2024, rust-version) — copied minus license/authors (rule 7). |
| `Cargo.toml` | 51–57 | `cudarc = { version = "0.19", default-features = false, features = ["std","driver","nvrtc","fallback-dynamic-loading","cuda-version-from-build-system"] }` — copied verbatim into shisu workspace deps. |
| `Cargo.toml` | 60, 66, 69, 70, 80 | `serde 1 (derive)`, `serde_json 1 (preserve_order)`, `thiserror 2`, `anyhow 1`, `parking_lot 0.12` — copied verbatim. |
| `Cargo.toml` | 83–86 | `atlas-core = { path = …, default-features = false }` + comment "so each consumer crate forwards the right backend feature explicitly … every member would silently pull in cudarc" — the per-target pattern the master plan cites; copied for `shisu-core`. |
| `Cargo.toml` | 96–112 | `[workspace.lints.rust] warnings = "deny"`; `[workspace.lints.clippy] all deny priority -1` + 8 allow entries — copied verbatim. |
| `rust-toolchain.toml` | 1–8 | `channel = "1.93.1"`, components rustfmt+clippy — copied (comment adapted, xgrammar rationale dropped). |
| `deny.toml` | 1–86 | cargo-deny config: `[graph] all-features`, advisories + RUSTSEC-2024-0436 ignore (18–23), licenses allow-list (31–53), bans `wildcards = "warn"` rationale (57–68), sources deny (70–86) — adapted (see Step 9 for what is dropped). Note: line 28's comment references a `[licenses] private.ignore` section that does not exist in the file (stale comment); shisu uses `publish = false` instead. |
| `.github/workflows/ci.yml` | 125–142, 144–187 | `test` + `test-macos-metal` job command sequences (`cargo test --workspace --locked`, `cargo check -p … --no-default-features --features metal`, metal test, `otool -L` no-libcuda assertion :181–187) — **not copied as a workflow** (GitHub Actions out of scope); the command sequence is ported into `scripts/check.sh`. |
| `.github/workflows/ci.yml` | 40, 43 | workflow-level `env: ATLAS_SKIP_BUILD: "1"` (→ renamed `SHISU_SKIP_BUILD`) — becomes the `export` line at the top of `scripts/check.sh`. |
| `.github/workflows/file-size-cap.yml` | 451–472 | the gate loop (`find crates -name '*.rs' -print0 | xargs -0 wc -l` + per-file cap + error messaging 466–488) — ported into `scripts/file_size_check.sh` with the cap at **1500**; atlas's ~90-entry legacy allow-list is NOT copied (shisu starts empty). Triggers/permissions/concurrency/action-pin rows (4–32) are irrelevant without CI. |
| `crates/atlas-core/Cargo.toml` | 9–14 | backend-feature pattern: `default = ["cuda"]`, `cuda = ["dep:cudarc"]`, `metal = []` — adapted (see ⚠ DEVIATION 2). |
| `crates/atlas-core/src/lib.rs` | 1–4 | crate stub pattern: SPDX line (dropped per rule 7) + `#![deny(warnings)]` + `#![deny(clippy::all)]` — the two deny attrs are copied into every shisu `lib.rs`. |
| `crates/spark-runtime/Cargo.toml` | 79–101, 105 | `objc2 0.6`, `objc2-metal 0.3` (feature list: MTLDevice, MTLBuffer, MTLResource, MTLCommandQueue, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipeline, MTLLibrary, MTLBlitCommandEncoder, MTLEvent, block2), `objc2-foundation 0.3` (NSData, NSString, NSError), `dispatch2 0.3`, `safetensors 0.8` — versions/features copied into shisu workspace deps. |
| `crates/spark-server/Cargo.toml` | 53–62, 79–85, 91, 99 | `clap 4 (derive)`, `axum 0.8 (json)`, `http-body 1`, `hyper 1 (server,http1,http2,client)`, `hyper-util 0.1 (server-auto,client-legacy,http1,tokio)`, `tower 0.5 (util)`, `tokio 1 (full,parking_lot)`, `tokio-stream 0.1`, `futures 0.3`, `tracing 0.1`, `tracing-subscriber 0.3 (env-filter)`, `tokenizers 0.23 (default-features=false, onig)`, `minijinja 2 (builtins,adjacent_loop_items,json,preserve_order)`, `tower-http 0.7 (catch-panic,cors)` — copied. |
| `crates/spark-storage/Cargo.toml` | 49–50 | `io-uring = "0.7"` under `[target.'cfg(target_os = "linux")'.dependencies]` — version copied; shisu declares it in `[workspace.dependencies]` and the CUDA phase puts it behind a linux target table. |
| ds4 `ds4_kvstore.c` | (19 `sha1` occurrences) | justifies the `sha1`+`hex` workspace deps (sha1-hex checkpoint filenames, master plan Task 7). |

⚠ **DEVIATION 1 (master-plan reference fix):** Task 1's Interfaces block says these workspace deps are "copied from atlas root Cargo.toml (… axum 0.8, hyper 1, tokio, clap 4, tracing, objc2 stack, minijinja, tokenizers 0.23, safetensors, sha1, hex …)". Verified: the atlas **root** manifest (112 ln) contains only cudarc, serde, serde_json, thiserror, anyhow, sha2, parking_lot + path crates. The rest live in per-crate manifests (rows above); their exact versions/features are taken from there. `sha1` and `hex` exist **nowhere** in atlas (grep: no matches) — they are pinned from crates.io (`sha1 = "0.10"`, `hex = "0.4"`) because `shisu-kvstore` needs sha1-hex names (ds4_kvstore.c). `sha2` (atlas:74, kernel-closure provenance) is NOT copied — no shisu consumer.

⚠ **DEVIATION 2 (master-plan reference fix):** "Default features: `metal` on macOS, `cuda` on Linux" cannot be literal Cargo: feature resolution is target-independent, so a `default` list cannot be OS-conditional. Resolution: root `[workspace.dependencies]` carries no backend choice; each binary crate sets `default = []` (empty) and the executor passes explicit `--features metal` in every macOS build command. The `cuda` default flips on only when `shisu-cuda` joins the workspace in Task 16 (which also binds `cuda = ["dep:cudarc"]`).

## Implementation Steps

- [ ] **Step 1: Preflight** — from the repo root run `ls shisu` and expect `No such file or directory` (verified absent when this plan was written; if it exists, STOP and reconcile — do not overwrite). Run `rustc --version` from any directory; rustup will auto-install 1.93.1 on first cargo command in Step 10 (needs network once).

- [ ] **Step 2: Create the directory skeleton**

```bash
mkdir -p shisu/crates/{shisu-core,shisu-gguf,shisu-metal,shisu-engine,shisu-kvstore,shisu-ir,shisu-server,shisu-bench}/src
```

- [ ] **Step 3: Write `shisu/.gitignore`** (the ds4 root `.gitignore` has no `target` entry — verified — so this file is required before the commit step):

```gitignore
/target
```

- [ ] **Step 4: Write `shisu/rust-toolchain.toml`** (adapted from atlas `rust-toolchain.toml:1-8`; xgrammar rationale replaced with the master-plan rationale):

```toml
[toolchain]
# Pinned to a specific stable for reproducible builds (master plan Tech
# Stack: Rust edition 2024, pin 1.93.1 — same pin as atlas). Bump
# deliberately rather than tracking `stable` — auto-upgrades produce
# subtle clippy lint surprises across kernel-launching code.
channel = "1.93.1"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 5: Write `shisu/Cargo.toml`** (structure from atlas `Cargo.toml:1-20,30-41,43-94,96-112`; versions per Source References):

```toml
[workspace]
resolver = "2"
members = [
    "crates/shisu-core",
    "crates/shisu-gguf",
    "crates/shisu-metal",
    "crates/shisu-engine",
    "crates/shisu-kvstore",
    "crates/shisu-ir",
    "crates/shisu-server",
    "crates/shisu-bench",
]
# shisu-kernels, shisu-cuda and shisu-ple are deliberately NOT members
# yet — they are created in the CUDA phase (Tasks 14/16/19). Same for
# kernels/ + kernels/cumetal_compat/ (Tasks 3b/14/15), test-vectors/
# (Tasks 2/18).

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.93"
# No license field: master plan rule 7 (license set later). Every crate
# is publish = false, which also makes `cargo deny check licenses` skip
# the workspace crates (they are not shipped artifacts yet).

[workspace.dependencies]
# Serialization (HF config.json). `preserve_order` keeps JSON object keys
# in insertion order (IndexMap) — required for the transformers `tojson`
# key-order match in the <tools> prompt block (atlas Cargo.toml:61-66).
serde = { version = "1", features = ["derive"] }
serde_json = { version = "1", features = ["preserve_order"] }

# Error handling: thiserror for library errors, anyhow at binary
# boundaries (global constraint 6).
thiserror = "2"
anyhow = "1"

# parking_lot: faster than std::sync, no poisoning, .lock() returns the
# guard directly (global constraint 6; atlas Cargo.toml:76-80).
parking_lot = "0.12"

# Logging (global constraint 6).
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# HTTP stack — versions/features from atlas crates/spark-server/Cargo.toml:53-62,79-85,91,99
# (the Task 10-13 server pattern: manual hyper accept loop with
# header_read_timeout, CORS + catch-panic, SSE).
clap = { version = "4", features = ["derive"] }
axum = { version = "0.8", features = ["json"] }
http-body = "1"
hyper = { version = "1", features = ["server", "http1", "http2", "client"] }
hyper-util = { version = "0.1", features = ["server-auto", "client-legacy", "http1", "tokio"] }
tower = { version = "0.5", features = ["util"] }
tower-http = { version = "0.7", features = ["catch-panic", "cors"] }
tokio = { version = "1", features = ["full", "parking_lot"] }
tokio-stream = "0.1"
futures = "0.3"

# Tokenizer + chat templates (atlas spark-server:85,91).
minijinja = { version = "2", features = ["builtins", "adjacent_loop_items", "json", "preserve_order"] }
tokenizers = { version = "0.23", default-features = false, features = ["onig"] }

# Model files: GGUF is hand-rolled (shisu-gguf); safetensors for the
# OptiQ loader (Task 6b); sha1+hex for kvstore checkpoint names
# (ds4_kvstore.c; not present anywhere in atlas — pinned from crates.io).
safetensors = "0.8"
sha1 = "0.10"
hex = "0.4"

# Apple Metal stack — versions + protocol-feature surface from atlas
# crates/spark-runtime/Cargo.toml:79-101. Task 4 extends the feature
# list if a protocol is missing.
objc2 = "0.6"
objc2-metal = { version = "0.3", features = [
    "MTLDevice",
    "MTLBuffer",
    "MTLResource",
    "MTLCommandQueue",
    "MTLCommandBuffer",
    "MTLCommandEncoder",
    "MTLComputeCommandEncoder",
    "MTLComputePipeline",
    "MTLLibrary",
    "MTLBlitCommandEncoder",
    "MTLEvent",
    "block2",
] }
objc2-foundation = { version = "0.3", features = [
    "NSData",
    "NSString",
    "NSError",
] }
# `newLibraryWithData` takes a DispatchData (libdispatch byte container),
# not NSData — dispatch2 ships the wrapper (atlas spark-runtime:100-101).
dispatch2 = "0.3"

# CUDA phase ONLY — declared now so Phase 4 manifests only need
# `{ workspace = true }`. No Task 1 crate references them, so
# `cargo check` never downloads or builds them. cudarc keeps atlas's
# driver+nvrtc-only feature set (atlas Cargo.toml:44-57); atlas's
# vendored-cudarc [patch.crates-io] (atlas:22-28) is deliberately NOT
# copied — it is an atlas SCALE/AMD symbol-stub hack. io-uring is used
# behind a linux target table by shisu-ple (Task 19).
cudarc = { version = "0.19", default-features = false, features = [
    "std",
    "driver",
    "nvrtc",
    "fallback-dynamic-loading",
    "cuda-version-from-build-system",
] }
io-uring = "0.7"

# Workspace crates. `shisu-core` uses default-features = false so each
# consumer forwards the backend feature explicitly (atlas pattern,
# atlas Cargo.toml:83-86) — without it, every member would silently
# pull in cudarc via shisu-core's default on macOS.
shisu-core = { path = "crates/shisu-core", default-features = false }
shisu-gguf = { path = "crates/shisu-gguf" }
shisu-metal = { path = "crates/shisu-metal" }
shisu-engine = { path = "crates/shisu-engine" }
shisu-kvstore = { path = "crates/shisu-kvstore" }
shisu-ir = { path = "crates/shisu-ir" }
shisu-server = { path = "crates/shisu-server" }
shisu-bench = { path = "crates/shisu-bench" }

# Workspace-wide lint policy (global constraint 6). Each crate also adds
# #![deny(warnings)] / #![deny(clippy::all)] to its lib.rs/main.rs; this
# section extends the policy to integration tests, benches and examples
# (atlas Cargo.toml:96-112, allow-list copied verbatim).
[workspace.lints.rust]
warnings = "deny"

[workspace.lints.clippy]
all = { level = "deny", priority = -1 }
too_many_arguments = "allow"
needless_range_loop = "allow"
large_enum_variant = "allow"
doc_lazy_continuation = "allow"
doc_overindented_list_items = "allow"
type_complexity = "allow"
ptr_arg = "allow"
if_same_then_else = "allow"
```

- [ ] **Step 6: Write `shisu/crates/shisu-core/Cargo.toml`:**

```toml
[package]
name = "shisu-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish = false
description = "shisu: shapes/config (HF config.json + ds4 shape profiles), dtypes, error types, Backend + Model traits, q4e page geometry."

[features]
# Backend selection mirrors atlas-core (atlas-core/Cargo.toml:9-14:
# default=["cuda"], cuda=["dep:cudarc"], metal=[]). Cargo feature
# resolution is target-independent, so a target-conditional `default`
# is impossible; the default here is EMPTY and the backend is chosen
# explicitly:
#   * macOS builds: `--features metal` (the workspace default)
#   * Linux CUDA phase (Task 16): binds `cuda = ["dep:cudarc"]` and
#     flips the default once shisu-cuda joins the workspace.
default = []
cuda = []
metal = []

[lints]
workspace = true
```

and `shisu/crates/shisu-core/src/lib.rs` (deny attrs from atlas-core/src/lib.rs:3-4; NO SPDX line, rule 7):

```rust
//! shisu-core — shapes/config (HF config.json + ds4 shape profiles),
//! dtypes, error types, the `Backend` + `Model` traits, and q4e page
//! geometry constants. Filled in by Task 2.

#![deny(warnings)]
#![deny(clippy::all)]
```

- [ ] **Step 7: Write `shisu/crates/shisu-metal/Cargo.toml`:**

```toml
[package]
name = "shisu-metal"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish = false
description = "shisu: Metal backend (objc2) — device init, pipeline cache, runtime MSL compile (newLibraryWithSource, ds4_metal.m pattern)."

[features]
# This crate IS the Metal backend; `metal` stays a real feature so
# consumers can forward it (`shisu-server/metal -> shisu-metal/metal`)
# and so Task 4 can bind the optional objc2 deps
# (`metal = ["dep:objc2", ...]`, atlas spark-runtime/Cargo.toml:83-101
# pattern). The objc2 dependencies themselves arrive in Task 4.
default = ["metal"]
metal = []

[lints]
workspace = true
```

and `shisu/crates/shisu-metal/src/lib.rs`:

```rust
//! shisu-metal — Metal `Backend` implementation (objc2): device init,
//! buffer alloc, pipeline cache, runtime MSL compile. Filled in by
//! Task 4.

#![deny(warnings)]
#![deny(clippy::all)]
```

- [ ] **Step 8: Write the six remaining crate manifests.** They are identical except `name` and `description` (no `[features]` section — backend features live on `shisu-core`/`shisu-metal` only). Template:

```toml
[package]
name = "shisu-<NAME>"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish = false
description = "<DESCRIPTION>"

[lints]
workspace = true
```

Exact `<NAME>` / `<DESCRIPTION>` pairs:

| name | description |
|---|---|
| `shisu-gguf` | `shisu: GGUF parser, quant block formats (q8_0/q4_k/q4e/iq2/mxfp4/exl3), tensor binding. Filled in by Task 3.` |
| `shisu-engine` | `shisu: backend-agnostic orchestration — qwen35 forward, sampling, KV cache, sessions. Filled in by Task 6.` |
| `shisu-kvstore` | `shisu: disk KV checkpoints (ds4_kvstore.c port — sha1 names, 48B header, LRU-by-utility). Filled in by Task 7.` |
| `shisu-ir` | `shisu: neutral request/stream IR (atlas spark-server/src/ir copy). Filled in by Task 9.` |
| `shisu-server` | `shisu: the shisu binary — clap CLI, axum router, OpenAI/Anthropic adapters, scheduler. Filled in by Task 10 (lib stub until then; main.rs lands in Task 10).` |
| `shisu-bench` | `shisu: perf gates — decode/prefill tok/s, kvstore resume latency, baselines. Filled in by Task 20.` |

and each `src/lib.rs` — one doc line naming the crate and its filling task, then the two deny attrs, e.g. for `shisu-gguf`:

```rust
//! shisu-gguf — GGUF parser, quant block formats, tensor binding.
//! Filled in by Task 3.

#![deny(warnings)]
#![deny(clippy::all)]
```

(`shisu-server` is a **lib stub in this task**; its `main.rs` arrives with Task 10 — the master plan's Task 1 Files block specifies lib.rs stubs for all eight crates.)

- [ ] **Step 9: Write `shisu/deny.toml`** (adapted from atlas `deny.toml:1-86`; dropped: `AGPL-3.0-only` + `CDLA-Permissive-2.0` allow entries — shisu has no license yet (rule 7) and workspace crates are `publish = false`, which the licenses check skips; dropped: the lattice-db `allow-git` entry (atlas-governance-only) and the vendored-cudarc patch rationale):

```toml
# cargo-deny configuration for shisu
# Adapted from atlas deny.toml. Workspace crates are publish = false,
# so `cargo deny check licenses` skips them; the allow-list below
# governs third-party code only.

[graph]
all-features = true

[advisories]
db-urls = ["https://github.com/rustsec/advisory-db"]
yanked = "warn"
# Only `unmaintained` notices on transitive deps we don't control —
# same entry and rationale as atlas deny.toml:18-23 (pulled in through
# the tokenizers dependency tree). Re-evaluate if it becomes a real
# vulnerability.
ignore = [
    { id = "RUSTSEC-2024-0436", reason = "paste is unmaintained; proc-macro is feature-complete and pulled in by transitive deps we don't control" },
]

[licenses]
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "CC0-1.0",
    "Unicode-3.0",
    "Unicode-DFS-2016",
    "Zlib",
]
confidence-threshold = 0.8
exceptions = []

[bans]
multiple-versions = "warn"
# `wildcards = "warn"` (not deny): shisu workspace crates inherit each
# other via `path` deps that resolve to `*` in Cargo.lock (atlas
# deny.toml:59-65 rationale). Defense-in-depth stays: Cargo.lock is
# committed, advisories check is strict.
wildcards = "warn"
highlight = "all"
skip = []
skip-tree = []

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
# No allow-git entries: shisu has no git dependencies. Keep it that way.
```

- [ ] **Step 10: Verify the workspace compiles** — from `shisu/`:

```bash
cargo check --workspace --features metal
```

Expected: `Compiling` lines for the 8 shisu crates, ending `Finished dev profile`. No cudarc/objc2/tokio downloads appear (unused `[workspace.dependencies]` entries are never resolved into the graph). Then also verify the no-feature build passes (proves the empty default pulls nothing):

```bash
cargo check --workspace
```

Expected: `Finished` with no new work beyond feature-off rebuilds. If the pinned cargo rejects bare `--features metal` in the virtual workspace (it should not — shisu-core and shisu-metal define it), use `--features shisu-core/metal,shisu-metal/metal` in both commands and in `scripts/check.sh`.

- [ ] **Step 11: Local gate dry-runs** — from `shisu/`:

```bash
cargo test --workspace --features metal        # expect: "0 passed" per crate, exit 0
cargo clippy --workspace --features metal --all-targets   # expect: clean (warnings=deny)
bash scripts/file_size_check.sh
# expect: exit 0; every file far below 1500 lines (stubs are ~6 lines)
grep -rn "ATLAS" . --include='*' --exclude-dir=target && echo "FAIL: ATLAS_ string present" || echo "OK: no ATLAS references"
```

`cargo deny check` is NOT required here (Task 20 owns the deny gate; run it anyway if `cargo-deny` happens to be installed — expected clean, but a missing binary is not a failure).

## Tests

This task ships no unit tests — the deliverable is the build skeleton, and its proof is the command matrix above (macOS-only: no crate links libcuda yet, and there is no CI — GitHub Actions is out of scope, so the matrix above IS the gate).

- **Before** (negative control, run after Step 5 — the root manifest exists but no crate manifest does yet): `cargo check --workspace` from `shisu/` MUST fail with `failed to load manifest for workspace member .../crates/shisu-core` — proves the member list is real. (Skip this control if Steps 2–8 were done in one pass; it is a manifest-integrity check, not a deliverable.)
- **After**: Step 12 + Step 13 all green.
- **Contract checks** (exact commands, run from `shisu/`):
  - `cargo tree -e no-dev | grep -c cudarc || true` → `0` (nothing pulls cudarc on macOS — the master plan's stated purpose of the per-target pattern).
  - `grep -rn "ATLAS" . --exclude-dir=target` → no output (SHISU_ rename complete, rule 5).
  - `rustup run "$(cat rust-toolchain.toml | sed -n 's/^channel = "//;s/"$//p')" rustc --version` → `rustc 1.93.1` (or simply `rustc --version` from `shisu/`).

## Acceptance Criteria

Matches master plan Task 1's checkboxes; each verifiable by command or file inspection (no project-wide gates — Task 20 owns clippy/deny/file-size final):

- [ ] `shisu/Cargo.toml` exists; `members` lists exactly the 8 crates; `shisu-kernels`/`shisu-cuda`/`shisu-ple` are absent from members and from disk (CUDA phase).
- [ ] Workspace deps present with the exact versions/features in Step 5 (serde_json preserve_order, thiserror 2, anyhow, parking_lot, axum 0.8, hyper 1, tokio, clap 4, tracing, objc2 stack, minijinja, tokenizers 0.23, safetensors, sha1, hex; cudarc + io-uring declared, unused).
- [ ] `shisu-core` workspace path dep has `default-features = false`; `shisu-core` features `default=[]/cuda/metal`; `shisu-metal` features `default=["metal"]/metal` (⚠ DEVIATION 2 documented above).
- [ ] `[workspace.lints]` = warnings deny + clippy all deny (atlas allow-list); every crate has `[lints] workspace = true` and `#![deny(warnings)] #![deny(clippy::all)]` in `lib.rs`; no SPDX headers (rule 7).
- [ ] `shisu/rust-toolchain.toml` pins 1.93.1 with rustfmt+clippy; `shisu/deny.toml` present (no AGPL/CDLA/allow-git entries).
- [ ] `cargo check --workspace --features metal` and `cargo check --workspace` pass on macOS from `shisu/`; `cargo test --workspace --features metal` exits 0.
- [ ] `shisu/.gitignore` contains `/target` and `git status` after the check shows no `target/` entries.

## Commit

From the repo root (Cargo.lock is committed — cargo-deny's bans check depends on it, atlas precedent `deny.toml:59-65`):

```bash
git add shisu/Cargo.toml shisu/Cargo.lock shisu/rust-toolchain.toml shisu/deny.toml \
        shisu/.gitignore shisu/crates shisu/scripts
git commit -m "chore: shisu workspace scaffold"
```

Expected: 25 files (root `Cargo.toml` + `Cargo.lock` + `rust-toolchain.toml` + `deny.toml` + `.gitignore` = 5, 8×(`Cargo.toml`+`src/lib.rs`) = 16, 2 scripts), working tree clean for `shisu/`.
