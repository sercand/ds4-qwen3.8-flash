# Task 7: shisu-kvstore — the radix-tree KV cache

**Spec:** `docs/superpowers/plans/2026-09-05-shisu-rust-port.md` § Task 7 (read it before executing; the spec argues this plan). This crate is **the** KV-cache system for every model on every platform (master Scope line "one KV-cache system for every model on macOS"): radix tree of token spans + refcounted paged KV + GDN/SSM checkpoint store + Marconi utility eviction, ported from ds4.c 55072-56265, with the ds4_kvstore disk store demoted to its spill/persistence tier.

**Depends on:** Task 1 only — workspace member `crates/shisu-kvstore` + `[workspace.dependencies]` entries `thiserror`, `parking_lot`, `tracing`, `sha1 = "0.10"`, `hex = "0.4"`, `tempfile = "3"` (dev) (task-01 Produces). No `shisu-core` dependency: the store is model-agnostic — geometry arrives as `CacheGeometry` (Deviation 2). First consumers: Task 6/8 (qwen35 engine), Task 17 (qwen4exp — same crate, no second bookkeeping), Task 10 (`/cache` stats), Task 13 (session page tables).

**Produces** (signatures later tasks rely on; the master-plan API words `lookup`/`commit`/`evict_to`/`spill`/`restore` are kept):

```rust
pub type PageId = i32;  pub type NodeId = u32;  pub type SlotId = u32;
pub struct SpillId(pub String);                       // 40-char lowercase sha1 hex

pub struct CacheGeometry {                             // geom.rs — nothing hardcoded per model
    pub page_tokens: u32,                              // 256 (shisu_core::q4e_page const, task-02 §Step 4)
    pub page_bytes: u64, pub ckpt_bytes: u64,          // derived by the engine, see Step 2
    pub prefill_s_per_token: f64, pub decay_s: f64,    // Marconi scale + idle decay
}
pub trait Backing: Send {                              // engine moves real bytes; store does bookkeeping
    fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError>;   // ds4 q4e_page_copy
    fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError>; // spill
    fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError>; // restore
}
pub struct PageTable { /* map: Vec<PageId>, len */ }   // Drop unrefs every holder it took
pub struct CkptRef { pub slot: SlotId, pub pos: u32, pub has_logits: bool }
pub struct Frontier { pub node: NodeId, pub table: PageTable }   // what a live session owns
pub enum ReuseSource { Cold, Live, Checkpoint, Tree }            // ds4 ds4_reuse_source
pub struct Lookup { pub node: NodeId, pub matched: u32, pub ckpt: Option<CkptRef>,
                    pub ckpt_pos: u32, pub source: ReuseSource }
pub struct Resume { pub frontier: Frontier, pub ckpt: Option<CkptRef>,
                    pub ckpt_pos: u32, pub matched: u32 }
pub struct CommitOpts<'a> { pub with_logits: bool, pub chunk_only: bool,
                            pub hints: &'a [u32], pub state: Option<&'a [u8]>,
                            pub logits: Option<&'a [u8]> }       // state = ckpt_bytes blob
pub struct EvictReport { pub pages_freed: u32, pub nodes_dropped: u32,
                         pub ckpts_dropped: u32 }
pub struct CacheStats { pub nodes: u32, pub pool_pages: u32, pub pages_used: u32,
    pub ckpt_used: u32, pub ckpt_capacity: u32, pub page_evictions: u64,
    pub ckpt_evictions: u64, pub spill: Option<SpillStats> }

pub struct KvCache;   // one parking_lot::Mutex inside — ds4's one mutex over tree+pool+slots
impl KvCache {
    pub fn open(geo: &CacheGeometry, backing: Box<dyn Backing>, pool_pages: u32,
                ckpt_slots: u32, spill: Option<SpillConfig>) -> Result<KvCache, KvError>;
    pub fn plan(&self, tokens: &[i32]) -> Lookup;               // pure query, takes no refs
    pub fn lookup(&self, tokens: &[i32]) -> Option<Resume>;     // plan + adopt pages + CoW
    pub fn commit(&self, fr: &mut Frontier, tokens: &[i32], q: u32,
                  opts: CommitOpts<'_>) -> Result<(), KvError>; // span + ckpt admission
    pub fn evict_to(&self, want_free_pages: u32) -> EvictReport;
    pub fn spill(&self, node: NodeId, text: &[u8]) -> Result<Option<SpillId>, KvError>;
    pub fn restore(&self, id: &SpillId) -> Result<Option<Restored>, KvError>;
    pub fn ckpt_state(&self, ckpt: &CkptRef) -> Vec<u8>;        // engine restores from this
    pub fn mark_live(&self, node: NodeId, stop: Option<NodeId>, delta: i32);
    pub fn next_admission(&self, fr: &Frontier, tokens: &[i32], hints: &[u32]) -> Option<u32>;
    pub fn stats(&self) -> CacheStats;
}
pub struct Restored { pub frontier: Frontier, pub tokens: Vec<i32>,
                      pub logits: Option<Vec<u8>> }             // logits = path-final row
```

`next_admission` is host-facing by design (ds4 `q4e_next_admission`, 67720–67756 — Source table row): T13's scheduler asks it for the next checkpoint stop before choosing a chunk end, and T6 snapshots the ckpt blob only when it answers `Some(new_end)` (Deviation 3's d2h cost). `commit` re-derives admission internally, so a host that never calls it stays correct — it just pays a blob snapshot on every commit.

`SpillConfig { dir: PathBuf, budget_mb: u64 }` → `SpillStore` (Step 8): `open` (budget 0 → 4096, one budget pass at open, `ds4_kvstore.c:609-645`), `write(sha, text, hdr, reason, payload)` atomic tmp+rename, `read(sha) -> Option<(Vec<u8> text, Vec<u8> payload)>` with re-hash guard, `touch(sha)`, `stats() -> SpillStats { dir, budget_bytes, total_bytes, entries }`. `Reason` enum + 48B header layout + scoring constants unchanged from the ds4_kvstore port (Step 8 table).

## Global Constraints

Master plan Rules 1–7, one line each:

1. All new code under `shisu/`; pure Rust — ds4.c is an algorithm reference, nothing copied verbatim except constants and the on-disk byte layout.
2. 1500-LoC cap per file (local gate `scripts/file_size_check.sh`); module split below already respects it.
3. The **utility formulas and the spill byte layout are the byte-for-byte artifacts**: tree/ckpt utility exactly as ds4.c 55447-55453 / 56115-56121; disk header exactly as `ds4_kvstore.c:393-415`.
4. No perf gate here; Marconi-beats-LRU is an ordering test, resume latency is recorded in Task 8/20.
5. Env knob: `SHISU_KVSTORE_HIT_HALF_LIFE_SECONDS` (u64 s, default `21600`, spill scoring only; invalid → default + `tracing::warn`). No other env reads.
6. `thiserror` `KvError { Io, PoolEmpty, PathGap, NodeRefused, SlotRefused, BadHeader, ShaMismatch, OverBudget { required, budget } }`; `parking_lot::Mutex`; `tracing` for evict/claim logs (ds4 messages at 56165-56167, 67644-67650).
7. No license headers.

Task-specific:
- Node storage is an arena (`Vec<TreeNode>` + `NodeId` indices, `u32::MAX` = none) — Deviation 1. No `unsafe`, no `Rc<RefCell>`.
- Timestamps: tree/ckpt `hit` uses a monotonic clock **passed in by the caller** (`now: f64` params, as ds4 threads `now_sec()`); spill header `created_at/last_used` are wall-clock unix seconds (ds4 `time(NULL)`).
- The store never touches GPU/Metal buffers: page bytes move only through `Backing`; ckpt state is an opaque `ckpt_bytes` blob the engine snapshots/restores — Deviation 3.
- Pure `std::fs`; no io-uring (ds4_kvstore.c uses stdio — verified zero `uring` hits); only `#[cfg(unix)]` dir-mode `0o700` allowed.

## Source References (verified)

Every ds4.c line below was opened while writing this plan. The ds4_kvstore rows were verified when the previous revision of this plan was written and remain valid for the spill tier.

| Source | Lines | What lives there / how it is used |
|---|---|---|
| `ds4.c` | 55072–55119 | Part-1 design comment: pages immutable past the frontier, resume-inside-a-page copies via `q4e_page_copy`, refs = tree nodes + live page tables, Marconi rationale ("LRU throws away the large SSM entries that are worth the most"), pages equal-value → recency, leaves before interior, ckpt value = prefill from nearest lower ckpt. The doc comments of `tree.rs`/`evict.rs` restate this. |
| `ds4.c` | 55121–55127 | `Q4E_PREFILL_S_PER_TOKEN 0.0014`, `Q4E_PAGE_MIB 7.5`, `Q4E_CKPT_MIB 113.0`, `Q4E_CACHE_DECAY_S 600.0` — q4e's per-model scale; only `decay_s` is model-agnostic (Step 2). |
| `ds4.c` | 55129–55207 | `q4e_page_pool` (free_ids stack + refs), take/ref/unref, `reverse` init flag (contiguous vs scrambled layout — the fingerprint test hook) → `pool.rs`. |
| `ds4.c` | 55209–55265 | `q4e_page_table` + `grow` (returns first_new/n_new, refuses when pool short) + `from_path` (deepest node wins on the one shared boundary page) → `page_table.rs`. |
| `ds4.c` | 55142–55161, 55267–55341 | `q4e_tree_node` fields, `q4e_span_tree`, init/free (iterative post-order), `walk` (partial-span match, `matched` out), `ckpt_below`, `lower_ckpt_pos` → `tree.rs`. |
| `ds4.c` | 55343–55441 | `relink`, `split` (lower half gets children? NO — upper keeps children+ckpt; straddled page double-held), `add` (first-token uniqueness guard, `pg_hi > table.len` refuse) → `tree.rs`. |
| `ds4.c` | 55443–55504 | `node_utility` (saved_s / (pages·MiB) / (1+age/decay)), `pick_victim` (leaf ∧ live==0, `extra` pricing hook), `drop` (refuses child/ckpt/live), `mark_live` (bounded to `stop`) → `evict.rs`/`tree.rs`. |
| `ds4.c` | 55506–55626 | `ds4_test_q4e_page_table` + `ds4_test_q4e_span_tree` — ported scenario-for-scenario as Rust tests (Step 9). |
| `ds4.c` | 55628–55636 | width shorthands; `Q4E_GDN_IN = 2·K + V` — the ckpt conv-window width formula (Step 2). |
| `ds4.c` | 55892–55955 | Part-2 design comment (ckpt = everything the recurrent stack cannot rewind; cache owns pool/tree/slots, graph borrows); `Q4E_CKPT_DEFAULT 40`/`MAX 256`, pool caps 600k tokens, `Q4E_ADMISSION_STRIDE 2048` → ckpt caps + admission stride. |
| `ds4.c` | 55957–56002 | `q4e_ckpt` slot fields (used/ready/speculative/pos/owner/logits_valid/pend/hit) and `q4e_cache` (one mutex, page_bytes, ckpt_evictions) → `ckpt.rs`. |
| `ds4.c` | 56008–56043 | `q4e_page_bytes`/`q4e_ckpt_bytes` — the *derivation pattern* Step 2 copies (count attn/gdn layers from the shape, never restate totals). |
| `ds4.c` | 56045–56109 | ckpt budget (explicit-or-derived, 0 disables), `ckpt_copy/alloc/free` — lazy slot alloc, cap shrink on OOM. |
| `ds4.c` | 56111–56197 | `ckpt_utility` (distance to lower ckpt / MiB / (1+age/decay)), `release`, `claim` (free slot → else evict speculative pass 0 → least-useful pass 1, only when a rule asked), `node_ckpt_utility` extra-pricer → `ckpt.rs`/`evict.rs`. |
| `ds4.c` | 56199–56230 | `cache_free_pages` — victim loop with progress guarantee (stale-ckpt spin story) → `evict_to`. |
| `ds4.c` | 56232–56265 | `q4e_page_copy` — every KV row of one page; the `Backing::copy_page` contract. |
| `ds4.c` | 67446–67527 | `plan_locked`/`cache_plan` — live-vs-checkpoint-vs-tree source decision, prompt-final ckpt needs `logits_valid`, image-token clamp (N/A: no vision in shisu — skip, note in code). |
| `ds4.c` | 67529–67610 | `cache_resume` — pin target **before** detach (use-after-free story), `from_path`, CoW of the interior page when `ckpt_pos & MASK != 0`, ckpt state copy, logits/pend restore → `lookup`. |
| `ds4.c` | 67612–67718 | `cache_commit` — walk-down to `matched_end` re-walking (no stale node pointers), `tree_add` refuse → rc 2 "lost span" warn, ckpt claim+fill, `speculative` flag; `commit_interrupted` (rule 5: cancelled prefill commits its frontier so retries converge). |
| `ds4.c` | 67720–67756 | `next_admission` — stop positions: sequence end, hints (message boundaries), branch point (`matched_end`), and stride-2048 multiples only while a slot is free → `CommitOpts.hints` + `chunk_only`. |
| `ds4.c` | 67771–67816 | `cache_stats` (entries/capacity/evictions/pool/tree_nodes) → `CacheStats`; `cache_path` (tokens/pages/page_bytes/state_bytes of the live path) → what `spill` serializes. |
| `ds4.c` | 67823–67848 | Disk-tier comment + payload layout: `u32 h[19]` header (magic `Q4EP`, version, page_bytes, tokens, page_tokens, pages, kv_width, layers, attn/gdn layers, idx/mtp flags, vocab, conv-window, state-matrix, ple-window, hc-width, pend flag/pos0) then tokens, logits, state, pages → `spill.rs` payload. |
| `ds4.c` | 68013–68080 | `payload_save` — logits-freshness gate (`logits_pos != pos` refuse), header array, tokens, logits, state, pages → `spill` writer. |
| `ds4.c` | 68082–68148 | `payload_check` — rc 2 = wrong build (discard), rc 1 = wrong-as-configured (keep); per-field geometry table → `restore` validator. |
| `ds4.c` | 68150–68316 | `payload_load` — walk+split at the match, pin-before-detach, adopt prefix, `own_from` (the match-interior page is re-read, not copied), grow, state+pages read, then `commit` the restored span (rc-2 warn path) → `restore`. |
| `ds4_q4e_page.h` | 1–47 | 256-token page geometry + why 256; 30 KiB/position = 7.5 MiB/page is **q4e-only** (12 attn layers incl. indexer+MTP planes) — never reused for qwen35. |
| `ds4_kvstore.h` | 11–13 | `DS4_KVSTORE_FIXED_HEADER 48u`, `DS4_KVSTORE_DEFAULT_MB 4096`, `DS4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull*60ull*60ull)` — spill-tier consts copied. |
| `ds4_kvstore.h` | 15–28 | `DS4_KVSTORE_EXT_*` flags + `ds4_kvstore_reason` enum 0–6 — copied (`Reason`, ext flags). |
| `ds4_kvstore.c` | 25–56 | magic `KVC`, `KV_CACHE_VERSION 1`, `KV_CACHE_PAYLOAD_ABI 2`, `MIN_EFFECTIVE_HITS 0.01`, continued-prefix factors 0.05/0.45, `ANCHOR_REASON_SCORE_FACTOR 2.0` — spill scoring consts verbatim. |
| `ds4_kvstore.c` | 174–189 | `reason_code` strings + `key_kind` — `Reason::from_str`/`key_kind`. |
| `ds4_kvstore.c` | 312–370 | sha1-hex name filter (40 hex + `.kv`), `path_for_sha`, `kv_mkdir_p` 0700 — ported (hand-rolled SHA-1 :222–310 replaced by `sha1` crate). |
| `ds4_kvstore.c` | 393–440 | `fill_header`/`read_header` — **the 48-byte layout** (table in Step 8), validity gate `tokens != 0 && quant_bits ∈ {2,4}`. |
| `ds4_kvstore.c` | 485–502 | `touch_file` — hits+1, `last_used = time(NULL)` header rewrite. |
| `ds4_kvstore.c` | 504–607 | spill eviction scoring (`effective_hits = hits·exp2(-elapsed/half_life)`, density `× anchor × continued-prefix factor`) + `evict` (rescan, lowest-score, tie-break older `last_used`) — ported into `spill.rs`. |
| `ds4_kvstore.c` | 609–645 | `open` — mkdir, budget default 4096 MB, initial evict pass. |
| `ds4_kvstore.c` | 803–844 | `file_size_bytes`/`budget_required` (+1 % ceil slack)/`file_size_fits`. |
| `ds4_kvstore.c` | 846–897, 1032–1048, 1073–1204 | existing-file compat check, already-stored fast path, pre-write budget check → evict → atomic `tmp.<pid>` + rename with cleanup — ported order exactly. |

⚠ **DEVIATION 1 (arena, not pointers):** ds4 links nodes with `parent/child/next` pointers; the port uses arena indices. Same topology and algorithms; every pointer op has an index twin. `Drop` for `KvCache` frees the arena; `PageTable::Drop` unrefs.

⚠ **DEVIATION 2 (geometry injected):** ds4 derives `page_bytes`/`ckpt_bytes` from global shape macros (56012-56043); the crate takes them in `CacheGeometry` so qwen35 (Task 6) and q4e (Task 17) each derive theirs from `ShapeProfile`. The crate stores **no** model constants.

⚠ **DEVIATION 3 (host-side ckpt blobs):** ds4 keeps each slot's GDN tensors device-resident and copies on-device (56060-56080). The port stores opaque host blobs; the engine snapshots/restores across the device boundary (qwen35 ckpt ≈ 51.2 MiB → single-digit ms on M1 Max). Revisit only if Task 20's resume-latency gate demands it.

⚠ **DEVIATION 4 (decay formula):** the master-plan Task 7 line paraphrases utility as `exp2(-idle/600)`. ds4's tree/ckpt utility is hyperbolic — `saved / MiB / (1.0 + age/600.0)` (55452, 56120). Port the **ds4 formula**; `exp2` decay stays only where ds4 uses it, the spill-tier `effective_hits` scoring (`ds4_kvstore.c:532-559`).

⚠ **DEVIATION 5 (no image clamp):** `plan_locked`'s image-token clamp (67466-67483) guards a vision path shisu does not ship; omitted with a comment citing those lines.

## Plan

- [ ] **Step 1: `Cargo.toml`** — deps `{ workspace = true }`: `thiserror`, `parking_lot`, `tracing`, `sha1`, `hex`; dev-dep `tempfile`. `lib.rs`: Task 1 deny attrs + `mod {geom,pool,page_table,tree,ckpt,evict,spill,error};` + flat re-exports.

- [ ] **Step 2: `geom.rs` — `CacheGeometry` + derivation contract.** The engine derives; the crate validates (`page_tokens == 1 << log2`, `page_bytes > 0`, `ckpt_bytes > 0`, `decay_s > 0`). Documented derivations (formula shape from ds4.c 56012-56043, inputs from `ShapeProfile`):
  - **qwen35 page_bytes** = n_attn × 2(K,V) × page_tokens × (kv_heads × attn_head_dim) × 2 B(f16) = 8 × 2 × 256 × (4 × 256) × 2 = **8,388,608 B = 8.0 MiB**. (8 attn layers = 32/4; head_dim **256** per task-02's config-verified DEVIATION, not the master plan's 128; KV f16 per task-06's buffers.)
  - **qwen35 ckpt_bytes** = n_gdn × ((conv_kernel−1) × gdn_in + v_heads × gdn_head_dim²) × 4 B + vocab × 4 B, with `gdn_in = 2·(qk_heads·hd) + v_heads·hd` (ds4.c:55635) = 2·2048 + 4096 = 8192 → 24 × (3×8192 + 32×128×128) × 4 + 248320×4 = 52,690,944 + 993,280 = **53,684,224 B ≈ 51.2 MiB** (not q4e's 113 — q4e adds PLE + HC + draft-row terms, ds4.c:56040-56042; qwen35 has neither).
  - **q4e page_bytes** = 7,880,704 B derived by Task 17 from the same formula incl. indexer + MTP planes (ds4.c 56012-56026; task-17 §"Source References" row `ds4.c 56008–56043`); the 7.5 MiB header figure excludes them.
  - `prefill_s_per_token`: q4e 0.0014 (ds4.c:55124); qwen35 = measured prefill s/token from `shisu-bench/baseline/m1max-qwen35.json` at Task 6 time (recorded, not guessed). `decay_s` = 600.0 both (ds4.c:55127).

- [ ] **Step 3: `pool.rs`** — `PagePool { free_ids: Vec<PageId>, refs: Vec<u32> }`; `init(n, reverse)` (55168-55183 rationale comment), `take` (refs=1), `ref_`, `unref` (last holder → free stack), `n_free`. Unit tests = ds4_test_q4e_page_table's pool half (55512-55542).

- [ ] **Step 4: `page_table.rs`** — `PageTable { map: Vec<PageId> }`; `grow(pool, n_pos) -> Option<(first_new, n_new)>` (refuses when short, 55220-55234), `from_path(tree, node, n_pos)` (deepest-node-wins, `PathGap` error when a page is unowned, 55236-55265), `Drop` → unref. Boundary-page rule (child's copy wins) gets its own test.

- [ ] **Step 5: `tree.rs`** — arena `TreeNode { parent, child, next, start, end, tok: Vec<i32>, pages: Vec<PageId>, ckpt: SlotId-sentinel, live, hit }`; `walk` (partial match → `matched`), `split` (upper keeps children+ckpt, straddled page ref'd into both, 55351-55402), `add` (first-token uniqueness + `pg_hi > table.len` → `NodeRefused`, 55404-55441), `relink`, `drop_leaf` (refuses child/ckpt/live, 55481-55493), `mark_live(node, stop, ±1)` (55495-55504), `ckpt_below`, `lower_ckpt_pos`, `is_ancestor`.

- [ ] **Step 6: `ckpt.rs`** — `CkptStore { slots: Vec<Slot>, cap, evictions }`, `Slot { used, speculative, pos, owner: NodeId, has_logits, pend: Option<Vec<u8>>, hit, state: Vec<u8> }` (pend row is q4e-only; opaque bytes). `claim(allow_evict, now)`: lowest free (lazy `state` alloc; OOM → shrink cap once, warn 56156-56169) → pass 0 evict least-useful **speculative** → pass 1 least-useful, only when `allow_evict` (rules 1-3 asked; a chunk boundary never evicts, 56141-56189). `release` (owner-clearing rule + the stale-owner story comment), `utility(k, now)` = `(pos − lower_ckpt_pos)·prefill_s_per_token / (ckpt_bytes/MiB) / (1 + age/decay)` (56111-56121).

- [ ] **Step 7: `evict.rs` + facade ops.** `node_utility(n)` = `(end−start)·prefill_s_per_token / (pages·page_bytes_MiB) / (1 + age/decay)` (55443-55453); `pick_victim(extra: Option<&dyn Fn(NodeId) -> f64>)` — leaves with `live == 0`, least utility, `extra` = the ckpt at its end priced on the same scale (55455-55479, 56191-56197); `evict_to(want)` = ds4's `cache_free_pages` loop incl. the progress guarantee (56199-56230). Facade: `plan` = `plan_locked` port (live frontier vs ckpt-vs-tree source, prompt-final ckpt requires `has_logits`, 67449-67513); `lookup` = `cache_resume` port (pin-before-detach comment 67548-67553, `from_path`, CoW via `Backing::copy_page` when `ckpt_pos % page_tokens != 0` 67561-67580, `hit` stamps); `commit` = `cache_commit` port (re-walk to `matched_end`, never trust stale nodes, `tree_add` failure → `tracing::warn` + span lost, ckpt claim+fill with `state`/`logits`, `speculative = chunk_only && !refresh`, 67620-67688); admission stop positions = `next_admission` (hints ∧ branch point ∧ stride-2048-while-spare, 67731-67756) exposed as `pub fn next_admission_pos(len, pos, hints, spare) -> u32` for Task 13's quanta loop; `commit_interrupted` semantics = plain `commit(.., with_logits = frontier_has_logits, chunk_only = false)` (rule 5, 67690-67718).

- [ ] **Step 8: `spill.rs`** — `SpillStore` keeping the previous revision's ported disk format intact: 48-byte header byte-exact (`ds4_kvstore.c:393-415`; offsets 0–2 magic `KVC`, 3 version, 4 quant_bits, 5 reason, 6 ext_flags, 7 model_id, 8 tokens, 12 hits, 16 ctx_size, 20 payload_abi, 21–23 reserved, 24 created_at, 32 last_used, 40 payload_bytes; then `text_len u32 ‖ text ‖ payload`), sha1-hex `.kv` names, atomic tmp+rename, budget pass with the `effective_hits·density` scoring (504-607), `touch`, `stats`. On top: `encode_path(tokens, logits, state, pages_bytes) -> Vec<u8>` and `decode_path` for the `Q4EP` payload (h[19] layout per ds4.c 67835-67848; qwen35 writes 0 for the indexer/MTP/PLE/hc fields; `decode` validates every field and returns `rc 1` vs `rc 2` semantics per 68082-68093). `KvCache::spill(node, text)`: refuse (`Ok(None)`) unless the node's ckpt has logits (68019-68030); serialize root..node tokens, ckpt state+logits, and every path page via `Backing::page_to_host`; key = `sha1(text)`; `Reason::Cold` for engine-initiated spills, `Shutdown` for the exit sweep. `KvCache::restore(id)`: read → validate → walk+split at the match → pin-before-detach → adopt prefix → fresh pages above `own_from` (the match-interior page is re-read from the file, not copied — 68229-68236) → `page_from_host` → `commit` the span (68311).

- [ ] **Step 9: tests** (see Tests) — port `ds4_test_q4e_page_table` (55506-55547) and `ds4_test_q4e_span_tree` (55549-55626) scenario-for-scenario (same numbers: 700-token paths, 300-prefix, 3+2+2 pages, victim order).

## Tests

`cargo test -p shisu-kvstore` — pure host code, no model, no GPU, no `#[ignore]`, macOS == Linux.

- **Ported ds4 scenarios** (`tests/tree.rs`): the two test functions above, exact assertions (refcounts 2→1 across split, straddled page in both halves, `pick_victim` recency order A-before-B, interior node not a victim while a child lives, drop→leaf cascade, pool fully returned).
- **Refcount invariants** (`tests/pool.rs`): grow/take/ref/unref sequences incl. full-pool refusal, second-holder keeps a page out of the free stack, `PageTable::Drop` returns everything.
- **CoW on resume** (`tests/lookup.rs`): fake `Backing` (Vec<u8> pages); ckpt at a non-multiple position → `lookup` copies exactly one page (assert `copy_page` call count == 1, source page refs unchanged); ckpt at a 256-multiple → zero copies; pin-before-detach: with the pool at capacity, `lookup` on a tree whose target leaf is the only droppable node still succeeds (the victim scan must not drop the target — 67548-67553).
- **Marconi beats LRU** (`tests/evict.rs`): workload = one long path with a lone ckpt at 20k + several short recent leaves; under page pressure the leaves go and the ckpt survives (LRU would drop it); a ckpt 300 tokens above its neighbour is evicted before one 20k away (55112-55115 numbers); `claim(allow_evict=false)` never evicts; speculative slots evict before rule-earned ones (56141-56150).
- **Admission positions** (unit): hints/branch/stride interaction per 67731-67756 — stride positions only while a slot is free, absolute multiples, never inside a chunk.
- **Spill/restore byte-exact** (`tests/spill.rs`): `tempfile::TempDir`; a 48-byte header fixture (hand-built hex, magic `4b 56 43 01`, gate `tokens != 0 && quant_bits ∈ {2,4}`) round-trips; `encode_path`→`decode_path` round-trips a synthetic path (tokens, logits, state, page bytes) and the restored tree yields the same `lookup` result as the original; a file whose h[2] (page_bytes) disagrees → `rc 1` keep, wrong magic → `rc 2` discard; budget pass evicts lowest-score first, tie-break older `last_used`; atomic write leaves no `.tmp.*` on injected failure.
- **Env knob**: `SHISU_KVSTORE_HIT_HALF_LIFE_SECONDS` in its own integration-test file (own process); invalid/absent → 21600 + warn.

## Acceptance

- [ ] `shisu-kvstore/src/{lib,error,geom,pool,page_table,tree,ckpt,evict,spill}.rs` exist; no file > 1500 LoC (`scripts/file_size_check.sh`).
- [ ] Public API matches Produces (master-plan words `lookup`/`commit`/`evict_to`/`spill`/`restore` present).
- [ ] Utility formulas byte-equal in structure to ds4.c 55447-55453 / 56115-56121 (hyperbolic decay, Deviation 4); spill header byte-identical to the `ds4_kvstore.c:393-415` fixture.
- [ ] No model constants in the crate: `grep -nE '2560|248320|8192|113|7\\.5' crates/shisu-kvstore/src` hits only doc comments citing derivations.
- [ ] `cargo test -p shisu-kvstore` green on macOS; same suite green on Linux unchanged.
- [ ] `SHISU_KVSTORE_HIT_HALF_LIFE_SECONDS` honored; no other env reads; no `ATLAS_*`/`DS4_*` env strings; no license headers; thiserror/parking_lot/tracing per rule 6.

## Commit

```
git add shisu/crates/shisu-kvstore shisu/Cargo.toml shisu/Cargo.lock
git commit -m "feat: shisu-kvstore radix-tree KV cache (paged KV + GDN checkpoints + Marconi + disk spill)"
```

(Master plan Task 7 says "Commit." with no message; message follows the Task 1 `type: summary` convention.)
