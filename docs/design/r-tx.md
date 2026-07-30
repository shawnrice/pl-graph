# R-TX — transactions & deferred constraint checks

Status: **v1 shipped** (foundation). Both engines, byte-identical.

## Why

Neither engine had any staging or rollback: writes applied eagerly, and a
mid-sequence failure left partial writes committed (round-7: a two-leg transfer
where leg 2 throws leaves the books unbalanced, `sum=1000`). And the built-in
constraints checked one write at a time, so cross-write invariants
(`debits==credits`) and "node + mandatory edge"–style rules were inexpressible.

R-TX adds an **atomic mutation boundary with rollback + deferred constraint
checks**. Because lenke is in-memory, single-writer, and synchronous, of the
classic ACID four only **Atomicity** and **Consistency** are in play — Isolation
is trivial (no concurrency) and Durability is the host's job (see the temporal
design note). So "transaction" here is deliberately smaller than a database
transaction manager: no MVCC, no isolation levels, no savepoints.

## Mechanism: eager-apply + undo-log + deferred-check-at-commit

Identical in both engines:

1. **Begin** — open a frame (a depth counter; nesting joins the outer frame, the
   outermost owns commit/rollback). Writes still apply immediately, so reads
   inside see their own writes — but each mutation records an **inverse op**.
2. **During** — the built-in required/type/unique gates do not throw per-write;
   they record the touched element and defer.
3. **Commit** — run the deferred checks against the fully-staged graph. On
   failure, roll the whole transaction back and surface
   `E_CONSTRAINT_VIOLATION`. On success, finalize and dispatch buffered events.
4. **Rollback** — replay the inverse ops newest-first; discard buffered events.

Chosen over snapshot/restore (O(graph) per transaction; the Rust `Graph` has no
`Clone`) and a staging overlay (invasive to every read path). The undo-log is
O(writes-in-transaction), and Rust's tombstone delete model makes inverses cheap
(undo of an insert = set the tombstone; undo of a delete = clear it in place, as
columns survive a delete). The frame allocates lazily, so a graph doing no
transactional writes pays nothing.

**Events** buffer during a transaction and dispatch as one batch on commit
(discarded on rollback), preserving the "an emitted event == a committed write"
contract that React reactivity and the sync `WriteLog`/CDC rely on. A buffered
event captures its reactive tokens at buffer time, because a removal evicts the
element before the event dispatches at commit.

**Per-statement atomicity** falls out of the same mechanism and unifies the
engines: every top-level write statement runs in an auto-commit frame, so a
faulting multi-row `INSERT`/`SET` rolls its earlier rows back instead of leaving
the write half-applied. This is Gremlin's "auto-commit per traversal" and the
fix for the round-7 partial-write bug, in one.

## Surface

The transaction surface is **engine-neutral and programmatic**, because the two
query languages sit at opposite ends: ISO GQL has _language-level_
`START TRANSACTION`/`COMMIT`/`ROLLBACK`, while Gremlin/TinkerPop has _no_
transaction language at all — only a host API (`graph.tx()`). So one host API
serves both:

- `graph.transaction(fn)` — runs `fn`, auto-commits on return, auto-rolls-back
  on throw, returns `fn`'s result. The common case.
- `graph.tx()` — a TinkerPop-style handle: `commit()` / `rollback()` explicitly.
- `beginTransaction()` / `commitTransaction()` / `rollbackTransaction()` — the
  primitives the above are built on.

The same surface exists on the native `RustGraph` (over the FFI/wasm
`lnk_begin_tx` / `lnk_commit_tx` / `lnk_rollback_tx` triple), so a transaction
behaves identically whether the store is TS or Rust.

## Also shipped

- **ISO GQL `START TRANSACTION [READ ONLY | READ WRITE]` / `COMMIT [WORK]` /
  `ROLLBACK [WORK]`** — a thin veneer over the primitives, both engines. The
  graph is the ISO session: transaction state persists across `query()` calls, so
  `query(g, 'START TRANSACTION')` … `query(g, 'COMMIT')` spans the statements
  between, with the per-statement auto-frames nesting inside. Contextual parsing
  (no new global reserved words); `READ ONLY` rejects a write statement; nesting a
  `START`, or `COMMIT`/`ROLLBACK` with no active transaction, is a coded error.

## Not in scope (follow-ups)

- ~~The R-CONSTRAINTS items that build on R-TX~~ — **SHIPPED**: edge-side
  constraints, min/`exactly one` cardinality, declarative (GQL-predicate)
  validators, and graph-level (cross-write) invariants all now use these deferred
  checks.
- Concurrency/MVCC, savepoints, and true nested (savepoint) transactions —
  nesting is flat in v1 (an inner rollback rolls the whole transaction back).

## Known v1 limitations

- `truncate()` is rejected inside a transaction (it can't be captured as a
  bounded undo-log — it would clone the whole graph).
- A restored edge is re-appended to its endpoints' adjacency, so _neighbor
  iteration order_ for a delete-then-rollback edge may differ from a
  never-touched graph. Serialization is edge-index-ordered and stays
  byte-identical.
