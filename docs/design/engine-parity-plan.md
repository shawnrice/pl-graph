# From-scratch engine: the path to parity

`crates/lenke-engine` is, today, a proof-of-architecture: one neutral IR that GQL
and Gremlin compile into, executing a read-only subset end to end, competitive
with `lenke-core` on aggregation/grouping and level on traversal. This document
is the honest build order from that slice to a real engine — capability parity
with `lenke-core`.

Rules the loop follows, every slice:

- One slice per iteration: implement, test with HAND-COMPUTED expected values
  (never trust the code to define correctness), `cargo test`, then
  `cargo clippy --all-targets -- -D warnings` (check ITS OWN exit code, do not
  chain), then `cargo fmt`, then commit.
- For a parser change, cross-check parse→run against the equivalent hand-built
  `Plan`.
- `value.rs` stays the single home for order/equality/coercion/null/NaN — never
  restate a rule in an operator.
- Do NOT touch `lenke-core`; do not push.
- Stop and surface if a genuine SEMANTIC fork appears — do not guess.

**Out of scope for this loop:** rebuilding the pure-TS engine. The design keeps
the TS engine and has the two AGREE via a conformance suite (Phase J), not a
second rewrite.

## Build order

Legend: `[ ]` todo · `[~]` in progress · `[x]` done (commit).

### Phase A — Mutation & transactions

- [x] A1. Store mutation primitives: `add_node`, `add_edge`, `set_prop`,
      `remove_prop`; edge-type interning; per-edge id counter. (columns grow with the
      node set; typed columns promote to `Gen` on a type change; `eid` monotonic.)
- [x] A2. Deletion: `delete_edge`, `delete_node` (tombstone + adjacency cleanup).
      (tombstone bitmap; dense ids never reused; mirrors detached by eid; scans
      skip via all_nodes/label buckets — no exec change needed.)
- [x] A3. Transactions: undo-log wrapper, `commit`/`rollback`, per-statement
      atomicity. (each mutation records its inverse; begin/commit/rollback +
      savepoint/rollback_to + transaction(); deferred checks & events = Phase H.)

### Phase B — Write statements in both languages

- [ ] B1. GQL `INSERT` (node & edge patterns) → store writes.
- [ ] B2. GQL `SET` / `REMOVE`.
- [ ] B3. GQL `MERGE` / `_MERGE` keyed upsert.
- [ ] B4. Gremlin `addV`/`addE`/`property`/`drop`.

### Phase C — Persistence

- [ ] C1. NDJSON egress (nodes + edges).
- [ ] C2. NDJSON ingest.
- [ ] C3. Schema/snapshot round-trip.

### Phase D — Indexes & planner seeding

- [ ] D1. Property (hash) index + planner seek on `=` / inline `{k:v}` / `$param`.
- [ ] D2. Range index + seek on `<,<=,>,>=` and `BETWEEN`.
- [ ] D3. Edge-type index; interval/temporal index.

### Phase E — Expression & function surface

- [ ] E1. Arithmetic (`+ - * / %`, unary) in IR + eval + both parsers.
- [ ] E2. Scalar functions (abs, sqrt, floor/ceil/round, coalesce, …).
- [ ] E3. `CASE` / conditional.
- [ ] E4. String & list functions.
- [ ] E5. `CAST` + cross-type coercion semantics.
- [ ] E6. 3VL completeness: `IS NULL`, property-exists, `AND`/`OR`/`NOT` gaps.

### Phase F — Query surface

- [ ] F1. GQL `WITH` (chained query parts, carry + aggregate + filter).
- [ ] F2. GQL `EXISTS { … }` subquery.
- [ ] F3. GQL `CALL` (named procedure + inline correlated subquery).
- [ ] F4. Path values: `ANY SHORTEST p = …`, accessors (length/nodes/rels).
- [ ] F5. Gremlin step breadth (select, where(P), order(local), groupCount, …).

### Phase G — Data model

- [ ] G1. Temporal values + typed temporal columns.
- [ ] G2. Map/record values (storage, dotted-path, construction, access).
- [ ] G3. Numeric edge-case parity audit against `value.rs`.

### Phase H — Semantics services

- [ ] H1. Constraints / validators (commit-time checks).
- [ ] H2. Events / CDC (observation-only notifications).
- [ ] H3. Typed nodes (host-side schema validate-before-write).

### Phase I — Algorithms & egress

- [ ] I1. Graph algorithms: degree, WCC, label-prop, PageRank, shortest-path.
- [ ] I2. Arrow IPC egress.

### Phase J — Agreement

- [ ] J1. Conformance suite: run matched shapes on `lenke-engine` and
      `lenke-core`, assert same results (agreement, not byte-identity).

## Standing

Update this file as slices land — tick the box and note the commit. The loop is
done when every box is `[x]` and the conformance suite (J1) is green.
