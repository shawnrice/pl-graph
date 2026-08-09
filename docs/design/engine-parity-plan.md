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

- [x] B1. GQL `INSERT` (node & edge patterns) → store writes. (`Plan::Insert`
      leaf + mutable `exec::execute`; nodes with labels+inline props, directed
      typed edges, per-INSERT variable scoping. NOTE: edge PROPERTIES deferred —
      the store has no edge-property model yet; needs its own slice (see B-note).)
- [x] B2. GQL `SET` / `REMOVE`. (`Plan::Update` over a read sub-plan; collect
      writes in a read pass then apply; `MATCH … WHERE … (SET k=expr | REMOVE k)+`;
      SET null stores a present null, REMOVE deletes — the null policy. Node
      properties only; edge props await B5. RETURN-after-update deferred.)
- [x] B3a. Unique-constraint primitive: `create_unique_constraint(label, keys)` +
      INSERT-time enforcement (a plain INSERT that violates it errors, rolled back
      via the txn). `execute` now returns `Result<Rows,String>`. Key equality uses
      group semantics (NULL keys collide — matches null-first-class, not SQL). B3b
      helpers `unique_keys_for`/enforcement seam ready.
- [x] B3b. GQL `_MERGE` keyed upsert (sigil convention): node form, key inferred
      from the unique constraint at execute time, default clobber / `_ON_CREATE` /
      `_ON_UPDATE [WHERE]` / `_ON_UPDATE_NOTHING`, txn-wrapped. NOTE: key inference + no-constraint error happen at execute (parser has no store) not parse —
      still an error, just later. Params ($x), edge form, multi-hop v2 deferred.
- [x] B4. Gremlin `addV`/`property`/`drop` over the same `execute` path (addV→
      Insert; property folds into addV or wraps a read traversal in Update;
      drop→node delete via new `SetOp::Delete`). Read-after-write is rejected.
      `addE` DEFERRED to B5 (needs V(id)/from/to vertex-refs + the edge model).
- B5. Relationship variables + edge properties (discovered in B1), split:
  - [x] B5a. Store edge-property model: `set_edge_prop`/`edge_prop`/
        `remove_edge_prop`/`has_edge_prop` keyed by `eid`, undo-logged
        (RestoreEdgeCell). Boxed key→(eid→value) map, not columnar (edges are
        cooler); dead-eid props linger on delete (safe — eids never reused).
  - [x] B5b. Bind the edge as a slot in `Expand` (opt-in `bind_edge` → `Col::Edges`
        frontier; edge slot at W, node at W+1); `Prop` on an edge slot reads
        `store.edge_prop`. `for_each_nbr` now yields `(nbr, eid)`; width/chain_width
        count +2 for a bound edge; node-only Expand unchanged (default false).
  - [x] B5c. GQL edge language surface: `[r:T]` relationship-variable binding →
        `expand_edge` (r at slot W, node W+1); `r.key` in RETURN/WHERE; inline
        `[:T {props}]` = edge props in INSERT and a match filter in MATCH; SET/
        REMOVE on a bound `r` writes edge props. Rel var on var-length rejected.
  - [x] B6. Gremlin `addE` (Plan::AddEdge leaf): `g.V(a).addE('T').to(V(b))` and
        `g.addE('T').from(V(a)).to(V(b))`, with `.property(k,v)`; minimal `V(<id>)`
        arg supported only as an addE anchor. Out-of-range/deleted endpoint errors;
        missing to/from is a parse error. General `V(id)` read traversals deferred.

### Phase C — Persistence

- [x] C1. NDJSON egress (nodes + edges): dependency-free hand-rolled JSON writer
      (`ndjson::to_ndjson`); one object per live node `{id,labels,props}` then per
      edge `{from,to,type,props}`; deterministic (nodes by id, keys sorted);
      NaN/Inf→null. Serializes only — value semantics stay in value.rs.
- [x] C2. NDJSON ingest (`ndjson::from_ndjson`): hand-rolled dependency-free JSON
      parser; loads node then edge lines. Ids are NOT preserved — file ids may be
      gapped (deletions), so nodes get fresh dense ids and edges are remapped.
      Round-trip is exact for a gap-free dump, stable from the first reload
      otherwise. Round-trip + hand-parse + remap + value-kinds + error tests.
- [x] C3. Schema/snapshot round-trip: `dump_schema` emits unique constraints as
      leading `{"schema":"unique",…}` lines; `snapshot` = schema-then-data;
      `load_snapshot`/`from_ndjson` apply schema BEFORE data (empty store, always
      valid) so reloaded INSERT-enforcement matches. Round-trip preserves data AND
      constraints (a reloaded constraint still rejects a violating INSERT).

### Phase D — Indexes & planner seeding

- D1. Property (hash) index + planner seek, split:
  - [x] D1a. Store hash index (`create_index(key)`, key-only, maintained through
        the mutation primitives so rollback stays consistent; `index_lookup`) + IR
        `IndexSeek{label,key,value}` executed as index-or-scan with EXACT `=`
        semantics (NaN/null match nothing; group_key == equals for finite non-null;
        candidates intersect the label). NOTE: label-scoped index (true
        O(candidates), avoiding the label HashSet build) is a perf follow-up.
  - [x] D1b. Optimizer rule: `Scan(Some(label))+Filter(prop = lit)` → `IndexSeek`
        for BOTH spellings (`seek_target` matches `prop = v` and `v = prop`);
        semantic no-op, rows preserved. Test asserts both spellings land the SAME
        seek target (equivalent-spellings invariant); ranges & unlabelled scans not
        seeded. Composes with pushdown (a pushed eq-filter over a labelled scan
        seeds too).
- D2. Range index + seek on `<,<=,>,>=` and `BETWEEN`, split:
  - [x] D2a. Store range index (`create_range_index`, BTreeMap keyed by `OrdVal`
        = `cmp_total` order, non-null values only; maintained through the
        primitives so rollback stays consistent; `range_lookup`) + IR
        `RangeSeek{label,key,op,value}` executed index-or-scan, matching Filter's
        `cmp_total` ordering exactly (null operand drops; NaN greatest; cross-type
        by rank — this engine's total order, noted vs lenke-core's throw for J1).
  - [x] D2b. Optimizer rule: `Scan(Some(label))+Filter(prop <op> lit)` → `RangeSeek`
        for a range op, BOTH spellings via `flip_range` (`n > 5` and `5 < n`
        normalize to `prop > 5`), same target + rows-preserved test. Composes with
        pushdown. NOTE: BETWEEN / conjunction-splitting (seed one bound of an
        `And`, keep the other as a Filter) deferred — GQL has no BETWEEN keyword
        yet and `x>=lo AND x<=hi` is an And filter; a small follow-up.
- D3. Edge-type index; interval/temporal index — RELOCATED. The interval index
  is a bitemporal edge structure that needs temporal values (Phase G1), so it is
  moved to G4 (after temporal). The edge-type index is a niche adjacency
  optimization (expand already walks adjacency correctly) tracked as G5. Phase D's
  correctness-relevant work (hash + range seek) is complete.

### Phase E — Expression & function surface

- [x] E1. Arithmetic (`+ - * / %`, unary minus) — IR `Expr::Arith{op}` + eval
      (finite-or-null via `value::as_num`; NULL/non-numeric/non-finite operand OR
      result → NULL) + GQL parser (new `+ / %` tokens; precedence add<mul<unary;
      unary `-x` desugars to `0 - x`). Gremlin inline arithmetic N/A (its subset
      has no arithmetic expressions — `math()`-style steps deferred).
- [x] E2. Scalar functions: IR `Expr::Call{name,args}` + GQL `name(args)` in
      primary() (name/arity validated at parse; no collision with aggregates).
      Deterministic numerics abs/sign/floor/ceil/round/sqrt (finite-or-null via
      `value::as_num`; `sign(0)=0`) + variadic `coalesce` (first non-null). Every
      Expr match got the Call arm. Transcendentals (exp/ln/sin/… + power last-ulp)
      deferred to J1 per numeric-determinism.md.
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
- [ ] G4. Interval/temporal edge index (relocated from D3; needs G1): RI-tree-style
      as-of/overlap seek over temporal edge bounds.
- [ ] G5. Edge-type index (relocated from D3): adjacency grouped by edge type for
      O(matching-degree) type-filtered expand — an optimization, not correctness.

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
