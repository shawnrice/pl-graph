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
- [x] E3. `CASE` / conditional: IR `Expr::Case{branches, otherwise}` + eval (first
      branch whose condition is literally TRUE — three-valued, FALSE/NULL skip —
      else ELSE, else NULL) + GQL searched form `CASE (WHEN e THEN e)+ [ELSE e]
END` (WHEN/THEN/ELSE/END contextual keywords). Case arm added to every Expr
      match. Simple form `CASE x WHEN v …` deferred (would desugar to `x = v`).
- E4. String & list functions, split:
  - [x] E4a. String functions over `Expr::Call`: upper/lower/trim, length (char
        count), substring(s,start[,len]) (0-based, char-indexed, clamped),
        replace, starts_with/ends_with/contains (→ Bool). NULL/wrong-type → NULL;
        eval unified into `call_scalar`. Arity validated at parse.
  - [x] E4b. List literal `[a,b,…]` (new `Expr::List`, per-row → `Value::List`,
        non-constant elements ok; parsed in primary(); List arm added to every Expr
        match) + size/head/last via `call_scalar` (non-list → NULL; head/last of
        empty → NULL). NOTE: `list[i]` index access still deferred (own slice).
- E5. `CAST` + coercion. USER DECISION (2026-08-09): failed cast THROWS
  `E_INVALID_VALUE`; `INTEGER` truncates toward zero (FLOAT/NUMBER keep the
  fraction; all stored as f64); broad conversions (string↔number, →string,
  number↔bool `0=false/nonzero=true`, string→bool `'true'/'false'`, null→null).
  Split (throw needs a fallible read pipeline):
  - [x] E5a. Make the read pipeline fallible: `eval`/`pull`/`aggregate`/
        `order_page`/`try_frontier_aggregate` return `Result<_,String>`; added
        `try_run`, `run` wraps it with `.expect`; `execute` uses `try_run` and
        MERGE eval faults roll back. Pure refactor, all 190 tests green.
  - [x] E5b. `CAST(<expr> AS <TYPE>)`: IR `Expr::Cast`/`CastTarget` +
        `value::cast()` (the coercion home) + GQL `CAST(e AS TYPE)` parsing
        (INTEGER/INT, FLOAT/…, STRING/…, BOOL/BOOLEAN). Throws E_INVALID_VALUE
        on failure, INTEGER truncates toward zero, null→null, broad conversions.
        9 tests (value table, per-row eval, fault via try_run, parse vs hand plan).
- [x] E6. 3VL completeness: `<e> IS [NOT] NULL` (a definite Bool, never
      UNKNOWN — value test) and `PROPERTY_EXISTS(<var>, <key>)` (presence test,
      the one predicate that separates an absent property from a present-null).
      IR `Expr::IsNull`/`PropertyExists`, eval + GQL parse, 4 tests (value-vs-
      presence distinction cross-checked against hand plans). AND/OR/NOT already
      complete from Phase E.

### Phase F — Query surface

- [x] F1. GQL `WITH` (chained query parts): projects/aggregates like RETURN,
      rebinds scope to the carried columns (a bare variable keeps its name), rides
      ORDER BY/SKIP/LIMIT, and a trailing WHERE is a post-projection (HAVING)
      filter — matching lenke-core. A continuing MATCH extends the working table
      from a carried node (shared `extend_chain`); an unbound continuation errors.
      4 tests (aggregate+HAVING, carry-into-MATCH, order+limit paging, error),
      cross-checked against hand plans.
- [x] F2. GQL `EXISTS { <pattern> [WHERE] }`: a correlated existence predicate
      (definite Bool, composes under NOT). IR `Expr::Exists{body,outer_width}` +
      `Plan::Row` (the correlated leaf); evaluated whole-batch with a provenance
      column so survivors map back to their outer row (`pull_body`). Body reuses
      `extend_chain`; the start node must be a bound outer variable. 5 tests
      (correlated hop, inner WHERE, outer-correlated WHERE, NOT EXISTS, error),
      cross-checked against a hand plan.
- [x] F3. GQL `CALL (scope) { … }` — the inline correlated (lateral) subquery.
      IR `Plan::CallInline{input,body,yields,outer_width}`; the body is a
      `Plan::Row`-rooted pattern (reuses `extend_chain`) run over the outer rows,
      emitting one row per sub-row (outer slots + yield exprs) — an inner lateral
      join, so zero-match outer rows drop. Imports only the declared scope vars;
      internal subquery vars don't survive. 5 tests (lateral join vs hand plan,
      inner WHERE, yield correlated on outer, named-form-deferred + unbound errors).
      The named-procedure form `CALL name(cfg) YIELD …` is RELOCATED to I3 — its
      catalog IS the graph algorithms (I1), so it lands with them (same
      dependency-driven move as D3 → G4/G5). OPTIONAL CALL and an aggregating
      subquery body are deferred.
- [x] F4. Path values: `MATCH p = ANY SHORTEST (a)-[:R]->*(b)` binds `p` to the
      row's path (lineage); `shortest_path` now reconstructs the full node chain
      via BFS predecessors, and accessors `nodes(p)` / `path_length(p)` read it.
      Parser: `path_vars` resolve to `Expr::Path`; shared `query_tail`; `*`/`+`
      quantifier (edge type required). 5 tests (length + hand plan, node-chain
      reconstruction, two errors). `relationships(p)`/`elements(p)` need edge-level
      lineage → F4b.
- [x] F4b. Edge-level path lineage: `Lineage` now carries a parallel edge list
      (edges/edge_offsets), populated by Expand (collects eids only when a bound
      edge or lineage needs them — hot path unchanged) and ShortestPath (BFS
      predecessor EDGE). Accessors are now `Expr::PathAccess{part}` reading the
      sidecar directly — `nodes`/`relationships`/`path_length`/`elements` — not
      scalar Call fns. 4 tests (relationships + elements over a chain, expand-level
      edge lineage, non-path-arg error).
- [x] F5a. Gremlin step breadth, part 1: value aggregates min/max/sum/mean (fold
      the current value stream), where(P) (filter the current traverser by a
      predicate), and a shared predicate parser so has(...) also accepts a bare
      op like gt(28) (not just P.gt). values()/aggregates now retrack the current
      slot. 3 tests. (Baseline already had V/addV/addE/hasLabel/has/out/in/both/
      values/count/dedup/limit/range/order().by()/groupCount().by()/property/drop.)
- [x] F5b. Gremlin step breadth, part 2: as('x') step labels + single-label
      select('x') (projects the labelled slot), within(...)/without(...) membership
      predicates (OR-of-equals and its negation, shared by has and where), and a
      bare groupCount() (groups by the current element, .by('k') still optional).
      4 tests. Multi-label select (a Map value) and order(local) (within-list sort)
      → F5c, which depends on G2 maps / list ops.

### Phase G — Data model

- [x] G1a. Temporal values (zone-less trio): a dependency-free `temporal` module
      (Date/Time/DateTime, Hinnant civil math, ISO-8601 parse/format) ported from
      lenke-core for AGREEMENT; `Value::Temporal` with all value.rs contract arms
      (equals same-kind, cmp_total chronological + cross-type rank Str<Temporal<
      List<Null, group_key by kind+ISO, cast→string); GQL typed literals DATE/TIME/
      DATETIME '…'; tagged-JSON egress. Stored in Gen columns for now. 8 tests.
- [x] G1b. Temporal value model complete: DURATION (months/days/secs/nanos kept
      separate, canonical P..M..DT..S form, total-order-only) + ZONED TIME/DATETIME
      (numeric offset preserved, split_offset, instant-then-offset order). All six
      kinds now parse/format/order via the temporal module; GQL literals DURATION
      '…' and two-word ZONED TIME/DATETIME '…'. 5 tests. (Duration/zoned relational
      `<` uses the total order like the rest of lenke-engine, not rel_cmp UNKNOWN.)
- [x] G1c. Temporal codec + accessors: NDJSON temporal DECODE (a single-key
      {"@tag":"iso"} object round-trips back to `Value::Temporal`), and the six
      component accessors year/month/day/hour/minute/second (ported `date_part`,
      euclidean, zoned decomposed in their own offset; NULL when undefined for the
      kind). 2 tests (store→NDJSON→store date round trip, all accessors + NULLs).
- [x] G1d. Typed `Column::Temporal` storage: temporals de-box from the Gen column
      into a homogeneous per-kind column (Vec<Temporal> + present bitmap); a
      different-kind or non-temporal write promotes to Gen, matching lenke-core's
      one-kind-per-column model. Both the SET/add_node path (new_absent) and the
      Builder (materialize) build it; reads/egress/round-trip observably identical.
      1 store test (typed column + mixed-kind promotion).
- [x] G1e. Temporal constructors + duration_between: date()/local_time()/
      datetime()(+local_datetime)/zoned_time()/zoned_datetime()/duration() parse a
      string or coerce between kinds (date↔datetime midnight, datetime→time part,
      date-str→midnight); duration_between(a,b) is the exact span (days for
      dates, secs+nanos for datetimes, cross-kind→NULL). Ported from lenke-core.
      2 tests. Duration arithmetic (instant ± duration, duration ± duration,
      duration × int, with overflow faults) → G1f.
- [ ] G1f. Temporal arithmetic: instant ± duration (add_months clamped, then
      days, then time; date-overflow throws), duration ± duration (component-wise),
      duration × integer — wired into the `+`/`-`/`*` operator via value.rs.
- [ ] G2. Map/record values (storage, dotted-path, construction, access).
- [ ] F5c. Gremlin multi-label select('a','b') (a Map value — needs G2) and
      order(local) (within-list sort — needs list ops). Relocated from F5b; placed
      after G2 because it depends on the map/list value model.
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
- [ ] I3. Named-procedure `CALL name(cfg) [YIELD col [AS a], …]` (relocated from
      F3): the ISO-conformant home for the I1 algorithms — the procedure catalog
      IS those algorithms, so it must land after them. `OPTIONAL CALL` keeps the
      outer row (yields null-filled) when the call is empty.

### Phase J — Agreement

- [ ] J1. Conformance suite: run matched shapes on `lenke-engine` and
      `lenke-core`, assert same results (agreement, not byte-identity).

## Standing

Update this file as slices land — tick the box and note the commit. The loop is
done when every box is `[x]` and the conformance suite (J1) is green.
