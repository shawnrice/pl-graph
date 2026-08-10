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
- [x] G1f. Temporal arithmetic (completes G1): instant ± duration (calendar
      months clamped to the new month's length, then days, then time; zoned forms
      apply it to the local wall clock; date-overflow THROWS), instant − instant =
      the exact span, duration ± duration (component-wise), duration × integer
      (non-integer → NULL). Wired into the Arith `+`/`-`/`*` eval (temporal path
      when either operand is temporal; numeric path unchanged); all rules in the
      temporal module. 2 tests (month-clamp leap/non-leap, span, dur sum/scale,
      overflow throw).
- [x] G2a. Record values (the ISO GQL `<record>`): `Value::Record` (sorted
      string keys, Arc-boxed, canonicalized via `value::make_record` — last-wins
      dedup) with all value.rs contract arms (equals/cmp_total/group_key,
      cross-type rank List<Record<Null, cast→error). GQL `{k: expr, …}` literal
      (`Expr::Record`) and field access `r.k` (extended `read_property` for record
      columns; absent field→NULL). NDJSON encode/decode (JSON object ↔ record,
      round-trips). Stored in Gen columns. 3 tests (contract, literal+field+hand
      plan, NDJSON round trip).
- [x] G2b. Gremlin `Value::Map` (any-key, INSERTION-ordered — positional
      equality/grouping, unlike Record's sorted keys; value.rs contract + rank
      Record<Map<Null) and `Expr::Field` for `{lit}.field` / `(expr).field` /
      chains (the general form of Prop; reuses read_property, which now reads a Map
      entry too). 2 tests (positional Map contract, field access + nested + hand
      plan). Map has no producer yet — F5c (multi-select) is its first, now
      unblocked. NDJSON Map egress is best-effort (not a stored type).
- [x] G2c. Nested field access on a variable: `n.rec.field` (and deeper chains) —
      the first `.key` stays a `Prop` (optimizer-seekable), further `.key` become
      `Field` accesses on the value it produced. Completes record access (stored
      record properties, not just `{lit}.field`). 1 test (stored record, nested
      read + absent→NULL).
- [x] G2d. Store-side dotted-path hash index: `create_index("meta.city")` keys on
      the value found by descending record fields (`resolve_path`); `Index` is now
      path-based (a plain property is a length-1 path — behaviour unchanged).
      Maintained through set/remove/delete (`reindex_node`) so rollback stays
      consistent; `index_lookup("meta.city", v)` returns candidates. 1 store test
      (build-from-data, maintenance on write, delete, no-index→None).
- [x] G2e. Planner seek for the dotted index: `seek_target` now recognizes a
      dotted property PATH (`Field{Prop{0}}` chain, via `prop_path`) on either
      side of `=`, seeding a dotted `IndexSeek`; the no-index scan fallback reads
      the path with `store.prop_path`. Both spellings land the same target. 1 opt
      test (seed shape + rows with index AND via the fallback). Completes the
      dotted-path index end to end (the memory's 140x seek path).
- [x] F5c. Gremlin multi-label `select('a','b')` builds an insertion-ordered
      `Value::Map` keyed by the labels (via a new `Expr::MapLit`, Gremlin-only);
      a single-label select still projects the element, an unknown label errors.
      1 test (ordered map of node ids). order(local) → F5d.
- [x] F5d. Gremlin `fold()` + `order(local)` (was deferred for want of a
      list-producing step — so this slice built the producer too). `fold()` is a
      new `AggFn::Collect` (barrier aggregate, no keys → one row holding a
      `Value::List` in group-row order, nulls kept, empty stream → one empty
      list). `order(local)` is a new transparent `Plan::SortLocal` that sorts
      inside slot 0's cell via `value::cmp_total` (List → elements; Map → by
      value; scalar → passthrough), `.by(asc|desc)` optional, `Scope.local`
      spelling accepted. 6 tests (fold set, fold-of-empty, order(local) asc on
      strings, .by(desc) on numbers, Scope.local, scalar passthrough). fold(seed,
      biFn) still deferred (no consumer).
- [x] G3. Numeric edge-case parity audit against `value.rs`: confirmed agreement
      with lenke-core on the f64 model — `as_num` is finite-Num-only (NaN/Inf →
      None → arithmetic NULL), `-0.0 == 0.0`, NaN unequal under `equals` but
      greatest-and-self-equal under `cmp_total`, `num_group_bits` canonicalizes all
      NaN payloads and signed zero, div/mod-by-zero and overflow → NULL. 4 tests
      (as_num gate, NaN-payload/signed-zero grouping, non-finite arithmetic → NULL,
      cross-type ordering). ONE recorded divergence (for J1): cross-type ORDERING
      is a single deterministic total order here (rank-based, never faults),
      whereas lenke-core's GQL raises E_INVALID_VALUE — a deliberate choice so
      sort/group/min-max stay total; equality already agrees (cross-type false).
- G4. Interval/temporal edge index, built measured-first, split. The consumer
  IS expressible in this adjacency-driven engine after all: a high-degree node
  whose edges carry `[vf, vt]` intervals, queried "as of T"
  (`MATCH (p)-[r:HELD]->() WHERE r.vf <= T AND r.vt >= T`) — the bitemporal shape.
  - [x] G4a. Benchmark harness: `examples/interval_bench.rs` times the as-of
        overlap count over a node's time-versioned edges. BASELINE (min of 7,
        release): 20k×8 = 31ms, 20k×64 = 361ms, 20k×512 = **4.08s**, 100k×64 =
        2.26s (~400ns/edge). The finding: the cost is NOT adjacency scan — it is
        the BOXED edge-prop post-filter (`vf` and `vt` are each a String-keyed +
        eid-keyed HashMap probe, per edge). An interval index that stores the
        intervals INLINE and seeks overlaps sidesteps those probes entirely, so
        the win should be large (unlike G5's marginal case).
  - [x] G4b. Opt-in interval index + seek (store level): `create_interval_index(
        lo_key, hi_key)` stores each node's OUT-edge intervals `(lo, hi, eid, nbr)`
        BOTH sorted by lo and by hi (read inline from the boxed props at build).
        `for_each_overlap(node, qlo, qhi, f)` seeds from whichever axis is more
        selective (`partition_point`) and post-filters the other — the FIRM
        bitemporal rule (never intersect both stabs). Maintained through the
        primitives + undo: per-node reindex on adjacency change, a full rebuild on
        an interval-axis edge-prop change (source node not cheaply known from eid)
        and once after rollback (prop+adjacency undo ordering can't be tracked per
        record). OFF by default. 3 store tests (seek == brute-force across points
        AND ranges incl. both seed axes; write tracking; rollback restore).
        MEASURED (`interval_bench`, numbers in the file): **96–226× faster** than
        the boxed post-filter scan — a large unambiguous win (the baseline cost was
        the boxed edge-prop probe, which the inline seek sidesteps). Build is a
        one-time pass over the props.
  - [x] G4c. Query/planner integration: new `Plan::IntervalExpand` (a seek-or-scan
        hop like IndexSeek) + an optimizer rule that fuses `Filter(r.lo <= X AND
        r.hi >= Y)` over a bind-edge `Expand` into it (recognizes both spellings
        via `interval_side`/`flip_cmp`; bounds must reference only slots below the
        hop). exec seeks via `for_each_overlap` when an OUT hop meets a matching
        interval index, else scans the adjacency applying the overlap itself — so
        rows are IDENTICAL either way (a non-numeric/absent bound or edge interval
        drops the edge, matching the `<=`/`>=` filter). 1 exec test: the optimizer
        fuses the pattern, and the plan returns the same count AND the same rows
        with the index off (scan) vs on (seek), equal to the hand-computed answer.
        Completes G4 — GQL as-of queries now get the interval seek transparently.
- G5. Edge-type index (type-filtered `expand`), built measured-first, split:
  - [x] G5a. Benchmark harness: `examples/expand_bench.rs` sweeps degree ×
        edge-type count (per CLAUDE.md — vary degree and the edge:type ratio) and
        times `MATCH (n:V)-[:T0]->() RETURN count(*)`. BASELINE (min of 7, release,
        this box): degree 4 = 157µs; degree 32 = 1.5–2.0ms; degree 256 = 8.7–10.0ms
        **independent of type count** (1/8/64 types all ~9ms). That is the finding:
        the cost is scanning the WHOLE adjacency, so an index that seeks one type
        only wins when the queried type is a small fraction of a high degree
        (deg 256 / 64 types → T0 is ~4 of 256). Low-degree / single-type can't
        benefit → the index must be OPT-IN so those pay nothing (G5b).
  - [x] G5b. Opt-in edge-type index: `create_edge_type_index` builds per-node
        `etype → adjacency` buckets (`out_typed`/`in_typed`); `for_each_nbr` seeks
        the bucket for a type-filtered hop instead of scanning. Maintained across
        writes/deletes/rollback by a per-node rebuild off the authoritative flat
        adjacency (O(1) push on the `add_edge` hot path), so no delta bookkeeping
        can drift; OFF by default (zero cost, and existing tests unchanged). 6 tests
        (buckets match a flat scan; add/delete/delete-node maintenance incl.
        neighbour mirrors; rollback restores exactly; grows with a new node) + a
        query-level equivalence test (same rows/counts with the index on vs off).
        MEASURED (`expand_bench`, numbers kept in the file): WINS 6.5–8.1× in the
        high-degree × many-type × selective regime (deg 256 / 8–64 types), but
        LOSES at degree 4 (0.32×) and at 200k nodes / low type-count (0.65–0.87×) —
        the per-node `HashMap` chases scattered heap while the flat scan is
        contiguous (the cache-transition effect). Opt-in is exactly why that's
        safe: a graph outside the winning regime never creates it. A CSR /
        sorted-by-type adjacency is the future broad-win representation (deferred,
        no workload needs it yet). Completes G5.

### Phase H — Semantics services

- [x] H1. Required-property constraint (validator): `create_required_constraint
(label, key)` — every live node with `label` must carry a PRESENT value for
      `key` (present-null passes; only absence violates, per null-first-class).
      Declaration errors on already-violating data; write statements (INSERT and
      \_MERGE) enforce it alongside unique and roll back on violation
      (`E_REQUIRED`); it round-trips through the snapshot schema
      (`{"schema":"required",…}`). 3 tests (store check/declare, INSERT reject +
      rollback, snapshot survival). NOTE: like unique, SET/REMOVE-time re-check is
      not yet wired (a REMOVE of a required key over a MATCH isn't caught) — a
      shared follow-up when Update gets constraint enforcement.
- [x] H2. Events / CDC change-list core (observation-only): every mutation records
      a `Change` (NodeAdded/Deleted, NodeProp, EdgeAdded/Deleted, EdgeProp) into the
      open transaction, 1:1 with the undo log; `commit` publishes it as
      `last_commit_changes()`, `rollback` discards it (not an event). Observation
      only — read AFTER commit, cannot veto. A node delete cascades its edges but
      reports one NodeDeleted. 4 tests (commit list + order, rollback-nothing,
      delete cascade, INSERT observed). NOTE: only txn-wrapped statements
      (INSERT/\_MERGE) emit CDC; SET/REMOVE/drop await wrapping the Update path in a
      txn (shared with the H1 SET/REMOVE-enforcement follow-up).
- [x] H2b. CDC value-scope routing (engine side): `touched_scopes(scope_key)`
      derives the DISTINCT scopes a commit wrote from the change list — a node
      change's scope is its `scope_key` property (host decides what that names);
      an edge change or an absent/deleted-node scope sets a fail-OPEN flag
      ("visible to all"). A subscriber to scope S treats the commit as relevant iff
      `open || scopes∋S`. Optimization-not-boundary; the host owns the scope-key
      authority (the engine derives, never mints). Scopes cmp_total-sorted. 1 test
      (distinct rooms A/B, dedup, fail-open on an unscoped node).
- [x] H3. Typed nodes — HOST-SIDE by design (not an engine capability). R-TYPED
      is `defineNode` with a bring-your-own Standard Schema (Zod/Valibot/ArkType)
      validating on the HOST before the write; a JS schema cannot be an engine
      validator, and rebuilding a schema DSL in Rust would duplicate the host's
      job. The ENGINE seam is already provided: H1 constraints (unique/required,
      enforced + rolled back at write time) for the invariants the engine CAN
      check, and the host validates-before-write for the rest. Done by delegation;
      no engine code.

### Phase I — Algorithms & egress

- [x] I1a. Graph algorithms (deterministic trio): a new `algo` module over the
      store — `degree` (out/in/both, optional edge type, dense-id order), weakly
      connected components (`weakly_connected_components`, union-by-min so a
      component's id is its smallest member, order-independent), and `bfs_distances`
      (shortest hop distances from a source, order-independent). Each returns
      `Vec<(node, result)>` in ascending-id order. 3 tests (triangle + isolated
      node: degrees, 2 components, BFS out/both). Rust API only — the GQL/Gremlin
      CALL surface is I3.
- [x] I1b. Iterative algorithms (completes I1): PageRank (pull model, ported from
      lenke-core — damping 0.85, 20 fixed iterations, dangling mass redistributed,
      per-target pull summed in `in_adj`/edge-insertion order, dangling in node-id
      order → mass-conserving and reproducible) and synchronous label propagation
      (10-round bound, early-stop, undirected, tie → smallest label id). 3 tests:
      label-prop collapses a triangle to its min-id label; PageRank on a 2-cycle is
      exactly 0.5/0.5 summing to 1; PageRank ranks higher-in-degree higher, sums to
      1, and is bit-reproducible. NOTE (J1): label-prop tiebreaks on the dense node
      id (this engine has no external string ids) vs lenke-core's lexicographic
      string tiebreak — they agree when id order matches string order.
- [x] I2a. Arrow columnar egress — the `ARW1` blob (a new `arrow` module):
      `to_arrow(&Rows)` lays a query result out as lenke-core's dependency-free
      carrier — 24-byte header (magic/version/nrows/ncols), 40-byte column
      descriptors, 8-aligned body buffers that ARE Arrow's physical layout
      (LE, LSB-first validity bitmap, i32 Utf8 offsets). Scalar column inference
      matches lenke-core (present Nums→Float64, present Bools→Bool, else Utf8),
      cell stringification matches for scalars/temporals. 3 tests via a hand-rolled
      reader (header+types, null round-trip through the validity bitmap, mixed
      num+bool→Utf8).
- I2b. Arrow egress completion, split (the "TS-verifier-dependent" blocker
  dissolved once lenke-core became a dev-dependency: byte-parity against the
  reference encoder IS the verification, no apache-arrow needed):
  - [x] I2b-1. Nested ARW1 columns: `FixedSizeList<Float64>` (all present cells
        same-length numeric lists; dim rides `buf2_len`) and `Struct`
        (record/map columns → typed child columns, pre-order flattened, child
        count rides `buf2_len`; header `ncols` counts top-level only). Mirrors
        lenke-core's `ArrowColumn`/`flatten_descs` exactly; engine `Record` and
        string-keyed `Map` both become a `Struct` (matching core's result-side
        Map→Struct). `tests/arrow_parity.rs` asserts BYTE-IDENTICAL `to_arrow`
        blobs vs lenke-core for scalar, FixedSizeList, Struct, and nested
        struct-with-list shapes (4 tests); 4 in-crate layout unit tests
        (fixed-list flat values + dim, struct pre-order children, struct null
        row, string-keyed map → struct).
  - [x] I2b-2. Arrow-IPC flatbuffer envelope: `to_arrow_ipc(rows, file)` frames
        the ARW1 buffers as standard Apache Arrow IPC (`Schema`/`RecordBatch`/
        `Footer` messages via a back-to-front FlatBuffers builder) — stream layout
        or file/Feather-v2 — so DuckDB/Polars/pandas consume it via `tableFromIPC`.
        A VERBATIM port of lenke-core's framing over the identical ARW1 blob; the
        `to_arrow_ipc` signature takes `&Rows` (the only adaptation). 5 tests in
        `arrow_parity.rs` assert BYTE-IDENTICAL IPC vs lenke-core in BOTH layouts
        for scalar/FixedSizeList/Struct/nested shapes, plus an envelope-shape check
        (stream continuation + end-of-stream markers; file `ARROW1` magic + footer).
        Completes I2b.
- [x] I3. Named-procedure `CALL name(cfg) [YIELD col [AS a], …]` (relocated from
      F3): the ISO home for the I1 algorithms. `Plan::CallProcedure{name,config}`
      runs the algorithm over the store into a `[node, <result>]` batch; the parser
      (a top-level `CALL` entry point) validates the snake_case name against the
      catalog (degree/pagerank/connected_components/label_propagation, result cols
      degree/score/componentId/label), parses the `{k:v}` config map, and wraps it
      in a Project applying YIELD (select/rename; default = all columns), which can
      feed further clauses. 3 tests (degree yield/default/rename + hand plan,
      degree({direction}) + connected_components, unknown-proc/unknown-yield
      errors). `OPTIONAL CALL` (null-fill) deferred.

### Phase J — Agreement

- [x] J1. Conformance suite: run matched shapes on `lenke-engine` and
      `lenke-core`, assert same results (agreement, not byte-identity).
      DONE — `crates/lenke-engine/tests/conformance.rs`. `lenke-core` is a
      path DEV-dependency (Cargo.toml; test-only, normal build stays
      standalone). One fixture defined as Rust data, serialized into each
      engine's own NDJSON dialect (they differ on the wire); 25 matched GQL
      shapes (19 unordered → multiset equality, 6 `ORDER BY` → ordered
      equality) plus a load-sanity check, all green. Comparison is by VALUE
      (scalars only), so the engines' independent dense-id assignments never
      enter it; numbers via `num_key` (integers exact, non-integers to 1e-9 —
      float bit-identity is a core-vs-TS invariant, not a lenke-engine one).

  Documented divergences (accounted for, not failures):
  - **ORDER BY scope.** `lenke-engine` scopes `ORDER BY` to OUTPUT columns
    (by alias/name); `lenke-core` also accepts a non-projected expression
    (`ORDER BY n.age` when only `n.name` is projected). The matched shapes
    order by a projected alias, which both accept. (Engine choice, gql.rs:439.)
  - **Cross-type ORDERING.** Engine gives a total order across types (by kind
    rank); core THROWS `E_INVALID_VALUE` on `<=` across types. Equality already
    agrees (cross-type = false in both). Not exercised by matched shapes. (§120,
    §323.)
  - **label-prop tiebreak.** Engine breaks label-propagation ties on the dense
    node id; the two engines' dense ids are assigned independently, so a tie can
    land differently. Algorithm outputs are not part of the matched GQL shapes.
    (§396.)

- [x] J2. Differential fuzzer (`tests/differential_fuzz.rs`): the Rust-native
      analogue of `differential-fuzz.test.ts` — a seeded PRNG generates random
      graphs (absent / present-null / adversarial-value props) and random GQL from
      the shared, type-safe surface (nested WHERE, whole-set + grouped aggregates
      incl. `count(DISTINCT)`, projections with arithmetic, DISTINCT, deterministic
      ORDER BY + paging), runs BOTH engines and asserts agreement (multiset or
      ordered). Default 400 iters/seed 1 under `cargo test`; `FUZZ_SEED`/`FUZZ_ITERS`
      to replay/scale. It IMMEDIATELY earned its keep: found `sum` over an
      empty/all-null group returning NULL (SQL-style) vs lenke-core's 0 (the
      GQL/Cypher convention) — fixed in `fold_grouped` (SUM of nothing = 0, AVG
      stays NULL). 24k queries × 8 seeds now agree, 0 skips.

## Standing

Update this file as slices land — tick the box and note the commit. The loop is
done when every box is `[x]` and the conformance suite (J1) is green.
