# Scoping: inline `CALL (scope) { … }` rewrite

Status: **scoping only, not started.** This documents what it would take to bring the
engine's inline correlated-subquery `CALL` up to the ISO surface pure-TS already
supports. Pure-TS `@lenke/gql` is the **oracle** here — it is correct; the engine is
the limited one.

## What the engine supports today

`gql::Parser::call_inline` (correlated form) accepts exactly ONE shape:

```
CALL (v) { MATCH (v)-[…]->(…) [WHERE …] RETURN <items | single COUNT> }
```

- The body MUST begin with `MATCH` whose **first node is a declared scope variable**
  (`node_plain()` + the `scope_vars.contains(&v)` guard); it may not be re-labeled or
  re-constrained.
- The body is built with `extend_chain(Plan::Row, …, from = scope-var slot)` — an
  Expand rooted at the scope var.
- No set operators. A single `COUNT(*)` RETURN is special-cased to `Expr::CountSubquery`;
  any other aggregate errors ("an aggregating RETURN inside CALL { … } is not supported").
- `call_inline_uncorrelated` handles the empty-scope `CALL () { … }` form (also no set-ops).

Exec `Plan::CallInline { input, body, yields, outer_width }` (exec.rs ~2900):

```rust
let outer = pull(input, store, track)?;
let sub   = pull_body(body, store, &outer)?;      // body EXTENDS the outer batch (lateral join)
let mut out = (0..outer_width).map(|i| sub.slot(i)) ++ yields.map(eval);
```

It is an **inner** lateral join: the body extends each outer row, and outer rows with no
sub-match simply vanish (they carry no sub-row). Outer-row identity is only the outer
slot _values_ — there is no per-outer provenance id.

## What ISO / pure-TS requires (the two failing conformance tests)

`runCallInline` (clauses.ts) is row-at-a-time and shows the target semantics:

```ts
flatMap((outer) => {
  const seed = pick scope vars from outer;
  let nested = runLinearClauses(body, seed);
  for (const {op, part} of bodyMore)                 // set-op tail
    nested = combineRows(op, nested, runLinearClauses(part, seed));
  if (nested.length === 0 && optional)               // OPTIONAL → left-outer
    return [outer + null-filled returnColumns];
  return nested.map(row => outer + row);              // inner → duplicate outer per nested row
}, bindings)
```

The two red tests need, beyond today's shape:

| feature                                                  | "RETURN \* / element columns" | "set operators"      |
| -------------------------------------------------------- | ----------------------------- | -------------------- |
| `RETURN *` (yield every body binding)                    | ✅                            | —                    |
| `OPTIONAL CALL` (left-outer null-fill)                   | ✅                            | ✅                   |
| body from a FRESH `MATCH (s:Software)` (not a scope var) | —                             | ✅ (the EXCEPT case) |
| `UNION [ALL]` / `EXCEPT` / `INTERSECT` in the body       | —                             | ✅                   |
| uncorrelated `CALL () { … UNION … }`                     | —                             | ✅                   |

## The architectural problem

Pure-TS is interpreted **row-at-a-time**: it literally re-runs the body query per outer
row against a fresh seed, so set-ops, OPTIONAL, aggregates, and fresh scans "just work"
by delegating to the normal query machinery.

The engine is **vectorized/columnar**: it must run each body part ONCE over all outer
rows and group the results back per outer row. That grouping needs a **provenance id per
outer row** — which the current lateral-extend `CallInline` does not carry. This is the
same mechanism `Expr::CollectSubquery` / `ScalarSubquery` / `Exists` already use (seed =
outer columns + a `prov` column at slot `outer_width`, run the body, results carry `prov`).

So the rewrite is fundamentally: **reimplement the CALL body as a provenance-tagged
correlated subquery that yields multiple columns and multiple rows per outer row**, with
set-op combination and OPTIONAL left-outer, all grouped on `prov`.

Note the **prerequisite**: the CALL body is currently built at `sub_slots = outer_width`,
but a provenance-tagged body needs `sub_slots = outer_width + 1` (prov reserved at
`outer_width`, exactly like `correlated_subquery_body`). This is the _same_ slot-layout
blocker that stalls the CALL-body scalar-aggregate work — doing it here unblocks that too.

## Design (recommended: provenance-tagged, engine-native)

1. **IR** — extend `Plan::CallInline`:

   ```rust
   CallInline {
     input, body, yields, outer_width,
     optional: bool,                          // OPTIONAL CALL → left-outer
     parts: Vec<(CombineOp, bool, Plan)>,     // set-op tail: (op, all, body-part)
     return_columns: Vec<String>,             // for OPTIONAL null-fill
   }
   ```

   Every `body`/`part` built with `prov` reserved at `outer_width` and yields at `+1…`.

2. **Parser** (`call_inline`):
   - Dispatch `OPTIONAL CALL` — today `query_tail`'s `OPTIONAL` branch only does
     `OPTIONAL MATCH`; add a lookahead so `OPTIONAL` + `CALL` routes here with
     `optional = true`.
   - **Relax the body start**: allow a fresh `MATCH (s:Label)` (a Scan) as well as a
     scope-var start. A fresh-scan body is a correlated CROSS-JOIN of the prov-seed with
     the scan; a scope-var body stays the Expand-from-slot it is now.
   - **Set-op tail in the body**: refactor the top-level UNION loop (`parse()`
     lines ~335–359) into a reusable `parse_setop_tail` and call it for the body,
     re-seeding each arm's scope from the same scope vars.
   - **`RETURN *`**: expand to yields over every body-bound sub-scope variable (needs a
     RETURN-\* path the engine doesn't have yet — small addition).

3. **Exec** (`Plan::CallInline`):
   - Provenance-tag: seed `outer + prov(0..n)`, run `body` and each `part` → each carries `prov`.
   - **Combine set-op parts per outer row**: group each part's rows by `prov`, then apply
     `combineRows(op, all)` on the yield-tuples _within each prov group_ (UNION distinct/all,
     EXCEPT, INTERSECT multiset rules).
   - **Inner vs OPTIONAL**: per outer row — ≥1 result → emit outer + yields per result;
     0 results and `optional` → emit outer + null yields; 0 and not optional → drop.

4. **Fold in the CALL-aggregate** (the deferred `sum`/`avg`/`min`/`max`): once the body
   carries `prov`, `CollectSubquery` over the arg works (that was its only blocker).

## Byte-identity risks (the hard part)

Row **order** and set-op semantics must match pure-TS's `combineRows` exactly:

- UNION distinct dedup **order** (first-seen), UNION ALL multiplicity, EXCEPT / INTERSECT
  multiset rules — per outer row.
- `RETURN *` column order (which sub-scope vars, in what order).
- OPTIONAL row placement relative to the outer batch, and null-fill column set.
- Interaction with a trailing `ORDER BY` on the outer query (the tests all `ORDER BY`,
  which masks intra-group order — but the differential fuzzer will not).

These are the same class of ordering pins that `group-by-vs-order-by-scope` and
`order-is-unspecified` already document; verify with the two conformance tests **and** an
extension of the differential fuzzer (it currently generates no inline-CALL).

## Suggested phasing

- **Phase 0 (prerequisite):** build the CALL body with `prov` reserved (`sub_slots =
outer_width + 1`); adapt the current `CallInline` exec + non-aggregate yields to the new
  layout. No behavior change; unblocks everything below (and the scalar-aggregate work).
- **Phase 1 (done):** `RETURN *` + `OPTIONAL CALL` (correlated-from-scope-var only, no
  set-ops). `RETURN *` / bare-element yields already worked via `return_items`'s `*`
  expansion; the only new work was `OPTIONAL CALL` — the parser dispatch (`OPTIONAL` +
  `CALL` lookahead) and a left-outer null-fill in the exec that reads the Phase-0
  provenance column, keeping imported scope vars intact and nulling only fresh body vars
  (node/edge yields stay node/edge columns via the `u32::MAX` sentinel so `f.name`
  resolves to NULL rather than downgrading to `Gen`). **Closed "inline subquery CALL with
  RETURN \* / element columns".**
- **Phase 2 (done):** fresh-scan body start + set operators (`UNION [ALL]`/`EXCEPT`/
  `INTERSECT`), correlated and uncorrelated. **Closed "inline subquery CALL with set
  operators".** Implementation:
  - Parser: `call_inline` now parses arm 0 + a set-op tail via a shared `call_arm` that
    dispatches on the arm's first node — a declared scope var roots an Expand (as before),
    anything else is a fresh `(x:Label)` Scan. Arms with a tail build `Plan::CallInline`
    with a `parts: Vec<CallPart>` set-op tail; single-arm bodies keep `parts` empty and
    the exact Phase-0/1 fast path (incl. the single-`COUNT(*)` special case). The
    uncorrelated `CALL () { … }` handles its tail by combining arms with the ordinary
    top-level `Plan::Union` operator and cross-joining the whole global result.
  - Exec: a fresh-scan arm runs through a new `pull_body` `Plan::Scan` arm (a correlated
    cross-join of the prov-seed with the label's nodes). `call_inline_setop` collects each
    arm's yield tuples grouped by provenance, combines them left-associatively per group
    (`combine_call_groups`, the same multiset rules as the top-level set-op), then lays
    out rows in outer order (imported columns gathered natively so `x.name` resolves; yield
    columns materialized to `Gen` like the top-level set-op) with OPTIONAL null-fill.
  - Byte-identity: verified against the two conformance tests (UNION/UNION ALL/EXCEPT with
    a fresh-scan arm/INTERSECT/OPTIONAL+set-op/uncorrelated/three-part, all ORDER BY'd).
    The differential fuzzer still generates no inline-CALL — extending it is the remaining
    open coverage item.
- **Phase 3 (done):** scalar-aggregate CALL body (`sum`/`avg`/`min`/`max`). A single-arm
  body whose RETURN is one non-distinct scalar aggregate lowers to a new `Expr::AggSubquery`
  (the parser generalized the `COUNT(*)` special case). The exec mirrors `CollectSubquery`
  — the same provenance-tagged sub-run — but REDUCES the argument per outer row instead of
  gathering a list. Empty / all-null group: `SUM` → 0 (GQL, not SQL's NULL — pure-TS is the
  oracle), `AVG`/`MIN`/`MAX` → NULL; `COUNT` still routes through `CountSubquery` (→ 0).
  Closed "correlated CALL sum → decorrelated". All three inline-CALL conformance tests now
  pass against the engine backend (184 → 187 vs engine over Phases 1–3).

## Effort

Multi-day, not a bounded fix. Phase 0 + Phase 1 are moderate (parser + a null-pad exec
path over the existing lateral join, once prov is carried). Phase 2 is the large one — a
new per-outer-row set-op combine in the vectorized exec, plus the byte-identity work.
Phase 3 is small once prov is carried (an aggregate twin of `CollectSubquery`).

## Status: DONE (Phases 0–3, fuzzed)

All four phases are implemented and verified byte-identical against the conformance
suite. The differential fuzzer now also generates inline CALL (a `genCall` producing the
plain/OPTIONAL lateral join, `RETURN *`, the set-op combine with scope-var AND fresh-scan
arms, the uncorrelated global set-op, and the scalar-aggregate body — ~12% of queries,
~2,400 per seed). It compares structurally (row multiset), which is the correct invariant
since intra-group order without ORDER BY is unspecified. 16 seeds × ~2,400 CALL queries
turned up ZERO CALL divergences; a coverage probe confirmed 2000/2000 generated CALL
queries execute (not vacuous both-error). The only divergences the run surfaces are the
pre-existing non-bool-in-AND/FILTER class (engine lenient, TS strict) — unrelated.

Measured perf: the `call/inline` micro-bench is 0.360 ms at the pre-rewrite baseline vs
0.370 ms at HEAD (min of 8) — a ~3% delta, below the repo's ~10% noise floor. The plain
lateral-join exec path is unchanged code; the one structural addition (Phase 0's prov
column) is one extra `Col::Num` per CALL, necessary for every feature above and neutral
at the bench's outer-set size.
