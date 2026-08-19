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
slot *values* — there is no per-outer provenance id.

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

| feature | "RETURN * / element columns" | "set operators" |
|---|---|---|
| `RETURN *` (yield every body binding) | ✅ | — |
| `OPTIONAL CALL` (left-outer null-fill) | ✅ | ✅ |
| body from a FRESH `MATCH (s:Software)` (not a scope var) | — | ✅ (the EXCEPT case) |
| `UNION [ALL]` / `EXCEPT` / `INTERSECT` in the body | — | ✅ |
| uncorrelated `CALL () { … UNION … }` | — | ✅ |

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
`outer_width`, exactly like `correlated_subquery_body`). This is the *same* slot-layout
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
     RETURN-* path the engine doesn't have yet — small addition).

3. **Exec** (`Plan::CallInline`):
   - Provenance-tag: seed `outer + prov(0..n)`, run `body` and each `part` → each carries `prov`.
   - **Combine set-op parts per outer row**: group each part's rows by `prov`, then apply
     `combineRows(op, all)` on the yield-tuples *within each prov group* (UNION distinct/all,
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
- **Phase 1:** `RETURN *` + `OPTIONAL CALL` (correlated-from-scope-var only, no set-ops).
  **Closes "inline subquery CALL with RETURN * / element columns".**
- **Phase 2:** fresh-scan body start + set operators (`UNION [ALL]`/`EXCEPT`/`INTERSECT`),
  correlated and uncorrelated. **Closes "inline subquery CALL with set operators".** This
  is the bulk of the byte-identity risk (per-outer-row combine).
- **Phase 3 (optional):** fold in the scalar-aggregate CALL body (`sum`/`avg`/`min`/`max`).

## Effort

Multi-day, not a bounded fix. Phase 0 + Phase 1 are moderate (parser + a null-pad exec
path over the existing lateral join, once prov is carried). Phase 2 is the large one — a
new per-outer-row set-op combine in the vectorized exec, plus the byte-identity work.
