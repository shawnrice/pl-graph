# The engine, from scratch

This is a self-contained design for the query engine, built from the ground up.
It does not reference or depend on any existing implementation — read it as if
nothing has been written yet.

Scope: two query languages (GQL and Gremlin), a graph store (multi-label
nodes/edges, typed properties, temporal values, interned strings, adjacency),
property and interval indexes, pattern matching, hops, filters, projection,
aggregation and grouping, ordering, paging, joins, variable-length and shortest
paths, path tracking, tags, sack, side-effects, a graph-algorithm library,
transactions, mutations and upsert, and columnar (Arrow) egress.

Two design commitments shape everything below:

- **One neutral algebra.** Both languages compile into a single language-agnostic
  IR. Everything beneath the front-end is blind to which language it came from.
- **One execution model.** A single columnar batch model, where path/tag/sack
  lineage is an optional per-operator strategy — not a second, separate engine.

## Two implementations, one design

The engine ships as **two implementations of this one design — one in Rust, one
in TypeScript.** Both are first-class. They are not kept in lockstep byte-identity
by continuous fuzzing; they are kept in **agreement**, which is a looser and more
maintainable coupling, and the neutral IR is what makes it tractable:

> **Note (current practice).** This section describes the intended coupling. In the
> shipped project the two engines are in fact held to **byte-identity**, enforced
> per-change by the differential / codec / write / injection fuzzers (TS vs FFI) and
> the backend-parity fuzzer (wasm vs FFI) — tighter than the "agreement" this design
> anticipated.

- **The IR and its execution semantics are the specification.** Both engines
  implement the _same_ algebra. Agreement is therefore structural — it follows
  from both implementing one spec — rather than enforced by two hand-written
  engines chasing each other.
- **A single language-agnostic conformance suite** (query → expected result)
  runs against both. It is the shared source of truth; a divergence is a
  conformance failure in CI, not a subtle byte-drift discovered later.
- **Order that is genuinely unspecified stays unspecified.** Where results are
  set-based (row order, adjacency order, map-key order absent an `ORDER BY`), the
  spec says "unordered" and the conformance compares as sets. The two engines do
  not have to make identical arbitrary choices — only agree on what is actually
  defined. This is where lockstep byte-identity overreaches, and dropping it is
  the point.
- Differential testing between the two implementations is a periodic check, not a
  per-commit gate.

The design is implementable in both: value columns are typed arrays
(`Float64Array` and friends in TS), lineage sidecars are offset+value buffers.
Nothing in the batch model requires a systems language.

## The core idea: one batch model, lineage as an operator strategy

A naive engine has two temptations that pull apart: a _row_ model that carries a
traverser (value + path + tags + sack) so it can answer path/tag queries, and a
_columnar_ model that carries bare value columns so it can go fast. Built
separately, these become two whole worlds with a bridge between them. Built
together, they are one:

**One batch type.** A batch is a columnar block of elements: a value column
(unboxed where the type allows), plus an _optional lineage sidecar_ — path as a
list column (a values buffer + per-row offsets), tags as named columns, sack as a
column. The lineage columns exist only when the plan needs them.

**One operator set; each operator has two strategies.** An operator (hop, filter,
project, aggregate, dedup, …) has:

- a **bulk strategy** — the fast set/columnar algorithm: count without
  enumerating, a backward reachability sweep for a selective `EXISTS`, a
  degree-product for a grouped count, bitset deduplication; and
- a **lineage-preserving strategy** — the per-element form that also extends the
  path/tag/sack sidecar.

Both run over the _same_ batch, share the _same_ storage access and value
contract, and live in the _same_ function. The strategy is chosen by whether the
plan requires lineage above this operator.

**Why this is one model and not two in disguise.** There is no separate
interpreter and no translation layer, because there is only one batch type and
one operator set — the per-element path is a branch inside each operator, not a
parallel world. And it is _correct_ to choose per operator, because path and
bulk-collapse are only "opposed" in that a bulk op (count, reach, degree-product)
produces no per-element result to attach a path to. But an operator needs the
lineage strategy exactly when a consumer above it reads lineage — and such a
consumer is enumerating, so the bulk strategy was never valid there anyway. The
lineage requirement is precisely the signal for which strategy applies. No query
needs both for the same operator.

## The layers

```
GQL text ─▶ GQL front-end ─┐
                           ├─▶ NEUTRAL IR ─▶ optimizer ─▶ physical plan ─▶ BATCH EXECUTION ─▶ storage
Gremlin text ▶ Grem front-end               (rewrite      (strategy per      (one batch type;
                           ┘                  rules)        operator, from     bulk or lineage
                                     │                      lineage need)      strategy per op)
                                     └──────────────── VALUE CONTRACT ──────────────────────┘
```

### Value contract — representation and semantics, defined once

One module owns both the value representation (a single numeric type — f64;
interned string ids; temporal; bool; list; map/record; null as a first-class
stored value) **and** its semantics: total order, equality, coercion rules, null
policy, NaN policy. Storage columns and runtime batches share the same
representation and the same comparators. There is exactly one place that answers
"how do two values compare," "are these equal," "what does summing a
non-number do," "where does NaN sort." Every operator and the storage layer
consult it; none restate it.

This module is the natural seam between the two implementations — it is small,
total, and the thing most worth having a shared conformance suite pin exactly.

### Storage — columnar, because the physics demands it

Typed columnar property store; interned strings; adjacency in a
compressed-sparse-row layout; temporal values in per-type struct-of-arrays;
property indexes and interval (relationship-tree) indexes for temporal/range
queries; multi-label nodes and edges. This shape is forced by cache locality and
bulk scan speed, not by taste — a from-scratch design arrives here on the merits.
The batch model reads directly from these columns, so a value column in a batch
is often a borrow of a storage column, not a copy.

### Neutral IR — a language-agnostic graph-relational algebra

One algebra, designed so that neither language's surface concepts leak into it.

- **Relational operators:** Scan (a label bucket, an index range, or the
  universe), Filter, Project, Aggregate (group keys + aggregate functions), Join,
  Order, Page, Distinct.
- **Graph operators:** Expand (one hop: direction, edge-type set, optional
  per-hop node/edge predicate), VarLength (quantified expansion, carrying
  trail-vs-walk as an explicit flag so the two are never conflated), ShortestPath,
  AlgorithmCall (the graph-algorithm library entered as a first-class operator).
- **Effect operators:** Insert, Set, Remove, Delete, Merge/upsert, and the
  side-effecting collectors (aggregate-to-a-named-bag, store, subgraph, sack).

Binding is uniform: a GQL variable and a Gremlin `as()` label are both "bind slot
N." A path is a lineage annotation on the plan, not a step in it. Every operator
carries a **lineage requirement** — whether any consumer above it reads path,
tags, or sack — computed once on the built IR and read later by the physical
planner to pick each operator's strategy.

The IR does not know which language produced it, and nothing below the front-end
branches on the source language.

### Front-ends — thin, and the only language-aware code

A GQL parser lowers to the IR. A Gremlin parser lowers to the IR. Each language's
genuine contract differences — GQL treating NaN as no-value inside a predicate
versus Gremlin filtering it; trail semantics; the group/row ordering a language
guarantees — are encoded as **attributes on IR nodes**, not as forks in the
executor. The executor is language-blind; the front-end is the single place any
language concept exists.

This is the property that pays off most: a lexer/parser per language, each a thin
compiler into the shared algebra, and one engine underneath.

### Optimizer — rewrite rules on the IR, written once

A fixed set of meaning-preserving IR→IR rewrite rules, applied to a fixpoint (an
ordered pass to begin; a cost model added only when a rule genuinely needs
cardinality — e.g. choosing which end of a pattern to seed from). Each rule is a
pure function of the IR, tested in isolation, and fires on the _normalized_ IR —
so it applies to every surface spelling in both languages at once. The
optimizations that a naive engine would hand-write per shape are, here, rules:

- predicate pushdown and constant folding,
- aggregate pushdown (count without enumerating),
- semi-join reordering (a selective `EXISTS` evaluated from the narrow end
  backward),
- group-by-aggregate as a degree product,
- duplicate-elimination pushdown (distinct as a membership bitset / frontier
  collapse),
- a comma-join off a shared start as a product of independent branches,
- seed-side selection.

Because a rule matches the algebra, not the syntax, there is no way for a
differently-worded-but-equivalent query to miss it. A conformance group asserts,
per rule, that equivalent spellings produce the same result _and_ cost within a
factor of each other.

### Physical planning and execution — batches, strategy from lineage

The physical planner lowers each logical operator to its execution and chooses
the bulk or lineage-preserving strategy from the operator's lineage requirement.
Execution pulls batches through the operator pipeline: vectorized where no lineage
is required, per-element (still over the one batch type) where it is. One batch
type, one operator set, one value contract, one storage — no second engine to
fall back to.

## The graph-algorithm library, transactions, egress

- **Algorithms** (degree, connected components, label propagation, PageRank,
  shortest path, centrality) are `AlgorithmCall` operators over the batch engine.
  Their determinism rules (e.g. a fixed summation order for reproducibility) are
  value-contract rules, so both implementations get them for free.
- **Transactions** (an undo log, deferred constraint checks, buffered events) and
  **constraints/validation** wrap mutation execution as a layer above the
  operators.
- **Columnar egress** is nearly free: the batch type is already columnar, so
  emitting Arrow is a framing of buffers the engine already holds, not a
  re-encode.

## Build order

A self-contained order for constructing the engine; the two implementations
proceed against the shared conformance suite, which is written first as the
specification.

1. **Value contract + storage.** The foundation both implementations share; pin
   the value contract with conformance tests before anything reads it.
2. **The batch type and a first operator (Scan → Filter → Project).** Establish
   the one-batch, two-strategy shape end to end on the simplest pipeline.
3. **Operators, in dependency order:** Expand, Aggregate/group, Order/Page,
   Distinct, Join, VarLength, ShortestPath, the effect operators. Each lands with
   both strategies and its conformance cases.
4. **Optimizer rules,** added as their operators exist; each with its
   equivalent-spellings conformance group.
5. **Front-ends:** GQL and Gremlin parsers lowering to the IR; the full
   per-language conformance suites run against the engine.
6. **Algorithms, transactions, egress** as the layers above.

Both the Rust and TS implementations follow this order and are checked against the
same conformance suite at each step; periodic differential runs between them catch
anything the suite misses.

## What this design buys, honestly

- **No fastpath ladder.** Optimizations are rules on the algebra, so one idea is
  written once and cannot be missed by a reworded query or a second language.
- **No per-language execution asymmetry.** One executor beneath two thin
  front-ends; a fix or a speedup lands for both languages at once.
- **No two-execution-model duplication.** One batch model with a per-operator
  lineage strategy, instead of a row interpreter and a columnar path with a
  bridge between them.
- **Agreement without lockstep.** Two implementations of one specified algebra,
  kept in step by a shared conformance suite rather than continuous byte-identity
  fuzzing — the coupling is looser and the drift surface is smaller.
- **One place for value semantics,** which is where cross-implementation
  disagreements otherwise breed.

The irreducible costs remain and are not hidden: two languages need two parsers,
two implementations need the work done twice (bounded by a shared spec and a
shared conformance suite, not by lockstep), and path-carrying queries are
genuinely per-element work no columnar trick removes — they are simply the
lineage strategy of the same operators, not a separate world.

## Open decisions

1. **Optimizer ambition.** Ordered fixpoint pass first; a cost-based optimizer
   only when a rule needs cardinality estimates. Recommend starting simple.
2. **Execution model** (pull vs push; batch width). Settle early with a
   microbenchmark, not by argument.
3. **How much the two implementations share by generation vs by hand.** The value
   contract and the conformance suite are shared artifacts; whether any operator
   logic is generated from one source or written twice against the spec is a
   build-tooling choice to make once the operator shape is stable.
