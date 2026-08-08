# The engine, from scratch

This is the design for the query engine as it would be built knowing what we know
now — not a migration of the current code. The current engine is not the
foundation here; it is the **oracle**. It, plus the six differential fuzzers,
defines the behavior the new engine must reproduce, which is what lets us build
fresh without flying blind (see "Build sequence").

Everything the current system supports is in scope: two query languages (GQL,
Gremlin), the graph store (multi-label nodes/edges, typed columnar properties,
temporal, interned strings, CSR adjacency), property and interval indexes,
pattern matching, hops, filters, projection, aggregation/grouping, ordering,
paging, joins, var-length and shortest paths, path tracking, tags, sack,
side-effects, the graph-algorithm library, transactions, mutations/upsert, and
Arrow egress. The sync engine sits above the core and is out of scope for the
core design (it consumes the engine; it does not change it).

The point of building fresh is to escape the two things that produced the
accidental complexity: **two execution models** (row `Trav`/`apply` and columnar
`Col`) that reimplement every hot step twice, and **two hand-written engines**
(Rust and TS) kept byte-identical by fuzzing. This design collapses the first to
one model, and forces a real decision on the second.

## The foundational fork: one engine, or two? (this is #3, and it is first)

Building fresh, the very first decision is whether there is one engine or two,
because it determines whether a byte-identity problem exists at all.

- **Two hand-written engines (today).** Rust + a pure-TS engine, kept
  byte-identical by six differential fuzzers. This is the single largest source
  of the bugs this codebase keeps fixing — the sync `BigInt` throw, the NaN
  ordering divergence, walk-vs-trail, the `by(__.path())` scoping bug were all
  "two engines drifted." It taxes every feature twice, forever.
- **One engine, Rust → WASM (recommended).** The pure-TS engine is deleted, and
  with it all six differential fuzzers and the byte-identity invariant. TS
  callers use the WASM build. Cost: ~600 KB brotli in the bundle, WASM startup,
  and no pure-JS-only environments (some edge/RN/SSR targets).

**Recommendation: one engine, Rust → WASM** — _if_ the product can tolerate WASM.
It deletes roughly half the surface and the entire class of drift bugs. The
pure-TS engine exists for reach (bundle size, no-WASM runtimes), not speed (Rust
is 1.5–4× it). This is the one genuinely product-level call in this document, and
it should be made before anything else, because it changes the value contract,
the test strategy, and the size of the whole effort.

**If two engines are required**, the rest of this design is unchanged — you
implement the same IR, optimizer, and batch model twice and keep the fuzzers. The
win is smaller (you keep _one thing_ twice instead of _two things and a bridge_
twice), but the architecture is the same. Everything below assumes one engine;
where two changes something, it is noted.

## The core idea: one batch model, lineage as an operator strategy

The current engine has two whole worlds — `Trav` (a row carrying value + path +
tags + sack) run by the `apply` interpreter, and `Col` (a bare value column) run
by the columnar path — with a translation layer and decline machinery between
them. They exist as two worlds only because they were built separately. From
scratch there is one:

**One batch type.** A batch is a columnar block of elements: a value column
(unboxed where the type allows), plus an _optional lineage sidecar_ — path as an
Arrow-style list column (values buffer + offsets), tags as named columns, sack as
a column. Lineage columns are present only when the plan needs them.

**One operator set, each operator with two strategies.** An operator (hop,
filter, project, aggregate, dedup, …) has a **bulk strategy** — the fast
columnar/set algorithm (`walk_count`, backward `reach_back`, degree products,
bitset dedup) — and a **lineage-preserving strategy** — the per-element form that
also extends the path/tag/sack sidecar. Both operate on the _same_ batch, share
the _same_ storage access and value contract, and live in the _same_ function.
The strategy is chosen by whether the plan requires lineage above this operator.

This is the columnar-traverser idea, and it works from scratch precisely because
there is no `apply` to relocate: you write each operator once, with a conditional
on lineage, over one batch type. There is no `Trav`-vs-`Col` type split, no
second interpreter, no translation, no decline path. The per-element logic still
exists (path tracking is inherently per-element), but it is a branch inside one
operator, not a parallel universe.

Why this is correct and not a fantasy: the reason path and bulk-collapse are
"opposed" is that a bulk op (count, reach, degree-product) _produces no
per-element result to attach a path to_. But an operator only needs the
lineage-preserving strategy when a consumer above it reads lineage — and a
consumer that reads lineage is, by definition, enumerating, so the bulk strategy
was never applicable there anyway. The lineage requirement is exactly the signal
for which strategy is valid. There is no case where you need both.

## The layers

```
GQL text ─▶ GQL front-end ─┐
                           ├─▶ LOGICAL IR ─▶ optimizer ─▶ physical plan ─▶ BATCH EXECUTION ─▶ storage
Gremlin text ▶ Grem front-end                (rewrite      (strategy per      (one batch type;
                           ┘                   rules)        operator from      operators run bulk
                                     │                       lineage need)      or lineage strategy)
                                     └──────────────── VALUE CONTRACT ───────────────────────┘
```

### Storage — kept, because it is driven by physics

Columnar typed property store, interned strings, CSR adjacency, SoA temporal,
property and interval (RI-tree) indexes, multi-label nodes and edges. A
from-scratch design lands here anyway — this shape is forced by cache locality
and columnar access, not by history. The current storage is mature and correct;
it is ported, not redesigned. This is the honest part where "from scratch" means
"the same, deliberately."

### Value contract — one module, representation and semantics together

The value model (f64 numeric, interned string ids, temporal, bool, list,
map/record, null-as-a-stored-value) **and** its semantics — total order,
equality, coercion, null policy, NaN policy — are one module, defined once. The
storage columns and the runtime batch share the same representation and the same
comparators. Every place that today re-states NaN handling or equality (eval,
aggregation, sort, seek) instead consults the contract.

With one engine this module exists once. With two, it is the one module you
mirror, and the fuzzers guard exactly it — which is where most drift lived.

### Logical IR — a language-neutral graph-relational algebra

One algebra, designed so neither language's concepts leak in. Operators:

- **Relational:** Scan (label bucket or universe), Filter, Project, Aggregate
  (group keys + aggregate functions), Join, Order, Page, Distinct.
- **Graph:** Expand (a hop: direction + edge-type set + optional per-hop node/edge
  predicate), VarLength (quantified expansion, with trail-vs-walk as an explicit
  flag — the `{2,2}` trap is a property of the node, not a spelling), ShortestPath,
  AlgorithmCall.
- **Effects:** Insert/Set/Remove/Delete/Merge, and the side-effecting collectors
  (aggregate-to-bag/store/subgraph/sack).

Every node carries a **lineage requirement**: does any consumer above it read
path, tags, or sack. This is computed once, on the built IR — the generalization
of today's `needs_path`/`reads_tags`, promoted from a retrofit gate to a
first-class plan property that the physical planner reads.

Binding is uniform: GQL variables and Gremlin `as()` tags both become "bind slot
N." A path is a lineage annotation, not a step. The IR does not know which
language produced it.

### Front-ends — thin, and the only language-aware code

GQL parser → lower to IR. Gremlin parser → lower to IR. Each language's _contract
quirks_ — GQL's NaN-as-no-value in predicates vs Gremlin's filtering; GQL trail
semantics vs Gremlin's; row-order and group-order conventions — are encoded as
**IR attributes**, not as forks in the executor. The executor is language-blind;
the front-end is the only place a language concept exists.

### Optimizer — rewrite rules on the IR, written once

A fixed rule set applied to fixpoint (an ordered pass to start; a cost model only
when a rule genuinely needs cardinality — e.g. seed-side selection). Each rule is
a pure, meaning-preserving IR→IR function, tested in isolation, and it fires on
the normalized IR regardless of surface spelling or source language. The wins
this codebase hand-rolled as fastpaths are the rules:

- aggregate pushdown (count without enumerate)
- semi-join reordering (selective EXISTS sweeps backward)
- group-by-aggregate as degree product
- duplicate-elimination pushdown (distinct as bitmap / frontier collapse)
- comma-join as a product of independent branches
- predicate pushdown, constant folding, seed-side selection

This is the layer whose _absence_ caused the fastpath ladder — the same idea
implemented eight times because a shape recognizer matched syntax, not meaning. A
rule on the IR cannot have that failure mode.

### Physical execution — batches, strategy chosen by lineage

The physical planner lowers each logical operator to its execution, choosing the
bulk or lineage-preserving strategy from the lineage requirement. Execution is
the batch engine: pull (or push) batches through the operator pipeline, vectorized
where lineage-free, per-element (still over the batch) where lineage is required.
One batch type, one operator set, one value contract, one storage.

## What sits above the core (ported, not redesigned)

- **Graph-algorithm library** (degree, WCC, label-prop, PageRank, shortest-path,
  centrality): calls into the batch engine / `seek`; the byte-identity summation
  rules become value-contract rules. Ported.
- **Transactions** (undo-log, deferred constraint checks, event buffering) and
  **constraints/validation**: a layer wrapping mutation execution. Ported.
- **Arrow egress**: the batch type _is_ already columnar and Arrow-shaped, so
  egress is close to free — arguably better than today, where it re-encodes.
- **Sync engine** (CDC, demand-fill, retry): sits entirely above; consumes the
  engine unchanged.

## Build sequence — grow against the oracle, then swap

The existing engine and the six differential fuzzers are an **executable spec**.
That is what makes building fresh safe.

1. **New engine as a separate crate/module**, not wired into the product. The old
   engine keeps shipping untouched throughout.
2. **Value contract + storage first** (storage largely ported). Establish the
   batch type and the value semantics; conformance-test the value contract
   against the old engine's semantics directly.
3. **Grow operator by operator.** For each logical operator, implement both
   strategies, then run the _existing fuzzer corpus_ new-vs-old and require the
   new engine to match the old (the old is the oracle). Add the operator's rules
   as they come.
4. **Both front-ends lower to the IR.** GQL and Gremlin conformance suites run
   against the new engine; must match the old.
5. **Soak: run the full fuzzer suite + all conformance tests against the new
   engine until it passes everything the old one did.** Only now is it proven.
6. **Swap.** New engine becomes the product engine; the old one runs in parallel
   as an oracle for a soak period, then is deleted. If #3 chose one engine, the
   entire pure-TS engine and the six fuzzers are deleted at this step — the
   largest single reduction in the effort.

At no point does the product ship the unproven engine, and the swap is gated on
"reproduces everything the oracle does." This is the safety the migration plan
was reaching for, achieved without letting the old code's structure dictate the
new one's.

## Honest assessment

- **Does it get smaller?** The row/column duality collapses to one model, which
  removes `apply` (~1,100 lines), the `Col`/`Trav` split, `to_gql`, and the
  decline machinery — replaced by one operator set that is larger per-operator
  (two strategies) but singular. Plausibly meaningfully smaller; the certain win
  is one world instead of two-plus-a-bridge. If #3 chooses one engine, the whole
  pure-TS engine and six fuzzers go too — that is the large, certain reduction.
- **What it definitely fixes:** the fastpath-ladder bug class (optimizations are
  rules on the IR), the per-language asymmetry (one executor), the drift bugs (if
  #3 chooses one engine, there is nothing to drift), and the scattered value
  policy.
- **What it costs:** a real rebuild. Storage and algorithms port cheaply; the IR,
  optimizer, and batch engine are new construction. The oracle+fuzzer strategy
  makes it safe but not fast — this is a multi-phase effort measured in the same
  order as the original build, discounted by everything already learned and by
  the storage/algorithm layers porting largely intact.
- **The one thing that must be decided by a human first:** #3. Everything
  downstream — the value contract's shape, the test strategy, the total size —
  turns on it.

## Open decisions

1. **#3: one engine (Rust→WASM) or two.** The foundational fork. Recommended:
   one, if the product tolerates WASM. Needs the product constraints (bundle
   budget, required no-WASM runtimes) to settle.
2. **Optimizer ambition.** Ordered fixpoint pass first; cost-based (Cascades)
   only when a rule needs cardinality. Recommend starting simple.
3. **Batch width / execution model** (pull vs push, batch size). An
   implementation choice to settle early with a microbenchmark against the
   oracle's numbers, not by argument.
