# TinkerPop parity gaps

Tracked gaps surfaced while importing TinkerPop reference scenarios as
tests. Each item is something where our v2 implementation diverges from
the canonical TinkerPop semantics in a way that's worth fixing (versus
intentional design differences, which live as inline `// doc:` comments
in the test files and aren't tracked here).

## Data-model differences (not gaps)

TinkerPop's reference graph is a **multi-graph with first-class
properties** — each property is itself an `Element` with an ID, label,
and meta-properties, and a key can hold multiple values (cardinality:
single/list/set). Our property-label graph is simpler: properties are
a flat `Record<string, unknown>`, one value per key, no property
identity.

Several apparent "drifts" from TinkerPop docs are direct consequences
of this model choice and are _not_ tracked as gaps:

- `valueMap()` returns `{key: value}` instead of `{key: [value]}`.
  TinkerPop wraps everything in lists to be cardinality-polymorphic; we
  don't need to because we don't have list cardinality.
- `propertyMap()` returns flat values instead of property objects
  (`vp[name->marko]`). We don't have property objects.
- `id()` on the result of `properties()` emits `undefined`. Properties
  don't have identity in our model.

If we ever adopt multi-property semantics in core, these become real
gaps. Until then, they're documented in `// doc:` comments inline.

### Null is a first-class property value (deliberate divergence)

TinkerPop has no null property values — `property(k, null)` is not a way to
store a null. **We store null as a first-class value**, present and distinct
from an absent property:

- `property(k, null)` **stores** a present null. `values(k)` yields `null`,
  `valueMap()` includes `k`, and `has(k)` is true. (A read alone can't tell a
  stored null from an absent key — both yield `null` — but `k in
element.properties`, the codecs, and the conformance corpus can.)
- Because setting null no longer removes, **deleting a property is a separate
  operation**: `.properties(k).drop()` — the TinkerPop-canonical idiom, wired in
  both engines (traverse to the property element, `drop()` it). GQL uses
  `REMOVE x.k`; `SET x.k = null` likewise stores a null.

This is symmetric across the pure-TS engine (JS objects keep `null` keys), the
Rust core, all five serialization codecs, and GQL's SQL-style null-typed value
model. It sidesteps the "`{ b: string | null }` becomes an unrepresentable
union" footgun a JS library would otherwise hand its users, and aligns with ISO
GQL (which has a first-class `null` type and a separate `REMOVE` statement)
rather than Cypher's "`SET = null` removes" convention. Sibling policy: the
NaN-ordering decision.

### Labels and property keys are validated (well-formedness)

Gremlin takes arbitrary strings as labels/keys, but a name that can't round-trip
through every codec is rejected at the graph's mutation boundary (a coded
`InvalidValue` — via the fault channel, surfaced by `try_run`):

- A **label / edge type** must be non-empty and free of `::` (GraphSON joins a
  node's labels with `::`, so a `::` inside one label is ambiguous there; an
  empty label collapses to "no labels" in GraphSON/CSV). A single `:` is fine.
- A **property key** must be non-empty (an empty key has no CSV column header /
  `key:value` pg-text form). Keys may contain `::` — they're never `::`-joined.

Enforced identically in the Rust core (`validate_label` / `validate_prop_key`,
at codec ingestion + `addV`/`addE`/`property`) and the TS core (`validateLabel` /
`validatePropertyKey`, at `addVertex`/`addEdge`/`addLabelTo*`/`setProperty`).

## Feature gaps surfaced by docs

These are missing capabilities, not behavioral drift. Tests note them
where applicable.

(The TS engine is currently gap-free here: `match()`, `subgraph()`, and
`shortestPath()` are implemented, `math()` resolves `as_`-bound names projected
by `by()`, and `by()` supports comparator/token forms. The gremlin test package
has no skipped tests.)

> **Known limitation — `order().by()` over Map/projection rows (BUG A).** The
> `by('<key>')` / `.by(select('<key>'))` modulators only project a property off a
> `Vertex`/`Edge`. Over `project()` rows (plain objects) or `group`/`groupCount`
> Maps they do NOT extract the named key — the whole row/Map reaches the
> comparator, which faults ("cannot order an element with an element" /
> "cannot order null with null"). This is **at parity with the Rust core**
> (`eval_by` has the identical element-only behavior by design), so it is a shared
> limitation, not TS drift. TinkerPop's `by(String)` is likewise element-only; the
> map-extraction form would be a `select`-from-Map feature, deferred. Sort such
> rows in application code after materializing them.

### Cross-engine parity (TS engine ⟷ Rust core)

The Rust gremlin engine (`crates/lenke-core/src/gremlin`) mirrors the TS package
step-for-step. Parity is verified by a differential runner
(`packages/native/src/gremlin-conformance.test.ts`): a single TS `Plan` is run
through the TS engine in-process and the Rust core over `bun:ffi`, and the
canonical JSON results are compared.

The only remaining divergence is **closures** — the TS-superset steps that
cannot exist in a text-driven engine:

- **Closures** (`map(fn)`/`filter(fn)`/`sideEffect(fn)`/`fold(seed, fn)`): a JS
  closure can't cross the Groovy-text boundary at all. This is permanent; use
  the sub-traversal forms for cross-engine plans. `planToGremlin` classifies
  these as `tsOnly` (asserted, not silently skipped).

Now at parity (previously TS-only):

- **`math()`** — the Rust engine ships the same infix evaluator (`+ - * /`,
  parens, `_`/`as_`-bound operands, cycling `by()` projection); non-numeric
  operands fault on both engines.
- **`branch()`** — `branch(test).option(match, …)…option(none, …)`; the `none`
  default is TinkerPop's `Pick.none`. Routes each traverser by its test result.
- **`regex`** predicate — same unanchored match semantics as JS `RegExp.test`
  (backed by the `regex` crate). **Two narrow divergences on the native engine:**
  (1) the `regex` crate is linear-time and does not support backreferences or
  lookaround, so a pattern using those (which JS `RegExp`/Java `Pattern` accept)
  is rejected at parse time; (2) to keep the binary lean, only the `unicode-perl`
  and `unicode-case` tables are compiled in — so Unicode-aware `\d \w \s \b` and
  case-insensitive folding work, but exotic property classes (`\p{Script=…}`,
  `\p{Age=…}`, general-category `\p{L}`, grapheme `\X`) are unavailable natively.
  Anchors, character classes, quantifiers, and alternation behave identically on
  both engines.
