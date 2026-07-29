# Property indexes

Label lookups are O(1) (`verticesByLabel`), but matching on a property value —
`has('name', 'marko')`, `WHERE n.age > 30` — is otherwise a full scan over every
element. A `PropertyIndex` is the property analog of the label index: an opt-in,
per-key secondary index that turns those into an index seek.

## Declaring indexes

Indexes are **opt-in per key** — an ephemeral in-memory graph (especially in a
browser tab) shouldn't pay heap for keys nobody filters on. Declare the ones you
query with `createIndex({ on, kind, keys })`:

```ts
graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] }); // backfills existing vertices, then stays current
graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
graph.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });

graph.vertexIndexes(); // ['age', 'name']
graph.dropVertexIndex('age');
```

`on` is `'vertex'` or `'edge'`; `keys` is the property name(s). **`kind: 'hash'`
is what you want for almost everything** — it's an ordered map (a `BTreeMap` in
the native engine; a bucket map + sorted distinct-value list here) that serves
equality, `IN`, and range from one structure. The other kind, `'interval'`, is a
specialized native-only index for `[lo, hi)` containment/overlap — see below.

**Nested / dotted keys.** Index a field _inside_ a map/record property by naming
the dotted path; the whole path is the key:

```ts
graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['meta.errorId'] });
// WHERE e.meta.errorId = $x  then seeks the dotted index instead of scanning.
```

Maintenance is automatic and synchronous: inserts, removes, and property writes
keep the index current, and it survives `clone()`/`snapshot()` without aliasing
the source. Bulk loads
that run with events disabled are still indexed, because maintenance lives in the
graph's mutation methods, not in event listeners.

## Querying directly

Each indexed key answers both equality and range queries from one structure:

```ts
graph.getVerticesByProperty('name', 'marko'); // Set<Vertex>, O(1)
graph.getVerticesByPropertyRange('age', { gte: 30, lt: 40 }); // half-open / closed
graph.getEdgesByProperty('weight', 1.0);
```

Internally an equality query hits a type-tagged bucket; a range query binary-
searches a sorted list of the key's _distinct_ values and unions their buckets.
Because the sorted list holds distinct values, adding or removing an element at
an already-present value is O(1) — only a value's first appearance or last
removal moves the sorted list.

## Automatic query seeding

You rarely call the above directly. The Gremlin and GQL engines **seed from a
declared index automatically** — no query change needed. The optimization is
inert unless an index exists, so existing queries are unaffected.

**Gremlin** — a `V()` / `E()` source followed by a `has(...)` on an indexed key:

```ts
traversal(V(), has('name', 'marko')); // eq      → bucket
traversal(V(), has('name', within('a', 'b'))); // within  → union of buckets
traversal(V(), has('age', gt(30))); // range   → sorted-index slice
traversal(V(), has('name', startsWith('m'))); // prefix  → [m, n) slice
traversal(E(), has('weight', 1.0)); // edges, same machinery
```

**GQL** — an element-pattern equality, or a seekable conjunct of a clause /
inline `WHERE` (drawn only from `AND`-chains, so each is a necessary condition):

```sql
MATCH (p:Person {name: 'marko'}) ...
MATCH (p:Person) WHERE p.age > 30 ...
MATCH (p:Person) WHERE p.name IN ['marko', 'josh'] ...
MATCH (p:Person) WHERE p.age >= 29 AND p.age < 35 ...   -- coalesced to [29, 35)
```

When several constraints are indexable, each one's cardinality is estimated
cheaply (`countEquals` is O(1); `countRange` sums bucket sizes over the sorted
slice — neither builds a set) and the **most selective** is seeded; the rest stay
as residual filters the engine re-applies. So the seed is always a _superset_ of
the true matches and results are identical to the unindexed scan.

GQL goes one step further: for a fixed-length path it scores **both ends** from
those same counts and seeds from whichever is more selective, walking the
relationship backwards if needed. `MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name =
'josh'` seeds from `josh` and walks back to `a`, rather than scanning every `a`.
This is the only cardinality-driven _planning_ decision in the engine — there's
no cost model or join-order search beyond picking the cheaper anchor.

## The type-strict caveat

Range and prefix seeks use the index's **type-strict** ordering: a numeric bound
matches only numeric values, never JS-coercible strings (`"40" > 30` is `true`
in plain JS, but `40`-the-number and `"40"`-the-string sort into different type
blocks in the index). For type-consistent data — the norm — results are
identical to the scan. For a key that genuinely mixes numbers and numeric
strings, declaring an index changes range results to the type-strict reading.
Equality and `within` are always exact (`===`), so they're never affected.

## Interval indexes (native engine)

A hash index answers "value = / < / IN". It cannot answer, in one seek, "which
rows' interval **contains** this point" or "**overlaps** this window" — that's a
2-D predicate over a _pair_ of endpoints. When a row carries a half-open interval
`[lo, hi)` (**lo inclusive, hi exclusive**) and you query it by containment or
overlap, use an interval index instead — an RI-tree, native-only
(`@lenke/native`; the pure-TS engine throws on `kind: 'interval'`):

```ts
g.createIndex({ on: 'edge', kind: 'interval', keys: ['vf', 'vt'] }); // valid-from, valid-to
// as-of — which edges were valid at $v? (vf inclusive, vt exclusive):
//   MATCH ()-[r:EMPLOYS]->() WHERE r.vf <= $v AND r.vt > $v RETURN r
```

Half-open means `hi` is the first instant **not** covered: `[10, 20)` contains
`10`, excludes `20`. Model an open-ended ("still current") interval with a
far-future sentinel for `hi`, never `null` — the stab needs both endpoints; a
zero-width `[x, x)` contains nothing. Reach for it for **bitemporal / valid-time
history** (two interval indexes — valid `[vf, vt)` + transaction `[tf, tt)` —
cover a bitemporal as-of), **reservations/calendars** (overlap), or any **numeric
range** (price bands, version ranges) — it isn't temporal-only. A row that stores
a single instant, not a `[start, end)` pair, is a hash index, not an interval one.
