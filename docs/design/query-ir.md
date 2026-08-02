# What GQL and Gremlin could share

Status: **not built, deliberately deferred.** This records the shape, what it
would be worth, and what has to exist before it is safe to attempt. It replaces
an earlier note that covered only the GQL half of the same problem.

## What is already shared

Both engines already bottom out in one place. `Graph::vertices_by_prop`,
`Graph::edges_by_prop`, the `RangeBound` / `IdxKey` range seeks, the label
buckets, the edge-type buckets — all of `graph.rs`, reached identically from
`gql/eval/scan.rs` and from `gremlin/exec.rs`.

The storage access layer is not the problem.

## What is not

One layer up sit two independent **recognizers**, which compute the identical
thing — `Option<Vec<u32>>`, a seed of element ids — from two different plan
shapes:

|            | GQL                                                                                                               | Gremlin                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------ |
| reads      | `CExpr`, `CNode`                                                                                                  | `&[Step]`                |
| lives in   | `prop_index_hint` + `node_index_seed` + `edge_prop_seed` + `scan_start_seed` (`scan.rs`), `cmp_bound` (`eval.rs`) | `index_seed` (`exec.rs`) |
| roughly    | 350 lines across two files                                                                                        | 118 lines                |
| gaps found | 5                                                                                                                 | 3                        |

Eight gaps, and every one was the same bug: the recognizer pattern-matches on
the _surface shape_ of a plan, so each new spelling needs its own arm, arms drift
apart, and nobody can enumerate the spellings still missing.

| spelling                                                      | cost vs. the seeking form |
| ------------------------------------------------------------- | ------------------------- |
| GQL: clause `WHERE u.k = $x` before a traversal vs. `{k: $x}` | 60x                       |
| GQL: `u.k IN [$a, $b]`                                        | 220x                      |
| GQL: `$x = u.k` (constant first)                              | 107x                      |
| GQL: `u.k = $a OR u.k = $b`                                   | 220x                      |
| GQL: `5 <= u.n AND 9 >= u.n` (grouped path)                   | 197x                      |
| Gremlin: `V().hasLabel('P').has('k', v)`                      | 207x                      |
| Gremlin: `E().hasLabel('R').has('w', 5)`                      | 553x                      |
| Gremlin: the same with a traversal on the end                 | 346x                      |

Every one returned the **correct answer**, so no correctness test could catch
them. Each was found by hand, or by an equivalence test written after the
previous one.

### They have already drifted

Not hypothetically — today, on `main`:

- **Choosing among several seekable filters.** GQL's `And` arm evaluates every
  seekable conjunct and takes `min_by_key(Vec::len)`, the smallest candidate set.
  Gremlin takes the **first** one it finds. So
  `has('country', 'US').has('ssn', $x)` seeds from the wrong side in one engine
  and the right side in the other.
- **Disjunctions.** GQL unions the branches of an `OR` of equalities, seeding
  when — and only when — every branch can. Gremlin's `or(...)` step stops the
  search outright and falls back to a scan.
- **Reversed operands.** GQL handles `$x = u.k`. Gremlin has no equivalent
  spelling to get wrong today, but it also has no `flip_compare`, so the first
  time one appears it will be a third implementation.

Nothing prevents this drift. The two files are fixed months apart by whoever
notices.

## The layer worth building

Not a new IR. Both engines already lower to a plan the planner reads; adding a
third representation above them would not help. What is missing is a shared
representation of the thing both recognizers are trying to _produce_:

```rust
/// Everything a front end has learned about ONE element variable.
struct ElementConstraints {
    labels: Option<LabelSet>,
    /// Conjunctive — all must hold.
    props: Vec<(Arc<str>, CompareOp, IdxKey)>,
    /// Disjunctive — IN-lists and OR-of-equalities, already folded together.
    unions: Vec<(Arc<str>, Vec<IdxKey>)>,
}

fn seed(graph: &Graph, edge: bool, c: &ElementConstraints) -> Option<Vec<u32>>;
```

Each front end writes a small **lowering** into it — GQL walks a `CExpr`,
Gremlin walks the leading run of element-filter steps — and everything below
that line exists once.

### Normalization lives there

Rewrites that make equivalent predicates identical, applied before anything
inspects them:

```
$x = u.k                  ->  u.k = $x          (constant to the right)
5 <= u.n                  ->  u.n >= 5          (operator flipped with operands)
u.k = $a OR u.k = $b      ->  u.k IN [$a, $b]   (same key, all equality)
(a OR b) OR c             ->  a OR b OR c       (flatten)
u.k IN [$a]               ->  u.k = $a          (singleton)
```

The recognizer then handles canonical forms only. It gets **smaller** while
covering **more** spellings, which is the opposite of the trajectory it is on
now.

### What it would fix on the day it lands

- Gremlin gets selectivity-based seed choice, `OR`-union seeding, and every
  future spelling, without a second implementation.
- GQL gets whatever Gremlin's recognizer learns next, for the same reason.
- The `cmp_bound` path in `eval.rs` — which is separate code, and which the fix
  for the single-comparison operand order did **not** reach — stops being
  separate.

### Cost

Roughly 250 shared lines replacing roughly 440 duplicated ones, plus about 150
lines of GQL lowering and 50 of Gremlin lowering. Normalization runs once per
plan, and prepared plans are cached, so its cost lands on parse rather than
execution — where the current approach pays per _execution_, trying every arm
against every predicate on every query.

### Why it is still deferred

A rewrite pass that changes what the planner sees is exactly the kind of change
that silently returns wrong rows: fold a disjunction wrongly and matches vanish;
flip an operator wrongly and you get the complement. Both failure modes are
invisible to a rate check and to most correctness tests.

The safety net is the pair of `#[ignore]`d equivalence tests —
`equivalent_spellings_cost_the_same` in `gql/index_seed_tests.rs` and
`equivalent_gremlin_spellings_cost_the_same` in `gremlin/index_seed_tests.rs` —
which assert that groups of equivalent queries return identical rows AND run
within a factor of each other. They are what makes this refactor checkable rather
than scary; the GQL one already caught a gap inside the fix for another, and the
Gremlin one found the gap it now guards.

**Before attempting this, broaden those groups.** Seven and three today, all
predicate forms. They want path shapes, quantifier spellings, and the negations
(`NOT IN`, `<>`, `IS NOT NULL`, `not(has(…))`) whose whole point is that they
must NOT seed — a normalization that helpfully "simplifies" one of those into a
seekable form is the most likely way this goes wrong.

## The larger version, and why not yet

The full form of the question is a shared **logical plan**: Scan/Seek, Expand,
Filter, Project, Aggregate, Sort, Limit, Union, Optional. Both front ends lower
to it, one optimizer rewrites it, one executor runs it. That is where predicate
pushdown, join ordering and cardinality-driven orientation would live once
instead of twice.

It is a quarter of work, not a week, and the reasons are specific:

- **Traverser semantics do not fit an operator tree.** `path()`, `simplePath()`,
  `sack()`, `store` / `aggregate`, `repeat().until()`'s barrier-versus-streaming
  emission, `local()`, and bulk counting all assume an object with identity and
  history. GQL has bindings and columnar batches — `ScanCols`, `compact`,
  `par_project`. Bridging them means path and bulk columns in the IR. Other
  engines do this, so it is possible; it is also most of the cost.
- **Ninety-one `Step` variants**, of which perhaps thirty lower cleanly. The rest
  need an opaque "run these steps over this stream" operator — and an optimizer
  stops dead at one of those, which is precisely where it already stops today.
- **Output order is a contract in both engines, differently.** `GROUP BY`
  first-seen order is pinned; Gremlin stream order is pinned. A shared executor
  has to reproduce both, per front end.
- **Byte-identity with the TS engines.** Each language exists twice, and results
  must match byte for byte. The access-path layer is safe because it is strictly
  a performance change — same rows, fewer of them read. A shared executor that
  changes evaluation order changes tie-breaks, which means mirroring the whole
  thing in TypeScript or giving up the invariant.

The access-path layer captures essentially all of the observed value — eight of
eight gaps found so far were access-path gaps, none were plan-shape gaps — for
about a tenth of the cost. It is also the precondition: there is no point sharing
an optimizer before sharing the thing it decides.

## What the value merge changed about this note

`Val` and `GVal` are now one type (`crate::value::Value`), which this note did
not anticipate. The relevant correction is to the "traverser semantics" bullet
above: that argument is about the EXECUTION MODEL, and it still holds. It was
never an argument about the value carrier, and the carrier turned out to be
cheap to merge —

- eight of eleven variants were already identical;
- `Record` (ISO, sorted string keys) and `Map` (TinkerPop, insertion-ordered,
  any-value keys) are kept as separate variants, because merging THOSE would
  impose one language's rules on the other;
- boxing `Path` and `Property` made the union 40 bytes: Gremlin unchanged, GQL
  17% smaller than before;
- only six matches across both engines needed new arms.

What it bought was not line count. It was a place for the conversions that both
engines had been maintaining separately — and the first one moved,
`Value::index_key`, had **already drifted**: Gremlin's copy had no `Temporal`
arm, so `has('when', <date>)` scanned where the identical GQL predicate seeked.
Neither disagreement could produce a failing test, because both were "correct
but slower".

The cost, recorded because it is a real one: the merged type carries a
`PartialEq` (TinkerPop's), and GQL must not use it. That used to be enforced by
`Val` having no such impl at all; now it is a convention.

**Semantics did not merge and should not.** Ordering, null placement, equality
and rendering stay per-language — see the table under "Not on the table".

## Merging `Record` and `Map`: tried, measured, reverted

The obvious next step after unifying the value type, and the reasoning against it
was originally "semantics" — which was wrong. Both order choices are
implementation conveniences (TS reached for a JS `Map`, Rust wanted sorted keys
for dedup and columnar work), and neither order is observable in the wire format:
Gremlin **sorts map keys on output** to match `serde_json::Map`, and stringifies
every key at the boundary.

So it was built. Branch `merge-record-map`, commit `000b680`: one
`Map(Arc<[(Value, Value)]>)` variant, GQL keeping its sorted-string-key
invariant by construction, Gremlin keeping insertion order. All 1639 tests pass —
it WORKS. The blocker is cost, not correctness.

**Key width.** `(Arc<str>, Value)` is 56 bytes; `(Value, Value)` is 80. Measured
on `map_bench`:

| workload              | separate | merged  | delta    |
| --------------------- | -------- | ------- | -------- |
| construct record/row  | 5184.8   | 6993.2  | **+35%** |
| read whole stored map | 5157.8   | 7072.5  | **+37%** |
| order by map          | 12363.7  | 14390.4 | +16%     |
| map equality filter   | 16414.8  | 18945.1 | +15%     |
| nested field access   | 3480.6   | 3444.2  | −1%      |

And it buys nothing to offset that. The merge ADDED seven `as_key_str()`
downcasts to GQL and shares no algorithm: GQL binary-searches sorted string
keys, Gremlin scans insertion-ordered any-value keys. Those are different data
structures that happen to have the same shape.

### Shrinking `Value` to make it free: also tried, also reverted

`Value` is 40 bytes because `Temporal` is, and `Temporal` is 40 because of
`Duration { months: i64, days: i64, secs: i64, nanos: u32 }` = 32. Every other
temporal variant is ≤ 16. So boxing that one variant — the same trick that makes
`Path` and `Property` free — projects to:

|                             | Temporal | Value | record entry | merged entry |
| --------------------------- | -------- | ----- | ------------ | ------------ |
| today                       | 40       | 40    | 56           | 80           |
| box `Duration`              | 24       | 32    | 48           | 64           |
| + `List`/`Map` as `Arc<[]>` | 24       | 24    | 40           | **48**       |

The bottom row is the one that would make the merge free — 48 against today's 56.

Built on branch `slim-temporal`, commit `4861a79`. `Temporal` 40 → 24 and `Value`
40 → 32 exactly as projected, 1639 tests pass, and the map workloads DID improve
(construct record −7%, map equality filter −11%).

It is still a clear loss, because durations pay for it:

| duration workload | inline  | boxed   |           |
| ----------------- | ------- | ------- | --------- |
| project K         | 2.19 ms | 7.54 ms | **+244%** |
| order by K        | 23.76   | 36.30   | +53%      |
| top-k 20          | 3.03    | 5.13    | +69%      |
| filter>p count    | 1.67    | 2.68    | +60%      |
| min/max K         | 1.94    | 3.03    | +56%      |

Reproduced across runs (7.54 / 7.18 against 2.19 / 2.22). The cause is
structural rather than incidental: the temporal columns are packed SoA, so
materializing a duration builds a fresh `Temporal` per row — and boxed, that is
a heap ALLOCATION per row where it used to be a register copy. `Arc<Duration>`
would make the clone cheap but not the construction, so it does not help either.

**This is why `Temporal` is `Copy` and `Duration` is inline**, and it should stay
that way. A −20% `Value` bought ~7% on maps and cost 3.3x on durations.

Two facts underpin it, both checked: a `Temporal` is NEVER mutated (no
`&mut Temporal` exists in the crate), and there is nothing to reference — the
columns are packed SoA, so `TemporalCol::get(i)` ASSEMBLES a `Temporal` from two
to four parallel arrays. Construction, not copying, is the hot operation, which
is exactly the wrong place to add an allocation.

### The version that would work, if the semantics are acceptable

Narrowing rather than boxing keeps `Copy`, so construction stays a register copy
and the duration regression never happens:

```text
  Duration { months: i32, days: i32, nanos: i64 }   16 bytes, still Copy
  Temporal  40 -> 24        Value  40 -> 32        record entry  56 -> 48
```

It costs a SEMANTIC change and so is not something to do quietly: folding
`secs: i64` + `nanos: u32` into one `i64` of nanoseconds caps a duration's
seconds component at ±292 years rather than ±292 billion. Months (`i32`, ±178M
years) and days (`i32`, ±5.8M years) are stored separately and unaffected, so
only an enormous SECONDS count is truncated (`PT9999999999S`).

No alternative avoids it: keeping `secs: i64 + nanos: u32` leaves 4 bytes for
months and days together, forcing days down to ±89 years — worse. It also
changes the stored representation, so it needs codec work, a fuzzer pass for
byte-identity, and an overflow policy (the precedent is that date overflow
throws).

Payoff: ~7-11% on map/record workloads and a 20% smaller value in every binding
slot.

## Where the remaining gap actually is

Both engines now share the access path and the value type. What they do NOT
share is EXECUTION, and that is where the cost is. Equivalent queries, same
graph, same process, 50k vertices / 100k edges:

| equivalent work    | GQL    | Gremlin   | ratio      |
| ------------------ | ------ | --------- | ---------- |
| count all by label | 0.1 us | 1409.6 us | **11747x** |
| filter + count     | 137.9  | 2080.7    | **15x**    |
| project one column | 698.8  | 6176.9    | **9x**     |
| 1-hop count        | 374.0  | 8866.7    | **24x**    |

None of that is semantic. GQL has machinery Gremlin has no access to:

- a **count shortcut** that answers `count(*)` without materializing rows;
- a **vectorized column gather** (`gather_num` / `gather_str` /
  `gather_temporal`) instead of per-row dispatch;
- **typed comparators** (`str_eq_vec`, `temporal_cmp_vec`) that compare packed
  columns directly;
- **columnar projection** that fills a `RowSet` without a `Val` per row.

Gremlin walks `Vec<Trav>` one traverser at a time for all of it.

The shape of the fix is the same one the access path used: recognize the
COLUMNAR-ELIGIBLE PREFIX of a traversal — the leading run of element filters
before anything that needs traverser identity — run it through the columnar
machinery, and hand the result to the ordinary walk. `path()`, `simplePath()`,
`sack()` and the rest keep the traverser semantics they require, because a
prefix like `V().hasLabel(X).has(k, v)` has none.

Two things to know before starting. The label BUCKET (`by_label`) is NOT a valid
shortcut for `hasLabel`: it indexes a vertex under every label it carries, while
Gremlin's `hasLabel` matches the FIRST label only, so a bucket count matches a
`[A, B]` vertex on `hasLabel('B')` where the traversal does not. And the cheapest
win in that list was not a fast path at all — `hasLabel` was allocating an
`Arc<str>` per element to compare two integers (fixed, −17%).

## The write path is already shared

Audited because "one way of querying OR MUTATING" is the goal and mutation had
never been checked. It turns out to need nothing:

- Both engines call the same primitives — `Graph::add_vertex`, `add_edge`,
  `set_prop`, `remove_vertex`. There is one write API and always was.
- Name validation AGREES. `validate_label` / `validate_prop_key` reject the same
  inputs from either side (empty label, `::` in a label, empty property key),
  and both accept `::` INSIDE a property key. Gremlin checks at the step, GQL at
  commit via the `names_checked` watermark — two mechanisms, same verdict.
- A property set to `null` is a PRESENT null in both, per the null-first-class
  policy.

The one difference is deliberate: `DELETE` on a vertex that still has edges is
`InvalidGraphOp` in GQL — ISO wants `DETACH DELETE` — while TinkerPop's `drop()`
cascades to the incident edges. Same class as the label rules and the self-loop
rule: a language contract, carried as behaviour rather than reconciled.

## Not on the table

Unifying the two surface languages. Users pick GQL or Gremlin, and the parallel
names across them are intentional. Sharing an access path is invisible to both.

Unifying the two total ORDERS. They are deliberately different contracts, not
drift:

|       | GQL                      | Gremlin                        |
| ----- | ------------------------ | ------------------------------ |
| ranks | Num, Str, Bool, Temporal | Null, Bool, Num, Str, Temporal |
| null  | sorts LARGEST            | sorts FIRST                    |
| NaN   | Equal to every number    | last (`total_cmp`)             |

GQL's is pinned byte-for-byte to TS `compareValues`; Gremlin's to TinkerPop.
Injecting the comparator into a shared sort would share `slice::sort_by` and
nothing else — GQL's `ORDER BY` is columnar with typed fast paths
(`dense_sort_key`) that never call a comparator at all, plus `nulls_first`, which
Gremlin has no concept of. The test for whether something is worth sharing is
whether the SHARED part is the substance: for seeking it was, for ordering it is
not.

## Related

`starts_with` still scans in GQL, and that one is genuinely different: a missing
prefix-range seek is an absent feature, not an unrecognized spelling.
Normalization would not help it. (Gremlin's `P::StartsWith` does seek.)
