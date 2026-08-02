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

## Not on the table

Unifying the two surface languages. Users pick GQL or Gremlin, and the parallel
names across them are intentional. Sharing an access path is invisible to both.

Unifying the two total ORDERS. They are deliberately different contracts, not
drift:

| | GQL | Gremlin |
| ------ | ---------------------------- | ------------------------------- |
| ranks | Num, Str, Bool, Temporal | Null, Bool, Num, Str, Temporal |
| null | sorts LARGEST | sorts FIRST |
| NaN | Equal to every number | last (`total_cmp`) |

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
