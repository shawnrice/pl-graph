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

## The join is the next frontier, and it has a real bug

`multi-pattern` is the largest remaining bail, and probing it turned up the
biggest cost asymmetry found in this whole effort.

GQL's comma-pattern join (`visit_patterns`, `gql/eval/pathfind.rs`) recurses
through patterns in **written order**. Gremlin's `match()` (`match_solve`) calls
`pick_runnable` — it picks a pattern whose start is already bound. So Gremlin has
an optimization GQL does not, and the two spellings of one query cost wildly
different amounts:

```text
MATCH (x:S)-[:R]->(b), (b)-[:R]->(c) WHERE x.k = 'target'   -- anchor FIRST
MATCH (b)-[:R]->(c), (x:S)-[:R]->(b) WHERE x.k = 'target'   -- anchor LAST

  A-nodes   anchor FIRST   anchor LAST      ratio
    3,000       0.000 ms      0.618 ms     1,278x
   30,000       0.001 ms      6.203 ms    12,356x
  300,000       0.001 ms     62.286 ms   121,336x
```

Both return the same single row. `anchor FIRST` is CONSTANT — it seeks the
selective anchor and expands from it. `anchor LAST` is LINEAR in the graph: the
unanchored pattern is enumerated as the outer loop and the anchor only prunes
afterwards. The ratio is unbounded, growing with the graph.

This is the `equivalent_spellings_cost_the_same` class from CLAUDE.md — every
instance of which returned the correct answer, so no correctness test could see
it — but one level up, at the join rather than the predicate, and far larger than
the 100-300x those cost.

### A chained comma join IS one path

`MATCH (a)-[r]->(b), (b)-[s]->(c)` and `MATCH (a)-[r]->(b)-[s]->(c)` are the same
query. The second vectorizes; the first stayed on the scalar join purely because
it arrived as two `CPath`s. `fuse_chain` splices them when each pattern after the
first starts on the node the previous one ended on and adds nothing to it (no
label, no inline props, no `WHERE` — a bare back-reference).

```text
  20k vertices, degree 3      before      after   one-path equivalent
  (a)-[]->(b), (b)-[]->(c)   12.593 ms   0.094 ms       0.095 ms   134x
  … , (c)-[]->(d)            40.615      0.117          0.118      347x
```

The comma spellings now cost what their chain equivalents cost, to the noise
floor. Two things make it sound, both checked rather than assumed: the trail
restriction (no repeated edge) applies to a QUANTIFIED walk, not a fixed-length
path — on a self-loop both spellings return the same row with `r = s` — and the
row ORDER is unchanged, because the scalar join nests pattern 2 inside pattern 1,
which is exactly the order one fused path enumerates.

A constrained shared node fuses too — `(a)-[]->(b:N), (b:M {k: 1})-[]->(c)` names
ONE variable, so the node must satisfy both sides and `merge_node` conjoins them
(`CLabelExpr::And`, concatenated props, `CExpr::And` on the node `WHERE`).

**What does NOT fuse, and why it barely matters.** Instrumenting the declines:

```text
  2188  disconnected cartesian   the patterns share no variable at all
    12  diverging / mid-join     shares a variable, but not at an END
```

Diverging — `(b)-[]->(a), (b)-[]->(c)` — would splice to `(a)<-[]-(b)-[]->(c)`,
which is the right rows, but a linear path can only be enumerated from an END, so
the fused form drives from `a` while the join drives from `b` and the rows come
out regrouped (same multiset, different order). That was attempted and backed
out; at 12 occurrences it buys nothing to accept an order change for. The
remaining 2188 are cartesian products, which have no shared variable and so are
not a fusion problem at all — cross-producting two columnar frames is a different
operation, and one worth doing separately if it ever matters.

Pinned in `equivalent_spellings_cost_the_same`, and guarded by two differential
tests that run each shape with fusion on and off and compare row-by-row. The
second exists because the first could not discriminate: its fixture gives every
node the same single label and has no inline node `WHERE`, so deleting the label
conjunction or the `where_` merge left it green. Both mutations now fail two
tests.

### Picking the pattern to drive

The fix is the one Gremlin already has: choose the next pattern by whether its
start is bound, instead of by where the user typed it. Doing that in the shared
IR is the point of the exercise — it is one optimization that both languages
would then be on the correct side of.

**Fixed**, in both engines (`pattern_rank`/`pick_pattern` in
`gql/eval/pathfind.rs`, `patternRank`/`pickPattern` in
`gql/src/executor/clauses.ts`). Each pattern is ranked by what it costs to START:

```text
  0   a variable is already bound   — continues a binding, no fresh scan
  1   start node has a label or inline props   — a restricted scan
  2   otherwise                     — every vertex
```

Ties keep the WRITTEN order, which is what made this safe: the reorder only fires
where a later pattern is strictly cheaper to start, and not one of the 1680 Rust
or 481 TS tests changed its rows. All four byte-identity fuzzers pass. The row
order of a multi-pattern `MATCH` without `ORDER BY` is unspecified anyway, but in
practice nothing moved.

After, native: both spellings 0.001 ms at every size — 121,336x becomes 1.01x.
And the pure-TS engine, which had the same bug and where optimization wins
usually do NOT reach:

```text
  A-nodes   anchor FIRST   anchor LAST (before)   after
    3,000       0.045 ms              7.387 ms   0.060 ms
   30,000       0.021 ms             76.893 ms   0.039 ms
  100,000       0.019 ms            270.223 ms   0.035 ms
```

The two rank functions must agree or the engines emit rows in different orders,
so they are deliberate mirrors and each says so at its definition.

## The experiment, concluded

The question was: can GQL and Gremlin compile to one IR, so an optimization is
written once and both get it, and the two per-language execution layers can be
deleted? Here is what actually happened.

### It works, and it found bugs neither engine's tests could

Sharing the seek and the expansion is real. `crate::seek` (1437 lines) and
`crate::value` (234) are the shared layer, and both front ends lower into them.
The payoff was not mainly speed — it was that putting the two engines side by
side made one engine's missing optimization VISIBLE:

- Gremlin's `match()` had always picked a pattern whose start was bound;
  both GQL engines joined comma patterns in written order. **121,336x** apart at
  300k vertices, and 270 ms vs 0.019 ms in pure-TS. Fixed in both.
- Gremlin's `Trav::tags` and GQL's group variables are the SAME per-repetition
  list (`select(Pop.all,'x')` is `((x)-[e]->(y)){1,4}`), and both deep-copied it.
  One `Arc` fixed both.
- `Value::index_key` had drifted: Gremlin's copy had no `Temporal` arm, so
  `has('when', DATE '…')` could not seek a temporal index while the same GQL
  predicate could. Merging the type fixed it.

None of these were findable from inside one engine. That is the strongest
argument the experiment produced.

### The speed result

GQL, `gql_bench`, interleaved against the pre-IR baseline to cancel drift:

```text
  var-length 1..2               1500.0us     37.1us   40.4x
  with carry then match        14550.0us   5070.0us    2.9x
  edge prop filter              1310.0us    707.4us    1.85x
  project over join             1620.0us    997.9us    1.62x
  with then match expand        1560.0us   1140.0us    1.37x
  scan + filter count            137.8us    107.3us    1.28x
  1-hop join count               497.7us    568.0us    0.88x  ← slower
  [b] scan+count+pred            149.8us    183.5us    0.82x  ← slower
```

Plus, off this benchmark: comma-pattern fusion 134x/347x, adjacent `MATCH`
clauses 37x/169x, disconnected cross products 38,000x, `ORDER BY` an output alias
2.2x (5.6x with a LIMIT), a carried numeric column 2.4-5.8x. Gremlin: 16 lowered
terminals at 3.6-34x, two-hop count 71x, `as()`-tag carry up to 1.5x.

The two regressions are the honest cost. Both are the shared path being slower
than the bespoke one it replaced on a shape where the bespoke one had nothing to
do. `[b] scan+count+pred` has a diagnosis: the clause `WHERE` is lowered into the
seek's conjuncts AND applied again as a mask afterwards. `ElementSeek::
answers_exactly` exists to report "the seek already settled this predicate" and
is **never called** — wiring it through `build_scan` is the fix.

### The surface result: it did NOT shrink

This is the goal that was not met.

```text
  non-test Rust in lenke-core/src:  53267 -> 57122   (+3855, +7.2%)
```

The shared layer was ADDED (+1671) and the per-language layers were only partly
deleted. One duplicate expander did go — `expand_frame`, a third copy of the
fan-out loop reached only through `vectorized_linear`, is now a seed plus a call
into `expand_scan` (−119 lines). But `gremlin/exec.rs` grew +1074 for the
lowering that lets Gremlin reach the shared path at all, and `scan.rs` +539.

The reason is structural, and worth stating plainly: **the two engines have
different execution models.** GQL's is columnar — a frame of columns, filtered
and projected in bulk. Gremlin's is a traverser stream, where each traverser
carries its own path, tags and sack. The shared IR covers what they genuinely
have in common — WHERE to start (`ElementSeek`) and HOW to fan out
(`Frontier`) — but a Gremlin step that reads `t.path` cannot be a column
operation without changing what TinkerPop promises. Measured: of ~1100 Gremlin
traversals in the suite, 33% lower fully onto the shared columnar path, 24% seed
through the shared IR then stream, and 43% are pure stream.

So the layers cannot both be deleted. What CAN be deleted — and was — is
duplicate implementations of the same operation. What cannot is one engine's
execution model.

### Where GQL now runs

Instrumenting row production across the GQL suite, baseline vs now:

```text
             columnar     scalar    columnar share
  baseline      28082       9826       74.1%
  now           30870       7172       81.1%
```

Scalar row production fell 27%. The remaining scalar work is: `agg-no-where`
(deliberate — the scalar driver stream-folds without materializing), a `MATCH`
after a `WITH` (handled by `vectorized_linear`, a different columnar entry), path
variables (the columnar frame cannot build a `Path` value), and
`multiseg-limit-dfs` (deliberate — DFS stops at the LIMIT, BFS cannot).

### Audit: which "contracts" were real

The conclusion above first claimed the two engines cannot share more because
their EXECUTION MODELS differ. Re-auditing each claimed contract against the
specs rather than against the existing code, that was too strong — one of them
was not a contract at all, and another was self-imposed.

| claimed                                           | verdict                                                                                                                                                                                                                                              |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `out('a','b')` emits in label-argument order      | **NOT a contract.** TinkerPop specifies no order; TinkerGraph groups only because it stores `Map<label, Set<Edge>>`. This repo already records adjacency order as unspecified. Deleted, −172 lines, six steps now on `crate::seek::adj`.             |
| 63% of traversals need per-traverser path history | **Self-imposed.** Only FIVE steps read `Trav::path`. `path_free` was an allowlist, so everything unlisted paid by default, and a looping shape could never be listed at all. Now derived + recursive: 63% → 34%, `repeat(out())` 2.3x, `union` 2.5x. |
| `hasLabel` matches only the FIRST label           | **Real, and consistent.** TinkerPop's model is one label per vertex, `addV` takes one, and `label()` returns the first — so `hasLabel` matching the first agrees with the rest of the surface. Multi-label vertices only arise from GQL or import.   |
| The two total orders                              | **Real.** GQL's is pinned to TS `compareValues` (null largest, NaN equal to every number), Gremlin's to TinkerPop (null first, NaN last).                                                                                                            |
| `DELETE` vs cascading `drop()`                    | **Real.** ISO requires `DETACH DELETE` to remove a connected vertex; TinkerPop's `drop()` cascades.                                                                                                                                                  |
| `Record` (sorted) vs `Map` (insertion-ordered)    | **Real** as data models, though map KEY order is separately documented as unspecified. Merging them measured +35% and was rejected earlier.                                                                                                          |

So the honest revision: the traverser stream is genuinely different **where a step
reads path, tags or sack** — and that is now 34% of traversals, not 63%. The
adjacency walk underneath was never model-specific, just duplicated. The lesson
is that "this is how the engine works" and "this is what the language requires"
had been conflated, and only reading the spec separates them.

**Kept though neutral**: `present_key_ids` keeps the column id `present_keys`
derives and discards, so `valueMap()` stops hashing each name back to the id it
just had — twice per key per element. Measured 79.6 ms → 79.2-82.6 ms over 50k ×
12 properties: no change. Kept anyway, because deriving a value, throwing it away
and re-deriving it twice is worth removing on its own. What the timing DOES say is
that `valueMap`'s cost is building a `GVal::Map` — an `Arc` key and a boxed value
per entry, 600k of them — so that is the thing to optimize next.

### If this is kept

The branch is worth keeping for the speed and the cross-engine bug class it
exposes, NOT for the surface reduction it was meant to deliver. Anyone continuing
should know the surface only starts shrinking when a whole per-language path can
be deleted, and that requires the two execution models to converge — which is a
much larger question than sharing an IR.

## The next one: a carrying WITH costs 4.8x

The remaining fallbacks after the join work are `incoming-bindings` — a `MATCH`
that follows a `WITH` or `INSERT`. The vectorized frame refuses outright:

```rust
if incoming.len() != 1 || incoming[0].0.iter().any(|c| c.is_some()) {
    return None; // a prior WITH/INSERT already produced bindings
}
```

Most of that shape is already fine, because the `WITH`'s own `MATCH` vectorizes
and what follows is a small expansion. One is not:

```text
  50k vertices, degree 3                                       best
  MATCH (a:V)-[:R]->(b) WHERE b.n > a.n RETURN count(*)       2.373 ms
  MATCH (a:V) WITH a, a.n AS m
    MATCH (a)-[:R]->(b) WHERE b.n > m RETURN count(*)        11.297 ms   4.8x
```

Same answer. Carrying a value through a `WITH` — instead of reading it from the
element in place — costs 4.8x, because everything after the `WITH` runs scalar.

**Two ways to fix it, and why the obvious one is the wrong one.**

_Inline the pass-through `WITH`_ at the AST level: substitute `m` → `a.n` in the
following clauses and turn its `WHERE` into a `FILTER`. Tempting, because
`decorrelate_clauses` already rewrites clauses by NAME before lowering, which is
far easier than slot surgery. But it needs a total substituter over the `Expr`
tree, and a variant it forgets leaves `m` unbound — which reads as `null` and
returns a wrong answer silently. That is the same failure mode that made
`Program::read_slots` report "assume everything" on an opaque `Op::Tree`, and it
is worth the same caution.

_Seed the frame from the incoming bindings_ instead. No expression rewriting at
all: build the `Frontier` from the incoming rows' value for the pattern's start
slot, carry their other bound slots as columns (`Frontier` gained value columns
for the group-variable work, so this now has somewhere to put a non-element
binding), and expand as usual. It needs a `set_column` beside `set_values` and a
new entry beside `build_scan`. When it does not apply it fails to BUILD, which is
loud, rather than producing a row with a null in it.

The second is the one to write.

## How much is left, measured

Every point where GQL declines the vectorized frame was labelled and the GQL
suite run once. ~9300 bails, by reason:

```text
  2683  multi-pattern            MATCH (a)-[]->(b), (c)-[]->(d)  — a JOIN
  2507  order+distinct/alias     ORDER BY an output alias; DISTINCT + ORDER BY
  2489  agg-no-where             deliberate: scalar stream-folds, no materialize
  1375  multi-clause-or-star     several MATCH clauses, or RETURN *
   126  incoming-bindings        a prior WITH/INSERT already produced bindings
    88  build_scan itself        ← the access path
    51  path-variable            needs the Path value the columnar frame can't build
     1  multiseg-limit-dfs       deliberate: DFS stops at the LIMIT, BFS can't
```

**The access path is done.** It accounts for 88 of ~9300 — under 1%. Everything
else is a layer the IR never claimed: clause composition, joins, and projection.
Two of the rows (`agg-no-where`, `multiseg-limit-dfs`) are not gaps at all —
they are routing decisions where the scalar driver is genuinely faster, and each
has its reasoning recorded at the branch.

So "how much of the scan is left to port" has the answer _almost none_, and the
next frontier is a different question: `multi-pattern` is a join, and Gremlin's
`match()` step is the same join. That is where a shared IR would pay next.

This is also why the code surface is currently +4.9% (53267 → 55866 non-test
lines in `lenke-core/src`): the shared layer was added, and the per-language
layers above it cannot be deleted until the join/projection layer is shared too.
The surface shrinks at the END of that work, not this one.

## Group variables: the port that was wrong to decline

Group variables in a parenthesized quantified unit — `((x)-[e]->(y)){1,4}` —
were the last shape `build_scan` declined, and the first attempt at them measured
1.25-2.65x SLOWER than the scalar matcher. That was nearly written off as
"GQL-only plumbing Gremlin never touches". It is not:

```text
  GQL      MATCH (a)((x)-[e]->(y)){1,4}(b)          x, e, y are per-rep LISTS
  Gremlin  repeat(__.as('x').out()).times(4)
             .select(Pop.all, 'x')                  the SAME per-rep list
```

`Trav::tags` is `Vec<(String, Vec<GVal>)>` and `Pop::All` hands back
`GVal::List(list.clone())`. It is the same operation, written twice — GQL-only
because of who wrote it, not because the logic differs. And the SLOWNESS was
shared too: both engines deep-copied those lists on their hot paths, GQL in
`scalar_col`'s per-row `Binding` rebuild and Gremlin in `Trav::step`'s per-step
`tags` clone.

So the fix was not to decline the port. It was three shared changes underneath it:

1. **`Value::List` is `Arc<[Self]>`**, like `Value::Record` already was. Cloning
   a list is a refcount bump in BOTH engines now.
2. **`Frontier` carries value columns**, not just id columns. A group variable is
   one list per row and has to fan out with the ids — without that, a later hop
   leaves the column short and every row past the first reads off the end.
3. **A `size()` kernel over a value column**, so the shape every group variable
   is actually read through never rebuilds a row binding at all.

4. **Group columns nothing reads are not built.** `Program::read_slots` collects
   the input slots the projection, GROUP BY / ORDER BY keys, lifted aggregates and
   clause WHERE actually read, and `build_scan` skips the rest. It returns
   `false` — meaning "assume everything" — the moment it meets an opaque
   `Op::Tree`, because this only ever SKIPS work and a missed slot would read
   back as `null` rather than failing.

   The trap: `CProjection::order_overlay` is EVERY input slot, populated whether
   or not the query sorts (it exists so a sort key can name an input the
   projection dropped). Folding it in unconditionally made the needed-set
   universal and the elision a silent no-op — it measured as noise, which is
   exactly how it would have shipped.

Measured (`layered_dense(6,6)`, forced both ways, min of 7):

```text
                                columnar   scalar   ratio   first attempt
  -[:R]->{1,4} (no group vars)    2171 us  2433 us   0.89x       0.92x
  ((x)-[]->(y)){1,4} rows         2310     4532      0.51x       1.28x
  ((x)-[e]->(y)){1,4} + size      4210     4347      0.97x       1.95x
  … WHERE size(e) >= 2            5122     5142      1.00x       2.65x
```

Every shape is now at or better than the scalar matcher, where every shape that
exposed a group variable had been 1.3-2.7x worse. The one that gains most is the
one that binds group variables and reads none of them — which is the common case,
because `((x)-[]->(y)){1,4}` is often written for the WALK, not for `x` and `y`.

The lesson for the rest of this effort: "this is one language's problem" was
wrong here, and the thing that made it wrong was reading the other engine's code
rather than reasoning about the feature. Same as the join-order bug below.

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
