# Baseline before the shared access-path work

Recorded 2026-08-01, at commit `3672add`, before any of the work in
[`query-ir.md`](./query-ir.md) started. One machine, one binary — these numbers
are a **before/after pair**, not a portable claim. Re-run the same commands after
the change and compare against this file; comparing against a run on any other
machine proves nothing.

The point of recording it is that the change could be a net loss and there would
otherwise be no way to tell. Normalization moves work from execution to plan
time, and plan time is the cheaper place only if the plan is reused.

## What this change can move, and what it cannot

| axis                             | at risk? | measured by                                      |
| -------------------------------- | -------- | ------------------------------------------------ |
| lex + parse                      | no       | `plan_bench`, `parse ns` column                  |
| **lower (where the pass lands)** | **yes**  | **`plan_bench`, `lower ns` column**              |
| which seek a query picks         | yes      | `bench:usage` (- / +), the two equivalence tests |
| per-shape query cost             | yes      | `gql_bench`, `gremlin_bench`                     |
| index seek vs scan ratios        | yes      | `edge_type_index_bench`, `gremlin_index_bench`   |
| ingest, codecs, algorithms       | no       | not re-run — nothing in this change touches them |

The last row matters as much as the others: re-running benchmarks a change
cannot affect produces noise that later reads as signal.

## Plan time — `cargo run --release --example plan_bench`

The one axis with no prior coverage, and the one a rewrite pass lands directly
on. Lowering is a consistent **27–34% of prepare** today.

```
shape                           parse ns  prepare ns    lower ns   lower %
--------------------------------------------------------------------------
point equality                      1452        2002         549       27%
reversed operands                   1376        2002         626       31%
inline property                     1220        1744         523       30%
IN list of 2                        1755        2524         770       30%
OR of equalities                    1917        2744         826       30%
range pair                          1929        2595         665       26%
reversed range pair                 1892        2647         755       29%
dotted path                         1480        2252         772       34%
bare label scan                      862        1313         451       34%
one-hop traversal                   1884        2596         713       27%
three-hop traversal                 2125        3302        1177       36%
var-length                          2076        2904         828       29%
group + aggregate                   3256        4171         914       22%
multi-clause                        3672        5004        1333       27%
wide AND (32 terms)                16460       22999        6539       28%
deep OR (32 terms)                 17244       23822        6578       28%
IN list of 256                     46253       67744       21491       32%
```

```
Gremlin parse (no lower step — its seek recogniser runs per EXECUTION)
point equality                       280 ns
label then has                       374 ns
within of 2                          411 ns
range                                345 ns
has then traverse                    448 ns
ten steps                           1074 ns
```

**What would count as a regression.** Ordinary shapes are 0.5–1.3 us to lower.
A normalization pass that adds a full extra tree walk should cost tens of
nanoseconds, so anything above roughly **+15% on `lower ns` for the ordinary
shapes** wants explaining. The three wide-tree rows are the ones to watch for
superlinearity: `IN list of 256` at 21.5 us is the current worst case, and a
fold that is accidentally quadratic in list length shows up there first and only
there.

**The Gremlin asymmetry is the real design risk.** GQL lowers once and caches the
plan; Gremlin re-runs its recogniser on every execution. Moving Gremlin's
recognition into a shared pass that costs 500 ns would add 500 ns **per run** to
a 280 ns parse — which is why the shared layer has to be reachable from a cached
plan on the Gremlin side too, or measured very carefully if it is not.

## Query shapes — `cargo run --release --example gql_bench`

52,000 vertices / 225,000 edges.

```
label scan + count               0.1 us       group by + aggregate     334.9 us
scan + filter count            135.7 us       group by 2 keys          529.1 us
projection LIMIT 100             1.6 us       exists subquery           3.81 ms
project many rows              939.3 us       edge prop filter          1.31 ms
1-hop join count               496.0 us       project over join         1.60 ms
var-length 1..2                 1.48 ms       order by + limit         782.1 us
order by num, no limit          5.31 ms       distinct 1 col           243.7 us
distinct 2 col                 498.3 us       with filter carry        719.7 us
with agg then filter           262.5 us       with then match expand    1.54 ms
with carry then match          14.93 ms       expr-heavy filter count  969.6 us
expr-heavy project              4.35 ms       2-hop join count          2.19 ms
```

Index seeding, same run:

```
  eq inline {name}   scan    1.29 ms   index     0.5 us   (2787x)
  where name =       scan    86.7 us   index     0.5 us   (167x)
  where age > 78     scan   122.0 us   index     7.2 us   (17x)
  where age 30..40   scan   311.3 us   index    99.7 us   (3x)
```

## Gremlin traversals — `cargo run --release --example gremlin_bench`

```
  V().hasLabel(P).count                1.45 ms   V(P).out(KNOWS).count           13.56 ms
  V().has(age>50).count                2.13 ms   V(P).out.out(KNOWS).count      156.39 ms
  V().hasLabel.values(name)            6.24 ms   V(P).out(KNOWS).values(name)    72.05 ms
  V().has(age>50).values(name)         4.25 ms   V(P).out(KNOWS).dedup.count     34.88 ms
  V(P).both(KNOWS).count              43.40 ms   V(P).out(CREATED).hasLabel(SW)   4.61 ms
```

## Index seeks — the two index benchmarks

`edge_type_index_bench`, 100,000 vertices / 400,100 edges:

```
  :RARE   scan   13.10 ms   seek   0.1 us   (98929x)   bucket 100/400100
  :KNOWS  scan   13.35 ms   seek   0.1 us   (104880x)  bucket 400000/400100
```

`gremlin_index_bench`, 100,000 vertices:

```
                        scan        seek
  eq point lookup     4.47 ms      0.3 us
  within (3 values)   4.61 ms      0.6 us
  range age > 75      3.97 ms    128.7 us
  between age [30,40) 4.04 ms    319.0 us
  startsWith          5.75 ms      3.4 us
```

## Equivalence ratios — the two `#[ignore]`d tests

```
cargo test --release equivalent_spellings_cost_the_same         -- --ignored --nocapture
cargo test --release equivalent_gremlin_spellings_cost_the_same -- --ignored --nocapture
```

Every group is within **1.4x** on the GQL side and **1.2x** on the Gremlin side,
against a `MAX_RATIO` of 12. That headroom is the safety margin the refactor
spends: if a normalization pass fails to canonicalize some spelling it used to
recognize by hand, this is where it shows up, and it has 8x of room before the
assertion fires. **Tighten `MAX_RATIO` only after the change**, not before —
lowering it now would be measuring the machine's noise, not the planner.

## Usage-shaped workloads — `cd packages/native && bun run bench:usage`

20,000 users, 5,000 ops per batch, best of 3, all three engines, without and with
indexes. This is the run that found the 60x planner cliff, so it is the one that
would notice a seed choice getting worse.

It takes upwards of an hour at the default settings, because the unindexed cells
are full scans by construction (~100 s each). Run it, do something else, and
paste the table here.

_Pending — see the note above._

## Reproducing

```
cargo run --release --example plan_bench
cargo run --release --example gql_bench
cargo run --release --example gremlin_bench
cargo run --release --example edge_type_index_bench
cargo run --release --example gremlin_index_bench
cargo test  --release equivalent_spellings_cost_the_same         -- --ignored --nocapture
cargo test  --release equivalent_gremlin_spellings_cost_the_same -- --ignored --nocapture
cd packages/native && bun run bench:usage
```
