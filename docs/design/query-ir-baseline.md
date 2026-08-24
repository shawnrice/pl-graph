# Baseline before the shared access-path work

> **Historical.** Baseline for the `lenke-core` IR work (see
> [query-ir.md](./query-ir.md), now historical). The example binaries referenced below
> lived in the retired `lenke-core` crate and are not in the current tree.

Recorded 2026-08-01 at commit `14e223c`, before any of the work in
[`query-ir.md`](./query-ir.md) started. One machine, one binary — these numbers
are a **before/after pair**, not a portable claim. Re-run the same commands after
the change and compare against this file; comparing against a run on any other
machine proves nothing.

The point of recording it is that the change could be a net loss and there would
otherwise be no way to tell. Normalization moves work from execution to plan
time, and plan time is the cheaper place only if the plan is reused.

Everything here is a single coherent snapshot at `14e223c`. Two things landed
during the recording and everything affected was re-run afterwards:

- **`14fa700`** — a string `$param` now takes the interned-id kernel, like the
  literal always did. It was 35x, and it moved several rows below.
- **`14e223c`** — `bench:usage` batches are time-boxed rather than fixed-count,
  which took the matrix from ~40 minutes to ~3 and fixed a precision problem at
  the fast end. Rates are unchanged within noise.

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
on. Lowering is a consistent **27–36% of prepare** today.

```
shape                           parse ns  prepare ns    lower ns   lower %
--------------------------------------------------------------------------
point equality                      1375        1971         596       30%
reversed operands                   1345        1970         624       32%
inline property                     1151        1730         579       33%
IN list of 2                        1660        2487         827       33%
OR of equalities                    1816        2699         883       33%
range pair                          1864        2562         698       27%
reversed range pair                 1844        2607         763       29%
dotted path                         1424        2201         777       35%
bare label scan                       829        1296         467       36%
one-hop traversal                   1861        2578         717       28%
three-hop traversal                 2096        3175        1079       34%
var-length                          1981        2808         827       29%
group + aggregate                   3157        4128         972       24%
multi-clause                        3574        4919        1345       27%
wide AND (32 terms)                15969       22446        6477       29%
deep OR (32 terms)                 16567       23028        6461       28%
IN list of 256                     43150       66113       22963       35%
```

```
Gremlin parse (no lower step — its seek recogniser runs per EXECUTION)
point equality                       279 ns
label then has                       370 ns
within of 2                          401 ns
range                                344 ns
has then traverse                    447 ns
ten steps                           1087 ns
```

Cross-check: `gql_bench` independently reports `+2.3 us parse/lower` for a point
lookup, against `plan_bench`'s 1971 ns prepare. Two different harnesses, same
number — which is the evidence that `plan_bench` measures what it claims.

**What would count as a regression.** Ordinary shapes are 0.5–1.3 us to lower.
A normalization pass that adds a full extra tree walk should cost tens of
nanoseconds, so anything above roughly **+15% on `lower ns` for the ordinary
shapes** wants explaining. The three wide-tree rows are the ones to watch for
superlinearity: `IN list of 256` at 23 us is the current worst case, and a fold
that is accidentally quadratic in list length shows up there first and only
there.

**The Gremlin asymmetry is the real design risk.** GQL lowers once and caches the
plan; Gremlin re-runs its recogniser on every execution. Moving Gremlin's
recognition into a shared pass that costs 500 ns would add 500 ns **per run** to
a 279 ns parse — which is why the shared layer has to be reachable from a cached
plan on the Gremlin side too, or measured very carefully if it is not.

## Query shapes — `cargo run --release --example gql_bench`

52,000 vertices / 225,000 edges.

```
label scan + count               0.1 us       group by + aggregate     362.8 us
scan + filter count            145.8 us       group by 2 keys          575.4 us
projection LIMIT 100             1.7 us       exists subquery           4.05 ms
project many rows               1.03 ms       edge prop filter          1.33 ms
1-hop join count               536.7 us       project over join         1.60 ms
var-length 1..2                 1.48 ms       order by + limit         777.8 us
order by num, no limit          5.25 ms       distinct 1 col           244.7 us
distinct 2 col                 499.0 us       with filter carry        732.3 us
with agg then filter           262.8 us       with then match expand    1.56 ms
with carry then match          14.53 ms       expr-heavy filter count   1.01 ms
expr-heavy project              4.38 ms       2-hop join count          2.17 ms
```

Parameterized point lookup over 52,000 Person, prepared plan:

```
  prepared.execute : 91.6 us      (was 2673.9 us before 14fa700 — 29x)
  parse+execute    : 93.8 us      (+2.3 us parse/lower)
```

Index seeding, same run:

```
  eq inline {name}   scan    1.28 ms   index     0.5 us   (2770x)
  where name =       scan    86.8 us   index     0.5 us   (159x)
  where age > 78     scan   122.9 us   index     7.2 us   (17x)
  where age 30..40   scan   331.0 us   index   102.9 us   (3x)
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

20,000 users, 250 ms batches, best of 3, all three engines, without and with
indexes. This is the run that found the 60x planner cliff, so it is the one that
would notice a seed choice getting worse. Operations per second.

```
workload                               ts(-)     ts(+)    ffi(-)    ffi(+)   wasm(-)   wasm(+)
----------------------------------------------------------------------------------------------
read: point lookup                       134     91.7k     23.8k    177.4k     12.4k    165.1k
read: permission check                     8     47.0k       303    110.8k       217    104.7k
read: 2-hop recommendation                20     65.0k      1.1k    135.6k       866    111.0k
read: keyed dedup lookup                 149      7.5k       600     39.5k       383     27.5k
write: property update                   156    108.0k      1.1k    187.0k       641    162.3k
write: append node                    118.2k     79.9k    236.2k    220.6k    232.8k    208.3k
interleaved: write + point read           67     51.5k      1.0k     88.2k       595     82.9k
interleaved: write + traversal            29     45.1k       684     77.6k       433     71.3k
read: permission check (inline anchor)     9     46.4k       154    115.4k       113    111.5k
write: 100 updates in one transaction    149    109.2k      1.1k    193.3k       638    169.4k
write: 100 updates in one statement      8.3k      8.7k      4.8k    580.3k      4.7k    505.6k
analytic: fan-out spread 1-3 hops        173     42.9k     23.0k    117.1k     12.2k    106.2k
analytic: cycle detection 2-4 hops         6     45.8k       164    116.2k       118    102.8k
interleaved: append + link               169     59.8k      1.0k    126.5k       642    110.8k
```

What the `$param` fix (`14fa700`) did to the unindexed native columns:

| workload                   | ffi(-) before | ffi(-) after | wasm(-) before | wasm(-) after |
| -------------------------- | ------------- | ------------ | -------------- | ------------- |
| read: point lookup         | 939           | **23.8k**    | 809            | **12.4k**     |
| analytic: fan-out 1-3 hops | 903           | **23.0k**    | 811            | **12.2k**     |
| read: 2-hop recommendation | 507           | 1.1k         | 399            | 866           |
| interleaved: write + read  | 495           | 1.0k         | 386            | 595           |

`ts(-)` did not move, and that is the control: the pure-TS engine has no
vectorized interned-id kernel to fall off, so it never had the defect. Measured
rather than assumed — 20k scan, literal 6181 us vs param 4790 us, no asymmetry.
On the point-lookup shape native went from 4.8x SLOWER than pure-TS to 178x
faster.

**Reading the (-) columns.** They are the no-index configuration, which is where
a planner regression shows up first — an index hides a bad access path. The (+)
columns are what a tuned deployment sees. Both are load-bearing.

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

Total, from cold: about six minutes.
