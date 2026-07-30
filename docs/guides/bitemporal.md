# Bitemporal modeling on lenke

**A recipe, not a feature.** lenke ships the _primitives_ you need to model bitemporality — `DATE` properties, parameterized `WHERE` predicates, a host clock, and atomic transactions — but it does **not** ship a bitemporal engine. There is no `PERIOD` type, no system-time column, no `AS OF` keyword. You compose bitemporality from the pieces below, and you keep full control over the schema. This guide shows the working pattern end to end; every query here is executable (verified against `@lenke/gql` on the pure-TS engine, and byte-identical on the Rust core).

> The [`.dogfood`](../../.dogfood/round9/sofia/bitemporal.ts) scratch script is the reference implementation this guide is distilled from — a cold reader built the whole slice from public docs and exported types alone.

## The model: four DATE columns on the fact

A _bitemporal_ fact is true along two independent time axes:

- **Valid time** — when the fact is true in the modeled world. Two columns: `validFrom`, `validTo`.
- **Transaction time** — when the _system_ believed the fact. Two columns: `txFrom`, `txTo`.

Put all four on whichever element carries the fact. Relationships are the natural home (an employment, a price, a role assignment is an _edge_), but nodes work identically.

```ts
import { Graph, LocalDateTime } from '@lenke/core';
import { query } from '@lenke/gql';

const g = new Graph();
g.createUniqueConstraint('Person', 'id');
g.createUniqueConstraint('Company', 'id');

query(g, `INSERT (:Person {id: 'alice', name: 'Alice'})`);
query(g, `INSERT (:Company {id: 'acme', name: 'Acme Corp'})`);

// v1 belief (recorded 2020-01-05): Alice employed by Acme from 2020-01-01,
// believed permanent (both "to" columns open-ended).
query(
  g,
  `MATCH (p:Person {id: 'alice'}), (c:Company {id: 'acme'})
   INSERT (p)-[:EMPLOYED_BY {
     validFrom: DATE '2020-01-01', validTo: DATE '9999-12-31',
     txFrom:    DATE '2020-01-05', txTo:    DATE '9999-12-31'
   }]->(c)`,
);
```

### Half-open intervals `[from, to)`

Every interval is **half-open**: `from` is inclusive, `to` is exclusive. This is the one convention that makes the predicates clean — adjacent versions abut without overlap (`validTo` of one equals `validFrom` of the next), and a point-in-time test is always `from <= t AND t < to`, never a fencepost puzzle.

### The open-end sentinel

An interval that is still open has no natural "to" value. Use a fixed far-future sentinel — **`DATE '9999-12-31'`** — rather than `null`. Two reasons:

1. The `t < to` half of every predicate stays a plain comparison; you never special-case `IS NULL`.
2. `null` is a _stored, present value_ on lenke (not absence — see the null policy), so it wouldn't shortcut the comparison anyway. A sentinel is both correct and faster to reason about.

Define it once as a constant and interpolate it into the _query text_ (it's a literal, not a value binding):

```ts
const INF = "DATE '9999-12-31'";
```

## As-of queries: a parameterized WHERE clause

An "as-of" query asks: _at transaction time `$tx`, believing what we believed then, what was true at valid time `$valid`?_ That is one four-term predicate over your chosen period columns — no special syntax:

```ts
function asOf(txDate: string, validAt: string) {
  return query(
    g,
    `MATCH (p:Person {id: 'alice'})-[e:EMPLOYED_BY]->(c:Company)
     WHERE e.validFrom <= date($valid) AND date($valid) < e.validTo
       AND e.txFrom    <= date($tx)    AND date($tx)    < e.txTo
     RETURN c.name AS employer,
            e.validFrom AS vFrom, e.validTo AS vTo,
            e.txFrom AS tFrom, e.txTo AS tTo`,
    { tx: txDate, valid: validAt },
  );
}
```

`date($valid)` and `date($tx)` cast a string param to a `DATE` engine-side, so the comparison is date-vs-date. **Send the instants as params — never build the query text from user input.** The same clause drives every temporal question: freeze `$tx` at today and sweep `$valid` to get a valid-time timeline; freeze `$valid` and sweep `$tx` to get the belief history.

## "Current belief" via the host clock

For _right now_ on both axes, you don't want a hardcoded date — you want the wall clock. Install a **host clock** and query with `current_date`:

```ts
g.setClock(() => LocalDateTime.fromJSDate(new Date(), { zone: 'utc' }));

const currentBeliefs = query(
  g,
  `MATCH (p:Person)-[e:EMPLOYED_BY]->(c:Company)
   WHERE current_date < e.txTo                                  -- belief still open
     AND e.validFrom <= current_date AND current_date < e.validTo  -- true today
   RETURN p.name AS person, c.name AS company
   ORDER BY person`,
);
```

`current_date` reads the clock you installed, so it's deterministic and testable — pin the clock to a fixed instant in tests, let it track `new Date()` in production. `current_date < e.txTo` selects only the _currently-believed_ version of each fact (the row whose transaction interval is still open); the two valid-time terms then keep only what is true today.

## A correction is an atomic event

The bitemporal payoff is that you _never mutate history_ — you record a new belief and close the old one, atomically. Suppose on 2021-06-10 we learn Alice actually left Acme on 2021-05-31. Two writes, one transaction:

1. **Close the old belief**: set the currently-open version's `txTo` to the correction date (we stopped believing "permanent" on 2021-06-10).
2. **Insert the corrected version**: same valid-time start, a real valid-time end, and a fresh transaction interval that is now open.

```ts
g.transaction(() => {
  query(
    g,
    `MATCH (p:Person {id: 'alice'})-[e:EMPLOYED_BY]->(c:Company {id: 'acme'})
     WHERE e.txTo = ${INF}
     SET e.txTo = DATE '2021-06-10'`,
  );
  query(
    g,
    `MATCH (p:Person {id: 'alice'}), (c:Company {id: 'acme'})
     INSERT (p)-[:EMPLOYED_BY {
       validFrom: DATE '2020-01-01', validTo: DATE '2021-05-31',
       txFrom:    DATE '2021-06-10', txTo:    ${INF}
     }]->(c)`,
  );
});
```

`graph.transaction(fn)` makes the pair atomic: either both writes land or neither does (undo-log rollback on throw), and downstream events fire once at commit. After this:

- An as-of query at `tx = 2021-01-01` still returns the _old_ belief (she was believed permanently employed) — history is intact.
- An as-of query at `tx = today, valid = 2021-06-15` returns **no row** — the corrected belief says she had left by then.
- The full transaction-time history is queryable by simply dropping the `txTo` filter, giving you a superseded-belief audit trail for free.

## The version-node variant (and its supersession trap)

The model above puts the period columns **on the fact itself** — the `EMPLOYED_BY` edge _is_ the versioned thing. That is the cleanest shape when the fact is a relationship. But a lot of apps version a whole **entity snapshot** instead: a stable identity node plus one _version node_ per belief, each carrying the four period columns and a copy of the mutable fields. This is the `ProfileVersion` shape teams reach for first — and it's where the correction logic is easiest to get subtly wrong, so it's worth spelling out.

```ts
const INF = "DATE '9999-12-31'";

g.createUniqueConstraint('Person', 'id');
query(g, `INSERT (:Person {id: 'alice'})`); // stable identity — never versioned

// v1 belief (recorded 2020-01-05): Alice = Engineer from 2020-01-01, believed permanent.
query(
  g,
  `INSERT (:PersonVersion {vid: 1, title: 'Engineer',
  validFrom: DATE '2020-01-01', validTo: ${INF},
  txFrom:    DATE '2020-01-05', txTo:    ${INF}})`,
);
query(
  g,
  `MATCH (p:Person {id:'alice'}), (v:PersonVersion {vid:1})
          INSERT (p)-[:HAS_VERSION]->(v)`,
);
```

The identity node holds only the immutable key; every mutable attribute lives on the version nodes, fanned out from it by `:HAS_VERSION`. An **as-of query is the same four-term predicate**, now matched over the version nodes:

```ts
const asOf = (tx: string, valid: string) =>
  query(
    g,
    `MATCH (p:Person {id:'alice'})-[:HAS_VERSION]->(v:PersonVersion)
     WHERE v.validFrom <= date($valid) AND date($valid) < v.validTo
       AND v.txFrom    <= date($tx)    AND date($tx)    < v.txTo
     RETURN v.title AS title`,
    { tx, valid },
  );
```

### The correction: close one belief, open a _split_ of two

Say on 2021-06-10 we learn Alice actually became **Senior Engineer** on 2021-06-01. The old belief was "Engineer, forever." The corrected belief splits valid time in two — Engineer `[2020-01-01, 2021-06-01)`, Senior Engineer `[2021-06-01, ∞)` — both recorded under a fresh, open transaction interval. One transaction:

```ts
g.transaction(() => {
  // 1. Close the currently-believed version (the one whose tx interval is open).
  query(
    g,
    `MATCH (p:Person {id:'alice'})-[:HAS_VERSION]->(v:PersonVersion)
            WHERE v.txTo = ${INF}
            SET v.txTo = DATE '2021-06-10'`,
  );
  // 2a. Re-assert the unchanged pre-correction slice under the new belief.
  query(
    g,
    `MATCH (p:Person {id:'alice'})
            INSERT (p)-[:HAS_VERSION]->(:PersonVersion {vid: 2, title: 'Engineer',
              validFrom: DATE '2020-01-01', validTo: DATE '2021-06-01',
              txFrom:    DATE '2021-06-10', txTo:    ${INF}})`,
  );
  // 2b. Insert the corrected post-change slice.
  query(
    g,
    `MATCH (p:Person {id:'alice'})
            INSERT (p)-[:HAS_VERSION]->(:PersonVersion {vid: 3, title: 'Senior Engineer',
              validFrom: DATE '2021-06-01', validTo: ${INF},
              txFrom:    DATE '2021-06-10', txTo:    ${INF}})`,
  );
});
```

After it, history is intact and the split is clean:

- `asOf('2021-01-01', '2021-07-01')` → **Engineer** — the old belief is untouched (an audit trail, not a mutation).
- `asOf('2021-08-01', '2021-07-01')` → **Senior Engineer** — the corrected belief.
- `asOf('2021-08-01', '2020-03-01')` → **Engineer** — the pre-change slice is still there under the new belief.

### Supersession pitfalls (the part teams get wrong)

The version-node shape has four failure modes the edge-period shape mostly sidesteps. Each produces _wrong answers with no error_, so guard for them:

1. **Re-assert the unchanged slice — don't just close and append one.** A correction that splits an interval must re-insert BOTH halves under the new belief (steps 2a **and** 2b). Skipping 2a leaves a hole: `asOf('2021-08-01', '2020-03-01')` would return no row, silently losing the still-true early history.
2. **Only one open version may cover a given valid instant.** Within a single belief (`txTo = INF`), the version nodes' valid intervals must **partition** — no overlap. Two open versions valid at the same instant makes an as-of query return _two_ rows for one entity. After any correction, assert the invariant: `MATCH (p)-[:HAS_VERSION]->(v) WHERE v.txTo = <INF> AND v.validFrom <= date($t) AND date($t) < v.validTo` must return **exactly one** row per entity.
3. **Close by the open-tx match, not by id.** Step 1 closes "the version whose `txTo` is the sentinel," never a hardcoded `vid`. Closing the wrong (already-superseded) version corrupts the belief history; matching on `txTo = INF` always finds the live one.
4. **Link every new version to the identity node in the same statement.** An `INSERT (:PersonVersion {…})` with no `(p)-[:HAS_VERSION]->` makes an **orphan** — invisible to every as-of query (which traverses from the identity), yet occupying space and, worse, matching a bare `MATCH (v:PersonVersion)`. The `INSERT (p)-[:HAS_VERSION]->(:PersonVersion {…})` form above creates the node and its link atomically, so an orphan is impossible.

Wrap the close-and-split in `graph.transaction(fn)` (as shown) so a throw between the writes rolls the whole correction back — never a half-applied belief.

## Why no `AS OF` keyword

An as-of query on lenke is _just a `WHERE` clause over the period columns you chose_. Adding an `AS OF` keyword would buy nothing here — worse, it would be a lie about what the engine does.

SQL:2011 earns `AS OF SYSTEM_TIME` only because it also mandates `PERIOD FOR SYSTEM_TIME (sys_start, sys_end)` — the engine _owns_ those two columns, auto-stamps them on every write, and forbids you from setting them. The keyword is sugar over an engine-managed schema contract. lenke deliberately declines that contract: **you** pick the column names, **you** decide how many axes you model (uni- or bitemporal), and **you** stamp them in your own writes (a correction's `txFrom` is _your_ correction date, not the engine's commit instant). That flexibility is the point — a knowledge graph often wants valid time without system time, or four columns with domain-specific names. A hardcoded `AS OF` over engine-owned `PERIOD` columns would impose exactly the schema mandate lenke avoids. The predicate is three lines; keep the control.

## Temporal caveats

A few behaviors worth internalizing before you build windows or velocity checks on top of the period columns:

- **Datetimes are instants, not wall clocks — and the math is not JavaScript `Date`.** A duration adds elapsed **seconds**, and a day is exactly 86,400 s, so `P1D` and `PT24H` are the _same_ operation (there is no DST-shortened calendar day). A `ZONED DATETIME` carries a fixed **offset** (`-05:00`), not a zone (`America/New_York`), and lenke has no IANA tz database — so adding a duration keeps the offset **frozen** and never "springs forward". The resulting _instant_ is exactly correct (e.g. `zoned_datetime('2026-03-07T20:00:00-05:00') + P1D` is `2026-03-08T20:00:00-05:00`, which as UTC is precisely +24 h), but the wall-clock **label** can go stale across a DST change — a tz-aware library would render that same instant as `21:00-04:00`. Because `_hour`/`_day` read the value's own frozen offset, a `GROUP BY` over data that crossed a DST edge can bucket an hour — or, near midnight, a whole day — off. **Store instants as UTC / `Z`, treat lenke datetimes as instants, and do all zone/DST conversion at the boundary with a real tz library** (TC39 `Temporal`, Luxon, `date-fns-tz`). The upside of this model: unlike JS `Date`, lenke never reads the host machine's timezone, so results are deterministic across machines and byte-identical on both engines. (Four-column `DATE`-only modeling as shown above sidesteps all of this — a `DATE` has no time-of-day and no offset.)
- **Never compare two durations relationally.** `duration <op> duration` (e.g. `WHERE (e.txTo - e.txFrom) < DURATION 'P30D'`) is `UNKNOWN` under GQL's three-valued logic, and a `WHERE` that evaluates to `UNKNOWN` **silently drops the row** — the query returns nothing, with no error. Compare **instants** instead: anchor the duration to a point in time and compare the resulting instants — `WHERE e.txTo < e.txFrom + DURATION 'P30D'` ("open less than 30 days") or `WHERE e.txTo >= e.txFrom + DURATION 'P30D'`. The same rule governs any "within N days / at least N days" window.
- **Temporal literal prefixes are BARE.** Write `DATETIME '…'`, `DATE '…'`, `TIMESTAMP '…'`, `DURATION '…'` — never `LOCAL DATETIME '…'`. Even though the underlying _type_ is named `LOCAL DATETIME` / `LOCAL TIME`, `LOCAL` is a reserved word in that position, so a `LOCAL DATETIME '…'` literal is `E_SYNTAX`.
- **Temporal arithmetic overflow THROWS — it is never a silent null.** An out-of-range result is a loud `E_DATA_EXCEPTION` (like division by zero), byte-identical in both engines, so a swallowed overflow can't masquerade as data:
  - `date/datetime ± duration` that lands outside the representable date range (a `DATE` is `i32` days, ≈±5.88M years) → `E_DATA_EXCEPTION` (`FAULT_DATE_OVERFLOW`).
  - `duration ± duration` / `duration × n` whose component leaves the f64-safe-integer range (≥ 2⁵³) → `E_DATA_EXCEPTION` (`FAULT_DURATION_OVERFLOW`).
  - A `DURATION '…'` literal past 2⁵³ is rejected earlier still, at parse (`E_SYNTAX`).
- **Temporal aggregates: `min`/`max`/`sum` compute; `avg` throws.** `min`/`max` over any temporal use the total order. `sum` is defined **only for `DURATION`** (folded component-wise, with the same overflow-throw as `dur + dur`) — a real "total tenure" duration, not a `NaN → null`. But:
  - `sum` over a **non-`DURATION`** temporal (`DATE`/`TIME`/…) is `E_DATA_EXCEPTION` — dates and times aren't summable.
  - `avg` over **any** temporal is `E_DATA_EXCEPTION` — averaging needs `duration ÷ count`, which is often non-representable (`avg(P1M, P2M) = P1.5M`, and a half-month has no integer-component form). Use `sum()` and divide in the host, or `min()`/`max()`.

## Durability

Because the temporal columns are ordinary `DATE` properties, they survive serialization with everything else — snapshot the whole graph (`serialize(g, 'ndjson')`), restore into a fresh `Graph`, and every as-of query answers identically against the restored copy. Bitemporality needs no special persistence path.
