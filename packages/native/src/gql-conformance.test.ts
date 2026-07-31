// Differential conformance for GQL rich results: the TS GQL engine
// (@lenke/gql, in-process over @lenke/core) vs the Rust core (this package,
// over bun:ffi), driven from ONE source of truth — the same NDJSON loaded into
// both — so a `RETURN n` / `RETURN r` shape can't drift between the two forms.
//
//   load once:   identical NDJSON (same ids/labels/properties)
//   TS engine:   JSON.stringify(query(tsGraph, q))
//   Rust core:   JSON.stringify(nativeGraph.query(q))
//   assert:      the two serializations are byte-identical
//
// This pins the "rich results" contract: a returned node serializes to
// `{id, labels, properties}` and a returned edge to
// `{id, from, to, labels, properties}`, with labels and property keys in
// sorted order (the columnar core has no per-element key order, so both
// engines canonicalize to sorted). A bare-id regression on either side, or a
// key-ordering divergence, shows up here as a red diff.
//
// Run: bun test packages/native/src/gql-conformance.test.ts
import { afterAll, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph, parseDate, parseDateTime } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './backend-ffi.js';
import { graphFromNdjson } from './graph.js';

// --- native library bootstrap (mirrors gremlin-conformance.test.ts) ---------
const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[gql-conformance] skipping: ${LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

// Same ids/labels/properties as the TinkerPop "modern" graph. Property keys are
// authored in NON-sorted insertion order (`name` before `age`, `weight` before
// `since`) precisely so the test proves both engines re-sort them on output.
const MODERN_NDJSON = [
  '{"type":"node","id":"1","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"2","labels":["Person"],"properties":{"name":"vadas","age":27}}',
  '{"type":"node","id":"4","labels":["Person"],"properties":{"name":"josh","age":32}}',
  '{"type":"node","id":"3","labels":["Software"],"properties":{"name":"lop","lang":"java"}}',
  '{"type":"edge","id":"7","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5,"since":2018}}',
  '{"type":"edge","id":"8","from":"1","to":"4","labels":["KNOWS"],"properties":{"weight":1.0,"since":2020}}',
  '{"type":"edge","id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"weight":0.4,"since":2009}}',
].join('\n');

suite('GQL differential: rich RETURN results (TS vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, MODERN_NDJSON);
  const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());

  const both = (q: string, params?: Record<string, unknown>): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q, params)),
    JSON.stringify(nativeGraph.query(q, params)),
  ];

  test('extreme-magnitude numbers render in exponential form (JS toString), byte-identical', () => {
    // Rust gql once used Display (never exponential): 1e21 → "1000000000000000000000",
    // TS → "1e+21". Both the numeric cell and toString(n) now match JS Number::toString.
    for (const [expr, want] of [
      ['1e21', '1e+21'],
      ['1e-7', '1e-7'],
      ['1e100', '1e+100'],
    ] as const) {
      const [ts, native] = both(`RETURN ${expr} AS n, toString(${expr}) AS s`);
      expect(ts).toBe(native);
      expect(ts).toBe(`[{"n":${want},"s":"${want}"}]`);
    }
  });

  test('count(DISTINCT ...) dedups structurally, not by object identity, byte-identical', () => {
    // A constant temporal is a FRESH instance per row; `[x]` a fresh array. Rust
    // dedups by val_key (structural), so both collapse. TS once used a reference-
    // identity Set (kept every instance separate) — now it dedups by valueKey too.
    for (const q of [
      // every Person row yields the same date value → 1 distinct.
      `MATCH (n:Person) RETURN count(DISTINCT date('2020-01-01')) AS c`,
      // both Software nodes have lang 'java' → [n.lang] is [java] both times → 1.
      `MATCH (n:Software) RETURN count(DISTINCT [n.lang]) AS c`,
    ]) {
      const [ts, native] = both(q);
      expect(ts).toBe(native);
      expect(ts).toBe(`[{"c":1}]`);
    }
  });

  test('string comparison is by code point, not UTF-16 unit (astral vs BMP), byte-identical', () => {
    // '😀' is U+1F600 (code point 0x1F600 > 0xE000), so '😀' > ''. JS `<`/`>`
    // once said the opposite (the high surrogate 0xD83D < 0xE000 by code unit),
    // diverging from Rust str::cmp. Now both engines order by code point.
    const [ts, native] = both(
      `RETURN ('😀' < '\\uE000') AS lt, ('😀' > '\\uE000') AS gt, ('😀' = '😀') AS eq`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"lt":false,"gt":true,"eq":true}]`);
  });

  test('min/max over a computed NaN agrees across engines (vectorized == scalar)', () => {
    // sqrt(age-30): marko(29)→NaN, vadas(27)→NaN, josh(32)→~1.41. A first-seen NaN
    // sticks under cmp_total/compareValues (min/max is a reduce), so both are NaN →
    // null. The vectorized Rust fold once used f64::min/max, which DROP NaN → ~1.41,
    // diverging from the scalar path and from TS. Now all three agree.
    const [ts, native] = both(
      `MATCH (n:Person) RETURN min(sqrt(n.age - 30)) AS lo, max(sqrt(n.age - 30)) AS hi`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"lo":null,"hi":null}]`);
  });

  test('sum/avg over a mixed numeric+temporal group faults in both engines', () => {
    // p is a number on one T and a DATE on another — an unsummable heterogeneous
    // group. Rust once checked only the first value (numeric) → DATE coerced to
    // NaN → null (no error), while TS scanned all values and threw. Now Rust scans
    // all values too, so both fault (order-independent).
    const mixed = [
      '{"type":"node","id":"1","labels":["T"],"properties":{"p":5}}',
      '{"type":"node","id":"2","labels":["T"],"properties":{"p":{"@date":"2020-01-01"}}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, mixed);
    const ts = tsDeserialize(mixed, 'ndjson', new Graph());

    for (const q of [`MATCH (n:T) RETURN sum(n.p) AS x`, `MATCH (n:T) RETURN avg(n.p) AS x`]) {
      expect(() => tsQuery(ts, q)).toThrow();
      expect(() => nat.query(q)).toThrow();
    }
  });

  test('scalar numeric fns coerce like Rust num_of (not JS Number), byte-identical', () => {
    // JS Number('0x10') is 16 and Number([5]) is 5; Rust num_of gives NaN → null.
    const [ts, native] = both(`RETURN abs('0x10') AS a, abs([5]) AS b, abs('12') AS c`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"a":null,"b":null,"c":12}]`);
  });

  test('-0 and +0 are ONE GROUP BY/DISTINCT key, byte-identical', () => {
    // (age-30)*0 yields -0 for marko/vadas (age<30) and +0 for josh (age>30).
    //
    // These used to be two groups because the Rust `val_key` keyed by raw bit
    // pattern and TS was aligned to it. That matched an implementation detail
    // rather than a rule: `-0 = 0` is TRUE, and the engine normalizes the
    // distinction everywhere else — ORDER BY and the total order sort them equal,
    // `sign()` returns 0 for both, the result JSON and `to_string` both render
    // `0`, an indexed equality seek finds both, `1 / ±0` faults rather than
    // yielding ±∞, and the Gremlin engine's `dedup_key` already collapsed them.
    // Grouping was the ONE place they differed, which produced two groups whose
    // rendered values were both `0` — a distinction no result could show.
    const [ts, native] = both(`MATCH (n:Person) RETURN count(DISTINCT (n.age - 30) * 0) AS c`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":1}]`);
    // The group holds every row, and its key renders as a plain `0`.
    const [gts, gnative] = both(
      `MATCH (n:Person) RETURN (n.age - 30) * 0 AS k, count(*) AS c GROUP BY k`,
    );

    expect(gts).toBe(gnative);
    expect(gts).toBe(`[{"k":0,"c":3}]`);
  });

  test('list subscript with a non-number index is null in both engines', () => {
    // A number index works; a string / boolean / non-integer index is null (the
    // ISO "non-integer list index → null" contract). Rust once coerced ['1']→index 1;
    // TS once threw on a non-number index. Now both return null.
    const [ts, native] = both(
      `RETURN [10,20,30][1] AS a, [10,20,30]['1'] AS b, [10,20,30][true] AS c, [10,20,30][1.5] AS d`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"a":20,"b":null,"c":null,"d":null}]`);
  });

  test('numeric-string coercion is strict + byte-identical (inf/nan/hex)', () => {
    // One strict grammar across both engines: non-finite spellings and hex/octal
    // coerce to NaN (scalar fns / aggregates) or NULL (to_integer/to_float). Rust
    // once accepted 'inf' (str::parse) and TS aggregates once accepted hex (Number()).
    const [ts, native] = both(
      `RETURN sign('inf') AS a, abs('inf') AS b, (to_integer('nan') IS NULL) AS c, (to_float('inf') IS NULL) AS d`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"a":null,"b":null,"c":true,"d":true}]`);

    // Aggregates coerce a hex / inf string to NaN → null in both, not JS Number()'s 16.
    const [ts2, nat2] = both(`MATCH (n:Person) RETURN sum('0x10') AS s, sum('inf') AS t`);
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(`[{"s":null,"t":null}]`);
  });

  test('RETURN n — rich node object, byte-identical, keys sorted', () => {
    const q = `MATCH (n:Person {name: 'marko'}) RETURN n`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(
      `[{"n":{"id":"1","labels":["Person"],"properties":{"age":29,"name":"marko"}}}]`,
    );
  });

  test('RETURN r — rich edge object, byte-identical, keys sorted', () => {
    const q = `MATCH (:Person {name: 'marko'})-[r:KNOWS]->(:Person {name: 'josh'}) RETURN r`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(
      `[{"r":{"id":"8","from":"1","to":"4","labels":["KNOWS"],"properties":{"since":2020,"weight":1}}}]`,
    );
  });

  test('RETURN * — a whole node column serializes richly and identically', () => {
    const [ts, native] = both(`MATCH (n:Person {name: 'vadas'}) RETURN *`);
    expect(ts).toBe(native);
  });

  test('RETURN both endpoints — every element column is rich and identical', () => {
    const q = `MATCH (a:Person {name: 'marko'})-[:CREATED]->(b:Software) RETURN a, b ORDER BY b.name`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
  });

  test('a scalar projection is unaffected (still a plain value, identical)', () => {
    const [ts, native] = both(`MATCH (n:Person) RETURN n.name AS name ORDER BY name`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"josh"},{"name":"marko"},{"name":"vadas"}]`);
  });

  test('property_names(n) — ISO GQL element function, sorted, byte-identical', () => {
    const q = `MATCH (n:Person {name: 'marko'}) RETURN property_names(n) AS ks`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    // Sorted property-name list (both engines canonicalize to sorted order).
    expect(ts).toBe(`[{"ks":["age","name"]}]`);
    // `keys(n)` is the openCypher spelling of the same function — identical result.
    const [tsK, natK] = both(`MATCH (n:Person {name: 'marko'}) RETURN keys(n) AS ks`);
    expect(tsK).toBe(natK);
    expect(tsK).toBe(ts);
  });

  test('stddev_pop / stddev_samp — ISO aggregates, byte-identical f64', () => {
    // Global over the three Person ages (29, 27, 32).
    const [ts, native] = both(
      `MATCH (n:Person) RETURN stddev_pop(n.age) AS sp, stddev_samp(n.age) AS ss`,
    );
    expect(ts).toBe(native);
    // Grouped stddev (per label bucket) — exercises the group-fold path.
    const [tsG, natG] = both(
      `MATCH (n)-[e:KNOWS]->(m) RETURN stddev_pop(e.weight) AS sp, count(*) AS c`,
    );
    expect(tsG).toBe(natG);
    // Edge cases: 1 value ⇒ pop = 0, samp = null; the exact numeric shape.
    const [ts1, nat1] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN stddev_pop(n.age) AS sp, stddev_samp(n.age) AS ss`,
    );
    expect(ts1).toBe(nat1);
    expect(ts1).toBe(`[{"sp":0,"ss":null}]`);
  });

  test('VALUE { … } — ISO scalar subquery, correlated + aggregate, byte-identical', () => {
    // Correlated single-row: marko's one CREATED target is "lop".
    const [ts1, nat1] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN VALUE { MATCH (n)-[:CREATED]->(s) RETURN s.name } AS made`,
    );
    expect(ts1).toBe(nat1);
    expect(ts1).toBe(`[{"made":"lop"}]`);

    // 0 rows → NULL: vadas has no out-edges.
    const [ts2, nat2] = both(
      `MATCH (n:Person {name: 'vadas'}) RETURN VALUE { MATCH (n)-[:KNOWS]->(m) RETURN m.name } AS f`,
    );
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(`[{"f":null}]`);

    // Aggregate RETURN folds the group: three Person nodes.
    const [ts3, nat3] = both(`RETURN VALUE { MATCH (n:Person) RETURN count(*) } AS c`);
    expect(ts3).toBe(nat3);
    expect(ts3).toBe(`[{"c":3}]`);

    // Correlated aggregate: marko's KNOWS out-degree is 2 (no cardinality error).
    const [ts4, nat4] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN VALUE { MATCH (n)-[:KNOWS]->(m) RETURN count(*) } AS deg`,
    );
    expect(ts4).toBe(nat4);
    expect(ts4).toBe(`[{"deg":2}]`);

    // No patterns → a constant scalar.
    const [ts5, nat5] = both(`RETURN VALUE { RETURN 1 + 2 } AS v`);
    expect(ts5).toBe(nat5);
    expect(ts5).toBe(`[{"v":3}]`);
  });

  test('VALUE { … } — a multi-row non-aggregate RETURN is a cardinality error in both', () => {
    // marko has two KNOWS neighbours; a non-aggregate scalar subquery must fault.
    const q = `MATCH (n:Person {name: 'marko'}) RETURN VALUE { MATCH (n)-[:KNOWS]->(m) RETURN m.name } AS f`;
    expect(() => tsQuery(tsGraph, q)).toThrow();
    expect(() => nativeGraph.query(q)).toThrow();
  });

  test('LET … IN … END — ISO scalar let-expression, byte-identical', () => {
    // Constant fold.
    const [ts1, nat1] = both(`RETURN LET x = 2 + 3 IN x * x END AS v`);
    expect(ts1).toBe(nat1);
    expect(ts1).toBe(`[{"v":25}]`);

    // Multiple bindings; a later binding sees an earlier one.
    const [ts2, nat2] = both(`RETURN LET x = 4, y = x + 1 IN x * y END AS v`);
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(`[{"v":20}]`);

    // Correlated: the binding reads an outer variable.
    const [ts3, nat3] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN LET u = n.name IN u || '!' END AS greet`,
    );
    expect(ts3).toBe(nat3);
    expect(ts3).toBe(`[{"greet":"marko!"}]`);

    // The binding RHS ends at the structural IN (bare IN operator suppressed).
    const [ts4, nat4] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN LET nm = n.name IN nm = 'marko' END AS ok`,
    );
    expect(ts4).toBe(nat4);
    expect(ts4).toBe(`[{"ok":true}]`);

    // A parenthesized IN predicate inside a binding still works (parens re-enable
    // the operator).
    const [ts5, nat5] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN LET hit = (n.name IN ['marko', 'josh']) IN hit END AS present`,
    );
    expect(ts5).toBe(nat5);
    expect(ts5).toBe(`[{"present":true}]`);
  });

  test('parenthesized-subpath WHERE — ISO, byte-identical, distinct from clause WHERE', () => {
    // MODERN: marko(29)→vadas(27), marko(29)→josh(32) (KNOWS); marko(29)→lop (CREATED).
    // Subpath WHERE spanning both endpoints: KNOWS pairs where age(x) < age(y).
    const [ts1, nat1] = both(
      `MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) RETURN y.name AS n ORDER BY n`,
    );
    expect(ts1).toBe(nat1);
    expect(ts1).toBe(`[{"n":"josh"}]`); // only marko(29)→josh(32)

    // The SAME predicate as a clause WHERE yields the SAME rows (proves neither is
    // misinterpreted for a single non-quantified pattern).
    const [ts2, nat2] = both(
      `MATCH (x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age RETURN y.name AS n ORDER BY n`,
    );
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(ts1);

    // A subpath WHERE AND a trailing clause WHERE compose (both applied, AND).
    const [ts3, nat3] = both(
      `MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) WHERE y.name = 'josh' RETURN x.name AS n`,
    );
    expect(ts3).toBe(nat3);
    expect(ts3).toBe(`[{"n":"marko"}]`);

    // Same subpath, a clause WHERE that excludes the row → empty (clause really runs).
    const [ts4, nat4] = both(
      `MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) WHERE y.name = 'vadas' RETURN x.name AS n`,
    );
    expect(ts4).toBe(nat4);
    expect(ts4).toBe(`[]`);

    // Subpath WHERE referencing the edge; single-node subpath.
    const [ts5, nat5] = both(
      `MATCH ((x:Person)-[e:KNOWS]->(y:Person) WHERE e.weight > 0.7) RETURN y.name AS n`,
    );
    expect(ts5).toBe(nat5);
    expect(ts5).toBe(`[{"n":"josh"}]`); // marko→josh weight 1.0; marko→vadas 0.5
    const [ts6, nat6] = both(
      `MATCH ((n:Person) WHERE n.age >= 29) RETURN n.name AS nm ORDER BY nm`,
    );
    expect(ts6).toBe(nat6);
    expect(ts6).toBe(`[{"nm":"josh"},{"nm":"marko"}]`);
  });

  test('a quantified / path-var subpath is rejected in BOTH engines', () => {
    for (const q of [
      `MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age)+ RETURN x.name AS n`,
      `MATCH p = ((x:Person)-[:KNOWS]->(y:Person)) RETURN x.name AS n`,
    ]) {
      expect(() => tsQuery(tsGraph, q)).toThrow();
      expect(() => nativeGraph.query(q)).toThrow();
    }
  });

  test('SELECT statement + HAVING — ISO, byte-identical', () => {
    // MODERN: Person marko(29)/vadas(27)/josh(32); Software lop. SELECT desugars
    // to MATCH + RETURN.
    const [ts1, nat1] = both(`SELECT 1 + 2 AS v`);
    expect(ts1).toBe(nat1);
    expect(ts1).toBe(`[{"v":3}]`);

    // GROUP BY label with a count.
    const [ts2, nat2] = both(
      `SELECT labels(n)[0] AS lab, count(*) AS c FROM MATCH (n) GROUP BY labels(n)[0] ORDER BY lab`,
    );
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(`[{"lab":"Person","c":3},{"lab":"Software","c":1}]`);

    // HAVING on the grouped count: keep only labels with >1 member → Person.
    const [ts3, nat3] = both(
      `SELECT labels(n)[0] AS lab, count(*) AS c FROM MATCH (n) ` +
        `GROUP BY labels(n)[0] HAVING count(*) > 1 ORDER BY lab`,
    );
    expect(ts3).toBe(nat3);
    expect(ts3).toBe(`[{"lab":"Person","c":3}]`);

    // HAVING on a global aggregate — keep or drop the single row.
    const [ts4, nat4] = both(`SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 2`);
    expect(ts4).toBe(nat4);
    expect(ts4).toBe(`[{"c":3}]`);
    const [ts5, nat5] = both(`SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 100`);
    expect(ts5).toBe(nat5);
    expect(ts5).toBe(`[]`);

    // A pre-aggregation WHERE, then HAVING referencing an aggregate not projected.
    const [ts6, nat6] = both(
      `SELECT labels(n)[0] AS lab FROM MATCH (n) WHERE n.name <> 'lop' ` +
        `GROUP BY labels(n)[0] HAVING count(*) >= 3`,
    );
    expect(ts6).toBe(nat6);
    expect(ts6).toBe(`[{"lab":"Person"}]`);
  });

  test('HAVING is SELECT-only — rejected on a bare RETURN in both engines', () => {
    const q = `MATCH (n:Person) RETURN count(*) AS c HAVING count(*) > 1`;
    expect(() => tsQuery(tsGraph, q)).toThrow();
    expect(() => nativeGraph.query(q)).toThrow();
  });

  test('record constructor + field access — ISO map value, byte-identical', () => {
    // Constructor canonicalizes (sorted keys, dup last-wins); access by dot and
    // by string subscript; nested; missing field → null.
    const cases: Array<[string, string]> = [
      [`RETURN {name: 'marko', age: 29} AS r`, `[{"r":{"age":29,"name":"marko"}}]`],
      [`RETURN {} AS r`, `[{"r":{}}]`],
      [`RETURN {a: 1, a: 2} AS r`, `[{"r":{"a":2}}]`],
      [`RETURN {a: 1, b: 2}.a AS x`, `[{"x":1}]`],
      [`RETURN {a: 1, b: 2}['b'] AS x`, `[{"x":2}]`],
      [`RETURN {a: 1}.zzz AS x`, `[{"x":null}]`],
      [`RETURN {p: {n: 5}}.p.n AS x`, `[{"x":5}]`],
      // A field value is any expression; correlated to a matched element.
      [
        `MATCH (n:Person {name: 'marko'}) RETURN {who: n.name, yrs: n.age} AS r`,
        `[{"r":{"who":"marko","yrs":29}}]`,
      ],
      // Equality is structural; ordering / DISTINCT are total.
      [`RETURN {a: 1, b: 2} = {b: 2, a: 1} AS eq`, `[{"eq":true}]`],
      [`RETURN {a: 1} = {a: 2} AS eq`, `[{"eq":false}]`],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);
      expect(ts).toBe(native);
      expect(ts).toBe(want);
    }
  });

  test('list[i] — ISO GQL 0-based subscript, null-safe, byte-identical', () => {
    const cases: Array<[string, string]> = [
      // 0-based: [0] is the first element.
      [`RETURN [10, 20, 30][0] AS a`, `[{"a":10}]`],
      [`RETURN [10, 20, 30][2] AS a`, `[{"a":30}]`],
      // Out of range / negative / null index → null (null-safe).
      [`RETURN [10, 20, 30][5] AS a`, `[{"a":null}]`],
      [`RETURN [10, 20, 30][-1] AS a`, `[{"a":null}]`],
      [`RETURN [10, 20, 30][null] AS a`, `[{"a":null}]`],
      // Index is any expression; chained subscripts nest left to right.
      [`RETURN [10, 20, 30][1 + 1] AS a`, `[{"a":30}]`],
      [`RETURN [[1, 2], [3, 4]][1][0] AS a`, `[{"a":3}]`],
      // Non-list base → null (not an error).
      [`RETURN 5[0] AS a`, `[{"a":null}]`],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }

    // Indexing a collected list over a bound variable.
    const [tsC, natC] = both(
      `MATCH (n:Person) WITH collect_list(n.name) AS names RETURN names[0] AS first`,
    );
    expect(tsC).toBe(natC);
  });

  test('cardinality(list) — ISO GQL name for collection size, == size', () => {
    const [ts, native] = both(`RETURN cardinality([10, 20, 30]) AS c, size([10, 20, 30]) AS s`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":3,"s":3}]`);
    // Over a collected list bound to a variable.
    const [tsC, natC] = both(
      `MATCH (n:Person) WITH collect_list(n.name) AS names RETURN cardinality(names) AS c`,
    );
    expect(tsC).toBe(natC);
  });

  // --- tagged-temporal param revival: a single-key `{'@date':'…'}` param (the
  // engine's OWN tagged output shape, for @date/@datetime/@localtime/@zoned_time/
  // @zoned_datetime/@duration) is revived into its temporal value, so the output
  // round-trips as an input param. Native revives while parsing the param string;
  // this pins the TS engine to the same behavior (was a silent divergence: TS
  // left the plain object un-revived → temporal-vs-object compare → UNKNOWN → 0).
  test('tagged-temporal param revives + round-trips, byte-identical', () => {
    const [ts, native] = both(`RETURN $asof AS d`, { asof: { '@date': '2020-07-01' } });
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"d":{"@date":"2020-07-01"}}]`);

    // The revived param compares as a temporal against a temporal literal.
    const [tsCmp, natCmp] = both(`RETURN (DATE '2020-06-01' <= $asof) AS le`, {
      asof: { '@date': '2020-07-01' },
    });
    expect(tsCmp).toBe(natCmp);
    expect(tsCmp).toBe(`[{"le":true}]`);

    // Every tagged kind revives; also inside a list param. @zoned_time is here
    // explicitly: the host param gate once rejected it (its accepted-tag set had
    // a bogus '@time' and omitted the real '@zoned_time'), so a valid ZONED TIME
    // param threw E_INVALID_JSON before reaching the engine. Now the gate derives
    // from the canonical TEMPORAL_TAG_KEYS, so every kind fromTaggedJson accepts
    // passes through byte-identically.
    const [tsAll, natAll] = both(`RETURN $dt AS dt, $dur AS dur, $zt AS zt, $xs AS xs`, {
      dt: { '@datetime': '2020-06-15T08:30:00' },
      dur: { '@duration': 'P1Y2M3DT4H' },
      zt: { '@zoned_time': '12:00:00Z' },
      xs: [{ '@date': '2020-01-01' }, { '@localtime': '08:30:00' }],
    });
    expect(tsAll).toBe(natAll);
    expect(tsAll).toContain(`"@zoned_time"`);

    // A bogus tag ('@time' is never emitted by any toJSON) is rejected by the
    // host gate — not silently shipped to the crate to fault differently.
    expect(() => nativeGraph.query(`RETURN $t AS t`, { t: { '@time': '12:00:00' } })).toThrow();
  });

  // --- nested non-finite under DISTINCT: persons have `age` (sqrt(age-100)=NaN),
  // software nodes don't (sqrt(null)=null), so `[sqrt(age-100)]` yields the two
  // groups [NaN] and [null] in BOTH engines. The TS distinct key once JSON.stringify'd
  // the list, folding NaN into null (one group) — Rust's bit-keyed val_key never did.
  // Now the key tags non-finite at any depth, so the partition matches.
  test('nested non-finite in a DISTINCT list stays distinct, byte-identical', () => {
    const q = `MATCH (n) RETURN DISTINCT [sqrt(n.age - 100)] AS k`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(2);
  });

  // --- ANY SHORTEST: the path value serializes byte-identically across engines.
  test('RETURN p — a shortest Path is {vertices, edges, length}, byte-identical', () => {
    const q = `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop' RETURN p`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(
      `[{"p":{"vertices":[` +
        `{"id":"1","labels":["Person"],"properties":{"age":29,"name":"marko"}},` +
        `{"id":"3","labels":["Software"],"properties":{"lang":"java","name":"lop"}}` +
        `],"edges":[` +
        `{"id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"since":2009,"weight":0.4}}` +
        `],"length":1}}]`,
    );
  });

  test('ANY SHORTEST endpoint set + per-endpoint path, identical under ORDER BY', () => {
    const [ts, native] = both(
      `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' RETURN b.name AS n, p ORDER BY n`,
    );
    expect(ts).toBe(native);
  });

  test('edges(path) — ISO name for the path edge list; relationships() is rejected', () => {
    const base = `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop'`;
    const [tsE, natE] = both(`${base} RETURN path_length(p) AS len, edges(p) AS es`);
    expect(tsE).toBe(natE);
    // Cypher's `relationships` is deliberately NOT accepted — GQL vocabulary is
    // node/edge. Both engines reject it as an unknown function.
    const q = `${base} RETURN relationships(p) AS es`;
    expect(() => tsQuery(tsGraph, q)).toThrow();
    expect(() => nativeGraph.query(q)).toThrow();
  });

  test('temporal component fns _year()/…/_second() — byte-identical; string rejected', () => {
    // Date parts, time parts, zoned-in-own-offset, pre-epoch, and null-in→null-out
    // all agree bit-for-bit across engines.
    const agree = [
      `RETURN _year(DATE '2024-03-15') AS y, _month(DATE '2024-03-15') AS mo, _day(DATE '2024-03-15') AS d`,
      `RETURN _hour(DATETIME '2024-03-15T13:47:09') AS h, _minute(DATETIME '2024-03-15T13:47:09') AS mi, _second(DATETIME '2024-03-15T13:47:09') AS s`,
      `RETURN _hour(local_time('13:47:09')) AS h, _minute(local_time('13:47:09')) AS mi`,
      // A zoned value reads its own offset (local wall clock), not UTC.
      `RETURN _day(zoned_datetime('2024-03-15T23:30:00+05:00')) AS d, _hour(zoned_datetime('2024-03-15T23:30:00+05:00')) AS h`,
      `RETURN _hour(zoned_time('01:15:00+02:00')) AS h`,
      // Pre-epoch date (negative epoch-day count).
      `RETURN _year(DATE '1969-12-31') AS y, _day(DATE '1969-12-31') AS d`,
      `RETURN _year(null) AS y`,
    ];

    for (const q of agree) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
    }

    // A string is NOT coerced, and a temporal lacking the component faults — both throw.
    for (const q of [
      `RETURN _year('2024-03-15') AS y`,
      `RETURN _hour(DATE '2024-03-15') AS h`,
      `RETURN _year(local_time('13:47:09')) AS y`,
    ]) {
      expect(() => tsQuery(tsGraph, q)).toThrow();
      expect(() => nativeGraph.query(q)).toThrow();
    }

    // The bare (sigil-less) names are NOT functions — date-parts are a lenke
    // extension, so only the `_`-prefixed form resolves. Both engines reject.
    const bare = `RETURN year(DATE '2024-03-15') AS y`;

    expect(() => tsQuery(tsGraph, bare)).toThrow();
    expect(() => nativeGraph.query(bare)).toThrow();
  });

  test('PROPERTY_EXISTS(n, key) — byte-identical presence predicate', () => {
    const [ts, native] = both(
      `MATCH (n:Person {name: 'marko'}) RETURN property_exists(n, name) AS hn, property_exists(n, age) AS ha, property_exists(n, nope) AS hx`,
    );

    expect(ts).toBe(native);
    expect(ts).toContain('"hn":true');
    expect(ts).toContain('"hx":false');

    // Edge properties agree.
    const [tsE, natE] = both(
      `MATCH ()-[e:CREATED]->() RETURN property_exists(e, weight) AS hw, property_exists(e, gone) AS hg`,
    );

    expect(tsE).toBe(natE);

    // A NULL element (unmatched OPTIONAL) → NULL, not false — both engines.
    const [tsN, natN] = both(
      `MATCH (n:Person {name: 'marko'}) OPTIONAL MATCH (n)-[:NOSUCH]->(m) RETURN property_exists(m, x) AS hx`,
    );

    expect(tsN).toBe(natN);
  });

  test('IS [NOT] TYPED <type> [NOT NULL] — byte-identical value-type predicate', () => {
    const q =
      `RETURN 5 IS TYPED INTEGER AS a, 5.5 IS TYPED INTEGER AS b, 5.5 IS TYPED FLOAT AS c, ` +
      `5 IS TYPED FLOAT AS d, 'x' IS TYPED STRING AS e, true IS TYPED BOOL AS f, ` +
      `[1,2] IS TYPED LIST AS g, DATE '2020-01-01' IS TYPED DATE AS h, ` +
      `DATETIME '2020-01-01T00:00:00' IS TYPED LOCAL DATETIME AS i, ` +
      `duration('P1D') IS TYPED DURATION AS j, 5 IS NOT TYPED STRING AS k, ` +
      `null IS TYPED INTEGER AS l, null IS TYPED INTEGER NOT NULL AS m, ` +
      `null IS TYPED NULL AS n, null IS TYPED ANY NOT NULL AS o`;
    const [ts, native] = both(q);

    expect(ts).toBe(native);
    // spot-check the tricky ones (numeric inference + null conformance)
    expect(ts).toContain('"a":true');
    expect(ts).toContain('"b":false');
    expect(ts).toContain('"d":true');
    expect(ts).toContain('"l":true');
    expect(ts).toContain('"m":false');
  });

  test('IS TYPED [ANY] RECORD — the open record type, byte-identical', () => {
    const q =
      `RETURN {a: 1} IS TYPED ANY RECORD AS a, {a: 1} IS TYPED RECORD AS b, ` +
      `5 IS TYPED ANY RECORD AS c, [1,2] IS TYPED RECORD AS d, ` +
      `5 IS NOT TYPED RECORD AS e, null IS TYPED ANY RECORD AS f, ` +
      `null IS TYPED ANY RECORD NOT NULL AS g`;
    const [ts, native] = both(q);

    expect(ts).toBe(native);
    expect(ts).toContain('"a":true');
    expect(ts).toContain('"c":false');
    expect(ts).toContain('"e":true');
    expect(ts).toContain('"g":false');
  });

  test('IS TYPED RECORD {…} — the closed record type, byte-identical', () => {
    const q =
      `RETURN {a: 1, b: 'x'} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS a, ` +
      `{a: 1} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS b, ` +
      `{a: 1, b: 'x', c: 9} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS c, ` +
      `{a: 1.5} IS TYPED RECORD {a :: INTEGER} AS d, ` +
      `{} IS TYPED RECORD {a :: INTEGER NOT NULL} AS e, ` +
      `{geo: {lat: 1, lng: 2}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS f`;
    const [ts, native] = both(q);

    expect(ts).toBe(native);
    expect(ts).toContain('"a":true');
    expect(ts).toContain('"c":false'); // closed on extras
    expect(ts).toContain('"d":false'); // 1.5 is not INTEGER
    expect(ts).toContain('"e":false'); // NOT NULL field absent
    expect(ts).toContain('"f":true'); // nested record match
  });

  test('graph-element predicates + ! — byte-identical', () => {
    // IS DIRECTED / IS SOURCE|DESTINATION OF / ALL_DIFFERENT / SAME over a real edge.
    const [ts, native] = both(
      `MATCH (m:Person {name: 'marko'})-[e:CREATED]->(s) ` +
        `RETURN e IS DIRECTED AS d, m IS SOURCE OF e AS msrc, s IS DESTINATION OF e AS sdst, ` +
        `s IS SOURCE OF e AS ssrc, ALL_DIFFERENT(m, s) AS diff, SAME(m, m) AS allsame, ` +
        `SAME(m, s) AS msame ORDER BY sdst LIMIT 1`,
    );

    expect(ts).toBe(native);
    expect(ts).toContain('"d":true');
    expect(ts).toContain('"msrc":true');
    expect(ts).toContain('"ssrc":false');

    // `!` unary-not, tight-binding.
    const [tsB, natB] = both(`RETURN !(1=2) AS a, !true AS b, (!(1=2) = true) AS c`);

    expect(tsB).toBe(natB);

    // NULL element (unmatched OPTIONAL) → NULL, both engines.
    const [tsN, natN] = both(
      `MATCH (m:Person {name: 'marko'}) OPTIONAL MATCH (m)-[:NOSUCH]->(x) ` +
        `RETURN x IS DIRECTED AS d, ALL_DIFFERENT(m, x) AS ad`,
    );

    expect(tsN).toBe(natN);
  });

  test('trim family (2-arg char set) + TRIM(… FROM …) — byte-identical', () => {
    const [ts, native] = both(
      `RETURN btrim('xxhixx','x') AS a, ltrim('xxhixx','x') AS b, rtrim('xxhixx','x') AS c, ` +
        `btrim('xyxhixyx','xy') AS d, trim('  hi  ') AS e, ` +
        `TRIM(LEADING 'x' FROM 'xxhi') AS f, TRIM('x' FROM 'xxhixx') AS g, ` +
        `TRIM(TRAILING FROM 'hi  ') AS h`,
    );

    expect(ts).toBe(native);
    expect(ts).toContain('"a":"hi"');
    expect(ts).toContain('"b":"hixx"');
  });

  test('explicit GROUP BY (RETURN) — byte-identical, incl. group-by-non-returned', () => {
    // marko/vadas/josh/peter are Person; group by age presence etc. Use a stable
    // grouping key (labels) over the modern graph.
    const q1 = `MATCH (n) RETURN labels(n)[0] AS lbl, count(*) AS c GROUP BY labels(n)[0] ORDER BY lbl`;
    const [ts1, nat1] = both(q1);

    expect(ts1).toBe(nat1);

    // GROUP BY a key that is NOT in the RETURN list — implicit grouping can't do this.
    const q2 = `MATCH (n) RETURN count(*) AS c GROUP BY labels(n)[0] ORDER BY c`;
    const [ts2, nat2] = both(q2);

    expect(ts2).toBe(nat2);

    // GROUP BY with no aggregate == DISTINCT on the key.
    const q3 = `MATCH (n) RETURN labels(n)[0] AS lbl GROUP BY labels(n)[0] ORDER BY lbl`;
    const [ts3, nat3] = both(q3);

    expect(ts3).toBe(nat3);
  });

  test('multi-MATCH EXISTS / COUNT { … } — byte-identical', () => {
    for (const q of [
      `RETURN EXISTS { MATCH (a:Person) MATCH (s:Software) } AS e`,
      `RETURN EXISTS { MATCH (a:Person) WHERE a.name='marko' MATCH (s:Software) WHERE s.name='lop' } AS e`,
      `RETURN EXISTS { MATCH (a:Person) MATCH (z:Nope) } AS e`,
      `RETURN COUNT { MATCH (a:Person {name:'marko'}) MATCH (s:Software {name:'lop'}) } AS c`,
    ]) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
    }
  });

  test('match modes REPEATABLE ELEMENTS / DIFFERENT EDGES — byte-identical', () => {
    // marko knows josh knows … (walkable). REPEATABLE ELEMENTS (WALK) allows
    // re-treading; DIFFERENT EDGES (= default TRAIL) does not.
    for (const q of [
      `MATCH REPEATABLE ELEMENTS (m:Person {name:'marko'})-[:KNOWS]->{2}(y) RETURN y.name AS n ORDER BY n`,
      `MATCH DIFFERENT EDGES (m:Person {name:'marko'})-[:KNOWS]->{1}(y) RETURN y.name AS n ORDER BY n`,
      `MATCH (m:Person {name:'marko'})-[:KNOWS]->{1}(y) RETURN y.name AS n ORDER BY n`,
    ]) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
    }
  });

  test('list[i].prop — property access chains off a subscript, byte-identical', () => {
    const base = `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop'`;
    // edges(p)[0].weight, nodes(p)[i].name, and out-of-range → null all
    // agree across engines.
    const [ts, native] = both(
      `${base} RETURN edges(p)[0].weight AS w, nodes(p)[0].name AS a, nodes(p)[1].name AS b, edges(p)[9].weight AS oob`,
    );
    expect(ts).toBe(native);

    // Value check on a single column (robust to object key ordering).
    const [tsW] = both(`${base} RETURN edges(p)[0].weight AS w`);
    expect(tsW).toBe(`[{"w":0.4}]`);

    // Consecutive-index comparison — the per-hop path-predicate motif — agrees.
    const [tsC, natC] = both(`${base} RETURN (edges(p)[0].weight < edges(p)[0].weight) AS lt`);
    expect(tsC).toBe(natC);
  });

  test('named procedure CALL (algorithms) is byte-identical across engines', () => {
    // `node` is a live vertex handle; `node.name` reads its property.
    const [tsD, natD] = both(
      'CALL degree() YIELD node, degree RETURN node.name AS n, degree ORDER BY n',
    );
    expect(tsD).toBe(natD);

    // pagerank scores (f64) through the CALL surface, ordered deterministically.
    const [tsP, natP] = both(
      'CALL pagerank() YIELD node, score RETURN node.name AS n, score ORDER BY score DESC, n',
    );
    expect(tsP).toBe(natP);

    // YIELD aliasing + WITH…WHERE filtering.
    const [tsF, natF] = both(
      'CALL degree() YIELD node AS v, degree AS d WITH v, d WHERE d >= 2 RETURN v.name AS n ORDER BY n',
    );
    expect(tsF).toBe(natF);

    // Returning the whole node hydrates the rich {id,labels,properties} map —
    // byte-identical across engines, exactly like `MATCH (n) RETURN n`.
    const [tsN, natN] = both('CALL degree() YIELD node RETURN node ORDER BY node.name LIMIT 2');
    expect(tsN).toBe(natN);
  });

  test('correlated OPTIONAL MATCH after MATCH (no barrier) is byte-identical', () => {
    // Regression: the native OPTIONAL null-fill used to leak into the next start
    // binding and drop real matches; must match TS for every start vertex.
    const [ts, native] = both(
      `MATCH (p:Person) OPTIONAL MATCH (p)-[:CREATED]->(w) ` +
        `RETURN p.name AS pn, w.name AS wn ORDER BY pn, wn`,
    );
    expect(ts).toBe(native);
  });

  test('inline subquery CALL (correlated lateral join) is byte-identical', () => {
    // Per-person created-count via a correlated subquery.
    const [tsC, natC] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN count(w) AS created } ` +
        `RETURN p.name AS name, created ORDER BY name`,
    );
    expect(tsC).toBe(natC);

    // Row duplication (marko's KNOWS neighbours) via the subquery.
    const [tsD, natD] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend } ` +
        `RETURN friend ORDER BY friend`,
    );
    expect(tsD).toBe(natD);

    // Scope isolation: `()` imports nothing, so the inner MATCH is unbound.
    const [tsS, natS] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL () { MATCH (n) RETURN count(n) AS total } RETURN total`,
    );
    expect(tsS).toBe(natS);

    // Non-agg subquery over MULTIPLE start vertices (native decorrelates this to a
    // flat join; TS runs it correlated) — the outputs must still match exactly.
    const [tsM, natM] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS thing } ` +
        `RETURN p.name AS pn, thing ORDER BY pn, thing`,
    );
    expect(tsM).toBe(natM);

    // AGGREGATING subquery, deliberately NO `ORDER BY`: native decorrelates it to
    // OPTIONAL MATCH + grouped WITH, TS runs it correlated — the row ORDER (not
    // just the set) must still match, proving the grouped first-seen order equals
    // the correlated outer order.
    const [tsA, natA] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN count(w) AS c } ` +
        `RETURN p.name AS pn, c`,
    );
    expect(tsA).toBe(natA);
  });

  test('inline subquery CALL with set operators is byte-identical', () => {
    // UNION (distinct) inside the correlated body: per person, KNOWS-neighbour
    // names ∪ CREATED-thing names. marko → {vadas, josh} ∪ {lop}; others empty.
    const [tsU, natU] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS x ` +
        `UNION MATCH (p)-[:CREATED]->(w) RETURN w.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsU).toBe(natU);
    expect(tsU).not.toBe('[]');

    // UNION ALL keeps duplicates: each KNOWS-neighbour twice.
    const [tsUA, natUA] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS x ` +
        `UNION ALL MATCH (p)-[:KNOWS]->(f) RETURN f.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsUA).toBe(natUA);
    expect(tsUA).not.toBe('[]');

    // EXCEPT where the correlation feeds the RIGHT side: all Software names
    // EXCEPT those p created. marko created lop ⇒ empty; vadas/josh ⇒ {lop}.
    const [tsE, natE] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (s:Software) RETURN s.name AS x ` +
        `EXCEPT MATCH (p)-[:CREATED]->(w) RETURN w.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsE).toBe(natE);
    expect(tsE).not.toBe('[]');

    // INTERSECT: p's created things that are Software. marko ⇒ {lop}; others ∅.
    const [tsI, natI] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS x ` +
        `INTERSECT MATCH (s:Software) RETURN s.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsI).toBe(natI);
    expect(tsI).not.toBe('[]');

    // OPTIONAL + a set-op body that is EMPTY for vadas/josh ⇒ null-filled rows.
    const [tsO, natO] = both(
      `MATCH (p:Person) ` +
        `OPTIONAL CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS x ` +
        `UNION MATCH (p)-[:CREATED]->(w) RETURN w.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsO).toBe(natO);

    // Uncorrelated `CALL () { … UNION … }`: a global union, one outer row.
    const [tsG, natG] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL () { MATCH (n:Person) RETURN n.name AS x ` +
        `UNION MATCH (n:Software) RETURN n.name AS x } ` +
        `RETURN x ORDER BY x`,
    );
    expect(tsG).toBe(natG);
    expect(tsG).not.toBe('[]');

    // Three parts (left-associative): UNION then UNION, correlated.
    const [tsT, natT] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS x ` +
        `UNION MATCH (p)-[:CREATED]->(w) RETURN w.name AS x ` +
        `UNION MATCH (s:Software) RETURN s.name AS x } ` +
        `RETURN p.name AS pn, x ORDER BY pn, x`,
    );
    expect(tsT).toBe(natT);
    expect(tsT).not.toBe('[]');
  });

  test('inline subquery CALL with RETURN * / element columns is byte-identical', () => {
    // RETURN * carries the newly-bound var (f) into the outer scope.
    const [tsS, natS] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN * } ` +
        `RETURN f.name AS fn ORDER BY fn`,
    );
    expect(tsS).toBe(natS);
    expect(tsS).not.toBe('[]');

    // RETURN * carries BOTH the imported var (p) and the new one (f).
    const [tsB, natB] = both(
      `MATCH (p:Person) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN * } ` +
        `RETURN p.name AS pn, f.name AS fn ORDER BY pn, fn`,
    );
    expect(tsB).toBe(natB);
    expect(tsB).not.toBe('[]');

    // OPTIONAL + empty `RETURN *` body: the outer row survives with the imported
    // var intact and the fresh var unbound (→ null on access) — NOT null-filling
    // the imported var. vadas/josh have no KNOWS edge, so their bodies are empty.
    const [tsO, natO] = both(
      `MATCH (p:Person) ` +
        `OPTIONAL CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN * } ` +
        `RETURN p.name AS pn, f.name AS fn ORDER BY pn, fn`,
    );
    expect(tsO).toBe(natO);
    // Every person appears at least once (marko twice, vadas/josh/… null-filled).
    expect(tsO).toContain('"pn":"vadas","fn":null');

    // A bare element column (`RETURN f`) merges the node handle back so `f.name`
    // resolves in the outer query — previously lost to null in the native engine.
    const [tsE, natE] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f } ` +
        `RETURN f.name AS fn ORDER BY fn`,
    );
    expect(tsE).toBe(natE);
    expect(tsE).not.toBe('[]');

    // The carried node re-serializes to the SAME rich {id,labels,properties} map
    // in both engines when returned whole.
    const [tsR, natR] = both(
      `MATCH (p:Person {name: 'marko'}) ` +
        `CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN * } ` +
        `RETURN f ORDER BY f.name`,
    );
    expect(tsR).toBe(natR);
    expect(tsR).toContain('"labels":["Person"]');
  });

  test('FILTER statement (ISO §14.6) is byte-identical', () => {
    // Bare condition (no WHERE) drops rows where the predicate is not TRUE.
    const [tsF, natF] = both(`MATCH (p:Person) FILTER p.age > 28 RETURN p.name AS n ORDER BY n`);
    expect(tsF).toBe(natF);
    expect(tsF).toBe(`[{"n":"josh"},{"n":"marko"}]`);

    // The optional WHERE keyword form is equivalent.
    const [tsW, natW] = both(
      `MATCH (p:Person) FILTER WHERE p.age > 28 RETURN p.name AS n ORDER BY n`,
    );
    expect(tsW).toBe(natW);
    expect(tsW).toBe(tsF);

    // FILTER over a projected working table (after WITH).
    const [tsP, natP] = both(
      `MATCH (p:Person) WITH p.name AS nm, p.age AS a FILTER a >= 29 RETURN nm ORDER BY nm`,
    );
    expect(tsP).toBe(natP);

    // Three-valued: a null-yielding predicate drops the row (not TRUE).
    const [tsN, natN] = both(`MATCH (p:Person) FILTER p.missing > 1 RETURN p.name AS n`);
    expect(tsN).toBe(natN);
    expect(tsN).toBe('[]');
  });

  test('LET statement (ISO §14.7) is byte-identical', () => {
    // Additive binding of a computed value.
    const [tsL, natL] = both(
      `MATCH (p:Person) LET doubled = p.age * 2 RETURN p.name AS n, doubled ORDER BY n`,
    );
    expect(tsL).toBe(natL);
    expect(tsL).toContain('"doubled":58'); // marko 29*2

    // Comma-list, left-to-right: y references x bound in the same LET.
    const [tsS, natS] = both(
      `MATCH (p:Person) LET x = p.age, y = x + 1 RETURN p.name AS n, y ORDER BY n`,
    );
    expect(tsS).toBe(natS);
    expect(tsS).toContain('"y":30'); // marko 29+1

    // A LET var feeds a subsequent FILTER.
    const [tsC, natC] = both(
      `MATCH (p:Person) LET a = p.age FILTER a > 28 RETURN p.name AS n ORDER BY n`,
    );
    expect(tsC).toBe(natC);
    expect(tsC).toBe(`[{"n":"josh"},{"n":"marko"}]`);

    // LET binding a value pulled from a matched neighbour, then projected.
    const [tsE, natE] = both(
      `MATCH (p:Person)-[:KNOWS]->(f) LET fn = f.name RETURN p.name AS pn, fn ORDER BY pn, fn`,
    );
    expect(tsE).toBe(natE);
    expect(tsE).not.toBe('[]');

    // LET binding a string-valued property, then returning it under the new name.
    const [tsG, natG] = both(`MATCH (p:Person {name: 'marko'}) LET who = p.name RETURN who`);
    expect(tsG).toBe(natG);
    expect(tsG).toBe(`[{"who":"marko"}]`);
  });

  test('NEXT statement composition (ISO) is byte-identical', () => {
    // Pipe a statement's RETURN output as the next statement's driving table.
    const [tsF, natF] = both(
      `MATCH (p:Person) RETURN p.name AS n, p.age AS a NEXT FILTER a > 28 RETURN n ORDER BY n`,
    );
    expect(tsF).toBe(natF);
    expect(tsF).toBe(`[{"n":"josh"},{"n":"marko"}]`);

    // An ELEMENT carried across NEXT stays a node handle, so it can be re-matched.
    const [tsE, natE] = both(
      `MATCH (p:Person) RETURN p AS person ` +
        `NEXT MATCH (person)-[:KNOWS]->(f) RETURN person.name AS pn, f.name AS fn ORDER BY pn, fn`,
    );
    expect(tsE).toBe(natE);
    expect(tsE).toBe(`[{"pn":"marko","fn":"josh"},{"pn":"marko","fn":"vadas"}]`);

    // YIELD selects (and can rename) the piped columns.
    const [tsY, natY] = both(
      `MATCH (p:Person) RETURN p.name AS n, p.age AS a NEXT YIELD n AS who RETURN who ORDER BY who`,
    );
    expect(tsY).toBe(natY);
    expect(tsY).toBe(`[{"who":"josh"},{"who":"marko"},{"who":"vadas"}]`);

    // Chained NEXT with LET + FILTER + ORDER BY across the boundaries.
    const [tsC, natC] = both(
      `MATCH (p:Person) RETURN p.age AS a NEXT LET b = a * 2 RETURN b ORDER BY b ` +
        `NEXT FILTER b > 55 RETURN b ORDER BY b`,
    );
    expect(tsC).toBe(natC);
    expect(tsC).toBe(`[{"b":58},{"b":64}]`);

    // Set operators around NEXT are a documented limitation — both engines reject.
    const threw = (run: () => unknown): boolean => {
      try {
        run();

        return false;
      } catch {
        return true;
      }
    };
    const setOpNext = `MATCH (p:Person) RETURN p.name AS n UNION MATCH (s:Software) RETURN s.name AS n NEXT RETURN n`;

    expect(threw(() => tsQuery(tsGraph, setOpNext))).toBe(true);
    expect(threw(() => nativeGraph.query(setOpNext))).toBe(true);
  });

  test('LOCAL TIME temporal type (ISO) is byte-identical', () => {
    // Constructor from a string, incl. fractional seconds; the wire form is the
    // tagged ISO-8601 string.
    const [tsC, natC] = both(`RETURN local_time('13:45:30') AS a, local_time('08:00:00.25') AS b`);
    expect(tsC).toBe(natC);
    expect(tsC).toBe(`[{"a":{"@localtime":"13:45:30"},"b":{"@localtime":"08:00:00.25"}}]`);

    // A non-time string → null (lenient, like the other temporal constructors).
    const [tsN, natN] = both(`RETURN local_time('2020-01-01') AS bad`);
    expect(tsN).toBe(natN);
    expect(tsN).toBe(`[{"bad":null}]`);

    // local_time(datetime) → the time-of-day part.
    const [tsF, natF] = both(`RETURN local_time(local_datetime('2020-06-15T13:45:30')) AS t`);
    expect(tsF).toBe(natF);
    expect(tsF).toBe(`[{"t":{"@localtime":"13:45:30"}}]`);

    // Time ± duration wraps within the 24h day (25h past 01:00 → 02:00).
    const [tsA, natA] = both(
      `RETURN local_time('01:00:00') + duration('PT25H') AS wrap, ` +
        `local_time('05:00:00') - duration('PT2H') AS minus`,
    );
    expect(tsA).toBe(natA);
    expect(tsA).toBe(`[{"wrap":{"@localtime":"02:00:00"},"minus":{"@localtime":"03:00:00"}}]`);

    // Relational comparison + ORDER BY total order over times.
    const [tsO, natO] = both(
      `FOR t IN [local_time('12:00:00'), local_time('06:30:00'), local_time('23:15:00')] ` +
        `RETURN t ORDER BY t`,
    );
    expect(tsO).toBe(natO);
    expect(tsO).toContain(`"06:30:00"`);
    expect(tsO.indexOf('06:30:00')).toBeLessThan(tsO.indexOf('23:15:00'));
  });

  test('ZONED temporal types (ISO) are byte-identical', () => {
    // Offset and `Z` both round-trip byte-for-byte (offset preserved, not normalized).
    const [tsC, natC] = both(
      `RETURN zoned_datetime('2020-01-01T12:00:00+05:00') AS a, ` +
        `zoned_datetime('2020-06-15T08:30:00.25Z') AS b, ` +
        `zoned_time('12:20:02+08:00') AS c`,
    );
    expect(tsC).toBe(natC);
    expect(tsC).toBe(
      `[{"a":{"@zoned_datetime":"2020-01-01T12:00:00+05:00"},` +
        `"b":{"@zoned_datetime":"2020-06-15T08:30:00.25Z"},` +
        `"c":{"@zoned_time":"12:20:02+08:00"}}]`,
    );

    // A datetime string with no offset is not a zoned value → null.
    const [tsN, natN] = both(`RETURN zoned_datetime('2020-01-01T12:00:00') AS bad`);
    expect(tsN).toBe(natN);
    expect(tsN).toBe(`[{"bad":null}]`);

    // Ordering/relational is by UTC instant: 09:00Z is before 12:00Z; a later
    // instant sorts after regardless of wall-clock/offset.
    const [tsL, natL] = both(
      `RETURN zoned_datetime('2020-01-01T12:00:00Z') < zoned_datetime('2020-01-01T12:00:01Z') AS lt`,
    );
    expect(tsL).toBe(natL);
    expect(tsL).toBe(`[{"lt":true}]`);

    // Zoned + duration applies in the value's own zone (crossing local midnight)
    // and keeps the offset.
    const [tsA, natA] = both(
      `RETURN zoned_datetime('2020-06-15T23:00:00+02:00') + duration('PT3H') AS plus`,
    );
    expect(tsA).toBe(natA);
    expect(tsA).toBe(`[{"plus":{"@zoned_datetime":"2020-06-16T02:00:00+02:00"}}]`);

    // ORDER BY sorts by the absolute instant.
    const [tsO, natO] = both(
      `FOR t IN [zoned_datetime('2020-01-01T12:00:00Z'), zoned_datetime('2020-01-01T09:00:00Z')] ` +
        `RETURN t ORDER BY t`,
    );
    expect(tsO).toBe(natO);
    expect(tsO.indexOf('09:00:00')).toBeLessThan(tsO.indexOf('12:00:00'));
  });

  test('ISO path functions on a bound path are byte-identical', () => {
    const q =
      `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop' ` +
      `RETURN path_length(p) AS len, length(p) AS len2, ` +
      `nodes(p) AS ns, edges(p) AS es, elements(p) AS el`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    // Length is the hop count; nodes/edges/elements are rich element lists.
    expect(ts).toBe(
      `[{"len":1,"len2":1,` +
        `"ns":[` +
        `{"id":"1","labels":["Person"],"properties":{"age":29,"name":"marko"}},` +
        `{"id":"3","labels":["Software"],"properties":{"lang":"java","name":"lop"}}],` +
        `"es":[{"id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"since":2009,"weight":0.4}}],` +
        `"el":[` +
        `{"id":"1","labels":["Person"],"properties":{"age":29,"name":"marko"}},` +
        `{"id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"since":2009,"weight":0.4}},` +
        `{"id":"3","labels":["Software"],"properties":{"lang":"java","name":"lop"}}]}]`,
    );
  });

  // --- var-length {1,2} count: native uses a degree-product fast path, TS
  // enumerates trails. They must agree, including with parallel edges + self-loops
  // (which the degree product would double-count without the correction). -------
  test('var-length {1,2} count matches trail enumeration (TS vs native)', () => {
    const VARLEN_NDJSON = [
      '{"type":"node","id":"a","labels":["Person","VIP"],"properties":{}}',
      '{"type":"node","id":"b","labels":["Person"],"properties":{}}',
      '{"type":"node","id":"c","labels":["Person"],"properties":{}}',
      '{"type":"edge","id":"e0","from":"a","to":"b","labels":["KNOWS"],"properties":{}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["KNOWS"],"properties":{}}',
      '{"type":"edge","id":"e2","from":"b","to":"c","labels":["KNOWS"],"properties":{}}',
      '{"type":"edge","id":"e3","from":"b","to":"b","labels":["KNOWS"],"properties":{}}',
      '{"type":"edge","id":"e4","from":"a","to":"a","labels":["KNOWS"],"properties":{}}',
      '{"type":"edge","id":"e5","from":"c","to":"a","labels":["KNOWS"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, VARLEN_NDJSON);
    const ts = tsDeserialize(VARLEN_NDJSON, 'ndjson', new Graph());

    for (const q of [
      `MATCH (x)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c`,
      `MATCH (x:VIP)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c`,
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // ISO quantified parenthesized subpath `((x)-[e]->(y) WHERE …){n,m}` — the
  // per-repetition predicate names the hop's source/edge/target. Byte-identical.
  test('quantified parenthesized subpath: per-hop node + cross-element predicate (TS vs native)', () => {
    const BAL_NDJSON = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a","bal":100.0}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b","bal":200.0}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c","bal":5.0}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d","bal":200.0}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":30.0}}',
      '{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":20.0}}',
      '{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, BAL_NDJSON);
    const ts = tsDeserialize(BAL_NDJSON, 'ndjson', new Graph());

    for (const q of [
      // Per-hop source/target/cross-element predicates; `(t)` is the endpoint.
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt <= x.bal){1,3} (t) RETURN t.id AS id ORDER BY t.id",
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE y.bal >= 100){1,3} (t) RETURN t.id AS id ORDER BY t.id",
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 1){1,3} (t) RETURN t.id AS id ORDER BY t.id",
      // GROUP variables: x/e/y exposed as lists, endpoint + list-index + size.
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) RETURN t.id AS tid, size(e) AS ne, size(x) AS nx, size(y) AS ny, x[0].id AS x0, y[1].id AS y1, e[0].amt AS e0",
      // Dual context: per-hop WHERE reads scalars, size(e) reads the list.
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 15){1,3} (t) RETURN t.id AS tid, size(e) AS ne ORDER BY t.id",
      // Zero-hop inclusion + anonymous endpoint.
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){0,1} (t) RETURN t.id AS tid, size(e) AS ne ORDER BY t.id, ne",
      "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} RETURN size(e) AS ne",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }

    // A labelled inner node is rejected by BOTH engines (Phase 1 restriction).
    const throws = (fn: () => void): boolean => {
      try {
        fn();

        return false;
      } catch {
        return true;
      }
    };
    const bad = 'MATCH (s:N) ((x:N)-[e]->(y)){1,2} RETURN 1 AS c';
    expect(throws(() => nat.query(bad))).toBe(true);
    expect(throws(() => tsQuery(ts, bad))).toBe(true);
  });

  // ISO MULTI-element repetition unit `((x)-[e1]->(m)-[e2]->(y)){n,m}`: each
  // repetition advances k hops. Intermediate node + every edge are group vars.
  // Byte-identical across engines.
  test('multi-element repetition unit + group variables (TS vs native)', () => {
    const CHAIN = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
      '{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":10.0}}',
      '{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":10.0}}',
      '{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}',
      '{"type":"edge","id":"e4","from":"d","to":"e","labels":["R"],"properties":{"amt":10.0}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, CHAIN);
    const ts = tsDeserialize(CHAIN, 'ndjson', new Graph());

    for (const q of [
      // A 2-hop unit lands only on even hop counts: {1}→c, {1,2}→{c,e}.
      "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){1} (t) RETURN t.id AS id ORDER BY t.id",
      "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){1,2} (t) RETURN t.id AS id ORDER BY t.id",
      // Group vars: intermediate `m` and BOTH edges are lists sized by repetition count.
      "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){2} (t) RETURN t.id AS tid, size(e1) AS n1, size(e2) AS n2, size(m) AS nm, x[0].id AS x0, x[1].id AS x1, m[0].id AS m0, m[1].id AS m1, y[1].id AS y1",
      // Per-unit WHERE spanning BOTH hops (interior node `m` shared).
      "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt <= e1.amt){1,2} (t) RETURN t.id AS id ORDER BY t.id",
      "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt < e1.amt){1,2} (t) RETURN t.id AS id ORDER BY t.id",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // ANY/ALL SHORTEST now honour a per-hop edge predicate — the BFS expands only over
  // passing edges (shortest path in the filtered subgraph). Byte-identical both engines.
  test('shortest selector honours a per-hop edge predicate (TS vs native)', () => {
    const W = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"w":1.0}}',
      '{"type":"edge","id":"e2","from":"a","to":"c","labels":["R"],"properties":{"w":10.0}}',
      '{"type":"edge","id":"e3","from":"c","to":"b","labels":["R"],"properties":{"w":10.0}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, W);
    const ts = tsDeserialize(W, 'ndjson', new Graph());

    for (const q of [
      // Unfiltered: 1 hop. Filtered (w>5): the direct edge is blocked → 2 hops (a→c→b).
      "MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R]->*(b:N {id:'b'}) RETURN path_length(p) AS len",
      "MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p) AS len",
      "MATCH p = ALL SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p) AS len",
      // A predicate blocking every seed-edge → empty.
      "MATCH ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 100]->*(b:N {id:'b'}) RETURN b.id AS id",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // A pattern may BEGIN with a quantified subpath (no anchor node), and a path variable
  // may bind the whole repeated walk — ISO, byte-identical both engines.
  test('unanchored quantified subpath + path variable (TS vs native)', () => {
    const CHAIN = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, CHAIN);
    const ts = tsDeserialize(CHAIN, 'ndjson', new Graph());

    for (const q of [
      // Path variable over a leading quantified subpath (no anchor).
      'MATCH p = ((x)-[e:R]->(y)){2} (t) RETURN t.id AS tid, path_length(p) AS len ORDER BY tid',
      // Unanchored, no path variable.
      'MATCH ((x)-[:R]->(y)){1,2} (t) RETURN t.id AS id ORDER BY id',
      // nodes(p) over the repeated walk.
      "MATCH p = ((x)-[e:R]->(y)){2} (t) WHERE t.id = 'c' RETURN size(nodes(p)) AS n",
      // The bare grouping (no quantifier) still parses as a WHERE-scoped grouping.
      'MATCH ((a)-[:R]->(b) WHERE a.id < b.id) RETURN a.id AS id ORDER BY id',
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // NESTED quantifiers `( … {a,b} … ){n,m}` — the outer repetition repeats an inner
  // variable-length sub-walk (the pushdown matcher's `Sub` path). Byte-identical.
  test('nested quantifiers (TS vs native)', () => {
    const CHAIN = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
      '{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}',
      '{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}',
      '{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{}}',
      '{"type":"edge","from":"c","to":"d","labels":["R"],"properties":{}}',
      '{"type":"edge","from":"d","to":"e","labels":["R"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, CHAIN);
    const ts = tsDeserialize(CHAIN, 'ndjson', new Graph());

    for (const q of [
      "MATCH (s:N {id:'a'}) ( ()-[:R]->{1,3}() ){1} (t) RETURN t.id AS id ORDER BY id",
      "MATCH (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){2} (t) RETURN t.id AS id ORDER BY id",
      "MATCH (s:N {id:'a'}) ( ()-[:R]->()-[:R]->{1,2}() ){1} (t) RETURN t.id AS id ORDER BY id",
      // count(*) form (per-trail multiplicity).
      "MATCH (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){1,2} (t) RETURN count(*) AS c",
      // Nested under a path mode.
      "MATCH ACYCLIC (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){2} (t) RETURN t.id AS id ORDER BY id",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }

    // The v1 rejections are identical on both engines.
    const code = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: unknown }).code;
      }

      return 'ok';
    };
    const bad = 'MATCH (s) ( (x)-[:R]->{1,2}(m) ){2} (t) RETURN t';
    expect(code(() => nat.query(bad))).toBe(code(() => tsQuery(ts, bad)));
  });

  // A per-hop EDGE predicate on a nested inner hop filters every edge of the inner walk
  // (the tractable slice of "WHERE inside a nested quantifier"). Byte-identical.
  test('nested quantifier per-hop edge predicate (TS vs native)', () => {
    const W = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
      '{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{"amt":10.0}}',
      '{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{"amt":1.0}}',
      '{"type":"edge","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, W);
    const ts = tsDeserialize(W, 'ndjson', new Graph());

    for (const q of [
      "MATCH (s:N {id:'a'}) ( ()-[e:R WHERE e.amt >= 5]->{1,2}() ){1,2} (t) RETURN t.id AS id ORDER BY id",
      "MATCH (s:N {id:'a'}) ( ()-[:R {amt:10.0}]->{1,3}() ){1} (t) RETURN t.id AS id ORDER BY id",
      "MATCH (s:N {id:'a'}) ( ()-[e:R WHERE e.amt >= 5]->{1,3}() ){1,3} (t) RETURN count(*) AS c",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // A five-node chain a→b→c→d→e for the nested group-variable exposure tests below.
  const NCHAIN = [
    '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
    '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
    '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
    '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
    '{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}',
    '{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}',
    '{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{}}',
    '{"type":"edge","from":"c","to":"d","labels":["R"],"properties":{}}',
    '{"type":"edge","from":"d","to":"e","labels":["R"],"properties":{}}',
  ].join('\n');

  // #1 — a nested quantifier's NODE group variables exposed as FLAT lists (the source `x`
  // per outer rep, and a nested hop's landing `-[]->{a,b}(y)`). Byte-identical.
  test('nested quantifier outer group variables (TS vs native)', () => {
    const nat = graphFromNdjson(backend, NCHAIN);
    const ts = tsDeserialize(NCHAIN, 'ndjson', new Graph());

    for (const q of [
      "MATCH (s:N {id:'a'}) ( (x)-[:R]->{2,2}(y) ){2} (t) RETURN t.id AS tid, size(x) AS nx, x[0].id AS x0, x[1].id AS x1, y[0].id AS y0, y[1].id AS y1",
      "MATCH (s:N {id:'a'}) ( (x)-[:R]->{1,2}(y) ){1} (t) RETURN t.id AS tid, size(x) AS nx, x[0].id AS x0, y[0].id AS y0 ORDER BY tid",
      "MATCH (s:N {id:'a'}) ( (x)-[:R]->{1,2}(y) ){2} (t) RETURN t.id AS id ORDER BY id",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // #2 — a nested PARENTHESIZED subpath exposes its inner variables as LIST-OF-LISTS
  // (one list level per enclosing quantifier). Byte-identical.
  test('nested parenthesized subpath list-of-lists (TS vs native)', () => {
    const nat = graphFromNdjson(backend, NCHAIN);
    const ts = tsDeserialize(NCHAIN, 'ndjson', new Graph());

    for (const q of [
      "MATCH (s:N {id:'a'}) ( ((x)-[:R]->(y)){2,2} ){2} (t) RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, x[0][0].id AS a00, x[0][1].id AS a01, x[1][0].id AS a10, y[0][1].id AS y01, y[1][1].id AS y11",
      "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,2} ){1} (t) RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, size(e[0]) AS ne0 ORDER BY tid",
      "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,1} ){2} (t) RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, x[0][0].id AS a00, x[1][0].id AS a10, y[1][0].id AS y10",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // A subpath-level WHERE on a NESTED quantifier: a per-OUTER-rep predicate with the inner
  // variables bound as LISTS (`size(e)`, `x[0]`). Byte-identical both engines.
  test('nested quantifier per-rep WHERE over grouped vars (TS vs native)', () => {
    const nat = graphFromNdjson(backend, NCHAIN);
    const ts = tsDeserialize(NCHAIN, 'ndjson', new Graph());

    for (const q of [
      // Abbreviated inner: `size(e)` constrains each outer rep's inner-walk length.
      "MATCH (s:N {id:'a'}) ( ()-[e:R]->{1,2}() WHERE size(e) = 2 ){2} (t) RETURN t.id AS id ORDER BY id",
      "MATCH (s:N {id:'a'}) ( ()-[e:R]->{1,2}() WHERE size(e) = 1 ){2} (t) RETURN t.id AS id ORDER BY id",
      // Nested parenthesized subpath: same per-rep constraint, list-of-lists still exposed.
      "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,2} WHERE size(e) = 2 ){2} (t) RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0",
      // Per-rep WHERE over list ELEMENTS.
      "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){2,2} WHERE x[0].id <> y[1].id ){2} (t) RETURN t.id AS id ORDER BY id",
      // A per-rep WHERE that prunes EVERYTHING → empty, both engines.
      "MATCH (s:N {id:'a'}) ( ()-[e:R]->{1,2}() WHERE size(e) = 5 ){2} (t) RETURN t.id AS id",
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // The fused matcher marks PER HOP, so ACYCLIC/SIMPLE forbid a multi-element unit from
  // repeating a vertex INTERNALLY (a self-loop `s→p, p→p` revisits p within one unit).
  // Both engines must agree: TRAIL keeps it (distinct edges), ACYCLIC/SIMPLE reject it.
  test('multi-element unit: per-hop vertex marking, byte-identical (TS vs native)', () => {
    const SELF = [
      '{"type":"node","id":"s","labels":["N"],"properties":{"id":"s"}}',
      '{"type":"node","id":"p","labels":["N"],"properties":{"id":"p"}}',
      '{"type":"edge","id":"e1","from":"s","to":"p","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e2","from":"p","to":"p","labels":["R"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, SELF);
    const ts = tsDeserialize(SELF, 'ndjson', new Graph());

    for (const mode of ['', 'ACYCLIC', 'SIMPLE']) {
      const q = `MATCH ${mode} (s:N {id:'s'}) ((x)-[:R]->(m)-[:R]->(y)){1} (t) RETURN t.id AS id`;
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // A dense multi-element unit fans out d^k from ONE vertex — both engines must FAULT
  // with the same code (E_RESOURCE_EXHAUSTED) rather than one OOMing. `{1}` (a single
  // repetition) proves the guard fires inside a single `expand_unit`, not just across
  // repetitions.
  test('dense multi-element unit expansion faults (not OOM), byte-identical (TS vs native)', () => {
    const lines: string[] = [];
    const n = 64;

    for (let i = 0; i < n; i += 1) {
      lines.push(`{"type":"node","id":"${i}","labels":["N"],"properties":{}}`);
    }

    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i !== j) {
          lines.push(`{"type":"edge","from":"${i}","to":"${j}","labels":["R"],"properties":{}}`);
        }
      }
    }

    const K = lines.join('\n');
    const nat = graphFromNdjson(backend, K);
    const ts = tsDeserialize(K, 'ndjson', new Graph());
    const q =
      'MATCH (s:N) ((a)-[:R]->(b)-[:R]->(c)-[:R]->(d)-[:R]->(e)){1} (t) RETURN count(*) AS c';
    const code = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: unknown }).code;
      }

      return 'ok';
    };
    const natCode = code(() => nat.query(q));
    expect(natCode).toBe('E_RESOURCE_EXHAUSTED');
    expect(code(() => tsQuery(ts, q))).toBe(natCode);
    // The TS reference engine must grind the full TRAIL_BUDGET (~1M steps) before it
    // faults — inherently a few seconds, and 2–3× slower on a CI runner — so this
    // deliberate stress test needs more than bun's 5s default per-test timeout.
  }, 30_000);

  // ANTI-DRIFT: the abbreviated `-[]->{n,m}` form (each engine's single-edge
  // fast-path) and an equivalent single-edge parenthesized subpath (the general
  // unit matcher) return IDENTICAL endpoints — across every path mode, on BOTH
  // engines. Pins the two matchers together so neither can silently diverge.
  test('abbreviated == single-edge subpath at k=1, every mode (TS vs native)', () => {
    const TRI = [
      '{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}',
      '{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}',
      '{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}',
      '{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}',
      '{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e3","from":"c","to":"a","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"e4","from":"a","to":"d","labels":["R"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, TRI);
    const ts = tsDeserialize(TRI, 'ndjson', new Graph());

    for (const mode of ['WALK', 'TRAIL', 'SIMPLE', 'ACYCLIC']) {
      for (const quant of ['{1,3}', '{0,2}', '{2}', '{1,4}']) {
        const abbrev = `MATCH ${mode} (s:N {id:'a'})-[:R]->${quant}(x) RETURN x.id AS id ORDER BY x.id`;
        const subpath = `MATCH ${mode} (s:N {id:'a'}) ((y)-[:R]->(z))${quant} (x) RETURN x.id AS id ORDER BY x.id`;
        const key = `${mode} ${quant}`;
        // Both engines agree abbreviated == subpath, AND the two engines agree with
        // each other (transitively pinning all four result sets equal).
        expect(JSON.stringify(nat.query(abbrev)), `native ${key}`).toBe(
          JSON.stringify(nat.query(subpath)),
        );
        expect(JSON.stringify(tsQuery(ts, abbrev)), `ts ${key}`).toBe(
          JSON.stringify(tsQuery(ts, subpath)),
        );
        expect(JSON.stringify(nat.query(abbrev)), `cross ${key}`).toBe(
          JSON.stringify(tsQuery(ts, abbrev)),
        );
      }
    }
  });

  // --- string `id` as element identity: `INSERT (:P {id: 'x'})` makes 'x' the
  // element id (so element_id === n.id and it round-trips), a numeric id stays an
  // ordinary property, dup/SET-id are rejected. Must be byte-identical. ---------
  test('string id property is the element identity (TS vs native)', () => {
    const nat = graphFromNdjson(backend, '');
    const ts = tsDeserialize('', 'ndjson', new Graph());

    for (const q of [
      "INSERT (:P {id: 'alice', name: 'A'})",
      'INSERT (:Q {id: 7})', // numeric → ordinary property
    ]) {
      nat.query(q);
      tsQuery(ts, q);
    }

    // element_id === the domain id, on both engines.
    for (const q of [
      "MATCH (n:P {id: 'alice'}) RETURN element_id(n) AS e, n.id AS p",
      'MATCH (n:Q {id: 7}) RETURN n.id AS i',
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }

    // Both reject a duplicate string id and a SET on the string-identity id, and
    // both allow SET on the numeric id — same coded outcome either side.
    const code = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: unknown }).code;
      }

      return 'ok';
    };
    expect(code(() => nat.query("INSERT (:P {id: 'alice'})"))).toBe(
      code(() => tsQuery(ts, "INSERT (:P {id: 'alice'})")),
    );
    expect(code(() => nat.query("MATCH (n:P {id: 'alice'}) SET n.id = 'bob'"))).toBe(
      code(() => tsQuery(ts, "MATCH (n:P {id: 'alice'}) SET n.id = 'bob'")),
    );
    expect(code(() => nat.query('MATCH (n:Q {id: 7}) SET n.id = 8'))).toBe('ok');
    expect(code(() => tsQuery(ts, 'MATCH (n:Q {id: 7}) SET n.id = 8'))).toBe('ok');
  });

  // --- edges have unique ids too: a string edge `id` is its identity (element_id
  // === r.id, round-trips), unique, SET-rejected; numeric stays an ordinary prop.
  // Must be byte-identical. --------------------------------------------------
  test('string id property is the edge identity (TS vs native)', () => {
    const seed = [
      "INSERT (:P {id: 'a'}), (:P {id: 'b'}), (:P {id: 'c'})",
      "MATCH (a:P {id: 'a'}), (b:P {id: 'b'}) INSERT (a)-[:R {id: 'e1', w: 5}]->(b)",
    ];
    const nat = graphFromNdjson(backend, '');
    const ts = tsDeserialize('', 'ndjson', new Graph());

    for (const q of seed) {
      nat.query(q);
      tsQuery(ts, q);
    }

    // element_id(r) === r.id, identical either side.
    const q = 'MATCH ()-[r:R]->() RETURN element_id(r) AS e, r.id AS p';
    expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));

    const code = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: unknown }).code;
      }

      return 'ok';
    };
    // dup edge id, SET on identity edge id → same coded rejection both engines.
    const dupQ = "MATCH (a:P {id: 'a'}), (c:P {id: 'c'}) INSERT (a)-[:R {id: 'e1'}]->(c)";
    expect(code(() => nat.query(dupQ))).toBe(code(() => tsQuery(ts, dupQ)));
    const setQ = "MATCH ()-[r:R {id: 'e1'}]->() SET r.id = 'e2'";
    expect(code(() => nat.query(setQ))).toBe(code(() => tsQuery(ts, setQ)));
  });

  // --- fixed-length multi-hop with a per-hop WHERE + LIMIT: native routes this to
  // the scalar depth-first driver (filters during traversal, stops at the LIMIT)
  // instead of the breadth-first vectorized path (which materializes the whole
  // cross-product of partial matches, and on a dense graph OOMs the host). TS has
  // always streamed it. They must return the same rows. Regression: guards against
  // native OOM-killing the host on exactly this dense-graph shape. -------------
  test('multi-hop with per-hop WHERE + LIMIT agrees (TS vs native)', () => {
    const CHAIN_NDJSON = [
      '{"type":"node","id":"a","labels":["A"],"properties":{"nm":"a"}}',
      '{"type":"node","id":"b","labels":["A"],"properties":{"nm":"b"}}',
      '{"type":"node","id":"c","labels":["A"],"properties":{"nm":"c"}}',
      '{"type":"node","id":"d","labels":["A"],"properties":{"nm":"d"}}',
      '{"type":"node","id":"e","labels":["A"],"properties":{"nm":"e"}}',
      '{"type":"node","id":"f","labels":["A"],"properties":{"nm":"f"}}',
      '{"type":"edge","from":"a","to":"b","labels":["E"],"properties":{"amt":1}}',
      '{"type":"edge","from":"b","to":"d","labels":["E"],"properties":{"amt":3}}',
      '{"type":"edge","from":"d","to":"f","labels":["E"],"properties":{"amt":6}}',
      '{"type":"edge","from":"a","to":"c","labels":["E"],"properties":{"amt":2}}',
      '{"type":"edge","from":"c","to":"e","labels":["E"],"properties":{"amt":1}}',
      '{"type":"edge","from":"e","to":"f","labels":["E"],"properties":{"amt":9}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, CHAIN_NDJSON);
    const ts = tsDeserialize(CHAIN_NDJSON, 'ndjson', new Graph());
    const q =
      'MATCH (v0:A)-[e1:E]->(v1:A)-[e2:E]->(v2:A)-[e3:E]->(v3:A) ' +
      'WHERE e1.amt < e2.amt AND e2.amt < e3.amt ' +
      'RETURN v0.nm AS s, v3.nm AS t ORDER BY s, t LIMIT 100';
    expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
  });

  // --- correlated EXISTS count: native uses a reverse semi-join (seed the selective
  // inner endpoint), TS tests every outer row. They must agree. Software (1 vertex)
  // is more selective than Person, so the fast path fires. ---------------------
  test('EXISTS / NOT EXISTS count matches per-row evaluation (TS vs native)', () => {
    for (const q of [
      `MATCH (a:Person) WHERE EXISTS { (a)-[:CREATED]->(:Software) } RETURN count(*) AS c`,
      `MATCH (a:Person) WHERE NOT EXISTS { (a)-[:CREATED]->(:Software) } RETURN count(*) AS c`,
    ]) {
      expect(JSON.stringify(nativeGraph.query(q)), q).toBe(JSON.stringify(tsQuery(tsGraph, q)));
    }
  });

  // --- count(DISTINCT endpoint): native marks a reachable frontier, TS enumerates
  // then dedups. Same reachable-set size. --------------------------------------
  test('count(DISTINCT endpoint) matches enumerated dedup (TS vs native)', () => {
    for (const q of [
      `MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b) AS c`,
      `MATCH (a:Person)-[:CREATED]->(b:Software) RETURN count(DISTINCT b) AS c`,
      `MATCH (a:Person)-[:KNOWS]->()-[:CREATED]->(c) RETURN count(DISTINCT c) AS c`,
    ]) {
      expect(JSON.stringify(nativeGraph.query(q)), q).toBe(JSON.stringify(tsQuery(tsGraph, q)));
    }
  });

  // --- percentile_cont / percentile_disc: ISO ordered-set aggregates, newly
  // implemented in both engines — must compute byte-identically. ---------------
  test('percentile_cont / percentile_disc agree (TS vs native)', () => {
    for (const q of [
      `MATCH (n:Person) RETURN percentile_cont(n.age, 0.5) AS x`,
      `MATCH (n:Person) RETURN percentile_disc(n.age, 0.5) AS x`,
      `MATCH (n:Person) RETURN percentile_cont(n.age, 0.9) AS x, percentile_disc(n.age, 0.9) AS y`,
      `MATCH (n:Person) RETURN percentile_cont(n.age, 0.0) AS lo, percentile_cont(n.age, 1.0) AS hi`,
    ]) {
      expect(JSON.stringify(nativeGraph.query(q)), q).toBe(JSON.stringify(tsQuery(tsGraph, q)));
    }
  });

  // --- COUNT { } degree: native takes an adjacency-count fast path, TS enumerates
  // the sub-pattern. Same per-row count. ---------------------------------------
  test('COUNT { } single-segment degree matches enumeration (TS vs native)', () => {
    for (const q of [
      `MATCH (a:Person) RETURN a.name AS name, COUNT { (a)-[:KNOWS]->() } AS deg ORDER BY name`,
      `MATCH (a:Person) RETURN a.name AS name, COUNT { (a)-[:CREATED]->(:Software) } AS d ORDER BY name`,
      `MATCH (a:Person) RETURN a.name AS name, COUNT { (a)<-[:KNOWS]-() } AS indeg ORDER BY name`,
      // reverse degree — the correlated node is the endpoint (native anchors there)
      `MATCH (s:Software) RETURN s.name AS name, COUNT { (:Person)-[:CREATED]->(s) } AS pop ORDER BY name`,
      `MATCH (a:Person) RETURN a.name AS name, COUNT { (b)-[:KNOWS]->(a) } AS indeg ORDER BY name`,
    ]) {
      expect(JSON.stringify(nativeGraph.query(q)), q).toBe(JSON.stringify(tsQuery(tsGraph, q)));
    }
  });

  // --- unbounded var-length + DISTINCT: native BFSes the reachable set, TS
  // enumerates trails then dedups. On a small graph (enumeration completes) they
  // must agree — including ->+ vs ->* seed inclusion and a cycle. -------------
  test('unbounded var-length DISTINCT matches trail enumeration (TS vs native)', () => {
    const REACH_NDJSON = [
      '{"type":"node","id":"s0","labels":["Node"],"properties":{"name":"s0"}}',
      '{"type":"node","id":"a1","labels":["Node"],"properties":{"name":"a1"}}',
      '{"type":"node","id":"a2","labels":["Node"],"properties":{"name":"a2"}}',
      '{"type":"node","id":"t3","labels":["Node","Target"],"properties":{"name":"t3"}}',
      '{"type":"edge","id":"r1","from":"s0","to":"a1","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"r2","from":"a1","to":"a2","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"r3","from":"a2","to":"a1","labels":["R"],"properties":{}}',
      '{"type":"edge","id":"r4","from":"a2","to":"t3","labels":["R"],"properties":{}}',
    ].join('\n');
    const nat = graphFromNdjson(backend, REACH_NDJSON);
    const ts = tsDeserialize(REACH_NDJSON, 'ndjson', new Graph());

    // DISTINCT rows with no ORDER BY are a set — the native BFS and TS enumeration
    // legitimately differ in row order, so compare the sorted name sets.
    const names = (rowset: Array<{ n: string }>): string[] => rowset.map((r) => r.n).sort();

    for (const q of [
      `MATCH (a:Node {name: 's0'})-[:R]->+(b) RETURN DISTINCT b.name AS n`,
      `MATCH (a:Node {name: 's0'})-[:R]->*(b) RETURN DISTINCT b.name AS n`,
      `MATCH (a:Node {name: 's0'})-[:R]->+(b:Target) RETURN DISTINCT b.name AS n`,
    ]) {
      expect(names(nat.query(q)), q).toEqual(names(tsQuery(ts, q)));
    }

    // count(DISTINCT) is a single deterministic value.
    const cq = `MATCH (a:Node {name: 's0'})-[:R]->+(b) RETURN count(DISTINCT b) AS c`;
    expect(JSON.stringify(nat.query(cq)), cq).toBe(JSON.stringify(tsQuery(ts, cq)));

    // EXISTS { reachability }: both engines BFS (was: trail enumeration, faulted on
    // an unreachable target). Reachable t3, unreachable 'nope', endpoint WHERE.
    for (const q of [
      `MATCH (a:Node {name: 's0'}) RETURN EXISTS { (a)-[:R]->+(b:Target) } AS r`,
      `MATCH (a:Node {name: 's0'}) RETURN EXISTS { (a)-[:R]->+(b:Node {name: 'nope'}) } AS r`,
      `MATCH (a:Node {name: 's0'}) RETURN EXISTS { (a)-[:R]->+(b) WHERE b.name = 'a2' } AS r`,
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // --- FOR (ISO list unwind / UNWIND) ---------------------------------------

  test('FOR unwinds a literal list identically', () => {
    const [ts, native] = both(`FOR x IN [1, 2, 3] RETURN x`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":1},{"x":2},{"x":3}]`);
  });

  test('FOR WITH ORDINALITY (1-based) is identical', () => {
    const [ts, native] = both(`FOR x IN ['a', 'b'] WITH ORDINALITY i RETURN x, i`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":"a","i":1},{"x":"b","i":2}]`);
  });

  test('FOR WITH OFFSET (0-based) is identical', () => {
    const [ts, native] = both(`FOR x IN ['a', 'b'] WITH OFFSET i RETURN x, i`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":"a","i":0},{"x":"b","i":1}]`);
  });

  test('FOR over null yields no rows on both engines', () => {
    const [ts, native] = both(`FOR x IN null RETURN x`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[]`);
  });

  test('FOR over a scalar unwinds as a singleton, identically', () => {
    const [ts, native] = both(`FOR x IN 5 RETURN x`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":5}]`);
  });

  test('FOR multiplies a prior MATCH row identically', () => {
    const [ts, native] = both(
      `MATCH (p:Person {name: 'marko'}) FOR t IN ['x', 'y'] RETURN p.name, t`,
    );
    expect(ts).toBe(native);
  });

  test('the FOR list can reference a bound var, identically', () => {
    const [ts, native] = both(`MATCH (p:Person {name: 'marko'}) FOR x IN [p.name, p.age] RETURN x`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":"marko"},{"x":29}]`);
  });

  test('the `FOR` clause drives a batch OPTIONAL MATCH (allow + deny) byte-identically', () => {
    // One row per requested name; josh exists (age 32), nobody does not (null).
    const [ts, native] = both(
      `FOR name IN ['josh', 'nobody'] OPTIONAL MATCH (p:Person {name: name}) RETURN name, p.age`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"josh","p.age":32},{"name":"nobody","p.age":null}]`);
  });

  // --- temporal literals + comparison (Phase 1) -----------------------------

  test('a DATE literal serializes to the tagged form byte-identically', () => {
    const [ts, native] = both(`RETURN DATE '2020-02-29' AS d`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"d":{"@date":"2020-02-29"}}]`);
  });

  test('a DURATION literal normalizes (years->months) identically', () => {
    const [ts, native] = both(`RETURN DURATION 'P1Y2M3DT4H5M6S' AS d`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"d":{"@duration":"P14M3DT14706S"}}]`);
  });

  test('temporal comparison (incl. cross-kind UNKNOWN) is byte-identical', () => {
    const cases: [string, string][] = [
      [`RETURN DATE '2020-01-01' < DATE '2020-06-01' AS x`, `[{"x":true}]`],
      [`RETURN DATE '2020-06-01' < DATE '2020-01-01' AS x`, `[{"x":false}]`],
      [`RETURN DATE '2020-01-01' = DATE '2020-01-01' AS x`, `[{"x":true}]`],
      [
        `RETURN TIMESTAMP '2021-06-15T08:30:00.5' >= DATETIME '2021-06-15T08:30:00' AS x`,
        `[{"x":true}]`,
      ],
      [`RETURN DATE '2020-01-01' < DATETIME '2020-01-01T00:00:00' AS x`, `[{"x":null}]`],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }
  });

  test('ORDER BY over temporal literals sorts chronologically, byte-identical', () => {
    const [ts, native] = both(
      `FOR d IN [DATE '2020-06-01', DATE '2020-01-01', DATE '2020-03-01'] RETURN d ORDER BY d`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(
      `[{"d":{"@date":"2020-01-01"}},{"d":{"@date":"2020-03-01"}},{"d":{"@date":"2020-06-01"}}]`,
    );
  });

  test('temporal constructor functions are byte-identical', () => {
    const cases: [string, string][] = [
      [`RETURN date('2020-02-29') AS d`, `[{"d":{"@date":"2020-02-29"}}]`],
      [
        `RETURN local_datetime('2021-06-15T08:30:00') AS d`,
        `[{"d":{"@datetime":"2021-06-15T08:30:00"}}]`,
      ],
      [`RETURN duration('P1Y2M') AS d`, `[{"d":{"@duration":"P14M"}}]`],
      [`RETURN date(local_datetime('2020-02-29T13:45:00')) AS d`, `[{"d":{"@date":"2020-02-29"}}]`],
      [`RETURN date('nope') AS d`, `[{"d":null}]`],
      // The point of the function form: convert a runtime string into a temporal.
      [`FOR s IN ['2019-03-15'] RETURN date(s) < DATE '2020-01-01' AS x`, `[{"x":true}]`],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }
  });

  test('duration_between returns the exact span, byte-identical', () => {
    const cases: [string, string][] = [
      // Two dates → whole days (96 days from Jan 15 to Apr 20, 2020).
      [
        `RETURN duration_between(DATE '2020-01-15', DATE '2020-04-20') AS d`,
        `[{"d":{"@duration":"P96D"}}]`,
      ],
      // Two datetimes → seconds (1h 1m 1s = 3661s), no month/day rollup.
      [
        `RETURN duration_between(DATETIME '2020-01-01T00:00:00', DATETIME '2020-01-01T01:01:01') AS d`,
        `[{"d":{"@duration":"PT3661S"}}]`,
      ],
      // Cross-kind → UNKNOWN (null).
      [
        `RETURN duration_between(DATE '2020-01-01', DATETIME '2020-01-01T00:00:00') AS d`,
        `[{"d":null}]`,
      ],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }
  });

  test('as-of WHERE filter over temporal values is byte-identical', () => {
    // Model the as-of over FOR-supplied dates + a WITH…WHERE window: keep the
    // date that falls inside the half-open [from, to) interval.
    const q = `FOR d IN [DATE '2020-06-01', DATE '2021-06-01'] WITH d WHERE DATE '2020-01-01' <= d AND d < DATE '2021-01-01' RETURN d`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"d":{"@date":"2020-06-01"}}]`);
  });

  test('temporal arithmetic is byte-identical', () => {
    const cases: [string, string][] = [
      [`RETURN DATE '2020-01-31' + DURATION 'P1M' AS d`, `[{"d":{"@date":"2020-02-29"}}]`],
      [`RETURN DATE '2021-01-31' + DURATION 'P1M' AS d`, `[{"d":{"@date":"2021-02-28"}}]`],
      [`RETURN DATE '2020-01-15' + DURATION 'P2M3D' AS d`, `[{"d":{"@date":"2020-03-18"}}]`],
      [
        `RETURN DATETIME '2020-01-01T10:00:00' + DURATION 'PT1H30M' AS d`,
        `[{"d":{"@datetime":"2020-01-01T11:30:00"}}]`,
      ],
      [`RETURN DATE '2020-03-18' - DURATION 'P2M3D' AS d`, `[{"d":{"@date":"2020-01-15"}}]`],
      [`RETURN DATE '2020-04-20' - DATE '2020-01-15' AS d`, `[{"d":{"@duration":"P96D"}}]`],
      [`RETURN DURATION 'P1M' + DURATION 'P2D' AS d`, `[{"d":{"@duration":"P1M2D"}}]`],
      [`RETURN DURATION 'P1M2DT3S' * 3 AS d`, `[{"d":{"@duration":"P3M6DT9S"}}]`],
      // A non-integer multiplier is invalid (a calendar duration has no
      // fractional multiple) → null on both engines, never a truncated value.
      [`RETURN DURATION 'P10D' * 1.5 AS d`, `[{"d":null}]`],
      [`RETURN DURATION 'P10D' * 2 AS d`, `[{"d":{"@duration":"P20D"}}]`],
      [`RETURN 0.5 * DURATION 'P10D' AS d`, `[{"d":null}]`],
    ];

    for (const [q, want] of cases) {
      const [ts, native] = both(q);

      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }
  });

  test('current_* read an injected now byte-identically (engine stays pure)', () => {
    // A FIXED `now` is handed to BOTH engines (via the reserved $__now param), so
    // the non-deterministic functions become deterministic and byte-identical.
    const now = { __now: parseDateTime('2026-07-12T10:30:45') };

    for (const [q, want] of [
      [`RETURN current_timestamp AS t`, `[{"t":{"@datetime":"2026-07-12T10:30:45"}}]`],
      [`RETURN local_timestamp AS t`, `[{"t":{"@datetime":"2026-07-12T10:30:45"}}]`],
      [`RETURN current_date AS d`, `[{"d":{"@date":"2026-07-12"}}]`],
    ] as [string, string][]) {
      const [ts, native] = both(q, now);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }

    // Without an injected now, both engines return null (no clock read).
    const [ts0, native0] = both(`RETURN current_date AS d`);
    expect(ts0).toBe(native0);
    expect(ts0).toBe(`[{"d":null}]`);
  });

  test('a host-wired clock (setClock) supplies $__now across the FFI, byte-identically', () => {
    // The clock lives in the JS host, not the engine — the same function wired
    // via setClock into both a native RustGraph and the TS core Graph. The
    // clock's LocalDateTime serializes to a tagged param, crosses the FFI, and
    // the crate revives it as $__now — so `current_*` reads it identically.
    const clock = () => parseDateTime('2026-07-13T09:00:00');
    const nat = graphFromNdjson(backend, MODERN_NDJSON).setClock(clock);
    const ts = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph()).setClock(clock);

    for (const [q, want] of [
      [`RETURN current_date AS d`, `[{"d":{"@date":"2026-07-13"}}]`],
      [`RETURN current_timestamp AS t`, `[{"t":{"@datetime":"2026-07-13T09:00:00"}}]`],
    ] as [string, string][]) {
      const native = JSON.stringify(nat.query(q));
      const tsOut = JSON.stringify(tsQuery(ts, q));
      expect(native, q).toBe(tsOut);
      expect(native, q).toBe(want);
    }

    // An explicit $__now still overrides the wired clock, on both sides.
    const pin = { __now: parseDateTime('2000-01-01T00:00:00') };
    expect(JSON.stringify(nat.query(`RETURN current_date AS d`, pin))).toBe(
      JSON.stringify(tsQuery(ts, `RETURN current_date AS d`, pin)),
    );
    nat.free();
  });

  test('current_timestamp coerces a DATE $__now to a DATETIME, byte-identically', () => {
    // A DATE `$__now` must not leak a DATE out of `current_timestamp` — the
    // datetime now-functions wrap in local_datetime(), coercing to midnight.
    const dateNow = { __now: parseDate('2026-07-12') };

    for (const [q, want] of [
      [`RETURN current_timestamp AS t`, `[{"t":{"@datetime":"2026-07-12T00:00:00"}}]`],
      [`RETURN current_date AS d`, `[{"d":{"@date":"2026-07-12"}}]`],
    ] as [string, string][]) {
      const [ts, native] = both(q, dateNow);
      expect(ts, q).toBe(native);
      expect(ts, q).toBe(want);
    }
  });

  test('UTF-16 slices (substring/left/right) across a surrogate pair are byte-identical', () => {
    // A slice can cut an astral pair; a lone surrogate must render U+FFFD on
    // BOTH engines (the native UTF-8 string cannot carry a lone surrogate).
    for (const q of [
      `RETURN substring('Rocket 🚀 go', 8, 1) AS s`,
      `RETURN substring('🚀🚀', 1, 1) AS s`,
      `RETURN left('🚀ab', 1) AS s`,
      `RETURN right('ab🚀', 1) AS s`,
    ]) {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
    }
  });
});

// Columnar grouped aggregation (`MATCH … WITH <key>, <agg> … RETURN`) runs the
// native side through the vectorized `with_frame` path (group by raw ids + folded
// columns), the TS side through its scalar accumulator. This block pins that the
// vectorized path stays byte-identical to the scalar one across key kinds (node
// identity, edge identity, property, multi-key), every aggregate, group-then-WHERE,
// rich-element carry-through, and the OPTIONAL fallback (which stays scalar).
suite('GQL differential: columnar grouped aggregation (TS vs native)', () => {
  // A small directed KNOWS graph where multiple sources have out-edges, so a
  // group-by-source produces several groups (a→2, b→1, c→1) — exercising the
  // group_ids refinement + first-seen ordering, not just a single group.
  const NDJSON = [
    '{"type":"node","id":"1","labels":["Person"],"properties":{"name":"a","age":30}}',
    '{"type":"node","id":"2","labels":["Person"],"properties":{"name":"b","age":20}}',
    '{"type":"node","id":"3","labels":["Person"],"properties":{"name":"c","age":40}}',
    '{"type":"node","id":"4","labels":["Person"],"properties":{"name":"d","age":20}}',
    '{"type":"edge","id":"10","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5,"since":2018}}',
    '{"type":"edge","id":"11","from":"1","to":"3","labels":["KNOWS"],"properties":{"weight":1.0,"since":2020}}',
    '{"type":"edge","id":"12","from":"2","to":"3","labels":["KNOWS"],"properties":{"weight":0.3,"since":2019}}',
    '{"type":"edge","id":"13","from":"3","to":"1","labels":["KNOWS"],"properties":{"weight":0.7,"since":2021}}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  const cases: Array<[string, string]> = [
    // group by node identity → count; the driving var is the group key.
    [
      'group by node → count',
      `MATCH (p:Person)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN p.name AS name, c ORDER BY name`,
    ],
    // group by node identity, RETURN the rich node — proves the element handle is
    // carried through the grouped frame (not flattened to an id).
    [
      'group by node → rich node carried',
      `MATCH (p:Person)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN p ORDER BY p.name`,
    ],
    // sum/avg/min/max over an edge property, grouped by the source node.
    [
      'group by node → sum/avg/min/max over edge prop',
      `MATCH (p:Person)-[e:KNOWS]->(f)
       WITH p, count(*) AS c, sum(e.weight) AS s, avg(e.weight) AS a, min(e.since) AS mn, max(e.since) AS mx
       RETURN p.name AS name, c, s, a, mn, mx ORDER BY name`,
    ],
    // group by a property key (edge.since) → count.
    [
      'group by edge property → count',
      `MATCH (p:Person)-[e:KNOWS]->(f) WITH e.since AS yr, count(*) AS c RETURN yr, c ORDER BY yr`,
    ],
    // multi-key grouping (two property keys) — refinement over two columns.
    [
      'group by two property keys',
      `MATCH (p:Person)-[:KNOWS]->(f) WITH p.age AS pa, f.age AS fa, count(*) AS c
       RETURN pa, fa, c ORDER BY pa, fa`,
    ],
    // group by edge identity (bare edge var) → count.
    [
      'group by edge identity',
      `MATCH (p:Person)-[e:KNOWS]->(f) WITH e, count(*) AS c RETURN e.since AS yr, c ORDER BY yr`,
    ],
    // group-then-filter (HAVING via WITH … WHERE) over the aggregate.
    [
      'group by node then WHERE on the aggregate',
      `MATCH (p:Person)-[:KNOWS]->(f) WITH p, count(f) AS c WHERE c > 1 RETURN p.name AS name, c ORDER BY name`,
    ],
    // global aggregate through the same path (ngroups == 1 fused fold).
    [
      'global aggregate over a traversal',
      `MATCH (p:Person)-[e:KNOWS]->(f) RETURN count(*) AS c, sum(e.weight) AS s, min(e.since) AS mn`,
    ],
    // OPTIONAL MATCH grouped agg — now VECTORIZED (expand_frame_optional): person
    // `d` has no out-edges, so its group is null-filled → count 0 (the null-fill row
    // path). Byte-identical to the scalar accumulator.
    [
      'OPTIONAL MATCH grouped → count (null-fill = 0)',
      `MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN p.name AS name, c ORDER BY name`,
    ],
    // OPTIONAL non-aggregating: `f`/`e` are nullable value columns — `f.name` is
    // null for the unmatched outer row `d` (val-column property access).
    [
      'OPTIONAL MATCH non-agg → null property for unmatched row',
      `MATCH (p:Person) OPTIONAL MATCH (p)-[e:KNOWS]->(f) RETURN p.name AS pn, f.name AS fn ORDER BY pn, fn`,
    ],
    // OPTIONAL aggregate over an edge property: `d`'s sum over zero edges is null,
    // avg is null, min is null — the folded aggregate on a null-only group.
    [
      'OPTIONAL MATCH → sum/avg/min over edge prop (empty group = null)',
      `MATCH (p:Person) OPTIONAL MATCH (p)-[e:KNOWS]->(f)
       WITH p, count(f) AS c, sum(e.weight) AS s, avg(e.weight) AS a, min(e.since) AS mn
       RETURN p.name AS name, c, s, a, mn ORDER BY name`,
    ],
    // OPTIONAL with an inline label on the optional node — the label filter runs as
    // a match check (a non-matching candidate is not a match → may null-fill).
    [
      'OPTIONAL MATCH with inline node label',
      `MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WITH p, count(f) AS c RETURN p.name AS name, c ORDER BY name`,
    ],
    // Agg-subquery DECORRELATION (native rewrites the correlated CALL to
    // `OPTIONAL MATCH … WITH p, count(f)`; TS stays correlated) — the strongest
    // check that the two forms agree. Friendless `d` → count 0 via the null-fill.
    [
      'correlated CALL count → decorrelated, byte-identical',
      `MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN p.name AS name, c ORDER BY name`,
    ],
    // Same, WITHOUT an outer ORDER BY — pins that decorrelation preserves the
    // correlated form's row order (outer scan order = first-seen group order).
    [
      'correlated CALL count → decorrelated, order preserved (no ORDER BY)',
      `MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN p.name AS name, c`,
    ],
    // Decorrelated sum over an edge property: friendless `d`'s sum over the empty
    // match is null (not 0) — the aggregate-over-null semantics.
    [
      'correlated CALL sum → decorrelated (empty = null)',
      `MATCH (p:Person) CALL (p) { MATCH (p)-[e:KNOWS]->(f) RETURN sum(e.weight) AS sw } RETURN p.name AS name, sw ORDER BY name`,
    ],
    // Terminal grouped aggregate + ORDER BY — vectorized_aggregate sorts the group
    // rows internally (was scalar). Order by the group key.
    [
      'terminal grouped agg + ORDER BY group key',
      `MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c ORDER BY n`,
    ],
    // Order by the AGGREGATE descending, tiebreak by the group key.
    [
      'terminal grouped agg + ORDER BY aggregate DESC',
      `MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c ORDER BY c DESC, n`,
    ],
    // Order by aggregate + LIMIT with a genuine TIE (b and c both count 1): the
    // tiebreak must resolve to first-seen group order on BOTH engines, else the
    // LIMIT keeps a different row. The strongest tie-order check.
    [
      'terminal grouped agg + ORDER BY agg DESC + LIMIT (tie)',
      `MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c ORDER BY c DESC LIMIT 2`,
    ],
  ];

  for (const [name, q] of cases) {
    test(name, () => {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
    });
  }

  test('group by node → count — exact expected shape', () => {
    const [ts, native] = both(
      `MATCH (p:Person)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN p.name AS name, c ORDER BY name`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"a","c":2},{"name":"b","c":1},{"name":"c","c":1}]`);
  });
});

// Grouped bounded var-length count — native takes the guarded-frequency-propagation
// shortcut (`try_grouped_varlen_1_2`, O(V+E), no trail enumeration); TS enumerates
// trails and groups. Byte-identity here proves the shortcut's per-endpoint trail
// multiplicity, its self-loop correction, and its replayed first-seen group order
// all match the enumerating engine. The graph deliberately includes a self-loop and
// a 2-cycle (the trail-vs-walk edge cases at bound ≤2).
suite('GQL differential: grouped var-length count shortcut (TS vs native)', () => {
  const NDJSON = [
    '{"type":"node","id":"1","labels":["Person"],"properties":{"city":"A"}}',
    '{"type":"node","id":"2","labels":["Person"],"properties":{"city":"B"}}',
    '{"type":"node","id":"3","labels":["Person"],"properties":{"city":"A"}}',
    '{"type":"node","id":"4","labels":["City"],"properties":{"city":"C"}}',
    // a 3-cycle 1→2→3→1, a chord 1→3, a self-loop 1→1, and an edge into the City node.
    '{"type":"edge","id":"10","from":"1","to":"2","labels":["KNOWS"]}',
    '{"type":"edge","id":"11","from":"2","to":"3","labels":["KNOWS"]}',
    '{"type":"edge","id":"12","from":"3","to":"1","labels":["KNOWS"]}',
    '{"type":"edge","id":"13","from":"1","to":"3","labels":["KNOWS"]}',
    '{"type":"edge","id":"14","from":"1","to":"1","labels":["KNOWS"]}',
    '{"type":"edge","id":"15","from":"2","to":"4","labels":["KNOWS"]}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  const cases: Array<[string, string]> = [
    // The headline shape: {1,2} grouped by endpoint property, first-seen order.
    [
      '{1,2} group by endpoint city',
      `MATCH (a)-[:KNOWS]->{1,2}(b) RETURN b.city AS c, count(*) AS n`,
    ],
    // Length-1 only — no self-loop-twice correction, no length-2 term.
    [
      '{1,1} group by endpoint city',
      `MATCH (a)-[:KNOWS]->{1,1}(b) RETURN b.city AS c, count(*) AS n`,
    ],
    // Length-2 only — isolates the length-2 term + self-loop correction.
    [
      '{2,2} group by endpoint city',
      `MATCH (a)-[:KNOWS]->{2,2}(b) RETURN b.city AS c, count(*) AS n`,
    ],
    // {0,2} includes the length-0 start-as-endpoint term.
    [
      '{0,2} group by endpoint city',
      `MATCH (a)-[:KNOWS]->{0,2}(b) RETURN b.city AS c, count(*) AS n`,
    ],
    // Start label filter (only Person starts).
    [
      '{1,2} with start label',
      `MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.city AS c, count(*) AS n`,
    ],
    // Endpoint label filter (only City endpoints).
    [
      '{1,2} with endpoint label',
      `MATCH (a)-[:KNOWS]->{1,2}(b:City) RETURN b.city AS c, count(*) AS n`,
    ],
    // Group by the endpoint node identity (not a property).
    [
      '{1,2} group by endpoint id',
      `MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN element_id(b) AS id, count(*) AS n ORDER BY id`,
    ],
    // Fixed two-hop grouped by the endpoint (try_grouped_2hop) — WALK semantics, so
    // the self-loop 1→1 makes 1→1→x paths count with NO trail correction (unlike the
    // var-length cases above). Same graph ⇒ the difference is provable.
    [
      'fixed 2-hop group by endpoint city',
      `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n`,
    ],
    [
      'fixed 2-hop with middle + end labels',
      `MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n`,
    ],
    [
      'fixed 2-hop group by endpoint id',
      `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN element_id(c) AS id, count(*) AS n ORDER BY id`,
    ],
  ];

  for (const [name, q] of cases) {
    test(name, () => {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
    });
  }
});

// Comma-join count shortcut: native takes `try_count_comma_join` (Σ_a filtered-
// out-degree(b) × filtered-out-degree(c), O(deg)); TS enumerates the cross product.
// Byte-identity proves the product equals the enumerated count — including the
// homomorphism cases where `b` and `c` can bind the SAME neighbour (overlapping
// predicates), which would diverge if either engine enforced node/edge uniqueness.
suite('GQL differential: comma-join count shortcut (TS vs native)', () => {
  const NDJSON = [
    '{"type":"node","id":"1","labels":["Person"],"properties":{"age":40}}',
    '{"type":"node","id":"2","labels":["Person"],"properties":{"age":70}}',
    '{"type":"node","id":"3","labels":["Person"],"properties":{"age":20}}',
    '{"type":"node","id":"4","labels":["Person"],"properties":{"age":65}}',
    '{"type":"node","id":"5","labels":["Person"],"properties":{"age":22}}',
    '{"type":"node","id":"6","labels":["Account"],"properties":{"age":80}}',
    '{"type":"edge","id":"10","from":"1","to":"2","labels":["KNOWS"]}',
    '{"type":"edge","id":"11","from":"1","to":"3","labels":["KNOWS"]}',
    '{"type":"edge","id":"12","from":"1","to":"4","labels":["KNOWS"]}',
    '{"type":"edge","id":"13","from":"1","to":"5","labels":["KNOWS"]}',
    '{"type":"edge","id":"14","from":"1","to":"6","labels":["OWNS"]}',
    '{"type":"edge","id":"15","from":"2","to":"3","labels":["KNOWS"]}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  const cases: Array<[string, string]> = [
    // Disjoint per-branch predicates (b>60 and c<25 can't be the same vertex).
    [
      'disjoint per-branch predicates',
      `MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 60 AND c.age < 25 RETURN count(*) AS n`,
    ],
    // OVERLAPPING predicates: a neighbour can be BOTH b and c — the count must
    // include the b==c diagonal (homomorphism), which the product does.
    [
      'overlapping predicates (b == c allowed)',
      `MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 30 AND c.age > 30 RETURN count(*) AS n`,
    ],
    // No WHERE-free branch: an anchor predicate (references only `a`).
    [
      'anchor predicate + one branch predicate',
      `MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE a.age = 40 AND b.age > 60 RETURN count(*) AS n`,
    ],
    // Different endpoint labels + different rel types per branch.
    [
      'different labels and rel types per branch',
      `MATCH (a:Person)-[:KNOWS]->(b:Person), (a)-[:OWNS]->(c:Account) WHERE b.age > 60 RETURN count(*) AS n`,
    ],
    // `WITH n, sum(...) RETURN count(*)` = count of distinct endpoints (the sum is
    // discarded) — native takes try_count_distinct_endpoint; TS materializes+groups.
    [
      'WITH endpoint, agg RETURN count(*) = distinct endpoints',
      `MATCH (m:Person)-[:KNOWS]->(n) WITH n, sum(m.age) AS s RETURN count(*) AS c`,
    ],
    // Endpoint label filter on the distinct-count.
    [
      'distinct endpoints with endpoint label',
      `MATCH (m:Person)-[:KNOWS]->(n:Person) WITH n, count(*) AS k RETURN count(*) AS c`,
    ],
  ];

  for (const [name, q] of cases) {
    test(name, () => {
      const [ts, native] = both(q);
      expect(ts, q).toBe(native);
    });
  }
});

// ISO GQL `LIMIT`/`OFFSET` accept a dynamic `$param` (nonNegativeIntegerSpecification,
// opengql:2268), and the COLON label-test predicate `WHERE n:Label` (opengql:2078).
// Both must produce byte-identical rows across the two engines.
suite('GQL differential: LIMIT/OFFSET $param + label-test predicate (TS vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, MODERN_NDJSON);
  const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());
  const both = (q: string, params?: Record<string, unknown>): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q, params)),
    JSON.stringify(nativeGraph.query(q, params)),
  ];

  test('LIMIT $n — dynamic bound resolves identically', () => {
    const q = `MATCH (n:Person) RETURN n.name AS name ORDER BY name LIMIT $n`;
    const [ts, native] = both(q, { n: 2 });
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"josh"},{"name":"marko"}]`);
  });

  test('OFFSET $o LIMIT $n — both bounds dynamic', () => {
    const q = `MATCH (n:Person) RETURN n.name AS name ORDER BY name OFFSET $o LIMIT $n`;
    const [ts, native] = both(q, { o: 1, n: 1 });
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"marko"}]`);
  });

  test('LIMIT $n over an unordered stream — set-based, still identical', () => {
    const [ts, native] = both(`MATCH (n:Person) RETURN count(*) AS c LIMIT $n`, { n: 5 });
    expect(ts).toBe(native);
  });

  test('WHERE n:Label — COLON label test, identical to IS LABELED', () => {
    const q = `MATCH (n) WHERE n:Person RETURN n.name AS name ORDER BY name`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"name":"josh"},{"name":"marko"},{"name":"vadas"}]`);
    // Same result as the spelled-out predicate.
    const [tsL] = both(`MATCH (n) WHERE n IS LABELED Person RETURN n.name AS name ORDER BY name`);
    expect(tsL).toBe(ts);
  });

  test('WHERE n:A|B — COLON with a label expression (disjunction)', () => {
    const q = `MATCH (n) WHERE n:Person|Software RETURN count(*) AS c`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":4}]`);
  });
});

// --- D1: a non-finite JSON number in a loaded document coerces to null on BOTH
// engines. TS coerces via `normalizeBag` on decode; native's ndjson/pg-json
// codecs map ±Infinity / NaN → null at the parse boundary. Storing a real
// non-finite float would corrupt count/sum/min/max/`IS NULL` and diverge.
suite('GQL differential: non-finite number coerces to null (D1)', () => {
  const backend = createFfiBackend(LIB);
  // `1e400` overflows an f64 to +Infinity; `-1e400` to -Infinity.
  const NF_NDJSON = [
    '{"type":"node","id":"1","labels":["N"],"properties":{"k":1,"v":1e400}}',
    '{"type":"node","id":"2","labels":["N"],"properties":{"k":2,"v":-1e400}}',
    '{"type":"node","id":"3","labels":["N"],"properties":{"k":3,"v":2.5}}',
  ].join('\n');
  const nativeGraph = graphFromNdjson(backend, NF_NDJSON);
  const tsGraph = tsDeserialize(NF_NDJSON, 'ndjson', new Graph());

  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  test('an overflowing literal reads back as a PRESENT null', () => {
    const [ts, native] = both(`MATCH (n:N) RETURN n.v AS v ORDER BY n.k`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"v":null},{"v":null},{"v":2.5}]`);
  });

  test('IS NULL sees the coerced value as null (the repro)', () => {
    const [ts, native] = both(`MATCH (n:N) WHERE n.v IS NULL RETURN count(*) AS c`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":2}]`);
  });

  test('aggregates ignore the coerced nulls identically (no NaN poisoning)', () => {
    const [ts, native] = both(
      `MATCH (n:N) RETURN count(n.v) AS c, sum(n.v) AS s, min(n.v) AS mn, max(n.v) AS mx`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":1,"s":2.5,"mn":2.5,"mx":2.5}]`);
  });
});

// --- D2/D3: TS param validation matches native's FFI param decoder. Both engines
// accept and reject exactly the same param shapes with the same error code.
suite('GQL differential: param value validation (D2/D3)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, MODERN_NDJSON);
  const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());

  const outcome = (run: () => unknown): { ok: true } | { code: unknown } => {
    try {
      run();

      return { ok: true };
    } catch (e) {
      return { code: (e as { code?: unknown }).code };
    }
  };
  const both = (q: string, params: Record<string, unknown>) => ({
    ts: outcome(() => tsQuery(tsGraph, q, params)),
    native: outcome(() => nativeGraph.query(q, params)),
  });

  const Q = `MATCH (n:Person) WHERE n.age = $x RETURN count(*) AS c`;

  // D2: a bound value of undefined / a function / a symbol is dropped by native's
  // JSON.stringify marshalling → MISSING; TS must not silently evaluate it to
  // undefined (which returns [] with no error).
  for (const [label, value] of [
    ['undefined', undefined],
    ['a function', () => 1],
    ['a symbol', Symbol('x')],
  ] as const) {
    test(`D2: ${label} param faults as E_MISSING_PARAMETER on both engines`, () => {
      const { ts, native } = both(Q, { x: value });
      expect(native).toEqual(ts);
      expect(ts).toEqual({ code: 'E_MISSING_PARAMETER' });
    });
  }

  // D3: a nested object / nested array is outside the LPG param model → both
  // engines reject with E_INVALID_JSON (native's `params.rs` grammar).
  for (const [label, value] of [
    ['a nested object', { a: 1 }],
    ['a nested array', [[1]]],
  ] as const) {
    test(`D3: ${label} param faults as E_INVALID_JSON on both engines`, () => {
      const { ts, native } = both(`RETURN $x AS x`, { x: value });
      expect(native).toEqual(ts);
      expect(ts).toEqual({ code: 'E_INVALID_JSON' });
    });
  }

  test('D3: a bigint param faults as E_INVALID_VALUE on both engines', () => {
    const { ts, native } = both(Q, { x: 10n });
    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_INVALID_VALUE' });
  });

  // Guardrails: valid scalar, flat-list, and tagged-temporal params still run and
  // stay byte-identical (the fix must NOT reject these).
  test('valid params (scalar, flat list, tagged temporal) still run identically', () => {
    const scalar = both(Q, { x: 29 });
    expect(scalar.ts).toEqual({ ok: true });
    expect(scalar.native).toEqual({ ok: true });

    const list = [
      JSON.stringify(tsQuery(tsGraph, `RETURN $xs AS xs`, { xs: [1, 'two', true, null] })),
      JSON.stringify(nativeGraph.query(`RETURN $xs AS xs`, { xs: [1, 'two', true, null] })),
    ];
    expect(list[0]).toBe(list[1]);
    expect(list[0]).toBe(`[{"xs":[1,"two",true,null]}]`);

    const temporal = [
      JSON.stringify(tsQuery(tsGraph, `RETURN $d AS d`, { d: { '@date': '2020-07-01' } })),
      JSON.stringify(nativeGraph.query(`RETURN $d AS d`, { d: { '@date': '2020-07-01' } })),
    ];
    expect(temporal[0]).toBe(temporal[1]);
    expect(temporal[0]).toBe(`[{"d":{"@date":"2020-07-01"}}]`);
  });
});

// ALL SHORTEST enumerates every path tied for the fewest-hop length (ISO
// per-path multiplicity), unlike ANY SHORTEST which keeps one. Needs a graph
// with multiple equal-length paths, so it builds its own diamond on both engines
// from shared NDJSON (identical ids → path values compare byte-for-byte).
suite('gql conformance: ALL SHORTEST — every tied path, byte-identical', () => {
  // a→b→d and a→c→d (two shortest a..d), then d→e→f (the tail extends both).
  const NDJSON = [
    { type: 'node', id: 'a', labels: ['N'], properties: { id: 'a' } },
    { type: 'node', id: 'b', labels: ['N'], properties: { id: 'b' } },
    { type: 'node', id: 'c', labels: ['N'], properties: { id: 'c' } },
    { type: 'node', id: 'd', labels: ['N'], properties: { id: 'd' } },
    { type: 'node', id: 'e', labels: ['N'], properties: { id: 'e' } },
    { type: 'node', id: 'f', labels: ['N'], properties: { id: 'f' } },
    { type: 'edge', id: 'ea', from: 'a', to: 'b', labels: ['R'], properties: {} },
    { type: 'edge', id: 'eb', from: 'a', to: 'c', labels: ['R'], properties: {} },
    { type: 'edge', id: 'ec', from: 'b', to: 'd', labels: ['R'], properties: {} },
    { type: 'edge', id: 'ed', from: 'c', to: 'd', labels: ['R'], properties: {} },
    { type: 'edge', id: 'ee', from: 'd', to: 'e', labels: ['R'], properties: {} },
    { type: 'edge', id: 'ef', from: 'e', to: 'f', labels: ['R'], properties: {} },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  test('two tied shortest paths a..d — full Path values agree', () => {
    const [ts, native] = both(
      `MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN p`,
    );
    expect(ts).toBe(native);
    // Both length-2 paths, in predecessor-recording order (via b, then via c).
    expect(ts).toContain('"b"');
    expect(ts).toContain('"c"');
    expect(JSON.parse(ts)).toHaveLength(2);
  });

  test('the tail extends both shortest paths (a..f), byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'f'}) RETURN p`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(2);
  });

  test('endpoint multiplicity (rows per endpoint) is byte-identical', () => {
    const [ts, native] = both(
      `MATCH ALL SHORTEST (a:N {id:'a'})-[:R]->*(x) RETURN x.id AS id ORDER BY id`,
    );
    expect(ts).toBe(native);
  });

  test('ANY SHORTEST still keeps exactly one path here', () => {
    const [ts, native] = both(
      `MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN p`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(1);
  });
});

// Path modes (WALK/TRAIL/SIMPLE/ACYCLIC) restrict which repeats a matched path
// may contain; byte-identical across engines. Triangle a→b→c→a + a→d + b→a on a
// shared graph. Default (no mode) == TRAIL.
suite('gql conformance: path modes — byte-identical restrictors', () => {
  const NDJSON = [
    { type: 'node', id: 'a', labels: ['N'], properties: { id: 'a' } },
    { type: 'node', id: 'b', labels: ['N'], properties: { id: 'b' } },
    { type: 'node', id: 'c', labels: ['N'], properties: { id: 'c' } },
    { type: 'node', id: 'd', labels: ['N'], properties: { id: 'd' } },
    { type: 'edge', id: 'e1', from: 'a', to: 'b', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e2', from: 'b', to: 'c', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e3', from: 'c', to: 'a', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e4', from: 'a', to: 'd', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e5', from: 'b', to: 'a', labels: ['R'], properties: {} },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  for (const mode of ['', 'WALK ', 'TRAIL ', 'SIMPLE ', 'ACYCLIC ']) {
    test(`${mode || '(default)'}(a)-[:R]->{1,3}(x) — endpoints agree`, () => {
      const [ts, native] = both(
        `MATCH ${mode}(a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY id`,
      );
      expect(ts).toBe(native);
    });

    test(`${mode || '(default)'} count(*) agrees (general matcher for non-trail)`, () => {
      const [ts, native] = both(`MATCH ${mode}(a:N {id:'a'})-[:R]->{1,3}(x) RETURN count(*) AS c`);
      expect(ts).toBe(native);
    });
  }

  test('ACYCLIC excludes the cycle-back-to-seed; default (TRAIL) includes it', () => {
    const [acTs] = both(
      `MATCH ACYCLIC (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY id`,
    );
    const [defTs] = both(`MATCH (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY id`);
    expect(acTs).not.toContain('"a"');
    expect(defTs).toContain('"a"');
  });
});

// Bare path-variable binding over a single quantified segment (`p = (a)-[]->{m,n}(b)`,
// no selector) enumerates EVERY walk under the pattern's mode and binds each as a
// full Path value — the `all_walk` driver, byte-identical across engines. Same
// triangle-with-tail as the path-modes suite (shared ids → Path values compare
// byte-for-byte).
suite('gql conformance: bare path binding — every walk as a Path, byte-identical', () => {
  const NDJSON = [
    { type: 'node', id: 'a', labels: ['N'], properties: { id: 'a' } },
    { type: 'node', id: 'b', labels: ['N'], properties: { id: 'b' } },
    { type: 'node', id: 'c', labels: ['N'], properties: { id: 'c' } },
    { type: 'node', id: 'd', labels: ['N'], properties: { id: 'd' } },
    { type: 'edge', id: 'e1', from: 'a', to: 'b', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e2', from: 'b', to: 'c', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e3', from: 'c', to: 'a', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e4', from: 'a', to: 'd', labels: ['R'], properties: {} },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  test('p bound over {1,3} — full Path values agree (a-b, a-b-c, a-b-c-a, a-d)', () => {
    const [ts, native] = both(
      `MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN nodes(p) AS ns ORDER BY path_length(p), x.id`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(4);
  });

  test('the whole Path value (vertices + edges) is byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN p ORDER BY path_length(p), x.id`,
    );
    expect(ts).toBe(native);
  });

  test('path_length agrees per row', () => {
    const [ts, native] = both(
      `MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len ORDER BY len, x.id`,
    );
    expect(ts).toBe(native);
  });

  test('SIMPLE p back to the seed keeps only the closing cycle', () => {
    const [ts, native] = both(
      `MATCH p = SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN nodes(p) AS ns`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(1);
  });

  test('bare ALL selector == default enumeration, byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = ALL (a:N {id:'a'})-[:R]->{1,3}(x) RETURN nodes(p) AS ns ORDER BY path_length(p), x.id`,
    );
    const [bareTs, bareNative] = both(
      `MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN nodes(p) AS ns ORDER BY path_length(p), x.id`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(bareTs);
    expect(native).toBe(bareNative);
  });

  test('bare ALL composes with SIMPLE mode, byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = ALL SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN nodes(p) AS ns`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(1);
  });
});

// Per-hop edge predicates on variable-length segments: the predicate (inline
// props / WHERE, optionally naming each hop's edge) filters every edge of the
// walk. Byte-identical across engines on a shared weighted chain a→b→c→d.
suite('gql conformance: per-hop edge predicate on var-length — byte-identical', () => {
  const NDJSON = [
    { type: 'node', id: 'a', labels: ['N'], properties: { id: 'a' } },
    { type: 'node', id: 'b', labels: ['N'], properties: { id: 'b' } },
    { type: 'node', id: 'c', labels: ['N'], properties: { id: 'c' } },
    { type: 'node', id: 'd', labels: ['N'], properties: { id: 'd' } },
    { type: 'edge', id: 'e1', from: 'a', to: 'b', labels: ['R'], properties: { amt: 10 } },
    { type: 'edge', id: 'e2', from: 'b', to: 'c', labels: ['R'], properties: { amt: 20 } },
    { type: 'edge', id: 'e3', from: 'c', to: 'd', labels: ['R'], properties: { amt: 5 } },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  test('WHERE e.amt >= 10 filters the low-weight hop (d unreachable)', () => {
    const [ts, native] = both(
      `MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) RETURN x.id AS id ORDER BY id`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toEqual([{ id: 'b' }, { id: 'c' }]);
  });

  test('loosening the threshold restores full reach', () => {
    const [ts, native] = both(
      `MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 1]->{1,3}(x) RETURN x.id AS id ORDER BY id`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toEqual([{ id: 'b' }, { id: 'c' }, { id: 'd' }]);
  });

  test('inline property predicate {amt:20} filters each hop', () => {
    const [ts, native] = both(
      `MATCH (b:N {id:'b'})-[:R {amt:20}]->{1,3}(x) RETURN x.id AS id ORDER BY id`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toEqual([{ id: 'c' }]);
  });

  test('predicate composes with a bound path variable', () => {
    const [ts, native] = both(
      `MATCH p = (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) ` +
        `RETURN nodes(p) AS ns ORDER BY path_length(p), x.id`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(2);
  });

  test('count(*) over a predicated var-length agrees (general matcher routing)', () => {
    const [ts, native] = both(
      `MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) RETURN count(*) AS c`,
    );
    expect(ts).toBe(native);
  });

  test('per-hop predicate + ANY SHORTEST runs (filtered BFS), byte-identical', () => {
    const q = `MATCH ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.amt > 1]->*(x) RETURN x.id AS id ORDER BY id`;
    expect(JSON.stringify(nativeGraph.query(q))).toBe(JSON.stringify(tsQuery(tsGraph, q)));
  });
});

// Bare `ANY` (one arbitrary path per endpoint) and `SHORTEST k [GROUP]` (the k
// shortest / the k smallest length-groups). Fixture: to `d` there is one length-1
// path (a→d) and two length-2 paths (a→b→d, a→c→d). Byte-identical across engines.
suite('gql conformance: ANY / SHORTEST k — byte-identical', () => {
  const NDJSON = [
    { type: 'node', id: 'a', labels: ['N'], properties: { id: 'a' } },
    { type: 'node', id: 'b', labels: ['N'], properties: { id: 'b' } },
    { type: 'node', id: 'c', labels: ['N'], properties: { id: 'c' } },
    { type: 'node', id: 'd', labels: ['N'], properties: { id: 'd' } },
    { type: 'edge', id: 'e1', from: 'a', to: 'd', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e2', from: 'a', to: 'b', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e3', from: 'b', to: 'd', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e4', from: 'a', to: 'c', labels: ['R'], properties: {} },
    { type: 'edge', id: 'e5', from: 'c', to: 'd', labels: ['R'], properties: {} },
  ]
    .map((r) => JSON.stringify(r))
    .join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  test('bare ANY yields one path per endpoint', () => {
    const [ts, native] = both(`MATCH ANY (a:N {id:'a'})-[:R]->*(x) RETURN x.id AS id ORDER BY id`);
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toEqual([{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }]);
  });

  test('ANY p = … binds one Path per endpoint, byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = ANY (a:N {id:'a'})-[:R]->*(x) RETURN nodes(p) AS ns ORDER BY x.id`,
    );
    expect(ts).toBe(native);
  });

  test('SHORTEST 2 keeps the two shortest paths to d', () => {
    const [ts, native] = both(
      `MATCH p = SHORTEST 2 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len ORDER BY len`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toEqual([{ len: 1 }, { len: 2 }]);
  });

  test('SHORTEST 2 GROUP keeps all paths in the two smallest length groups', () => {
    const [ts, native] = both(
      `MATCH p = SHORTEST 2 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) ` +
        `RETURN nodes(p) AS ns ORDER BY path_length(p)`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(3);
  });

  test('SHORTEST 1 GROUP == ALL SHORTEST here', () => {
    const [grpTs, grpNative] = both(
      `MATCH p = SHORTEST 1 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len`,
    );
    const [allTs] = both(
      `MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len`,
    );
    expect(grpTs).toBe(grpNative);
    expect(grpTs).toBe(allTs);
  });

  test('GROUPS is a synonym for GROUP', () => {
    const [ts, native] = both(
      `MATCH p = SHORTEST 2 GROUPS (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len ORDER BY len`,
    );
    expect(ts).toBe(native);
    expect(JSON.parse(ts)).toHaveLength(3);
  });

  test('SHORTEST k over the whole graph (all endpoints) is byte-identical', () => {
    const [ts, native] = both(
      `MATCH p = SHORTEST 2 (a:N {id:'a'})-[:R]->*(x) RETURN x.id AS id, path_length(p) AS len ORDER BY id, len`,
    );
    expect(ts).toBe(native);
  });
});

// A graph with stored map/record properties (nested, out-of-order keys) — proves
// the map round-trips through BOTH engines' storage and reads back byte-identical.
const MAP_NDJSON = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","meta":{"city":"NYC","zip":"10001"}}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"id":"b","meta":{"city":"LA","zip":"90001"}}}',
  '{"type":"node","id":"c","labels":["P"],"properties":{"id":"c","meta":{"city":"NYC","zip":"10002"}}}',
].join('\n');

suite('GQL differential: stored map/record properties (TS vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, MAP_NDJSON);
  const tsGraph = tsDeserialize(MAP_NDJSON, 'ndjson', new Graph());

  const both = (q: string, params?: Record<string, unknown>): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q, params)),
    JSON.stringify(nativeGraph.query(q, params)),
  ];

  test('read a whole stored map — canonical (sorted keys), byte-identical', () => {
    const [ts, native] = both(`MATCH (n:P {id: 'a'}) RETURN n.meta AS m`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"m":{"city":"NYC","zip":"10001"}}]`);
  });

  test('nested field access on a stored map', () => {
    const [ts, native] = both(`MATCH (n:P {id: 'b'}) RETURN n.meta.city AS c, n.meta.zip AS z`);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"c":"LA","z":"90001"}]`);
    // A missing nested field → null.
    const [ts2, nat2] = both(`MATCH (n:P {id: 'b'}) RETURN n.meta.nope AS x`);
    expect(ts2).toBe(nat2);
    expect(ts2).toBe(`[{"x":null}]`);
  });

  test('WHERE on a nested map field filters rows identically', () => {
    const [ts, native] = both(
      `MATCH (n:P) WHERE n.meta.city = 'NYC' RETURN n.id AS id ORDER BY id`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"id":"a"},{"id":"c"}]`);
  });

  test('construct a record from a stored map field, and compare', () => {
    const [ts, native] = both(
      `MATCH (n:P {id: 'a'}) RETURN {here: n.meta.city, eq: n.meta = {city: 'NYC', zip: '10001'}} AS r`,
    );
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"r":{"eq":true,"here":"NYC"}}]`);
  });

  test('SET a map property, then read it back — byte-identical write path', () => {
    // Mutating both graphs in lockstep; assert the read matches.
    const q = `MATCH (n:P {id: 'c'}) SET n.tag = {b: 2, a: 1} RETURN n.tag AS t`;
    const [ts, native] = both(q);
    expect(ts).toBe(native);
    expect(ts).toBe(`[{"t":{"a":1,"b":2}}]`);
  });
});

// ORDER BY + LIMIT top-k: the engines must project exactly the same rows, so a
// projection that would fault on a row OUTSIDE the top-k must not fault in
// either. Native has always kept only the top-k input bindings when the sort
// keys don't read the output; the TS engine projected every row first, so it
// faulted where native returned a row.
suite('ORDER BY + LIMIT projects only the emitted rows', () => {
  const NDJSON = [
    '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a"}}',
    '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z"}}',
    '{"type":"node","id":"3","labels":["T"],"properties":{"n":5,"s":"m"}}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  // Release the native handle deterministically — the GC backstop warns (and is
  // not guaranteed to run) once enough graphs accumulate across this file.
  afterAll(() => {
    nativeGraph.free();
  });

  test('a faulting projection outside the top-k is never evaluated', () => {
    // `n.n = 7` divides by zero, but ORDER BY n.n LIMIT 1 emits only n.n = 3.
    const [ts, native] = both(`MATCH (n:T) RETURN 1/(n.n - 7) AS x ORDER BY n.n LIMIT 1`);

    expect(ts).toBe(native);
    expect(ts).toBe(`[{"x":-0.25}]`);
  });

  test('a sort key that READS the output still projects every row', () => {
    // The sort key is the projected column, so every row must be projected to
    // sort at all — both engines fault. Same for an alias of the input column.
    for (const q of [
      `MATCH (n:T) RETURN 1/(n.n - 7) AS x ORDER BY x LIMIT 1`,
      `MATCH (n:T) RETURN 1/(n.n - 7) AS x, n.n AS t ORDER BY t LIMIT 1`,
    ]) {
      expect(() => tsQuery(tsGraph, q)).toThrow();
      expect(() => nativeGraph.query(q)).toThrow();
    }
  });

  test('the ordinary ORDER BY + LIMIT results are unchanged', () => {
    for (const q of [
      `MATCH (n:T) RETURN n.n AS v ORDER BY n.n LIMIT 2`,
      `MATCH (n:T) RETURN n.n AS v ORDER BY n.n DESC LIMIT 2`,
      `MATCH (n:T) RETURN n.n AS v ORDER BY n.n SKIP 1 LIMIT 1`,
      `MATCH (n:T) RETURN n.s AS v ORDER BY n.n DESC LIMIT 2`,
      `MATCH (n:T) RETURN DISTINCT n.n AS v ORDER BY n.n LIMIT 2`,
      `MATCH (n:T) RETURN * ORDER BY n.n LIMIT 1`,
      `MATCH (n:T) RETURN n.n AS v ORDER BY n.n`,
      `MATCH (n:T) RETURN n.n AS v ORDER BY n.n LIMIT 99`,
    ]) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
    }
  });
});

// Grouping keys must agree with equality. `-0 = 0` is true, and the distinction is
// normalized everywhere else (ORDER BY, sign(), the result JSON, the property
// index), so both engines collapse the two zeroes into ONE group — including
// inside a record, which the TS side used to key by stringifying (and which had
// disagreed with native in the other direction).
suite('signed zero is one grouping key, nested or not', () => {
  const NDJSON = [
    '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"x":-1}}',
    '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"x":4}}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());

  afterAll(() => {
    nativeGraph.free();
  });

  test('-0 and 0 collapse inside a record, a list, and bare', () => {
    for (const q of [
      // `0 * -1` is -0 and `0 * 4` is 0 — ONE group, and it renders as `0`.
      `MATCH (n:T) RETURN count(DISTINCT {b: (0 * n.x)}) AS x`,
      `MATCH (n:T) RETURN (0 * n.x) AS k, count(*) AS c GROUP BY k`,
      `MATCH (n:T) RETURN count(DISTINCT {a: 1, b: (0 * n.x)}) AS x`,
      `MATCH (n:T) RETURN count(DISTINCT {b: [0 * n.x]}) AS x`,
      `MATCH (n:T) RETURN count(DISTINCT (0 * n.x)) AS x`,
      `MATCH (n:T) RETURN collect_list(DISTINCT {a: 1, b: (0 * n.x)}) AS x`,
      // …and equal records still collapse.
      `MATCH (n:T) RETURN count(DISTINCT {b: 1}) AS x`,
      `MATCH (n:T) RETURN count(DISTINCT {b: n.n}) AS x`,
    ]) {
      const ts = JSON.stringify(tsQuery(tsGraph, q));
      const native = JSON.stringify(nativeGraph.query(q));

      expect(ts).toBe(native);
    }
  });
});

// ISO `<order by and page statement>` in STATEMENT position — the grammar puts
// `orderByAndPageStatement` both trailing a RETURN (`primitiveResultStatement`)
// and as a pipeline step of its own (`primitiveQueryStatement`). This covers the
// second form, which sorts/slices the working BINDING table before any projection
// runs, so a later RETURN only ever projects the survivors.
suite('ISO standalone ORDER BY / OFFSET / LIMIT statement', () => {
  const NDJSON = [
    '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a"}}',
    '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z"}}',
    '{"type":"node","id":"3","labels":["T"],"properties":{"n":5,"s":"m"}}',
  ].join('\n');

  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());
  const both = (q: string): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q)),
    JSON.stringify(nativeGraph.query(q)),
  ];

  afterAll(() => {
    nativeGraph.free();
  });

  test('the statement form sorts, slices, and composes identically', () => {
    for (const q of [
      `MATCH (n:T) ORDER BY n.n RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n DESC RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.s RETURN n.s AS x`,
      `MATCH (n:T) ORDER BY n.n LIMIT 2 RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n DESC LIMIT 2 RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n OFFSET 1 RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n OFFSET 1 LIMIT 1 RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n OFFSET 99 RETURN n.n AS x`,
      // Bare LIMIT / OFFSET with no ORDER BY.
      `MATCH (n:T) LIMIT 1 RETURN n.n AS x`,
      `MATCH (n:T) OFFSET 1 RETURN n.n AS x`,
      // Composes with FILTER, with an aggregate, and with itself.
      `MATCH (n:T) FILTER n.n > 3 ORDER BY n.n LIMIT 1 RETURN n.n AS x`,
      `MATCH (n:T) ORDER BY n.n LIMIT 2 RETURN count(*) AS c`,
      `MATCH (n:T) ORDER BY n.n LIMIT 2 ORDER BY n.n DESC LIMIT 1 RETURN n.n AS x`,
      // The trailing form is unaffected.
      `MATCH (n:T) RETURN n.n AS x ORDER BY x LIMIT 2`,
    ]) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
    }
  });

  test('paging trims the binding table BEFORE the projection runs', () => {
    // The semantic point of the statement form: a projection that would fault on
    // a dropped row never runs, because the row is gone before RETURN sees it.
    for (const [q, want] of [
      [`MATCH (n:T) ORDER BY n.n LIMIT 0 RETURN 1/0 AS x`, `[]`],
      [`MATCH (n:T) ORDER BY n.n LIMIT 1 RETURN 1/(n.n - 7) AS x`, `[{"x":-0.25}]`],
    ] as const) {
      const [ts, native] = both(q);

      expect(ts).toBe(native);
      expect(ts).toBe(want);
    }
  });

  test('an unbound dynamic bound is a clean MissingParameter in both', () => {
    const q = `MATCH (n:T) ORDER BY n.n LIMIT $k RETURN n.n AS x`;

    expect(() => tsQuery(tsGraph, q)).toThrow();
    expect(() => nativeGraph.query(q)).toThrow();
  });
});
