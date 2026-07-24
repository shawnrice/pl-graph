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
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph, parseDate, parseDateTime } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './backend-ffi.js';
import { graphFromFormat } from './graph.js';

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
  const nativeGraph = graphFromFormat(backend, MODERN_NDJSON, 'ndjson');
  const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());

  const both = (q: string, params?: Record<string, unknown>): [string, string] => [
    JSON.stringify(tsQuery(tsGraph, q, params)),
    JSON.stringify(nativeGraph.query(q, params)),
  ];

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

    // Every tagged kind revives; also inside a list param.
    const [tsAll, natAll] = both(`RETURN $dt AS dt, $dur AS dur, $xs AS xs`, {
      dt: { '@datetime': '2020-06-15T08:30:00' },
      dur: { '@duration': 'P1Y2M3DT4H' },
      xs: [{ '@date': '2020-01-01' }, { '@localtime': '08:30:00' }],
    });
    expect(tsAll).toBe(natAll);
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

  test('edges(path) — ISO name for the path edge list, == relationships(path)', () => {
    const base = `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop'`;
    const [tsE, natE] = both(`${base} RETURN path_length(p) AS len, edges(p) AS es`);
    expect(tsE).toBe(natE);
    // `edges` (ISO) and `relationships` (openCypher) are the same accessor.
    const [tsR, natR] = both(`${base} RETURN edges(p) AS es`);
    const [tsR2] = both(`${base} RETURN relationships(p) AS es`);
    expect(tsR).toBe(natR);
    expect(tsR).toBe(tsR2);
  });

  test('list[i].prop — property access chains off a subscript, byte-identical', () => {
    const base = `MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop'`;
    // relationships(p)[0].weight, nodes(p)[i].name, and out-of-range → null all
    // agree across engines.
    const [ts, native] = both(
      `${base} RETURN relationships(p)[0].weight AS w, nodes(p)[0].name AS a, nodes(p)[1].name AS b, relationships(p)[9].weight AS oob`,
    );
    expect(ts).toBe(native);

    // Value check on a single column (robust to object key ordering).
    const [tsW] = both(`${base} RETURN relationships(p)[0].weight AS w`);
    expect(tsW).toBe(`[{"w":0.4}]`);

    // Consecutive-index comparison — the per-hop path-predicate motif — agrees.
    const [tsC, natC] = both(
      `${base} RETURN (relationships(p)[0].weight < relationships(p)[0].weight) AS lt`,
    );
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
      `nodes(p) AS ns, relationships(p) AS es, elements(p) AS el`;
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
    const nat = graphFromFormat(backend, VARLEN_NDJSON, 'ndjson');
    const ts = tsDeserialize(VARLEN_NDJSON, 'ndjson', new Graph());

    for (const q of [
      `MATCH (x)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c`,
      `MATCH (x:VIP)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c`,
    ]) {
      expect(JSON.stringify(nat.query(q)), q).toBe(JSON.stringify(tsQuery(ts, q)));
    }
  });

  // --- string `id` as element identity: `INSERT (:P {id: 'x'})` makes 'x' the
  // element id (so element_id === n.id and it round-trips), a numeric id stays an
  // ordinary property, dup/SET-id are rejected. Must be byte-identical. ---------
  test('string id property is the element identity (TS vs native)', () => {
    const nat = graphFromFormat(backend, '', 'ndjson');
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
    const nat = graphFromFormat(backend, '', 'ndjson');
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
  // always streamed it. They must return the same rows. Regression: the round-16
  // dogfood sim drove native to an OOM kill on exactly this shape. -------------
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
    const nat = graphFromFormat(backend, CHAIN_NDJSON, 'ndjson');
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
    const nat = graphFromFormat(backend, REACH_NDJSON, 'ndjson');
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

  test('R-BATCH: FOR drives a batch OPTIONAL MATCH (allow + deny) byte-identically', () => {
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
    const nat = graphFromFormat(backend, MODERN_NDJSON, 'ndjson').setClock(clock);
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
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, MODERN_NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, NF_NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, MODERN_NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
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
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
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
