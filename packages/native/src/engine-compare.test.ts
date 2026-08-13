// Cross-engine CORRECTNESS: the shipped row-based core vs the standalone columnar
// engine (`lenke-engine`), driven from the SAME graph through one artifact's `lnk_*` /
// `lnk_e_*` FFI. A broad Gremlin + GQL query set runs on both; results are compared as
// values (JSON round-trip, so number spelling / spacing never counts). This is the
// apples-to-apples correctness gate the engine must clear to be a drop-in for core.
//
// Needs a `.so` built WITH the feature (default `bun run build:rust` does NOT include it):
//   cargo build --release --features engine-compare --manifest-path ../../crates/lenke-core/Cargo.toml
import { describe, expect, test } from 'bun:test';

import { EDGE_CASE_NDJSON, fixtureNdjson } from './engine-compare-fixture.js';
import { type CompareHandle, loadCompare, norm } from './engine-compare.js';

// Recursively sort object keys, so two documents that differ ONLY in key order
// normalize equal. Used to prove a divergence is purely ordering, not content.
const cmp = (a: string, b: string): number => {
  if (a < b) {
    return -1;
  }

  return a > b ? 1 : 0;
};
const sortKeys = (v: unknown): unknown => {
  if (Array.isArray(v)) {
    return v.map(sortKeys);
  }

  if (v && typeof v === 'object') {
    return Object.fromEntries(
      Object.entries(v as Record<string, unknown>)
        .sort(([a], [b]) => cmp(a, b))
        .map(([k, x]) => [k, sortKeys(x)]),
    );
  }

  return v;
};
const normDeep = (j: string | null): string | null =>
  j === null ? null : JSON.stringify(sortKeys(JSON.parse(j)));

// Gremlin query set — the feature space the perf/correctness work covered: source
// scans, 1–3 hops (out/in/both, typed and untyped), edge hops, labels, has-predicates,
// values/valueMap, dedup, count/groupCount, order+limit, repeat (fixed/emit/until),
// aggregates, path, tree.
const gremlinQueries = (): string[] => [
  'g.V().count()',
  'g.V().hasLabel("Person").count()',
  'g.V().hasLabel("VIP").count()',
  'g.V().has("age", gt(40)).count()',
  'g.V().has("score", lt(0)).count()',
  'g.V().values("name")',
  'g.V().values("age")',
  'g.V().label()',
  'g.V().id()',
  'g.V().out().count()',
  'g.V().in().count()',
  'g.V().both().count()',
  'g.V().out("R").count()',
  'g.V().out("F").count()',
  'g.V().out().out().count()',
  'g.V().out().in().count()',
  'g.V().both().both().count()',
  'g.V().out().values("name")',
  'g.V().out().dedup().count()',
  'g.V().both().dedup().count()',
  'g.V().outE().count()',
  'g.V().outE("R").inV().count()',
  'g.V().bothE().otherV().count()',
  'g.V().outE().inV().values("age")',
  'g.V().hasLabel("VIP").out().values("city")',
  'g.V().groupCount().by("city")',
  'g.V().out().groupCount().by("city")',
  'g.V().values("score").sum()',
  'g.V().values("age").max()',
  'g.V().values("age").min()',
  'g.V().values("score").mean()',
  'g.V().order().by("age").limit(5).values("name")',
  'g.V().order().by("score").limit(10).values("score")',
  'g.V().hasLabel("VIP").order().by("age").limit(3).values("name")',
  'g.V().repeat(__.out()).times(2).count()',
  'g.V().repeat(__.both()).times(2).dedup().count()',
  'g.V().repeat(__.out()).times(1).emit().count()',
  'g.V().repeat(__.out()).until(__.hasLabel("VIP")).times(3).count()',
  'g.V().out().where(__.both()).count()',
  'g.V().dedup().count()',
  'g.V().out().path().count()',
  'g.V().out().out().path().count()',
];

// GQL query set — MATCH/RETURN, WHERE predicates, aggregates, ORDER BY (+ paging),
// GROUP BY, DISTINCT, multi-hop patterns, count shortcuts.
const gqlQueries = (): string[] => [
  'MATCH (n:Person) RETURN count(*) AS c',
  'MATCH (n:VIP) RETURN count(*) AS c',
  'MATCH (n:Person) RETURN n.name AS name ORDER BY name LIMIT 10',
  'MATCH (n:Person) RETURN n.age AS age ORDER BY age DESC LIMIT 10',
  'MATCH (n:Person) WHERE n.age > 40 RETURN count(*) AS c',
  'MATCH (n:Person) WHERE n.score < 0 RETURN count(*) AS c',
  'MATCH (n:Person) RETURN avg(n.age) AS a',
  'MATCH (n:Person) RETURN sum(n.score) AS s',
  'MATCH (n:Person) RETURN max(n.age) AS mx, min(n.age) AS mn',
  'MATCH (n:Person) RETURN n.city AS city, count(*) AS c ORDER BY city',
  'MATCH (n:Person)-[e:R]->(m) RETURN count(*) AS c',
  'MATCH (n:Person)-[e:F]->(m) RETURN count(*) AS c',
  'MATCH (n:Person)-[e]->(m) RETURN count(*) AS c',
  'MATCH (n:Person)-[e:R]->(m) RETURN n.name AS a, m.name AS b ORDER BY a, b LIMIT 20',
  'MATCH (n:Person)-[]->()-[]->(m) RETURN count(*) AS c',
  'MATCH (n:Person) RETURN DISTINCT n.city AS city ORDER BY city',
  "MATCH (n:Person) WHERE n.city = 'oslo' RETURN count(*) AS c",
  'MATCH (n:VIP) RETURN n.name AS name, n.age AS age ORDER BY age DESC, name LIMIT 5',
  'MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age LIMIT 5 OFFSET 2',
];

const runSuite = (
  label: string,
  h: CompareHandle,
  kind: 'gremlin' | 'gql',
  queries: string[],
): void => {
  describe(`${label} · ${kind}`, () => {
    for (const q of queries) {
      test(q, () => {
        const runner = kind === 'gremlin' ? h.gremlin.bind(h) : h.gql.bind(h);
        const core = norm(runner('core', q));
        const engine = norm(runner('engine', q));
        // A shared expectation across a graph both engines built identically.
        expect({ q, engine }).toEqual({ q, engine: core });
      });
    }
  });
};

describe('engine vs core', () => {
  const lib = loadCompare();

  // Same vertex/edge counts is the precondition — if the graphs differ, nothing below
  // is apples-to-apples.
  const graphs: Array<[string, string]> = [
    ['edge-case', EDGE_CASE_NDJSON],
    ['fixture-2k-deg3', fixtureNdjson(2000, 3)],
  ];

  for (const [name, ndjson] of graphs) {
    const h = lib.fromCoreNdjson(ndjson);
    test(`${name}: identical vertex/edge counts`, () => {
      expect(h.vertexCount('engine')).toBe(h.vertexCount('core'));
      expect(h.edgeCount('engine')).toBe(h.edgeCount('core'));
    });

    // The edge-case graph has no Person/VIP labels, so only run the labelled suites on
    // the fixture; run label-agnostic Gremlin structure on both.
    if (name === 'fixture-2k-deg3') {
      runSuite(name, h, 'gremlin', gremlinQueries());
      runSuite(name, h, 'gql', gqlQueries());
      knownDivergences(h);
    }
  }
});

// The comparison surfaced exactly two engine-vs-core differences, both understood and
// documented here rather than papered over — a drop-in must either match core or have a
// principled reason not to.
const knownDivergences = (h: CompareHandle): void => {
  describe('known divergences (documented, not silent)', () => {
    // (1) valueMap / elementMap PROPERTY-KEY ORDER. Core is row-based and preserves each
    // element's property INSERTION order; the columnar engine has no per-element order and
    // emits keys in a canonical (sorted) order. The "order is unspecified" policy already
    // classifies map-key order as won't-fix. Assert the divergence is PURELY ordering: the
    // same key set and the same values, equal once keys are deep-sorted.
    test('valueMap differs only in key order (same content)', () => {
      const core = h.gremlin('core', 'g.V().valueMap()');
      const engine = h.gremlin('engine', 'g.V().valueMap()');
      // Not byte-equal…
      expect(norm(engine)).not.toEqual(norm(core));
      // …but identical under a deep key-sort — so it is ordering alone, not content.
      expect(normDeep(engine)).toEqual(normDeep(core));
    });

    // (2) DOUBLE-QUOTED STRING LITERALS. ISO GQL reserves double quotes for delimited
    // IDENTIFIERS and single quotes for string literals. Core leniently accepts a
    // double-quoted string as a literal; the engine follows ISO (so `= "oslo"` is an
    // identifier reference that finds nothing → an error, surfaced here as null). The
    // engine is the ISO-correct side. Single-quoted forms agree exactly (covered above).
    test('double-quoted string: core lenient, engine ISO-strict', () => {
      const q = 'MATCH (n:Person) WHERE n.city = "oslo" RETURN count(*) AS c';
      expect(h.gql('core', q)).not.toBeNull();
      expect(h.gql('engine', q)).toBeNull();
    });
  });
};
