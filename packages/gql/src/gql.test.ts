import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import type { MatchClause, Query, Statement } from './ast.js';
import { createFinancialGraph } from './fixtures/createFinancialGraph.js';
import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { compile, gql, parseQuery, prepare, query } from './index.js';

/** The first clause of a query's first linear part is its MATCH (in these tests). */
const firstMatch = (q: Statement): MatchClause => (q as Query).parts[0].clauses[0] as MatchClause;

const g = createTestSocialGraph();
const names = (rows: { [k: string]: unknown }[], col: string): unknown[] =>
  rows.map((r) => r[col]).sort();

describe('GQL: MATCH / WHERE / RETURN', () => {
  test('the motivating example: friends-of older people', () => {
    const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN b.name`);
    // Only josh (32) and peter (35) are over 30, but only josh has KNOWS
    // out-edges... actually marko (29) is the one who KNOWS. So nobody over 30
    // has an outgoing KNOWS — expect empty.
    expect(rows).toEqual([]);
  });

  test('all KNOWS targets', () => {
    const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('WHERE on the source binds correctly', () => {
    const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'marko' RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('CREATED edges to software', () => {
    const rows = query(g, `MATCH (p:Person)-[:CREATED]->(s:Software) RETURN p.name, s.name`);
    expect(rows).toHaveLength(4);
    expect(names(rows, 's.name')).toEqual(['lop', 'lop', 'lop', 'ripple']);
  });

  test('RETURN DISTINCT', () => {
    const rows = query(g, `MATCH (p:Person)-[:CREATED]->(s:Software) RETURN DISTINCT s.name`);
    expect(names(rows, 's.name')).toEqual(['lop', 'ripple']);
  });

  test('incoming direction <-', () => {
    const rows = query(
      g,
      `MATCH (s:Software)<-[:CREATED]-(p:Person) WHERE s.name = 'ripple' RETURN p.name`,
    );
    expect(names(rows, 'p.name')).toEqual(['josh']);
  });

  test('two-hop pattern', () => {
    const rows = query(
      g,
      `MATCH (a:Person)-[:KNOWS]->(b:Person)-[:CREATED]->(s:Software) RETURN a.name, s.name`,
    );
    // marko KNOWS josh; josh CREATED ripple + lop. marko KNOWS vadas (no creates).
    expect(names(rows, 's.name')).toEqual(['lop', 'ripple']);
  });

  test('AND / OR / NOT in WHERE', () => {
    const rows = query(g, `MATCH (p:Person) WHERE p.age >= 29 AND p.age < 33 RETURN p.name`);
    expect(names(rows, 'p.name')).toEqual(['josh', 'marko']);
  });

  test('AS alias and LIMIT', () => {
    const rows = query(g, `MATCH (p:Person) RETURN p.name AS who LIMIT 2`);
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => 'who' in r)).toBe(true);
  });

  test('comma-joined patterns share a variable', () => {
    const rows = query(
      g,
      `MATCH (a:Person)-[:KNOWS]->(b), (a)-[:CREATED]->(s) RETURN a.name, b.name, s.name`,
    );
    // Only marko both KNOWS someone and CREATED something.
    expect(rows.every((r) => r['a.name'] === 'marko')).toBe(true);
    expect(rows).toHaveLength(2); // {vadas,josh} x {lop}
  });

  test('tagged-template form', () => {
    const rows = gql(g)`MATCH (p:Person) WHERE p.name = 'vadas' RETURN p.age`;
    expect(rows).toEqual([{ 'p.age': 27 }]);
  });

  test('template substitutions bind as params, never spliced text', () => {
    const name = 'vadas';
    const rows = gql(g)`MATCH (p:Person) WHERE p.name = ${name} RETURN p.age`;
    expect(rows).toEqual([{ 'p.age': 27 }]);

    // Injection-shaped values are inert data: no parse, no clause smuggling.
    const hostile = "' RETURN p.age //";
    expect(gql(g)`MATCH (p:Person) WHERE p.name = ${hostile} RETURN p.age`).toEqual([]);
  });

  test('string form takes an explicit bindings object', () => {
    const rows = gql(g)('MATCH (p:Person) WHERE p.name = $name RETURN p.age', { name: 'vadas' });
    expect(rows).toEqual([{ 'p.age': 27 }]);
  });
});

describe('GQL: property maps & inline WHERE', () => {
  const g = createTestSocialGraph();

  test('node property map', () => {
    const rows = query(g, `MATCH (n {name: 'marko'}) RETURN n.age`);
    expect(rows).toEqual([{ 'n.age': 29 }]);
  });

  test('node property map with label', () => {
    const rows = query(g, `MATCH (s:Software {lang: 'java'}) RETURN s.name`);
    expect(names(rows, 's.name')).toEqual(['lop', 'ripple']);
  });

  test('node property map with no match', () => {
    expect(query(g, `MATCH (n {name: 'nobody'}) RETURN n.name`)).toEqual([]);
  });

  test('edge property map', () => {
    const rows = query(g, `MATCH (a)-[:KNOWS {weight: 1}]->(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh']);
  });

  test('inline WHERE on a node', () => {
    const rows = query(g, `MATCH (n:Person WHERE n.age > 30) RETURN n.name`);
    expect(names(rows, 'n.name')).toEqual(['josh', 'peter']);
  });

  test('inline WHERE on an edge', () => {
    const rows = query(g, `MATCH (a)-[r:KNOWS WHERE r.weight > 0.5]->(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh']);
  });

  test('property map and inline WHERE combine', () => {
    const rows = query(g, `MATCH (n:Person {name: 'josh'} WHERE n.age > 30) RETURN n.age`);
    expect(rows).toEqual([{ 'n.age': 32 }]);
  });

  test('empty property map matches anything', () => {
    const rows = query(g, `MATCH (n:Person {}) RETURN n.name`);
    expect(rows).toHaveLength(4);
  });

  test('property value can reference an earlier binding', () => {
    // edges where the target software name equals... contrived: match a person
    // who created lop, bound by name.
    const rows = query(g, `MATCH (a:Person)-[:CREATED]->(s {name: 'lop'}) RETURN a.name`);
    expect(names(rows, 'a.name')).toEqual(['josh', 'marko', 'peter']);
  });
});

describe('GQL: expressions & three-valued logic', () => {
  const g = createTestSocialGraph();

  test('NOT of a null comparison is UNKNOWN (excluded), not TRUE', () => {
    // The conformance bug: ISO yields 0 rows here.
    expect(query(g, `MATCH (n:Person) WHERE NOT (n.foo = 1) RETURN n.name`)).toEqual([]);
  });

  test('IS NULL / IS NOT NULL', () => {
    expect(names(query(g, `MATCH (n:Person) WHERE n.foo IS NULL RETURN n.name`), 'n.name')).toEqual(
      ['josh', 'marko', 'peter', 'vadas'],
    );
    expect(query(g, `MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.name`)).toHaveLength(4);
  });

  test('IN and NOT IN', () => {
    expect(
      names(query(g, `MATCH (n:Person) WHERE n.name IN ['marko', 'josh'] RETURN n.name`), 'n.name'),
    ).toEqual(['josh', 'marko']);
    expect(
      names(query(g, `MATCH (n:Person) WHERE n.name NOT IN ['marko'] RETURN n.name`), 'n.name'),
    ).toEqual(['josh', 'peter', 'vadas']);
  });

  test('XOR', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WHERE (n.age > 30) XOR (n.name = 'marko') RETURN n.name`,
    );
    expect(names(rows, 'n.name')).toEqual(['josh', 'marko', 'peter']);
  });

  test('arithmetic with precedence', () => {
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN n.age + 1 AS x`)).toEqual([{ x: 30 }]);
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN 1 + 2 * 3 AS x`)).toEqual([{ x: 7 }]);
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN -n.age AS x`)).toEqual([{ x: -29 }]);
  });

  test('string concatenation ||', () => {
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN n.name || '!' AS x`)).toEqual([
      { x: 'marko!' },
    ]);
  });

  test('arithmetic with null is null', () => {
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN n.foo + 1 AS x`)).toEqual([
      { x: null },
    ]);
  });
});

describe('GQL: RETURN *, ORDER BY, SKIP/OFFSET', () => {
  const g = createTestSocialGraph();

  test('RETURN * returns all bound variables', () => {
    const rows = query(g, `MATCH (n:Person {name: 'marko'}) RETURN *`);
    expect(rows).toHaveLength(1);
    expect(Object.keys(rows[0])).toEqual(['n']);
  });

  // A returned node/edge is a live element in the pure-TS engine (so consumers
  // keep methods/traversal), but its SERIALIZATION is byte-identical to the
  // native engine's plain `{id, labels, properties}` object: top-level field
  // order fixed, labels and property keys sorted. (Cross-engine parity across
  // the JSON/sync boundary is pinned by native/src/gql-conformance.test.ts.)
  test('a returned node serializes to a rich {id, labels, properties} object, keys sorted', () => {
    const rows = query(g, `MATCH (n:Person {name: 'marko'}) RETURN n`);
    expect(JSON.stringify(rows[0])).toBe(
      `{"n":{"id":"1","labels":["Person"],"properties":{"age":29,"name":"marko"}}}`,
    );
  });

  test('a returned edge serializes to a rich {id, from, to, labels, properties} object', () => {
    const rows = query(
      g,
      `MATCH (:Person {name: 'marko'})-[r:KNOWS]->(:Person {name: 'josh'}) RETURN r`,
    );
    expect(JSON.stringify(rows[0])).toBe(
      `{"r":{"id":"8","from":"1","to":"4","labels":["KNOWS"],"properties":{"weight":1}}}`,
    );
  });

  test('ORDER BY ascending (default)', () => {
    const rows = query(g, `MATCH (n:Person) RETURN n.name ORDER BY n.age`);
    expect(rows.map((r) => r['n.name'])).toEqual(['vadas', 'marko', 'josh', 'peter']);
  });

  test('ORDER BY descending', () => {
    const rows = query(g, `MATCH (n:Person) RETURN n.name ORDER BY n.age DESC`);
    expect(rows.map((r) => r['n.name'])).toEqual(['peter', 'josh', 'marko', 'vadas']);
  });

  test('ORDER BY an alias', () => {
    const rows = query(g, `MATCH (n:Person) RETURN n.name AS who, n.age AS a ORDER BY a DESC`);
    expect(rows.map((r) => r['who'])).toEqual(['peter', 'josh', 'marko', 'vadas']);
  });

  test('SKIP and LIMIT', () => {
    const rows = query(g, `MATCH (n:Person) RETURN n.name ORDER BY n.age SKIP 1 LIMIT 2`);
    expect(rows.map((r) => r['n.name'])).toEqual(['marko', 'josh']);
  });

  test('OFFSET is a synonym for SKIP', () => {
    const rows = query(g, `MATCH (n:Person) RETURN n.name ORDER BY n.age OFFSET 2`);
    expect(rows.map((r) => r['n.name'])).toEqual(['josh', 'peter']);
  });
});

describe('GQL: aggregation', () => {
  const g = createTestSocialGraph();

  test('count(*)', () => {
    expect(query(g, `MATCH (n:Person) RETURN count(*) AS c`)).toEqual([{ c: 4 }]);
  });

  test('count(*) over no matches is 0', () => {
    expect(query(g, `MATCH (n:Robot) RETURN count(*) AS c`)).toEqual([{ c: 0 }]);
  });

  test('sum / avg / min / max', () => {
    expect(query(g, `MATCH (n:Person) RETURN sum(n.age) AS s`)).toEqual([{ s: 123 }]);
    expect(query(g, `MATCH (n:Person) RETURN avg(n.age) AS a`)).toEqual([{ a: 30.75 }]);
    expect(query(g, `MATCH (n:Person) RETURN min(n.age) AS lo, max(n.age) AS hi`)).toEqual([
      { lo: 27, hi: 35 },
    ]);
  });

  test('sum(DURATION) computes; avg(DURATION) / sum(non-duration temporal) throw', () => {
    // sum folds component-wise (P1M10D + P2M5D = P3M15D), not a NaN → null.
    expect(
      String(query(g, `FOR d IN [DURATION 'P1M10D', DURATION 'P2M5D'] RETURN sum(d) AS s`)[0].s),
    ).toBe('P3M15D');
    // avg over durations is a loud error (needs unrepresentable duration/count).
    expect(() =>
      query(g, `FOR d IN [DURATION 'P1M10D', DURATION 'P2M5D'] RETURN avg(d) AS a`),
    ).toThrow();
    // dates aren't summable → loud error.
    expect(() =>
      query(g, `FOR d IN [DATE '2020-01-01', DATE '2020-02-01'] RETURN sum(d) AS s`),
    ).toThrow();
  });

  test('collect_list (ISO; Cypher collect is not a function here)', () => {
    const rows = query(g, `MATCH (n:Person) RETURN collect_list(n.name) AS names`);
    expect((rows[0].names as string[]).sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
    expect(() => query(g, `MATCH (n:Person) RETURN collect(n.name) AS names`)).toThrow();
  });

  test('implicit grouping', () => {
    const rows = query(
      g,
      `MATCH (:Person)-[:CREATED]->(s:Software) RETURN s.name, count(*) AS c ORDER BY s.name`,
    );
    expect(rows).toEqual([
      { 's.name': 'lop', c: 3 },
      { 's.name': 'ripple', c: 1 },
    ]);
  });

  test('count(DISTINCT …)', () => {
    const rows = query(g, `MATCH (:Person)-[:CREATED]->(s) RETURN count(DISTINCT s.name) AS c`);
    expect(rows).toEqual([{ c: 2 }]);
  });

  test('scalar function (upper)', () => {
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN upper(n.name) AS u`)).toEqual([
      { u: 'MARKO' },
    ]);
  });
});

describe('GQL: OPTIONAL MATCH & WITH', () => {
  const g = createTestSocialGraph();

  test('OPTIONAL MATCH keeps unmatched rows with nulls', () => {
    const rows = query(
      g,
      `MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name`,
    );
    // marko → vadas, josh; vadas/josh/peter → null.
    expect(rows).toHaveLength(5);
    const friends = rows.map((r) => r['b.name']).filter(Boolean) as string[];
    expect(friends.sort()).toEqual(['josh', 'vadas']);
  });

  test('WITH projects and chains a WHERE', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WITH n.age AS age WHERE age > 30 RETURN age ORDER BY age`,
    );
    expect(rows.map((r) => r['age'])).toEqual([32, 35]);
  });

  test('WITH carries an element into the next MATCH', () => {
    const rows = query(
      g,
      `MATCH (a:Person {name: 'marko'}) WITH a MATCH (a)-[:KNOWS]->(b) RETURN b.name`,
    );
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('WITH aggregation then filter (HAVING-style)', () => {
    const rows = query(
      g,
      `MATCH (:Person)-[:CREATED]->(s) WITH s.name AS name, count(*) AS c WHERE c > 1 RETURN name, c`,
    );
    expect(rows).toEqual([{ name: 'lop', c: 3 }]);
  });
});

describe('GQL: parameters & numeric literals', () => {
  const g = createTestSocialGraph();

  test('$param in WHERE', () => {
    const rows = query(g, `MATCH (n:Person) WHERE n.name = $name RETURN n.age`, { name: 'marko' });
    expect(rows).toEqual([{ 'n.age': 29 }]);
  });

  test('$param as a list for IN', () => {
    const rows = query(g, `MATCH (n:Person) WHERE n.name IN $names RETURN n.name`, {
      names: ['marko', 'josh'],
    });
    expect(names(rows, 'n.name')).toEqual(['josh', 'marko']);
  });

  test('hex, scientific, and underscored numbers', () => {
    const one = (lit: string) =>
      query(g, `MATCH (n:Person {name: 'marko'}) RETURN ${lit} AS x`)[0].x;
    expect(one('0xFF')).toBe(255);
    expect(one('0o17')).toBe(15);
    expect(one('0b1010')).toBe(10);
    expect(one('1e3')).toBe(1000);
    expect(one('1_000')).toBe(1000);
    expect(one('3.5e-1')).toBe(0.35);
  });
});

describe('GQL: variable-length paths', () => {
  const g = createTestSocialGraph();

  test('+ (one or more hops)', () => {
    const rows = query(g, `MATCH (a:Person {name: 'marko'})-[:KNOWS]->+(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('* includes zero hops (the start node itself)', () => {
    const rows = query(g, `MATCH (a:Person {name: 'marko'})-[:KNOWS]->*(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'marko', 'vadas']);
  });

  test('bounded {1,1}', () => {
    const rows = query(g, `MATCH (a:Person {name: 'marko'})-[:KNOWS]->{1,1}(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('{2,3} finds nothing (no 2-hop KNOWS chain)', () => {
    const rows = query(g, `MATCH (a:Person {name: 'marko'})-[:KNOWS]->{2,3}(b) RETURN b.name`);
    expect(rows).toEqual([]);
  });

  test('undirected var-length uses trail semantics (no repeated relationship)', () => {
    const rows = query(g, `MATCH (a:Person {name: 'josh'})~[:KNOWS]~{1,2}(b) RETURN b.name`);
    // josh~marko (1 hop, via the marko–josh edge), then marko~vadas (2). The walk
    // back to josh would reuse that same edge, which a trail forbids — so unlike
    // Gremlin's walk semantics, josh is not reached.
    expect(names(rows, 'b.name')).toEqual(['marko', 'vadas']);
  });
});

describe('GQL: set operations', () => {
  const g = createTestSocialGraph();

  test('UNION (distinct)', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN n.name AS x UNION MATCH (s:Software) RETURN s.name AS x`,
    );
    expect((rows.map((r) => r['x']) as string[]).sort()).toEqual([
      'josh',
      'lop',
      'marko',
      'peter',
      'ripple',
      'vadas',
    ]);
  });

  test('UNION removes duplicates; UNION ALL keeps them', () => {
    const distinct = query(
      g,
      `MATCH (n:Person {name: 'marko'}) RETURN n.name AS x UNION MATCH (n:Person {name: 'marko'}) RETURN n.name AS x`,
    );
    expect(distinct).toHaveLength(1);
    const all = query(
      g,
      `MATCH (n:Person {name: 'marko'}) RETURN n.name AS x UNION ALL MATCH (n:Person {name: 'marko'}) RETURN n.name AS x`,
    );
    expect(all).toHaveLength(2);
  });

  test('EXCEPT', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN n.name AS x EXCEPT MATCH (n:Person {name: 'marko'}) RETURN n.name AS x`,
    );
    expect((rows.map((r) => r['x']) as string[]).sort()).toEqual(['josh', 'peter', 'vadas']);
  });

  test('INTERSECT', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN n.name AS x INTERSECT MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS x`,
    );
    expect((rows.map((r) => r['x']) as string[]).sort()).toEqual(['josh', 'peter']);
  });
});

describe('GQL: delimited identifiers', () => {
  test('backtick-delimited variable and property', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) SET n.\`full name\` = 'Marko P'`);
    const rows = query(
      g,
      `MATCH (\`the node\`:Person {name: 'marko'}) RETURN \`the node\`.\`full name\` AS x`,
    );
    expect(rows).toEqual([{ x: 'Marko P' }]);
  });
});

describe('GQL: write statements', () => {
  test('INSERT a node', () => {
    const g = createTestSocialGraph();
    query(g, `INSERT (n:Person {name: 'newbie', age: 99})`);
    expect(query(g, `MATCH (n:Person {name: 'newbie'}) RETURN n.age`)).toEqual([{ 'n.age': 99 }]);
  });

  test('INSERT … RETURN binds the created node', () => {
    const g = createTestSocialGraph();
    expect(query(g, `INSERT (n:Person {name: 'z'}) RETURN n.name`)).toEqual([{ 'n.name': 'z' }]);
  });

  test('INSERT an edge between matched nodes', () => {
    const g = createTestSocialGraph();
    query(
      g,
      `MATCH (a:Person {name: 'marko'}), (b:Person {name: 'peter'}) INSERT (a)-[:KNOWS]->(b)`,
    );
    const rows = query(g, `MATCH (:Person {name: 'marko'})-[:KNOWS]->(b) RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'peter', 'vadas']);
  });

  test('INSERT rejects an ambiguous node label / a typeless or disjunction edge', () => {
    const g = createTestSocialGraph();
    // A non-conjunction node label can't be created (which one?).
    expect(() => query(g, `INSERT (a:Foo|Bar)`)).toThrow(/conjunction|not creatable/);
    expect(() => query(g, `INSERT (a:!Foo)`)).toThrow();
    // An edge must carry exactly one type.
    expect(() => query(g, `INSERT (a)-[r]->(b)`)).toThrow(); // typeless
    expect(() => query(g, `INSERT (a)-[:A|B]->(b)`)).toThrow(); // disjunction type
    // A conjunction node and an unlabelled node stay valid.
    query(g, `INSERT (a:Foo&Bar)`);
    query(g, `INSERT (a)`);
  });

  test('SET a property', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) SET n.age = 30`);
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN n.age`)).toEqual([{ 'n.age': 30 }]);
  });

  test('SET a label', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) SET n:Verified`);
    expect(query(g, `MATCH (n:Verified) RETURN n.name`)).toEqual([{ 'n.name': 'marko' }]);
  });

  test('REMOVE a property', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) REMOVE n.age`);
    expect(query(g, `MATCH (n:Person {name: 'marko'}) WHERE n.age IS NULL RETURN n.name`)).toEqual([
      { 'n.name': 'marko' },
    ]);
  });

  test('DETACH DELETE a node', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) DETACH DELETE n`);
    expect(query(g, `MATCH (n:Person) RETURN count(*) AS c`)).toEqual([{ c: 3 }]);
  });
});

describe('GQL: parser', () => {
  test('parses direction into the AST', () => {
    const q = parseQuery(`MATCH (a)<-[:R]-(b) RETURN a`);
    expect(firstMatch(q).patterns[0].segments[0].rel.direction).toBe('in');
  });

  test('parses an edge label expression', () => {
    const q = parseQuery(`MATCH (a)-[:KNOWS|CREATED]->(b) RETURN a`);
    expect(firstMatch(q).patterns[0].segments[0].rel.label?.kind).toBe('or');
  });
});

describe('GQL: ISO (not Cypher) syntax', () => {
  const g = createTestSocialGraph();

  test('-- is a line comment, not an undirected edge', () => {
    const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b) -- only marko KNOWS\nRETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('// line comment', () => {
    const rows = query(g, `// find software\nMATCH (s:Software) RETURN s.name`);
    expect(names(rows, 's.name')).toEqual(['lop', 'ripple']);
  });

  test('/* block comment */', () => {
    const rows = query(g, `MATCH (s:Software) /* inline */ RETURN s.name`);
    expect(names(rows, 's.name')).toEqual(['lop', 'ripple']);
  });

  test('~ is an undirected edge', () => {
    const q = parseQuery(`MATCH (a)~[:KNOWS]~(b) RETURN a`);
    expect(firstMatch(q).patterns[0].segments[0].rel.direction).toBe('both');
  });

  test('undirected edge matches either traversal direction', () => {
    // josh is reached from marko via KNOWS; undirected should also find marko
    // from josh.
    const rows = query(g, `MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name`);
    expect(names(rows, 'b.name')).toEqual(['marko']);
  });

  test('label disjunction :A|B', () => {
    const rows = query(g, `MATCH (n:Person|Software) RETURN n.name`);
    expect(rows).toHaveLength(6); // all nodes
  });

  test('label conjunction :A&B', () => {
    // No fixture node is both Person and Software.
    const rows = query(g, `MATCH (n:Person&Software) RETURN n.name`);
    expect(rows).toEqual([]);
    // …but it parses as a conjunction.
    const q = parseQuery(`MATCH (n:Person&Software) RETURN n`);
    expect(firstMatch(q).patterns[0].start.label?.kind).toBe('and');
  });

  test('label negation :!A', () => {
    const rows = query(g, `MATCH (n:!Software) RETURN n.name`);
    expect(names(rows, 'n.name')).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });

  test('label wildcard :%', () => {
    const rows = query(g, `MATCH (n:%) RETURN n.name`);
    expect(rows).toHaveLength(6);
  });

  test('IS as the label introducer', () => {
    const rows = query(g, `MATCH (n IS Person) RETURN n.name`);
    expect(names(rows, 'n.name')).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });

  test('grouped label expression :(A|B)&!C', () => {
    const q = parseQuery(`MATCH (n:(Person|Robot)&!Software) RETURN n`);
    expect(firstMatch(q).patterns[0].start.label?.kind).toBe('and');
  });

  test('edge label disjunction [:A|B]', () => {
    const rows = query(g, `MATCH (a:Person)-[:KNOWS|CREATED]->(b) RETURN b.name`);
    // marko: KNOWS vadas, KNOWS josh, CREATED lop; josh: CREATED ripple, lop;
    // peter: CREATED lop.
    expect(rows).toHaveLength(6);
  });

  test('edge label negation [:!CREATED]', () => {
    const rows = query(g, `MATCH (a:Person)-[:!CREATED]->(b) RETURN b.name`);
    // Only the two KNOWS edges remain.
    expect(names(rows, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('colon-chained :A:B (Cypher) is rejected', () => {
    expect(() => parseQuery(`MATCH (n:Person:Software) RETURN n`)).toThrow();
  });

  test('abbreviated arrows ->, <-, <->', () => {
    expect(
      firstMatch(parseQuery(`MATCH (a)->(b) RETURN a`)).patterns[0].segments[0].rel.direction,
    ).toBe('out');
    expect(
      firstMatch(parseQuery(`MATCH (a)<-(b) RETURN a`)).patterns[0].segments[0].rel.direction,
    ).toBe('in');
    expect(
      firstMatch(parseQuery(`MATCH (a)<->(b) RETURN a`)).patterns[0].segments[0].rel.direction,
    ).toBe('both');
  });
});

describe('GQL: compile / prepare (reusable plans)', () => {
  test('a prepared plan runs without re-parsing', () => {
    const plan = prepare(`MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name`);
    expect(names(plan(g), 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('one plan, many param bindings (reentrant — no shared state)', () => {
    const plan = prepare(`MATCH (n:Person) WHERE n.name = $who RETURN n.age`);
    expect(plan(g, { who: 'marko' })).toEqual([{ 'n.age': 29 }]);
    expect(plan(g, { who: 'peter' })).toEqual([{ 'n.age': 35 }]);
    // Re-running an earlier binding still yields its own result (no carryover).
    expect(plan(g, { who: 'marko' })).toEqual([{ 'n.age': 29 }]);
  });

  test('one plan runs against independent graphs', () => {
    const plan = prepare(`MATCH (n:Person) RETURN count(*) AS c`);
    const g1 = createTestSocialGraph();
    const g2 = createTestSocialGraph();
    query(g2, `INSERT (n:Person {name: 'newbie', age: 1})`);
    expect(plan(g1)).toEqual([{ c: 4 }]);
    expect(plan(g2)).toEqual([{ c: 5 }]);
  });

  test('compile accepts a pre-parsed AST', () => {
    const plan = compile(parseQuery(`MATCH (n:Person) WHERE n.age > $min RETURN n.name`) as Query);
    expect(names(plan(g, { min: 31 }), 'n.name')).toEqual(['josh', 'peter']);
  });
});

// The running example from the GQL / SQL/PGQ research literature: a financial
// network layered over a social one. Every expected result below is computed by
// hand from `createFinancialGraph`'s fixed instance. See that fixture for the
// full data and the laundering-trail diagram.
describe('GQL: financial graph (GQL/SQL-PGQ literature example)', () => {
  const fg = createFinancialGraph();

  test('label expressions partition the two node kinds', () => {
    expect(query(fg, `MATCH (p:Person) RETURN count(*) AS n`)).toEqual([{ n: 5 }]);
    expect(query(fg, `MATCH (a:Account) RETURN count(*) AS n`)).toEqual([{ n: 4 }]);
  });

  test('implicit grouping: people per city', () => {
    const rows = query(fg, `MATCH (p:Person) RETURN p.city AS city, count(*) AS n ORDER BY city`);
    expect(rows).toEqual([
      { city: 'Berlin', n: 1 },
      { city: 'London', n: 3 },
      { city: 'Paris', n: 1 },
    ]);
  });

  test('aggregation over a numeric edge property: total money moved', () => {
    expect(
      query(
        fg,
        `MATCH (:Account)-[t:TRANSFER]->(:Account) RETURN sum(t.amount) AS total, count(*) AS n`,
      ),
    ).toEqual([{ total: 2400, n: 3 }]);
  });

  test('incoming aggregation: amount received per account', () => {
    const rows = query(
      fg,
      `MATCH (a:Account)<-[t:TRANSFER]-(:Account) RETURN a.name AS account, sum(t.amount) AS received ORDER BY account`,
    );
    // acc-dave never receives, so it isn't in the result.
    expect(rows).toEqual([
      { account: 'acc-alice', received: 500 },
      { account: 'acc-bob', received: 900 },
      { account: 'acc-carol', received: 1000 },
    ]);
  });

  test('the motivating "money-laundering" query', () => {
    // Friends in the same city who transfer to each other via a common friend
    // who lives elsewhere, with a decreasing amount along the trail.
    const rows = query(
      fg,
      `MATCH (x:Person)-[:FRIENDS]-(y:Person),
             (x)-[:OWNS]->(ax),
             (y)-[:OWNS]->(ay),
             (z:Person)-[:OWNS]->(az),
             (ax)-[t1:TRANSFER]->(az)-[t2:TRANSFER]->(ay)
       WHERE x.city = y.city AND x.city <> z.city AND t2.amount < t1.amount
       RETURN x.name AS name1, y.name AS name2`,
    );
    // alice & bob (London friends) launder via carol (Paris): 1000 then 900.
    expect(rows).toEqual([{ name1: 'alice', name2: 'bob' }]);
  });

  test('variable-length money flow follows the whole trail', () => {
    const rows = query(
      fg,
      `MATCH (s:Account {name: 'acc-dave'})-[:TRANSFER]->+(r:Account) RETURN r.name`,
    );
    // acc-dave → acc-alice → acc-carol → acc-bob.
    expect(names(rows, 'r.name')).toEqual(['acc-alice', 'acc-bob', 'acc-carol']);
  });

  test('OPTIONAL MATCH keeps the account-less person', () => {
    const rows = query(
      fg,
      `MATCH (p:Person) OPTIONAL MATCH (p)-[:OWNS]->(a) RETURN p.name, a.name`,
    );
    expect(rows).toHaveLength(5);
    const accounts = rows.map((r) => r['a.name']).filter(Boolean) as string[];
    expect(accounts.sort()).toEqual(['acc-alice', 'acc-bob', 'acc-carol', 'acc-dave']);
    // erin owns nothing → a property access on her null account column is NULL.
    expect(rows.find((r) => r['p.name'] === 'erin')!['a.name']).toBeNull();
  });

  test('WITH aggregation then HAVING-style filter: who sent ≥ 900', () => {
    const rows = query(
      fg,
      `MATCH (p:Person)-[:OWNS]->(:Account)-[t:TRANSFER]->(:Account)
       WITH p.name AS name, sum(t.amount) AS sent
       WHERE sent >= 900
       RETURN name, sent ORDER BY name`,
    );
    // alice's account sent 1000, carol's 900; dave's 500 is filtered out.
    expect(rows).toEqual([
      { name: 'alice', sent: 1000 },
      { name: 'carol', sent: 900 },
    ]);
  });

  test('undirected friendship matches from either endpoint', () => {
    // bob is only ever the *target* of a FRIENDS edge (alice→bob), yet an
    // undirected pattern still finds his friends.
    const rows = query(fg, `MATCH (:Person {name: 'bob'})-[:FRIENDS]-(f) RETURN f.name`);
    expect(names(rows, 'f.name')).toEqual(['alice', 'dave']);
  });
});

describe('GQL: ORDER BY NULLS FIRST / LAST (ISO <null ordering>)', () => {
  const g = createTestSocialGraph();
  // Software nodes (lop, ripple) have no `age`, so `n.age` is null for them.
  const ages = (q: string) => query(g, q).map((r) => r['age']);

  test('ORDER BY / min / max impose a total order across types (num < str < bool)', () => {
    const g2 = new Graph();

    for (const v of [2, 'a', 1, true, 'b']) {
      g2.addVertex({ labels: ['X'], properties: { v } });
    }

    expect(query(g2, `MATCH (n:X) RETURN n.v AS v ORDER BY n.v`).map((r) => r.v)).toEqual([
      1,
      2,
      'a',
      'b',
      true,
    ]);
    expect(query(g2, `MATCH (n:X) RETURN n.v AS v ORDER BY n.v DESC`).map((r) => r.v)).toEqual([
      true,
      'b',
      'a',
      2,
      1,
    ]);
    expect(query(g2, `MATCH (n:X) RETURN min(n.v) AS m`)[0].m).toBe(1);
    expect(query(g2, `MATCH (n:X) RETURN max(n.v) AS m`)[0].m).toBe(true);
  });

  test('default: nulls sort last in both directions', () => {
    expect(ages(`MATCH (n) RETURN n.age AS age ORDER BY n.age ASC`)).toEqual([
      27,
      29,
      32,
      35,
      null,
      null,
    ]);
    // Nulls sort LAST by default in DESC too (our pinned default).
    expect(ages(`MATCH (n) RETURN n.age AS age ORDER BY n.age DESC`)).toEqual([
      35,
      32,
      29,
      27,
      null,
      null,
    ]);
  });

  test('NULLS FIRST overrides the ascending default', () => {
    expect(ages(`MATCH (n) RETURN n.age AS age ORDER BY n.age ASC NULLS FIRST`)).toEqual([
      null,
      null,
      27,
      29,
      32,
      35,
    ]);
  });

  test('NULLS LAST overrides the descending default', () => {
    expect(ages(`MATCH (n) RETURN n.age AS age ORDER BY n.age DESC NULLS LAST`)).toEqual([
      35,
      32,
      29,
      27,
      null,
      null,
    ]);
  });

  test('NULLS / FIRST / LAST stay usable as ordinary identifiers', () => {
    // Only `NULLS FIRST|LAST` following a sort key is special (contextual parse).
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN n.name AS first`)).toEqual([
      { first: 'marko' },
    ]);
  });
});

describe('GQL: IS TRUE / FALSE / UNKNOWN (ISO <boolean test>)', () => {
  const g = createTestSocialGraph();

  test('truth-value tests collapse three-valued logic to a definite boolean', () => {
    const r = query(
      g,
      `MATCH (n:Person {name: 'marko'}) RETURN
         true IS TRUE AS a,
         (1 = 2) IS FALSE AS b,
         null IS UNKNOWN AS c,
         true IS NOT FALSE AS d,
         null IS NOT TRUE AS e,
         null IS TRUE AS f`,
    );
    expect(r).toEqual([{ a: true, b: true, c: true, d: true, e: true, f: false }]);
  });

  test('IS TRUE / IS NOT TRUE resolve UNKNOWN predicates in WHERE', () => {
    // n.foo is missing → `n.foo = 1` is UNKNOWN. IS TRUE makes that definite-false
    // (excluded); IS NOT TRUE makes it definite-true (kept).
    expect(query(g, `MATCH (n:Person) WHERE (n.foo = 1) IS TRUE RETURN n.name`)).toEqual([]);
    expect(query(g, `MATCH (n:Person) WHERE (n.foo = 1) IS NOT TRUE RETURN n.name`)).toHaveLength(
      4,
    );
  });
});

describe('GQL: CASE expression (ISO <case expression>)', () => {
  const g = createTestSocialGraph();
  // Evaluate an expression against a single, fixed row.
  const one = (e: string): unknown =>
    query(g, `MATCH (n:Person {name: 'marko'}) RETURN ${e} AS r`)[0].r;

  test('searched CASE returns the first TRUE branch', () => {
    expect(one(`CASE WHEN 1 > 2 THEN 'a' WHEN 2 > 1 THEN 'b' ELSE 'c' END`)).toBe('b');
  });

  test('searched CASE with no match falls to ELSE', () => {
    expect(one(`CASE WHEN false THEN 'a' ELSE 'z' END`)).toBe('z');
  });

  test('searched CASE with no ELSE and no match is NULL', () => {
    expect(one(`CASE WHEN false THEN 'a' END`)).toBeNull();
  });

  test('an UNKNOWN condition is not TRUE, so its branch is skipped', () => {
    // n.foo is missing → `n.foo = 1` is UNKNOWN, which is not a match.
    expect(one(`CASE WHEN n.foo = 1 THEN 'x' ELSE 'y' END`)).toBe('y');
  });

  test('simple CASE over integers (TCK Conditional2)', () => {
    const r = (v: string) =>
      one(
        `CASE ${v} WHEN -10 THEN 'minus ten' WHEN 0 THEN 'zero' WHEN 5 THEN 'five' ELSE 'else' END`,
      );
    expect(r('0')).toBe('zero');
    expect(r('5')).toBe('five');
    expect(r('-10')).toBe('minus ten');
    expect(r('42')).toBe('else');
  });

  test('simple CASE: a NULL subject never matches (→ ELSE)', () => {
    expect(one(`CASE n.foo WHEN 1 THEN 'a' ELSE 'none' END`)).toBe('none');
  });

  test('CASE drives a computed RETURN column', () => {
    const rows = query(
      g,
      `MATCH (n:Person)
       RETURN n.name AS name,
              CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS band
       ORDER BY name`,
    );
    expect(rows).toEqual([
      { name: 'josh', band: 'senior' },
      { name: 'marko', band: 'junior' },
      { name: 'peter', band: 'senior' },
      { name: 'vadas', band: 'junior' },
    ]);
  });

  test('CASE inside an aggregate (conditional sum)', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN sum(CASE WHEN n.age >= 30 THEN 1 ELSE 0 END) AS seniors`,
    );
    expect(rows).toEqual([{ seniors: 2 }]); // josh (32) and peter (35)
  });

  test('NULLIF (ISO case abbreviation)', () => {
    expect(one(`nullif(7, 7)`)).toBeNull();
    expect(one(`nullif(7, 8)`)).toBe(7);
  });
});

describe('GQL: ISO numeric & string value functions', () => {
  const g = createTestSocialGraph();
  const v = (e: string): unknown =>
    query(g, `MATCH (n:Person {name: 'marko'}) RETURN ${e} AS r`)[0].r;

  test('numeric value functions', () => {
    expect(v('abs(-5)')).toBe(5);
    expect(v('ceil(2.1)')).toBe(3);
    expect(v('ceiling(2.1)')).toBe(3);
    expect(v('floor(2.9)')).toBe(2);
    expect(v('sqrt(9)')).toBe(3);
    expect(v('power(2, 10)')).toBe(1024);
    expect(v('mod(7, 3)')).toBe(1);
    expect(v('log10(1000)')).toBe(3);
    expect(v('log(2, 8)')).toBe(3); // general log: base 2 of 8
  });

  test('trigonometric and angle conversion', () => {
    expect(v('radians(180)')).toBeCloseTo(Math.PI);
    expect(v('degrees(radians(90))')).toBeCloseTo(90);
    expect(v('sin(0)')).toBe(0);
    // atan2(y, x): quadrant-correct angle (ISO GQL binary arctangent).
    expect(v('atan2(1, 1)')).toBe(Math.PI / 4);
    expect(v('atan2(0, 1)')).toBe(0);
    expect(v('atan2(1, 0)')).toBe(Math.PI / 2);
    expect(v('atan2(null, 1)')).toBeNull();
  });

  test('string value functions', () => {
    expect(v(`char_length('hello')`)).toBe(5);
    expect(v(`character_length('hello')`)).toBe(5);
    expect(v(`upper('abc')`)).toBe('ABC');
    expect(v(`lower('ABC')`)).toBe('abc');
    expect(v(`left('hello', 2)`)).toBe('he');
    expect(v(`right('hello', 2)`)).toBe('lo');
    expect(v(`right('hi', 0)`)).toBe('');
    expect(v(`ltrim('  hi ')`)).toBe('hi ');
    expect(v(`rtrim('  hi ')`)).toBe('  hi');
    expect(v(`btrim('  hi  ')`)).toBe('hi');
  });

  test('null argument yields null', () => {
    expect(v('sqrt(null)')).toBeNull();
    expect(v('power(null, 2)')).toBeNull();
    expect(v(`left(null, 2)`)).toBeNull();
  });
});

describe('GQL: EXISTS subquery (ISO <exists predicate>)', () => {
  const g = createTestSocialGraph();

  test('EXISTS keeps rows whose correlated sub-pattern matches', () => {
    const rows = query(g, `MATCH (n:Person) WHERE EXISTS { (n)-[:CREATED]->(s) } RETURN n.name`);
    expect(names(rows, 'n.name')).toEqual(['josh', 'marko', 'peter']);
  });

  test('NOT EXISTS negates the predicate', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WHERE NOT EXISTS { (n)-[:CREATED]->(:Software) } RETURN n.name`,
    );
    expect(names(rows, 'n.name')).toEqual(['vadas']);
  });

  test('an inner WHERE in the subquery is correlated to the outer row', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WHERE EXISTS { (n)-[:KNOWS]->(f) WHERE f.age < 30 } RETURN n.name`,
    );
    // marko KNOWS vadas (27) and josh (32); vadas < 30 → marko qualifies. No one
    // else has outgoing KNOWS.
    expect(names(rows, 'n.name')).toEqual(['marko']);
  });

  test('EXISTS composes inside arbitrary boolean logic', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WHERE n.age > 34 OR EXISTS { (n)-[:KNOWS]->() } RETURN n.name`,
    );
    // peter (35) by age, marko by the EXISTS.
    expect(names(rows, 'n.name')).toEqual(['marko', 'peter']);
  });

  test('EXISTS works as a RETURN value', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN n.name AS name, EXISTS { (n)-[:CREATED]->() } AS creates ORDER BY name`,
    );
    expect(rows).toEqual([
      { name: 'josh', creates: true },
      { name: 'marko', creates: true },
      { name: 'peter', creates: true },
      { name: 'vadas', creates: false },
    ]);
  });

  test('EXISTS is a reserved word; bare use as a name is rejected', () => {
    const h = createTestSocialGraph();
    h.addVertex({ labels: ['Flag'], properties: { exists: true } });
    // A reserved word can still name a property via a delimited identifier.
    expect(query(h, 'MATCH (n:Flag) RETURN n.`exists` AS e')).toEqual([{ e: true }]);
    // Bare `exists` as a variable is a syntax error.
    expect(() => query(h, `MATCH (exists:Flag) RETURN exists`)).toThrow();
  });
});

describe('GQL: COUNT subquery (ISO count subquery)', () => {
  const g = createTestSocialGraph();

  test('COUNT { … } returns the correlated match count per row', () => {
    const rows = query(
      g,
      `MATCH (n:Person) RETURN n.name AS name, COUNT { (n)-[:CREATED]->() } AS c ORDER BY name`,
    );
    expect(rows).toEqual([
      { name: 'josh', c: 2 }, // ripple + lop
      { name: 'marko', c: 1 }, // lop
      { name: 'peter', c: 1 }, // lop
      { name: 'vadas', c: 0 }, // creates nothing
    ]);
  });

  test('COUNT subquery in WHERE', () => {
    const rows = query(
      g,
      `MATCH (n:Person) WHERE COUNT { (n)-[:CREATED]->() } > 1 RETURN n.name AS name`,
    );
    expect(names(rows, 'name')).toEqual(['josh']);
  });

  test('the count(...) aggregate and COUNT { } subquery coexist (paren vs brace)', () => {
    expect(query(g, `MATCH (n:Person) RETURN count(*) AS c`)).toEqual([{ c: 4 }]);
    // `count` is reserved; a delimited identifier still names a `count` property.
    const h = createTestSocialGraph();
    h.addVertex({ labels: ['Tally'], properties: { count: 9 } });
    expect(query(h, 'MATCH (n:Tally) RETURN n.`count` AS c')).toEqual([{ c: 9 }]);
  });
});

describe('GQL: IS LABELED predicate & ELEMENT_ID (ISO)', () => {
  const g = createTestSocialGraph();

  test('IS LABELED tests an element against a label expression', () => {
    expect(names(query(g, `MATCH (x) WHERE x IS LABELED Person RETURN x.name`), 'x.name')).toEqual([
      'josh',
      'marko',
      'peter',
      'vadas',
    ]);
    expect(query(g, `MATCH (x) WHERE x IS LABELED Software RETURN count(*) AS c`)).toEqual([
      { c: 2 },
    ]);
  });

  test('IS NOT LABELED negates the predicate', () => {
    expect(query(g, `MATCH (x) WHERE x IS NOT LABELED Person RETURN count(*) AS c`)).toEqual([
      { c: 2 }, // the two Software nodes
    ]);
  });

  test('IS LABELED accepts a boolean label expression', () => {
    const h = createTestSocialGraph();
    h.addVertex({ labels: ['Person', 'Admin'], properties: { name: 'boss' } });
    expect(
      names(query(h, `MATCH (x) WHERE x IS LABELED Person & Admin RETURN x.name`), 'x.name'),
    ).toEqual(['boss']);
    // disjunction spans both kinds: 5 Person (incl. boss) + 2 Software.
    expect(query(h, `MATCH (x) WHERE x IS LABELED Person | Software RETURN count(*) AS c`)).toEqual(
      [{ c: 7 }],
    );
  });

  // ISO COLON label-test predicate (opengql:2078): `n:Label` desugars to the same
  // predicate as `IS LABELED`, reusing the pattern parser's label-expression grammar.
  test('WHERE n:Label — COLON form, same as IS LABELED', () => {
    expect(names(query(g, `MATCH (x) WHERE x:Person RETURN x.name`), 'x.name')).toEqual(
      names(query(g, `MATCH (x) WHERE x IS LABELED Person RETURN x.name`), 'x.name'),
    );
    // a label expression after the colon works too.
    expect(query(g, `MATCH (x) WHERE x:Person|Software RETURN count(*) AS c`)).toEqual([
      { c: query(g, `MATCH (x) RETURN count(*) AS c`)[0].c },
    ]);
  });

  test('ELEMENT_ID returns the element identifier', () => {
    expect(query(g, `MATCH (n:Person {name: 'marko'}) RETURN element_id(n) AS id`)).toEqual([
      { id: '1' },
    ]);
  });
});

describe('GQL: ISO reserved words', () => {
  test('a reserved word is rejected in a BINDING position (but not as a property name)', () => {
    const g = createTestSocialGraph();
    // Binding positions — variable, alias, label — still reject a bare reserved word.
    expect(() => query(g, `MATCH (select) RETURN select`)).toThrow(/reserved word/);
    expect(() => query(g, `MATCH (n:Person) RETURN n AS count`)).toThrow();
    expect(() => query(g, `MATCH (n:Match) RETURN n`)).toThrow();
    // A property NAME is NOT a binding position: a reserved word names a property fine
    // (matches the native engine), returning null for an absent one rather than throwing.
    expect(query(g, `MATCH (n:Person) RETURN n.value AS v LIMIT 1`)).toEqual([{ v: null }]);
  });

  test('a reserved word is allowed as a delimited identifier or a function name', () => {
    const g = createTestSocialGraph();
    g.addVertex({ labels: ['Misc'], properties: { value: 7 } });
    expect(query(g, 'MATCH (n:Misc) RETURN n.`value` AS v')).toEqual([{ v: 7 }]);
    expect(query(g, `RETURN upper('x') AS u`)).toEqual([{ u: 'X' }]);
  });

  test('non-reserved words (FIRST, LAST, LABELED, TYPE) work as identifiers', () => {
    const g = createTestSocialGraph();
    g.addVertex({ labels: ['First'], properties: { last: 'z', type: 't' } });
    expect(query(g, `MATCH (first:First) RETURN first.last AS last, first.type AS type`)).toEqual([
      { last: 'z', type: 't' },
    ]);
  });

  test('COLLECT_LIST is the ISO name for the collect aggregate', () => {
    const g = createTestSocialGraph();
    const rows = query(g, `MATCH (n:Person) RETURN collect_list(n.name) AS names`);
    expect((rows[0].names as string[]).sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });
});

describe('GQL: unbound variables are a compile-time error', () => {
  const g = createTestSocialGraph();

  test('a reference to a variable no clause binds faults (not a silent null)', () => {
    // The native engine rejects this while parsing; ISO GQL / Cypher call it "variable
    // not defined". Previously pure-TS resolved it to null, hiding the typo.
    expect(() => query(g, `RETURN n.x AS y`)).toThrow();
    expect(() => query(g, `RETURN unknownVar AS y`)).toThrow();
    expect(() => query(g, `FOR v IN [n.x] RETURN v AS x`)).toThrow();
    expect(() => query(g, `MATCH (m:Person) WHERE n.age > 0 RETURN m.name AS y`)).toThrow();
    // A variable dropped by a WITH projection is out of scope afterwards.
    expect(() => query(g, `MATCH (p:Person) WITH p.name AS nm RETURN p.age AS a`)).toThrow();
  });

  test('every binding form puts its variable in scope', () => {
    // MATCH pattern var, LET, FOR alias (+ ordinality), WITH alias, and a correlated
    // subquery seeing the outer variable — none of these should fault.
    expect(query(g, `MATCH (p:Person) RETURN p.name AS n`).length).toBeGreaterThan(0);
    expect(query(g, `LET x = 1 RETURN x AS y`)).toEqual([{ y: 1 }]);
    expect(query(g, `FOR v IN [1, 2] WITH ORDINALITY i RETURN v AS a, i AS b`)).toEqual([
      { a: 1, b: 1 },
      { a: 2, b: 2 },
    ]);
    expect(query(g, `MATCH (p:Person) WITH p.name AS nm RETURN nm AS out`).length).toBeGreaterThan(
      0,
    );
    expect(
      query(g, `MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(f) } RETURN p.name AS n`).length,
    ).toBeGreaterThan(0);
    // ORDER BY may reference an output alias.
    expect(query(g, `MATCH (p:Person) RETURN p.age AS a ORDER BY a`).length).toBeGreaterThan(0);
  });
});
