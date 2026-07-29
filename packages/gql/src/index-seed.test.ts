import { describe, expect, test } from 'bun:test';

import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { query } from './index.js';

const sorted = (rows: { [k: string]: unknown }[], col: string): unknown[] =>
  rows.map((r) => r[col]).sort();

describe('GQL property-index seeding', () => {
  test('an equality property constraint returns the same rows whether or not name is indexed', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });

    const q = `MATCH (p:Person {name: 'marko'})-[:KNOWS]->(b) RETURN b.name`;
    expect(sorted(query(indexed, q), 'b.name')).toEqual(sorted(query(plain, q), 'b.name'));
    expect(sorted(query(indexed, q), 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('the label constraint still excludes a same-named wrong-label vertex', () => {
    const g = createTestSocialGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // lop is Software; seeding from the name bucket must still honor :Person.
    expect(query(g, `MATCH (p:Person {name: 'lop'}) RETURN p.name`)).toEqual([]);
    expect(query(g, `MATCH (s:Software {name: 'lop'}) RETURN s.name`)).toEqual([
      { 's.name': 'lop' },
    ]);
  });

  test('a non-indexed key still works via the scan fallback', () => {
    const g = createTestSocialGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] }); // age is NOT indexed
    const rows = query(g, `MATCH (p:Person {age: 32}) RETURN p.name`);
    expect(sorted(rows, 'p.name')).toEqual(['josh']);
  });

  test('seeding reflects live mutations', () => {
    const g = createTestSocialGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    g.addVertex({ id: '99', labels: ['Person'], properties: { name: 'marko', age: 50 } });
    const rows = query(g, `MATCH (p:Person {name: 'marko'}) RETURN p.age`);
    expect(sorted(rows, 'p.age')).toEqual([29, 50]);
  });

  test('an empty bucket yields no rows', () => {
    const g = createTestSocialGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    expect(query(g, `MATCH (p:Person {name: 'nobody'}) RETURN p.name`)).toEqual([]);
  });
});

describe('GQL WHERE-derived seed hints', () => {
  // Ages: marko=29, vadas=27, josh=32, peter=35.
  const both = (q: string) => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });

    return { plain: query(plain, q), indexed: query(indexed, q) };
  };

  test('WHERE equality seeds and matches the scan', () => {
    const { plain, indexed } = both(`MATCH (p:Person) WHERE p.name = 'marko' RETURN p.age`);
    expect(indexed).toEqual(plain);
    expect(indexed).toEqual([{ 'p.age': 29 }]);
  });

  test('WHERE range seeds and matches the scan', () => {
    const { plain, indexed } = both(`MATCH (p:Person) WHERE p.age > 30 RETURN p.name`);
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh', 'peter']);
  });

  test('a two-sided WHERE range works (each bound is a sound conjunct)', () => {
    const { plain, indexed } = both(
      `MATCH (p:Person) WHERE p.age >= 29 AND p.age < 35 RETURN p.name`,
    );
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh', 'marko']);
  });

  test('flipped comparison (const on the left) seeds too', () => {
    const { plain, indexed } = both(`MATCH (p:Person) WHERE 30 < p.age RETURN p.name`);
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh', 'peter']);
  });

  test('WHERE IN seeds from a union and matches the scan', () => {
    const { plain, indexed } = both(
      `MATCH (p:Person) WHERE p.name IN ['marko', 'josh'] RETURN p.age`,
    );
    expect(sorted(indexed, 'p.age')).toEqual(sorted(plain, 'p.age'));
    expect(sorted(indexed, 'p.age')).toEqual([29, 32]);
  });

  test('an OR predicate is NOT seeded (would miss a branch)', () => {
    const { plain, indexed } = both(
      `MATCH (p:Person) WHERE p.name = 'marko' OR p.age > 30 RETURN p.name`,
    );
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh', 'marko', 'peter']);
  });

  test('inline node WHERE seeds the start node', () => {
    const { plain, indexed } = both(`MATCH (p:Person WHERE p.age > 30) RETURN p.name`);
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh', 'peter']);
  });

  test('WHERE seeding still honors the rest of the pattern', () => {
    const { plain, indexed } = both(
      `MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'marko' RETURN b.name`,
    );
    expect(sorted(indexed, 'b.name')).toEqual(sorted(plain, 'b.name'));
    expect(sorted(indexed, 'b.name')).toEqual(['josh', 'vadas']);
  });

  test('multiple seekable conjuncts seed from the most selective one', () => {
    const { plain, indexed } = both(
      `MATCH (p:Person) WHERE p.age > 28 AND p.name = 'josh' RETURN p.name`,
    );
    expect(sorted(indexed, 'p.name')).toEqual(sorted(plain, 'p.name'));
    expect(sorted(indexed, 'p.name')).toEqual(['josh']);
  });

  test('an element-map equality and a WHERE range together still match the scan', () => {
    const { plain, indexed } = both(
      `MATCH (p:Person {name: 'marko'}) WHERE p.age < 30 RETURN p.age`,
    );
    expect(sorted(indexed, 'p.age')).toEqual(sorted(plain, 'p.age'));
    expect(sorted(indexed, 'p.age')).toEqual([29]);
  });
});

describe('GQL smaller-side seed selection', () => {
  // The selective constraint is on the *far* end of the pattern, so the planner
  // should seed from there and walk the relationship backwards.
  test('seeds from the selective far end and walks back (results match the scan)', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });

    const q = `MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.name = 'josh' RETURN a.name`;
    expect(sorted(query(indexed, q), 'a.name')).toEqual(sorted(query(plain, q), 'a.name'));
    expect(sorted(query(indexed, q), 'a.name')).toEqual(['marko']); // marko KNOWS josh
  });

  test('far-end element-map constraint also drives the seed side', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const q = `MATCH (a:Person)-[:KNOWS]->(b:Person {name: 'vadas'}) RETURN a.name`;
    expect(sorted(query(indexed, q), 'a.name')).toEqual(sorted(query(plain, q), 'a.name'));
    expect(sorted(query(indexed, q), 'a.name')).toEqual(['marko']); // marko KNOWS vadas
  });

  test('a variable-length segment keeps its orientation and still matches', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const q = `MATCH (a:Person)-[:KNOWS]->{1,2}(b:Person) WHERE b.name = 'josh' RETURN a.name`;
    expect(sorted(query(indexed, q), 'a.name')).toEqual(sorted(query(plain, q), 'a.name'));
  });

  test('an unlabeled start seeds the indexed far end instead of a full scan', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // `a` has no label (a whole-graph scan); seeding from b=josh avoids it.
    const q = `MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name = 'josh' RETURN a.name`;
    expect(sorted(query(indexed, q), 'a.name')).toEqual(sorted(query(plain, q), 'a.name'));
    expect(sorted(query(indexed, q), 'a.name')).toEqual(['marko']);
  });

  test('multi-hop pattern seeds from the selective end either way', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // marko -KNOWS-> josh -CREATED-> ripple/lop
    const q = `MATCH (a:Person {name: 'marko'})-[:KNOWS]->(b)-[:CREATED]->(c) RETURN c.name`;
    expect(sorted(query(indexed, q), 'c.name')).toEqual(sorted(query(plain, q), 'c.name'));
  });
});
