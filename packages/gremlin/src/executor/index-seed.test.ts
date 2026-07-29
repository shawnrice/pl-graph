import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { between, eq, gt, inside, startsWith, within } from '../predicates.js';
import { E, has, hasLabel, out, V, values } from '../steps.js';
import { traversal } from '../traversal.js';
import { seedFromIndex } from './index-seed.js';
import { run } from './index.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('index-seeded V() source', () => {
  test('equality has() returns the same result whether or not name is indexed', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });

    const plan = () => traversal(V(), has('name', 'marko'), values('age'));
    expect(arr(run(plan(), indexed))).toEqual(arr(run(plan(), plain)));
    expect(arr(run(plan(), indexed))).toEqual([29]);
  });

  test('3-arg has(label, key, value) keeps the label constraint when seeded', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // lop is SOFTWARE; the PERSON label must still exclude it.
    expect(arr(run(traversal(V(), has('PERSON', 'name', 'lop'), values('name')), g))).toEqual([]);
    expect(arr(run(traversal(V(), has('PERSON', 'name', 'marko'), values('name')), g))).toEqual([
      'marko',
    ]);
  });

  test('downstream steps still run after an index seed', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // marko -created-> lop, so out() then name should reach lop/vadas/josh.
    const result = arr(run(traversal(V(), has('name', 'marko'), out(), values('name')), g));
    expect((result as string[]).sort()).toEqual(['josh', 'lop', 'vadas']);
  });

  test('seedFromIndex drops a plain has but keeps other filters', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const plan = traversal(V(), hasLabel('PERSON'), has('name', eq('marko')));
    const seeded = seedFromIndex(plan, g, false);
    expect(seeded).not.toBeNull();
    // The consumed has('name', ...) is gone; hasLabel stays as a residual.
    expect(seeded!.steps.map((s) => s.kind)).toEqual(['hasLabel']);
    expect(arr(seeded!.stream).length).toBe(1); // only marko in the bucket
  });

  test('falls back (null) for an unindexed key', () => {
    const g = createTestTinkerGraph();
    const plan = traversal(V(), has('name', eq('marko')));
    expect(seedFromIndex(plan, g, false)).toBeNull();
  });

  test('range predicates are seeded but kept as a residual filter', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    // Ages: marko=29, vadas=27, josh=32, peter=35.
    const seeded = seedFromIndex(traversal(V(), has('age', gt(30))), g, false);
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(2); // josh, peter
    expect(seeded!.steps.map((s) => s.kind)).toEqual(['has']); // residual kept
  });

  test('range results match the unindexed scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    const cases = [gt(30), between(28, 33), inside(28, 33)];

    for (const pred of cases) {
      const plan = () => traversal(V(), has('age', pred), values('name'));
      expect((arr(run(plan(), indexed)) as string[]).sort()).toEqual(
        (arr(run(plan(), plain)) as string[]).sort(),
      );
    }
  });

  test('within is seeded from a union of buckets and dropped', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const seeded = seedFromIndex(
      traversal(V(), has('name', within('marko', 'josh', 'nobody'))),
      g,
      false,
    );
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(2); // marko, josh (nobody → empty)
    expect(seeded!.steps).toEqual([]); // exact match set → has dropped
  });

  test('startsWith is seeded from a prefix range and matches the scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // Names: marko, vadas, josh, peter, lop, ripple.
    const plan = () => traversal(V(), has('name', startsWith('r')), values('name'));
    expect((arr(run(plan(), indexed)) as string[]).sort()).toEqual(
      (arr(run(plan(), plain)) as string[]).sort(),
    );
    expect(arr(run(plan(), indexed))).toEqual(['ripple']);
  });

  test('startsWith seeds the prefix slice as a residual filter', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const seeded = seedFromIndex(traversal(V(), has('name', startsWith('m'))), g, false);
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(1); // marko
    expect(seeded!.steps.map((s) => s.kind)).toEqual(['has']); // residual kept
  });

  test('within results match the unindexed scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const plan = () => traversal(V(), has('name', within('vadas', 'josh')), values('name'));
    expect((arr(run(plan(), indexed)) as string[]).sort()).toEqual(
      (arr(run(plan(), plain)) as string[]).sort(),
    );
  });

  test('falls back (null) for a non-seedable predicate', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // `neq` has no bucket to seed from.
    const plan = traversal(V(), has('name', { op: 'neq', value: 'marko' }));
    expect(seedFromIndex(plan, g, false)).toBeNull();
  });

  test('an empty bucket short-circuits to no results', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    expect(arr(run(traversal(V(), has('name', 'nobody'), values('name')), g))).toEqual([]);
  });

  test('picks the most selective filter by count; the rest stay residual', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['lang'] });
    // lang=java → 2 (lop, ripple); name=lop → 1; name wins on count.
    const plan = traversal(V(), has('lang', eq('java')), has('name', eq('lop')));
    const seeded = seedFromIndex(plan, g, false);
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(1); // seeds from name=lop
    // The chosen exact name filter is dropped; the lang filter stays a residual.
    expect(seeded!.steps.map((s) => s.kind)).toEqual(['has']);
  });

  test('a more selective equality is chosen over a wider range', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    // name=josh → 1; age>30 → 2 (josh, peter); name wins, age stays residual.
    const plan = traversal(V(), has('name', eq('josh')), has('age', gt(30)));
    const seeded = seedFromIndex(plan, g, false);
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(1);
    expect(seeded!.steps.map((s) => s.kind)).toEqual(['has']); // age range residual
  });

  test('multi-filter results match the unindexed scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    const plan = () =>
      traversal(V(), has('age', gt(28)), has('name', startsWith('j')), values('name'));
    expect((arr(run(plan(), indexed)) as string[]).sort()).toEqual(
      (arr(run(plan(), plain)) as string[]).sort(),
    );
  });
});

describe('index-seeded E() source', () => {
  // Edge weights: 7=0.5, 8=1.0, 9=0.4, 10=1.0, 11=0.4, 12=0.2.
  test('equality has() on an indexed edge property seeds and matches the scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });

    const plan = () => traversal(E(), has('weight', 1.0));
    const got = (arr(run(plan(), indexed)) as Array<{ id: string }>).map((e) => e.id).sort();
    const want = (arr(run(plan(), plain)) as Array<{ id: string }>).map((e) => e.id).sort();
    expect(got).toEqual(want);
    expect(got).toEqual(['10', '8']);
  });

  test('seedFromIndex seeds E() from the edge property index', () => {
    const g = createTestTinkerGraph();
    g.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });
    const seeded = seedFromIndex(traversal(E(), has('weight', eq(0.4))), g, false);
    expect(seeded).not.toBeNull();
    expect(arr(seeded!.stream).length).toBe(2); // edges 9 and 11
    expect(seeded!.steps).toEqual([]); // exact eq → has dropped
  });

  test('range on an indexed edge property matches the scan', () => {
    const plain = createTestTinkerGraph();
    const indexed = createTestTinkerGraph();
    indexed.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });
    const plan = () => traversal(E(), has('weight', gt(0.5)), values('weight'));
    expect((arr(run(plan(), indexed)) as number[]).sort()).toEqual(
      (arr(run(plan(), plain)) as number[]).sort(),
    );
  });

  test('falls back (null) when the edge key is not indexed', () => {
    const g = createTestTinkerGraph();
    expect(seedFromIndex(traversal(E(), has('weight', eq(0.4))), g, false)).toBeNull();
  });
});
