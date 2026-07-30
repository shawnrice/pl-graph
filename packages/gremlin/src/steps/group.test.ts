import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, count, group, hasLabel, label, sum, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('STEP, group', () => {
  const g = createTestTinkerGraph();

  // doc: g.V().hasLabel('person').values('age').group() — group by the value itself.
  test('group by self (no keyBy/valueBy) collects ages into a Map', () => {
    const result = arr(run(traversal(V(), hasLabel('PERSON'), values('age'), group()), g));
    expect(result).toHaveLength(1);
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map).toBeInstanceOf(Map);
    expect(map.get(29)).toEqual([29]);
    expect(map.get(27)).toEqual([27]);
    expect(map.get(32)).toEqual([32]);
    expect(map.get(35)).toEqual([35]);
  });

  // doc: g.V().hasLabel('person').group().by('age').by('name') — group names by age.
  test('group by name keyed by age', () => {
    const result = arr(
      run(traversal(V(), hasLabel('PERSON'), group({ keyBy: 'age', valueBy: 'name' })), g),
    );
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map.get(29)).toEqual(['marko']);
    expect(map.get(27)).toEqual(['vadas']);
    expect(map.get(32)).toEqual(['josh']);
    expect(map.get(35)).toEqual(['peter']);
  });

  // doc: g.V().group().by('lang') — software vertices grouped by lang; non-software produce undefined keys.
  test('group by property (lang) collapses missing-key vertices into one bucket', () => {
    const result = arr(run(traversal(V(), group({ keyBy: 'lang', valueBy: 'name' })), g));
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map.get('java')).toEqual(['lop', 'ripple']);
    // Persons lack `lang`, so they all bucket under `undefined`.
    expect(map.get(undefined)).toEqual(['marko', 'vadas', 'josh', 'peter']);
  });

  // doc: g.V().group().by(label) — [software:[v[3],v[5]],person:[v[1],v[2],v[4],v[6]]]
  test('group by(label())', () => {
    const result = arr(run(traversal(V(), group().by(label())), g));
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map.get('PERSON')).toHaveLength(4);
    expect(map.get('SOFTWARE')).toHaveLength(2);
  });

  // doc: g.V().group().by(label).by('name') — [software:[lop,ripple],person:[marko,vadas,josh,peter]]
  test('group by(label()).by("name")', () => {
    const result = arr(run(traversal(V(), group().by(label()).by('name')), g));
    const map = result[0] as Map<unknown, unknown[]>;
    expect((map.get('SOFTWARE') as string[]).slice().sort()).toEqual(['lop', 'ripple']);
    expect((map.get('PERSON') as string[]).slice().sort()).toEqual([
      'josh',
      'marko',
      'peter',
      'vadas',
    ]);
  });

  // doc: g.V().group().by(label).by(count()) → {PERSON: 4, SOFTWARE: 2}
  // A reducing value-by (count/sum/min/max/mean/fold) folds over the whole group
  // as a barrier, yielding a single value per bucket — not a per-traverser list
  // of 1s (the behavior before local aggregation folded groups).
  test('group by(label()).by(count()) yields a per-bucket count', () => {
    const result = arr(run(traversal(V(), group().by(label()).by(count())), g));
    const map = result[0] as Map<unknown, number>;
    expect(map.get('PERSON')).toBe(4);
    expect(map.get('SOFTWARE')).toBe(2);
  });

  // A reducing sum over a mapped value: group ages by label, sum per bucket.
  test('group by(label()).by(traversal(values(age), sum())) sums per bucket', () => {
    const result = arr(
      run(
        traversal(
          V(),
          group()
            .by(label())
            .by(traversal(values('age'), sum())),
        ),
        g,
      ),
    );
    const map = result[0] as Map<unknown, number | null>;
    expect(map.get('PERSON')).toBe(29 + 27 + 32 + 35); // 123
    expect(map.get('SOFTWARE')).toBeNull(); // no ages → sum of empty is null
  });

  // doc: g.V().group().by('age').by('name') — [32:[josh],35:[peter],27:[vadas],29:[marko]]
  // Software vertices have no 'age', so they bucket under undefined.
  test('group by(age) valued by(name)', () => {
    const result = arr(run(traversal(V(), group({ keyBy: 'age', valueBy: 'name' })), g));
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map.get(29)).toEqual(['marko']);
    expect(map.get(27)).toEqual(['vadas']);
    expect(map.get(32)).toEqual(['josh']);
    expect(map.get(35)).toEqual(['peter']);
  });

  // doc: g.V().group().by('name').by('age') — software vertices map to [] in TinkerPop
  // because they have no age. Our impl drops the missing-age values, so software
  // names bucket to empty arrays.
  test('group by(name) valued by(age)', () => {
    const result = arr(run(traversal(V(), group({ keyBy: 'name', valueBy: 'age' })), g));
    const map = result[0] as Map<unknown, unknown[]>;
    expect(map.get('marko')).toEqual([29]);
    expect(map.get('vadas')).toEqual([27]);
    expect(map.get('josh')).toEqual([32]);
    expect(map.get('peter')).toEqual([35]);
    // Software vertices: name keys exist, but value-by 'age' yields nothing.
    expect(map.has('lop')).toBe(true);
    expect(map.has('ripple')).toBe(true);
  });
});
