import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import {
  Order,
  Scope,
  T,
  V,
  fold,
  groupCount,
  hasLabel,
  order,
  project,
  tail,
  values,
} from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  // Round-11 BUG A: order().by('<key>') over project() Map rows sorts by the keyed
  // value instead of throwing "cannot order an element with an element".
  test('order().by(key) sorts project() rows by the keyed value', () => {
    const rows = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          project(['name', 'age'], ['name', 'age']),
          order().by('age'),
        ),
        tinkerGraph,
      ),
    );
    expect(rows).toEqual([
      { name: 'vadas', age: 27 },
      { name: 'marko', age: 29 },
      { name: 'josh', age: 32 },
      { name: 'peter', age: 35 },
    ]);
  });

  test('simple ordering works', () => {
    const result = arr(run(traversal(V(), values('name'), order()), tinkerGraph));
    expect(result).toEqual(['josh', 'lop', 'marko', 'peter', 'ripple', 'vadas']);
  });

  test('we can order by desc', () => {
    const result = arr(run(traversal(V(), values('name'), order({ desc: true })), tinkerGraph));
    expect(result).toEqual(['vadas', 'ripple', 'peter', 'marko', 'lop', 'josh']);
  });

  test('we can order by a property key', () => {
    const result = arr(
      run(traversal(V(), hasLabel('PERSON'), order({ key: 'age' }), values('name')), tinkerGraph),
    );
    expect(result).toEqual(['vadas', 'marko', 'josh', 'peter']);
  });

  // Same query via the modulator form — `order().by('age')` should match the
  // legacy `order({ key: 'age' })` form exactly.
  test('order().by(key) matches the legacy config form', () => {
    const result = arr(
      run(traversal(V(), hasLabel('PERSON'), order().by('age'), values('name')), tinkerGraph),
    );
    expect(result).toEqual(['vadas', 'marko', 'josh', 'peter']);
  });

  // doc: g.V().values('name').order().tail() — vadas
  test('order then tail() yields the last element', () => {
    const r = arr(run(traversal(V(), values('name'), order(), tail()), tinkerGraph));
    expect(r).toEqual(['vadas']);
  });

  // doc: g.V().values('name').order().tail(3) — peter; ripple; vadas
  test('order then tail(3) yields the last three', () => {
    const r = arr(run(traversal(V(), values('name'), order(), tail(3)), tinkerGraph));
    expect(r).toEqual(['peter', 'ripple', 'vadas']);
  });

  // doc: g.V().values('name').order().by(Order.desc) — sorted descending.
  test('order().by(Order.desc) reverses natural order', () => {
    const r = arr(run(traversal(V(), values('name'), order().by(Order.desc)), tinkerGraph));
    expect(r).toEqual(['vadas', 'ripple', 'peter', 'marko', 'lop', 'josh']);
  });

  // doc: g.V().values('name').order().by(Order.asc) — explicit asc same as default.
  test('order().by(Order.asc) matches default natural order', () => {
    const r = arr(run(traversal(V(), values('name'), order().by(Order.asc)), tinkerGraph));
    expect(r).toEqual(['josh', 'lop', 'marko', 'peter', 'ripple', 'vadas']);
  });

  // doc: g.V().hasLabel('person').order().by('age',Order.desc).values('name')
  // — peter (35), josh (32), marko (29), vadas (27)
  test('order().by(key, Order.desc) sorts by property descending', () => {
    const r = arr(
      run(
        traversal(V(), hasLabel('PERSON'), order().by('age', Order.desc), values('name')),
        tinkerGraph,
      ),
    );
    expect(r).toEqual(['peter', 'josh', 'marko', 'vadas']);
  });

  // order(Scope.local) sorts WITHIN each traverser's value, not across the stream.
  // The canonical use: rank a groupCount() Map by its counts (was silently a
  // no-op — order(Scope.local) spread a Symbol and did nothing).
  test('order(Scope.local) sorts a group Map by value desc (top-N-from-groupCount)', () => {
    const [m] = arr(
      run(traversal(V(), groupCount().by(T.label), order(Scope.local).by(Order.desc)), tinkerGraph),
    ) as Map<unknown, number>[];
    expect([...m.entries()]).toEqual([
      ['PERSON', 4],
      ['SOFTWARE', 2],
    ]);
  });

  test('order(Scope.local) sorts a folded list (asc default, desc via by)', () => {
    const asc = arr(
      run(
        traversal(V(), hasLabel('PERSON'), values('age'), fold(), order(Scope.local)),
        tinkerGraph,
      ),
    ) as number[][];
    expect(asc[0]).toEqual([27, 29, 32, 35]);

    const desc = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          values('age'),
          fold(),
          order(Scope.local).by(Order.desc),
        ),
        tinkerGraph,
      ),
    ) as number[][];
    expect(desc[0]).toEqual([35, 32, 29, 27]);
  });
});
