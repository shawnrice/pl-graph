import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { findClosures, isSerializable, serialize } from '../serialize.js';
import { V, filter, flatMap, fold, hasLabel, map, pipe, sideEffect, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('closure-bearing step variants', () => {
  const tinkerGraph = createTestTinkerGraph();

  test('map with closure projects each value', () => {
    const r = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          map((v: unknown) =>
            (v as { properties: { name: string } }).properties.name.toUpperCase(),
          ),
        ),
        tinkerGraph,
      ),
    );
    expect(r.sort()).toEqual(['JOSH', 'MARKO', 'PETER', 'VADAS']);
  });

  test('map with sub-plan still works (dispatch)', () => {
    const r = arr(run(traversal(V(), hasLabel('PERSON'), map(pipe(values('name')))), tinkerGraph));
    expect(r.sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });

  test('filter with closure', () => {
    const r = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          filter((v: unknown) => (v as { properties: { age: number } }).properties.age > 30),
          values('name'),
        ),
        tinkerGraph,
      ),
    );
    expect(r.sort()).toEqual(['josh', 'peter']);
  });

  test('flatMap with closure', () => {
    const r = arr(
      run(
        traversal(
          V('1'),
          flatMap((v: unknown) => {
            const props = (v as { properties: Record<string, unknown> }).properties;

            return Object.entries(props).map(([k, val]) => `${k}=${String(val)}`);
          }),
        ),
        tinkerGraph,
      ),
    );
    expect(r.sort()).toEqual(['age=29', 'name=marko']);
  });

  test('sideEffect with closure runs and passes through', () => {
    const seen: string[] = [];
    const r = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          sideEffect((v: unknown) =>
            seen.push((v as { properties: { name: string } }).properties.name),
          ),
          values('name'),
        ),
        tinkerGraph,
      ),
    );
    expect(seen.sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
    expect((r as string[]).sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });

  test('fold with seed and reducer (closure form)', () => {
    const r = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          values('age'),
          fold(0, (acc, v) => (acc as number) + (v as number)),
        ),
        tinkerGraph,
      ),
    );
    expect(r).toEqual([29 + 27 + 32 + 35]);
  });

  test('fold() without args still produces an array', () => {
    const r = arr(run(traversal(V(), hasLabel('PERSON'), values('name'), fold()), tinkerGraph));
    expect(r).toHaveLength(1);
    expect((r[0] as string[]).sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
  });
});

describe('serialize', () => {
  test('pure plans round-trip through JSON', () => {
    const plan = traversal(V(), hasLabel('PERSON'), values('name'));
    expect(isSerializable(plan)).toBe(true);
    const s = serialize(plan);
    const parsed = JSON.parse(s);
    expect(parsed).toEqual(plan);
  });

  test('plans with closure-bearing steps are NOT serializable', () => {
    const plan = traversal(
      V(),
      map((v: unknown) => v),
    );
    expect(isSerializable(plan)).toBe(false);
    expect(findClosures(plan)).toEqual(['0.V', '1.mapFn'].filter((s) => s.endsWith('mapFn')));
    expect(() => serialize(plan)).toThrow(/closure-bearing/);
  });

  test('findClosures finds closures in nested sub-plans', () => {
    const plan = traversal(V(), filter(pipe(map((v: unknown) => v))));
    const closures = findClosures(plan);
    expect(closures.length).toBeGreaterThan(0);
  });
});
