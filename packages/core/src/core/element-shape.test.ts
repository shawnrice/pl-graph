import { describe, expect, test } from 'bun:test';

import { mathSign } from '../numeric.js';
import { isEdgeShaped, isVertexShaped } from './Element.js';
import { Graph } from './Graph.js';

// The structural guards and `sign` used to be duplicated per engine. They are
// shared now BECAUSE they must mean one thing in both, so what these pin is the
// agreement, not the implementation.
describe('shared value predicates', () => {
  test('real graph elements are recognized by shape', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['V'], properties: {} });
    const b = g.addVertex({ id: 'b', labels: ['V'], properties: {} });
    const e = g.addEdge({ id: 'e', from: a, to: b, labels: ['R'], properties: {} });

    expect(isVertexShaped(a)).toBe(true);
    expect(isEdgeShaped(a)).toBe(false);
    expect(isEdgeShaped(e)).toBe(true);
    expect(isVertexShaped(e)).toBe(false);
    expect(isVertexShaped(b)).toBe(true);
  });

  test('a vertex is anything with an id and no `from`', () => {
    // The case the two engines disagreed on: `{id, from}` with no `to`. The GQL
    // copy read "an element that is not an edge" and called it a vertex; the
    // Gremlin copy did not. The stricter reading is the shared one.
    expect(isVertexShaped({ id: 'x', from: 'y' })).toBe(false);
    expect(isEdgeShaped({ id: 'x', from: 'y' })).toBe(false);

    expect(isVertexShaped({ id: 'x' })).toBe(true);
    expect(isEdgeShaped({ from: 'a', to: 'b' })).toBe(true);
  });

  test('non-objects and empty shapes are neither', () => {
    for (const v of [null, undefined, 1, 'x', true, [], {}]) {
      expect(isVertexShaped(v)).toBe(false);
      expect(isEdgeShaped(v)).toBe(false);
    }
  });

  test('sign is -1/0/1 and passes NaN through', () => {
    // Not `Math.sign`: it returns -0 for -0, and Rust's `signum` returns +1 for
    // 0.0. Both would diverge, which is why this exists at all.
    expect(mathSign(5)).toBe(1);
    expect(mathSign(-5)).toBe(-1);
    expect(mathSign(0)).toBe(0);
    expect(Object.is(mathSign(-0), 0)).toBe(true);
    expect(Number.isNaN(mathSign(Number.NaN))).toBe(true);
    expect(mathSign(Infinity)).toBe(1);
    expect(mathSign(-Infinity)).toBe(-1);
  });
});
