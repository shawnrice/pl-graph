import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, count, fold, hasLabel, in_, out, pipe, unfold, union, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('union tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  // doc: g.V().union(fold(),fold()).unfold().values('name')
  // — marko;marko;vadas;vadas;lop;lop;josh;josh;ripple;ripple;peter;peter
  // (each vertex appears twice via two fold() branches).
  test('union runs each sub-plan and merges outputs', () => {
    const r = arr(
      run(traversal(V(), union(fold(), fold()), unfold(), values('name')), tinkerGraph),
    );
    // union() runs each arm over the WHOLE stream and concatenates ARM-BY-ARM: the first
    // fold() folds all 6 vertices into one list, the second folds all 6 again, so unfold()
    // then values('name') yields the 6 names in V() order, TWICE (arm 1's, then arm 2's) —
    // not interleaved per vertex.
    expect(r).toEqual([
      'marko',
      'vadas',
      'josh',
      'peter',
      'lop',
      'ripple',
      'marko',
      'vadas',
      'josh',
      'peter',
      'lop',
      'ripple',
    ]);
  });

  // legacy: g.V('4').union(in_(), out()).values('age', 'lang') — 29, java, java
  // josh's in: marko (age 29). josh's out: ripple, lop (lang java).
  test('union of in_() and out() emits both', () => {
    const r = arr(run(traversal(V('4'), union(in_(), out()), values('age', 'lang')), tinkerGraph));
    expect(r).toEqual([29, 'java', 'java']);
  });

  // doc: g.V(1,4).union(out().values('name'), in_().values('name')) — names
  // of out-neighbors and in-neighbors merged per starting vertex.
  test('union of out and in flattens both branches', () => {
    const r = arr(
      run(
        traversal(V('1', '4'), union(pipe(out(), values('name')), pipe(in_(), values('name')))),
        tinkerGraph,
      ),
    );
    // marko(1): out -> {vadas, josh, lop}; in -> {} (no incoming).
    // josh(4): out -> {ripple, lop}; in -> {marko}.
    expect((r as string[]).sort()).toEqual(['josh', 'lop', 'lop', 'marko', 'ripple', 'vadas']);
  });

  // g.V(1,4).union(out().count(), in_().count()) — WHOLE-STREAM counts: each arm's count()
  // reduces the whole {marko, josh} stream, not per vertex, matching TinkerPop.
  test('union terminal counts reduce the whole stream, not per element', () => {
    const r = arr(
      run(traversal(V('1', '4'), union(pipe(out(), count()), pipe(in_(), count()))), tinkerGraph),
    );
    // out() over {marko, josh} = 3 + 2 = 5; in() over {marko, josh} = 0 + 1 = 1.
    expect(r).toEqual([5, 1]);
  });

  // doc: parent traversal continues after union; output of union feeds the
  // next step. Here: g.V(1,4).union(out(), in_()).hasLabel('PERSON').values('name').
  test('union output feeds the parent traversal', () => {
    const r = arr(
      run(
        traversal(V('1', '4'), union(out(), in_()), hasLabel('PERSON'), values('name')),
        tinkerGraph,
      ),
    );
    // marko: out PERSON = {vadas, josh}; in = {}; josh: out PERSON = {}; in = {marko}.
    expect((r as string[]).sort()).toEqual(['josh', 'marko', 'vadas']);
  });
});
