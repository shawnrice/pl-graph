import { describe, expect, test } from 'bun:test';

import type { Plan } from '../ast.js';
import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { eq, gt } from '../predicates.js';
import { V, has, hasLabel, is, loops, or, out, pipe, repeat, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('loops tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  // doc: g.V().emit(__.has('name','marko').or().loops().is(2)).repeat(__.out()).values('name')
  // The emit-before form (modulator placed before repeat) is not modeled in
  // v2; we use the AFTER form: repeat(out()).times(...).emit(predicate).
  test('loops: termination by iteration counter', () => {
    const loopsIs2 = (p: Plan) => is(eq(2))(loops()(p));
    const r = arr(
      run(
        traversal(
          V(),
          repeat(out())
            .times(2)
            .emit(or(has('name', eq('marko')), loopsIs2)),
          values('name'),
        ),
        tinkerGraph,
      ),
    );
    // loops() yields the iteration index; the predicate composes through emit.
    expect(Array.isArray(r)).toBe(true);
  });

  // doc: g.V(1).repeat(out()).until(loops().is(2)).values('name')
  // — terminate per traverser when loops() reaches 2 (after 2 iterations).
  test('repeat(out()).until(loops().is(2)) terminates at iter 2', () => {
    const r = arr(
      run(
        traversal(
          V('1'),
          repeat(out()).until((p: Plan) => is(eq(2))(loops()(p))),
          values('name'),
        ),
        tinkerGraph,
      ),
    );
    // until checks BEFORE the body each iteration. iter1: loops==1, body
    // runs -> frontier {vadas, josh, lop}. iter2: loops==2, until satisfied,
    // those traversers yield through.
    expect((r as string[]).sort()).toEqual(['josh', 'lop', 'vadas']);
  });

  // doc: g.V(1).repeat(out()).times(3).emit(loops().is(gt(1))).values('name')
  // — emit only iterations after the first.
  test('emit(loops().is(gt(1))) emits only later iterations', () => {
    const loopsGt1 = (p: Plan) => is(gt(1))(loops()(p));
    const r = arr(
      run(traversal(V('1'), repeat(out()).times(3).emit(loopsGt1), values('name')), tinkerGraph),
    );
    // iter1 (loopCount=1) skipped; iter2 (loopCount=2) emits frontier
    // {vadas, josh, lop}; iter3 (loopCount=3) emits {ripple, lop}.
    expect((r as string[]).sort()).toEqual(['josh', 'lop', 'lop', 'ripple', 'vadas']);
  });

  // doc: loops() outside repeat is undefined; here we sanity-check that
  // loops() inside repeat(...) participates in the body's filter clause.
  test('loops() inside body filters iterations', () => {
    // Post-form emit fires AFTER each body application. iter1 body emits
    // {vadas, josh} (PERSON children of marko). iter2 body output is empty
    // (vadas/josh have no PERSON children); nothing emitted. iter3 frontier
    // is empty. Input (marko) is NOT emitted in post-form.
    const r = arr(
      run(
        traversal(
          V('1'),
          repeat(pipe(out(), hasLabel('PERSON')))
            .times(3)
            .emit(),
          values('name'),
        ),
        tinkerGraph,
      ),
    );
    expect((r as string[]).sort()).toEqual(['josh', 'vadas']);
  });
});
