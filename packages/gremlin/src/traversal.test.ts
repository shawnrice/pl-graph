import { describe, expect, test } from 'bun:test';

import { V, count, dedupe, gt, has, out, repeat, simplePath, take, traversal } from './index.js';

describe('traversal AST', () => {
  test('builds a flat plan', () => {
    const plan = traversal(V(1), out('knows'), has('age', gt(30)), take(5));

    expect(plan).toEqual({
      steps: [
        { kind: 'V', ids: [1] },
        { kind: 'out', labels: ['knows'] },
        { kind: 'has', key: 'age', pred: { op: 'gt', value: 30 } },
        { kind: 'take', n: 5 },
      ],
    });
  });

  test('V() with no args has no ids', () => {
    const plan = traversal(V(), count());
    expect(plan.steps[0]).toEqual({ kind: 'V', ids: undefined });
  });

  test('repeat builds a body plan and supports modifiers', () => {
    const plan = traversal(V(1), repeat(out('knows')).times(3), dedupe());

    expect(plan.steps).toEqual([
      { kind: 'V', ids: [1] },
      {
        kind: 'repeat',
        body: { steps: [{ kind: 'out', labels: ['knows'] }] },
        times: 3,
      },
      { kind: 'dedupe', labels: undefined, bys: undefined },
    ]);
  });

  test('repeat composes body, until, and emit', () => {
    const plan = traversal(
      V(1),
      repeat(out('knows'))
        .until(has('label', gt('user')))
        .emit(simplePath()),
    );

    const [, repeatStep] = plan.steps;
    expect(repeatStep.kind).toBe('repeat');

    if (repeatStep.kind !== 'repeat') {
      return;
    }

    expect(repeatStep.body.steps).toEqual([{ kind: 'out', labels: ['knows'] }]);
    expect(repeatStep.until?.steps).toEqual([
      { kind: 'has', key: 'label', pred: { op: 'gt', value: 'user' } },
    ]);
    expect(repeatStep.emit?.steps).toEqual([{ kind: 'simplePath' }]);
  });

  test('AST is JSON-serializable end-to-end', () => {
    const plan = traversal(V(1), out('knows'), has('age', gt(30)), take(5));
    // eslint-disable-next-line unicorn/prefer-structured-clone -- the assertion is specifically about JSON-serializability, not deep cloning
    const roundTripped = JSON.parse(JSON.stringify(plan));
    expect(roundTripped).toEqual(plan);
  });
});
