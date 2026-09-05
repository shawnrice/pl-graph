// The `sack` — per-traverser local data (TinkerPop). LAZY: `withSack` records
// the default on the context; a traverser's `sack` stays `undefined` until a
// `sack(op)` write allocates one, and a read before that returns the default.
// Split-on-branch is the clone `extend`/`{...t}` already do; OLAP merge is out
// of scope (this engine is OLTP).

import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { By, SackOp } from '../ast.js';
import { byOr, evalBy } from './runtime.js';
import { extend, type RunContext, type Traverser } from './runtime.js';

export const withSackStep = function* (
  stream: Iterable<Traverser<unknown>>,
  init: unknown,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  ctx.sackInit = { value: init };

  yield* stream;
};

const num = (v: unknown): number | null => (typeof v === 'number' ? v : null);

// Merge a projected value into the sack: `newSack = op(currentSack, projected)`.
// Byte-identical to native `apply_sack_op`: a non-numeric fold yields null, and
// min/max use `<=`/`>=` so NaN resolves the same way in both engines.
const applySackOp = (op: SackOp, current: unknown, projected: unknown): unknown => {
  if (op === 'assign') {
    return projected;
  }

  const a = num(current);
  const b = num(projected);

  if (a === null || b === null) {
    return null;
  }

  switch (op) {
    case 'sum':
      return a + b;
    case 'mult':
      return a * b;
    case 'min':
      return a <= b ? a : b;
    case 'max':
      return a >= b ? a : b;
  }
};

export const sackStep = function* (
  stream: Iterable<Traverser<unknown>>,
  op: SackOp | undefined,
  bys: readonly By[] | undefined,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  const cfg = ctx.sackInit;

  if (!cfg) {
    throw new LenkeError('sack() requires a preceding withSack()', {
      code: ErrorCode.InvalidGraphOp,
    });
  }

  const by = bys?.[0];

  for (const t of stream) {
    const current = t.sack === undefined ? cfg.value : t.sack;

    if (op === undefined) {
      // `sack()` read — emit the sack as the traverser's value.
      yield extend(t, current);
    } else {
      // `sack(op).by(proj)` write — update the sack, pass the traverser through.
      // A no-value by() coerces to `undefined` (unchanged from before NO_VALUE).
      const projected = by ? byOr(evalBy(by, t.value, graph, ctx)) : t.value;

      yield { ...t, sack: applySackOp(op, current, projected) };
    }
  }
};
