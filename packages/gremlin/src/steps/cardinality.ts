import { ErrorCode, LenkeError } from '@lenke/errors';

import { appendStep, scopeTokenOf, type StepFn } from './framework.js';

// range(start, end) with end < start (end >= 0) is an invalid window. TinkerPop throws
// IllegalArgumentException at BUILD time (before any element flows, so even an unreached
// choose/coalesce arm faults) and the native engine rejects it at parse — validate here at
// construction to match, rather than only at runtime dispatch (which an unreached arm skips).
const checkRangeWindow = (start: number, end: number): void => {
  if (end >= 0 && end < start) {
    throw new LenkeError(
      `range(${start}, ${end}) is not a valid window — the high bound must be ` +
        `greater than or equal to the low bound`,
      { code: ErrorCode.InvalidValue },
    );
  }
};

// Each takes an optional first `Scope` argument. With `Scope.local`, the
// operation slices each traverser's iterable value (typical use: after
// `fold()` or on list-shaped projections); with `Scope.global` (default,
// equivalent to omitting the scope) it slices the stream of traversers.
//
// The Symbol → string-token translation happens in `scopeTokenOf` so the
// AST stays JSON-serializable.

export function take(n: number): StepFn;
export function take(scope: symbol, n: number): StepFn;
export function take(a: number | symbol, b?: number): StepFn {
  if (typeof a === 'symbol') {
    return appendStep({ kind: 'take', n: b!, scope: scopeTokenOf(a) });
  }

  return appendStep({ kind: 'take', n: a });
}

export function skip(n: number): StepFn;
export function skip(scope: symbol, n: number): StepFn;
export function skip(a: number | symbol, b?: number): StepFn {
  if (typeof a === 'symbol') {
    return appendStep({ kind: 'skip', n: b!, scope: scopeTokenOf(a) });
  }

  return appendStep({ kind: 'skip', n: a });
}

// `limit(n)` is identical to `take(n)` (TinkerPop alias).
export function limit(n: number): StepFn;
export function limit(scope: symbol, n: number): StepFn;
export function limit(a: number | symbol, b?: number): StepFn {
  if (typeof a === 'symbol') {
    return appendStep({ kind: 'take', n: b!, scope: scopeTokenOf(a) });
  }

  return appendStep({ kind: 'take', n: a });
}

export function range(start: number, end: number): StepFn;
export function range(scope: symbol, start: number, end: number): StepFn;
export function range(a: number | symbol, b: number, c?: number): StepFn {
  if (typeof a === 'symbol') {
    checkRangeWindow(b, c!);

    return appendStep({ kind: 'range', start: b, end: c!, scope: scopeTokenOf(a) });
  }

  checkRangeWindow(a, b);

  return appendStep({ kind: 'range', start: a, end: b });
}

export function tail(n?: number): StepFn;
export function tail(scope: symbol, n: number): StepFn;
export function tail(a: number | symbol = 1, b?: number): StepFn {
  if (typeof a === 'symbol') {
    return appendStep({ kind: 'tail', n: b!, scope: scopeTokenOf(a) });
  }

  return appendStep({ kind: 'tail', n: a });
}

// Random subset of N traversers (materializes the stream).
export const sample = (n: number): StepFn => appendStep({ kind: 'sample', n });
