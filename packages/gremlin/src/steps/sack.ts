// The `sack` — per-traverser local data carried along a traversal (TinkerPop).
// `withSack(init)` enables it; `sack()` reads it; `sack(op).by(proj)` merges a
// projected value in via a merge Operator. See `Operator` in `./framework.js`.

import { ErrorCode, LenkeError } from '@lenke/errors';

import { appendStep, type SackOp, type Step } from '../ast.js';
import {
  type ByableStep,
  makeByable,
  OPERATOR_TO_SACKOP,
  type OperatorSym,
  type StepFn,
} from './framework.js';

/**
 * Enable the per-traverser sack, initialized to `init`. With no `withSack` in
 * the traversal, `sack()` faults and no sack machinery runs (lazy).
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#sack-step
 */
export const withSack = (init: unknown): StepFn => appendStep({ kind: 'withSack', init });

type SackStep = Extract<Step, { kind: 'sack' }>;

/**
 * `sack()` emits the current traverser's sack. `sack(Operator.x).by(proj)`
 * merges the projected value into the sack via `x` and passes the traverser
 * through unchanged (its value is untouched).
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#sack-step
 */
export function sack(op?: OperatorSym): ByableStep<SackStep> {
  let sackOp: SackOp | undefined;

  if (op !== undefined) {
    sackOp = OPERATOR_TO_SACKOP.get(op);

    if (sackOp === undefined) {
      throw new LenkeError('sack(): unrecognized operator (use Operator.sum/mult/min/max/assign)', {
        code: ErrorCode.Unsupported,
      });
    }
  }

  return makeByable<SackStep>((bys) => ({
    kind: 'sack',
    ...(sackOp === undefined ? {} : { op: sackOp }),
    ...(bys && bys.length > 0 ? { bys } : {}),
  }));
}
