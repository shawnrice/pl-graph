import type { Plan } from '../ast.js';
import { STEP_FN, type StepFn } from './framework.js';

/**
 * Compose multiple step constructors into a single branded StepFn.
 *
 * Use to build sub-plans inline:
 *
 *     filter(pipe(label(), is(eq('PERSON'))))   // sub-plan form, branded
 *     filter((v) => v.id === 1)                 // closure form
 *
 * `traversal(...)` works in the same slots now, so reach for whichever is
 * more readable for the call site.
 */
export const pipe = (...steps: StepFn[]): StepFn => {
  const fn = (plan: Plan): Plan => steps.reduce((p, s) => s(p), plan);
  Object.defineProperty(fn, STEP_FN, { value: true, enumerable: false });

  return fn;
};
