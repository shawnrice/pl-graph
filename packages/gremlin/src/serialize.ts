import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Plan, Step } from './ast.js';

/**
 * The set of step kinds that carry a JS closure. Plans containing any of
 * these cannot be serialized — the closure won't survive `JSON.stringify`.
 */
const CLOSURE_KINDS: ReadonlySet<Step['kind']> = new Set([
  'mapFn',
  'flatMapFn',
  'filterFn',
  'sideEffectFn',
  'foldFn',
] as const);

const isPlan = (x: unknown): x is Plan => !!x && typeof x === 'object' && 'steps' in x;

const findClosuresInArrayItem = (v: unknown, path: string): string[] => {
  if (isPlan(v)) {
    return findClosures(v, path);
  }

  // `branch.options` is `[{ match, plan }, ...]` — recurse via `.plan`.
  if (v && typeof v === 'object' && 'plan' in v) {
    const inner = v.plan;

    if (isPlan(inner)) {
      return findClosures(inner, `${path}.plan`);
    }
  }

  return [];
};

/**
 * Recursively walk a plan and collect any closure-bearing step kinds. Returns
 * a list of paths (e.g. `['repeat.body', 'mapFn']`) so callers can pinpoint
 * which steps prevent serialization.
 */
export const findClosures = (plan: Plan, prefix = ''): string[] => {
  const found: string[] = [];

  for (let i = 0; i < plan.steps.length; i++) {
    const step = plan.steps[i];
    const path = prefix ? `${prefix}.${i}.${step.kind}` : `${i}.${step.kind}`;

    if (CLOSURE_KINDS.has(step.kind)) {
      found.push(path);
    }

    // Recurse into sub-plans.
    for (const [field, value] of Object.entries(step)) {
      if (isPlan(value)) {
        found.push(...findClosures(value, `${path}.${field}`));
      }

      if (Array.isArray(value)) {
        for (let j = 0; j < value.length; j++) {
          found.push(...findClosuresInArrayItem(value[j], `${path}.${field}[${j}]`));
        }
      }
    }
  }

  return found;
};

/**
 * Whether a plan can be serialized (i.e. contains no closure-bearing steps).
 */
export const isSerializable = (plan: Plan): boolean => findClosures(plan).length === 0;

/**
 * Serialize a plan to JSON. Throws if any step in the plan (or any of its
 * sub-plans) carries a closure — those don't survive the round-trip and
 * silently dropping them would be a footgun.
 *
 * If you need to ship a closure-bearing plan, you'll have to either rewrite
 * the closures as sub-plans (e.g. `map(v => v.id)` → `map(pipe(id()))`), or
 * build a custom serializer that registers named closures somewhere both
 * sides can resolve.
 */
export const serialize = (plan: Plan): string => {
  const closures = findClosures(plan);

  if (closures.length > 0) {
    throw new LenkeError(
      `Plan contains closure-bearing steps and cannot be serialized:\n  ${closures.join('\n  ')}\n` +
        `Rewrite these as sub-plans (e.g. map(pipe(...))) to make the plan serializable.`,
      { code: ErrorCode.Unsupported },
    );
  }

  // A plan can embed literal predicate values (`is(eq(10n))`); `JSON.stringify`
  // throws on a BigInt and on a cyclic value, and silently drops `undefined` /
  // turns `NaN`/`Infinity` into `null`. Surface the throwing cases cleanly.
  try {
    return JSON.stringify(plan);
  } catch (cause) {
    throw new LenkeError(
      'Plan contains a non-serializable value (e.g. a BigInt or a cyclic structure).',
      { code: ErrorCode.InvalidValue, cause },
    );
  }
};

/**
 * Deserialize a plan from JSON. The resulting plan only contains the
 * sub-plan forms — closures cannot round-trip.
 */
export const deserialize = (json: string): Plan => JSON.parse(json) as Plan;
