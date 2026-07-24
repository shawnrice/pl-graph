/**
 * `planToGremlin` — the Plan → Groovy-text emitter.
 *
 * This is what makes the typed fluent API portable: author once as a `Plan`, run
 * it directly on the TS engine, or emit Groovy and run the identical traversal on
 * the Rust core via `graph.gremlin(text)`. It also absorbs the two dialects'
 * spelling differences (`in_`/`dedupe` here, `in`/`dedup` in the text dialect), so
 * callers never see them.
 *
 * It lived inside `packages/native/src/gremlin-conformance.test.ts`, where the
 * cross-engine suite used it to prove both engines agree — implemented, tested,
 * and unreachable by anyone outside that file. Moved here and exported so the
 * portability story is actually usable.
 *
 * Kinds that cannot cross the text boundary — JS closures, non-finite literals —
 * throw `EmitUnsupported`, tagged so a caller can distinguish "TS-only superset"
 * from "gap in the emitter".
 */
import { isTemporal, temporalLiteralParts } from '@lenke/core';

import type { By, Plan, Predicate, Step } from './ast.js';
import { isPlan } from './steps/framework.js';

//
// The reason a case can't be dual-authored by hand: this derives the Groovy
// string mechanically from the same Plan the TS engine runs. Unsupported kinds
// throw with a tag so the runner can classify (tsOnly) vs surface (emitter gap).

type UnsupportedKind = 'closure' | 'nonfinite' | 'unsupported';

export class EmitUnsupported extends Error {
  constructor(
    readonly reason: UnsupportedKind,
    detail: string,
  ) {
    super(`planToGremlin: ${reason}: ${detail}`);
  }
}

/** True for the kinds that only exist in the TS superset (no native form). */
export const isTsOnly = (e: unknown): e is EmitUnsupported =>
  e instanceof EmitUnsupported && e.reason !== 'unsupported';

const emitLiteral = (v: unknown): string => {
  if (typeof v === 'string') {
    return `'${v.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
  }

  if (typeof v === 'boolean') {
    return String(v);
  }

  if (typeof v === 'number') {
    // NaN / ±Infinity have no Groovy literal and can't survive JSON — this is
    // exactly why the NaN drifts are unreachable through this boundary.
    if (!Number.isFinite(v)) {
      throw new EmitUnsupported('nonfinite', `${v} has no Groovy literal`);
    }

    return String(v);
  }

  // A temporal emits as the dialect's constructor call — `date('2020-01-01')`.
  // Its `toJSON` tag is the discriminant, so the emitted spelling and the
  // parser's accepted spelling are derived from one table rather than two.
  if (isTemporal(v)) {
    const parts = temporalLiteralParts(v);

    if (parts === null) {
      throw new EmitUnsupported('unsupported', 'unrecognized temporal kind');
    }

    return `${parts.kind}(${emitLiteral(parts.iso)})`;
  }

  throw new EmitUnsupported('unsupported', `literal of type ${typeof v}`);
};

const emitPredicate = (p: Predicate): string => {
  switch (p.op) {
    case 'eq':
    case 'neq':
    case 'gt':
    case 'gte':
    case 'lt':
    case 'lte':
      return `${p.op}(${emitLiteral(p.value)})`;
    case 'within':
    case 'without':
      return `${p.op}(${p.values.map(emitLiteral).join(', ')})`;
    case 'between':
    case 'inside':
    case 'outside':
      return `${p.op}(${emitLiteral(p.min)}, ${emitLiteral(p.max)})`;
    case 'regex':
      return `regex(${emitLiteral(p.value)})`;
    case 'not':
      return `not(${emitPredicate(p.predicate)})`;
    default:
      // startsWith/containing/... — extend as the corpus grows.
      throw new EmitUnsupported('unsupported', `predicate ${p.op}`);
  }
};

const emitBy = (by: By): string => {
  switch (by.kind) {
    case 'identity':
      return by.direction ? `.by(${by.direction})` : '.by()';
    case 'key':
      return by.direction
        ? `.by(${emitLiteral(by.key)}, ${by.direction})`
        : `.by(${emitLiteral(by.key)})`;
    case 'traversal': {
      const sub = emitSubPlan(by.plan);

      return by.direction ? `.by(${sub}, ${by.direction})` : `.by(${sub})`;
    }
    default:
      // token / column by()s — extend as the corpus grows.
      throw new EmitUnsupported('unsupported', `by(${by.kind})`);
  }
};

// A `.with(<option>, <value>)` modulator suffix, or '' when the field is unset.
const emitWith = (option: string, value: string | number | undefined): string =>
  value === undefined ? '' : `.with(${option}, ${emitLiteral(value)})`;

// OLAP algorithm steps → Groovy, reconstructing the `.with(...)` modulators from
// the step's config fields. Split out of `emitStep` to keep its complexity low.
const emitAlgoStep = (
  step: Extract<Step, { kind: 'pageRank' | 'connectedComponent' | 'peerPressure' }>,
): string => {
  switch (step.kind) {
    case 'pageRank': {
      const base = step.alpha === undefined ? 'pageRank()' : `pageRank(${emitLiteral(step.alpha)})`;

      return `${base}${emitWith('PageRank.propertyName', step.property)}${emitWith('PageRank.times', step.times)}`;
    }
    case 'connectedComponent':
      return `connectedComponent()${emitWith('ConnectedComponent.propertyName', step.property)}`;
    case 'peerPressure':
      return `peerPressure()${emitWith('PeerPressure.propertyName', step.property)}${emitWith('PeerPressure.times', step.times)}`;
  }
};

/**
 * Edge steps and the vertex steps that come back off them. Split out of
 * `emitStep` purely to keep any one function's branch count reasonable.
 */
/** Steps that emit as a bare `name()` — no arguments, no modulators. */
const NILADIC: ReadonlySet<Step['kind']> = new Set([
  'id',
  'label',
  'value',
  'sum',
  'min',
  'max',
  'mean',
  'fold',
  'unfold',
  'identity',
  'count',
] as const);

const emitEdgeStep = (step: Step): string | null => {
  switch (step.kind) {
    case 'E':
      return `E(${(step.ids ?? []).map(emitLiteral).join(', ')})`;
    case 'outE':
    case 'inE':
    case 'bothE':
      return `${step.kind}(${step.labels.map(emitLiteral).join(', ')})`;
    case 'outV':
    case 'inV':
    case 'bothV':
    case 'otherV':
      return `${step.kind}()`;
    default:
      return null;
  }
};

/** Slicing. The TS names differ from the dialect's: `take` is `limit`, and a
 *  local scope is spelled as a leading `local` argument. */
const emitSliceStep = (step: Step): string | null => {
  switch (step.kind) {
    case 'take':
    case 'tail': {
      const name = step.kind === 'take' ? 'limit' : 'tail';

      return step.scope === 'local' ? `${name}(local, ${step.n})` : `${name}(${step.n})`;
    }
    case 'skip':
      return step.scope === 'local' ? `skip(local, ${step.n})` : `skip(${step.n})`;
    case 'range':
      return step.scope === 'local'
        ? `range(local, ${step.start}, ${step.end})`
        : `range(${step.start}, ${step.end})`;
    default:
      return null;
  }
};

/** The `by()`-modulated steps. Legacy scalar `by`/`keyBy`/`valueBy` are property
 *  names; the `bys` modulator form takes precedence wherever both exist. */
const emitModulatedStep = (step: Step): string | null => {
  const mods = (bys: readonly By[] | undefined): string => (bys ?? []).map(emitBy).join('');

  switch (step.kind) {
    case 'path':
      return `path()${mods(step.bys)}`;
    case 'valueMap':
      return `valueMap(${(step.keys ?? []).map(emitLiteral).join(', ')})`;
    case 'groupCount': {
      if (step.bys?.length) {
        return `groupCount()${mods(step.bys)}`;
      }

      return step.by === undefined ? 'groupCount()' : `groupCount().by(${emitLiteral(step.by)})`;
    }
    case 'group': {
      if (step.bys?.length) {
        return `group()${mods(step.bys)}`;
      }

      const legacy = [step.keyBy, step.valueBy]
        .filter((k): k is string => k !== undefined)
        .map((k) => `.by(${emitLiteral(k)})`)
        .join('');

      return `group()${legacy}`;
    }
    case 'select': {
      const pop = step.pop && step.pop !== 'last' ? `${step.pop}, ` : '';

      return `select(${pop}${step.labels.map(emitLiteral).join(', ')})${mods(step.bys)}`;
    }
    case 'selectColumn':
      return `select(${step.column})`;
    case 'project':
      return `project(${step.keys.map(emitLiteral).join(', ')})${mods(step.bys)}`;
    case 'order': {
      // `order()` used to drop its modulators, which would have emitted a
      // differently-ordered result rather than an error.
      const scope = step.scope === 'local' ? 'local' : '';

      if (step.bys?.length) {
        return `order(${scope})${mods(step.bys)}`;
      }

      if (step.key !== undefined) {
        return `order(${scope}).by(${emitLiteral(step.key)}${step.desc ? ', desc' : ''})`;
      }

      return step.desc ? `order(${scope}).by(desc)` : `order(${scope})`;
    }
    default:
      return null;
  }
};

/** Steps carrying whole sub-plans. */
const emitNestedStep = (step: Step): string | null => {
  switch (step.kind) {
    case 'union':
      return `union(${step.plans.map(emitSubPlan).join(', ')})`;
    case 'not':
      return `not(${emitSubPlan(step.plan)})`;
    case 'repeat': {
      // Placement is semantic, not cosmetic: `until(c).repeat(b)` is while-do,
      // `repeat(b).until(c)` is do-while, and the same distinction applies to
      // `emit`. Emitting the wrong side would silently change the result, so a
      // pre-form modulator is emitted BEFORE the repeat.
      const until = step.until ? `until(${emitSubPlan(step.until)})` : '';
      const emit = step.emit ? `emit(${emitSubPlan(step.emit)})` : '';
      const pre =
        (step.untilBefore && until ? `${until}.` : '') +
        (step.emitBefore && emit ? `${emit}.` : '');
      const post =
        (!step.untilBefore && until ? `.${until}` : '') +
        (!step.emitBefore && emit ? `.${emit}` : '') +
        (step.times === undefined ? '' : `.times(${step.times})`);

      return `${pre}repeat(${emitSubPlan(step.body)})${post}`;
    }
    default:
      return null;
  }
};

const emitStep = (step: Step): string => {
  // Edge / slice / modulated / nested families first — each returns null for a
  // kind it does not own, so the switch below stays the simple-step case.
  const family =
    emitEdgeStep(step) ?? emitSliceStep(step) ?? emitModulatedStep(step) ?? emitNestedStep(step);

  if (family !== null) {
    return family;
  }

  switch (step.kind) {
    case 'V':
      return `V(${(step.ids ?? []).map(emitLiteral).join(', ')})`;
    case 'inject':
      return `inject(${step.values.map(emitLiteral).join(', ')})`;
    case 'out':
    case 'in':
    case 'both':
      return `${step.kind}(${step.labels.map(emitLiteral).join(', ')})`;
    case 'hasLabel':
      return `hasLabel(${step.labels.map(emitLiteral).join(', ')})`;
    case 'has':
      return `has(${emitLiteral(step.key)}, ${emitPredicate(step.pred)})`;
    case 'is':
      return `is(${emitPredicate(step.pred)})`;
    case 'values':
      return `values(${step.keys.map(emitLiteral).join(', ')})`;
    case 'dedupe':
      return `dedup(${(step.labels ?? []).map(emitLiteral).join(', ')})`;
    case 'constant':
      return `constant(${emitLiteral(step.value)})`;
    case 'property': {
      // A traversal-induced value emits as an anonymous sub-traversal; a literal
      // as a Groovy literal. (cardinality is TS-superset — not emitted.)
      const val = isPlan(step.value) ? emitSubPlan(step.value) : emitLiteral(step.value);

      return `property(${emitLiteral(step.key)}, ${val})`;
    }
    // The TS-superset kinds: no native form. Classified tsOnly.
    case 'mapFn':
    case 'flatMapFn':
    case 'filterFn':
    case 'sideEffectFn':
    case 'foldFn':
      throw new EmitUnsupported('closure', step.kind);
    case 'math': {
      const bys = (step.bys ?? []).map(emitBy).join('');

      return `math(${emitLiteral(step.expr)})${bys}`;
    }
    case 'branch': {
      const opts = step.options
        .map((o) => `.option(${emitLiteral(o.match)}, ${emitSubPlan(o.plan)})`)
        .join('');
      const def = step.default ? `.option(none, ${emitSubPlan(step.default)})` : '';

      return `branch(${emitSubPlan(step.test)})${opts}${def}`;
    }
    default:
      // Arity-0 steps emit as a bare `name()`; kept in a set rather than a case
      // arm each so this switch's branch count stays reviewable.
      if (NILADIC.has(step.kind)) {
        return `${step.kind}()`;
      }

      throw new EmitUnsupported('unsupported', `step ${step.kind}`);
  }
};

// OLAP kinds route to `emitAlgoStep`; everything else to the core `emitStep`
// switch — kept separate so neither function's cyclomatic complexity grows.
const ALGO_KINDS = new Set<Step['kind']>(['pageRank', 'connectedComponent', 'peerPressure']);

const emitAnyStep = (step: Step): string => {
  // `as('label')` — kept out of `emitStep`'s switch to hold its complexity budget.
  if (step.kind === 'as') {
    return `as(${emitLiteral(step.label)})`;
  }

  return ALGO_KINDS.has(step.kind)
    ? emitAlgoStep(
        step as Extract<Step, { kind: 'pageRank' | 'connectedComponent' | 'peerPressure' }>,
      )
    : emitStep(step);
};

// An anonymous sub-traversal (no `g.` prefix): `out('KNOWS').values('name')`.
const emitSubPlan = (p: Plan): string => p.steps.map(emitAnyStep).join('.');

export const planToGremlin = (plan: Plan): string => `g.${plan.steps.map(emitAnyStep).join('.')}`;
