// Step dispatcher. Routes a single step's `kind` to the per-category impl
// function in `./{category}.ts`. Also hosts `applyPlanToStream`, which is
// called recursively by step impls that take sub-plans (where, filter,
// repeat, branch, optional, choose, union, coalesce, local, flatMap, map,
// sideEffect, fold, etc.).
//
// Cycle: this module imports per-category step impls; those impls import
// `applyPlanToStream` from here. ESM resolves the cycle correctly because
// neither side dereferences the other at module-init — `applyStep` and
// `applyPlanToStream` are only called inside generator bodies.

import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Plan, Step } from '../ast.js';
import { matches } from '../predicates.js';
// Per-category step impls.
import {
  aggregateComparable,
  aggregateNumber,
  countLocal,
  countStep,
  maxLocal,
  meanLocal,
  minLocal,
  orderLocalStep,
  orderStep,
  sumLocal,
} from './aggregation.js';
import { connectedComponentStep, pageRankStep, peerPressureStep } from './algorithms.js';
import {
  sampleStep,
  skipTraversers,
  sliceLocal,
  tailLocal,
  tailTraversers,
  takeTraversers,
} from './cardinality.js';
import {
  failStep,
  hasRevisit,
  whereCompareStep,
  whereCurrentStep,
  whereSubPlanStep,
} from './filters.js';
import {
  branchStep,
  chooseStep,
  coalesceStep,
  localStep,
  optionalStep,
  repeatStep,
  unionStep,
} from './iteration.js';
import { matchStep } from './match.js';
import { evalMath, mathStep } from './misc.js';
import { edgeToVertex, traverseToEdge, traverseToVertex } from './movement.js';
import { addEStep, addVStep, dropStep, propertyStep } from './mutation.js';
import {
  asStep,
  groupCountStep,
  groupStep,
  pathStep,
  projectElementMap,
  projectProperties,
  projectPropertyMap,
  projectStep,
  projectValueMap,
  projectValues,
  selectColumnStep,
  selectStep,
  treeStep,
} from './projection.js';
import {
  closureView,
  filterStream,
  filterTraverser,
  firstLabel,
  hasAny,
  isEdge,
  isVertex,
  keyToBy,
  mapTraverser,
  newContext,
  normalizeBys,
  type RunContext,
  type Traverser,
  tupleKey,
  evalBy,
} from './runtime.js';
import { sackStep, withSackStep } from './sack.js';
import { shortestPathStep } from './shortest-path.js';
import { aggregateStep, barrierStep, capStep, subgraphStep } from './side-effects.js';
import {
  flatMapFnStep,
  flatMapStep,
  foldFnStep,
  foldStep,
  indexStep,
  injectMidStream,
  mapStep,
  sideEffectFnStep,
  sideEffectStep,
  unfoldStream,
} from './stream.js';

export const applyPlanToStream = (
  plan: Plan,
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  ctx: RunContext = newContext(),
): Iterable<Traverser<unknown>> => {
  let s = stream;

  for (const step of plan.steps) {
    s = applyStep(step, s, graph, ctx);
  }

  return s;
};

// eslint-disable-next-line complexity -- step-kind dispatch; complexity is inherent to the switch arity, not cognitive load
export const applyStep = (
  step: Step,
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  ctx: RunContext = newContext(),
): Iterable<Traverser<unknown>> => {
  switch (step.kind) {
    case 'V':
    case 'E':
      throw new LenkeError(`${step.kind} can only appear as the first step`, {
        code: ErrorCode.Syntax,
      });

    case 'out':
    case 'in':
    case 'both':
      return traverseToVertex(stream, graph, step);

    case 'outE':
    case 'inE':
    case 'bothE':
      return traverseToEdge(stream, graph, step);

    case 'outV':
    case 'inV':
    case 'bothV':
    case 'otherV':
      return edgeToVertex(stream, step);

    case 'has':
      return filterStream(stream, (v) => {
        if (!isVertex(v) && !isEdge(v)) {
          return false;
        }

        return matches(step.pred, v.properties[step.key]);
      });

    case 'hasLabel':
      return filterStream(stream, (v) => {
        if (!isVertex(v) && !isEdge(v)) {
          return false;
        }

        return step.labels.some((l) => v.labels.has(l));
      });

    case 'hasId':
      return filterStream(stream, (v) => {
        if (!isVertex(v) && !isEdge(v)) {
          return false;
        }

        return step.ids.includes(v.id);
      });

    case 'hasKey':
      return filterStream(stream, (v) => {
        // Element form: vertex/edge with one of the given property keys.
        if (isVertex(v) || isEdge(v)) {
          return step.keys.some((k) => k in v.properties);
        }

        // Property-object form: filter the stream produced by `properties()`,
        // which yields `{key, value}` per property. Match if the object's
        // `key` field equals one of the given keys.
        if (v !== null && typeof v === 'object' && 'key' in v) {
          return step.keys.includes((v as { key: unknown }).key as string);
        }

        return false;
      });

    case 'simplePath':
      return filterTraverser(stream, (t) => !hasRevisit(t.path));

    case 'cyclicPath':
      return filterTraverser(stream, (t) => hasRevisit(t.path));

    case 'dedupe': {
      // Primitives and stable graph-element references dedupe directly in this
      // Set — value/identity equality, no allocation (the hot path). Composite
      // values (lists / plain-object maps) are distinct references even when
      // structurally equal, so they upgrade to a structural string key in a
      // second, lazily-created Set. A recurring *reference* short-circuits via
      // the WeakSet, so each composite is keyed at most once: slow the first
      // time, an O(1) pointer hit on every repeat.
      const seen = new Set<unknown>();
      let seenComposite: Set<string> | null = null;
      let seenRefs: WeakSet<object> | null = null;
      const { labels } = step;
      const by = step.bys?.[0];

      return filterTraverser(stream, (t) => {
        // Multi-label form: dedupe by the tuple of tagged values at the given
        // labels — a stable NUL-joined string key.
        if (labels && labels.length > 0) {
          const k = tupleKey(labels.map((l) => t.tags.get(l)));

          if (seen.has(k)) {
            return false;
          }

          seen.add(k);

          return true;
        }

        const v = by !== undefined ? evalBy(by, t.value, graph, ctx) : t.value;

        // Only plain lists/maps need structural keying; elements (stable refs),
        // JS `Map`s, and other class instances keep cheap reference identity.
        const proto =
          v !== null && typeof v === 'object' ? (Object.getPrototypeOf(v) as unknown) : false;
        const isComposite = Array.isArray(v) || proto === Object.prototype || proto === null;

        if (isComposite) {
          const ref = v as object;

          seenRefs ??= new WeakSet();

          if (seenRefs.has(ref)) {
            return false; // same reference already processed → duplicate, no re-key
          }

          seenRefs.add(ref);

          const k = tupleKey([v]);
          seenComposite ??= new Set();

          if (seenComposite.has(k)) {
            return false;
          }

          seenComposite.add(k);

          return true;
        }

        if (seen.has(v)) {
          return false;
        }

        seen.add(v);

        return true;
      });
    }

    case 'take':
      return step.scope === 'local'
        ? mapTraverser(stream, (v) => sliceLocal(v, 0, step.n))
        : takeTraversers(stream, step.n);

    case 'skip':
      return step.scope === 'local'
        ? mapTraverser(stream, (v) => sliceLocal(v, step.n, Infinity))
        : skipTraversers(stream, step.n);

    case 'range':
      if (step.scope === 'local') {
        const end = step.end < 0 ? Infinity : step.end;

        return mapTraverser(stream, (v) => sliceLocal(v, step.start, end));
      }

      if (step.end < 0) {
        return skipTraversers(stream, step.start);
      }

      return takeTraversers(skipTraversers(stream, step.start), Math.max(0, step.end - step.start));

    case 'tail':
      return step.scope === 'local'
        ? mapTraverser(stream, (v) => tailLocal(v, step.n))
        : tailTraversers(stream, step.n);

    case 'is':
      return filterTraverser(stream, (t) => matches(step.pred, t.value));

    case 'identity':
      return stream;

    case 'inject':
      return injectMidStream(stream, step.values);

    case 'unfold':
      return unfoldStream(stream);

    case 'sum':
      return step.scope === 'local'
        ? mapTraverser(stream, sumLocal)
        : aggregateNumber(stream, 'sum');
    case 'min':
      return step.scope === 'local'
        ? mapTraverser(stream, minLocal)
        : aggregateComparable(stream, 'min');
    case 'max':
      return step.scope === 'local'
        ? mapTraverser(stream, maxLocal)
        : aggregateComparable(stream, 'max');
    case 'mean':
      return step.scope === 'local'
        ? mapTraverser(stream, meanLocal)
        : aggregateNumber(stream, 'mean');

    case 'values':
      return projectValues(stream, step.keys);

    case 'valueMap':
      return projectValueMap(stream, step.keys);

    case 'properties':
      return projectProperties(stream, step.keys);

    case 'order': {
      const bys = normalizeBys(step.bys, step.key);
      const desc = step.desc ?? false;

      return step.scope === 'local'
        ? orderLocalStep(stream, bys, desc, graph, ctx)
        : orderStep(stream, bys, desc, graph, ctx);
    }

    case 'fail':
      return failStep(stream, step.message);

    case 'where':
      // Three AST shapes share kind 'where'; TS narrows on which fields are set:
      // a sub-plan, a two-key compare (`startKey`), or a current-vs-label predicate.
      if ('plan' in step) {
        return whereSubPlanStep(stream, step.plan, graph, ctx);
      }

      return 'startKey' in step
        ? whereCompareStep(stream, step, graph, ctx)
        : whereCurrentStep(stream, step);

    case 'and':
      return filterTraverser(stream, (t) =>
        step.plans.every((p) => hasAny(applyPlanToStream(p, [t], graph, ctx))),
      );

    case 'or':
      return filterTraverser(stream, (t) =>
        step.plans.some((p) => hasAny(applyPlanToStream(p, [t], graph, ctx))),
      );

    case 'not':
      return filterTraverser(stream, (t) => !hasAny(applyPlanToStream(step.plan, [t], graph, ctx)));

    case 'union':
      return unionStep(stream, step.plans, graph, ctx);

    case 'coalesce':
      return coalesceStep(stream, step.plans, graph, ctx);

    case 'optional':
      return optionalStep(stream, step.plan, graph, ctx);

    case 'choose':
      return chooseStep(stream, step.test, step.thenPlan, step.elsePlan, graph, ctx);

    case 'filter':
      return whereSubPlanStep(stream, step.plan, graph, ctx);

    case 'hasNot':
      return filterStream(stream, (v) => {
        if (!isVertex(v) && !isEdge(v)) {
          return false;
        }

        return step.keys.every((k) => !(k in v.properties));
      });

    case 'value':
      return mapTraverser(stream, (v) => {
        if (v !== null && typeof v === 'object' && 'key' in v && 'value' in v) {
          return (v as { value: unknown }).value;
        }

        return v;
      });

    case 'index':
      return indexStep(stream);

    case 'math':
      return mathStep(stream, step, graph, ctx);

    case 'hasValue':
      return filterStream(stream, (v) => {
        if (v !== null && typeof v === 'object' && 'value' in v) {
          return step.values.includes((v as { value: unknown }).value);
        }

        return step.values.includes(v);
      });

    case 'match':
      return matchStep(stream, step.patterns, graph, ctx);

    case 'subgraph':
      return subgraphStep(stream, step.key, ctx);

    case 'shortestPath':
      return shortestPathStep(stream, step, graph);

    case 'pageRank':
      return pageRankStep(stream, step, graph);

    case 'connectedComponent':
      return connectedComponentStep(stream, step, graph);

    case 'peerPressure':
      return peerPressureStep(stream, step, graph);

    case 'hasLabelAnd':
      return filterStream(stream, (v) => {
        if (!isVertex(v) && !isEdge(v)) {
          return false;
        }

        return v.labels.has(step.label) && matches(step.pred, v.properties[step.key]);
      });

    case 'elementMap':
      return projectElementMap(stream, step.keys);

    case 'propertyMap':
      return projectPropertyMap(stream, step.keys);

    case 'constant':
      return mapTraverser(stream, () => step.value);

    case 'loops':
      return mapTraverser(stream, (_v, t) => t.loopCount);

    case 'sideEffect':
      return sideEffectStep(stream, step.plan, graph, ctx);

    case 'local':
      return localStep(stream, step.plan, graph, ctx);

    case 'none':
      if (step.pred === undefined) {
        // Legacy: drain and emit nothing.
        // eslint-disable-next-line require-yield -- generator-shaped but intentionally yields nothing
        return (function* () {
          for (const _ of stream) {
            // intentionally drop
          }
        })();
      }

      // TinkerPop 3.8: keep traversers whose iterable value has NO element
      // satisfying the predicate. Non-iterable values are filtered out.
      return filterTraverser(stream, (t) => {
        const v = t.value;

        if (
          v === null ||
          v === undefined ||
          typeof (v as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== 'function'
        ) {
          return false;
        }

        for (const x of v as Iterable<unknown>) {
          if (matches(step.pred!, x)) {
            return false;
          }
        }

        return true;
      });

    case 'aggregate':
    case 'store':
      // Today both append per-traverser without a barrier. Distinct AST kinds
      // so a future optimizer can introduce a barrier on `aggregate` only.
      return aggregateStep(stream, step.key, ctx);

    case 'barrier':
      return barrierStep(stream);

    case 'cap':
      return capStep(stream, ctx, step.key);

    case 'id':
      return mapTraverser(stream, (v) => (isVertex(v) || isEdge(v) ? v.id : undefined));

    case 'label':
      return mapTraverser(stream, (v) => {
        if (isVertex(v) || isEdge(v)) {
          return firstLabel(v.labels) ?? null;
        }

        // For `{key, value}` property objects produced by `properties()`,
        // `label()` returns the key — matching TinkerPop's behavior of
        // treating the property's key field as its "label".
        if (v !== null && typeof v === 'object' && 'key' in v) {
          return (v as { key: unknown }).key;
        }

        return undefined;
      });

    case 'path':
      return pathStep(stream, step.bys, graph, ctx);

    case 'count':
      return step.scope === 'local' ? mapTraverser(stream, countLocal) : countStep(stream);

    case 'fold':
    case 'toList':
      return foldStep(stream);

    case 'repeat':
      return repeatStep(stream, step, graph, ctx);

    case 'as':
      return asStep(stream, step.label);

    case 'select':
      return selectStep(stream, step.labels, step.pop ?? 'last', step.bys, graph, ctx);

    case 'selectColumn':
      return selectColumnStep(stream, step.column);

    case 'group': {
      const keyBy = step.bys?.[0] ?? keyToBy(step.keyBy);
      const valueBy = step.bys?.[1] ?? keyToBy(step.valueBy);

      return groupStep(stream, keyBy, valueBy, graph, ctx);
    }

    case 'groupCount': {
      const by = step.bys?.[0] ?? keyToBy(step.by);

      return groupCountStep(stream, by, graph, ctx);
    }

    case 'project':
      return projectStep(stream, step.keys, step.bys, graph, ctx);

    case 'tree':
      return treeStep(stream, step.bys, graph, ctx);

    case 'branch':
      return branchStep(stream, step.test, step.options, step.default, graph, ctx);

    case 'flatMap':
      return flatMapStep(stream, step.plan, graph, ctx);

    case 'map':
      return mapStep(stream, step.plan, graph, ctx);

    case 'mapFn':
      return mapTraverser(stream, (v, t) => step.fn(v, closureView(t, ctx)));

    case 'flatMapFn':
      return flatMapFnStep(stream, step.fn, ctx);

    case 'filterFn':
      return filterTraverser(stream, (t) => step.fn(t.value, closureView(t, ctx)));

    case 'sideEffectFn':
      return sideEffectFnStep(stream, step.fn, ctx);

    case 'foldFn':
      return foldFnStep(stream, step.seed, step.fn, ctx);

    case 'sample':
      return sampleStep(stream, step.n);

    case 'addV':
      return addVStep(stream, graph, step.label);

    case 'addE':
      return addEStep(stream, graph, step, ctx);

    case 'property':
      return propertyStep(stream, step.key, step.value, graph, ctx);

    case 'withSack':
      return withSackStep(stream, step.init, ctx);

    case 'sack':
      return sackStep(stream, step.op, step.bys, graph, ctx);

    case 'drop':
      return dropStep(stream, graph);
  }
};

// Re-export `evalMath` for any consumer that wants it; only the dispatcher
// references it through `mathStep`, so this is mostly future-proofing.
export { evalMath };
