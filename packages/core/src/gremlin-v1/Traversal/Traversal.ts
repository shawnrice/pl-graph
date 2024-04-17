import type { UnaryFn } from '@pl-graph/fp/src';
import { List } from '@pl-graph/list/src';
import { arraysAreEqual } from '@pl-graph/utils/src';

import { Edge } from '../../core/Edge';
import { Graph } from '../../core/Graph';
import { Vertex } from '../../core/Vertex';
import type { Predicate } from '../Predicates';
import {
  and,
  as,
  both,
  bothE,
  bothV,
  by,
  choose,
  coalesce,
  count,
  dedupe,
  E,
  fail,
  filter,
  fold,
  has,
  hasId,
  hasKey,
  hasLabel,
  id,
  identity,
  in_,
  inE,
  inject,
  inV,
  is,
  label,
  limit,
  map,
  max,
  mean,
  min,
  none,
  not,
  optional,
  or,
  order,
  otherV,
  out,
  outE,
  outV,
  path,
  range,
  select,
  sideEffect,
  skip,
  Step,
  sum,
  tail,
  toArray,
  toList,
  toSet,
  unfold,
  union,
  V,
  valueMap,
  values,
  where,
} from '../Steps';
import * as tokens from '../tokens';
import { Traverser } from '../Traverser';
import { NotImplemented } from './NotImplemented';

type TraversalParams = {
  graph: Graph<any, any>;
  value?: Iterable<any>;
};

// From the tinkerpop docs
// GraphTraversal is a monoid in that it is an algebraic structure that has a single binary operation that is associative. The binary operation is function composition (i.e. method chaining) and its identity is the step identity(). This is related to a monad as popularized by the functional programming community.

/**
 * A Traversal is the record of the steps that a walk shall take
 *
 * Its terminating methods turn these into usable JS values
 */
export class Traversal<S = any, E = S> {
  graph: Graph<any, any>;

  steps: Step<any, any>[];

  current: any;

  static tokens = tokens;

  static isTraversal<In = any, Out = In>(x: unknown): x is Traversal<In, Out> {
    return x instanceof Traversal;
  }

  static with<S, E>(graph: Graph<any, any>): Traversal<S, E> {
    return new Traversal({ graph });
  }

  constructor(params: TraversalParams) {
    const { graph, value } = params;
    this.graph = graph;
    this.steps = [];
    this.current = value ?? []; // We hold a value of the last eagerly evaluated step (barrier steps)
  }

  // *[Symbol.iterator]() {
  //   for (const step of this.steps) {
  //     yield* step.run(this.value, {});
  //   }
  // }

  isEqual(t: Traversal<S, E>): boolean {
    if (this.steps.length !== t.steps.length) {
      return false;
    }

    for (let i = 0; i < this.steps.length; i++) {
      const a = this.steps[i];
      const b = t.steps[i];

      if (a.species !== b.species || a.genus !== b.genus) {
        return false;
      }

      if (a.callback !== b.callback) {
        return false;
      }

      if (!arraysAreEqual(a.args, b.args)) {
        return false;
      }
    }

    return true;
  }

  addStep(step: Step<any, any>): Traversal<S, E> {
    this.steps.push(step);
    return this;
  }

  clone(): Traversal<any> {
    const next = new Traversal({ graph: this.graph });
    next.steps = Array.from(this.steps);
    next.current = Array.from(this.current);
    return next;
  }

  /**
   * Runs the traversal and sets the value to the outcome. It also removes all the steps
   *
   * This is intended to be called by Barrier steps
   */
  flush(): Traversal<S, E> {
    this.current = toArray(this);
    this.steps = [];

    return this;
  }

  getExecutor(): UnaryFn<Iterable<Traverser<any>>> {
    let fn = null;
    for (const step of this.steps) {
      fn = fn ? step.callback(fn) : step.callback;
    }

    return fn as unknown as UnaryFn<Iterable<Traverser<any>>>;
  }

  *run(): Generator<Iterable<Traverser<any>>> {
    const executor = this.getExecutor();

    if (typeof this.current !== 'string' && typeof this.current[Symbol.iterator] === 'function') {
      yield* executor(this.current);
    } else {
      yield* executor(this.current);
    }
  }

  /**
   * Steps
   */

  aggregate(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  and(...traversals: any[]): Traversal {
    return and(...traversals)(this);
  }

  as<S, E>(...keys: string[]): Traversal<S, E> {
    return as(...keys)(this);
  }

  barrier(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  both<S, E>(...labels: string[]): Traversal<S, E> {
    return both(...labels)(this);
  }

  bothE<S, E>(...labels: string[]): Traversal<S, E> {
    return bothE(...labels)(this);
  }

  bothV<S, E>(): Traversal<S, E> {
    return bothV()(this);
  }

  branch(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  by<S, E>(...args: any[]): Traversal<S, E> {
    return by(...args)(this);
  }

  cap(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  choose(
    condition: UnaryFn<Traversal>,
    t: UnaryFn<Traversal>,
    f?: UnaryFn<Traversal>,
  ): Traversal<S, E> {
    return choose(condition, t, f)(this);
  }

  coalesce(...traversals: UnaryFn<Traversal>[]): Traversal {
    return coalesce(...traversals)(this);
  }

  constant(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  count(): Traversal<S, E> {
    return count()(this);
  }

  /**
   * Deduplicates objects
   *
   * Uses strict comparison
   *
   * The dedup(String...) signature does nothing right now
   *
   * @see https://tinkerpop.apache.org/docs/current/reference/#dedup-step
   */
  dedup(...args: string[]): Traversal<S, E> {
    return dedupe(...args)(this);
  }

  E<S, E>(x?: Edge<any, any, any> | string | null): Traversal<S, E> {
    return E<S, E>(x)(this);
  }

  fail(message?: string): Traversal<S, E> {
    return fail(message)(this);
  }

  filter<T>(predicate: UnaryFn<T, boolean>): Traversal<S, E> {
    return filter(predicate)(this);
  }

  fold<T, TT>(...args: any[]): Traversal<T, TT> {
    return fold(...args)(this);
  }

  has<S, E = S>(key: string, predicate: Predicate<S>): Traversal<S, E>;
  has<S, E = S>(key: string, value: any): Traversal<S, E>;
  has<S, E = S>(label: string, key: string, value: S): Traversal<S, E>;
  has<S, E = S>(...args: (string | Predicate<S> | any)[]): Traversal<S, E> {
    return has(...args)(this);
  }

  hasId<T>(...ids: string[]): Traversal<T> {
    return hasId<T>(...ids)(this);
  }

  hasKey<T>(...keys: string[]): Traversal<T> {
    return hasKey<T>(...keys)(this);
  }

  hasLabel<S, E>(...labels: string[]): Traversal<S, E> {
    return hasLabel<S, E>(...labels)(this);
  }

  id(): Traversal<S, E> {
    return id()(this);
  }

  identity(): Traversal<S, E> {
    return identity()(this);
  }

  /**
   * Move to incoming adjacent vertices given edge labels
   */
  ['in'](...labels: string[]): Traversal<S, E> {
    return in_(...labels)(this);
  }

  /**
   * Move to incoming adjacent vertices given edge labels
   */
  in_(...labels: string[]): Traversal<S, E> {
    return this.in(...labels);
  }

  /**
   * Move to incoming edges given edge labels
   */
  inE(...labels: string[]): Traversal<S, E> {
    return inE(...labels)(this);
  }

  inject(...args: any[]): Traversal<S, E> {
    return inject(...args)(this);
  }

  inV(): Traversal<S, E> {
    return inV()(this);
  }

  is(x0: any): Traversal<S, E> {
    return is(x0)(this);
  }

  label(): Traversal<S, E> {
    return label()(this);
  }

  limit<S, E>(x0: number): Traversal<S, E> {
    return limit<S, E>(x0)(this);
  }

  map<S, E>(mapper: UnaryFn<S, E>): Traversal<S, E> {
    return map(mapper)(this);
  }

  /**
   * The max()-step (map) operates on a stream of comparable objects and determines which is the last object
   * according to its natural order in the stream.
   */
  max(): Traversal {
    return max()(this);
  }

  mean(): Traversal {
    return mean()(this);
  }

  min(): Traversal {
    return min()(this);
  }

  /**
   * Filter all traversers in the traversal.
   *
   * This step has narrow use cases and is primarily intended for use as a signal to remote servers
   * that iterate() was called. While it may be directly used, it is often a sign that a traversal
   * should be re-written in another form.
   *
   * __FILTER__
   *
   * __HERE BE DRAGONS__: Since we cannot pass on any traversers, if anything comes after this,
   * then we'll lose reference to the graph
   */
  none(): Traversal {
    return none()(this);
  }

  not(traversal: UnaryFn<Traversal>): Traversal {
    return not(traversal)(this);
  }

  optional(x0: UnaryFn<Traversal>): Traversal {
    return optional(x0)(this);
  }

  or(...traversals: any[]): Traversal {
    return or(...traversals)(this);
  }

  /**
   * NEEDS BY MODULATION
   */
  order(...args: any[]): Traversal {
    return order(...args)(this);
  }

  otherV(): Traversal<S, E> {
    return otherV()(this);
  }

  /**
   * Move to outgoing adjacent vertices given edge labels
   */
  out(...labels: string[]): Traversal {
    return out(...labels)(this);
  }

  /**
   * Move to outgoing edges given edge labels
   */
  outE(...labels: string[]): Traversal {
    return outE(...labels)(this);
  }

  outV(): Traversal {
    return outV()(this);
  }

  path(): Traversal {
    return path()(this);
  }

  range(low: number, high: number): Traversal {
    return range(low, high)(this);
  }

  select(...keys: string[]): Traversal {
    return select(...keys)(this);
  }

  sideEffect(effect: UnaryFn<Traverser<any>, any>): Traversal {
    return sideEffect(effect)(this);
  }

  skip(x0: number): Traversal {
    return skip(x0)(this);
  }

  sum(): Traversal {
    return sum()(this);
  }

  tail(num?: number): Traversal {
    return tail(num)(this);
  }

  unfold(): Traversal {
    return unfold()(this);
  }

  union(...traversals: UnaryFn<Traversal>[]): Traversal {
    return union(...traversals)(this);
  }

  V(x?: Vertex<any> | string | null): Traversal {
    return V(x)(this);
  }

  values(...keys: string[]): Traversal {
    return values(...keys)(this);
  }

  valueMap(...keys: string[]): Traversal {
    return valueMap(...keys)(this);
  }

  where(...x: any[]): Traversal {
    return where(...x)(this);
  }

  /**
   * Terminating Methods
   */

  /**
   * Converts to an array. This is super weird, and I think we'll have to change it
   *
   * Terminating step
   */
  toArray(): E[] {
    return toArray<S, E>(this);
  }

  /**
   * Converts the traversal to a List
   *
   * (Basically an iterator of the objects
   *
   * Terminating step
   */
  toList(): List<E> {
    return toList<S, E>(this);
  }

  /**
   * Converts the traversal result to a set
   *
   * Terminating step
   */
  toSet(): Set<E> {
    return toSet<S, E>(this);
  }

  /* eslint-disable */

  elementMap(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  flatMap(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  group(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  groupCount(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  key(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  match(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  option(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  project(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  properties(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  propertyMap(): never {
    throw new NotImplemented('TODO: IMPLEMENT THIS');
  }

  /**
   * The following methods are not yet supported, and we have no scheduled plans to support them
   */

  /* eslint-disable class-methods-use-this  */

  /**
   * @throws NotImplemented
   */
  addE(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  addV(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  call(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  coin(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  connectedComponent(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  cyclicPath(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  drop(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  element(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  emit(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  explain(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  from(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  index(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  IO(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  local(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  loops(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  math(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  mergeEdge(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  mergeVertex(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  pageRank(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  peerPressure(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  profile(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  program(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  property(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  read(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  repeat(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  sack(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  sample(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  shortestPath(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  simplePath(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  subgraph(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  timeLimit(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  to(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  tree(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  until(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  value(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  with(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /**
   * @throws NotImplemented
   */
  write(): never {
    throw new NotImplemented('Not yet implemented');
  }

  /* eslint-enable class-methods-use-this  */
}
