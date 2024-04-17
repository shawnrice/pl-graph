import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

export type StartStep = 'start';

export type StepModulator = 'modulator';

export type GeneralStep = 'map' | 'flatmap' | 'filter' | 'sideEffect' | 'branch';

export type TerminatingStep = 'terminal';

export type StepGenus = StartStep | GeneralStep | TerminatingStep | StepModulator;

export type TraverserFunction = (x0: Iterable<Traverser<any>>) => Generator<Traverser<any>>;

export type TraverserCallback = TraverserFunction & { traverse: UnaryFn<Traversal<any, any>> };

export type GremlinStep<Args extends any[] = any[]> = (...args: Args) => {
  (traversal: Traversal): Traversal;
  (traversers: Iterable<Traverser<any>>): Generator<Traverser<any>>;
};

export type Step<S, E = S> = {
  args: any[];
  isBarrier?: boolean;
  genus: StepGenus;
  species: string;
  callback: UnaryFn<Iterable<Traverser<S>>, Generator<Traverser<E>, void>>;
};
