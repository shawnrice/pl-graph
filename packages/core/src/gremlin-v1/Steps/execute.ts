/* eslint-disable func-style */
/* eslint-disable no-restricted-syntax, functional/immutable-data */
import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { TraverserFunction } from './types';

type OverloadedFn = {
  (x: Traversal): Traversal;
  (x: Iterable<Traverser<any>>): Generator<Traverser<any>>;
};

/**
 * This is not a step but a helper to ensure that we can use these steps in either functional
 * composition or something else
 */
export const execute = (addStep: UnaryFn<Traversal>, callback: TraverserFunction): OverloadedFn => {
  function exec(x: Traversal): Traversal;
  function exec(x: Iterable<Traverser<any>>): Generator<Traverser<any>>;
  function exec(x: Traversal | Iterable<Traverser<any>>): Traversal | Generator<Traverser<any>> {
    return Traversal.isTraversal(x) ? addStep(x) : callback(x);
  }

  return exec;
};
