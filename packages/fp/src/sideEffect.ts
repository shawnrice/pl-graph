import { UnaryFn } from './types';

function* internalSideEffect<T>(effect: UnaryFn<T, any>, iterable: Iterable<T>): Iterable<T> {
  for (const iteration of iterable) {
    effect(iteration);
    yield iteration;
  }
}

export function sideEffect<T>(effect: UnaryFn<T, any>): UnaryFn<Iterable<T>>;
export function sideEffect<T>(effect: UnaryFn<T, any>, iterable: Iterable<T>): Iterable<T>;
export function sideEffect<T>(
  effect: UnaryFn<T, any>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalSideEffect(effect, iterable) : x0 => internalSideEffect(effect, x0);
}
