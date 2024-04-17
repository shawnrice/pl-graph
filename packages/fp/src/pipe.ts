import { UnaryFn } from './types';
import { maybeOptimizeIterable } from './utils';

export function pipe<A, B>(x0: UnaryFn<A, B>): UnaryFn<A, B>;
export function pipe<A, B, C>(x0: UnaryFn<A, B>, x1: UnaryFn<B, C>): UnaryFn<A, C>;
export function pipe<A, B, C, D>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
): UnaryFn<A, D>;
export function pipe<A, B, C, D, E>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
): UnaryFn<A, E>;
export function pipe<A, B, C, D, E, F>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
): UnaryFn<A, F>;
export function pipe<A, B, C, D, E, F, G>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
  x5: UnaryFn<F, G>,
): UnaryFn<A, G>;
export function pipe<A, B, C, D, E, F, G, H>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
  x5: UnaryFn<F, G>,
  x6: UnaryFn<G, H>,
): UnaryFn<A, H>;
export function pipe<A, B, C, D, E, F, G, H, I>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
  x5: UnaryFn<F, G>,
  x6: UnaryFn<G, H>,
  x7: UnaryFn<H, I>,
): UnaryFn<A, I>;
export function pipe<A, B, C, D, E, F, G, H, I, J>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
  x5: UnaryFn<F, G>,
  x6: UnaryFn<G, H>,
  x7: UnaryFn<H, I>,
  x8: UnaryFn<I, J>,
): UnaryFn<A, J>;
export function pipe<A, B, C, D, E, F, G, H, I, J, K>(
  x0: UnaryFn<A, B>,
  x1: UnaryFn<B, C>,
  x2: UnaryFn<C, D>,
  x3: UnaryFn<D, E>,
  x4: UnaryFn<E, F>,
  x5: UnaryFn<F, G>,
  x6: UnaryFn<G, H>,
  x7: UnaryFn<H, I>,
  x8: UnaryFn<I, J>,
  x9: UnaryFn<J, K>,
): UnaryFn<A, K>;

export function pipe<T>(...fns: any[]): UnaryFn<Iterable<T>> {
  return x0 => fns.reduce((g, f) => f(g), maybeOptimizeIterable(x0));
}
