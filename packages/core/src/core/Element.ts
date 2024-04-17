import { Edge } from './Edge';
import { Vertex } from './Vertex';

type AnyObject = Record<string, any>;

/**
 * An element is a either a vertex or an edge in a graph
 * note: We need to deal with the generics here....
 */
export type Element<
  F extends AnyObject = AnyObject,
  T extends AnyObject = AnyObject,
  P extends AnyObject = AnyObject,
> = Vertex<F> | Edge<Vertex<F>, Vertex<T>, P>;

/**
 * An alias of `Element` so that we don't clash with the HTMLElement as much
 */
export type GraphElement<
  F extends AnyObject = AnyObject,
  T extends AnyObject = AnyObject,
  P extends AnyObject = AnyObject,
> = Element<F, T, P>;

export const isElement = <
  F extends AnyObject = AnyObject,
  T extends AnyObject = AnyObject,
  P extends AnyObject = AnyObject,
>(
  x: unknown,
): x is Element<F, T, P> => {
  return Vertex.isVertex(x) || Edge.isEdge(x);
};
