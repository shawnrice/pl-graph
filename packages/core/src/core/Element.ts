import { Edge } from './Edge.js';
import { Vertex } from './Vertex.js';

/**
 * An element is either a vertex or an edge in a graph.
 */
export type Element = Vertex | Edge;

/**
 * Alias of `Element` to avoid clashing with the global HTMLElement.
 */
export type GraphElement = Element;

export const isElement = (x: unknown): x is Element => {
  return Vertex.isVertex(x) || Edge.isEdge(x);
};

/**
 * STRUCTURAL element guards, for values flowing through a query.
 *
 * `isElement` above asks `instanceof`, which is right for the store: everything
 * in the graph is a real `Vertex`/`Edge`. A query engine cannot ask that. Its
 * streams carry plain objects — decoded documents, `project()` rows, records —
 * and it has to decide what a value IS from its shape.
 *
 * Both engines had their own copy of this pair, and they had drifted: one asked
 * "is an element and not an edge" for a vertex, the other "has an id and no
 * `from`". Those disagree on `{id, from}` with no `to` — not reachable today
 * (`from` is a GQL reserved word) but a real difference in a predicate that must
 * mean one thing.
 *
 * The stricter reading wins: anything carrying `from` is not a vertex.
 */
export const isEdgeShaped = (x: unknown): boolean =>
  typeof x === 'object' && x !== null && 'from' in x && 'to' in x;

export const isVertexShaped = (x: unknown): boolean =>
  typeof x === 'object' && x !== null && 'id' in x && !('from' in x);
