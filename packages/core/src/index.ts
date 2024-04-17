import { Traversal, Traverser } from './gremlin-v1';
import * as Predicate from './gremlin-v1/Predicates';
import * as Step from './gremlin-v1/Steps';
import * as Token from './gremlin-v1/tokens';
import { NotImplemented } from './gremlin-v1/Traversal/NotImplemented';

// export { useGraphContext } from './react/GraphContext';
// export { GraphProvider } from './react/GraphProvider';
// export { useGraphTraversal } from './react/useGraphTraversal';
// export { useGraphSubscription } from './react/useGraphSubscription';

export const Gremlin = {
  NotImplemented,
  Predicate,
  Step,
  Token,
  Traversal,
  Traverser,
};

// export type { GraphState } from './react/GraphContext';

export { Edge, Graph, isElement, Vertex } from './core';
export { Traversal, Traverser } from './gremlin-v1';

export type {
  EdgeAddedEvent,
  EdgeRemovedEvent,
  Element,
  GraphElement,
  GraphEvents as GraphEvent,
  VertexAddedEvent,
  VertexRemovedEvent,
} from './core';

export * as P from './gremlin-v1/Predicates';
export * as TraversalSteps from './gremlin-v1/Steps';
