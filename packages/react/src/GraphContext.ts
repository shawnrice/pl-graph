import * as React from 'react';

import { Edge, Graph, Vertex } from '@pl-graph/core/src';

export type GraphState<V extends Vertex = Vertex, E extends Edge = Edge> = { graph: Graph<V, E> };

const defaultGraph = new Graph();

export const GraphContext = React.createContext<GraphState<any, any>>({ graph: defaultGraph });
GraphContext.displayName = 'GraphContext';

export const useGraphContext = <V extends Vertex = Vertex, E extends Edge = Edge>(): GraphState<
  V,
  E
> => React.useContext(GraphContext);
