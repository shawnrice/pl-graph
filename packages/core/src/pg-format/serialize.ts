import { Edge, Graph, Vertex } from '../';
import { PGFormat } from './types';

export const graph2PGJSON = <
  V extends Vertex<any> = Vertex<any>,
  E extends Edge<any, any, any> = Edge<any, any, any>,
>(graph: Graph<V, E>): PGFormat => {
  const nodes: PGFormat['nodes'] = [];
  const edges: PGFormat['edges'] = [];

  for (const vertex of graph.vertices) {
    nodes.push({
      id: vertex.id,
      labels: [...vertex.labels],
      properties: vertex.properties,
    });
  }

  for (const edge of graph.edges) {
    edges.push({
      id: edge.id,
      from: edge.from.id,
      to: edge.to.id,
      undirected: false,
      labels: [...edge.labels],
      properties: edge.properties,
    });
  }

  return { nodes, edges };
};

export const serialize = <
  V extends Vertex<any> = Vertex<any>,
  E extends Edge<any, any, any> = Edge<any, any, any>,
>(graph: Graph<V, E>, space?: string | number): string =>
  JSON.stringify(graph2PGJSON(graph), null, space);
