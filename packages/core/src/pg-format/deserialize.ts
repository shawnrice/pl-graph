import { Edge } from '../core/Edge';
import { Graph } from '../core/Graph';
import { Vertex } from '../core/Vertex';

/**
 * Deserialize a pg-format string into a graph
 *
 * @see https://pg-format.readthedocs.io/en/0.3/contents/reference.html
 *
 */
export const deserialize = <
  V extends Vertex<any> = Vertex<any>,
  E extends Edge<any, any, any> = Edge<any, any, any>,
>(
  value: string,
  graph: Graph<V, E>,
): Graph<V, E> => {
  // Todo, guard this in case there's bad json data
  const data = JSON.parse(value);

  const { nodes, edges } = data;

  const eventsEnabled = graph.eventsEnabled();

  if (eventsEnabled) {
    graph.disableEvents();
  }

  for (const node of nodes) {
    graph.addVertex({
      id: node.id,
      labels: node.labels,
      properties: node.properties,
    });
  }

  for (const edge of edges) {
    const from = graph.getVertexById(edge.from);
    const to = graph.getVertexById(edge.to);

    if (!from || !to) {
      console.log({ from: edge.from, to: edge.to });
      throw new Error(`Edge references a non-existent vertex`);
    }

    graph.addEdge({
      id: edge.id ?? `${from.id}-[${edge.labels.join(',')}]->${to.id}`,
      from,
      to,
      labels: edge.labels,
      properties: edge.properties,
    });
  }

  if (eventsEnabled) {
    // Go ahead and re-enable events
    graph.enableEvents();
    // Force a snapshot to be recreated as this would
    // have happened if the events were enabled through this process
    graph.snapshot();
  }

  return graph;
};
