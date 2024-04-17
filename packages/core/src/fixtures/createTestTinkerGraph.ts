/* eslint-disable @typescript-eslint/no-magic-numbers */
/* eslint-disable sort-keys */
import { Edge } from '../core/Edge';
import { Graph } from '../core/Graph';
import { Vertex } from '../core/Vertex';

type Person = Vertex<{ name: string; age: number }>;
type Software = Vertex<{ name: string; lang: string }>;

type TinkerGraphVertex = Person | Software;

type PersonKnows = Edge<Person, Person, { weight: number }>;
type PersonCreated = Edge<Person, Software, { weight: number }>;

type TinkerGraphEdge = PersonKnows | PersonCreated;

type TinkerGraph = Graph<TinkerGraphVertex, TinkerGraphEdge>;

/**
 * Creates an instance of the example graph from the gremlin documentation
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#graph-computing
 * @see https://tinkerpop.apache.org/docs/current/images/tinkerpop-modern.png
 */
export const createTestTinkerGraph = (): TinkerGraph => {
  const graph = new Graph<TinkerGraphVertex, TinkerGraphEdge>();
  graph.disableEvents();

  const people = [
    { id: '1', name: 'marko', age: 29 },
    { id: '2', name: 'vadas', age: 27 },
    { id: '4', name: 'josh', age: 32 },
    { id: '6', name: 'peter', age: 35 },
  ];

  const software = [
    { id: '3', name: 'lop', lang: 'java' },
    { id: '5', name: 'ripple', lang: 'java' },
  ];

  for (const person of people) {
    graph.addVertex({
      id: person.id,
      labels: ['PERSON'],
      properties: { name: person.name, age: person.age },
    });
  }

  for (const program of software) {
    graph.addVertex({
      id: program.id,
      labels: ['SOFTWARE'],
      properties: { name: program.name, lang: program.lang },
    });
  }

  graph.addEdge({
    id: '7',
    from: graph.getVertexById('1')!,
    to: graph.getVertexById('2')!,
    labels: ['KNOWS'],
    properties: { weight: 0.5 },
  });

  graph.addEdge({
    id: '8',
    from: graph.getVertexById('1')!,
    to: graph.getVertexById('4')!,
    labels: ['KNOWS'],
    properties: { weight: 1.0 },
  });

  graph.addEdge({
    id: '9',
    from: graph.getVertexById('1')!,
    to: graph.getVertexById('3')!,
    labels: ['CREATED'],
    properties: { weight: 0.4 },
  });

  graph.addEdge({
    id: '10',
    from: graph.getVertexById('4')!,
    to: graph.getVertexById('5')!,
    labels: ['CREATED'],
    properties: { weight: 1.0 },
  });

  graph.addEdge({
    id: '11',
    from: graph.getVertexById('4')!,
    to: graph.getVertexById('3')!,
    labels: ['CREATED'],
    properties: { weight: 0.4 },
  });

  graph.addEdge({
    id: '12',
    from: graph.getVertexById('6')!,
    to: graph.getVertexById('3')!,
    labels: ['CREATED'],
    properties: { weight: 0.2 },
  });

  graph.enableEvents();

  return graph;
};
