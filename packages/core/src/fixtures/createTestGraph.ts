/* eslint-disable no-magic-numbers */
/* eslint-disable sort-keys */
import fs from 'node:fs';
import path from 'node:path';

import { Edge } from '../core/Edge';
import { Graph } from '../core/Graph';
import { Vertex } from '../core/Vertex';

const idFrom = (type: string, id: string) => [type, id].join(':');

const movies = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'movies.json'), 'utf-8'));

/**
 * Test helper
 */
export const vertexId = (id: string): string => idFrom('vertex', id);

/**
 * Test helper
 */
export const edgeId = (id: string): string => idFrom('edge', id);

type Movie = Vertex<{
  tagline: string;
  title: string;
  released: number;
}>;

type Person = Vertex<{
  born: number;
  name: string;
}>;

type ActedIn = Edge<Person, Movie, { roles: string[] }>;

type Directed = Edge<Person, Movie, {}>;

type Wrote = Edge<Person, Movie, {}>;

type Produced = Edge<Person, Movie, {}>;

type MovieVertices = Movie | Person;
type MovieEdges = ActedIn | Directed | Wrote | Produced;

/**
 * Test helper
 */
export const createTestGraph = (): Graph<MovieVertices, MovieEdges> => {
  const graph = new Graph<MovieVertices, MovieEdges>();
  graph.disableEvents();
  movies.forEach(x => {
    if (x.type === 'node') {
      graph.addVertex({
        id: vertexId(x.id),
        labels: x.labels,
        properties: x.properties,
      });
    } else if (x.type === 'relationship') {
      graph.addEdge<Person, Movie, any>({
        id: edgeId(x.id),
        from: graph.getVertexById(vertexId(x.start.id))!,
        to: graph.getVertexById(vertexId(x.end.id))!,
        properties: x.properties,
        labels: [x.label as string],
      });
    }
  });

  graph.enableEvents();

  return graph;
};
