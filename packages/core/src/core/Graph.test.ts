/* eslint-disable yoda */
import { describe, expect, mock, test } from 'bun:test';

import { createTestGraph, edgeId, vertexId } from '../fixtures/createTestGraph';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph';

describe('Graph Tests', () => {
  const graph = createTestGraph();

  test('it can get Gene Hackman by id', () => {
    expect(graph.getVertexById(vertexId('89'))!.properties.name).toBe('Gene Hackman');
  });

  test('it can get an edge by id', () => {
    const edge = graph.getEdgeById(edgeId('130'))!;
    expect(edge.from.properties.name).toBe('Gene Hackman');
    expect(edge.hasLabel('ACTED_IN')).toBeTruthy();
    expect(edge.getProperty('roles')[0]).toBe('Little Bill Daggett');
    expect(edge.to.properties.title).toBe('Unforgiven');
    expect(edge.to.properties.released).toBe(1992);
  });

  test('We can filter manually', () => {
    // Note, this is us using the indices manually, which isn't the best practice
    const moviesFromTheEarlyMidNineties = Array.from(graph.getVerticesByLabel('Movie'))
      .filter(x => {
        if (!x.properties.released) {
          return false;
        }

        return 1992 <= x.properties.released && x.properties.released < 1995;
      })
      .map(x => x.properties.title);

    expect(moviesFromTheEarlyMidNineties).toEqual([
      'A Few Good Men',
      'Sleepless in Seattle',
      'Unforgiven',
      'Hoffa',
      'A League of Their Own',
    ]);
  });

  test('We can manually walk from a node', () => {
    const unforgiven = graph.getVertexById(vertexId('97'))!;
    const [{ from }] = unforgiven.edgesToByLabel('DIRECTED')!;
    expect(from.properties.name).toBe('Clint Eastwood');
  });

  test('we get an empty set for a non-existent label', () => {
    const unforgiven = graph.getVertexById(vertexId('97'))!;
    expect(Array.from(unforgiven.edgesToByLabel('This Label Does Not Exist'))).toEqual([]);
  });

  test('we can walk the other way', () => {
    const clientEastwood = graph.getVertexById(vertexId('99'))!;
    expect(clientEastwood.properties.name).toBe('Clint Eastwood');
    const movies = clientEastwood.edgesFromByLabel('DIRECTED');

    const [unforgiven] = Array.from(movies)
      .filter(x => x.to.id === vertexId('97'))
      .map(x => x.to);
    expect(unforgiven?.properties.title).toBe('Unforgiven');
  });

  test('we can add labels and then get them', () => {
    const apollo13 = graph.getVertexById(vertexId('142'))!;
    expect(apollo13.properties.title).toBe('Apollo 13');
    expect(apollo13.labels.size).toBe(1);
    expect(apollo13.labels.values().next().value).toBe('Movie');
    apollo13.addLabel('Space');
    expect(apollo13.labels.size).toBe(2);
    expect(graph.getVerticesByLabel('Space').values().next().value).toBe(apollo13);
    apollo13.removeLabel('Space');
    expect(apollo13.labels.size).toBe(1);
    expect(Array.from(graph.getVerticesByLabel('Space'))).toHaveLength(0);
  });

  test('the emitter works', () => {
    let type: string | null = null;
    let id: string | null = null;
    const listener = mock(event => {
      type = event.type;
      id = event.value.id;
    });
    graph.emitter.once('@graph/VertexAdded', listener);

    graph.enableEvents();

    graph.addVertex({
      id: '99999',
      labels: ['Movie'],
      properties: { a: 1 },
    });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(type).toBe('@graph/VertexAdded');
    expect(id).toBe('99999');
  });

  test('we can double-up on the once emitters', () => {
    // vi.useFakeTimers();
    const tinkerGraph = createTestTinkerGraph();
    // vi.runOnlyPendingTimers();

    const vertexAdded = mock(() => {});
    const vertexAddedOnce = mock(() => {});
    const vertexAddedOnce2 = mock(() => {});
    const vertexRemoved = mock(() => {});
    const edgeAdded = mock(() => {});
    const edgeRemoved = mock(() => {});

    tinkerGraph.emitter.on('@graph/VertexAdded', vertexAdded);

    tinkerGraph.emitter.once('@graph/VertexAdded', vertexAddedOnce);
    tinkerGraph.emitter.once('@graph/VertexAdded', vertexAddedOnce2);
    tinkerGraph.emitter.once('@graph/VertexRemoved', vertexRemoved);
    tinkerGraph.emitter.on('@graph/EdgeAdded', edgeAdded);
    tinkerGraph.emitter.on('@graph/EdgeRemoved', edgeRemoved);

    const v = tinkerGraph.addVertex({
      id: '99999',
      labels: ['Movie'],
      properties: { a: 1 },
    });

    const v2 = tinkerGraph.addVertex({
      id: '999999',
      labels: ['Movie'],
      properties: { a: 1 },
    });

    tinkerGraph.addEdge({ from: v, to: v2, labels: ['Not Real'] });
    tinkerGraph.removeVertex(v);
    tinkerGraph.removeVertex(v2);

    expect(vertexAdded).toHaveBeenCalledTimes(2);
    expect(vertexAddedOnce).toHaveBeenCalledTimes(1);
    expect(vertexAddedOnce2).toHaveBeenCalledTimes(1);
    expect(vertexRemoved).toHaveBeenCalledTimes(1);
    expect(edgeAdded).toHaveBeenCalledTimes(1);
    expect(edgeRemoved).toHaveBeenCalledTimes(1);
  });
});
