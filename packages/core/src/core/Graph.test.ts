/* eslint-disable yoda */
import { describe, expect, mock, test } from 'bun:test';

import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { createTestGraph, edgeId, vertexId } from '../fixtures/createTestGraph.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { Graph } from './Graph.js';
import { Vertex } from './Vertex.js';

const thrownBy = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return e;
  }

  return undefined;
};

describe('Graph Tests', () => {
  const graph = createTestGraph();

  test('it can get Gene Hackman by id', () => {
    expect(graph.getVertexById(vertexId('89'))!.properties.name).toBe('Gene Hackman');
  });

  test('it can get an edge by id', () => {
    const edge = graph.getEdgeById(edgeId('130'))!;
    expect(edge.from.properties.name).toBe('Gene Hackman');
    expect(edge.hasLabel('ACTED_IN')).toBeTruthy();
    // getProperty<T> types the read — no cast (returns string[] here, not unknown).
    expect(edge.getProperty<string[]>('roles')[0]).toBe('Little Bill Daggett');
    expect(edge.to.properties.title).toBe('Unforgiven');
    expect(edge.to.properties.released).toBe(1992);
  });

  test('We can filter manually', () => {
    // Note, this is us using the indices manually, which isn't the best practice
    const moviesFromTheEarlyMidNineties = Array.from(graph.getVerticesByLabel('Movie'))
      .filter((x) => {
        const released = x.properties.released as number;

        if (!released) {
          return false;
        }

        return 1992 <= released && released < 1995;
      })
      .map((x) => x.properties.title);

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
    const [{ from }] = unforgiven.edgesToByLabel('DIRECTED');
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
      .filter((x) => x.to.id === vertexId('97'))
      .map((x) => x.to);
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
    // Capture into an object: reads of `captured.type`/`.id` use the declared
    // property type, sidestepping the control-flow narrowing-to-`null` that a
    // `let` assigned only inside this listener closure would suffer.
    const captured: { type: string | null; id: string | null } = { type: null, id: null };
    const listener = mock((event) => {
      captured.type = event.type;
      captured.id = event.value.id;
    });
    graph.emitter.once('@graph/VertexAdded', listener);

    graph.enableEvents();

    graph.addVertex({
      id: '99999',
      labels: ['Movie'],
      properties: { a: 1 },
    });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(captured.type).toBe('@graph/VertexAdded');
    expect(captured.id).toBe('99999');
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

    tinkerGraph.addEdge({ from: v, to: v2, labels: ['Not Real'], properties: {} });
    tinkerGraph.removeVertex(v);
    tinkerGraph.removeVertex(v2);

    expect(vertexAdded).toHaveBeenCalledTimes(2);
    expect(vertexAddedOnce).toHaveBeenCalledTimes(1);
    expect(vertexAddedOnce2).toHaveBeenCalledTimes(1);
    expect(vertexRemoved).toHaveBeenCalledTimes(1);
    expect(edgeAdded).toHaveBeenCalledTimes(1);
    expect(edgeRemoved).toHaveBeenCalledTimes(1);
  });

  test('addEdge rejects a label-less edge (removal-cascade invariant)', () => {
    const g = createTestTinkerGraph();
    const a = g.getVertexById('1')!;
    const b = g.getVertexById('2')!;

    // A label-less edge would never land in a label bucket, so removeVertex
    // could not cascade it — reject it up front with a coded error.
    const err = thrownBy(() => g.addEdge({ from: a, to: b, labels: [], properties: {} }));
    expect(hasErrorCode(err, ErrorCode.InvalidGraphOp)).toBe(true);
  });

  test('size is the total element count (vertices + edges), distinct from vertexCount', () => {
    const g = createTestTinkerGraph();
    expect(g.size).toBe(g.vertexCount + g.edgeCount);
    // and not just the vertex count (the historical, misleading behavior).
    expect(g.size).toBeGreaterThan(g.vertexCount);
  });

  test('addEdge throws on a missing endpoint and leaves no orphaned state', () => {
    const g = createTestTinkerGraph();
    const a = g.getVertexById('1')!;
    const stranger = new Vertex({ id: 'not-in-graph', labels: ['X'], properties: {}, graph: g });

    const edgesBefore = g.edgeCount;
    const propBagsBefore = g.elementProperties.size;

    const err = thrownBy(() =>
      g.addEdge({ from: a, to: stranger, labels: ['KNOWS'], properties: { w: 1 } }),
    );
    expect(hasErrorCode(err, ErrorCode.MissingVertex)).toBe(true);

    // Validation happens before construction, so the rejected edge wrote no
    // labels/properties into the graph's element maps.
    expect(g.edgeCount).toBe(edgesBefore);
    expect(g.elementProperties.size).toBe(propBagsBefore);
  });

  describe('clone is a fully independent deep copy', () => {
    test('mutating a clone does not corrupt the source (and vice versa)', () => {
      const g = createTestTinkerGraph();
      const clone = g.clone();

      clone.getVertexById('1')!.setProperty('name', 'CHANGED');
      expect(g.getVertexById('1')!.properties.name).toBe('marko'); // source untouched

      g.getVertexById('2')!.setProperty('name', 'ALSO-CHANGED');
      expect(clone.getVertexById('2')!.properties.name).toBe('vadas'); // clone untouched
    });

    test('clone holds distinct element instances with equal content', () => {
      const g = createTestTinkerGraph();
      const clone = g.clone();

      expect(clone.getVertexById('1')).not.toBe(g.getVertexById('1')); // not aliased
      expect(clone.vertexCount).toBe(g.vertexCount);
      expect(clone.edgeCount).toBe(g.edgeCount);
      expect(clone.getVertexById('1')!.properties).toEqual(g.getVertexById('1')!.properties);

      // Removing from the clone leaves the source intact.
      clone.removeVertex('1');
      expect(clone.getVertexById('1')).toBeNull();
      expect(g.getVertexById('1')).not.toBeNull();
    });

    describe('copy() — async, event-loop-friendly deep copy', () => {
      test('is independent and preserves ids like clone()', async () => {
        const g = createTestTinkerGraph();
        const copy = await g.copy();

        expect(copy.vertexCount).toBe(g.vertexCount);
        expect(copy.edgeCount).toBe(g.edgeCount);
        expect(copy.getVertexById('1')).not.toBe(g.getVertexById('1'));

        copy.getVertexById('1')!.setProperty('name', 'CHANGED');
        expect(g.getVertexById('1')!.properties.name).toBe('marko');
      });

      test('copies indexes (declared + functional) and every constraint kind', async () => {
        const g = new Graph();

        g.createIndex({ on: 'vertex', kind: 'hash', keys: ['v'] });
        g.createIndex({ on: 'edge', kind: 'hash', keys: ['w'] });
        g.createUniqueConstraint('P', 'id');
        g.createRequiredConstraint('P', 'id');
        g.createTypeConstraint('P', 'age', 'number');
        g.createCardinalityConstraint('P', 'R', 'out', 0, 1);
        g.addVertex({ id: 'a', labels: ['P'], properties: { id: 'a', v: 1, age: 30 } });
        g.addVertex({ id: 'b', labels: ['P'], properties: { id: 'b', v: 2, age: 40 } });
        g.addEdge({
          from: g.getVertexById('a')!,
          to: g.getVertexById('b')!,
          labels: ['R'],
          properties: { w: 5 },
        });

        const copy = await g.copy();

        // Indexes: declared on the copy, and functional (a seek finds the row).
        expect([...copy.vertexPropertyIndex.indexedKeys()]).toContain('v');
        expect([...copy.edgePropertyIndex.indexedKeys()]).toContain('w');
        expect([...copy.getVerticesByProperty('v', 1)].map((n) => n.id)).toEqual(['a']);

        // Every constraint kind is enforced on the copy — not just unique.
        expect(() =>
          copy.addVertex({ id: 'dup', labels: ['P'], properties: { id: 'a' } }),
        ).toThrow();
        expect(() => copy.addVertex({ id: 'x', labels: ['P'], properties: { v: 9 } })).toThrow();
        expect(() =>
          copy.addVertex({ id: 'y', labels: ['P'], properties: { id: 'y', age: 'nope' } }),
        ).toThrow();
        expect(() =>
          copy.addEdge({
            from: copy.getVertexById('a')!,
            to: copy.getVertexById('b')!,
            labels: ['R'],
            properties: {},
          }),
        ).toThrow();
      });

      test('yields the event loop while copying a large graph', async () => {
        const g = new Graph();

        for (let i = 0; i < 30_000; i += 1) {
          g.addVertex({ id: `n${i}`, labels: ['P'], properties: { v: i } });
        }

        let timerFired = false;
        const timer = setTimeout(() => {
          timerFired = true;
        }, 0);

        const copy = await g.copy();
        clearTimeout(timer);

        expect(copy.vertexCount).toBe(30_000);
        expect(timerFired).toBe(true);
      });
    });
  });

  describe('the property bag is frozen against external mutation', () => {
    test('a stray top-level write throws instead of silently corrupting', () => {
      const g = createTestTinkerGraph();
      const v = g.getVertexById('1')!;

      // Strict mode (ES module): assigning to a frozen object throws.
      expect(() => {
        (v.properties as { name: string }).name = 'CHANGED';
      }).toThrow();

      // The legitimate path still works and keeps the index/value consistent.
      v.setProperty('name', 'CHANGED');
      expect(v.properties.name).toBe('CHANGED');
    });
  });

  describe('well-formed name validation (ingestion gate)', () => {
    // A `::` label is unrepresentable in GraphSON; an empty label/key is
    // unrepresentable across codecs. Reject at the graph's mutation boundary.
    test('addVertex rejects a :: label, an empty label, and an empty property key', () => {
      const g = createTestTinkerGraph();
      expect(() => g.addVertex({ labels: ['a::b'], properties: {} })).toThrow(/::/);
      expect(() => g.addVertex({ labels: [''], properties: {} })).toThrow(/non-empty/);
      expect(() => g.addVertex({ labels: ['N'], properties: { '': 1 } })).toThrow(/property key/);
      expect(
        hasErrorCode(
          thrownBy(() => g.addVertex({ labels: ['x::y'], properties: {} })),
          ErrorCode.InvalidValue,
        ),
      ).toBe(true);
    });

    test('setProperty rejects an empty key; a single-colon label is fine', () => {
      const g = createTestTinkerGraph();
      const v = g.addVertex({ labels: ['a:b'], properties: { k: 1 } }); // single colon OK
      expect(v.properties.k).toBe(1);
      expect(() => v.setProperty('', 2)).toThrow(/property key/);
    });

    test('addLabelToVertex rejects a :: label', () => {
      const g = createTestTinkerGraph();
      const v = g.addVertex({ labels: ['N'], properties: {} });
      expect(() => g.addLabelToVertex('a::b', v)).toThrow(/::/);
    });

    // The numeric model is float64 (the Rust engine has no bigint; every codec +
    // the FFI param boundary already reject it). A raw JS bigint used to be
    // stored as-is in the pure-JS graph, then handled inconsistently downstream
    // (a `WHERE bigint > n` silently dropped every row). Reject it at the gate.
    test('a bigint property value is rejected with InvalidValue (every write path)', () => {
      const g = createTestTinkerGraph();
      const v = g.addVertex({ labels: ['N'], properties: { n: 1 } });
      const a = g.addVertex({ labels: ['A'], properties: {} });

      // message points at the escape hatch, code is E_INVALID_VALUE
      expect(() => g.addVertex({ labels: ['N'], properties: { big: 1n } })).toThrow(/Number\(/);

      for (const attempt of [
        () => g.addVertex({ labels: ['N'], properties: { big: 1n } }),
        () => g.addVertex({ labels: ['N'], properties: { list: [1, 2n] } }), // hidden in a list
        () => v.setProperty('big', 9007199254740993n),
        () => v.setProperties({ big: 5n }),
        () => g.addEdge({ from: a, to: v, labels: ['E'], properties: { w: 10n } }),
      ]) {
        expect(hasErrorCode(thrownBy(attempt), ErrorCode.InvalidValue)).toBe(true);
      }

      // the escape hatch — an explicit Number() — stores fine
      const ok = g.addVertex({ labels: ['N'], properties: { big: Number(123n) } });
      expect(ok.properties.big).toBe(123);
    });
  });

  describe('ergonomics', () => {
    test('addVertex/addEdge accept no properties (a plain labeled element)', () => {
      const g = createTestGraph();
      const a = g.addVertex({ id: 'x', labels: ['N'] }); // no properties
      const b = g.addVertex({ id: 'y', labels: ['N'] });
      const e = g.addEdge({ from: a, to: b, labels: ['E'] }); // no properties
      expect([...a.labels]).toEqual(['N']);
      expect(e.hasLabel('E')).toBe(true);
      expect(a.properties).toEqual({});
    });

    test('graph.on returns a working unsubscribe', () => {
      const g = createTestGraph();
      let hits = 0;
      const off = g.on('@graph/VertexAdded', () => (hits += 1));
      g.addVertex({ id: 'a1', labels: ['N'], properties: {} });
      expect(hits).toBe(1);
      off();
      g.addVertex({ id: 'a2', labels: ['N'], properties: {} });
      expect(hits).toBe(1); // detached — no further hits
    });

    test('VertexPropertyChanged carries `previous` (for undo without pre-commit reads)', () => {
      const g = createTestGraph();
      const v = g.addVertex({ id: 'v', labels: ['N'], properties: { color: 'red' } });
      const seen: Array<{ previous: unknown; value: unknown }> = [];
      g.on('@graph/VertexPropertyChanged', (ev) =>
        seen.push({ previous: ev.value.previous, value: ev.value.value }),
      );
      v.setProperty('color', 'blue'); // present → new
      v.setProperty('size', 'L'); // absent → present
      expect(seen).toEqual([
        { previous: 'red', value: 'blue' },
        { previous: undefined, value: 'L' },
      ]);
    });

    test('removal + bulk property events carry `previous` (audit can recover deleted values)', () => {
      const g = createTestGraph();
      const v = g.addVertex({ id: 'v', labels: ['N'], properties: { a: 1, b: 2, c: 3 } });

      let removed: unknown;
      let removedBulk: unknown;
      let bulkPrev: unknown;
      g.on('@graph/VertexPropertyRemoved', (ev) => (removed = ev.value.previous));
      g.on('@graph/VertexPropertiesRemoved', (ev) => (removedBulk = ev.value.previous));
      g.on('@graph/VertexPropertiesChanged', (ev) => (bulkPrev = ev.value.previous));

      v.removeProperty('a'); // previous = the removed value
      expect(removed).toBe(1);

      v.setProperties({ b: 20, d: 4 }); // previous = prior values of the written keys
      expect(bulkPrev).toEqual({ b: 2, d: undefined });

      v.removeProperties(['c', 'z']); // previous = only the actually-removed keys
      expect(removedBulk).toEqual({ c: 3 });
    });

    test('truncate() emits a removal event for every element (audit visibility)', () => {
      const g = new Graph();
      const a = g.addVertex({ id: 'a', labels: ['N'], properties: { x: 1 } });
      const b = g.addVertex({ id: 'b', labels: ['N'], properties: { x: 2 } });
      g.addEdge({ id: 'e', from: a, to: b, labels: ['E'], properties: {} });

      const removedVertices: string[] = [];
      const removedEdges: string[] = [];
      let mutations = 0;
      g.on('@graph/VertexRemoved', (ev) => removedVertices.push(ev.value.id));
      g.on('@graph/EdgeRemoved', (ev) => removedEdges.push(ev.value.id));
      g.on('@graph/mutate', () => (mutations += 1));

      g.truncate();

      // Every element is announced (edges first, then vertices), so an audit
      // journal / the CDC WriteLog / a React store sees the most destructive op
      // without special-casing — it used to emit nothing.
      expect(removedEdges).toEqual(['e']);
      expect(removedVertices.sort()).toEqual(['a', 'b']);
      expect(mutations).toBe(3); // all three flow through @graph/mutate
      // …and the graph is actually empty afterward.
      expect([...g.vertices]).toHaveLength(0);
      expect([...g.edges]).toHaveLength(0);
    });
  });
});
