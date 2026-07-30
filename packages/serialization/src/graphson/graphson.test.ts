import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { graphContentEqual, randomLpgGraph } from '../testkit.js';
import { decode, encode, graphsonCodec } from './index.js';

const errOf = (input: string): unknown => {
  try {
    decode(input, new Graph());
  } catch (e) {
    return e;
  }

  return undefined;
};

type Typed = { '@type': string; '@value': unknown };

type Wrapper = { '@type': string; '@value': any };

const parse = (g: Graph): { vertices: Wrapper[]; edges: Wrapper[] } => JSON.parse(encode(g));

describe('graphson typed-value shapes', () => {
  test('integers encode as g:Int64, floats as g:Double', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: ['Person'], properties: { age: 42, height: 1.75 } });
    const doc = parse(g);
    const props = doc.vertices[0]['@value'].properties;
    const age = props.age[0]['@value'].value as Typed;
    const height = props.height[0]['@value'].value as Typed;
    expect(age).toEqual({ '@type': 'g:Int64', '@value': 42 });
    expect(height).toEqual({ '@type': 'g:Double', '@value': 1.75 });
  });

  test('strings and booleans encode as plain JSON, null as null', () => {
    const g = new Graph();
    g.addVertex({
      id: 'n0',
      labels: ['T'],
      properties: { s: 'hi', b: true, z: null },
    });
    const props = parse(g).vertices[0]['@value'].properties;
    expect(props.s[0]['@value'].value).toBe('hi');
    expect(props.b[0]['@value'].value).toBe(true);
    expect(props.z[0]['@value'].value).toBeNull();
  });

  test('lists encode as g:List of typed values', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: ['T'], properties: { tags: [1, 'a', 2.5] } });
    const value = parse(g).vertices[0]['@value'].properties.tags[0]['@value'].value as Typed;
    expect(value['@type']).toBe('g:List');
    expect(value['@value']).toEqual([
      { '@type': 'g:Int64', '@value': 1 },
      'a',
      { '@type': 'g:Double', '@value': 2.5 },
    ]);
  });

  test('vertex wrapper has g:Vertex + single-element g:VertexProperty arrays', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: ['Person'], properties: { name: 'x' } });
    const [wrapper] = parse(g).vertices;
    expect(wrapper['@type']).toBe('g:Vertex');
    expect(wrapper['@value'].id).toBe('n0');
    expect(wrapper['@value'].label).toBe('Person');
    const arr = wrapper['@value'].properties.name;
    expect(Array.isArray(arr)).toBe(true);
    expect(arr).toHaveLength(1);
    expect(arr[0]['@type']).toBe('g:VertexProperty');
    expect(arr[0]['@value'].label).toBe('name');
  });

  test('edge wrapper has g:Edge with inV/outV and g:Property values', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['T'], properties: {} });
    const b = g.addVertex({ id: 'b', labels: ['T'], properties: {} });
    g.addEdge({ id: 'e0', from: a, to: b, labels: ['KNOWS'], properties: { w: 3 } });
    const [wrapper] = parse(g).edges;
    expect(wrapper['@type']).toBe('g:Edge');
    expect(wrapper['@value'].id).toBe('e0');
    expect(wrapper['@value'].label).toBe('KNOWS');
    expect(wrapper['@value'].outV).toBe('a');
    expect(wrapper['@value'].inV).toBe('b');
    const { w } = wrapper['@value'].properties;
    expect(w['@type']).toBe('g:Property');
    expect(w['@value'].key).toBe('w');
    expect(w['@value'].value).toEqual({ '@type': 'g:Int64', '@value': 3 });
  });
});

describe('graphson multi-label :: convention', () => {
  test('decode preserves the target graph events flag (does not force-enable)', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: ['A'], properties: { x: 1 } });
    const doc = encode(g);

    // Decoding inside an events-off batch must leave events off — this once
    // unconditionally re-enabled, diverging from the other four codecs.
    const off = new Graph();
    off.disableEvents();
    decode(doc, off);
    expect(off.eventsEnabled()).toBe(false);

    // Default (events on) still ends up on.
    const on = new Graph();
    decode(doc, on);
    expect(on.eventsEnabled()).toBe(true);
  });

  test('multiple labels join with :: and split back', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: ['A', 'B'], properties: {} });
    expect(parse(g).vertices[0]['@value'].label).toBe('A::B');
    const back = decode(encode(g), new Graph());
    expect([...back.getVertexById('n0')!.labels].sort()).toEqual(['A', 'B']);
  });

  test('empty label set encodes as "" and decodes to []', () => {
    const g = new Graph();
    g.addVertex({ id: 'n0', labels: [], properties: {} });
    expect(parse(g).vertices[0]['@value'].label).toBe('');
    const back = decode(encode(g), new Graph());
    expect([...back.getVertexById('n0')!.labels]).toEqual([]);
  });
});

describe('graphson: malformed input is a clean error, not a crash', () => {
  test('invalid JSON → InvalidJson (not a raw SyntaxError)', () => {
    expect(hasErrorCode(errOf('{bad'), ErrorCode.InvalidJson)).toBe(true);
  });

  test('non-object top level → InvalidShape', () => {
    expect(hasErrorCode(errOf('[]'), ErrorCode.InvalidShape)).toBe(true);
  });

  test('vertex missing @value / id / label → InvalidShape (was a TypeError)', () => {
    expect(hasErrorCode(errOf('{"vertices":[{"@type":"g:Vertex"}]}'), ErrorCode.InvalidShape)).toBe(
      true,
    );
    expect(
      hasErrorCode(errOf('{"vertices":[{"@value":{"label":""}}]}'), ErrorCode.InvalidShape),
    ).toBe(true);
    expect(
      hasErrorCode(errOf('{"vertices":[{"@value":{"id":"a"}}]}'), ErrorCode.InvalidShape),
    ).toBe(true);
  });

  test('non-string vertex label → InvalidShape (was a TypeError)', () => {
    expect(
      hasErrorCode(
        errOf('{"vertices":[{"@value":{"id":"a","label":42}}]}'),
        ErrorCode.InvalidShape,
      ),
    ).toBe(true);
  });

  test('g:List with a non-array @value → InvalidShape (was a TypeError)', () => {
    const doc =
      '{"vertices":[{"@value":{"id":"a","label":"","properties":' +
      '{"k":[{"@value":{"value":{"@type":"g:List","@value":5}}}]}}}]}';
    expect(hasErrorCode(errOf(doc), ErrorCode.InvalidShape)).toBe(true);
  });

  test('a well-formed minimal document still decodes', () => {
    const g = decode('{"vertices":[{"@value":{"id":"a","label":""}}]}', new Graph());
    expect(g.vertexCount).toBe(1);
  });
});

describe('graphson round-trip property test', () => {
  test('decode(encode(g)) === g for >=300 seeds', () => {
    for (let seed = 0; seed < 320; seed += 1) {
      const original = randomLpgGraph(seed);
      const roundTripped = decode(encode(original), new Graph());

      if (!graphContentEqual(roundTripped, randomLpgGraph(seed))) {
        throw new Error(`round-trip mismatch at seed ${seed}`);
      }
    }
  });

  test('codec object exposes name + encode/decode', () => {
    expect(graphsonCodec.name).toBe('graphson');
    const g = randomLpgGraph(7);
    expect(graphContentEqual(graphsonCodec.decode(graphsonCodec.encode(g), new Graph()), g)).toBe(
      true,
    );
  });
});

describe('graphson throughput smoke', () => {
  test('encode/decode ~10k elements', () => {
    const g = new Graph();
    g.disableEvents();
    const n = 5000;
    const verts = [];

    for (let i = 0; i < n; i += 1) {
      verts.push(
        g.addVertex({
          id: `v${i}`,
          labels: ['Node'],
          properties: { i, name: `node-${i}`, ratio: i + 0.5, tags: [i, 'x'] },
        }),
      );
    }

    for (let i = 0; i < n; i += 1) {
      g.addEdge({
        id: `e${i}`,
        from: verts[i],
        to: verts[(i + 1) % n],
        labels: ['NEXT'],
        properties: { w: i * 0.25 },
      });
    }

    g.enableEvents();

    const start = performance.now();
    const json = encode(g);
    const back = decode(json, new Graph());
    const elapsed = performance.now() - start;

    expect(back.vertexCount).toBe(n);
    expect(back.edgeCount).toBe(n);
    expect(graphContentEqual(back, g)).toBe(true);
    // eslint-disable-next-line no-console
    console.log(`graphson throughput: ${2 * n} elements in ${elapsed.toFixed(1)}ms`);
  });
});
