import type { Graph } from '@lenke/core';
import {
  graphsonTag,
  graphsonType,
  isTemporal,
  LenkeRecord,
  temporalFormat,
  temporalParse,
} from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Codec } from '../codec.js';
import { normalizeBag } from '../value.js';
import type { PropertyValue } from '../value.js';

/**
 * GraphSON v3.0 (Apache TinkerPop) codec for the `@lenke/core` LPG model.
 *
 * The whole graph is serialized as a single JSON document
 * `{ "vertices": [ <g:Vertex> ... ], "edges": [ <g:Edge> ... ] }`. This is a
 * clean whole-graph wrapper rather than TinkerPop's inline `outE`/`inE`
 * adjacency, but each element uses standard GraphSON v3 typed values of the
 * form `{ "@type": <type>, "@value": <value> }`.
 *
 * LPG ↔ GraphSON mapping and lossiness
 * ------------------------------------
 *   - **Single-value properties.** TinkerPop allows multi-properties and
 *     meta-properties; our LPG models exactly one value per key. We therefore
 *     emit each vertex property key as a *single-element* `g:VertexProperty`
 *     array, and read back only the first element. Documents that carry true
 *     multi-properties lose all but the first value on decode.
 *   - **Multi-label `::` convention.** GraphSON's `label` is a single string,
 *     but our vertices carry a `Set<string>` of labels. To preserve round-trip
 *     fidelity we join multiple labels with `::` (TinkerPop's multi-label
 *     convention) and split on decode; an empty label set encodes as `""` and
 *     decodes back to `[]`. Single-label graphs are therefore standard
 *     GraphSON; only multi/zero-label graphs use the convention.
 *   - **int/float inference.** JS has a single `number` type. We infer the
 *     GraphSON scalar type from the runtime value: `Number.isInteger(n)` →
 *     `g:Int64`, otherwise `g:Double`. A whole-valued float (e.g. `2.0`) thus
 *     round-trips through `g:Int64`; the LPG model does not distinguish the two.
 *   - **Ids** are strings in our model and are preserved exactly, wrapped as
 *     plain JSON strings.
 */

const VERTICES = 'vertices';
const EDGES = 'edges';

type Typed = { '@type': string; '@value': unknown };

/** Encode a single LPG scalar/list value as a GraphSON v3 typed value. */
const encodeValue = (value: PropertyValue): unknown => {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? { '@type': 'g:Int64', '@value': value }
      : { '@type': 'g:Double', '@value': value };
  }

  if (isTemporal(value)) {
    return { '@type': graphsonType(value), '@value': temporalFormat(value) };
  }

  if (value instanceof LenkeRecord) {
    // g:Map — a FLAT [k1, v1, k2, v2, …] array; a string key is bare.
    const flat: unknown[] = [];

    for (const [k, v] of value) {
      flat.push(k);
      flat.push(encodeValue(v as PropertyValue));
    }

    return { '@type': 'g:Map', '@value': flat };
  }

  // Array → g:List of typed values.
  return { '@type': 'g:List', '@value': value.map(encodeValue) };
};

const shapeError = (msg: string): never => {
  throw new LenkeError(`graphson: ${msg}`, { code: ErrorCode.InvalidShape });
};

// Bounds g:List recursion against an adversarial deeply-nested value; mirrors
// the Rust decoder, which serde caps at 128 levels during parsing.
const MAX_NESTING = 128;

/** Decode a GraphSON v3 typed value (or plain JSON scalar) back to an LPG value. */
const decodeValue = (node: unknown, depth = 0): PropertyValue => {
  if (
    node === null ||
    typeof node === 'string' ||
    typeof node === 'boolean' ||
    typeof node === 'number' // a bare JSON number (untyped) — accept as-is
  ) {
    return node;
  }

  if (typeof node !== 'object') {
    return shapeError(`unsupported property value of type ${typeof node}`);
  }

  const typed = node as Typed;

  switch (typed['@type']) {
    case 'g:Int64':
    case 'g:Int32':
    case 'g:Double':
    case 'g:Float': {
      const v = typed['@value'];

      if (typeof v !== 'number') {
        return shapeError(`${typed['@type']} value must be a number`);
      }

      return v;
    }
    case 'g:List': {
      const v = typed['@value'];

      if (!Array.isArray(v)) {
        return shapeError('g:List value must be an array');
      }

      if (depth >= MAX_NESTING) {
        return shapeError('g:List nesting exceeds the maximum depth');
      }

      return v.map((el) => decodeValue(el, depth + 1));
    }
    case 'g:Map': {
      // A flat `[k1, v1, k2, v2, …]` array → a canonical record. Stored map keys
      // are strings, so a non-string key is rejected.
      const v = typed['@value'];

      if (!Array.isArray(v) || v.length % 2 !== 0) {
        return shapeError('g:Map value must be an even-length array');
      }

      if (depth >= MAX_NESTING) {
        return shapeError('g:Map nesting exceeds the maximum depth');
      }

      const entries: [string, unknown][] = [];

      for (let i = 0; i < v.length; i += 2) {
        const key = decodeValue(v[i], depth + 1);

        if (typeof key !== 'string') {
          return shapeError('a stored g:Map key must be a string');
        }

        entries.push([key, decodeValue(v[i + 1], depth + 1)]);
      }

      return LenkeRecord.from(entries);
    }
    default: {
      // A temporal wrapper (`gx:LocalDate`/`gx:LocalDateTime`/`gx:Duration`).
      const tag = graphsonTag(String(typed['@type']));

      if (tag) {
        const v = typed['@value'];

        if (typeof v !== 'string') {
          return shapeError('temporal @value must be a string');
        }

        return temporalParse(tag, v);
      }

      // An unknown/missing wrapper is outside the LPG model (Rust rejects it too,
      // rather than silently storing a raw out-of-model object).
      return shapeError(`unknown typed value '${String(typed['@type'])}'`);
    }
  }
};

const LABEL_SEP = '::';

/** Join an LPG label set into a GraphSON single-label string (`::` convention). */
const joinLabels = (labels: Iterable<string>): string => [...labels].join(LABEL_SEP);

/** Split a GraphSON label string back into an LPG label list. */
const splitLabels = (label: string): string[] => (label === '' ? [] : label.split(LABEL_SEP));

export const encode = (graph: Graph): string => {
  const vertices: unknown[] = [];

  for (const vertex of graph.vertices) {
    const bag = normalizeBag(vertex.properties);
    const properties: Record<string, unknown[]> = {};

    for (const key of Object.keys(bag)) {
      properties[key] = [
        {
          '@type': 'g:VertexProperty',
          '@value': {
            id: `${vertex.id}/${key}`,
            value: encodeValue(bag[key]),
            label: key,
          },
        },
      ];
    }

    vertices.push({
      '@type': 'g:Vertex',
      '@value': {
        id: vertex.id,
        label: joinLabels(vertex.labels),
        properties,
      },
    });
  }

  const edges: unknown[] = [];

  for (const edge of graph.edges) {
    const bag = normalizeBag(edge.properties);
    const properties: Record<string, unknown> = {};

    for (const key of Object.keys(bag)) {
      properties[key] = {
        '@type': 'g:Property',
        '@value': { key, value: encodeValue(bag[key]) },
      };
    }

    edges.push({
      '@type': 'g:Edge',
      '@value': {
        id: edge.id,
        label: joinLabels(edge.labels),
        inV: edge.to.id,
        outV: edge.from.id,
        properties,
      },
    });
  }

  return JSON.stringify({ [VERTICES]: vertices, [EDGES]: edges });
};

type VertexValue = {
  id: string;
  label: string;
  properties?: Record<string, Array<{ '@value': { value: unknown } }>>;
};

type EdgeValue = {
  id: string;
  label: string;
  inV: string;
  outV: string;
  properties?: Record<string, { '@value': { value: unknown } }>;
};

export const decode = (input: string, graph: Graph): Graph => {
  let parsed: unknown;

  try {
    parsed = JSON.parse(input);
  } catch (cause) {
    throw new LenkeError(`graphson: invalid JSON: ${input.slice(0, 80)}`, {
      code: ErrorCode.InvalidJson,
      cause,
    });
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    shapeError('expected a top-level object');
  }

  const { vertices, edges } = parsed as { vertices?: unknown; edges?: unknown };

  if (vertices !== undefined && !Array.isArray(vertices)) {
    shapeError("'vertices' must be an array");
  }

  if (edges !== undefined && !Array.isArray(edges)) {
    shapeError("'edges' must be an array");
  }

  graph.disableEvents();

  for (const wrapper of (vertices as Array<{ '@value'?: unknown }> | undefined) ?? []) {
    const value = wrapper['@value'];

    if (typeof value !== 'object' || value === null) {
      shapeError('each vertex must have an @value object');
    }

    const v = value as VertexValue;

    if (typeof v.id !== 'string' && typeof v.id !== 'number') {
      shapeError('vertex @value.id must be a string or number');
    }

    if (typeof v.label !== 'string') {
      shapeError('vertex @value.label must be a string');
    }

    const properties: Record<string, PropertyValue> = {};

    for (const key of Object.keys(v.properties ?? {})) {
      const entries = v.properties![key];
      // LPG is single-value: read only the first element of the array.
      const [first] = entries ?? [];

      if (first !== undefined) {
        properties[key] = decodeValue(first['@value']?.value);
      }
    }

    graph.addVertex({ id: v.id, labels: splitLabels(v.label), properties });
  }

  for (const wrapper of (edges as Array<{ '@value'?: unknown }> | undefined) ?? []) {
    const value = wrapper['@value'];

    if (typeof value !== 'object' || value === null) {
      shapeError('each edge must have an @value object');
    }

    const e = value as EdgeValue;

    if (typeof e.label !== 'string') {
      shapeError('edge @value.label must be a string');
    }

    const from = graph.getVertexById(e.outV);
    const to = graph.getVertexById(e.inV);

    if (from === null || to === null) {
      throw new LenkeError(
        `GraphSON edge ${e.id} references missing vertex (outV=${e.outV}, inV=${e.inV})`,
        {
          code: ErrorCode.MissingVertex,
          details: { id: e.id, outV: e.outV, inV: e.inV },
        },
      );
    }

    const properties: Record<string, PropertyValue> = {};

    for (const key of Object.keys(e.properties ?? {})) {
      properties[key] = decodeValue(e.properties![key]?.['@value']?.value);
    }

    graph.addEdge({ id: e.id, from, to, labels: splitLabels(e.label), properties });
  }

  graph.enableEvents();

  return graph;
};

export const graphsonCodec: Codec = { name: 'graphson', encode, decode };
