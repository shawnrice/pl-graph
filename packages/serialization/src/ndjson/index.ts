import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Codec } from '../codec.js';
import { type ChunkSource, linesFromChunks } from '../streaming.js';
import { normalizeBag, type PropertyValue } from '../value.js';

/**
 * **NDJSON** (newline-delimited JSON) — one element per line, each a JSON object
 * tagged with `type`. The streaming-native JSON format: it round-trips the full
 * LPG value model like PG-JSON (JSON natively carries string/number/boolean/
 * null/list, and edges keep their id), but because each element is its own line
 * it streams incrementally — unlike a single JSON document, which can't be
 * `JSON.parse`d from a stream.
 *
 *   {"type":"node","id":"n0","labels":["Person"],"properties":{"name":"marko"}}
 *   {"type":"edge","id":"e0","from":"n0","to":"n1","labels":["CREATED"],"properties":{"weight":0.4}}
 *
 * Node and edge ids are preserved; an edge endpoint not declared as its own node
 * line is created as a bare node. Property values pass through `normalizeBag` at
 * the boundary (see `value.ts`). No lossiness within the LPG model.
 */

type NodeRecord = {
  type: 'node';
  id: string;
  labels: string[];
  properties: Record<string, PropertyValue>;
};
type EdgeRecord = {
  type: 'edge';
  id: string;
  from: string;
  to: string;
  labels: string[];
  properties: Record<string, PropertyValue>;
};

type VertexLike = { id: string; labels: Iterable<string>; properties: Record<string, unknown> };
type EdgeLike = VertexLike & { from: { id: string }; to: { id: string } };

const nodeLine = (vertex: VertexLike): string =>
  JSON.stringify({
    type: 'node',
    id: vertex.id,
    labels: [...vertex.labels],
    properties: normalizeBag(vertex.properties),
  });

const edgeLine = (edge: EdgeLike): string =>
  JSON.stringify({
    type: 'edge',
    id: edge.id,
    from: edge.from.id,
    to: edge.to.id,
    labels: [...edge.labels],
    properties: normalizeBag(edge.properties),
  });

/** Serialize a graph to NDJSON: a node line per vertex, then an edge line per edge. */
export const encode = (graph: Graph): string => {
  const lines: string[] = [];

  for (const vertex of graph.vertices) {
    lines.push(nodeLine(vertex));
  }

  for (const edge of graph.edges) {
    lines.push(edgeLine(edge));
  }

  return lines.join('\n');
};

/** Parse one NDJSON line into a record, or `null` for a blank line. */
const parseLine = (line: string): NodeRecord | EdgeRecord | null => {
  const trimmed = line.trim();

  if (trimmed === '') {
    return null;
  }

  let record: unknown;

  try {
    record = JSON.parse(trimmed);
  } catch (cause) {
    throw new LenkeError(`ndjson: invalid JSON: ${trimmed.slice(0, 80)}`, {
      code: ErrorCode.InvalidJson,
      cause,
    });
  }

  if (typeof record !== 'object' || record === null) {
    throw new LenkeError(
      `ndjson: each line must be a node or edge object: ${trimmed.slice(0, 80)}`,
      {
        code: ErrorCode.InvalidShape,
      },
    );
  }

  const { type } = record as { type?: unknown };

  if (type !== 'node' && type !== 'edge') {
    throw new LenkeError(`ndjson: line is not a 'node' or 'edge' record: ${trimmed.slice(0, 80)}`, {
      code: ErrorCode.InvalidShape,
    });
  }

  return record as NodeRecord | EdgeRecord;
};

const addNode = (graph: Graph, record: NodeRecord): void => {
  graph.addVertex({
    id: String(record.id),
    labels: record.labels ?? [],
    properties: normalizeBag(record.properties ?? {}),
  });
};

// Lenient endpoint policy (streaming): a missing endpoint is created bare.
const ensureVertex = (graph: Graph, id: string) =>
  graph.getVertexById(id) ?? graph.addVertex({ id, labels: [], properties: {} });

// Strict endpoint policy (one-shot decode): a truly-dangling edge is rejected,
// matching native `decode` and the pg-json/graphson/csv codecs. Safe in the
// two-pass `decode` because all node lines are applied before any edge.
const requireVertex = (graph: Graph, id: string) => {
  const v = graph.getVertexById(id);

  if (!v) {
    throw new LenkeError(`edge references a non-existent vertex '${id}'`, {
      code: ErrorCode.MissingVertex,
    });
  }

  return v;
};

const addEdge = (
  graph: Graph,
  record: EdgeRecord,
  resolve: (graph: Graph, id: string) => ReturnType<typeof ensureVertex> = ensureVertex,
): void => {
  const from = resolve(graph, String(record.from));
  const to = resolve(graph, String(record.to));
  graph.addEdge({
    id: record.id != null ? String(record.id) : undefined,
    from,
    to,
    labels: record.labels ?? [],
    properties: normalizeBag(record.properties ?? {}),
  });
};

const apply = (graph: Graph, record: NodeRecord | EdgeRecord): void => {
  if (record.type === 'node') {
    addNode(graph, record);
  } else {
    addEdge(graph, record);
  }
};

/**
 * Deserialize an NDJSON string into `graph`. Two passes — nodes first, then
 * edges — so records may appear in any order. A dangling edge (an endpoint with
 * no node line anywhere in the input) is rejected with `MissingVertex`, matching
 * native `decode` and the other document codecs. (The streaming `decodeStream`
 * keeps the lenient create-bare policy for its single-pass, batch-at-a-time use.)
 */
export const decode = (input: string, graph: Graph): Graph => {
  const wasEnabled = graph.eventsEnabled();

  if (wasEnabled) {
    graph.disableEvents();
  }

  const nodes: NodeRecord[] = [];
  const edges: EdgeRecord[] = [];

  for (const line of input.split('\n')) {
    const record = parseLine(line);

    if (record?.type === 'node') {
      nodes.push(record);
    } else if (record?.type === 'edge') {
      edges.push(record);
    }
  }

  for (const record of nodes) {
    addNode(graph, record);
  }

  for (const record of edges) {
    addEdge(graph, record, requireVertex);
  }

  if (wasEnabled) {
    graph.enableEvents();
    graph.snapshot();
  }

  return graph;
};

/**
 * Stream the document in batched line chunks, without building the whole string.
 * The concatenation of the chunks is BYTE-IDENTICAL to `encode` (the codec-fuzz
 * canonical) — lines are `\n`-separated with NO trailing newline, so the `\n`
 * is emitted as a separator BEFORE each chunk except the first.
 */
export async function* encodeStream(graph: Graph): AsyncGenerator<string> {
  const batchSize = 1024;
  let batch: string[] = [];
  let first = true;
  const flush = (): string => {
    const chunk = first ? batch.join('\n') : `\n${batch.join('\n')}`;
    first = false;
    batch = [];

    return chunk;
  };

  for (const vertex of graph.vertices) {
    batch.push(nodeLine(vertex));

    if (batch.length >= batchSize) {
      yield flush();
    }
  }

  for (const edge of graph.edges) {
    batch.push(edgeLine(edge));

    if (batch.length >= batchSize) {
      yield flush();
    }
  }

  if (batch.length > 0) {
    yield flush();
  }
}

/**
 * Slurp a large graph from a chunk source one line at a time (memory bounded by
 * the longest line). Single pass, so — unlike `decode` — node records must
 * precede the edges that reference them (which `encodeStream` guarantees); an
 * as-yet-unseen endpoint is created as a bare node.
 */
export const decodeStream = async (source: ChunkSource, graph: Graph): Promise<Graph> => {
  const wasEnabled = graph.eventsEnabled();

  if (wasEnabled) {
    graph.disableEvents();
  }

  for await (const line of linesFromChunks(source)) {
    const record = parseLine(line);

    if (record) {
      apply(graph, record);
    }
  }

  if (wasEnabled) {
    graph.enableEvents();
    graph.snapshot();
  }

  return graph;
};

export const ndjsonCodec: Codec = { name: 'ndjson', encode, decode, encodeStream, decodeStream };
