import type { Graph } from '@lenke/core';
import { query } from '@lenke/gql';

// The render-facing shape: plain data extracted from a @lenke/core Graph.
export type GNode = { id: string; labels: string[]; properties: Record<string, unknown> };
export type GEdge = { id: string; from: string; to: string; labels: string[] };
export type GraphModel = { nodes: GNode[]; edges: GEdge[] };

// The graph's schema at a glance — every vertex label and edge type with its
// count, plus the property keys in use. This is what lets you explore a graph
// you didn't author: you can see what's there before writing a query.
export type LabelCount = { label: string; count: number };
export type Schema = {
  vertexLabels: LabelCount[];
  edgeLabels: LabelCount[];
  vertexKeys: string[];
  edgeKeys: string[];
};

const tally = (m: Map<string, number>, key: string): void => {
  m.set(key, (m.get(key) ?? 0) + 1);
};

const byLabel = (m: Map<string, number>): LabelCount[] =>
  [...m].map(([label, count]) => ({ label, count })).sort((a, b) => a.label.localeCompare(b.label));

export const schemaOf = (graph: Graph): Schema => {
  const vLabels = new Map<string, number>();
  const vKeys = new Set<string>();
  const eLabels = new Map<string, number>();
  const eKeys = new Set<string>();

  for (const v of graph.vertices) {
    for (const l of v.labels) {
      tally(vLabels, l);
    }

    for (const k of Object.keys(v.properties)) {
      vKeys.add(k);
    }
  }

  for (const e of graph.edges) {
    for (const l of e.labels) {
      tally(eLabels, l);
    }

    for (const k of Object.keys(e.properties)) {
      eKeys.add(k);
    }
  }

  return {
    vertexLabels: byLabel(vLabels),
    edgeLabels: byLabel(eLabels),
    vertexKeys: [...vKeys].sort(),
    edgeKeys: [...eKeys].sort(),
  };
};

export const toModel = (graph: Graph): GraphModel => ({
  nodes: [...graph.vertices].map((v) => ({
    id: v.id,
    labels: [...v.labels],
    properties: v.properties,
  })),
  edges: [...graph.edges].map((e) => ({
    id: e.id,
    from: e.from.id,
    to: e.to.id,
    labels: [...e.labels],
  })),
});

// Run a GQL query and collect the vertex ids to highlight: any returned value
// that IS a vertex (a node object carries `.id`) or that equals a vertex id (so
// `RETURN p`, `RETURN element_id(p)`, and `RETURN p.id AS id` all light up the
// matched nodes). Property-only returns (`RETURN p.name`) highlight nothing —
// there's no node to point at.
export const highlightFromQuery = (graph: Graph, text: string): Set<string> => {
  const rows = query(graph, text);
  const valid = new Set([...graph.vertices].map((v) => v.id));
  const ids = new Set<string>();

  const consider = (value: unknown): void => {
    if (typeof value === 'string' && valid.has(value)) {
      ids.add(value);

      return;
    }

    if (value && typeof value === 'object' && 'id' in value) {
      const { id } = value;

      if (typeof id === 'string' && valid.has(id)) {
        ids.add(id);
      }
    }
  };

  for (const row of rows) {
    for (const value of Object.values(row)) {
      consider(value);
    }
  }

  return ids;
};
