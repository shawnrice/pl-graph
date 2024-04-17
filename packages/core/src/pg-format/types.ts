type NodeId = string | number;

type EdgeId = string | number;

type PropValue = string | number;

type Node = {
  id: NodeId;
  labels: string[];
  properties: Record<string, PropValue>;
}

type Edge = {
  /**
   * This is not part of the PG format spec, so it's optional.
   */
  id?: EdgeId;

  from: NodeId;
  to: NodeId;
  undirected: boolean;
  labels: string[];
  properties: Record<string, PropValue>;
}

export type PGFormat = {
  nodes: Node[];
  edges: Edge[];
};

export const isPGFormat = (value: any): value is PGFormat =>
  typeof value === 'object' &&
  Array.isArray(value.nodes) &&
  (typeof value.edges === 'undefined' || Array.isArray(value.edges));
