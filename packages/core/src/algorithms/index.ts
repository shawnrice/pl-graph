// In-engine graph algorithms as data-last, dual-form free functions over the
// public TS `Graph` surface (vertices, adjacency, get/setProperty) — so each
// doubles as a worked example of the "escape hatch": a user writes their own the
// exact same way. Each mirrors the native `@lenke/native` implementation
// vertex-for-vertex (insertion order, no sorting) so results are byte-identical.
export type { AlgorithmConfig, AlgorithmRow, GraphAlgorithm } from './types.js';
export { degree, type DegreeRow } from './degree.js';
export { connectedComponents, type ComponentRow } from './connected-components.js';
export {
  stronglyConnectedComponents,
  onCycle,
  type OnCycleRow,
} from './strongly-connected-components.js';
export { labelPropagation, type LabelRow } from './label-propagation.js';
export { peerPressure, type ClusterRow } from './peer-pressure.js';
export { pagerank, personalizedPagerank, type PageRankRow } from './pagerank.js';
export { betweenness, closeness, type CentralityRow } from './centrality.js';
export { shortestPath, type ShortestPathRow } from './shortest-path.js';
export { neighborAggregate, type NeighborAggregateRow } from './neighbor-aggregate.js';
export { runAlgorithmSync, type AlgorithmName } from './run-sync.js';
