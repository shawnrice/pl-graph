export { Edge, Graph, isElement, Path, PropertyIndex, Vertex } from './core/index.js';
export { validateLabel, validatePropertyKey, validatePropertyValue } from './core/index.js';

export type {
  CardinalityConstraint,
  Element,
  GraphElement,
  GraphOptions,
  IndexableValue,
  InvariantInfo,
  PathElement,
  PathJSON,
  PathStep,
  RangeBound,
  ScalarTypeName,
  ValidatorInfo,
} from './core/index.js';
// The full graph-event surface: every per-event payload type (e.g.
// `VertexPropertyChanged`, `LabelAddedToVertex`), the `GraphEvents` map keyed by
// event name, the `GraphEvent` UNION of all payloads, and `GraphEventType`. (The
// old export mis-aliased the *map* as `GraphEvent` and omitted the payload
// types, so a typed reducer over the union wasn't writable.)
export type * from './core/GraphEvents.js';

// ISO temporal values (DATE / LOCAL DATETIME / DURATION) — the value-model
// foundation shared by the serialization codecs and the query engines.
export {
  LocalDate,
  LocalTime,
  LocalDateTime,
  ZonedTime,
  ZonedDateTime,
  Duration,
  isTemporal,
  coerceTemporal,
  temporalTag,
  temporalFormat,
  temporalParse,
  temporalCmpTotal,
  temporalLiteralParts,
  temporalRelCmp,
  temporalArith,
  durationBetween,
  civilFromDays,
  graphsonType,
  graphsonTag,
  fromTaggedJson,
  parseDate,
  parseLocalTime,
  parseDateTime,
  parseZonedTime,
  parseZonedDateTime,
  parseDuration,
} from './temporal.js';
export type { Clock, Temporal } from './temporal.js';
export { defineEdge, defineNode } from './schema.js';
export type {
  EdgeSchema,
  InferInput,
  InferOutput,
  NodeSchema,
  StandardSchemaV1,
  VertexRef,
} from './schema.js';

// In-engine graph algorithms (degree centrality, …) — data-last, dual-form free
// functions mirroring the native engine byte-for-byte. See ./algorithms.
export {
  betweenness,
  closeness,
  connectedComponents,
  degree,
  labelPropagation,
  neighborAggregate,
  onCycle,
  pagerank,
  personalizedPagerank,
  peerPressure,
  runAlgorithmSync,
  shortestPath,
  stronglyConnectedComponents,
  type AlgorithmConfig,
  type AlgorithmRow,
  type CentralityRow,
  type ClusterRow,
  type ComponentRow,
  type DegreeRow,
  type GraphAlgorithm,
  type LabelRow,
  type NeighborAggregateRow,
  type OnCycleRow,
  type AlgorithmName,
  type PageRankRow,
  type ShortestPathRow,
} from './algorithms/index.js';
