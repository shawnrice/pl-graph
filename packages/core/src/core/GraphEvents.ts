import type { EmitterEvent } from '@lenke/emitter';

import type { Edge } from './Edge.js';
import type { Vertex } from './Vertex.js';

export type VertexAddedEvent = EmitterEvent<'@graph/VertexAdded', Vertex>;

export type VertexRemovedEvent = EmitterEvent<'@graph/VertexRemoved', Vertex>;

export type EdgeAddedEvent = EmitterEvent<'@graph/EdgeAdded', Edge>;

export type EdgeRemovedEvent = EmitterEvent<'@graph/EdgeRemoved', Edge>;

export type LabelAddedToVertex = EmitterEvent<
  '@graph/LabelAddedToVertex',
  { label: string; vertex: Vertex }
>;

export type LabelRemovedFromVertex = EmitterEvent<
  '@graph/LabelRemovedFromVertex',
  { label: string; vertex: Vertex }
>;
export type LabelAddedToEdge = EmitterEvent<
  '@graph/LabelAddedToEdge',
  { label: string; edge: Edge }
>;
export type LabelRemovedFromEdge = EmitterEvent<
  '@graph/LabelRemovedFromEdge',
  { label: string; edge: Edge }
>;

export type VertexPropertyChanged = EmitterEvent<
  '@graph/VertexPropertyChanged',
  // `previous` is the value before the write (`undefined` if the key was absent),
  // so a listener can reverse the change without reading pre-commit state.
  { key: string; value: unknown; previous: unknown; vertex: Vertex }
>;

export type VertexPropertiesChanged = EmitterEvent<
  '@graph/VertexPropertiesChanged',
  // `previous` holds the prior value of each key in `next` (`undefined` if the
  // key was absent), so an audit/undo listener can reverse a bulk write.
  { vertex: Vertex; next: { [key: string]: unknown }; previous: { [key: string]: unknown } }
>;

export type VertexPropertyRemoved = EmitterEvent<
  '@graph/VertexPropertyRemoved',
  // `previous` is the removed value, so an audit can recover it post-commit.
  { vertex: Vertex; key: string; previous: unknown }
>;

export type VertexPropertiesRemoved = EmitterEvent<
  '@graph/VertexPropertiesRemoved',
  // `previous` maps each actually-removed key to its removed value.
  { vertex: Vertex; keys: string[]; previous: { [key: string]: unknown } }
>;

export type EdgePropertyChanged = EmitterEvent<
  '@graph/EdgePropertyChanged',
  // `previous` is the value before the write (`undefined` if the key was absent).
  { key: string; value: unknown; previous: unknown; edge: Edge }
>;

export type EdgePropertiesChanged = EmitterEvent<
  '@graph/EdgePropertiesChanged',
  // `previous` holds the prior value of each key in `next` (`undefined` if absent).
  { edge: Edge; next: { [key: string]: unknown }; previous: { [key: string]: unknown } }
>;

export type EdgePropertyRemoved = EmitterEvent<
  '@graph/EdgePropertyRemoved',
  // `previous` is the removed value.
  { edge: Edge; key: string; previous: unknown }
>;

export type EdgePropertiesRemoved = EmitterEvent<
  '@graph/EdgePropertiesRemoved',
  // `previous` maps each actually-removed key to its removed value.
  { edge: Edge; keys: string[]; previous: { [key: string]: unknown } }
>;

export type OnMutate = EmitterEvent<
  '@graph/mutate',
  {
    original: EmitterEvent<any, any>;
  }
>;

export type GraphEvents = {
  '@graph/VertexAdded': VertexAddedEvent;
  '@graph/VertexRemoved': VertexRemovedEvent;
  '@graph/EdgeAdded': EdgeAddedEvent;
  '@graph/EdgeRemoved': EdgeRemovedEvent;
  '@graph/LabelAddedToVertex': LabelAddedToVertex;
  '@graph/LabelRemovedFromVertex': LabelRemovedFromVertex;
  '@graph/LabelAddedToEdge': LabelAddedToEdge;
  '@graph/LabelRemovedFromEdge': LabelRemovedFromEdge;
  '@graph/VertexPropertyChanged': VertexPropertyChanged;
  '@graph/VertexPropertiesChanged': VertexPropertiesChanged;
  '@graph/VertexPropertyRemoved': VertexPropertyRemoved;
  '@graph/VertexPropertiesRemoved': VertexPropertiesRemoved;
  '@graph/EdgePropertyChanged': EdgePropertyChanged;
  '@graph/EdgePropertiesChanged': EdgePropertiesChanged;
  '@graph/EdgePropertyRemoved': EdgePropertyRemoved;
  '@graph/EdgePropertiesRemoved': EdgePropertiesRemoved;
  '@graph/mutate': OnMutate;
};

type Primitive = string | number | boolean | bigint | symbol | null | undefined;
type Expand<T> = T extends Primitive ? T : { [K in keyof T]: T[K] };

export type GraphEventType = Expand<keyof GraphEvents>;

export type GraphEvent =
  | VertexAddedEvent
  | VertexRemovedEvent
  | EdgeAddedEvent
  | EdgeRemovedEvent
  | LabelAddedToVertex
  | LabelRemovedFromVertex
  | LabelAddedToEdge
  | LabelRemovedFromEdge
  | VertexPropertyChanged
  | VertexPropertiesChanged
  | VertexPropertyRemoved
  | VertexPropertiesRemoved
  | EdgePropertyChanged
  | EdgePropertiesChanged
  | EdgePropertyRemoved
  | EdgePropertiesRemoved
  | OnMutate;
