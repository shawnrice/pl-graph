import { EmitterEvent } from '../../../emitter/src';
import { Edge } from './Edge';
import { Vertex } from './Vertex';

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
  { key: string; value: any; vertex: Vertex }
>;

export type VertexPropertiesChanged = EmitterEvent<
  '@graph/VertexPropertiesChanged',
  { vertex: Vertex; next: { [key: string]: any } }
>;

export type VertexPropertyRemoved = EmitterEvent<
  '@graph/VertexPropertyRemoved',
  { vertex: Vertex; key: string }
>;

export type VertexPropertiesRemoved = EmitterEvent<
  '@graph/VertexPropertiesRemoved',
  { vertex: Vertex; keys: string[] }
>;

export type EdgePropertyChanged = EmitterEvent<
  '@graph/EdgePropertyChanged',
  { key: string; value: any; edge: Edge }
>;

export type EdgePropertiesChanged = EmitterEvent<
  '@graph/EdgePropertiesChanged',
  { edge: Edge; next: { [key: string]: any } }
>;

export type EdgePropertyRemoved = EmitterEvent<
  '@graph/EdgePropertyRemoved',
  { edge: Edge; key: string }
>;

export type EdgePropertiesRemoved = EmitterEvent<
  '@graph/EdgePropertiesRemoved',
  { edge: Edge; keys: string[] }
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
type Builtin = Primitive | Function | Date | Error | RegExp;
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
