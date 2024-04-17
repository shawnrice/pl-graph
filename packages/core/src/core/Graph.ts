/* eslint-disable @typescript-eslint/unified-signatures */
import { Emitter, EmitterEvent } from '@pl-graph/emitter/src';
import { timer } from '@pl-graph/utils/src/timer';

import { Traversal } from '../gremlin-v1/Traversal';
import { deserialize } from '../pg-format/deserialize';
import { serialize } from '../pg-format/serialize';
import { Edge } from './Edge';
import { Vertex } from './Vertex';

import type { UnaryFn } from '@pl-graph/fp/src';
import type { UnknownObject } from '@pl-graph/utils/src/types';

import type { AddEdgeParams } from './Edge';
import type {
  GraphEvent,
  GraphEvents,
  GraphEventType,
} from './GraphEvents';
import type { VertexParams } from './Vertex';
type AddVertexParams<V> = Pick<VertexParams<V>, 'id' | 'labels' | 'properties'>;

type GraphOptions = {
  eagerSnapshot: boolean;
};

/**
 * A Property-Label graph.
 *
 * More
 */
export class Graph<
  V extends Vertex<any> = Vertex<any>,
  E extends Edge<any, any, any> = Edge<any, any, any>,
> {
  verticesById: Map<string, V>;

  verticesByLabel: Map<string, Set<V>>;

  vertices: Set<V>;

  edgesById: Map<string, E>;

  edges: Set<E>;

  edgesByLabel: Map<string, Set<E>>;

  edgesFromByLabel: Map<string, Map<string, Set<E>>>;

  edgesToByLabel: Map<string, Map<string, Set<E>>>;

  edgesByVertex: Map<string, Set<E>>;

  eagerSnapshot: boolean;

  elementLabels: Map<string, Set<string>>;

  elementProperties: Map<string, UnknownObject>;

  private readonly listeners: Set<() => any>;

  public nextSnapshot: Graph<V, E> | null;

  private nextSnapshotIsStale: boolean;

  private createNextSnapshot: number | undefined;

  emitter: Emitter<keyof GraphEvents, GraphEvents>;

  constructor(options: Partial<GraphOptions> = {}) {
    this.verticesById = new Map();
    this.vertices = new Set();
    this.edgesById = new Map();
    this.verticesByLabel = new Map();
    this.edges = new Set();
    this.edgesFromByLabel = new Map();
    this.edgesToByLabel = new Map();
    this.edgesByVertex = new Map();
    this.edgesByLabel = new Map();

    this.elementLabels = new Map();
    this.elementProperties = new Map();

    this.eagerSnapshot = options.eagerSnapshot ?? true;

    this.nextSnapshot = null;
    this.nextSnapshotIsStale = true;
    this.emitter = new Emitter({ enabled: true });

    this.listeners = new Set();

    this.createNextSnapshot = undefined;

    const callback = () => {
      this.prepareNextSnapshotLazy();
    };

    const markIsStale = (event: GraphEvent) => {
      // We're creating this as a function so that we can delay its execution with `queueMicrotask`
      // This should allow other user-defined
      const doTheWork = () => {
        if (event.defaultPrevented) {
          return;
        }

        this.nextSnapshotIsStale = true;

        if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
          window.cancelIdleCallback(this.createNextSnapshot!);
          this.createNextSnapshot = window.requestIdleCallback(callback);
        } else {
          global.clearTimeout(this.createNextSnapshot);
          global.setTimeout(callback, 1);
        }
      };

      queueMicrotask(doTheWork);
    };

    const graphMutationEvents: GraphEventType[] = [
      '@graph/VertexAdded',
      '@graph/VertexRemoved',
      '@graph/EdgeAdded',
      '@graph/EdgeRemoved',
      '@graph/LabelAddedToVertex',
      '@graph/LabelRemovedFromVertex',
      '@graph/LabelAddedToEdge',
      '@graph/LabelRemovedFromEdge',
      '@graph/VertexPropertyChanged',
      '@graph/VertexPropertiesChanged',
      '@graph/VertexPropertyRemoved',
      '@graph/VertexPropertiesRemoved',
      '@graph/EdgePropertyChanged',
      '@graph/EdgePropertiesChanged',
      '@graph/EdgePropertyRemoved',
      '@graph/EdgePropertiesRemoved',
    ];

    const onMutate = (event: GraphEventType) => {
      this.emit(new EmitterEvent('@graph/mutate', { original: event }));
    };

    graphMutationEvents.forEach(type => {
      this.on(type, markIsStale);
      this.on(type, onMutate);
    });

    this.eagerSnapshot && this.prepareNextSnapshotLazy();
  }

  /**
   * Creates a graph from a pg-format string
   * @returns Graph
   */
  static from<V extends Vertex<any>, E extends Edge<any, any, any>>(value: string): Graph<V, E> {
    return deserialize(value, new Graph<V, E>());
  }

  get size(): number {
    return this.vertices.size;
  }

  get vertexCount(): number {
    return this.vertices.size;
  }

  get stats(): Record<'vertices' | 'edges', Record<string, number>> {
    return {
      vertices: Object.fromEntries(
        Array.from(this.verticesByLabel.entries()).map(([key, value]) => [key, value.size]),
      ),
      edges: Object.fromEntries(
        Array.from(this.edgesByLabel.entries()).map(([key, value]) => [key, value.size]),
      ),
    };
  }

  get edgeCount(): number {
    return this.edges.size;
  }

  public subscribe = (callback: () => any): (() => void) => {
    this.listeners.add(callback);

    const unsubscribe = () => {
      this.listeners.delete(callback);
    };

    return unsubscribe;
  };

  /**
   * Clones the graph as well as all the vertices and edges
   *
   * @returns Graph<V,E>
   */
  public clone = (options: Partial<GraphOptions> = {}): Graph<V, E> => {
    // So, we need to copy EVERYTHING
    const next = new Graph<V, E>(options);
    next.disableEvents();

    next.verticesById = new Map(this.verticesById);
    next.vertices = new Set(this.vertices);
    next.edgesById = new Map(this.edgesById);
    next.verticesByLabel = new Map(this.verticesByLabel);
    next.edges = new Set(this.edges);
    next.edgesFromByLabel = new Map(this.edgesFromByLabel);
    next.edgesToByLabel = new Map(this.edgesToByLabel);
    next.edgesByVertex = new Map(this.edgesByVertex);
    next.edgesByLabel = new Map(this.edgesByLabel);
    next.elementLabels = new Map(this.elementLabels);
    next.elementProperties = new Map(this.elementProperties);

    return next;
  };

  /**
   * Serializes the current graph to a pg-json string
   */
  public serialize = (space?: string | number): string => {
    return serialize(this, space);
  };

  public truncate = (): void => {
    this.verticesById = new Map();
    this.vertices = new Set();
    this.edgesById = new Map();
    this.verticesByLabel = new Map();
    this.edges = new Set();
    this.edgesFromByLabel = new Map();
    this.edgesToByLabel = new Map();
    this.edgesByVertex = new Map();
    this.edgesByLabel = new Map();
    this.elementLabels = new Map();
    this.elementProperties = new Map();

    this.nextSnapshot = null;
  };

  private readonly notify = (): void => {
    for (const listener of this.listeners) {
      listener();
    }
  };

  private readonly prepareNextSnapshot = (): Graph<V, E> => {
    if (this.nextSnapshot && !this.nextSnapshotIsStale) {
      return this.nextSnapshot;
    }

    /* eslint-disable functional/immutable-data */
    const timeSnapshotCreation = timer(
      `Cloning the graph with ${this.vertexCount} vertices and ${this.edgeCount} edges`,
    );

    this.nextSnapshot = this.clone({ eagerSnapshot: false });
    timeSnapshotCreation();
    this.nextSnapshotIsStale = false;
    /* eslint-enable functional/immutable-data */

    // Notify listeners that there is a new snapshot available
    this.notify();

    return this.nextSnapshot;
  };

  private readonly prepareNextSnapshotLazy = (): void => {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    const rIC =
      typeof window !== 'undefined' && 'requestIdleCallback' in window
        ? window.requestIdleCallback
        : (cb: () => any) => setTimeout(cb, 1);
    rIC(() => this.prepareNextSnapshot(), { timeout: undefined });
  };

  public snapshot = (): Graph<V, E> => {
    return this.prepareNextSnapshot();
  };

  /* Mutation methods */

  /**
   * Adds a Vertex to the Graph
   */
  public addVertex: {
    <T extends Vertex<any>>(vertex: Vertex<T>): Vertex<T>;
    <T extends V['properties']>(params: AddVertexParams<T>): Vertex<T>;
  } = <T extends V['properties'] | Vertex<any>>(
    params: AddVertexParams<T> | Vertex<T>,
  ): Vertex<T> => {
    if (params.id && this.getVertexById(params.id)) {
      // We have already added a vertex with this ID
      return this.getVertexById(params.id)!;
    }

    const vertex: Vertex<T> = Vertex.isVertex<T>(params)
      ? params
      : new Vertex<T>({ ...params, graph: this });

    const event = this.emit(new EmitterEvent('@graph/VertexAdded', vertex));

    if (event.defaultPrevented) {
      return vertex;
    }

    this.vertices.add(vertex);
    this.verticesById.set(vertex.id, vertex);

    for (const label of vertex.labels) {
      this.indexVertexLabel(label, vertex);
    }

    return vertex;
  };

  /**
   * Removes a Vertex from the Graph
   *
   * This also removes any edges that the Vertex was part of
   */
  public removeVertex = (vertex: V | string): V => {
    if (typeof vertex === 'string') {
      if (!this.verticesById.has(vertex)) {
        return null as any;
      }

      return this.removeVertex(this.verticesById.get(vertex)!);
    }

    const event = this.emit(new EmitterEvent('@graph/VertexRemoved', vertex));

    if (event.defaultPrevented) {
      return vertex;
    }

    const edges = this.edgesByVertex.get(vertex.id) ?? new Set();

    for (const edge of edges) {
      this.removeEdge(edge);
    }

    for (const label of vertex.labels) {
      this.deIndexVertexLabel(label, vertex);
    }

    this.vertices.delete(vertex);
    this.verticesById.delete(vertex.id);

    vertex.evict();

    return vertex;
  };

  public addLabelToVertex = (label: string, vertex: Vertex<any>): Vertex<any> => {
    const event = this.emit(new EmitterEvent('@graph/LabelAddedToVertex', { label, vertex }));

    if (event.defaultPrevented) {
      return vertex;
    }

    this.indexVertexLabel(label, vertex);
    const next = new Set(this.elementLabels.get(vertex.id) ?? []);
    next.add(label);
    this.elementLabels.set(vertex.id, next);

    return vertex;
  };

  public removeLabelFromVertex = (label: string, vertex: Vertex<any>): Vertex<any> => {
    const event = this.emit(new EmitterEvent('@graph/LabelRemovedFromVertex', { label, vertex }));

    if (event.defaultPrevented) {
      return vertex;
    }

    this.deIndexVertexLabel(label, vertex);
    const next = new Set(this.elementLabels.get(vertex.id) ?? []);
    next.delete(label);
    this.elementLabels.set(vertex.id, next);
    return vertex;
  };

  // TODO fix the type signature so that it's internally consistent. It's currently optimized for use
  public addEdge: {
    <From extends V, To extends V, P extends UnknownObject>(
      params: Omit<AddEdgeParams<From, To, P>, 'graph'>,
    ): Edge<From, To, P>;
    <From extends E['from'], To extends E['to'], P extends UnknownObject>(
      edge: Edge<From, To, P>,
    ): Edge<Vertex<From>, Vertex<To>, P>;
  } = <From extends E['from'], To extends E['to'], P extends UnknownObject>(
    params:
      | Omit<AddEdgeParams<Vertex<From>, Vertex<To>, P>, 'graph'>
      | Edge<Vertex<From>, Vertex<To>, P>,
  ) => {
    if (params.id && this.getEdgeById(params.id)) {
      return this.getEdgeById(params.id)!;
    }

    const edge: E = Edge.isEdge(params)
      ? params
      : new Edge<From, To, P>({ ...params, graph: this });

    if (!edge.from || !edge.to) {
      console.error('Cannot create edge with missing vertices.');
      return edge;
    }

    const event = this.emit(new EmitterEvent('@graph/EdgeAdded', edge));

    if (event.defaultPrevented) {
      return edge;
    }

    this.edges.add(edge);
    this.edgesById.set(edge.id, edge);

    for (const label of edge.labels) {
      this.indexEdgeLabel(label, edge);
    }

    if (!this.edgesByVertex.has(edge.to.id)) {
      this.edgesByVertex.set(edge.to.id, new Set());
    }

    if (!this.edgesByVertex.has(edge.from.id)) {
      this.edgesByVertex.set(edge.from.id, new Set());
    }

    this.edgesByVertex.get(edge.to.id)?.add(edge);
    this.edgesByVertex.get(edge.from.id)?.add(edge);

    return edge;
  };

  public removeEdge = (edge: E): E => {
    const event = this.emit(new EmitterEvent('@graph/EdgeRemoved', edge));

    if (event.defaultPrevented) {
      return edge;
    }

    for (const label of edge.labels) {
      this.deIndexEdgeLabel(label, edge);
    }

    this.edgesByVertex.get(edge.to.id)?.delete(edge);
    this.edgesByVertex.get(edge.from.id)?.delete(edge);

    this.edgesById.delete(edge.id);
    this.edges.delete(edge);

    edge.evict();

    return edge;
  };

  public addLabelToEdge = <From extends V, To extends V, P extends UnknownObject>(
    label: string,
    edge: Edge<From, To, P>,
  ): Edge<From, To, P> => {
    if (edge.labels.has(label)) {
      // Already added....
      return edge;
    }

    const event = this.emit(new EmitterEvent('@graph/LabelAddedToEdge', { label, edge }));

    if (event.defaultPrevented) {
      return edge;
    }

    const next = new Set(this.elementLabels.get(edge.id) ?? []);
    next.add(label);
    this.elementLabels.set(edge.id, next);

    this.indexEdgeLabel(label, edge);

    return edge;
  };

  public removeLabelFromEdge = <From extends V, To extends V, P extends UnknownObject>(
    label: string,
    edge: Edge<From, To, P>,
  ): Edge<From, To, P> => {
    if (!edge.labels.has(label)) {
      // Already removed...
      return edge;
    }

    const event = this.emit(new EmitterEvent('@graph/LabelRemovedFromEdge', { label, edge }));

    if (event.defaultPrevented) {
      return edge;
    }

    const next = new Set(this.elementLabels.get(edge.id) ?? []);
    next.delete(label);
    this.elementLabels.set(edge.id, next);

    this.deIndexEdgeLabel(label, edge);

    return edge;
  };

  /* Query methods */

  public hasVertex = (vertex: V | string): boolean => {
    if (typeof vertex === 'string') {
      return this.verticesById.has(vertex);
    }

    return this.vertices.has(vertex);
  };

  public owns = (x: V | E): boolean => {
    return (Vertex.isVertex(x) && this.vertices.has(x)) || (Edge.isEdge(x) && this.edges.has(x));
  };

  /**
   * Walk a Graph using a Gremlin Query
   */
  public traverse: {
    <T = any>(query: UnaryFn<Traversal, Traversal | any[]>): Traversal<T> | any[];
    <T = any>(...fns: any[]): Traversal<any, T>;
  } = (...fns: any[]) => {
    const traversal = Traversal.with<any, any>(this);

    if (fns.length) {
      return fns.reduce((g, f) => f(g), traversal);
    }

    return traversal;
  };

  public getVertexById = <X extends V>(id: string): X | null => {
    return (this.verticesById.get(id) ?? null) as X | null;
  };

  public getEdgeById = <X extends E>(id: string): X | null => {
    return (this.edgesById.get(id) ?? null) as X | null;
  };

  public getVerticesByLabel = (label: string): Set<V> => {
    return new Set(this.verticesByLabel.get(label) ?? []);
  };

  public getEdgesByLabel = (label: string): Set<E> => {
    return new Set(this.edgesByLabel.get(label) ?? []);
  };

  /**
   * Deserializes a string from pg-format into this graph
   */
  deserialize(value: string): Graph<V, E> {
    return deserialize(value, this);
  }

  /* Event Emitter Proxy */

  public eventsEnabled = (): boolean => {
    return this.emitter.isEnabled();
  };

  /**
   * Enables the GraphEvents to fire
   */
  public enableEvents = (): void => {
    this.emitter.enable();
  };

  /**
   * Disables the GraphEvents
   *
   * This is potentially useful when hydrating a Graph so as not to create too many snapshots
   */
  public disableEvents = (): void => {
    this.emitter.disable();
  };

  /**
   * Adds an EventListener on a GraphEvent type
   */
  public on = <T extends keyof GraphEvents>(
    type: T,
    listener: (event: GraphEvents[T]) => any,
  ): void => {
    this.emitter.on(type, listener);
  };

  /**
   * Adds an EventListener on a GraphEvent type that will be run only once
   */
  public once = <T extends keyof GraphEvents>(
    type: T,
    listener: (event: GraphEvents[T]) => any,
  ): void => {
    this.emitter.once(type, listener);
  };

  /**
   * Emits a GraphEvent
   */
  public emit = <T extends EmitterEvent<any, any>>(event: T): T => {
    return this.emitter.emit(event);
  };

  /* Internal Methods */

  /**
   * Adds a vertex to all appropriate indices based on a label
   */
  private readonly indexVertexLabel = (label: string, vertex: V): void => {
    if (!this.verticesByLabel.has(label)) {
      this.verticesByLabel.set(label, new Set());
    }

    this.verticesByLabel.get(label)?.add(vertex);
  };

  /**
   * Removes a Vertex from indices based on a label
   */
  private readonly deIndexVertexLabel = (label: string, vertex: V): void => {
    if (this.verticesByLabel.get(label)?.delete(vertex)) {
      if (this.verticesByLabel.get(label)?.size === 0) {
        this.verticesByLabel.delete(label);
      }
    }
  };

  /**
   * Convenience for adding an edge to all the indices based on labels
   */
  private readonly indexEdgeLabel = (label: string, edge: E) => {
    if (!this.edgesByLabel.has(label)) {
      this.edgesByLabel.set(label, new Set());
    }

    this.edgesByLabel.get(label)!.add(edge);

    if (!this.edgesFromByLabel.has(edge.from.id)) {
      this.edgesFromByLabel.set(edge.from.id, new Map());
    }

    const edgesFrom = this.edgesFromByLabel.get(edge.from.id)!;

    if (!edgesFrom.has(label)) {
      edgesFrom.set(label, new Set());
    }

    edgesFrom.get(label)!.add(edge);

    if (!this.edgesToByLabel.has(edge.to.id)) {
      this.edgesToByLabel.set(edge.to.id, new Map());
    }

    const edgesTo = this.edgesToByLabel.get(edge.to.id)!;

    if (!edgesTo.has(label)) {
      edgesTo.set(label, new Set());
    }

    edgesTo.get(label)!.add(edge);
  };

  /**
   * Convenience for removing an edge from all the indices
   *
   * This should be called when removing a label from an edge or deleting an edge
   */
  private readonly deIndexEdgeLabel = (label: string, edge: E) => {
    this.edgesByLabel.get(label)?.delete(edge);

    const fromId = edge.from.id;
    if (this.edgesFromByLabel.get(fromId)?.get(label)?.delete(edge)) {
      if (this.edgesFromByLabel.get(fromId)?.get(label)?.size === 0) {
        this.edgesFromByLabel.get(fromId)?.delete(label);
      }
    }

    const toId = edge.to.id;
    if (this.edgesToByLabel.get(toId)?.get(label)?.delete(edge)) {
      if (this.edgesToByLabel.get(toId)?.get(label)?.size === 0) {
        this.edgesToByLabel.get(toId)?.delete(label);
      }
    }
  };
}
