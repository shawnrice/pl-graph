import { EmitterEvent } from '@pl-graph/emitter/src';
import { rando } from '@pl-graph/utils/src';

import { Edge } from './Edge';
import { Graph } from './Graph';

import type { UnknownObject } from '@pl-graph/utils/src/types';

export type VertexParams<P extends { [key: string]: any }> = {
  id?: string;
  labels: string[];
  graph: Graph<any, any>;
  properties: P;
};

export type VertexJSON<T extends Record<string, any>> = {
  id: string;
  labels: string[];
  properties: T;
};

export class Vertex<
  T extends Record<string, any> = Record<string, any>, // Properties
> {
  #id: string;

  #graph: Graph<any, any> | null;

  static from<T extends UnknownObject>(params: VertexParams<T>): Vertex<T> {
    return new Vertex<T>(params);
  }

  /**
   * TypeCheck if something is a `Vertex`
   *
   * Does not actually check the `T` of the `Vertex<T>`
   */
  static isVertex<T extends UnknownObject = UnknownObject>(x: unknown): x is Vertex<T> {
    return x instanceof Vertex;
  }

  constructor(params: VertexParams<T>) {
    const { id = rando(), labels = [], graph, properties = {} } = params;
    this.#id = id;
    this.#graph = graph;
    // @ts-expect-error: this is inevitable
    this.properties = Object.assign({}, properties);
    this.labels = labels;
  }

  get graph(): Graph<any, any> {
    return this.#graph!;
  }

  set graph(graph: Graph<any, any>) {
    this.#graph = graph;
  }

  get id(): string {
    return this.#id;
  }

  get labels(): Set<string> {
    return this.#graph!.elementLabels.get(this.id) ?? new Set();
  }

  set labels(labels: string[] | Set<string>) {
    this.#graph!.elementLabels.set(this.id, new Set(labels));
  }

  get properties(): T {
    return (this.#graph?.elementProperties.get(this.#id) ?? {}) as T;
  }

  set properties(properties: T) {
    this.#graph!.elementProperties.set(this.#id, { ...properties });
  }

  getProperty<K extends string>(key: K): T[K] {
    return this.properties[key];
  }

  setProperty<K extends string>(key: K, value: T[K]): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/VertexPropertyChanged', { vertex: this, key, value }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = { ...this.properties, [key]: value };
  }

  setProperties(props: Partial<T>): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/VertexPropertiesChanged', {
        vertex: this,
        next: props,
      }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    if (!this.#graph) {
      console.error('NO GRAPH');
      throw new Error('No graph');
    }

    this.properties = { ...this.properties, ...props };
  }

  removeProperty<K extends string>(key: K): void {
    if (!this.hasProperty(key)) {
      return;
    }

    const event = this.#graph?.emit(
      new EmitterEvent('@graph/VertexPropertyRemoved', { vertex: this, key }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = Object.fromEntries(
      Object.entries(this.properties).filter(([k]) => key !== k),
    ) as T;
  }

  removeProperties(keys: string[]): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/VertexPropertiesRemoved', { vertex: this, keys }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = Object.fromEntries(
      Object.entries(this.properties).filter(([k]) => !keys.includes(k)),
    ) as T;
  }

  hasProperty(key: string): boolean {
    return key in this.properties;
  }

  hasLabel(label: string): boolean {
    return this.labels.has(label);
  }

  addLabel(label: string): Vertex<T> {
    this.#graph?.addLabelToVertex(label, this);
    return this;
  }

  removeLabel(label: string): Vertex<T> {
    this.#graph?.removeLabelFromVertex(label, this);
    return this;
  }

  evict(): void {
    this.#graph = null;
  }

  edgesFromByLabel(label: string): Set<Edge<Vertex<T>, any, any>> {
    return this.#graph?.edgesFromByLabel.get(this.id)?.get(label) ?? new Set();
  }

  edgesToByLabel(label: string): Set<Edge<any, Vertex<T>, any>> {
    return this.#graph?.edgesToByLabel.get(this.id)?.get(label) ?? new Set();
  }

  toString(): string {
    return `Vertex (${this.id}) {}`;
  }

  toJSON(): VertexJSON<T> {
    return {
      id: this.id,
      labels: Array.from(this.labels),
      properties: this.properties,
    };
  }
}
