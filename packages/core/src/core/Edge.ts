import { EmitterEvent } from '@pl-graph/emitter/src';
import { rando } from '@pl-graph/utils/src';

import { Graph } from './Graph';
import { Vertex } from './Vertex';

export type AddEdgeParams<F extends Vertex, T extends Vertex, P extends Record<string, unknown>> = {
  id?: string;
  from: F;
  graph: Graph<any, any>;
  to: T;
  labels: string[];
  properties: P;
};

export class Edge<
  F extends Vertex = Vertex, // From
  T extends Vertex = F, // To
  P extends Record<string, any> = Record<string, any>, // Properties
> {
  #id: string;

  #from: string;

  #to: string;

  #graph: Graph<any, any> | null;

  /**
   * TypeCheck if something is an `Edge`
   *
   * Does not actually check the `F`, `T`, or `P` of the `Edge<F, T, P>`
   */
  static isEdge<F, T, P>(x: unknown): x is Edge<F, T, P> {
    return x instanceof Edge;
  }

  constructor(params: AddEdgeParams<F, T, P>) {
    const { id = rando(), from, graph, to, labels = [], properties = {} } = params;
    this.#id = id;
    this.#graph = graph;
    this.labels = labels;
    this.properties = properties as P;
    this.#from = from instanceof Vertex ? from.id : from;
    this.#to = to instanceof Vertex ? to.id : to;
  }

  get id(): string {
    return this.#id;
  }

  get graph(): Graph<any, any> {
    return this.#graph!;
  }

  set graph(graph: Graph<any, any>) {
    this.#graph = graph;
  }

  get from(): F {
    return this.#graph!.getVertexById(this.#from)! as F;
  }

  get to(): T {
    return this.#graph!.getVertexById(this.#to)! as T;
  }

  get labels(): Set<string> {
    return this.#graph?.elementLabels.get(this.id) ?? new Set();
  }

  set labels(labels: string[] | Set<string>) {
    this.#graph!.elementLabels.set(this.id, new Set(labels));
  }

  get properties(): P {
    return (this.#graph?.elementProperties.get(this.#id) ?? {}) as P;
  }

  set properties(properties: P) {
    this.#graph!.elementProperties.set(this.id, { ...properties });
  }

  addLabel(label: string): Edge<F, T, P> | null {
    return this.#graph?.addLabelToEdge(label, this) ?? null;
  }

  removeLabel(label: string): Edge<F, T, P> | null {
    return this.#graph?.removeLabelFromEdge(label, this) ?? null;
  }

  hasLabel(label: string): boolean;
  hasLabel(...labels: string[]): boolean;
  hasLabel(...labels: string[]): boolean {
    for (const label of labels) {
      if (this.labels.has(label)) {
        return true;
      }
    }

    return false;
  }

  hasProperty(prop: string): boolean {
    return prop in this.properties;
  }

  getProperty(prop: string): any {
    return this.properties[prop];
  }

  setProperty<V extends any>(key: string, value: V): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/EdgePropertyChanged', { edge: this, key, value }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = { ...this.properties, [key]: value };
  }

  setProperties(props: Partial<P>): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/EdgePropertiesChanged', {
        edge: this,
        next: props,
      }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = { ...this.properties, ...props };
  }

  removeProperty(key: string): void {
    if (!this.hasProperty(key)) {
      return;
    }

    const event = this.#graph?.emit(
      new EmitterEvent('@graph/EdgePropertyRemoved', { edge: this, key }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = Object.fromEntries(
      Object.entries(this.properties).filter(([k]) => key !== k),
    ) as P;
  }

  removeProperties(keys: string[]): void {
    const event = this.#graph?.emit(
      new EmitterEvent('@graph/EdgePropertiesRemoved', { edge: this, keys }),
    );

    if (event?.defaultPrevented) {
      return;
    }

    this.properties = Object.fromEntries(
      Object.entries(this.properties).filter(([k]) => !keys.includes(k)),
    ) as P;
  }

  /**
   * Removes the reference from this to the graph container
   */
  evict(): void {
    this.#graph = null;
    // @ts-expect-error: this is fine
    this.#from = this.#from.id;
    // @ts-expect-error: this is fine
    this.#to = this.#to.id;
  }

  toJSON(): Record<string, any> {
    return {
      id: this.id,
      from: this.from.id,
      to: this.to.id,
      labels: Array.from(this.labels),
      properties: this.properties,
    };
  }
}
