import { rando } from '@pl-graph/utils/src';

import { Graph } from './Graph';
import { Vertex } from './Vertex';

type Property<T = any> = Record<string, unknown>;

type GraphElement = {
  get id(): string;
  get label(): string;

  get graph(): Graph<any, any>;

  keys: () => Set<string>;

  property: <V = any>(key: string, value?: V) => Property<V> | null;

  value: <V = any>(key: string) => Property<V> | null;

  remove: () => void;

  values: <V>(...keys: string[]) => Iterable<V>;

  properties: <V>(...keys: string[]) => Iterable<Property<V>>;
};

type VertexPropertyParams = {
  id?: string;
  labels: string[];
  vertex: Vertex;
};

export const Cardinality = {
  set: Symbol.for('set'),
  list: Symbol.for('list'),
  single: Symbol.for('single'),
} as const;

export class VertexProperty implements GraphElement {
  #id: string;

  #labels: Set<string>;

  #vertex: Vertex;

  #store: Map<string, any[]>;

  static Cardinality = Cardinality;

  static DEFAULT_LABEL = 'vertexProperty';

  constructor(params: VertexPropertyParams) {
    this.#id = params.id ?? rando();
    this.#labels = new Set(params.labels ?? []);
    this.#vertex = params.vertex;

    this.#store = new Map();
  }

  get id(): string {
    return this.#id;
  }

  get label(): string {
    // this is a bit wonky
    return Array.from(this.#labels)[0]!;
  }

  get graph(): Graph<any, any> {
    return this.#vertex.graph;
  }

  keys(): Set<string> {
    return new Set(this.#store.keys());
  }

  element(): Vertex {
    return this.#vertex;
  }

  property(key: string): any;
  property(key: string, value: any): any;
  property(cardinality: symbol, key: string, value: any): any;
  *property(...args: any[]): any {
    if (args.length === 1) {
      const [key] = args;
      const set = this.#store.get(key) ?? new Set();
      for (const v of set) {
        yield { key, value: v };
      }
    } else if (args.length === 2) {
      const [key, value] = args;
      if (!this.#store.has(key)) {
        this.#store.set(key, []);
      }

      this.#store.get(key)!.push(value);
    } else if (args.length === 3) {
      const [cardinality, key, value] = args as [symbol, string, any];
      if (!this.#store.has(key)) {
        this.#store.set(key, []);
      }

      if (cardinality === Symbol.for('set')) {
        const store = this.#store.get(key)!;
        if (!store.includes(value)) {
          store.push(value);
        }
      } else if (cardinality === Symbol.for('list')) {
        this.#store.get(key)!.push(value);
      } else if (cardinality === Symbol.for('single')) {
        this.#store.set(key, [value]);
      }
    }
  }

  *properties<V>(...keys: string[]): Iterable<Property<V>> {
    if (keys.length === 0) {
      for (const [key, set] of this.#store) {
        for (const value of set) {
          yield { key, value };
        }
      }
    } else {
      for (const key of keys) {
        for (const value of this.#store.get(key) ?? new Set()) {
          yield { key, value };
        }
      }
    }
  }

  /**
   * Remove the property from the associated element
   */
  remove(): void {
    // this.#vertex.removeProperty(this);
  }

  *values(...keys: string[]): Iterable<any> {
    if (keys.length === 0) {
      for (const [set] of this.#store) {
        yield* set;
      }
    } else {
      for (const key of keys) {
        yield* this.#store.get(key) ?? new Set();
      }
    }
  }
}
