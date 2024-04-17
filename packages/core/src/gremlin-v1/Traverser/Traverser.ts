import { Graph } from '../../core/Graph';

type TraverserParams<S> = {
  pathMap?: Traverser<any>[];
  sideEffects?: Map<string, any>;
  graph: Graph<any, any>;
  value: S;
};

type CloneTraverserParams<S> = Partial<TraverserParams<S>>;

/**
 * A Traverser wraps the object (Vertex, Edge) being traversed.
 *
 * It holds both the object and metadata about the traversal
 */
export class Traverser<S> {
  graph: Graph<any, any>;

  sack: Record<string, any>;

  sideEffects: Map<string, any>;

  pathMap: Traverser<any>[];

  value: S;

  constructor(params: TraverserParams<S>) {
    this.pathMap = Array.from(params.pathMap ?? []);
    this.sack = {};
    this.sideEffects = new Map(params.sideEffects ?? []);
    this.graph = params.graph;
    this.value = params.value;
    this.pathMap.push(this);
  }

  clone<T = S>(params: CloneTraverserParams<T> = {}): Traverser<T> {
    return new Traverser<T>({
      graph: this.graph,
      pathMap: this.pathMap,
      sideEffects: this.sideEffects,
      // @ts-expect-error: womp
      value: this.value as T,
      ...params,
    });
  }

  get(): S {
    return this.value;
  }

  /**
   * Bulk is used as an optimization
   *
   * @see https://tinkerpop.apache.org/docs/current/reference/#barrier-step notes on "bulking optimization" in
   */
  bulk(): number {
    if ('length' in this.value && typeof this.value.length === 'number') {
      return this.value.length;
    }

    if ('size' in this.value && typeof this.value.size === 'number') {
      return this.value.size;
    }

    if (this.value) {
      return 1;
    }

    return 0;
  }

  /**
   * Not implemented
   */
  path(): any[] {
    return this.pathMap.map(x => x.get());
  }

  loops(): number {
    throw new Error('Not implemented');
    return 0;
  }

  sideEffect(key: string): any {
    return this.sideEffects.get(key);
  }
}
