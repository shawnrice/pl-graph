import type { Backend } from '@lenke/native';

/**
 * Build a {@link Backend} backed by the native napi addon, so the whole
 * `@lenke/native` facade — `graphFromNdjson`, the `RustGraph` tagged-template
 * `query`/`gremlin`, `createStore` + `liveQuery` — runs on Node against the fast
 * N-API engine with no code changes.
 *
 * @example
 * ```ts
 * import { createNodeBackend } from '@lenke/node/backend';
 * import {
  createStore,
  graphFromNdjson,
} from '@lenke/native';
 *
 * const backend = createNodeBackend();
 * const g = graphFromNdjson(backend, await readFile('graph.ndjson'));
 * const people = createStore(g).liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
 * ```
 */
export declare function createNodeBackend(): Backend;
