/**
 * `@lenke/native` — the Rust columnar core, callable from JS/TS.
 *
 * One C ABI, two backends behind a single {@link Backend} contract. The
 * backends live behind subpath exports so that loading this package in a
 * browser never pulls in the Bun-only `bun:ffi` builtin:
 *   - `@lenke/native/ffi` → `createFfiBackend` loads the native dynamic
 *     library over `bun:ffi` (server / CLI), and
 *   - `@lenke/native/wasm` → `createWasmBackend` instantiates the wasm
 *     artifact (browser).
 *
 * This entry is environment-neutral: shared types plus the engine-neutral
 * {@link RustGraph} facade — GQL via `query`, Gremlin via `gremlin`, both with
 * the same tagged-template ergonomics as `@lenke/gql`.
 *
 * @example bun / server
 * ```ts
 * import { createFfiBackend } from '@lenke/native/ffi';
 * import {
  graphFromNdjson,
} from '@lenke/native';
 * const backend = createFfiBackend('/path/to/liblenke_core.dylib');
 * const g = graphFromNdjson(backend, await Bun.file('graph.ndjson').bytes());
 * g.query`MATCH (a:Person) RETURN a.name`;
 * ```
 *
 * @example browser
 * ```ts
 * import { createWasmBackend } from '@lenke/native/wasm';
 * import { graphFromNdjson } from '@lenke/native';
 * const backend = await createWasmBackend(fetch('/lenke_core.wasm'));
 * const g = graphFromNdjson(backend, ndjsonBytes);
 * ```
 */

export { ABI_VERSION } from './abi.js';
// Re-exported so consumers can name the `type` arg of `createTypeConstraint` /
// `createEdgeTypeConstraint` (and build replicable schema ops) without reaching
// past this facade into `@lenke/core`.
export type { ScalarTypeName } from '@lenke/core';
export type { Backend, GraphHandle, MergeReport } from './backend.js';
export {
  applySchemaOp,
  attachGraph,
  createEmptyGraph,
  decodeArrow,
  escapeGremlin,
  graphFromFormat,
  graphFromNdjson,
  composeGremlin,
  type GremlinLiteral,
  gremlin,
  type QueryParams,
  type RustGraph,
  type Row,
  type SchemaOp,
} from './graph.js';
export { createStore, inferDeps, type Store, type LiveQuery } from './store.js';
// The napi adapter rebuilds coded LenkeErrors from the wire-code tag its N-API
// exceptions carry, matching the errors the bun:ffi / wasm backends throw.
export { errorFromNapi } from './marshal.js';

/** True when running under Bun, where the native FFI backend is available. */
export const isBun = typeof (globalThis as { Bun?: unknown }).Bun !== 'undefined';
