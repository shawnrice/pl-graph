/**
 * `@lenke/native` — the Rust columnar engine, callable from JS/TS.
 *
 * One C ABI, two backends over the lenke-engine C ABI behind a single
 * {@link Backend} contract. The backends live behind subpath exports so that
 * loading this package in a browser never pulls in the Bun-only `bun:ffi`
 * builtin:
 *   - `@lenke/native/ffi-engine` → `createFfiEngineBackend` loads the native
 *     dynamic library over `bun:ffi` (server / CLI), and
 *   - `@lenke/native/wasm-engine` → `createWasmEngineBackend` instantiates the
 *     wasm artifact (browser).
 *
 * This entry is environment-neutral: shared types plus the engine-neutral
 * {@link RustGraph} facade — GQL via `query`, Gremlin via `gremlin`, both with
 * the same tagged-template ergonomics as `@lenke/gql`.
 *
 * @example bun / server
 * ```ts
 * import { createFfiEngineBackend } from '@lenke/native/ffi-engine';
 * import { graphFromNdjson } from '@lenke/native';
 * const backend = createFfiEngineBackend('/path/to/liblenke_engine.so');
 * const g = graphFromNdjson(backend, await Bun.file('graph.ndjson').bytes());
 * g.query`MATCH (a:Person) RETURN a.name`;
 * ```
 *
 * @example browser
 * ```ts
 * import { createWasmEngineBackend } from '@lenke/native/wasm-engine';
 * import { graphFromNdjson } from '@lenke/native';
 * const backend = await createWasmEngineBackend(fetch('/lenke_engine.wasm'));
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
// The engine-backed builder + thin ABI shape, re-exported so an out-of-tree host
// (the `@lenke/node` napi adapter) can drive the same `Backend`-assembling logic
// over its own `lnk_*` transport instead of reimplementing it.
export { buildEngineBackend, encodeInput, type EngineAbi } from './backend-engine.js';
// The napi adapter rebuilds coded LenkeErrors from the wire-code tag its N-API
// exceptions carry, matching the errors the bun:ffi / wasm backends throw.
export { errorFromNapi } from './marshal.js';

/** True when running under Bun, where the native FFI backend is available. */
export const isBun = typeof (globalThis as { Bun?: unknown }).Bun !== 'undefined';
