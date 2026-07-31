// Adapter: expose the napi `Graph` class as the shared `Backend` contract from
// `@lenke/native`. The contract is handle-based (an opaque numeric token), while
// the addon hands back live `Graph` objects — so we keep a small id→object
// registry and let napi's GC reclaim a graph once its handle is dropped. The
// per-call Map lookup is nanoseconds against query compute; the doc's own
// measurements put the compute-to-transfer ratio around 2000:1.
import { errorFromNapi } from '@lenke/native';

import { Graph, abiVersion, prepare } from './index.js';

// The facade passes Uint8Array; the addon wants a Node Buffer. Wrap (no copy)
// rather than reallocate when we already hold a Uint8Array view.
const asBuffer = (u8) =>
  Buffer.isBuffer(u8) ? u8 : Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength);

// Decodes the JSON MergeReport the addon returns from mergeNdjson.
const mergeDecoder = new TextDecoder();

// The addon throws N-API exceptions tagged with the stable wire code
// (`… [E_SYNTAX]`); rebuild them as coded LenkeErrors so a consumer matches
// `hasErrorCode(e, ErrorCode.Syntax)` identically to the bun:ffi / wasm
// backends. (The getters — counts / version / epoch — are infallible in the
// addon, so only the fallible ops are wrapped.)
const coded = (fn) => {
  try {
    return fn();
  } catch (e) {
    throw errorFromNapi(e && typeof e.message === 'string' ? e.message : undefined);
  }
};

// Async twin of `coded`: rebuild an addon rejection (or a synchronous throw before
// the promise is returned) as a coded LenkeError.
const codedAsync = async (fn) => {
  try {
    return await fn();
  } catch (e) {
    throw errorFromNapi(e && typeof e.message === 'string' ? e.message : undefined);
  }
};

/** @returns {import('@lenke/native').Backend} */
export function createNodeBackend() {
  /** @type {Map<number, InstanceType<typeof Graph>>} */
  const registry = new Map();
  let nextHandle = 1;

  const put = (graph) => {
    const handle = nextHandle++;
    registry.set(handle, graph);

    return handle;
  };
  const get = (handle) => {
    const graph = registry.get(handle);

    if (graph === undefined) {
      throw new Error(`lenke: invalid graph handle ${handle}`);
    }

    return graph;
  };

  // A parallel registry for prepared statements (their own opaque handle,
  // independent of any graph — matching the C ABI's `*mut Prepared`).
  /** @type {Map<number, InstanceType<typeof import('./index.js').PreparedQuery>>} */
  const prepared = new Map();
  let nextPrepared = 1;
  const putPrepared = (pq) => {
    const handle = nextPrepared++;
    prepared.set(handle, pq);

    return handle;
  };
  const getPrepared = (handle) => {
    const pq = prepared.get(handle);

    if (pq === undefined) {
      throw new Error(`lenke: invalid prepared handle ${handle}`);
    }

    return pq;
  };

  return {
    abiVersion: abiVersion(),

    graphFromNdjson: (bytes, parallel) =>
      coded(() => put(Graph.fromNdjson(asBuffer(bytes), parallel))),
    mergeNdjson: (handle, bytes) =>
      coded(() => JSON.parse(mergeDecoder.decode(get(handle).mergeNdjson(asBuffer(bytes))))),
    // Drop the reference; the underlying lenke-core graph is freed when napi GCs
    // the object. No explicit native free to call.
    graphClone: (handle) => coded(() => put(get(handle).cloneGraph())),
    graphFree: (handle) => {
      registry.delete(handle);
    },

    vertexCount: (handle) => get(handle).vertexCount,
    edgeCount: (handle) => get(handle).edgeCount,
    version: (handle) => get(handle).version(),
    epoch: (handle, name) => get(handle).epoch(name),
    // napi-rs camelCases the Rust `create_vertex_index` / `create_edge_index`.
    createIndex: (handle, on, kind, keys) => get(handle).createIndex(on, kind, keys),
    setConfig: (handle, id, value) => get(handle).setConfig(id, value),
    dropVertexIndex: (handle, key) => get(handle).dropVertexIndex(key),
    dropEdgeIndex: (handle, key) => get(handle).dropEdgeIndex(key),
    vertexIndexes: (handle) => get(handle).vertexIndexes(),
    lastWriteScope: (handle, key) => get(handle).lastWriteScope(key),
    edgeIndexes: (handle) => get(handle).edgeIndexes(),
    // JSON string over the addon boundary (like the C ABI's lnk_dump_schema), parsed
    // to the SchemaOp[] the Backend contract returns.
    dumpSchema: (handle) => JSON.parse(get(handle).dumpSchema()),

    // `prepare` is a module-level addon function (a Prepared needs no graph);
    // execute binds it to a graph at call time.
    prepare: (text, maxOperatorChain) => coded(() => putPrepared(prepare(text, maxOperatorChain))),
    preparedFree: (handle) => {
      prepared.delete(handle);
    },
    preparedQueryRows: (prep, graph, params) =>
      coded(() => getPrepared(prep).query(get(graph), params)),
    preparedQueryArrow: (prep, graph, params) =>
      coded(() => getPrepared(prep).queryArrow(get(graph), params)),

    // `params` arrives pre-serialized (a flat JSON object of $name bindings)
    // per the Backend contract; the addon decodes it crate-side.
    queryRows: (handle, query, params) => coded(() => get(handle).query(query, params)),
    queryArrow: (handle, query, params) => coded(() => get(handle).queryArrow(query, params)),
    queryArrowIpc: (handle, query, file, params) =>
      coded(() => get(handle).queryArrowIpc(query, params, file)),
    gremlinJson: (handle, query) => coded(() => get(handle).gremlin(query)),
    algo: (handle, name, config) => coded(() => get(handle).algo(name, config)),
    // Off-thread on a libuv worker → Promise<Buffer>; the event loop stays free.
    algoAsync: (handle, name, config) => codedAsync(() => get(handle).algoAsync(name, config)),

    encodeNdjson: (handle) => get(handle).encodeNdjson(),
    serialize: (handle, format) => coded(() => get(handle).serialize(format)),
    deserialize: (bytes, format) => coded(() => put(Graph.deserialize(asBuffer(bytes), format))),
  };
}
