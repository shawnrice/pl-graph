// Adapter: drive the shared `Backend` contract from `@lenke/native` over the napi
// addon. The addon exposes the engine's THIN abi (`Graph` = the 12 `lnk_*`
// primitives: open / clone / config / stat / query / tx / schemaApply /
// schemaDump / encode / command / commandAsync + the free `abiVersion`), and
// `buildEngineBackend` assembles the full high-level Backend (codecs, schema DDL,
// algorithms, prepared statements, CDC) over it — the exact same builder the
// bun:ffi and wasm backends use, so this addon inherits their behavior for free.
//
// The Backend contract is handle-based (an opaque numeric token) while the addon
// hands back live `Graph` objects, so we keep a small id→object registry and let
// napi's GC reclaim a graph (`Drop` → `lnk_close`) once its handle is dropped. The
// per-call Map lookup is nanoseconds against query compute.
import { buildEngineBackend, encodeInput, errorFromNapi } from '@lenke/native';

import { Graph, abiVersion } from './index.js';

// The facade passes Uint8Array; the addon wants a Node Buffer. Wrap (no copy)
// rather than reallocate when we already hold a Uint8Array view.
const asBuffer = (u8) =>
  Buffer.isBuffer(u8) ? u8 : Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength);

const encoder = new TextEncoder();

// The addon throws N-API exceptions tagged with the stable wire code
// (`… [E_SYNTAX]`); rebuild them as coded LenkeErrors so a consumer matches
// `hasErrorCode(e, ErrorCode.Syntax)` identically to the bun:ffi / wasm backends.
const toLenkeError = (e) =>
  errorFromNapi(e && typeof e.message === 'string' ? e.message : undefined);
const coded = (fn) => {
  try {
    return fn();
  } catch (e) {
    throw toLenkeError(e);
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

  // The thin engine abi, one method per `lnk_*` primitive, mapped onto the napi
  // `Graph`. Every fallible call is `coded()` so a native fault surfaces as the
  // same LenkeError the FFI/wasm abis throw — `buildEngineBackend` relies on that.
  /** @type {import('@lenke/native').EngineAbi} */
  const abi = {
    abiVersion: abiVersion(),
    open: (bytes, format) =>
      coded(() => put(Graph.open(bytes ? asBuffer(bytes) : undefined, format))),
    // Dropping the reference frees the underlying store when napi GCs the object
    // (`Graph`'s `Drop` → `lnk_close`); there is no explicit native free to call.
    close: (handle) => {
      registry.delete(handle);
    },
    clone: (handle) => coded(() => put(get(handle).cloneGraph())),
    config: (handle, id, value) => get(handle).config(id, value),
    stat: (handle, which) => get(handle).stat(which),
    query: (handle, lang, query, params, format) =>
      coded(() => get(handle).query(lang, query, params ?? undefined, format)),
    tx: (handle, action) => coded(() => get(handle).tx(action)),
    schemaApply: (handle, json) => coded(() => get(handle).schemaApply(json)),
    schemaDump: (handle) => coded(() => get(handle).schemaDump()),
    encode: (handle, format) => coded(() => get(handle).encode(format)),
    command: (handle, name, input) =>
      coded(() => {
        const bytes = encodeInput(input);

        return get(handle).command(name, bytes ? asBuffer(bytes) : undefined);
      }),
  };

  const backend = buildEngineBackend(abi);

  // `buildEngineBackend` has no off-thread algorithm path (the FFI/wasm hosts are
  // single-threaded). Node has a libuv threadpool, so add the async twin over the
  // addon's `commandAsync`, wrapping the same `{name, config}` payload `algo` uses.
  backend.algoAsync = async (handle, name, config) => {
    try {
      const payload = encoder.encode(
        JSON.stringify({ name, config: config ? JSON.parse(config) : {} }),
      );

      return await get(handle).commandAsync('algo', asBuffer(payload));
    } catch (e) {
      throw toLenkeError(e);
    }
  };

  return backend;
}
