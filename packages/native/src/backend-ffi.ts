import { dlopen, FFIType, type Pointer, ptr, toArrayBuffer } from 'bun:ffi';

import { ErrorCode, LenkeError } from '@lenke/errors';

import { assertAbi } from './abi.js';
import type { Backend, GraphHandle, MergeReport } from './backend.js';
import type { SchemaOp } from './graph.js';
import { asByteLength, type ErrorReport, parseErrorReport } from './marshal.js';

// usize / pointer are 64-bit on the native targets we load (arm64 / x86_64);
// the wasm backend uses 32-bit equivalents and lives in its own module.
const U = FFIType.u64_fast;

const SYMBOLS = {
  lnk_abi_version: { args: [], returns: FFIType.u32 },
  lnk_graph_from_ndjson: { args: [FFIType.ptr, U, FFIType.u32], returns: FFIType.ptr },
  lnk_merge_ndjson: { args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr], returns: FFIType.ptr },
  lnk_graph_clone: { args: [FFIType.ptr], returns: FFIType.ptr },
  lnk_graph_free: { args: [FFIType.ptr], returns: FFIType.void },
  lnk_graph_vertex_count: { args: [FFIType.ptr], returns: U },
  lnk_graph_edge_count: { args: [FFIType.ptr], returns: U },
  lnk_graph_version: { args: [FFIType.ptr], returns: U },
  lnk_graph_epoch: { args: [FFIType.ptr, FFIType.ptr, U], returns: U },
  lnk_graph_set_max_operator_chain: { args: [FFIType.ptr, U], returns: FFIType.void },
  // (g, element:u8, kind:u8, k0_ptr, k0_len, k1_ptr, k1_len) -> i32
  lnk_create_index: {
    args: [FFIType.ptr, FFIType.u8, FFIType.u8, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_unique_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_required_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_type_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_edge_unique_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_edge_required_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_edge_type_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_cardinality_constraint: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.u8, FFIType.u32, FFIType.i64],
    returns: FFIType.i32,
  },
  lnk_create_validator: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_create_invariant: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U],
    returns: FFIType.i32,
  },
  lnk_drop_vertex_index: { args: [FFIType.ptr, FFIType.ptr, U], returns: FFIType.i32 },
  lnk_drop_edge_index: { args: [FFIType.ptr, FFIType.ptr, U], returns: FFIType.i32 },
  lnk_begin_tx: { args: [FFIType.ptr], returns: FFIType.i32 },
  lnk_commit_tx: { args: [FFIType.ptr], returns: FFIType.i32 },
  lnk_rollback_tx: { args: [FFIType.ptr], returns: FFIType.i32 },
  lnk_vertex_indexes: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },
  lnk_last_write_scope: { args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr], returns: FFIType.ptr },
  lnk_edge_indexes: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },
  lnk_dump_schema: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },
  lnk_prepare: { args: [FFIType.ptr, U, U], returns: FFIType.ptr },
  lnk_prepared_free: { args: [FFIType.ptr], returns: FFIType.void },
  lnk_prepared_query_rows: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_prepared_query_arrow: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_query_rows: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_query_arrow: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_query_arrow_ipc: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.u32, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_gremlin_json: { args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr], returns: FFIType.ptr },
  lnk_algo: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_encode_ndjson: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },
  lnk_serialize: { args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr], returns: FFIType.ptr },
  lnk_deserialize: { args: [FFIType.ptr, U, FFIType.ptr, U], returns: FFIType.ptr },
  lnk_free_buf: { args: [FFIType.ptr, U], returns: FFIType.void },
  lnk_free_arrow: { args: [FFIType.ptr, U], returns: FFIType.void },
  lnk_last_error_json: { args: [FFIType.ptr], returns: FFIType.ptr },
} as const;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// bun:ffi `ptr()` rejects a zero-length view, so an empty payload — an empty
// query string, an empty file to deserialize, an empty Gremlin traversal — would
// crash with a raw `TypeError` *before* reaching Rust (where it would otherwise
// produce a normal coded error / empty graph, as the pure-TS engine does). Hand
// `ptr()` a 1-byte scratch for the empty case while still passing the real (0)
// byte length, so the crate reads an empty slice instead of the binding throwing.
const EMPTY_SCRATCH = new Uint8Array(1);
const bytesPtr = (b: Uint8Array): Pointer => ptr(b.byteLength === 0 ? EMPTY_SCRATCH : b);

// A graph handle is an opaque token in the public contract (a JS `number` so the
// wasm offset and the native pointer share one type). At the bun boundary it is
// a branded `Pointer`; these two helpers cross that seam in one place.
const asPtr = (h: GraphHandle): Pointer => h as unknown as Pointer;
const asHandle = (p: Pointer | null): GraphHandle => p as unknown as GraphHandle;

/**
 * Load the native dynamic library over `bun:ffi`. Pass the absolute path to the
 * built `liblenke_core.{dylib,so,dll}` (see `build:rust`).
 */
export const createFfiBackend = (libPath: string): Backend => {
  const { symbols } = dlopen(libPath, SYMBOLS);

  const abiVersion = symbols.lnk_abi_version();
  assertAbi(abiVersion);

  // Read (and clear) the crate's out-of-band last-error after a null/-1 return.
  // The error rides this side channel, never the data return — so the binary
  // Arrow carrier stays a pure column blob. Returns null if nothing is pending.
  const readLastError = (): ErrorReport | null => {
    const outLen = new BigUint64Array(1);
    const errPtr = symbols.lnk_last_error_json(ptr(outLen));

    if (!errPtr) {
      return null;
    }

    // Validate OUTSIDE the try: a throw here means the crate wrote an
    // unrepresentable `out_len`, so we don't know the buffer's true size and
    // `lnk_free_buf(errPtr, <wrong len>)` would be undefined behavior — leaking
    // once on corrupt ABI data is the safe choice. With a valid len, the
    // `finally` frees even if the copy step OOMs.
    const len = asByteLength(outLen[0], 'last-error');
    let json: string;

    try {
      json = decoder.decode(new Uint8Array(toArrayBuffer(errPtr, 0, len)).slice());
    } finally {
      symbols.lnk_free_buf(errPtr, len);
    }

    // A malformed report is itself an FFI fault; `parseErrorReport` returns null
    // and `fail` falls back to a generic code.
    return parseErrorReport(json);
  };

  // Turn a failure sentinel into a `LenkeError` carrying the shared code, so a
  // consumer matches `hasErrorCode(e, ErrorCode.Syntax)` identically whether the
  // error came from the TS engine or this native one. `fallback` is used only if
  // the crate left no report (e.g. an older lib, or a non-instrumented path).
  const fail = (op: string, fallback: ErrorCode): never => {
    const report = readLastError();

    if (report) {
      throw new LenkeError(`lenke: ${op}: ${report.message}`, {
        code: report.code,
        details: report.details ?? undefined,
      });
    }

    throw new LenkeError(`lenke: ${op} failed`, { code: fallback });
  };

  // A query/encode call returns a crate-owned buffer (pointer + out_len). Read
  // it into a JS-owned copy and hand the crate buffer straight back to its
  // matching free, so nothing leaks and the copy outlives the native memory.
  const takeBuf = (
    call: (outLen: Pointer) => Pointer | null,
    free: (ptr: Pointer, len: number | bigint) => void,
    op: string,
  ): Uint8Array => {
    const outLen = new BigUint64Array(1);
    const resPtr = call(ptr(outLen));

    if (!resPtr) {
      return fail(op, ErrorCode.Ffi);
    }

    // Validate before the try (see readLastError): an unrepresentable len can't
    // be freed safely, so it leaks by design; a valid len is freed in `finally`
    // even if the copy OOMs.
    const len = asByteLength(outLen[0], op);

    try {
      return new Uint8Array(toArrayBuffer(resPtr, 0, len)).slice();
    } finally {
      free(resPtr, len);
    }
  };

  return {
    abiVersion,

    graphFromNdjson: (bytes, parallel) => {
      const h = symbols.lnk_graph_from_ndjson(bytesPtr(bytes), bytes.byteLength, parallel ? 1 : 0);

      if (!h) {
        return fail('graphFromNdjson', ErrorCode.InvalidJson);
      }

      return asHandle(h);
    },
    mergeNdjson: (handle, bytes) =>
      JSON.parse(
        decoder.decode(
          takeBuf(
            (outLen) =>
              symbols.lnk_merge_ndjson(asPtr(handle), bytesPtr(bytes), bytes.byteLength, outLen),
            symbols.lnk_free_buf,
            'mergeNdjson',
          ),
        ),
      ) as MergeReport,
    graphClone: (handle) => {
      const h = symbols.lnk_graph_clone(asPtr(handle));

      if (!h) {
        return fail('graphClone', ErrorCode.InvalidGraphOp);
      }

      return asHandle(h);
    },
    graphFree: (handle) => symbols.lnk_graph_free(asPtr(handle)),
    vertexCount: (handle) => Number(symbols.lnk_graph_vertex_count(asPtr(handle))),
    edgeCount: (handle) => Number(symbols.lnk_graph_edge_count(asPtr(handle))),
    version: (handle) => Number(symbols.lnk_graph_version(asPtr(handle))),
    epoch: (handle, name) => {
      const n = encoder.encode(name);

      return Number(symbols.lnk_graph_epoch(asPtr(handle), ptr(n), n.byteLength));
    },
    setMaxOperatorChain: (handle, n) => {
      symbols.lnk_graph_set_max_operator_chain(asPtr(handle), n);
    },
    createIndex: (handle, on, kind, keys) => {
      const element = on === 'edge' ? 1 : 0;
      const k = kind === 'interval' ? 1 : 0;
      const k0 = encoder.encode(keys[0] ?? '');
      const hasK1 = keys.length > 1;
      const k1 = hasK1 ? encoder.encode(keys[1]) : null;

      symbols.lnk_create_index(
        asPtr(handle),
        element,
        k,
        ptr(k0),
        k0.byteLength,
        k1 === null ? null : ptr(k1),
        k1 === null ? 0 : k1.byteLength,
      );
    },
    createUniqueConstraint: (handle, label, key) => {
      const l = encoder.encode(label);
      const k = encoder.encode(key);
      const r = symbols.lnk_create_unique_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createUniqueConstraint(${label}, ${key}): existing data already violates the unique constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createUniqueConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createRequiredConstraint: (handle, label, key) => {
      const l = encoder.encode(label);
      const k = encoder.encode(key);
      const r = symbols.lnk_create_required_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createRequiredConstraint(${label}, ${key}): existing data already violates the required constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createRequiredConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createTypeConstraint: (handle, label, key, type) => {
      const l = encoder.encode(label);
      const k = encoder.encode(key);
      const t = encoder.encode(type);
      const r = symbols.lnk_create_type_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
        ptr(t),
        t.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createTypeConstraint(${label}, ${key}, ${type}): existing data already violates the type constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r === -3) {
        throw new LenkeError(`lenke: createTypeConstraint: unknown scalar type '${type}'`, {
          code: ErrorCode.InvalidValue,
        });
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createTypeConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createEdgeUniqueConstraint: (handle, edgeType, key) => {
      const l = encoder.encode(edgeType);
      const k = encoder.encode(key);
      const r = symbols.lnk_create_edge_unique_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createEdgeUniqueConstraint(${edgeType}, ${key}): existing data already violates the unique constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createEdgeUniqueConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createEdgeRequiredConstraint: (handle, edgeType, key) => {
      const l = encoder.encode(edgeType);
      const k = encoder.encode(key);
      const r = symbols.lnk_create_edge_required_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createEdgeRequiredConstraint(${edgeType}, ${key}): existing data already violates the required constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createEdgeRequiredConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createEdgeTypeConstraint: (handle, edgeType, key, type) => {
      const l = encoder.encode(edgeType);
      const k = encoder.encode(key);
      const t = encoder.encode(type);
      const r = symbols.lnk_create_edge_type_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(k),
        k.byteLength,
        ptr(t),
        t.byteLength,
      );

      if (r === -2) {
        throw new LenkeError(
          `lenke: createEdgeTypeConstraint(${edgeType}, ${key}, ${type}): existing data already violates the type constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r === -3) {
        throw new LenkeError(`lenke: createEdgeTypeConstraint: unknown scalar type '${type}'`, {
          code: ErrorCode.InvalidValue,
        });
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createEdgeTypeConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createCardinalityConstraint: (handle, label, edgeType, direction, min, max) => {
      const l = encoder.encode(label);
      const e = encoder.encode(edgeType);
      // direction: 0 = out, 1 = in; max: i64 with -1 = unbounded (null).
      const r = symbols.lnk_create_cardinality_constraint(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(e),
        e.byteLength,
        direction === 'out' ? 0 : 1,
        min,
        BigInt(max ?? -1),
      );

      if (r === -1) {
        throw new LenkeError(
          `lenke: createCardinalityConstraint(${label}, ${edgeType}, ${direction}): existing data already violates the cardinality constraint`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createCardinalityConstraint failed', { code: ErrorCode.Ffi });
      }
    },
    createValidator: (handle, label, varName, predicate) => {
      const l = encoder.encode(label);
      const v = encoder.encode(varName);
      const p = encoder.encode(predicate);
      const r = symbols.lnk_create_validator(
        asPtr(handle),
        ptr(l),
        l.byteLength,
        ptr(v),
        v.byteLength,
        ptr(p),
        p.byteLength,
      );

      if (r === -1) {
        throw new LenkeError(
          `lenke: createValidator(${label}, ${varName}): existing data already violates the predicate '${predicate}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r === -2) {
        throw new LenkeError(
          `lenke: createValidator(${label}, ${varName}): could not parse the predicate '${predicate}'`,
          { code: ErrorCode.Syntax },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createValidator failed', { code: ErrorCode.Ffi });
      }
    },
    createInvariant: (handle, name, query) => {
      const n = encoder.encode(name);
      const q = encoder.encode(query);
      const r = symbols.lnk_create_invariant(
        asPtr(handle),
        ptr(n),
        n.byteLength,
        ptr(q),
        q.byteLength,
      );

      if (r === -1) {
        throw new LenkeError(
          `lenke: createInvariant(${name}): existing data already violates the invariant '${query}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (r === -2) {
        throw new LenkeError(
          `lenke: createInvariant(${name}): could not parse the query '${query}'`,
          { code: ErrorCode.Syntax },
        );
      }

      if (r !== 0) {
        throw new LenkeError('lenke: createInvariant failed', { code: ErrorCode.Ffi });
      }
    },
    dropVertexIndex: (handle, key) => {
      const k = encoder.encode(key);

      // -2 == the key backs a unique constraint; dropping it is refused so
      // enforcement can't be silently downgraded (matches the TS core).
      if (symbols.lnk_drop_vertex_index(asPtr(handle), ptr(k), k.byteLength) === -2) {
        throw new LenkeError(
          `lenke: cannot drop the vertex index on '${key}' — it backs a unique constraint; drop the constraint first`,
          { code: ErrorCode.InvalidGraphOp },
        );
      }
    },
    dropEdgeIndex: (handle, key) => {
      const k = encoder.encode(key);

      if (symbols.lnk_drop_edge_index(asPtr(handle), ptr(k), k.byteLength) === -2) {
        throw new LenkeError(
          `lenke: cannot drop the edge index on '${key}' — it backs a unique constraint; drop the constraint first`,
          { code: ErrorCode.InvalidGraphOp },
        );
      }
    },
    beginTransaction: (handle) => {
      if (symbols.lnk_begin_tx(asPtr(handle)) !== 0) {
        throw new LenkeError('lenke: beginTransaction failed', { code: ErrorCode.Ffi });
      }
    },
    commitTransaction: (handle) => {
      const r = symbols.lnk_commit_tx(asPtr(handle));

      // -1 == a deferred constraint check failed at commit (the transaction has
      // already been rolled back), matching the TS core's ConstraintViolation.
      if (r === -1) {
        throw new LenkeError('lenke: transaction commit failed a constraint check', {
          code: ErrorCode.ConstraintViolation,
        });
      }

      // -2 == commit with no open transaction.
      if (r === -2) {
        throw new LenkeError('lenke: commit called with no open transaction', {
          code: ErrorCode.InvalidGraphOp,
        });
      }

      if (r !== 0) {
        throw new LenkeError('lenke: commitTransaction failed', { code: ErrorCode.Ffi });
      }
    },
    rollbackTransaction: (handle) => {
      if (symbols.lnk_rollback_tx(asPtr(handle)) !== 0) {
        throw new LenkeError('lenke: rollbackTransaction failed', { code: ErrorCode.Ffi });
      }
    },
    vertexIndexes: (handle) =>
      JSON.parse(
        decoder.decode(
          takeBuf(
            (outLen) => symbols.lnk_vertex_indexes(asPtr(handle), outLen),
            symbols.lnk_free_buf,
            'vertexIndexes',
          ),
        ),
      ) as string[],
    edgeIndexes: (handle) =>
      JSON.parse(
        decoder.decode(
          takeBuf(
            (outLen) => symbols.lnk_edge_indexes(asPtr(handle), outLen),
            symbols.lnk_free_buf,
            'edgeIndexes',
          ),
        ),
      ) as string[],
    dumpSchema: (handle) =>
      JSON.parse(
        decoder.decode(
          takeBuf(
            (outLen) => symbols.lnk_dump_schema(asPtr(handle), outLen),
            symbols.lnk_free_buf,
            'dumpSchema',
          ),
        ),
      ) as SchemaOp[],
    lastWriteScope: (handle, key) => {
      const k = encoder.encode(key);

      return JSON.parse(
        decoder.decode(
          takeBuf(
            (outLen) =>
              symbols.lnk_last_write_scope(asPtr(handle), bytesPtr(k), k.byteLength, outLen),
            symbols.lnk_free_buf,
            'lastWriteScope',
          ),
        ),
      ) as string[];
    },

    queryRows: (handle, query, params) => {
      const q = encoder.encode(query);
      // Params ride their own buffer; null pointer = "no params" on the C side.
      const p = params === undefined ? null : encoder.encode(params);

      return takeBuf(
        (outLen) =>
          symbols.lnk_query_rows(
            asPtr(handle),
            bytesPtr(q),
            q.byteLength,
            p ? bytesPtr(p) : null,
            p?.byteLength ?? 0,
            outLen,
          ),
        symbols.lnk_free_buf,
        'query',
      );
    },

    queryArrow: (handle, query, params) => {
      const q = encoder.encode(query);
      const p = params === undefined ? null : encoder.encode(params);

      return takeBuf(
        (outLen) =>
          symbols.lnk_query_arrow(
            asPtr(handle),
            bytesPtr(q),
            q.byteLength,
            p ? bytesPtr(p) : null,
            p?.byteLength ?? 0,
            outLen,
          ),
        symbols.lnk_free_arrow,
        'queryArrow',
      );
    },

    queryArrowIpc: (handle, query, file, params) => {
      const q = encoder.encode(query);
      const p = params === undefined ? null : encoder.encode(params);

      return takeBuf(
        (outLen) =>
          symbols.lnk_query_arrow_ipc(
            asPtr(handle),
            bytesPtr(q),
            q.byteLength,
            p ? bytesPtr(p) : null,
            p?.byteLength ?? 0,
            file ? 1 : 0,
            outLen,
          ),
        symbols.lnk_free_arrow,
        'queryArrowIpc',
      );
    },

    gremlinJson: (handle, query) => {
      const q = encoder.encode(query);

      return takeBuf(
        (outLen) => symbols.lnk_gremlin_json(asPtr(handle), bytesPtr(q), q.byteLength, outLen),
        symbols.lnk_free_buf,
        'gremlin',
      );
    },

    algo: (handle, name, config) => {
      const n = encoder.encode(name);
      // Config rides its own buffer; null pointer = "defaults" on the C side.
      const c = config === undefined ? null : encoder.encode(config);

      return takeBuf(
        (outLen) =>
          symbols.lnk_algo(
            asPtr(handle),
            bytesPtr(n),
            n.byteLength,
            c ? bytesPtr(c) : null,
            c?.byteLength ?? 0,
            outLen,
          ),
        symbols.lnk_free_buf,
        'algo',
      );
    },

    encodeNdjson: (handle) =>
      takeBuf(
        (outLen) => symbols.lnk_encode_ndjson(asPtr(handle), outLen),
        symbols.lnk_free_buf,
        'encodeNdjson',
      ),

    serialize: (handle, format) => {
      const f = encoder.encode(format);

      return takeBuf(
        (outLen) => symbols.lnk_serialize(asPtr(handle), ptr(f), f.byteLength, outLen),
        symbols.lnk_free_buf,
        `serialize(${format})`,
      );
    },

    prepare: (text, maxOperatorChain) => {
      const t = encoder.encode(text);
      const h = symbols.lnk_prepare(ptr(t), t.byteLength, maxOperatorChain ?? 10_000);

      if (!h) {
        return fail('prepare', ErrorCode.Syntax);
      }

      return asHandle(h);
    },
    preparedFree: (prepared) => symbols.lnk_prepared_free(asPtr(prepared)),
    preparedQueryRows: (prepared, graph, params) => {
      const p = params === undefined ? null : encoder.encode(params);

      return takeBuf(
        (outLen) =>
          symbols.lnk_prepared_query_rows(
            asPtr(prepared),
            asPtr(graph),
            p ? bytesPtr(p) : null,
            p?.byteLength ?? 0,
            outLen,
          ),
        symbols.lnk_free_buf,
        'preparedQuery',
      );
    },
    preparedQueryArrow: (prepared, graph, params) => {
      const p = params === undefined ? null : encoder.encode(params);

      return takeBuf(
        (outLen) =>
          symbols.lnk_prepared_query_arrow(
            asPtr(prepared),
            asPtr(graph),
            p ? bytesPtr(p) : null,
            p?.byteLength ?? 0,
            outLen,
          ),
        symbols.lnk_free_arrow,
        'preparedQueryArrow',
      );
    },

    deserialize: (input, format) => {
      const f = encoder.encode(format);
      const h = symbols.lnk_deserialize(bytesPtr(input), input.byteLength, ptr(f), f.byteLength);

      if (!h) {
        return fail(`deserialize(${format})`, ErrorCode.UnknownFormat);
      }

      return asHandle(h);
    },
  };
};
