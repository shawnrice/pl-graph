/**
 * The bun:ffi backend over the STANDALONE engine's 16-symbol C ABI
 * (`liblenke_engine.{so,dylib}`, built with `bun run engine:build:rust`). Mirrors
 * `backend-ffi.ts`'s marshalling, but loads the lean `lnk_*` surface and hands it
 * to {@link buildEngineBackend}, which maps the {@link Backend} contract onto it.
 */
import { dlopen, FFIType, type Pointer, ptr, toArrayBuffer } from 'bun:ffi';

import { ErrorCode, LenkeError } from '@lenke/errors';

import { assertAbi } from './abi.js';
import { buildEngineBackend, encodeInput, type EngineAbi } from './backend-engine.js';
import type { GraphHandle } from './backend.js';
import type { Backend } from './backend.js';
import { asByteLength, type ErrorReport, parseErrorReport } from './marshal.js';

// usize / u64 on the native targets (arm64 / x86_64).
const U = FFIType.u64_fast;

const SYMBOLS = {
  lnk_abi_version: { args: [], returns: FFIType.u32 },
  lnk_free: { args: [FFIType.ptr, U], returns: FFIType.void },
  lnk_last_error_json: { args: [FFIType.ptr], returns: FFIType.ptr },
  lnk_open: { args: [FFIType.ptr, U, FFIType.u8, FFIType.u32], returns: FFIType.ptr },
  lnk_close: { args: [FFIType.ptr], returns: FFIType.void },
  lnk_clone: { args: [FFIType.ptr], returns: FFIType.ptr },
  lnk_config: { args: [FFIType.ptr, FFIType.u32, U], returns: FFIType.u32 },
  lnk_stat: { args: [FFIType.ptr, FFIType.u32], returns: U },
  lnk_query: {
    args: [FFIType.ptr, FFIType.u8, FFIType.ptr, U, FFIType.ptr, U, FFIType.u8, FFIType.ptr],
    returns: FFIType.ptr,
  },
  lnk_tx: { args: [FFIType.ptr, FFIType.u8], returns: FFIType.i32 },
  lnk_schema_apply: { args: [FFIType.ptr, FFIType.ptr, U], returns: FFIType.i32 },
  lnk_schema_dump: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },
  lnk_encode: { args: [FFIType.ptr, FFIType.u8, FFIType.ptr], returns: FFIType.ptr },
  lnk_command: {
    args: [FFIType.ptr, FFIType.ptr, U, FFIType.ptr, U, FFIType.ptr],
    returns: FFIType.ptr,
  },
} as const;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// bun:ffi `ptr()` rejects a zero-length view; hand it a 1-byte scratch for an empty
// payload while still passing the real (0) length (see backend-ffi.ts).
const EMPTY_SCRATCH = new Uint8Array(1);
const bytesPtr = (b: Uint8Array): Pointer => ptr(b.byteLength === 0 ? EMPTY_SCRATCH : b);
const asPtr = (h: GraphHandle): Pointer => h as unknown as Pointer;
const asHandle = (p: Pointer | null): GraphHandle => p as unknown as GraphHandle;

/** Load `liblenke_engine` over bun:ffi. Pass the absolute path to the built cdylib. */
export const createFfiEngineBackend = (libPath: string): Backend => {
  const lib = dlopen(libPath, SYMBOLS);
  const { symbols } = lib;

  const abiVersion = symbols.lnk_abi_version();
  assertAbi(abiVersion);

  const readLastError = (): ErrorReport | null => {
    const outLen = new BigUint64Array(1);
    const errPtr = symbols.lnk_last_error_json(ptr(outLen));

    if (!errPtr) {
      return null;
    }

    const len = asByteLength(outLen[0], 'last-error');
    let json: string;

    try {
      json = decoder.decode(new Uint8Array(toArrayBuffer(errPtr, 0, len)).slice());
    } finally {
      symbols.lnk_free(errPtr, len);
    }

    return parseErrorReport(json);
  };

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

  // A buffer-returning call: read the crate-owned (ptr, out_len) into a JS copy,
  // then hand the crate buffer back to `lnk_free`.
  const takeResult = (call: (outLen: Pointer) => Pointer | null, op: string): Uint8Array => {
    const outLen = new BigUint64Array(1);
    const resPtr = call(ptr(outLen));

    if (!resPtr) {
      return fail(op, ErrorCode.Ffi);
    }

    const len = asByteLength(outLen[0], op);

    try {
      return new Uint8Array(toArrayBuffer(resPtr, 0, len)).slice();
    } finally {
      symbols.lnk_free(resPtr, len);
    }
  };

  const abi: EngineAbi = {
    abiVersion,
    open: (bytes, format, threads = 1) => {
      const h = symbols.lnk_open(
        bytes ? bytesPtr(bytes) : null,
        bytes ? bytes.byteLength : 0,
        format,
        threads,
      );

      if (!h) {
        return fail('open', ErrorCode.InvalidJson);
      }

      return asHandle(h);
    },
    close: (handle) => symbols.lnk_close(asPtr(handle)),
    clone: (handle) => {
      const c = symbols.lnk_clone(asPtr(handle));

      if (!c) {
        return fail('clone', ErrorCode.InvalidGraphOp);
      }

      return asHandle(c);
    },
    config: (handle, id, value) => symbols.lnk_config(asPtr(handle), id, value),
    stat: (handle, which) => Number(symbols.lnk_stat(asPtr(handle), which)),
    query: (handle, lang, query, params, format) => {
      const q = encoder.encode(query);
      const p = params === null ? null : encoder.encode(params);

      return takeResult(
        (outLen) =>
          symbols.lnk_query(
            asPtr(handle),
            lang,
            bytesPtr(q),
            q.byteLength,
            p ? ptr(p) : null,
            p ? p.byteLength : 0,
            format,
            outLen,
          ),
        'query',
      );
    },
    tx: (handle, action) => {
      if (symbols.lnk_tx(asPtr(handle), action) !== 0) {
        fail('tx', ErrorCode.Ffi);
      }
    },
    schemaApply: (handle, json) => {
      const j = encoder.encode(json);

      if (symbols.lnk_schema_apply(asPtr(handle), bytesPtr(j), j.byteLength) !== 0) {
        fail('schemaApply', ErrorCode.Ffi);
      }
    },
    schemaDump: (handle) =>
      takeResult((outLen) => symbols.lnk_schema_dump(asPtr(handle), outLen), 'schemaDump'),
    encode: (handle, format) =>
      takeResult((outLen) => symbols.lnk_encode(asPtr(handle), format, outLen), 'encode'),
    command: (handle, name, input) => {
      const n = encoder.encode(name);
      const inBytes = encodeInput(input);

      return takeResult(
        (outLen) =>
          symbols.lnk_command(
            asPtr(handle),
            bytesPtr(n),
            n.byteLength,
            inBytes ? bytesPtr(inBytes) : null,
            inBytes ? inBytes.byteLength : 0,
            outLen,
          ),
        'command',
      );
    },
  };

  const backend = buildEngineBackend(abi);
  // Retain the bun:ffi Library on the backend: bun closes (dlclose's) a Library when
  // it is garbage-collected, which invalidates every `symbols` pointer. Destructuring
  // only `symbols` left the Library unreferenced, so under GC pressure the native lib
  // could be unloaded WHILE graphs were still calling into it — surfacing as silent
  // wrong results (e.g. a constraint check that no longer ran), not a clean fault.
  (backend as { __lib?: unknown }).__lib = lib;

  return backend;
};
