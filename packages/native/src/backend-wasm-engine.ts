/**
 * The WebAssembly backend over the STANDALONE engine's 16-symbol C ABI
 * (`lenke_engine.wasm`, built with `bun run engine:build:wasm`). The wasm twin of
 * `backend-ffi-engine.ts`: everything is 32-bit linear-memory offsets, u64 returns
 * arrive as BigInt, and inputs are copied into the module's memory via `lnk_alloc`.
 * The marshalled ABI is handed to {@link buildEngineBackend}.
 */
import { ErrorCode, LenkeError } from '@lenke/errors';

import { assertAbi } from './abi.js';
import { buildEngineBackend, encodeInput, type EngineAbi } from './backend-engine.js';
import type { Backend } from './backend.js';
import { type ErrorReport, parseErrorReport } from './marshal.js';

export type WasmSource =
  | WebAssembly.Module
  | ArrayBuffer
  | ArrayBufferView
  | Response
  | Promise<Response>;

/* eslint-disable max-params -- the wasm `lnk_*` declarations mirror the C ABI arity 1:1; lnk_query legitimately takes 8 offset args and can't drop params */
type WasmExports = {
  memory: WebAssembly.Memory;
  lnk_abi_version: () => number;
  lnk_alloc: (len: number) => number;
  lnk_dealloc: (ptr: number, len: number) => void;
  lnk_free: (ptr: number, len: number) => void;
  lnk_last_error_json: (outLen: number) => number;
  lnk_open: (ptr: number, len: number, format: number) => number;
  lnk_close: (h: number) => void;
  lnk_clone: (h: number) => number;
  // `value` is u64 → an i64 wasm param, so it crosses the boundary as a BigInt.
  lnk_config: (h: number, id: number, value: bigint) => number;
  // returns u64 → BigInt.
  lnk_stat: (h: number, which: number) => bigint;
  lnk_query: (
    h: number,
    lang: number,
    qp: number,
    ql: number,
    pp: number,
    pl: number,
    format: number,
    outLen: number,
  ) => number;
  lnk_tx: (h: number, action: number) => number;
  lnk_schema_apply: (h: number, jp: number, jl: number) => number;
  lnk_schema_dump: (h: number, outLen: number) => number;
  lnk_encode: (h: number, format: number, outLen: number) => number;
  lnk_command: (
    h: number,
    np: number,
    nl: number,
    ip: number,
    il: number,
    outLen: number,
  ) => number;
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const instantiate = async (source: WasmSource): Promise<WebAssembly.Instance> => {
  if (source instanceof Response || source instanceof Promise) {
    const { instance } = await WebAssembly.instantiateStreaming(source, {});

    return instance;
  }

  const result = (await WebAssembly.instantiate(source as ArrayBuffer, {})) as unknown as
    | WebAssembly.Instance
    | { instance: WebAssembly.Instance };

  return 'instance' in result ? result.instance : result;
};

/** Instantiate the engine wasm backend from `lenke_engine.wasm`. */
export const createWasmEngineBackend = async (source: WasmSource): Promise<Backend> => {
  const instance = await instantiate(source);
  const ex = instance.exports as unknown as WasmExports;

  const abiVersion = ex.lnk_abi_version();
  assertAbi(abiVersion);

  // memory.buffer is replaced when the heap grows, so views must be fresh on every
  // access — never cache a Uint8Array across a call that can allocate.
  const u8 = (): Uint8Array => new Uint8Array(ex.memory.buffer);
  const dv = (): DataView => new DataView(ex.memory.buffer);

  const writeBytes = (bytes: Uint8Array): number => {
    const p = ex.lnk_alloc(bytes.byteLength);
    u8().set(bytes, p);

    return p;
  };

  const readBytes = (ptr: number, len: number, op: string): Uint8Array => {
    const mem = u8();

    if (ptr < 0 || len < 0 || ptr + len > mem.length) {
      throw new LenkeError(
        `lenke: ${op}: native result [${ptr}, ${ptr + len}) escapes wasm memory (${mem.length} bytes)`,
        { code: ErrorCode.Ffi, details: { ptr, len, memBytes: mem.length } },
      );
    }

    return mem.slice(ptr, ptr + len);
  };

  const readLastError = (): ErrorReport | null => {
    const outLenPtr = ex.lnk_alloc(4);

    try {
      const errPtr = ex.lnk_last_error_json(outLenPtr);

      if (!errPtr) {
        return null;
      }

      const len = dv().getUint32(outLenPtr, true);
      const json = decoder.decode(readBytes(errPtr, len, 'last-error'));
      ex.lnk_free(errPtr, len);

      return parseErrorReport(json);
    } finally {
      ex.lnk_dealloc(outLenPtr, 4);
    }
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

  // A result-returning call whose only marshalling is the 4-byte out_len slot
  // (the arg-free reads: schema dump, encode).
  const resultCall = (call: (outLenPtr: number) => number, op: string): Uint8Array => {
    const outLenPtr = ex.lnk_alloc(4);

    try {
      const resPtr = call(outLenPtr);

      if (!resPtr) {
        return fail(op, ErrorCode.Ffi);
      }

      const len = dv().getUint32(outLenPtr, true);
      const copy = readBytes(resPtr, len, op);
      ex.lnk_free(resPtr, len);

      return copy;
    } finally {
      ex.lnk_dealloc(outLenPtr, 4);
    }
  };

  const abi: EngineAbi = {
    abiVersion,
    open: (bytes, format) => {
      const p = bytes ? writeBytes(bytes) : 0;

      try {
        const h = ex.lnk_open(p, bytes ? bytes.byteLength : 0, format);

        if (!h) {
          return fail('open', ErrorCode.InvalidJson);
        }

        return h;
      } finally {
        if (p) {
          ex.lnk_dealloc(p, bytes!.byteLength);
        }
      }
    },
    close: (handle) => ex.lnk_close(handle),
    clone: (handle) => {
      const c = ex.lnk_clone(handle);

      if (!c) {
        return fail('clone', ErrorCode.InvalidGraphOp);
      }

      return c;
    },
    config: (handle, id, value) => ex.lnk_config(handle, id, BigInt(value)),
    stat: (handle, which) => Number(ex.lnk_stat(handle, which)),
    query: (handle, lang, query, params, format) => {
      const q = encoder.encode(query);
      const p = params === null ? null : encoder.encode(params);
      const qp = writeBytes(q);
      const pp = p ? writeBytes(p) : 0;
      const outLenPtr = ex.lnk_alloc(4);

      try {
        const resPtr = ex.lnk_query(
          handle,
          lang,
          qp,
          q.byteLength,
          pp,
          p ? p.byteLength : 0,
          format,
          outLenPtr,
        );

        if (!resPtr) {
          return fail('query', ErrorCode.Ffi);
        }

        const len = dv().getUint32(outLenPtr, true);
        const copy = readBytes(resPtr, len, 'query');
        ex.lnk_free(resPtr, len);

        return copy;
      } finally {
        ex.lnk_dealloc(qp, q.byteLength);

        if (pp) {
          ex.lnk_dealloc(pp, p!.byteLength);
        }

        ex.lnk_dealloc(outLenPtr, 4);
      }
    },
    tx: (handle, action) => {
      if (ex.lnk_tx(handle, action) !== 0) {
        fail('tx', ErrorCode.Ffi);
      }
    },
    schemaApply: (handle, json) => {
      const j = encoder.encode(json);
      const jp = writeBytes(j);

      try {
        if (ex.lnk_schema_apply(handle, jp, j.byteLength) !== 0) {
          fail('schemaApply', ErrorCode.Ffi);
        }
      } finally {
        ex.lnk_dealloc(jp, j.byteLength);
      }
    },
    schemaDump: (handle) =>
      resultCall((outLenPtr) => ex.lnk_schema_dump(handle, outLenPtr), 'schemaDump'),
    encode: (handle, format) =>
      resultCall((outLenPtr) => ex.lnk_encode(handle, format, outLenPtr), 'encode'),
    command: (handle, name, input) => {
      const n = encoder.encode(name);
      const inBytes = encodeInput(input);
      const np = writeBytes(n);
      const ip = inBytes ? writeBytes(inBytes) : 0;
      const outLenPtr = ex.lnk_alloc(4);

      try {
        const resPtr = ex.lnk_command(
          handle,
          np,
          n.byteLength,
          ip,
          inBytes ? inBytes.byteLength : 0,
          outLenPtr,
        );

        if (!resPtr) {
          return fail('command', ErrorCode.Ffi);
        }

        const len = dv().getUint32(outLenPtr, true);
        const copy = readBytes(resPtr, len, 'command');
        ex.lnk_free(resPtr, len);

        return copy;
      } finally {
        ex.lnk_dealloc(np, n.byteLength);

        if (ip) {
          ex.lnk_dealloc(ip, inBytes!.byteLength);
        }

        ex.lnk_dealloc(outLenPtr, 4);
      }
    },
  };

  return buildEngineBackend(abi);
};
