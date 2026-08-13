// The wasm surface of the cross-engine comparison. Same shape as the FFI loader
// (`engine-compare.ts`) — build a paired graph, run queries on core (`lnk_*`) and engine
// (`lnk_e_*`), compare — but through a WebAssembly instance instead of bun:ffi. This is
// what makes the comparison "across all surfaces": the SAME artifact's exports, now in a
// 32-bit linear-memory world (usize → i32, inputs copied into the module's memory via
// `lnk_alloc`, `memory.buffer` re-read after any call that may have grown the heap).
//
// Requires a wasm built WITH the feature:
//   cargo build --release --target wasm32-unknown-unknown --no-default-features \
//     --features gql,gremlin,ndjson,codecs,arrow,engine-compare \
//     --manifest-path ../../crates/lenke-core/Cargo.toml
import { readFileSync } from 'node:fs';

import { type CompareHandle, type Engine, type Loaded, toEngineDialect } from './engine-compare.js';

export const WASM_PATH = new URL(
  '../../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

type WasmExports = {
  memory: WebAssembly.Memory;
  lnk_alloc: (len: number) => number;
  lnk_dealloc: (ptr: number, len: number) => void;
  lnk_free_buf: (ptr: number, len: number) => void;
  lnk_graph_from_ndjson: (ptr: number, len: number, parallel: number) => number;
  lnk_graph_free: (h: number) => void;
  lnk_graph_vertex_count: (h: number) => bigint;
  lnk_graph_edge_count: (h: number) => bigint;
  lnk_gremlin_json: (h: number, q: number, ql: number, o: number) => number;
  lnk_query_rows: (h: number, q: number, ql: number, p: number, pl: number, o: number) => number;
  lnk_e_graph_from_ndjson: (ptr: number, len: number) => number;
  lnk_e_graph_free: (h: number) => void;
  lnk_e_graph_vertex_count: (h: number) => bigint;
  lnk_e_graph_edge_count: (h: number) => bigint;
  lnk_e_gremlin_json: (h: number, q: number, ql: number, o: number) => number;
  lnk_e_query_rows: (h: number, q: number, ql: number, o: number) => number;
};

const enc = new TextEncoder();
const dec = new TextDecoder();

export const loadCompareWasm = async (wasmPath: string = WASM_PATH): Promise<Loaded> => {
  const bytes = readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const ex = instance.exports as unknown as WasmExports;

  // memory.buffer is replaced when the heap grows, so views must be fetched fresh.
  const u8 = (): Uint8Array => new Uint8Array(ex.memory.buffer);

  const writeBytes = (b: Uint8Array): number => {
    const p = ex.lnk_alloc(b.byteLength);

    u8().set(b, p);

    return p;
  };

  // Run `call` with a freshly-allocated 4-byte out-len cell, copy the result bytes out,
  // free both, and decode — or null if the call returned a null pointer.
  const withResult = (call: (outLenPtr: number) => number): string | null => {
    const outLenPtr = ex.lnk_alloc(4);

    try {
      const res = call(outLenPtr);

      if (!res) {
        return null;
      }

      const len = new DataView(ex.memory.buffer).getUint32(outLenPtr, true);

      try {
        return dec.decode(u8().slice(res, res + len));
      } finally {
        ex.lnk_free_buf(res, len);
      }
    } finally {
      ex.lnk_dealloc(outLenPtr, 4);
    }
  };

  // Stage a query string, run it, free the staging buffer.
  const withQuery = (
    q: string,
    call: (qPtr: number, qLen: number, outLenPtr: number) => number,
  ): string | null => {
    const qb = enc.encode(q);
    const qPtr = writeBytes(qb);

    try {
      return withResult((o) => call(qPtr, qb.byteLength, o));
    } finally {
      ex.lnk_dealloc(qPtr, qb.byteLength);
    }
  };

  const fromCoreNdjson = (coreNdjson: string): CompareHandle => {
    const cb = enc.encode(coreNdjson);
    const eb = enc.encode(toEngineDialect(coreNdjson));
    const cPtr = writeBytes(cb);
    const core = ex.lnk_graph_from_ndjson(cPtr, cb.byteLength, 0);

    ex.lnk_dealloc(cPtr, cb.byteLength);

    const ePtr = writeBytes(eb);
    const engine = ex.lnk_e_graph_from_ndjson(ePtr, eb.byteLength);

    ex.lnk_dealloc(ePtr, eb.byteLength);

    if (!core || !engine) {
      throw new Error('failed to build one of the graphs');
    }

    return {
      vertexCount: (e: Engine) =>
        Number(
          e === 'core' ? ex.lnk_graph_vertex_count(core) : ex.lnk_e_graph_vertex_count(engine),
        ),
      edgeCount: (e: Engine) =>
        Number(e === 'core' ? ex.lnk_graph_edge_count(core) : ex.lnk_e_graph_edge_count(engine)),
      gremlin: (e: Engine, q: string) =>
        e === 'core'
          ? withQuery(q, (qp, ql, o) => ex.lnk_gremlin_json(core, qp, ql, o))
          : withQuery(q, (qp, ql, o) => ex.lnk_e_gremlin_json(engine, qp, ql, o)),
      gql: (e: Engine, q: string) =>
        e === 'core'
          ? withQuery(q, (qp, ql, o) => ex.lnk_query_rows(core, qp, ql, 0, 0, o))
          : withQuery(q, (qp, ql, o) => ex.lnk_e_query_rows(engine, qp, ql, o)),
      free: () => {
        ex.lnk_graph_free(core);
        ex.lnk_e_graph_free(engine);
      },
    };
  };

  // A wasm instance has no explicit teardown — GC reclaims it once unreferenced.
  return { fromCoreNdjson, close: () => {} };
};
