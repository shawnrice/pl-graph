// Cross-engine comparison harness: load BOTH engines behind one artifact and drive
// them from the same graph + queries. `lnk_*` is the shipped row-based core; `lnk_e_*`
// is the standalone columnar engine (`lenke-engine`), exposed under the crate's
// `engine-compare` feature. This module is the shared loader — a correctness test and a
// perf bench both build on it.
//
// It deliberately bypasses the full `Backend` contract and talks to the raw FFI: the
// comparison is read-only (build a graph, run queries, compare JSON), so it needs only
// build / count / query / free from each side, and pairing them here keeps the two
// engines' surfaces literally side by side.
//
// Requires a `.so` built WITH the feature:
//   cargo build --release --features engine-compare --manifest-path ../../crates/lenke-core/Cargo.toml
import { dlopen, FFIType, type Pointer, ptr, toArrayBuffer } from 'bun:ffi';

const U = FFIType.u64_fast;
const P = FFIType.ptr;

export const LIB_PATH = new URL(
  '../../../crates/lenke-core/target/release/liblenke_core.so',
  import.meta.url,
).pathname;

const SYMBOLS = {
  // shipped core
  lnk_graph_from_ndjson: { args: [P, U, FFIType.u32], returns: P },
  lnk_graph_free: { args: [P], returns: FFIType.void },
  lnk_graph_vertex_count: { args: [P], returns: U },
  lnk_graph_edge_count: { args: [P], returns: U },
  lnk_gremlin_json: { args: [P, P, U, P], returns: P },
  lnk_query_rows: { args: [P, P, U, P, U, P], returns: P },
  lnk_free_buf: { args: [P, U], returns: FFIType.void },
  // standalone engine (engine-compare feature)
  lnk_e_graph_from_ndjson: { args: [P, U], returns: P },
  lnk_e_graph_free: { args: [P], returns: FFIType.void },
  lnk_e_graph_vertex_count: { args: [P], returns: U },
  lnk_e_graph_edge_count: { args: [P], returns: U },
  lnk_e_gremlin_json: { args: [P, P, U, P], returns: P },
  lnk_e_query_rows: { args: [P, P, U, P], returns: P },
} as const;

const enc = new TextEncoder();
const dec = new TextDecoder();
const bytesPtr = (b: Uint8Array): Pointer => ptr(b.byteLength === 0 ? new Uint8Array(1) : b);

/** core-dialect NDJSON (`{type,id,labels,properties}` / edges with `from`/`to`) →
 *  engine-dialect (`{id,labels,props}` / `{from,to,labels,props}`). Same logical
 *  graph, no discriminator: the engine dispatches on `from` (edge) vs `id` (node). */
export const toEngineDialect = (ndjson: string): string =>
  ndjson
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => {
      const { type: _t, properties, ...rest } = JSON.parse(l) as Record<string, unknown>;

      return JSON.stringify({ ...rest, props: properties ?? {} });
    })
    .join('\n');

export type Engine = 'core' | 'engine';

export type CompareHandle = {
  vertexCount(e: Engine): number;
  edgeCount(e: Engine): number;
  /** Run a Gremlin query; returns raw JSON text, or null on parse/exec failure. */
  gremlin(e: Engine, q: string): string | null;
  /** Run a GQL query; returns raw JSON text, or null on parse/exec failure. */
  gql(e: Engine, q: string): string | null;
  free(): void;
};

// A comparison surface — an FFI or wasm loader — that builds a paired graph and closes
// the underlying artifact. Both surfaces implement this so one harness drives either.
export type Loaded = {
  fromCoreNdjson(coreNdjson: string): CompareHandle;
  close(): void;
};

/** dlopen the artifact and return a factory that builds a paired graph from one
 *  core-dialect NDJSON string (translating to the engine dialect internally). */
export const loadCompare = (libPath: string = LIB_PATH): Loaded => {
  const lib = dlopen(libPath, SYMBOLS);
  const s = lib.symbols;

  const takeBuf = (call: (outLen: Pointer) => Pointer | null): string | null => {
    const outLen = new BigUint64Array(1);
    const res = call(ptr(outLen));

    if (!res) {
      return null;
    }

    const len = Number(outLen[0]);

    try {
      return dec.decode(new Uint8Array(toArrayBuffer(res, 0, len)).slice());
    } finally {
      s.lnk_free_buf(res, len);
    }
  };

  const fromCoreNdjson = (coreNdjson: string): CompareHandle => {
    const cb = enc.encode(coreNdjson);
    const eb = enc.encode(toEngineDialect(coreNdjson));
    const core = s.lnk_graph_from_ndjson(bytesPtr(cb), cb.byteLength, 0);
    const engine = s.lnk_e_graph_from_ndjson(bytesPtr(eb), eb.byteLength);

    if (!core || !engine) {
      throw new Error('failed to build one of the graphs');
    }

    return {
      vertexCount: (e) =>
        Number(e === 'core' ? s.lnk_graph_vertex_count(core) : s.lnk_e_graph_vertex_count(engine)),
      edgeCount: (e) =>
        Number(e === 'core' ? s.lnk_graph_edge_count(core) : s.lnk_e_graph_edge_count(engine)),
      gremlin: (e, q) => {
        const qb = enc.encode(q);

        return e === 'core'
          ? takeBuf((o) => s.lnk_gremlin_json(core, bytesPtr(qb), qb.byteLength, o))
          : takeBuf((o) => s.lnk_e_gremlin_json(engine, bytesPtr(qb), qb.byteLength, o));
      },
      gql: (e, q) => {
        const qb = enc.encode(q);

        return e === 'core'
          ? takeBuf((o) => s.lnk_query_rows(core, bytesPtr(qb), qb.byteLength, null, 0, o))
          : takeBuf((o) => s.lnk_e_query_rows(engine, bytesPtr(qb), qb.byteLength, o));
      },
      free: () => {
        s.lnk_graph_free(core);
        s.lnk_e_graph_free(engine);
      },
    };
  };

  return { fromCoreNdjson, close: () => lib.close() };
};

/** Normalize result JSON to a canonical value form for comparison — a JSON round-trip
 *  so textual formatting (number spelling, spacing) never counts as a divergence; only
 *  the decoded value structure does. */
export const norm = (j: string | null): string | null =>
  j === null ? null : JSON.stringify(JSON.parse(j));
