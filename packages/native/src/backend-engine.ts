/**
 * The engine backend translator: implements the {@link Backend} contract (shaped
 * around the former lenke-core's 44-symbol ABI) over the standalone engine's lean 16-symbol
 * ABI. Written ONCE here; the FFI and wasm engine backends each supply the
 * low-level {@link EngineAbi} (how to call `lnk_*` and marshal buffers) and wrap
 * it with {@link buildEngineBackend}.
 *
 * The 44 -> 16 fold lives here: a variant family collapses into an enum arg
 * (`query(lang, format)`, `tx(action)`, `stat(which)`, `open/encode(format)`) and
 * the exotic tiers ride `command(name, input)`. The schema-DDL family becomes one
 * `schemaApply(json)` op vocabulary.
 *
 * The full contract is wired: every serialization codec (ndjson, binary, and —
 * via the shared `lenke-codec` crate — pg-json, pg-text, graphson, csv); the whole
 * schema-DDL surface (indexes + drop, vertex/edge unique/required/type,
 * cardinality, validators, invariants); direct `algo`; and prepared statements
 * (JSON + Arrow). A failure throws a coded `LenkeError` read from the engine's
 * out-of-band last-error channel, so callers branch on `error.code` exactly as
 * with the former core backend.
 */
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Backend, GraphHandle, MergeReport } from './backend.js';
import type { SchemaOp } from './graph.js';

/**
 * The engine's 16-symbol ABI, buffer-marshalled and error-checked by the concrete
 * backend. Every op THROWS a coded `LenkeError` on failure (reading the engine's
 * out-of-band last-error), so the translator handles only success values.
 */
export type EngineAbi = {
  readonly abiVersion: number;
  /**
   * `lnk_open` — format 0 = NDJSON (null bytes = empty graph), 1 = binary,
   * 2 = pg-json, 3 = pg-text, 4 = graphson, 5 = csv. `threads` (default 1)
   * parallelizes the NDJSON PARSE only; every other format decodes serially.
   */
  open: (bytes: Uint8Array | null, format: number, threads?: number) => GraphHandle;
  /** `lnk_close`. */
  close: (handle: GraphHandle) => void;
  /** `lnk_clone`. */
  clone: (handle: GraphHandle) => GraphHandle;
  /** `lnk_config` — returns its raw `u32` (1 = applied). */
  config: (handle: GraphHandle, id: number, value: number) => number;
  /** `lnk_stat` — which 0 = vertex count, 1 = edge count, 2 = version. */
  stat: (handle: GraphHandle, which: number) => number;
  /** `lnk_query` — lang 0 = GQL, 1 = Gremlin; format 0 = JSON, 1 = Arrow, 2 = Arrow IPC. */
  query: (
    handle: GraphHandle,
    lang: number,
    query: string,
    params: string | null,
    format: number,
  ) => Uint8Array;
  /** `lnk_tx` — action 0 = begin, 1 = commit, 2 = rollback. Throws on a rejected commit. */
  tx: (handle: GraphHandle, action: number) => void;
  /** `lnk_schema_apply` — throws on `-1` (bad/unsupported op) and `-2` (data violates). */
  schemaApply: (handle: GraphHandle, json: string) => void;
  /** `lnk_schema_dump` — the `{op:…}` op-list as NDJSON bytes. */
  schemaDump: (handle: GraphHandle) => Uint8Array;
  /**
   * `lnk_encode` — format 0 = NDJSON, 1 = binary, 2 = pg-json, 3 = pg-text,
   * 4 = graphson, 5 = csv.
   */
  encode: (handle: GraphHandle, format: number) => Uint8Array;
  /** `lnk_command` — a named exotic-tier op (algo/CDC/epoch/merge/prepared). */
  command: (handle: GraphHandle, name: string, input: string | Uint8Array | null) => Uint8Array;
};

// Enum constants mirroring the engine ABI (see crates/lenke-engine/docs/abi.md).
const LANG_GQL = 0;
const LANG_GREMLIN = 1;
const FMT_JSON = 0;
const FMT_ARROW = 1;
const FMT_ARROW_IPC = 2; // Arrow IPC file / Feather layout
const FMT_ARROW_IPC_STREAM = 3; // Arrow IPC stream layout
// lnk_open / lnk_encode `format` bytes. 0/1 are the engine's native channels;
// 2..5 route through the shared lenke-codec bridge (byte-identical with the TS engine).
const FMT_NDJSON = 0;
const FMT_BINARY = 1;
/** The textual codecs handled by the shared crate, mapped to their format byte. */
const CODEC_FORMAT: Record<string, number> = {
  'pg-json': 2,
  'pg-text': 3,
  graphson: 4,
  csv: 5,
};
const TX_BEGIN = 0;
const TX_COMMIT = 1;
const TX_ROLLBACK = 2;
const STAT_VERTEX = 0;
const STAT_EDGE = 1;
const STAT_VERSION = 2;

const decoder = new TextDecoder();
const encoder = new TextEncoder();

const parseJson = <T>(bytes: Uint8Array): T => JSON.parse(decoder.decode(bytes)) as T;

/** Normalize a command input to bytes: a string is UTF-8 encoded, bytes pass
 * through, null stays null (no payload). Shared by the FFI and wasm backends. */
export const encodeInput = (input: string | Uint8Array | null): Uint8Array | null => {
  if (input === null) {
    return null;
  }

  if (typeof input === 'string') {
    return encoder.encode(input);
  }

  return input;
};

const unsupported = (feature: string): never => {
  throw new LenkeError(`lenke: ${feature} is not supported by the engine backend`, {
    code: ErrorCode.Unsupported,
  });
};

/** One decoded engine schema-dump line (`{op:…}` vocabulary). */
type EngineSchemaLine = {
  op: string;
  on?: 'vertex' | 'edge';
  kind?: string;
  keys?: string[];
  key?: string;
  label?: string;
  etype?: string;
  type?: string;
  edgeType?: string;
  direction?: 'out' | 'in';
  min?: number;
  max?: number | null;
  var?: string;
  predicate?: string;
  name?: string;
  query?: string;
};

const dumpLines = (bytes: Uint8Array): EngineSchemaLine[] =>
  decoder
    .decode(bytes)
    .split('\n')
    .filter((l) => l.length > 0)
    .map((l) => JSON.parse(l) as EngineSchemaLine);

/** The indexed property keys for `on`, from the engine's schema dump (sorted). */
const indexKeys = (bytes: Uint8Array, on: 'vertex' | 'edge'): string[] =>
  dumpLines(bytes)
    .filter((o) => o.op === 'createIndex' && o.on === on && Array.isArray(o.keys))
    .flatMap((o) => o.keys as string[])
    .sort();

// Each engine `{op:…}` family maps to one core SchemaOp (or `[]` when it has no
// core equivalent — e.g. the engine's opt-in edge-type index). Split per-op so
// each mapper stays simple.
const indexOp = (o: EngineSchemaLine): SchemaOp[] => {
  if (o.on === 'vertex' && o.keys?.[0] !== undefined) {
    return [{ op: 'createVertexIndex', key: o.keys[0] }];
  }

  if (o.on === 'edge' && o.kind === 'interval' && o.keys?.length === 2) {
    return [{ op: 'createEdgeIntervalIndex', loKey: o.keys[0], hiKey: o.keys[1] }];
  }

  return [];
};

const uniqueOp = (o: EngineSchemaLine): SchemaOp[] => {
  if (o.on === 'edge' && o.etype !== undefined && o.keys?.[0] !== undefined) {
    return [{ op: 'createEdgeUniqueConstraint', edgeType: o.etype, key: o.keys[0] }];
  }

  if (o.label !== undefined && o.keys?.[0] !== undefined) {
    return [{ op: 'createUniqueConstraint', label: o.label, key: o.keys[0] }];
  }

  return [];
};

const requiredOp = (o: EngineSchemaLine): SchemaOp[] => {
  if (o.on === 'edge' && o.etype !== undefined && o.key !== undefined) {
    return [{ op: 'createEdgeRequiredConstraint', edgeType: o.etype, key: o.key }];
  }

  if (o.label !== undefined && o.key !== undefined) {
    return [{ op: 'createRequiredConstraint', label: o.label, key: o.key }];
  }

  return [];
};

const typeOp = (o: EngineSchemaLine): SchemaOp[] => {
  if (o.key === undefined || o.type === undefined) {
    return [];
  }

  if (o.on === 'edge' && o.etype !== undefined) {
    return [
      { op: 'createEdgeTypeConstraint', edgeType: o.etype, key: o.key, type: o.type } as SchemaOp,
    ];
  }

  if (o.label !== undefined) {
    return [{ op: 'createTypeConstraint', label: o.label, key: o.key, type: o.type } as SchemaOp];
  }

  return [];
};

const cardinalityOp = (o: EngineSchemaLine): SchemaOp[] =>
  o.label !== undefined &&
  o.edgeType !== undefined &&
  o.direction !== undefined &&
  o.min !== undefined
    ? [
        {
          op: 'createCardinalityConstraint',
          label: o.label,
          edgeType: o.edgeType,
          direction: o.direction,
          min: o.min,
          max: o.max ?? null,
        },
      ]
    : [];

const validatorOp = (o: EngineSchemaLine): SchemaOp[] =>
  o.label !== undefined && o.var !== undefined && o.predicate !== undefined
    ? [{ op: 'createValidator', label: o.label, varName: o.var, predicate: o.predicate }]
    : [];

const invariantOp = (o: EngineSchemaLine): SchemaOp[] =>
  o.name !== undefined && o.query !== undefined
    ? [{ op: 'createInvariant', name: o.name, query: o.query }]
    : [];

/** Map one engine `{op:…}` schema-dump line to core's {@link SchemaOp}. */
const toSchemaOp = (o: EngineSchemaLine): SchemaOp[] => {
  const mapper: Record<string, (o: EngineSchemaLine) => SchemaOp[]> = {
    createIndex: indexOp,
    unique: uniqueOp,
    required: requiredOp,
    type: typeOp,
    cardinality: cardinalityOp,
    validator: validatorOp,
    invariant: invariantOp,
  };

  return mapper[o.op]?.(o) ?? [];
};

/**
 * Wrap a low-level {@link EngineAbi} as a full {@link Backend}. Shared by the FFI
 * and wasm engine backends.
 */
export const buildEngineBackend = (abi: EngineAbi): Backend => {
  // The graph-independent prepared-statement commands (prepare / preparedFree) still
  // need a store pointer for lnk_command, but do not read it — an empty scratch graph
  // serves. Prepared handles are kept in a JS-side table so the caller holds a small
  // number, decoupled from the engine's opaque (possibly >2^53) handle string.
  const scratch = abi.open(null, FMT_NDJSON);
  const prepared = new Map<number, string>();
  let nextPrepared = 1;

  return {
    abiVersion: abi.abiVersion,

    graphFromNdjson: (bytes, threads) => abi.open(bytes, FMT_NDJSON, threads),
    mergeNdjson: (handle, bytes): MergeReport =>
      // First-wins bulk merge (matching the TS engine): the engine returns the full report
      // (added counts + skipped ids + phantom endpoints) directly.
      parseJson<MergeReport>(abi.command(handle, 'merge', bytes)),
    graphClone: (handle) => abi.clone(handle),
    graphFree: (handle) => abi.close(handle),

    vertexCount: (handle) => abi.stat(handle, STAT_VERTEX),
    edgeCount: (handle) => abi.stat(handle, STAT_EDGE),
    version: (handle) => abi.stat(handle, STAT_VERSION),
    epoch: (handle, name) => parseJson<{ epoch: number }>(abi.command(handle, 'epoch', name)).epoch,

    createIndex: (handle, on, kind, keys) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'createIndex', on, kind, keys })),
    setConfig: (handle, id, value) => abi.config(handle, id, value) === 1,

    createUniqueConstraint: (handle, label, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'unique', on: 'vertex', label, key })),
    createRequiredConstraint: (handle, label, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'required', on: 'vertex', label, key })),
    createTypeConstraint: (handle, label, key, type) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'type', on: 'vertex', label, key, type })),
    createEdgeUniqueConstraint: (handle, edgeType, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'unique', on: 'edge', etype: edgeType, key })),
    createEdgeRequiredConstraint: (handle, edgeType, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'required', on: 'edge', etype: edgeType, key })),
    createEdgeTypeConstraint: (handle, edgeType, key, type) =>
      abi.schemaApply(
        handle,
        JSON.stringify({ op: 'type', on: 'edge', etype: edgeType, key, type }),
      ),
    createCardinalityConstraint: (handle, label, edgeType, direction, min, max) =>
      abi.schemaApply(
        handle,
        JSON.stringify({ op: 'cardinality', label, edgeType, direction, min, max }),
      ),
    createValidator: (handle, label, varName, predicate) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'validator', label, var: varName, predicate })),
    createInvariant: (handle, name, query) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'invariant', name, query })),
    dropVertexIndex: (handle, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'dropIndex', on: 'vertex', key })),
    dropEdgeIndex: (handle, key) =>
      abi.schemaApply(handle, JSON.stringify({ op: 'dropIndex', on: 'edge', key })),

    beginTransaction: (handle) => abi.tx(handle, TX_BEGIN),
    commitTransaction: (handle) => abi.tx(handle, TX_COMMIT),
    rollbackTransaction: (handle) => abi.tx(handle, TX_ROLLBACK),

    vertexIndexes: (handle) => indexKeys(abi.schemaDump(handle), 'vertex'),
    edgeIndexes: (handle) => indexKeys(abi.schemaDump(handle), 'edge'),
    dumpSchema: (handle) => dumpLines(abi.schemaDump(handle)).flatMap(toSchemaOp),
    lastWriteScope: (handle, key) =>
      parseJson<{ scopes: unknown[]; open: boolean }>(
        abi.command(handle, 'last_write_scope', key),
      ).scopes.map((v) => String(v)),

    queryRows: (handle, query, params) =>
      abi.query(handle, LANG_GQL, query, params ?? null, FMT_JSON),
    queryArrow: (handle, query, params) =>
      abi.query(handle, LANG_GQL, query, params ?? null, FMT_ARROW),
    // Arrow IPC: format 2 = FILE (Feather), 3 = STREAM. Honor the caller's `file`
    // flag — dropping it silently emitted the file framing for a stream request.
    queryArrowIpc: (handle, query, file, params) =>
      abi.query(
        handle,
        LANG_GQL,
        query,
        params ?? null,
        file ? FMT_ARROW_IPC : FMT_ARROW_IPC_STREAM,
      ),
    gremlinJson: (handle, query) => abi.query(handle, LANG_GREMLIN, query, null, FMT_JSON),

    // Run a native algorithm directly (also reachable via a GQL `CALL` query).
    algo: (handle, name, config) =>
      abi.command(
        handle,
        'algo',
        JSON.stringify({ name, config: config ? (JSON.parse(config) as unknown) : {} }),
      ),

    encodeNdjson: (handle) => abi.encode(handle, FMT_NDJSON),
    serialize: (handle, format) => {
      if (format === 'ndjson') {
        return abi.encode(handle, FMT_NDJSON);
      }

      if (format === 'binary') {
        return abi.encode(handle, FMT_BINARY);
      }

      const fmt = CODEC_FORMAT[format];

      if (fmt !== undefined) {
        return abi.encode(handle, fmt);
      }

      return unsupported(`serialize('${format}')`);
    },
    deserialize: (input, format) => {
      if (format === 'ndjson') {
        return abi.open(input, FMT_NDJSON);
      }

      if (format === 'binary') {
        return abi.open(input, FMT_BINARY);
      }

      const fmt = CODEC_FORMAT[format];

      if (fmt !== undefined) {
        return abi.open(input, fmt);
      }

      return unsupported(`deserialize('${format}')`);
    },

    prepare: (text) => {
      const { handle } = parseJson<{ handle: string }>(abi.command(scratch, 'prepare', text));
      const id = nextPrepared;
      nextPrepared += 1;
      prepared.set(id, handle);

      return id;
    },
    preparedFree: (p) => {
      const handle = prepared.get(p);

      if (handle === undefined) {
        return;
      }

      abi.command(scratch, 'prepared_free', JSON.stringify({ handle }));
      prepared.delete(p);
    },
    preparedQueryRows: (p, graph, params) => {
      const handle = prepared.get(p);

      if (handle === undefined) {
        throw new LenkeError('lenke: prepared statement is not live', { code: ErrorCode.Ffi });
      }

      const payload = params
        ? JSON.stringify({ handle, params: JSON.parse(params) as unknown })
        : JSON.stringify({ handle });

      return abi.command(graph, 'prepared_run', payload);
    },
    preparedQueryArrow: (p, graph, params) => {
      const handle = prepared.get(p);

      if (handle === undefined) {
        throw new LenkeError('lenke: prepared statement is not live', { code: ErrorCode.Ffi });
      }

      const payload = JSON.stringify({
        handle,
        format: 'arrow',
        ...(params ? { params: JSON.parse(params) as unknown } : {}),
      });

      return abi.command(graph, 'prepared_run', payload);
    },
  };
};
