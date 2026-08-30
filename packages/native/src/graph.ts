import type {
  AlgorithmConfig,
  CentralityRow,
  Clock,
  ClusterRow,
  ComponentRow,
  DegreeRow,
  GraphLimits,
  LabelRow,
  NeighborAggregateRow,
  OnCycleRow,
  PageRankRow,
  ScalarTypeName,
  ShortestPathRow,
  Temporal,
} from '@lenke/core';
import {
  CONFIG_IDS,
  fromTaggedJson,
  isTemporal,
  PARALLELISM_CONFIG_ID,
  TEMPORAL_TAG_KEYS,
  temporalLiteralParts,
} from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type {
  Backend,
  GraphHandle,
  IndexKind,
  IndexTarget,
  MergeReport,
  PreparedHandle,
} from './backend.js';
import { ensureDisposeSymbol } from './dispose.js';

/**
 * The document formats the native codecs read and write. A union, not a bare
 * `string`, so a typo (`'njson'`) is a compile error rather than a runtime one.
 * Mirrors the Rust `codec::decode` dispatch; the pure-TS `FormatName` is the
 * separate TS-codec list, and `@lenke/native` deliberately doesn't depend on it.
 */
export type GraphFormat = 'ndjson' | 'pg-json' | 'pg-text' | 'graphson' | 'csv';

/**
 * Construction-time settings for a native graph. Settings are constructor-only —
 * they are host policy, fixed for the graph's life, and pinning them removes any
 * question about what a mid-transaction change would mean. The one exception is
 * the clock (`RustGraph.setClock`), a host dependency rather than a resource bound.
 */
export type GraphConfigOptions = {
  /** Resource ceilings (see `GraphLimits`). */
  limits?: Partial<GraphLimits>;
  /** Shorthand for `limits.operatorChain`; the two are the same setting. */
  maxOperatorChain?: number;
  /**
   * Opt-in multicore for the in-engine graph algorithms: the max worker threads they
   * may use (default `1` = serial). A value > 1 runs betweenness / closeness /
   * PageRank / label-propagation / peer-pressure on a DEDICATED, bounded pool of that
   * size — never the global pool, so it will not starve the host event loop. Results
   * are byte-identical at any thread count. No effect on the wasm build (no threads);
   * a per-call `threads` on an algorithm config overrides this graph-level default.
   */
  parallelism?: number;
};

/** A decoded result row: column name → cell value. */
export type Row = Record<string, unknown>;

export type { IndexKind, IndexTarget } from './backend.js';

/**
 * The spec passed to {@link RustGraph.createIndex}: which element (`on`), the
 * index `kind`, and the property `keys` (`[k]` for a hash index, `[loKey, hiKey]`
 * for an interval index).
 */
export type IndexSpec = {
  on: IndexTarget;
  kind: IndexKind;
  keys: string[];
};

/**
 * The tagged wire form of a temporal — what `toJSON` emits and what comes back
 * through NDJSON, snapshots and `elementMap`: `{"@date": "2020-01-01"}`,
 * `{"@duration": "P1D"}`, and so on.
 */
export type TaggedTemporal = Readonly<Record<`@${string}`, string>>;

/**
 * A **schema** declaration as structured data — a constraint / validator /
 * invariant / index (or an index drop). These are set up via the programmatic API
 * ({@link RustGraph.createUniqueConstraint} …), so this union is the serializable
 * form: {@link applySchemaOp} applies one to a graph, {@link RustGraph.dumpSchema}
 * reads them all back out. `@lenke/sync` rides them over the CDC log so a replica
 * stays schema-in-lock-step, and the snapshot codec persists them for cold boot.
 * (`defineNode`/`defineEdge` are NOT here: they bind a label to a host-side JS
 * validator, which isn't engine state and can't be serialized — their *writes*
 * replicate instead.)
 */
export type SchemaOp =
  | { op: 'createVertexIndex'; key: string }
  | { op: 'createEdgeIndex'; key: string }
  | { op: 'createEdgeIntervalIndex'; loKey: string; hiKey: string }
  | { op: 'dropVertexIndex'; key: string }
  | { op: 'dropEdgeIndex'; key: string }
  | { op: 'createUniqueConstraint'; label: string; key: string }
  | { op: 'createRequiredConstraint'; label: string; key: string }
  | { op: 'createTypeConstraint'; label: string; key: string; type: ScalarTypeName }
  | { op: 'createEdgeUniqueConstraint'; edgeType: string; key: string }
  | { op: 'createEdgeRequiredConstraint'; edgeType: string; key: string }
  | { op: 'createEdgeTypeConstraint'; edgeType: string; key: string; type: ScalarTypeName }
  | {
      op: 'createCardinalityConstraint';
      label: string;
      edgeType: string;
      direction: 'out' | 'in';
      min: number;
      max: number | null;
    }
  | { op: 'createValidator'; label: string; varName: string; predicate: string }
  | { op: 'createInvariant'; name: string; query: string };

/**
 * Apply one {@link SchemaOp} to a graph by calling the matching declaration
 * method — the inverse of {@link RustGraph.dumpSchema}, and the single dispatch
 * shared by CDC replay (`@lenke/sync`) and snapshot cold-boot. Any throw (e.g.
 * existing data already violates the constraint) surfaces to the caller.
 */
export const applySchemaOp = (g: RustGraph, s: SchemaOp): void => {
  switch (s.op) {
    case 'createVertexIndex':
      g.createIndex({ on: 'vertex', kind: 'hash', keys: [s.key] });
      break;
    case 'createEdgeIndex':
      g.createIndex({ on: 'edge', kind: 'hash', keys: [s.key] });
      break;
    case 'createEdgeIntervalIndex':
      g.createIndex({ on: 'edge', kind: 'interval', keys: [s.loKey, s.hiKey] });
      break;
    case 'dropVertexIndex':
      g.dropVertexIndex(s.key);
      break;
    case 'dropEdgeIndex':
      g.dropEdgeIndex(s.key);
      break;
    case 'createUniqueConstraint':
      g.createUniqueConstraint(s.label, s.key);
      break;
    case 'createRequiredConstraint':
      g.createRequiredConstraint(s.label, s.key);
      break;
    case 'createTypeConstraint':
      g.createTypeConstraint(s.label, s.key, s.type);
      break;
    case 'createEdgeUniqueConstraint':
      g.createEdgeUniqueConstraint(s.edgeType, s.key);
      break;
    case 'createEdgeRequiredConstraint':
      g.createEdgeRequiredConstraint(s.edgeType, s.key);
      break;
    case 'createEdgeTypeConstraint':
      g.createEdgeTypeConstraint(s.edgeType, s.key, s.type);
      break;
    case 'createCardinalityConstraint':
      g.createCardinalityConstraint(s.label, s.edgeType, s.direction, s.min, s.max);
      break;
    case 'createValidator':
      g.createValidator(s.label, s.varName, s.predicate);
      break;
    case 'createInvariant':
      g.createInvariant(s.name, s.query);
      break;
  }
};

type RowSetJson = { columns: string[]; rows: unknown[][] };

const decoder = new TextDecoder();

// Decode a JSON carrier the crate handed back. The bytes are ours, but a decode
// failure here means the FFI result contract drifted — surface it as a coded
// `Ffi` fault rather than a bare `SyntaxError` with no provenance.
const parseJson = (bytes: Uint8Array, op: string): unknown => {
  try {
    return JSON.parse(decoder.decode(bytes));
  } catch (cause) {
    throw new LenkeError(`lenke: ${op} returned a non-JSON carrier`, {
      code: ErrorCode.Ffi,
      cause,
    });
  }
};

const isRowSet = (doc: unknown): doc is RowSetJson =>
  typeof doc === 'object' &&
  doc !== null &&
  Array.isArray((doc as RowSetJson).columns) &&
  Array.isArray((doc as RowSetJson).rows);

const decodeRows = (bytes: Uint8Array): Row[] => {
  const doc = parseJson(bytes, 'query');

  if (!isRowSet(doc)) {
    throw new LenkeError('lenke: query result was not a {columns, rows} document', {
      code: ErrorCode.Ffi,
    });
  }

  return doc.rows.map((row) => {
    const out: Row = {};
    doc.columns.forEach((col, i) => {
      out[col] = row[i];
    });

    return out;
  });
};

const isTemplate = (x: unknown): x is TemplateStringsArray =>
  Array.isArray((x as TemplateStringsArray)?.raw);

/** GQL bindings for the string-call form: `g.query('… $name', { name })`. */
export type QueryParams = Record<string, unknown>;

/**
 * Compile a GQL call into `(text, paramsJson)`. This is the injection-safety
 * seam: template substitutions become `$p0…$pn` **bindings**, serialized as a
 * flat JSON object and decoded by the crate — a value never touches the GQL
 * parser, so quotes/operators/keywords in it stay inert data. (The old
 * behavior spliced `String(sub)` into the query text.)
 *
 * The string form takes an optional explicit bindings object referencing its
 * own `$name`s. Templates own the `$p<n>` namespace — don't hand-write `$p0`
 * inside a template that also has `${}` substitutions.
 */
const compileGql = (
  q: string | TemplateStringsArray,
  subs: unknown[],
  clock?: Clock,
): { text: string; params: string | undefined } => {
  if (!isTemplate(q)) {
    const explicit = subs[0] as QueryParams | undefined;
    const bindings = withClock(explicit, clock);

    return {
      text: q,
      params: bindings && Object.keys(bindings).length ? stringifyParams(bindings) : undefined,
    };
  }

  if (subs.length === 0) {
    const bindings = withClock(undefined, clock);

    return { text: q.join(''), params: bindings ? stringifyParams(bindings) : undefined };
  }

  const bindings: QueryParams = {};
  const text = q.reduce((acc, part, i) => {
    if (i >= subs.length) {
      return acc + part;
    }

    bindings[`p${i}`] = subs[i];

    return `${acc + part}$p${i}`;
  }, '');

  return { text, params: stringifyParams(withClock(bindings, clock) ?? bindings) };
};

/**
 * Fold a wired {@link Clock} into a query's bindings: bind `$__now` from the
 * clock unless the caller supplied it explicitly (explicit always wins). Returns
 * the bindings unchanged when no clock is wired — zero cost on the common path.
 * The `LocalDateTime` the clock returns serializes (via `toJSON`) to a tagged
 * temporal the crate revives into the reserved `$__now` param.
 */
const withClock = (
  params: QueryParams | undefined,
  clock: Clock | undefined,
): QueryParams | undefined => {
  if (!clock || (params && Object.hasOwn(params, '__now'))) {
    return params;
  }

  return { ...params, __now: clock() };
};

// `JSON.stringify` throws a raw `TypeError` on a bigint value; surface a coded
// error instead. (Serializing a bigint as a JSON number would silently lose
// precision above 2^53, so a param rejects it rather than corrupt it — pass such
// a value as a string, or as a number if it is within the safe integer range.)
/** A tagged-temporal plain object, e.g. `{ '@date': '2027-05-25' }` (one @-key,
 *  string value) — the only object shape valid as a param value. The accepted
 *  keys come from the canonical `TEMPORAL_TAG_KEYS` (the exact set `fromTaggedJson`
 *  accepts), so this gate can't drift from what the engine actually ingests. */
const isTaggedTemporalObject = (v: object): boolean => {
  const keys = Object.keys(v);

  return (
    keys.length === 1 &&
    TEMPORAL_TAG_KEYS.has(keys[0]) &&
    typeof (v as Record<string, unknown>)[keys[0]] === 'string'
  );
};

/**
 * Validate a param value against the LPG param model BEFORE it JSON-encodes,
 * matching the TS engine's `validateParam` and native `gql/params.rs` so both
 * engines accept/reject exactly the same inputs. Without this a JS `Date` (or any
 * object with a `toJSON`) silently coerced to a string across the FFI while TS
 * rejected it, and a plain object faulted with a different code. Accepts a scalar,
 * a lenke `Temporal`, a tagged-temporal object, or a flat list of scalars/temporals.
 */
const assertParamModel = (name: string, value: unknown, inList = false): void => {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'boolean' ||
    typeof value === 'number'
  ) {
    return;
  }

  if (typeof value === 'bigint') {
    throw new LenkeError(
      `a bigint parameter ($${name}) is not supported: the numeric model is float64 — pass Number(x) or a string`,
      { code: ErrorCode.InvalidValue, details: { param: name } },
    );
  }

  if (isTemporal(value)) {
    return;
  }

  if (Array.isArray(value)) {
    if (inList) {
      throw new LenkeError(
        `parameter $${name} is a nested list; only a flat list of scalars is a valid param value`,
        { code: ErrorCode.InvalidValue, details: { param: name } },
      );
    }

    for (const el of value) {
      assertParamModel(name, el, true);
    }

    return;
  }

  if (typeof value === 'object' && isTaggedTemporalObject(value)) {
    return;
  }

  // `undefined` / function / symbol are dropped by `JSON.stringify`, so the binding
  // is simply absent → `E_MISSING_PARAMETER` at execute (both engines already agree
  // on this). Only a value that *would* encode to a non-model shape (a `Date` via
  // `toJSON`, a plain object) is rejected here.
  if (value === undefined || typeof value === 'function' || typeof value === 'symbol') {
    return;
  }

  throw new LenkeError(
    `parameter $${name} is outside the LPG param model: only a scalar, a flat list ` +
      `of scalars, or a tagged-temporal object is a valid param value`,
    { code: ErrorCode.InvalidValue, details: { param: name } },
  );
};

const stringifyParams = (params: object): string => {
  for (const [name, value] of Object.entries(params)) {
    assertParamModel(name, value);
  }

  return JSON.stringify(params, (_key, value: unknown) => {
    if (typeof value === 'bigint') {
      throw new LenkeError(
        'lenke: a bigint parameter cannot cross the native boundary without precision loss — pass it as a string or a safe-range number',
        { code: ErrorCode.InvalidValue },
      );
    }

    return value;
  });
};

/** Serialize a prepared statement's `$name` bindings (empty/absent → no params). */
const serializeParams = (params?: QueryParams): string | undefined =>
  params && Object.keys(params).length ? stringifyParams(params) : undefined;

/**
 * Serialize a JS scalar to a **safe** Gremlin literal — the injection-proof way
 * to put a value into traversal text, since Gremlin has no engine-side param
 * binding (unlike GQL's `$name`). Strings are single-quoted with `\` and `'`
 * escaped exactly as the lexer decodes them, so a value can never break out of
 * the literal; finite non-exponential numbers and `bigint`s pass through;
 * booleans map to `true`/`false`. `null`/`undefined` and non-scalars throw —
 * the engine has no literal for them.
 */
export const escapeGremlin = (value: unknown): string => {
  switch (typeof value) {
    case 'string':
      return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
    case 'boolean':
      return value ? 'true' : 'false';
    case 'bigint':
      return value.toString();
    case 'number': {
      const text = String(value);

      // The lexer's number grammar is `-?[0-9.]+` — no exponent form, no NaN.
      if (!/^-?\d+(\.\d+)?$/.test(text)) {
        throw new LenkeError(`lenke: gremlin cannot embed the number ${text} as a literal`, {
          code: ErrorCode.InvalidGraphOp,
        });
      }

      return text;
    }
    default: {
      // A temporal embeds as the dialect's literal constructor — `date('…')`.
      // Accepts both a stored instance and the tagged wire form `{"@date": …}`,
      // since a caller most often passes back a value it read out of a graph or
      // wrote into one. Without this a temporal could only be hand-spelled, which
      // is exactly what this helper exists to avoid.
      const temporal = isTemporal(value) ? value : fromTaggedJson(value);

      if (temporal) {
        const parts = temporalLiteralParts(temporal);

        if (parts) {
          return `${parts.kind}(${escapeGremlin(parts.iso)})`;
        }
      }

      throw new LenkeError(
        `lenke: gremlin cannot embed a ${value === null ? 'null' : typeof value} value as a literal`,
        { code: ErrorCode.InvalidGraphOp },
      );
    }
  }
};

/**
 * A value {@link escapeGremlin} can embed as a Gremlin literal: the scalars the
 * lexer has literal forms for, plus a temporal (either a stored instance or the
 * tagged wire form `{"@date": "2020-01-01"}` that `toJSON` emits). Anything else
 * — `null`, arrays, plain objects, a `Vertex` — has no literal and is rejected.
 */
export type GremlinLiteral = string | number | bigint | boolean | Temporal | TaggedTemporal;

/**
 * Build Gremlin traversal text, escaping every `${value}` via
 * {@link escapeGremlin}.
 *
 * **This is Gremlin's parameter binding.** Unlike GQL — where `query(text, {p})`
 * sends bindings alongside the text and a value never reaches the parser —
 * Gremlin here has no engine-side `$name`. Values are escaped *into* the text at
 * compose time, and this tagged template is the seam that does it safely:
 *
 * ```ts
 * const asOf = { '@date': '2021-06-01' };
 * g.gremlin`g.V().has('id', eq(${userId})).outE('R').has('vf', lte(${asOf}))`;
 * ```
 *
 * A string interpolation is single-quoted with `\` and `'` escaped exactly as
 * the lexer decodes them, so a hostile value collapses to one inert literal
 * rather than closing the quote and injecting steps. Never build traversal text
 * with `+` or a bare template literal — that is the injection you are avoiding.
 *
 * There is deliberately **no** `gremlin(text, params)` form. It reads like GQL's
 * and would silently discard the bindings (a plain string has no interpolation
 * sites), so it is a type error, and a runtime throw if the types are bypassed.
 * A plain string with nothing to substitute passes through unchanged.
 *
 * @see escapeGremlin for the per-value rules.
 * @see RustGraph.gremlin to compose and run in one step.
 */
export function gremlin(q: string): string;
export function gremlin(q: TemplateStringsArray, ...subs: readonly GremlinLiteral[]): string;
export function gremlin(q: string | TemplateStringsArray, ...subs: unknown[]): string {
  return composeGremlin(q, ...subs);
}

/**
 * {@link gremlin} without the overloads — for forwarding wrappers whose own
 * parameter is the `string | TemplateStringsArray` union (a tagged-template
 * method cannot narrow its own argument before passing it on). Same runtime
 * behaviour, including the `(text, params)` refusal; prefer {@link gremlin} in
 * user-facing code, where the overloads make that call a compile error.
 */
export function composeGremlin(q: string | TemplateStringsArray, ...subs: unknown[]): string {
  if (!isTemplate(q)) {
    // A plain string has no interpolation sites, so any extra arguments would be
    // silently discarded. That reads exactly like GQL's `query(text, params)` and
    // fails far downstream as a parse error on the un-substituted text, so refuse
    // it here and say what to use instead. The overloads above make this
    // unreachable from typed code; this catches JS callers and `as any` casts.
    if (subs.length > 0) {
      throw new LenkeError(
        'lenke: gremlin(text, params) is not a binding form — Gremlin has no engine-side ' +
          'parameter binding. Use the tagged template so values are escaped into the text: ' +
          "g.gremlin`g.V().has('k', eq(<value>))` — every interpolation is escaped.",
        { code: ErrorCode.InvalidGraphOp },
      );
    }

    return q;
  }

  return q.reduce(
    (acc, part, i) => acc + part + (i < subs.length ? escapeGremlin(subs[i]) : ''),
    '',
  );
}

// ARW1 column type tags (mirrors crates/lenke-engine/src/arrow.rs).
const ARW_FLOAT64 = 1;
const ARW_BOOL = 2;
const ARW_UTF8 = 3;
const ARW_FIXED_LIST = 4; // FixedSizeList<Float64>[dim]; dim rides buf2_len

/**
 * Decode an ARW1 columnar blob (from {@link RustGraph.queryArrow} /
 * `lnk_query_arrow`) back into {@link Row}s. The blob is a 24-byte header
 * (`"ARW1"` magic [4 bytes] | `version` u32 LE [4 bytes] | `nrows` **u64 LE**
 * [8 bytes] | `ncols` **u64 LE** [8 bytes]) + `ncols` × 40-byte descriptors + the
 * referenced Apache-Arrow little-endian buffers; this reads them in place (no
 * Arrow dependency). Its purpose on the wire: ship/transfer the columnar bytes
 * instead of JSON rows, and materialize here only when a consumer wants objects.
 *
 * **Scalar columns only.** ARW1 has three column types — float64, bool, utf8 —
 * so a projection of scalars (numbers, booleans, strings) round-trips
 * byte-identical to the JSON `query` path. A column holding a **list** (e.g.
 * `collect_list(...)`) or a whole **element** (`RETURN n`) is flattened into its
 * text form in the utf8 column, so it does NOT reconstruct as the structured
 * array/object the JSON path returns. Use `query()` (JSON) for non-scalar
 * projections; reserve `queryArrow`/`decodeArrow` for scalar analytical columns.
 *
 * Pass a row shape to type the result: `decodeArrow<{ n: string }>(blob)`.
 */
export const decodeArrow = <R extends Row = Row>(blob: Uint8Array): R[] => {
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const td = new TextDecoder();

  if (blob.length < 24 || td.decode(blob.subarray(0, 4)) !== 'ARW1') {
    throw new LenkeError('lenke: not an ARW1 arrow blob', { code: ErrorCode.Ffi });
  }

  const nrows = Number(dv.getBigUint64(8, true));
  const ncols = Number(dv.getBigUint64(16, true));
  const rows: Row[] = Array.from({ length: nrows }, () => ({}));

  for (let c = 0; c < ncols; c += 1) {
    const d = 24 + c * 40;
    const type = dv.getUint32(d, true);
    const nameOff = dv.getUint32(d + 8, true);
    const nameLen = dv.getUint32(d + 12, true);
    const validityOff = dv.getUint32(d + 16, true);
    const validityLen = dv.getUint32(d + 20, true);
    const buf1Off = dv.getUint32(d + 24, true);
    const buf2Off = dv.getUint32(d + 32, true);
    const name = td.decode(blob.subarray(nameOff, nameOff + nameLen));

    // LSB-first validity bitmap; length 0 ⇒ every row valid (no nulls). The
    // column type is invariant, so decide it ONCE and run a tight per-row loop
    // (rather than re-testing the type on every one of the nrows cells).
    const isNull = (i: number): boolean =>
      validityLen !== 0 && (blob[validityOff + (i >> 3)] & (1 << (i & 7))) === 0;

    if (type === ARW_FLOAT64) {
      for (let i = 0; i < nrows; i += 1) {
        rows[i][name] = isNull(i) ? null : dv.getFloat64(buf1Off + i * 8, true);
      }
    } else if (type === ARW_BOOL) {
      for (let i = 0; i < nrows; i += 1) {
        rows[i][name] = isNull(i) ? null : (blob[buf1Off + (i >> 3)] & (1 << (i & 7))) !== 0;
      }
    } else if (type === ARW_UTF8) {
      for (let i = 0; i < nrows; i += 1) {
        if (isNull(i)) {
          rows[i][name] = null;
          continue;
        }

        const start = dv.getInt32(buf1Off + i * 4, true);
        const end = dv.getInt32(buf1Off + (i + 1) * 4, true);
        rows[i][name] = td.decode(blob.subarray(buf2Off + start, buf2Off + end));
      }
    } else if (type === ARW_FIXED_LIST) {
      // FixedSizeList<Float64>[dim]: dim rides buf2_len; buf1 is the flat child
      // values (nrows × dim). A null list → null (not a zero vector).
      const dim = dv.getUint32(d + 36, true);

      for (let i = 0; i < nrows; i += 1) {
        if (isNull(i)) {
          rows[i][name] = null;
          continue;
        }

        const base = buf1Off + i * dim * 8;
        rows[i][name] = Array.from({ length: dim }, (_, j) => dv.getFloat64(base + j * 8, true));
      }
    } else {
      throw new LenkeError(`lenke: arrow column '${name}' has unknown type ${type}`, {
        code: ErrorCode.Ffi,
      });
    }
  }

  return rows as R[];
};

/**
 * An ergonomic handle over a Rust-backed graph. Wraps a {@link Backend} and an
 * opaque graph handle; decodes the JSON/Arrow carriers the crate returns.
 *
 * The underlying graph is heap-owned by the native/wasm module (and pinned by
 * the napi backend's id→object registry), so it must be released. Three ways,
 * uniform across the ffi / napi / wasm backends:
 *   - `using g = graphFromNdjson(...)` — released at scope exit (preferred);
 *   - `g.free()` — release explicitly (idempotent; the handle is dead after);
 *   - forget both — a {@link FinalizationRegistry} backstop reclaims the handle
 *     when the wrapper is collected. Best-effort (the GC may never run it before
 *     exit), so it is a leak-net, not a substitute for `using`/`free()`.
 */
export type RustGraph = {
  readonly vertexCount: number;
  readonly edgeCount: number;
  /** Monotonic mutation counter — O(1) "did anything change?" for snapshots. */
  readonly version: number;
  /** Per-token change epoch (label / edge-type / property-key) for finer invalidation. */
  epoch: (name: string) => number;
  /**
   * Declare an opt-in secondary index (backfills existing elements, then stays
   * current; idempotent). The single index-creation entry point:
   *   - `on`: `'vertex'` or `'edge'`.
   *   - `kind`: `'hash'` — an equality seek over one key, turning
   *     `WHERE x.k = …` / `x.k IN […]` / range constraints into seeks instead of
   *     scans (a large win for repeated point lookups, e.g. bulk edge inserts that
   *     `MATCH` their endpoints by id); or `'interval'` — an RI-tree over an edge
   *     `[loKey, hiKey)` temporal pair, so an as-of (`lo <= v AND hi > v`) /
   *     overlap predicate seeds from it (two — valid `[vf,vt)` + transaction
   *     `[tf,tt)` — cover a bitemporal as-of).
   *   - `keys`: `[k]` for a hash index, `[loKey, hiKey]` for an interval index.
   *
   * An interval index is edge-only.
   *
   * @example g.createIndex({ on: 'vertex', kind: 'hash', keys: ['email'] })
   * @example g.createIndex({ on: 'edge', kind: 'interval', keys: ['vf', 'vt'] })
   */
  createIndex: (spec: IndexSpec) => void;
  /**
   * Declare a UNIQUE constraint on `(label, key)`: at most one vertex carrying
   * `label` may hold a given non-null value for `key`. Index-backed. Throws
   * `ConstraintViolation` if the current data already violates it. The Pattern-B
   * primitive `_MERGE` keys on; see docs/design/gql-extensions.md §3.
   */
  createUniqueConstraint: (label: string, key: string) => void;
  createRequiredConstraint: (label: string, key: string) => void;
  createTypeConstraint: (label: string, key: string, type: ScalarTypeName) => void;
  /**
   * Declare a UNIQUE / REQUIRED / TYPE constraint on `(edgeType, key)` — the edge
   * analogue of the vertex constraints above, keyed by edge type and enforced
   * against edge properties. Throws `ConstraintViolation` (or `InvalidValue` for
   * an unknown type name) exactly as the vertex forms do.
   */
  createEdgeUniqueConstraint: (edgeType: string, key: string) => void;
  createEdgeRequiredConstraint: (edgeType: string, key: string) => void;
  createEdgeTypeConstraint: (edgeType: string, key: string, type: ScalarTypeName) => void;
  /**
   * Declare a CARDINALITY constraint bounding the degree of every vertex carrying
   * `label` over `edgeType` in `direction` (`out` = source, `in` = target) to
   * `min..=max` (`max: null` unbounded). Throws `ConstraintViolation` if the
   * current data already violates it. Max is enforced at each statement's
   * auto-commit; min is commit-time only. See docs/design/transactions.md.
   */
  createCardinalityConstraint: (
    label: string,
    edgeType: string,
    direction: 'out' | 'in',
    min: number,
    max: number | null,
  ) => void;
  /**
   * Declare a custom VALIDATOR on `label` (a vertex label OR an edge type): every
   * element carrying the label must satisfy the GQL boolean `predicate` (pure ISO
   * WHERE-clause syntax), with the element bound to `varName`
   * (`g.createValidator('User', 'u', 'u.age >= 0 AND u.age < 150')`). SQL-`CHECK`
   * semantics — rejected only on a definite `false`; a null/unknown result passes.
   * Throws `ConstraintViolation` if existing data already violates it, or `Syntax`
   * if the predicate can't be parsed. The native twin of `@lenke/gql`'s
   * `createValidator`, enforced byte-identically in the Rust GQL evaluator.
   */
  createValidator: (label: string, varName: string, predicate: string) => void;
  /**
   * Declare a graph-level INVARIANT `name` = a whole-graph GQL `query` (`MATCH …
   * RETURN`) that must hold after every write transaction — the cross-write twin
   * of a per-element validator (`g.createInvariant('balanced', 'MATCH (a:Acct)
   * RETURN sum(a.balance) = 0')`). Evaluated ONCE per commit against the fully-
   * staged graph; `false`-only-fails: VIOLATED iff any result cell is boolean
   * `false` (`true`/`null`/non-boolean/empty all hold). Throws
   * `ConstraintViolation` if existing data already violates it, or `Syntax` if the
   * query can't be parsed. The native twin of `@lenke/gql`'s `createInvariant`,
   * enforced byte-identically in the Rust GQL evaluator.
   */
  createInvariant: (name: string, query: string) => void;
  /** Drop a vertex / edge property index (no-op if absent). */
  dropVertexIndex: (key: string) => void;
  dropEdgeIndex: (key: string) => void;
  /** The vertex / edge property keys that currently carry a secondary index (sorted). */
  vertexIndexes: () => string[];
  edgeIndexes: () => string[];
  /**
   * The full active schema as replayable {@link SchemaOp}s — constraints,
   * validators, invariants, and indexes — in a deterministic order. The read side
   * of the `create*` declarations (their inverse): feed each back through
   * {@link applySchemaOp} to reconstruct the schema on another graph. This is how
   * the snapshot codec persists schema alongside the graph data, so a cold boot
   * restores the constraints/validators/indexes that can't be derived from data.
   * `defineNode`/`defineEdge` are NOT here — a host-side JS validator isn't engine
   * state.
   */
  dumpSchema: () => SchemaOp[];
  /**
   * The distinct values of property `key` across the vertices the most recent
   * committed write touched — that write's content-derived **value-scope**, for CDC
   * interest routing (`lastWriteScope('room')` → `['42']` right after a write into
   * room 42). Empty when the last write touched no vertex carrying `key`. Reads a
   * handful of columns off the already-collected touched set.
   */
  lastWriteScope: (key: string) => string[];
  /**
   * Run `fn` as one atomic transaction. Every write inside applies to the
   * graph immediately (so reads see their own writes), but if `fn` throws — or a
   * deferred constraint check fails at commit — the whole batch rolls back and
   * nothing persists. On success the writes commit together. Returns whatever
   * `fn` returns. Nesting joins the outer transaction (flat, savepoint-less): the
   * outermost frame owns commit/rollback. The engine-neutral transaction surface,
   * mirroring the TS core (`packages/core/src/core/Graph.ts`).
   */
  transaction: <T>(fn: (graph: RustGraph) => T) => T;
  /**
   * Lower-level transaction handle mirroring TinkerPop's `graph.tx()`: the
   * transaction opens now; call `commit()` or `rollback()` explicitly. Prefer
   * {@link RustGraph.transaction} for the common auto-managed case.
   */
  tx: () => { commit: () => void; rollback: () => void };
  /** Open a transaction frame. Nesting increments depth; the outermost frame owns commit/rollback. */
  beginTransaction: () => void;
  /**
   * Close the current frame. The outermost commit runs the deferred constraint
   * checks against the fully-staged graph and throws `ConstraintViolation` (after
   * rolling the whole transaction back) if one fails.
   */
  commitTransaction: () => void;
  /** Roll the current transaction back: reverse every staged write. No-op if none open. */
  rollbackTransaction: () => void;
  /**
   * Wire (or clear, with `null`) the host {@link Clock} that supplies `$__now`
   * for the ISO now-functions (`current_date`/`current_timestamp`). Read once
   * per `query`/`queryArrow`, and only when that call didn't pass an explicit
   * `$__now` (so an explicit value still wins). The engine never reads a clock
   * itself — the clock's `LocalDateTime` is serialized and bound as a param, so
   * native and wasm stay identical. Returns the graph for chaining.
   */
  setClock: (clock: Clock | null) => RustGraph;
  /**
   * Run a GQL query → decoded rows. Two safe, parameterized forms:
   * - tagged template — each `${sub}` compiles to a `$p<n>` **binding**, never
   *   spliced text: ``g.query`MATCH (p:Person) WHERE p.name = ${name} RETURN p` ``
   * - string + bindings — `g.query('… WHERE p.name = $name RETURN p', { name })`
   *
   * Pass a row shape to type the result — `g.query<{ name: string }>('…')` — an
   * opt-in, caller-side assertion (rows are `Record<string, unknown>` at
   * runtime; nothing is validated). Defaults to {@link Row}.
   */
  query: {
    <R extends Row = Row>(text: string, params?: QueryParams): R[];
    <R extends Row = Row>(strings: TemplateStringsArray, ...subs: unknown[]): R[];
  };
  /** Run a GQL query → raw Arrow ("ARW1") columnar blob. Same two forms as {@link RustGraph.query}. */
  queryArrow: {
    (text: string, params?: QueryParams): Uint8Array;
    (strings: TemplateStringsArray, ...subs: unknown[]): Uint8Array;
  };
  /**
   * Run a GQL query → standard **Apache Arrow IPC** bytes, framed **natively** (no
   * JS re-encode) — the interop handoff to DuckDB / Polars / pandas. `opts.format`
   * picks the IPC stream layout (default) or the file / Feather-v2 layout; the
   * tagged-template form (injection-safe `${}` interpolation, like {@link query})
   * always emits the stream layout. For a JS-side transcode of an existing ARW1
   * blob instead, see `toArrowIPC` in `@lenke/native/arrow`.
   */
  queryArrowIpc: {
    (text: string, opts?: { params?: QueryParams; format?: 'stream' | 'file' }): Uint8Array;
    (strings: TemplateStringsArray, ...subs: unknown[]): Uint8Array;
  };
  /**
   * Run a textual Gremlin traversal → JSON-decoded result stream.
   *
   * **Use the tagged template — it is Gremlin's parameter binding.** GQL sends
   * bindings beside the text (`query(text, { p })`) so a value never reaches the
   * parser; Gremlin has no engine-side `$name`, so values are escaped *into* the
   * text at compose time via {@link escapeGremlin}:
   *
   * ```ts
   * g.gremlin`g.V().has('name', ${userInput}).values('age')`;
   * g.gremlin`g.V().outE('R').has('vf', lte(${asOfDate}))`;
   * ```
   *
   * A string interpolation becomes one quoted literal with `\` and `'` escaped,
   * so a hostile value matches nothing instead of closing the quote and
   * injecting steps. Numbers, booleans, bigints and temporals get their literal
   * forms; anything else throws rather than guessing.
   *
   * Do **not** hand-build the text with `+` or a bare template literal, and note
   * there is no `gremlin(text, params)` form — it reads like GQL's but would
   * silently drop the bindings, so it is a type error. Pass a plain string only
   * when it is a constant. To build text for elsewhere (a sync client, a log)
   * without running it, use the {@link gremlin} tag directly.
   */
  gremlin: {
    (q: string): unknown[];
    (q: TemplateStringsArray, ...subs: readonly GremlinLiteral[]): unknown[];
  };
  /**
   * The graph algorithms — each runs the whole computation natively and resolves a
   * `Promise` of `{ node, … }` rows in insertion order. On the Node/napi backend
   * the run happens **off the JS thread** (on a libuv threadpool thread), so the
   * event loop stays free; by default the algorithm itself runs single-threaded —
   * pass `threads` (or set the graph's `parallelism`) to opt into a bounded multicore
   * pool for the float-heavy ones (betweenness/closeness/PageRank/label-propagation/
   * peer-pressure), byte-identical to serial. While the promise
   * is pending the graph is single-flight-locked (any other call on it throws until
   * it settles — the off-thread read must not race a mutation). On the bun:ffi /
   * wasm backends (no threadpool) they resolve the same rows but the run blocks;
   * prefer the `@lenke/core` cooperative-yield functions there. A `writeProperty`
   * config writes each result back to that vertex property.
   *
   * `degree`: `direction` (`'out'` default / `'in'` / `'both'`) + optional
   * `edgeLabel`. `connectedComponents`: `componentId` = the component's first-inserted
   * vertex id. `labelPropagation`: `iterations` (default 10). `pagerank`:
   * `iterations` (default 20), `dampingFactor` (default 0.85), optional
   * `weightProperty`. `shortestPath`: from `config.source`, unweighted BFS or (with
   * `weightProperty`) Dijkstra. `betweenness` / `closeness`: shortest-path centrality
   * over out-edges (optionally `edgeLabel` / `weightProperty`), directed and
   * unnormalized — **O(V·E)**, so not for very large graphs. Each has a data-last
   * twin in `@lenke/core`.
   */
  degree: (config?: AlgorithmConfig) => Promise<DegreeRow[]>;
  connectedComponents: (config?: AlgorithmConfig) => Promise<ComponentRow[]>;
  /**
   * Strongly-connected components (directed): two vertices share a component iff
   * each is reachable from the other along directed edges. Each `componentId` is the
   * component's first-inserted member's external id. Optional `edgeLabel` filter.
   */
  stronglyConnectedComponents: (config?: AlgorithmConfig) => Promise<ComponentRow[]>;
  /**
   * Per-vertex cycle membership: `onCycle` is true iff the vertex lies on a directed
   * cycle — its SCC has more than one member, or it has a self-loop. Optional
   * `edgeLabel` filter.
   */
  onCycle: (config?: AlgorithmConfig) => Promise<OnCycleRow[]>;
  labelPropagation: (config?: AlgorithmConfig) => Promise<LabelRow[]>;
  peerPressure: (config?: AlgorithmConfig) => Promise<ClusterRow[]>;
  pagerank: (config?: AlgorithmConfig) => Promise<PageRankRow[]>;
  /**
   * Personalized PageRank / random-walk-with-restart: like {@link pagerank} but the
   * surfer restarts to the `sourceNodes` seed set (external ids) instead of
   * uniformly, ranking by proximity to those seeds. Same `iterations` /
   * `dampingFactor` / `weightProperty` / `edgeLabel` knobs; an empty/all-unknown
   * seed set degenerates to global PageRank.
   */
  personalizedPagerank: (config?: AlgorithmConfig) => Promise<PageRankRow[]>;
  betweenness: (config?: AlgorithmConfig) => Promise<CentralityRow[]>;
  closeness: (config?: AlgorithmConfig) => Promise<CentralityRow[]>;
  shortestPath: (config?: AlgorithmConfig) => Promise<ShortestPathRow[]>;
  /**
   * Vectorized neighbour aggregation (message passing): for each vertex, aggregate
   * its neighbours' list-valued `config.feature` vectors element-wise over the whole
   * block — `config.op` (`'mean'` default / `'sum'` / `'max'` / `'min'`), by
   * `config.direction`, optionally `config.includeSelf`. One native pass; the result
   * list is written to `config.writeProperty` when set.
   */
  neighborAggregate: (config?: AlgorithmConfig) => Promise<NeighborAggregateRow[]>;
  /**
   * Deep-copy this graph into a fresh, fully independent {@link RustGraph} — the
   * fast fork/branch substrate. An O(V+E) native clone of the columnar store, not
   * a serialize→parse NDJSON round-trip: no text, no re-validation, no re-indexing,
   * and element ids are preserved exactly (a base-vs-copy diff by id is exact).
   * The copy owns its own native memory and must be `free`d (or `using`-bound)
   * independently — freeing one does not affect the other.
   */
  copy: () => RustGraph;
  /** Serialize the graph back to NDJSON bytes. */
  toNdjson: () => Uint8Array;
  /**
   * Bulk-append NDJSON bytes into this graph — a `COPY FROM` for a live store.
   * Ingests at bulk speed (no per-`INSERT` parse); a node whose id already
   * exists is first-wins-skipped, edge endpoints resolve against the graph.
   * Equivalent to `deserialize(bytes, 'ndjson', existingGraph)` on the TS core.
   * Returns a {@link MergeReport} of what applied vs. skipped.
   */
  mergeNdjson: (bytes: Uint8Array) => MergeReport;
  /** Serialize the graph in a named format (`pg-json | pg-text | graphson | csv | ndjson`). */
  serialize: (format: string) => string;
  /**
   * Compile a GQL query once into a reusable {@link PreparedQuery}, bound to this
   * graph — parse/lower is paid once, then each `.query(params)` skips it. The
   * win over `query()` is the ~per-call parse cost; results are identical. A
   * syntax error throws here, at prepare time. `prepare` takes a plain string
   * with `$name` params (no tagged-template form — the text is fixed). Pass a
   * row shape to type `.query(params)`'s result. `maxOperatorChain` overrides the
   * operator-chain ceiling for this parse (default 10_000; prepared statements are
   * graph-independent, so this is a prepare-time option, not the graph's setting).
   */
  prepare: <R extends Row = Row>(
    text: string,
    opts?: { maxOperatorChain?: number },
  ) => PreparedQuery<R>;
  /** Release the underlying graph. Idempotent; the handle is invalid afterwards. */
  free: () => void;
  /** `using`-compatible alias of {@link RustGraph.free} (Explicit Resource Management). */
  [Symbol.dispose]: () => void;
};

/**
 * A compiled GQL query bound to a graph (from {@link RustGraph.prepare}). Rerun
 * it cheaply with fresh `$name` params. `free()` releases the compiled plan; the
 * graph it was prepared on must still be live at execute time.
 */
export type PreparedQuery<R extends Row = Row> = {
  /** Execute with optional `$name` bindings → decoded rows. */
  query: (params?: QueryParams) => R[];
  /** Execute → the raw Arrow ("ARW1") columnar blob. */
  queryArrow: (params?: QueryParams) => Uint8Array;
  /** Release the compiled plan. Idempotent; the prepared query is dead afterwards. */
  free: () => void;
  /** `using`-compatible alias of {@link PreparedQuery.free}. */
  [Symbol.dispose]: () => void;
};

// GC backstop for a leaked RustGraph. The crate owns the graph's heap and the
// napi backend's registry pins it, so a wrapper collected without free() would
// leak. This reclaims the handle when the wrapper becomes unreachable — a
// leak-net only (the GC is not guaranteed to run it). Skipped on runtimes with
// no FinalizationRegistry; free()/`using` still work there.
const reclaim: FinalizationRegistry<() => void> | undefined =
  typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry<() => void>((release) => {
        release();
      })
    : undefined;

type DisposalState = { freed: boolean; busy: boolean };

// Disposal state shared per (backend, handle) — NOT per wrapper — so two
// wrappers attached to the same handle share one freed-once guard: a.free()
// then b.free() (or b's GC backstop) reaches backend.graphFree exactly once,
// never a double-free. Entries are removed on free, so a backend that recycles
// a handle value hands the next attachment a fresh state; the per-backend Map
// lives inside a WeakMap and dies with its backend.
const disposal = new WeakMap<Backend, Map<GraphHandle, DisposalState>>();

const stateFor = (backend: Backend, handle: GraphHandle): DisposalState => {
  let states = disposal.get(backend);

  if (!states) {
    states = new Map();
    disposal.set(backend, states);
  }

  let state = states.get(handle);

  if (!state) {
    state = { freed: false, busy: false };
    states.set(handle, state);
  }

  return state;
};

// Dev-only leak signal. The GC backstop firing at all means a wrapper was
// collected without free()/`using`: the handle still gets reclaimed below, but
// that relies on the GC running the finalizer — which the spec never
// guarantees — so a leak may sit unreleased indefinitely. Warn once to nudge
// the caller toward deterministic disposal. Silent in production and when
// LENKE_SILENCE_LEAK_WARNING is set; the reclaim itself is never affected.
let leakWarned = false;

const leakWarningsEnabled = (): boolean => {
  // Browser/worker builds have no `process` — keep the dev aid on there, since
  // there is no NODE_ENV to mark a production bundle. Node/Bun honor both the
  // explicit opt-out and NODE_ENV=production.
  if (typeof process === 'undefined') {
    return true;
  }

  const { env } = process;

  return !env.LENKE_SILENCE_LEAK_WARNING && env.NODE_ENV !== 'production';
};

const warnLeak = (): void => {
  if (leakWarned || !leakWarningsEnabled()) {
    return;
  }

  leakWarned = true;
  console.warn(
    'lenke: a RustGraph was garbage-collected without free() — the handle was ' +
      'reclaimed by a GC backstop, but finalizers are not guaranteed to run, so ' +
      'leaked graphs can sit unreleased. Dispose deterministically with ' +
      '`using g = ...` or an explicit g.free(). ' +
      '(Set LENKE_SILENCE_LEAK_WARNING to silence this warning.)',
  );
};

// Test-only: clear the one-shot latch. Tests share a single process, so a graph
// leaked by an earlier test can reclaim and consume the once-per-process warning
// before the dedicated leak test runs — leaving it with nothing to observe.
// Resetting first makes that test deterministic. Not part of the public surface.
export const __resetLeakWarnedForTests = (): void => {
  leakWarned = false;
};

// Built at module scope so the registered thunk closes over `backend` /
// `handle` / `state` only — never the wrapper — or it would pin the very
// object it exists to reclaim. Only ever invoked from the GC backstop (free()
// unregisters its token), so a live `!freed` here is, by construction, a leak.
const reclaimThunk = (backend: Backend, handle: GraphHandle, state: DisposalState) => (): void => {
  if (!state.freed) {
    warnLeak();
    state.freed = true;
    disposal.get(backend)?.delete(handle);
    backend.graphFree(handle);
  }
};

/** Wrap an existing backend + handle as a {@link RustGraph}. */
/**
 * Build a {@link PreparedQuery} over `backend`, bound to the graph guarded by
 * `live`. Owns its own compiled-plan handle (freed independently of the graph);
 * `live()` still guards the graph, so a prepared query on a freed graph throws.
 */
const makePrepared = <R extends Row = Row>(
  backend: Backend,
  live: () => GraphHandle,
  text: string,
  maxOperatorChain?: number,
): PreparedQuery<R> => {
  ensureDisposeSymbol();
  // throws on a syntax error
  const prepHandle: PreparedHandle = backend.prepare(text, maxOperatorChain);
  let freed = false;

  const prepLive = (): PreparedHandle => {
    if (freed) {
      throw new LenkeError('lenke: prepared query used after free()', {
        code: ErrorCode.InvalidGraphOp,
      });
    }

    return prepHandle;
  };

  const free = (): void => {
    if (freed) {
      return;
    }

    backend.preparedFree(prepHandle);
    freed = true;
  };

  return {
    query: (params) =>
      decodeRows(backend.preparedQueryRows(prepLive(), live(), serializeParams(params))) as R[],
    queryArrow: (params) => backend.preparedQueryArrow(prepLive(), live(), serializeParams(params)),
    free,
    [Symbol.dispose]: free,
  };
};

export const attachGraph = (backend: Backend, handle: GraphHandle): RustGraph => {
  ensureDisposeSymbol(); // so the [Symbol.dispose] key below resolves on any runtime

  // The host wall-clock wired via `setClock`, read on every `query`/`queryArrow`
  // to supply `$__now` for the ISO now-functions. Null → those read null: the
  // engine never invents a clock, so it stays pure across native and wasm.
  let clock: Clock | null = null;

  // Shared with every other wrapper on this (backend, handle). Note free()
  // deletes the map entry, so attaching AFTER a free gets fresh state — that is
  // deliberate: a backend may recycle handle values (ffi handles are pointers),
  // and a recycled handle is a brand-new graph, not a freed one.
  const state = stateFor(backend, handle);
  const token = {};

  // Every member passes the handle through this gate: a freed graph answers
  // with a coded error instead of handing the backend a dangling handle (on
  // the ffi backend that read would be a native use-after-free, not a throw).
  const live = (): GraphHandle => {
    if (state.freed) {
      throw new LenkeError('lenke: graph used after free()', { code: ErrorCode.InvalidGraphOp });
    }

    // Single-flight guard for the async algorithm path: while an off-thread run is
    // reading this graph, any other native call would risk a data race, so it throws
    // until the promise settles. (This is what makes `algoAsync` sound.)
    if (state.busy) {
      throw new LenkeError(
        'lenke: graph is busy running an async algorithm — await it before the next call',
        { code: ErrorCode.InvalidGraphOp },
      );
    }

    return handle;
  };

  const free = (): void => {
    if (state.freed) {
      return;
    }

    // Release FIRST, mark after: if graphFree throws, the state stays live so
    // a retry (or the GC backstop) can still reclaim the graph.
    backend.graphFree(handle);
    state.freed = true;
    disposal.get(backend)?.delete(handle);
    reclaim?.unregister(token);
  };

  // Run an algorithm off the JS thread (Promise-returning). `live()` first rejects a
  // freed/busy graph, then the busy flag is set so no other native call can touch the
  // graph while the off-thread run reads it (the soundness guard for the native
  // `algoAsync`); it clears when the promise settles. Backends without a real
  // threadpool (bun:ffi, wasm) have no `algoAsync`, so this falls back to the
  // synchronous `algo` — the API stays a Promise, but that run does block.
  const runAlgoAsync = async (name: string, config?: AlgorithmConfig): Promise<Row[]> => {
    const handleForRun = live();
    const cfg = config && JSON.stringify(config);
    state.busy = true;

    try {
      const bytes = backend.algoAsync
        ? await backend.algoAsync(handleForRun, name, cfg)
        : backend.algo(handleForRun, name, cfg);

      return decodeRows(bytes);
    } finally {
      state.busy = false;
    }
  };

  const graph: RustGraph = {
    get vertexCount() {
      return backend.vertexCount(live());
    },
    get edgeCount() {
      return backend.edgeCount(live());
    },
    get version() {
      return backend.version(live());
    },
    epoch: (name) => backend.epoch(live(), name),
    createIndex: (spec) => backend.createIndex(live(), spec.on, spec.kind, spec.keys),
    createUniqueConstraint: (label, key) => backend.createUniqueConstraint(live(), label, key),
    createRequiredConstraint: (label, key) => backend.createRequiredConstraint(live(), label, key),
    createTypeConstraint: (label, key, type) =>
      backend.createTypeConstraint(live(), label, key, type),
    createEdgeUniqueConstraint: (edgeType, key) =>
      backend.createEdgeUniqueConstraint(live(), edgeType, key),
    createEdgeRequiredConstraint: (edgeType, key) =>
      backend.createEdgeRequiredConstraint(live(), edgeType, key),
    createEdgeTypeConstraint: (edgeType, key, type) =>
      backend.createEdgeTypeConstraint(live(), edgeType, key, type),
    createCardinalityConstraint: (label, edgeType, direction, min, max) =>
      backend.createCardinalityConstraint(live(), label, edgeType, direction, min, max),
    createValidator: (label, varName, predicate) =>
      backend.createValidator(live(), label, varName, predicate),
    createInvariant: (name, query) => backend.createInvariant(live(), name, query),
    dropVertexIndex: (key) => backend.dropVertexIndex(live(), key),
    dropEdgeIndex: (key) => backend.dropEdgeIndex(live(), key),
    vertexIndexes: () => backend.vertexIndexes(live()),
    lastWriteScope: (key) => backend.lastWriteScope(live(), key),
    edgeIndexes: () => backend.edgeIndexes(live()),
    dumpSchema: () => backend.dumpSchema(live()),
    beginTransaction: () => backend.beginTransaction(live()),
    commitTransaction: () => backend.commitTransaction(live()),
    rollbackTransaction: () => backend.rollbackTransaction(live()),
    tx: () => {
      backend.beginTransaction(live());

      return {
        commit: () => backend.commitTransaction(live()),
        rollback: () => backend.rollbackTransaction(live()),
      };
    },
    transaction: <T>(fn: (graph: RustGraph) => T): T => {
      backend.beginTransaction(live());

      let result: T;

      try {
        result = fn(graph);
      } catch (error) {
        backend.rollbackTransaction(live());

        throw error;
      }

      backend.commitTransaction(live()); // may throw ConstraintViolation after rolling back

      return result;
    },
    setClock: (c) => {
      // The clock is a callback, not a value — it never crosses the native
      // boundary; the host resolves it once per query into `$__now`. It is also
      // the one setting that stays runtime-settable: a host dependency rather
      // than a resource bound, and clearing it with `null` is part of its
      // contract. Every other setting is fixed at construction.
      clock = c;

      return graph;
    },
    query: <R extends Row = Row>(q: string | TemplateStringsArray, ...subs: unknown[]): R[] => {
      const { text, params } = compileGql(q, subs, clock ?? undefined);

      return decodeRows(backend.queryRows(live(), text, params)) as R[];
    },
    queryArrow: (q: string | TemplateStringsArray, ...subs: unknown[]) => {
      const { text, params } = compileGql(q, subs, clock ?? undefined);

      return backend.queryArrow(live(), text, params);
    },
    queryArrowIpc: (q: string | TemplateStringsArray, ...rest: unknown[]) => {
      // String form: (text, { params?, format? }). Tagged-template form: stream only.
      if (typeof q === 'string') {
        const opts = (rest[0] ?? {}) as { params?: QueryParams; format?: 'stream' | 'file' };
        const { text, params } = compileGql(
          q,
          opts.params === undefined ? [] : [opts.params],
          clock ?? undefined,
        );

        return backend.queryArrowIpc(live(), text, opts.format === 'file', params);
      }

      const { text, params } = compileGql(q, rest, clock ?? undefined);

      return backend.queryArrowIpc(live(), text, false, params);
    },
    // `gremlin(...)` here is the module-level composer (safe escaping), not this
    // property — object keys don't bind in scope.
    gremlin: (q, ...subs) =>
      parseJson(backend.gremlinJson(live(), composeGremlin(q, ...subs)), 'gremlin') as unknown[],
    degree: (config) => runAlgoAsync('degree', config) as Promise<DegreeRow[]>,
    connectedComponents: (config) =>
      runAlgoAsync('connectedComponents', config) as Promise<ComponentRow[]>,
    stronglyConnectedComponents: (config) =>
      runAlgoAsync('stronglyConnectedComponents', config) as Promise<ComponentRow[]>,
    onCycle: (config) => runAlgoAsync('onCycle', config) as Promise<OnCycleRow[]>,
    labelPropagation: (config) => runAlgoAsync('labelPropagation', config) as Promise<LabelRow[]>,
    peerPressure: (config) => runAlgoAsync('peerPressure', config) as Promise<ClusterRow[]>,
    pagerank: (config) => runAlgoAsync('pagerank', config) as Promise<PageRankRow[]>,
    personalizedPagerank: (config) =>
      runAlgoAsync('personalizedPagerank', config) as Promise<PageRankRow[]>,
    betweenness: (config) => runAlgoAsync('betweenness', config) as Promise<CentralityRow[]>,
    closeness: (config) => runAlgoAsync('closeness', config) as Promise<CentralityRow[]>,
    shortestPath: (config) => runAlgoAsync('shortestPath', config) as Promise<ShortestPathRow[]>,
    neighborAggregate: (config) =>
      runAlgoAsync('neighborAggregate', config) as Promise<NeighborAggregateRow[]>,
    copy: () => attachGraph(backend, backend.graphClone(live())),
    toNdjson: () => backend.encodeNdjson(live()),
    mergeNdjson: (bytes) => backend.mergeNdjson(live(), bytes),
    serialize: (format) => decoder.decode(backend.serialize(live(), format)),
    prepare: <R extends Row = Row>(text: string, opts?: { maxOperatorChain?: number }) =>
      makePrepared<R>(backend, live, text, opts?.maxOperatorChain),
    free,
    [Symbol.dispose]: free,
  };

  reclaim?.register(graph, reclaimThunk(backend, handle, state), token);

  return graph;
};

/**
 * Decode NDJSON bytes into a graph and return a {@link RustGraph} facade.
 * Empty input yields an empty graph (a cold boot), not an FFI fault — the
 * boundary can't take a zero-length buffer, so it crosses as one newline,
 * which the decoder treats as zero elements.
 */
/**
 * Apply construction-time settings to a freshly created handle. Settings are
 * constructor-only: they are host policy, fixed for the graph's life, so the
 * id-keyed backend setter stays INTERNAL rather than becoming a public mutator.
 * `maxOperatorChain` is the shorthand for `limits.operatorChain`.
 */
const applyGraphConfig = (
  backend: Backend,
  handle: GraphHandle,
  opts: GraphConfigOptions,
): void => {
  const limits: Partial<GraphLimits> = {
    ...(opts.maxOperatorChain === undefined ? {} : { operatorChain: opts.maxOperatorChain }),
    ...opts.limits,
  };

  for (const [name, value] of Object.entries(limits) as [keyof GraphLimits, number][]) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new LenkeError(`limits.${name} must be a positive integer, got ${value}`, {
        code: ErrorCode.InvalidValue,
      });
    }

    if (!backend.setConfig(handle, CONFIG_IDS[`limits.${name}`], value)) {
      throw new LenkeError(
        `the loaded lenke artifact does not support the \`limits.${name}\` setting; rebuild the native crate`,
        { code: ErrorCode.InvalidValue },
      );
    }
  }

  // Opt-in multicore for the graph algorithms. `1` is the serial default (the store's
  // default too), so only a value > 1 crosses the boundary — this keeps the common
  // case working on an older artifact that predates the `parallelism` config id.
  if (opts.parallelism !== undefined) {
    if (!Number.isInteger(opts.parallelism) || opts.parallelism < 1) {
      throw new LenkeError(`parallelism must be a positive integer, got ${opts.parallelism}`, {
        code: ErrorCode.InvalidValue,
      });
    }

    if (
      opts.parallelism > 1 &&
      !backend.setConfig(handle, PARALLELISM_CONFIG_ID, opts.parallelism)
    ) {
      throw new LenkeError(
        'the loaded lenke artifact does not support the `parallelism` setting; rebuild the native crate',
        { code: ErrorCode.InvalidValue },
      );
    }
  }
};

export const graphFromNdjson = (
  backend: Backend,
  input: string | Uint8Array,
  // `parallel` is a legacy no-op (kept so old call sites still type-check); use
  // `parallelism` — it both parallelizes THIS decode's parse AND becomes the graph's
  // default for later algorithm/encode runs.
  opts: GraphConfigOptions & { parallel?: boolean } = {},
): RustGraph => {
  // Accepts the text form as well as bytes: NDJSON is a text format, and needing
  // to hand-encode it was why most callers reached for `graphFromFormat(…,
  // 'ndjson')` instead of this, the dedicated entry point.
  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;
  // Parallelize the parse when the caller opted into multicore (a valid, >1 value);
  // applyGraphConfig below validates `parallelism` fully and persists it on the store.
  const threads =
    typeof opts.parallelism === 'number' &&
    Number.isInteger(opts.parallelism) &&
    opts.parallelism > 1
      ? opts.parallelism
      : 1;
  const handle = backend.graphFromNdjson(
    bytes.byteLength === 0 ? new TextEncoder().encode('\n') : bytes,
    threads,
  );

  applyGraphConfig(backend, handle, opts);

  return attachGraph(backend, handle);
};

/**
 * A fresh, empty {@link RustGraph} to `INSERT` / `mergeNdjson` into — the
 * self-documenting cold boot. (Equivalent to `graphFromNdjson(backend, <empty>)`
 * without the encode-an-empty-buffer incantation.) Pass `{ maxOperatorChain }` to
 * override the GQL operator-chain ceiling (default 10_000).
 */
export const createEmptyGraph = (backend: Backend, opts: GraphConfigOptions = {}): RustGraph =>
  graphFromNdjson(backend, new Uint8Array(0), opts);

/**
 * Deserialize a document into a {@link RustGraph}. Accepts a string or raw bytes.
 *
 * The format and any settings travel together in one options object, so the call
 * reads as `graphFromFormat(backend, doc, { format: 'graphson' })` rather than as
 * a run of positional arguments whose meanings you have to remember. For NDJSON
 * use {@link graphFromNdjson}, which needs no format at all.
 */
export const graphFromFormat = (
  backend: Backend,
  input: string | Uint8Array,
  options: GraphConfigOptions & { format: GraphFormat },
): RustGraph => {
  const { format, ...opts } = options;
  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;
  const handle = backend.deserialize(bytes, format);

  applyGraphConfig(backend, handle, opts);

  return attachGraph(backend, handle);
};
