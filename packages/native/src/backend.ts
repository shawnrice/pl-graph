/**
 * The backend contract. Defined in terms of *logical operations* — never raw
 * pointers — so the facade is identical across environments. Each backend hides
 * its own memory marshalling:
 *   - the FFI backend hands the crate a pointer to a JS-owned buffer and reads
 *     results in place (then frees the crate-owned result buffer);
 *   - the wasm backend copies bytes into linear memory via `lnk_alloc`, calls,
 *     then copies results back out.
 *
 * A graph handle is opaque: a native pointer or a wasm linear-memory offset,
 * both representable as a JS `number`. Treat it as a token, never arithmetic.
 */
// Type-only import — a `SchemaOp` is the shape `dumpSchema` returns. The reverse
// edge (graph.ts imports `Backend`) makes this a type-level cycle, which is erased
// at compile and so carries no runtime dependency.
import type { SchemaOp } from './graph.js';

export type GraphHandle = number;

/** An opaque handle to a compiled, reusable GQL query (see {@link Backend.prepare}). */
export type PreparedHandle = number;

/** Which element an index covers (see {@link Backend.createIndex}). */
export type IndexTarget = 'vertex' | 'edge';

/**
 * The kind of secondary index (see {@link Backend.createIndex}): `'hash'` is an
 * equality seek over a single key; `'interval'` is an RI-tree over an edge
 * `[loKey, hiKey)` temporal pair for as-of / overlap seeks (edge-only).
 */
export type IndexKind = 'hash' | 'interval';

/**
 * What a {@link Backend.mergeNdjson} applied vs. skipped — so a caller sees
 * anything that didn't land cleanly. Empty `*Skipped`/`phantomVertices` arrays
 * mean a fully clean merge.
 */
export type MergeReport = {
  /** Vertices actually inserted. */
  nodesAdded: number;
  /** Edges actually inserted. */
  edgesAdded: number;
  /** Batch node ids skipped because the id already existed (first-wins). */
  nodesSkipped: string[];
  /** Batch edge ids dropped because that explicit id already existed. */
  edgesSkipped: string[];
  /** Ids referenced as an edge endpoint but never declared as a node — created as bare vertices. */
  phantomVertices: string[];
};

export type Backend = {
  /** Value of `lnk_abi_version()` for the loaded artifact. */
  readonly abiVersion: number;

  /** Decode NDJSON bytes into a graph; returns an owning handle. */
  graphFromNdjson: (bytes: Uint8Array, parallel: boolean) => GraphHandle;
  /**
   * Bulk-append NDJSON bytes into an existing graph — a `COPY FROM` for a live
   * store. Ingests at bulk speed (no per-`INSERT` parse); a node whose id
   * already exists is first-wins-skipped. Returns a {@link MergeReport} of what
   * applied vs. skipped. Throws a coded error on a parse fault.
   */
  mergeNdjson: (handle: GraphHandle, bytes: Uint8Array) => MergeReport;
  /**
   * Deep-copy a graph into a fresh, fully independent handle — the fast
   * fork/branch substrate. An O(V+E) clone of the columnar store, not a
   * serialize→parse round-trip: no text, no re-validation, no re-indexing, and
   * element ids are preserved exactly. The returned handle owns its own memory and
   * must be `graphFree`d.
   */
  graphClone: (handle: GraphHandle) => GraphHandle;
  /** Release a handle from `graphFromNdjson`. */
  graphFree: (handle: GraphHandle) => void;

  vertexCount: (handle: GraphHandle) => number;
  edgeCount: (handle: GraphHandle) => number;

  /** Monotonic mutation counter — O(1) change signal for reactive snapshots. */
  version: (handle: GraphHandle) => number;
  /** Per-token change epoch (label / edge-type / property-key) for finer invalidation. */
  epoch: (handle: GraphHandle, name: string) => number;

  /**
   * Declare an opt-in secondary index (backfills existing elements, then stays
   * current; idempotent). The single parametric index creator:
   *   - `on`: `'vertex'` or `'edge'` — which element the index covers.
   *   - `kind`: `'hash'` (equality seek, one key — turns `WHERE x.k = …` into a
   *     seek) or `'interval'` (RI-tree over an edge `[k0, k1)` temporal pair — an
   *     as-of `k0 <= v AND k1 > v` / overlap predicate seeds from it; two of them,
   *     valid `[vf,vt)` + transaction `[tf,tt)`, cover a bitemporal as-of).
   *   - `keys`: `[k]` for a hash index, `[loKey, hiKey]` for an interval index.
   * An interval index is edge-only; a hash index takes exactly one key.
   */
  createIndex: (handle: GraphHandle, on: IndexTarget, kind: IndexKind, keys: string[]) => void;

  /**
   * Set the GQL operator-chain ceiling for this graph (the `maxOperatorChain`
   * option); the parser rejects a longer `a AND b AND …` / `x + y + …` chain with
   * `E_SYNTAX`. Anti-resource-abuse only — the n-ary AST never overflows the stack.
   */
  setMaxOperatorChain: (handle: GraphHandle, n: number) => void;
  /**
   * Declare a UNIQUE constraint on `(label, key)`. Throws `ConstraintViolation`
   * if the current data already violates it. See docs/design/gql-extensions.md §3.
   */
  createUniqueConstraint: (handle: GraphHandle, label: string, key: string) => void;
  createRequiredConstraint: (handle: GraphHandle, label: string, key: string) => void;
  createTypeConstraint: (handle: GraphHandle, label: string, key: string, type: string) => void;
  /**
   * Declare a UNIQUE / REQUIRED / TYPE constraint on `(edgeType, key)` — the edge
   * analogue of the vertex constraints above, keyed by edge type. Throws
   * `ConstraintViolation` (or `InvalidValue` for an unknown type name) as the
   * vertex forms do.
   */
  createEdgeUniqueConstraint: (handle: GraphHandle, edgeType: string, key: string) => void;
  createEdgeRequiredConstraint: (handle: GraphHandle, edgeType: string, key: string) => void;
  createEdgeTypeConstraint: (
    handle: GraphHandle,
    edgeType: string,
    key: string,
    type: string,
  ) => void;
  /**
   * Declare a CARDINALITY constraint bounding the degree of every vertex carrying
   * `label` over `edgeType` in `direction` to `min..=max` (`max: null`
   * unbounded). Throws `ConstraintViolation` if the current data already violates
   * it. The degree-bound member of R-CONSTRAINTS (see docs/design/r-tx.md).
   */
  createCardinalityConstraint: (
    handle: GraphHandle,
    label: string,
    edgeType: string,
    direction: 'out' | 'in',
    min: number,
    max: number | null,
  ) => void;
  /**
   * Declare a custom VALIDATOR on `label` (a vertex label OR an edge type): every
   * element carrying the label must satisfy the GQL boolean `predicate` (pure ISO
   * WHERE-clause syntax), with the element bound to `varName`. SQL-`CHECK`
   * semantics — rejected only on a definite `false`; a null/unknown result passes.
   * Throws `ConstraintViolation` if existing data already violates the predicate,
   * or `Syntax` (`E_SYNTAX`) if the predicate can't be parsed. The native
   * counterpart of `@lenke/gql`'s `createValidator` — same `(label, varName,
   * predicate)`, enforced byte-identically in the Rust GQL evaluator.
   */
  createValidator: (handle: GraphHandle, label: string, varName: string, predicate: string) => void;
  /**
   * Declare a graph-level INVARIANT `name` = a whole-graph GQL `query` (`MATCH …
   * RETURN`) that must hold after every write transaction. `false`-only-fails:
   * VIOLATED iff any cell in the result set is boolean `false` (`true`/`null`/
   * non-boolean/empty all hold). Throws `ConstraintViolation` if existing data
   * already violates it, or `Syntax` (`E_SYNTAX`) if the query can't be parsed.
   * The native counterpart of `@lenke/gql`'s `createInvariant` — same `(name,
   * query)`, enforced byte-identically in the Rust GQL evaluator.
   */
  createInvariant: (handle: GraphHandle, name: string, query: string) => void;
  /** Drop a vertex / edge property index (no-op if absent). */
  dropVertexIndex: (handle: GraphHandle, key: string) => void;
  dropEdgeIndex: (handle: GraphHandle, key: string) => void;

  /**
   * Transaction primitives (R-TX). `beginTransaction` opens a frame (writes still
   * apply eagerly, but record undo ops); nesting joins the outer frame.
   * `commitTransaction` closes it — the outermost commit runs the deferred
   * constraint checks and **throws `ConstraintViolation`** (after rolling back) if
   * one fails. `rollbackTransaction` reverses every staged write. See
   * packages/core/src/core/Graph.ts.
   */
  beginTransaction: (handle: GraphHandle) => void;
  commitTransaction: (handle: GraphHandle) => void;
  rollbackTransaction: (handle: GraphHandle) => void;
  /** The currently-indexed vertex / edge property keys (sorted). */
  vertexIndexes: (handle: GraphHandle) => string[];
  edgeIndexes: (handle: GraphHandle) => string[];
  /** The full active schema as replayable {@link SchemaOp}s (constraints, validators,
   * invariants, indexes), deterministic order — the read side of the `create*`
   * declarations, for snapshot persistence + CDC replication. */
  dumpSchema: (handle: GraphHandle) => SchemaOp[];
  /** The distinct values of property `key` across the last committed write's touched
   * vertices — that write's content-derived CDC value-scope. */
  lastWriteScope: (handle: GraphHandle, key: string) => string[];

  /**
   * Run a GQL query; returns the `{columns, rows}` JSON document as bytes.
   * `params` is an optional pre-serialized flat JSON object of `$name`
   * bindings — values bind to already-parsed param slots at execute time and
   * never touch the GQL parser (the injection-safety contract).
   */
  queryRows: (handle: GraphHandle, query: string, params?: string) => Uint8Array;
  /** Run a GQL query; returns the Arrow ("ARW1") columnar blob bytes. Same optional `params`. */
  queryArrow: (handle: GraphHandle, query: string, params?: string) => Uint8Array;
  /** Run a GQL query and return standard Apache Arrow IPC bytes, framed natively.
   * `file` selects the file / Feather-v2 layout, else the IPC stream layout. */
  queryArrowIpc: (handle: GraphHandle, query: string, file: boolean, params?: string) => Uint8Array;
  /** Run a textual Gremlin query; returns the JSON-array result bytes. */
  gremlinJson: (handle: GraphHandle, query: string) => Uint8Array;

  /**
   * Run a native graph algorithm (`degree`, `pagerank`, `connectedComponents`,
   * `labelPropagation`, `shortestPath`) over the whole graph in one call; returns
   * the `{columns, rows}` JSON document bytes. `config` is the algorithm's
   * pre-serialized JSON config object (omitted = defaults); a `writeProperty`
   * config mutates the graph.
   */
  algo: (handle: GraphHandle, name: string, config?: string) => Uint8Array;

  /**
   * Non-blocking {@link Backend.algo}: runs the algorithm off the JS thread and
   * resolves the same `{columns, rows}` bytes. Present only on backends that have a
   * real threadpool (the Node/napi backend); absent on bun:ffi and wasm, where the
   * facade falls back to the synchronous {@link Backend.algo}. While the promise is
   * pending the graph must not be touched by another call (the facade guards this).
   */
  algoAsync?: (handle: GraphHandle, name: string, config?: string) => Promise<Uint8Array>;

  /** Serialize the whole graph back to NDJSON bytes. */
  encodeNdjson: (handle: GraphHandle) => Uint8Array;

  /** Serialize the graph in a named format (`pg-json | pg-text | graphson | csv | ndjson`). */
  serialize: (handle: GraphHandle, format: string) => Uint8Array;
  /** Deserialize bytes in a named format into a new graph handle. */
  deserialize: (input: Uint8Array, format: string) => GraphHandle;

  /**
   * Compile a GQL query into a reusable prepared statement (lex/parse/lower
   * once). Graph-independent; execute it against any graph with fresh params via
   * {@link Backend.preparedQueryRows}. Throws a coded error on a syntax error.
   * `maxOperatorChain` is the anti-resource-abuse operator-chain ceiling applied
   * while parsing (default 10_000 when omitted).
   */
  prepare: (text: string, maxOperatorChain?: number) => PreparedHandle;
  /** Release a handle from {@link Backend.prepare}. */
  preparedFree: (prepared: PreparedHandle) => void;
  /** Execute a prepared statement against `graph` → the `{columns, rows}` JSON bytes. */
  preparedQueryRows: (prepared: PreparedHandle, graph: GraphHandle, params?: string) => Uint8Array;
  /** Execute a prepared statement against `graph` → the Arrow ("ARW1") blob bytes. */
  preparedQueryArrow: (prepared: PreparedHandle, graph: GraphHandle, params?: string) => Uint8Array;
};
