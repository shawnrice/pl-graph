import { Emitter, EmitterEvent } from '@lenke/emitter';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Clock } from '../temporal.js';
import { Edge } from './Edge.js';
import type { GraphEvent, GraphEvents, GraphEventType } from './GraphEvents.js';
import { isRecord } from './LenkeRecord.js';
import { PropertyIndex, type RangeBound } from './PropertyIndex.js';
import { validateElementNames, validateLabel } from './validate.js';
import { Vertex } from './Vertex.js';

type AddVertexParams = {
  id?: string;
  labels: string[];
  // Optional: a plain labeled vertex needs no properties (defaults to `{}`).
  properties?: Record<string, unknown>;
};

type AddEdgeArgs = {
  id?: string;
  from: Vertex;
  to: Vertex;
  labels: string[];
  // Optional: a plain labeled edge needs no properties (defaults to `{}`).
  properties?: Record<string, unknown>;
};

/** Which element an index covers (see {@link Graph.createIndex}). */
export type IndexTarget = 'vertex' | 'edge';

/**
 * The kind of secondary index (see {@link Graph.createIndex}): `'hash'` is an
 * equality seek over a single key; `'interval'` is an RI-tree over an edge
 * `[loKey, hiKey)` temporal pair — native-engine-only (throws in the pure-TS
 * engine).
 */
export type IndexKind = 'hash' | 'interval';

/**
 * The spec passed to {@link Graph.createIndex}: which element (`on`), the index
 * `kind`, and the property `keys` (`[k]` for a hash index, `[loKey, hiKey]` for
 * an interval index).
 */
export type IndexSpec = {
  on: IndexTarget;
  kind: IndexKind;
  keys: string[];
};

export type GraphOptions = {
  /**
   * Invoked when a graph-event listener or a `subscribe()` callback throws.
   * Isolation: one failing listener can neither stop the others nor break a
   * deferred `notify()` (which runs from an idle/timeout callback, where an
   * escaping throw would be an unhandled error with no context). Defaults to
   * re-throwing on a microtask — visible to the host's unhandled-error handling
   * without interrupting dispatch. Mirrors the Emitter's `onError`.
   */
  onError?: (error: unknown) => void;

  /**
   * The maximum length of a single left-associative operator chain (`a AND b AND
   * …`, `x + y + …`) a GQL query may contain before the parser rejects it with
   * `E_SYNTAX`. The associative operator AST is n-ary, so a long chain never
   * overflows the stack regardless of this value — it is a pure anti-resource-
   * abuse ceiling (each operand is an allocation + an eval step). Defaults to
   * 10_000, far beyond any hand- or machine-generated predicate; raise it only if
   * a legitimate generated query needs longer chains. Mirrors the native engine's
   * `createEmptyGraph(backend, { maxOperatorChain })`.
   */
  maxOperatorChain?: number;
};

/**
 * The scalar type a TYPE constraint (R-CONSTRAINTS) can require of a property
 * value — every stored non-null value maps to exactly one of these. `list` means
 * "an array" (element types are not constrained); the three temporals match the
 * `LocalDate`/`LocalDateTime`/`Duration` value classes.
 */
export type ScalarTypeName =
  | 'string'
  | 'number'
  | 'boolean'
  | 'date'
  | 'datetime'
  | 'duration'
  | 'list';

/** A declared property type for a constraint: a scalar, or a closed ISO
 *  `RECORD { field :: type, … }` (an exact field set, each field a `TypeSpec`,
 *  so records nest). Mirrors the native `TypeSpec`. */
export type TypeSpec =
  | { kind: 'scalar'; type: ScalarTypeName }
  // Each record field is `[name, type, notNull]`. A field is nullable/optional by
  // default (absent OR null both satisfy it); `NOT NULL` makes it required.
  | { kind: 'record'; fields: ReadonlyArray<readonly [string, TypeSpec, boolean]> }
  // The ISO OPEN record type (`ANY RECORD` / bare `RECORD`): any map, any shape.
  | { kind: 'anyRecord' };

const SCALARS = new Set<ScalarTypeName>([
  'string',
  'number',
  'boolean',
  'date',
  'datetime',
  'duration',
  'list',
]);

/** Lexicographic compare of two field-name strings (the canonical key sort). */
const cmpField = (a: string, b: string): number => {
  if (a < b) {
    return -1;
  }

  return a > b ? 1 : 0;
};

/** Parse a constraint type name: a scalar, or `record { field :: type, … }`
 *  (`:`/`::`, whitespace-tolerant, nesting allowed). `null` if malformed.
 *  Mirrors the native `TypeSpec::parse`. */
/**
 * Parse a constraint type name, also capturing an optional TOP-LEVEL `NOT NULL`
 * (`string NOT NULL`) as `notNull`. A top-level `NOT NULL` makes the property
 * REQUIRED (present + non-null) — the type-surface spelling of a required
 * constraint, mirroring per-field `NOT NULL` inside a record.
 */
export const parseTypeSpecWithNotNull = (
  input: string,
): { spec: TypeSpec; notNull: boolean } | null => {
  let i = 0;
  const s = input;
  const ws = (): void => {
    while (i < s.length && /\s/.test(s[i])) {
      i++;
    }
  };
  const ident = (): string | null => {
    ws();
    const start = i;

    while (i < s.length && /[A-Za-z0-9_]/.test(s[i])) {
      i++;
    }

    return i > start ? s.slice(start, i) : null;
  };
  const eat = (c: string): boolean => {
    ws();

    if (i < s.length && s[i] === c) {
      i++;

      return true;
    }

    return false;
  };
  const eatKw = (word: string): boolean => {
    const save = i;
    const w = ident();

    if (w?.toLowerCase() === word) {
      return true;
    }

    i = save;

    return false;
  };
  const parseType = (): TypeSpec | null => {
    const word = ident();

    if (word === null) {
      return null;
    }

    // ISO `[ANY] RECORD [{…}]`: `any record` / bare `record` → the OPEN type.
    if (word.toLowerCase() === 'any') {
      return eatKw('record') ? { kind: 'anyRecord' } : null;
    }

    if (word.toLowerCase() === 'record') {
      ws();

      if (i >= s.length || s[i] !== '{') {
        return { kind: 'anyRecord' }; // bare `record` → open
      }

      if (!eat('{')) {
        return null;
      }

      const fields: Array<[string, TypeSpec, boolean]> = [];
      ws();

      if (!eat('}')) {
        for (;;) {
          const name = ident();

          if (name === null || !eat(':')) {
            return null;
          }

          eat(':'); // optional second colon
          const ty = parseType();

          if (ty === null) {
            return null;
          }

          // ISO `NOT NULL` → a required (present, non-null) field.
          let notNull = false;

          if (eatKw('not')) {
            if (!eatKw('null')) {
              return null;
            }

            notNull = true;
          }

          // Duplicate field → last wins; keep canonical sorted order.
          const at = fields.findIndex(([k]) => k === name);

          if (at >= 0) {
            fields[at] = [name, ty, notNull];
          } else {
            fields.push([name, ty, notNull]);
          }

          if (eat(',')) {
            continue;
          }

          if (eat('}')) {
            break;
          }

          return null;
        }
      }

      fields.sort((x, y) => cmpField(x[0], y[0]));

      return { kind: 'record', fields };
    }

    return SCALARS.has(word as ScalarTypeName)
      ? { kind: 'scalar', type: word as ScalarTypeName }
      : null;
  };

  const t = parseType();

  if (t === null) {
    return null;
  }

  // Optional TOP-LEVEL `NOT NULL`.
  let notNull = false;

  if (eatKw('not')) {
    if (!eatKw('null')) {
      return null;
    }

    notNull = true;
  }

  ws();

  return i === s.length ? { spec: t, notNull } : null;
};

/** Parse a constraint type name (strict — a trailing `NOT NULL` is rejected; use
 * {@link parseTypeSpecWithNotNull} to accept it). */
export const parseTypeSpec = (input: string): TypeSpec | null => {
  const parsed = parseTypeSpecWithNotNull(input);

  return parsed !== null && !parsed.notNull ? parsed.spec : null;
};

/** The canonical type name for a `TypeSpec` (round-trips through `parseTypeSpec`);
 *  used by the schema dump. */
export const typeSpecName = (spec: TypeSpec): string => {
  if (spec.kind === 'scalar') {
    return spec.type;
  }

  if (spec.kind === 'anyRecord') {
    return 'any record';
  }

  return `record{${spec.fields
    .map(([k, t, notNull]) => `${k}::${typeSpecName(t)}${notNull ? ' NOT NULL' : ''}`)
    .join(',')}}`;
};

/** Does a value satisfy a `TypeSpec`? A null is exempt (REQUIRED is separate); a
 *  scalar constraint exempts a non-scalar (a map — the record path governs those);
 *  a record must match the closed shape exactly. Mirrors native `value_matches`. */
export const valueMatches = (
  v: unknown,
  spec: TypeSpec,
  scalarTypeOf: (x: unknown) => ScalarTypeName | null,
): boolean => {
  if (v === null || v === undefined) {
    return true;
  }

  if (spec.kind === 'scalar') {
    const got = scalarTypeOf(v);

    return got === null || got === spec.type;
  }

  // The OPEN record type — any map value, any shape. A map is a `LenkeRecord` or a
  // bare `{…}` plain object (a class instance like a temporal/element is NOT a map),
  // mirroring both `normalizePropertyValue` and native `matches!(v, Value::Map)`.
  if (spec.kind === 'anyRecord') {
    return (
      isRecord(v) ||
      (typeof v === 'object' && v !== null && Object.getPrototypeOf(v) === Object.prototype)
    );
  }

  // Accept the value in any record-like form — the write path checks it BEFORE
  // canonicalizing to a `LenkeRecord`, so a bare `Map` or plain object must match
  // too. Sort entries to align with the (sorted) declared fields.
  let raw: Array<[string, unknown]>;

  if (isRecord(v) || v instanceof Map) {
    raw = [...(v as Map<string, unknown>)];
  } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
    raw = Object.entries(v);
  } else {
    return false;
  }

  const present = new Map(raw);
  const declared = new Set(spec.fields.map(([k]) => k));

  // Closed record: any value key that isn't a declared field is a violation.
  for (const k of present.keys()) {
    if (!declared.has(k)) {
      return false;
    }
  }

  // Each field is nullable/optional by default — absent OR null both satisfy it.
  // `NOT NULL` makes the field required (must be present AND non-null).
  return spec.fields.every(([k, ft, notNull]) => {
    if (!present.has(k)) {
      return !notNull;
    }

    const fv = present.get(k);

    if (fv === null || fv === undefined) {
      return !notNull;
    }

    return valueMatches(fv, ft, scalarTypeOf);
  });
};

/**
 * A declared CARDINALITY constraint (R-CONSTRAINTS): every vertex carrying
 * `label` must have `min <= degree <= max` over `edgeType` in `direction`
 * (`out` = the vertex is the edge source, `in` = the target). `max: null` is
 * unbounded. Returned by {@link Graph.cardinalityConstraints} for introspection.
 */
export type CardinalityConstraint = {
  label: string;
  edgeType: string;
  direction: 'out' | 'in';
  min: number;
  max: number | null;
};

/**
 * A declared VALIDATOR (custom constraint, R-CONSTRAINTS): every element carrying
 * `label` — vertex label OR edge type, one namespace — must satisfy the GQL
 * boolean predicate `src`, with the element bound to variable `varName`.
 * SQL-`CHECK` semantics: the element is rejected only when the predicate is a
 * *definite* `false`; a `null`/unknown result PASSES (an absent optional property
 * isn't a violation). The compiled predicate closure is internal — only
 * `@lenke/gql` (which can parse+compile a GQL expression) registers one, via
 * {@link Graph.registerValidator}. Returned by {@link Graph.validators}.
 */
export type ValidatorInfo = {
  label: string;
  varName: string;
  src: string;
};

/** A compiled validator predicate: the element (vertex or edge) → three-valued
 *  result (`true` / `false` / `null` = UNKNOWN). Rejects only on a definite `false`. */
type ValidatorFn = (element: Vertex | Edge) => boolean | null;

/** A registered validator: its bind variable, its GQL source, and the compiled predicate. */
type ValidatorEntry = { varName: string; src: string; fn: ValidatorFn };

/**
 * Introspection shape for a declared graph-level INVARIANT (a cross-write
 * assertion). The compiled query closure is internal — only `@lenke/gql` (which
 * can parse+compile a GQL query) registers one, via
 * {@link Graph.registerInvariant}. Returned by {@link Graph.invariants}.
 */
export type InvariantInfo = {
  name: string;
  src: string;
};

/**
 * A compiled graph-level INVARIANT: a whole-graph GQL query → its result rows.
 * The invariant is VIOLATED iff any returned cell is boolean `false`; everything
 * else (`true`, `null`, a non-boolean value, an empty result) HOLDS. Runs once,
 * against the fully-staged graph, at a write transaction's commit.
 */
type InvariantFn = (graph: Graph) => ReadonlyArray<Record<string, unknown>>;

/** A registered invariant: its GQL query source (for messaging/introspection)
 *  and the compiled query closure. */
type InvariantEntry = { src: string; fn: InvariantFn };

/**
 * Where a buffered transaction event stashes its reactive tokens, captured while
 * the element is still live (see {@link Graph.emit}); `markMutated` reads them at
 * commit-time dispatch instead of re-deriving from a possibly-evicted element.
 */
const TX_TOKENS = Symbol('lenke.txTokens');

/**
 * A Property-Label graph.
 *
 * Vertices and Edges are concrete (non-generic) classes. Property types are
 * `Record<string, unknown>` at this layer; cast at the application boundary
 * if you need typed access (`vertex.properties as Person`), or wrap the
 * graph with a schema-aware layer.
 */
/** Max wall-clock a {@link Graph.copy} run holds the thread before yielding (ms).
 *  Time-based, not element-count-based, so the loop is released at a bounded
 *  interval regardless of how heavy each element is. */
const COPY_YIELD_MS = 5;

/** Release the event loop so a large {@link Graph.copy} doesn't block it. A
 *  `setTimeout(0)` macrotask — the honest yield: it lets pending timers and I/O
 *  run before the copy resumes. NOT `await Promise.resolve()`, which is a
 *  microtask: a self-replenishing microtask loop never reaches the macrotask
 *  phase, so it looks async but starves timers/I/O entirely (measured: 0 turns for
 *  a concurrent timer, vs a fair ~1 per yield here). */
const yieldToLoop = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

/** Deep-copy a `label → Set<key>` constraint map (used by {@link Graph.clone} /
 *  {@link Graph.copy}). */
const copySetMap = (src: ReadonlyMap<string, Set<string>>, dst: Map<string, Set<string>>): void => {
  for (const [k, v] of src) {
    dst.set(k, new Set(v));
  }
};

/** Deep-copy a `label → Map<key, type>` constraint map. */
const copyTypeMap = (
  src: ReadonlyMap<string, Map<string, ScalarTypeName>>,
  dst: Map<string, Map<string, ScalarTypeName>>,
): void => {
  for (const [k, v] of src) {
    dst.set(k, new Map(v));
  }
};

export class Graph {
  verticesById: Map<string, Vertex>;
  verticesByLabel: Map<string, Set<Vertex>>;

  edgesById: Map<string, Edge>;
  edgesByLabel: Map<string, Set<Edge>>;
  edgesFromByLabel: Map<string, Map<string, Set<Edge>>>;
  edgesToByLabel: Map<string, Map<string, Set<Edge>>>;

  elementLabels: Map<string, Set<string>>;
  elementProperties: Map<string, Record<string, unknown>>;

  /** Opt-in secondary indexes over property values (see {@link PropertyIndex}). */
  vertexPropertyIndex: PropertyIndex<Vertex>;
  edgePropertyIndex: PropertyIndex<Edge>;

  private readonly listeners: Set<() => unknown>;

  // Single error sink for graph-event listeners and `subscribe()` callbacks —
  // see {@link GraphOptions.onError}.
  private readonly onError: (error: unknown) => void;

  // A `requestIdleCallback` handle (number) in the browser, or a `setTimeout`
  // handle in Node — the pending debounced subscriber notification, if any.
  private notifyHandle: ReturnType<typeof setTimeout> | number | undefined;

  // Reactive change tracking (mirrors the Rust core, for useSyncExternalStore):
  // `mutationVersion` is an O(1) "did anything change?" signal; `tokenEpochs`
  // are per-token (label / edge-type / property-key) counters for *selective*
  // invalidation — a selector that depends only on `name` recomputes only when
  // `epoch('name')` moved, not on every mutation. Both advance on the same
  // deferred, veto-checked step, so a vetoed mutation bumps nothing.
  private mutationVersion: number;
  private readonly tokenEpochs: Map<string, number>;

  emitter: Emitter<keyof GraphEvents, GraphEvents>;

  // R-TX transaction state. `txDepth > 0` means an open transaction: writes still
  // apply eagerly to the live store (so reads inside see their own writes), but
  // each mutation records an inverse op in `txUndo`, emitted events buffer in
  // `txEvents` (dispatched together on commit, discarded on rollback), and the
  // built-in constraint checks defer to commit — the touched vertex ids collect
  // in `txTouched`. `applyingUndo` is true only while a rollback replays inverse
  // ops, which must neither re-record undo nor re-run constraint checks.
  private txDepth = 0;
  private txUndo: Array<() => void> = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private txEvents: Array<EmitterEvent<any, any>> = [];
  private txTouched = new Set<string>();
  // Edge analogue of `txTouched`: edge ids whose built-in edge constraints must be
  // re-checked at commit (R-TX deferral for edge writes).
  private txTouchedEdges = new Set<string>();
  private applyingUndo = false;
  // Access mode of the active explicit transaction opened by ISO GQL
  // `START TRANSACTION READ ONLY` (see @lenke/gql's executor). Set true by that
  // statement, cleared on commit/rollback. Only the GQL statement executor reads
  // it — the core mutators are access-mode agnostic; a read-only write is rejected
  // one level up, at the statement boundary, before any mutation applies.
  private txReadOnly = false;

  // The host wall-clock this graph's GQL now-functions (`current_date`,
  // `current_timestamp`) read, wired via `setClock`. Null → those functions read
  // null: the engine never invents a clock. The clock lives here, not in the
  // engine, so the engine stays a pure function of (graph, params).
  private queryClock: Clock | null = null;

  // Anti-resource-abuse ceiling on operator-chain length (see GraphOptions).
  // Read by `@lenke/gql`'s query entries and passed to the parser. Default 10k.
  private maxChain = 10_000;

  /**
   * Wire (or clear, with `null`) the host {@link Clock} that supplies `$__now`
   * for the ISO now-functions. `@lenke/gql`'s `query`/`gql` call it once per
   * query — and only when that query didn't pass an explicit `$__now`, so an
   * explicit value still wins (deterministic for tests). The engine never reads
   * a clock itself; this is the host opting into wall time. Returns `this` for
   * chaining: `new Graph().setClock(() => LocalDateTime.fromJSDate(new Date()))`.
   */
  setClock(clock: Clock | null): this {
    this.queryClock = clock;

    return this;
  }

  /** The wired host {@link Clock}, or null. Read by the GQL query entries. */
  get clock(): Clock | null {
    return this.queryClock;
  }

  /** The operator-chain ceiling (see {@link GraphOptions.maxOperatorChain}). */
  get maxOperatorChain(): number {
    return this.maxChain;
  }

  constructor(options: GraphOptions = {}) {
    this.verticesById = new Map();
    this.edgesById = new Map();
    this.verticesByLabel = new Map();
    this.edgesFromByLabel = new Map();
    this.edgesToByLabel = new Map();
    this.edgesByLabel = new Map();

    this.elementLabels = new Map();
    this.elementProperties = new Map();

    this.vertexPropertyIndex = new PropertyIndex();
    this.edgePropertyIndex = new PropertyIndex();

    if (options.maxOperatorChain !== undefined) {
      this.maxChain = options.maxOperatorChain;
    }

    this.mutationVersion = 0;
    this.tokenEpochs = new Map();
    // Default: surface a listener error on a microtask — visible to the host's
    // unhandled-error handling, but it never breaks dispatch or a deferred notify.
    this.onError =
      options.onError ??
      ((error: unknown) => {
        queueMicrotask(() => {
          throw error;
        });
      });
    this.emitter = new Emitter({ enabled: true, onError: (error) => this.onError(error) });

    this.listeners = new Set();

    this.notifyHandle = undefined;

    // Coalesce subscriber notifications: many mutations in a tick collapse into
    // one deferred `notify()`. Idle-scheduled in the browser, a 1ms timer in Node.
    const scheduleNotify = () => {
      if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
        window.cancelIdleCallback(this.notifyHandle as number);
        this.notifyHandle = window.requestIdleCallback(this.notify);
      } else {
        globalThis.clearTimeout(this.notifyHandle);
        // Store the handle so the next mutation's clearTimeout above actually
        // cancels this one — otherwise every mutation leaks a fresh timer and
        // the debounce never coalesces.
        this.notifyHandle = globalThis.setTimeout(this.notify, 1);
      }
    };

    const markMutated = (event: GraphEvent) => {
      // Capture the touched tokens NOW, synchronously, while the element is
      // still attached — a removal nulls the element's graph ref before the
      // deferred step below runs, so reading labels/keys there would throw. A
      // buffered transaction event already captured them at buffer time (the
      // element is evicted by the time it dispatches at commit), so prefer those.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const tokens = (event as any)[TX_TOKENS] ?? this.tokensOf(event);
      const doTheWork = () => {
        // Advance the reactive counters (deferred so a burst of mutations
        // coalesces into one React notify). A mutation event always means a
        // committed write — events are observation-only, nothing vetoes.
        this.mutationVersion += 1;

        for (const token of tokens) {
          this.tokenEpochs.set(token, (this.tokenEpochs.get(token) ?? 0) + 1);
        }

        scheduleNotify();
      };

      queueMicrotask(doTheWork);
    };

    const graphMutationEvents: GraphEventType[] = [
      '@graph/VertexAdded',
      '@graph/VertexRemoved',
      '@graph/EdgeAdded',
      '@graph/EdgeRemoved',
      '@graph/LabelAddedToVertex',
      '@graph/LabelRemovedFromVertex',
      '@graph/LabelAddedToEdge',
      '@graph/LabelRemovedFromEdge',
      '@graph/VertexPropertyChanged',
      '@graph/VertexPropertiesChanged',
      '@graph/VertexPropertyRemoved',
      '@graph/VertexPropertiesRemoved',
      '@graph/EdgePropertyChanged',
      '@graph/EdgePropertiesChanged',
      '@graph/EdgePropertyRemoved',
      '@graph/EdgePropertiesRemoved',
    ];

    const onMutate = (event: GraphEventType) => {
      this.emit(new EmitterEvent('@graph/mutate', { original: event }));
    };

    graphMutationEvents.forEach((type) => {
      // The same listener is registered against many event types; we erase
      // the per-event listener type here.
      this.on(type, markMutated as never);
      this.on(type, onMutate as never);
    });
  }

  /**
   * The graph's vertices, as a re-iterable view over `verticesById` (insertion
   * order). Not a stored `Set` — membership/identity lives in `verticesById`,
   * so iterating here costs nothing extra and inserts pay no second structure.
   */
  get vertices(): Iterable<Vertex> {
    return { [Symbol.iterator]: () => this.verticesById.values() };
  }

  /** The graph's edges, as a re-iterable view over `edgesById` (insertion order). */
  get edges(): Iterable<Edge> {
    return { [Symbol.iterator]: () => this.edgesById.values() };
  }

  /**
   * Total element count — vertices **plus** edges — as a quick "how big is
   * this?" scalar. For the per-type counts use {@link vertexCount} /
   * {@link edgeCount}; for the per-label breakdown use {@link stats}.
   */
  get size(): number {
    return this.verticesById.size + this.edgesById.size;
  }

  get vertexCount(): number {
    return this.verticesById.size;
  }

  get stats(): Record<'vertices' | 'edges', Record<string, number>> {
    return {
      vertices: Object.fromEntries(
        Array.from(this.verticesByLabel.entries()).map(([key, value]) => [key, value.size]),
      ),
      edges: Object.fromEntries(
        Array.from(this.edgesByLabel.entries()).map(([key, value]) => [key, value.size]),
      ),
    };
  }

  get edgeCount(): number {
    return this.edgesById.size;
  }

  public subscribe = (callback: () => unknown): (() => void) => {
    this.listeners.add(callback);

    return () => {
      this.listeners.delete(callback);
    };
  };

  /**
   * Monotonic mutation counter — an O(1) "did anything change?" signal for
   * `useSyncExternalStore`-style snapshots. Advances on the same deferred,
   * veto-checked step as snapshot-staleness (so it reflects committed changes
   * by the time subscribers are notified).
   */
  get version(): number {
    return this.mutationVersion;
  }

  /**
   * The change epoch for a single token — a label, edge-type, or property-key
   * name (0 if never touched). A selector that depends only on specific tokens
   * can fingerprint `deps.map(epoch)` and recompute *only* when one of them
   * moved, instead of on every mutation.
   */
  public epoch = (name: string): number => this.tokenEpochs.get(name) ?? 0;

  /**
   * The tokens (label / edge-type / property-key names) a mutation event
   * touched — used to bump the right epochs. A topology change (add/remove an
   * element) touches that element's labels *and* property keys; a property
   * write touches just the key; a label change touches just the label. This
   * mirrors the Rust core's epoch rule, so `epoch('Person')` moves iff Person
   * membership changed and `epoch('age')` iff some age value changed.
   */
  private tokensOf(event: GraphEvent): string[] {
    const value = event.value as Record<string, any>;

    switch (event.type) {
      case '@graph/VertexAdded':
      case '@graph/VertexRemoved':
      case '@graph/EdgeAdded':
      case '@graph/EdgeRemoved': {
        const labels: string[] = value?.labels ? [...value.labels] : [];
        const keys: string[] = value?.properties ? Object.keys(value.properties) : [];

        return [...labels, ...keys];
      }
      case '@graph/LabelAddedToVertex':
      case '@graph/LabelRemovedFromVertex':
      case '@graph/LabelAddedToEdge':
      case '@graph/LabelRemovedFromEdge':
        return value?.label ? [value.label as string] : [];
      case '@graph/VertexPropertyChanged':
      case '@graph/EdgePropertyChanged':
      case '@graph/VertexPropertyRemoved':
      case '@graph/EdgePropertyRemoved':
        return value?.key ? [value.key as string] : [];
      case '@graph/VertexPropertiesChanged':
      case '@graph/EdgePropertiesChanged':
        return value?.next ? Object.keys(value.next) : [];
      case '@graph/VertexPropertiesRemoved':
      case '@graph/EdgePropertiesRemoved':
        return (value?.keys as string[]) ?? [];
      default:
        return [];
    }
  }

  /**
   * Clones the graph as well as all the vertices and edges
   */
  public clone = (): Graph => {
    const next = new Graph();
    next.disableEvents();

    // Recreate the declared (but empty) indexes so the backfill below populates
    // the clone's own buckets rather than aliasing the source's.
    for (const key of this.vertexPropertyIndex.indexedKeys()) {
      next.vertexPropertyIndex.createIndex(key);
    }

    for (const key of this.edgePropertyIndex.indexedKeys()) {
      next.edgePropertyIndex.createIndex(key);
    }

    // Deep copy: each vertex/edge is a *fresh* instance bound to `next`, with
    // its own labels and property bag. Nothing is shared with the source, so a
    // mutation on either graph can't reach into the other. Endpoints resolve to
    // the clone's own vertices by id.
    for (const vertex of this.verticesById.values()) {
      next.addVertex(
        new Vertex({
          id: vertex.id,
          labels: [...vertex.labels],
          properties: { ...vertex.properties },
          graph: next,
        }),
      );
    }

    for (const edge of this.edgesById.values()) {
      next.addEdge(
        new Edge({
          id: edge.id,
          from: next.getVertexById(edge.from.id)!,
          to: next.getVertexById(edge.to.id)!,
          labels: [...edge.labels],
          properties: { ...edge.properties },
          graph: next,
        }),
      );
    }

    this.#copyRegistriesInto(next);

    if (this.eventsEnabled()) {
      next.enableEvents();
    }

    return next;
  };

  /**
   * Deep-copy the constraint / validator / invariant registries into `next`.
   * Copied directly (not re-declared via `create*Constraint`, which would re-scan
   * the data) — the elements came from a valid graph, so the copy already
   * satisfies them. Shared by {@link clone} and {@link copy}.
   */
  #copyRegistriesInto = (next: Graph): void => {
    copySetMap(this.vertexUniqueConstraints, next.vertexUniqueConstraints);
    copySetMap(this.vertexRequiredConstraints, next.vertexRequiredConstraints);
    copyTypeMap(this.vertexTypeConstraints, next.vertexTypeConstraints);
    copySetMap(this.vertexTypeNotNull, next.vertexTypeNotNull);
    copySetMap(this.edgeUniqueConstraints, next.edgeUniqueConstraints);
    copySetMap(this.edgeRequiredConstraints, next.edgeRequiredConstraints);
    copyTypeMap(this.edgeTypeConstraints, next.edgeTypeConstraints);
    copySetMap(this.edgeTypeNotNull, next.edgeTypeNotNull);

    // Record-typed constraints: `TypeSpec` values are immutable, so a shallow
    // inner-map copy is a full copy.
    for (const [k, v] of this.vertexRecordConstraints) {
      next.vertexRecordConstraints.set(k, new Map(v));
    }

    for (const [k, v] of this.edgeRecordConstraints) {
      next.edgeRecordConstraints.set(k, new Map(v));
    }

    for (const [k, v] of this.vertexCardinalityConstraints) {
      next.vertexCardinalityConstraints.set(k, { ...v });
    }

    // Validator / invariant entries are immutable descriptors holding a compiled
    // fn — share the entries, copy the containers.
    for (const [k, entries] of this.validatorRegistry) {
      next.validatorRegistry.set(k, [...entries]);
    }

    for (const [k, entry] of this.invariantRegistry) {
      next.invariantRegistry.set(k, entry);
    }
  };

  /**
   * Async deep copy — the event-loop-friendly {@link clone}. Same faithful result
   * (fresh vertices/edges bound to the copy, own labels/property bags, indexes,
   * and all constraints), but it yields to the event loop whenever a run has held
   * the thread for {@link COPY_YIELD_MS}, so copying a large graph doesn't block it. The parity with
   * the native engine's `copy()`; async because it is pure-JS O(V+E) work, where
   * the native copy is an off-thread-fast columnar memcpy.
   */
  public copy = async (): Promise<Graph> => {
    const next = new Graph();
    next.disableEvents();

    for (const key of this.vertexPropertyIndex.indexedKeys()) {
      next.vertexPropertyIndex.createIndex(key);
    }

    for (const key of this.edgePropertyIndex.indexedKeys()) {
      next.edgePropertyIndex.createIndex(key);
    }

    let lastYield = performance.now();

    for (const vertex of this.verticesById.values()) {
      next.addVertex(
        new Vertex({
          id: vertex.id,
          labels: [...vertex.labels],
          properties: { ...vertex.properties },
          graph: next,
        }),
      );

      if (performance.now() - lastYield >= COPY_YIELD_MS) {
        // eslint-disable-next-line no-await-in-loop
        await yieldToLoop();
        lastYield = performance.now();
      }
    }

    for (const edge of this.edgesById.values()) {
      next.addEdge(
        new Edge({
          id: edge.id,
          from: next.getVertexById(edge.from.id)!,
          to: next.getVertexById(edge.to.id)!,
          labels: [...edge.labels],
          properties: { ...edge.properties },
          graph: next,
        }),
      );

      if (performance.now() - lastYield >= COPY_YIELD_MS) {
        // eslint-disable-next-line no-await-in-loop
        await yieldToLoop();
        lastYield = performance.now();
      }
    }

    this.#copyRegistriesInto(next);

    if (this.eventsEnabled()) {
      next.enableEvents();
    }

    return next;
  };

  public truncate = (): void => {
    // A wholesale reset can't be captured as a bounded undo-log (it would clone
    // the entire graph), so it's not reversible inside a transaction. Callers who
    // need it should truncate outside the transaction boundary.
    if (this.txDepth > 0) {
      throw new LenkeError('truncate() is not supported inside a transaction', {
        code: ErrorCode.InvalidGraphOp,
      });
    }

    // Emit a removal event for every element before clearing, so `truncate` — the
    // most destructive op — is visible to every listener/journal that already
    // handles `@graph/EdgeRemoved`/`@graph/VertexRemoved` (React sync, an audit
    // stream via `@graph/mutate`, the CDC WriteLog), with no special-casing.
    // Edges first, then vertices (a clean teardown order a journal can replay).
    // The elements are still live at emit time. Events are observation-only, so
    // these are notifications — nothing can veto the reset.
    const edges = [...this.edgesById.values()];
    const vertices = [...this.verticesById.values()];

    for (const edge of edges) {
      this.emit(new EmitterEvent('@graph/EdgeRemoved', edge));
    }

    for (const vertex of vertices) {
      this.emit(new EmitterEvent('@graph/VertexRemoved', vertex));
    }

    this.verticesById = new Map();
    this.edgesById = new Map();
    this.verticesByLabel = new Map();
    this.edgesFromByLabel = new Map();
    this.edgesToByLabel = new Map();
    this.edgesByLabel = new Map();
    this.elementLabels = new Map();
    this.elementProperties = new Map();

    // Keep declared indexes but drop their contents — `truncate` empties the
    // graph, not its schema.
    this.vertexPropertyIndex.clear();
    this.edgePropertyIndex.clear();
  };

  private readonly notify = (): void => {
    // Snapshot first: a subscriber that (un)subscribes from within its own
    // callback must not perturb the in-flight pass, and mutating the Set
    // mid-iteration is itself a hazard. Then isolate each subscriber — `notify`
    // runs from a deferred idle/timeout callback, so an escaping throw would skip
    // every later subscriber and surface as an unhandled error with no context.
    // Mirrors the Emitter's listener isolation.
    for (const listener of new Set(this.listeners)) {
      try {
        listener();
      } catch (error) {
        this.onError(error);
      }
    }
  };

  /**
   * The graph to read for a `useSyncExternalStore` snapshot. Reads are served
   * from the live graph itself — no copy is made. Referential stability across
   * renders is provided by the consumer's epoch/version fingerprinting (see the
   * React `useGraphSelector` gate), not by isolating a frozen copy here. For an
   * independent, mutable copy of the graph use {@link clone} instead.
   */
  public snapshot = (): Graph => this;

  /* Mutation methods */

  /**
   * Adds a Vertex to the Graph. Accepts either an existing Vertex or params
   * to construct a new one.
   */
  public addVertex = (params: AddVertexParams | Vertex): Vertex => {
    // Ingestion gate: reject a malformed label / property key before the Vertex
    // constructor writes it into the graph's element maps.
    validateElementNames(params.labels, params.properties ?? {});

    if (params.id && this.getVertexById(params.id)) {
      return this.getVertexById(params.id)!;
    }

    const vertex = Vertex.isVertex(params)
      ? params
      : new Vertex({ ...params, properties: params.properties ?? {}, graph: this });

    // Constraint gate: reject before emitting/committing, so a rejected write
    // leaves no trace and every write path (direct API, GQL, Gremlin, ingest)
    // is covered by one chokepoint. Inside a transaction the checks defer to
    // commit (record the touched vertex); a rollback replay skips them entirely.
    if (this.applyingUndo) {
      // rollback replay restores known-good state — never re-check.
    } else if (this.txDepth > 0) {
      this.txTouched.add(vertex.id);
    } else {
      const missing = this.missingRequired(vertex.labels, vertex.properties);

      if (missing) {
        throw new LenkeError(
          `missing required property '${missing.key}' for label '${missing.label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const badType = this.typeViolation(vertex.labels, vertex.properties);

      if (badType) {
        throw new LenkeError(
          `property '${badType.key}' must be ${badType.expected} on '${badType.label}', got ${badType.got}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const dup = this.uniqueConflict(vertex.labels, vertex.properties, vertex);

      if (dup) {
        throw new LenkeError(
          `unique constraint on '${dup.label}.${dup.key}' violated by value ${JSON.stringify(dup.existing.properties[dup.key])}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const invalid = this.validatorViolationOf(vertex);

      if (invalid) {
        throw new LenkeError(`validator '${invalid.src}' on '${invalid.label}' violated`, {
          code: ErrorCode.ConstraintViolation,
        });
      }
    }

    this.emit(new EmitterEvent('@graph/VertexAdded', vertex));

    this.verticesById.set(vertex.id, vertex);

    for (const label of vertex.labels) {
      this.indexVertexLabel(label, vertex);
    }

    this.vertexPropertyIndex.add(vertex, vertex.properties);

    this.recordUndo(() => this.removeVertex(vertex.id));

    return vertex;
  };

  /**
   * Removes a Vertex from the Graph. Also removes any edges the Vertex was
   * part of.
   */
  public removeVertex = (vertex: Vertex | string): Vertex | null => {
    if (typeof vertex === 'string') {
      const found = this.verticesById.get(vertex);

      if (!found) {
        return null;
      }

      return this.removeVertex(found);
    }

    // Idempotent: dropping an already-removed vertex is a no-op. Without this,
    // a second removal re-emits `VertexRemoved` for an evicted vertex whose
    // `labels` getter dereferences a now-null graph → TypeError. (The string
    // overload above already short-circuits; this guards the object overload.)
    if (!this.verticesById.has(vertex.id)) {
      return null;
    }

    // Capture the vertex's state while it's still live, so a rollback can
    // reconstruct it. Its incident edges reconstruct via each `removeEdge`'s own
    // inverse below (recorded before this vertex's re-add, so on reverse replay
    // the vertex comes back first, then its edges).
    const undoId = vertex.id;
    const undoLabels = [...vertex.labels];
    const undoProps = { ...vertex.properties };

    this.emit(new EmitterEvent('@graph/VertexRemoved', vertex));

    for (const edge of this.incidentEdges(vertex.id)) {
      this.removeEdge(edge);
    }

    for (const label of vertex.labels) {
      this.deIndexVertexLabel(label, vertex);
    }

    // De-index properties while they're still readable (evict() severs the
    // vertex from the graph, after which `vertex.properties` reads empty).
    this.vertexPropertyIndex.remove(vertex, vertex.properties);

    this.verticesById.delete(vertex.id);

    vertex.evict();

    this.recordUndo(() =>
      this.addVertex({ id: undoId, labels: undoLabels, properties: undoProps }),
    );

    return vertex;
  };

  public addLabelToVertex = (label: string, vertex: Vertex): Vertex => {
    validateLabel(label);

    const hadLabel = (this.elementLabels.get(vertex.id) ?? new Set()).has(label);

    // Adding a label brings its required keys into force for this vertex. Inside
    // a transaction the check defers to commit (via the touched set).
    if (this.applyingUndo) {
      // rollback replay — skip the check.
    } else if (this.txDepth > 0) {
      this.txTouched.add(vertex.id);
    } else {
      for (const key of this.vertexRequiredConstraints.get(label) ?? []) {
        if (!this.isPresent(vertex.properties[key])) {
          throw new LenkeError(
            `cannot add label '${label}': it requires property '${key}', which is missing`,
            { code: ErrorCode.ConstraintViolation },
          );
        }
      }
    }

    this.emit(new EmitterEvent('@graph/LabelAddedToVertex', { label, vertex }));

    this.indexVertexLabel(label, vertex);
    const next = new Set(this.elementLabels.get(vertex.id) ?? []);
    next.add(label);
    this.elementLabels.set(vertex.id, next);

    if (!hadLabel) {
      this.recordUndo(() => this.removeLabelFromVertex(label, vertex));
    }

    return vertex;
  };

  public removeLabelFromVertex = (label: string, vertex: Vertex): Vertex => {
    const hadLabel = (this.elementLabels.get(vertex.id) ?? new Set()).has(label);

    this.emit(new EmitterEvent('@graph/LabelRemovedFromVertex', { label, vertex }));

    this.deIndexVertexLabel(label, vertex);
    const next = new Set(this.elementLabels.get(vertex.id) ?? []);
    next.delete(label);
    this.elementLabels.set(vertex.id, next);

    if (hadLabel) {
      this.recordUndo(() => this.addLabelToVertex(label, vertex));
    }

    return vertex;
  };

  /**
   * Adds an Edge to the Graph. Accepts either an existing Edge or params to
   * construct a new one.
   */
  public addEdge = (params: AddEdgeArgs | Edge): Edge => {
    // Same gate as addVertex — validate before the Edge constructor writes.
    validateElementNames(params.labels, params.properties ?? {});

    if (params.id && this.getEdgeById(params.id)) {
      return this.getEdgeById(params.id)!;
    }

    if (Edge.isEdge(params)) {
      this.assertValidEdge(params.from, params.to, params.labels.size);

      return this.insertEdge(params);
    }

    // A missing `from`/`to` must surface as a coded error, not a raw TypeError
    // from dereferencing `params.to.id` below.
    if (!params.from || !params.to) {
      throw new LenkeError('Cannot add an edge with missing endpoint vertices.', {
        code: ErrorCode.MissingVertex,
      });
    }

    // Validate the *params* before constructing: the `Edge` constructor eagerly
    // writes labels/properties into the graph's element maps via its setters, so
    // rejecting after construction would leave orphaned map entries behind.
    // `params.from`/`params.to` are vertices that must already live in this
    // graph — resolve them by id the same way the edge's getters will.
    this.assertValidEdge(
      this.getVertexById(params.from.id),
      this.getVertexById(params.to.id),
      params.labels.length,
    );

    return this.insertEdge(
      new Edge({ ...params, properties: params.properties ?? {}, graph: this }),
    );
  };

  /**
   * Reject an edge that can't be a valid LPG member: both endpoints must resolve
   * to vertices in this graph, and it must carry ≥1 label — the invariant the
   * removal cascade depends on (a label-less edge never lands in a label bucket,
   * so `incidentEdges`/`removeVertex` can't find it, leaving a dangling edge).
   */
  private assertValidEdge(from: Vertex | null, to: Vertex | null, labelCount: number): void {
    if (!from || !to) {
      throw new LenkeError('Cannot add an edge with missing endpoint vertices.', {
        code: ErrorCode.MissingVertex,
      });
    }

    if (labelCount === 0) {
      throw new LenkeError('Cannot add an edge with no labels: every edge must carry ≥1 label.', {
        code: ErrorCode.InvalidGraphOp,
      });
    }
  }

  /** Shared insertion tail: emit, then register the edge in the id and label indexes. */
  private readonly insertEdge = (edge: Edge): Edge => {
    // Constraint gate (mirror addVertex): reject before emitting/committing, so a
    // rejected edge write leaves no trace. Inside a transaction the checks defer
    // to commit (record the touched edge); a rollback replay skips them entirely.
    if (!this.deferEdgeConstraint(edge)) {
      const missing = this.edgeMissingRequired(edge.labels, edge.properties);

      if (missing) {
        throw new LenkeError(
          `missing required property '${missing.key}' for edge type '${missing.label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const badType = this.edgeTypeViolation(edge.labels, edge.properties);

      if (badType) {
        throw new LenkeError(
          `property '${badType.key}' must be ${badType.expected} on edge type '${badType.label}', got ${badType.got}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const dup = this.edgeUniqueConflict(edge.labels, edge.properties, edge);

      if (dup) {
        throw new LenkeError(
          `unique constraint on edge type '${dup.label}.${dup.key}' violated by value ${JSON.stringify(dup.existing.properties[dup.key])}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const invalid = this.validatorViolationOf(edge);

      if (invalid) {
        throw new LenkeError(`validator '${invalid.src}' on '${invalid.label}' violated`, {
          code: ErrorCode.ConstraintViolation,
        });
      }
    }

    // Cardinality gate: an edge write changes both endpoints' degree. Runs here,
    // before the edge lands in the adjacency indexes below (so the endpoints'
    // degree is still pre-insert). Outside a transaction it enforces MAX eagerly;
    // inside one it records the touched endpoints for the commit-time recheck.
    this.cardinalityGateOnInsert(edge);

    this.emit(new EmitterEvent('@graph/EdgeAdded', edge));

    this.edgesById.set(edge.id, edge);

    for (const label of edge.labels) {
      this.indexEdgeLabel(label, edge);
    }

    this.edgePropertyIndex.add(edge, edge.properties);

    this.recordUndo(() => this.removeEdge(edge));

    return edge;
  };

  public removeEdge = (edge: Edge): Edge => {
    // Capture the edge's shape while live, so a rollback can reconstruct it.
    const undoId = edge.id;
    const undoLabels = [...edge.labels];
    const undoProps = { ...edge.properties };
    const undoFrom = edge.from.id;
    const undoTo = edge.to.id;

    // Cardinality: removing an edge drops both endpoints' degree, which may fall
    // below `min` — record them for the commit-time recheck (min is never
    // enforced eagerly). No-op outside a transaction / during a rollback replay.
    this.cardinalityNoteOnRemove(edge);

    this.emit(new EmitterEvent('@graph/EdgeRemoved', edge));

    for (const label of edge.labels) {
      this.deIndexEdgeLabel(label, edge);
    }

    this.edgePropertyIndex.remove(edge, edge.properties);

    this.edgesById.delete(edge.id);

    edge.evict();

    this.recordUndo(() =>
      this.addEdge({
        id: undoId,
        from: this.getVertexById(undoFrom)!,
        to: this.getVertexById(undoTo)!,
        labels: undoLabels,
        properties: undoProps,
      }),
    );

    return edge;
  };

  public addLabelToEdge = (label: string, edge: Edge): Edge => {
    validateLabel(label);

    if (edge.labels.has(label)) {
      return edge;
    }

    // Adding an edge type brings its required keys into force for this edge.
    // Inside a transaction the check defers to commit (via the touched set).
    if (this.applyingUndo) {
      // rollback replay — skip the check.
    } else if (this.txDepth > 0) {
      this.txTouchedEdges.add(edge.id);
    } else {
      for (const key of this.edgeRequiredConstraints.get(label) ?? []) {
        if (!this.isPresent(edge.properties[key])) {
          throw new LenkeError(
            `cannot add edge type '${label}': it requires property '${key}', which is missing`,
            { code: ErrorCode.ConstraintViolation },
          );
        }
      }
    }

    this.emit(new EmitterEvent('@graph/LabelAddedToEdge', { label, edge }));

    const next = new Set(this.elementLabels.get(edge.id) ?? []);
    next.add(label);
    this.elementLabels.set(edge.id, next);

    this.indexEdgeLabel(label, edge);

    this.recordUndo(() => this.removeLabelFromEdge(label, edge));

    return edge;
  };

  public removeLabelFromEdge = (label: string, edge: Edge): Edge => {
    if (!edge.labels.has(label)) {
      return edge;
    }

    this.emit(new EmitterEvent('@graph/LabelRemovedFromEdge', { label, edge }));

    const next = new Set(this.elementLabels.get(edge.id) ?? []);
    next.delete(label);
    this.elementLabels.set(edge.id, next);

    this.deIndexEdgeLabel(label, edge);

    this.recordUndo(() => this.addLabelToEdge(label, edge));

    return edge;
  };

  /* Query methods */

  public getVertexById = (id: string): Vertex | null => {
    return this.verticesById.get(id) ?? null;
  };

  public getEdgeById = (id: string): Edge | null => {
    return this.edgesById.get(id) ?? null;
  };

  public getVerticesByLabel = (label: string): Set<Vertex> => {
    return new Set(this.verticesByLabel.get(label) ?? []);
  };

  public getEdgesByLabel = (label: string): Set<Edge> => {
    return new Set(this.edgesByLabel.get(label) ?? []);
  };

  /**
   * The first live edge of `label` from `from` to `to`, if any — the structural
   * key `_MERGE`'s edge form upserts on (ensures at most one such edge).
   * First-by-insertion-order, matching the Rust core.
   */
  public findEdge = (from: Vertex, to: Vertex, label: string): Edge | undefined => {
    for (const edge of this.edgesFromByLabel.get(from.id)?.get(label) ?? []) {
      if (edge.to === to) {
        return edge;
      }
    }

    return undefined;
  };

  /* Property indexes */

  /**
   * Declare an opt-in secondary index and backfill it from the elements already
   * in the graph (subsequent mutations keep it current). The single
   * index-creation entry point:
   *   - `on`: `'vertex'` or `'edge'`.
   *   - `kind`: `'hash'` — an equality seek over one key (turns `WHERE x.k = …`
   *     into a seek). `'interval'` is only available in the native engine
   *     (`@lenke/native`) and throws here.
   *   - `keys`: `[k]` for a hash index.
   *
   * @example g.createIndex({ on: 'vertex', kind: 'hash', keys: ['email'] })
   */
  public createIndex = (spec: IndexSpec): void => {
    if (spec.kind === 'interval') {
      throw new LenkeError(
        'lenke: interval indexes are only available in the native engine (@lenke/native), not the pure-TS engine',
        { code: ErrorCode.InvalidGraphOp },
      );
    }

    const [key] = spec.keys;

    if (spec.on === 'vertex') {
      this.vertexPropertyIndex.createIndex(key);

      for (const vertex of this.verticesById.values()) {
        this.vertexPropertyIndex.addForKey(vertex, key, vertex.properties[key]);
      }
    } else {
      this.edgePropertyIndex.createIndex(key);

      for (const edge of this.edgesById.values()) {
        this.edgePropertyIndex.addForKey(edge, key, edge.properties[key]);
      }
    }
  };

  public dropVertexIndex = (key: string): void => {
    // A unique constraint is index-backed; dropping its index would silently
    // downgrade enforcement (or lose it), so refuse — drop the constraint first.
    // (Byte-identical to the Rust core, which rejects the same.)
    for (const keys of this.vertexUniqueConstraints.values()) {
      if (keys.has(key)) {
        throw new LenkeError(
          `lenke: cannot drop the vertex index on '${key}' — it backs a unique constraint; drop the constraint first`,
          { code: ErrorCode.InvalidGraphOp },
        );
      }
    }

    this.vertexPropertyIndex.dropIndex(key);
  };

  public dropEdgeIndex = (key: string): void => {
    // See dropVertexIndex: an edge unique constraint is index-backed, so refuse.
    for (const keys of this.edgeUniqueConstraints.values()) {
      if (keys.has(key)) {
        throw new LenkeError(
          `lenke: cannot drop the edge index on '${key}' — it backs a unique constraint; drop the constraint first`,
          { code: ErrorCode.InvalidGraphOp },
        );
      }
    }

    this.edgePropertyIndex.dropIndex(key);
  };

  /** The property keys currently indexed for vertices / edges. */
  public vertexIndexes = (): string[] => this.vertexPropertyIndex.indexedKeys();
  public edgeIndexes = (): string[] => this.edgePropertyIndex.indexedKeys();

  /** Vertices with `key === value`. Empty set if `key` isn't indexed. */
  public getVerticesByProperty = (key: string, value: unknown): Set<Vertex> => {
    return new Set(this.vertexPropertyIndex.equals(key, value) ?? []);
  };

  /** Vertices whose `key` falls within `bound`. Empty set if `key` isn't indexed. */
  public getVerticesByPropertyRange = (key: string, bound: RangeBound): Set<Vertex> => {
    return this.vertexPropertyIndex.range(key, bound) ?? new Set();
  };

  public getEdgesByProperty = (key: string, value: unknown): Set<Edge> => {
    return new Set(this.edgePropertyIndex.equals(key, value) ?? []);
  };

  public getEdgesByPropertyRange = (key: string, bound: RangeBound): Set<Edge> => {
    return this.edgePropertyIndex.range(key, bound) ?? new Set();
  };

  /**
   * Keep the vertex property index current after a property write actually
   * lands. Called by {@link Vertex} once a mutation clears its event guard, so
   * a prevented mutation never touches the index. No-ops for unindexed keys.
   */
  public reindexVertexProperty = (
    vertex: Vertex,
    key: string,
    oldValue: unknown,
    newValue: unknown,
  ): void => {
    this.vertexPropertyIndex.update(vertex, key, oldValue, newValue);
  };

  public reindexEdgeProperty = (
    edge: Edge,
    key: string,
    oldValue: unknown,
    newValue: unknown,
  ): void => {
    this.edgePropertyIndex.update(edge, key, oldValue, newValue);
  };

  /* Unique constraints (declared over `(label, property key)`) */
  // At most one live vertex carrying `label` may hold a given non-null scalar
  // value for `key`. Backed by the vertex property index (so lookups seek). This
  // is the Pattern-B primitive `_MERGE` keys on; byte-identical to the Rust core.
  // See docs/design/gql-extensions.md §3.

  private readonly vertexUniqueConstraints = new Map<string, Set<string>>();

  /**
   * A value participates in uniqueness only if it's a non-null scalar — null and
   * lists/objects are exempt (SQL: NULLs distinct), matching the Rust `IdxKey`
   * domain so both engines agree on what a constraint can bucket.
   */
  private isUniqueKeyable = (value: unknown): value is string | number | boolean =>
    typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean';

  /**
   * Declare a UNIQUE constraint on `(label, key)`. Creates the backing vertex
   * index if absent, then registers it. Idempotent. Throws
   * {@link ErrorCode.ConstraintViolation} if the current data already violates it
   * — an already-broken constraint is meaningless (as SQL rejects the unique
   * index build).
   */
  public createUniqueConstraint = (label: string, key: string): void => {
    if (!this.vertexIndexes().includes(key)) {
      this.createIndex({ on: 'vertex', kind: 'hash', keys: [key] });
    }

    const seen = new Set<unknown>();

    for (const vertex of this.getVerticesByLabel(label)) {
      const value = vertex.properties[key];

      if (!this.isUniqueKeyable(value)) {
        continue;
      }

      if (seen.has(value)) {
        throw new LenkeError(
          'existing data already violates the unique constraint being declared',
          { code: ErrorCode.ConstraintViolation },
        );
      }

      seen.add(value);
    }

    let keys = this.vertexUniqueConstraints.get(label);

    if (!keys) {
      keys = new Set();
      this.vertexUniqueConstraints.set(label, keys);
    }

    keys.add(key);
  };

  /**
   * Drop a unique constraint. The backing index is left in place (drop it via
   * {@link dropVertexIndex} if unwanted). Idempotent.
   */
  public dropUniqueConstraint = (label: string, key: string): void => {
    const keys = this.vertexUniqueConstraints.get(label);

    if (keys) {
      keys.delete(key);

      if (keys.size === 0) {
        this.vertexUniqueConstraints.delete(label);
      }
    }
  };

  /** Property keys under a unique constraint for `label` (sorted; empty if none). */
  public uniqueKeys = (label: string): string[] =>
    [...(this.vertexUniqueConstraints.get(label) ?? [])].sort();

  /** True iff `(label, key)` carries a unique constraint. */
  public hasUniqueConstraint = (label: string, key: string): boolean =>
    this.vertexUniqueConstraints.get(label)?.has(key) ?? false;

  /** Every declared unique constraint as sorted `[label, key]` pairs. */
  public uniqueConstraints = (): Array<[string, string]> =>
    [...this.vertexUniqueConstraints]
      .flatMap(([label, keys]) => [...keys].map((key): [string, string] => [label, key]))
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  // --- REQUIRED constraints (R-CONSTRAINTS) --------------------------------
  // A required `(label, key)` means: every vertex carrying `label` must have a
  // present, non-null value under `key`. Enforced at the core mutation boundary
  // (addVertex + property removal/null-set + addLabel), so every write path —
  // direct API, GQL, Gremlin, ingest — is covered. Byte-identical to the Rust
  // core. Declarative (no closures), so it mirrors across engines like `unique`.

  private readonly vertexRequiredConstraints = new Map<string, Set<string>>();

  /** Is a value "present" for a required constraint? Absent (`undefined`) and
   *  `null` both fail; every other stored value (incl. `''`, `0`, `false`, `[]`)
   *  satisfies presence. */
  private isPresent = (value: unknown): boolean => value !== undefined && value !== null;

  /**
   * Declare a REQUIRED constraint on `(label, key)`. Idempotent. Throws
   * {@link ErrorCode.ConstraintViolation} if any existing vertex with `label`
   * lacks a present, non-null `key` — an already-violated constraint is
   * meaningless.
   */
  public createRequiredConstraint = (label: string, key: string): void => {
    for (const vertex of this.getVerticesByLabel(label)) {
      if (!this.isPresent(vertex.properties[key])) {
        throw new LenkeError(
          `existing data already violates the required constraint being declared on (${label}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    let keys = this.vertexRequiredConstraints.get(label);

    if (!keys) {
      keys = new Set();
      this.vertexRequiredConstraints.set(label, keys);
    }

    keys.add(key);
  };

  /** Drop a required constraint. Idempotent. */
  public dropRequiredConstraint = (label: string, key: string): void => {
    const keys = this.vertexRequiredConstraints.get(label);

    if (keys) {
      keys.delete(key);

      if (keys.size === 0) {
        this.vertexRequiredConstraints.delete(label);
      }
    }
  };

  /** Property keys required for `label` (sorted; empty if none). */
  public requiredKeys = (label: string): string[] =>
    [...(this.vertexRequiredConstraints.get(label) ?? [])].sort();

  /** True iff `(label, key)` carries a required constraint. */
  public hasRequiredConstraint = (label: string, key: string): boolean =>
    this.vertexRequiredConstraints.get(label)?.has(key) ?? false;

  /** Every declared required constraint as sorted `[label, key]` pairs. */
  public requiredConstraints = (): Array<[string, string]> =>
    [...this.vertexRequiredConstraints]
      .flatMap(([label, keys]) => [...keys].map((key): [string, string] => [label, key]))
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  /**
   * The first `(label, key)` a new element with these `labels`/`properties`
   * would violate by omitting a required key, or `undefined` if all satisfied.
   */
  public missingRequired = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
  ): { label: string; key: string } | undefined => {
    if (this.vertexRequiredConstraints.size === 0 && this.vertexTypeNotNull.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      // Required keys = declared required constraints UNION scalar `NOT NULL` keys.
      const keys = new Set([
        ...(this.vertexRequiredConstraints.get(label) ?? []),
        ...(this.vertexTypeNotNull.get(label) ?? []),
      ]);

      for (const key of keys) {
        if (!this.isPresent(properties[key])) {
          return { label, key };
        }
      }
    }

    return undefined;
  };

  /** True iff `key` is required by any of `vertex`'s labels (so it can't be
   *  removed or set to null). */
  public isRequiredKey = (vertex: Vertex, key: string): boolean => {
    if (this.vertexRequiredConstraints.size === 0 && this.vertexTypeNotNull.size === 0) {
      return false;
    }

    for (const label of vertex.labels) {
      if (
        this.vertexRequiredConstraints.get(label)?.has(key) ||
        this.vertexTypeNotNull.get(label)?.has(key)
      ) {
        return true;
      }
    }

    return false;
  };

  /**
   * Inside a transaction (or a rollback replay) the per-write constraint gates
   * defer to the commit-time recheck rather than throwing immediately. Returns
   * true if the caller should skip its inline check; records the touched vertex
   * so {@link runDeferredChecks} revisits it at commit (a replay records nothing).
   */
  private deferConstraint = (vertex: Vertex): boolean => {
    if (this.applyingUndo) {
      return true;
    }

    if (this.txDepth > 0) {
      this.txTouched.add(vertex.id);

      return true;
    }

    return false;
  };

  /** Throw if setting `vertex.key = value` would null out a required key. Called
   *  by the property mutators so every write path is guarded, not just the API. */
  public assertRequiredOnSet = (vertex: Vertex, key: string, value: unknown): void => {
    if (this.deferConstraint(vertex)) {
      return;
    }

    if (!this.isPresent(value) && this.isRequiredKey(vertex, key)) {
      throw new LenkeError(`cannot set required property '${key}' to null`, {
        code: ErrorCode.ConstraintViolation,
      });
    }
  };

  /** Throw if removing `vertex.key` would drop a required key. */
  public assertRequiredOnRemove = (vertex: Vertex, key: string): void => {
    if (this.deferConstraint(vertex)) {
      return;
    }

    if (this.isRequiredKey(vertex, key)) {
      throw new LenkeError(`cannot remove required property '${key}'`, {
        code: ErrorCode.ConstraintViolation,
      });
    }
  };

  // --- TYPE constraints (R-CONSTRAINTS) ------------------------------------
  // A type `(label, key, type)` means: every present, non-null value under `key`
  // on a vertex carrying `label` must be of the given scalar type. Null/absent
  // are exempt (a null has no type — use a `required` constraint for presence).
  // Enforced at the core mutation boundary; byte-identical to the Rust core.

  private readonly vertexTypeConstraints = new Map<string, Map<string, ScalarTypeName>>();

  private readonly vertexRecordConstraints = new Map<string, Map<string, TypeSpec>>();

  /** `label` → scalar keys declared `NOT NULL` (`string NOT NULL`). A `NOT NULL`
   *  key is REQUIRED (present + non-null); folded into the required checks. Kept
   *  separate from {@link vertexRequiredConstraints} so dropping the type
   *  constraint leaves an independently-declared required constraint intact. */
  private readonly vertexTypeNotNull = new Map<string, Set<string>>();

  /** The scalar type of a stored value, or `null` for null/absent/record
   *  (type-exempt from a scalar constraint — the record path governs maps). */
  private valueType = (v: unknown): ScalarTypeName | null => {
    if (v === undefined || v === null || isRecord(v)) {
      return null;
    }

    if (Array.isArray(v)) {
      return 'list';
    }

    const t = typeof v;

    if (t === 'string' || t === 'number' || t === 'boolean') {
      return t;
    }

    // Temporals carry a `kind` discriminant ('date' | 'datetime' | 'duration').
    if (typeof v === 'object' && 'kind' in v) {
      const k = (v as { kind: unknown }).kind;

      if (k === 'date' || k === 'datetime' || k === 'duration') {
        return k;
      }
    }

    return 'string'; // unreachable for a validated scalar value
  };

  /**
   * Declare a TYPE constraint on `(label, key)`. Idempotent (re-declaring
   * replaces). Throws {@link ErrorCode.ConstraintViolation} if any existing
   * vertex with `label` holds a present, non-null `key` of a different type.
   */
  public createTypeConstraint = (label: string, key: string, type: string): void => {
    const parsed = parseTypeSpecWithNotNull(type);

    if (parsed === null) {
      throw new LenkeError(
        `unknown or malformed type '${type}' for the type constraint on (${label}, ${key})`,
        { code: ErrorCode.InvalidValue },
      );
    }

    const { spec, notNull } = parsed;

    for (const vertex of this.getVerticesByLabel(label)) {
      const v = vertex.properties[key];

      if (!valueMatches(v, spec, this.valueType)) {
        throw new LenkeError(
          `existing data already violates the type constraint being declared on (${label}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (notNull && !this.isPresent(v)) {
        throw new LenkeError(
          `existing data already violates the NOT NULL constraint being declared on (${label}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    const markNotNull = (): void => {
      if (!notNull) {
        return;
      }

      let nn = this.vertexTypeNotNull.get(label);

      if (!nn) {
        nn = new Set();
        this.vertexTypeNotNull.set(label, nn);
      }

      nn.add(key);
    };

    // Scalar → the existing map; a record / any-record shape → the parallel map.
    if (spec.kind === 'scalar') {
      let keys = this.vertexTypeConstraints.get(label);

      if (!keys) {
        keys = new Map();
        this.vertexTypeConstraints.set(label, keys);
      }

      keys.set(key, spec.type);
    } else {
      let keys = this.vertexRecordConstraints.get(label);

      if (!keys) {
        keys = new Map();
        this.vertexRecordConstraints.set(label, keys);
      }

      keys.set(key, spec);
    }

    markNotNull(); // record-level (or scalar) NOT NULL makes the property required
  };

  /** Drop a type constraint (scalar or record). Idempotent. Removes this
   *  constraint's `NOT NULL` (leaving any independent required constraint). */
  public dropTypeConstraint = (label: string, key: string): void => {
    for (const map of [
      this.vertexTypeConstraints,
      this.vertexRecordConstraints,
      this.vertexTypeNotNull,
    ]) {
      const keys = map.get(label);

      if (keys) {
        keys.delete(key);

        if (keys.size === 0) {
          map.delete(label);
        }
      }
    }
  };

  /** The declared type for `(label, key)`, or `undefined`. */
  public typeConstraint = (label: string, key: string): ScalarTypeName | undefined =>
    this.vertexTypeConstraints.get(label)?.get(key);

  /** Every declared type constraint as sorted `[label, key, type]` triples. */
  public typeConstraints = (): Array<[string, string, ScalarTypeName]> =>
    [...this.vertexTypeConstraints]
      .flatMap(([label, keys]) =>
        [...keys].map(([key, type]): [string, string, ScalarTypeName] => [label, key, type]),
      )
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  /**
   * The first type violation a new element with these `labels`/`properties`
   * would cause, or `undefined`. Null/absent values are exempt.
   */
  public typeViolation = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
  ): { label: string; key: string; expected: ScalarTypeName; got: ScalarTypeName } | undefined => {
    if (this.vertexTypeConstraints.size === 0 && this.vertexRecordConstraints.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      const cs = this.vertexTypeConstraints.get(label);

      if (cs) {
        for (const [key, type] of cs) {
          const got = this.valueType(properties[key]);

          if (got !== null && got !== type) {
            return { label, key, expected: type, got };
          }
        }
      }

      const rs = this.vertexRecordConstraints.get(label);

      if (rs) {
        for (const [key, spec] of rs) {
          if (key in properties && !valueMatches(properties[key], spec, this.valueType)) {
            return { label, key, expected: 'string', got: 'string' }; // shape mismatch
          }
        }
      }
    }

    return undefined;
  };

  /** Throw if setting `vertex.key = value` would break a type constraint. Null
   *  is exempt (a null has no type — `required` governs presence). */
  public assertTypeOnSet = (vertex: Vertex, key: string, value: unknown): void => {
    if (this.deferConstraint(vertex)) {
      return;
    }

    if (this.vertexTypeConstraints.size === 0 && this.vertexRecordConstraints.size === 0) {
      return;
    }

    // Record constraints: the value must match the declared shape (a null is
    // exempt via `valueMatches`).
    for (const label of vertex.labels) {
      const spec = this.vertexRecordConstraints.get(label)?.get(key);

      if (spec && !valueMatches(value, spec, this.valueType)) {
        throw new LenkeError(
          `property '${key}' must match the declared record type on '${label}'`,
          {
            code: ErrorCode.ConstraintViolation,
          },
        );
      }
    }

    // Scalar constraints: a non-scalar (null/record) has no scalar type → exempt.
    const got = this.valueType(value);

    if (got === null) {
      return;
    }

    for (const label of vertex.labels) {
      const type = this.vertexTypeConstraints.get(label)?.get(key);

      if (type && got !== type) {
        throw new LenkeError(`property '${key}' must be ${type} on '${label}', got ${got}`, {
          code: ErrorCode.ConstraintViolation,
        });
      }
    }
  };

  /**
   * Throw if setting `vertex.key = value` would collide with another vertex
   * under a unique constraint. Called from the property-write chokepoint so the
   * direct API enforces the same invariant the GQL `SET` path does.
   */
  public assertUniqueOnSet = (vertex: Vertex, key: string, value: unknown): void => {
    if (this.deferConstraint(vertex)) {
      return;
    }

    const conflict = this.uniqueConflictOnSet(vertex, key, value);

    if (conflict) {
      throw new LenkeError(
        `unique constraint on '${conflict.label}.${key}' violated by value ${JSON.stringify(value)}`,
        { code: ErrorCode.ConstraintViolation },
      );
    }
  };

  /**
   * The single live vertex carrying `label` whose `key === value`, if any (≤1
   * under the constraint). The `_MERGE` create-vs-update decision. Null/list
   * values yield `undefined` (exempt).
   */
  public uniqueLookup = (label: string, key: string, value: unknown): Vertex | undefined => {
    if (!this.isUniqueKeyable(value)) {
      return undefined;
    }

    for (const vertex of this.vertexPropertyIndex.equals(key, value) ?? []) {
      if (vertex.hasLabel(label)) {
        return vertex;
      }
    }

    return undefined;
  };

  /**
   * If adding a vertex with `labels` + `properties` would break a unique
   * constraint, the offending `{ label, key, existing }`. Drives INSERT
   * enforcement; `exclude` skips one vertex. Only constrained keys present with a
   * non-null scalar value are checked.
   */
  public uniqueConflict = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
    exclude?: Vertex,
  ): { label: string; key: string; existing: Vertex } | undefined => {
    if (this.vertexUniqueConstraints.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      for (const key of this.uniqueKeys(label)) {
        if (!(key in properties)) {
          continue;
        }

        const value = properties[key];

        if (!this.isUniqueKeyable(value)) {
          continue;
        }

        for (const vertex of this.vertexPropertyIndex.equals(key, value) ?? []) {
          if (vertex !== exclude && vertex.hasLabel(label)) {
            return { label, key, existing: vertex };
          }
        }
      }
    }

    return undefined;
  };

  /**
   * If setting `vertex.key = value` would break a unique constraint on one of
   * `vertex`'s labels, the offending `{ label, existing }`.
   */
  public uniqueConflictOnSet = (
    vertex: Vertex,
    key: string,
    value: unknown,
  ): { label: string; existing: Vertex } | undefined => {
    if (this.vertexUniqueConstraints.size === 0 || !this.isUniqueKeyable(value)) {
      return undefined;
    }

    for (const [label, keys] of this.vertexUniqueConstraints) {
      if (!keys.has(key) || !vertex.hasLabel(label)) {
        continue;
      }

      for (const other of this.vertexPropertyIndex.equals(key, value) ?? []) {
        if (other !== vertex && other.hasLabel(label)) {
          return { label, existing: other };
        }
      }
    }

    return undefined;
  };

  // --- EDGE constraints (R-CONSTRAINTS, edge types) ------------------------
  // A direct mirror of the vertex unique/required/type constraints, keyed by edge
  // TYPE instead of node label, enforced against each edge's properties and the
  // edge property index. Byte-identical to the Rust core. Enforcement is at the
  // core mutation boundary (addEdge gate + Edge property asserts + addLabelToEdge),
  // and defers to commit inside a transaction exactly like the vertex ones.

  private readonly edgeUniqueConstraints = new Map<string, Set<string>>();
  private readonly edgeRequiredConstraints = new Map<string, Set<string>>();
  private readonly edgeTypeConstraints = new Map<string, Map<string, ScalarTypeName>>();
  private readonly edgeRecordConstraints = new Map<string, Map<string, TypeSpec>>();

  /** Edge-type → scalar keys declared `NOT NULL`. Edge analogue of
   *  {@link vertexTypeNotNull}. */
  private readonly edgeTypeNotNull = new Map<string, Set<string>>();

  /**
   * Inside a transaction (or a rollback replay) the per-write edge-constraint
   * gates defer to the commit-time recheck. Returns true if the caller should skip
   * its inline check; records the touched edge so {@link runDeferredChecks}
   * revisits it at commit (a replay records nothing). Edge analogue of
   * {@link deferConstraint}.
   */
  private deferEdgeConstraint = (edge: Edge): boolean => {
    if (this.applyingUndo) {
      return true;
    }

    if (this.txDepth > 0) {
      this.txTouchedEdges.add(edge.id);

      return true;
    }

    return false;
  };

  /* Edge UNIQUE constraints (declared over `(edge type, key)`) */

  /**
   * Declare a UNIQUE constraint on `(edgeType, key)`. Creates the backing edge
   * index if absent. Idempotent. Throws {@link ErrorCode.ConstraintViolation} if
   * the current data already violates it.
   */
  public createEdgeUniqueConstraint = (edgeType: string, key: string): void => {
    if (!this.edgeIndexes().includes(key)) {
      this.createIndex({ on: 'edge', kind: 'hash', keys: [key] });
    }

    const seen = new Set<unknown>();

    for (const edge of this.getEdgesByLabel(edgeType)) {
      const value = edge.properties[key];

      if (!this.isUniqueKeyable(value)) {
        continue;
      }

      if (seen.has(value)) {
        throw new LenkeError(
          'existing data already violates the edge unique constraint being declared',
          { code: ErrorCode.ConstraintViolation },
        );
      }

      seen.add(value);
    }

    let keys = this.edgeUniqueConstraints.get(edgeType);

    if (!keys) {
      keys = new Set();
      this.edgeUniqueConstraints.set(edgeType, keys);
    }

    keys.add(key);
  };

  /** Drop an edge unique constraint. The backing index is left in place. Idempotent. */
  public dropEdgeUniqueConstraint = (edgeType: string, key: string): void => {
    const keys = this.edgeUniqueConstraints.get(edgeType);

    if (keys) {
      keys.delete(key);

      if (keys.size === 0) {
        this.edgeUniqueConstraints.delete(edgeType);
      }
    }
  };

  /** Property keys under a unique constraint for `edgeType` (sorted; empty if none). */
  public edgeUniqueKeys = (edgeType: string): string[] =>
    [...(this.edgeUniqueConstraints.get(edgeType) ?? [])].sort();

  /** True iff `(edgeType, key)` carries a unique constraint. */
  public hasEdgeUniqueConstraint = (edgeType: string, key: string): boolean =>
    this.edgeUniqueConstraints.get(edgeType)?.has(key) ?? false;

  /** Every declared edge unique constraint as sorted `[edgeType, key]` pairs. */
  public edgeUniqueConstraintList = (): Array<[string, string]> =>
    [...this.edgeUniqueConstraints]
      .flatMap(([edgeType, keys]) => [...keys].map((key): [string, string] => [edgeType, key]))
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  /**
   * If adding an edge with `labels` + `properties` would break a unique
   * constraint, the offending `{ label, key, existing }`. Only constrained keys
   * present with a non-null scalar value are checked. `exclude` skips one edge.
   */
  public edgeUniqueConflict = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
    exclude?: Edge,
  ): { label: string; key: string; existing: Edge } | undefined => {
    if (this.edgeUniqueConstraints.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      for (const key of this.edgeUniqueKeys(label)) {
        if (!(key in properties)) {
          continue;
        }

        const value = properties[key];

        if (!this.isUniqueKeyable(value)) {
          continue;
        }

        for (const edge of this.edgePropertyIndex.equals(key, value) ?? []) {
          if (edge !== exclude && edge.hasLabel(label)) {
            return { label, key, existing: edge };
          }
        }
      }
    }

    return undefined;
  };

  /**
   * If setting `edge.key = value` would break a unique constraint on one of
   * `edge`'s types, the offending `{ label, existing }`.
   */
  public edgeUniqueConflictOnSet = (
    edge: Edge,
    key: string,
    value: unknown,
  ): { label: string; existing: Edge } | undefined => {
    if (this.edgeUniqueConstraints.size === 0 || !this.isUniqueKeyable(value)) {
      return undefined;
    }

    for (const [label, keys] of this.edgeUniqueConstraints) {
      if (!keys.has(key) || !edge.hasLabel(label)) {
        continue;
      }

      for (const other of this.edgePropertyIndex.equals(key, value) ?? []) {
        if (other !== edge && other.hasLabel(label)) {
          return { label, existing: other };
        }
      }
    }

    return undefined;
  };

  /**
   * Throw if setting `edge.key = value` would collide with another edge under a
   * unique constraint. Called from the edge property-write chokepoint.
   */
  public assertEdgeUniqueOnSet = (edge: Edge, key: string, value: unknown): void => {
    if (this.deferEdgeConstraint(edge)) {
      return;
    }

    const conflict = this.edgeUniqueConflictOnSet(edge, key, value);

    if (conflict) {
      throw new LenkeError(
        `unique constraint on edge type '${conflict.label}.${key}' violated by value ${JSON.stringify(value)}`,
        { code: ErrorCode.ConstraintViolation },
      );
    }
  };

  /* Edge REQUIRED constraints */

  /**
   * Declare a REQUIRED constraint on `(edgeType, key)`. Idempotent. Throws
   * {@link ErrorCode.ConstraintViolation} if any existing edge of `edgeType`
   * lacks a present, non-null `key`.
   */
  public createEdgeRequiredConstraint = (edgeType: string, key: string): void => {
    for (const edge of this.getEdgesByLabel(edgeType)) {
      if (!this.isPresent(edge.properties[key])) {
        throw new LenkeError(
          `existing data already violates the edge required constraint being declared on (${edgeType}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    let keys = this.edgeRequiredConstraints.get(edgeType);

    if (!keys) {
      keys = new Set();
      this.edgeRequiredConstraints.set(edgeType, keys);
    }

    keys.add(key);
  };

  /** Drop an edge required constraint. Idempotent. */
  public dropEdgeRequiredConstraint = (edgeType: string, key: string): void => {
    const keys = this.edgeRequiredConstraints.get(edgeType);

    if (keys) {
      keys.delete(key);

      if (keys.size === 0) {
        this.edgeRequiredConstraints.delete(edgeType);
      }
    }
  };

  /** Property keys required for `edgeType` (sorted; empty if none). */
  public edgeRequiredKeys = (edgeType: string): string[] =>
    [...(this.edgeRequiredConstraints.get(edgeType) ?? [])].sort();

  /** True iff `(edgeType, key)` carries a required constraint. */
  public hasEdgeRequiredConstraint = (edgeType: string, key: string): boolean =>
    this.edgeRequiredConstraints.get(edgeType)?.has(key) ?? false;

  /** Every declared edge required constraint as sorted `[edgeType, key]` pairs. */
  public edgeRequiredConstraintList = (): Array<[string, string]> =>
    [...this.edgeRequiredConstraints]
      .flatMap(([edgeType, keys]) => [...keys].map((key): [string, string] => [edgeType, key]))
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  /**
   * The first `(edgeType, key)` a new edge with these `labels`/`properties` would
   * violate by omitting a required key, or `undefined`.
   */
  public edgeMissingRequired = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
  ): { label: string; key: string } | undefined => {
    if (this.edgeRequiredConstraints.size === 0 && this.edgeTypeNotNull.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      const keys = new Set([
        ...(this.edgeRequiredConstraints.get(label) ?? []),
        ...(this.edgeTypeNotNull.get(label) ?? []),
      ]);

      for (const key of keys) {
        if (!this.isPresent(properties[key])) {
          return { label, key };
        }
      }
    }

    return undefined;
  };

  /** True iff `key` is required by any of `edge`'s types (so it can't be removed
   *  or set to null). */
  public isEdgeRequiredKey = (edge: Edge, key: string): boolean => {
    if (this.edgeRequiredConstraints.size === 0 && this.edgeTypeNotNull.size === 0) {
      return false;
    }

    for (const label of edge.labels) {
      if (this.edgeTypeNotNull.get(label)?.has(key)) {
        return true;
      }

      if (this.edgeRequiredConstraints.get(label)?.has(key)) {
        return true;
      }
    }

    return false;
  };

  /** Throw if setting `edge.key = value` would null out a required key. */
  public assertEdgeRequiredOnSet = (edge: Edge, key: string, value: unknown): void => {
    if (this.deferEdgeConstraint(edge)) {
      return;
    }

    if (!this.isPresent(value) && this.isEdgeRequiredKey(edge, key)) {
      throw new LenkeError(`cannot set required edge property '${key}' to null`, {
        code: ErrorCode.ConstraintViolation,
      });
    }
  };

  /** Throw if removing `edge.key` would drop a required key. */
  public assertEdgeRequiredOnRemove = (edge: Edge, key: string): void => {
    if (this.deferEdgeConstraint(edge)) {
      return;
    }

    if (this.isEdgeRequiredKey(edge, key)) {
      throw new LenkeError(`cannot remove required edge property '${key}'`, {
        code: ErrorCode.ConstraintViolation,
      });
    }
  };

  /* Edge TYPE constraints */

  /**
   * Declare a TYPE constraint on `(edgeType, key)`. Idempotent (re-declaring
   * replaces). Throws {@link ErrorCode.InvalidValue} for an unknown scalar type,
   * or {@link ErrorCode.ConstraintViolation} if any existing edge of `edgeType`
   * holds a present, non-null `key` of a different type.
   */
  public createEdgeTypeConstraint = (edgeType: string, key: string, type: string): void => {
    const parsed = parseTypeSpecWithNotNull(type);

    if (parsed === null) {
      throw new LenkeError(
        `unknown or malformed type '${type}' for the edge type constraint on (${edgeType}, ${key})`,
        { code: ErrorCode.InvalidValue },
      );
    }

    const { spec, notNull } = parsed;

    for (const edge of this.getEdgesByLabel(edgeType)) {
      const v = edge.properties[key];

      if (!valueMatches(v, spec, this.valueType)) {
        throw new LenkeError(
          `existing data already violates the edge type constraint being declared on (${edgeType}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      if (notNull && !this.isPresent(v)) {
        throw new LenkeError(
          `existing data already violates the NOT NULL constraint being declared on (${edgeType}, ${key})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    const markNotNull = (): void => {
      if (!notNull) {
        return;
      }

      let nn = this.edgeTypeNotNull.get(edgeType);

      if (!nn) {
        nn = new Set();
        this.edgeTypeNotNull.set(edgeType, nn);
      }

      nn.add(key);
    };

    if (spec.kind === 'scalar') {
      let keys = this.edgeTypeConstraints.get(edgeType);

      if (!keys) {
        keys = new Map();
        this.edgeTypeConstraints.set(edgeType, keys);
      }

      keys.set(key, spec.type);
    } else {
      let keys = this.edgeRecordConstraints.get(edgeType);

      if (!keys) {
        keys = new Map();
        this.edgeRecordConstraints.set(edgeType, keys);
      }

      keys.set(key, spec);
    }

    markNotNull();
  };

  /** Drop an edge type constraint (scalar or record). Idempotent. */
  public dropEdgeTypeConstraint = (edgeType: string, key: string): void => {
    for (const map of [
      this.edgeTypeConstraints,
      this.edgeRecordConstraints,
      this.edgeTypeNotNull,
    ]) {
      const keys = map.get(edgeType);

      if (keys) {
        keys.delete(key);

        if (keys.size === 0) {
          map.delete(edgeType);
        }
      }
    }
  };

  /** The declared type for `(edgeType, key)`, or `undefined`. */
  public edgeTypeConstraint = (edgeType: string, key: string): ScalarTypeName | undefined =>
    this.edgeTypeConstraints.get(edgeType)?.get(key);

  /** Every declared edge type constraint as sorted `[edgeType, key, type]` triples. */
  public edgeTypeConstraintList = (): Array<[string, string, ScalarTypeName]> =>
    [...this.edgeTypeConstraints]
      .flatMap(([edgeType, keys]) =>
        [...keys].map(([key, type]): [string, string, ScalarTypeName] => [edgeType, key, type]),
      )
      .sort((a, b) => (a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0])));

  /**
   * The first type violation a new edge with these `labels`/`properties` would
   * cause, or `undefined`. Null/absent values are exempt.
   */
  public edgeTypeViolation = (
    labels: Iterable<string>,
    properties: Readonly<Record<string, unknown>>,
  ): { label: string; key: string; expected: ScalarTypeName; got: ScalarTypeName } | undefined => {
    if (this.edgeTypeConstraints.size === 0 && this.edgeRecordConstraints.size === 0) {
      return undefined;
    }

    for (const label of labels) {
      const cs = this.edgeTypeConstraints.get(label);

      if (cs) {
        for (const [key, type] of cs) {
          const got = this.valueType(properties[key]);

          if (got !== null && got !== type) {
            return { label, key, expected: type, got };
          }
        }
      }

      const rs = this.edgeRecordConstraints.get(label);

      if (rs) {
        for (const [key, spec] of rs) {
          if (key in properties && !valueMatches(properties[key], spec, this.valueType)) {
            return { label, key, expected: 'string', got: 'string' };
          }
        }
      }
    }

    return undefined;
  };

  /** Throw if setting `edge.key = value` would break a type constraint. Null is
   *  exempt (a null has no type — `required` governs presence). */
  public assertEdgeTypeOnSet = (edge: Edge, key: string, value: unknown): void => {
    if (this.deferEdgeConstraint(edge)) {
      return;
    }

    if (this.edgeTypeConstraints.size === 0 && this.edgeRecordConstraints.size === 0) {
      return;
    }

    for (const label of edge.labels) {
      const spec = this.edgeRecordConstraints.get(label)?.get(key);

      if (spec && !valueMatches(value, spec, this.valueType)) {
        throw new LenkeError(
          `property '${key}' must match the declared record type on edge type '${label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    const got = this.valueType(value);

    if (got === null) {
      return;
    }

    for (const label of edge.labels) {
      const type = this.edgeTypeConstraints.get(label)?.get(key);

      if (type && got !== type) {
        throw new LenkeError(
          `property '${key}' must be ${type} on edge type '${label}', got ${got}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }
  };

  // --- CARDINALITY constraints (R-CONSTRAINTS, degree bounds) --------------
  // A cardinality constraint bounds the DEGREE of each vertex carrying `label`
  // over `edgeType` in `direction` (out = the vertex is the edge source; in =
  // the target): for every such vertex V, `min <= degree(V) <= max`. "exactly
  // one" = min:1,max:1; "at most one" = min:0,max:1; "at least one" = min:1,
  // max:null. A self-loop (V—[e]->V) counts once for out AND once for in (it
  // appears in both adjacency directions), matching the Rust core.
  //
  // Enforcement splits by satisfiability at a single write (see docs/design/r-tx.md):
  //   - MAX is reachable by one write, so it's eager on a bare `addEdge` outside
  //     a transaction (the over-the-limit edge throws) and deferred to commit
  //     inside one.
  //   - MIN is NOT reachable by one write (a fresh vertex has degree 0), so it is
  //     commit-time ONLY, checked in `runDeferredChecks` against touched vertices.
  //     Every GQL statement auto-commits and `transaction(fn)` commits at the end,
  //     so those are min's enforcement boundaries. A bare `addVertex(...)` outside
  //     any transaction has NO commit boundary, so it does not trip a min check —
  //     intended: min inherently needs a second write (the edge), which is why
  //     R-TX was its prerequisite. Declaring the constraint scans existing data.

  private readonly vertexCardinalityConstraints = new Map<string, CardinalityConstraint>();

  /** Composite registry key for a `(label, edgeType, direction)` cardinality constraint. */
  private cardKey = (label: string, edgeType: string, direction: 'out' | 'in'): string =>
    `${label} ${edgeType} ${direction}`;

  /** Number of `edgeType` edges for which `vertex` is the SOURCE (out-degree). */
  public outDegree = (vertex: Vertex, edgeType: string): number =>
    this.edgesFromByLabel.get(vertex.id)?.get(edgeType)?.size ?? 0;

  /** Number of `edgeType` edges for which `vertex` is the TARGET (in-degree). */
  public inDegree = (vertex: Vertex, edgeType: string): number =>
    this.edgesToByLabel.get(vertex.id)?.get(edgeType)?.size ?? 0;

  /**
   * Declare a CARDINALITY constraint bounding the degree of every vertex carrying
   * `label` over `edgeType` in `direction`. Re-declaring `(label, edgeType,
   * direction)` replaces the bounds. Throws {@link ErrorCode.ConstraintViolation}
   * if any existing vertex already violates `min`/`max` (mirrors unique/required).
   */
  public createCardinalityConstraint = (
    label: string,
    edgeType: string,
    direction: 'out' | 'in',
    min: number,
    max: number | null,
  ): void => {
    for (const vertex of this.getVerticesByLabel(label)) {
      const degree =
        direction === 'out' ? this.outDegree(vertex, edgeType) : this.inDegree(vertex, edgeType);

      if (degree < min || (max !== null && degree > max)) {
        throw new LenkeError(
          `existing data already violates the cardinality constraint being declared on '${label}' (${edgeType} ${direction})`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    this.vertexCardinalityConstraints.set(this.cardKey(label, edgeType, direction), {
      label,
      edgeType,
      direction,
      min,
      max,
    });
  };

  /** Drop a cardinality constraint on `(label, edgeType, direction)`. Idempotent. */
  public dropCardinalityConstraint = (
    label: string,
    edgeType: string,
    direction: 'out' | 'in',
  ): void => {
    this.vertexCardinalityConstraints.delete(this.cardKey(label, edgeType, direction));
  };

  /** Every declared cardinality constraint, sorted by `(label, edgeType, direction)`. */
  public cardinalityConstraints = (): CardinalityConstraint[] =>
    [...this.vertexCardinalityConstraints.values()].sort((a, b) => {
      if (a.label !== b.label) {
        return a.label.localeCompare(b.label);
      }

      if (a.edgeType !== b.edgeType) {
        return a.edgeType.localeCompare(b.edgeType);
      }

      return a.direction.localeCompare(b.direction);
    });

  /**
   * The first cardinality constraint `vertex` currently violates (degree below
   * `min` or above `max`), or `undefined`. The commit-time / declare-time check.
   */
  private cardinalityViolationOf = (vertex: Vertex): CardinalityConstraint | undefined => {
    for (const c of this.vertexCardinalityConstraints.values()) {
      if (!vertex.hasLabel(c.label)) {
        continue;
      }

      const degree =
        c.direction === 'out'
          ? this.outDegree(vertex, c.edgeType)
          : this.inDegree(vertex, c.edgeType);

      if (degree < c.min || (c.max !== null && degree > c.max)) {
        return c;
      }
    }

    return undefined;
  };

  /**
   * The first MAX cardinality constraint the about-to-be-inserted `edge` would
   * push its constrained endpoint over, or `undefined`. Called from
   * {@link insertEdge} before the edge is indexed, so each endpoint's degree is
   * still pre-insert — hence `+ 1`.
   */
  private cardinalityMaxExceededOnAdd = (
    from: Vertex,
    to: Vertex,
    edgeLabels: Set<string>,
  ): CardinalityConstraint | undefined => {
    for (const c of this.vertexCardinalityConstraints.values()) {
      if (c.max === null || !edgeLabels.has(c.edgeType)) {
        continue;
      }

      if (c.direction === 'out') {
        if (from.hasLabel(c.label) && this.outDegree(from, c.edgeType) + 1 > c.max) {
          return c;
        }
      } else if (to.hasLabel(c.label) && this.inDegree(to, c.edgeType) + 1 > c.max) {
        return c;
      }
    }

    return undefined;
  };

  /**
   * Cardinality gate for an edge insert. Outside a transaction, enforce MAX
   * eagerly (a single edge can push a degree over its bound). Inside one, record
   * both endpoints as touched so {@link runDeferredChecks} re-checks min AND max
   * at commit. A rollback replay skips entirely (restores known-good state).
   */
  private cardinalityGateOnInsert = (edge: Edge): void => {
    if (this.vertexCardinalityConstraints.size === 0 || this.applyingUndo) {
      return;
    }

    if (this.txDepth > 0) {
      this.txTouched.add(edge.from.id);
      this.txTouched.add(edge.to.id);

      return;
    }

    const exceeded = this.cardinalityMaxExceededOnAdd(edge.from, edge.to, edge.labels);

    if (exceeded) {
      throw new LenkeError(
        `cardinality constraint on '${exceeded.label}' (${exceeded.edgeType} ${exceeded.direction}) violated: degree would exceed max ${exceeded.max}`,
        { code: ErrorCode.ConstraintViolation },
      );
    }
  };

  /**
   * Note both endpoints of a removed edge for the commit-time cardinality
   * recheck (their degree dropped, possibly below `min`). No-op outside a
   * transaction — a bare `removeEdge` has no commit boundary, so min is not
   * enforced there — or during a rollback replay.
   */
  private cardinalityNoteOnRemove = (edge: Edge): void => {
    if (this.vertexCardinalityConstraints.size === 0 || this.applyingUndo || this.txDepth === 0) {
      return;
    }

    this.txTouched.add(edge.from.id);
    this.txTouched.add(edge.to.id);
  };

  // --- VALIDATORS (R-CONSTRAINTS, custom GQL-predicate constraints) ---------
  // A validator attaches a GQL boolean expression to a label (vertex label OR
  // edge type — one string namespace): every element carrying `label` must
  // satisfy the predicate, with the element bound to `varName`. SQL-`CHECK`
  // semantics — a write is rejected only when the predicate is a *definite*
  // `false`; a `null`/unknown result passes (an absent optional property is not a
  // violation). Enforced at the core mutation boundary (addVertex / insertEdge
  // gates) and deferred to commit inside a transaction (via the same touched-set
  // machinery the other constraints use), so every write path is covered.
  //
  // Core cannot parse or evaluate a GQL expression, so `@lenke/gql` owns the
  // public `createValidator`: it parses+compiles the predicate into the closure
  // registered here. The Rust core stores the predicate STRING and evaluates it
  // in its own byte-identical GQL evaluator — the same predicate accepts/rejects
  // identically on both engines.

  private readonly validatorRegistry = new Map<string, ValidatorEntry[]>();

  /**
   * Register a compiled VALIDATOR predicate on `label`. Appends (a label may
   * carry several validators). Declare-time scan: rejects
   * ({@link ErrorCode.ConstraintViolation}) if any existing element carrying
   * `label` — vertex OR edge — currently evaluates to a definite `false` (an
   * already-violated validator is meaningless, mirroring unique/required/type/
   * cardinality). Called by `@lenke/gql`'s `createValidator`, which supplies the
   * parsed+compiled `fn`; core never parses the predicate itself.
   */
  public registerValidator = (
    label: string,
    varName: string,
    src: string,
    fn: ValidatorFn,
  ): void => {
    for (const element of [...this.getVerticesByLabel(label), ...this.getEdgesByLabel(label)]) {
      if (fn(element) === false) {
        throw new LenkeError(
          `existing data already violates the validator '${src}' being declared on '${label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }
    }

    let list = this.validatorRegistry.get(label);

    if (!list) {
      list = [];
      this.validatorRegistry.set(label, list);
    }

    list.push({ varName, src, fn });
  };

  /** Drop every validator declared on `label`. Idempotent. */
  public dropValidator = (label: string): void => {
    this.validatorRegistry.delete(label);
  };

  /**
   * Every declared validator as `{ label, varName, src }`, sorted by
   * `(label, src)`. The compiled predicate closure is internal (not exposed).
   */
  public validators = (): ValidatorInfo[] =>
    [...this.validatorRegistry]
      .flatMap(([label, entries]) =>
        entries.map((e): ValidatorInfo => ({ label, varName: e.varName, src: e.src })),
      )
      .sort((a, b) =>
        a.label === b.label ? a.src.localeCompare(b.src) : a.label.localeCompare(b.label),
      );

  /**
   * The first validator `element` (vertex or edge) currently fails — a *definite*
   * `false` over any label it carries — or `undefined`. A `null`/unknown result
   * passes (SQL-`CHECK`). The per-write / declare-time / commit-time check.
   */
  private validatorViolationOf = (
    element: Vertex | Edge,
  ): { label: string; src: string } | undefined => {
    if (this.validatorRegistry.size === 0) {
      return undefined;
    }

    for (const label of element.labels) {
      for (const entry of this.validatorRegistry.get(label) ?? []) {
        if (entry.fn(element) === false) {
          return { label, src: entry.src };
        }
      }
    }

    return undefined;
  };

  /* Graph-level INVARIANTS (cross-write assertions) */
  //
  // An invariant is a whole-graph GQL assertion query that must hold after every
  // write transaction (`createInvariant(g, 'balanced', 'MATCH (a:Acct) RETURN
  // sum(a.balance) = 0')`). Unlike a per-element validator, it's evaluated ONCE
  // per commit against the fully-staged graph. `false`-only-fails: VIOLATED iff a
  // returned cell is boolean `false` (everything else — `true`/`null`/non-boolean
  // /empty — holds). Enforced in `runDeferredChecks`, after the per-element loops,
  // and only when the transaction actually wrote something (a pure-read commit
  // skips them). Core cannot parse a GQL query, so `@lenke/gql` owns the public
  // `createInvariant`: it compiles the query into the closure registered here. The
  // Rust core stores the query and evaluates it in its byte-identical engine.

  private readonly invariantRegistry = new Map<string, InvariantEntry>();

  /**
   * Register a compiled graph-level INVARIANT under `name` (re-declaring the same
   * `name` replaces the prior query). Declare-time run: evaluates `fn` against the
   * current graph immediately and rejects ({@link ErrorCode.ConstraintViolation})
   * if it already returns a `false` cell — an already-violated invariant is
   * meaningless (mirroring the validators/constraints). Called by `@lenke/gql`'s
   * `createInvariant`, which supplies the parsed+compiled query closure; core
   * never parses the query itself.
   */
  public registerInvariant = (name: string, src: string, fn: InvariantFn): void => {
    if (this.invariantViolated(fn(this))) {
      throw new LenkeError(`existing data already violates the invariant '${name}' (${src})`, {
        code: ErrorCode.ConstraintViolation,
      });
    }

    this.invariantRegistry.set(name, { src, fn });
  };

  /** Drop the graph-level invariant named `name`. Idempotent. */
  public dropInvariant = (name: string): void => {
    this.invariantRegistry.delete(name);
  };

  /** Every declared invariant as `{ name, src }`, sorted by `name`. The compiled
   *  query closure is internal (not exposed). */
  public invariants = (): InvariantInfo[] =>
    [...this.invariantRegistry]
      .map(([name, e]): InvariantInfo => ({ name, src: e.src }))
      .sort((a, b) => a.name.localeCompare(b.name));

  /**
   * `false`-only-fails: a result set VIOLATES an invariant iff any cell is boolean
   * `false`. A `true`, a `null`, a non-boolean value, or an empty result all HOLD.
   * Byte-identical to the Rust `invariant_violated` (`Value::Bool(false)`).
   */
  private invariantViolated = (rows: ReadonlyArray<Record<string, unknown>>): boolean =>
    rows.some((row) => Object.values(row).some((cell) => cell === false));

  /* Event Emitter Proxy */

  public eventsEnabled = (): boolean => {
    return this.emitter.isEnabled();
  };

  public enableEvents = (): void => {
    this.emitter.enable();
  };

  public disableEvents = (): void => {
    this.emitter.disable();
  };

  /** Subscribe to a graph event. Returns an unsubscribe function (call it to detach). */
  public on = <T extends keyof GraphEvents>(
    type: T,
    listener: (event: GraphEvents[T]) => unknown,
  ): (() => void) => this.emitter.on(type, listener);

  public once = <T extends keyof GraphEvents>(
    type: T,
    listener: (event: GraphEvents[T]) => unknown,
  ): void => {
    this.emitter.once(type, listener);
  };

  public emit = <T extends EmitterEvent<any, any>>(event: T): T => {
    // Inside a transaction, buffer instead of dispatching — an emitted event is
    // meant to signal a *committed* write (React reactivity + the sync WriteLog
    // both treat it that way), so staged writes that might roll back must not
    // fire until commit. Flushed on commit; discarded on rollback.
    if (this.txDepth > 0) {
      // Capture the reactive tokens now, while the element is still live — a
      // removal evicts the element (nulls its graph ref) before this event is
      // dispatched at commit, after which its `labels` getter would throw.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (event as any)[TX_TOKENS] = this.tokensOf(event as never);
      this.txEvents.push(event);

      return event;
    }

    return this.emitter.emit(event as never) as T;
  };

  /* Transactions (R-TX) */

  /**
   * Run `fn` as one atomic transaction. Every write inside applies to the live
   * graph immediately (so reads see their own writes), but if `fn` throws — or a
   * deferred constraint check fails at commit — the whole batch rolls back and
   * nothing is observed. On success the writes commit together and their events
   * fire as a single batch. Returns whatever `fn` returns. Nesting joins the
   * outer transaction (flat, savepoint-less): the outermost frame owns
   * commit/rollback.
   *
   * This is the engine-neutral transaction surface — the same mechanism backs
   * GQL and Gremlin (Gremlin has no transaction *language*, only this host API;
   * the ISO GQL `START TRANSACTION`/`COMMIT`/`ROLLBACK` keywords are a thin layer
   * over the same primitives).
   */
  public transaction = <T>(fn: (graph: this) => T): T => {
    this.beginTransaction();

    let result: T;

    try {
      result = fn(this);
    } catch (error) {
      this.rollbackTransaction();

      throw error;
    }

    this.commitTransaction(); // may throw after rolling back if a deferred check fails

    return result;
  };

  /**
   * Lower-level transaction handle mirroring TinkerPop's `graph.tx()`: the
   * transaction opens now; call `commit()` or `rollback()` explicitly. Prefer
   * {@link transaction} for the common auto-managed case.
   */
  public tx = (): { commit: () => void; rollback: () => void } => {
    this.beginTransaction();

    return {
      commit: () => this.commitTransaction(),
      rollback: () => this.rollbackTransaction(),
    };
  };

  /** True while a transaction is open and recording writes (not during a rollback replay). */
  public isTransacting = (): boolean => this.txDepth > 0 && !this.applyingUndo;

  /**
   * Is the active explicit transaction READ ONLY? Set by ISO GQL
   * `START TRANSACTION READ ONLY`, cleared on commit/rollback. The GQL statement
   * executor consults this to reject a write statement in a read-only transaction.
   */
  public isReadOnlyTransaction = (): boolean => this.txReadOnly;

  /** Set/clear the active transaction's READ ONLY access mode (see {@link isReadOnlyTransaction}). */
  public setTransactionReadOnly = (readOnly: boolean): void => {
    this.txReadOnly = readOnly;
  };

  /** Open a transaction frame. Nesting increments depth; the outermost frame owns commit/rollback. */
  public beginTransaction = (): void => {
    this.txDepth += 1;
  };

  /**
   * Close the current frame. The outermost commit runs the deferred constraint
   * checks against the fully-staged graph — on failure it rolls the whole
   * transaction back and throws — then dispatches the buffered events as one batch.
   */
  public commitTransaction = (): void => {
    if (this.txDepth === 0) {
      throw new LenkeError('commit called with no open transaction', {
        code: ErrorCode.InvalidGraphOp,
      });
    }

    this.txDepth -= 1;

    if (this.txDepth > 0) {
      return; // an inner commit — the outermost frame finalizes
    }

    try {
      this.runDeferredChecks();
    } catch (error) {
      this.applyUndoAndReset();

      throw error;
    }

    const events = this.txEvents;
    this.txUndo = [];
    this.txEvents = [];
    this.txTouched.clear();
    this.txTouchedEdges.clear();

    for (const event of events) {
      this.emitter.emit(event as never);
    }
  };

  /** Roll the current transaction back: replay inverse ops in reverse, discard buffered events. Idempotent. */
  public rollbackTransaction = (): void => {
    if (this.txDepth === 0) {
      return;
    }

    this.applyUndoAndReset();
  };

  /** Record an inverse op to replay if the current transaction rolls back (no-op outside a transaction). */
  public recordUndo = (inverse: () => void): void => {
    if (this.txDepth > 0 && !this.applyingUndo) {
      this.txUndo.push(inverse);
    }
  };

  private applyUndoAndReset = (): void => {
    this.applyingUndo = true;

    const undo = this.txUndo;

    // Replay inverse ops newest-first; each reverses exactly one forward write.
    for (let i = undo.length - 1; i >= 0; i -= 1) {
      undo[i]();
    }

    this.applyingUndo = false;
    this.txDepth = 0;
    this.txUndo = [];
    this.txEvents = []; // discard everything buffered (forward writes and undo replay alike)
    this.txTouched.clear();
    this.txTouchedEdges.clear();
  };

  /**
   * Re-run the built-in vertex constraints (required / type / unique) against
   * every vertex touched during the transaction, now that all writes are staged.
   * The per-write gates defer to here so an intermediate state — a node added
   * before its mandatory property, two rows that momentarily collide — doesn't
   * trip a constraint the final state satisfies.
   */
  private runDeferredChecks = (): void => {
    for (const id of this.txTouched) {
      const vertex = this.verticesById.get(id);

      if (!vertex) {
        continue; // added then removed within the transaction — nothing to check
      }

      const missing = this.missingRequired(vertex.labels, vertex.properties);

      if (missing) {
        throw new LenkeError(
          `missing required property '${missing.key}' for label '${missing.label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const badType = this.typeViolation(vertex.labels, vertex.properties);

      if (badType) {
        throw new LenkeError(
          `property '${badType.key}' must be ${badType.expected} on '${badType.label}', got ${badType.got}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const dup = this.uniqueConflict(vertex.labels, vertex.properties, vertex);

      if (dup) {
        throw new LenkeError(
          `unique constraint on '${dup.label}.${dup.key}' violated by value ${JSON.stringify(dup.existing.properties[dup.key])}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      // Cardinality: a vertex lands in `txTouched` when it's added OR when an
      // incident edge is added/removed (either endpoint's degree changed). This
      // is where BOTH bounds are enforced: max (also caught eagerly for a bare
      // addEdge) and min (only satisfiable across writes, so commit-time only —
      // this per-statement / transaction commit is min's single enforcement point).
      const card = this.cardinalityViolationOf(vertex);

      if (card) {
        throw new LenkeError(
          `cardinality constraint on '${card.label}' (${card.edgeType} ${card.direction}) violated: degree out of bounds [${card.min}, ${card.max ?? '∞'}]`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const invalid = this.validatorViolationOf(vertex);

      if (invalid) {
        throw new LenkeError(`validator '${invalid.src}' on '${invalid.label}' violated`, {
          code: ErrorCode.ConstraintViolation,
        });
      }
    }

    // Edge constraints: re-check every edge touched during the transaction against
    // the fully-staged graph (edge analogue of the vertex loop above).
    for (const id of this.txTouchedEdges) {
      const edge = this.edgesById.get(id);

      if (!edge) {
        continue; // added then removed within the transaction — nothing to check
      }

      const missing = this.edgeMissingRequired(edge.labels, edge.properties);

      if (missing) {
        throw new LenkeError(
          `missing required property '${missing.key}' for edge type '${missing.label}'`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const badType = this.edgeTypeViolation(edge.labels, edge.properties);

      if (badType) {
        throw new LenkeError(
          `property '${badType.key}' must be ${badType.expected} on edge type '${badType.label}', got ${badType.got}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const dup = this.edgeUniqueConflict(edge.labels, edge.properties, edge);

      if (dup) {
        throw new LenkeError(
          `unique constraint on edge type '${dup.label}.${dup.key}' violated by value ${JSON.stringify(dup.existing.properties[dup.key])}`,
          { code: ErrorCode.ConstraintViolation },
        );
      }

      const invalid = this.validatorViolationOf(edge);

      if (invalid) {
        throw new LenkeError(`validator '${invalid.src}' on '${invalid.label}' violated`, {
          code: ErrorCode.ConstraintViolation,
        });
      }
    }

    // Graph-level invariants (cross-write assertions): run ONCE against the fully
    // staged graph, AFTER the per-element loops above, but only if this
    // transaction actually wrote something. The undo log is non-empty iff a write
    // was recorded during the frame, so a pure-read commit skips them (no spurious
    // cost/throw). Each invariant is a whole-graph GQL query — VIOLATED iff any
    // returned cell is boolean `false`. A read query run here does NOT re-enter the
    // transaction machinery (reads skip the auto-commit frame), so this is safe.
    if (this.invariantRegistry.size > 0 && this.txUndo.length > 0) {
      for (const [name, entry] of this.invariantRegistry) {
        if (this.invariantViolated(entry.fn(this))) {
          throw new LenkeError(`invariant '${name}' (${entry.src}) violated`, {
            code: ErrorCode.ConstraintViolation,
          });
        }
      }
    }
  };

  /* Internal Methods */

  /**
   * Every edge incident to `id`, in either direction, reconstructed from the
   * directional label indexes (the union of every per-label bucket under the
   * vertex in `edgesFromByLabel` and `edgesToByLabel`). A `Set` dedupes the two
   * directions for self-loops. This is the cascade source for `removeVertex` —
   * it relies on the LPG invariant that every edge carries at least one label
   * (enforced by both the GQL and Gremlin layers), so it appears under some
   * bucket. We keep no separate per-vertex edge index, since vertex removal is
   * the only reader and it is far colder than the edge-insert path that index
   * would tax.
   */
  private readonly incidentEdges = (id: string): Set<Edge> => {
    const incident = new Set<Edge>();

    for (const byLabel of [this.edgesFromByLabel.get(id), this.edgesToByLabel.get(id)]) {
      for (const bucket of byLabel?.values() ?? []) {
        for (const edge of bucket) {
          incident.add(edge);
        }
      }
    }

    return incident;
  };

  private readonly indexVertexLabel = (label: string, vertex: Vertex): void => {
    if (!this.verticesByLabel.has(label)) {
      this.verticesByLabel.set(label, new Set());
    }

    this.verticesByLabel.get(label)?.add(vertex);
  };

  private readonly deIndexVertexLabel = (label: string, vertex: Vertex): void => {
    if (
      this.verticesByLabel.get(label)?.delete(vertex) &&
      this.verticesByLabel.get(label)?.size === 0
    ) {
      this.verticesByLabel.delete(label);
    }
  };

  private readonly indexEdgeLabel = (label: string, edge: Edge) => {
    if (!this.edgesByLabel.has(label)) {
      this.edgesByLabel.set(label, new Set());
    }

    this.edgesByLabel.get(label)!.add(edge);

    if (!this.edgesFromByLabel.has(edge.from.id)) {
      this.edgesFromByLabel.set(edge.from.id, new Map());
    }

    const edgesFrom = this.edgesFromByLabel.get(edge.from.id)!;

    if (!edgesFrom.has(label)) {
      edgesFrom.set(label, new Set());
    }

    edgesFrom.get(label)!.add(edge);

    if (!this.edgesToByLabel.has(edge.to.id)) {
      this.edgesToByLabel.set(edge.to.id, new Map());
    }

    const edgesTo = this.edgesToByLabel.get(edge.to.id)!;

    if (!edgesTo.has(label)) {
      edgesTo.set(label, new Set());
    }

    edgesTo.get(label)!.add(edge);
  };

  private readonly deIndexEdgeLabel = (label: string, edge: Edge) => {
    this.edgesByLabel.get(label)?.delete(edge);

    const fromId = edge.from.id;

    if (
      this.edgesFromByLabel.get(fromId)?.get(label)?.delete(edge) &&
      this.edgesFromByLabel.get(fromId)?.get(label)?.size === 0
    ) {
      this.edgesFromByLabel.get(fromId)?.delete(label);
    }

    const toId = edge.to.id;

    if (
      this.edgesToByLabel.get(toId)?.get(label)?.delete(edge) &&
      this.edgesToByLabel.get(toId)?.get(label)?.size === 0
    ) {
      this.edgesToByLabel.get(toId)?.delete(label);
    }
  };
}
