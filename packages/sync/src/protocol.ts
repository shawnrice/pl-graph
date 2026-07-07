/**
 * The lenke live-query wire protocol, v1.
 *
 * The frontend asks **declaratively**: the primitive is a standing query, not a
 * fetch. All messages are tagged plain data — nothing callable ever crosses —
 * so the same protocol rides any port-shaped transport: a Worker `postMessage`
 * channel in the browser, a WebSocket to a server host. Conformance is
 * **structural**: a host conforms by emitting these tags; consumers may write
 * the same shapes down independently with no dependency in either direction.
 *
 * v1 is deliberately brutal-minimal (~6 messages). Arrow-buffer negotiation and
 * resumable subscriptions remain extensions; **keyed row diffs** have landed as
 * a backward-compatible one (declare `key` on subscribe → patch/remove/order
 * pushes; without it, the full-`rows` v1 shape is unchanged).
 *
 * ```
 * client → host:  subscribe   { sub, query, deps, params?, key?, window? }
 * host → client:  rows        { sub, rows | (patch, remove, order), version, complete }  // now, then on change
 * client → host:  unsubscribe { sub }
 * client → host:  query       { req, query, params?, lang? }      // one-shot (gql | gremlin)
 * host → client:  result      { req, rows | values | error }
 * client → host:  mutate      { req, text, lang?, params? }   // gql | gremlin
 * host → client:  ack         { req, ok, error? }                // UI effect arrives via rows pushes
 * host → client:  status      { connected, pendingWrites, protocol }
 * ```
 *
 * `params` everywhere is a flat object of `$name` bindings. Values bind to
 * already-parsed param slots engine-side and never touch the GQL parser — the
 * wire's injection-safety contract. Send values as params; never build query
 * text from user input.
 */

import { isLenkeError } from '@lenke/errors';
import {
  applySchemaOp,
  type QueryParams,
  type Row,
  type RustGraph,
  type SchemaOp,
} from '@lenke/native';

// `SchemaOp` + `applySchemaOp` are native primitives (they describe / dispatch a
// `RustGraph`'s own schema methods, and pair with `RustGraph.dumpSchema`). Re-
// exported here so wire consumers keep importing them from `@lenke/sync`.
export { applySchemaOp, type SchemaOp } from '@lenke/native';

/** A failure crossing the wire: the stable code is the contract, the message is free to change. */
export type WireError = {
  code: string;
  message: string;
};

/** Shape any thrown failure into the wire's coded-error contract. */
export const toWireError = (e: unknown): WireError => {
  if (isLenkeError(e)) {
    return { code: e.code, message: e.message };
  }

  return { code: 'Unknown', message: e instanceof Error ? e.message : String(e) };
};

/** One replicable write: query text, its language, and (GQL only) `$name` bindings. */
export type SyncWrite = {
  /** Query text — GQL DML, or a Gremlin mutation traversal when `lang: 'gremlin'`. */
  text: string;
  /**
   * Language, default `'gql'`. `'gremlin'` executes `text` through the Gremlin
   * engine (mutation steps: `addV` / `addE` / `property` / `drop`). Gremlin has
   * no param binding — pre-escape values with the `gremlin` tag and leave
   * `params` unset.
   */
  lang?: 'gql' | 'gremlin';
  /** `$name` bindings (GQL only). */
  params?: QueryParams;
  /**
   * A bulk NDJSON payload — a `COPY FROM` into the graph, applied via
   * `mergeNdjson` at bulk speed instead of a per-record `INSERT`. When set, `text`
   * is ignored (use `''`). Intended for LOCAL demand-fill (populating a cache
   * from a batch of rows), not for wire-replicated user mutations, since bytes
   * don't cross the JSON protocol.
   */
  ndjson?: Uint8Array;
  /**
   * A schema declaration to (re)apply instead of a data write. When set, `text`
   * is ignored (use `''`). This is how constraints/validators/invariants/indexes
   * replicate across the CDC log — see [`SchemaOp`].
   */
  schema?: SchemaOp;
};

/**
 * Apply one write to a graph: GQL via `query`, Gremlin via `gremlin`, or a schema
 * declaration via [`applySchemaOp`]. The ONE write-language dispatch — the
 * engine's loop and the host's default `applyMutation` both route through it, so a
 * new language (or a per-language step) lands in exactly one place. Lives here
 * because both import protocol and host can't import engine (cycle).
 */
export const runWrite = (g: RustGraph, w: SyncWrite): void => {
  if (w.schema) {
    applySchemaOp(g, w.schema);

    return;
  }

  if (w.ndjson) {
    // Bulk COPY FROM — the fast demand-fill path. Surface anything that didn't
    // land cleanly (a rare event for disjoint batches) so it isn't swallowed.
    const report = g.mergeNdjson(w.ndjson);

    if (report.nodesSkipped.length || report.edgesSkipped.length || report.phantomVertices.length) {
      // eslint-disable-next-line no-console -- dev-visible signal, matching the store's other diagnostics
      console.warn(
        `lenke: mergeNdjson had conflicts — ${report.nodesSkipped.length} node id(s), ` +
          `${report.edgesSkipped.length} edge id(s) skipped; ${report.phantomVertices.length} phantom endpoint(s).`,
      );
    }

    return;
  }

  if (w.lang === 'gremlin') {
    g.gremlin(w.text);
  } else {
    g.query(w.text, w.params);
  }
};

// ---------------------------------------------------------------------------
// client → host
// ---------------------------------------------------------------------------

/**
 * Open a standing query. The host answers with a `rows` push immediately, then
 * again whenever a mutation moves one of the query's dependency epochs.
 * Re-subscribing with the same `sub` replaces the subscription (that is how a
 * windowed grid scrolls: same `sub`, new `window`).
 */
export type SubscribeMessage = {
  type: 'subscribe';
  /** Client-chosen subscription id, unique per connection. */
  sub: string;
  /** GQL text (values belong in `params`, not in the text). */
  query: string;
  /** `$name` bindings — part of the standing query's identity. */
  params?: QueryParams;
  /**
   * Dependency posture for epoch-gated invalidation — **required**, declared
   * explicitly (no silent omission, no host-side inference):
   * - `[...]` — recompute only when one of these label / edge-type /
   *   property-key epochs moves.
   * - `[]` — depends on nothing → never recomputes (a constant query).
   * - `null` — depends on everything → recompute on every change.
   *
   * A client that wants inference derives the array itself with
   * `inferDeps(query)` before subscribing.
   *
   * **Demand-fill routes off these tokens.** A query over a demand-fill
   * collection MUST list that collection's labels here, or it will neither
   * load nor gate on it: `[]`/`null` match no collection, so the host reports
   * `complete: true` immediately over whatever is already local (no skeleton,
   * no load). Recompute-always (`null`) and demand-fill are therefore mutually
   * exclusive — declare the labels to get demand-fill.
   */
  deps: readonly string[] | null;
  /**
   * Row-identity column for **keyed diffs**. When present, the value of this
   * column identifies a row across pushes, so the host sends only what changed
   * (`patch` / `remove` / `order`) instead of the whole result each time. When
   * absent, every push carries the full `rows` — the shape a keyless query
   * (aggregates) needs, and the only one a minimal v1 consumer must understand.
   * Ignored for `lang: 'gremlin'` (Gremlin results aren't keyed rows).
   */
  key?: string;
  /**
   * Query language, default `'gql'`. `'gremlin'` makes this a standing Gremlin
   * traversal: pushes carry `values` (arbitrary JSON) instead of `rows`, full
   * each time (no keyed diffs). `deps` gating works identically. Gremlin has no
   * engine param binding — build the text with the `gremlin` tag / `escapeGremlin`
   * to interpolate values safely.
   */
  lang?: 'gql' | 'gremlin';
  /**
   * Windowed read for grids: the host sends only `rows.slice(offset, offset +
   * limit)` of the result on each push (a keyless subscription — `key`/keyed
   * diffs and `lang: 'gremlin'` ignore it). Scroll by re-subscribing the same
   * `sub` with a new window; pair with `ORDER BY` so the slice is stable.
   * `complete` still reflects the whole scope, not the page.
   */
  window?: { offset: number; limit: number };
  /**
   * Demand-fill priority hint. Higher runs first when loads contend for the
   * scheduler's limited concurrency (default 0). Routed to the load job so an
   * app-supplied `loadScheduler` — or the default priority-ordered gate — can
   * front-run a user-facing query ahead of background prefetches.
   */
  priority?: number;
};

/** Tear down a standing query. */
export type UnsubscribeMessage = {
  type: 'unsubscribe';
  sub: string;
};

/** One-shot query (loaders, event handlers) — answered once with `result`. */
export type QueryMessage = {
  type: 'query';
  /** Client-chosen request id. */
  req: string;
  query: string;
  /** `$name` bindings. Ignored for `lang: 'gremlin'` (Gremlin has no param binding). */
  params?: QueryParams;
  /**
   * Query language, default `'gql'`. `'gremlin'` runs the text through the
   * Gremlin engine and answers with `values` (arbitrary JSON) instead of
   * `rows`. Gremlin has no engine param binding, so interpolate values with the
   * `gremlin` tag / `escapeGremlin` (they escape into safe literals) rather than
   * string concatenation. (`client.gremlin` as a tagged template does this for
   * you.)
   */
  lang?: 'gql' | 'gremlin';
  /**
   * Result encoding, default `'json'`. `'arrow'` answers with an `arrow`
   * columnar blob instead of `rows` — smaller on the wire and decodable without
   * a JSON parse, worth it for large one-shot loads. Requires a **binary-capable
   * transport** (a Worker `MessagePort`, where the `Uint8Array` also transfers
   * zero-copy, or a binary WebSocket); a JSON-stringifying transport must keep
   * `'json'`. GQL only — Gremlin (`lang: 'gremlin'`) answers with `values`.
   */
  format?: 'json' | 'arrow';
};

/**
 * Apply a mutation. The host answers `ack` with only success/failure — the UI
 * effect arrives through `rows` pushes on whichever subscriptions the mutation
 * touched, exactly as if another client had written.
 */
export type MutateMessage = {
  type: 'mutate';
  req: string;
  /**
   * Mutating query text. GQL (`INSERT` / `SET` / `REMOVE` / `DELETE`) with values
   * in `params`, or — with `lang: 'gremlin'` — a Gremlin mutation traversal
   * (`addV` / `addE` / `property` / `drop`) with values pre-escaped via the
   * `gremlin` tag.
   */
  text: string;
  /**
   * Query language, default `'gql'`. `'gremlin'` executes `text` through the
   * Gremlin engine; Gremlin has no engine param binding, so `params` is ignored —
   * interpolate values with the `gremlin` tag / `escapeGremlin`.
   */
  lang?: 'gql' | 'gremlin';
  /** `$name` bindings (GQL only). */
  params?: QueryParams;
  /**
   * The committing client's **stable id** (the same id embedded in `req`). The
   * host tags this write's op-log entry with it so origin-skip can filter the
   * write out of the author's OWN CDC backlog — even across a reconnect, where a
   * fresh host would otherwise mint a new per-connection origin and echo the
   * write back to its author. Absent (a legacy client) → a per-connection
   * fallback (echo-skip within one connection only).
   */
  clientId?: string;
};

/**
 * Opt into the CDC write stream: after this, the host pushes every OTHER
 * client's committed write as a {@link WritesMessage}, so a local optimistic
 * engine can `ingest` them (cross-client optimism over a socket). `since`
 * requests catch-up — replay the ops after that cursor from a prior session;
 * omit (or `0`) to start from the current head. If `since` has fallen off the
 * server's retained op log, the host answers with a `resync` writes message and
 * the client must cold-boot from a snapshot.
 */
export type SubscribeWritesMessage = {
  type: 'subscribeWrites';
  since?: number;
  /**
   * The subscribing client's **stable id** — so the host knows which op-log
   * entries are this client's own and skips them in the backlog + live tail.
   * Stable across reconnect, so a re-dialed client never re-ingests a write it
   * already applied optimistically. Absent (a legacy client) → per-connection
   * fallback (the prior behavior; breaks on reconnect).
   */
  clientId?: string;
  /**
   * Value-scope filter — the set of scope-key values this client cares about
   * (e.g. `['42']` for room 42). When set, the host forwards only writes whose
   * content-derived value-scope intersects it, so a many-room app replicates just
   * the client's rooms rather than the whole graph. Requires the host to be
   * configured with a `scopeKey` (the property that names the scope, e.g.
   * `'room'`); without one, or omitted here, the stream is unscoped (every write,
   * subject to token routing). Interest routing, not a security boundary — the
   * host may still enforce/derive scopes from auth.
   */
  scopes?: readonly string[];
};

/**
 * Register the **ephemeral cleanup** for THIS connection: writes the host runs
 * when the connection closes (and broadcasts over the CDC stream), so a client's
 * presence node vanishes for everyone the moment it drops. Presence itself is a
 * normal `_MERGE` upsert; this is just the tear-down. Re-registering replaces
 * the prior set; the client re-sends it on reconnect. See the README on presence.
 */
export type OnDisconnectMessage = {
  type: 'onDisconnect';
  writes: SyncWrite[];
};

export type ClientMessage =
  | SubscribeMessage
  | UnsubscribeMessage
  | QueryMessage
  | MutateMessage
  | SubscribeWritesMessage
  | OnDisconnectMessage;

// ---------------------------------------------------------------------------
// host → client
// ---------------------------------------------------------------------------

/** One upsert in a keyed diff: the row's key value + the columns that changed
 *  (all columns when the row is new). The client merges `set` into the prior row. */
export type RowPatch = {
  key: unknown;
  set: Row;
};

/**
 * A subscription push: the current result of a standing query. Sent once on
 * subscribe and again on every relevant change. On a subscription failure
 * (bad query, rejected params) this carries `error` instead of rows, and the
 * subscription is closed.
 *
 * Two shapes, chosen by whether the subscription declared a `key`:
 * - **Keyless** — `rows` carries the full result every push (v1's only shape).
 * - **Keyed diff** — `patch` (upserts, changed cells only), `remove` (gone key
 *   values), and `order` (the full key order, present only when it changed);
 *   the client applies them onto its retained rows. `rows` is absent.
 */
export type RowsMessage = {
  type: 'rows';
  sub: string;
  /** Full result (keyless subscriptions). Absent on keyed-diff pushes. */
  rows?: Row[];
  /** Keyed diff: rows to upsert (changed cells only; all cells when new). */
  patch?: RowPatch[];
  /** Keyed diff: key values whose rows were removed. */
  remove?: unknown[];
  /** Keyed diff: the full key order after applying — sent only when it changed. */
  order?: unknown[];
  /** A `lang: 'gremlin'` subscription's result values (arbitrary JSON), full each push. */
  values?: unknown[];
  /** The graph's mutation version at snapshot time. */
  version?: number;
  /**
   * Whether the underlying scope is fully loaded — so a query over a
   * partially-synced collection can distinguish "no results" from "not loaded
   * yet" and the UI can render skeletons honestly. When the host is driven by a
   * {@link createSyncEngine}, this reflects real per-collection completeness
   * (`empty`/`loading`/`complete`) and flips as demand-fill lands. A **bare**
   * host (no engine, a local-only store) is trivially `complete: true`.
   */
  complete?: boolean;
  error?: WireError;
  /**
   * How the client should treat `error`. `true` — a demand-fill LOAD failed
   * (the standing query is fine, the fetch for its data errored): surface the
   * error but keep the subscription and its warm rows; a later successful load
   * clears it. `false`/absent — the standing query itself failed: the host
   * closed the subscription (the client goes wire-inactive; a re-subscribe
   * retries). Set by the engine, which knows whether it caught a loader vs a
   * query-execution failure — not inferred from the code.
   */
  retryable?: boolean;
};

/** The single answer to a one-shot `query`. */
export type ResultMessage = {
  type: 'result';
  req: string;
  /** Rows for a GQL query. */
  rows?: Row[];
  /** ARW1 columnar blob for a `format: 'arrow'` query — the client decodes it to rows. */
  arrow?: Uint8Array;
  /** Result values for a `lang: 'gremlin'` query (arbitrary JSON). */
  values?: unknown[];
  error?: WireError;
};

/** The answer to a `mutate` — success/failure only; data flows via `rows`. */
export type AckMessage = {
  type: 'ack';
  req: string;
  ok: boolean;
  error?: WireError;
};

/** Host status — sent on attach; a built-in subscription in later versions. */
export type StatusMessage = {
  type: 'status';
  connected: boolean;
  pendingWrites: number;
  /** Protocol version for forward-compat negotiation. */
  protocol: 1;
};

/**
 * A CDC push: committed writes from OTHER clients, in seq order, for a local
 * engine to `ingest`. `cursor` is the seq of the last write here — the client
 * retains it as its resume point across reconnects. `resync: true` (with no
 * writes) means the requested `since` fell off the server's retained op log: the
 * local write-stream is unrecoverable and the client must cold-boot from a
 * fresh snapshot.
 */
export type WritesMessage = {
  type: 'writes';
  writes: SyncWrite[];
  cursor: number;
  /**
   * The cursor this batch contiguously FOLLOWS (the last cursor the host sent
   * this client). A client compares it to its own cursor to detect a gap or a
   * reordered message on the live tail — `from !== myCursor` means a batch was
   * lost or delivered out of order, so it cold-boots (resyncs) rather than
   * silently skipping the missing writes. Optional for back-compat: a host that
   * omits it (older, or none configured) simply disables the check, and the tail
   * falls back to assuming in-order delivery. Own-origin/uninterested writes tick
   * the cursor via empty batches, so `from` still chains contiguously.
   */
  from?: number;
  resync?: boolean;
};

export type HostMessage = RowsMessage | ResultMessage | AckMessage | StatusMessage | WritesMessage;

export type SyncMessage = ClientMessage | HostMessage;

// ---------------------------------------------------------------------------
// guards
// ---------------------------------------------------------------------------

const CLIENT_TYPES = new Set([
  'subscribe',
  'unsubscribe',
  'query',
  'mutate',
  'subscribeWrites',
  'onDisconnect',
]);
const HOST_TYPES = new Set(['rows', 'result', 'ack', 'status', 'writes']);

/** Cheap structural gate: is this a tagged message at all? (Not a validator.) */
const tagOf = (msg: unknown): string | null => {
  if (typeof msg !== 'object' || msg === null) {
    return null;
  }

  const t = (msg as { type?: unknown }).type;

  return typeof t === 'string' ? t : null;
};

export const isClientMessage = (msg: unknown): msg is ClientMessage => {
  const t = tagOf(msg);

  return t !== null && CLIENT_TYPES.has(t);
};

export const isHostMessage = (msg: unknown): msg is HostMessage => {
  const t = tagOf(msg);

  return t !== null && HOST_TYPES.has(t);
};

/**
 * Canonical, collision-free string for a keyed row's key value. The host (diff
 * computation) and client (diff application) MUST agree byte-for-byte, so this
 * lives in the shared protocol module rather than being copied into each. An
 * absent key column (`undefined`) is kept distinct from a literal `null`.
 *
 * The key column must be **present and unique per row**; multiple rows sharing
 * one key value (e.g. several `null`s) collapse to a single row in the diff.
 */
export const keyOf = (value: unknown): string => {
  // JSON.stringify returns the value `undefined` for an undefined input, and a
  // JSON text (quoted string / number / null / …) otherwise. The bare sentinel
  // `undefined` can never be a JSON text, so it can't collide with a real key.
  const s = JSON.stringify(value);

  return s ?? 'undefined';
};
