/**
 * The client half of the v1 protocol — `liveQuery`'s port-crossing shadow.
 *
 * Where {@link createSyncHost} sits next to the store, the client sits next to
 * the UI and speaks the same tagged messages over the same transport seam: you
 * hand it a `send` function and feed inbound messages to `receive`.
 *
 * ```ts
 * // Browser ↔ Worker:
 * const client = createSyncClient({ send: (m) => worker.postMessage(m) });
 * worker.onmessage = (e) => client.receive(e.data);
 *
 * // Browser/Node ↔ server WebSocket:
 * const client = createSyncClient({ send: (m) => ws.send(JSON.stringify(m)) });
 * ws.onmessage = (e) => client.receive(JSON.parse(String(e.data)));
 * ```
 *
 * What it manages (the design doc's client registry):
 * - **Dedupe by query signature** — N local consumers of the same
 *   `(query, params, deps)` share ONE wire subscription. The signature is
 *   forgiving where semantics allow: query text is whitespace/comment-
 *   normalized (values untouched) and deps compare as a set — but case is
 *   never folded (labels/properties are case-sensitive).
 * - **Refcounted unsubscribe** — the wire subscription is torn down when the
 *   last local subscriber leaves; the entry retires into a bounded LRU
 *   (`maxInactiveQueries`) so a quick revival is warm, while a session's worth
 *   of distinct signatures can't grow retained-result memory without bound.
 * - **Referentially-stable snapshots** — `getSnapshot()` returns the same
 *   object until a push replaces it, so `useSyncExternalStore(h.subscribe,
 *   h.getSnapshot)` plugs in directly (no React dependency here, same as
 *   `@lenke/native`'s store).
 * - **Honest loading state** — a snapshot is `{ rows, complete, error? }`;
 *   before the first push `complete` is `false`, so a UI can render skeletons
 *   instead of lying with an empty result.
 * - **Promise-shaped one-shots** — `query()` resolves rows, `mutate()`
 *   resolves on `ack ok` and rejects with the coded error otherwise. The UI
 *   effect of a mutation arrives via subscription pushes, exactly as if
 *   another client had written.
 *
 * Reconnect *policy* is deliberately NOT here: back-off, re-dial, and request
 * parking live in {@link createReconnectingClient}, which composes this client
 * and drives its one seam — {@link SyncClient.replay} re-emits every active
 * subscription and unanswered one-shot over a fresh transport. Resumable
 * subscriptions (server-side cursor catch-up) remain a later protocol
 * extension; replay re-runs standing queries from scratch, which the snapshot
 * model makes correct (the host just re-answers current rows).
 */

import { ErrorCode, LenkeError } from '@lenke/errors';
import {
  composeGremlin as buildGremlin,
  decodeArrow,
  type GremlinLiteral,
  type QueryParams,
  type Row,
} from '@lenke/native';

import {
  isHostMessage,
  keyOf,
  type ClientMessage,
  type RowsMessage,
  type SyncWrite,
  type WireError,
  type WritesMessage,
} from './protocol.js';

/** What a standing query currently knows. Stable reference between pushes. */
export type ClientSnapshot = {
  /** Last pushed rows (a stable `[]` before the first push). Empty for a Gremlin query. */
  rows: Row[];
  /** A `lang: 'gremlin'` subscription's result values; `undefined` for a GQL query. */
  values?: unknown[];
  /** False until the host has answered (and on error) — render skeletons, not lies. */
  complete: boolean;
  /** Graph version at snapshot time, when known. */
  version?: number;
  /** Set when the host closed the subscription with an error. */
  error?: WireError;
};

/** A `useSyncExternalStore`-ready handle over one standing query. */
export type ClientLiveQuery = {
  /** Register a change callback; returns an unsubscribe fn (refcounted). */
  subscribe: (onChange: () => void) => () => void;
  /** Current snapshot — the same reference until a push replaces it. */
  getSnapshot: () => ClientSnapshot;
};

export type SyncClient = {
  /**
   * This client's stable identity — the `clientId` option verbatim, or the
   * random default generated when it was omitted. Threaded into every `mutate`
   * and `subscribeWrites` for exactly-once dedupe and origin-skip. Read it to
   * **persist a generated default** (e.g. to `localStorage`) so a later
   * `createSyncClient` can pass it back and keep origin-skip across a restart.
   */
  readonly clientId: string;
  /** Feed one inbound (already-parsed) host message. Unknown tags are ignored. */
  receive: (msg: unknown) => void;
  /**
   * A standing query. Consumers with the same `(query, params, deps)` share
   * one wire subscription; the wire teardown happens when the last local
   * subscriber unsubscribes.
   */
  liveQuery: (
    query: string,
    opts: {
      /**
       * Dependency posture — **required**: token array (epoch-gated), `[]`
       * (never recomputes), or `null` (recompute on every change). No inference.
       */
      deps: readonly string[] | null;
      params?: QueryParams;
      /** Row-identity column → keyed diff pushes (patch/remove) instead of full rows. */
      key?: string;
      /**
       * `'gremlin'` makes this a standing Gremlin traversal — the snapshot's
       * `values` (not `rows`) carry the result. `key` ignored. No engine param
       * binding — build `query` with the `gremlin` tag / `escapeGremlin`.
       */
      lang?: 'gql' | 'gremlin';
      /**
       * Windowed read for grids (keyless GQL only): the snapshot carries just
       * `rows.slice(offset, offset + limit)`. Scroll by calling `liveQuery`
       * again with a new window (it's a distinct standing query). Pair with
       * `ORDER BY` so the page is stable.
       */
      window?: { offset: number; limit: number };
      /**
       * Demand-fill priority hint (default 0). Higher-priority subscriptions'
       * loads run first when they contend for the engine's load concurrency —
       * a user-facing query can jump ahead of background prefetches.
       */
      priority?: number;
    },
  ) => ClientLiveQuery;
  /**
   * One-shot GQL query → rows. Pass `{ format: 'arrow' }` to fetch the result
   * as a columnar blob and decode it here (smaller wire, no JSON parse) — needs
   * a binary-capable transport; the returned rows are identical either way.
   */
  query: <R extends Row = Row>(
    query: string,
    params?: QueryParams,
    opts?: { format?: 'arrow' },
  ) => Promise<R[]>;
  /**
   * One-shot Gremlin traversal → its JSON result values. Use it as a tagged
   * template to interpolate values safely — each `${v}` is escaped into a
   * Gremlin literal (Gremlin has no param binding), so
   * ``client.gremlin`g.V().has('name', ${userInput}).values('age')` `` is
   * injection-safe. A plain string is sent as-is (you own its safety).
   */
  gremlin: {
    (traversal: string): Promise<unknown[]>;
    (traversal: TemplateStringsArray, ...subs: readonly GremlinLiteral[]): Promise<unknown[]>;
  };
  /**
   * Apply a mutation; resolves on `ack ok`, rejects with the coded error. GQL
   * by default (values ride `params`). `lang: 'gremlin'` sends the text as a
   * Gremlin mutation — this is the replication bridge's seam (`upstream.push`
   * forwards a queued `SyncWrite`'s language here so a Gremlin write doesn't
   * degrade to GQL on the wire); for hand-written Gremlin prefer
   * {@link SyncClient.mutateGremlin}, which escapes interpolated values.
   */
  mutate: (text: string, params?: QueryParams, lang?: 'gql' | 'gremlin') => Promise<void>;
  /**
   * Apply a Gremlin mutation traversal (`addV` / `addE` / `property` / `drop`).
   * Use it as a tagged template to interpolate values safely — each `${v}` is
   * escaped into a Gremlin literal (Gremlin has no param binding), so
   * ``client.mutateGremlin`g.addV('Person').property('name', ${name})` `` is
   * injection-safe. A plain string is sent as-is (you own its safety). Resolves
   * on `ack ok`, rejects with the coded error.
   */
  mutateGremlin: {
    (traversal: string): Promise<void>;
    (traversal: TemplateStringsArray, ...subs: readonly GremlinLiteral[]): Promise<void>;
  };
  /**
   * Replicate a whole queued {@link SyncWrite} upstream — the drop-in for a
   * sync engine's `upstream.push`, so `upstream: { push: client.pushWrite }` is
   * the entire wiring. It forwards the write's `text`, `params`, AND `lang`
   * together, so a Gremlin write can't lose its language and silently degrade
   * to GQL on the wire (the exact footgun of hand-calling
   * `mutate(w.text, w.params)` and forgetting the third argument). A local
   * bulk `ndjson` load is never replicated, so passing one is a programming
   * error and rejects.
   */
  pushWrite: (write: SyncWrite) => Promise<void>;
  /**
   * The host's last `status` message, if any. Carries `pendingWrites` (the
   * write-back backlog). It intentionally does NOT report connectivity: a status
   * message only arrives while the wire is up, so a bound client can never see it
   * flip to offline — use {@link createReconnectingClient}'s `connected()` /
   * `onConnectivity()` for outage detection.
   */
  getStatus: () => { pendingWrites: number } | null;
  /**
   * Subscribe to host `status` pushes (pending-write count); returns an
   * unsubscribe fn. Pairs with {@link getStatus} for a poll-free
   * `useSyncExternalStore(onStatus, getStatus)` — the snapshot reference is
   * stable between pushes.
   */
  onStatus: (cb: () => void) => () => void;
  /** Live wire-subscription count — for tests and debugging. */
  subscriptionCount: () => number;
  /**
   * Opt into the CDC write stream: `onWrites` receives every OTHER client's
   * committed writes, in order, to hand to a local optimistic engine's `ingest`
   * — so cross-client changes appear locally without re-querying. Returns an
   * unsubscribe. `onResync` (optional) fires when the server's op log has moved
   * past this client's cursor (a long disconnect): the local write-stream is
   * stale and the app should cold-boot from a fresh snapshot. `onIngestError`
   * (optional) fires when handing a batch to `onWrites` throws — the batch is
   * isolated so it can't wedge the transport, but ingest isn't atomic yet
   * so a partial apply means the app should cold-boot to re-sync.
   * Survives reconnect (resumes from the last cursor via {@link replay}).
   */
  subscribeWrites: (
    onWrites: (writes: readonly SyncWrite[]) => void,
    opts?: {
      onResync?: () => void;
      onIngestError?: (error: unknown, writes: readonly SyncWrite[]) => void;
      /**
       * Value-scope filter — only receive writes whose content-derived scope
       * intersects these values (e.g. `['42']` for room 42), so a many-room app
       * replicates just this client's rooms. Requires the host to be configured
       * with a `scopeKey`. Survives reconnect (re-sent via {@link replay}).
       */
      scopes?: readonly string[];
    },
  ) => () => void;
  /**
   * Register ephemeral cleanup writes — run by the host when this connection
   * closes (and broadcast over the CDC stream), so a presence node vanishes for
   * everyone on disconnect. Presence itself is a normal `mutate` (an `_MERGE`
   * upsert); this is just the teardown, e.g. `[{ text: 'MATCH (p:Presence {sid:
   * $s}) DETACH DELETE p', params: { s } }]`. Re-registering replaces; survives
   * reconnect (re-sent via {@link replay}).
   */
  onDisconnect: (writes: readonly SyncWrite[]) => void;
  /**
   * Re-emit every active subscription and every unanswered one-shot over the
   * current transport. A reconnect manager calls this once a fresh connection
   * is open: subscribes are idempotent (the host replaces by `sub` id), reads
   * re-run harmlessly, and writes replay at-least-once (host/engine dedupe is
   * the deferred protocol concern). Pending promises are untouched — they
   * settle when the replayed request is answered.
   */
  replay: () => void;
  /** Tear down every subscription and reject every pending request. */
  close: () => void;
};

export type SyncClientOptions = {
  /** Deliver one message to the host. */
  send: (msg: ClientMessage) => void;
  /**
   * How many **wire-inactive** standing queries to retain (default 64), as an
   * LRU. An entry with subscribers is never evicted. A retained inactive entry
   * revives warm — same handle, last rows kept for identity-preserving diffs
   * (StrictMode remounts and back-navigation stay cheap); one evicted past the
   * cap just re-subscribes cold on next use: the host re-answers from its
   * (local) store — a re-query, not a refetch — at the cost of one wholesale
   * re-render. This bounds the registry: without it, a session issuing
   * many distinct `(query, params)` signatures (per-id detail queries,
   * search-as-you-type) grows the retained-result memory without limit.
   */
  maxInactiveQueries?: number;
  /**
   * A **stable identity for this client**, threaded into every `mutate` and
   * `subscribeWrites` message. The host uses it two ways: to dedupe a re-sent
   * write (exactly-once) and — crucially — for **origin-skip**, so the CDC
   * stream never echoes your own writes back to you. Supply a durable value
   * (persisted per device/tab) to keep origin-skip working **across a
   * reconnect**: a fresh host re-learns which origin is "you" from this id, so
   * writes you made before the drop aren't replayed to you as if foreign. If
   * omitted, a random per-instance id is generated — unique, but it changes on
   * every `createSyncClient`, so it can't survive a process restart.
   */
  clientId?: string;
};

const wireToError = (e: WireError): LenkeError =>
  // Wire codes are the shared ErrorCode vocabulary; the cast keeps the
  // ecosystem's one error type without re-validating strings at this layer.
  // Only add the `lenke:` prefix if the origin didn't already (a native/engine
  // LenkeError arrives pre-prefixed) — otherwise it doubles to `lenke: lenke:`.
  new LenkeError(e.message.startsWith('lenke:') ? e.message : `lenke: ${e.message}`, {
    code: e.code as ErrorCode,
  });

/** Stable JSON for the dedupe key: object keys sorted, arrays kept in order. */
const canonical = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map(canonical).join(',')}]`;
  }

  if (typeof value === 'object' && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`);

    return `{${entries.join(',')}}`;
  }

  return JSON.stringify(value) ?? 'null';
};

const QUOTES = new Set(["'", '"', '`']);

/**
 * Whitespace/comment-normalize query text for the dedupe signature ONLY (the
 * wire carries the author's text). Outside quoted regions, whitespace runs —
 * and, for GQL, comments (`//` / `--` line, `/* *​/` block; they are token
 * separators, exactly as the lexer treats them) — collapse to one space, so
 * formatting differences don't mint duplicate entries. Quoted regions
 * (`'…' "…" `…``) copy verbatim, backslash-aware: values and delimited
 * identifiers never change. Gremlin has no comments, so only whitespace
 * normalizes there.
 *
 * Case is deliberately left alone: GQL keywords are case-insensitive, but
 * labels / property names / strings are NOT — folding case could merge two
 * genuinely different queries and serve one of them the other's cached rows.
 * `MATCH` vs `match` not deduping is the safe miss. The same invariant governs
 * the whole function: every edge case degrades to LESS normalization (a missed
 * dedupe), never to a false merge.
 */
const normalizeForSignature = (text: string, lang?: 'gql' | 'gremlin'): string => {
  const gql = lang !== 'gremlin';
  let out = '';
  let i = 0;

  const space = (): void => {
    if (out !== '' && !out.endsWith(' ')) {
      out += ' ';
    }
  };

  while (i < text.length) {
    const c = text[i];

    if (QUOTES.has(c)) {
      // Copy the quoted region verbatim; a backslash escapes the next char.
      out += c;
      i += 1;

      while (i < text.length) {
        const ch = text[i];
        out += ch;
        i += 1;

        if (ch === '\\' && i < text.length) {
          out += text[i];
          i += 1;
        } else if (ch === c) {
          break;
        }
      }

      continue;
    }

    const two = text.slice(i, i + 2);

    if (gql && (two === '//' || two === '--')) {
      while (i < text.length && text[i] !== '\n') {
        i += 1;
      }

      space(); // a comment separates tokens, same as whitespace

      continue;
    }

    if (gql && two === '/*') {
      i += 2;

      while (i < text.length && text.slice(i, i + 2) !== '*/') {
        i += 1;
      }

      i = Math.min(i + 2, text.length);
      space();

      continue;
    }

    if (/\s/.test(c)) {
      while (i < text.length && /\s/.test(text[i])) {
        i += 1;
      }

      space();

      continue;
    }

    out += c;
    i += 1;
  }

  return out.trimEnd();
};

const EMPTY_ROWS: Row[] = [];
const INITIAL: ClientSnapshot = { rows: EMPTY_ROWS, complete: false };

/** Same columns, same values — used to keep a row's identity when a re-push doesn't change it. */
const shallowEqualRow = (a: Row, b: Row): boolean => {
  const keys = Object.keys(a);

  if (keys.length !== Object.keys(b).length) {
    return false;
  }

  return keys.every((k) => Object.is(a[k], b[k]));
};

type Entry = {
  /** The dedupe signature this entry is registered under. */
  signature: string;
  /** Wire sub id — reassigned when a torn-down handle is revived. */
  sub: string;
  /** The subscribe payload, retained so {@link SyncClient.replay} can re-emit it. */
  query: string;
  params?: QueryParams;
  deps: readonly string[] | null;
  /** Row-identity column, if this subscription requested keyed diffs. */
  key?: string;
  /** `'gremlin'` → snapshots carry `values`, applied whole (no keyed diffs). */
  lang?: 'gql' | 'gremlin';
  /** Windowed read (keyless GQL only), retained for replay. */
  window?: { offset: number; limit: number };
  /** Demand-fill priority hint, retained so {@link SyncClient.replay} re-emits it. */
  priority?: number;
  /** Current rows by canonical key — the base each keyed diff is applied onto. */
  rowsByKey?: Map<string, Row>;
  snapshot: ClientSnapshot;
  listeners: Set<() => void>;
  handle: ClientLiveQuery;
};

type Pending = {
  resolve: (value: never) => void;
  reject: (reason: LenkeError) => void;
  kind: 'query' | 'gremlin' | 'mutate';
  /** The exact message sent, retained for replay across a reconnect. */
  msg: ClientMessage;
};

/** Replace an entry's snapshot and wake its subscribers. */
const settle = (entry: Entry, snapshot: ClientSnapshot): void => {
  entry.snapshot = snapshot;

  for (const l of entry.listeners) {
    l();
  }
};

export const createSyncClient = (options: SyncClientOptions): SyncClient => {
  const { send } = options;
  const maxInactive = options.maxInactiveQueries ?? 64;

  const entries = new Map<string, Entry>(); // signature → entry
  // Wire-inactive entries, insertion-ordered = LRU (oldest first). Bounds the
  // registry: entries with subscribers are never here; an entry past the cap is
  // dropped from `entries` so its retained rows can be collected. The data
  // itself lives in the host's store — eviction costs a re-query on revival,
  // never a refetch.
  const inactive = new Set<Entry>();

  const retire = (entry: Entry): void => {
    inactive.delete(entry); // refresh recency if already retired
    inactive.add(entry);

    while (inactive.size > maxInactive) {
      const oldest = inactive.values().next().value as Entry;
      inactive.delete(oldest);

      // Only drop the registration if it's still ours — a stale-handle revival
      // may have re-registered a different entry under this signature.
      if (entries.get(oldest.signature) === oldest) {
        entries.delete(oldest.signature);
      }
    }
  };
  const bySub = new Map<string, Entry>(); // wire sub id → entry
  const pending = new Map<string, Pending>(); // req id → resolver
  let nextId = 0;
  // A stable per-client id — makes each mutate `req` globally unique so the
  // server can dedupe a re-sent write (exactly-once) across a reconnect, where
  // the connection (and per-connection counters) would otherwise reset. It's
  // ALSO the origin-skip key: an explicit `clientId` persisted by the caller
  // keeps the host filtering out our own writes across a reconnect (a fresh host
  // re-learns "us" from it). Omitted → a random per-instance id (unique, but
  // reborn on every construction, so origin-skip resets on a process restart).
  const clientId =
    options.clientId ??
    (globalThis as { crypto?: { randomUUID?: () => string } }).crypto?.randomUUID?.() ??
    `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  let status: { pendingWrites: number } | null = null;
  const statusListeners = new Set<() => void>();

  // CDC write stream: the handler that ingests other clients' writes, the
  // last-applied op cursor (retained across reconnects to resume via `replay`),
  // and the cold-boot hook for when the server's log has moved past the cursor.
  let writeHandler: ((writes: readonly SyncWrite[]) => void) | undefined;
  let writeResync: (() => void) | undefined;
  let writeIngestError: ((error: unknown, writes: readonly SyncWrite[]) => void) | undefined;
  let writeCursor = 0;
  let writesSubscribed = false;
  // Value-scope filter for the CDC stream (retained so `replay` re-sends it).
  let writeScopes: readonly string[] | undefined;
  // Ephemeral cleanup writes registered for this connection (presence teardown);
  // re-sent on reconnect so the fresh host runs them when THAT connection drops.
  let disconnectWrites: readonly SyncWrite[] | undefined;

  const liveQuery = (
    query: string,
    opts: {
      deps: readonly string[] | null;
      params?: QueryParams;
      key?: string;
      lang?: 'gql' | 'gremlin';
      /** Keyless GQL only: fetch a `slice(offset, offset+limit)` page; scroll by re-subscribing with a new window. */
      window?: { offset: number; limit: number };
      priority?: number;
    },
  ): ClientLiveQuery => {
    // Deps are semantically a SET (epoch gating sums them; collection matching
    // is membership), so sort them for the signature: two consumers declaring
    // the same tokens in different orders share one entry. Sorted HERE only —
    // `canonical` must keep arrays ordered in general, because array-valued
    // params are order-significant values. `null` (recompute-always) stays
    // distinct from `[]` (never). The query text is whitespace/comment-
    // normalized the same way (signature only — the wire carries the original).
    const signature = canonical([
      normalizeForSignature(query, opts.lang),
      opts.params ?? null,
      opts.deps === null ? null : [...opts.deps].sort(),
      opts.key ?? null,
      opts.lang ?? null,
      opts.window ?? null, // a different window is a different standing query
    ]);
    const existing = entries.get(signature);

    if (existing) {
      return existing.handle;
    }

    // The entry is the canonical handle for its signature; only its WIRE
    // subscription activates/deactivates ('' = inactive). This makes a
    // subscribe/unsubscribe/subscribe cycle (React StrictMode's mount dance)
    // revive cleanly — fresh wire sub, last snapshot kept as the
    // stale-but-honest starting point — with no duplicate-entry races.
    // Inactive entries are retained in a bounded LRU (`maxInactiveQueries`);
    // one evicted past the cap re-registers itself on its next subscribe.
    const entry: Entry = {
      signature,
      sub: '',
      query,
      params: opts.params,
      deps: opts.deps,
      key: opts.key,
      lang: opts.lang,
      window: opts.window,
      priority: opts.priority,
      snapshot: INITIAL,
      listeners: new Set(),
      handle: {
        subscribe: (onChange) => {
          inactive.delete(entry); // live again — off the eviction path

          // A stale handle can outlive its registration (evicted past the LRU
          // cap): re-register so future liveQuery calls dedupe onto it again.
          // If another entry claimed the signature meanwhile, leave it — two
          // wire subs for one signature is wasteful but correct.
          if (!entries.has(entry.signature)) {
            entries.set(entry.signature, entry);
          }

          if (entry.sub === '') {
            activate();
          }

          entry.listeners.add(onChange);

          return () => {
            entry.listeners.delete(onChange);

            // Last local subscriber gone → tear down the wire subscription and
            // retire the entry into the inactive LRU (bounded keep-warm).
            if (entry.listeners.size === 0) {
              if (entry.sub !== '') {
                bySub.delete(entry.sub);
                send({ type: 'unsubscribe', sub: entry.sub });
                entry.sub = '';
              }

              retire(entry);
            }
          };
        },
        getSnapshot: () => entry.snapshot,
      },
    };

    const activate = (): void => {
      entry.sub = `s${++nextId}`;
      bySub.set(entry.sub, entry);
      // The retained rows (rowsByKey) are KEPT across a re-subscribe: the fresh
      // host re-pushes every row as a full patch, and applyDiff keeps unchanged
      // rows' identity by diffing against this base — so a reconnect (or a
      // StrictMode remount) doesn't churn the whole list. The last snapshot
      // stays on screen as the stale-but-honest starting point meanwhile.
      send({
        type: 'subscribe',
        sub: entry.sub,
        query,
        deps: opts.deps,
        params: opts.params,
        key: opts.key,
        lang: opts.lang,
        window: opts.window,
        priority: opts.priority,
      });
    };

    entries.set(signature, entry);
    activate();

    return entry.handle;
  };

  const query = <R extends Row = Row>(
    text: string,
    params?: QueryParams,
    opts?: { format?: 'arrow' },
  ): Promise<R[]> =>
    new Promise<R[]>((resolve, reject) => {
      const req = `q${++nextId}`;
      const msg: ClientMessage = { type: 'query', req, query: text, params, format: opts?.format };
      pending.set(req, { resolve: resolve as (v: never) => void, reject, kind: 'query', msg });
      send(msg);
    });

  const gremlin = (
    traversal: string | TemplateStringsArray,
    ...subs: unknown[]
  ): Promise<unknown[]> =>
    new Promise<unknown[]>((resolve, reject) => {
      const req = `g${++nextId}`;
      // Tagged-template subs are escaped into safe literals; a plain string
      // passes through (buildGremlin is @lenke/native's `gremlin` composer).
      const text = buildGremlin(traversal, ...subs);
      const msg: ClientMessage = { type: 'query', req, query: text, lang: 'gremlin' };
      pending.set(req, { resolve: resolve as (v: never) => void, reject, kind: 'gremlin', msg });
      send(msg);
    });

  const mutate = (text: string, params?: QueryParams, lang?: 'gql' | 'gremlin'): Promise<void> => {
    if (lang === 'gremlin' && params !== undefined) {
      // Gremlin has no param binding — silently dropping the bindings would
      // run the traversal with unbound `$name` literals. Fail at the call site.
      return Promise.reject(
        new LenkeError(
          'lenke: a gremlin write has no param binding — interpolate values with the gremlin tag',
          { code: ErrorCode.InvalidGraphOp },
        ),
      );
    }

    return new Promise<void>((resolve, reject) => {
      const req = `m-${clientId}-${++nextId}`;
      const msg: ClientMessage = { type: 'mutate', req, text, params, lang, clientId };
      pending.set(req, { resolve: resolve as (v: never) => void, reject, kind: 'mutate', msg });
      send(msg);
    });
  };

  const mutateGremlin = (
    traversal: string | TemplateStringsArray,
    ...subs: unknown[]
  ): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      const req = `m-${clientId}-${++nextId}`;
      // Tagged-template subs are escaped into safe literals; a plain string
      // passes through (buildGremlin is @lenke/native's `gremlin` composer).
      const text = buildGremlin(traversal, ...subs);
      const msg: ClientMessage = { type: 'mutate', req, text, lang: 'gremlin', clientId };
      pending.set(req, { resolve: resolve as (v: never) => void, reject, kind: 'mutate', msg });
      send(msg);
    });

  const pushWrite = (write: SyncWrite): Promise<void> => {
    if (write.ndjson) {
      // A bulk ndjson batch is a local demand-fill load, not a user mutation —
      // it never enters the upstream queue. Reaching here means one was routed
      // to replication by mistake; fail loud rather than send an empty `text`.
      return Promise.reject(
        new LenkeError('lenke: a bulk ndjson load is never replicated upstream', {
          code: ErrorCode.InvalidGraphOp,
        }),
      );
    }

    return mutate(write.text, write.params, write.lang);
  };

  // Apply a keyed diff (patch / remove / order) onto the entry's retained rows.
  // Unchanged rows keep their object identity, so React list reconciliation
  // skips them — including across a reconnect, where the fresh host re-pushes
  // every row as a full patch: a patch that doesn't actually change a row keeps
  // the old object (see the shallow-equal check below).
  const applyDiff = (entry: Entry, msg: RowsMessage): void => {
    const key = entry.key as string;
    const structural =
      (msg.patch?.length ?? 0) > 0 || (msg.remove?.length ?? 0) > 0 || msg.order !== undefined;

    // A complete/version-only push carries no ops: keep the same rows array so
    // the reference stays stable, and just refresh the flags.
    if (!structural) {
      settle(entry, {
        rows: entry.snapshot.rows,
        complete: msg.complete ?? true,
        version: msg.version,
      });

      return;
    }

    const map = entry.rowsByKey ?? new Map<string, Row>();
    entry.rowsByKey = map;

    for (const kv of msg.remove ?? []) {
      map.delete(keyOf(kv));
    }

    for (const p of msg.patch ?? []) {
      const ck = keyOf(p.key);
      const prev = map.get(ck);
      const merged = prev ? { ...prev, ...p.set } : { ...p.set };
      // Reuse the prior object when the patch changed nothing (a reconnect
      // re-push of an untouched row) so its identity survives the reconnect.
      map.set(ck, prev && shallowEqualRow(prev, merged) ? prev : merged);
    }

    if (msg.order !== undefined) {
      // Rebuild in the given key order, then re-key the base to exactly those
      // rows — a fresh host after a reconnect sends the current `order` but no
      // `remove`s, so rows that vanished during the outage are pruned here.
      const rows = msg.order
        .map((kv) => map.get(keyOf(kv)))
        .filter((r): r is Row => r !== undefined);
      entry.rowsByKey = new Map(rows.map((r) => [keyOf(r[key]), r]));
      settle(entry, { rows, complete: msg.complete ?? true, version: msg.version });

      return;
    }

    // No `order` (a pure cell change): keep the prior order, swap in updated rows.
    const rows = entry.snapshot.rows.map((r) => map.get(keyOf(r[key])) ?? r);
    settle(entry, { rows, complete: msg.complete ?? true, version: msg.version });
  };

  // Apply a CDC `writes` batch: cold-boot on a resync/gap/reorder, skip a
  // stale/duplicate, otherwise advance the cursor and ingest. Split out of
  // `receive` to keep that dispatcher's branching in check.
  const handleWrites = (msg: WritesMessage): void => {
    if (msg.resync) {
      // The op log moved past us: adopt the resume point and cold-boot.
      writeCursor = msg.cursor;
      writeResync?.();

      return;
    }

    // Idempotence guard: a batch at or below the cursor was already applied (a
    // duplicate, or a stale echo from before a reconnect). Skip it; never regress.
    if (msg.cursor <= writeCursor) {
      return;
    }

    // Gap/reorder guard: `from` is the cursor this batch contiguously follows. If
    // it doesn't match ours, a live-tail message was lost or delivered out of
    // order (a non-FIFO transport / reordered send) — applying it would silently
    // skip the missing writes, so cold-boot instead. `from === undefined` (an
    // older host) disables the check, keeping the prior in-order assumption.
    if (msg.from !== undefined && msg.from !== writeCursor) {
      writeCursor = msg.cursor;
      writeResync?.();

      return;
    }

    // Advance even on an empty batch (own-origin cursor ticks) so a later resume
    // asks from the right place — and PAST a failed batch (a poison op is usually
    // deterministic, so holding the cursor would replay it forever). Move on and
    // surface the error instead.
    writeCursor = msg.cursor;

    if (msg.writes.length > 0) {
      // Isolate ingest: an un-appliable write must not escape `receive()` and
      // wedge the transport pump. Since ingest isn't atomic yet the batch
      // may have partially applied, so the app should cold-boot to re-sync.
      try {
        writeHandler?.(msg.writes);
      } catch (error) {
        writeIngestError?.(error, msg.writes);
      }
    }
  };

  const receive = (msg: unknown): void => {
    if (!isHostMessage(msg)) {
      return; // forward-compat: unknown tags fall through silently
    }

    switch (msg.type) {
      case 'rows': {
        const entry = bySub.get(msg.sub);

        if (!entry) {
          return; // a push that raced our unsubscribe — drop it
        }

        if (msg.error) {
          if (msg.retryable) {
            // A demand-fill LOAD failed (the standing query is fine). Surface the
            // error but KEEP the subscription and its warm rows — the next
            // successful load clears it with a fresh push. Tearing down here
            // would drop rows and force a re-subscribe on every transient blip.
            settle(entry, {
              ...entry.snapshot,
              complete: msg.complete ?? entry.snapshot.complete,
              error: msg.error,
            });

            return;
          }

          // The host closed this subscription: surface the error and go wire-
          // inactive. The handle stays canonical — a later subscribe retries.
          bySub.delete(msg.sub);
          entry.sub = '';
          entry.rowsByKey = undefined;
          settle(entry, { rows: EMPTY_ROWS, complete: false, error: msg.error });

          return;
        }

        if (entry.lang === 'gremlin') {
          // Gremlin pushes carry full `values` (no rows, no diffs) each time.
          settle(entry, {
            rows: EMPTY_ROWS,
            values: msg.values ?? [],
            complete: msg.complete ?? true,
            version: msg.version,
          });

          return;
        }

        if (entry.key !== undefined) {
          applyDiff(entry, msg); // keyed subscription → diff push

          return;
        }

        settle(entry, {
          rows: msg.rows ?? EMPTY_ROWS,
          complete: msg.complete ?? true,
          version: msg.version,
        });

        return;
      }
      case 'result': {
        const p = pending.get(msg.req);

        if (p) {
          pending.delete(msg.req);

          if (msg.error) {
            p.reject(wireToError(msg.error));
          } else if (p.kind === 'gremlin') {
            (p.resolve as (values: unknown[]) => void)(msg.values ?? []);
          } else if (msg.arrow) {
            // format: 'arrow' → decode the columnar blob to rows. A decode
            // failure (e.g. a JSON transport mangled the Uint8Array into an
            // object) must REJECT the promise, not throw out of receive() and
            // leave it hanging forever.
            try {
              (p.resolve as (rows: Row[]) => void)(decodeArrow(msg.arrow));
            } catch {
              p.reject(
                new LenkeError('lenke: could not decode arrow result — needs a binary transport', {
                  code: ErrorCode.Ffi,
                }),
              );
            }
          } else {
            (p.resolve as (rows: Row[]) => void)(msg.rows ?? []);
          }
        }

        return;
      }
      case 'ack': {
        const p = pending.get(msg.req);

        if (p) {
          pending.delete(msg.req);

          if (msg.ok) {
            (p.resolve as () => void)();
          } else {
            // A not-ok ack without a report is itself a boundary fault.
            p.reject(
              msg.error
                ? wireToError(msg.error)
                : new LenkeError('lenke: mutate failed', { code: ErrorCode.Ffi }),
            );
          }
        }

        return;
      }
      case 'status': {
        // A fresh object only on an actual push, so getStatus() stays a stable
        // reference between messages (useSyncExternalStore-safe).
        status = { pendingWrites: msg.pendingWrites };

        for (const l of statusListeners) {
          l();
        }

        return;
      }
      case 'writes':
        handleWrites(msg);

        return;
      default:
    }
  };

  return {
    clientId,
    receive,
    liveQuery,
    query,
    gremlin,
    mutate,
    mutateGremlin,
    pushWrite,
    getStatus: () => status,
    onStatus: (cb) => {
      statusListeners.add(cb);

      return () => statusListeners.delete(cb);
    },
    subscriptionCount: () => bySub.size,
    subscribeWrites: (onWrites, opts) => {
      writeHandler = onWrites;
      writeResync = opts?.onResync;
      writeIngestError = opts?.onIngestError;
      writeScopes = opts?.scopes;
      writesSubscribed = true;
      send({ type: 'subscribeWrites', since: writeCursor, clientId, scopes: writeScopes });

      return () => {
        writeHandler = undefined;
        writeResync = undefined;
        writeIngestError = undefined;
        writesSubscribed = false;
      };
    },
    onDisconnect: (writes) => {
      disconnectWrites = writes;
      send({ type: 'onDisconnect', writes: [...writes] });
    },
    replay: () => {
      // Re-subscribe every active standing query (inactive entries — no local
      // subscribers — stay silent), then re-send every unanswered one-shot. The
      // retained rows (rowsByKey) are KEPT: the fresh host re-pushes full rows
      // and applyDiff preserves unchanged-row identity against this base, so a
      // reconnect after a long sleep catches up without re-rendering the world.
      for (const entry of bySub.values()) {
        send({
          type: 'subscribe',
          sub: entry.sub,
          query: entry.query,
          params: entry.params,
          deps: entry.deps,
          key: entry.key,
          lang: entry.lang,
          window: entry.window,
          priority: entry.priority,
        });
      }

      for (const p of pending.values()) {
        send(p.msg);
      }

      // Resume the CDC write stream from the last applied cursor — the host
      // replays the op tail after it, or answers `resync` if we've fallen off.
      // `clientId` re-identifies us to the FRESH host so origin-skip filters our
      // own writes out of the replayed backlog (the reconnect re-apply bug).
      if (writesSubscribed) {
        send({ type: 'subscribeWrites', since: writeCursor, clientId, scopes: writeScopes });
      }

      // Re-register the ephemeral cleanup so the fresh host tears it down if THIS
      // connection drops.
      if (disconnectWrites) {
        send({ type: 'onDisconnect', writes: [...disconnectWrites] });
      }
    },
    close: () => {
      for (const entry of bySub.values()) {
        send({ type: 'unsubscribe', sub: entry.sub });
        entry.sub = '';
      }

      entries.clear();
      bySub.clear();
      inactive.clear();

      // The transport seam is gone from under these requests — a boundary fault.
      const closing = new LenkeError('lenke: client closed', { code: ErrorCode.Ffi });

      for (const p of pending.values()) {
        p.reject(closing);
      }

      pending.clear();
      statusListeners.clear();
    },
  };
};
