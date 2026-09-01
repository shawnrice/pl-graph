/**
 * A reconnecting {@link SyncClient} — connection lifecycle around the v1 client.
 *
 * The v1 client (see {@link createSyncClient}) is bound to one transport: it
 * cannot outlive a dropped socket, and its pending requests reject when the
 * transport closes. That is the right primitive, but a real app on a flaky
 * network wants one durable handle whose live queries survive an outage and
 * whose writes wait for the wire to come back. This wraps the client with that
 * policy and nothing more:
 *
 * - **Re-dial with back-off** — a `connect` callback establishes one transport;
 *   on close the manager re-dials with exponential back-off (reset on success).
 * - **Re-subscribe on reconnect** — every active standing query is re-emitted
 *   against the fresh transport via {@link SyncClient.replay}; the host
 *   re-answers current rows. Snapshots hold their last value meanwhile (warm,
 *   marked stale by `connected()` / the status line), so the UI never blanks.
 * - **Park, don't reject** — a `query` / `mutate` issued while offline is held;
 *   `replay` re-sends it once reconnected, and the original promise settles on
 *   the eventual answer.
 *
 * **This assumes a durable engine sits behind it.** The manager gives you
 * at-least-once delivery (a write whose ack was lost on a dying socket replays
 * and may apply twice); exactly-once needs server-side request-id dedupe, a
 * protocol concern this layer does not own. The persisted write-back queue that
 * makes an outage lossless lives in {@link createSyncEngine}, not here — this is
 * connection lifecycle, not durability.
 *
 * ```ts
 * const client = createReconnectingClient({
 *   connect: ({ opened, received, closed }) => {
 *     const ws = new WebSocket(url);
 *     ws.onopen = opened;
 *     ws.onmessage = (e) => received(JSON.parse(String(e.data)));
 *     ws.onclose = closed;
 *     ws.onerror = () => ws.close();
 *     return { send: (m) => ws.send(JSON.stringify(m)), close: () => ws.close() };
 *   },
 * });
 * ```
 */

import { createSyncClient, type SyncClient } from './client.js';
import type { ClientMessage, HostMessage } from './protocol.js';

/** One live transport, from the manager's point of view. */
export type ReconnectingConnection = {
  /** Post one client message over this transport. */
  send: (msg: ClientMessage) => void;
  /** Abandon this transport (the manager calls this on teardown). */
  close: () => void;
};

/**
 * Establish one transport, wiring its lifecycle to the manager's handlers.
 * Called once per connection attempt. `opened` is expected to fire
 * asynchronously (as every real socket does); `closed` covers both clean close
 * and error and triggers a re-dial.
 */
export type ReconnectingConnect = (handlers: {
  /** The transport is open and ready to carry messages. */
  opened: () => void;
  /** One inbound host message arrived, already parsed to an object. */
  received: (msg: HostMessage) => void;
  /** The transport closed or errored — the manager will re-dial. */
  closed: () => void;
}) => ReconnectingConnection;

export type ReconnectingClientOptions = {
  connect: ReconnectingConnect;
  /**
   * Re-dial back-off. Delay is `min(maxMs, baseMs * 2 ** attempt)`, reset to
   * attempt 0 on every successful open. Defaults: `baseMs` 500, `maxMs` 5000.
   */
  retry?: { baseMs?: number; maxMs?: number };
  /**
   * This client's stable identity, passed through to the inner client. Crucial for
   * multiplayer over a reconnecting transport: the SAME `clientId` re-attaches on
   * every re-dial, so the host's origin-skip and write-stream dedupe hold ACROSS a
   * reconnect (a re-sent write isn't echoed back as if from a new peer). Omit for a
   * per-connection random id (fine for a pure read/live-query client).
   */
  clientId?: string;
  /**
   * Passed through to the inner client (which lives for this manager's whole
   * life, so the bound matters most here): how many wire-inactive standing
   * queries to keep warm. Default 64.
   */
  maxInactiveQueries?: number;
};

/**
 * The client surface plus connectivity. Includes the full CDC/multiplayer surface
 * — `clientId`, `subscribeWrites`, `onDisconnect` — which survive reconnect via the
 * manager's internal {@link SyncClient.replay}, so multiplayer + reconnect compose.
 * `receive` and `replay` are absent by design: the manager owns the transport, so
 * inbound messages and re-emits are driven internally, never by the caller.
 */
export type ReconnectingClient = Pick<
  SyncClient,
  | 'clientId'
  | 'liveQuery'
  | 'query'
  | 'gremlin'
  | 'mutate'
  | 'mutateGremlin'
  | 'pushWrite'
  | 'subscribeWrites'
  | 'onDisconnect'
  | 'getStatus'
  | 'onStatus'
  | 'subscriptionCount'
  | 'close'
> & {
  /** Is a transport currently open? */
  connected: () => boolean;
  /** Observe connectivity flips (open/close); returns an unsubscribe fn. */
  onConnectivity: (cb: (up: boolean) => void) => () => void;
};

export const createReconnectingClient = (
  options: ReconnectingClientOptions,
): ReconnectingClient => {
  const baseMs = options.retry?.baseMs ?? 500;
  const maxMs = options.retry?.maxMs ?? 5000;

  let conn: ReconnectingConnection | null = null;
  let up = false;
  let stopped = false;
  let attempt = 0;
  let redial: ReturnType<typeof setTimeout> | null = null;
  const connectivity = new Set<(up: boolean) => void>();

  // One inner client for the manager's whole life: its entries and pending
  // requests survive transport drops. While offline `send` drops the message —
  // the state stays in the client, and replay() re-emits it on reconnect.
  const inner = createSyncClient({
    send: (m) => {
      if (up && conn) {
        conn.send(m);
      }
    },
    clientId: options.clientId,
    maxInactiveQueries: options.maxInactiveQueries,
  });

  const setUp = (next: boolean): void => {
    if (up === next) {
      return;
    }

    up = next;

    for (const cb of connectivity) {
      cb(next);
    }
  };

  const dial = (): void => {
    if (stopped) {
      return;
    }

    // `opened`/`closed` may fire either synchronously during connect() (a
    // MessagePort or test double) or asynchronously (real sockets). Per-dial
    // state lets us handle both without a temporal-dead-zone crash and without
    // acting on a half-built connection:
    // - `goLive()` runs the open work only once BOTH the connection is assigned
    //   AND `opened` has fired, so a synchronous open still replays over a live
    //   `conn` (not `null`) rather than dropping every re-subscribe.
    // - `settled` makes `closed` idempotent — a transport that signals close
    //   more than once won't fork a second dial chain.
    const held: { c: ReconnectingConnection | null; opened: boolean; settled: boolean } = {
      c: null,
      opened: false,
      settled: false,
    };

    const goLive = (): void => {
      if (!held.opened || held.c === null) {
        return;
      }

      conn = held.c;
      attempt = 0;
      setUp(true);
      inner.replay(); // re-subscribe + re-send parked one-shots, over the live conn
    };

    held.c = options.connect({
      opened: () => {
        held.opened = true;
        goLive();
      },
      received: (m) => inner.receive(m),
      closed: () => {
        if (held.settled) {
          return;
        }

        held.settled = true;

        if (conn === held.c) {
          conn = null;
        }

        setUp(false);

        if (stopped) {
          return;
        }

        redial = setTimeout(dial, Math.min(maxMs, baseMs * 2 ** attempt++));
      },
    });

    if (held.settled) {
      // `closed()` fired SYNCHRONOUSLY during connect() (e.g. an immediate failure):
      // `held.c` was still null inside that callback, so its `conn === held.c` check
      // could not null it and the just-returned conn is orphaned. Release it now
      // (the redial is already scheduled by that callback).
      held.c?.close();
      held.c = null;
    } else {
      goLive(); // if opened fired synchronously during connect(), run it now
    }
  };

  dial();

  return {
    clientId: inner.clientId,
    liveQuery: inner.liveQuery,
    query: inner.query,
    gremlin: inner.gremlin,
    mutate: inner.mutate,
    mutateGremlin: inner.mutateGremlin,
    pushWrite: inner.pushWrite,
    // The CDC surface — writes subscription + presence teardown. Both live on the
    // long-lived inner client and re-emit on reconnect via replay(), so a
    // multiplayer app keeps its cross-client stream and presence across drops.
    subscribeWrites: inner.subscribeWrites,
    onDisconnect: inner.onDisconnect,
    getStatus: inner.getStatus,
    onStatus: inner.onStatus,
    subscriptionCount: inner.subscriptionCount,
    connected: () => up,
    onConnectivity: (cb) => {
      connectivity.add(cb);

      return () => connectivity.delete(cb);
    },
    close: () => {
      stopped = true;

      if (redial) {
        clearTimeout(redial);
        redial = null;
      }

      conn?.close();
      conn = null;
      setUp(false);
      inner.close(); // rejects any still-pending request — the app is tearing down
    },
  };
};
