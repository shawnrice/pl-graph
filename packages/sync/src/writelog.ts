import type { SyncWrite } from './protocol.js';

/**
 * The server-side op log behind the CDC write stream. All `SyncHost`s on one
 * server share a single `WriteLog` (the way they share one `Store`): a host
 * `append`s each committed write, and every host with a stream subscriber
 * `subscribe`s to fan the tail out to its client. Ordering + resumability come
 * from a monotonic `seq`; a bounded ring caps memory, so a client that has
 * fallen too far behind gets `null` from `since` and must cold-boot from a
 * snapshot. This is statement-based replication — the op *is* the `SyncWrite`
 * (write-language text + resolved params), replayed through `runWrite`, which is
 * deterministic because the two engines are byte-identical.
 */
export type WriteLogEntry = {
  /** Monotonic sequence number (1-based; `0` means "before the first op"). */
  seq: number;
  /** The **stable client id** that committed it — for origin-skip so a client
   *  never re-ingests the write it already applied optimistically. Keyed on the
   *  client's durable id (not the connection), so the skip survives a reconnect:
   *  a re-dialed client presents the same id and its own writes are still
   *  filtered out of its backlog. (Legacy clients that send no id get a
   *  per-connection fallback — echo-skip within one connection only.) */
  origin: string;
  write: SyncWrite;
  /**
   * The label / edge-type / property-key tokens this write touches (as by
   * `inferDeps`), for interest routing — a host forwards the write only to
   * clients whose subscriptions depend on one of these tokens. `undefined` means
   * "affects everything / can't infer" (e.g. a Gremlin write) → forward to all.
   */
  tokens?: readonly string[];
  /**
   * The write's content-derived **value-scope** — the distinct values of the
   * host's configured scope key across the elements this write touched (e.g.
   * `['42']` for a write into room 42). A host with a `scopes` filter forwards the
   * write only if this intersects it. `undefined` means "unscoped / can't derive"
   * (no scope key configured, or the write touched no scoped element) → forward
   * regardless of any scope filter.
   */
  scope?: readonly string[];
};

export type WriteLogOptions = {
  /**
   * Max retained entries (ring buffer). Older entries drop; a `since` reaching
   * past them returns `null`, signalling the client to cold-boot from a
   * snapshot rather than apply a gapped stream. Default 1024.
   */
  capacity?: number;
};

export type WriteLog = {
  /** Append a committed write tagged with the committing client's stable id (with
   *  the tokens it touches, and its value-scope, for interest routing); assigns +
   *  returns its `seq` and notifies subscribers. */
  append(
    origin: string,
    write: SyncWrite,
    tokens?: readonly string[],
    scope?: readonly string[],
  ): number;
  /** Subscribe to the live tail. Returns an unsubscribe. */
  subscribe(cb: (entry: WriteLogEntry) => void): () => void;
  /**
   * Entries strictly after `seq`, ascending. `[]` if the caller is already
   * current; `null` if `seq` has fallen off the retained tail (there'd be a gap
   * → the caller must cold-boot). `since(0)` means "from the very start".
   */
  since(seq: number): WriteLogEntry[] | null;
  /** The latest assigned seq (`0` if none yet). */
  head(): number;
};

export const createWriteLog = (options: WriteLogOptions = {}): WriteLog => {
  const capacity = Math.max(1, options.capacity ?? 1024);
  const buffer: WriteLogEntry[] = []; // retained tail, ascending, contiguous seq
  const subscribers = new Set<(entry: WriteLogEntry) => void>();
  let seq = 0;

  return {
    append: (origin, write, tokens, scope) => {
      seq += 1;
      const entry: WriteLogEntry = { seq, origin, write, tokens, scope };
      buffer.push(entry);

      if (buffer.length > capacity) {
        buffer.shift();
      }

      // Deliver to current subscribers. (Callbacks here are host forwards — they
      // don't (un)subscribe mid-fan-out, so iterating the set directly is safe.)
      for (const cb of subscribers) {
        cb(entry);
      }

      return seq;
    },

    subscribe: (cb) => {
      subscribers.add(cb);

      return () => {
        subscribers.delete(cb);
      };
    },

    since: (from) => {
      if (from === seq) {
        return []; // exactly current
      }

      if (from > seq) {
        // The caller is AHEAD of this log — impossible unless the log REGRESSED
        // (server restart, LB failover to a peer, in-memory reset: seq dropped back).
        // Treating it as "current" ([]) would silently drop every future write (the
        // client skips cursors <= its own). Force a cold-boot resync instead.
        return null;
      }

      // The buffer holds a contiguous run [oldest … seq]. The caller needs
      // (from … seq]; that's a gap unless `from + 1` is still retained.
      const oldest = buffer.length > 0 ? buffer[0].seq : seq + 1;

      if (from + 1 < oldest) {
        return null; // fell off the tail → cold boot
      }

      return buffer.filter((e) => e.seq > from);
    },

    head: () => seq,
  };
};
