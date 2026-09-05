/**
 * The sync loop — the worker-side machinery between the local store and the
 * network. One producer and one mechanism per arrow of the design's full loop:
 *
 *   frontend declares interest      → host `onSubscribe` fires `ensure`
 *   worker fills what that implies  → collection loaders write into the graph
 *   server pushes what changed      → `ingest` applies writes; epochs route
 *   local writes go back up         → `mutate` applies optimistically + queues
 *
 * **Collections** are the completeness unit: an app-defined scope
 * (`"people"`, `"cycle-2026"`) declaring which labels it covers and how to
 * load itself. Demand-fill needs no protocol addition — a subscription's
 * dependency tokens already name the labels it reads, so `ensure(deps)` fires
 * the loaders for every intersecting collection that isn't loaded. Deps no
 * collection covers are local-only data, complete by definition.
 *
 * **Loaders return writes, not graphs**: `SyncWrite[]` (GQL text + `$name`
 * bindings, or a Gremlin mutation traversal), applied in one `store.mutate`.
 * That keeps the loader an ordinary
 * async function next to the data (fetch, decode, map), keeps values on the
 * params path (never spliced), and lets epochs route the resulting pushes
 * with no knowledge of subscriptions.
 *
 * **Write-back** is optimistic and FIFO: `mutate` applies locally at once
 * (subscribers see it immediately), then queues the write for
 * `upstream.push`, one in flight at a time, with exponential backoff. A write
 * that exhausts its retries is dropped and reported via `onWriteError` —
 * rollback-and-correct arrives with server cursors (a later step), not v1.
 *
 * The loop is persistence-agnostic: hydrate by building the store from a
 * snapshot before constructing the engine, and pass `initiallyComplete` for
 * the collections that snapshot already covers.
 */

import { ErrorCode, LenkeError } from '@lenke/errors';
import type { QueryParams, Store } from '@lenke/native';

import { createSyncHost, type SyncHost, type SyncHostOptions } from './host.js';
import { runWrite, toWireError, type SyncWrite, type WireError } from './protocol.js';

export type { SyncWrite } from './protocol.js';

export type CollectionState = 'empty' | 'loading' | 'complete' | 'error';

export type CollectionDefinition = {
  /** Labels / edge-types this scope covers — matched against subscription deps. */
  labels: readonly string[];
  /**
   * Param name(s) that scope this collection to one slice by VALUE. A keyed
   * collection tracks completeness and demand-fills per distinct bound value
   * (`cluster = 'prod-east'` vs `'prod-west'`), reading that value straight off
   * the subscription's `params` — no synthetic label, no side channel. A
   * subscription that omits a key param neither demand-fills nor counts against
   * this collection's completeness: value-scoped collections serve scoped
   * subscriptions. Omit `key` for a single whole-collection scope.
   */
  key?: string | readonly string[];
  /** Fetch the scope (its bound key values, if keyed) → writes that materialize it. */
  load: (scope: QueryParams) => Promise<SyncWrite[]>;
};

/**
 * A collection at one scope: a bare `name` for a whole collection, or
 * `{ name, scope }` (the key params' bound values) for one slice of a keyed one.
 */
export type CollectionScope = { name: string; scope?: QueryParams };

/**
 * One unit of demand-fill work handed to a {@link LoadScheduler}: load this
 * `collection` at this `scope`. `attempt` is 0 for a fresh demand, ≥1 for the
 * Nth retry after a failure; `priority` (default 0, higher first) is the
 * subscription's hint. `run()` performs the load and rejects on failure — the
 * engine, not the scheduler, decides whether a rejection earns another attempt.
 */
export type LoadJob = {
  readonly collection: string;
  readonly scope: QueryParams;
  readonly attempt: number;
  readonly priority: number;
  run: () => Promise<void>;
};

/**
 * The demand-fill escape hatch: an app-supplied gate deciding WHEN, in what
 * ORDER, and at what CONCURRENCY load jobs run. Called once per job; returns a
 * cancel fn the engine invokes when the job is superseded (a fresh explicit
 * demand, or `retryAll`) so a stale backoff timer never double-fires.
 *
 * The split is deliberate: the scheduler owns timing/ordering/concurrency; the
 * engine owns the retry POLICY (how many attempts, and that only a rejection
 * earns the next one). This is why a custom scheduler cannot storm — the engine
 * only ever hands it one live job per (collection, scope), and only mints the
 * next attempt when the prior `run()` rejects, up to `loadRetry.attempts`.
 *
 * Omit it for the default gate: attempt-0 jobs run as soon as a concurrency
 * slot frees (highest priority first), retries wait an exponential, capped,
 * jittered backoff.
 */
export type LoadScheduler = (job: LoadJob) => () => void;

export type SyncEngineOptions = {
  store: Store;
  /** The app's demand-fill scopes, keyed by collection name. */
  collections?: Record<string, CollectionDefinition>;
  /**
   * Collections (or keyed-collection slices) the boot snapshot already covers —
   * their first load is skipped. A bare string names a whole collection;
   * `{ name, scope }` names one slice of a keyed one.
   */
  initiallyComplete?: readonly (string | CollectionScope)[];
  /**
   * Pending writes restored from a snapshot. Their effects are already IN the
   * snapshot's graph (they were applied optimistically before it was saved),
   * so they re-enqueue for upstream without re-applying locally.
   */
  initialWrites?: readonly SyncWrite[];
  /** Where local writes replicate to. Omit for a local-only engine. */
  upstream?: {
    push: (write: SyncWrite) => Promise<void>;
  };
  /**
   * Write-back retry policy: `attempts` tries, `baseMs * 2^n` between them,
   * capped at `maxMs` (default 30s) so long outages back off politely instead
   * of exploding the wait.
   */
  retry?: { attempts?: number; baseMs?: number; maxMs?: number };
  /**
   * Backpressure cap on the write-back queue (default 1000). When the queue is
   * full, `mutate` REFUSES the write (a coded `E_RESOURCE_EXHAUSTED` throw,
   * before the optimistic local apply) rather than growing without bound — a
   * runaway write loop hits a wall instead of an infinite queue, and every
   * queued write stops bloating each snapshot. Sized for human-scale offline
   * editing; raise it if your app legitimately batches more. Restored
   * `initialWrites` are exempt (they are truth already applied to the
   * snapshot's graph); the cap gates NEW writes only.
   */
  maxPendingWrites?: number;
  /**
   * Demand-fill retry policy: after a load fails, the engine schedules up to
   * `attempts` total tries (default 5) with `baseMs * 2^(n-1)` backoff between
   * them (default base 1s), capped at `maxMs` (default 30s). Distinct from the
   * write-back `retry` so reads and writes back off independently.
   */
  loadRetry?: { attempts?: number; baseMs?: number; maxMs?: number };
  /**
   * The demand-fill scheduler escape hatch (see {@link LoadScheduler}). Omit for
   * the default backoff + priority + concurrency gate; supply one to route loads
   * through an app-owned queue (custom prioritization, a global rate limit, a
   * circuit breaker).
   */
  loadScheduler?: LoadScheduler;
  /**
   * Max loads the DEFAULT scheduler runs at once (default 4) — the choke point
   * that keeps a burst of standing subscriptions from stampeding the single
   * upstream connection. Ignored when `loadScheduler` is supplied (a custom gate
   * owns its own concurrency).
   */
  maxConcurrentLoads?: number;
  /** A write exhausted its retries and was dropped from the queue. */
  onWriteError?: (write: SyncWrite, error: unknown) => void;
  /** A collection load failed (state → 'error'; a retry is scheduled if any remain). */
  onLoadError?: (collection: string, error: unknown) => void;
};

export type SyncEngine = {
  readonly store: Store;
  /**
   * Completeness of one collection (for status surfaces and tests). Pass
   * `scope` (the key params' values) for a keyed collection; `undefined` for an
   * unknown collection or a keyed one addressed without its scope.
   */
  collectionState: (name: string, scope?: QueryParams) => CollectionState | undefined;
  /** Are the collections these deps + params imply all complete? (`null` deps → yes.) */
  isComplete: (deps: readonly string[] | null, params?: QueryParams) => boolean;
  /**
   * Fire loaders for every intersecting, unloaded (collection, scope). An
   * explicit demand: it supersedes any pending backoff with a fresh immediate
   * attempt. `priority` (default 0, higher first) is passed to the scheduler.
   */
  ensure: (deps: readonly string[] | null, params?: QueryParams, priority?: number) => void;
  /**
   * Reset every errored collection to a fresh immediate load — for a reconnect
   * handler that knows the backend just returned and shouldn't wait out each
   * slice's backoff. A no-op for collections that aren't in `error`.
   */
  retryAll: () => void;
  /**
   * Apply a local write optimistically and queue it for upstream. GQL by default
   * (values ride `params`); pass `lang: 'gremlin'` to run `text` as a Gremlin
   * mutation traversal (no params — pre-escape values with the `gremlin` tag).
   */
  mutate: (text: string, params?: QueryParams, lang?: 'gql' | 'gremlin') => void;
  /** Apply server-pushed writes locally (never re-replicated upstream). */
  ingest: (writes: readonly SyncWrite[]) => void;
  /** Queued-or-in-flight write count (feeds the status message). */
  pendingWrites: () => number;
  /** The queue's current contents — persist these in the snapshot. */
  queuedWrites: () => readonly SyncWrite[];
  /** Loads and queue-length changes re-notify here (hosts refresh on it). */
  onChange: (cb: () => void) => () => void;
  /** A host for one client connection, wired into this loop. */
  createHost: (options: Pick<SyncHostOptions, 'send'>) => SyncHost;
};

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/**
 * The default demand-fill gate. Attempt-0 jobs (fresh demands) run as soon as a
 * concurrency slot frees, highest `priority` first; retries first wait an
 * exponential, capped, jittered backoff. A single in-flight cap across all
 * collections is the choke point on the one upstream connection. The returned
 * cancel fn drops a job whether it's still waiting on its backoff timer or
 * already queued — so a superseding demand never leaves a ghost retry behind.
 */
const defaultScheduler = (cfg: {
  baseMs: number;
  maxMs: number;
  maxConcurrent: number;
}): LoadScheduler => {
  let inFlight = 0;
  const waiting: { job: LoadJob; live: boolean }[] = [];

  const pump = (): void => {
    // Highest priority first; the sort is stable so equal priorities keep
    // arrival order (a fair-enough FIFO within a band).
    waiting.sort((a, b) => b.job.priority - a.job.priority);

    while (inFlight < cfg.maxConcurrent && waiting.length > 0) {
      const slot = waiting.shift();

      if (!slot || !slot.live) {
        continue;
      }

      inFlight += 1;
      void slot.job.run().finally(() => {
        inFlight -= 1;
        pump();
      });
    }
  };

  return (job) => {
    const slot = { job, live: true };
    const backoff =
      job.attempt === 0 ? 0 : Math.min(cfg.maxMs, cfg.baseMs * 2 ** (job.attempt - 1));

    // A fresh demand (no backoff) enqueues synchronously so `ensure` flips the
    // slice to 'loading' on the spot (up to the concurrency cap) — no wasted
    // tick, and the documented "immediately loading" contract holds.
    if (backoff === 0) {
      waiting.push(slot);
      pump();

      return () => {
        slot.live = false;
      };
    }

    // A retry waits full jitter over [backoff/2, backoff]: spreads a thundering
    // herd of same-attempt retries without ever collapsing the delay to ~0.
    const delay = backoff / 2 + Math.random() * (backoff / 2);
    const timer = setTimeout(() => {
      if (!slot.live) {
        return;
      }

      waiting.push(slot);
      pump();
    }, delay);

    return () => {
      slot.live = false;
      clearTimeout(timer);
    };
  };
};

/** The param name(s) that scope a collection — normalized from `key`. */
const keyNamesOf = (def: CollectionDefinition): readonly string[] => {
  if (def.key === undefined) {
    return [];
  }

  return typeof def.key === 'string' ? [def.key] : def.key;
};

export const createSyncEngine = (options: SyncEngineOptions): SyncEngine => {
  const { store, upstream } = options;
  const collections = options.collections ?? {};
  const attempts = options.retry?.attempts ?? 5;
  const baseMs = options.retry?.baseMs ?? 250;
  const maxMs = options.retry?.maxMs ?? 30_000;
  const maxPending = options.maxPendingWrites ?? 1000;

  const loadAttempts = options.loadRetry?.attempts ?? 5;
  const loadBaseMs = options.loadRetry?.baseMs ?? 1_000;
  const loadMaxMs = options.loadRetry?.maxMs ?? 30_000;
  const scheduler: LoadScheduler =
    options.loadScheduler ??
    defaultScheduler({
      baseMs: loadBaseMs,
      maxMs: loadMaxMs,
      maxConcurrent: options.maxConcurrentLoads ?? 4,
    });

  // State is keyed per (collection, scope value): an unkeyed collection uses its
  // bare name; a keyed one appends its bound key values. Absent → 'empty', so
  // only 'complete' slices need seeding from the snapshot.
  const states = new Map<string, CollectionState>();

  // Resolve a collection + a subscription's params to its state key and scope.
  // null = a keyed collection whose key params the subscription didn't supply:
  // it neither demand-fills nor gates completeness for that subscription.
  const scopeOf = (
    name: string,
    def: CollectionDefinition,
    params?: QueryParams,
  ): { stateKey: string; scope: QueryParams } | null => {
    const keys = keyNamesOf(def);

    if (keys.length === 0) {
      return { stateKey: name, scope: {} };
    }

    const scope: Record<string, unknown> = {};

    for (const k of keys) {
      if (params == null || !(k in params)) {
        return null;
      }

      scope[k] = (params as Record<string, unknown>)[k];
    }

    // Keys are in the definition's fixed order, so this tag is deterministic.
    const tag = keys.map((k) => JSON.stringify(scope[k])).join('\x01');

    return { stateKey: `${name}\u0000${tag}`, scope: scope };
  };

  const stateOf = (stateKey: string): CollectionState => states.get(stateKey) ?? 'empty';

  for (const entry of options.initiallyComplete ?? []) {
    const { name, scope } = typeof entry === 'string' ? { name: entry, scope: undefined } : entry;
    const def = collections[name];
    const resolved = def && scopeOf(name, def, scope);

    if (resolved) {
      states.set(resolved.stateKey, 'complete');
    }
  }

  const changeListeners = new Set<() => void>();
  const notifyChange = (): void => {
    for (const l of changeListeners) {
      l();
    }
  };

  // ---- demand-fill -----------------------------------------------------

  type Match = { name: string; stateKey: string; scope: QueryParams };

  // Collections whose labels a subscription reads, each resolved to the scope
  // its params select. A keyed collection missing its key params drops out.
  // `null`/empty deps declare no label to route on → no collection to fill.
  const matchesFor = (deps: readonly string[] | null, params?: QueryParams): Match[] => {
    const out: Match[] = [];

    if (!deps || deps.length === 0) {
      return out;
    }

    for (const [name, def] of Object.entries(collections)) {
      if (!def.labels.some((l) => deps.includes(l))) {
        continue;
      }

      const resolved = scopeOf(name, def, params);

      if (resolved) {
        out.push({ name, ...resolved });
      }
    }

    return out;
  };

  const isComplete = (deps: readonly string[] | null, params?: QueryParams): boolean =>
    matchesFor(deps, params).every((m) => stateOf(m.stateKey) === 'complete');

  // A scope's last load failure, keyed by stateKey. Surfaced to standing queries
  // as a RETRYABLE error (the sub stays alive; the next demand re-attempts).
  const loadErrors = new Map<string, WireError>();

  // The load error a subscription should see: the first errored scope it matches
  // (undefined if every matched scope loaded, or is empty/loading, cleanly).
  const loadError = (
    deps: readonly string[] | null,
    params?: QueryParams,
  ): WireError | undefined => {
    for (const m of matchesFor(deps, params)) {
      const e = loadErrors.get(m.stateKey);

      if (e) {
        return e;
      }
    }

    return undefined;
  };

  // At most one pending-or-running job per (collection, scope). `attemptOf` is
  // the failure count driving backoff; `cancelPending` supersedes a scheduled
  // job when a fresh demand or `retryAll` arrives; `matchByKey`/`priorityOf`
  // let the retry loop and `retryAll` re-address a slice by its state key alone.
  const attemptOf = new Map<string, number>();
  const cancelPending = new Map<string, () => void>();
  const matchByKey = new Map<string, Match>();
  const priorityOf = new Map<string, number>();

  const runLoad = (match: Match): Promise<void> => {
    cancelPending.delete(match.stateKey); // running now, no longer pending
    states.set(match.stateKey, 'loading');
    loadErrors.delete(match.stateKey); // a fresh attempt clears the stale error → UI shows loading

    return collections[match.name].load(match.scope).then(
      (writes) => {
        // One mutate for the whole scope: subscribers hear a single version
        // bump, and epochs route it to exactly the affected live queries.
        store.mutate((g) => {
          for (const w of writes) {
            runWrite(g, w);
          }
        });
        states.set(match.stateKey, 'complete');
        attemptOf.delete(match.stateKey);
        // Even an empty load changes what `complete` means for standing
        // queries — hosts must re-push. (A non-empty load also bumped the
        // version, but the flip itself must be observable either way.)
        // `safeNotify`, not `notifyChange`: this reaches `host.refresh` → `send`,
        // which can throw on a dead socket — and a throw here would reject the
        // load and escape as an unhandled rejection (the scheduler runs it as
        // `void job.run().finally(...)`, with no `.catch`).
        safeNotify();
      },
      (e) => {
        // The load failed. Record the error and — while attempts remain — hand the
        // scheduler the NEXT attempt (timer-driven backoff) FIRST, so a throwing
        // `onLoadError` can't abort the retry/bookkeeping. This scheduled job is the
        // ONLY retry path: no refresh/push re-triggers a load, so a failure can never
        // storm. Attempts run 0,1,…,loadAttempts-1.
        states.set(match.stateKey, 'error');
        loadErrors.set(match.stateKey, toWireError(e));

        const next = (attemptOf.get(match.stateKey) ?? 0) + 1;

        if (next < loadAttempts) {
          schedule(match, next, priorityOf.get(match.stateKey) ?? 0);
        } else {
          attemptOf.delete(match.stateKey); // budget spent; next demand starts fresh
        }

        // The user callback and `notifyChange` (→ host `refresh` → `send`) are
        // side-channels that can throw — contain both, exactly as the write-back
        // path does (`reportWriteError`/`safeNotify`), so a throw never rejects this
        // load and escapes as an unhandled rejection.
        try {
          options.onLoadError?.(match.name, e);
        } catch {
          // best-effort: a user callback fault must not abort load bookkeeping
        }

        safeNotify();
      },
    );
  };

  const schedule = (match: Match, attempt: number, priority: number): void => {
    attemptOf.set(match.stateKey, attempt);
    priorityOf.set(match.stateKey, priority);
    matchByKey.set(match.stateKey, match);
    cancelPending.get(match.stateKey)?.(); // supersede any prior pending job

    const job: LoadJob = {
      collection: match.name,
      scope: match.scope,
      attempt,
      priority,
      run: () => runLoad(match),
    };
    cancelPending.set(match.stateKey, scheduler(job));
  };

  const ensure = (deps: readonly string[] | null, params?: QueryParams, priority = 0): void => {
    for (const match of matchesFor(deps, params)) {
      const state = stateOf(match.stateKey);

      // 'loading' → a job is already running; 'complete' → nothing to do. Both
      // 'empty' (never loaded) and 'error' (failed, maybe mid-backoff) fire a
      // fresh attempt-0 job — an explicit demand supersedes any pending backoff
      // so a user action isn't stuck behind a 30s timer.
      if (state === 'loading' || state === 'complete') {
        continue;
      }

      schedule(match, 0, priority);
    }
  };

  const retryAll = (): void => {
    // Reset every errored slice to a fresh attempt-0 job.
    for (const [stateKey, state] of states) {
      const match = state === 'error' ? matchByKey.get(stateKey) : undefined;

      if (match) {
        schedule(match, 0, priorityOf.get(stateKey) ?? 0);
      }
    }
  };

  // ---- write-back ------------------------------------------------------

  // Restored writes re-enqueue as-is: their effects are already in the
  // snapshot's graph, they just never reached upstream.
  const queue: SyncWrite[] = [...(options.initialWrites ?? [])];
  let pumping = false;

  // Side-channel callbacks (a user `onWriteError`, and `notifyChange` → host
  // `refresh` → `send`, e.g. a `send` on a CLOSING socket) can throw. Contain
  // them: a thrown reporter/notification must never abort the drain loop — see
  // `pump`'s `finally`, which is what actually stops a throw from wedging the
  // queue (a stuck `pumping` would early-return every future pump forever).
  const reportWriteError = (write: SyncWrite, e: unknown): void => {
    try {
      options.onWriteError?.(write, e);
    } catch {
      // best-effort: a user callback fault must not block replication
    }
  };

  const safeNotify = (): void => {
    try {
      notifyChange(); // pendingWrites moved
    } catch {
      // non-fatal (e.g. send on a dead socket): the next drain re-notifies
    }
  };

  const pump = async (): Promise<void> => {
    if (pumping) {
      return;
    }

    pumping = true;

    // `finally` is load-bearing: if anything below throws, `pumping` must still
    // clear, or every future pump early-returns and the queue never drains
    // again — writes pile up and bloat each snapshot, silently.
    try {
      while (queue.length > 0) {
        const [write] = queue; // FIFO; stays queued (and counted) until settled
        let sent = false;

        for (let attempt = 0; attempt < attempts && !sent; attempt += 1) {
          try {
            // upstream is present by construction: writes only enqueue when it is.
            await upstream!.push(write);
            sent = true;
          } catch (e) {
            if (attempt + 1 >= attempts) {
              // Terminal: drop and report. Roll-back-and-correct needs server
              // cursors (a later step) — silently retrying forever would just
              // hide a dead upstream.
              reportWriteError(write, e);
            } else {
              await sleep(Math.min(maxMs, baseMs * 2 ** attempt));
            }
          }
        }

        queue.shift();
        safeNotify();
      }
    } finally {
      pumping = false;
    }
  };

  const mutate = (text: string, params?: QueryParams, lang?: 'gql' | 'gremlin'): void => {
    if (lang === 'gremlin' && params !== undefined) {
      // Refuse loudly: Gremlin has no param binding, so silently dropping the
      // bindings would run the traversal with unbound `$name` literals.
      throw new LenkeError(
        'lenke: a gremlin write has no param binding — interpolate values with the gremlin tag',
        { code: ErrorCode.InvalidGraphOp },
      );
    }

    if (upstream && queue.length >= maxPending) {
      // Backpressure BEFORE the optimistic apply: a refused write must not
      // diverge the local graph from what will ever reach upstream.
      throw new LenkeError(
        `lenke: write-back queue is full (${maxPending} pending) — upstream is not draining`,
        { code: ErrorCode.ResourceExhausted },
      );
    }

    const before = store.version;
    let write: SyncWrite;

    if (lang === 'gremlin') {
      write = { text, lang };
    } else if (params) {
      write = { text, params };
    } else {
      write = { text };
    }

    store.mutate((g) => runWrite(g, write)); // optimistic: local readers see it now

    // Version-gated enqueue: a write that changed nothing replicates nothing.
    if (upstream && store.version !== before) {
      queue.push(write);
      notifyChange();
      void pump();
    }
  };

  const ingest = (writes: readonly SyncWrite[]): void => {
    store.mutate((g) => {
      for (const w of writes) {
        runWrite(g, w);
      }
    });
  };

  // ---- assembly --------------------------------------------------------

  // Flush aggressively: restored writes start replicating immediately, not on
  // the next local mutation.
  if (upstream && queue.length > 0) {
    void pump();
  }

  return {
    store,
    collectionState: (name, scope) => {
      const def = collections[name];
      const resolved = def && scopeOf(name, def, scope);

      return resolved ? stateOf(resolved.stateKey) : undefined;
    },
    isComplete,
    ensure,
    retryAll,
    mutate,
    ingest,
    pendingWrites: () => queue.length,
    queuedWrites: () => [...queue],
    onChange: (cb) => {
      changeListeners.add(cb);

      return () => {
        changeListeners.delete(cb);
      };
    },
    createHost: ({ send }) => {
      const host = createSyncHost(store, {
        send,
        applyMutation: mutate,
        isComplete,
        loadError,
        onSubscribe: ensure,
        pendingWrites: () => queue.length,
      });

      // Completeness flips and queue movement must reach standing queries and
      // the status surface even when the graph version never moved.
      const refresh = (): void => {
        host.refresh();
        host.sendStatus();
      };
      changeListeners.add(refresh);

      return {
        ...host,
        close: () => {
          changeListeners.delete(refresh);
          host.close();
        },
      };
    },
  };
};
