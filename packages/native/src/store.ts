import { ErrorCode, LenkeError } from '@lenke/errors';

import { ensureDisposeSymbol } from './dispose.js';
import type { QueryParams, RustGraph, Row } from './graph.js';

/**
 * A framework-agnostic reactive store over a {@link RustGraph}, built for React's
 * `useSyncExternalStore` (and anything with the same `subscribe`/`getSnapshot`
 * contract). It is *not* React-coupled — the package has no React dependency.
 *
 * The hard requirement `useSyncExternalStore` imposes is that `getSnapshot()`
 * return a **referentially-stable** value — the same reference until the data
 * actually changes — or React re-renders forever. The wasm/native graph is
 * mutable state behind a handle, so we bridge it with the engine's O(1) mutation
 * `version`: `getSnapshot` re-runs its query only when the version moved, and
 * caches the result otherwise. With declared `deps`, it goes finer still — it
 * recomputes only when one of its dependency *epochs* (per label / edge-type /
 * property-key) changed, so an unrelated mutation doesn't recompute it.
 *
 * Identical standing queries are **pooled**: two `liveQuery` calls with the same
 * `(kind, deps, params, text)` share one recompute and one cached array, so many
 * consumers of the same query — most notably several connections on one store
 * (e.g. several tabs on a SharedWorker's graph) — don't each pay for it.
 *
 * @example
 * ```ts
 * const store = createStore(graph);
 * const people = store.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
 * // in a component:
 * const rows = useSyncExternalStore(people.subscribe, people.getSnapshot);
 * // mutate (this notifies subscribers iff the graph actually changed):
 * store.mutate((g) => g.query("INSERT (:Person {name: 'zoe'})"));
 * ```
 */
export type LiveQuery<T = Row> = {
  /** Register a change callback; returns an unsubscribe function. */
  subscribe: (onChange: () => void) => () => void;
  /** Current result — a stable reference until a relevant mutation occurs. */
  getSnapshot: () => T[];
};

export type Store = {
  /** The underlying graph (read it directly for one-off queries). */
  readonly graph: RustGraph;
  /** The graph's current mutation version. */
  readonly version: number;
  /**
   * Run a mutating operation and notify subscribers **iff** it actually changed
   * the graph (decided by the version counter, so a read-only call is silent).
   */
  mutate: <T>(fn: (graph: RustGraph) => T) => T;
  /**
   * A `useSyncExternalStore`-ready live query. `deps` is **required** — declare
   * the dependency posture explicitly (React's array semantics, no silent
   * omission):
   * - `[...]` — recompute only when one of these label/property/edge-type
   *   epochs moves (`graph.epoch(token)`).
   * - `[]` — depends on nothing → computed once, never recomputed.
   * - `null` — depends on everything → recompute on every graph change.
   *
   * `params` are `$name` bindings for the query text — part of the standing
   * query's identity, bound safely at execute time (never spliced). Use
   * {@link inferDeps} if you want to derive `deps` from the query text rather
   * than hand-declare it.
   */
  liveQuery: <R extends Row = Row>(
    text: string,
    opts: { deps: readonly string[] | null; params?: QueryParams },
  ) => LiveQuery<R>;
  /**
   * The Gremlin twin of {@link liveQuery}: a standing traversal whose result
   * values (arbitrary JSON, not rows) are recomputed under the same `deps`
   * gating. Gremlin has no engine param binding, so interpolate values with the
   * {@link gremlin} tag / {@link escapeGremlin} (safe literals). The traversal
   * must be a **read**; a mutating step in a live query would rewrite the graph
   * on every recompute.
   */
  liveGremlin: (text: string, opts: { deps: readonly string[] | null }) => LiveQuery<unknown>;
  /**
   * Deterministically release the store: sever every live subscription and
   * {@link RustGraph.free | free} the underlying graph handle. The imperative twin
   * of `[Symbol.dispose]` — call it when a runtime/tool can't use `using` (a replica
   * fleet freed in a loop, a test `afterEach`, a `finally`), so you get deterministic
   * release without leaning on the {@link FinalizationRegistry} backstop (which only
   * fires when the GC happens to run, and warns about the leak until it does).
   * Idempotent. After it, snapshots stay on their last cached value and `mutate`
   * throws a coded error.
   */
  free: () => void;
  /**
   * Dispose the store and {@link RustGraph.free | free} its underlying graph, so
   * `using store = createStore(graph)` releases the native/wasm handle at scope
   * exit. Idempotent; identical to {@link Store.free}.
   */
  [Symbol.dispose]: () => void;
};

/** Canonical, key-sorted JSON string of a value — a collision-free identity. */
const canonical = (v: unknown): string => {
  if (v === null || typeof v !== 'object') {
    return JSON.stringify(v) ?? 'null';
  }

  if (Array.isArray(v)) {
    return `[${v.map(canonical).join(',')}]`;
  }

  return `{${Object.keys(v as Record<string, unknown>)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`)
    .join(',')}}`;
};

export const createStore = (graph: RustGraph): Store => {
  ensureDisposeSymbol(); // so the [Symbol.dispose] key below resolves on any runtime

  // A Set of per-subscription ENTRIES, not of the callbacks themselves: two
  // subscriptions that happen to pass the same `onChange` reference (e.g. one
  // callback wired to two live queries) must be independent registrations, so
  // unsubscribing one can't delete the other. Keying by callback identity would
  // collapse them into one and silently kill the survivor's notifications.
  const listeners = new Set<{ notify: () => void }>();
  // Disposal must sever the subscription graph, not just free the handle: a
  // still-mounted live query would otherwise call query/gremlin on the freed
  // graph on its next getSnapshot. After dispose, snapshots stay on their last
  // cached value (stable reference — safe for useSyncExternalStore), new
  // subscriptions are no-ops, and mutate throws a coded error.
  let disposed = false;
  const notify = (): void => {
    for (const l of listeners) {
      l.notify();
    }
  };

  // Deterministic release: sever subscriptions (so a still-mounted live query never
  // calls into the freed graph on its next getSnapshot) then free the handle.
  // Idempotent — the graph's own freed-once guard makes a double free a no-op too.
  const free = (): void => {
    if (disposed) {
      return;
    }

    disposed = true;
    listeners.clear();
    graph.free();
  };

  // A pooled, epoch-gated computation shared by every handle with the same
  // signature. `run` is the only difference between liveQuery (rows) and
  // liveGremlin (values); the gating keys off the graph version and the declared
  // `deps` epochs, not the query text.
  type Cell<T> = {
    readonly deps: readonly string[] | null;
    readonly run: () => T[];
    seenVersion: number; // -1 is unreachable for a u64 version/epoch → the first call primes
    seenFingerprint: number;
    cached: T[];
    refs: number; // active subscribers across all handles sharing this cell
  };
  const pool = new Map<string, Cell<unknown>>();

  const snapshotOf = <T>(cell: Cell<T>): T[] => {
    if (disposed) {
      return cell.cached; // the graph is freed — hold the last snapshot, don't touch it
    }

    const v = graph.version;

    if (v === cell.seenVersion) {
      return cell.cached; // nothing mutated since last read → stable reference
    }

    // `null` deps → gate on the global version (recompute every change). `[]` →
    // a constant fingerprint, so it never recomputes after the first prime.
    // Otherwise sum the declared epochs and recompute only when one moved.
    let fingerprint: number;

    if (cell.deps === null) {
      fingerprint = v;
    } else if (cell.deps.length === 0) {
      fingerprint = 0;
    } else {
      fingerprint = cell.deps.reduce((acc, d) => acc + graph.epoch(d), 0);
    }

    // Recompute only when our dependencies moved. Crucially, advance the bookkeeping
    // (seenVersion / seenFingerprint) ONLY after run() returns: a run() that throws —
    // a query made invalid by the mutation (a CAST error, an incompatible ORDER BY on
    // newly-inserted data) — must re-throw on every subsequent read, not leave the cell
    // marked "seen" so the next same-version read silently returns stale rows.
    if (fingerprint !== cell.seenFingerprint) {
      cell.cached = cell.run();
      cell.seenFingerprint = fingerprint;
    }

    cell.seenVersion = v;

    return cell.cached;
  };

  const makeLive = <T>(
    signature: string,
    deps: readonly string[] | null,
    run: () => T[],
  ): LiveQuery<T> => {
    let cell = pool.get(signature) as Cell<T> | undefined;

    if (!cell) {
      cell = { deps, run, seenVersion: -1, seenFingerprint: -1, cached: [], refs: 0 };
      pool.set(signature, cell as Cell<unknown>);
    }

    // The handle closes over the cell, so it stays usable even after the cell is
    // evicted from the pool (a still-held handle just stops sharing with future
    // callers).
    const held = cell;

    return {
      subscribe: (onChange) => {
        if (disposed) {
          return () => {};
        }

        held.refs += 1;
        const entry = { notify: onChange }; // fresh identity per subscribe
        listeners.add(entry);

        return () => {
          listeners.delete(entry);
          held.refs -= 1;

          // Last subscriber gone → stop sharing this cell with future callers.
          // A fresh identical liveQuery after this mints a new cell (correct,
          // just briefly un-shared); the pool never accumulates dead cells.
          if (held.refs === 0 && pool.get(signature) === held) {
            pool.delete(signature);
          }
        };
      },
      getSnapshot: () => snapshotOf(held),
    };
  };

  // Pool key: a collision-free JSON identity of the standing query. `deps` are a
  // set (the fingerprint is a commutative sum) so they sort; `null` stays
  // distinct from `[]`; params key-sort; text is exact. No false collision could
  // ever share two different queries' results.
  const signatureOf = (
    kind: 'q' | 'g',
    text: string,
    deps: readonly string[] | null,
    params?: QueryParams,
  ): string => canonical([kind, deps === null ? null : [...deps].sort(), params ?? null, text]);

  return {
    graph,
    get version() {
      return graph.version;
    },
    mutate: (fn) => {
      if (disposed) {
        throw new LenkeError('lenke: store used after dispose', {
          code: ErrorCode.InvalidGraphOp,
        });
      }

      const before = graph.version;
      const result = fn(graph);

      if (graph.version !== before) {
        notify();
      }

      return result;
    },
    liveQuery: <R extends Row = Row>(
      text: string,
      opts: { deps: readonly string[] | null; params?: QueryParams },
    ) =>
      makeLive<R>(signatureOf('q', text, opts.deps, opts.params), opts.deps, () =>
        graph.query<R>(text, opts.params),
      ),
    liveGremlin: (text, opts) =>
      makeLive(signatureOf('g', text, opts.deps), opts.deps, () => graph.gremlin(text)),
    free,
    [Symbol.dispose]: free,
  };
};

/**
 * Best-effort extraction of a query's dependency tokens — the `:Label` / `:TYPE`
 * names and `.key` property keys it references. **Over-grabbing is safe**
 * (causes an unnecessary recompute); under-grabbing risks a stale snapshot, so
 * prefer passing `deps` explicitly for correctness-critical live queries, and
 * use `null` deps (recompute-always) for whole-element returns like `RETURN n`.
 */
export const inferDeps = (text: string): string[] => {
  const tokens = new Set<string>();

  // :Label or :TYPE   and   [:TYPE]
  for (const m of text.matchAll(/:([A-Za-z_]\w*)/g)) {
    tokens.add(m[1]);
  }

  // .key property access
  for (const m of text.matchAll(/\.([A-Za-z_]\w*)/g)) {
    tokens.add(m[1]);
  }

  // Inline-map filter keys — `(o:Purchase {status:'paid', qty:2})`. These read
  // properties without a `.` prefix, so the property-access regex above misses
  // them; without this a `SET o.status=…` would not invalidate the query.
  // Over-grabbing is safe, so we accept the occasional false positive (a map key
  // that shadows a non-property token).
  for (const m of text.matchAll(/[{,]\s*([A-Za-z_]\w*)\s*:/g)) {
    tokens.add(m[1]);
  }

  return [...tokens];
};
