// Proves the reactive store satisfies the useSyncExternalStore contract:
// getSnapshot is referentially stable until a *relevant* mutation, version-gated
// caching works, finer (per-token) invalidation skips unaffected queries, and
// subscribers fire on mutate. Run: bun test packages/native/src/store.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { graphFromNdjson } from './graph.js';
import { createStore, inferDeps } from './store.js';

// Host-specific shared-library extension (macOS `.dylib`, Linux `.so`, Windows
// `.dll`); `build:rust` emits the one for this platform.
const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;

// Built by `bun run build:rust`, not the test — skip cleanly when it's absent.
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[store.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}',
].join('\n');

const newStore = () => {
  const backend = createFfiEngineBackend(LIB);
  const g = graphFromNdjson(backend, new TextEncoder().encode(NDJSON));

  return createStore(g);
};

suite('@lenke/native reactive store', () => {
  test('store.free() releases the graph imperatively and is idempotent', () => {
    const store = newStore();
    const live = store.liveQuery('MATCH (n:P) RETURN n.name', { deps: null });
    const before = live.getSnapshot();
    expect(before.length).toBe(2);

    store.free();

    // The native handle is released: a mutate now throws a coded error.
    expect(() => store.mutate((g) => g.query("INSERT (:P {name: 'zoe'})"))).toThrow();
    // A still-mounted live query is severed — its snapshot stays on the last value
    // (stable reference, safe for useSyncExternalStore) instead of hitting the freed graph.
    expect(live.getSnapshot()).toBe(before);
    // Idempotent — a second free (and the dispose symbol) are no-ops.
    expect(() => store.free()).not.toThrow();
    expect(() => store[Symbol.dispose]()).not.toThrow();
  });

  test('getSnapshot is referentially stable with no mutation', () => {
    const store = newStore();
    const names = store.liveQuery('MATCH (n:P) RETURN n.name ORDER BY n.name', { deps: null });
    const s1 = names.getSnapshot();
    const s2 = names.getSnapshot();
    expect(s1).toBe(s2); // same reference → React won't loop
    expect(s1).toEqual([{ 'n.name': 'marko' }, { 'n.name': 'vadas' }]);
    store.graph.free();
  });

  test('mutate bumps version, notifies, and changes the snapshot', () => {
    const store = newStore();
    const ages = store.liveQuery('MATCH (n:P) RETURN n.age ORDER BY n.age', { deps: ['P', 'age'] });
    const before = ages.getSnapshot();
    const v0 = store.version;

    let fired = 0;
    const unsub = ages.subscribe(() => {
      fired += 1;
    });
    store.mutate((g) => g.query("MATCH (n:P) WHERE n.name = 'marko' SET n.age = 99"));

    expect(store.version).toBeGreaterThan(v0);
    expect(fired).toBe(1);
    const after = ages.getSnapshot();
    expect(after).not.toBe(before); // an `age` dep moved → new snapshot
    expect(after).toEqual([{ 'n.age': 27 }, { 'n.age': 99 }]);
    unsub();
    store.graph.free();
  });

  test('finer invalidation: an unaffected query keeps its snapshot reference', () => {
    const store = newStore();
    const names = store.liveQuery('MATCH (n:P) RETURN n.name ORDER BY n.name', {
      deps: ['P', 'name'],
    });
    const before = names.getSnapshot();

    // mutate only `age` — the names query depends on P + name, not age
    store.mutate((g) => g.query("MATCH (n:P) WHERE n.name = 'marko' SET n.age = 99"));

    const after = names.getSnapshot();
    expect(after).toBe(before); // version moved, but our deps didn't → same ref
    store.graph.free();
  });

  test('null deps recompute on any mutation (coarse, always-correct)', () => {
    const store = newStore();
    const all = store.liveQuery('MATCH (n:P) RETURN n.name ORDER BY n.name', { deps: null });
    const before = all.getSnapshot();
    store.mutate((g) => g.query("MATCH (n:P) WHERE n.name = 'marko' SET n.age = 99"));
    const after = all.getSnapshot();
    expect(after).not.toBe(before); // null: any mutation invalidates
    expect(after).toEqual(before); // ...but the names are unchanged
    store.graph.free();
  });

  test('[] deps depend on nothing → computed once, never recomputed', () => {
    const store = newStore();
    const constants = store.liveQuery('MATCH (n:P) RETURN n.name ORDER BY n.name', { deps: [] });
    const before = constants.getSnapshot();

    // Even a mutation this query WOULD reflect leaves the snapshot untouched:
    // `[]` is a promise that it depends on nothing, so it never re-runs.
    store.mutate((g) => g.query("INSERT (:P {name: 'zoe'})"));

    expect(constants.getSnapshot()).toBe(before); // same reference — never recomputed
    store.graph.free();
  });

  test('liveGremlin recomputes values under the same deps gating', () => {
    const store = newStore();
    const count = store.liveGremlin('g.V().count()', { deps: ['P'] });
    expect(count.getSnapshot()).toEqual([2]); // marko, vadas
    const stable = count.getSnapshot();
    expect(count.getSnapshot()).toBe(stable); // referentially stable between reads

    // A `P` epoch move recomputes; the values (not rows) reflect it.
    store.mutate((g) => g.query("INSERT (:P {name: 'zoe'})"));
    expect(count.getSnapshot()).toEqual([3]);
    store.graph.free();
  });

  test('a read-only mutate() call does not notify', () => {
    const store = newStore();
    const q = store.liveQuery('MATCH (n:P) RETURN n.name', { deps: null });
    let fired = 0;
    q.subscribe(() => {
      fired += 1;
    });
    const v0 = store.version;
    store.mutate((g) => g.query('MATCH (n:P) RETURN n.name')); // read-only
    expect(store.version).toBe(v0);
    expect(fired).toBe(0);
    store.graph.free();
  });

  test('inferDeps extracts labels and property keys', () => {
    expect(
      inferDeps('MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN a.name').sort(),
    ).toEqual(['KNOWS', 'Person', 'age', 'name']);
  });

  test('a live query made invalid by a mutation re-throws on every read, never serves stale', () => {
    using store = newStore();
    // Empty until an :N node exists, so the first read is a clean []. A later insert of
    // non-numeric data makes the CAST throw at exec time.
    const q = store.liveQuery('MATCH (n:N) RETURN CAST(n.x AS INTEGER) AS v', { deps: null });
    expect(q.getSnapshot()).toEqual([]);

    store.mutate((g) => g.query("INSERT (:N {x: 'notanumber'})"));

    // The recompute throws — and CRUCIALLY it throws AGAIN on the next same-version read,
    // rather than silently returning the stale [] (the bookkeeping must not advance past a
    // failed run()).
    expect(() => q.getSnapshot()).toThrow();
    expect(() => q.getSnapshot()).toThrow();
  });
});
