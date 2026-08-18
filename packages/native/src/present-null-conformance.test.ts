import { describe, expect, test } from 'bun:test';
// Cross-engine differential for STORED PRESENT-NULL — the value `SET n.k = null`
// stores, distinct from an absent (REMOVE'd) key. lenke-core boxes such a column to
// `Mixed`; lenke-engine boxes it to `Gen` today and (future) will keep it typed with a
// nulls bit. Whatever the storage, the two engines must agree on every observable:
// query results, aggregates, string search, and the NDJSON round-trip. Loads BOTH
// backends and compares them op-for-op, so it guards the drop-in directly.
//
//   bun test src/present-null-conformance.test.ts
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { createFfiBackend } from './backend-ffi.js';
import type { Backend } from './backend.js';

const LIB_EXT = { darwin: 'dylib', win32: 'dll' }[process.platform as string] ?? 'so';
const CORE_LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const ENGINE_LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;

const decoder = new TextDecoder();
const encoder = new TextEncoder();

// One node per state: value / present-null / absent, plus decoys with a real 0 and a
// string, so `= 0`, `min`, `count`, and string search all have something to get wrong.
const NDJSON = [
  { type: 'node', id: 'a', labels: ['P'], properties: { id: 'a', age: 10, city: 'oslo' } },
  { type: 'node', id: 'b', labels: ['P'], properties: { id: 'b', age: 20, city: 'oskar' } },
  { type: 'node', id: 'c', labels: ['P'], properties: { id: 'c', age: 30, city: 'bergen' } },
  { type: 'node', id: 'd', labels: ['P'], properties: { id: 'd', age: 0, city: 'oslo' } },
]
  .map((r) => JSON.stringify(r))
  .join('\n');

// After this, `age`/`city` carry a present-null (c) and an absent (d) → the column
// de-opts on both engines, exercising the boxed read paths.
const MUTATIONS = [
  "MATCH (n:P {id:'c'}) SET n.age = null",
  "MATCH (n:P {id:'c'}) SET n.city = null",
  "MATCH (n:P {id:'d'}) REMOVE n.age",
  "MATCH (n:P {id:'d'}) REMOVE n.city",
];

const QUERIES = [
  'MATCH (n:P) RETURN n.age AS x, n.id AS t ORDER BY t',
  'MATCH (n:P) WHERE n.age = 0 RETURN n.id AS x ORDER BY x',
  'MATCH (n:P) WHERE n.age IS NULL RETURN n.id AS x ORDER BY x',
  'MATCH (n:P) WHERE n.age IS NOT NULL RETURN n.id AS x ORDER BY x',
  'MATCH (n:P) RETURN min(n.age) AS a, max(n.age) AS b, sum(n.age) AS c, count(n.age) AS d, count(*) AS e',
  "MATCH (n:P) WHERE n.city STARTS WITH 'os' RETURN n.id AS x ORDER BY x",
  "MATCH (n:P) WHERE n.city CONTAINS 'ka' RETURN n.id AS x ORDER BY x",
  'MATCH (n:P) RETURN n.city AS x, count(*) AS c GROUP BY n.city ORDER BY x',
  'MATCH (n:P) RETURN DISTINCT n.age AS x ORDER BY x',
];

const run = (make: () => Backend): { rows: string[]; ndjson: string } => {
  const backend = make();
  const h = backend.graphFromNdjson(encoder.encode(NDJSON), false);

  try {
    for (const m of MUTATIONS) {
      backend.queryRows(h, m, '{}');
    }

    const rows = QUERIES.map((q) => decoder.decode(backend.queryRows(h, q, '{}')));
    // A present-null must survive the round-trip AS a null (not vanish to absent).
    const ndjson = decoder.decode(backend.encodeNdjson(h));

    return { rows, ndjson };
  } finally {
    backend.graphFree(h);
  }
};

// Compare NDJSON as a set of canonical records — property order within a node is each
// engine's own business (unspecified), only the (id → props) mapping is observable.
const canonNdjson = (nd: string): string[] =>
  nd
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      const r = JSON.parse(line) as { properties?: Record<string, unknown> };
      if (r.properties) {
        r.properties = Object.fromEntries(
          Object.entries(r.properties).sort(([x], [y]) => x.localeCompare(y)),
        );
      }
      return JSON.stringify(r);
    })
    .sort();

describe('present-null: engine ⟷ core agree', () => {
  const ready = existsSync(CORE_LIB) && existsSync(ENGINE_LIB);
  const t = ready ? test : test.skip;

  t('query results and the NDJSON round-trip match op-for-op', () => {
    const core = run(() => createFfiBackend(CORE_LIB));
    const engine = run(() => createFfiEngineBackend(ENGINE_LIB));

    for (let i = 0; i < QUERIES.length; i++) {
      expect(engine.rows[i], `query diverged: ${QUERIES[i]}`).toBe(core.rows[i]);
    }
    expect(canonNdjson(engine.ndjson), 'NDJSON round-trip diverged').toEqual(
      canonNdjson(core.ndjson),
    );
  });
});
