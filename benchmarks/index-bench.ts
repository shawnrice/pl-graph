// Ad-hoc benchmark: property-index seeding vs. full-scan, for the GQL and
// Gremlin engines. Builds one graph, then runs each query against an
// unindexed copy and an indexed copy and reports the speedup.
//
//   bun run benchmarks/index-bench.ts [nVertices] [avgDegree]

import { Graph } from '@lenke/core';

import { query } from '../packages/gql/src/index.js';
import { gt, has, traversal, toArray, V } from '../packages/gremlin/src/index.js';
import { ndjsonCodec } from '../packages/serialization/src/index.js';
import { genNdjson } from './datagen.js';

const N = Number(process.argv[2] ?? 50_000);
const DEG = Number(process.argv[3] ?? 4);

const load = (ndjson: string): Graph => ndjsonCodec.decode(ndjson, new Graph());

/** Run `fn` for ~`budgetMs`, return mean ms/op and ops/sec. */
const bench = (fn: () => void, budgetMs = 800): { mean: number; ops: number } => {
  // warm up
  for (let i = 0; i < 3; i++) {
    fn();
  }

  let iters = 0;
  const start = performance.now();
  let now = start;

  while (now - start < budgetMs) {
    fn();
    iters++;
    now = performance.now();
  }

  const elapsed = now - start;

  return { mean: elapsed / iters, ops: (iters / elapsed) * 1000 };
};

// A stable multiset key for a row array (order-insensitive comparison).
const bag = (rs: unknown[]): string =>
  rs
    .map((r) => JSON.stringify(r))
    .sort()
    .join('\n');

const pad = (s: string, n: number): string => s.padEnd(n);
const fmt = (n: number): string => {
  if (n >= 100) {
    return n.toFixed(0);
  }

  return n >= 1 ? n.toFixed(2) : n.toFixed(4);
};

const { ndjson } = genNdjson(N, DEG);
console.log(
  `\nDataset: ${N.toLocaleString()} :Person vertices, ${(N * DEG).toLocaleString()} :KNOWS edges`,
);
console.log('Props: name (unique-ish), age (100 values), active (bool), dept (10 values)\n');

const plain = load(ndjson);

const indexed = load(ndjson);
const t0 = performance.now();
indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
const buildMs = performance.now() - t0;
console.log(`Index build (name + age backfill): ${fmt(buildMs)} ms\n`);

type Case = { label: string; run: (g: Graph) => unknown };

const cases: Case[] = [
  {
    label: "GQL  WHERE n.name = 'p49999' (unique eq)",
    run: (g) => query(g, `MATCH (n:Person) WHERE n.name = 'p49999' RETURN n.age`),
  },
  {
    label: "GQL  (n:Person {name: 'p49999'}) (element-map eq)",
    run: (g) => query(g, `MATCH (n:Person {name: 'p49999'}) RETURN n.age`),
  },
  {
    label: 'GQL  WHERE n.age = 42 (eq, ~1%)',
    run: (g) => query(g, `MATCH (n:Person) WHERE n.age = 42 RETURN n.name`),
  },
  {
    label: 'GQL  WHERE n.age >= 98 (range, ~2%)',
    run: (g) => query(g, `MATCH (n:Person) WHERE n.age >= 98 RETURN n.name`),
  },
  {
    label: "GQL  (a)-[:KNOWS]->(b) WHERE b.name='p123' (far-end seed)",
    run: (g) =>
      query(g, `MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.name = 'p123' RETURN a.name`),
  },
  {
    label: "Grem V().has('name','p49999')",
    run: (g) => toArray(traversal(V(), has('name', 'p49999')), g),
  },
  {
    label: "Grem V().has('age', gt(98))",
    run: (g) => toArray(traversal(V(), has('age', gt(98))), g),
  },
];

console.log(
  `${pad('Query', 56)} ${pad('scan ms', 10)} ${pad('index ms', 10)} ${pad('speedup', 9)} rows`,
);
console.log('-'.repeat(96));

for (const c of cases) {
  const plainRows = c.run(plain) as unknown[];
  const indexedRows = c.run(indexed) as unknown[];
  const rows = Array.isArray(indexedRows) ? indexedRows.length : 0;
  // Compare as multisets — index seeding may reorder rows (spec-legal without
  // ORDER BY), but the set of rows must be identical.
  const same = bag(plainRows) === bag(indexedRows);

  const a = bench(() => c.run(plain));
  const b = bench(() => c.run(indexed));
  const speedup = a.mean / b.mean;

  console.log(
    `${pad(c.label, 56)} ${pad(fmt(a.mean), 10)} ${pad(fmt(b.mean), 10)} ${pad(`${fmt(speedup)}x`, 9)} ${rows}${same ? '' : '  ⚠ MISMATCH'}`,
  );
}

console.log();
