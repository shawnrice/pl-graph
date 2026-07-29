// Write-side benchmark: how much does maintaining a property index cost on
// node creation? Builds N vertices three ways — no index, a low-cardinality
// index (age, ~100 distinct), and a high-cardinality index (name, unique) —
// across sizes, to expose linear vs. quadratic scaling.
//
//   bun run benchmarks/write-bench.ts

import { Graph } from '@lenke/core';

const SIZES = [25_000, 50_000, 100_000];

const fmt = (n: number): string => {
  if (n >= 100) {
    return n.toFixed(0);
  }

  return n >= 1 ? n.toFixed(2) : n.toFixed(3);
};
const pad = (s: string, n: number): string => s.padEnd(n);

/** Time building `n` vertices; `index` declares an index up front, or not. */
const buildTime = (n: number, index: 'none' | 'age' | 'name'): number => {
  const g = new Graph();
  g.disableEvents();

  if (index !== 'none') {
    g.createIndex({ on: 'vertex', kind: 'hash', keys: [index] });
  }

  const start = performance.now();

  for (let i = 0; i < n; i++) {
    g.addVertex({
      id: `n${i}`,
      labels: ['Person'],
      properties: { name: `p${i}`, age: i % 100, active: i % 2 === 0, dept: `d${i % 10}` },
    });
  }

  return performance.now() - start;
};

console.log('\nNode-creation cost (ms) by index, and ns/node:\n');
console.log(
  `${pad('N', 10)} ${pad('no index', 18)} ${pad('age idx (100 vals)', 22)} ${pad('name idx (unique)', 22)}`,
);
console.log('-'.repeat(74));

for (const n of SIZES) {
  const none = buildTime(n, 'none');
  const age = buildTime(n, 'age');
  const name = buildTime(n, 'name');
  const perNode = (ms: number): string => `${fmt(ms)}ms (${fmt((ms / n) * 1e6)}ns)`;
  console.log(
    `${pad(n.toLocaleString(), 10)} ${pad(perNode(none), 18)} ${pad(perNode(age), 22)} ${pad(perNode(name), 22)}`,
  );
}

console.log();
