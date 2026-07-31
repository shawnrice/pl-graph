// PageRank: old in-engine GQL fixpoint loop (per-iteration WITH-aggregation + SET,
// dangling scalar carried through JS) vs the new one-call native `g.pagerank()`.
// Same graph, same damping (0.85), same 20 iterations — compares wall-clock and
// verifies the result agrees (pr sum ~1.0, same top nodes).
//
// Run: N=100000 M=10 bun .dogfood/round4/marcus/pagerank_bench.ts
import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

import { generate } from './gen.ts';

const N = Number(process.env.N ?? 100000);
const M = Number(process.env.M ?? 10);
const ITERS = Number(process.env.PR_ITERS ?? 20);
const d = 0.85;

const time = async <T>(label: string, fn: () => T | Promise<T>): Promise<T> => {
  const t0 = performance.now();
  const r = await fn();
  console.log(`  [${(performance.now() - t0).toFixed(0).padStart(7)} ms] ${label}`);
  return r;
};

const gen = generate(N, M);
const backend = createNodeBackend();
const g = graphFromNdjson(backend, gen.bytes);
console.log(`graph: ${g.vertexCount} vertices, ${g.edgeCount} edges; PageRank ${ITERS} iters\n`);

// --- old approach: GQL fixpoint loop (needs outdeg materialized first) ---
g.query`MATCH (n:V) SET n.outdeg = COUNT { (n)-[:LINK]->() }`;
g.query`MATCH (n:V) SET n.pr = ${1 / N}`;
await time(`GQL fixpoint loop (${ITERS} iters, WITH-agg + SET per iter)`, () => {
  for (let it = 0; it < ITERS; it++) {
    const dm = g.query`MATCH (n:V) WHERE n.outdeg = 0 RETURN sum(n.pr) AS m` as { m: number }[];
    const base = (1 - d) / N + (d * ((dm[0]?.m ?? 0) || 0)) / N;
    g.query`MATCH (n:V) SET n.pr_new = ${base}`;
    g.query`MATCH (m:V)-[:LINK]->(n:V) WITH n, sum(m.pr / m.outdeg) AS inc SET n.pr_new = n.pr_new + ${d} * inc`;
    g.query`MATCH (n:V) SET n.pr = n.pr_new`;
  }
});
const gqlSum = (g.query`MATCH (n:V) RETURN sum(n.pr) AS s` as { s: number }[])[0]!.s;

// --- new approach: one native call ---
const rows = await time('native g.pagerank({}) — whole computation in one call', () =>
  g.pagerank({ iterations: ITERS }),
);
const nativeSum = rows.reduce((s, r) => s + r.score, 0);

console.log();
console.log(`  GQL-loop    pr sum = ${gqlSum.toFixed(6)}`);
console.log(`  native      pr sum = ${nativeSum.toFixed(6)}`);
const top = [...rows].sort((a, b) => b.score - a.score).slice(0, 5);
console.log(`  native top-5: ${JSON.stringify(top)}`);
