import {
  decodeArrow,
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

import { generate } from './gen.ts';

const N = Number(process.env.N ?? 100000);
const M = Number(process.env.M ?? 10); // attachment => ~M*N edges
const PR_ITERS = Number(process.env.PR_ITERS ?? 20);
const CC_ITERS = Number(process.env.CC_ITERS ?? 30);

const mb = (b: number) => (b / 1048576).toFixed(1) + ' MB';
const rss = () => process.memoryUsage().rss;
function time<T>(label: string, fn: () => T): T {
  const t0 = performance.now();
  const r = fn();
  const ms = performance.now() - t0;
  console.log(`  [${ms.toFixed(0).padStart(6)} ms] ${label}`);
  return r;
}

console.log(
  `=== lenke graph-analytics slice — N=${N} vertices, M=${M} (target ~${M * N} edges) ===\n`,
);

// ---- generate ----
const gen = time(`generate NDJSON (${mb(0)})`, () => generate(N, M));
console.log(
  `  -> ${gen.nVertices} vertices, ${gen.nEdges} edges, ndjson=${mb(gen.bytes.length)}\n`,
);

// ---- load on native backend ----
const backend = createNodeBackend();
const rssBefore = rss();
const g = time('bulk-load NDJSON on native (N-API) backend', () =>
  graphFromNdjson(backend, gen.bytes),
);
const rssAfter = rss();
console.log(`  -> vertexCount=${g.vertexCount} edgeCount=${g.edgeCount}`);
console.log(`  -> RSS load delta: ${mb(rssAfter - rssBefore)} (rss now ${mb(rssAfter)})\n`);

// ============================================================
// 1. DEGREE + WEIGHTED CENTRALITY  — fully in-engine (GQL aggregation)
// ============================================================
console.log('--- Degree / weighted centrality (in-engine GQL aggregation) ---');
time(
  'SET outdeg  = COUNT{ (n)-[:LINK]->() }',
  () => g.query`MATCH (n:V) SET n.outdeg = COUNT { (n)-[:LINK]->() }`,
);
time(
  'SET indeg   = COUNT{ (n)<-[:LINK]-() }',
  () => g.query`MATCH (n:V) SET n.indeg = COUNT { (n)<-[:LINK]-() }`,
);
// weighted in-degree via edge-weight sum (grouped aggregation write)
time('SET wdeg base 0 then sum(weight)', () => {
  g.query`MATCH (n:V) SET n.wdeg = 0.0`;
  g.query`MATCH ()-[e:LINK]->(n:V) WITH n, sum(e.weight) AS s SET n.wdeg = s`;
});
const topDeg = g.query`MATCH (n:V) RETURN n.idx AS idx, n.indeg AS indeg, n.wdeg AS wdeg ORDER BY indeg DESC LIMIT 5`;
console.log('  top-5 by in-degree:', JSON.stringify(topDeg));
console.log();

// ============================================================
// 2. PAGERANK — iteration in-engine (GQL SET + neighbor aggregation),
//    fixpoint loop + dangling-mass scalar carried through JS.
// ============================================================
console.log(`--- PageRank (${PR_ITERS} iters, in-engine SET; loop + dangling scalar in JS) ---`);
const d = 0.85;
g.query`MATCH (n:V) SET n.pr = ${1 / N}`;
time(`PageRank ${PR_ITERS} iterations`, () => {
  for (let it = 0; it < PR_ITERS; it++) {
    // dangling mass = sum of pr over nodes with no out-edges (whole-graph scalar)
    const dm = g.query`MATCH (n:V) WHERE n.outdeg = 0 RETURN sum(n.pr) AS m` as { m: number }[];
    const dangling = (dm[0]?.m ?? 0) || 0;
    const base = (1 - d) / N + (d * dangling) / N;
    g.query`MATCH (n:V) SET n.pr_new = ${base}`;
    g.query`MATCH (m:V)-[:LINK]->(n:V) WITH n, sum(m.pr / m.outdeg) AS inc SET n.pr_new = n.pr_new + ${d} * inc`;
    g.query`MATCH (n:V) SET n.pr = n.pr_new`;
  }
});
const prSum = (g.query`MATCH (n:V) RETURN sum(n.pr) AS s` as { s: number }[])[0].s;
const topPr = g.query`MATCH (n:V) RETURN n.idx AS idx, n.pr AS pr, n.indeg AS indeg ORDER BY pr DESC LIMIT 5`;
console.log(`  pr sum = ${prSum.toFixed(6)} (should be ~1.0)`);
console.log('  top-5 by PageRank:', JSON.stringify(topPr));
console.log();

// ============================================================
// 3. CONNECTED COMPONENTS — min-label propagation, fully in-engine (GQL SET).
// ============================================================
console.log(
  `--- Connected components (min-label propagation, in-engine SET; <=${CC_ITERS} iters) ---`,
);
g.query`MATCH (n:V) SET n.comp = n.idx`;
let ccIters = 0;
time('connected components to fixpoint', () => {
  for (let it = 0; it < CC_ITERS; it++) {
    ccIters++;
    // undirected: propagate min in both directions + measure change via global sum
    const before = (g.query`MATCH (n:V) RETURN sum(n.comp) AS s` as { s: number }[])[0].s;
    g.query`MATCH (n:V)-[:LINK]->(m:V) WITH m, min(n.comp) AS c SET m.comp = CASE WHEN c < m.comp THEN c ELSE m.comp END`;
    g.query`MATCH (n:V)-[:LINK]->(m:V) WITH n, min(m.comp) AS c SET n.comp = CASE WHEN c < n.comp THEN c ELSE n.comp END`;
    const after = (g.query`MATCH (n:V) RETURN sum(n.comp) AS s` as { s: number }[])[0].s;
    if (after === before) break;
  }
});
const nComp = (g.query`MATCH (n:V) RETURN count(DISTINCT n.comp) AS c` as { c: number }[])[0].c;
console.log(`  converged in ${ccIters} iterations; #components = ${nComp}`);
console.log();

// ============================================================
// 4. LABEL PROPAGATION (community) — argmax/mode is NOT an in-engine aggregate,
//    so gather neighbor labels in-engine (collect_list) but decide in JS.
// ============================================================
console.log('--- Label propagation (gather in-engine, majority-vote/argmax in JS) ---');
// seed labels = node idx; build adjacency in JS by pulling the edge list ONCE via Arrow.
const lpIters = 5;
time(`label propagation ${lpIters} sweeps (pull edges once, argmax in JS)`, () => {
  // pull edge endpoints as columnar Arrow (scalar float64 idx columns)
  const blob = g.queryArrow`MATCH (a:V)-[:LINK]->(b:V) RETURN a.idx AS s, b.idx AS t`;
  const edges = decodeArrow<{ s: number; t: number }>(blob);
  const nbr = new Map<number, number[]>();
  for (const { s, t } of edges) {
    (nbr.get(s) ?? nbr.set(s, []).get(s)!).push(t);
    (nbr.get(t) ?? nbr.set(t, []).get(t)!).push(s);
  }
  const label = new Map<number, number>();
  for (const k of nbr.keys()) label.set(k, k);
  for (let it = 0; it < lpIters; it++) {
    for (const [n, ns] of nbr) {
      const cnt = new Map<number, number>();
      for (const m of ns) cnt.set(label.get(m)!, (cnt.get(label.get(m)!) ?? 0) + 1);
      let best = label.get(n)!;
      let bestC = -1;
      for (const [lb, c] of cnt)
        if (c > bestC || (c === bestC && lb < best)) ((best = lb), (bestC = c));
      label.set(n, best);
    }
  }
  const communities = new Set(label.values());
  console.log(`  -> ${communities.size} communities over ${nbr.size} connected nodes`);
});
console.log();

// ============================================================
// 5. ARROW FEATURE EXPORT — wide numeric feature matrix, columnar.
// ============================================================
console.log('--- Arrow feature-vector export (columnar, in-engine -> decodeArrow) ---');
const featBlob = time(
  'queryArrow feature matrix',
  () =>
    g.queryArrow`MATCH (n:V)
    RETURN element_id(n) AS id, n.idx AS idx, n.indeg AS indeg, n.outdeg AS outdeg,
           n.wdeg AS wdeg, n.pr AS pr, n.comp AS comp
    ORDER BY pr DESC`,
);
console.log(`  arrow blob = ${mb(featBlob.length)}`);
const feats = time('decodeArrow -> row objects', () => decodeArrow(featBlob));
console.log(`  decoded ${feats.length} feature vectors; sample:`);
console.log('  ', JSON.stringify(feats[0]));
console.log('  ', JSON.stringify(feats[1]));

// precision check: comp is an integer stored/round-tripped through float64
const compIsInt = feats.every((f: any) => Number.isInteger(f.comp) && Number.isInteger(f.idx));
console.log(`  integer columns (idx/comp) survived float64 round-trip exactly: ${compIsInt}`);

console.log(`\n=== done. final RSS ${mb(rss())} ===`);
g.free();
