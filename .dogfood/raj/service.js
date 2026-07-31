// Raj's "who-to-follow" + fraud-signal slice on lenke (N-API backend).
// Run: bun .dogfood/raj/service.js
//
// Exercises: createNodeBackend -> graphFromNdjson (bulk) -> mergeNdjson (+report)
// -> createVertexIndex -> prepare() in a hot loop -> friend-of-friend / mutuals.

import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

import { makeBatch, toNdjson } from './seed.js';

const hr = (label) => console.log(`\n=== ${label} ===`);
const bench = (label, iters, fn) => {
  // warmup
  for (let i = 0; i < 50; i++) fn(i);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  const ms = performance.now() - t0;
  console.log(`${label}: ${iters} iters in ${ms.toFixed(1)}ms  (${(ms / iters).toFixed(3)}ms/op)`);
  return ms;
};

// ---------------------------------------------------------------------------
hr('1. backend + bulk load');
const backend = createNodeBackend();
console.log('abiVersion:', backend.abiVersion);

// initial bulk load: batch A (users 0..1999)
const a = makeBatch({ start: 0, count: 2000, avgOut: 8, seed: 42 });
const ndjsonA = toNdjson([...a.nodeLines, ...a.edgeLines]);
console.log(
  `batch A ndjson: ${(ndjsonA.length / 1024).toFixed(0)}KB, ${a.nodeLines.length} nodes, ${a.edgeLines.length} edges`,
);

using g = graphFromNdjson(backend, ndjsonA); // freed at scope exit
console.log(`loaded -> vertexCount=${g.vertexCount} edgeCount=${g.edgeCount}`);

// ---------------------------------------------------------------------------
hr('2. mergeNdjson bulk-append + MergeReport');
// batch B (users 2000..2999) PLUS a deliberate duplicate node (p0) and an edge
// whose endpoint (p999999) was never declared as a node -> phantom vertex.
const b = makeBatch({ start: 2000, count: 1000, avgOut: 8, seed: 7 });
const dupNode = JSON.stringify({
  type: 'node',
  id: 'p0',
  labels: ['Person'],
  properties: { uid: 0, name: 'DUP' },
});
const phantomEdge = JSON.stringify({
  type: 'edge',
  from: 'p2000',
  to: 'p999999',
  labels: ['FOLLOWS'],
  properties: {},
});
const ndjsonB = toNdjson([...b.nodeLines, dupNode, ...b.edgeLines, phantomEdge]);

const report = g.mergeNdjson(ndjsonB);
console.log('MergeReport:', {
  nodesAdded: report.nodesAdded,
  edgesAdded: report.edgesAdded,
  nodesSkipped: report.nodesSkipped, // expect ['p0'] (dup, first-wins)
  edgesSkipped: report.edgesSkipped,
  phantomVertices: report.phantomVertices, // expect ['p999999']
});
console.log(`after merge -> vertexCount=${g.vertexCount} edgeCount=${g.edgeCount}`);

// ---------------------------------------------------------------------------
hr('3. sanity queries');
const [{ n: personCount }] = g.query('MATCH (p:Person) RETURN count(*) AS n');
console.log('persons:', personCount);
const topByCity = g.query('MATCH (p:Person) WHERE p.city = $c RETURN count(*) AS n', { c: 'sf' });
console.log('persons in sf:', topByCity[0].n);

// ---------------------------------------------------------------------------
hr('4. index: point-lookup by uid, unindexed vs indexed');
const SEED_UID = 123;
const pointLookup = (uid) =>
  g.query('MATCH (p:Person) WHERE p.uid = $uid RETURN p.name AS name', { uid });

console.log('vertexIndexes before:', g.vertexIndexes());
const msScan = bench('point lookup (full scan)', 3000, (i) => pointLookup(i % 3000));

g.createVertexIndex('uid');
console.log('vertexIndexes after createVertexIndex("uid"):', g.vertexIndexes());
const msSeek = bench('point lookup (index seek)', 3000, (i) => pointLookup(i % 3000));
console.log(`index speedup: ${(msScan / msSeek).toFixed(1)}x`);

console.log('lookup uid=123 ->', pointLookup(SEED_UID));

// ---------------------------------------------------------------------------
hr('5. prepared statement in a hot loop (prepared vs unprepared)');
const QTEXT =
  'MATCH (me:Person {uid: $uid})-[:FOLLOWS]->(f)-[:FOLLOWS]->(c) ' +
  'WHERE c.uid <> $uid AND NOT EXISTS { MATCH (me)-[:FOLLOWS]->(c) } ' +
  'RETURN c.uid AS candidate, count(*) AS mutuals ' +
  'ORDER BY mutuals DESC, candidate ASC LIMIT 5';

using prepared = g.prepare(QTEXT);

// correctness: prepared and one-shot agree
const sampleUid = 50;
const viaPrepared = prepared.query({ uid: sampleUid });
const viaOneShot = g.query(QTEXT, { uid: sampleUid });
console.log('who-to-follow for uid=50 (prepared):', viaPrepared);
console.log('prepared == one-shot:', JSON.stringify(viaPrepared) === JSON.stringify(viaOneShot));

const msUnprepared = bench('recommend (one-shot query)', 2000, (i) =>
  g.query(QTEXT, { uid: i % 3000 }),
);
const msPrepared = bench('recommend (prepared.query)', 2000, (i) =>
  prepared.query({ uid: i % 3000 }),
);
console.log(`prepared speedup (traversal-bound): ${(msUnprepared / msPrepared).toFixed(2)}x`);

// A cheap (parse-dominated) query — this is where prepare() actually pays off,
// since the per-call lex/parse/lower is a bigger share of total work.
const CHEAP = 'MATCH (p:Person {uid: $uid}) RETURN p.name AS name';
using cheapPrepared = g.prepare(CHEAP);
const msCheapOneShot = bench('cheap (one-shot query)', 5000, (i) =>
  g.query(CHEAP, { uid: i % 3000 }),
);
const msCheapPrepared = bench('cheap (prepared.query)', 5000, (i) =>
  cheapPrepared.query({ uid: i % 3000 }),
);
console.log(`prepared speedup (parse-bound): ${(msCheapOneShot / msCheapPrepared).toFixed(2)}x`);

// ---------------------------------------------------------------------------
hr('6. recommendation queries');
// friend-of-friend (2-hop reach, dedup)
const fof = g.query(
  'MATCH (me:Person {uid: $uid})-[:FOLLOWS]->()-[:FOLLOWS]->(c) ' +
    'WHERE c.uid <> $uid RETURN count(DISTINCT c.uid) AS reach',
  { uid: sampleUid },
);
console.log('uid=50 2-hop reach:', fof[0].reach);

// mutual-connection count between two users (shared people they both follow)
const mutuals = g.query(
  'MATCH (a:Person {uid: $a})-[:FOLLOWS]->(x)<-[:FOLLOWS]-(b:Person {uid: $b}) ' +
    'RETURN count(DISTINCT x.uid) AS shared',
  { a: 50, b: 51 },
);
console.log('mutual follows(50,51):', mutuals[0].shared);

// fraud signal: reciprocal-follow ring detection (mutual A<->B follows)
const reciprocal = g.query(
  'MATCH (a:Person)-[:FOLLOWS]->(b:Person)-[:FOLLOWS]->(a) ' +
    'WHERE a.uid < b.uid RETURN count(*) AS reciprocal_pairs',
);
console.log('reciprocal follow pairs (fraud signal):', reciprocal[0].reciprocal_pairs);

hr('done');
