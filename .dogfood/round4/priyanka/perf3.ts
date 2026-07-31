import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const dir = import.meta.dir;
const g = graphFromNdjson(createNodeBackend(), await Bun.file(`${dir}/graph.ndjson`).bytes());
g.createVertexIndex('uid');
g.createVertexIndex('rid');
g.createVertexIndex('gid');

const rndUser = (i: number) => `u${(i * 2654435761) % 50000}`;
const rndRes = (i: number) => `r${(i * 40503) % 120000}`;

function bench(label: string, iters: number, fn: (i: number) => void) {
  fn(0);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  const t1 = performance.now();
  console.log(`  ${label}: ${((t1 - t0) / iters).toFixed(4)} ms/op`);
}

// Single-seed on r; ancestors small; incoming grants per ancestor small;
// EXISTS re-seeds u (fast single seek) and walks membership to the candidate.
const q1 = g.prepare<{ allowed: boolean }>(`
  MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p)
  RETURN EXISTS {
    MATCH (u:User {uid: $u})-[:MEMBER_OF]->*(p)
  } AS allowed LIMIT 1`);
bench('r-seed, EXISTS re-seed u', 5000, (i) => q1.query({ u: rndUser(i), r: rndRes(i) }));
q1.free();

// Same, but as a top-level EXISTS returning a single boolean row (no LIMIT needed).
const q2 = g.prepare<{ allowed: boolean }>(`
  MATCH (u:User {uid: $u})
  RETURN EXISTS {
    MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p),
          (u)-[:MEMBER_OF]->*(p)
  } AS allowed`);
bench('u-seed outer, EXISTS r-driven', 5000, (i) => q2.query({ u: rndUser(i), r: rndRes(i) }));
q2.free();

// Correctness spot check on q2 for the deep case
const q2c = g.prepare<{ allowed: boolean }>(`
  MATCH (u:User {uid: $u})
  RETURN EXISTS {
    MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc)<-[:EDITOR|OWNER]-(p),
          (u)-[:MEMBER_OF]->*(p)
  } AS allowed`);
console.log('  deep edit (expect true):', q2c.query({ u: 'u-deep', r: 'r-deepdoc' })[0]?.allowed);
console.log('  none edit (expect false):', q2c.query({ u: 'u-none', r: 'r-deepdoc' })[0]?.allowed);
console.log('  owner edit (expect true):', q2c.query({ u: 'u-owner', r: 'r-owned' })[0]?.allowed);
console.log(
  '  viewer edit (expect false):',
  q2c.query({ u: 'u-viewer', r: 'r-shared' })[0]?.allowed,
);
q2c.free();

g.free();
