import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const dir = import.meta.dir;
const g = graphFromNdjson(createNodeBackend(), await Bun.file(`${dir}/graph.ndjson`).bytes());
g.createVertexIndex('uid');
g.createVertexIndex('rid');

const rndUser = (i: number) => `u${(i * 2654435761) % 50000}`;
const rndRes = (i: number) => `r${(i * 40503) % 120000}`;

function bench(label: string, iters: number, fn: (i: number) => void) {
  fn(0);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  const t1 = performance.now();
  console.log(`  ${label}: ${((t1 - t0) / iters).toFixed(4)} ms/op`);
}

// A) single connected pattern, seed r, reach u by reverse membership, uid inline
const A = g.prepare<{ c: number }>(`
  MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p)<-[:MEMBER_OF]-*(m:User {uid: $u})
  RETURN count(*) AS c`);
bench('A: r-seed, connected, uid inline (random)', 5000, (i) =>
  A.query({ u: rndUser(i), r: rndRes(i) }),
);
bench('A: deep case', 3000, () => A.query({ u: 'u-deep', r: 'r-deepdoc' }));
console.log('   A deep c=', A.query({ u: 'u-deep', r: 'r-deepdoc' })[0]?.c, '(expect >0)');
console.log('   A none c=', A.query({ u: 'u-none', r: 'r-deepdoc' })[0]?.c, '(expect 0)');
console.log('   A owner c=', A.query({ u: 'u-owner', r: 'r-owned' })[0]?.c, '(expect >0)');
A.free();

// B) same but seed u, reach r by forward chain, rid inline
const B = g.prepare<{ c: number }>(`
  MATCH (u:User {uid: $u})-[:MEMBER_OF]->*(p)-[:EDITOR|OWNER|VIEWER]->(anc)<-[:PARENT]-*(r:Resource {rid: $r})
  RETURN count(*) AS c`);
bench('B: u-seed, connected, rid inline (random)', 5000, (i) =>
  B.query({ u: rndUser(i), r: rndRes(i) }),
);
bench('B: deep case', 3000, () => B.query({ u: 'u-deep', r: 'r-deepdoc' }));
console.log('   B deep c=', B.query({ u: 'u-deep', r: 'r-deepdoc' })[0]?.c, '(expect >0)');
console.log('   B none c=', B.query({ u: 'u-none', r: 'r-deepdoc' })[0]?.c, '(expect 0)');
B.free();

// C) EXISTS wrapper on A for short-circuit boolean (no full count)
const C = g.prepare<{ allowed: boolean }>(`
  MATCH (r:Resource {rid: $r})
  RETURN EXISTS {
    MATCH (r)-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p)<-[:MEMBER_OF]-*(m:User {uid: $u})
  } AS allowed`);
bench('C: r-seed + EXISTS short-circuit (random)', 5000, (i) =>
  C.query({ u: rndUser(i), r: rndRes(i) }),
);
bench('C: deep case', 3000, () => C.query({ u: 'u-deep', r: 'r-deepdoc' }));
console.log('   C deep=', C.query({ u: 'u-deep', r: 'r-deepdoc' })[0]?.allowed);
console.log('   C none=', C.query({ u: 'u-none', r: 'r-deepdoc' })[0]?.allowed);
console.log('   C owner=', C.query({ u: 'u-owner', r: 'r-owned' })[0]?.allowed);
console.log('   C viewerEdit(should be false w/o VIEWER)=');
C.free();

g.free();
