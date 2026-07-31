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
  console.log(`  ${label}: ${((t1 - t0) / iters).toFixed(4)} ms/op  (${iters} iters)`);
}

// 1. marshalling floor: trivial prepared query
const p1 = g.prepare('RETURN 1 AS x');
bench('RETURN 1 (marshalling floor)', 5000, () => p1.query());
p1.free();

// 2. single index seek
const p2 = g.prepare('MATCH (u:User {uid: $u}) RETURN u.uid AS uid');
bench('index seek user', 5000, (i) => p2.query({ u: rndUser(i) }));
p2.free();

// 3. two seeds (u and r), no traversal
const p3 = g.prepare('MATCH (u:User {uid: $u}), (r:Resource {rid: $r}) RETURN 1 AS x');
bench('two index seeds', 5000, (i) => p3.query({ u: rndUser(i), r: rndRes(i) }));
p3.free();

// 4. principals only (transitive membership)
const p4 = g.prepare('MATCH (u:User {uid: $u})-[:MEMBER_OF]->*(p) RETURN count(*) AS c');
bench('principals of u (member*)', 5000, (i) => p4.query({ u: rndUser(i) }));
p4.free();

// 5. ancestors only
const p5 = g.prepare('MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc) RETURN count(*) AS c');
bench('ancestors of r (parent*)', 5000, (i) => p5.query({ r: rndRes(i) }));
p5.free();

// 6. the real EXISTS check (current formulation)
const p6 = g.prepare(`
  MATCH (u:User {uid: $u}), (r:Resource {rid: $r})
  RETURN EXISTS {
    MATCH (u)-[:MEMBER_OF]->*(p)-[:EDITOR|OWNER|VIEWER]->(anc),
          (r)-[:PARENT]->*(anc)
  } AS allowed`);
bench('EXISTS check (u forward, r forward)', 3000, (i) =>
  p6.query({ u: rndUser(i), r: rndRes(i) }),
);
p6.free();

// 7. alternative: drive from r ancestors, reverse grant, reverse membership to pinned u
const p7 = g.prepare(`
  MATCH (u:User {uid: $u}), (r:Resource {rid: $r})
  RETURN EXISTS {
    MATCH (r)-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p),
          (p)<-[:MEMBER_OF]-*(u)
  } AS allowed`);
bench('EXISTS check (r-driven, reverse)', 3000, (i) => p7.query({ u: rndUser(i), r: rndRes(i) }));
p7.free();

g.free();
