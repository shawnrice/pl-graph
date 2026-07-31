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

const a = g.prepare('MATCH (r:Resource {rid: $r}) RETURN r.rid AS rid');
bench('single Resource seek (rid, inline)', 5000, (i) => a.query({ r: rndRes(i) }));
a.free();

const b = g.prepare('MATCH (r:Resource) WHERE r.rid = $r RETURN r.rid AS rid');
bench('single Resource seek (rid, WHERE)', 5000, (i) => b.query({ r: rndRes(i) }));
b.free();

// order swapped: resource first, user second
const c = g.prepare('MATCH (r:Resource {rid: $r}), (u:User {uid: $u}) RETURN 1 AS x');
bench('two seeds (resource first)', 5000, (i) => c.query({ u: rndUser(i), r: rndRes(i) }));
c.free();

// same-label second seed: two users
const d = g.prepare('MATCH (u:User {uid: $u}), (v:User {uid: $v}) RETURN 1 AS x');
bench('two seeds (both User)', 5000, (i) => d.query({ u: rndUser(i), v: rndUser(i + 1) }));
d.free();

// group seek alone
const e = g.prepare('MATCH (x:Group {gid: $x}) RETURN x.gid AS g');
bench('single Group seek (gid)', 5000, (i) => e.query({ x: `g${i % 4000}` }));
e.free();

g.free();
