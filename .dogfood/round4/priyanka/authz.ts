import {
  graphFromNdjson,
  type RustGraph,
} from '@lenke/native';
// Zanzibar-style ReBAC authorization service on lenke (native N-API backend).
//
//   check(user, permission, resource)  — resolves:
//     * transitive group membership   (user -MEMBER_OF->* principal)
//     * resource-hierarchy inheritance (resource -PARENT->* ancestor)
//     * role -> permission rules       (OWNER/EDITOR => edit+view; VIEWER => view)
//
// Measures load time and check() latency at scale, and exercises the two
// "expand" queries a real authz service needs: reverse (list a user's
// resources) and forward (who can act on a resource).
import { createNodeBackend } from '@lenke/node/backend';

const dir = import.meta.dir;

// ---------------------------------------------------------------- load
const bytes = await Bun.file(`${dir}/graph.ndjson`).bytes();
const tLoad0 = performance.now();
const g = graphFromNdjson(createNodeBackend(), bytes);
const tLoad1 = performance.now();
console.log(`\n=== LOAD ===`);
console.log(`vertices=${g.vertexCount.toLocaleString()} edges=${g.edgeCount.toLocaleString()}`);
console.log(`load (graphFromNdjson) = ${(tLoad1 - tLoad0).toFixed(1)} ms`);

// Indexes so the check() seed lookups (u.uid / r.rid) seek instead of scan.
const tIdx0 = performance.now();
g.createVertexIndex('uid');
g.createVertexIndex('rid');
g.createVertexIndex('gid');
const tIdx1 = performance.now();
console.log(`build 3 vertex indexes = ${(tIdx1 - tIdx0).toFixed(1)} ms  ->`, g.vertexIndexes());

// ---------------------------------------------------------------- check()
// edit  = a principal of u holds OWNER|EDITOR on r or any ancestor of r
// view  = a principal of u holds OWNER|EDITOR|VIEWER on r or any ancestor of r
//
// PERF-CRITICAL SHAPE: exactly ONE index seed (the resource, via {rid:$r}),
// then a single fully-connected traversal inside EXISTS that reaches the user
// by walking up PARENT, back along the grant edge, then *reverse* MEMBER_OF to
// the requesting user (uid is an inline FILTER on a traversal-bound node, not a
// second seed). A second comma-separated {k:$x} seed is NOT index-backed and
// full-scans its label — the naive two-seed form is ~175x slower (see notes).
const EDIT_ROLES = '[:EDITOR|OWNER]';
const VIEW_ROLES = '[:EDITOR|OWNER|VIEWER]';
const checkSql = (roles: string) => `
  MATCH (r:Resource {rid: $r})
  RETURN EXISTS {
    MATCH (r)-[:PARENT]->*(anc)<-${roles}-(p)<-[:MEMBER_OF]-*(m:User {uid: $u})
  } AS allowed`;

// Prepared statements — parse/lower once, run many.
const editStmt = g.prepare<{ allowed: boolean }>(checkSql(EDIT_ROLES));
const viewStmt = g.prepare<{ allowed: boolean }>(checkSql(VIEW_ROLES));

function check(uid: string, perm: 'edit' | 'view', rid: string): boolean {
  const stmt = perm === 'edit' ? editStmt : viewStmt;
  return stmt.query({ u: uid, r: rid })[0]?.allowed === true;
}

// ---------------------------------------------------------------- correctness
console.log(`\n=== CORRECTNESS (known-answer cases) ===`);
type Case = { uid: string; rid: string; edit: boolean; view: boolean; why: string };
const cases: Case[] = await Bun.file(`${dir}/cases.json`).json();
let pass = 0;
for (const c of cases) {
  const e = check(c.uid, 'edit', c.rid);
  const v = check(c.uid, 'view', c.rid);
  const ok = e === c.edit && v === c.view;
  if (ok) pass++;
  console.log(
    `  ${ok ? 'PASS' : 'FAIL'}  check(${c.uid}, {edit:${e},view:${v}}, ${c.rid})` +
      `  expect{edit:${c.edit},view:${c.view}}  — ${c.why}`,
  );
}
console.log(`  ${pass}/${cases.length} correct`);

// ---------------------------------------------------------------- latency
function bench(label: string, iters: number, fn: (i: number) => void) {
  fn(0); // warm
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  const t1 = performance.now();
  const per = (t1 - t0) / iters;
  console.log(
    `  ${label}: ${iters} iters, ${(t1 - t0).toFixed(1)} ms total, ${per.toFixed(3)} ms/op`,
  );
  return per;
}

console.log(`\n=== check() LATENCY at scale ===`);
// Random real users/resources from the bulk population (index-seeded seeds).
const N = 50_000;
const R = 120_000;
const rndUser = (i: number) => `u${(i * 2654435761) % N}`;
const rndRes = (i: number) => `r${(i * 40503) % R}`;

bench('single check() edit  (random u,r)', 2000, (i) => check(rndUser(i), 'edit', rndRes(i)));
bench('single check() view  (random u,r)', 2000, (i) => check(rndUser(i), 'view', rndRes(i)));
bench('deep transitive check() (u-deep, r-deepdoc)', 2000, () =>
  check('u-deep', 'edit', 'r-deepdoc'),
);

// Batched: a single GQL round-trip that answers many (u,r) pairs via UNWIND-less
// list param is not available, so batch via one query over a param list of pairs
// is emulated by an IN-style expansion. Instead measure amortized throughput of
// the prepared-statement loop (what a service's /batchCheck endpoint would do).
console.log(`\n=== batched check() throughput ===`);
{
  const pairs = Array.from({ length: 5000 }, (_, i) => [rndUser(i), rndRes(i)] as const);
  const t0 = performance.now();
  let allowed = 0;
  for (const [u, r] of pairs) if (check(u, 'view', r)) allowed++;
  const t1 = performance.now();
  console.log(
    `  5000 view checks in ${(t1 - t0).toFixed(1)} ms ` +
      `(${((t1 - t0) / 5000).toFixed(3)} ms/op, ${allowed} allowed) — ${((5000 / (t1 - t0)) * 1000).toFixed(0)} checks/s`,
  );
}

// ---------------------------------------------------------------- expand: reverse
// "list all resources user U can VIEW" — a resource r is viewable if some
// ancestor(-or-self) of r carries a VIEW-role grant to a principal of U.
console.log(`\n=== EXPAND: resources u-deep can view ===`);
{
  const t0 = performance.now();
  const rows = g.query<{ rid: string }>(
    `MATCH (u:User {uid: $u})-[:MEMBER_OF]->*(p)-${VIEW_ROLES}->(gres),
           (r:Resource)-[:PARENT]->*(gres)
     RETURN DISTINCT r.rid AS rid ORDER BY rid`,
    { u: 'u-deep' },
  );
  const t1 = performance.now();
  console.log(
    `  ${rows.length} resources in ${(t1 - t0).toFixed(1)} ms:`,
    rows.map((r) => r.rid).slice(0, 10),
  );
}

// ---------------------------------------------------------------- expand: forward
// "who can EDIT resource R" — expand granted principals (incl. group members).
console.log(`\n=== EXPAND: who can edit r-deepdoc ===`);
{
  const t0 = performance.now();
  const rows = g.query<{ uid: string }>(
    `MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc),
           (principal)-${EDIT_ROLES}->(anc),
           (u:User)-[:MEMBER_OF]->*(principal)
     RETURN DISTINCT u.uid AS uid ORDER BY uid`,
    { r: 'r-deepdoc' },
  );
  const t1 = performance.now();
  console.log(
    `  ${rows.length} users can edit r-deepdoc in ${(t1 - t0).toFixed(1)} ms:`,
    rows.map((r) => r.uid).slice(0, 10),
  );
}

// a bulk-population "who can edit" on a random deep resource, for a scale feel
console.log(`\n=== EXPAND: who can edit a random bulk resource ===`);
{
  const rid = rndRes(7);
  const t0 = performance.now();
  const rows = g.query<{ uid: string }>(
    `MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc),
           (principal)-${EDIT_ROLES}->(anc),
           (u:User)-[:MEMBER_OF]->*(principal)
     RETURN DISTINCT u.uid AS uid`,
    { r: rid },
  );
  const t1 = performance.now();
  console.log(`  ${rows.length} users can edit ${rid} in ${(t1 - t0).toFixed(1)} ms`);
}

editStmt.free();
viewStmt.free();
g.free();
console.log('\ndone.');
