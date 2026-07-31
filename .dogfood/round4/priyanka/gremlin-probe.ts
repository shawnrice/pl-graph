import {
  graphFromNdjson,
  gremlin,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';
const g = graphFromNdjson(
  createNodeBackend(),
  await Bun.file(`${import.meta.dir}/graph.ndjson`).bytes(),
);
g.createVertexIndex('rid');

// Gremlin edit-check. To get zero-hop-INCLUSIVE transitive closure (self +
// ancestors, self + members) — matching GQL `->*` — neither post-form
// `repeat(...).emit()` nor prefix `emit().repeat(...)` works: both drop the
// start vertex. The working form is `union(identity(), repeat(...).emit())`.
// Textual Gremlin uses `in('MEMBER_OF')` (there is no `__in` / anonymous `__`).
function canEdit(uid: string, rid: string): boolean {
  const t = gremlin`
    g.V().has('rid', ${rid})
      .union(identity(), repeat(out('PARENT')).emit()).dedup()
      .in('EDITOR','OWNER')
      .union(identity(), repeat(in('MEMBER_OF')).emit()).dedup()
      .has('uid', ${uid})
      .limit(1).count()`;
  return ((g.gremlin(t) as number[])[0] ?? 0) > 0;
}
const cases: [string, string, boolean][] = [
  ['u-deep', 'r-deepdoc', true],
  ['u-none', 'r-deepdoc', false],
  ['u-owner', 'r-owned', true],
  ['u-editor', 'r-shared', true],
  ['u-deep', 'r-owned', false],
];
let pass = 0;
for (const [u, r, exp] of cases) {
  const got = canEdit(u, r);
  const ok = got === exp;
  if (ok) pass++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} gremlin canEdit(${u},${r})=${got} expect ${exp}`);
}
canEdit('u-deep', 'r-deepdoc'); // warm
const t0 = performance.now();
const IT = 2000;
for (let i = 0; i < IT; i++) canEdit(`u${(i * 2654435761) % 50000}`, `r${(i * 40503) % 120000}`);
console.log(`  gremlin check latency: ${((performance.now() - t0) / IT).toFixed(3)} ms/op`);
console.log(`  ${pass}/${cases.length} correct`);
g.free();
