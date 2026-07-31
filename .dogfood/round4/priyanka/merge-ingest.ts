import {
  graphFromNdjson,
} from '@lenke/native';
// Bulk tuple ingest into a LIVE graph via mergeNdjson (COPY FROM), with indexes
// staying current — the write path a Zanzibar service uses to apply a batch of
// new relationship tuples without a full reload.
import { createNodeBackend } from '@lenke/node/backend';

const dir = import.meta.dir;
const g = graphFromNdjson(createNodeBackend(), await Bun.file(`${dir}/graph.ndjson`).bytes());
g.createVertexIndex('rid');
g.createVertexIndex('uid');

const editCheck = g.prepare<{ allowed: boolean }>(`
  MATCH (r:Resource {rid: $r})
  RETURN EXISTS {
    MATCH (r)-[:PARENT]->*(anc)<-[:EDITOR|OWNER]-(p)<-[:MEMBER_OF]-*(m:User {uid: $u})
  } AS allowed`);
const canEdit = (u: string, r: string) => editCheck.query({ u, r })[0]?.allowed === true;

console.log('before ingest: canEdit(u-none, r-deepdoc) =', canEdit('u-none', 'r-deepdoc'));

// A batch of new tuples: a brand-new group with u-none in it, granted EDITOR on
// r-folderA (an ancestor of r-deepdoc). Plus a couple of new vertices.
const batch = [
  { type: 'node', id: 'g-fresh', labels: ['Group'], properties: { gid: 'g-fresh' } },
  {
    type: 'edge',
    id: 'm-fresh',
    from: 'u-none',
    to: 'g-fresh',
    labels: ['MEMBER_OF'],
    properties: {},
  },
  {
    type: 'edge',
    id: 'grant-fresh',
    from: 'g-fresh',
    to: 'r-folderA',
    labels: ['EDITOR'],
    properties: {},
  },
]
  .map((l) => JSON.stringify(l))
  .join('\n');

const t0 = performance.now();
const report = g.mergeNdjson(new TextEncoder().encode(batch));
const t1 = performance.now();
console.log(`mergeNdjson report:`, report, `in ${(t1 - t0).toFixed(2)} ms`);
console.log(`now vertices=${g.vertexCount} edges=${g.edgeCount}`);

// index still current + inheritance re-resolves through the freshly-merged tuples
console.log(
  'after ingest:  canEdit(u-none, r-deepdoc) =',
  canEdit('u-none', 'r-deepdoc'),
  '(expect true)',
);

editCheck.free();
g.free();
