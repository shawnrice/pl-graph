import {
  graphFromNdjson,
} from '@lenke/native';
// Probe GQL features needed for ReBAC check(): variable-length paths,
// edge-label disjunction, EXISTS subquery, *0.. semantics.
import { createNodeBackend } from '@lenke/node/backend';

// Tiny fixture:
//   alice -MEMBER_OF-> eng -MEMBER_OF-> staff
//   docFolder <-PARENT- doc   (doc's parent is docFolder)
//   staff -EDITOR-> docFolder
//   => alice should be able to edit doc (transitive group + inherited grant)
//   bob has nothing.
const lines = [
  { type: 'node', id: 'alice', labels: ['User'], properties: { uid: 'alice' } },
  { type: 'node', id: 'bob', labels: ['User'], properties: { uid: 'bob' } },
  { type: 'node', id: 'eng', labels: ['Group'], properties: { gid: 'eng' } },
  { type: 'node', id: 'staff', labels: ['Group'], properties: { gid: 'staff' } },
  { type: 'node', id: 'doc', labels: ['Resource'], properties: { rid: 'doc' } },
  { type: 'node', id: 'docFolder', labels: ['Resource'], properties: { rid: 'docFolder' } },
  { type: 'edge', id: 'e1', from: 'alice', to: 'eng', labels: ['MEMBER_OF'], properties: {} },
  { type: 'edge', id: 'e2', from: 'eng', to: 'staff', labels: ['MEMBER_OF'], properties: {} },
  { type: 'edge', id: 'e3', from: 'doc', to: 'docFolder', labels: ['PARENT'], properties: {} },
  { type: 'edge', id: 'e4', from: 'staff', to: 'docFolder', labels: ['EDITOR'], properties: {} },
];
const ndjson = new TextEncoder().encode(lines.map((l) => JSON.stringify(l)).join('\n'));

const g = graphFromNdjson(createNodeBackend(), ndjson);
console.log('counts', g.vertexCount, g.edgeCount);

function tryq(label: string, text: string, params?: Record<string, unknown>) {
  try {
    const rows = params ? g.query(text, params) : g.query(text);
    console.log(`OK  ${label}:`, JSON.stringify(rows));
  } catch (e) {
    console.log(`ERR ${label}:`, (e as Error).message);
  }
}

// 1. plain variable-length membership *(0 or more)
tryq('vlen *', 'MATCH (u:User {uid:$u})-[:MEMBER_OF]->*(g) RETURN g.gid AS gid', { u: 'alice' });
// 1b. edge-label disjunction
tryq('edge disjunction', 'MATCH (p)-[:EDITOR|OWNER]->(r) RETURN count(*) AS c');
// 2. variable-length + edge disjunction + inherited grant, single combined pattern
tryq(
  'combined edit path',
  `MATCH (u:User {uid:$u})-[:MEMBER_OF]->*(p)-[:EDITOR|OWNER]->(anc),
         (r:Resource {rid:$r})-[:PARENT]->*(anc)
   RETURN count(*) AS c`,
  { u: 'alice', r: 'doc' },
);
// 3. EXISTS subquery form → boolean
tryq(
  'EXISTS allowed alice',
  `MATCH (u:User {uid:$u}), (r:Resource {rid:$r})
   RETURN EXISTS {
     MATCH (u)-[:MEMBER_OF]->*(p)-[:EDITOR|OWNER]->(anc),
           (r)-[:PARENT]->*(anc)
   } AS allowed`,
  { u: 'alice', r: 'doc' },
);
// 4. negative case: bob
tryq(
  'EXISTS bob',
  `MATCH (u:User {uid:$u}), (r:Resource {rid:$r})
   RETURN EXISTS {
     MATCH (u)-[:MEMBER_OF]->*(p)-[:EDITOR|OWNER]->(anc),
           (r)-[:PARENT]->*(anc)
   } AS allowed`,
  { u: 'bob', r: 'doc' },
);
// 5. {0,} form
tryq('vlen {0,}', 'MATCH (u:User {uid:$u})-[:MEMBER_OF]->{0,}(g) RETURN count(*) AS c', {
  u: 'alice',
});

g.free();
