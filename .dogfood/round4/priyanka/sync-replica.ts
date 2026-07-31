import {
  createStore,
  graphFromNdjson,
} from '@lenke/native';
// BONUS: propagate a tuple write to a second in-process replica via @lenke/sync.
//
// Two independent authz graphs (primary + replica), each fronted by a SyncHost.
// A single sync client drives a live "who can view r-shared" query against the
// primary, and issues ONE grant-tuple write over the v1 protocol. A tiny
// replication bus fans the same mutate message to the replica's host — modeling
// a write replicated over the sync protocol to a second in-process store. We
// prove both graphs re-authorize (and the live query pushes the new grantee).
import { createNodeBackend } from '@lenke/node/backend';
import { createSyncClient, createSyncHost } from '@lenke/sync';

// Small, clear fixture: u-new is a plain user with no grant on r-shared yet.
const fixture = [
  { type: 'node', id: 'u-new', labels: ['User'], properties: { uid: 'u-new' } },
  { type: 'node', id: 'u-old', labels: ['User'], properties: { uid: 'u-old' } },
  { type: 'node', id: 'r-shared', labels: ['Resource'], properties: { rid: 'r-shared' } },
  { type: 'edge', id: 'g0', from: 'u-old', to: 'r-shared', labels: ['VIEWER'], properties: {} },
]
  .map((l) => JSON.stringify(l))
  .join('\n');
const bytes = new TextEncoder().encode(fixture);

const backend = createNodeBackend();
const primary = createStore(graphFromNdjson(backend, bytes));
const replica = createStore(graphFromNdjson(backend, bytes));
primary.graph.createVertexIndex('rid');
replica.graph.createVertexIndex('rid');

// who-can-VIEW r-shared, resolving transitive membership + inheritance.
const VIEWERS = `
  MATCH (r:Resource {rid: $r})-[:PARENT]->*(anc)<-[:EDITOR|OWNER|VIEWER]-(p)<-[:MEMBER_OF]-*(m:User)
  RETURN DISTINCT m.uid AS uid ORDER BY uid`;
const viewers = (store: typeof primary) =>
  store.graph.query<{ uid: string }>(VIEWERS, { r: 'r-shared' }).map((x) => x.uid);

console.log('BEFORE write:');
console.log('  primary viewers:', viewers(primary));
console.log('  replica viewers:', viewers(replica));

// --- wire the v1 protocol in-process (send/receive are the whole transport) ---
const hostPrimary = createSyncHost(primary, { send: (m) => client.receive(m) });
const hostReplica = createSyncHost(replica, { send: () => {} }); // replica: apply-only
const client = createSyncClient({
  // replication bus: a client message fans out to BOTH hosts. The primary answers
  // subscriptions back to the client; the replica just applies the write.
  send: (m) => {
    hostPrimary.receive(m);
    if (m.type === 'mutate') hostReplica.receive(m);
  },
});

// standing live query over the primary; deps=null → recompute on any change.
const live = client.liveQuery(VIEWERS, { deps: null, params: { r: 'r-shared' } });
const pushes: string[][] = [];
live.subscribe(() => pushes.push((live.getSnapshot().rows as { uid: string }[]).map((r) => r.uid)));
console.log(
  '\nlive query initial snapshot complete=',
  live.getSnapshot().complete,
  'rows=',
  (live.getSnapshot().rows as { uid: string }[]).map((r) => r.uid),
);

// --- the tuple write: grant u-new VIEWER on r-shared, over the protocol ---
console.log('\nwriting tuple: (u-new)-[:VIEWER]->(r-shared) via client.mutate ...');
await client.mutate('MATCH (u:User {uid: $u}), (r:Resource {rid: $r}) INSERT (u)-[:VIEWER]->(r)', {
  u: 'u-new',
  r: 'r-shared',
});

console.log('\nAFTER write:');
console.log('  primary viewers:', viewers(primary));
console.log('  replica viewers:', viewers(replica));
console.log('  live-query pushes observed:', JSON.stringify(pushes));

const ok =
  viewers(primary).includes('u-new') &&
  viewers(replica).includes('u-new') &&
  pushes.some((p) => p.includes('u-new'));
console.log(
  `\n${ok ? 'PASS' : 'FAIL'}: tuple write propagated to primary + replica and pushed to the live query`,
);

live.unsubscribe?.();
hostPrimary.close();
hostReplica.close();
primary[Symbol.dispose]();
replica[Symbol.dispose]();
