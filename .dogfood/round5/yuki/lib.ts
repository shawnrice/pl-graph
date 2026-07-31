// Shared harness for the round-5 multiplayer dogfood.
//
// Topology under test: ONE authoritative server (shared Store + WriteLog + optional
// DedupRegistry) with one SyncHost per connection; N clients, each with its own
// optimistic local SyncEngine + SyncClient, wired to the server over an in-process
// "socket" I can CUT (to simulate a dropped connection / lost ack) and re-dial.
import { existsSync } from 'node:fs';

import {
  createEmptyGraph,
  createStore,
  graphFromNdjson,
  type RustGraph,
  type Store,
} from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';
import {
  createSyncClient,
  createSyncHost,
  createSyncEngine,
  createWriteLog,
  createDedupRegistry,
  type SyncClient,
  type SyncEngine,
  type SyncHost,
  type WriteLog,
  type DedupRegistry,
  type SyncWrite,
} from '@lenke/sync';

const LIB_EXT = { darwin: 'dylib', win32: 'dll' }[process.platform as string] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
if (!existsSync(LIB)) throw new Error(`native lib missing: ${LIB}`);

export const backend = createFfiBackend(LIB);

// Schema every store (server + each client's optimistic store) must set up
// identically, since CDC replicates WRITES, not schema/constraints/indexes.
export const installSchema = (g: RustGraph): void => {
  g.createUniqueConstraint('Presence', 'sid');
  g.createUniqueConstraint('Card', 'id');
};

export const emptyGraph = (): RustGraph => {
  const g = createEmptyGraph(backend);
  installSchema(g);
  return g;
};

// A one-directional link that can be cut: after cut(), delivered messages are
// dropped until reconnect() points it at a fresh target.
export type Cut = {
  send: (m: unknown) => void;
  cut: () => void;
  reconnect: (target: (m: unknown) => void) => void;
};
export const cuttableLink = (initial: ((m: unknown) => void) | null): Cut => {
  let target = initial;
  return {
    send: (m) => target?.(m),
    cut: () => {
      target = null;
    },
    reconnect: (t) => {
      target = t;
    },
  };
};

export type Server = {
  store: Store;
  writeLog: WriteLog;
  dedup: DedupRegistry;
  host: (send: (m: unknown) => void) => SyncHost;
};

export const newServer = (opts: { seed?: string } = {}): Server => {
  const g = opts.seed
    ? graphFromNdjson(backend, new TextEncoder().encode(opts.seed))
    : createEmptyGraph(backend);
  installSchema(g);
  const store = createStore(g);
  const writeLog = createWriteLog();
  const dedup = createDedupRegistry();
  return {
    store,
    writeLog,
    dedup,
    host: (send) => createSyncHost(store, { send: (m) => send(m), writeLog, dedup }),
  };
};

export type Client = {
  name: string;
  engine: SyncEngine;
  client: SyncClient;
  local: Store; // the client's optimistic store
  toHost: Cut; // client -> host
  fromHost: Cut; // host -> client
  host: () => SyncHost;
  reconnect: () => void; // rebuild host (new origin) + client.replay()
};

export const connect = (
  server: Server,
  name: string,
  opts: { onResync?: () => void } = {},
): Client => {
  const local = createStore(emptyGraph());
  const engine = createSyncEngine({
    store: local,
    upstream: { push: (w: SyncWrite) => client.pushWrite(w) },
    retry: { attempts: 2, baseMs: 5 },
  });

  // host -> client link (repointed to client once it exists)
  const fromHost = cuttableLink(null);
  let host = server.host((m) => fromHost.send(m));
  // client -> host link (repointed on reconnect)
  const toHost = cuttableLink((m) => host.receive(m));
  const client = createSyncClient({ send: (m) => toHost.send(m) });
  fromHost.reconnect((m) => client.receive(m));

  // CDC: pipe other clients' writes into our optimistic engine.
  client.subscribeWrites((writes) => engine.ingest(writes), {
    onResync: () => opts.onResync?.(),
  });

  const reconnect = (): void => {
    // A fresh transport replaces BOTH directions.
    fromHost.reconnect((m) => client.receive(m));
    host = server.host((m) => fromHost.send(m)); // fresh host, new origin id
    toHost.reconnect((m) => host.receive(m));
    client.replay();
  };

  return {
    name,
    engine,
    client,
    local,
    toHost,
    fromHost,
    host: () => host,
    reconnect,
  };
};

export const cardCount = (s: Store): number =>
  s.mutate((g) => g.query<{ c: number }>('MATCH (n:Card) RETURN count(*) AS c')[0].c);

export const presenceCount = (s: Store): number =>
  s.mutate((g) => g.query<{ c: number }>('MATCH (n:Presence) RETURN count(*) AS c')[0].c);

export const cards = (s: Store): { id: unknown; title: unknown; column: unknown }[] =>
  s.mutate((g) =>
    g.query('MATCH (c:Card) RETURN c.id AS id, c.title AS title, c.column AS column ORDER BY c.id'),
  ) as { id: unknown; title: unknown; column: unknown }[];

export const presences = (s: Store): { sid: unknown; card: unknown }[] =>
  s.mutate((g) =>
    g.query('MATCH (p:Presence) RETURN p.sid AS sid, p.card AS card ORDER BY p.sid'),
  ) as { sid: unknown; card: unknown }[];

export const settle = (ms = 30): Promise<void> => new Promise((r) => setTimeout(r, ms));
