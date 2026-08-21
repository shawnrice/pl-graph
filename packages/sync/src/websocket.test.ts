// The design's load-bearing claim, proven end-to-end: a WebSocket is
// structurally a port, so the *same* createSyncHost that would sit behind a
// Worker's postMessage serves real sockets unchanged. A Bun.serve server hosts
// one store; two genuine WebSocket clients subscribe and mutate; JSON text
// frames carry protocol v1. Run: bun test packages/sync/src/websocket.test.ts
import { afterAll, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncClient, type ClientSnapshot } from './client.js';
import { createSyncHost, type SyncHost } from './host.js';
import type { HostMessage, RowsMessage } from './protocol.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[websocket.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"marko"}}',
  '{"type":"node","id":"b","labels":["Person"],"properties":{"name":"vadas"}}',
].join('\n');

suite('@lenke/sync over a real WebSocket', () => {
  if (!hasLib) {
    return;
  }

  // One shared store per server — every connection gets its own host over it,
  // exactly the multi-client topology a production server would run.
  const store = createStore(
    graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
  );

  const hosts = new Map<unknown, SyncHost>();
  const server = Bun.serve({
    port: 0,
    fetch(req, srv) {
      return srv.upgrade(req) ? undefined : new Response('websocket only', { status: 400 });
    },
    websocket: {
      open(ws) {
        // The entire transport adapter: JSON text frames in, JSON text frames out.
        hosts.set(ws, createSyncHost(store, { send: (m) => ws.send(JSON.stringify(m)) }));
      },
      message(ws, raw) {
        hosts.get(ws)?.receive(JSON.parse(String(raw)));
      },
      close(ws) {
        hosts.get(ws)?.close();
        hosts.delete(ws);
      },
    },
  });

  afterAll(() => server.stop(true));

  /** A tiny promise-based client: every inbound frame lands in an awaitable queue. */
  const connect = async () => {
    const ws = new WebSocket(`ws://localhost:${server.port}`);
    const queue: HostMessage[] = [];
    let wake: (() => void) | null = null;

    ws.onmessage = (e) => {
      queue.push(JSON.parse(String(e.data)) as HostMessage);
      wake?.();
    };
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error('websocket failed to connect'));
    });

    return {
      ws,
      send: (msg: unknown) => ws.send(JSON.stringify(msg)),
      next: async (): Promise<HostMessage> => {
        while (queue.length === 0) {
          await new Promise<void>((resolve) => {
            wake = resolve;
          });
          wake = null;
        }

        return queue.shift() as HostMessage;
      },
    };
  };

  test('subscribe → rows → remote mutate → push, across two real sockets', async () => {
    const alice = await connect();
    const bob = await connect();

    // Both clients get the status handshake.
    expect((await alice.next()).type).toBe('status');
    expect((await bob.next()).type).toBe('status');

    // Alice opens a standing query and gets rows immediately.
    alice.send({ type: 'subscribe', sub: 'people', query: 'MATCH (p:Person) RETURN p.name' });
    const first = (await alice.next()) as RowsMessage;
    expect(first).toMatchObject({ type: 'rows', sub: 'people', complete: true });
    expect(first.rows).toHaveLength(2);

    // Bob writes over *his* socket; the ack is his, the push is Alice's.
    bob.send({ type: 'mutate', req: 'w1', text: "INSERT (:Person {name: 'carol'})" });
    expect(await bob.next()).toEqual({ type: 'ack', req: 'w1', ok: true });

    const pushed = (await alice.next()) as RowsMessage;
    expect(pushed.type).toBe('rows');
    expect(pushed.rows).toHaveLength(3);

    alice.ws.close();
    bob.ws.close();
  });

  test('one-shot query round-trips the socket', async () => {
    const client = await connect();
    await client.next(); // status

    client.send({ type: 'query', req: 'q1', query: 'MATCH (p:Person) RETURN p.name' });
    const result = await client.next();
    expect(result.type).toBe('result');
    expect((result as { rows?: unknown[] }).rows?.length).toBeGreaterThanOrEqual(2);

    client.ws.close();
  });

  // The full stack: createSyncClient (the registry the UI consumes) over a
  // genuine socket against the same host — subscribe, push-on-remote-write,
  // params binding, promise-shaped mutate. This is the browser story minus
  // the browser.
  test('createSyncClient speaks to the host over a real WebSocket', async () => {
    const openSocket = (): Promise<WebSocket> =>
      new Promise((resolve, reject) => {
        const ws = new WebSocket(`ws://localhost:${server.port}`);
        ws.onopen = () => resolve(ws);
        ws.onerror = () => reject(new Error('websocket failed to connect'));
      });

    const attach = (ws: WebSocket) => {
      const client = createSyncClient({ send: (m) => ws.send(JSON.stringify(m)) });
      ws.onmessage = (e) => client.receive(JSON.parse(String(e.data)));

      return client;
    };

    const alice = attach(await openSocket());
    const bobWs = await openSocket();
    const bob = attach(bobWs);

    // Alice stands up a parameterized live query with ONE persistent
    // subscriber (dropping to zero refs would tear down the wire sub); each
    // push resolves the waiters queued at that moment.
    const live = alice.liveQuery('MATCH (p:Person) WHERE p.name = $n RETURN p.name', {
      deps: null,
      params: { n: 'nils' },
    });
    const waiters: Array<(s: ClientSnapshot) => void> = [];
    const stopLive = live.subscribe(() => {
      for (const w of waiters.splice(0)) {
        w(live.getSnapshot());
      }
    });
    const nextChange = (): Promise<ClientSnapshot> =>
      new Promise((resolve) => {
        waiters.push(resolve);
      });

    const first = await nextChange();
    expect(first.complete).toBe(true);
    expect(first.rows).toEqual([]); // nils doesn't exist yet

    // Bob writes over his socket; Alice's standing query hears it.
    const changed = nextChange();
    await bob.mutate('INSERT (:Person {name: $n})', { n: 'nils' });
    const after = await changed;
    expect(after.rows).toEqual([{ 'p.name': 'nils' }]);

    stopLive();
    bobWs.close();
  });
});
