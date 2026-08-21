// createReconnectingClient over a controllable in-process transport: a real
// createSyncHost sits on one shared store (the "server"); the transport can be
// cut and will re-dial. Proves the three guarantees — standing queries
// re-subscribe after a reconnect, a write issued while offline lands once the
// wire returns, and connectivity flips are observable — without a real socket.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncClient } from './client.js';
import { createSyncHost } from './host.js';
import type { HostMessage } from './protocol.js';
import { createReconnectingClient } from './reconnect.js';
import { createWriteLog, type WriteLog } from './writelog.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[reconnect.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const NDJSON = ['a', 'b']
  .map((id) => `{"type":"node","id":"${id}","labels":["Person"],"properties":{"name":"${id}"}}`)
  .join('\n');

/** Spin until `pred()` holds, pumping micro/macro tasks; fails loudly on timeout. */
const until = async (pred: () => boolean, label: string): Promise<void> => {
  for (let i = 0; i < 500; i++) {
    if (pred()) {
      return;
    }

    await new Promise((r) => setTimeout(r, 1));
  }

  throw new Error(`until: condition never held — ${label}`);
};

const suite = hasLib ? describe : describe.skip;

suite('createReconnectingClient', () => {
  if (!hasLib) {
    return;
  }

  // A controllable transport: each connection gets a fresh host over the shared
  // store (exactly the per-socket-host topology a server runs). `cut()` drops
  // the live connection, which triggers the manager's re-dial.
  const makeTransport = (
    store: ReturnType<typeof createStore>,
    opts?: { syncOpen?: boolean; writeLog?: WriteLog },
  ) => {
    let liveClosed: (() => void) | null = null;
    let liveHost: ReturnType<typeof createSyncHost> | null = null;
    let allow = true; // when false, every dial fails to open (a held outage)
    let connects = 0;

    const connect: Parameters<typeof createReconnectingClient>[0]['connect'] = ({
      opened,
      received,
      closed,
    }) => {
      if (!allow) {
        queueMicrotask(closed); // dial fails; the manager backs off and retries

        return { send: () => {}, close: () => {} };
      }

      connects++;
      const host = createSyncHost(store, {
        send: (m: HostMessage) => queueMicrotask(() => received(m)),
        writeLog: opts?.writeLog,
      });
      liveHost = host;
      liveClosed = closed;

      // A real socket opens asynchronously; a MessagePort/test double may fire
      // opened() synchronously during connect() — exercise both.
      if (opts?.syncOpen) {
        opened();
      } else {
        queueMicrotask(opened);
      }

      return {
        send: (m) => {
          if (liveHost === host) {
            queueMicrotask(() => host.receive(m));
          }
        },
        close: () => {
          if (liveHost === host) {
            liveHost = null;
            liveClosed = null;
            host.close();
          }
        },
      };
    };

    const cut = (): void => {
      const closed = liveClosed;
      liveHost?.close();
      liveHost = null;
      liveClosed = null;
      closed?.(); // the manager schedules a re-dial
    };

    // Hold the wire down (drop the live socket, refuse new dials) / let it back.
    const goOffline = (): void => {
      allow = false;
      cut();
    };
    const goOnline = (): void => {
      allow = true;
    };

    return { connect, cut, goOffline, goOnline, connects: () => connects };
  };

  test('re-subscribes a standing query after a reconnect', async () => {
    const store = createStore(
      graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
    );
    const t = makeTransport(store);
    const client = createReconnectingClient({ connect: t.connect, retry: { baseMs: 1, maxMs: 5 } });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    live.subscribe(() => {}); // one persistent subscriber keeps the wire sub alive
    await until(() => live.getSnapshot().rows.length === 2, 'initial rows');

    // Drop the socket, then write to the shared store while offline.
    t.cut();
    await until(() => !client.connected(), 'goes offline on cut');
    store.mutate((g) => g.query("INSERT (:Person {name: 'carol'})"));

    // The manager re-dials; the re-subscribed query re-answers current rows.
    await until(() => client.connected(), 'reconnects');
    await until(() => live.getSnapshot().rows.length === 3, 'rows reflect the offline write');
    expect(t.connects()).toBeGreaterThanOrEqual(2);

    client.close();
  });

  test('a mutate issued while offline lands after reconnect', async () => {
    const store = createStore(
      graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
    );
    const t = makeTransport(store);
    const client = createReconnectingClient({ connect: t.connect, retry: { baseMs: 1, maxMs: 5 } });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    live.subscribe(() => {});
    await until(() => live.getSnapshot().rows.length === 2, 'initial rows');

    // Hold the wire down so the write genuinely parks (not a timing race).
    t.goOffline();
    await until(() => !client.connected(), 'offline');
    let acked = false;
    const done = client.mutate("INSERT (:Person {name: 'dave'})").then(() => {
      acked = true;
    });
    // While the outage is held open, the parked write must not settle.
    await new Promise((r) => setTimeout(r, 10));
    expect(acked).toBe(false);

    t.goOnline(); // the next re-dial succeeds; replay re-sends the parked write
    await done; // resolves only once the wire returns and the ack arrives
    expect(acked).toBe(true);
    await until(() => live.getSnapshot().rows.length === 3, 'the parked write is visible');

    client.close();
  });

  test('the CDC write stream + clientId survive a reconnect (multiplayer + reconnect)', async () => {
    const store = createStore(
      graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
    );
    // The CDC stream needs a shared op log across every per-connection host.
    const writeLog = createWriteLog();
    const t = makeTransport(store, { writeLog });
    const client = createReconnectingClient({
      connect: t.connect,
      retry: { baseMs: 1, maxMs: 5 },
      clientId: 'me',
    });

    // clientId is now exposed (it used to be Pick<>'d out of the reconnect surface).
    expect(client.clientId).toBe('me');

    // A second, independent client on the same server — the "other player".
    const box: { c?: ReturnType<typeof createSyncClient> } = {};
    const otherHost = createSyncHost(store, {
      send: (m) => queueMicrotask(() => box.c?.receive(m)),
      writeLog,
    });
    box.c = createSyncClient({ send: (m) => queueMicrotask(() => otherHost.receive(m)) });
    const other = box.c;

    await until(() => client.connected(), 'first open');

    // Subscribe to the CDC stream on the reconnecting client (the widened surface).
    const seen: string[] = [];
    client.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));
    // onDisconnect is exposed too — registering ephemeral teardown must not throw.
    client.onDisconnect([{ text: "MATCH (p:Presence {sid: 'me'}) DETACH DELETE p" }]);

    // The other player's write arrives on the reconnecting client's CDC stream.
    await other.mutate("INSERT (:Person {name: 'carol'})");
    await until(() => seen.length === 1, 'first cross-client write');

    // Drop + auto-reconnect; replay() must re-subscribe the write stream from the cursor.
    t.cut();
    await until(() => !client.connected(), 'offline');
    await until(() => client.connected(), 'reconnect');

    // A write AFTER the reconnect still reaches the stream — proving replay
    // re-subscribed it (this is exactly "multiplayer + reconnect together").
    await other.mutate("INSERT (:Person {name: 'dave'})");
    await until(() => seen.length === 2, 'cross-client write after reconnect');
    expect(seen).toEqual(["INSERT (:Person {name: 'carol'})", "INSERT (:Person {name: 'dave'})"]);

    client.close();
  });

  test('reports connectivity flips and stops re-dialing on close', async () => {
    const store = createStore(
      graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
    );
    const t = makeTransport(store);
    const flips: boolean[] = [];
    const client = createReconnectingClient({ connect: t.connect, retry: { baseMs: 1, maxMs: 5 } });
    client.onConnectivity((u) => flips.push(u));

    await until(() => client.connected(), 'first open');
    t.cut();
    await until(() => !client.connected(), 'drop observed');
    await until(() => client.connected(), 'auto-reconnect');

    expect(flips).toEqual([true, false, true]);

    client.close();
    expect(client.connected()).toBe(false);

    // After close, a cut must not resurrect the connection.
    const before = t.connects();
    t.cut();
    await new Promise((r) => setTimeout(r, 10));
    expect(t.connects()).toBe(before);
  });

  test('a synchronously-opening transport still re-subscribes on reconnect', async () => {
    // opened() fires DURING connect() (a MessagePort / test double). The manager
    // must still replay over the live connection, not a null one.
    const store = createStore(
      graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
    );
    const t = makeTransport(store, { syncOpen: true });
    const client = createReconnectingClient({ connect: t.connect, retry: { baseMs: 1, maxMs: 5 } });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    live.subscribe(() => {});
    await until(
      () => live.getSnapshot().rows.length === 2,
      'initial rows over a sync-open transport',
    );

    // Reconnect: the re-subscribe must reach the fresh host (a null-conn replay
    // would drop it and the new row would never appear).
    t.cut();
    store.mutate((g) => g.query("INSERT (:Person {name: 'carol'})"));
    await until(
      () => live.getSnapshot().rows.length === 3,
      're-subscribed after a sync-open reconnect',
    );

    client.close();
  });
});
