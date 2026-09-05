import { describe, expect, test } from 'bun:test';

import { connectSharedWorker, servePort, serveSharedWorker, type PortLike } from './port.js';

// A fake host tracking the lifecycle calls servePort drives.
const makeFakeHost = () => {
  const calls = { received: [] as unknown[], closed: 0, statuses: 0 };

  return {
    host: {
      receive: (m: unknown) => calls.received.push(m),
      refresh: () => {},
      sendStatus: () => {
        calls.statuses += 1;
      },
      close: () => {
        calls.closed += 1;
      },
    },
    calls,
  };
};

// A fake engine that hands out fresh hosts and records how many it made.
const makeFakeEngine = () => {
  const hosts: ReturnType<typeof makeFakeHost>[] = [];

  return {
    hosts,
    engine: {
      createHost: () => {
        const h = makeFakeHost();
        hosts.push(h);

        return h.host as never;
      },
    },
  };
};

// A fake MessagePort: `deliver` simulates an inbound message; `sent` records
// outbound; `fireClose` simulates the (Chromium-only) close event.
const makeFakePort = () => {
  const sent: unknown[] = [];
  let onmessage: ((e: MessageEvent) => void) | null = null;
  let onClose: (() => void) | null = null;

  const port: PortLike = {
    postMessage: (m) => sent.push(m),
    start: () => {},
    set onmessage(fn: ((e: MessageEvent) => void) | null) {
      onmessage = fn;
    },
    get onmessage() {
      return onmessage;
    },
    addEventListener: (type, listener) => {
      if (type === 'close') {
        onClose = listener;
      }
    },
  };

  return {
    port,
    sent,
    // The helpers only read `.data`, so a minimal event stands in for a real one.
    deliver: (data: unknown) => onmessage?.({ data } as MessageEvent),
    fireClose: () => onClose?.(),
  };
};

describe('servePort', () => {
  test('opens a host, routes messages, and tears down on bye then revives', () => {
    const { engine, hosts } = makeFakeEngine();
    const fake = makeFakePort();

    servePort(engine, fake.port);
    expect(hosts).toHaveLength(1);

    fake.deliver({ type: 'subscribe', sub: 's1' });
    expect(hosts[0].calls.received).toHaveLength(1);

    // bye tears the host down (revivable), NOT terminal.
    fake.deliver({ type: 'bye' });
    expect(hosts[0].calls.closed).toBe(1);

    // a post-bye message revives a FRESH host (bfcache).
    fake.deliver({ type: 'subscribe', sub: 's2' });
    expect(hosts).toHaveLength(2);
    expect(hosts[1].calls.received).toHaveLength(1);
  });

  test('close event is terminal: shuts the host and fires onClose', () => {
    const { engine, hosts } = makeFakeEngine();
    const fake = makeFakePort();
    let closed = false;

    servePort(engine, fake.port, { onClose: () => (closed = true) });
    fake.fireClose();

    expect(hosts[0].calls.closed).toBe(1);
    expect(closed).toBe(true);
  });

  test('sendStatus targets the live host (no-op after bye)', () => {
    const { engine, hosts } = makeFakeEngine();
    const fake = makeFakePort();

    const served = servePort(engine, fake.port);
    served.sendStatus();
    expect(hosts[0].calls.statuses).toBe(1);

    fake.deliver({ type: 'bye' });
    served.sendStatus(); // host is gone → no throw, no extra status
    expect(hosts[0].calls.statuses).toBe(1);
  });
});

describe('serveSharedWorker', () => {
  test('sets onconnect, serves each connection once the engine resolves, and broadcasts status', async () => {
    const { engine, hosts } = makeFakeEngine();
    const svc = serveSharedWorker(Promise.resolve(engine as never));

    const global = globalThis as unknown as {
      onconnect: (e: { ports: readonly PortLike[] }) => void;
    };
    expect(typeof global.onconnect).toBe('function');

    const a = makeFakePort();
    const b = makeFakePort();
    global.onconnect({ ports: [a.port] });
    global.onconnect({ ports: [b.port] });

    // engine is a promise → serving is deferred a microtask.
    await Promise.resolve();
    await Promise.resolve();

    expect(svc.connectionCount()).toBe(2);
    expect(hosts).toHaveLength(2);

    svc.broadcastStatus();
    expect(hosts[0].calls.statuses).toBe(1);
    expect(hosts[1].calls.statuses).toBe(1);
  });
});

describe('connectSharedWorker', () => {
  test('wires a client over a port (accepts a SharedWorker-like or a bare port)', () => {
    const fake = makeFakePort();
    const client = connectSharedWorker({ port: fake.port });

    // The client sends over the port; a mutate posts a wire message.
    void client.mutate('INSERT (:Person {name: $n})', { n: 'ada' });
    expect(fake.sent.length).toBeGreaterThan(0);
    expect((fake.sent.at(-1) as { type?: string }).type).toBe('mutate');
  });
});
