# service map — a local-first vertical slice

A small but complete lenke application: a live **service-dependency map**. ~240
services across 4 clusters, `CALLS` edges between them, and one editable
`status` column. It is deliberately small — a bare HTML `<table>`, no UI library
— because the app itself isn't the point. The point is the **plumbing**: this one
feature is threaded through every layer lenke offers, so you can read one repo
and see how the pieces compose into a real local-first app.

Concretely, the same status edit travels:

```
React tab  ──►  sync client  ──►  SharedWorker (wasm store + sync engine)  ──►  WebSocket  ──►  Node server (native store)
    ▲                                      │
    └──────────── live push ◄──────────────┘   (and to every other open tab)
```

## What it demonstrates

- **One in-memory graph store, queried with GQL and Gremlin** — no database.
- **Live queries**: the UI _subscribes_ to a standing query and is pushed a new
  result only when that result actually changes, instead of re-fetching.
- **The same sync host over two transports** — a `MessagePort` between tab and
  worker, and a `WebSocket` between worker and server. A socket is structurally
  a port, so both ends run the identical host.
- **Local-first**: optimistic edits apply instantly, survive a page reload via an
  OPFS snapshot, and replicate to the server when it's reachable — offline is
  just "the socket isn't up yet," not a separate code path.
- **Both Rust reach-paths**: the browser runs the engine as **WebAssembly**; the
  server runs it as the **native N-API addon** — same core, same queries.

## Run it

```sh
bun install

# once: build the two engine artifacts the demo loads
bun run --cwd ../../packages/native build:wasm     # → lenke_engine.wasm  (browser)
bunx nx build @lenke/node                          # → native addon     (server)

bun run dev       # runs BOTH the ws server and vite; open the printed URL
```

`dev` starts the WebSocket server _and_ the Vite dev server together — the app
is inert without the server, because each cluster's rows are demand-filled by
querying it. (Need them apart? `bun run server` and `bun run frontend` run each
half on its own.)

Open it in Chrome first — it leans on `SharedWorker` and OPFS.

Two ways to check it without clicking around:

- `bun run smoke` — spawns the real Node server and drives it with a protocol
  client over a real socket (no browser). Fast; covers the server + protocol.
- `bun run e2e` — `bunx playwright install chromium` once, then this boots the
  server + vite and drives the slice in a real Chromium. It's the only harness
  that exercises the browser-only paths: the SharedWorker, the wasm engine,
  OPFS, and the cross-tab `MessagePort` push.

## Things to try

1. **Live everywhere + blast radius.** Open two tabs, pick a cluster, flip a
   service to `down` in one tab — the other updates instantly, and every service
   that transitively depends on it lights up amber in **both the table and the
   force-directed graph**, tracing the failure along the call edges. That
   cascade is one live variable-length GQL query (`-[:CALLS]->{1,}`); the
   per-row **blast radius** button lists the affected callers explicitly. Sort
   the table by `← callers` to find the services whose failure hurts most, and
   drag nodes in the graph to untangle it.
2. **Demand-fill.** Switch to a cluster you haven't opened. It renders
   `loading…` first (the subscription's result is marked incomplete), the worker
   fetches just that cluster from the server, the rows arrive, and it flips to
   complete. Data is loaded lazily, per cluster, on first view.
3. **Pull the cable.** Kill the server (Ctrl-C in terminal 1). Edits still apply
   instantly and the status bar counts the unsynced changes. Reload the tab
   _while still offline_ — the OPFS snapshot warm-boots the store with your
   queued edits intact. Restart the server and the queue drains to zero.

## How it's built

Four files, one per layer. Read them in this order.

### `src/main.tsx` — the tab

A React view over a **sync client**. It creates one `SharedWorker` for the origin
and a `createSyncClient` that talks to it over `worker.port`. Views subscribe to
live queries through `useSyncExternalStore`, so a component re-renders exactly
when its query's snapshot changes — nothing polls.

A subscription declares its **dependency tokens** (`['Service', 'status', …]`) —
the labels and property keys whose mutation must re-run it. A `status` flip
touches `status`, so the service grid re-runs; it doesn't touch anything the
blast-radius query reads, so that one stays put.

Because a `MessagePort` can't reliably tell the worker its tab has closed, the
tab sends an explicit `bye` on `pagehide` (and `replay()`s its subscriptions on
bfcache restore). Without it the worker would keep re-running a dead tab's
queries forever — the kind of lifecycle detail a real port-based app has to get
right, so it's here rather than hidden.

### `worker.ts` — the SharedWorker (the heart of it)

One lenke store per origin, shared by every tab. It owns:

- **The graph** — a `createStore` over the **wasm** backend.
- **A snapshot store** (`createSnapshotStore`, OPFS-backed) — warm-boots the
  store on load and persists the store + pending write queue on a debounce, so a
  reload (even offline) comes back where you left off.
- **The sync engine** (`createSyncEngine`) — holds the optimistic write queue and
  the **demand-fill collections**. One `services` collection is `key`ed on
  `cluster`, so the engine tracks completeness and lazy-loading _per cluster
  value_, reading the value straight off each subscription's `$cluster` param.
- **The server link** (`createReconnectingClient`) — a reconnecting WebSocket
  that re-dials with back-off and re-subscribes on reconnect. Loaders are
  one-shot `query`s against it; write-back is `mutate`. While it's down, writes
  simply stay queued.
- **One host per tab** — the SharedWorker's `connect` event hands over a
  `MessagePort` per tab, and each gets its own `engine.createHost`. This is the
  identical one-host-per-connection shape the server uses for sockets.

### `server.ts` — the authoritative server

The whole fleet in an embedded lenke store — here on the **native N-API addon**,
the fast production Node path — with one `createSyncHost` per WebSocket. This is
the payoff of the transport symmetry: the _same_ `createSyncHost` that sits
behind the worker's `postMessage` serves TCP sockets here, unchanged. ~45 lines.

### `datagen.ts` — the fixture

Generates the synthetic fleet (services, clusters, `CALLS` edges) as NDJSON, used
to seed both the server store and the tests. No lenke concepts — just data.

## Why it's built this way

- **Store in a SharedWorker, not per tab.** One store for the origin means every
  tab shares state and cross-tab live updates come for free — a write in any tab
  bumps the graph version and the engine pushes to every subscriber whose result
  moved. The write path never knows subscriptions exist.
- **Subscribe, don't fetch.** The client asks _"keep me current on this query"_;
  the host recomputes a subscription only when its dependency tokens move and
  pushes only when the snapshot reference actually changes. The grid subscribes
  with a row `key`, so a single status flip ships one changed cell (`patch`), not
  the whole cluster.
- **Offline is the default, not a mode.** Optimistic writes land in a durable,
  version-gated queue; the OPFS snapshot carries that queue across reloads; the
  reconnecting client replays it when the socket returns. There is no "offline
  branch" — the socket just isn't settling yet.
- **Injection-safe by construction.** GQL values are sent as `params` and bind to
  already-parsed slots — they never touch the query parser. Gremlin has no engine
  param binding, so its values are escaped through the `` gremlin`…` `` tag /
  `escapeGremlin` into safe literals. User input never becomes query structure.

The [`../../docs/guides`](../../docs/guides) explain each layer on its own; this
example is where they meet.
