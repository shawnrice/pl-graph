# @lenke/node

> Fast native Node.js addon (N-API via [napi-rs](https://napi.rs)) for the `lenke-engine` graph engine.

The idiomatic **Node** path to the engine. Where `@lenke/native/ffi` (Bun) and `@lenke/native/wasm` (browser) cross a hand-marshalled **C ABI**, this addon speaks **N-API directly**: JS strings and `Buffer`s arrive as real Rust values, results come back as `Buffer`s — no per-call pointer marshalling, and no dynamic library to locate at runtime (the engine is compiled straight in).

Node is the intended **production** consumer; the Bun FFI backend is a dev-speed convenience.

## Two ways to use it

### 1. The `Graph` class (direct, fastest)

```js
import { Graph } from '@lenke/node';

const g = Graph.fromNdjson(await readFile('graph.ndjson'));
const doc = JSON.parse(new TextDecoder().decode(g.query('MATCH (p:Person) RETURN p.name')));
const arrow = g.queryArrow('MATCH (p:Person) RETURN p.age'); // ARW1 columnar Buffer
```

`query` / `queryArrow` / `gremlin` / `serialize` / `encodeNdjson` return `Buffer`; `version()` / `epoch(name)` / `vertexCount` / `edgeCount` are plain numbers. `mergeNdjson(bytes)` bulk-appends and returns a `Buffer` of **JSON** — the merge report — which you `JSON.parse` yourself. (This raw `Graph` class hands back `Buffer`s throughout; if you want decoded results — parsed rows, a typed `MergeReport` — use the `Backend` facade below, which does the decoding for you.)

### 2. The `Backend` adapter (drop-in for the `@lenke/native` facade)

`createNodeBackend()` satisfies the shared `Backend` contract, so the whole `@lenke/native` facade — `graphFromNdjson`, the `RustGraph` tagged-template `query`/`gremlin`, `createStore` + `liveQuery` — runs on Node against this engine unchanged:

```js
import { createNodeBackend } from '@lenke/node/backend';
import { graphFromNdjson, createStore } from '@lenke/native';

const g = graphFromNdjson(createNodeBackend(), await readFile('graph.ndjson'));
const people = createStore(g).liveQuery('MATCH (p:Person) RETURN p.name', {
  deps: ['Person', 'name'],
});
// people.getSnapshot() → referentially stable until a mutation bumps the epoch
```

## Build

```
bunx nx build @lenke/node    # napi build --platform --release --esm
                             # → index.js (ESM), index.d.ts, lenke-node.<triple>.node
```

The generated `index.js` / `index.d.ts` / `*.node` are git-ignored artifacts. `napi build` cross-compiles for the platforms listed under `napi.targets` in `package.json`; CI produces the per-platform prebuilt binaries.

## ESM

The package is `"type": "module"`, like the rest of the monorepo. napi-rs v3's `--esm` emits a native-ESM binding (`import`/`export`, with `createRequire` used internally only to load the `.node` — which is the one thing ESM can't import directly). So consumers just `import { Graph } from '@lenke/node'` with no CJS interop.
