# The Rust engine in WebAssembly

**Engine:** Rust `lenke-engine` · **Reach-path:** WebAssembly (`@lenke/native/wasm-engine`) · **Runtime:** browser, and anything with a `WebAssembly` global (Node, Deno, Bun).

Use this to run the columnar Rust engine in a browser with no native addon — or in any host where you'd rather ship a `.wasm` than a platform binary. Same `RustGraph` API as the [native](./native.md) reach-paths; the differences are an **async load**, no threads, and that you build and ship the `.wasm` yourself.

## Loading

`createWasmEngineBackend` is async and accepts `.wasm` bytes, a `Response`, or a compiled `Module`:

```ts
import { createWasmEngineBackend } from '@lenke/native/wasm-engine';
import { graphFromNdjson } from '@lenke/native';

// Browser — stream-compile from a fetch:
const backend = await createWasmEngineBackend(fetch(new URL('./lenke_engine.wasm', import.meta.url)));

// Node — from bytes:
// import { readFile } from 'node:fs/promises';
// const backend = await createWasmEngineBackend(await readFile('lenke_engine.wasm'));

using g = graphFromNdjson(backend, ndjsonBytes);
const rows = g.query`MATCH (p:Person) RETURN p.name AS name`;
```

There's no `wasm-bindgen`/`wasm-pack` glue and no import object, so no special bundler wasm-loader config is required — a bundler that turns the `.wasm` into a URL or bytes is enough. The `./wasm` subpath export is separate from `./ffi` precisely so a browser bundle never pulls in the Bun-only `bun:ffi` builtin.

## Memory

The wasm graph is heap-owned by the module (its handle is a linear-memory offset) — release it with `using` or `g.free()`, exactly as for ffi. The `FinalizationRegistry` backstop applies here too, as a leak-net only. See the [memory model](./choosing-your-build.md#memory-model).

One wasm-specific caveat if you drop below the facade: the module's `memory.buffer` is replaced when the heap grows, so never cache a typed-array view across a call that can allocate. The `RustGraph`/`Backend` facade handles this for you — you only need to know it if you call the raw exports.

## Building the `.wasm`

```bash
# in packages/native
bun run build:wasm      # the full engine: GQL, Gremlin, NDJSON, binary, textual codecs, Arrow
```

This is `cargo build --release --target wasm32-unknown-unknown --features capi` (the `capi` feature exposes the C ABI the backend calls). wasm has no threads, so anything the native build parallelizes runs serially here.

### Trim the textual codecs

The one size lever today is the `codecs` feature (on by default), which pulls in the pg-json / pg-text / graphson / csv serializers. Build with `--no-default-features --features capi` to drop them — a smaller module that still runs GQL, Gremlin, NDJSON, binary snapshots, and Arrow, just without the extra textual formats. Both query languages are always compiled in; there is no separate GQL-only or Gremlin-only build.

## Packaging (roadmap)

Today, **you build the `.wasm` and hand its bytes/`Response` to `createWasmEngineBackend` yourself** — `@lenke/native` does not yet bundle or publish a prebuilt artifact, and there's no packaging step that copies it into a `dist/`. A packaged distribution (so you can `import` the wasm without a manual build) is planned but **not yet built**. Until then, wire the build output into your app's bundler (the [`examples/service-map`](../../examples/service-map) worker imports it with a Vite `?url` import).

## In a worker

The common browser deployment runs this wasm engine inside a web worker, driven by [`@lenke/sync`](./frontend-worker.md) so the graph lives off the main thread. See that guide for the wiring.
