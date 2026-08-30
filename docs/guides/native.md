# The Rust engine on a server or CLI

**Engine:** Rust `lenke-engine` · **Reach-paths:** N-API (`@lenke/node`) and bun:ffi (`@lenke/native/ffi-engine`) · **Runtime:** Node or Bun.

Use this when you want the columnar Rust engine in a server process or a command-line tool. Two reach-paths, by runtime: **N-API** on Node (the fast production path) and **bun:ffi** on Bun. Both present the identical `RustGraph` facade, so the only difference is how the engine loads. For the browser, see [wasm](./wasm.md); for embedding it as a cache/view machine, see [backend-embedded](./backend-embedded.md).

## N-API — `@lenke/node` (the Node path)

The engine is compiled _into_ a per-platform `.node` addon — no dynamic library to locate. Two ways to use it.

**Through the shared facade** (recommended — identical API to ffi/wasm):

```ts
import { createNodeBackend } from '@lenke/node/backend';
import { graphFromNdjson } from '@lenke/native';

const backend = createNodeBackend();
const g = graphFromNdjson(backend, ndjsonBytes);

const rows = g.query`MATCH (p:Person) RETURN p.name AS name`;
```

**The raw addon** (lower-level; returns `Buffer`s and takes a params JSON string):

```ts
import { Graph } from '@lenke/node';

const g = Graph.fromNdjson(ndjsonBuffer);
const buf = g.query('MATCH (p:Person) RETURN p.name', undefined);
```

**Memory:** the raw `@lenke/node` `Graph` is GC-managed — there's no `free()` and none is needed. The facade's `RustGraph` still gives you a uniform `free()`/`using` (it drops the facade's handle registry) so you can write one lifecycle across all reach-paths. See the [memory model](./choosing-your-build.md#memory-model).

**Build:** `napi build --platform --release --esm` (the package's `build` script) emits `index.js` (ESM), `index.d.ts`, and the `.node` for each configured target triple.

## bun:ffi — `@lenke/native/ffi-engine` (the Bun path)

Loads the Rust `cdylib` directly via `bun:ffi`. You supply the path to the built library — the package does **not** auto-locate it.

```ts
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';
import { graphFromNdjson } from '@lenke/native';

const libPath = new URL('./liblenke_engine.so', import.meta.url).pathname; // your build output
const backend = createFfiEngineBackend(libPath);

using g = graphFromNdjson(backend, ndjsonBytes); // freed at scope exit
const rows = g.query`MATCH (p:Person) RETURN p.name AS name`;
```

**Memory:** the ffi graph is heap-owned — release it with `using` or `g.free()`. A `FinalizationRegistry` backstop catches a forgotten free, but don't rely on it. See the [memory model](./choosing-your-build.md#memory-model).

**Build:** `bun run build:rust` — `cargo build --release --features capi` (the `capi` feature exposes the C ABI) producing `liblenke_engine.{so,dylib,dll}` under the crate's `target/release/`.

## The shared graph API

Whichever reach-path loaded the backend, `graphFromNdjson` / `graphFromFormat` return the same `RustGraph`:

```ts
g.query`MATCH (p:Person) WHERE p.name = ${name} RETURN p`; // GQL; ${} → a bound $param, never spliced
g.query('MATCH (p:Person) WHERE p.age > $min RETURN p', { min: 30 });
g.gremlin`g.V().has('name', ${name}).values('age')`; // Gremlin; ${} escaped to a safe literal
g.queryArrow`MATCH (p:Person) RETURN p.name, p.age`; // raw in-process Arrow (ARW1) columnar bytes
g.queryArrowIpc`MATCH (p:Person) RETURN p.name, p.age`; // standard Arrow IPC bytes (DuckDB/Polars/pandas)
g.toNdjson(); // serialize back out
g.free(); // or `using`
```

**Arrow egress → DuckDB / Polars / pandas.** `queryArrow` returns lenke's compact in-process columnar blob (zero-copy for JS consumers). To hand a result to another Arrow tool, `queryArrowIpc(query, { format: 'stream' | 'file' })` frames it as **standard Apache Arrow IPC** entirely in the engine — the stream layout (`pyarrow.ipc.open_stream`, `polars.read_ipc_stream`) or the file / Feather-v2 layout (`pandas.read_feather`, `polars.read_ipc`). A pure-JS transcode of an existing blob is also available (`toArrowIPC(blob, format)` from `@lenke/native/arrow`, which brings its own zero runtime deps); the two produce byte-identical bytes.

Writes are GQL DML run through the same `query` call (`INSERT` / `SET` / `REMOVE` / `DELETE`), or Gremlin mutation traversals (`addV` / `addE` / `property` / `drop`) through `gremlin`.

**Injection safety.** Never build query text from untrusted input by string concatenation. Interpolate values only via a **tagged template** — each interpolation is bound as a GQL `$param`, or escaped into a safe literal for Gremlin (which has no engine-side binding, so the tag is the boundary) — or via a **`$name` param bag**, like `query('… WHERE x = $x', { x })`. The plain-string form `query('…' + userInput)` — the same method, no tag — is unsafe; the `query-safety/no-raw-interpolation` lint rule (shipped in `@lenke/dev`'s config) flags it, and you should suppress it with an `oxlint-disable` comment only when the text is genuinely trusted.

## Picking only one query language

The Rust core compiles GQL and Gremlin as Cargo features. To ship the engine with just the one your shop uses, build with `--features gql` _or_ `--features gremlin` (dropping the other), plus whatever codecs you need. The default build (`build:rust`) includes the full feature set; a trimmed build is smaller. This is the compile-time equivalent of installing one TS frontend package — see [choosing-your-build](./choosing-your-build.md#axis-2--the-query-frontend).

## Embedding it

For the lifecycle of an embedded cache — bulk load, query, mutate, rebuild-from-source, warm-start, and multi-tenancy — see [backend-embedded](./backend-embedded.md). The [`examples/service-map`](../../examples/service-map) server is a worked Node/N-API host.
