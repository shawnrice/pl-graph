# Choosing your build

Three independent choices — **engine**, **query frontend**, **reach-path** — plus two cross-cutting concerns, **memory** and **numeric results**. This page is the matrix; the topology guides ([pure-ts](./pure-ts.md), [native](./native.md), [wasm](./wasm.md), and the frontend/backend guides) show each in context.

## Axis 1 — the engine

The graph substrate comes in two complete, interchangeable implementations. You pick one; you don't stack them (there's no "Rust storage behind a TS frontend" hybrid — that's nonsense, and lenke doesn't do it).

|                 | pure-TS — `@lenke/core`                                                                                                          | Rust core — `lenke-engine`                                                                      |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **What it is**  | A mutable in-memory labeled-property graph, driven by method calls (`addVertex`, `getVerticesByLabel`, opt-in property indexes). | A columnar graph you drive with a query language (GQL DML for writes, GQL/Gremlin for reads). |
| **Runs on**     | Anything that runs JS — browser, Node, Deno, Bun. No native artifact.                                                            | Reached from JS via one of three reach-paths (below).                                         |
| **Query**       | Bring a frontend: [`@lenke/gql`](../../packages/gql) or [`@lenke/gremlin`](../../packages/gremlin) over the core `Graph`.        | GQL and Gremlin compiled into the crate; also Apache Arrow columnar output.                   |
| **Strengths**   | Zero native deps, smallest footprint, trivial to embed, direct object access.                                                    | Throughput on large graphs and heavy queries; columnar scans; Arrow transfer.                 |
| **Use it when** | Small-to-medium graphs, quick embedding, or anywhere you can't ship a native/wasm artifact.                                      | Large data, query-heavy workloads, or anywhere you want the columnar engine.                  |

The two engines deliberately share only their **reactive change signal** — a monotonic `version` and per-token `epoch(name)` — which is what lets the React store and the sync engine work identically over either. See [pure-ts](./pure-ts.md) for the TS engine and [native](./native.md)/[wasm](./wasm.md) for the Rust one.

## Axis 2 — the query frontend

The graph is **query-language-agnostic**. A shop standardizes on GQL _or_ Gremlin — rarely both — so lenke lets you take only the one you use.

- **On the TS engine**, the frontend is a package you install over `@lenke/core`:
  - GQL → [`@lenke/gql`](../../packages/gql): `query(graph, 'MATCH …')`.
  - Gremlin → [`@lenke/gremlin`](../../packages/gremlin): `graph.toArray(traversal(V(id), values('name')))`.
  - Install one, tree-shake the other out entirely.
- **On the Rust engine**, the frontend is a **Cargo feature** compiled into the crate. Build with `--features gql` _or_ `--features gremlin` and drop the other. Same intent as the npm choice, coarser mechanism (you rebuild rather than re-install). This is also how you shrink a wasm bundle — see [wasm](./wasm.md).

GQL and Gremlin are faithful to their own ontologies, so where the two query models genuinely differ, semantics differ; otherwise they run over the same substrate and see the same data.

## Axis 3 — the reach-path (Rust engine only)

The Rust core is one crate reachable three ways. All three present the **same JS surface** — a `Backend` contract and the `RustGraph`/`Store` facades over it — so your code is identical regardless of which one loads.

|              | bun:ffi                                                                          | N-API                                                                                            | WebAssembly                                                                                           |
| ------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Package**  | `@lenke/native/ffi-engine`                                                              | [`@lenke/node`](../../packages/node)                                                             | `@lenke/native/wasm-engine`                                                                                  |
| **Load**     | `createFfiEngineBackend(libPath)` — synchronous `dlopen`; you supply the library path. | `createNodeBackend()` (facade) or raw `Graph` — synchronous `require` of a per-platform `.node`. | `await createWasmEngineBackend(source)` — **async**; `source` is `.wasm` bytes / a `Response` / a `Module`. |
| **Artifact** | `liblenke_engine.{so,dylib,dll}`                                                   | `lenke-node.<triple>.node`                                                                       | `lenke_engine.wasm`                                                                                     |
| **Build**    | `bun run build:rust` (in `@lenke/native`)                                        | `napi build --platform --release --esm` (in `@lenke/node`)                                       | `bun run build:wasm` (in `@lenke/native`)                                                             |
| **Runtime**  | Bun only (`bun:ffi`)                                                             | Node (the fast production path)                                                                  | Browser — and anything with a `WebAssembly` global (Node, Deno, Bun)                                  |
| **Threads**  | rayon (parallel NDJSON decode)                                                   | rayon                                                                                            | none — wasm has no threads, so the parallel decoder falls back to serial                              |

Guides: [native](./native.md) covers bun:ffi + N-API (server/CLI); [wasm](./wasm.md) covers WebAssembly (browser/universal).

## Numeric results across builds

**Arithmetic is exact everywhere.** `+ - * / %`, comparisons, `sqrt`, `round`/`floor`/`ceil`/`sign`, `degrees`/`radians`, the aggregations, and every [graph algorithm](./algorithms.md) give **bit-identical** answers on both engines and all three reach-paths. IEEE 754 pins those operations to one correctly-rounded result, and the algorithms additionally fix their summation order so floating-point non-associativity can't drift.

The **transcendental** functions are the exception — `exp`, `ln`, `log`, `log10`, `power`, `sin`, `cos`, `tan`, `cot`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`. IEEE 754 deliberately does _not_ mandate a correctly-rounded result for these, so implementations are free to differ in the last unit in the last place — and each build reaches a different implementation:

| build                    | where the math comes from                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| pure-TS — `@lenke/core`  | the JS runtime's `Math.*` (so it can also vary between V8, JSC and SpiderMonkey)                    |
| native — bun:ffi / N-API | the **host operating system's** math library — glibc on Linux, Apple's on macOS, the CRT on Windows |
| wasm                     | compiled into the module by Rust; `lenke_engine.wasm` imports nothing from the host                   |

Two consequences worth planning around:

- **A native result depends on the machine that produced it.** The same query answered on a Linux server and on a macOS laptop can differ in the last ulp, and so can two Linux hosts on different glibc versions. This is a property of the platform, not something lenke chooses.
- **wasm is the most reproducible reach-path.** Because the module carries its own implementations and imports nothing, it computes the same value on every machine that runs it — worth knowing if reproducibility across heterogeneous clients matters more to you than raw throughput.

Measured on Linux/glibc over 396 (function, argument) pairs, every disagreement is ~1 ulp and confined to the functions listed above: pure-TS and native agree on 391 of them, and the two Rust builds on 368. Those counts describe _that_ platform pairing rather than a constant — a macOS host, with a different system libm, would produce a different table.

A ~7% disagreement rate is unremarkable for two independent implementations. Nothing here is out of tolerance: a good libm targets under 1 ulp of error, and glibc's is nearer half that, so on the fraction of arguments where the two land on opposite sides of a rounding boundary they print different last digits. That is the expected behavior of implementations that are each individually correct, not a defect in either.

This is a well-trodden problem, and the standard answer is to **name one reference implementation and use it everywhere**. Java is the clearest case: [`java.lang.Math`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/StrictMath.html) is fast and platform-dependent, while `StrictMath` is specified to reproduce the published [fdlibm](https://www.netlib.org/fdlibm/readme) algorithms bit for bit on every platform — for very nearly the same function list as above — at some cost in speed. JS engines have converged the same way: ECMA-262 recommends (without requiring) fdlibm for `Math.sin`/`cos`/`tan`, and both V8 and SpiderMonkey adopted it, partly for reproducibility. The other end of the spectrum is consensus systems, which avoid floating point for these operations altogether.

lenke has not made that move yet. If cross-machine reproducibility matters more to you than matching the host's libm, say so — routing the Rust engine through its own compiled-in math on native as well as wasm would make **every** Rust build agree on every operating system, at the cost of widening the pure-TS gap. Today's default instead keeps pure-TS and native closely aligned and accepts that native tracks the host.

How far apart are they in practice? The worst single-call disagreement measured is **4 ulp — agreement to 15.3 significant digits** (`power(0.1, 10)`), and it does not meaningfully compound: an aggregate over 2000 rows still agrees to ~14 significant digits. So the gap is far below any tolerance you would set. The reason it matters anyway is **equality, not accuracy** — a 1-ulp difference still puts the same logical value in different `DISTINCT`/`GROUP BY` buckets on the two builds. The measured costs of each way out are recorded in [numeric-determinism](../design/numeric-determinism.md).

If a number has to match bit-for-bit across builds today, round it at the boundary (`round(x, 9)`) or compare it with a tolerance. Everything outside that function list — including all of the graph algorithms, the storage layer and every serialization codec — is safe to compare exactly.

## Memory model

The Rust engine's graph is heap-owned and must be released. lenke makes this **one rule across all three reach-paths**:

```ts
// Preferred — the same rule on every reach-path (see the N-API note below):
using g = graphFromNdjson(backend, bytes);
// ... use g ...
// (freed automatically here)

// Explicit — works on any build target, including older bundler outputs:
const g = graphFromNdjson(backend, bytes);
try {
  // ... use g ...
} finally {
  g.free(); // idempotent
}
```

- `using` needs a modern build target (TS ≥ 5.2 / esbuild down-levels it to `try/finally` and shims `Symbol.dispose`); `free()` is the universal fallback. lenke polyfills `Symbol.dispose` for runtimes that predate it, so `using` is safe to ship to browsers.
- A `Store` is disposable too: `using store = createStore(g)` frees the underlying graph.
- If you forget both, a `FinalizationRegistry` backstop reclaims the handle when the wrapper is garbage-collected. It's a leak-net (the GC may never run it before exit), **not** a substitute for `using`/`free()`.
- **N-API timing caveat:** on ffi and wasm, `free()` releases native memory _at that moment_. On N-API, `free()` drops the facade's reference and invalidates the handle, but the native memory itself is reclaimed when V8 garbage-collects the addon object — deterministic _invalidation_, GC-timed _reclamation_. A server cycling many graphs under memory pressure will see promptly-falling RSS on ffi, not necessarily on N-API.
- The pure-TS `@lenke/core` graph and the raw `@lenke/node` `Graph` class are ordinary GC-managed objects — nothing to free. The `RustGraph`/`Store` facades give even the N-API path a uniform `free()`/`using`, so you can write one lifecycle for all builds.

## Putting it together

A few common combinations:

- **Browser, GQL, local-first** → Rust engine · GQL · wasm, driven through [`@lenke/sync`](./frontend-worker.md) in a worker.
- **Node service cache, Gremlin** → Rust engine · Gremlin (`--features gremlin`) · N-API, embedded per [backend-embedded](./backend-embedded.md).
- **Anywhere, no native artifact** → TS engine · your frontend of choice, per [pure-ts](./pure-ts.md).
