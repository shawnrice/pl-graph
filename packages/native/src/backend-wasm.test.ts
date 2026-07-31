import { describe, expect, test } from 'bun:test';
// End-to-end proof of the wasm backend: instantiate the lenke_core.wasm
// artifact, build a graph from NDJSON, and drive GQL + Gremlin through the same
// `RustGraph` facade the FFI backend uses. This is the test that proves the
// linear-memory marshalling (lnk_alloc in, copy out) actually works in a JS
// runtime. Run: bun test packages/native/src/backend-wasm.test.ts
import { existsSync } from 'node:fs';

import { ErrorCode, hasErrorCode, isLenkeError } from '@lenke/errors';

import { ABI_VERSION } from './abi.js';
import { createWasmBackend } from './backend-wasm.js';
import { graphFromFormat, graphFromNdjson } from './graph.js';
import { createStore } from './store.js';

const WASM = new URL(
  '../../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

// The artifact is built by `bun run build:wasm` (not by the test). Skip cleanly
// with a hint when it's absent, rather than hard-erroring at module load.
const hasWasm = existsSync(WASM);

if (!hasWasm) {
  console.warn(
    `[backend-wasm.test] skipping: ${WASM} not found — run \`bun run build:wasm\` first.`,
  );
}

const suite = hasWasm ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}',
  '{"type":"edge","id":"e1","labels":["knows"],"from":"a","to":"b","properties":{"weight":0.5}}',
].join('\n');

const bytes = new TextEncoder().encode(NDJSON);
const wasmBytes = hasWasm ? await Bun.file(WASM).arrayBuffer() : new ArrayBuffer(0);

suite('@lenke/native wasm backend', () => {
  test('instantiates at the expected ABI version', async () => {
    const backend = await createWasmBackend(wasmBytes);
    expect(backend.abiVersion).toBe(ABI_VERSION);
  });

  test('builds a graph and reports counts', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);
    expect(g.vertexCount).toBe(2);
    expect(g.edgeCount).toBe(1);
    g.free();
  });

  test('runs a GQL query through the facade (string + template)', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    const rows = g.query('MATCH (n:P) RETURN n.name, n.age ORDER BY n.age');
    expect(rows).toEqual([
      { 'n.name': 'vadas', 'n.age': 27 },
      { 'n.name': 'marko', 'n.age': 29 },
    ]);

    const min = 28;
    const tpl = g.query`MATCH (n:P) WHERE n.age > ${min} RETURN n.name`;
    expect(tpl).toEqual([{ 'n.name': 'marko' }]);

    g.free();
  });

  test('runs a textual Gremlin query through the facade', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);
    const names = g.gremlin("g.V().has('name','marko').out('knows').values('name')");
    expect(names).toEqual(['vadas']);
    g.free();
  });

  test('match / subgraph / shortestPath work over wasm linear memory', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    // match(): marko —knows→ vadas, bound to (a, b).
    expect(
      g.gremlin("g.V().match(__.as('a').out('knows').as('b')).select('a','b').by('name')"),
    ).toEqual([{ a: 'marko', b: 'vadas' }]);

    // shortestPath(): marko → vadas, one hop (vertex ids a, b).
    const sp = g.gremlin(
      "g.V().has('name','marko').shortestPath().with(ShortestPath.target, __.has('name','vadas'))",
    ) as Array<Array<{ id: string }>>;
    expect(sp.map((p) => p.map((v) => v.id))).toEqual([['a', 'b']]);

    // subgraph(): the one knows edge → 2 vertices, 1 edge.
    const sg = g.gremlin("g.E().hasLabel('knows').subgraph('sg').cap('sg')") as [
      { vertices: unknown[]; edges: unknown[] },
    ];
    expect(sg[0].vertices).toHaveLength(2);
    expect(sg[0].edges).toHaveLength(1);

    g.free();
  });

  test('round-trips through NDJSON (and preserves the edge id)', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);
    const out = g.toNdjson();
    expect(new TextDecoder().decode(out)).toContain('"id":"e1"');
    const g2 = graphFromNdjson(backend, out);
    expect(g2.vertexCount).toBe(2);
    expect(g2.edgeCount).toBe(1);
    g.free();
    g2.free();
  });

  test('serializes + round-trips through every format (over linear memory)', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    for (const fmt of ['pg-json', 'pg-text', 'graphson', 'csv', 'ndjson'] as const) {
      const doc = g.serialize(fmt);
      expect(doc.length).toBeGreaterThan(0);
      const g2 = graphFromFormat(backend, doc, { format: fmt });
      expect(g2.vertexCount).toBe(2);
      expect(g2.edgeCount).toBe(1);
      g2.free();
    }

    g.free();
  });

  test('reactive store works over linear memory (version + finer invalidation)', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);
    const store = createStore(g);

    const names = store.liveQuery('MATCH (n:P) RETURN n.name ORDER BY n.name', {
      deps: ['P', 'name'],
    });
    const ages = store.liveQuery('MATCH (n:P) RETURN n.age ORDER BY n.age', { deps: ['P', 'age'] });
    const namesBefore = names.getSnapshot();
    const agesBefore = ages.getSnapshot();
    const v0 = store.version;

    store.mutate((gr) => gr.query("MATCH (n:P) WHERE n.name = 'marko' SET n.age = 99"));

    expect(store.version).toBeGreaterThan(v0); // u64 version crossed linear memory
    expect(ages.getSnapshot()).not.toBe(agesBefore); // `age` dep moved
    expect(names.getSnapshot()).toBe(namesBefore); // `name`/`P` deps did not → stable ref
    g.free();
  });

  test('params bind through linear memory (never spliced into the text)', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    const rows = g.query`MATCH (n:P) WHERE n.name = ${'marko'} RETURN n.age`;
    expect(rows).toHaveLength(1);

    // An injection-shaped value stays inert data on the wasm path too.
    const hostile = g.query`MATCH (n:P) WHERE n.name = ${"' RETURN n //"} RETURN n.name`;
    expect(hostile).toEqual([]);
    g.free();
  });

  test('a large insert exercises a memory grow', async () => {
    const backend = await createWasmBackend(wasmBytes);
    // ~5k nodes of NDJSON forces the wasm heap to grow past its initial pages.
    const lines: string[] = [];

    for (let i = 0; i < 5000; i += 1) {
      lines.push(`{"type":"node","id":"n${i}","labels":["P"],"properties":{"age":${i % 80}}}`);
    }

    const big = new TextEncoder().encode(lines.join('\n'));
    const g = graphFromNdjson(backend, big);
    expect(g.vertexCount).toBe(5000);
    const rows = g.query('MATCH (n:P) WHERE n.age = 42 RETURN count(*) AS c');
    expect(rows[0].c).toBe(62); // i%80==42 for i in 0..4999 → 42,122,…,4922 = 62
    g.free();
  });

  // The crate's structured last-error is read back *out of linear memory* and
  // rethrown with the shared `ErrorCode` — so a browser-side wasm graph reports
  // failures identically to the server-side FFI backend (no second-class errors).
  test('a GQL syntax error surfaces as a coded LenkeError with crate details', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    let caught: unknown;

    try {
      g.query('THIS IS NOT GQL');
    } catch (e) {
      caught = e;
    }

    expect(isLenkeError(caught)).toBe(true);
    expect(hasErrorCode(caught, ErrorCode.Syntax)).toBe(true);
    expect((caught as { details?: { pos?: number } }).details?.pos).toBeTypeOf('number');
    g.free();
  });

  test('an unknown serialize format throws a coded UnknownFormat', async () => {
    const backend = await createWasmBackend(wasmBytes);
    const g = graphFromNdjson(backend, bytes);

    let caught: unknown;

    try {
      g.serialize('nope');
    } catch (e) {
      caught = e;
    }

    expect(hasErrorCode(caught, ErrorCode.UnknownFormat)).toBe(true);
    g.free();
  });

  test('invalid-UTF-8 NDJSON fails as a coded LenkeError, not a raw Error', async () => {
    const backend = await createWasmBackend(wasmBytes);

    let caught: unknown;

    try {
      graphFromNdjson(backend, new Uint8Array([0xff, 0xfe, 0xfd]));
    } catch (e) {
      caught = e;
    }

    // The crate reports bad UTF-8 on the FFI channel; the point is it's a coded
    // LenkeError, never a bare `Error` (which is what the wasm backend used to
    // throw before it read the last-error).
    expect(isLenkeError(caught)).toBe(true);
    expect(hasErrorCode(caught, ErrorCode.Ffi)).toBe(true);
  });
});
