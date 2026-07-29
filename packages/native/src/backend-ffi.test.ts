// End-to-end proof of the FFI backend through the public facade: load the native
// library, build a graph from NDJSON, and run GQL + Gremlin through `RustGraph`.
// The wasm backend exercises the identical `Backend` contract; its own test runs
// in a browser/wasm host. Run: bun test packages/native/src/backend-ffi.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { ErrorCode, hasErrorCode, isLenkeError } from '@lenke/errors';

import { ABI_VERSION } from './abi.js';
import { createFfiBackend } from './backend-ffi.js';
import {
  createEmptyGraph,
  decodeArrow,
  escapeGremlin,
  graphFromFormat,
  graphFromNdjson,
  gremlin,
} from './graph.js';

// The shared-library extension is platform-specific: macOS `.dylib`, Linux
// `.so`, Windows `.dll`. `build:rust` emits the one for the host.
const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;

// The artifact is built by `bun run build:rust` (not by the test). Skip cleanly
// with a hint when it's absent, rather than hard-erroring at dlopen.
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[backend-ffi.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}',
  '{"type":"edge","id":"e1","labels":["knows"],"from":"a","to":"b","properties":{"weight":0.5}}',
].join('\n');

const bytes = new TextEncoder().encode(NDJSON);

suite('@lenke/native FFI backend', () => {
  test('loads at the expected ABI version', () => {
    const backend = createFfiBackend(LIB);
    expect(backend.abiVersion).toBe(ABI_VERSION);
  });

  test('builds a graph and reports counts', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);
    expect(g.vertexCount).toBe(2);
    expect(g.edgeCount).toBe(1);
    g.free();
  });

  test('createEmptyGraph cold-boots a blank graph you can INSERT into', () => {
    const backend = createFfiBackend(LIB);
    const g = createEmptyGraph(backend);
    expect(g.vertexCount).toBe(0);
    expect(g.edgeCount).toBe(0);

    g.query(`INSERT (:Person {name: 'ada'})`);
    expect(g.vertexCount).toBe(1);
    g.free();
  });

  test('createVertexIndex is exposed and an indexed param lookup is correct', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    // Idempotent; declaring an index must not change query results — only make
    // `{k: $x}` / `WHERE .k = $x` seek instead of scan.
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });

    expect(g.query('MATCH (n:P {name: $n}) RETURN n.age', { n: 'marko' })).toEqual([
      { 'n.age': 29 },
    ]);
    expect(g.query('MATCH (n:P) WHERE n.name = $n RETURN n.age', { n: 'vadas' })).toEqual([
      { 'n.age': 27 },
    ]);
    expect(g.query('MATCH (n:P {name: $n}) RETURN n.age', { n: 'nobody' })).toEqual([]);

    g.free();
  });

  test('the index API round-trips: create → list → drop (parity with the TS graph)', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    expect(g.vertexIndexes()).toEqual([]);
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    g.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });
    expect(g.vertexIndexes()).toEqual(['age', 'name']); // sorted
    expect(g.edgeIndexes()).toEqual(['weight']);

    g.dropVertexIndex('age');
    expect(g.vertexIndexes()).toEqual(['name']);
    g.dropVertexIndex('missing'); // no-op
    expect(g.vertexIndexes()).toEqual(['name']);

    g.free();
  });

  test('createIndex routes kind:interval to the RI-tree — an as-of seek is byte-identical to the scan', () => {
    const backend = createFfiBackend(LIB);

    // Three employment edges with a valid interval [vf, vt); an as-of query at
    // v=15 should match only the edge whose interval covers it. The consolidated
    // createIndex({ on:'edge', kind:'interval' }) must route to the edge interval
    // index and return exactly what the un-indexed scan does.
    const build = () => {
      const g = createEmptyGraph(backend);
      g.query(
        "INSERT (:E {id: 'a'})-[:EMPLOYS {vf: 0, vt: 10}]->(:C {id: 'x'}), " +
          "(:E {id: 'b'})-[:EMPLOYS {vf: 10, vt: 20}]->(:C {id: 'y'}), " +
          "(:E {id: 'c'})-[:EMPLOYS {vf: 20, vt: 30}]->(:C {id: 'z'})",
      );

      return g;
    };
    const asOf =
      'MATCH ()-[r:EMPLOYS]->() WHERE r.vf <= $v AND r.vt > $v RETURN r.vf ORDER BY r.vf';

    const scan = build();
    const expected = scan.query(asOf, { v: 15 });

    expect(expected).toEqual([{ 'r.vf': 10 }]);
    scan.free();

    const seek = build();

    seek.createIndex({ on: 'edge', kind: 'interval', keys: ['vf', 'vt'] });
    expect(seek.query(asOf, { v: 15 })).toEqual(expected);
    // A boundary probe (vt is exclusive): v=10 excludes [0,10), includes [10,20).
    expect(seek.query(asOf, { v: 10 })).toEqual([{ 'r.vf': 10 }]);
    seek.free();
  });

  test('mergeNdjson bulk-appends into a live graph (COPY FROM)', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes); // 2 nodes (marko, vadas), 1 edge
    expect(g.vertexCount).toBe(2);
    expect(g.edgeCount).toBe(1);

    const clean = g.mergeNdjson(
      new TextEncoder().encode(
        '{"type":"node","id":"c","labels":["P"],"properties":{"name":"josh","age":32}}\n' +
          '{"type":"edge","id":"e2","from":"a","to":"c","labels":["knows"],"properties":{}}',
      ),
    );
    // A clean merge reports what landed and nothing skipped.
    expect(clean).toEqual({
      nodesAdded: 1,
      edgesAdded: 1,
      nodesSkipped: [],
      edgesSkipped: [],
      phantomVertices: [],
    });

    // A dirty merge reports the conflicts: an existing id (first-wins), a
    // duplicate edge id, and an edge endpoint that was never declared.
    const dirty = g.mergeNdjson(
      new TextEncoder().encode(
        '{"type":"node","id":"a","labels":["P"],"properties":{"name":"IGNORED"}}\n' +
          '{"type":"edge","id":"e2","from":"a","to":"c","labels":["knows"],"properties":{}}\n' +
          '{"type":"edge","from":"a","to":"ghost","labels":["knows"],"properties":{}}',
      ),
    );
    expect(dirty.nodesSkipped).toEqual(['a']);
    expect(dirty.edgesSkipped).toEqual(['e2']);
    expect(dirty.phantomVertices).toEqual(['ghost']);
    expect(dirty.nodesAdded).toBe(0);

    expect(g.vertexCount).toBe(4); // c + the ghost phantom
    expect(g.edgeCount).toBe(3);
    expect(g.query('MATCH (n:P) RETURN n.name ORDER BY n.name').map((r) => r['n.name'])).toEqual([
      'josh',
      'marko',
      'vadas',
    ]);
    // An indexed key stays queryable — the append maintained the index.
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    g.mergeNdjson(
      new TextEncoder().encode(
        '{"type":"node","id":"d","labels":["P"],"properties":{"name":"peter"}}',
      ),
    );
    expect(g.query('MATCH (n:P {name: $n}) RETURN n.name', { n: 'peter' })).toEqual([
      { 'n.name': 'peter' },
    ]);

    g.free();
  });

  test('prepare() compiles a reusable query bound to the graph', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    const q = g.prepare('MATCH (n:P) WHERE n.age > $min RETURN n.name ORDER BY n.name');
    // Same compiled plan, rerun with fresh params — and identical to query().
    expect(q.query({ min: 28 })).toEqual([{ 'n.name': 'marko' }]);
    expect(q.query({ min: 100 })).toEqual([]);
    expect(q.query({ min: 26 })).toEqual([{ 'n.name': 'marko' }, { 'n.name': 'vadas' }]);
    expect(q.query({ min: 26 })).toEqual(
      g.query('MATCH (n:P) WHERE n.age > $min RETURN n.name ORDER BY n.name', { min: 26 }),
    );

    q.free();
    expect(() => q.query({ min: 28 })).toThrow(/used after free/);

    // A syntax error surfaces at prepare time.
    expect(() => g.prepare('MATCH (n RETURN n')).toThrow();

    g.free();
  });

  test('runs a GQL query through the facade (string + template)', () => {
    const backend = createFfiBackend(LIB);
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

  test('runs a textual Gremlin query through the facade', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);
    const names = g.gremlin("g.V().has('name','marko').out('knows').values('name')");
    expect(names).toEqual(['vadas']);
    g.free();
  });

  test('escapeGremlin serializes scalars to safe literals and rejects the rest', () => {
    expect(escapeGremlin("o'brien")).toBe("'o\\'brien'");
    expect(escapeGremlin(29)).toBe('29');
    expect(escapeGremlin(-3.5)).toBe('-3.5');
    expect(escapeGremlin(true)).toBe('true');
    expect(escapeGremlin(false)).toBe('false');
    expect(() => escapeGremlin(null)).toThrow(); // no gremlin null literal
    expect(() => escapeGremlin(1e21)).toThrow(); // exponent form isn't lexable
    expect(() => escapeGremlin({ a: 1 })).toThrow();

    expect(gremlin`g.V().has('name', ${'marko'}).count()`).toBe(
      "g.V().has('name', 'marko').count()",
    );
  });

  // Gremlin has no engine-side `$name` binding — the tagged template IS the
  // binding mechanism. It refused every non-scalar, so a temporal could only be
  // hand-spelled into the text, which defeats the point of an escaping helper and
  // is unsafe for any value that isn't a trusted constant.
  test('escapeGremlin embeds temporals as dialect literal constructors', () => {
    // Both the stored instance and the tagged wire form a caller reads back out.
    expect(escapeGremlin({ '@date': '2020-01-01' })).toBe("date('2020-01-01')");
    expect(escapeGremlin({ '@datetime': '2020-01-01T12:30:00' })).toBe(
      "datetime('2020-01-01T12:30:00')",
    );
    expect(escapeGremlin({ '@duration': 'P1D' })).toBe("duration('P1D')");
    // `@localtime` is spelled `time(...)`, so a plain tag-slice would be wrong.
    expect(escapeGremlin({ '@localtime': '12:30:00' })).toBe("time('12:30:00')");

    expect(gremlin`g.V().has('vf', lte(${{ '@date': '2021-06-01' }}))`).toBe(
      "g.V().has('vf', lte(date('2021-06-01')))",
    );
  });

  // `gremlin(text, params)` reads exactly like GQL's `query(text, params)`, but a
  // plain string has no interpolation sites — the params were silently discarded
  // and the failure surfaced far downstream as a parse error on un-substituted
  // text. Refuse it at the call instead, and name the form that does work.
  test('gremlin(text, params) is refused rather than silently dropping the params', () => {
    expect(() =>
      (gremlin as (q: string, ...s: unknown[]) => string)("g.V().has('vf', lte($v))", {
        v: { '@date': '2021-06-01' },
      }),
    ).toThrow(/not a binding form/);

    // A plain string with nothing to substitute still passes through untouched.
    expect(gremlin('g.V().count()')).toBe('g.V().count()');
  });

  test('the Gremlin tagged template escapes interpolations — injection stays inert', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);
    const before = g.vertexCount;

    // A value engineered to close the string and inject a drop must not run.
    const evil = "marko'); g.V().drop(); //";
    const rows = g.gremlin`g.V().has('name', ${evil}).values('name')`;

    expect(rows).toEqual([]); // matched nothing — it's one literal string
    expect(g.vertexCount).toBe(before); // the graph was NOT dropped
    // A legit value with a quote still round-trips and matches:
    expect(g.gremlin`g.V().has('name', ${'marko'}).count()`).toEqual([1]);
    g.free();
  });

  test('decodeArrow round-trips queryArrow back to the same rows, nulls included', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    // n.missing is absent on every node → a fully-null column (validity bitmap).
    const q = 'MATCH (n:P) RETURN n.name, n.age, n.missing ORDER BY n.name';
    const blob = g.queryArrow(q);

    expect(new TextDecoder().decode(blob.subarray(0, 4))).toBe('ARW1');
    expect(decodeArrow(blob)).toEqual(g.query(q)); // exact parity with the JSON path
    g.free();
  });

  test('decodeArrow reconstructs a FixedSizeList column (would have thrown before)', () => {
    const backend = createFfiBackend(LIB);
    const ndjson = new TextEncoder().encode(
      [
        '{"type":"node","id":"a","labels":["V"],"properties":{"name":"a","h":[1.5,2.5,3.5]}}',
        '{"type":"node","id":"b","labels":["V"],"properties":{"name":"b"}}', // no h → null list
      ].join('\n'),
    );
    const g = graphFromNdjson(backend, ndjson);
    const q = 'MATCH (n:V) RETURN n.h AS h ORDER BY n.name';

    // The list column egresses as a FixedSizeList; decodeArrow rebuilds number[].
    expect(decodeArrow(g.queryArrow(q))).toEqual([{ h: [1.5, 2.5, 3.5] }, { h: null }]);
    // …and matches the JSON `.query()` path exactly.
    expect(decodeArrow(g.queryArrow(q))).toEqual(g.query(q));
    g.free();
  });

  test('round-trips through NDJSON', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);
    const out = g.toNdjson();
    const g2 = graphFromNdjson(backend, out);
    expect(g2.vertexCount).toBe(2);
    expect(g2.edgeCount).toBe(1);
    g.free();
    g2.free();
  });

  test('serializes + round-trips through every format', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    for (const fmt of ['pg-json', 'pg-text', 'graphson', 'csv', 'ndjson']) {
      const doc = g.serialize(fmt);
      expect(doc.length).toBeGreaterThan(0);
      const g2 = graphFromFormat(backend, doc, fmt);
      expect(g2.vertexCount).toBe(2);
      expect(g2.edgeCount).toBe(1);
      // the GQL query gives the same answer regardless of the carrier format
      expect(g2.query('MATCH (n:P) RETURN n.name ORDER BY n.name')).toEqual([
        { 'n.name': 'marko' },
        { 'n.name': 'vadas' },
      ]);
      g2.free();
    }

    g.free();
  });

  test('graphson preserves the edge id; unknown format throws a coded error', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);
    const gson = g.serialize('graphson');
    expect(gson).toContain('"e1"'); // the edge id survives

    let caught: unknown;

    try {
      g.serialize('nope');
    } catch (e) {
      caught = e;
    }

    expect(hasErrorCode(caught, ErrorCode.UnknownFormat)).toBe(true);
    g.free();
  });

  // The failure crossing: a real crate error rides the last-error side channel,
  // gets read back, and arrives as a `LenkeError` carrying the *same*
  // `ErrorCode` a pure-TS engine would raise — identical to the wasm backend.
  test('a GQL syntax error surfaces as a coded LenkeError with crate details', () => {
    const backend = createFfiBackend(LIB);
    const g = graphFromNdjson(backend, bytes);

    let caught: unknown;

    try {
      g.query('THIS IS NOT GQL');
    } catch (e) {
      caught = e;
    }

    expect(isLenkeError(caught)).toBe(true);
    expect(hasErrorCode(caught, ErrorCode.Syntax)).toBe(true);
    // the parse offset carried over from the crate's structured report
    expect((caught as { details?: { pos?: number } }).details?.pos).toBeTypeOf('number');
    g.free();
  });
});
