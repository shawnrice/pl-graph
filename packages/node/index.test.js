// Proves the napi addon is callable from Node and that its Backend adapter drives
// the whole @lenke/native facade. Run: node --test (from packages/node), after
// `napi build`.
//
// The addon is a THIN layer over the engine's `lnk_*` C ABI (see src/lib.rs): the
// `Graph` class exposes the raw primitives (open / query(lang, q, params, format) /
// encode(format) / stat(which) / command …), and `buildEngineBackend` — driven
// here through `createNodeBackend` — assembles the high-level Backend on top. The
// first block smoke-tests the raw primitives; the rest proves the assembled facade.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ErrorCode, hasErrorCode, isLenkeError } from '@lenke/errors';
import { createStore, graphFromNdjson } from '@lenke/native';

import { createNodeBackend } from './backend.js';
import { Graph, abiVersion } from './index.js';

const NDJSON = Buffer.from(
  [
    JSON.stringify({
      type: 'node',
      id: 'n0',
      labels: ['Person'],
      properties: { name: 'marko', age: 29 },
    }),
    JSON.stringify({
      type: 'node',
      id: 'n1',
      labels: ['Person'],
      properties: { name: 'vadas', age: 27 },
    }),
    JSON.stringify({
      type: 'edge',
      id: 'e0',
      from: 'n0',
      to: 'n1',
      labels: ['KNOWS'],
      properties: {},
    }),
  ].join('\n'),
);
const dec = new TextDecoder();
const json = (buf) => JSON.parse(dec.decode(buf));

// Raw-abi format/lang bytes, mirroring src/lib.rs (and the bun:ffi abi).
const FMT_NDJSON = 0;
const FMT_JSON = 0;
const FMT_ARROW = 1;
const LANG_GQL = 0;
const LANG_GREMLIN = 1;
const openNdjson = () => Graph.open(NDJSON, FMT_NDJSON);

test('abiVersion matches the C ABI (18)', () => {
  assert.equal(abiVersion(), 18);
});

test('open decodes counts (stat: 0 = vertices, 1 = edges)', () => {
  const g = openNdjson();
  assert.equal(g.stat(0), 2);
  assert.equal(g.stat(1), 1);
});

test('query returns the {columns, rows} document', () => {
  const g = openNdjson();
  const doc = json(g.query(LANG_GQL, 'MATCH (p:Person) RETURN p.name', undefined, FMT_JSON));
  assert.deepEqual(new Set(doc.rows.flat()), new Set(['marko', 'vadas']));
});

test('query (format 1) returns an ARW1 columnar blob', () => {
  const g = openNdjson();
  const blob = g.query(LANG_GQL, 'MATCH (p:Person) RETURN p.age', undefined, FMT_ARROW);
  assert.ok(blob.length > 4);
  assert.equal(dec.decode(blob.subarray(0, 4)), 'ARW1');
});

test('gremlin (lang 1) returns a JSON result array', () => {
  const g = openNdjson();
  assert.deepEqual(json(g.query(LANG_GREMLIN, 'g.V().count()', undefined, FMT_JSON)), [2]);
});

test('version (stat: 2) advances on a mutating query', () => {
  const g = openNdjson();
  const before = g.stat(2);
  g.query(LANG_GQL, "MATCH (p:Person) WHERE p.name = 'marko' SET p.age = 99", undefined, FMT_JSON);
  assert.ok(g.stat(2) > before, 'version should advance after SET');
});

test('a bad query throws with a lenke-tagged, wire-coded message', () => {
  const g = openNdjson();
  assert.throws(() => g.query(LANG_GQL, 'NOT A QUERY', undefined, FMT_JSON), /lenke: query:/);
});

test('encode (format 0) round-trips the data as NDJSON', () => {
  const g = openNdjson();
  assert.match(dec.decode(g.encode(FMT_NDJSON)), /marko/);
});

test('params bind as data, never spliced (injection stays inert)', () => {
  const g = openNdjson();
  const rows = json(
    g.query(
      LANG_GQL,
      'MATCH (p:Person) WHERE p.name = $name RETURN p.age',
      JSON.stringify({ name: 'marko' }),
      FMT_JSON,
    ),
  );
  assert.equal(rows.rows.length, 1);
  assert.equal(rows.rows[0][0], 29);

  const before = g.stat(0);
  const hostile = json(
    g.query(
      LANG_GQL,
      'MATCH (p:Person) WHERE p.name = $name RETURN p.name',
      JSON.stringify({ name: "' DELETE p RETURN 1 //" }),
      FMT_JSON,
    ),
  );
  assert.equal(hostile.rows.length, 0);
  assert.equal(g.stat(0), before);

  // A nested-object param binds as a first-class MAP value (the engine's map/record
  // support) and is accepted — superseding lenke-core, which rejected any nested
  // param object as E_INVALID_VALUE. Only malformed JSON is refused at the boundary,
  // and it carries the stable WIRE code in its message tail (the same string the
  // ffi/wasm backends surface), not the Rust Debug name.
  assert.doesNotThrow(() =>
    g.query(LANG_GQL, 'MATCH (p:Person) RETURN p.name', '{"m":{"nested":1}}', FMT_JSON),
  );
  assert.throws(
    () => g.query(LANG_GQL, 'MATCH (p:Person) RETURN p', '{"bad": }', FMT_JSON),
    /E_INVALID_JSON/,
  );
});

test('createNodeBackend errors are coded LenkeErrors (parity with ffi/wasm)', () => {
  const g = graphFromNdjson(createNodeBackend(), NDJSON);

  // A GQL syntax error surfaces as a coded LenkeError, exactly as the bun:ffi
  // and wasm backends do — so `hasErrorCode` works uniformly across all three.
  let syntax;

  try {
    g.query('THIS IS NOT GQL');
  } catch (e) {
    syntax = e;
  }

  assert.ok(isLenkeError(syntax), 'a bad query should throw a LenkeError');
  assert.ok(hasErrorCode(syntax, ErrorCode.Syntax), 'code should be E_SYNTAX');
  assert.doesNotMatch(syntax.message, /\[E_/, 'the wire-code tag is stripped from the message');

  // A Gremlin parse error is coded too.
  let gremlin;

  try {
    g.gremlin('g.V().nope()');
  } catch (e) {
    gremlin = e;
  }

  assert.ok(hasErrorCode(gremlin, ErrorCode.Syntax), 'gremlin parse error → E_SYNTAX');

  // Bad NDJSON reports its own code (E_INVALID_JSON), not a coarse fallback.
  let bad;

  try {
    graphFromNdjson(createNodeBackend(), Buffer.from('not json'));
  } catch (e) {
    bad = e;
  }

  assert.ok(hasErrorCode(bad, ErrorCode.InvalidJson), 'bad NDJSON → E_INVALID_JSON');
});

test('createNodeBackend powers the @lenke/native facade + liveQuery', () => {
  const backend = createNodeBackend();
  assert.equal(backend.abiVersion, 18);

  const g = graphFromNdjson(backend, NDJSON);
  const store = createStore(g);
  const live = store.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
  assert.equal(live.getSnapshot().length, 2);

  // Referentially stable until a relevant mutation bumps the epoch.
  assert.strictEqual(live.getSnapshot(), live.getSnapshot());

  // A mutation touching a dependency ('name') recomputes to a fresh reference.
  const before = live.getSnapshot();
  store.mutate((graph) => graph.query("INSERT (:Person {name: 'zoe'})"));
  const after = live.getSnapshot();
  assert.notStrictEqual(after, before);
  assert.equal(after.length, 3);
});

test('algorithms run off-thread: resolve rows, non-blocking, single-flight', async () => {
  const backend = createNodeBackend();
  const g = graphFromNdjson(backend, NDJSON);

  // 1. Resolves the result rows (PageRank is a probability distribution ~ sums to 1).
  const rows = await g.pagerank({});
  const total = rows.reduce((s, r) => s + r.score, 0);
  assert.ok(Math.abs(total - 1) < 1e-9, 'PageRank mass should sum to 1');

  // 2. It does not block the event loop — a macrotask scheduled now runs before a
  //    long off-thread run resolves. On a synchronous (blocking) implementation the
  //    timer could not fire until after the result. The run must clearly outlast the
  //    timer's ~1ms floor, so this uses a big graph (a 4000-node ring) — the tiny
  //    NDJSON fixture resolves in well under a millisecond and would race the timer.
  const ringLines = [];

  for (let i = 0; i < 4000; i++) {
    ringLines.push(JSON.stringify({ type: 'node', id: `v${i}`, labels: ['P'], properties: {} }));
  }

  for (let i = 0; i + 1 < 4000; i++) {
    ringLines.push(
      JSON.stringify({ type: 'edge', id: `e${i}`, from: `v${i}`, to: `v${i + 1}`, labels: ['R'] }),
    );
  }

  const big = graphFromNdjson(createNodeBackend(), Buffer.from(ringLines.join('\n')));
  let ticked = false;
  setTimeout(() => {
    ticked = true;
  }, 0);
  await big.pagerank({ iterations: 2000 });
  assert.ok(ticked, 'event loop should have ticked during the off-thread run');

  // 3. writeProperty writes are applied (on the main thread) before it resolves.
  //    (g is the @lenke/native facade — query() returns decoded rows, not bytes.)
  await g.pagerank({ writeProperty: 'pr' });
  const back = g.query("MATCH (n:Person {name: 'marko'}) RETURN n.pr AS pr");
  assert.equal(typeof back[0].pr, 'number');

  // 4. Single-flight: touching the graph while a run is pending throws, and the
  //    graph is usable again once it settles.
  const pending = g.pagerank({});
  assert.throws(
    () => g.query('MATCH (n) RETURN n'),
    (e) => hasErrorCode(e, ErrorCode.InvalidGraphOp),
  );
  await pending;
  assert.ok(g.query('MATCH (n) RETURN count(*) AS c')[0].c >= 2);
});
