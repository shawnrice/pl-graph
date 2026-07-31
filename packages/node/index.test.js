// Proves the napi addon is callable from Node and that its Backend adapter
// drives the whole @lenke/native facade. Run: node --test (from packages/node),
// after `napi build`.
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

test('abiVersion matches the C ABI (18)', () => {
  assert.equal(abiVersion(), 18);
});

test('fromNdjson decodes counts', () => {
  const g = Graph.fromNdjson(NDJSON);
  assert.equal(g.vertexCount, 2);
  assert.equal(g.edgeCount, 1);
});

test('query returns the {columns, rows} document', () => {
  const g = Graph.fromNdjson(NDJSON);
  const doc = json(g.query('MATCH (p:Person) RETURN p.name'));
  assert.deepEqual(new Set(doc.rows.flat()), new Set(['marko', 'vadas']));
});

test('queryArrow returns an ARW1 columnar blob', () => {
  const g = Graph.fromNdjson(NDJSON);
  const blob = g.queryArrow('MATCH (p:Person) RETURN p.age');
  assert.ok(blob.length > 4);
  assert.equal(dec.decode(blob.subarray(0, 4)), 'ARW1');
});

test('gremlin returns a JSON result array', () => {
  const g = Graph.fromNdjson(NDJSON);
  assert.deepEqual(json(g.gremlin('g.V().count()')), [2]);
});

test('version advances on a mutating query', () => {
  const g = Graph.fromNdjson(NDJSON);
  const before = g.version();
  g.query("MATCH (p:Person) WHERE p.name = 'marko' SET p.age = 99");
  assert.ok(g.version() > before, 'version should advance after SET');
});

test('a bad query throws with a lenke-tagged message', () => {
  const g = Graph.fromNdjson(NDJSON);
  assert.throws(() => g.query('NOT A QUERY'), /lenke: query:/);
});

test('encodeNdjson round-trips the data', () => {
  const g = Graph.fromNdjson(NDJSON);
  assert.match(dec.decode(g.encodeNdjson()), /marko/);
});

// The payoff: the napi addon drives the shared Backend contract, so the whole
// @lenke/native facade (graphFromNdjson + createStore + liveQuery) runs on Node.
test('params bind as data, never spliced (injection stays inert)', () => {
  const g = Graph.fromNdjson(NDJSON);
  const rows = json(
    g.query(
      'MATCH (p:Person) WHERE p.name = $name RETURN p.age',
      JSON.stringify({ name: 'marko' }),
    ),
  );
  assert.equal(rows.rows.length, 1);
  assert.equal(rows.rows[0][0], 29);

  const before = g.vertexCount;
  const hostile = json(
    g.query(
      'MATCH (p:Person) WHERE p.name = $name RETURN p.name',
      JSON.stringify({ name: "' DELETE p RETURN 1 //" }),
    ),
  );
  assert.equal(hostile.rows.length, 0);
  assert.equal(g.vertexCount, before);

  // Malformed params reject with the coded error, not silent misbehavior. The
  // raw Graph message carries the stable WIRE code in its tail (`[E_INVALID_JSON]`),
  // the same string the ffi/wasm backends surface — not the Rust Debug name.
  assert.throws(
    () => g.query('MATCH (p:Person) RETURN p', '{"bad":{"nested":1}}'),
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
  //    long off-thread run (many PageRank iterations) resolves. On a synchronous
  //    (blocking) implementation the timer could not fire until after the result.
  let ticked = false;
  setTimeout(() => {
    ticked = true;
  }, 0);
  await g.pagerank({ iterations: 20_000 });
  assert.ok(ticked, 'event loop should have ticked during the off-thread run');

  // 3. writeProperty writes are applied (on the main thread) before it resolves.
  //    (g is the @lenke/native facade — query() returns decoded rows, not bytes.)
  await g.pagerank({ writeProperty: 'pr' });
  const back = g.query('MATCH (n:Person {name: "marko"}) RETURN n.pr AS pr');
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
