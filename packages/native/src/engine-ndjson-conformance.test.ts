// NDJSON interchange (engine <-> core): proves the two native engines speak the
// SAME NDJSON dialect now — a core-format document loads into both, and each
// engine's `encodeNdjson` output loads into the OTHER with the data intact. This
// is the drop-in tripwire for `graphFromNdjson` / snapshot interchange; the
// engine used to emit `{"id",…,"props"}` where core emits `{"type",…,"properties"}`.
//
// Build both libs first: `bun run build:rust && bun run engine:build:rust`.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { createFfiBackend } from './backend-ffi.js';
import type { Backend, GraphHandle } from './backend.js';

const CORE = new URL('../../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url)
  .pathname;
const ENGINE = new URL(
  '../../../crates/lenke-engine/target/release/liblenke_engine.so',
  import.meta.url,
).pathname;
const ready = existsSync(CORE) && existsSync(ENGINE);
const suite = ready ? describe : describe.skip;

const enc = new TextEncoder();
const dec = new TextDecoder();
const rows = (b: Uint8Array) =>
  JSON.parse(dec.decode(b)) as { columns: string[]; rows: unknown[][] };

// A canonical core-format NDJSON document (the SHIPPED shape) with two nodes,
// a temporal + a record property, and a typed edge.
const SEED = enc.encode(
  '{"type":"node","id":"1","labels":["P"],"properties":{"name":"a","age":30,"born":{"@date":"1990-05-01"}}}\n' +
    '{"type":"node","id":"2","labels":["P","Q"],"properties":{"name":"b","meta":{"x":1,"y":"hi"}}}\n' +
    '{"type":"edge","id":"e0","from":"1","to":"2","labels":["KNOWS"],"properties":{"w":1.5}}\n',
);

// The order-independent shape of a graph: sorted node names + edge endpoints +
// counts, read via GQL so it doesn't depend on either engine's property order.
const shape = (be: Backend, g: GraphHandle) => ({
  v: be.vertexCount(g),
  e: be.edgeCount(g),
  names: rows(be.queryRows(g, 'MATCH (n:P) RETURN n.name AS n ORDER BY n')).rows,
  edge: rows(be.queryRows(g, 'MATCH (a)-[r:KNOWS]->(b) RETURN a.name AS f, b.name AS t, r.w AS w'))
    .rows,
});

suite('ndjson interchange (engine <-> core)', () => {
  const core = createFfiBackend(CORE);
  const engine = createFfiEngineBackend(ENGINE);

  test('both engines parse the same core-format NDJSON identically', () => {
    const cg = core.graphFromNdjson(SEED, false);
    const eg = engine.graphFromNdjson(SEED, false);
    expect(shape(engine, eg)).toEqual(shape(core, cg));
    core.graphFree(cg);
    engine.graphFree(eg);
  });

  test("the engine's NDJSON output loads into core with data intact", () => {
    const eg = engine.graphFromNdjson(SEED, false);
    const dumped = engine.encodeNdjson(eg); // engine-emitted NDJSON
    const cg = core.graphFromNdjson(dumped, false); // …parsed by core
    expect(shape(core, cg)).toEqual(shape(engine, eg));
    engine.graphFree(eg);
    core.graphFree(cg);
  });

  test("core's NDJSON output loads into the engine with data intact", () => {
    const cg = core.graphFromNdjson(SEED, false);
    const dumped = core.encodeNdjson(cg); // core-emitted NDJSON
    const eg = engine.graphFromNdjson(dumped, false); // …parsed by the engine
    expect(shape(engine, eg)).toEqual(shape(core, cg));
    core.graphFree(cg);
    engine.graphFree(eg);
  });
});
