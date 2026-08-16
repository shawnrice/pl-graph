import { describe, expect, test } from 'bun:test';
// Exercises the engine backends (FFI + wasm) through the shared Backend contract.
// Build the artifacts first:
//   bun run engine:build:rust && bun run engine:build:wasm
import { existsSync, readFileSync } from 'node:fs';

import { hasErrorCode, ErrorCode } from '@lenke/errors';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { createWasmEngineBackend } from './backend-wasm-engine.js';
import type { Backend } from './backend.js';

const SO = new URL(
  '../../../crates/lenke-engine/target/release/liblenke_engine.so',
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../../crates/lenke-engine/target/wasm32-unknown-unknown/release/lenke_engine.wasm',
  import.meta.url,
).pathname;

const enc = new TextEncoder();
const dec = new TextDecoder();
const NDJSON = enc.encode(
  '{"id":"1","labels":["P"],"props":{"name":"alice","age":30}}\n' +
    '{"id":"2","labels":["P"],"props":{"name":"bob","age":25}}\n' +
    '{"from":"1","to":"2","id":"e0","type":"KNOWS","props":{}}\n',
);
const rows = (b: Uint8Array) =>
  JSON.parse(dec.decode(b)) as { columns: string[]; rows: unknown[][] };

// The behaviour every engine backend must satisfy, run against both transports.
const suite = (name: string, make: () => Promise<Backend> | Backend) => {
  describe(name, () => {
    test('load + counts + query', async () => {
      const be = await make();
      expect(be.abiVersion).toBe(18);
      const g = be.graphFromNdjson(NDJSON, false);
      expect(be.vertexCount(g)).toBe(2);
      expect(be.edgeCount(g)).toBe(1);
      expect(rows(be.queryRows(g, 'MATCH (n:P) RETURN count(*) AS c')).rows).toEqual([[2]]);
      be.graphFree(g);
    });

    test('params + gremlin', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      const r = rows(
        be.queryRows(g, 'MATCH (n:P) WHERE n.name = $nm RETURN n.age AS a', '{"nm":"alice"}'),
      );
      expect(r.rows).toEqual([[30]]);
      // Gremlin path routes through the same lnk_query(lang=1).
      const gr = JSON.parse(dec.decode(be.gremlinJson(g, 'g.V().count()')));
      expect(gr).toEqual([2]);
      be.graphFree(g);
    });

    test('version + index round-trip', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      const v0 = be.version(g);
      be.queryRows(g, "MATCH (n:P) WHERE n.name = 'alice' SET n.age = 31");
      expect(be.version(g)).toBeGreaterThan(v0);
      be.createIndex(g, 'vertex', 'hash', ['age']);
      expect(be.vertexIndexes(g)).toContain('age');
      be.graphFree(g);
    });

    test('unique constraint: clean ok, violation throws', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      be.createUniqueConstraint(g, 'P', 'name'); // distinct names → ok
      const dup = be.graphFromNdjson(
        enc.encode(
          '{"id":"1","labels":["P"],"props":{"e":"x"}}\n{"id":"2","labels":["P"],"props":{"e":"x"}}\n',
        ),
        false,
      );
      let threw = false;

      try {
        be.createUniqueConstraint(dup, 'P', 'e');
      } catch (err) {
        threw = true;
        expect(hasErrorCode(err, ErrorCode.ConstraintViolation)).toBe(true);
      }

      expect(threw).toBe(true);
      be.graphFree(g);
      be.graphFree(dup);
    });

    test('prepared statement: reuse, then use-after-free is a clean error', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      const p = be.prepare('MATCH (n:P) WHERE n.name = $nm RETURN n.age AS a');
      expect(rows(be.preparedQueryRows(p, g, '{"nm":"alice"}')).rows).toEqual([[30]]);
      expect(rows(be.preparedQueryRows(p, g, '{"nm":"bob"}')).rows).toEqual([[25]]);
      be.preparedFree(p);
      expect(() => be.preparedQueryRows(p, g, '{"nm":"alice"}')).toThrow();
      be.graphFree(g);
    });

    test('snapshot round-trips (ndjson + binary)', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      const asNdjson = be.encodeNdjson(g);
      const g2 = be.graphFromNdjson(asNdjson, false);
      expect(be.vertexCount(g2)).toBe(2);
      const asBinary = be.serialize(g, 'binary');
      const g3 = be.deserialize(asBinary, 'binary');
      expect(be.vertexCount(g3)).toBe(2);
      expect(be.edgeCount(g3)).toBe(1);
      [g, g2, g3].forEach((h) => be.graphFree(h));
    });

    test('unsupported methods throw E_UNSUPPORTED', async () => {
      const be = await make();
      const g = be.graphFromNdjson(NDJSON, false);
      expect(() => be.createValidator(g, 'P', 'p', 'p.age >= 0')).toThrow();
      expect(() => be.serialize(g, 'graphson')).toThrow();
      be.graphFree(g);
    });
  });
};

if (existsSync(SO)) {
  suite('engine FFI backend', () => createFfiEngineBackend(SO));
} else {
  test.skip(`engine FFI backend (missing ${SO} — run \`bun run engine:build:rust\`)`, () => {});
}

if (existsSync(WASM)) {
  suite('engine wasm backend', () => createWasmEngineBackend(readFileSync(WASM)));
} else {
  test.skip(`engine wasm backend (missing ${WASM} — run \`bun run engine:build:wasm\`)`, () => {});
}
