// `dumpSchema` reads the full active schema back out as replayable `SchemaOp`s —
// the inverse of the `create*` declarations. Proven on BOTH backends (ffi + wasm):
// declare one of every kind, dump it, and round-trip (apply the dump to a fresh
// graph, re-dump → identical). This is what lets a snapshot persist schema the
// graph NDJSON can't carry. Run: bun test packages/native/src/dump-schema.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createFfiBackend } from './backend-ffi.js';
import { createWasmBackend } from './backend-wasm.js';
import type { Backend } from './backend.js';
import { applySchemaOp, createEmptyGraph, type RustGraph, type SchemaOp } from './index.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

// Declare one of every schema kind. Cardinality/type carry their extra fields;
// the edge-unique below also proves an index-backed constraint dumps its auto-index.
const declareAll = (g: RustGraph): void => {
  g.createIndex({ on: 'vertex', kind: 'hash', keys: ['handle'] });
  g.createUniqueConstraint('User', 'email');
  g.createRequiredConstraint('User', 'name');
  g.createTypeConstraint('User', 'age', 'number');
  g.createTypeConstraint('Event', 'at', 'datetime');
  g.createEdgeUniqueConstraint('FOLLOWS', 'id'); // index-backed → also emits createEdgeIndex
  g.createEdgeRequiredConstraint('FOLLOWS', 'since');
  g.createEdgeTypeConstraint('RATED', 'stars', 'number');
  g.createCardinalityConstraint('User', 'FOLLOWS', 'out', 0, 5);
  g.createCardinalityConstraint('User', 'OWNS', 'in', 1, null); // unbounded max → null
  g.createValidator('User', 'u', 'u.age >= 0 AND u.age < 150');
  g.createInvariant('balanced', 'MATCH (a:Acct) RETURN sum(a.balance) = 0');
};

const backends: Array<{ name: string; make: () => Promise<Backend> | Backend; ok: boolean }> = [
  { name: 'ffi', make: () => createFfiBackend(LIB), ok: existsSync(LIB) },
  {
    name: 'wasm',
    make: async () => createWasmBackend(await Bun.file(WASM).arrayBuffer()),
    ok: existsSync(WASM),
  },
];

for (const { name, make, ok } of backends) {
  const suite = ok ? describe : describe.skip;

  suite(`dumpSchema (${name} backend)`, () => {
    test('captures every declaration kind with its fields', async () => {
      const g = createEmptyGraph(await make());
      declareAll(g);
      const dump = g.dumpSchema();
      const by = (op: string) => dump.filter((s) => s.op === op);

      // Structured ops, not write-language text — spot-check the field-carrying kinds.
      expect(by('createUniqueConstraint')).toEqual([
        { op: 'createUniqueConstraint', label: 'User', key: 'email' },
      ]);
      expect(by('createTypeConstraint')).toEqual([
        { op: 'createTypeConstraint', label: 'Event', key: 'at', type: 'datetime' },
        { op: 'createTypeConstraint', label: 'User', key: 'age', type: 'number' },
      ]);
      expect(by('createCardinalityConstraint')).toEqual([
        {
          op: 'createCardinalityConstraint',
          label: 'User',
          edgeType: 'FOLLOWS',
          direction: 'out',
          min: 0,
          max: 5,
        },
        {
          op: 'createCardinalityConstraint',
          label: 'User',
          edgeType: 'OWNS',
          direction: 'in',
          min: 1,
          max: null, // unbounded serializes to JSON null, not omitted
        },
      ]);
      expect(by('createValidator')).toEqual([
        {
          op: 'createValidator',
          label: 'User',
          varName: 'u',
          predicate: 'u.age >= 0 AND u.age < 150',
        },
      ]);
      expect(by('createInvariant')).toEqual([
        {
          op: 'createInvariant',
          name: 'balanced',
          query: 'MATCH (a:Acct) RETURN sum(a.balance) = 0',
        },
      ]);
      // The edge unique constraint is index-backed → its auto-created edge index is
      // in the dump too (so a replay reconstructs the index, not just the constraint).
      expect(by('createEdgeIndex')).toEqual([{ op: 'createEdgeIndex', key: 'id' }]);
    });

    test('round-trips: applying the dump to a fresh graph reproduces it exactly', async () => {
      const backend = await make();
      const src = createEmptyGraph(backend);
      declareAll(src);
      const dump = src.dumpSchema();

      const restored = createEmptyGraph(backend);

      for (const op of dump) {
        applySchemaOp(restored, op);
      }

      // Deterministic order + full fidelity → the re-dump is byte-identical.
      expect(restored.dumpSchema()).toEqual(dump);
    });

    test('an empty graph dumps an empty schema', async () => {
      const g = createEmptyGraph(await make());
      expect(g.dumpSchema()).toEqual([] as SchemaOp[]);
    });
  });
}
