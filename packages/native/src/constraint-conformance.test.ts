// Cross-engine differential for the unique-constraint primitive: the SAME
// createUniqueConstraint + INSERT/SET sequence runs on BOTH the TS engine
// (@lenke/core + @lenke/gql) and the Rust core (over bun:ffi), asserting the two
// agree on every write outcome (the same ConstraintViolation, or both succeed)
// AND on the resulting graph state byte-for-byte. This is the divergence
// tripwire the parallel unit tests can't be. See docs/design/gql-extensions.md.
//
// Run: bun test packages/native/src/constraint-conformance.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import {
  createInvariant as tsCreateInvariant,
  createValidator as tsCreateValidator,
  query as tsQuery,
} from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { graphFromNdjson } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  // eslint-disable-next-line no-console
  console.warn(`[constraint] skipping: ${LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

// A seed node of an unrelated label — both engines start byte-identical, and
// `Acct` / `Other` are a clean namespace for the constraint.
const SEED = '{"type":"node","id":"seed","labels":["Seed"],"properties":{}}';

type Outcome = { ok: true } | { code: unknown };

const outcome = (run: () => unknown): Outcome => {
  try {
    run();

    return { ok: true };
  } catch (e) {
    return { code: (e as { code?: unknown }).code };
  }
};

suite('unique-constraint differential (TS vs native)', () => {
  // One identical write sequence, applied to whichever engine drives it.
  const SCRIPT: Array<{ label: string; sql: string }> = [
    { label: 'insert first Acct', sql: `INSERT (:Acct {email: 'a@x.io', name: 'A'})` },
    { label: 'duplicate email → violation', sql: `INSERT (:Acct {email: 'a@x.io', name: 'B'})` },
    { label: 'different email ok', sql: `INSERT (:Acct {email: 'b@x.io', name: 'B'})` },
    { label: 'different label ok', sql: `INSERT (:Other {email: 'a@x.io'})` },
    { label: 'null email #1 (exempt)', sql: `INSERT (:Acct {email: null, name: 'N1'})` },
    { label: 'null email #2 (exempt)', sql: `INSERT (:Acct {email: null, name: 'N2'})` },
    {
      label: 'SET collision → violation',
      sql: `MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'a@x.io'`,
    },
    { label: 'SET self ok', sql: `MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'b@x.io'` },
  ];

  test('every write outcome and the final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createUniqueConstraint('Acct', 'email');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createUniqueConstraint('Acct', 'email');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    // The resulting graph must be byte-identical too — same Accts, same values.
    const READ = `MATCH (n:Acct) RETURN n.name, n.email ORDER BY n.name`;
    const tsRows = JSON.stringify(tsQuery(tsGraph, READ));
    const nativeRows = JSON.stringify(nativeGraph.query(READ));

    expect(nativeRows).toEqual(tsRows);
  });

  test('declare-time rejection agrees across engines (pre-existing duplicates)', () => {
    const dup = [
      '{"type":"node","id":"1","labels":["Acct"],"properties":{"email":"dup@x.io"}}',
      '{"type":"node","id":"2","labels":["Acct"],"properties":{"email":"dup@x.io"}}',
    ].join('\n');

    const tsGraph = tsDeserialize(dup, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, dup);

    const ts = outcome(() => tsGraph.createUniqueConstraint('Acct', 'email'));
    const native = outcome(() => nativeGraph.createUniqueConstraint('Acct', 'email'));

    expect(native).toEqual(ts);
  });

  // Dropping the index that backs a unique constraint would downgrade
  // enforcement to a scan (native) or silently lose it (TS). Both engines now
  // refuse it identically; a plain index still drops, and enforcement survives.
  test('dropping a unique-constraint-backing index is refused, identically', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const nativeGraph = graphFromNdjson(createFfiEngineBackend(LIB), SEED);
    const engines: Array<{
      g: {
        createUniqueConstraint: (l: string, k: string) => void;
        createEdgeUniqueConstraint: (t: string, k: string) => void;
        createIndex: (spec: {
          on: 'vertex' | 'edge';
          kind: 'hash' | 'interval';
          keys: string[];
        }) => void;
        dropVertexIndex: (k: string) => void;
        dropEdgeIndex: (k: string) => void;
      };
      run: (sql: string) => unknown;
    }> = [
      { g: tsGraph, run: (sql) => tsQuery(tsGraph, sql) },
      { g: nativeGraph, run: (sql) => nativeGraph.query(sql) },
    ];

    for (const { g } of engines) {
      g.createUniqueConstraint('Acct', 'email');
      g.createEdgeUniqueConstraint('LINK', 'id');
      g.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] }); // plain, droppable
    }

    for (const { g } of engines) {
      // Backing a unique constraint → E_INVALID_GRAPH_OP on both.
      expect(outcome(() => g.dropVertexIndex('email'))).toEqual({ code: 'E_INVALID_GRAPH_OP' });
      expect(outcome(() => g.dropEdgeIndex('id'))).toEqual({ code: 'E_INVALID_GRAPH_OP' });
      // A plain index still drops.
      expect(outcome(() => g.dropVertexIndex('age'))).toEqual({ ok: true });
    }

    // Enforcement survived the refused drop — a duplicate still violates, identically.
    const dup = `INSERT (:Acct {email: 'x@x.io'})`;
    const [ts, native] = engines.map(({ run }) =>
      outcome(() => {
        run(dup);
        run(dup);
      }),
    );
    expect(native).toEqual(ts);
    expect(native).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });

  test('_MERGE outcomes and final state agree across engines', () => {
    // Exercises every disposition + WHERE-gate + the no-constraint error, run on
    // both engines, comparing each outcome and the final graph byte-for-byte.
    const script = [
      `_MERGE (u:Acct {email: 'm@x.io', name: 'A'}) _ON_CREATE SET u.created = 1`, // create
      `_MERGE (u:Acct {email: 'm@x.io', name: 'B'})`, // clobber payload
      `_MERGE (u:Acct {email: 'm@x.io', name: 'IGN'}) _ON_UPDATE SET u.name = 'Upd'`, // replace
      `_MERGE (u:Acct {email: 'm@x.io', name: 'IGN2'}) _ON_UPDATE_NOTHING`, // no-op
      `_MERGE (u:Acct {email: 'k@x.io', v: 1})`, // create #2
      `_MERGE (u:Acct {email: 'k@x.io'}) _ON_UPDATE SET u.v = 9 WHERE u.v < 9`, // LWW: applies
      `_MERGE (u:Acct {email: 'k@x.io'}) _ON_UPDATE SET u.v = 2 WHERE u.v < 2`, // LWW: no-op
      `_MERGE (u:Nope {k: 1})`, // no constraint → error (both InvalidGraphOp)
    ];

    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createUniqueConstraint('Acct', 'email');
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createUniqueConstraint('Acct', 'email');

    for (const sql of script) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `_MERGE outcome mismatch: ${sql}`).toEqual(ts);
    }

    const READ = `MATCH (u:Acct) RETURN u.email, u.name, u.v, u.created ORDER BY u.email`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('_MERGE edge form outcomes and final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);

    for (const g of [tsGraph, nativeGraph]) {
      g.createUniqueConstraint('User', 'id');
      g.createUniqueConstraint('Team', 'id');
    }

    const script = [
      `INSERT (:User {id: 'u1'}), (:Team {id: 't1'})`,
      `_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 1}]->(t:Team {id:'t1'}) _ON_CREATE SET m.role = 'admin'`, // create
      `_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 2}]->(t:Team {id:'t1'})`, // clobber edge props
      `_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 9}]->(t:Team {id:'t1'}) _ON_UPDATE_NOTHING`, // no-op
      `_MERGE (u:User {id:'u1'})-[m:MEMBER]->(t:Team {id:'nope'})`, // missing endpoint → error
    ];

    for (const sql of script) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `edge _MERGE mismatch: ${sql}`).toEqual(ts);
    }

    const READ = `MATCH (:User)-[m:MEMBER]->(:Team) RETURN m.since, m.role`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });
});

suite('required-constraint differential (TS vs native)', () => {
  // Same INSERT/SET/REMOVE/label sequence on both engines, comparing every write
  // outcome (same ConstraintViolation, or both succeed) + the final state.
  const SCRIPT: Array<{ label: string; sql: string }> = [
    { label: 'insert with required ok', sql: `INSERT (:Acct {email: 'a@x.io', name: 'A'})` },
    { label: 'insert missing required → violation', sql: `INSERT (:Acct {name: 'B'})` },
    { label: 'insert null required → violation', sql: `INSERT (:Acct {email: null, name: 'C'})` },
    { label: 'different label unaffected', sql: `INSERT (:Other {name: 'O'})` },
    {
      label: 'set required to null → violation',
      sql: `MATCH (n:Acct {email: 'a@x.io'}) SET n.email = null`,
    },
    {
      label: 'set required to a new value ok',
      sql: `MATCH (n:Acct {email: 'a@x.io'}) SET n.email = 'a2@x.io'`,
    },
    {
      label: 'remove required → violation',
      sql: `MATCH (n:Acct {email: 'a2@x.io'}) REMOVE n.email`,
    },
    {
      label: 'set a non-required key ok',
      sql: `MATCH (n:Acct {email: 'a2@x.io'}) SET n.name = 'AA'`,
    },
    // Adding the constrained label to a node missing the key → violation.
    { label: 'seed a Person without email', sql: `INSERT (:Person {name: 'P'})` },
    { label: 'add :Acct to it → violation', sql: `MATCH (p:Person {name: 'P'}) SET p:Acct` },
  ];

  test('every write outcome and the final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createRequiredConstraint('Acct', 'email');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createRequiredConstraint('Acct', 'email');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH (n:Acct) RETURN n.name, n.email ORDER BY n.name`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('declare-time rejection agrees across engines (existing data missing the key)', () => {
    const missing = [
      '{"type":"node","id":"1","labels":["Acct"],"properties":{"email":"a@x.io"}}',
      '{"type":"node","id":"2","labels":["Acct"],"properties":{"name":"no-email"}}',
    ].join('\n');

    const tsGraph = tsDeserialize(missing, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, missing);

    const ts = outcome(() => tsGraph.createRequiredConstraint('Acct', 'email'));
    const native = outcome(() => nativeGraph.createRequiredConstraint('Acct', 'email'));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });
});

suite('type-constraint differential (TS vs native)', () => {
  // A `number` type constraint on Acct.age: the same INSERT/SET sequence on both
  // engines, comparing each write outcome + the final state.
  const SCRIPT: Array<{ label: string; sql: string }> = [
    { label: 'insert right type ok', sql: `INSERT (:Acct {age: 30, name: 'A'})` },
    { label: 'insert wrong type → violation', sql: `INSERT (:Acct {age: 'old', name: 'B'})` },
    { label: 'insert null (exempt) ok', sql: `INSERT (:Acct {age: null, name: 'N'})` },
    { label: 'insert absent (exempt) ok', sql: `INSERT (:Acct {name: 'X'})` },
    { label: 'different label unaffected', sql: `INSERT (:Other {age: 'text'})` },
    {
      label: 'set wrong type → violation',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = 'nope'`,
    },
    {
      label: 'set right type ok',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = 31`,
    },
    {
      label: 'set null (exempt) ok',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = null`,
    },
  ];

  test('every write outcome and the final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createTypeConstraint('Acct', 'age', 'number');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createTypeConstraint('Acct', 'age', 'number');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH (n:Acct) RETURN n.name, n.age ORDER BY n.name`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('declare-time rejection agrees across engines (existing data has wrong type)', () => {
    const wrong = [
      '{"type":"node","id":"1","labels":["Acct"],"properties":{"age":30}}',
      '{"type":"node","id":"2","labels":["Acct"],"properties":{"age":"old"}}',
    ].join('\n');

    const tsGraph = tsDeserialize(wrong, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, wrong);

    const ts = outcome(() => tsGraph.createTypeConstraint('Acct', 'age', 'number'));
    const native = outcome(() => nativeGraph.createTypeConstraint('Acct', 'age', 'number'));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });

  test('unknown scalar type name is rejected (InvalidValue) across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);

    const ts = outcome(() => tsGraph.createTypeConstraint('Acct', 'age', 'int' as never));
    const native = outcome(() => nativeGraph.createTypeConstraint('Acct', 'age', 'int' as never));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_INVALID_VALUE' });
  });
});

// Edge-side constraints: the SAME createEdge*Constraint + INSERT/SET edge script
// on both engines, asserting identical write outcomes and byte-identical final
// edge state. The edge analogue of the vertex differentials above. Edges are
// created inline via `INSERT (:P {..})-[:FOLLOWS {..}]->(:P {..})`.
suite('edge unique-constraint differential (TS vs native)', () => {
  const SCRIPT: Array<{ label: string; sql: string }> = [
    {
      label: 'first FOLLOWS ok',
      sql: `INSERT (:P {id: 'a'})-[:FOLLOWS {tag: 'x'}]->(:P {id: 'b'})`,
    },
    {
      label: 'dup tag → violation',
      sql: `INSERT (:P {id: 'c'})-[:FOLLOWS {tag: 'x'}]->(:P {id: 'd'})`,
    },
    {
      label: 'different tag ok',
      sql: `INSERT (:P {id: 'e'})-[:FOLLOWS {tag: 'y'}]->(:P {id: 'f'})`,
    },
    {
      label: 'different type ok',
      sql: `INSERT (:P {id: 'g'})-[:LIKES {tag: 'x'}]->(:P {id: 'h'})`,
    },
    {
      label: 'null tag #1 (exempt)',
      sql: `INSERT (:P {id: 'i'})-[:FOLLOWS {tag: null}]->(:P {id: 'j'})`,
    },
    {
      label: 'null tag #2 (exempt)',
      sql: `INSERT (:P {id: 'k'})-[:FOLLOWS {tag: null}]->(:P {id: 'l'})`,
    },
    {
      label: 'SET collision → violation',
      sql: `MATCH ()-[r:FOLLOWS {tag: 'y'}]->() SET r.tag = 'x'`,
    },
    { label: 'SET self ok', sql: `MATCH ()-[r:FOLLOWS {tag: 'y'}]->() SET r.tag = 'y'` },
  ];

  test('every write outcome and the final edge state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createEdgeUniqueConstraint('FOLLOWS', 'tag');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createEdgeUniqueConstraint('FOLLOWS', 'tag');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH ()-[r:FOLLOWS]->() RETURN r.tag ORDER BY r.tag`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('declare-time rejection agrees across engines (pre-existing duplicate edges)', () => {
    const build = `INSERT (:P {id: 'a'})-[:FOLLOWS {tag: 'dup'}]->(:P {id: 'b'}), (:P {id: 'c'})-[:FOLLOWS {tag: 'dup'}]->(:P {id: 'd'})`;
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsQuery(tsGraph, build);
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.query(build);

    const ts = outcome(() => tsGraph.createEdgeUniqueConstraint('FOLLOWS', 'tag'));
    const native = outcome(() => nativeGraph.createEdgeUniqueConstraint('FOLLOWS', 'tag'));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });
});

suite('edge required-constraint differential (TS vs native)', () => {
  const SCRIPT: Array<{ label: string; sql: string }> = [
    {
      label: 'insert with required ok',
      sql: `INSERT (:P {id: 'a'})-[:FOLLOWS {since: 1}]->(:P {id: 'b'})`,
    },
    {
      label: 'insert missing required → violation',
      sql: `INSERT (:P {id: 'c'})-[:FOLLOWS]->(:P {id: 'd'})`,
    },
    {
      label: 'insert null required → violation',
      sql: `INSERT (:P {id: 'e'})-[:FOLLOWS {since: null}]->(:P {id: 'f'})`,
    },
    {
      label: 'different type unaffected',
      sql: `INSERT (:P {id: 'g'})-[:LIKES]->(:P {id: 'h'})`,
    },
    {
      label: 'set required to null → violation',
      sql: `MATCH ()-[r:FOLLOWS]->() SET r.since = null`,
    },
    {
      label: 'set required to a new value ok',
      sql: `MATCH ()-[r:FOLLOWS]->() SET r.since = 2`,
    },
    {
      label: 'remove required → violation',
      sql: `MATCH ()-[r:FOLLOWS]->() REMOVE r.since`,
    },
  ];

  test('every write outcome and the final edge state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createEdgeRequiredConstraint('FOLLOWS', 'since');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createEdgeRequiredConstraint('FOLLOWS', 'since');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH ()-[r:FOLLOWS]->() RETURN r.since ORDER BY r.since`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });
});

suite('edge type-constraint differential (TS vs native)', () => {
  const SCRIPT: Array<{ label: string; sql: string }> = [
    {
      label: 'insert right type ok',
      sql: `INSERT (:P {id: 'a'})-[:FOLLOWS {since: 30}]->(:P {id: 'b'})`,
    },
    {
      label: 'insert wrong type → violation',
      sql: `INSERT (:P {id: 'c'})-[:FOLLOWS {since: 'old'}]->(:P {id: 'd'})`,
    },
    {
      label: 'insert null (exempt) ok',
      sql: `INSERT (:P {id: 'e'})-[:FOLLOWS {since: null}]->(:P {id: 'f'})`,
    },
    {
      label: 'set wrong type → violation',
      sql: `MATCH ()-[r:FOLLOWS {since: 30}]->() SET r.since = 'nope'`,
    },
    {
      label: 'set right type ok',
      sql: `MATCH ()-[r:FOLLOWS {since: 30}]->() SET r.since = 31`,
    },
  ];

  test('every write outcome and the final edge state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createEdgeTypeConstraint('FOLLOWS', 'since', 'number');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createEdgeTypeConstraint('FOLLOWS', 'since', 'number');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH ()-[r:FOLLOWS]->() RETURN r.since ORDER BY r.since`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('unknown scalar type name is rejected (InvalidValue) across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);

    const ts = outcome(() => tsGraph.createEdgeTypeConstraint('FOLLOWS', 'since', 'int' as never));
    const native = outcome(() =>
      nativeGraph.createEdgeTypeConstraint('FOLLOWS', 'since', 'int' as never),
    );

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_INVALID_VALUE' });
  });
});

// Cardinality (degree-bound) constraint: an "exactly one" bound on a Purchase's
// PLACED_BY out-degree, run identically on both engines. Because every GQL
// statement auto-commits, BOTH bounds land at the per-statement commit — the
// node+edge INSERT satisfies it, a bare Purchase INSERT trips min, and a second
// PLACED_BY trips max. (`Order` is a reserved GQL keyword, so we use `Purchase`.)
suite('cardinality-constraint differential (TS vs native)', () => {
  const SCRIPT: Array<{ label: string; sql: string }> = [
    {
      label: 'node+edge together satisfies exactly-one',
      sql: `INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})`,
    },
    {
      label: 'bare Purchase (degree 0) violates min',
      sql: `INSERT (:Purchase {id: 'o2'})`,
    },
    {
      label: 'a second PLACED_BY (degree 2) violates max',
      sql: `MATCH (o:Purchase {id: 'o1'}), (c:Customer {id: 'c1'}) INSERT (o)-[:PLACED_BY]->(c)`,
    },
    {
      label: 'a Customer (unconstrained label) is unaffected',
      sql: `INSERT (:Customer {id: 'c9'})`,
    },
    {
      label: 'another satisfying Purchase ok',
      sql: `INSERT (:Purchase {id: 'o3'})-[:PLACED_BY]->(:Customer {id: 'c1'})`,
    },
  ];

  test('every write outcome and the final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsGraph.createCardinalityConstraint('Purchase', 'PLACED_BY', 'out', 1, 1);

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createCardinalityConstraint('Purchase', 'PLACED_BY', 'out', 1, 1);

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    // Byte-identical final state: only the satisfying Purchases + their edges landed.
    const READ = `MATCH (o:Purchase)-[:PLACED_BY]->(c:Customer) RETURN o.id, c.id ORDER BY o.id`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
    const COUNT = `MATCH (o:Purchase) RETURN o.id ORDER BY o.id`;
    expect(JSON.stringify(nativeGraph.query(COUNT))).toEqual(
      JSON.stringify(tsQuery(tsGraph, COUNT)),
    );
  });

  test('declare-time rejection agrees across engines (existing vertex under min)', () => {
    // A Purchase with no PLACED_BY out-edge already violates a min:1 bound.
    const build = `INSERT (:Purchase {id: 'o1'})`;
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsQuery(tsGraph, build);
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.query(build);

    const ts = outcome(() =>
      tsGraph.createCardinalityConstraint('Purchase', 'PLACED_BY', 'out', 1, 1),
    );
    const native = outcome(() =>
      nativeGraph.createCardinalityConstraint('Purchase', 'PLACED_BY', 'out', 1, 1),
    );

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });
});

suite('validator differential (TS vs native)', () => {
  // A custom GQL-predicate validator declared identically on BOTH engines — TS
  // via `@lenke/gql`'s free `createValidator` (which compiles a closure into
  // core), native via `RustGraph.createValidator` (Rust parses + evaluates the
  // same predicate). The same predicate string must accept/reject byte-identically
  // and leave byte-identical state — the whole point of the dual-engine invariant.
  const SCRIPT: Array<{ label: string; sql: string }> = [
    { label: 'insert valid age ok', sql: `INSERT (:Acct {age: 20, name: 'A'})` },
    { label: 'insert negative age → violation', sql: `INSERT (:Acct {age: -5, name: 'B'})` },
    { label: 'insert over-max age → violation', sql: `INSERT (:Acct {age: 200, name: 'C'})` },
    // null / absent age is exempt (SQL-CHECK: UNKNOWN passes).
    { label: 'insert no age (null passes)', sql: `INSERT (:Acct {name: 'D'})` },
    { label: 'insert explicit null age (passes)', sql: `INSERT (:Acct {age: null, name: 'E'})` },
    { label: 'different label unaffected', sql: `INSERT (:Other {age: -99})` },
    {
      label: 'SET age below 0 → violation',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = -1`,
    },
    {
      label: 'SET age to a valid value ok',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = 30`,
    },
    {
      label: 'SET age to null (passes)',
      sql: `MATCH (n:Acct {name: 'A'}) SET n.age = null`,
    },
  ];

  test('every write outcome and the final state agree across engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    tsCreateValidator(tsGraph, 'Acct', 'u', 'u.age >= 0 AND u.age < 150');

    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);
    nativeGraph.createValidator('Acct', 'u', 'u.age >= 0 AND u.age < 150');

    for (const { label, sql } of SCRIPT) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `outcome mismatch: ${label}`).toEqual(ts);
    }

    const READ = `MATCH (n:Acct) RETURN n.name, n.age ORDER BY n.name`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('edge validator: outcomes + final state agree across engines', () => {
    const seed = [
      '{"type":"node","id":"a","labels":["P"],"properties":{}}',
      '{"type":"node","id":"b","labels":["P"],"properties":{}}',
    ].join('\n');

    const tsGraph = tsDeserialize(seed, 'ndjson', new Graph());
    tsCreateValidator(tsGraph, 'KNOWS', 'r', 'r.weight >= 0');
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, seed);
    nativeGraph.createValidator('KNOWS', 'r', 'r.weight >= 0');

    const script = [
      `MATCH (a:P), (b:P) WHERE a <> b INSERT (a)-[:KNOWS {weight: -1}]->(b)`, // violation
      `MATCH (a:P), (b:P) WHERE a <> b INSERT (a)-[:KNOWS {weight: 5}]->(b)`, // ok
      `MATCH (a:P), (b:P) WHERE a <> b INSERT (a)-[:KNOWS]->(b)`, // null weight passes
    ];

    for (const sql of script) {
      const ts = outcome(() => tsQuery(tsGraph, sql));
      const native = outcome(() => nativeGraph.query(sql));

      expect(native, `edge validator mismatch: ${sql}`).toEqual(ts);
    }

    const READ = `MATCH ()-[r:KNOWS]->() RETURN r.weight ORDER BY r.weight`;
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('declare-time rejection agrees across engines (existing violating data)', () => {
    const dup = [
      '{"type":"node","id":"1","labels":["Acct"],"properties":{"age":-5}}',
      '{"type":"node","id":"2","labels":["Acct"],"properties":{"age":30}}',
    ].join('\n');

    const tsGraph = tsDeserialize(dup, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, dup);

    const ts = outcome(() => tsCreateValidator(tsGraph, 'Acct', 'u', 'u.age >= 0'));
    const native = outcome(() => nativeGraph.createValidator('Acct', 'u', 'u.age >= 0'));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });

  test('an unparseable predicate is E_SYNTAX on both engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);

    const ts = outcome(() => tsCreateValidator(tsGraph, 'Acct', 'u', 'u.age >>>'));
    const native = outcome(() => nativeGraph.createValidator('Acct', 'u', 'u.age >>>'));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_SYNTAX' });
  });

  test('a predicate referencing the wrong variable is E_SYNTAX on both engines', () => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, SEED);

    // Predicate binds `u`, but references `x` — unbound → the validator would
    // silently never fire. Both engines reject it at declare time (E_SYNTAX).
    const wrong = 'x.age >= 0';
    const tsWrong = outcome(() => tsCreateValidator(tsGraph, 'Acct', 'u', wrong));
    const nativeWrong = outcome(() => nativeGraph.createValidator('Acct', 'u', wrong));
    expect(nativeWrong).toEqual(tsWrong);
    expect(tsWrong).toEqual({ code: 'E_SYNTAX' });

    // A correct-var predicate and a constant predicate are both accepted on both.
    const tsGood = outcome(() => tsCreateValidator(tsGraph, 'Acct', 'u', 'u.age >= 0'));
    const nativeGood = outcome(() => nativeGraph.createValidator('Acct', 'u', 'u.age >= 0'));
    expect(nativeGood).toEqual(tsGood);
    expect(tsGood).toEqual({ ok: true });

    const tsConst = outcome(() => tsCreateValidator(tsGraph, 'Acct', 'u', '1 = 1'));
    const nativeConst = outcome(() => nativeGraph.createValidator('Acct', 'u', '1 = 1'));
    expect(nativeConst).toEqual(tsConst);
    expect(tsConst).toEqual({ ok: true });
  });
});

suite('graph-level invariant differential (TS vs native)', () => {
  // A whole-graph GQL assertion (`sum(balance) = 0`) declared identically on BOTH
  // engines — TS via `@lenke/gql`'s free `createInvariant` (which compiles a query
  // closure into core), native via `RustGraph.createInvariant` (Rust parses +
  // evaluates the same query). The invariant runs once per write commit against
  // the fully-staged graph; `false`-only-fails. A balanced multi-statement
  // transaction commits on both; an unbalanced one rolls back on both — with
  // byte-identical final state.
  const LEDGER = [
    '{"type":"node","id":"a","labels":["Acct"],"properties":{"name":"a","balance":100}}',
    '{"type":"node","id":"b","labels":["Acct"],"properties":{"name":"b","balance":-100}}',
  ].join('\n');

  const INV = `MATCH (a:Acct) RETURN sum(a.balance) = 0`;
  const READ = `MATCH (a:Acct) RETURN a.name, a.balance ORDER BY a.name`;

  test('balanced transaction commits, unbalanced rolls back — outcomes + state agree', () => {
    const tsGraph = tsDeserialize(LEDGER, 'ndjson', new Graph());
    tsCreateInvariant(tsGraph, 'balanced', INV);
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, LEDGER);
    nativeGraph.createInvariant('balanced', INV);

    // A balance-preserving transfer (sum stays 0) commits on both engines.
    const balancedTs = outcome(() =>
      tsGraph.transaction(() => {
        tsQuery(tsGraph, `MATCH (a:Acct {name: 'a'}) SET a.balance = 70`);
        tsQuery(tsGraph, `MATCH (b:Acct {name: 'b'}) SET b.balance = -70`);
      }),
    );
    const balancedNative = outcome(() =>
      nativeGraph.transaction(() => {
        nativeGraph.query(`MATCH (a:Acct {name: 'a'}) SET a.balance = 70`);
        nativeGraph.query(`MATCH (b:Acct {name: 'b'}) SET b.balance = -70`);
      }),
    );
    expect(balancedNative, 'balanced-commit outcome mismatch').toEqual(balancedTs);
    expect(balancedTs).toEqual({ ok: true });

    // An unbalanced half-transfer (sum ≠ 0) rolls the whole transaction back on both.
    const unbalancedTs = outcome(() =>
      tsGraph.transaction(() => {
        tsQuery(tsGraph, `MATCH (a:Acct {name: 'a'}) SET a.balance = 999`);
      }),
    );
    const unbalancedNative = outcome(() =>
      nativeGraph.transaction(() => {
        nativeGraph.query(`MATCH (a:Acct {name: 'a'}) SET a.balance = 999`);
      }),
    );
    expect(unbalancedNative, 'unbalanced-rollback outcome mismatch').toEqual(unbalancedTs);
    expect(unbalancedTs).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });

    // Final state is byte-identical: the balanced commit landed (70 / -70), the
    // unbalanced transaction left no trace.
    expect(JSON.stringify(nativeGraph.query(READ))).toEqual(JSON.stringify(tsQuery(tsGraph, READ)));
  });

  test('declare-time rejection on an already-unbalanced graph agrees across engines', () => {
    const skewed = [
      '{"type":"node","id":"a","labels":["Acct"],"properties":{"name":"a","balance":100}}',
      '{"type":"node","id":"b","labels":["Acct"],"properties":{"name":"b","balance":-50}}',
    ].join('\n');

    const tsGraph = tsDeserialize(skewed, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, skewed);

    const ts = outcome(() => tsCreateInvariant(tsGraph, 'balanced', INV));
    const native = outcome(() => nativeGraph.createInvariant('balanced', INV));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_CONSTRAINT_VIOLATION' });
  });

  test('an unparseable invariant query is E_SYNTAX on both engines', () => {
    const tsGraph = tsDeserialize(LEDGER, 'ndjson', new Graph());
    const backend = createFfiEngineBackend(LIB);
    const nativeGraph = graphFromNdjson(backend, LEDGER);

    const ts = outcome(() => tsCreateInvariant(tsGraph, 'bad', `MATCH (a:Acct) RETURN >>>`));
    const native = outcome(() => nativeGraph.createInvariant('bad', `MATCH (a:Acct) RETURN >>>`));

    expect(native).toEqual(ts);
    expect(ts).toEqual({ code: 'E_SYNTAX' });
  });
});

// A temporal value ANYWHERE in a numeric aggregate makes it
// unrepresentable — a heterogeneous numeric+temporal column must throw, not
// silently coerce the temporal to null (native faulted per-value; TS only checked
// the first row). Both engines now throw E_INVALID_VALUE identically.
suite('temporal-in-aggregate differential (TS vs native)', () => {
  const MIXED = [
    '{"type":"node","id":"a","labels":["P"],"properties":{"k":5}}',
    '{"type":"node","id":"b","labels":["P"],"properties":{"k":{"@date":"2020-01-01"}}}',
  ].join('\n');

  for (const agg of ['sum', 'avg']) {
    test(`${agg}() over a mixed numeric+temporal column throws on both engines`, () => {
      const tsGraph = tsDeserialize(MIXED, 'ndjson', new Graph());
      const nativeGraph = graphFromNdjson(createFfiEngineBackend(LIB), MIXED);
      const q = `MATCH (n:P) RETURN ${agg}(n.k) AS v`;

      const ts = outcome(() => tsQuery(tsGraph, q));
      const native = outcome(() => nativeGraph.query(q));

      expect(native).toEqual(ts);
      expect(native).toEqual({ code: 'E_INVALID_VALUE' });
    });
  }
});

// Aggregates over LIST-valued columns. Prior art (SQL/Postgres/DuckDB/
// Cypher/ISO): sum/avg over an array is a type error; min/max order arrays
// element-wise. lenke's GQL now matches — sum/avg throw, min/max use the total
// order recursively (both engines). (Gremlin keeps its own TinkerPop semantics —
// tested in gremlin-conformance, not here.)
suite('list-in-aggregate differential (TS vs native)', () => {
  const seed = (rows: readonly string[]) => {
    const tsGraph = tsDeserialize(SEED, 'ndjson', new Graph());
    const nativeGraph = graphFromNdjson(createFfiEngineBackend(LIB), SEED);

    for (const s of rows) {
      tsQuery(tsGraph, s);
      nativeGraph.query(s);
    }

    return { tsGraph, nativeGraph };
  };

  for (const agg of ['sum', 'avg']) {
    test(`${agg}() over a list column throws E_INVALID_VALUE on both engines`, () => {
      const { tsGraph, nativeGraph } = seed([`INSERT (:P {k: [3]})`, `INSERT (:P {k: [1]})`]);
      const q = `MATCH (n:P) RETURN ${agg}(n.k) AS v`;

      const ts = outcome(() => tsQuery(tsGraph, q));
      const native = outcome(() => nativeGraph.query(q));

      expect(native).toEqual(ts);
      expect(native).toEqual({ code: 'E_INVALID_VALUE' });
    });
  }

  for (const agg of ['min', 'max']) {
    test(`${agg}() over lists compares element-wise (not string-coerced), byte-identical`, () => {
      // [10] vs [9] is the trap: element-wise 10 > 9, but string coercion gives
      // "10" < "9". Plus a length tie-break case.
      const { tsGraph, nativeGraph } = seed([
        `INSERT (:P {k: [10]})`,
        `INSERT (:P {k: [9]})`,
        `INSERT (:P {k: [10, 0]})`,
      ]);
      const q = `MATCH (n:P) RETURN ${agg}(n.k) AS v`;

      expect(JSON.stringify(nativeGraph.query(q))).toEqual(JSON.stringify(tsQuery(tsGraph, q)));
    });
  }
});
