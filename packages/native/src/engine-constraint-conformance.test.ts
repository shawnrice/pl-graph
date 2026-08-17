// Engine-vs-core constraint differential: the SAME declare/write sequence runs on
// BOTH native backends — lenke-core (the shipped row engine) and the standalone
// lenke-engine — through the shared `Backend` contract, asserting they agree on
// every outcome (both succeed, or both throw the same `E_*` code). This is the
// drop-in tripwire: a constraint the engine accepts that core rejects (or vice
// versa), or a mismatched error code, fails here.
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

const EMPTY = new Uint8Array(0);

// An outcome is `ok` or the thrown `E_*` wire code — the two things that must match.
type Outcome = string;
const run = (fn: () => void): Outcome => {
  try {
    fn();

    return 'ok';
  } catch (e) {
    return (e as { code?: string }).code ?? 'THREW';
  }
};

type Step = (be: Backend, g: GraphHandle) => void;
// `seed` is GQL run before the compared steps (format-agnostic, unlike NDJSON,
// whose dialects differ between the two engines).
type Scenario = { name: string; seed: string[]; steps: Step[] };

const SCENARIOS: Scenario[] = [
  {
    name: 'vertex type constraint: declare, conforming + violating writes, unknown name',
    seed: ["INSERT (:P {name: 'a', age: 30})", "INSERT (:P {name: 'b', age: 25})"],
    steps: [
      (be, g) => be.createTypeConstraint(g, 'P', 'age', 'number'),
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'c', age: 40})"),
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'd', age: 'old'})"), // wrong type
      (be, g) => be.createTypeConstraint(g, 'P', 'x', 'not-a-type'), // unknown name
    ],
  },
  {
    name: 'type constraint the existing data already breaks is rejected',
    seed: ["INSERT (:P {name: 'a', age: 'thirty'})"],
    steps: [(be, g) => be.createTypeConstraint(g, 'P', 'age', 'number')],
  },
  // NOTE: closed-record type-constraint parity is covered by the engine's store
  // unit tests, not here — the engine's GQL `INSERT` can't yet write a map literal
  // as a property value (`{m: {a:1}}` → "values must be literals"), which core
  // accepts, so a record can't be seeded identically through both backends.
  {
    name: 'unique constraint: declare then a duplicate write violates',
    seed: ["INSERT (:P {name: 'a', age: 30})", "INSERT (:P {name: 'b', age: 25})"],
    steps: [
      (be, g) => be.createUniqueConstraint(g, 'P', 'name'),
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'a', age: 9})"), // dup name
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'z', age: 9})"), // distinct → ok
    ],
  },
  {
    name: 'validator: declare, then violating + conforming writes',
    seed: ["INSERT (:P {name: 'a', age: 30})"],
    steps: [
      (be, g) => be.createValidator(g, 'P', 'p', 'p.age >= 0'),
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'x', age: -1})"), // violation
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'y', age: 5})"), // ok
    ],
  },
  {
    name: 'validator the existing data already breaks is rejected',
    seed: ["INSERT (:P {name: 'a', age: -5})"],
    steps: [(be, g) => be.createValidator(g, 'P', 'p', 'p.age >= 0')],
  },
  {
    name: 'invariant: declare, then a violating write',
    seed: ["INSERT (:P {name: 'a', age: 30})"],
    steps: [
      (be, g) => be.createInvariant(g, 'nonneg', 'MATCH (p:P) RETURN p.age >= 0'),
      (be, g) => be.queryRows(g, "INSERT (:P {name: 'x', age: -1})"),
    ],
  },
  {
    name: 'dropVertexIndex is rejected while it backs a unique constraint',
    seed: ["INSERT (:P {name: 'a', age: 30})", "INSERT (:P {name: 'b', age: 25})"],
    steps: [
      (be, g) => be.createIndex(g, 'vertex', 'hash', ['name']),
      (be, g) => be.createUniqueConstraint(g, 'P', 'name'),
      (be, g) => be.dropVertexIndex(g, 'name'), // backs the unique → rejected
    ],
  },
];

suite('constraint conformance (engine vs core)', () => {
  const core = createFfiBackend(CORE);
  const engine = createFfiEngineBackend(ENGINE);

  const outcomes = (be: Backend, s: Scenario): Outcome[] => {
    const g = be.graphFromNdjson(EMPTY, false);

    for (const stmt of s.seed) {
      be.queryRows(g, stmt);
    }

    const out = s.steps.map((step) => run(() => step(be, g)));
    be.graphFree(g);

    return out;
  };

  for (const scenario of SCENARIOS) {
    test(scenario.name, () => {
      const coreOut = outcomes(core, scenario);
      const engineOut = outcomes(engine, scenario);
      expect(engineOut).toEqual(coreOut);
    });
  }
});
