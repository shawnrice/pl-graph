import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { createValidator, query } from './index.js';

/** Run `fn`, expecting a thrown coded error; return its `.code`. */
const codeOf = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return (e as { code?: unknown }).code;
  }

  throw new Error('expected a throw, got a normal return');
};

// A declarative validator attaches a pure ISO GQL boolean predicate (WHERE-clause
// syntax) to a label; every element carrying the label must satisfy it at the
// mutation boundary. SQL-`CHECK` semantics: a write fails only on a *definite*
// `false`; a null/unknown result passes. Byte-identical to the Rust engine — see
// the differential in packages/native/src/constraint-conformance.test.ts.
describe('GQL: declarative validators', () => {
  test('declare-time scan rejects already-violating data', () => {
    const g = new Graph();
    g.addVertex({ labels: ['User'], properties: { age: -5 } });

    expect(codeOf(() => createValidator(g, 'User', 'u', 'u.age >= 0 AND u.age < 150'))).toBe(
      ErrorCode.ConstraintViolation,
    );

    // The rejected declaration registered nothing — a subsequent bad write is fine.
    expect(g.validators()).toEqual([]);
  });

  test('per-write rejection and acceptance on INSERT', () => {
    const g = new Graph();
    createValidator(g, 'User', 'u', 'u.age >= 0 AND u.age < 150');

    expect(codeOf(() => query(g, `INSERT (:User {age: -5})`))).toBe(ErrorCode.ConstraintViolation);
    // A rejected write leaves no trace.
    expect(g.vertexCount).toBe(0);

    query(g, `INSERT (:User {age: 20})`);
    expect(g.vertexCount).toBe(1);
  });

  test('null / unknown result passes (absent optional property)', () => {
    const g = new Graph();
    createValidator(g, 'User', 'u', 'u.age >= 0');

    // No `age` at all → `u.age` is null → `null >= 0` is UNKNOWN → passes.
    query(g, `INSERT (:User {name: 'Ada'})`);
    // An explicit null value is likewise exempt.
    query(g, `INSERT (:User {name: 'Bo', age: null})`);

    expect(g.vertexCount).toBe(2);
  });

  test('deferred inside a transaction: briefly-invalid-then-fixed commits', () => {
    const g = new Graph();
    createValidator(g, 'User', 'u', 'u.age >= 0');

    // Momentarily invalid (age -5), fixed to 5 before commit → the final state
    // satisfies the validator, so the transaction commits.
    g.transaction((tx) => {
      const v = tx.addVertex({ labels: ['User'], properties: { age: -5 } });
      v.setProperty('age', 5);
    });

    expect(g.vertexCount).toBe(1);
    expect([...g.vertices][0].properties.age).toBe(5);
  });

  test('deferred inside a transaction: left-invalid rolls the whole batch back', () => {
    const g = new Graph();
    createValidator(g, 'User', 'u', 'u.age >= 0');
    query(g, `INSERT (:User {age: 1})`); // a pre-existing good row

    expect(
      codeOf(() =>
        g.transaction((tx) => {
          tx.addVertex({ labels: ['User'], properties: { age: 7 } }); // fine on its own
          tx.addVertex({ labels: ['User'], properties: { age: -1 } }); // violates
        }),
      ),
    ).toBe(ErrorCode.ConstraintViolation);

    // Atomic rollback: neither staged row survives; only the pre-existing one.
    expect(g.vertexCount).toBe(1);
    expect([...g.vertices][0].properties.age).toBe(1);
  });

  test('edge validator enforced at the insert gate', () => {
    const g = new Graph();
    createValidator(g, 'KNOWS', 'r', 'r.weight >= 0');

    const a = g.addVertex({ labels: ['P'], properties: {} });
    const b = g.addVertex({ labels: ['P'], properties: {} });

    expect(
      codeOf(() => g.addEdge({ from: a, to: b, labels: ['KNOWS'], properties: { weight: -1 } })),
    ).toBe(ErrorCode.ConstraintViolation);
    expect(g.edgeCount).toBe(0);

    g.addEdge({ from: a, to: b, labels: ['KNOWS'], properties: { weight: 5 } });
    expect(g.edgeCount).toBe(1);

    // An edge with no `weight` passes (null → UNKNOWN).
    g.addEdge({ from: a, to: b, labels: ['KNOWS'], properties: {} });
    expect(g.edgeCount).toBe(2);
  });

  test('drop + introspection', () => {
    const g = new Graph();
    createValidator(g, 'User', 'u', 'u.age >= 0');
    createValidator(g, 'User', 'u', 'u.age < 150');

    expect(g.validators()).toEqual([
      { label: 'User', varName: 'u', src: 'u.age < 150' },
      { label: 'User', varName: 'u', src: 'u.age >= 0' },
    ]);

    g.dropValidator('User');
    expect(g.validators()).toEqual([]);

    // No validator left → a previously-rejected write now succeeds.
    query(g, `INSERT (:User {age: -5})`);
    expect(g.vertexCount).toBe(1);
  });

  test('an unparseable predicate throws E_SYNTAX at declaration time', () => {
    const g = new Graph();

    expect(codeOf(() => createValidator(g, 'User', 'u', 'u.age >>>'))).toBe(ErrorCode.Syntax);
    expect(codeOf(() => createValidator(g, 'User', 'u', ''))).toBe(ErrorCode.Syntax);
    // A predicate that smuggles in an extra clause is rejected too.
    expect(codeOf(() => createValidator(g, 'User', 'u', 'true RETURN 1'))).toBe(ErrorCode.Syntax);
  });

  test('a predicate referencing the wrong variable is rejected at declare time', () => {
    const g = new Graph();
    // Predicate references `x`, but the element binds to `u` — `x.age` would be
    // unbound → the predicate reads UNKNOWN → the SQL-CHECK never fires and the
    // validator silently does nothing. Reject it at DECLARE time (E_SYNTAX),
    // naming the offending variable. Both engines agree (see the differential).
    const code = codeOf(() => createValidator(g, 'User', 'u', 'x.age >= 0'));
    expect(code).toBe(ErrorCode.Syntax);

    // A bare unbound name (no dotted property) is rejected too.
    expect(codeOf(() => createValidator(g, 'User', 'u', 'age >= 0'))).toBe(ErrorCode.Syntax);

    // The rejected declaration registered nothing.
    expect(g.validators()).toEqual([]);
  });

  test('a correct-var predicate and a constant predicate are both accepted', () => {
    const g = new Graph();
    // The declared variable is fine…
    createValidator(g, 'User', 'u', 'u.age >= 0');
    // …and a predicate that references NO variable at all (a constant) is legit.
    createValidator(g, 'User', 'u', '1 = 1');
    expect(g.validators()).toHaveLength(2);

    // The correct-var validator still enforces on write.
    expect(codeOf(() => query(g, `INSERT (:User {age: -5})`))).toBe(ErrorCode.ConstraintViolation);
    query(g, `INSERT (:User {age: 5})`);
    expect(g.vertexCount).toBe(1);
  });

  test('an unknown function surfaces at write time as E_UNKNOWN_FUNCTION', () => {
    const g = new Graph();
    // No existing data, so the declare-time scan never evaluates the predicate —
    // the unknown function is only discovered when the first write evaluates it.
    createValidator(g, 'User', 'u', 'no_such_fn(u.age) >= 0');

    expect(codeOf(() => query(g, `INSERT (:User {age: 1})`))).toBe(ErrorCode.UnknownFunction);
  });
});
