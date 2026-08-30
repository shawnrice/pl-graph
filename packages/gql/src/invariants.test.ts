import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { createInvariant, query } from './index.js';

/** Run `fn`, expecting a thrown coded error; return its `.code`. */
const codeOf = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return (e as { code?: unknown }).code;
  }

  throw new Error('expected a throw, got a normal return');
};

/** A two-account double-entry ledger whose balances sum to zero. */
const ledger = (): Graph => {
  const g = new Graph();
  g.addVertex({ labels: ['Acct'], properties: { name: 'a', balance: 100 } });
  g.addVertex({ labels: ['Acct'], properties: { name: 'b', balance: -100 } });

  return g;
};

const sum = (g: Graph): number =>
  query<{ s: number }>(g, `MATCH (a:Acct) RETURN sum(a.balance) AS s`)[0]?.s ?? NaN;

// A graph-level INVARIANT is a whole-graph GQL assertion query that must hold
// after every write transaction. Unlike a per-element validator, it runs ONCE per
// commit against the fully-staged graph. `false`-only-fails: VIOLATED iff any
// result cell is boolean `false` (true/null/non-boolean/empty all hold).
// Byte-identical to the Rust engine — see the differential in
// packages/native/src/constraint-conformance.test.ts.
describe('GQL: graph-level invariants', () => {
  test('debits==credits: a balance-preserving transfer commits', () => {
    const g = ledger();
    createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`);

    // Move 30 from a→b in one atomic transaction; the sum stays 0 → commits.
    g.transaction(() => {
      query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 70`);
      query(g, `MATCH (b:Acct {name: 'b'}) SET b.balance = -70`);
    });

    expect(sum(g)).toBe(0);
    expect(query(g, `MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b`)).toEqual([{ b: 70 }]);
  });

  test('an unbalanced transaction rolls back with E_CONSTRAINT_VIOLATION', () => {
    const g = ledger();
    createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`);

    // Only one side of the transfer → the sum is no longer 0 → violated.
    expect(
      codeOf(() =>
        g.transaction(() => {
          query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 999`);
        }),
      ),
    ).toBe(ErrorCode.ConstraintViolation);

    // The whole transaction rolled back — the balance is untouched.
    expect(sum(g)).toBe(0);
    expect(query(g, `MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b`)).toEqual([{ b: 100 }]);
  });

  test('a single auto-committing unbalanced write is rejected + rolled back', () => {
    const g = ledger();
    createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`);

    // Every GQL write statement auto-commits, so a lone unbalanced SET trips the
    // invariant at its own commit boundary (no explicit transaction needed).
    expect(codeOf(() => query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 5`))).toBe(
      ErrorCode.ConstraintViolation,
    );
    expect(sum(g)).toBe(0);
  });

  test('a bare transaction(fn) that throws rolls back before the invariant runs', () => {
    const g = ledger();
    createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`);

    // The fn writes an unbalanced state, then throws — the throw rolls back first,
    // so the transaction never commits and the invariant never fires. Either way
    // the state is restored.
    expect(
      codeOf(() =>
        g.transaction(() => {
          query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 40`);

          throw new Error('boom');
        }),
      ),
    ).toBeUndefined(); // a plain Error has no `.code`

    expect(sum(g)).toBe(0);
    expect(query(g, `MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b`)).toEqual([{ b: 100 }]);
  });

  test('declare-time rejection when the graph already violates', () => {
    const g = new Graph();
    g.addVertex({ labels: ['Acct'], properties: { name: 'a', balance: 100 } });
    g.addVertex({ labels: ['Acct'], properties: { name: 'b', balance: -50 } }); // sum = 50

    expect(
      codeOf(() => createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`)),
    ).toBe(ErrorCode.ConstraintViolation);
    // The rejected declaration registered nothing.
    expect(g.invariants()).toEqual([]);
  });

  test('a count invariant: at least one Admin must remain', () => {
    const g = new Graph();
    g.addVertex({ labels: ['User'], properties: { name: 'u1', role: 'Admin' } });
    g.addVertex({ labels: ['User'], properties: { name: 'u2', role: 'Member' } });
    createInvariant(g, 'has_admin', `MATCH (u:User) WHERE u.role = 'Admin' RETURN count(u) > 0`);

    // Demote the member → one admin remains → holds.
    query(g, `MATCH (u:User {name: 'u2'}) SET u.role = 'Guest'`);
    // Demote the last admin → count drops to 0 → `0 > 0` is false → violated.
    expect(codeOf(() => query(g, `MATCH (u:User {name: 'u1'}) SET u.role = 'Guest'`))).toBe(
      ErrorCode.ConstraintViolation,
    );
    expect(query(g, `MATCH (u:User {role: 'Admin'}) RETURN count(u) AS n`)).toEqual([{ n: 1 }]);
  });

  test('drop + introspection', () => {
    const g = ledger();
    createInvariant(g, 'balanced', `MATCH (a:Acct) RETURN sum(a.balance) = 0`);
    createInvariant(g, 'has_acct', `MATCH (a:Acct) RETURN count(a) >= 0`);

    expect(g.invariants()).toEqual([
      { name: 'balanced', src: `MATCH (a:Acct) RETURN sum(a.balance) = 0` },
      { name: 'has_acct', src: `MATCH (a:Acct) RETURN count(a) >= 0` },
    ]);

    g.dropInvariant('balanced');
    expect(g.invariants()).toEqual([
      { name: 'has_acct', src: `MATCH (a:Acct) RETURN count(a) >= 0` },
    ]);

    // Dropped → a previously-rejected unbalanced write now succeeds.
    query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 5`);
    expect(sum(g)).toBe(-95);
  });

  test('an unparseable query throws E_SYNTAX at declare time', () => {
    const g = new Graph();
    expect(codeOf(() => createInvariant(g, 'bad', `MATCH (a:Acct) RETURN >>>`))).toBe(
      ErrorCode.Syntax,
    );
    expect(codeOf(() => createInvariant(g, 'empty', ``))).toBe(ErrorCode.Syntax);
    // The rejected declarations registered nothing.
    expect(g.invariants()).toEqual([]);
  });

  test('non-boolean / null / empty result sets all HOLD (false-only-fails)', () => {
    const g = ledger();
    // A number cell (the raw sum), a null cell, and an empty result set — none is
    // a boolean `false`, so all three hold and a write commits regardless.
    createInvariant(g, 'nonbool', `MATCH (a:Acct) RETURN sum(a.balance)`);
    createInvariant(g, 'nullcell', `MATCH (a:Acct) RETURN a.missing`);
    createInvariant(g, 'empty', `MATCH (z:NoSuchLabel) RETURN z.x = z.x`);

    query(g, `MATCH (a:Acct {name: 'a'}) SET a.balance = 12345`);
    expect(query(g, `MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b`)).toEqual([{ b: 12345 }]);
  });

  test('a pure-read transaction does NOT run the invariant (gated on writes)', () => {
    const g = ledger();
    // Register a spy invariant directly through core so we can count evaluations.
    let calls = 0;
    g.registerInvariant('spy', 'spy', (gr) => {
      calls += 1;
      // Read the graph so the closure is a genuine query-shaped read, then hold.
      void gr.getVerticesByLabel('Acct');

      return [];
    });
    calls = 0; // registerInvariant runs the check once at declare time — ignore it.

    // A transaction that only reads writes nothing → the invariant must not run.
    g.transaction(() => {
      query(g, `MATCH (a:Acct) RETURN a.balance`);
    });
    expect(calls).toBe(0);

    // A transaction that writes something runs the invariant exactly once.
    g.transaction(() => {
      query(g, `INSERT (:Temp {k: 1})`);
    });
    expect(calls).toBe(1);
  });
});
