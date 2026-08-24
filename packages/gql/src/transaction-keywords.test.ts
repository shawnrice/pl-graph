import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { isTxControl, parseQuery, query } from './index.js';

/** Run `fn`, expecting a thrown coded error; return its `.code`. */
const codeOf = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return (e as { code?: unknown }).code;
  }

  throw new Error('expected a throw, got a normal return');
};

const ids = (g: Graph): string[] =>
  query<{ id: string }>(g, `MATCH (n:Acct) RETURN n.id AS id ORDER BY n.id`).map((r) => r.id);

// Byte-identical with the Rust core's transaction-keyword tests
// (crates/lenke-engine/src/gql/tests.rs) and the cross-engine differential
// (packages/native/src/transaction-conformance.test.ts).
describe('GQL: ISO transaction keywords (START TRANSACTION / COMMIT / ROLLBACK)', () => {
  test('START … INSERT … INSERT … COMMIT persists all writes', () => {
    const g = new Graph();

    expect(query(g, `START TRANSACTION`)).toEqual([]);
    expect(g.isTransacting()).toBe(true);
    query(g, `INSERT (:Acct {id: 'a'})`);
    query(g, `INSERT (:Acct {id: 'b'})`);
    // Read-your-writes: the pending rows are visible inside the transaction.
    expect(ids(g)).toEqual(['a', 'b']);
    expect(query(g, `COMMIT`)).toEqual([]);

    expect(g.isTransacting()).toBe(false);
    expect(ids(g)).toEqual(['a', 'b']);
  });

  test('START … INSERT … ROLLBACK discards all writes', () => {
    const g = new Graph();
    query(g, `INSERT (:Acct {id: 'seed'})`);

    query(g, `START TRANSACTION`);
    query(g, `INSERT (:Acct {id: 'a'})`);
    query(g, `INSERT (:Acct {id: 'b'})`);
    expect(ids(g)).toEqual(['a', 'b', 'seed']);
    query(g, `ROLLBACK`);

    expect(g.isTransacting()).toBe(false);
    expect(ids(g)).toEqual(['seed']);
  });

  test('COMMIT WORK / ROLLBACK WORK parse and behave like the bare forms', () => {
    const g = new Graph();

    query(g, `START TRANSACTION`);
    query(g, `INSERT (:Acct {id: 'a'})`);
    query(g, `COMMIT WORK`);
    expect(ids(g)).toEqual(['a']);

    query(g, `START TRANSACTION`);
    query(g, `INSERT (:Acct {id: 'b'})`);
    query(g, `ROLLBACK WORK`);
    expect(ids(g)).toEqual(['a']);
  });

  test('a deferred required constraint holds across statements: commits when valid', () => {
    const g = new Graph();
    g.createRequiredConstraint('Acct', 'email');

    // The intermediate state (an Acct with no email) is allowed *inside* the
    // transaction; the required check runs only at COMMIT, once the email is set.
    query(g, `START TRANSACTION`);
    query(g, `INSERT (:Acct {id: 'a'})`);
    query(g, `MATCH (n:Acct {id: 'a'}) SET n.email = 'a@x.io'`);
    query(g, `COMMIT`);

    expect(query<{ e: string }>(g, `MATCH (n:Acct) RETURN n.email AS e`)).toEqual([
      { e: 'a@x.io' },
    ]);
  });

  test('a deferred required constraint that never becomes valid rolls the whole transaction back', () => {
    const g = new Graph();
    g.createRequiredConstraint('Acct', 'email');

    query(g, `START TRANSACTION`);
    query(g, `INSERT (:Acct {id: 'a', email: 'a@x.io'})`);
    query(g, `INSERT (:Acct {id: 'b'})`); // never gets an email

    // The deferred required check fires at COMMIT and rolls everything back.
    expect(codeOf(() => query(g, `COMMIT`))).toBe(ErrorCode.ConstraintViolation);
    expect(g.isTransacting()).toBe(false);
    expect(ids(g)).toEqual([]);
  });

  test('nested START TRANSACTION is a coded error (ISO forbids nesting)', () => {
    const g = new Graph();
    query(g, `START TRANSACTION`);

    expect(codeOf(() => query(g, `START TRANSACTION`))).toBe(ErrorCode.InvalidGraphOp);
    // The original transaction is untouched — still open.
    expect(g.isTransacting()).toBe(true);
    query(g, `ROLLBACK`);
  });

  test('COMMIT / ROLLBACK with no active transaction is a coded error', () => {
    const g = new Graph();

    expect(codeOf(() => query(g, `COMMIT`))).toBe(ErrorCode.InvalidGraphOp);
    expect(codeOf(() => query(g, `ROLLBACK`))).toBe(ErrorCode.InvalidGraphOp);
  });

  test('READ ONLY: a write statement is rejected, a read is allowed', () => {
    const g = new Graph();
    query(g, `INSERT (:Acct {id: 'seed'})`);

    query(g, `START TRANSACTION READ ONLY`);

    // A read is fine.
    expect(ids(g)).toEqual(['seed']);
    // Every write shape is rejected before it applies.
    expect(codeOf(() => query(g, `INSERT (:Acct {id: 'x'})`))).toBe(ErrorCode.InvalidGraphOp);
    expect(codeOf(() => query(g, `MATCH (n:Acct) SET n.touched = true`))).toBe(
      ErrorCode.InvalidGraphOp,
    );
    expect(codeOf(() => query(g, `MATCH (n:Acct {id: 'seed'}) DELETE n`))).toBe(
      ErrorCode.InvalidGraphOp,
    );

    query(g, `COMMIT`);

    // After commit the read-only mode is cleared — writes work again.
    query(g, `INSERT (:Acct {id: 'x'})`);
    expect(ids(g)).toEqual(['seed', 'x']);
  });

  test('READ WRITE (explicit) allows writes; access mode clears on rollback', () => {
    const g = new Graph();

    query(g, `START TRANSACTION READ WRITE`);
    query(g, `INSERT (:Acct {id: 'a'})`);
    query(g, `ROLLBACK`);

    // Read-only was never set; and after rollback a fresh write is fine.
    query(g, `INSERT (:Acct {id: 'b'})`);
    expect(ids(g)).toEqual(['b']);
  });

  describe('parsing', () => {
    test('produces a TxControl node with the right kind and access mode', () => {
      const start = parseQuery(`START TRANSACTION`);
      expect(isTxControl(start) && start).toEqual({ kind: 'start', accessMode: undefined });

      const ro = parseQuery(`START TRANSACTION READ ONLY`);
      expect(isTxControl(ro) && ro).toEqual({ kind: 'start', accessMode: 'read only' });

      const rw = parseQuery(`START TRANSACTION READ WRITE`);
      expect(isTxControl(rw) && rw).toEqual({ kind: 'start', accessMode: 'read write' });

      expect(parseQuery(`COMMIT`)).toEqual({ kind: 'commit' });
      expect(parseQuery(`COMMIT WORK`)).toEqual({ kind: 'commit' });
      expect(parseQuery(`ROLLBACK`)).toEqual({ kind: 'rollback' });
      expect(parseQuery(`ROLLBACK WORK`)).toEqual({ kind: 'rollback' });

      // Case-insensitive.
      expect(parseQuery(`start transaction read only`)).toEqual({
        kind: 'start',
        accessMode: 'read only',
      });
    });

    test('malformed transaction commands are syntax errors', () => {
      expect(codeOf(() => query(new Graph(), `START`))).toBe(ErrorCode.Syntax);
      expect(codeOf(() => query(new Graph(), `START FROBNICATE`))).toBe(ErrorCode.Syntax);
      expect(codeOf(() => query(new Graph(), `START TRANSACTION READ SIDEWAYS`))).toBe(
        ErrorCode.Syntax,
      );
      // Two access modes → a trailing-input syntax error.
      expect(codeOf(() => query(new Graph(), `START TRANSACTION READ ONLY READ WRITE`))).toBe(
        ErrorCode.Syntax,
      );
      expect(codeOf(() => query(new Graph(), `COMMIT ALL THE THINGS`))).toBe(ErrorCode.Syntax);
    });

    test('reserved-word non-regression: the soft keywords stay usable as identifiers', () => {
      const g = new Graph();
      // `read`, `write`, `only`, `work`, `transaction` are NOT reserved — a
      // variable / label / alias named after one still parses and runs.
      query(g, `INSERT (:read {write: 1})`);
      expect(query(g, `MATCH (read:read) RETURN read.write AS only`)).toEqual([{ only: 1 }]);
      expect(query(g, `MATCH (n:read) RETURN n.write AS transaction`)).toEqual([
        { transaction: 1 },
      ]);

      // `commit` IS an ISO reserved word (pre-existing) — usable only delimited.
      const c = parseQuery('MATCH (`commit`) RETURN `commit`');
      expect(isTxControl(c)).toBe(false);
    });
  });
});
