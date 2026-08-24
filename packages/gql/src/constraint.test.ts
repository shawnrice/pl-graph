import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { query } from './index.js';

/** Run `fn`, expecting a thrown coded error; return its `.code`. */
const codeOf = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return (e as { code?: unknown }).code;
  }

  throw new Error('expected a throw, got a normal return');
};

// Byte-identical with the Rust core's constraint tests
// (crates/lenke-engine/src/gql/tests.rs). See docs/design/gql-extensions.md §3.
describe('GQL: unique constraints', () => {
  test('enforced on INSERT and SET; per-label; a self-set is not a collision', () => {
    const g = createTestSocialGraph(); // no Acct/Other labels — a clean namespace
    g.createUniqueConstraint('Acct', 'email');

    query(g, `INSERT (:Acct {email: 'a@x.io', name: 'A'})`);

    // Duplicate email under the same label → violation (no partial write).
    expect(codeOf(() => query(g, `INSERT (:Acct {email: 'a@x.io', name: 'B'})`))).toBe(
      ErrorCode.ConstraintViolation,
    );

    // A different email is fine; a different label with the same email is fine.
    query(g, `INSERT (:Acct {email: 'b@x.io', name: 'B'})`);
    query(g, `INSERT (:Other {email: 'a@x.io'})`);

    // A SET that collides with an existing Acct email → violation …
    expect(codeOf(() => query(g, `MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'a@x.io'`))).toBe(
      ErrorCode.ConstraintViolation,
    );
    // … but setting a row to its OWN current value is not a self-collision.
    query(g, `MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'b@x.io'`);
  });

  test('the check is deferred to commit — a transient duplicate resolved before commit is allowed', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'id');
    query(g, `INSERT (:Acct {id: 1, bal: 100})`);

    // Inside one transaction: create a duplicate id, then resolve it before
    // commit. The eager per-statement check used to reject this; the native
    // engine defers to commit and accepts it — now TS matches.
    g.transaction(() => {
      query(g, `INSERT (:Acct {id: 1, bal: 999})`); // transiently two id=1
      query(g, `MATCH (n:Acct {id: 1, bal: 999}) DELETE n`); // resolved before commit
    });
    expect(query(g, `MATCH (n:Acct) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);

    // A SET that collides and is reverted within the same transaction also commits.
    query(g, `INSERT (:Acct {id: 2, bal: 50})`);
    g.transaction(() => {
      query(g, `MATCH (n:Acct {id: 2}) SET n.id = 1`); // collides transiently
      query(g, `MATCH (n:Acct {id: 1, bal: 50}) SET n.id = 2`); // reverted
    });
    expect(query(g, `MATCH (n:Acct) RETURN count(*) AS c`)).toEqual([{ c: 2 }]);

    // But a duplicate that SURVIVES to commit still rolls the whole tx back.
    expect(
      codeOf(() =>
        g.transaction(() => {
          query(g, `INSERT (:Acct {id: 3, bal: 1})`);
          query(g, `INSERT (:Acct {id: 3, bal: 2})`); // unresolved → violation at commit
        }),
      ),
    ).toBe(ErrorCode.ConstraintViolation);
    expect(query(g, `MATCH (n:Acct) RETURN count(*) AS c`)).toEqual([{ c: 2 }]);
  });

  test('null and absent values are exempt (SQL: NULLs distinct)', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'email');
    query(g, `INSERT (:Acct {email: null, name: 'A'})`);
    query(g, `INSERT (:Acct {email: null, name: 'B'})`);
    query(g, `INSERT (:Acct {name: 'C'})`);
  });

  test('createUniqueConstraint rejects pre-existing duplicates', () => {
    const g = new Graph();
    g.addVertex({ labels: ['Acct'], properties: { email: 'dup@x.io' } });
    g.addVertex({ labels: ['Acct'], properties: { email: 'dup@x.io' } });
    expect(codeOf(() => g.createUniqueConstraint('Acct', 'email'))).toBe(
      ErrorCode.ConstraintViolation,
    );
  });

  test('introspection + drop', () => {
    const g = new Graph();
    g.createUniqueConstraint('Acct', 'email');
    g.createUniqueConstraint('Acct', 'handle');
    expect(g.hasUniqueConstraint('Acct', 'email')).toBe(true);
    expect(g.uniqueKeys('Acct')).toEqual(['email', 'handle']);
    expect(g.uniqueConstraints()).toEqual([
      ['Acct', 'email'],
      ['Acct', 'handle'],
    ]);
    g.dropUniqueConstraint('Acct', 'email');
    expect(g.hasUniqueConstraint('Acct', 'email')).toBe(false);
    expect(g.hasUniqueConstraint('Acct', 'handle')).toBe(true);
  });
});
