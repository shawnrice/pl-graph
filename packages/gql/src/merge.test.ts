import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { query } from './index.js';
import { parse } from './parser.js';

// The `_MERGE` keyed-upsert extension (node form). Behavior spec:
// docs/design/gql-extensions.md §2. Rust parity + differential land in later
// slices; this pins the TS semantics.
describe('GQL: _MERGE (keyed upsert, node form)', () => {
  const codeOf = (fn: () => unknown): unknown => {
    try {
      fn();
    } catch (e) {
      return (e as { code?: unknown }).code;
    }

    throw new Error('expected a throw');
  };

  test('create path: inserts the pattern + runs _ON_CREATE', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'email');

    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'A'}) _ON_CREATE SET u.created = 1`);

    expect(query(g, `MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name, u.created`)).toEqual([
      { 'u.name': 'A', 'u.created': 1 },
    ]);
    expect(query(g, `MATCH (u:Acct) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });

  test('update path default = clobber payload; _ON_CREATE does not re-run; still one node', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'email');
    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'A'}) _ON_CREATE SET u.created = 1`);

    // Same email present → clobber the payload (name), leave the key; created
    // stays (birth-only, not re-run), and no second node is created.
    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'A2'})`);

    expect(query(g, `MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name, u.created`)).toEqual([
      { 'u.name': 'A2', 'u.created': 1 },
    ]);
    expect(query(g, `MATCH (u:Acct) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });

  test('_ON_UPDATE SET replaces the default clobber (payload not written)', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'email');
    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'A'})`);

    // The pattern payload `name: 'IGNORED'` is NOT written — the explicit
    // _ON_UPDATE replaces the default clobber entirely.
    query(
      g,
      `_MERGE (u:Acct {email: 'a@x.io', name: 'IGNORED'}) _ON_UPDATE SET u.name = 'FromUpdate'`,
    );

    expect(query(g, `MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name`)).toEqual([
      { 'u.name': 'FromUpdate' },
    ]);
  });

  test('_ON_UPDATE_NOTHING leaves the existing element untouched', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Acct', 'email');
    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'A'})`);

    query(g, `_MERGE (u:Acct {email: 'a@x.io', name: 'IGNORED'}) _ON_UPDATE_NOTHING`);

    expect(query(g, `MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name`)).toEqual([{ 'u.name': 'A' }]);
  });

  test('WHERE-gated update = last-write-wins (false predicate is a no-op)', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Doc', 'id');
    query(g, `_MERGE (d:Doc {id: 1, v: 1, body: 'first'})`);

    // Incoming v (5) is newer than stored (1) → applies.
    query(g, `_MERGE (d:Doc {id: 1}) _ON_UPDATE SET d.v = 5, d.body = 'newer' WHERE d.v < 5`);
    expect(query(g, `MATCH (d:Doc {id: 1}) RETURN d.v, d.body`)).toEqual([
      { 'd.v': 5, 'd.body': 'newer' },
    ]);

    // Stored (5) is not < 3 → predicate false → no-op.
    query(g, `_MERGE (d:Doc {id: 1}) _ON_UPDATE SET d.v = 3, d.body = 'older' WHERE d.v < 3`);
    expect(query(g, `MATCH (d:Doc {id: 1}) RETURN d.v, d.body`)).toEqual([
      { 'd.v': 5, 'd.body': 'newer' },
    ]);
  });

  test('presence idiom: bare _MERGE clobbers, tracking the moving value', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('Presence', 'sid');
    query(g, `_MERGE (p:Presence {sid: 's1', x: 0, y: 0})`);
    query(g, `_MERGE (p:Presence {sid: 's1', x: 10, y: 20})`);

    expect(query(g, `MATCH (p:Presence) RETURN p.x, p.y`)).toEqual([{ 'p.x': 10, 'p.y': 20 }]);
    expect(query(g, `MATCH (p:Presence) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });

  test('no unique constraint on the label → error (can’t define the key)', () => {
    const g = createTestSocialGraph();
    expect(codeOf(() => query(g, `_MERGE (x:Nope {k: 1})`))).toBe(ErrorCode.InvalidGraphOp);
  });

  test('parse: conflicting update dispositions rejected', () => {
    expect(() =>
      parse(`_MERGE (u:Acct {email: 'a'}) _ON_UPDATE SET u.n = 1 _ON_UPDATE_NOTHING`),
    ).toThrow();
  });

  test('iso-strict dialect does not recognize _MERGE (extension gated off)', () => {
    // Under iso-strict, `_MERGE` is a plain identifier → no clause → syntax error.
    expect(() => parse(`_MERGE (u:Acct {email: 'a'})`, { dialect: 'iso-strict' })).toThrow();
    // …but it parses fine under the default (lenke) dialect.
    expect(() => parse(`_MERGE (u:Acct {email: 'a'})`)).not.toThrow();
  });

  test('edge form: upserts the edge between key-matched endpoints (ensure-tuple)', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('User', 'id');
    g.createUniqueConstraint('Team', 'id');
    query(g, `INSERT (:User {id: 'u1'}), (:Team {id: 't1'})`);

    query(
      g,
      `_MERGE (u:User {id: 'u1'})-[m:MEMBER {since: 1}]->(t:Team {id: 't1'}) _ON_CREATE SET m.role = 'admin'`,
    );
    expect(
      query(g, `MATCH (:User {id:'u1'})-[m:MEMBER]->(:Team {id:'t1'}) RETURN m.since, m.role`),
    ).toEqual([{ 'm.since': 1, 'm.role': 'admin' }]);

    // Idempotent: clobbers edge props, no duplicate edge, _ON_CREATE not re-run.
    query(g, `_MERGE (u:User {id: 'u1'})-[m:MEMBER {since: 2}]->(t:Team {id: 't1'})`);
    expect(
      query(g, `MATCH (:User {id:'u1'})-[m:MEMBER]->(:Team {id:'t1'}) RETURN m.since, m.role`),
    ).toEqual([{ 'm.since': 2, 'm.role': 'admin' }]);
    expect(query(g, `MATCH (:User)-[m:MEMBER]->(:Team) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });

  test('edge form: a missing endpoint errors', () => {
    const g = createTestSocialGraph();
    g.createUniqueConstraint('User', 'id');
    g.createUniqueConstraint('Team', 'id');
    query(g, `INSERT (:User {id: 'u1'})`);
    expect(codeOf(() => query(g, `_MERGE (u:User {id:'u1'})-[m:MEMBER]->(t:Team {id:'t1'})`))).toBe(
      ErrorCode.InvalidGraphOp,
    );
  });

  test('iso-strict parses the whole ISO surface but rejects every extension', () => {
    // A spread of pure-ISO GQL — all must parse under iso-strict, proving the
    // ISO surface is self-contained (no extension leaked into it).
    const iso = [
      `MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b.name`,
      `INSERT (:Person {name: 'x', age: 1})`,
      `MATCH (n:Person) SET n.age = 2`,
      `MATCH (n:Person) REMOVE n.age`,
      `MATCH (n:Person) DETACH DELETE n`,
      `MATCH (n) RETURN count(*) AS c ORDER BY c DESC LIMIT 5`,
    ];

    for (const q of iso) {
      expect(() => parse(q, { dialect: 'iso-strict' }), q).not.toThrow();
    }

    // Every extension construct is a syntax error under iso-strict.
    for (const ext of [
      `_MERGE (u:Acct {email: 'a'})`,
      `_MERGE (u:Acct {email: 'a'}) _ON_CREATE SET u.x = 1`,
    ]) {
      expect(() => parse(ext, { dialect: 'iso-strict' }), ext).toThrow();
    }
  });

  // Edge `_MERGE` with endpoints bound by a preceding MATCH — the natural way to
  // upsert an edge between two known vertices. Regression: resolveMergeEndpoint
  // ignored the binding and re-inferred a unique key from the (empty) node
  // pattern, so the bound-variable edge-merge form was unusable (it threw
  // "_MERGE needs a unique constraint on the pattern's label(s) []"). Mirrors the
  // native `merge_edge_between_bound_endpoints_upserts` test for byte-identity.
  test('edge form: upserts between bound endpoints (create then update)', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['A'], properties: { id: 'a' } });
    g.addVertex({ id: 'b', labels: ['A'], properties: { id: 'b' } });
    g.createUniqueConstraint('A', 'id');

    const merge =
      "MATCH (a:A {id: 'a'}), (b:A {id: 'b'}) " +
      '_MERGE (a)-[r:R]->(b) _ON_CREATE SET r.n = 1 _ON_UPDATE SET r.n = r.n + 100 ' +
      'RETURN r.n AS n';

    expect(query(g, merge)).toEqual([{ n: 1 }]); // created
    expect(query(g, merge)).toEqual([{ n: 101 }]); // updated, not duplicated
    expect(query(g, 'MATCH (:A)-[r:R]->(:A) RETURN count(r) AS c')).toEqual([{ c: 1 }]);
  });
});
