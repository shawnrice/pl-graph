import { describe, expect, test } from 'bun:test';

import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { parseDate } from '../temporal.js';
import { Graph, parseTypeSpec, typeSpecName } from './Graph.js';

const thrown = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return e;
  }

  return undefined;
};

const isCV = (fn: () => unknown): boolean =>
  hasErrorCode(thrown(fn), ErrorCode.ConstraintViolation);

describe('R-CONSTRAINTS: required', () => {
  test('declaring over already-violating data throws', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['User'], properties: {} }); // no email
    expect(isCV(() => g.createRequiredConstraint('User', 'email'))).toBe(true);
  });

  test('addVertex is rejected when a required key is absent or null', () => {
    const g = new Graph();
    g.addVertex({ id: 'seed', labels: ['User'], properties: { email: 's@x.io' } });
    g.createRequiredConstraint('User', 'email');

    expect(isCV(() => g.addVertex({ id: 'b', labels: ['User'], properties: {} }))).toBe(true);
    expect(
      isCV(() => g.addVertex({ id: 'c', labels: ['User'], properties: { email: null } })),
    ).toBe(true);
    // present (even '' / false / 0) satisfies; the write commits
    expect(
      g
        .addVertex({ id: 'd', labels: ['User'], properties: { email: '' } })
        .getProperty<string>('email'),
    ).toBe('');
    // a vertex without the constrained label is unaffected
    expect(g.addVertex({ id: 'e', labels: ['Bot'], properties: {} })).toBeTruthy();
  });

  test('a required key cannot be nulled or removed', () => {
    const g = new Graph();
    const v = g.addVertex({ id: 'u', labels: ['User'], properties: { email: 'u@x.io' } });
    g.createRequiredConstraint('User', 'email');

    expect(isCV(() => v.setProperty('email', null))).toBe(true);
    expect(isCV(() => v.removeProperty('email'))).toBe(true);
    expect(isCV(() => v.setProperties({ email: null, name: 'U' }))).toBe(true);
    expect(isCV(() => v.removeProperties(['email']))).toBe(true);
    // a non-required key is free to change
    v.setProperty('name', 'U');
    expect(v.getProperty<string>('name')).toBe('U');
    // the required key survived every rejected write
    expect(v.getProperty<string>('email')).toBe('u@x.io');
  });

  test('adding a label brings its required keys into force', () => {
    const g = new Graph();
    g.createRequiredConstraint('User', 'email');
    const p = g.addVertex({ id: 'p', labels: ['Person'], properties: {} });
    expect(isCV(() => g.addLabelToVertex('User', p))).toBe(true); // missing email
    p.setProperty('email', 'p@x.io');
    expect(g.addLabelToVertex('User', p)).toBeTruthy(); // now satisfied
  });

  test('the constraint registry is queryable', () => {
    const g = new Graph();
    g.createRequiredConstraint('User', 'email');
    g.createRequiredConstraint('User', 'name');
    expect(g.hasRequiredConstraint('User', 'email')).toBe(true);
    expect(g.hasRequiredConstraint('User', 'age')).toBe(false);
    expect(g.requiredKeys('User')).toEqual(['email', 'name']);
    expect(g.requiredConstraints()).toEqual([
      ['User', 'email'],
      ['User', 'name'],
    ]);
    g.dropRequiredConstraint('User', 'email');
    expect(g.requiredKeys('User')).toEqual(['name']);
  });
});

describe('R-CONSTRAINTS: type', () => {
  test('declaring over wrong-typed data throws', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['P'], properties: { age: 'old' } });
    expect(isCV(() => g.createTypeConstraint('P', 'age', 'number'))).toBe(true);
  });

  test('addVertex/setProperty reject a wrong-typed value; null/absent are exempt', () => {
    const g = new Graph();
    g.createTypeConstraint('P', 'age', 'number');
    g.createTypeConstraint('P', 'name', 'string');
    g.createTypeConstraint('P', 'dob', 'date');
    g.createTypeConstraint('P', 'tags', 'list');

    expect(isCV(() => g.addVertex({ id: 'a', labels: ['P'], properties: { age: 'forty' } }))).toBe(
      true,
    );
    expect(isCV(() => g.addVertex({ id: 'b', labels: ['P'], properties: { name: 5 } }))).toBe(true);
    expect(isCV(() => g.addVertex({ id: 'c', labels: ['P'], properties: { dob: 'nope' } }))).toBe(
      true,
    );
    expect(isCV(() => g.addVertex({ id: 'l', labels: ['P'], properties: { tags: 'x' } }))).toBe(
      true,
    );

    // right type commits; a matching temporal/list is fine
    const ok = g.addVertex({
      id: 'ok',
      labels: ['P'],
      properties: { age: 40, name: 'A', dob: parseDate('2000-01-01'), tags: ['x'] },
    });
    expect(ok.getProperty<number>('age')).toBe(40);
    // null and absent are type-exempt (use `required` for presence)
    expect(g.addVertex({ id: 'n', labels: ['P'], properties: { age: null } })).toBeTruthy();
    expect(g.addVertex({ id: 'm', labels: ['P'], properties: {} })).toBeTruthy();

    // setProperty enforces the type; null is still exempt
    expect(isCV(() => ok.setProperty('age', 'x'))).toBe(true);
    ok.setProperty('age', 41);
    expect(ok.getProperty<number>('age')).toBe(41);
    ok.setProperty('age', null); // exempt
    expect(ok.getProperty('age')).toBeNull();
  });

  test('the registry is queryable', () => {
    const g = new Graph();
    g.createTypeConstraint('P', 'age', 'number');
    g.createTypeConstraint('P', 'name', 'string');
    expect(g.typeConstraint('P', 'age')).toBe('number');
    expect(g.typeConstraint('P', 'missing')).toBeUndefined();
    expect(g.typeConstraints()).toEqual([
      ['P', 'age', 'number'],
      ['P', 'name', 'string'],
    ]);
    g.dropTypeConstraint('P', 'age');
    expect(g.typeConstraint('P', 'age')).toBeUndefined();
  });
});

// V3 (Rafael, r7): a unique constraint used to be enforced only on the GQL
// `INSERT`/`SET` path — the direct `addVertex`/`setProperty` API bypassed it,
// yielding count=2 with no throw. It's now a core invariant at the mutation
// chokepoint, so every write path agrees.
describe('R-CONSTRAINTS: unique (direct-API enforcement, V3)', () => {
  test('addVertex rejects a duplicate under a unique constraint', () => {
    const g = new Graph();
    g.createUniqueConstraint('User', 'email');

    g.addVertex({ id: 'a', labels: ['User'], properties: { email: 'x@y.io' } });

    expect(
      isCV(() => g.addVertex({ id: 'b', labels: ['User'], properties: { email: 'x@y.io' } })),
    ).toBe(true);
    // A different value, a different label, and null are all fine.
    expect(
      g.addVertex({ id: 'c', labels: ['User'], properties: { email: 'z@y.io' } }),
    ).toBeTruthy();
    expect(
      g.addVertex({ id: 'd', labels: ['Other'], properties: { email: 'x@y.io' } }),
    ).toBeTruthy();
    expect(g.addVertex({ id: 'n1', labels: ['User'], properties: { email: null } })).toBeTruthy();
    expect(g.addVertex({ id: 'n2', labels: ['User'], properties: { email: null } })).toBeTruthy();
    // The rejected insert left no trace.
    const emails = [...g.getVerticesByLabel('User')].filter(
      (v) => v.getProperty('email') === 'x@y.io',
    );
    expect(emails).toHaveLength(1);
  });

  test('setProperty rejects a collision under a unique constraint', () => {
    const g = new Graph();
    g.createUniqueConstraint('User', 'email');
    g.addVertex({ id: 'a', labels: ['User'], properties: { email: 'x@y.io' } });
    const b = g.addVertex({ id: 'b', labels: ['User'], properties: { email: 'z@y.io' } });

    expect(isCV(() => b.setProperty('email', 'x@y.io'))).toBe(true);
    // Re-setting a vertex's own value is not a self-collision.
    b.setProperty('email', 'z@y.io');
    expect(b.getProperty<string>('email')).toBe('z@y.io');
  });
});

// Edge-side constraints are a direct mirror of the vertex ones, keyed by edge
// TYPE, enforced at the addEdge gate + Edge property mutators + addLabelToEdge,
// and deferred to commit inside a transaction (R-TX). Byte-identical to Rust.
describe('R-CONSTRAINTS: edge unique/required/type', () => {
  // Two anchor vertices for edges; every edge below runs between them (parallel
  // edges are fine — an LPG allows many edges between the same pair).
  const anchors = (g: Graph): [ReturnType<Graph['addVertex']>, ReturnType<Graph['addVertex']>] => [
    g.addVertex({ labels: ['N'], properties: {} }),
    g.addVertex({ labels: ['N'], properties: {} }),
  ];
  const edge = (
    g: Graph,
    a: ReturnType<Graph['addVertex']>,
    b: ReturnType<Graph['addVertex']>,
    type: string,
    properties: Record<string, unknown>,
  ) => g.addEdge({ from: a, to: b, labels: [type], properties });

  test('declaring over already-violating edges throws (unique/required/type)', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    edge(g, a, b, 'FOLLOWS', { tag: 'x' });
    edge(g, a, b, 'FOLLOWS', { tag: 'x' });
    expect(isCV(() => g.createEdgeUniqueConstraint('FOLLOWS', 'tag'))).toBe(true);

    edge(g, a, b, 'REL', {}); // no `since`
    expect(isCV(() => g.createEdgeRequiredConstraint('REL', 'since'))).toBe(true);

    edge(g, a, b, 'TYP', { since: 'old' });
    expect(isCV(() => g.createEdgeTypeConstraint('TYP', 'since', 'number'))).toBe(true);
  });

  test('the addEdge gate enforces unique (null/other-type/other-value exempt)', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    g.createEdgeUniqueConstraint('FOLLOWS', 'tag');
    edge(g, a, b, 'FOLLOWS', { tag: 'x' });

    expect(isCV(() => edge(g, a, b, 'FOLLOWS', { tag: 'x' }))).toBe(true);
    // A different value, a different type, and null are all fine.
    expect(edge(g, a, b, 'FOLLOWS', { tag: 'y' })).toBeTruthy();
    expect(edge(g, a, b, 'LIKES', { tag: 'x' })).toBeTruthy();
    expect(edge(g, a, b, 'FOLLOWS', { tag: null })).toBeTruthy();
    expect(edge(g, a, b, 'FOLLOWS', { tag: null })).toBeTruthy();
    // The rejected insert left no trace: exactly one FOLLOWS tag='x'.
    const x = [...g.getEdgesByLabel('FOLLOWS')].filter((e) => e.getProperty('tag') === 'x');
    expect(x).toHaveLength(1);
  });

  test('the addEdge gate enforces required + type; setProperty too', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    g.createEdgeRequiredConstraint('REL', 'since');
    g.createEdgeTypeConstraint('REL', 'since', 'number');

    expect(isCV(() => edge(g, a, b, 'REL', {}))).toBe(true); // missing required
    expect(isCV(() => edge(g, a, b, 'REL', { since: null }))).toBe(true); // null required
    expect(isCV(() => edge(g, a, b, 'REL', { since: 'old' }))).toBe(true); // wrong type
    const e = edge(g, a, b, 'REL', { since: 2020 }); // ok

    // A required key cannot be nulled or removed; a wrong type is rejected.
    expect(isCV(() => e.setProperty('since', null))).toBe(true);
    expect(isCV(() => e.removeProperty('since'))).toBe(true);
    expect(isCV(() => e.setProperty('since', 'nope'))).toBe(true);
    e.setProperty('since', 2021); // right type, present
    expect(e.getProperty<number>('since')).toBe(2021);
  });

  test('setProperty enforces edge unique; self-set is not a collision', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    g.createEdgeUniqueConstraint('FOLLOWS', 'tag');
    edge(g, a, b, 'FOLLOWS', { tag: 'x' });
    const e2 = edge(g, a, b, 'FOLLOWS', { tag: 'y' });

    expect(isCV(() => e2.setProperty('tag', 'x'))).toBe(true);
    e2.setProperty('tag', 'y'); // re-setting its own value is fine
    expect(e2.getProperty<string>('tag')).toBe('y');
  });

  test('adding an edge type brings its required keys into force', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    g.createEdgeRequiredConstraint('AUDITED', 'at');
    const e = edge(g, a, b, 'PLAIN', {}); // lacks `at`
    expect(isCV(() => g.addLabelToEdge('AUDITED', e))).toBe(true);
    e.setProperty('at', 1);
    expect(g.addLabelToEdge('AUDITED', e)).toBeTruthy(); // now satisfied
  });

  test('the edge registries are queryable', () => {
    const g = new Graph();
    g.createEdgeUniqueConstraint('FOLLOWS', 'tag');
    g.createEdgeRequiredConstraint('FOLLOWS', 'since');
    g.createEdgeTypeConstraint('FOLLOWS', 'since', 'number');
    expect(g.hasEdgeUniqueConstraint('FOLLOWS', 'tag')).toBe(true);
    expect(g.edgeUniqueKeys('FOLLOWS')).toEqual(['tag']);
    expect(g.edgeUniqueConstraintList()).toEqual([['FOLLOWS', 'tag']]);
    expect(g.hasEdgeRequiredConstraint('FOLLOWS', 'since')).toBe(true);
    expect(g.edgeRequiredKeys('FOLLOWS')).toEqual(['since']);
    expect(g.edgeTypeConstraint('FOLLOWS', 'since')).toBe('number');
    expect(g.edgeTypeConstraintList()).toEqual([['FOLLOWS', 'since', 'number']]);
    g.dropEdgeUniqueConstraint('FOLLOWS', 'tag');
    expect(g.hasEdgeUniqueConstraint('FOLLOWS', 'tag')).toBe(false);
  });

  test('DEFERRED: an intermediate edge violation resolved before commit commits; unresolved rolls back', () => {
    const g = new Graph();
    const [a, b] = anchors(g);
    g.createEdgeRequiredConstraint('REL', 'since');
    g.createEdgeUniqueConstraint('REL', 'tag');

    // Resolved-before-commit: an edge inserted without its required key, and two
    // edges that momentarily share a unique value, both settle before commit.
    g.transaction((tx) => {
      const e1 = tx.addEdge({
        from: a,
        to: b,
        labels: ['REL'],
        properties: { since: 1, tag: 'x' },
      });
      // A sibling that momentarily collides on `tag`, then is disambiguated.
      const e2 = tx.addEdge({
        from: a,
        to: b,
        labels: ['REL'],
        properties: { since: 2, tag: 'x' },
      });
      e2.setProperty('tag', 'y');
      // An edge added missing its required key, then given it before commit.
      const e3 = tx.addEdge({ from: a, to: b, labels: ['REL'], properties: { tag: 'z' } });
      e3.setProperty('since', 3);
      expect(e1.getProperty<string>('tag')).toBe('x');
    });
    expect(g.edgeCount).toBe(3); // all three committed

    // Unresolved required violation → the whole transaction rolls back.
    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.addEdge({ from: a, to: b, labels: ['REL'], properties: { tag: 'w' } }); // never given `since`
        }),
      ),
    ).toBe(true);
    expect(g.edgeCount).toBe(3); // rolled back — no new edge

    // Unresolved unique collision → rolls back too.
    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.addEdge({ from: a, to: b, labels: ['REL'], properties: { since: 9, tag: 'x' } }); // dup of e1.tag
        }),
      ),
    ).toBe(true);
    expect(g.edgeCount).toBe(3);
  });
});

describe('R-CONSTRAINTS: cardinality', () => {
  // An Order must be placed by exactly one Customer: bound the OUT-degree of every
  // :Order over PLACED_BY to [1, 1].
  const order = (g: Graph, id: string) => g.addVertex({ id, labels: ['Order'], properties: {} });
  const customer = (g: Graph, id: string) =>
    g.addVertex({ id, labels: ['Customer'], properties: {} });

  test('declare-time rejection of already-violating data', () => {
    const g = new Graph();
    order(g, 'o1'); // out-degree 0, but min:1
    expect(isCV(() => g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1))).toBe(true);

    // With the mandatory edge in place, declaring succeeds.
    const c1 = customer(g, 'c1');
    g.addEdge({ from: g.getVertexById('o1')!, to: c1, labels: ['PLACED_BY'], properties: {} });
    expect(() => g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1)).not.toThrow();

    // A max-only constraint (at most one) also rejects pre-existing over-max data.
    const o2 = order(g, 'o2');
    g.addEdge({ from: o2, to: c1, labels: ['SHIPPED_TO'], properties: {} });
    g.addEdge({ from: o2, to: customer(g, 'c2'), labels: ['SHIPPED_TO'], properties: {} });
    expect(isCV(() => g.createCardinalityConstraint('Order', 'SHIPPED_TO', 'out', 0, 1))).toBe(
      true,
    );
  });

  test('max is rejected EAGERLY on the over-the-limit addEdge (no transaction)', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 0, 1); // at most one
    const o1 = order(g, 'o1');
    const c1 = customer(g, 'c1');
    const c2 = customer(g, 'c2');

    g.addEdge({ from: o1, to: c1, labels: ['PLACED_BY'], properties: {} }); // degree 1, ok
    // The second PLACED_BY out-edge would make degree 2 > max 1 → thrown immediately.
    expect(isCV(() => g.addEdge({ from: o1, to: c2, labels: ['PLACED_BY'], properties: {} }))).toBe(
      true,
    );
    // The rejected edge left no trace.
    expect(g.outDegree(o1, 'PLACED_BY')).toBe(1);
    expect(g.edgeCount).toBe(1);
  });

  test('IN direction bounds the target vertex', () => {
    const g = new Graph();
    // A Customer may receive at most one PRIMARY_CONTACT in-edge.
    g.createCardinalityConstraint('Customer', 'PRIMARY_CONTACT', 'in', 0, 1);
    const c1 = customer(g, 'c1');
    g.addEdge({ from: order(g, 'o1'), to: c1, labels: ['PRIMARY_CONTACT'], properties: {} });
    expect(g.inDegree(c1, 'PRIMARY_CONTACT')).toBe(1);
    expect(
      isCV(() =>
        g.addEdge({ from: order(g, 'o2'), to: c1, labels: ['PRIMARY_CONTACT'], properties: {} }),
      ),
    ).toBe(true);
  });

  test('"exactly one" is satisfied when node+edge are created together in a transaction', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1);
    const c1 = customer(g, 'c1');

    // Node + mandatory edge in one transaction → the intermediate degree-0 state
    // is never checked; the committed state satisfies min:1,max:1.
    expect(() =>
      g.transaction((tx) => {
        const o = tx.addVertex({ id: 'o1', labels: ['Order'], properties: {} });
        tx.addEdge({ from: o, to: c1, labels: ['PLACED_BY'], properties: {} });
      }),
    ).not.toThrow();
    expect(g.getVertexById('o1')).toBeTruthy();
    expect(g.outDegree(g.getVertexById('o1')!, 'PLACED_BY')).toBe(1);
  });

  test('"exactly one" rolls back when the node is created without the mandatory edge', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1);

    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.addVertex({ id: 'o1', labels: ['Order'], properties: {} }); // no PLACED_BY edge
        }),
      ),
    ).toBe(true);
    expect(g.getVertexById('o1')).toBeNull(); // rolled back
  });

  test('min is NOT tripped by a bare direct-API addVertex outside a transaction', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1);
    // A bare addVertex has no commit boundary — min (only satisfiable across
    // writes) is deliberately not enforced here. The Order lands degree-0.
    expect(() => order(g, 'o1')).not.toThrow();
    expect(g.getVertexById('o1')).toBeTruthy();
    expect(g.outDegree(g.getVertexById('o1')!, 'PLACED_BY')).toBe(0);
  });

  test('removeEdge dropping below min is rejected at commit (rolls back)', () => {
    const g = new Graph();
    const c1 = customer(g, 'c1');
    const o1 = order(g, 'o1');
    const e = g.addEdge({ from: o1, to: c1, labels: ['PLACED_BY'], properties: {} });
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1); // now satisfied

    // Removing the only PLACED_BY edge inside a transaction drops o1 to degree 0
    // < min 1 → the commit fails and the removal rolls back.
    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.removeEdge(e);
        }),
      ),
    ).toBe(true);
    expect(g.outDegree(o1, 'PLACED_BY')).toBe(1); // restored
    expect(g.edgeCount).toBe(1);
  });

  test('a vertex-delete cascade re-checks the surviving neighbor at commit', () => {
    const g = new Graph();
    const c1 = customer(g, 'c1');
    const o1 = order(g, 'o1');
    g.addEdge({ from: o1, to: c1, labels: ['PLACED_BY'], properties: {} });
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1);

    // Deleting the Customer cascades to the PLACED_BY edge, dropping the surviving
    // Order to degree 0 < min → the whole delete rolls back.
    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.removeVertex('c1');
        }),
      ),
    ).toBe(true);
    expect(g.getVertexById('c1')).toBeTruthy(); // rolled back
    expect(g.outDegree(o1, 'PLACED_BY')).toBe(1);
  });

  test('a self-loop counts once for out AND once for in', () => {
    const g = new Graph();
    const v = g.addVertex({ id: 'v', labels: ['Node'], properties: {} });
    g.addEdge({ from: v, to: v, labels: ['SELF'], properties: {} });
    expect(g.outDegree(v, 'SELF')).toBe(1);
    expect(g.inDegree(v, 'SELF')).toBe(1);

    // So an out-bound "at most one" is satisfied, and a second self-loop trips it.
    g.createCardinalityConstraint('Node', 'SELF', 'out', 0, 1);
    expect(isCV(() => g.addEdge({ from: v, to: v, labels: ['SELF'], properties: {} }))).toBe(true);
  });

  test('an unbounded max (min:1, max:null) enforces only the lower bound', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'LINE_ITEM', 'out', 1, null); // at least one
    const c1 = customer(g, 'c1');
    // Many out-edges are fine (no upper bound); zero is not.
    g.transaction((tx) => {
      const o = tx.addVertex({ id: 'o1', labels: ['Order'], properties: {} });
      tx.addEdge({ from: o, to: c1, labels: ['LINE_ITEM'], properties: {} });
      tx.addEdge({ from: o, to: customer(tx, 'c2'), labels: ['LINE_ITEM'], properties: {} });
    });
    expect(g.outDegree(g.getVertexById('o1')!, 'LINE_ITEM')).toBe(2);
    expect(
      isCV(() =>
        g.transaction((tx) => tx.addVertex({ id: 'o2', labels: ['Order'], properties: {} })),
      ),
    ).toBe(true);
  });

  test('drop and introspection', () => {
    const g = new Graph();
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 1, 1);
    g.createCardinalityConstraint('Customer', 'PRIMARY_CONTACT', 'in', 0, 1);
    g.createCardinalityConstraint('Order', 'LINE_ITEM', 'out', 1, null);

    expect(g.cardinalityConstraints()).toEqual([
      { label: 'Customer', edgeType: 'PRIMARY_CONTACT', direction: 'in', min: 0, max: 1 },
      { label: 'Order', edgeType: 'LINE_ITEM', direction: 'out', min: 1, max: null },
      { label: 'Order', edgeType: 'PLACED_BY', direction: 'out', min: 1, max: 1 },
    ]);

    // Re-declaring the same (label, edgeType, direction) replaces the bounds.
    g.createCardinalityConstraint('Order', 'PLACED_BY', 'out', 0, 5);
    expect(g.cardinalityConstraints()).toContainEqual({
      label: 'Order',
      edgeType: 'PLACED_BY',
      direction: 'out',
      min: 0,
      max: 5,
    });

    g.dropCardinalityConstraint('Order', 'PLACED_BY', 'out');
    expect(
      g.cardinalityConstraints().map((c) => `${c.label}.${c.edgeType}.${c.direction}`),
    ).toEqual(['Customer.PRIMARY_CONTACT.in', 'Order.LINE_ITEM.out']);
    g.dropCardinalityConstraint('Order', 'PLACED_BY', 'out'); // idempotent
  });
});

describe('R-CONSTRAINTS: RECORD-typed', () => {
  test('parseTypeSpec + typeSpecName round-trip; declare validates existing data', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['P'], properties: { meta: { city: 'NYC', tier: 2 } } });
    // Matching shape declares OK.
    g.createTypeConstraint('P', 'meta', 'record{city::string,tier::number}');
    // Conflicting shape against existing data → rejected.
    const g2 = new Graph();
    g2.addVertex({ id: 'a', labels: ['P'], properties: { meta: { city: 'NYC', tier: 2 } } });
    expect(
      isCV(() => g2.createTypeConstraint('P', 'meta', 'record{city::number,tier::number}')),
    ).toBe(true);
  });

  test('set/insert enforce the closed record shape; null exempt', () => {
    const g = new Graph();
    g.createTypeConstraint('P', 'meta', 'record{city::string,tier::number}');
    const v = g.addVertex({
      id: 'a',
      labels: ['P'],
      properties: { meta: { city: 'NYC', tier: 1 } },
    });
    // A well-shaped plain-object write passes.
    expect(() => v.setProperty('meta', { city: 'LA', tier: 3 })).not.toThrow();
    // Fields are nullable/optional by default — a missing field is fine, and so
    // is an explicit null in a field.
    expect(() => v.setProperty('meta', { city: 'X' })).not.toThrow();
    expect(() => v.setProperty('meta', { city: 'X', tier: null })).not.toThrow();
    expect(() => v.setProperty('meta', {})).not.toThrow();
    // Wrong field type / extra field → rejected (closed on extras, typed fields).
    expect(isCV(() => v.setProperty('meta', { city: 1, tier: 3 }))).toBe(true);
    expect(isCV(() => v.setProperty('meta', { city: 'X', tier: 1, extra: 9 }))).toBe(true);
    // Null is exempt (top-level).
    expect(() => v.setProperty('meta', null)).not.toThrow();
    // A bad-shape insert is a violation.
    expect(
      isCV(() =>
        g.addVertex({ id: 'b', labels: ['P'], properties: { meta: { city: 9, tier: 1 } } }),
      ),
    ).toBe(true);
  });

  test('NOT NULL makes a record field required (present and non-null)', () => {
    const g = new Graph();
    g.createTypeConstraint('P', 'meta', 'record{city::string NOT NULL,tier::number}');
    // The name round-trips with the NOT NULL sigil.
    const spec = parseTypeSpec('record{city::string NOT NULL,tier::number}');

    expect(spec).not.toBeNull();
    expect(typeSpecName(spec!)).toBe('record{city::string NOT NULL,tier::number}');
    const v = g.addVertex({
      id: 'a',
      labels: ['P'],
      properties: { meta: { city: 'NYC', tier: 1 } },
    });
    // `tier` is nullable → may be omitted; `city` is NOT NULL → present + non-null.
    expect(() => v.setProperty('meta', { city: 'LA' })).not.toThrow();
    expect(isCV(() => v.setProperty('meta', { tier: 2 }))).toBe(true);
    expect(isCV(() => v.setProperty('meta', { city: null, tier: 2 }))).toBe(true);
    expect(isCV(() => v.setProperty('meta', {}))).toBe(true);
  });

  test('nested record type + edge record constraint + drop', () => {
    const g = new Graph();
    g.createTypeConstraint('P', 'addr', 'record{geo::record{lat::number,lng::number}}');
    const v = g.addVertex({
      id: 'a',
      labels: ['P'],
      properties: { addr: { geo: { lat: 1, lng: 2 } } },
    });
    expect(isCV(() => v.setProperty('addr', { geo: { lat: 'x', lng: 2 } }))).toBe(true);
    // Dropping lifts enforcement.
    g.dropTypeConstraint('P', 'addr');
    expect(() => v.setProperty('addr', { geo: { lat: 'x', lng: 2 } })).not.toThrow();

    // Edge record constraint.
    const g2 = new Graph();
    const a = g2.addVertex({ id: 'a', labels: ['P'], properties: {} });
    const b = g2.addVertex({ id: 'b', labels: ['P'], properties: {} });
    g2.createEdgeTypeConstraint('LINK', 'meta', 'record{w::number}');
    expect(
      isCV(() =>
        g2.addEdge({ from: a, to: b, labels: ['LINK'], properties: { meta: { w: 'x' } } }),
      ),
    ).toBe(true);
    expect(() =>
      g2.addEdge({ from: a, to: b, labels: ['LINK'], properties: { meta: { w: 0.5 } } }),
    ).not.toThrow();
  });
});

describe('R-CONSTRAINTS: scalar NOT NULL type constraint', () => {
  test('declaring over already absent/null data throws', () => {
    const absent = new Graph();
    absent.addVertex({ id: 'a', labels: ['P'], properties: {} });
    expect(isCV(() => absent.createTypeConstraint('P', 'name', 'string NOT NULL'))).toBe(true);

    const nulled = new Graph();
    nulled.addVertex({ id: 'a', labels: ['P'], properties: { name: null } });
    expect(isCV(() => nulled.createTypeConstraint('P', 'name', 'string NOT NULL'))).toBe(true);

    // A plain (nullable) type constraint is exempt from absent/null.
    const ok = new Graph();
    ok.addVertex({ id: 'a', labels: ['P'], properties: {} });
    expect(() => ok.createTypeConstraint('P', 'name', 'string')).not.toThrow();
  });

  test('NOT NULL requires present + non-null AND the declared type', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['P'], properties: { name: 'marko' } });
    g.createTypeConstraint('P', 'name', 'string NOT NULL');

    // absent / null → rejected (the required half)
    expect(isCV(() => g.addVertex({ id: 'b', labels: ['P'], properties: {} }))).toBe(true);
    expect(isCV(() => g.addVertex({ id: 'c', labels: ['P'], properties: { name: null } }))).toBe(
      true,
    );
    // wrong type → rejected (the type half)
    expect(isCV(() => g.addVertex({ id: 'd', labels: ['P'], properties: { name: 42 } }))).toBe(
      true,
    );
    // present, non-null, right type → OK
    expect(
      g
        .addVertex({ id: 'e', labels: ['P'], properties: { name: 'vadas' } })
        .getProperty<string>('name'),
    ).toBe('vadas');
  });

  test('a NOT NULL key cannot be nulled or removed', () => {
    const g = new Graph();
    const v = g.addVertex({ id: 'a', labels: ['P'], properties: { name: 'marko' } });
    g.createTypeConstraint('P', 'name', 'string NOT NULL');

    expect(isCV(() => v.setProperty('name', null))).toBe(true);
    expect(isCV(() => v.removeProperty('name'))).toBe(true);
    expect(v.getProperty<string>('name')).toBe('marko');
  });

  test('dropping the type constraint leaves an independent required intact', () => {
    const g = new Graph();
    const v = g.addVertex({ id: 'a', labels: ['P'], properties: { name: 'marko' } });
    g.createRequiredConstraint('P', 'name'); // declared independently
    g.createTypeConstraint('P', 'name', 'string NOT NULL');
    g.dropTypeConstraint('P', 'name');

    // The required survives; a wrong TYPE is now allowed (type constraint gone).
    expect(isCV(() => v.setProperty('name', null))).toBe(true);
    expect(() => v.setProperty('name', 42)).not.toThrow();
  });

  test('NOT NULL on a record-typed constraint is rejected', () => {
    const g = new Graph();
    expect(
      hasErrorCode(
        thrown(() => g.createTypeConstraint('P', 'meta', 'record{a::number} NOT NULL')),
        ErrorCode.InvalidValue,
      ),
    ).toBe(true);
  });

  test('edge scalar NOT NULL enforces present + non-null', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['P'], properties: {} });
    const b = g.addVertex({ id: 'b', labels: ['P'], properties: {} });
    g.addEdge({ from: a, to: b, labels: ['LINK'], properties: { w: 1.5 } });
    g.createEdgeTypeConstraint('LINK', 'w', 'number NOT NULL');

    expect(isCV(() => g.addEdge({ from: a, to: b, labels: ['LINK'], properties: {} }))).toBe(true);
    expect(
      isCV(() => g.addEdge({ from: a, to: b, labels: ['LINK'], properties: { w: null } })),
    ).toBe(true);
    expect(() =>
      g.addEdge({ from: a, to: b, labels: ['LINK'], properties: { w: 2.5 } }),
    ).not.toThrow();
  });
});
