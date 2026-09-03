import { describe, expect, test } from 'bun:test';

import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { Graph } from './Graph.js';

const isCV = (fn: () => unknown): boolean => {
  try {
    fn();
  } catch (e) {
    return hasErrorCode(e, ErrorCode.ConstraintViolation);
  }

  return false;
};

describe('atomic transaction (commit / rollback)', () => {
  test('transaction(fn) commits on return and returns fn result', () => {
    const g = new Graph();
    const out = g.transaction((tx) => {
      tx.addVertex({ id: 'a', labels: ['User'], properties: { name: 'A' } });
      tx.addVertex({ id: 'b', labels: ['User'], properties: { name: 'B' } });

      return 42;
    });

    expect(out).toBe(42);
    expect(g.vertexCount).toBe(2);
    expect(g.getVertexById('a')?.getProperty<string>('name')).toBe('A');
  });

  test('a throw inside the transaction rolls every write back (leaves no trace)', () => {
    const g = new Graph();
    g.addVertex({ id: 'seed', labels: ['User'], properties: { name: 'S' } });

    expect(() =>
      g.transaction((tx) => {
        tx.addVertex({ id: 'a', labels: ['User'], properties: { name: 'A' } });
        tx.getVertexById('seed')!.setProperty('name', 'MUTATED');

        throw new Error('boom');
      }),
    ).toThrow('boom');

    // Nothing from the transaction survived.
    expect(g.vertexCount).toBe(1);
    expect(g.getVertexById('a')).toBeNull();
    expect(g.getVertexById('seed')!.getProperty<string>('name')).toBe('S');
  });

  test('rollback reverses edges, labels, and property removals too', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['User'], properties: { name: 'A', age: 30 } });
    const b = g.addVertex({ id: 'b', labels: ['User'], properties: { name: 'B' } });
    g.addEdge({ id: 'e', from: a, to: b, labels: ['KNOWS'], properties: { since: 2020 } });

    expect(() =>
      g.transaction((tx) => {
        tx.getEdgeById('e')!.setProperty('since', 1999);
        tx.removeEdge(tx.getEdgeById('e')!);
        tx.getVertexById('a')!.removeProperty('age');
        tx.addLabelToVertex('Admin', tx.getVertexById('a')!);
        tx.removeVertex('b');

        throw new Error('undo everything');
      }),
    ).toThrow();

    // Original topology and values fully restored.
    expect(g.vertexCount).toBe(2);
    expect(g.edgeCount).toBe(1);
    expect(g.getVertexById('b')).not.toBeNull();
    expect(g.getEdgeById('e')!.getProperty<number>('since')).toBe(2020);
    expect(g.getVertexById('a')!.getProperty<number>('age')).toBe(30);
    expect(g.getVertexById('a')!.hasLabel('Admin')).toBe(false);
  });

  test('tx() handle: explicit commit and rollback', () => {
    const g = new Graph();

    const t1 = g.tx();
    g.addVertex({ id: 'keep', labels: ['X'], properties: {} });
    t1.commit();
    expect(g.getVertexById('keep')).not.toBeNull();

    const t2 = g.tx();
    g.addVertex({ id: 'drop', labels: ['X'], properties: {} });
    t2.rollback();
    expect(g.getVertexById('drop')).toBeNull();
  });
});

describe('deferred constraint checks', () => {
  test('required is checked at commit, not per-write (add node, fill mandatory key later)', () => {
    const g = new Graph();
    g.createRequiredConstraint('User', 'email');

    // Adding the node without email would throw immediately outside a tx; inside,
    // it's fine as long as the key is present by commit.
    g.transaction((tx) => {
      const u = tx.addVertex({ id: 'u', labels: ['User'], properties: {} });
      u.setProperty('email', 'u@x.io');
    });

    expect(g.getVertexById('u')!.getProperty<string>('email')).toBe('u@x.io');
  });

  test('a required violation that survives to commit rolls the whole tx back', () => {
    const g = new Graph();
    g.createRequiredConstraint('User', 'email');

    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.addVertex({ id: 'u', labels: ['User'], properties: {} }); // never gets an email
          tx.addVertex({ id: 'other', labels: ['Thing'], properties: {} });
        }),
      ),
    ).toBe(true);

    // Rolled back: neither vertex remains.
    expect(g.getVertexById('u')).toBeNull();
    expect(g.getVertexById('other')).toBeNull();
  });

  test('unique tolerates an intermediate collision that the final state resolves', () => {
    const g = new Graph();
    g.createUniqueConstraint('User', 'email');
    g.addVertex({ id: 'a', labels: ['User'], properties: { email: 'a@x.io' } });

    // Swap two emails within one tx — mid-transaction both briefly hold the same
    // value, which a per-write check would reject.
    g.addVertex({ id: 'b', labels: ['User'], properties: { email: 'b@x.io' } });
    g.transaction((tx) => {
      tx.getVertexById('a')!.setProperty('email', 'tmp@x.io');
      tx.getVertexById('b')!.setProperty('email', 'a@x.io');
      tx.getVertexById('a')!.setProperty('email', 'b@x.io');
    });

    expect(g.getVertexById('a')!.getProperty<string>('email')).toBe('b@x.io');
    expect(g.getVertexById('b')!.getProperty<string>('email')).toBe('a@x.io');
  });

  test('a genuine unique collision at commit rolls back', () => {
    const g = new Graph();
    g.createUniqueConstraint('User', 'email');
    g.addVertex({ id: 'a', labels: ['User'], properties: { email: 'a@x.io' } });

    expect(
      isCV(() =>
        g.transaction((tx) => {
          tx.addVertex({ id: 'b', labels: ['User'], properties: { email: 'a@x.io' } });
        }),
      ),
    ).toBe(true);

    expect(g.getVertexById('b')).toBeNull();
  });

  test('cross-write invariant: a user-thrown check rolls back a partial transfer', () => {
    const g = new Graph();
    g.addVertex({ id: 'acct1', labels: ['Acct'], properties: { balance: 1000 } });
    g.addVertex({ id: 'acct2', labels: ['Acct'], properties: { balance: 0 } });

    expect(() =>
      g.transaction((tx) => {
        const a = tx.getVertexById('acct1')!;
        const b = tx.getVertexById('acct2')!;
        a.setProperty('balance', (a.getProperty<number>('balance') ?? 0) - 400);

        // leg 2 "fails" before the credit lands
        throw new Error('remote leg failed');
        // eslint-disable-next-line no-unreachable
        b.setProperty('balance', (b.getProperty<number>('balance') ?? 0) + 400);
      }),
    ).toThrow();

    // Books stayed balanced — no half-applied transfer.
    expect(g.getVertexById('acct1')!.getProperty<number>('balance')).toBe(1000);
    expect(g.getVertexById('acct2')!.getProperty<number>('balance')).toBe(0);
  });
});

describe('event buffering', () => {
  test('events fire once on commit, not mid-transaction', () => {
    const g = new Graph();
    const seen: string[] = [];
    g.on('@graph/VertexAdded', (e) => seen.push((e.value as { id: string }).id));

    g.transaction((tx) => {
      tx.addVertex({ id: 'a', labels: ['X'], properties: {} });
      tx.addVertex({ id: 'b', labels: ['X'], properties: {} });
      // Nothing dispatched yet — buffered.
      expect(seen).toEqual([]);
    });

    expect(seen).toEqual(['a', 'b']);
  });

  test('a rolled-back transaction dispatches no events', () => {
    const g = new Graph();
    const seen: string[] = [];
    g.on('@graph/VertexAdded', (e) => seen.push((e.value as { id: string }).id));

    expect(() =>
      g.transaction((tx) => {
        tx.addVertex({ id: 'a', labels: ['X'], properties: {} });

        throw new Error('nope');
      }),
    ).toThrow();

    expect(seen).toEqual([]);
  });
});

// Found by the write-path differential fuzzer. An undo op must re-resolve its
// element by id at replay time, never hold the instance: a later write inside the
// same transaction can remove the element (a DETACH DELETE of an endpoint cascades
// to its edges), and `evict()` severs that instance from the graph while the replay
// re-creates a FRESH one for the same id. Touching the stale instance throws a raw
// TypeError mid-replay, which abandons every OLDER undo op — leaving the graph
// half-rolled-back rather than restored.
describe('rollback replays every op when an element is re-created mid-transaction', () => {
  const seed = (): Graph => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['P'], properties: { n: 1 } });
    const b = g.addVertex({ id: 'b', labels: ['P'], properties: { n: 2 } });

    g.addEdge({ id: 'e1', from: a, to: b, labels: ['R'], properties: { w: 1 } });

    return g;
  };
  const snapshot = (g: Graph): string =>
    JSON.stringify([
      [...g.vertices].map((v) => [v.id, [...v.labels].sort(), { ...v.properties }]).sort(),
      [...g.edges].map((e) => [e.id, e.from.id, e.to.id, [...e.labels].sort()]).sort(),
    ]);

  const rollsBackCleanly = (write: (g: Graph) => void): void => {
    const g = seed();
    const before = snapshot(g);

    expect(() =>
      g.transaction(() => {
        write(g);

        throw new Error('abort');
      }),
    ).toThrow('abort'); // the ABORT must surface — not a TypeError from the replay

    expect(snapshot(g)).toBe(before);
  };

  test('an edge added then removed by a cascading delete', () => {
    rollsBackCleanly((g) => {
      const a = g.getVertexById('a')!;
      const b = g.getVertexById('b')!;

      g.addEdge({ id: 'e2', from: a, to: b, labels: ['R'], properties: {} });
      g.removeVertex(b); // cascades to e1 AND the just-added e2
    });
  });

  test('an edge label added then the edge removed', () => {
    rollsBackCleanly((g) => {
      const e = g.getEdgeById('e1')!;

      g.addLabelToEdge('T', e);
      g.removeEdge(e);
    });
  });

  test('an edge label added then an endpoint detach-deleted', () => {
    rollsBackCleanly((g) => {
      g.addLabelToEdge('T', g.getEdgeById('e1')!);
      g.removeVertex(g.getVertexById('a')!);
    });
  });

  test('a vertex label added then the vertex removed', () => {
    rollsBackCleanly((g) => {
      const a = g.getVertexById('a')!;

      g.addLabelToVertex('Z', a);
      g.removeVertex(a);
    });
  });

  test('a property set then the element removed', () => {
    rollsBackCleanly((g) => {
      const a = g.getVertexById('a')!;

      a.setProperty('z', 1);
      g.removeVertex(a);
    });
  });
});

describe('schema ops roll back with the transaction', () => {
  const seed = (): Graph => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['N'], properties: { k: 1 } });
    g.addVertex({ id: 'b', labels: ['N'], properties: { k: 2 } });
    return g;
  };
  const rolledBack = (g: Graph, fn: () => void): void => {
    try {
      g.transaction(() => {
        fn();
        throw new Error('abort');
      });
    } catch {
      /* expected */
    }
  };

  test('an index / constraint created in a transaction is reverted on rollback', () => {
    const g = seed();
    rolledBack(g, () => {
      g.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] });
      g.createUniqueConstraint('N', 'k');
    });
    expect(g.vertexIndexes()).toEqual([]);
    expect(g.uniqueConstraints()).toEqual([]);
  });

  test('a schema op committed in a transaction persists', () => {
    const g = seed();
    g.transaction(() => g.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] }));
    expect(g.vertexIndexes()).toEqual(['k']);
  });

  test('an index dropped in a transaction is restored on rollback', () => {
    const g = seed();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] });
    rolledBack(g, () => g.dropVertexIndex('k'));
    expect(g.vertexIndexes()).toEqual(['k']);
  });

  test('a pre-existing index survives a rolled-back tx that also declares new schema', () => {
    const g = seed();
    g.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] });
    rolledBack(g, () => {
      g.addVertex({ id: 'c', labels: ['N'], properties: { k: 3 } });
      g.createUniqueConstraint('N', 'other');
    });
    expect(g.vertexIndexes()).toEqual(['k']); // pre-existing index intact
    expect(g.uniqueConstraints()).toEqual([]); // tx-created constraint reverted
    expect(g.vertexCount).toBe(2); // data write reverted
  });
});
