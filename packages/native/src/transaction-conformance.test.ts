// Cross-engine differential for transactions. The SAME transaction scripts — explicit
// begin/commit/rollback around GQL statements, plus per-statement atomicity and
// deferred constraint checks — run on BOTH the TS engine (@lenke/core + @lenke/gql)
// and the Rust core (over bun:ffi), asserting the two agree on every outcome AND
// on the resulting graph state byte-for-byte.
//
// Run: bun test packages/native/src/transaction-conformance.test.ts
import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { nativeBackend, NATIVE_LIB, nativeReady } from './conformance-harness.js';
import { graphFromNdjson, type RustGraph } from './graph.js';

const hasLib = nativeReady;

if (!hasLib) {
  // eslint-disable-next-line no-console
  console.warn(`[transaction] skipping: ${NATIVE_LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

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

// A single driver shape over either engine, so a script is written once.
type Engine = {
  query: (sql: string) => unknown;
  transaction: (fn: () => void) => void;
};

const tsEngine = (): Engine & { _g: Graph } => {
  const g = tsDeserialize(SEED, 'ndjson', new Graph());

  return { query: (sql) => tsQuery(g, sql), transaction: (fn) => g.transaction(fn), _g: g };
};

const nativeEngine = (): Engine & { _g: RustGraph } => {
  const backend = nativeBackend();
  const g = graphFromNdjson(backend, SEED);

  return { query: (sql) => g.query(sql), transaction: (fn) => g.transaction(fn), _g: g };
};

const READ = `MATCH (n:Acct) RETURN n.id, n.bal, n.email ORDER BY n.id`;

/** Run `script` on both engines; assert every outcome and the final Acct state agree. */
const differential = (
  declare: (e: Engine) => void,
  script: Array<{ label: string; run: (e: Engine) => unknown }>,
): void => {
  const ts = tsEngine();
  const native = nativeEngine();
  declare(ts);
  declare(native);

  for (const { label, run } of script) {
    const a = outcome(() => run(ts));
    const b = outcome(() => run(native));

    expect(b, `outcome mismatch: ${label}`).toEqual(a);
  }

  expect(JSON.stringify(native.query(READ)), 'final state mismatch').toEqual(
    JSON.stringify(ts.query(READ)),
  );
};

suite('transactions differential: explicit transactions (TS vs native)', () => {
  test('a committed transaction persists all its statements', () => {
    differential(() => {}, [
      {
        label: 'atomic two-insert commit',
        run: (e) =>
          e.transaction(() => {
            e.query(`INSERT (:Acct {id: 'a', bal: 100})`);
            e.query(`INSERT (:Acct {id: 'b', bal: 200})`);
          }),
      },
    ]);
  });

  test('a transaction whose body throws rolls every statement back', () => {
    differential(() => {}, [
      {
        label: 'rollback on throw leaves no trace',
        run: (e) => {
          try {
            e.transaction(() => {
              e.query(`INSERT (:Acct {id: 'x', bal: 1})`);
              e.query(`INSERT (:Acct {id: 'y', bal: 2})`);

              throw new Error('boom');
            });
          } catch {
            // swallow — the point is the graph state, compared below
          }
        },
      },
    ]);
  });
});

suite('transactions differential: per-statement atomicity (TS vs native)', () => {
  test('a multi-row INSERT that violates unique on a later row leaves zero rows', () => {
    differential(
      (e) => declareUnique(e),
      [
        {
          // One statement, two bindings via FOR-unwind: both rows carry id='dup',
          // so the second collides. Per-statement atomicity must roll the first
          // row back too — a partial write would diverge across engines.
          label: 'FOR-INSERT with a duplicate unique value → violation, zero rows',
          run: (e) => e.query(`FOR x IN [1, 2] INSERT (:Acct {id: 'dup', bal: x})`),
        },
      ],
    );
  });
});

suite('transactions differential: a faulted statement keeps the caller frame', () => {
  // Found by the write-path fuzzer. A GQL write statement runs in its OWN frame for
  // per-statement atomicity; when that statement faults inside a transaction the
  // CALLER opened, only the statement's own writes may unwind. The TS engine used
  // to tear the whole transaction down: the caller's earlier writes were lost, the
  // depth dropped to 0 so every later write escaped the transaction entirely, and
  // the caller's own commit then threw `commit called with no open transaction`.
  //
  // This is the realistic "skip the bad rows, commit the good ones" loop.
  const script = (abort: boolean): Array<{ label: string; run: (e: Engine) => unknown }> => [
    {
      label: `swallowed fault inside a caller transaction (abort=${abort})`,
      run: (e) =>
        e.transaction(() => {
          e.query(`MATCH (n:Acct) WHERE n.id = 'a' SET n.bal = 111`);

          try {
            // Collides with the unique constraint → faults, and the app skips it.
            e.query(`INSERT (:Acct {id: 'a', bal: 1})`);
          } catch {
            /* swallowed on purpose */
          }

          e.query(`MATCH (n:Acct) WHERE n.id = 'b' SET n.bal = 222`);

          if (abort) {
            throw new Error('abort');
          }
        }),
    },
  ];

  test('the writes around a swallowed fault still commit together', () => {
    differential((e) => declareUnique(e), script(false));
  });

  test('and an abort still rolls every one of them back', () => {
    differential((e) => declareUnique(e), script(true));
  });

  test('a faulted statement does not leak its own partial writes either', () => {
    differential(
      (e) => declareUnique(e),
      [
        {
          label: 'FOR-INSERT collides on the 2nd row inside a caller transaction',
          run: (e) =>
            e.transaction(() => {
              try {
                e.query(`FOR x IN [1, 2] INSERT (:Acct {id: 'dup2', bal: x})`);
              } catch {
                /* swallowed */
              }

              e.query(`MATCH (n:Acct) WHERE n.id = 'a' SET n.bal = 7`);
            }),
        },
      ],
    );
  });
});

suite('transactions differential: deferred constraint checks (TS vs native)', () => {
  test('required is checked at commit, not per statement (fill the key in a later statement)', () => {
    differential(
      (e) => declareRequired(e),
      [
        {
          label: 'insert without required, set it, commit — ok',
          run: (e) =>
            e.transaction(() => {
              e.query(`INSERT (:Acct {id: 'u'})`);
              e.query(`MATCH (n:Acct {id: 'u'}) SET n.email = 'u@x.io'`);
            }),
        },
      ],
    );
  });

  test('a required violation that survives to commit rolls the whole transaction back', () => {
    differential(
      (e) => declareRequired(e),
      [
        {
          label: 'insert without required, never set it, commit — violation',
          run: (e) => {
            try {
              e.transaction(() => {
                e.query(`INSERT (:Acct {id: 'v'})`);
                e.query(`INSERT (:Acct {id: 'w', email: 'w@x.io'})`);
              });
            } catch {
              // compared by final state below
            }
          },
        },
      ],
    );
  });
});

suite('transactions differential: ISO transaction keywords (TS vs native)', () => {
  test('START … INSERT … COMMIT via keywords persists on both engines', () => {
    differential(() => {}, [
      { label: 'START', run: (e) => e.query(`START TRANSACTION`) },
      { label: 'insert a', run: (e) => e.query(`INSERT (:Acct {id: 'a', bal: 100})`) },
      { label: 'insert b', run: (e) => e.query(`INSERT (:Acct {id: 'b', bal: 200})`) },
      { label: 'COMMIT', run: (e) => e.query(`COMMIT`) },
    ]);
  });

  test('START … INSERT … ROLLBACK via keywords discards on both engines', () => {
    differential(() => {}, [
      { label: 'seed', run: (e) => e.query(`INSERT (:Acct {id: 'seed', bal: 1})`) },
      { label: 'START', run: (e) => e.query(`START TRANSACTION`) },
      { label: 'insert a', run: (e) => e.query(`INSERT (:Acct {id: 'a', bal: 100})`) },
      { label: 'ROLLBACK', run: (e) => e.query(`ROLLBACK`) },
    ]);
  });

  test('COMMIT WORK / ROLLBACK WORK parse and behave identically', () => {
    differential(() => {}, [
      { label: 'START #1', run: (e) => e.query(`START TRANSACTION`) },
      { label: 'insert a', run: (e) => e.query(`INSERT (:Acct {id: 'a', bal: 1})`) },
      { label: 'COMMIT WORK', run: (e) => e.query(`COMMIT WORK`) },
      { label: 'START #2', run: (e) => e.query(`START TRANSACTION`) },
      { label: 'insert b', run: (e) => e.query(`INSERT (:Acct {id: 'b', bal: 2})`) },
      { label: 'ROLLBACK WORK', run: (e) => e.query(`ROLLBACK WORK`) },
    ]);
  });

  test('a deferred required constraint via keywords: commits when valid', () => {
    differential(
      (e) => declareRequired(e),
      [
        { label: 'START', run: (e) => e.query(`START TRANSACTION`) },
        { label: 'insert u (no email)', run: (e) => e.query(`INSERT (:Acct {id: 'u'})`) },
        {
          label: 'set email later',
          run: (e) => e.query(`MATCH (n:Acct {id: 'u'}) SET n.email = 'u@x.io'`),
        },
        { label: 'COMMIT', run: (e) => e.query(`COMMIT`) },
      ],
    );
  });

  test('a deferred required constraint via keywords: COMMIT rolls back when invalid', () => {
    differential(
      (e) => declareRequired(e),
      [
        { label: 'START', run: (e) => e.query(`START TRANSACTION`) },
        { label: 'insert v ok', run: (e) => e.query(`INSERT (:Acct {id: 'v', email: 'v@x.io'})`) },
        { label: 'insert w (no email)', run: (e) => e.query(`INSERT (:Acct {id: 'w'})`) },
        // Both engines throw ConstraintViolation and roll the whole tx back.
        { label: 'COMMIT → violation', run: (e) => e.query(`COMMIT`) },
      ],
    );
  });

  test('nested START TRANSACTION is the same coded error on both engines', () => {
    differential(() => {}, [
      { label: 'START', run: (e) => e.query(`START TRANSACTION`) },
      { label: 'nested START → error', run: (e) => e.query(`START TRANSACTION`) },
      // Close the still-open transaction so the final-state read agrees cleanly.
      { label: 'ROLLBACK', run: (e) => e.query(`ROLLBACK`) },
    ]);
  });

  test('COMMIT / ROLLBACK with no active transaction is the same coded error', () => {
    differential(() => {}, [
      { label: 'COMMIT no tx → error', run: (e) => e.query(`COMMIT`) },
      { label: 'ROLLBACK no tx → error', run: (e) => e.query(`ROLLBACK`) },
    ]);
  });

  test('READ ONLY rejects a write and allows a read, identically', () => {
    differential(() => {}, [
      { label: 'seed', run: (e) => e.query(`INSERT (:Acct {id: 'seed', bal: 1})`) },
      { label: 'START READ ONLY', run: (e) => e.query(`START TRANSACTION READ ONLY`) },
      { label: 'read is allowed', run: (e) => e.query(`MATCH (n:Acct) RETURN n.id`) },
      { label: 'INSERT rejected', run: (e) => e.query(`INSERT (:Acct {id: 'x', bal: 9})`) },
      { label: 'SET rejected', run: (e) => e.query(`MATCH (n:Acct) SET n.bal = 5`) },
      { label: 'DELETE rejected', run: (e) => e.query(`MATCH (n:Acct {id: 'seed'}) DELETE n`) },
      { label: 'COMMIT', run: (e) => e.query(`COMMIT`) },
      // After commit the mode clears — a write applies on both engines.
      { label: 'write after commit', run: (e) => e.query(`INSERT (:Acct {id: 'x', bal: 9})`) },
    ]);
  });
});

// A unique constraint isn't declarable via GQL DDL; both engines take the same
// programmatic call. Small shims keep the driver engine-neutral.
const declareRequired = (e: Engine): void => {
  const g = (e as unknown as { _g: Graph | RustGraph })._g;
  g.createRequiredConstraint('Acct', 'email');
};

const declareUnique = (e: Engine): void => {
  const g = (e as unknown as { _g: Graph | RustGraph })._g;
  g.createUniqueConstraint('Acct', 'id');
};
