// Differential fuzzer for the WRITE path. The read fuzzer (differential-fuzz.test.ts)
// generates queries; this generates random write SEQUENCES and, after every
// statement, compares BOTH engines on (a) the returned rows and (b) the ENTIRE
// resulting graph state. Writes are where a divergence corrupts data rather than
// just misreporting it, and they had no fuzzer.
//
// Graph state is compared canonically: element order and property-key order are
// unspecified (the two engines place the special `id` key differently on INSERT),
// so each NDJSON line is re-emitted with sorted keys and the lines are sorted.
//
// Every 4th sequence instead checks the ROLLBACK invariant — the same writes inside
// a transaction that throws must leave the graph byte-identical to where it started,
// in each engine. That is what caught a TS-side bug where a statement faulting inside
// a caller's transaction tore the whole frame down, so later writes escaped it.
//
// Seed: random each run (FUZZ_SEED=<n> to replay); the failing seed is printed.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize, serialize as tsSerialize } from '@lenke/serialization';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { graphFromNdjson } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const suite = existsSync(LIB) ? describe : describe.skip;

// Every element carries an explicit string `id` — that IS the element identity, so a
// created node's id is deterministic in both engines (an id-less INSERT mints a UUID
// in TS and a counter in native: inherent, not a divergence).
const SEED = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"n":1,"s":"x"}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"n":2,"s":"y","m":{"k":1}}}',
  '{"type":"node","id":"c","labels":["Q"],"properties":{"n":3}}',
  '{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"b","properties":{"w":5}}',
  '{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"c","properties":{"w":-1}}',
].join('\n');

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;

  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// Distinct `FUZZ_SEED`s must explore DISJOINT cases. `SEED + i` did not: seeds 1
// and 2 differ in one case out of four hundred, so running eight seeds was ~1.02x
// the coverage of running one, not 8x. Multiplying by a large odd constant gives
// each base seed its own region while keeping a reported seed reproducible.
const caseSeed = (base: number, i: number): number => base * 1_000_003 + i;
const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)];

const VALUES = [
  '1',
  '0',
  '-1',
  '2.5',
  '-0.0',
  '1e21',
  "'str'",
  "''",
  'true',
  'false',
  'null',
  "date('2020-01-01')",
  "duration('P1D')",
  '[1, 2]',
  '{k: 1}',
  "'a😀b'",
  '9007199254740992',
];
const KEYS = ['n', 's', 'm', 'z', 'w'];
const LABELS = ['P', 'Q', 'Z'];

const genStatement = (r: () => number, step: number): string => {
  const p = r();
  const v = (): string => pick(r, VALUES);
  const k = (): string => pick(r, KEYS);
  const where = (): string =>
    pick(r, [
      '',
      ` WHERE n.n > ${pick(r, ['0', '1', '2'])}`,
      " WHERE n.s = 'x'",
      ' WHERE n.n IS NOT NULL',
    ]);

  if (p < 0.16) {
    return `INSERT (:${pick(r, LABELS)} {id: 'g${step}', ${k()}: ${v()}})`;
  }

  if (p < 0.3) {
    return `MATCH (n:P)${where()} SET n.${k()} = ${v()} RETURN count(*) AS c`;
  }

  if (p < 0.4) {
    return `MATCH (n:P)${where()} SET n.${k()} = ${v()}, n.${k()} = ${v()} RETURN count(*) AS c`;
  }

  if (p < 0.5) {
    return `MATCH (n)${where()} REMOVE n.${k()} RETURN count(*) AS c`;
  }

  if (p < 0.58) {
    return `MATCH (n:P)${where()} SET n:${pick(r, LABELS)} RETURN count(*) AS c`;
  }

  if (p < 0.66) {
    return `MATCH (n:${pick(r, LABELS)}) REMOVE n:${pick(r, LABELS)} RETURN count(*) AS c`;
  }

  if (p < 0.74) {
    return `MATCH (n:${pick(r, LABELS)})${where()} DETACH DELETE n`;
  }

  if (p < 0.8) {
    return `MATCH (n:${pick(r, LABELS)})${where()} DELETE n`;
  }

  if (p < 0.86) {
    return `MATCH (a:P), (b:Q) INSERT (a)-[:R {id: 'ge${step}', w: ${v()}}]->(b)`;
  }

  if (p < 0.9) {
    return `MATCH ()-[e:R]->() SET e.${k()} = ${v()} RETURN count(*) AS c`;
  }

  // `_MERGE` keyed upsert — the key set is the inline props; a hit updates, a miss
  // creates. `_ON_CREATE` / `_ON_UPDATE` fire on their respective branch.
  if (p < 0.96) {
    const key = pick(r, ["{id: 'a'}", "{id: 'zz'}", `{n: ${pick(r, ['1', '2', '9'])}}`]);
    const tail = pick(r, [
      '',
      ` _ON_CREATE SET u.${k()} = ${v()}`,
      ` _ON_UPDATE SET u.${k()} = ${v()}`,
      ` _ON_CREATE SET u.${k()} = ${v()} _ON_UPDATE SET u.${k()} = ${v()}`,
    ]);

    return `_MERGE (u:${pick(r, LABELS)} ${key})${tail}`;
  }

  return `MATCH ()-[e:R]->() WHERE e.w < 0 DELETE e`;
};

// Compare graph STATE, not serialization order: parse each line and re-emit with
// sorted property keys and labels. (Property key order in NDJSON is insertion
// order, and the two engines place the special `id` key differently on INSERT —
// a serialization detail, not a difference in what the graph holds.)
const canon = (line: string): string => {
  const o = JSON.parse(line) as Record<string, unknown>;
  const props = (o.properties ?? {}) as Record<string, unknown>;
  const sorted: Record<string, unknown> = {};

  for (const k of Object.keys(props).sort()) {
    sorted[k] = props[k];
  }

  const labels = Array.isArray(o.labels) ? [...(o.labels as string[])].sort() : o.labels;

  return JSON.stringify({ ...o, labels, properties: sorted });
};
const norm = (s: string): string =>
  s.trim().split('\n').filter(Boolean).map(canon).sort().join('\n');

/**
 * A statement's outcome for comparison: its rows, or just "it failed".
 *
 * The error CODE is deliberately not compared. When a single statement has two
 * independent faults reachable — say `MATCH (n) WHERE n.n > 1 DELETE n` over a
 * graph holding both a node whose `n` is a DATE (invalid comparison) and a still-
 * connected node (invalid delete) — which fault is reported depends on the
 * interleaving: the TS engine filters every row before writing any, the native one
 * streams and writes as it matches. Both reject the statement and both leave the
 * graph untouched, so only the diagnostic differs; pinning it would mean forcing
 * filter-all-then-write and giving up streaming writes. Each fault ALONE reports
 * the same code in both engines. The read fuzzer takes the same stance.
 *
 * The graph STATE after every statement is still compared exactly — that is where
 * a write divergence actually costs something, and it is what caught the
 * transaction-frame bug this fuzzer was written for.
 */
const outcome = (run: () => unknown): string => {
  try {
    return JSON.stringify(run());
  } catch {
    return 'ERR';
  }
};

suite('differential fuzz: write path (TS engine vs Rust engine)', () => {
  const backend = createFfiEngineBackend(LIB);
  const SEED_COUNT =
    process.env.FUZZ_SEED === undefined
      ? Math.floor(Math.random() * 0x1_0000_0000)
      : Number(process.env.FUZZ_SEED) >>> 0;
  const ITERATIONS = 400;

  test(`${ITERATIONS} random write sequences agree on rows and graph state`, () => {
    const divergences: string[] = [];

    for (let i = 0; i < ITERATIONS && divergences.length < 5; i++) {
      const r = mulberry32(caseSeed(SEED_COUNT, i));
      const nat = graphFromNdjson(backend, SEED);
      const ts = tsDeserialize(SEED, 'ndjson', new Graph());

      try {
        // Every 4th sequence checks the rollback invariant instead.
        if (i % 4 === 3) {
          const before: [string, string] = [
            norm(tsSerialize(ts, 'ndjson')),
            norm(nat.serialize('ndjson')),
          ];
          const stmts = [0, 1, 2].map((step) => genStatement(r, step));

          for (const [graph, run] of [
            [ts, (q: string) => tsQuery(ts, q)],
            [nat, (q: string) => nat.query(q)],
          ] as const) {
            try {
              (graph as { transaction: (fn: () => void) => void }).transaction(() => {
                for (const q of stmts) {
                  try {
                    run(q);
                  } catch {
                    // A statement may legitimately fault; the app skips it and the
                    // enclosing transaction must still protect everything else.
                  }
                }

                throw new Error('fuzz rollback');
              });
            } catch {
              // expected: the abort propagates after the rollback
            }
          }

          const after: [string, string] = [
            norm(tsSerialize(ts, 'ndjson')),
            norm(nat.serialize('ndjson')),
          ];

          for (const [idx, engine] of (['ts', 'native'] as const).entries()) {
            if (before[idx] !== after[idx]) {
              divergences.push(
                `[seed ${SEED_COUNT + i}] ${engine}: rolled-back writes LEAKED\n    ${stmts.join('\n    ')}\n  before:\n${before[idx]}\n  after:\n${after[idx]}`,
              );
            }
          }

          continue;
        }

        const history: string[] = [];

        for (let step = 0; step < 4; step++) {
          const q = genStatement(r, step);

          history.push(q);

          const rowsTs = outcome(() => tsQuery(ts, q));
          const rowsNative = outcome(() => nat.query(q));
          const stateTs = norm(tsSerialize(ts, 'ndjson'));
          const stateNative = norm(nat.serialize('ndjson'));

          if (rowsTs !== rowsNative || stateTs !== stateNative) {
            const state =
              stateTs === stateNative
                ? ''
                : `\n  state ts:\n${stateTs}\n  state native:\n${stateNative}`;

            divergences.push(
              `[seed ${SEED_COUNT + i}] after:\n    ${history.join('\n    ')}\n  rows ts=${rowsTs}\n  rows native=${rowsNative}${state}`,
            );
            break; // a divergence poisons the rest of the sequence
          }
        }
      } finally {
        nat.free();
      }
    }

    const report = divergences.length
      ? `FUZZ_SEED=${SEED_COUNT} bun test <this file> to reproduce:\n\n${divergences.join('\n\n')}`
      : 'no divergences';

    expect(report).toBe('no divergences');
  });
});
