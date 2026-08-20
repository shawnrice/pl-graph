// Backend parity: the wasm build of the engine must agree with the FFI build.
//
// The other fuzzers prove "FFI == TS". wasm is the SAME Rust source compiled for a
// 32-bit target with a different allocator and a copy-in/copy-out memory boundary,
// so proving "wasm == FFI" gives wasm every guarantee already established for FFI,
// without running each suite twice against it.
//
// TRANSCENDENTALS ARE EXCLUDED, and that is a measured decision, not an oversight.
// On x86-64 Rust calls the system libm (glibc); wasm32 has none, so it links the
// pure-Rust `libm` crate — and the two disagree in the last ulp on exp, ln, sin,
// tan, cot, cosh, log, atan2 and power. Measured over 396 (function, argument)
// pairs: TS agrees with glibc 391 times but with the libm crate only 364, and the
// two Rust builds agree 368. So pointing native at the libm crate would fix
// wasm-vs-native by trading 5 TS-vs-native disagreements for 27 — strictly worse.
// Every IEEE-exact function (sqrt, abs, ceil, floor, round, sign, degrees,
// radians) agrees across all three, and those stay in the generator.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { createWasmEngineBackend } from './backend-wasm-engine.js';
import type { Backend } from './backend.js';
import { graphFromNdjson } from './graph.js';

const LIB = new URL(
  '../../../crates/lenke-engine/target/release/liblenke_engine.so',
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../../crates/lenke-engine/target/wasm32-unknown-unknown/release/lenke_engine.wasm',
  import.meta.url,
).pathname;

const hasBoth = existsSync(LIB) && existsSync(WASM);
const suite = hasBoth ? describe : describe.skip;
const ffi = hasBoth ? createFfiEngineBackend(LIB) : null;
const wasm = hasBoth ? await createWasmEngineBackend(await Bun.file(WASM).arrayBuffer()) : null;

const SEED = [
  '{"type":"node","id":"1","labels":["P"],"properties":{"n":3,"s":"a","x":-1,"m":{"k":1}}}',
  '{"type":"node","id":"2","labels":["P"],"properties":{"n":7,"s":"z","x":4}}',
  '{"type":"node","id":"3","labels":["Q"],"properties":{"n":5,"s":"m"}}',
  '{"type":"edge","id":"e1","labels":["R"],"from":"1","to":"2","properties":{"w":2}}',
  '{"type":"edge","id":"e2","labels":["R"],"from":"2","to":"3","properties":{"w":-1}}',
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

const NUMS = ['0', '-0.0', '1', '-1', '3.14', '1e21', '1e-7', '1e100', '9007199254740992', '0.1'];
const STRS = ["'a'", "''", "'a😀b'", "'中文'", "'inf'", "'nan'", "'0x10'"];
// IEEE-exact only — see the note at the top on why the transcendentals are out.
const FNS = [
  'abs',
  'sqrt',
  'ceil',
  'floor',
  'sign',
  'round',
  'degrees',
  'radians',
  'to_string',
  'to_integer',
  'to_float',
  'upper',
  'trim',
  'char_length',
  'byte_length',
  'reverse',
];
const PROPS = ['n', 's', 'x', 'm'];
const AGG = ['count', 'sum', 'avg', 'min', 'max', 'collect_list', 'stddev_pop'];

const expr = (r: () => number, d: number): string => {
  if (d <= 0 || r() < 0.35) {
    const leaf = [() => pick(r, NUMS), () => pick(r, STRS), () => `n.${pick(r, PROPS)}`];

    return pick(r, leaf)();
  }

  const e = (): string => expr(r, d - 1);
  const shapes = [
    () => `(${e()} ${pick(r, ['+', '-', '*', '/', '%'])} ${e()})`,
    () => `(${e()} ${pick(r, ['<', '>', '=', '<=', '<>'])} ${e()})`,
    () => `${pick(r, FNS)}(${e()})`,
    () => `[${e()}, ${e()}]`,
    () => `coalesce(${e()}, ${e()})`,
    () => `CASE WHEN (${e()} > ${e()}) THEN ${e()} ELSE ${e()} END`,
  ];

  return pick(r, shapes)();
};

const genQuery = (r: () => number): string => {
  const shapes = [
    () => `MATCH (n:P) RETURN ${pick(r, AGG)}(${expr(r, 2)}) AS x`,
    // `LET`-bound key: a RETURN alias is not a grouping element, and spelling it
    // that way keyed every row null — one group per query, whatever the expression.
    () => `MATCH (n:P) LET k = ${expr(r, 1)} RETURN k, count(*) AS c GROUP BY k ORDER BY k, c`,
    () => `MATCH (n:P) WHERE ${expr(r, 2)} RETURN n.n AS x ORDER BY x`,
    () => `MATCH (n:P) ORDER BY n.n LIMIT 2 RETURN ${expr(r, 2)} AS x, n.n AS t`,
    () => `MATCH (a:P)-[e:R]->(b) RETURN ${expr(r, 2)} AS x`,
    () => `FOR v IN ${expr(r, 2)} RETURN v AS x`,
    () => `MATCH (n:P) RETURN ${expr(r, 3)} AS x, n.n AS t ORDER BY t`,
  ];

  return pick(r, shapes)();
};

const WRITES = [
  'MATCH (n:P) SET n.z = 1 RETURN count(*) AS c',
  "INSERT (:Z {id: 'new', v: 2})",
  'MATCH (n:P) WHERE n.n > 3 REMOVE n.s RETURN count(*) AS c',
  'MATCH (n:Q) DETACH DELETE n',
  "_MERGE (u:P {id: '1'}) _ON_UPDATE SET u.hit = 1",
  'MATCH ()-[e:R]->() SET e.w = 9 RETURN count(*) AS c',
];
const GREMLIN = [
  'g.V().count()',
  "g.V().hasLabel('P').values('n')",
  "g.V().has('n', gt(3)).values('s')",
  'g.E().count()',
  "g.V().out('R').path()",
  "g.V().order().by('n').limit(2).values('n')",
  "g.V().group().by(label).by('n')",
];
const FORMATS = ['ndjson', 'pg-json', 'pg-text', 'graphson', 'csv'] as const;

const run = (backend: Backend, fn: (g: ReturnType<typeof graphFromNdjson>) => unknown): string => {
  const g = graphFromNdjson(backend, SEED);

  try {
    return JSON.stringify(fn(g));
  } catch (e) {
    return `ERR ${(e as { code?: string }).code ?? 'throw'}`;
  } finally {
    g.free();
  }
};

suite('backend parity: wasm agrees with ffi', () => {
  const SEED_BASE =
    process.env.FUZZ_SEED === undefined
      ? Math.floor(Math.random() * 0x1_0000_0000)
      : Number(process.env.FUZZ_SEED) >>> 0;
  const ITERATIONS = 300;

  test(`${ITERATIONS} random queries, writes, traversals and documents agree`, () => {
    expect(ffi!.abiVersion).toBe(wasm!.abiVersion);

    const divergences: string[] = [];
    const compare = (
      what: string,
      fn: (g: ReturnType<typeof graphFromNdjson>) => unknown,
    ): void => {
      if (divergences.length >= 5) {
        return;
      }

      const a = run(ffi!, fn);
      const b = run(wasm!, fn);

      if (a !== b) {
        divergences.push(`${what}\n    ffi:  ${a.slice(0, 200)}\n    wasm: ${b.slice(0, 200)}`);
      }
    };

    for (let i = 0; i < ITERATIONS && divergences.length === 0; i++) {
      const r = mulberry32(caseSeed(SEED_BASE, i));
      const q = genQuery(r);

      compare(`[seed ${SEED_BASE + i}] read: ${q}`, (g) => g.query(q));

      const w = pick(r, WRITES);

      compare(`[seed ${SEED_BASE + i}] write: ${w}`, (g) => {
        g.query(w);

        return g.serialize('ndjson');
      });

      const gr = pick(r, GREMLIN);

      compare(`[seed ${SEED_BASE + i}] gremlin: ${gr}`, (g) => g.gremlin(gr));

      const f = pick(r, FORMATS);

      compare(`[seed ${SEED_BASE + i}] codec: ${f}`, (g) => g.serialize(f));
    }

    const report = divergences.length
      ? `FUZZ_SEED=${SEED_BASE} bun test <this file> to reproduce:\n\n${divergences.join('\n\n')}`
      : 'no divergences';

    expect(report).toBe('no divergences');
  });
});
