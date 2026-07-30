// Differential fuzzer: generate random GQL scalar expressions from a seeded PRNG,
// run each `RETURN <expr>` through BOTH engines (the TS @lenke/gql engine and the
// Rust core over bun:ffi), and assert byte-identical behavior — the same JSON when
// both succeed, and both-error when either errors. Byte-identity is the hard
// invariant, so any value that renders/compares/coerces differently between the two
// engines is a bug. The generator deliberately favors the edge values that have
// bitten us before (extreme magnitudes, -0, NaN/Inf producers, astral/control
// chars, mixed types, deep nesting).
//
// Deterministic: a fixed SEED + iteration count, so CI is reproducible and a failure
// prints the exact reproducing expression. Bump ITERATIONS or SEED locally to hunt.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './backend-ffi.js';
import { graphFromFormat } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const suite = existsSync(LIB) ? describe : describe.skip;

// A tiny two-vertex graph so property access + aggregates over rows can be fuzzed.
const NDJSON = [
  '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a","x":-1}}',
  '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z","x":4}}',
].join('\n');

// --- seeded PRNG (mulberry32) -----------------------------------------------
const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;

  return () => {
    a |= 0;
    a = (a + 0x6d_2b_79_f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);

    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
};

const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)];

// Number literals: normal, signed-zero, extreme magnitudes (exponential threshold),
// safe-integer boundary, and values that stress toString/collation.
const NUMS = [
  '0',
  '1',
  '-1',
  '0.0',
  '3.14',
  '-2.5',
  '0.1',
  '1e21',
  '1e-7',
  '1e100',
  '1e-300',
  '123456.789',
  '9007199254740992',
  '2',
  '100',
] as const;

// String literals (GQL single-quoted), including astral, BMP-boundary, control,
// and escape chars — the collation / escaping edge space.
const STR_CHARS = ['a', 'Z', '0', ' ', '\\u0041', '\\uE000', '\\n', '\\t', 'ä', '中'] as const;

const genString = (r: () => number): string => {
  const len = Math.floor(r() * 4);
  let s = "'";

  for (let i = 0; i < len; i++) {
    s += pick(r, STR_CHARS);
  }

  // Occasionally splice an astral char (surrogate pair) — the code-point vs
  // code-unit collation trap.
  if (r() < 0.2) {
    s += '😀';
  }

  return `${s}'`;
};

const TEMPORALS = [
  "date('2020-01-01')",
  "date('0001-01-01')",
  "datetime('2020-06-15T08:30:00')",
  "duration('P1Y2M')",
  "duration('P3DT4H')",
  "zoned_datetime('2020-01-01T00:00:00Z')",
] as const;

// Strings that stress numeric-string coercion: non-finite spellings, radix
// prefixes, whitespace, empty — the exact forms JS Number() and Rust
// str::parse::<f64> disagree on.
const NUM_STRINGS = ["'inf'", "'nan'", "'0x10'", "'  5  '", "''", "'Infinity'", "'1e3'"] as const;

// A leaf: a literal value or a property reference.
const genLeaf = (r: () => number): string => {
  const p = r();

  if (p < 0.3) {
    return pick(r, NUMS);
  }

  if (p < 0.4) {
    return pick(r, NUM_STRINGS);
  }

  if (p < 0.55) {
    return genString(r);
  }

  if (p < 0.62) {
    return r() < 0.5 ? 'true' : 'false';
  }

  if (p < 0.68) {
    return 'null';
  }

  if (p < 0.8) {
    return pick(r, TEMPORALS);
  }

  // property access over the row (n: number, s: string, x: number)
  return `n.${pick(r, ['n', 's', 'x'])}`;
};

const ARITH = ['+', '-', '*', '/', '%'] as const;
const CMP = ['<', '>', '=', '<=', '>=', '<>'] as const;
const UNARY_FN = [
  'abs',
  'sqrt',
  'ceil',
  'floor',
  'sign',
  'to_string',
  'to_integer',
  'to_float',
  'size',
] as const;
const STR_FN = ['upper', 'lower', 'trim', 'char_length'] as const;

const genExpr = (r: () => number, depth: number): string => {
  if (depth <= 0 || r() < 0.35) {
    return genLeaf(r);
  }

  const p = r();

  const e = (): string => genExpr(r, depth - 1);

  if (p < 0.25) {
    return `(${e()} ${pick(r, ARITH)} ${e()})`;
  }

  if (p < 0.45) {
    return `(${e()} ${pick(r, CMP)} ${e()})`;
  }

  if (p < 0.6) {
    return `${pick(r, UNARY_FN)}(${e()})`;
  }

  if (p < 0.72) {
    return `${pick(r, STR_FN)}(${e()})`;
  }

  if (p < 0.82) {
    return `[${e()}, ${e()}]`;
  }

  if (p < 0.9) {
    return `coalesce(${e()}, ${e()})`;
  }

  return `CASE WHEN (${e()} ${pick(r, CMP)} ${e()}) THEN ${e()} ELSE ${e()} END`;
};

const AGG = ['count', 'sum', 'avg', 'min', 'max', 'collect_list'] as const;

// A full query: either a scalar RETURN (single-row constant) or a MATCH + aggregate
// over the two rows (exercises DISTINCT/GROUP BY keying, min/max/sum NaN handling).
const genQuery = (r: () => number): string => {
  if (r() < 0.35) {
    const distinct = r() < 0.5 ? 'DISTINCT ' : '';

    return `MATCH (n:T) RETURN ${pick(r, AGG)}(${distinct}${genExpr(r, 2)}) AS x`;
  }

  return `MATCH (n:T) RETURN ${genExpr(r, 3)} AS x`;
};

const codeOf = (e: unknown): string =>
  (e as { code?: string })?.code ?? (e instanceof Error ? e.name : 'unknown');

type Outcome = { ok: true; json: string } | { ok: false; code: string };

suite('differential fuzz: TS gql engine vs Rust core', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromFormat(backend, NDJSON, 'ndjson');
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());

  const run = (engine: 'ts' | 'native', q: string): Outcome => {
    try {
      const rows = engine === 'ts' ? tsQuery(tsGraph, q) : nativeGraph.query(q);

      return { ok: true, json: JSON.stringify(rows) };
    } catch (e) {
      return { ok: false, code: codeOf(e) };
    }
  };

  // Seed: random each run, so every run explores fresh expressions — fuzzing is
  // discovery, not a fixed corpus (the specific bugs found here are pinned as their
  // own deterministic unit tests, which are the permanent regression guards). Set
  // FUZZ_SEED=<n> to replay a run exactly; the seed is printed on failure so any
  // divergence is reproducible. Property-based-testing convention: random by
  // default, seed on failure (proptest, QuickCheck, fast-check all do this).
  const SEED =
    process.env.FUZZ_SEED !== undefined
      ? Number(process.env.FUZZ_SEED) >>> 0
      : Math.floor(Math.random() * 0x1_0000_0000);
  const ITERATIONS = 20_000;

  test(`${ITERATIONS} random expressions render byte-identically across engines`, () => {
    const divergences: string[] = [];

    for (let i = 0; i < ITERATIONS; i++) {
      const q = genQuery(mulberry32(SEED + i));
      const ts = run('ts', q);
      const nat = run('native', q);

      // Both errored → acceptable (both reject the input); a shape divergence is
      // when exactly one succeeds, or both succeed with different JSON.
      if (ts.ok && nat.ok) {
        if (ts.json !== nat.json) {
          divergences.push(
            `[seed ${SEED + i}] ${q}\n    ts:     ${ts.json}\n    native: ${nat.json}`,
          );
        }
      } else if (ts.ok !== nat.ok) {
        const tsSide = ts.ok ? `ok ${ts.json}` : `err ${(ts as { code: string }).code}`;
        const natSide = nat.ok ? `ok ${nat.json}` : `err ${(nat as { code: string }).code}`;
        divergences.push(`[seed ${SEED + i}] ${q}\n    ts:     ${tsSide}\n    native: ${natSide}`);
      }

      if (divergences.length >= 10) {
        break; // cap the report
      }
    }

    const report = divergences.length
      ? `FUZZ_SEED=${SEED} bun test <this file> to reproduce:\n\n${divergences.join('\n\n')}`
      : 'no divergences';
    expect(report).toBe('no divergences');
  });
});
